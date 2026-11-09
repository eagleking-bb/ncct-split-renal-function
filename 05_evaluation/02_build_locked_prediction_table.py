#!/usr/bin/env python3
"""Build the anonymized patient-side locked prediction master table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import MODEL_ORDER, anonymize_id, configure_logging, load_config, load_prediction, output_dir, write_environment_manifest


PRED_COLUMNS = {
    "clinical_only": "predicted_gfr_clinical_only",
    "radiomics_only": "predicted_gfr_radiomics_only",
    "image_only": "predicted_gfr_image_only",
    "early_concat": "predicted_gfr_early_concat",
    "cross_attention": "predicted_gfr_cross_attention",
    "dfme": "predicted_gfr_dfme",
}


EXPECTED_EXTERNAL_PATIENTS = 169


def apply_external_case_list(predictions: dict[str, pd.DataFrame], path: str | Path) -> dict[str, pd.DataFrame]:
    """Restrict the external predictions to the locked cohort.

    The analysis set is supplied as a list of case identifiers; the composition
    of that list is described in the Supplementary Materials.
    """
    case_list = pd.read_csv(path)
    column = "anonymized_patient_id" if "anonymized_patient_id" in case_list.columns else "case_id"
    values = case_list[column].astype(str)
    keep_ids = set(values.map(lambda v: v if v.startswith("DFME_") else anonymize_id(v)))
    filtered = {}
    for key, frame in predictions.items():
        keep = frame["case_id"].map(anonymize_id).isin(keep_ids)
        filtered[key] = frame.loc[keep].reset_index(drop=True)
    patients = filtered["dfme"]["case_id"].nunique()
    if patients != EXPECTED_EXTERNAL_PATIENTS:
        raise ValueError(
            f"external cohort must hold {EXPECTED_EXTERNAL_PATIENTS} patients, found {patients}"
        )
    return filtered


def load_disease(config: dict) -> pd.DataFrame:
    frames = []
    for key in ("disease_center1_csv", "disease_center2_csv", "disease_center3_csv"):
        frame = pd.read_csv(config["inputs"][key], dtype={"ID": str})
        frame = frame[frame["ID"].notna()].copy()
        frame["ID"] = frame["ID"].str.strip()
        if frame["ID"].duplicated().any():
            duplicates = frame[frame["ID"].duplicated(False)]
            if duplicates.groupby("ID")[["left_disease", "right_disease"]].nunique(dropna=False).max().max() > 1:
                raise ValueError(f"Conflicting disease labels in {key}")
            frame = frame.drop_duplicates("ID")
        frames.append(frame[["ID", "left_disease", "right_disease"]])
    disease = pd.concat(frames, ignore_index=True)
    if disease["ID"].duplicated().any():
        raise ValueError("Disease files overlap on patient ID")
    return disease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--external-case-list", type=Path, default=None,
                        help="optional CSV of the case identifiers forming the locked external cohort")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = configure_logging(config, "02_build_locked_prediction_table")
    case_dir = output_dir(config, "case_level")
    audit_dir = output_dir(config, "audit")

    clinical = pd.read_csv(config["inputs"]["clinical_csv"], dtype={"ID": str})
    clinical = clinical.loc[:, ~clinical.columns.str.startswith("Unnamed")].copy()
    clinical["ID"] = clinical["ID"].str.strip()
    if clinical["ID"].duplicated().any():
        raise ValueError("Clinical source contains duplicate patient IDs")
    audit = pd.read_csv(config["inputs"]["nifti_audit_csv"], dtype={"patient_id": str})
    audit["patient_id"] = audit["patient_id"].str.strip()
    disease = load_disease(config)

    all_rows = []
    audit_rows = []
    for split in ("internal", "external"):
        predictions = {key: load_prediction(config["models"][key][split]) for key in MODEL_ORDER}
        if split == "external" and args.external_case_list is not None:
            predictions = apply_external_case_list(predictions, args.external_case_list)
        reference = predictions["dfme"]
        reference_ids = set(reference["case_id"])
        for key, frame in predictions.items():
            if set(frame["case_id"]) != reference_ids:
                raise ValueError(f"{split}/{key}: model cohort differs from DFME")
            aligned = frame.set_index("case_id").loc[reference["case_id"]].reset_index()
            truth = ["left_true", "right_true", "left_cls_true", "right_cls_true"]
            if not np.allclose(aligned[truth], reference[truth], rtol=0, atol=1e-8):
                raise ValueError(f"{split}/{key}: truth or label mismatch")
        early = predictions["early_concat"].set_index("case_id").loc[reference["case_id"]]
        cross = predictions["cross_attention"].set_index("case_id").loc[reference["case_id"]]
        for column in ("left_pred", "right_pred", "left_cls_prob", "right_cls_prob"):
            source_column = column
            if column.endswith("pred") and "left_pred_raw" in early.columns and "left_pred_raw" in cross.columns:
                source_column = column + "_raw"
            expected = 0.5 * early[source_column].to_numpy(float) + 0.5 * cross[source_column].to_numpy(float)
            actual = reference[column].to_numpy(float)
            # GFR files may apply max(raw, 0) after arithmetic averaging.
            if column.endswith("pred"):
                expected = np.maximum(expected, 0)
            audit_rows.append({"split": split, "field": column, "component_source_field": source_column, "operation_order": "0.5/0.5 raw average then max(x,0) for GFR; direct average for probabilities", "max_abs_difference_dfme_vs_fixed_average": float(np.max(np.abs(expected - actual)))})
        wide = reference[["case_id", "left_true", "right_true", "left_cls_true", "right_cls_true", "left_cls_prob", "right_cls_prob"]].copy()
        for key, frame in predictions.items():
            indexed = frame.set_index("case_id").loc[wide["case_id"]]
            wide[f"{key}_left_pred"] = np.maximum(indexed["left_pred"].to_numpy(float), 0)
            wide[f"{key}_right_pred"] = np.maximum(indexed["right_pred"].to_numpy(float), 0)
        wide = wide.merge(clinical, left_on="case_id", right_on="ID", how="left", validate="one_to_one")
        wide = wide.merge(audit, left_on="case_id", right_on="patient_id", how="left", suffixes=("", "_audit"), validate="one_to_one")
        wide = wide.merge(disease, left_on="case_id", right_on="ID", how="left", suffixes=("", "_disease"), validate="one_to_one")
        if wide[["age", "sex", "height", "weight", "SCr"]].isna().all(axis=1).any():
            raise ValueError(f"{split}: clinical covariates could not be linked for some predictions")
        for _, row in wide.iterrows():
            for side, other in (("left", "right"), ("right", "left")):
                true_side = float(row[f"{side}_true"])
                true_other = float(row[f"{other}_true"])
                total = true_side + true_other
                record = {
                    "anonymized_patient_id": anonymize_id(row["case_id"]),
                    "center": row["Center"], "split": split, "side": side,
                    "true_side_gfr": true_side, "true_contralateral_gfr": true_other, "true_total_gfr": total,
                    "true_srf_percent": 100 * true_side / total if total > 0 else np.nan,
                    "true_srf_lt15": int(row[f"{side}_cls_true"]),
                    "predicted_probability_srf_lt15": float(row[f"{side}_cls_prob"]),
                    "serum_creatinine": row["SCr"], "age": row["age"], "sex": row["sex"], "height": row["height"], "weight": row["weight"],
                    "same_side_disease_label": row.get(f"{side}_disease", np.nan),
                    "kidney_volume": row.get(f"{side}_volume_ml", np.nan),
                    "kidney_HU_mean": row.get(f"{side}_hu_mean", np.nan),
                    "z_spacing": row.get("spacing_z", np.nan), "QC_status": row.get("qc_status", "not linked"),
                    "exclusion_reason": row.get("qc_reason", np.nan),
                }
                for key, column in PRED_COLUMNS.items():
                    record[column] = float(row[f"{key}_{side}_pred"])
                denominator = record["predicted_gfr_dfme"] + float(row[f"dfme_{other}_pred"])
                record["predicted_srf_dfme_percent"] = 100 * record["predicted_gfr_dfme"] / denominator if denominator > 0 else np.nan
                all_rows.append(record)
    master = pd.DataFrame(all_rows).sort_values(["split", "anonymized_patient_id", "side"]).reset_index(drop=True)
    if master.duplicated(["anonymized_patient_id", "side"]).any():
        raise ValueError("Master table contains duplicate patient-side rows")
    master.to_csv(case_dir / "locked_case_level_predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(audit_dir / "dfme_equal_weight_verification.csv", index=False)

    definitions = {
        "anonymized_patient_id": ("Deterministic study-specific SHA-256 pseudonym", "none", "derived from case_id"),
        "center": ("Center code", "category", "clinical table"), "split": ("Locked evaluation cohort", "category", "prediction source"),
        "side": ("Kidney laterality", "left/right", "prediction column"), "true_side_gfr": ("Reference side-specific GFR", "mL/min/1.73 m2", "locked prediction truth"),
        "true_contralateral_gfr": ("Contralateral reference GFR", "mL/min/1.73 m2", "locked prediction truth"),
        "true_total_gfr": ("Sum of left and right reference GFR", "mL/min/1.73 m2", "derived"),
        "true_srf_percent": ("100 × side GFR / bilateral total GFR", "%", "derived"), "true_srf_lt15": ("Same-side SRF <15%", "binary", "locked label; formula verified"),
        "predicted_srf_dfme_percent": ("100 × predicted side GFR / predicted bilateral total", "%", "derived from nonnegative DFME GFR"),
        "predicted_probability_srf_lt15": ("Same-side classification-head probability", "0-1", "locked DFME output"),
        "serum_creatinine": ("Serum creatinine", "source unit; presumed µmol/L, author confirmation needed", "clinical table"),
        "age": ("Age", "years", "clinical table"), "sex": ("Sex; 0 male, 1 female per codebook", "binary", "clinical table"),
        "height": ("Height", "m", "clinical table"), "weight": ("Weight", "kg", "clinical table"),
        "same_side_disease_label": ("Kidney-specific disease label", "category", "center disease file"),
        "kidney_volume": ("Mask-derived same-side volume", "mL", "NIfTI audit"), "kidney_HU_mean": ("Same-side mean intensity", "HU-like stored values", "NIfTI/radiomics audit"),
        "z_spacing": ("NIfTI z-axis voxel spacing", "mm", "NIfTI header"), "QC_status": ("Later NIfTI/mask QC status", "category", "NIfTI audit"),
        "exclusion_reason": ("Later QC reason", "text", "NIfTI audit"),
    }
    for key, column in PRED_COLUMNS.items():
        definitions[column] = (f"Nonnegative locked {config['models'][key]['label']} side-GFR prediction", "mL/min/1.73 m2", config["models"][key]["internal"] + " / external counterpart")
    lines = ["# Locked case-level prediction data dictionary", "", "The public-facing table contains pseudonyms only. Original IDs remain in protected source files and are not copied.", "", "| Column | Definition | Unit | Source | Missing n (%) |", "|---|---|---|---|---|"]
    for column in master.columns:
        definition, unit, source = definitions.get(column, (column, "as recorded", "derived/linked"))
        missing = int(master[column].isna().sum())
        lines.append(f"| {column} | {definition} | {unit} | {source} | {missing} ({100*missing/len(master):.1f}%) |")
    (case_dir / "locked_case_level_predictions_data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")

    inputs = [config["inputs"]["clinical_csv"], config["inputs"]["nifti_audit_csv"], *[config["inputs"][k] for k in ("disease_center1_csv", "disease_center2_csv", "disease_center3_csv")]]
    inputs += [spec[split] for spec in config["models"].values() for split in ("internal", "external")]
    write_environment_manifest(config, inputs, "02_build_locked_prediction_table")
    logger.info("Wrote %d patient-side rows (%d patients)", len(master), master["anonymized_patient_id"].nunique())


if __name__ == "__main__":
    main()
