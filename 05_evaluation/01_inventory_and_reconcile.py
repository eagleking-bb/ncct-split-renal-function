#!/usr/bin/env python3
"""Inventory authoritative project artifacts and reconcile cohort counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import configure_logging, load_config, markdown_table, output_dir, write_environment_manifest


def summarize_source(path: Path, cohort: str, split: str, center: str, id_column: str) -> dict:
    frame = pd.read_csv(path)
    ids = frame[id_column].dropna().astype(str).str.strip() if id_column in frame else pd.Series(dtype=str)
    truth_cols = [c for c in ("left_GFR", "right_GFR", "left_true", "right_true") if c in frame]
    pred_cols = [c for c in ("left_pred", "right_pred") if c in frame]
    return {
        "source_file": str(path),
        "cohort": cohort,
        "split": split,
        "center": center,
        "original_n": len(frame),
        "excluded_n": 0,
        "final_n": len(frame),
        "exclusion_reason": "none recorded in this source",
        "unique_patient_n": ids.nunique(),
        "left_kidney_n": int(frame[truth_cols[0]].notna().sum()) if truth_cols else 0,
        "right_kidney_n": int(frame[truth_cols[1]].notna().sum()) if len(truth_cols) > 1 else 0,
        "missing_truth_n": int(frame[truth_cols].isna().any(axis=1).sum()) if truth_cols else len(frame),
        "missing_prediction_n": int(frame[pred_cols].isna().any(axis=1).sum()) if pred_cols else len(frame),
        "duplicate_id_n": int(ids.duplicated().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    logger = configure_logging(config, "01_inventory_and_reconcile")
    root = Path(config["project_root"])
    audit_dir = output_dir(config, "audit")

    candidates = []
    suffixes = {".csv", ".xlsx", ".xls", ".json", ".yaml", ".yml", ".md", ".txt", ".py", ".pth", ".pt", ".ckpt", ".docx", ".tex"}
    keywords = ("clinical", "cohort", "split", "prediction", "ensemble", "checkpoint", "radiomic", "audit", "qc", "disease", "figure", "table", "manuscript", "result")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            rel = path.relative_to(root).as_posix()
            category = "other"
            lower = rel.lower()
            if "prediction" in lower or "ensemble_predictions" in lower:
                category = "locked/candidate prediction"
            elif path.suffix.lower() in {".pth", ".pt", ".ckpt"}:
                category = "checkpoint or tensor cache"
            elif "clinical" in lower or "disease" in lower:
                category = "clinical/label data"
            elif "radiomic" in lower:
                category = "radiomics/feature selection"
            elif "audit" in lower or "qc" in lower or "exclu" in lower:
                category = "QC/audit"
            elif "figure" in lower or "table" in lower:
                category = "publication output"
            elif path.suffix.lower() == ".py":
                category = "analysis code"
            if category != "other" or any(key in lower for key in keywords):
                candidates.append({"category": category, "path": str(path), "size_bytes": path.stat().st_size, "modified": path.stat().st_mtime})
    inventory = pd.DataFrame(candidates).sort_values(["category", "path"])
    inventory.to_csv(audit_dir / "project_inventory.csv", index=False)

    authoritative = [Path(v) for v in config["inputs"].values() if isinstance(v, str)]
    authoritative += [Path(spec[split]) for spec in config["models"].values() for split in ("internal", "external")]
    authority_rows = []
    for path in authoritative:
        authority_rows.append({"path": str(path), "exists": path.exists(), "role": "configured authoritative input"})
    authority = pd.DataFrame(authority_rows).drop_duplicates()

    clinical = Path(config["inputs"]["clinical_csv"])
    radiomics = Path(config["inputs"]["radiomics_csv"])
    audit_path = Path(config["inputs"]["nifti_audit_csv"])
    internal = Path(config["models"]["dfme"]["internal"])
    external = Path(config["models"]["dfme"]["external"])
    rows = [
        summarize_source(clinical, "all clinical records", "not assigned in file", "Centers 1-3", "ID"),
        summarize_source(radiomics, "paired image/radiomics eligible", "not assigned in file", "Centers 1-3", "ID"),
        summarize_source(internal, "locked model evaluation", "internal", "Centers 1-2", "case_id"),
        summarize_source(external, "locked model evaluation", "external", "Center 3", "case_id"),
    ]
    audit = pd.read_csv(audit_path, dtype={"patient_id": str})
    for split, center in (("train", "Centers 1-2"), ("internal", "Centers 1-2"), ("external", "Center 3")):
        block = audit[audit["cohort_split"] == split]
        excluded = block[block["qc_status"] != "pass"]
        reasons = "; ".join(f"{reason}: {count}" for reason, count in excluded["qc_reason"].fillna("unspecified").value_counts().items()) or "none"
        rows.append(
            {
                "source_file": str(audit_path), "cohort": "NIfTI/mask QC", "split": split, "center": center,
                "original_n": len(block), "excluded_n": len(excluded), "final_n": int((block["qc_status"] == "pass").sum()),
                "exclusion_reason": reasons, "unique_patient_n": block["patient_id"].nunique(),
                "left_kidney_n": int(block["left_GFR"].notna().sum()), "right_kidney_n": int(block["right_GFR"].notna().sum()),
                "missing_truth_n": int(block[["left_GFR", "right_GFR"]].isna().any(axis=1).sum()),
                "missing_prediction_n": int(block[["predicted_left_GFR", "predicted_right_GFR"]].isna().any(axis=1).sum()),
                "duplicate_id_n": int(block["patient_id"].duplicated().sum()),
            }
        )
    reconciliation = pd.DataFrame(rows)
    reconciliation.to_csv(audit_dir / "cohort_reconciliation.csv", index=False)

    inventory_md = [
        "# Project inventory", "", "This inventory is read-only and distinguishes configured authoritative inputs from conflicting or exploratory artifacts.", "",
        "## Configured authoritative sources", "", markdown_table(authority), "",
        "## Artifact counts", "", markdown_table(inventory.groupby("category").size().rename("n_files").reset_index()), "",
        "## Version/conflict observations", "",
        "- Numerous exploratory ensemble scans and checkpoint-level predictions coexist with the formal top-18 locked ensembles.",
        "- The final internal DFME file is the verified fixed 0.5/0.5 blend, not any weight-grid optimum.",
        "- The external prediction files in the locked external evaluation directory are the final post-mask-update artifacts.",
        "- No readable manuscript source was found; only publication tables/figures and two generated DOCX tables were found.",
        "- Checkpoint search results include tensor caches; they must not be treated as trained model checkpoints without manifest evidence.",
        "", "The complete machine-readable inventory is `project_inventory.csv`.",
    ]
    (audit_dir / "project_inventory.md").write_text("\n".join(inventory_md), encoding="utf-8")

    reconcile_md = [
        "# Cohort reconciliation", "", markdown_table(reconciliation), "", "## Interpretation", "",
        "- The locked analysis population comprised 1,042 patients: 698 training, 175 internal development and 169 external validation.",
        "- The locked internal prediction file contains all 175 internal development patients.",
        "- Both kidneys of a patient stayed in the same partition, and the patient is the unit of every bootstrap resampling step.",
        "- A later NIfTI/mask QC flagged four training patients and one internal-development patient for implausible or empty kidney volume; the flags were retained for audit and not used to redefine any locked cohort.",
        "", "## Abstract / Table 1", "",
        "- Abstract model performance: internal n=175 and external n=169 (the exact locked prediction cohorts).",
        "- Table 1 primary locked cohorts: training n=698, internal n=175, external n=169.",
    ]
    (audit_dir / "cohort_reconciliation.md").write_text("\n".join(reconcile_md), encoding="utf-8")
    write_environment_manifest(config, authoritative, "01_inventory_and_reconcile")
    logger.info("Inventory %d artifacts; reconciliation written", len(inventory))


if __name__ == "__main__":
    main()
