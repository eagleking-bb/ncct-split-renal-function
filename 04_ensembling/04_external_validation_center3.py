#!/usr/bin/env python3
"""External validation in Centre 3 on the locked 169-patient cohort.

This script reproduces Table 2B and the external panels of Figure 5.  It reads
the per-model inference outputs that were written once, after the models had
been frozen, and it never re-fits, re-weights, recalibrates or re-thresholds
anything.

Metrics
-------
Left- and right-kidney observations are pooled before CCC, MAE, RMSE and bias
are computed; side-specific metrics are never averaged.  AUROC and AUPRC use
the kidney-level reference label for SRF < 15% and the direct classification
probability.  Confidence intervals are two-sided 95% patient-level percentile
bootstrap intervals (5,000 replicates, seed 2026), with both kidneys of a
patient always resampled together and the same resampling indices shared by
all models.  The cohort is asserted to be exactly 169 patients, 338 kidneys
and 28 positive kidneys.

Usage
-----
    python 04_external_validation_center3.py \
        --predictions-dir /path/to/external_center3_eval \
        --output-dir results/external169
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ID_SALT = "DFME-final-locked-analysis::"
ID_PREFIX = "DFME_"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026
EXPECTED_PATIENTS = 169
EXPECTED_KIDNEYS = 338
EXPECTED_POSITIVES = 28

# model_key -> (sub-directory of the inference run, label used in Table 2B)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "clinical_only": ("clinical_ridge_logistic", "Clinical-only"),
    "radiomics_only": ("radiomics_random_forest", "Radiomics-only"),
    "ncct_only": ("vision_only", "NCCT-only"),
    "ncct_clinical": ("concat_clin_only", "NCCT + clinical"),
    "ncct_radiomics": ("concat_rad_only", "NCCT + radiomics"),
    "early_concat": ("concat_full", "Early concatenation"),
    "cross_attention": ("dual_attention_full", "Dual-branch cross-attention"),
    "dfme": ("fixed_equal_weight_concat_dual", "DFME"),
}
ENSEMBLE_COMPONENTS = ("early_concat", "cross_attention")
ENSEMBLE_WEIGHTS = (0.5, 0.5)


def anonymize_id(value: object) -> str:
    token = f"{ID_SALT}{str(value).strip()}".encode()
    return ID_PREFIX + hashlib.sha256(token).hexdigest()[:16]


def ccc(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (population moments)."""
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mean_ref, mean_pred = reference.mean(), prediction.mean()
    var_ref, var_pred = reference.var(ddof=0), prediction.var(ddof=0)
    covariance = np.mean((reference - mean_ref) * (prediction - mean_pred))
    denominator = var_ref + var_pred + (mean_ref - mean_pred) ** 2
    return float(2.0 * covariance / denominator) if denominator else float("nan")


def load_model(predictions_dir: Path, subdir: str) -> pd.DataFrame:
    path = predictions_dir / subdir / "external_predictions.csv"
    if not path.is_file():
        raise SystemExit(f"missing inference output: {path}")
    frame = pd.read_csv(path)
    required = {"case_id", "left_true", "right_true", "left_pred", "right_pred",
                "left_cls_true", "right_cls_true", "left_cls_prob", "right_cls_prob"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["anonymized_patient_id"] = frame.case_id.map(anonymize_id)
    if frame.anonymized_patient_id.duplicated().any():
        raise SystemExit(f"{path} has duplicate patients")
    return frame


def to_arrays(frame: pd.DataFrame, patient_index: pd.Index) -> dict[str, np.ndarray]:
    """Return (patients x 2) arrays ordered by patient_index; column 0 is left."""
    frame = frame.set_index("anonymized_patient_id").loc[patient_index]
    raw_columns = ("left_pred_raw", "right_pred_raw")
    has_raw = all(column in frame.columns for column in raw_columns)
    arrays = {}
    for name, columns in (
        ("reference", ("left_true", "right_true")),
        ("prediction", ("left_pred", "right_pred")),
        ("raw_prediction", raw_columns if has_raw else ("left_pred", "right_pred")),
        ("label", ("left_cls_true", "right_cls_true")),
        ("probability", ("left_cls_prob", "right_cls_prob")),
    ):
        arrays[name] = frame.loc[:, list(columns)].to_numpy(dtype=float)
    return arrays


def point_metrics(reference: np.ndarray, prediction: np.ndarray,
                  label: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    reference_flat = reference.ravel()
    prediction_flat = prediction.ravel()
    label_flat = label.ravel()
    probability_flat = probability.ravel()
    error = prediction_flat - reference_flat
    return {
        "n_patients": int(reference.shape[0]),
        "n_kidneys": int(reference_flat.size),
        "n_positive_kidneys": int(label_flat.sum()),
        "ccc": ccc(reference_flat, prediction_flat),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "pearson_r": float(np.corrcoef(reference_flat, prediction_flat)[0, 1]),
        "auroc": float(roc_auc_score(label_flat, probability_flat)),
        "auprc": float(average_precision_score(label_flat, probability_flat)),
    }


def bootstrap_metrics(models: dict[str, dict[str, np.ndarray]], n_patients: int,
                      replicates: int, seed: int) -> dict[str, dict[str, tuple[float, float]]]:
    # The locked analysis used the legacy numpy RandomState stream with seed
    # 2026; the same patient-level indices are shared by every model.
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_patients, size=(replicates, n_patients))
    names = ("ccc", "mae", "rmse", "auroc")
    collected = {key: {name: np.empty(replicates) for name in names} for key in models}
    for replicate in range(replicates):
        draw = indices[replicate]
        for key, arrays in models.items():
            reference = arrays["reference"][draw].ravel()
            prediction = arrays["prediction"][draw].ravel()
            label = arrays["label"][draw].ravel()
            probability = arrays["probability"][draw].ravel()
            error = prediction - reference
            collected[key]["ccc"][replicate] = ccc(reference, prediction)
            collected[key]["mae"][replicate] = np.mean(np.abs(error))
            collected[key]["rmse"][replicate] = np.sqrt(np.mean(error ** 2))
            collected[key]["auroc"][replicate] = (
                roc_auc_score(label, probability) if label.min() != label.max() else np.nan
            )
    intervals = {}
    for key in models:
        intervals[key] = {}
        for name in names:
            values = collected[key][name]
            values = values[np.isfinite(values)]
            intervals[key][name] = (float(np.percentile(values, 2.5)),
                                    float(np.percentile(values, 97.5)))
    return intervals


def verify_ensemble(models: dict[str, dict[str, np.ndarray]]) -> float:
    """DFME must be the fixed 0.5:0.5 average of its components, clipped at zero.

    The non-negative constraint of the final layer is applied once, to the
    averaged prediction, not to each component separately.
    """
    averaged = sum(weight * models[key]["raw_prediction"]
                   for key, weight in zip(ENSEMBLE_COMPONENTS, ENSEMBLE_WEIGHTS))
    expected = np.maximum(averaged, 0.0)
    maximum = float(np.nanmax(np.abs(models["dfme"]["prediction"] - expected)))
    if maximum > 1e-3:
        raise SystemExit(
            "the locked ensemble is not the prespecified 0.5:0.5 average "
            f"(maximum absolute difference {maximum:.4f})"
        )
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True, type=Path,
                        help="directory holding one sub-directory per model, each with "
                             "external_predictions.csv written by the frozen model")
    parser.add_argument("--external-case-list", type=Path,
                        help="optional CSV of the case identifiers that form the locked external "
                             "cohort; supply it when the inference directory holds more cases")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = {key: load_model(args.predictions_dir, subdir)
              for key, (subdir, _) in MODEL_REGISTRY.items()}

    shared = set.intersection(*(set(frame.anonymized_patient_id) for frame in frames.values()))
    if not shared:
        raise SystemExit("the model outputs share no patients")

    if args.external_case_list is not None:
        case_list = pd.read_csv(args.external_case_list)
        column = "anonymized_patient_id" if "anonymized_patient_id" in case_list.columns else "case_id"
        values = case_list[column].astype(str)
        included = set(values.map(lambda v: v if v.startswith(ID_PREFIX) else anonymize_id(v)))
    else:
        included = set(shared)
    patient_index = pd.Index(sorted(included & shared))

    if patient_index.size != EXPECTED_PATIENTS:
        raise SystemExit(f"expected {EXPECTED_PATIENTS} patients, got {patient_index.size}")

    models = {key: to_arrays(frame, patient_index) for key, frame in frames.items()}

    n_kidneys = int(models["dfme"]["reference"].size)
    n_positive = int(models["dfme"]["label"].sum())
    if n_kidneys != EXPECTED_KIDNEYS:
        raise SystemExit(f"expected {EXPECTED_KIDNEYS} kidney observations, got {n_kidneys}")
    if n_positive != EXPECTED_POSITIVES:
        raise SystemExit(f"expected {EXPECTED_POSITIVES} positive kidneys, got {n_positive}")

    maximum_ensemble_difference = verify_ensemble(models)

    intervals = bootstrap_metrics(models, patient_index.size, args.bootstrap, args.seed)

    rows = []
    for key, (_, label) in MODEL_REGISTRY.items():
        metrics = point_metrics(models[key]["reference"], models[key]["prediction"],
                                models[key]["label"], models[key]["probability"])
        for name, (low, high) in intervals[key].items():
            metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"] = low, high
        metrics["model_key"] = key
        metrics["model"] = label
        metrics["analysis_set"] = "external_center3_locked_169"
        rows.append(metrics)
    column_order = ["model_key", "model", "n_patients", "n_kidneys", "n_positive_kidneys",
                    "ccc", "ccc_ci_low", "ccc_ci_high", "mae", "mae_ci_low", "mae_ci_high",
                    "rmse", "rmse_ci_low", "rmse_ci_high", "bias", "pearson_r",
                    "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "analysis_set"]
    table = pd.DataFrame(rows)[column_order]
    metrics_path = args.output_dir / "external169_metrics.csv"
    table.to_csv(metrics_path, index=False)

    # de-identified kidney-level table for the external cohort
    long_rows = []
    for key in MODEL_REGISTRY:
        arrays = models[key]
        for position, patient in enumerate(patient_index):
            for side_index, side in enumerate(("left", "right")):
                long_rows.append({
                    "anonymized_patient_id": patient,
                    "model_key": key,
                    "side": side,
                    "true_gfr": arrays["reference"][position, side_index],
                    "true_srf_lt15": int(arrays["label"][position, side_index]),
                    "predicted_gfr": arrays["prediction"][position, side_index],
                    "predicted_probability_srf_lt15": arrays["probability"][position, side_index],
                })
    pd.DataFrame(long_rows).to_csv(args.output_dir / "external169_kidney_predictions.csv", index=False)

    manifest = {
        "cohort": "Centre 3 locked external validation",
        "n_patients": int(patient_index.size),
        "n_kidneys": n_kidneys,
        "n_positive_kidneys": n_positive,
        "models": {key: label for key, (_, label) in MODEL_REGISTRY.items()},
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed, "unit": "patient",
                      "shared_indices_across_models": True, "interval": "percentile"},
        "ensemble_check": {"weights": dict(zip(ENSEMBLE_COMPONENTS, ENSEMBLE_WEIGHTS)),
                           "max_abs_difference": maximum_ensemble_difference},
        "refitting": False,
        "recalibration": False,
        "threshold_optimisation": False,
    }
    (args.output_dir / "external169_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    display = table[["model", "ccc", "mae", "rmse", "auroc"]].copy()
    for column, digits in (("ccc", 3), ("mae", 2), ("rmse", 2), ("auroc", 3)):
        display[column] = display[column].map(lambda value, d=digits: f"{value:.{d}f}")
    print(f"locked external cohort: {patient_index.size} patients / {n_kidneys} kidneys / "
          f"{n_positive} positive kidneys")
    print(f"ensemble check passed (max |difference| = {maximum_ensemble_difference:.2e})")
    print(display.to_string(index=False))
    print(f"written: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
