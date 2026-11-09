#!/usr/bin/env python3
"""Generate external-cohort predictions from trained checkpoints.

This is the missing link between training and
``04_external_validation_center3.py``: the evaluator expects one
``external_predictions.csv`` per model, but nothing in the released package
could build those files from a checkpoint.  This script does exactly that, and
only that:

1. load each checkpoint and its ``inference_state`` (preprocessing statistics,
   image pipeline, model definition, patient split);
2. transform the external cohort with the **training** statistics - nothing is
   re-fitted on the external centre;
3. run the frozen model and write
   ``<subdir>/external_predictions.csv`` with ``left_pred_raw`` /
   ``right_pred_raw`` plus the non-negative constrained predictions;
4. build the DFME prediction by averaging the two component **raw** GFR values
   with the prespecified 0.5:0.5 weights and clipping the average at zero;
5. (optionally) apply the saved tabular estimator bundles for the clinical and
   radiomics baselines.

A checkpoint written before the ``inference_state`` schema is rejected: the
imputation and scaling parameters cannot be recovered from it, and guessing
them would invalidate the locked external evaluation.  Supply a compatible
state explicitly with ``--preprocessing-state`` only if you have
documentation for how that model was fitted.

Example
-------
    python 04_ensembling/05_external_inference_from_checkpoints.py \
        --manifest config/external_inference.json \
        --output-dir results/external_center3_eval
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "03_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from common.checkpoint import MissingInferenceStateError, require_inference_state  # noqa: E402
from common.monai_pipeline import (  # noqa: E402
    get_deterministic_transforms,
    get_val_runtime_transforms,
    PreprocessedImageDataset,
)
from common.preprocessing import (  # noqa: E402
    PreprocessingStateError,
    TabularPreprocessor,
    coerce_numeric,
    mask_inactive_modalities,
)

OUTPUT_COLUMNS = [
    "case_id",
    "left_true",
    "right_true",
    "left_pred",
    "right_pred",
    "left_pred_raw",
    "right_pred_raw",
    "left_cls_true",
    "right_cls_true",
    "left_cls_prob",
    "right_cls_prob",
]
DFME_KEY = "dfme"
DFME_COMPONENTS = ("early_concat", "cross_attention")
DFME_WEIGHTS = (0.5, 0.5)


class InferenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_value, base: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_external_frame(manifest: dict, base: Path, limit: int | None = None) -> pd.DataFrame:
    clinical = pd.read_csv(resolve(manifest["clinical_csv"], base))
    radiomics = pd.read_csv(resolve(manifest["radiomics_csv"], base))
    center_column = manifest.get("center_column", "Center")
    external_center = manifest["external_center"]
    frame = pd.merge(clinical, radiomics, on="ID", how="inner")
    frame = frame[frame[center_column].astype(str) == str(external_center)].copy()
    if frame.empty:
        raise InferenceError(f"no rows for external centre {external_center!r}")

    image_dirs = manifest.get("image_dirs", {}).get(external_center)
    if image_dirs:
        ct_dir = resolve(image_dirs["ct"], base)
        mask_dir = resolve(image_dirs["mask"], base)
        rows = []
        for _, row in frame.iterrows():
            patient = str(row["ID"]).strip()
            for suffix in (".nii.gz", ".nii"):
                ct_path = ct_dir / f"{patient}{suffix}"
                mask_path = mask_dir / f"{patient}{suffix}"
                if ct_path.is_file() and mask_path.is_file():
                    rows.append((str(ct_path), str(mask_path)))
                    break
            else:
                raise InferenceError(f"missing CT/mask pair for {patient}")
        frame["ct_path"] = [item[0] for item in rows]
        frame["mask_path"] = [item[1] for item in rows]

    case_list = manifest.get("case_list")
    if case_list:
        wanted = set(pd.read_csv(resolve(case_list, base))["case_id"].astype(str))
        frame = frame[frame["ID"].astype(str).isin(wanted)].copy()

    frame = frame.sort_values("ID").reset_index(drop=True)
    if limit:
        frame = frame.head(int(limit)).reset_index(drop=True)
    if frame["ID"].astype(str).duplicated().any():
        raise InferenceError("external cohort contains duplicate patient IDs")
    return frame


def load_fitted_preprocessor(state: dict) -> TabularPreprocessor:
    return TabularPreprocessor.from_state(state["preprocessing"])


def center_stats_override(manifest: dict, state: dict, base: Path) -> dict | None:
    """Optional explicit radiomics statistics for a centre without fitted values."""
    explicit = manifest.get("external_center_radiomics_stats")
    if explicit:
        payload = json.loads(resolve(explicit, base).read_text(encoding="utf-8"))
        return payload
    if manifest.get("radiomics_normalisation") == "center_stats_file" and not explicit:
        raise InferenceError(
            "radiomics_normalisation='center_stats_file' requires "
            "'external_center_radiomics_stats'"
        )
    return None


def build_tabular_inputs(frame: pd.DataFrame, state: dict, override: dict | None):
    preprocessor = load_fitted_preprocessor(state)
    rad_features = list(state["rad_features"])
    missing = [feature for feature in rad_features if feature not in frame.columns]
    if missing:
        raise InferenceError(f"radiomics CSV is missing checkpoint features: {missing}")
    clinical_features = list(state["clinical_features"])
    missing = [feature for feature in clinical_features if feature not in frame.columns]
    if missing:
        raise InferenceError(f"clinical CSV is missing checkpoint features: {missing}")

    numeric = clinical_features + rad_features + list(state.get("gfr_features", []))
    frame = coerce_numeric(frame, [c for c in numeric if c in frame.columns])
    scaled = preprocessor.transform(frame, unknown_center_stats=override)

    half = len(rad_features) // 2
    prior_l = np.concatenate(
        [scaled[clinical_features].to_numpy(dtype=np.float32),
         scaled[rad_features[:half]].to_numpy(dtype=np.float32)], axis=1
    )
    prior_r = np.concatenate(
        [scaled[clinical_features].to_numpy(dtype=np.float32),
         scaled[rad_features[half:]].to_numpy(dtype=np.float32)], axis=1
    )
    # Reproduce the ablation: a clin_only / rad_only / vision_only checkpoint was
    # trained with the removed block set to zero, so inference must zero it too.
    active = state.get("active_tabular_modalities", ["clinical", "radiomics"])
    prior_l = mask_inactive_modalities(prior_l, len(clinical_features), active)
    prior_r = mask_inactive_modalities(prior_r, len(clinical_features), active)
    return scaled, prior_l, prior_r, preprocessor.last_transform_notes


def load_model_module(code_entrypoint: str, extra_root: Path | None = None):
    """Resolve the training script that defines the checkpoint's model class.

    Repository-relative paths are resolved against the repository first, then
    against the manifest directory (which lets a self-contained inference bundle
    ship its own model code).
    """
    candidate = Path(code_entrypoint)
    path = None
    if candidate.is_absolute():
        path = candidate
    else:
        for root in (REPO_ROOT, extra_root):
            if root is None:
                continue
            possible = (Path(root) / candidate)
            if possible.is_file():
                path = possible.resolve()
                break
    if path is None or not path.is_file():
        raise InferenceError(
            f"model code entrypoint not found: {code_entrypoint!r} "
            f"(searched the repository root and the manifest directory)"
        )
    spec = importlib.util.spec_from_file_location(f"inference_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise InferenceError(f"cannot import model code: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model(state: dict, base_dir: Path | None = None):
    model_state = state["model"]
    module = load_model_module(model_state["code_entrypoint"], base_dir)
    model_class = getattr(module, model_state["class_name"], None)
    if model_class is None:
        raise InferenceError(
            f"{model_state['code_entrypoint']} does not define {model_state['class_name']}"
        )
    model = model_class(**model_state.get("kwargs", {}))
    return model


def denormalise(values: np.ndarray, stats: dict) -> np.ndarray:
    """Undo the training-set z-scoring of the GFR head (no imputation involved)."""
    return np.asarray(values, dtype=float) * float(stats["scale"]) + float(stats["mean"])


def predict_with_checkpoint(
    checkpoint_path: Path,
    frame: pd.DataFrame,
    state_override: dict | None,
    override_stats: dict | None,
    batch_size: int,
    device: str,
    base_dir: Path | None = None,
) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state_override is not None:
        checkpoint["inference_state"] = state_override
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise InferenceError(f"{checkpoint_path} is not a model checkpoint")
    state = require_inference_state(checkpoint, str(checkpoint_path))

    scaled, prior_l, prior_r, notes = build_tabular_inputs(frame, state, override_stats)
    samples = []
    for position, (_, row) in enumerate(scaled.iterrows()):
        samples.append(
            {
                "case_id": row["ID"],
                "ct": row["ct_path"],
                "mask": row["mask_path"],
                "prior_l": prior_l[position],
                "prior_r": prior_r[position],
            }
        )

    target_shape = tuple(state.get("target_shape", (16, 256, 256)))
    dataset = PreprocessedImageDataset(
        samples,
        transform=get_deterministic_transforms(target_shape),
        runtime_transform=get_val_runtime_transforms(),
        cache="none",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(state, base_dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    left_z, right_z, left_p, right_p = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].float().to(device)
            pl = batch["prior_l"].float().to(device)
            pr = batch["prior_r"].float().to(device)
            outputs = model(ct, pl, pr)
            reg_l, reg_r, cls_l, cls_r = outputs[0], outputs[1], outputs[2], outputs[3]
            left_z.append(reg_l.detach().cpu().numpy().reshape(-1))
            right_z.append(reg_r.detach().cpu().numpy().reshape(-1))
            left_p.append(torch.sigmoid(cls_l.detach().cpu()).numpy().reshape(-1))
            right_p.append(torch.sigmoid(cls_r.detach().cpu()).numpy().reshape(-1))

    gfr_stats = state["preprocessing"]["gfr"]
    left_raw = denormalise(np.concatenate(left_z), gfr_stats["left_GFR"])
    right_raw = denormalise(np.concatenate(right_z), gfr_stats["right_GFR"])

    return {
        "case_id": scaled["ID"].astype(str).to_numpy(),
        "left_pred_raw": left_raw,
        "right_pred_raw": right_raw,
        "left_pred": np.clip(left_raw, 0.0, None),
        "right_pred": np.clip(right_raw, 0.0, None),
        "left_cls_prob": np.concatenate(left_p),
        "right_cls_prob": np.concatenate(right_p),
        "normalisation_notes": notes,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "experiment_tag": checkpoint.get("experiment_tag"),
    }


def reference_columns(frame: pd.DataFrame, state: dict) -> dict:
    gfr_features = list(state.get("gfr_features", ["left_GFR", "right_GFR"]))
    left_true = pd.to_numeric(frame[gfr_features[0]], errors="coerce").to_numpy(dtype=float)
    right_true = pd.to_numeric(frame[gfr_features[1]], errors="coerce").to_numpy(dtype=float)
    total = left_true + right_true + 1e-8
    threshold = float(state.get("srf_threshold", 0.15))
    return {
        "left_true": left_true,
        "right_true": right_true,
        "left_cls_true": (left_true / total < threshold).astype(int),
        "right_cls_true": (right_true / total < threshold).astype(int),
    }


def predictions_frame(predictions: dict, reference: dict) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "case_id": predictions["case_id"],
            "left_true": reference["left_true"],
            "right_true": reference["right_true"],
            "left_pred": predictions["left_pred"],
            "right_pred": predictions["right_pred"],
            "left_pred_raw": predictions["left_pred_raw"],
            "right_pred_raw": predictions["right_pred_raw"],
            "left_cls_true": reference["left_cls_true"],
            "right_cls_true": reference["right_cls_true"],
            "left_cls_prob": predictions["left_cls_prob"],
            "right_cls_prob": predictions["right_cls_prob"],
        }
    )
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise InferenceError(f"prediction table is missing columns: {missing}")
    if frame.isna().any().any():
        raise InferenceError("prediction table contains missing values")
    return frame[OUTPUT_COLUMNS]


def combine_dfme(components: dict) -> dict:
    """Prespecified DFME: average RAW GFR, then clip at zero; average probabilities."""
    first = components[DFME_COMPONENTS[0]]
    second = components[DFME_COMPONENTS[1]]
    if not np.array_equal(first["case_id"], second["case_id"]):
        raise InferenceError("DFME components are not aligned by patient")
    left_raw = DFME_WEIGHTS[0] * first["left_pred_raw"] + DFME_WEIGHTS[1] * second["left_pred_raw"]
    right_raw = DFME_WEIGHTS[0] * first["right_pred_raw"] + DFME_WEIGHTS[1] * second["right_pred_raw"]
    return {
        "case_id": first["case_id"],
        "left_pred_raw": left_raw,
        "right_pred_raw": right_raw,
        "left_pred": np.clip(left_raw, 0.0, None),
        "right_pred": np.clip(right_raw, 0.0, None),
        "left_cls_prob": DFME_WEIGHTS[0] * first["left_cls_prob"] + DFME_WEIGHTS[1] * second["left_cls_prob"],
        "right_cls_prob": DFME_WEIGHTS[0] * first["right_cls_prob"] + DFME_WEIGHTS[1] * second["right_cls_prob"],
        "normalisation_notes": {},
        "checkpoint_sha256": None,
        "experiment_tag": DFME_KEY,
    }


def predict_tabular(bundle_path: Path, frame: pd.DataFrame, override_stats: dict | None) -> dict:
    import joblib

    bundle = joblib.load(bundle_path)
    if bundle.get("gfr_target_scaling") == "standardised with the saved training statistics":
        raise InferenceError(
            f"{bundle_path} was trained with the pre-fix GFR target handling "
            "(standardised twice); retrain the tabular baseline before external inference"
        )
    target_contract = bundle.get("gfr_target")
    if not isinstance(target_contract, dict) or target_contract.get("unit") != "mL/min/1.73 m2":
        raise InferenceError(
            f"{bundle_path} does not declare GFR outputs in mL/min/1.73 m2; "
            "retrain the tabular baseline with the current entrypoint"
        )
    state = {"preprocessing": bundle["preprocessing"]}
    preprocessor = TabularPreprocessor.from_state(bundle["preprocessing"])
    clinical_features = list(bundle["clinical_features"])
    rad_features = list(bundle["radiomics_features"])
    estimators = bundle["estimators"]

    numeric = clinical_features + rad_features + ["left_GFR", "right_GFR"]
    frame = coerce_numeric(frame, [c for c in numeric if c in frame.columns])
    scaled = preprocessor.transform(frame, unknown_center_stats=override_stats)

    layout = estimators["regressor_layout"]
    if layout == "both_sides_same_features":
        values = estimators["regressor"].predict(scaled[clinical_features].to_numpy(dtype=float))
        left_raw, right_raw = values[:, 0], values[:, 1]
    elif layout == "side_specific_features":
        half = len(rad_features) // 2
        left_raw = estimators["regressor_left"].predict(scaled[rad_features[:half]].to_numpy(dtype=float))
        right_raw = estimators["regressor_right"].predict(scaled[rad_features[half:]].to_numpy(dtype=float))
    else:
        raise InferenceError(f"unsupported regressor layout: {layout}")

    probs = []
    for side in ("left", "right"):
        model = estimators.get(f"classifier_{side}")
        if model is None:
            probs.append(np.full(len(scaled), np.nan))
            continue
        features = clinical_features if layout == "both_sides_same_features" else (
            rad_features[: len(rad_features) // 2] if side == "left" else rad_features[len(rad_features) // 2:]
        )
        matrix = scaled[features].to_numpy(dtype=float)
        probs.append(model.predict_proba(matrix)[:, 1] if hasattr(model, "predict_proba") else 1.0 / (1.0 + np.exp(-model.decision_function(matrix))))

    left_raw = np.asarray(left_raw, dtype=float)
    right_raw = np.asarray(right_raw, dtype=float)
    if not (np.isfinite(left_raw).all() and np.isfinite(right_raw).all()):
        raise InferenceError(f"{bundle_path} produced non-finite GFR predictions")
    return {
        "case_id": scaled["ID"].astype(str).to_numpy(),
        "left_pred_raw": left_raw,
        "right_pred_raw": right_raw,
        "target_unit": target_contract["unit"],
        "left_pred": np.clip(left_raw, 0.0, None),
        "right_pred": np.clip(right_raw, 0.0, None),
        "left_cls_prob": probs[0],
        "right_cls_prob": probs[1],
        "normalisation_notes": preprocessor.last_transform_notes,
        "checkpoint_sha256": sha256_file(bundle_path),
        "experiment_tag": bundle.get("model_name"),
    }


def write_model_output(output_dir: Path, subdir: str, table: pd.DataFrame, manifest_entry: dict) -> None:
    target = output_dir / subdir
    target.mkdir(parents=True, exist_ok=True)
    path = target / "external_predictions.csv"
    table.to_csv(path, index=False)
    (target / "inference_manifest.json").write_text(
        json.dumps(manifest_entry, indent=2), encoding="utf-8"
    )
    print(f"written: {path} ({len(table)} patients)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model keys to run (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N patients only")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--preprocessing-state", type=Path, default=None,
                        help="explicit JSON preprocessing state for checkpoints that lack one")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    output_dir = (args.output_dir or resolve(manifest["output_dir"], base)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_external_frame(manifest, base, args.limit)
    print(f"external cohort: {len(frame)} patients from {manifest['external_center']}")

    state_override = None
    if args.preprocessing_state:
        state_override = json.loads(args.preprocessing_state.read_text(encoding="utf-8"))

    requested = set(args.models) if args.models else None
    deep_models = manifest.get("deep_models", {})
    tabular_models = manifest.get("tabular_models", {})
    unknown = requested - set(deep_models) - set(tabular_models) if requested else set()
    if unknown:
        raise SystemExit(f"unknown model keys: {sorted(unknown)}")

    run_manifest = {
        "external_center": manifest["external_center"],
        "n_patients": int(len(frame)),
        "cohort": "external validation cohort only; never used for fitting",
        "ensemble": {"components": list(DFME_COMPONENTS), "weights": list(DFME_WEIGHTS),
                     "rule": "average raw GFR, then max(0, .); probabilities averaged directly"},
        "refitting": False,
        "recalibration": False,
        "threshold_optimisation": False,
        "models": {},
    }

    override_stats = center_stats_override(manifest, {}, base)
    deep_predictions = {}
    first_state = None
    for key, entry in deep_models.items():
        if requested and key not in requested and key not in DFME_COMPONENTS:
            continue
        checkpoint_path = resolve(entry["checkpoint"], base)
        predictions = predict_with_checkpoint(
            checkpoint_path, frame, state_override, override_stats,
            args.batch_size, args.device, base,
        )
        if first_state is None:
            first_state = predictions
        deep_predictions[key] = predictions
        reference = reference_columns(frame, _state_of(checkpoint_path, state_override))
        table = predictions_frame(predictions, reference)
        write_model_output(
            output_dir, entry.get("subdir", key), table,
            {
                "model_key": key,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": predictions["checkpoint_sha256"],
                "experiment_tag": predictions["experiment_tag"],
                "preprocessing": "from checkpoint inference_state; no external refitting",
                "radiomics_normalisation_notes": predictions["normalisation_notes"],
                "rows": int(len(table)),
            },
        )
        run_manifest["models"][key] = {"subdir": entry.get("subdir", key),
                                       "checkpoint_sha256": predictions["checkpoint_sha256"]}

    if all(component in deep_predictions for component in DFME_COMPONENTS):
        dfme = combine_dfme(deep_predictions)
        reference = reference_columns(frame, _state_of(
            resolve(deep_models[DFME_COMPONENTS[0]]["checkpoint"], base), state_override))
        table = predictions_frame(dfme, reference)
        write_model_output(
            output_dir, "fixed_equal_weight_concat_dual", table,
            {
                "model_key": DFME_KEY,
                "components": {k: run_manifest["models"][k]["checkpoint_sha256"] for k in DFME_COMPONENTS},
                "weights": dict(zip(DFME_COMPONENTS, DFME_WEIGHTS)),
                "rule": "mean of raw GFR -> max(0, .); probability mean",
                "rows": int(len(table)),
            },
        )
        run_manifest["models"][DFME_KEY] = {"subdir": "fixed_equal_weight_concat_dual"}

    for key, entry in tabular_models.items():
        if requested and key not in requested:
            continue
        bundle_path = resolve(entry["estimator"], base)
        predictions = predict_tabular(bundle_path, frame, override_stats)
        state = json.loads((bundle_path.parent / f"{key}_estimator_manifest.json").read_text(encoding="utf-8")) \
            if (bundle_path.parent / f"{key}_estimator_manifest.json").is_file() else {}
        reference = reference_columns(frame, {"gfr_features": ["left_GFR", "right_GFR"], "srf_threshold": 0.15})
        table = predictions_frame(predictions, reference)
        write_model_output(
            output_dir, entry.get("subdir", key), table,
            {
                "model_key": key,
                "estimator": str(bundle_path),
                "estimator_sha256": predictions["checkpoint_sha256"],
                "preprocessing_order": state.get("preprocessing_order"),
                "preprocessing": "from estimator bundle; no external refitting",
                "rows": int(len(table)),
            },
        )
        run_manifest["models"][key] = {"subdir": entry.get("subdir", key),
                                       "estimator_sha256": predictions["checkpoint_sha256"]}

    (output_dir / "external_inference_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(f"manifest: {output_dir / 'external_inference_manifest.json'}")
    return 0


def _state_of(checkpoint_path: Path, override: dict | None) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if override is not None:
        checkpoint["inference_state"] = override
    return require_inference_state(checkpoint, str(checkpoint_path))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MissingInferenceStateError, PreprocessingStateError, InferenceError) as error:
        raise SystemExit(f"external inference failed: {error}")
