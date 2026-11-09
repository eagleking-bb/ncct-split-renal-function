"""Unit contracts for GFR labels, targets and tabular external outputs.

The original implementation applied the tabular preprocessor to left/right GFR
before deriving SRF labels and before fitting the MLP target scaler.  That
standardised the target twice and changed the label definition.  These tests
use fixed, hand-checkable GFR values so a regression cannot hide behind real
patient data.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.preprocessing import TabularPreprocessor, build_development_bundle

ROOT = Path(__file__).resolve().parents[1]
RAD = [f"Left_f{i}" for i in range(3)] + [f"Right_f{i}" for i in range(3)]
CLIN = ["height", "weight", "sex", "age", "SCr"]
GFR_UNIT = "mL/min/1.73 m2"


def fixed_frame() -> pd.DataFrame:
    rows = []
    # Alternating high/low function gives labels that are obvious by hand:
    # 10/(10+90)=0.10 -> left SRF<15%; 60/(60+40)=0.60 -> neither side.
    gfr_pairs = [(60.0, 40.0), (10.0, 90.0)] * 5
    for index, (left, right) in enumerate(gfr_pairs):
        row = {
            "ID": f"P{index:03d}",
            "Center": "Center_1" if index % 2 == 0 else "Center_2",
            "height": 165.0 + index,
            "weight": 60.0 + index,
            "sex": float(index % 2),
            "age": 50.0 + index,
            "SCr": 80.0 + index,
            "left_GFR": left,
            "right_GFR": right,
            "ct_path": f"/tmp/{index}.nii.gz",
            "mask_path": f"/tmp/{index}_mask.nii.gz",
        }
        for feature in RAD:
            row[feature] = float(index + 1) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_raw_gfr_labels_originals_and_single_target_standardisation():
    frame = fixed_frame()
    bundle = build_development_bundle(
        frame,
        radiomics_features=RAD,
        clinical_features=CLIN,
        test_size=0.2,
        random_state=42,
    )
    raw = frame.set_index("ID")
    stats = bundle.state["preprocessing"]["gfr"]

    for sample in bundle.train_samples + bundle.val_samples:
        patient = sample["case_id"]
        left_raw = float(raw.loc[patient, "left_GFR"])
        right_raw = float(raw.loc[patient, "right_GFR"])

        assert sample["left_gfr_original"] == pytest.approx(left_raw)
        assert sample["right_gfr_original"] == pytest.approx(right_raw)

        left_srf = left_raw / (left_raw + right_raw)
        right_srf = right_raw / (left_raw + right_raw)
        assert sample["label_cls_l"][0] == pytest.approx(float(left_srf < 0.15))
        assert sample["label_cls_r"][0] == pytest.approx(float(right_srf < 0.15))

        left_mean = float(stats["left_GFR"]["mean"])
        left_scale = float(stats["left_GFR"]["scale"])
        right_mean = float(stats["right_GFR"]["mean"])
        right_scale = float(stats["right_GFR"]["scale"])
        expected_left_z = (left_raw - left_mean) / left_scale
        expected_right_z = (right_raw - right_mean) / right_scale
        assert sample["left_gfr"][0] == pytest.approx(expected_left_z, rel=1e-6, abs=1e-6)
        assert sample["right_gfr"][0] == pytest.approx(expected_right_z, rel=1e-6, abs=1e-6)

        # One inverse transform must recover the exact clinical value.
        recovered_left = sample["left_gfr"][0] * left_scale + left_mean
        recovered_right = sample["right_gfr"][0] * right_scale + right_mean
        assert recovered_left == pytest.approx(left_raw, rel=1e-6, abs=1e-6)
        assert recovered_right == pytest.approx(right_raw, rel=1e-6, abs=1e-6)


def test_bundle_state_declares_gfr_contract():
    bundle = build_development_bundle(
        fixed_frame(), radiomics_features=RAD, clinical_features=CLIN
    )
    contract = bundle.state["preprocessing"]["gfr_target_contract"]
    assert contract["unit"] == GFR_UNIT
    assert "raw" in contract["labels_and_original_values"]
    assert "once" in contract["model_target"]


# ---------------------------------------------------------------------------
# Tabular baseline and external-inference contracts require torch/joblib.
# ---------------------------------------------------------------------------
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_labels_ignore_standardised_columns_when_original_columns_exist():
    pytest.importorskip("torch")
    module = _load(
        ROOT / "03_training" / "09_train_clinical_and_radiomics_tabular_baselines.py",
        "tabular_baselines_gfr_test",
    )
    frame = pd.DataFrame(
        {
            "left_GFR": [-2.0, 3.0],          # deliberately wrong standardised values
            "right_GFR": [2.0, -3.0],
            "left_GFR_original": [10.0, 60.0],
            "right_GFR_original": [90.0, 40.0],
        }
    )
    y_reg, y_cls, total = module.labels_from_gfr(frame)
    assert np.allclose(y_reg, [[10.0, 90.0], [60.0, 40.0]])
    assert y_cls.tolist() == [[1, 0], [0, 0]]
    assert np.allclose(total[:1], [100.0], atol=1e-6)


class FixedGFRRegressor:
    """Predict fixed raw GFR values, independent of the input features."""

    def predict(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 55.0), np.full(n, 45.0)])


def test_tabular_external_outputs_are_in_original_gfr_units(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    joblib = pytest.importorskip("joblib")
    inference = _load(
        ROOT / "04_ensembling" / "05_external_inference_from_checkpoints.py",
        "external_inference_gfr_test",
    )
    frame = fixed_frame()
    pre = TabularPreprocessor(
        CLIN, RAD, gfr_features=["left_GFR", "right_GFR"]
    ).fit(frame[frame["Center"].isin(["Center_1", "Center_2"])])
    external = frame.iloc[:2].copy()
    external["Center"] = "Center_3"

    bundle = {
        "model_name": "fixed_gfr",
        "estimators": {
            "regressor_layout": "both_sides_same_features",
            "regressor": FixedGFRRegressor(),
            "classifier_left": None,
            "classifier_right": None,
        },
        "clinical_features": CLIN,
        "radiomics_features": RAD,
        "preprocessing": pre.state(),
        "gfr_target": {
            "unit": GFR_UNIT,
            "regression_estimators": "fitted on original GFR values",
        },
    }
    bundle_path = tmp_path / "fixed_gfr_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    result = inference.predict_tabular(bundle_path, external, None)
    assert result["target_unit"] == GFR_UNIT
    assert np.allclose(result["left_pred_raw"], [55.0, 55.0])
    assert np.allclose(result["right_pred_raw"], [45.0, 45.0])
    assert np.allclose(result["left_pred"], [55.0, 55.0])
    assert np.allclose(result["right_pred"], [45.0, 45.0])


def test_old_tabular_bundle_with_double_scaled_target_is_rejected(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    joblib = pytest.importorskip("joblib")
    inference = _load(
        ROOT / "04_ensembling" / "05_external_inference_from_checkpoints.py",
        "external_inference_gfr_legacy_test",
    )
    frame = fixed_frame()
    pre = TabularPreprocessor(
        CLIN, RAD, gfr_features=["left_GFR", "right_GFR"]
    ).fit(frame[frame["Center"].isin(["Center_1", "Center_2"])])
    bundle = {
        "model_name": "legacy",
        "estimators": {},
        "clinical_features": CLIN,
        "radiomics_features": RAD,
        "preprocessing": pre.state(),
        "gfr_target_scaling": "standardised with the saved training statistics",
    }
    path = tmp_path / "legacy.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(inference.InferenceError, match="retrain"):
        inference.predict_tabular(path, frame.iloc[:2].copy(), None)


def test_deep_output_denormalisation_returns_original_units():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    inference = _load(
        ROOT / "04_ensembling" / "05_external_inference_from_checkpoints.py",
        "external_inference_denormalise_test",
    )
    stats = {"mean": 40.0, "scale": 12.0}
    restored = inference.denormalise(np.array([0.0, 1.0, -1.0]), stats)
    assert np.allclose(restored, [40.0, 52.0, 28.0])
