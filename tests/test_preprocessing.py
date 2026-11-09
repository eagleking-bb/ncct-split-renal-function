"""Preprocessing: NaN handling, train-only statistics, serialisable state."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.preprocessing import (
    PREPROCESSING_ORDER,
    PreprocessingStateError,
    TabularPreprocessor,
    active_modalities_for_tag,
    build_development_bundle,
    mask_inactive_modalities,
)

RAD = [f"Left_f{i}" for i in range(3)] + [f"Right_f{i}" for i in range(3)]
CLIN = ["height", "weight", "sex", "age", "SCr"]


def synthetic_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array(["Center_1", "Center_2"] * (n // 2))
    frame = pd.DataFrame(
        {
            "ID": [f"P{i:03d}" for i in range(n)],
            "Center": centers,
            "height": rng.normal(165, 8, n),
            "weight": rng.normal(62, 10, n),
            "sex": rng.integers(0, 2, n).astype(float),
            "age": rng.normal(55, 12, n),
            "SCr": rng.normal(90, 25, n),
            "left_GFR": rng.normal(38, 12, n),
            "right_GFR": rng.normal(38, 12, n),
            "ct_path": [f"/tmp/{i}.nii.gz" for i in range(n)],
            "mask_path": [f"/tmp/{i}_mask.nii.gz" for i in range(n)],
        }
    )
    for feature in RAD:
        frame[feature] = rng.normal(0, 1, n)
    # introduce missing values in both blocks
    frame.loc[frame.index[:5], "SCr"] = np.nan
    frame.loc[frame.index[3:8], RAD[0]] = np.nan
    return frame


def test_nan_input_is_imputed_before_scaling():
    frame = synthetic_frame()
    pre = TabularPreprocessor(CLIN, RAD).fit(frame.iloc[:32])
    scaled = pre.transform(frame)  # must not raise on NaN
    assert np.isfinite(scaled[CLIN + RAD].to_numpy(dtype=float)).all()
    assert pre.state()["preprocessing_order"] == PREPROCESSING_ORDER


def test_validation_statistics_do_not_leak():
    frame = synthetic_frame()
    train = frame.iloc[:32]
    held_out = frame.iloc[32:].copy()
    pre = TabularPreprocessor(CLIN, RAD).fit(train)
    before = pre.state()
    pre.transform(held_out)
    after = pre.state()
    # transforming new rows must not change the fitted statistics
    assert before["clinical"] == after["clinical"]
    assert before["radiomics"]["Center_1"] == after["radiomics"]["Center_1"]


def test_state_round_trip():
    frame = synthetic_frame()
    pre = TabularPreprocessor(CLIN, RAD).fit(frame.iloc[:32])
    restored = TabularPreprocessor.from_state(pre.state())
    a = pre.transform(frame.iloc[32:])
    b = restored.transform(frame.iloc[32:])
    pd.testing.assert_frame_equal(a, b)


def test_unsupported_preprocessing_order_is_rejected():
    frame = synthetic_frame()
    pre = TabularPreprocessor(CLIN, RAD).fit(frame.iloc[:32])
    state = pre.state()
    state["preprocessing_order"] = "standardise_then_impute"
    with pytest.raises(PreprocessingStateError):
        TabularPreprocessor.from_state(state)


def test_bundle_split_is_patient_level_and_recorded():
    frame = synthetic_frame()
    bundle = build_development_bundle(
        frame, radiomics_features=RAD, clinical_features=CLIN, test_size=0.25, random_state=7
    )
    split = bundle.state["split"]
    assert split["n_train"] + split["n_validation"] == len(frame)
    assert not set(split["train_ids"]) & set(split["validation_ids"])
    assert len(bundle.train_samples) == split["n_train"]
    assert len(bundle.val_samples) == split["n_validation"]
    for sample in bundle.train_samples:
        assert sample["prior_l"].shape == (len(CLIN) + 3,)
        assert sample["prior_r"].shape == (len(CLIN) + 3,)


def test_unknown_centre_uses_pooled_training_block():
    frame = synthetic_frame()
    pre = TabularPreprocessor(CLIN, RAD).fit(frame.iloc[:32])
    external = frame.iloc[32:].copy()
    external["Center"] = "Center_3"
    scaled = pre.transform(external)
    assert pre.last_transform_notes == {"Center_3": "pooled_training_fallback"}
    assert np.isfinite(scaled[RAD].to_numpy(dtype=float)).all()


def test_ablation_modalities_are_masked_consistently():
    prior = np.ones((2, 8), dtype=np.float32)  # 5 clinical + 3 radiomics
    assert active_modalities_for_tag("concat_clin_only") == ["clinical"]
    assert active_modalities_for_tag("concat_rad_only") == ["radiomics"]
    assert active_modalities_for_tag("vision_only") == []
    assert active_modalities_for_tag("concat_full") == ["clinical", "radiomics"]

    radiomics_zeroed = mask_inactive_modalities(prior, 5, ["clinical"])
    assert np.all(radiomics_zeroed[:, :5] == 1.0)
    assert np.all(radiomics_zeroed[:, 5:] == 0.0)

    all_zeroed = mask_inactive_modalities(prior, 5, [])
    assert np.all(all_zeroed == 0.0)

    untouched = mask_inactive_modalities(prior, 5, ["clinical", "radiomics"])
    assert np.all(untouched == 1.0)
