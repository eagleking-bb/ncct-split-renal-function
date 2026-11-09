"""Bootstrap p values must come from a null-centred distribution."""
from __future__ import annotations

import numpy as np
import pytest

from bootstrap_common import bootstrap_share_above_zero, null_centred_bootstrap_p


def test_identical_models_give_p_one():
    replicates = np.zeros(1000)
    assert null_centred_bootstrap_p(replicates, 0.0) == 1.0


def test_zero_difference_with_noise_gives_p_one():
    rng = np.random.default_rng(0)
    replicates = rng.normal(0, 0.01, 500)
    assert null_centred_bootstrap_p(replicates, 0.0) == pytest.approx(1.0)


def test_large_consistent_difference_is_significant():
    rng = np.random.default_rng(1)
    replicates = rng.normal(0.5, 0.02, 2000)
    p = null_centred_bootstrap_p(replicates, 0.5)
    assert p < 0.01


def test_centred_around_zero_with_wide_spread_is_not_significant():
    rng = np.random.default_rng(2)
    replicates = rng.normal(0.1, 0.5, 2000)
    observed = 0.1
    p = null_centred_bootstrap_p(replicates, observed)
    assert 0.0 < p < 1.0
    assert p > 0.5


def test_add_one_smoothing_bounds():
    replicates = np.array([1.0, 2.0, 3.0, 4.0])
    p = null_centred_bootstrap_p(replicates, 2.5)
    assert 0.0 < p <= 1.0
    assert p >= 1 / 5


def test_share_above_zero_is_descriptive_only():
    assert bootstrap_share_above_zero(np.array([-1.0, 1.0, 2.0])) == pytest.approx(2 / 3)


def test_nan_handling():
    p = null_centred_bootstrap_p(np.array([np.nan, 1.0, 1.1]), 1.05)
    assert np.isfinite(p)
    assert np.isnan(null_centred_bootstrap_p(np.array([]), 1.0))
