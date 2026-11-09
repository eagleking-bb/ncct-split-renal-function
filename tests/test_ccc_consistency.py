"""The four Lin's CCC implementations must stay mutually consistent.

CCC is defined in the shared evaluation utilities and again in three standalone
plotting/evaluation scripts, because those scripts are meant to run on their own.
Two conventions are in use:

* ``ddof=0`` population moments (the three standalone scripts);
* ``ddof=1`` sample moments (``05_evaluation/utils.py``), the classical Lin
  estimator.

The two coincide exactly when the means of the two vectors are equal, because
the numerator and the variance sum then share the same n/(n-1) factor.  When the
means differ, the squared bias term ``(mean_a - mean_b)**2`` is not rescaled, so
the ddof choice shifts CCC slightly.  On the locked tables the shift is of the
order of 1e-4, i.e. far below the three decimals reported in the manuscript.

These tests pin both facts, so that editing one implementation (or "tidying"
one of them) cannot silently change a reported value.
"""
from __future__ import annotations

import numpy as np


def _implementations():
    from conftest import load_module

    return {
        "05_evaluation/utils.py": load_module(
            "05_evaluation/utils.py", "ccc_utils"
        ).ccc,
        "04_ensembling/04_external_validation_center3.py": load_module(
            "04_ensembling/04_external_validation_center3.py", "ccc_external"
        ).ccc,
        "04_ensembling/03_plot_concat_main_comparison.py": load_module(
            "04_ensembling/03_plot_concat_main_comparison.py", "ccc_concat_plot"
        ).concordance_ccc,
        "04_ensembling/02_plot_internal_architecture_comparison.py": load_module(
            "04_ensembling/02_plot_internal_architecture_comparison.py", "ccc_arch_plot"
        ).concordance_ccc,
    }


def test_ccc_implementations_agree_when_means_are_equal():
    """With no mean difference the ddof conventions cancel exactly."""
    rng = np.random.default_rng(2026)
    implementations = _implementations()
    reference = rng.normal(40.0, 12.0, size=350)
    prediction = 0.8 * reference + rng.normal(0.0, 5.0, size=350)
    prediction += reference.mean() - prediction.mean()

    values = {name: float(fn(reference, prediction)) for name, fn in implementations.items()}
    assert max(values.values()) - min(values.values()) < 1e-12, values


def test_ccc_implementations_differ_only_by_a_negligible_amount():
    """With a systematic offset the ddof choice moves CCC by well under 0.001."""
    rng = np.random.default_rng(2026)
    implementations = _implementations()
    for _ in range(5):
        reference = rng.normal(40.0, 12.0, size=350)
        prediction = 0.8 * reference + rng.normal(4.0, 5.0, size=350)
        values = {name: float(fn(reference, prediction)) for name, fn in implementations.items()}
        spread = max(values.values()) - min(values.values())
        assert spread < 2e-3, values


def test_ccc_reference_behaviour():
    """Perfect agreement is 1; a constant offset shrinks CCC by the bias term."""
    implementation = _implementations()["05_evaluation/utils.py"]
    x = np.array([10.0, 20.0, 30.0, 40.0])

    assert implementation(x, x) == 1.0

    variance = np.var(x, ddof=1)
    expected = 2.0 * variance / (2.0 * variance + 25.0)
    assert abs(implementation(x, x + 5.0) - expected) < 1e-12
