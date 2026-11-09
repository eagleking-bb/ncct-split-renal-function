"""Shared patient-level bootstrap helpers for the paired comparisons.

Terminology used by every statistics script in this folder:

``confidence interval``
    Percentile interval of the bootstrap distribution of the paired
    difference.  This is what the manuscripts report.
``descriptive bootstrap variability``
    The spread of that same distribution and the share of replicates on either
    side of zero.  Descriptive only, not a test.
``p value``
    A hypothesis test, and therefore requires a null distribution.  The
    bootstrap distribution is centred on the *observed* difference, so the share
    of replicates crossing zero is **not** a p value.  Where a p value is
    reported, :func:`null_centred_bootstrap_p` shifts the replicates onto the
    null (``delta* - delta_hat``), counts how often such a shifted replicate is
    at least as extreme as the observed difference, and smooths with
    ``(k + 1) / (n + 1)``.
"""

from __future__ import annotations

import numpy as np


def null_centred_bootstrap_p(
    delta_values,
    observed_delta: float,
    *,
    two_sided: bool = True,
) -> float:
    """Null-centred bootstrap p value with add-one smoothing.

    Parameters
    ----------
    delta_values
        Paired bootstrap replicates of the effect difference.
    observed_delta
        Effect difference estimated on the observed data.

    Returns
    -------
    float
        ``(k + 1) / (n + 1)`` where ``k`` counts shifted replicates at least as
        extreme as the observed difference.  Fully consistent models (all
        replicates and the observed difference equal to zero) give exactly 1.0.
    """
    values = np.asarray(delta_values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return float("nan")
    observed = float(observed_delta)
    if not np.isfinite(observed):
        return float("nan")
    if abs(observed) < 1e-12:
        return 1.0
    null_distribution = values - observed
    if two_sided:
        extreme = int(np.sum(np.abs(null_distribution) >= abs(observed)))
    else:
        extreme = int(np.sum(null_distribution >= observed if observed > 0 else null_distribution <= observed))
    return float((extreme + 1) / (n + 1))


def bootstrap_share_above_zero(delta_values) -> float:
    """Descriptive share of replicates above zero (NOT a p value)."""
    values = np.asarray(delta_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > 0))


P_VALUE_METHOD = (
    "null-centred patient-level bootstrap test with add-one smoothing; "
    "exploratory, not a confirmatory hypothesis test"
)
