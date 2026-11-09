"""Formal ablation entrypoint for the ``vision_only`` experiment.

This thin wrapper reuses the curated early-concatenation script so that the
data split, preprocessing, losses, optimiser, scheduler, SWA and top-k
checkpoint protocol stay identical to ``concat_full``. The experiment tag is
*forced*, not defaulted: a stale ``EXPERIMENT_TAG`` in the caller's environment
must never turn this entrypoint into a different experiment.
"""

import importlib.util
import os
from pathlib import Path

DECLARED_EXPERIMENT_TAG = "vision_only"
WRAPPER_NAME = Path(__file__).name

# Forced, never os.environ.setdefault: the wrapper owns the experiment identity.
os.environ["EXPERIMENT_TAG"] = DECLARED_EXPERIMENT_TAG
os.environ["GFR_WRAPPER_NAME"] = WRAPPER_NAME
os.environ["GFR_WRAPPER_TAG"] = DECLARED_EXPERIMENT_TAG

_BASELINE_PATH = Path(__file__).with_name("05_train_architecture_baseline_early_concat_full.py")
_SPEC = importlib.util.spec_from_file_location("early_concat_baseline", _BASELINE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load early-concatenation baseline: {_BASELINE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_ACTUAL_TAG = getattr(_MODULE, "ABLATION_MODE", None)
if _ACTUAL_TAG != DECLARED_EXPERIMENT_TAG:
    raise RuntimeError(
        f"{WRAPPER_NAME} declares EXPERIMENT_TAG={DECLARED_EXPERIMENT_TAG!r} but the loaded "
        f"baseline resolved ABLATION_MODE={_ACTUAL_TAG!r}; refusing to run the wrong experiment"
    )

main = _MODULE.main


if __name__ == "__main__":
    main()
