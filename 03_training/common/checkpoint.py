"""Checkpoint payload helpers.

A checkpoint written by these entrypoints carries everything required to
reconstruct predictions on a new cohort:

``inference_state.preprocessing``
    per-feature medians / means / scales for clinical variables, GFR targets and
    radiomics (per centre and pooled), plus the preprocessing order.
``inference_state.split``
    the patient-level split and its fingerprint.
``inference_state.image_preprocessing``
    spacing, crop, resize, intensity normalisation and augmentation spec.
``inference_state.model``
    model class, constructor kwargs and experiment label.

Old checkpoints (written before this schema) do not contain those fields.  They
are rejected by :func:`require_inference_state` instead of being silently
re-fitted, because re-fitting an imputer or scaler on external data would
invalidate the locked external evaluation.
"""

from __future__ import annotations

from typing import Sequence

INFERENCE_SCHEMA_VERSION = "dfme-inference-state-v1"


class MissingInferenceStateError(RuntimeError):
    """Raised when a checkpoint cannot support out-of-sample inference."""


def build_inference_state(
    dataset_state: dict,
    *,
    architecture: str,
    experiment_tag: str,
    model_class: str,
    model_kwargs: dict,
    code_entrypoint: str,
    rad_features: Sequence[str],
    clinical_features: Sequence[str],
    target_shape: Sequence[int],
    srf_threshold: float,
    run_seed: int,
    active_tabular_modalities: Sequence[str] | None = None,
    extra: dict | None = None,
) -> dict:
    state = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "preprocessing": dataset_state["preprocessing"],
        "split": dataset_state["split"],
        "image_preprocessing": dataset_state.get("image_preprocessing"),
        "model": {
            "architecture": architecture,
            "experiment_tag": experiment_tag,
            "class_name": model_class,
            "kwargs": dict(model_kwargs),
            "code_entrypoint": code_entrypoint,
        },
        "rad_features": list(rad_features),
        "clinical_features": list(clinical_features),
        "target_shape": list(target_shape),
        "srf_threshold": float(srf_threshold),
        "run_seed": int(run_seed),
        "active_tabular_modalities": list(
            active_tabular_modalities
            if active_tabular_modalities is not None
            else ("clinical", "radiomics")
        ),
        # Re-training is mandatory for this payload: the preprocessing order and
        # the augmentation schedule differ from the previously released runs.
        "reproducibility": {
            "rerun_required": "NEW_RUN_REQUIRED",
            "reason": (
                "median imputation now precedes standardisation, and random "
                "augmentation is sampled per epoch instead of being cached"
            ),
        },
    }
    if extra:
        state.update(extra)
    return state


def require_inference_state(checkpoint: dict, source: str = "checkpoint") -> dict:
    state = checkpoint.get("inference_state") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise MissingInferenceStateError(
            f"{source} does not contain an 'inference_state' block, so its "
            "imputation/scaling parameters and image preprocessing cannot be "
            "reconstructed.  Re-train the model with the current entrypoints "
            "(NEW_RUN_REQUIRED) or supply a compatible preprocessing state "
            "explicitly via the inference manifest.  Refitting on the external "
            "cohort is deliberately not supported."
        )
    schema = state.get("schema_version")
    if schema != INFERENCE_SCHEMA_VERSION:
        raise MissingInferenceStateError(
            f"{source} uses inference_state schema {schema!r}; expected "
            f"{INFERENCE_SCHEMA_VERSION!r}."
        )
    for key in ("preprocessing", "model", "rad_features"):
        if key not in state:
            raise MissingInferenceStateError(f"{source} inference_state is missing {key!r}")
    return state
