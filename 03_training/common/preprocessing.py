"""Tabular preprocessing shared by every GFR model entrypoint.

Design contract
---------------
* Every statistic is estimated on the internal training split only.
* The order is always **median imputation first, standardisation second**.
  The earlier implementation scaled first and raised on missing input; the two
  orders give identical inputs whenever the mean and scale are estimated from
  observed values, so the fitted feature values are unchanged (see
  ``PREPROCESSING_ORDER``).
* Radiomics features are standardised per development centre by default,
  matching the locked analysis; a pooled training-cohort block is also stored
  so that a centre without fitted statistics (external validation) can be
  transformed with training-only parameters instead of being re-fitted.
* The full fitted state is JSON-serialisable and is written into every
  checkpoint, so external inference never needs to re-fit a scaler.

.. note::
   Median filling commutes with standardisation, because the median of an
   affine transform is the transform of the median.  Imputing first therefore
   reproduces the feature values of the previously released runs while staying
   defined when a column is entirely missing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "dfme-preprocessing-v3"
PREPROCESSING_ORDER = "median_impute_then_train_standardise"
DEFAULT_CLINICAL_FEATURES = ["height", "weight", "sex", "age", "SCr"]
DEFAULT_GFR_FEATURES = ["left_GFR", "right_GFR"]
DEFAULT_DEVELOPMENT_CENTERS = ("Center_1", "Center_2")


class PreprocessingStateError(RuntimeError):
    """Raised when a saved preprocessing state cannot be applied faithfully."""


def coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``columns`` coerced to float."""
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            raise KeyError(f"missing required column: {column}")
        if out[column].dtype == object:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def fit_column_stats(values: Iterable[float]) -> dict:
    """Median / mean / scale for one column, tolerant of all-missing columns."""
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"median": 0.0, "mean": 0.0, "scale": 1.0, "n_finite": 0}
    median = float(np.median(finite))
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=0))
    return {
        "median": median,
        "mean": mean,
        "scale": sd if sd > 0 else 1.0,
        "n_finite": int(finite.size),
    }


def fit_feature_block(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, dict]:
    return {feature: fit_column_stats(frame[feature]) for feature in features}


def apply_column_stats(values: Iterable[float], stats: dict) -> np.ndarray:
    """Median-impute then z-score using previously fitted statistics."""
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    median = float(stats["median"])
    mean = float(stats["mean"])
    scale = float(stats["scale"]) or 1.0
    array = np.where(np.isfinite(array), array, median)
    return (array - mean) / scale


def apply_feature_block(frame: pd.DataFrame, features: Sequence[str], block: dict) -> np.ndarray:
    missing = [feature for feature in features if feature not in block]
    if missing:
        raise PreprocessingStateError(
            "preprocessing state is missing statistics for: " + ", ".join(missing)
        )
    columns = [apply_column_stats(frame[feature], block[feature]) for feature in features]
    return np.column_stack(columns).astype(np.float64)


#: Ablation tags that deliberately remove one tabular block from the prior vector.
ABLATION_MODALITIES = {
    "concat_clin_only": ("clinical",),
    "ours_clin_only": ("clinical",),
    "concat_rad_only": ("radiomics",),
    "ours_rad_only": ("radiomics",),
    "vision_only": (),
}
ALL_TABULAR_MODALITIES = ("clinical", "radiomics")


def active_modalities_for_tag(experiment_tag: str) -> list[str]:
    """Which tabular blocks a given experiment actually uses."""
    return list(ABLATION_MODALITIES.get(experiment_tag, ALL_TABULAR_MODALITIES))


def mask_inactive_modalities(prior, n_clinical: int, active_modalities) -> np.ndarray:
    """Zero the blocks an ablation model was trained with zeroed.

    An external-inference caller that fed the full vector to a ``vision_only``
    or ``concat_clin_only`` checkpoint would silently evaluate a different model
    from the one that was trained.
    """
    active = set(active_modalities)
    output = np.array(prior, dtype=np.float32, copy=True)
    if "clinical" not in active:
        output[:, :n_clinical] = 0.0
    if "radiomics" not in active:
        output[:, n_clinical:] = 0.0
    return output


class FittedScaler:
    """Minimal stand-in for ``StandardScaler`` used on the GFR targets.

    ``02`` and ``05`` referenced ``gfr_scaler_left``/``gfr_scaler_right`` without
    ever defining them, so the first validation pass raised ``NameError``.  The
    GFR statistics are now taken from the fitted preprocessing state, which keeps
    a single source of truth and makes the de-standardisation explicit.
    """

    def __init__(self, mean: float = 0.0, scale: float = 1.0):
        self.mean_ = np.array([float(mean)], dtype=float)
        self.scale_ = np.array([float(scale)], dtype=float)

    def update_from_stats(self, stats: dict) -> "FittedScaler":
        self.mean_ = np.array([float(stats["mean"])], dtype=float)
        self.scale_ = np.array([float(stats["scale"]) or 1.0], dtype=float)
        return self

    def transform(self, values) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean_[0]) / self.scale_[0]

    def inverse_transform(self, values) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale_[0] + self.mean_[0]


@dataclass
class TabularPreprocessor:
    """Fitted imputation + standardisation state for clinical/radiomics/GFR."""

    clinical_features: list[str] = field(default_factory=lambda: list(DEFAULT_CLINICAL_FEATURES))
    radiomics_features: list[str] = field(default_factory=list)
    gfr_features: list[str] = field(default_factory=lambda: list(DEFAULT_GFR_FEATURES))
    radiomics_granularity: str = "per_center"
    clinical: dict[str, dict] = field(default_factory=dict)
    gfr: dict[str, dict] = field(default_factory=dict)
    radiomics: dict[str, dict[str, dict]] = field(default_factory=dict)
    center_column: str = "Center"
    fitted_centers: list[str] = field(default_factory=list)
    last_transform_notes: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame, centers: Sequence[str] | None = None) -> "TabularPreprocessor":
        if self.radiomics_granularity not in {"per_center", "pooled"}:
            raise ValueError(f"unsupported radiomics granularity: {self.radiomics_granularity}")
        self.clinical = fit_feature_block(frame, self.clinical_features)
        self.gfr = fit_feature_block(frame, self.gfr_features)

        center_values = frame[self.center_column].astype(str)
        if centers is None:
            centers = sorted(center_values.unique())
        self.fitted_centers = list(dict.fromkeys(str(center) for center in centers))

        # Pooled block: always fitted, used for centres without their own block.
        if self.radiomics_features:
            self.radiomics["pooled"] = fit_feature_block(frame, self.radiomics_features)

        if self.radiomics_granularity == "per_center":
            for center in self.fitted_centers:
                subset = frame.loc[center_values == center]
                if len(subset) == 0:
                    raise ValueError(f"no training rows for centre {center!r}")
                self.radiomics[center] = fit_feature_block(subset, self.radiomics_features)
        return self

    # -------------------------------------------------------------- transform
    def transform(
        self,
        frame: pd.DataFrame,
        unknown_center_stats: dict[str, dict] | None = None,
        transform_gfr: bool = False,
    ) -> pd.DataFrame:
        """Return a copy with imputed and standardised input columns.

        GFR is a target, not a model input.  It is therefore left in its
        original unit by default; set ``transform_gfr=True`` only for an
        explicit diagnostic transform.  Model training standardises the target
        once from these original values.

        ``unknown_center_stats`` optionally maps a centre name to a radiomics
        statistics block supplied explicitly (for example a JSON file that
        documents how an external centre was normalised).  Without it, a centre
        that was not seen during fitting falls back to the pooled training
        block; nothing is ever fitted on the data being transformed.
        """
        if not self.clinical or not self.gfr:
            raise PreprocessingStateError("preprocessor is not fitted")
        out = frame.copy()
        clinical = apply_feature_block(out, self.clinical_features, self.clinical)
        for index, feature in enumerate(self.clinical_features):
            out[feature] = clinical[:, index]
        if transform_gfr:
            gfr = apply_feature_block(out, self.gfr_features, self.gfr)
            for index, feature in enumerate(self.gfr_features):
                out[feature] = gfr[:, index]

        if self.radiomics_features:
            centers = out[self.center_column].astype(str)
            seen = set(centers.unique())
            blocks = {str(k): v for k, v in (unknown_center_stats or {}).items()}
            notes = {}
            for center in sorted(seen):
                if center in blocks:
                    notes[center] = "explicitly_supplied"
                elif center in self.radiomics:
                    blocks[center] = self.radiomics[center]
                    notes[center] = "fitted_training_centre"
                else:
                    # No fitted block for this centre: reuse the pooled block
                    # that was fitted on the whole training split.  Nothing is
                    # fitted on the data being transformed.
                    blocks[center] = self.radiomics["pooled"]
                    notes[center] = "pooled_training_fallback"
            for center in sorted(seen):
                mask = centers == center
                block = apply_feature_block(
                    out.loc[mask], self.radiomics_features, blocks[center]
                )
                out.loc[out.index[mask], self.radiomics_features] = block
            self.last_transform_notes = notes
        return out

    # ------------------------------------------------------------ (de)serialise
    def state(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "preprocessing_order": PREPROCESSING_ORDER,
            "center_column": self.center_column,
            "radiomics_granularity": self.radiomics_granularity,
            "clinical_features": list(self.clinical_features),
            "radiomics_features": list(self.radiomics_features),
            "gfr_features": list(self.gfr_features),
            "fitted_centers": list(self.fitted_centers),
            "clinical": self.clinical,
            "gfr": self.gfr,
            "radiomics": self.radiomics,
            "gfr_target_contract": {
                "unit": "mL/min/1.73 m2",
                "labels_and_original_values": "computed from raw GFR values",
                "model_target": "training-set mean/scale applied once",
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> "TabularPreprocessor":
        if not isinstance(state, dict) or "clinical" not in state:
            raise PreprocessingStateError(
                "checkpoint does not contain a usable tabular preprocessing state"
            )
        schema = state.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise PreprocessingStateError(
                f"unsupported preprocessing schema {schema!r}; expected {SCHEMA_VERSION!r}. "
                "States written before the GFR target-unit fix are not compatible; "
                "refit the preprocessing state with the current entrypoints."
            )
        order = state.get("preprocessing_order")
        if order not in (None, PREPROCESSING_ORDER):
            raise PreprocessingStateError(
                f"unsupported preprocessing order {order!r}; expected {PREPROCESSING_ORDER!r}"
            )
        preprocessor = cls(
            clinical_features=list(state.get("clinical_features", DEFAULT_CLINICAL_FEATURES)),
            radiomics_features=list(state.get("radiomics_features", [])),
            gfr_features=list(state.get("gfr_features", DEFAULT_GFR_FEATURES)),
            radiomics_granularity=state.get("radiomics_granularity", "per_center"),
            center_column=state.get("center_column", "Center"),
        )
        preprocessor.clinical = state["clinical"]
        preprocessor.gfr = state["gfr"]
        preprocessor.radiomics = state.get("radiomics", {})
        preprocessor.fitted_centers = list(state.get("fitted_centers", []))
        return preprocessor

    def feature_order_fingerprint(self) -> str:
        payload = "|".join(
            [
                ",".join(self.clinical_features),
                ",".join(self.radiomics_features),
                ",".join(self.gfr_features),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- splits
def split_development_cohort(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = "Center",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level development split (one row per patient)."""
    from sklearn.model_selection import train_test_split

    if frame["ID"].astype(str).duplicated().any():
        raise ValueError("development table must contain one row per patient")
    stratify = frame[stratify_column] if stratify_column in frame.columns else None
    train, validation = train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train.copy(), validation.copy()


def split_fingerprint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    id_column: str = "ID",
    center_column: str = "Center",
) -> dict:
    """Deterministic fingerprint of a patient split, for checkpoint provenance."""
    def canonical(frame: pd.DataFrame) -> str:
        rows = sorted(
            f"{str(row[id_column]).strip()}|{str(row[center_column])}"
            for _, row in frame.iterrows()
        )
        return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

    return {
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "train_ids": sorted(str(value).strip() for value in train[id_column]),
        "validation_ids": sorted(str(value).strip() for value in validation[id_column]),
        "train_fingerprint": canonical(train),
        "validation_fingerprint": canonical(validation),
    }


# ------------------------------------------------------------ dataset builder
@dataclass
class DatasetBundle:
    train_samples: list
    val_samples: list
    state: dict


def build_development_bundle(
    frame: pd.DataFrame,
    radiomics_features: Sequence[str],
    clinical_features: Sequence[str] = DEFAULT_CLINICAL_FEATURES,
    gfr_features: Sequence[str] = DEFAULT_GFR_FEATURES,
    target_shape: Sequence[int] = (16, 256, 256),
    srf_threshold: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
    prior_ablation: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple] | None = None,
) -> DatasetBundle:
    """Split, fit preprocessing on the training split, and build sample dicts.

    ``frame`` must already contain ``ID``, ``Center``, ``ct_path``, ``mask_path``
    and every clinical / radiomics / GFR column.
    """
    radiomics_features = list(radiomics_features)
    clinical_features = list(clinical_features)
    gfr_features = list(gfr_features)
    numeric_columns = clinical_features + radiomics_features + gfr_features
    frame = coerce_numeric(frame, numeric_columns)

    train, validation = split_development_cohort(
        frame, test_size=test_size, random_state=random_state
    )

    preprocessor = TabularPreprocessor(
        clinical_features=clinical_features,
        radiomics_features=radiomics_features,
        gfr_features=gfr_features,
    ).fit(train)

    train_scaled = preprocessor.transform(train)
    # Validation is transformed with the training statistics only.
    validation_scaled = preprocessor.transform(validation)

    def build_dicts(raw: pd.DataFrame, scaled: pd.DataFrame) -> list:
        if len(raw) != len(scaled):
            raise ValueError("raw and transformed development frames are not aligned")
        samples = []
        for (_, raw_row), (_, row) in zip(raw.iterrows(), scaled.iterrows()):
            norm_clin = row[clinical_features].to_numpy(dtype=np.float64)
            half = len(radiomics_features) // 2
            norm_rad_l = row[radiomics_features[:half]].to_numpy(dtype=np.float64)
            norm_rad_r = row[radiomics_features[half:]].to_numpy(dtype=np.float64)
            if prior_ablation is not None:
                norm_clin, norm_rad_l, norm_rad_r = prior_ablation(
                    norm_clin, norm_rad_l, norm_rad_r
                )
            prior_l = np.concatenate([norm_clin, norm_rad_l]).astype(np.float32)
            prior_r = np.concatenate([norm_clin, norm_rad_r]).astype(np.float32)

            # Labels, class thresholds and the stored original values must come
            # from the raw clinical GFR measurement.  The transformed frame is
            # used only for model inputs; its GFR columns are never read here.
            left_gfr = float(raw_row[gfr_features[0]])
            right_gfr = float(raw_row[gfr_features[1]])
            if not (np.isfinite(left_gfr) and np.isfinite(right_gfr)):
                raise ValueError("GFR must be non-missing after cohort construction")
            # Standardise the target exactly once, with training-set statistics.
            left_z = float(apply_column_stats([left_gfr], preprocessor.gfr[gfr_features[0]])[0])
            right_z = float(apply_column_stats([right_gfr], preprocessor.gfr[gfr_features[1]])[0])

            total_gfr = left_gfr + right_gfr + 1e-8
            left_srf = left_gfr / total_gfr
            right_srf = right_gfr / total_gfr
            samples.append(
                {
                    "case_id": row["ID"],
                    "ct": row["ct_path"],
                    "mask": row["mask_path"],
                    "center": row["Center"],
                    "prior_l": prior_l,
                    "prior_r": prior_r,
                    "left_gfr": np.array([left_z], dtype=np.float32),
                    "right_gfr": np.array([right_z], dtype=np.float32),
                    "label_cls_l": np.array([1.0 if left_srf < srf_threshold else 0.0], dtype=np.float32),
                    "label_cls_r": np.array([1.0 if right_srf < srf_threshold else 0.0], dtype=np.float32),
                    "left_gfr_original": left_gfr,
                    "right_gfr_original": right_gfr,
                }
            )
        return samples

    state = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing": preprocessor.state(),
        "split": {
            "method": (
                f"train_test_split(test_size={test_size}, random_state={random_state}, "
                "stratify=Center); one row per patient"
            ),
            "unit": "patient",
            **split_fingerprint(train, validation),
        },
        "target_shape": list(target_shape),
        "srf_threshold": float(srf_threshold),
    }
    return DatasetBundle(
        build_dicts(train, train_scaled),
        build_dicts(validation, validation_scaled),
        state,
    )
