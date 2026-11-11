"""Image pipeline shared by the five deep-learning entrypoints.

Why this module exists
----------------------
The earlier implementations handed the *full* training transform, including
``RandAffined`` / ``RandGaussianNoised`` / ``RandAdjustContrastd``, to
``PersistentDataset`` or ``CacheDataset``.  MONAI caches the result of the
transform, so a volume kept the same augmentation for the whole training run
instead of receiving a fresh draw every epoch, and ablations that followed that
path were trained with weaker augmentation than the main model.

The pipeline here separates the two stages:

``deterministic`` (cacheable)
    load -> channel first -> spacing -> foreground crop -> resize -> tensor.
``runtime`` (never cached, executed in ``__getitem__`` on every access)
    random affine -> patient-specific kidney normalisation -> Gaussian noise ->
    contrast jitter -> tensor conversion.

Validation uses only the deterministic stage plus the deterministic kidney
normalisation, so no randomness reaches validation or inference.

.. note::
   Caching only the deterministic stage keeps augmentation per-epoch for every
   entry point, so the models reported here are trained with the same schedule
   and stay comparable.  The cached path is not exposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from monai.data import Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstD,
    LoadImaged,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    ResizeD,
    Spacingd,
    ToTensorD,
)

CACHE_POLICY_VERSION = "deterministic-cache+per-epoch-runtime-v2"


class PatientSpecificKidneyNormalizer(MapTransform):
    """z-score the CT inside the kidney mask and zero everything outside it."""

    def __init__(self, keys: list, mask_key: str = "mask"):
        super().__init__(keys, allow_missing_keys=False)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            ct, mask = d[key].squeeze(), d[self.mask_key].squeeze()
            valid_voxels = ct[mask > 0]
            if len(valid_voxels) <= 1:
                mean_val, std_val = 0.0, 1.0
            else:
                mean_val, std_val = valid_voxels.mean(), valid_voxels.std() + 1e-8
            ct_norm = (ct - mean_val) / std_val
            d[key] = (ct_norm * (mask > 0)).unsqueeze(0).float()
        return d


def _resize_size(target_shape: Sequence[int]) -> tuple[int, int, int]:
    # Kept identical to the locked pipeline: the spatial size is passed as
    # (H, W, D) while ``target_shape`` is stored as (D, H, W).
    return (target_shape[1], target_shape[2], target_shape[0])


def get_deterministic_transforms(target_shape: Sequence[int] = (16, 256, 256)) -> Compose:
    """Cacheable preprocessing; no random component."""
    size = _resize_size(target_shape)
    return Compose(
        [
            LoadImaged(keys=["ct", "mask"], image_only=False, reader="NibabelReader"),
            EnsureChannelFirstD(keys=["ct", "mask"], channel_dim=None),
            Spacingd(keys=["ct", "mask"], pixdim=(1.0, 1.0, 3.0), mode=("trilinear", "nearest")),
            CropForegroundd(keys=["ct", "mask"], source_key="mask"),
            ResizeD(keys=["ct"], spatial_size=size, mode="trilinear"),
            ResizeD(keys=["mask"], spatial_size=size, mode="nearest"),
            ToTensorD(keys=["ct", "mask"]),
        ]
    )


def get_train_runtime_transforms() -> Compose:
    """Random augmentation; executed on every ``__getitem__``."""
    return Compose(
        [
            RandAffined(
                keys=["ct", "mask"],
                prob=0.2,
                rotate_range=(0.05, 0.05, 0.05),
                scale_range=(0.05, 0.05, 0.05),
                mode=("trilinear", "nearest"),
            ),
            PatientSpecificKidneyNormalizer(keys=["ct"], mask_key="mask"),
            RandGaussianNoised(keys=["ct"], prob=0.2, std=0.02),
            RandAdjustContrastd(keys=["ct"], prob=0.2, gamma=(0.8, 1.2)),
            ToTensorD(keys=["ct"]),
        ]
    )


def get_val_runtime_transforms() -> Compose:
    return Compose(
        [
            PatientSpecificKidneyNormalizer(keys=["ct"], mask_key="mask"),
            ToTensorD(keys=["ct"]),
        ]
    )


def get_train_transforms(target_shape: Sequence[int] = (16, 256, 256)) -> Compose:
    """Uncached training pipeline (deterministic followed by runtime)."""
    return Compose(
        [
            get_deterministic_transforms(target_shape),
            get_train_runtime_transforms(),
        ]
    )


def get_val_transforms(target_shape: Sequence[int] = (16, 256, 256)) -> Compose:
    return Compose(
        [
            get_deterministic_transforms(target_shape),
            get_val_runtime_transforms(),
        ]
    )


def image_preprocessing_spec(target_shape: Sequence[int] = (16, 256, 256)) -> dict:
    """Machine-readable description stored in checkpoints and manifests."""
    return {
        "cache_policy": CACHE_POLICY_VERSION,
        "reader": "NibabelReader",
        "spacing_mm": [1.0, 1.0, 3.0],
        "crop": "CropForegroundd(source_key='mask')",
        "resize_hwd": list(_resize_size(target_shape)),
        "target_shape_dhw": list(target_shape),
        # ResizeD receives (H, W, D); the resulting tensor is channel-first
        # (C, H, W, D) = (1, target_shape[1], target_shape[2], target_shape[0]).
        "tensor_layout": "channel_first_CHWD",
        "intensity": "per-patient z-score inside kidney mask, zero outside",
        "train_augmentation": {
            "rand_affine": {"prob": 0.2, "rotate_range": [0.05, 0.05, 0.05],
                            "scale_range": [0.05, 0.05, 0.05]},
            "rand_gaussian_noise": {"prob": 0.2, "std": 0.02},
            "rand_adjust_contrast": {"prob": 0.2, "gamma": [0.8, 1.2]},
            "applied": "per __getitem__ (per epoch, per sample); never cached",
        },
        "validation": "deterministic preprocessing + kidney normalisation only",
    }


class PreprocessedImageDataset(Dataset):
    """Deterministic cache + per-access runtime transform over sample dicts.

    ``cache="persistent"`` writes the deterministic output to ``cache_dir`` and
    reuses it across epochs; ``cache="none"`` recomputes it every access.  In
    both cases the random augmentation runs on every access, so augmentation is
    never frozen.
    """

    def __init__(
        self,
        samples: list,
        transform: Compose,
        runtime_transform: Compose | None = None,
        cache_dir: str | Path | None = None,
        cache: str = "persistent",
    ):
        if cache not in {"persistent", "none"}:
            raise ValueError(f"unsupported cache mode: {cache}")
        self.data = samples
        self.runtime_transform = runtime_transform
        self.cache = cache
        image_records = [
            {"case_id": sample["case_id"], "ct": sample["ct"], "mask": sample["mask"]}
            for sample in samples
        ]
        if cache == "persistent":
            if cache_dir is None:
                raise ValueError("cache_dir is required for cache='persistent'")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.image_dataset = PersistentDataset(
                data=image_records, transform=transform, cache_dir=str(cache_dir)
            )
        else:
            self.image_dataset = Dataset(data=image_records, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        image_data = self.image_dataset[index]
        if self.runtime_transform is not None:
            image_data = self.runtime_transform(image_data)
        sample = dict(self.data[index])
        sample["ct"] = image_data["ct"]
        sample["mask"] = image_data.get("mask")
        return sample


def build_datasets(
    train_samples: list,
    val_samples: list,
    target_shape: Sequence[int] = (16, 256, 256),
    cache_root: str | Path | None = None,
    use_cache: bool = True,
) -> tuple[Dataset, Dataset]:
    """Build train/validation datasets with the corrected caching policy."""
    if use_cache:
        if cache_root is None:
            raise ValueError("cache_root is required when use_cache=True")
        cache_root = Path(cache_root)
        return (
            PreprocessedImageDataset(
                train_samples,
                transform=get_deterministic_transforms(target_shape),
                runtime_transform=get_train_runtime_transforms(),
                cache_dir=cache_root / "train",
                cache="persistent",
            ),
            PreprocessedImageDataset(
                val_samples,
                transform=get_deterministic_transforms(target_shape),
                runtime_transform=get_val_runtime_transforms(),
                cache_dir=cache_root / "val",
                cache="persistent",
            ),
        )
    # No in-memory CacheDataset: caching a random transform is exactly the bug
    # this module replaces.  Deterministic work is recomputed instead.
    return (
        PreprocessedImageDataset(
            train_samples,
            transform=get_deterministic_transforms(target_shape),
            runtime_transform=get_train_runtime_transforms(),
            cache="none",
        ),
        PreprocessedImageDataset(
            val_samples,
            transform=get_deterministic_transforms(target_shape),
            runtime_transform=get_val_runtime_transforms(),
            cache="none",
        ),
    )
