#!/usr/bin/env python3
"""Strict old-to-NCCT nnU-Net v2 transfer training for locked NFH data."""

from __future__ import annotations

import _codecs
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# Paths are supplied through the environment so that no institutional location is
# hard-coded in the released code.
#   GFR_SEG_ROOT                  nnU-Net working root (nnUNet_raw / preprocessed / results)
#   GFR_SEG_PRETRAINED_CHECKPOINT contrast-enhanced CT checkpoint used for initialisation
ROOT = Path(os.environ["GFR_SEG_ROOT"]).expanduser().resolve()
OLD_CHECKPOINT = Path(os.environ["GFR_SEG_PRETRAINED_CHECKPOINT"]).expanduser().resolve()
DATASET = "Dataset817_NFHNCCTKidney"
PLANS_ID = "nnUNetPlans_StrictTransferNN"
CONFIGURATION = "3d_fullres"
FOLD = 0
SEED = 2026
TRAINER_NAME = "nnUNetTrainer_250epochs_NoMirroring"

os.environ["nnUNet_raw"] = str(ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(ROOT / "nnUNet_results")
os.environ["nnUNet_compile"] = "false"
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import (
    nnUNetTrainer_250epochs_NoMirroring,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_load(path: Path, map_location="cpu"):
    dtypes = [
        np.float16, np.float32, np.float64, np.int8, np.int16, np.int32,
        np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
    ]
    torch.serialization.add_safe_globals(
        [np.core.multiarray.scalar, np.dtype, _codecs.encode]
        + [type(np.dtype(d)) for d in dtypes]
    )
    return torch.load(path, map_location=map_location, weights_only=True)


def atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)


def validate_inputs() -> tuple[dict, dict, dict]:
    # The interpreter is not asserted here: run this script from the nnU-Net environment
    # (see REQUIREMENTS.txt), which is what the locked run did.
    if not OLD_CHECKPOINT.is_file():
        raise FileNotFoundError(OLD_CHECKPOINT)
    old_hash = sha256(OLD_CHECKPOINT)
    if old_hash != "ec21514cd5c4d36db7cf3834bd85217d4c750d93e77200e77d8345553489585c":
        raise RuntimeError(f"Old checkpoint SHA-256 changed: {old_hash}")

    pp = ROOT / "nnUNet_preprocessed" / DATASET
    plans_path = pp / f"{PLANS_ID}.json"
    dataset_path = pp / "dataset.json"
    split_path = pp / "splits_final.json"
    plans = load_json(str(plans_path))
    dataset_json = load_json(str(dataset_path))
    splits = load_json(str(split_path))
    if len(splits) != 1 or len(splits[0]["train"]) != 332 or len(splits[0]["val"]) != 83:
        raise RuntimeError("Locked segmentation split is not exactly 332 train / 83 development")
    if set(splits[0]["train"]) & set(splits[0]["val"]):
        raise RuntimeError("Train/development overlap detected")
    data_dir = pp / plans["configurations"][CONFIGURATION]["data_identifier"]
    seg_kwargs = plans["configurations"][CONFIGURATION]["resampling_fn_seg_kwargs"]
    if seg_kwargs.get("order") != 0 or seg_kwargs.get("order_z") != 0:
        raise RuntimeError(f"Mask resampling is not nearest-neighbor: {seg_kwargs}")
    npz = {p.stem for p in data_dir.glob("*.npz")}
    pkl = {p.stem for p in data_dir.glob("*.pkl")}
    expected = set(splits[0]["train"]) | set(splits[0]["val"])
    if npz != expected or pkl != expected:
        raise RuntimeError(
            f"Preprocessed cache mismatch: expected={len(expected)}, npz={len(npz)}, pkl={len(pkl)}, "
            f"missing_npz={len(expected - npz)}, missing_pkl={len(expected - pkl)}"
        )
    return plans, dataset_json, splits[0]


def output_folder() -> Path:
    return (
        ROOT / "nnUNet_results" / DATASET
        / f"{TRAINER_NAME}__{PLANS_ID}__{CONFIGURATION}" / f"fold_{FOLD}"
    )


def choose_resume_checkpoint(folder: Path):
    candidates = [folder / "checkpoint_latest.pth", folder / "checkpoint_best.pth"]
    found = []
    for path in candidates:
        if path.is_file():
            ckpt = safe_load(path, map_location="cpu")
            found.append((int(ckpt.get("current_epoch", -1)), path, ckpt))
    if not found:
        return None
    return max(found, key=lambda item: item[0])


def make_trainer(plans: dict, dataset_json: dict, unpack: bool):
    return nnUNetTrainer_250epochs_NoMirroring(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        unpack_dataset=unpack,
        device=torch.device("cuda", 0),
    )


def strict_transfer(trainer, checkpoint: dict) -> dict:
    trainer.initialize()
    source = checkpoint.get("network_weights")
    if not isinstance(source, dict):
        raise RuntimeError("Old checkpoint has no network_weights state_dict")
    target = trainer.network.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    shape_mismatches = {
        k: {"source": list(source[k].shape), "target": list(target[k].shape)}
        for k in sorted(set(source) & set(target)) if tuple(source[k].shape) != tuple(target[k].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(json.dumps({
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
        }, indent=2))
    result = trainer.network.load_state_dict(source, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict=True returned incompatibility: {result}")
    return {
        "strict": True,
        "loaded_tensors": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "shape_mismatches": shape_mismatches,
        "segmentation_heads_included": True,
        "fresh_optimizer_and_scheduler": True,
    }


def cleanup_dataloaders(trainer) -> None:
    for name in ("dataloader_train", "dataloader_val"):
        loader = getattr(trainer, name, None)
        if loader is not None and hasattr(loader, "_finish"):
            try:
                loader._finish()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    start = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    set_seed()
    plans, dataset_json, split = validate_inputs()
    folder = output_folder()
    final_ckpt = folder / "checkpoint_final.pth"
    if final_ckpt.exists() and not args.smoke_test:
        print(f"Training already complete: {final_ckpt}", flush=True)
        return 0

    old = safe_load(OLD_CHECKPOINT, map_location="cpu")
    trainer = make_trainer(plans, dataset_json, unpack=not args.smoke_test)
    resume = choose_resume_checkpoint(folder) if not args.smoke_test else None
    if resume is None:
        load_audit = strict_transfer(trainer, old)
        mode = "smoke_test_fresh_transfer" if args.smoke_test else "fresh_transfer"
        resume_path = None
        resume_epoch = 0
    else:
        trainer.initialize()
        resume_epoch, resume_path, resume_dict = resume
        trainer.load_checkpoint(resume_dict)
        # load_checkpoint uses strict load_state_dict; verify exact keys and shapes independently as well.
        current = trainer.network.state_dict()
        saved = resume_dict["network_weights"]
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        mismatches = [k for k in set(current) & set(saved) if tuple(current[k].shape) != tuple(saved[k].shape)]
        if missing or unexpected or mismatches:
            raise RuntimeError(f"Resume state incompatibility: missing={missing}, unexpected={unexpected}, shapes={mismatches}")
        load_audit = {
            "strict": True,
            "loaded_tensors": len(saved),
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": {},
            "resume_optimizer_scheduler_scaler": True,
        }
        mode = "resume"

    cfg = trainer.configuration_manager
    manifest = {
        "started_utc": now_utc(),
        "status": "smoke_test_running" if args.smoke_test else "training_running",
        "mode": mode,
        "root": str(ROOT),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "seed": SEED,
        "seed_scope": ["python_random", "numpy", "torch_cpu", "torch_cuda"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "determinism_note": "Seeds fixed; deterministic kernels are not guaranteed because nnU-Net speed defaults are retained.",
        "dataset": DATASET,
        "fold": FOLD,
        "locked_train_cases": len(split["train"]),
        "locked_development_cases": len(split["val"]),
        "trainer": TRAINER_NAME,
        "plans_identifier": PLANS_ID,
        "configuration": CONFIGURATION,
        "patch_size": list(cfg.patch_size),
        "spacing": list(cfg.spacing),
        "batch_size": trainer.batch_size,
        "epochs": trainer.num_epochs,
        "iterations_per_epoch": trainer.num_iterations_per_epoch,
        "validation_iterations_per_epoch": trainer.num_val_iterations_per_epoch,
        "foreground_oversampling": trainer.oversample_foreground_percent,
        "optimizer": "SGD(momentum=0.99, nesterov=True)",
        "initial_learning_rate": trainer.initial_lr,
        "weight_decay": trainer.weight_decay,
        "schedule": "PolyLRScheduler over 250 epochs",
        "loss": str(trainer.loss),
        "deep_supervision": trainer.enable_deep_supervision,
        "augmentation_mirroring": False,
        "inference_allowed_mirroring_axes": trainer.inference_allowed_mirroring_axes,
        "checkpoint_selection": "checkpoint_best.pth: maximum EMA foreground pseudo-Dice on the fixed 83-case development split during training",
        "old_checkpoint": str(OLD_CHECKPOINT),
        "old_checkpoint_sha256": sha256(OLD_CHECKPOINT),
        "checkpoint_load": load_audit,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_current_epoch": resume_epoch,
        "output_folder": str(folder),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = ROOT / ("finetune_smoke_manifest.json" if args.smoke_test else "finetune_training_manifest.json")
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    try:
        if args.smoke_test:
            trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()
            batch = next(trainer.dataloader_train)
            trainer.network.train()
            result = trainer.train_step(batch)
            manifest.update({
                "status": "smoke_test_pass",
                "completed_utc": now_utc(),
                "elapsed_seconds": time.time() - start,
                "train_step_loss": float(result["loss"]),
                "batch_data_shape": list(batch["data"].shape),
            })
            cleanup_dataloaders(trainer)
        else:
            trainer.run_training()
            if not final_ckpt.is_file():
                raise RuntimeError("Training returned without checkpoint_final.pth")
            manifest.update({
                "status": "training_complete",
                "completed_utc": now_utc(),
                "elapsed_seconds": time.time() - start,
                "checkpoint_final": str(final_ckpt),
                "checkpoint_final_sha256": sha256(final_ckpt),
                "checkpoint_best": str(folder / "checkpoint_best.pth"),
                "checkpoint_best_sha256": sha256(folder / "checkpoint_best.pth"),
            })
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        cleanup_dataloaders(trainer)
        manifest.update({
            "status": "failed",
            "failed_utc": now_utc(),
            "elapsed_seconds": time.time() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
