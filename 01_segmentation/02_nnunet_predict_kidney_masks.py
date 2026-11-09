#!/usr/bin/env python3
"""Prepare pseudonymized locked GZH/GDPH inputs and run the frozen NCCT model."""

from __future__ import annotations

import _codecs
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Paths are supplied through the environment (no institutional location is hard-coded).
#   GFR_SEG_ROOT     nnU-Net working root holding nnUNet_results
#   GFR_PROJECT_ROOT project root holding the analysis trees
ROOT = Path(os.environ["GFR_SEG_ROOT"]).expanduser().resolve()
PROJECT = Path(os.environ["GFR_PROJECT_ROOT"]).expanduser().resolve()
MODEL = ROOT / "nnUNet_results/Dataset817_NFHNCCTKidney/nnUNetTrainer_250epochs_NoMirroring__nnUNetPlans_StrictTransferNN__3d_fullres/fold_0/checkpoint_best.pth"
RUN_SALT = "nnunet-ncct-finetune-v1"
os.environ["nnUNet_raw"] = str(ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(ROOT / "nnUNet_results")
os.environ["nnUNet_compile"] = "false"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def study_id(raw_id: str, center: str) -> str:
    prefix = {"Center_2": "GZH", "Center_3": "GDPH"}[center]
    token = hashlib.sha256(f"{RUN_SALT}|{center}|{raw_id}".encode()).hexdigest()[:16]
    return f"{prefix}_{token}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_load(path: Path):
    dtypes = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32,
              np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]
    torch.serialization.add_safe_globals(
        [np.core.multiarray.scalar, np.dtype, _codecs.encode]
        + [type(np.dtype(d)) for d in dtypes]
    )
    return torch.load(path, map_location="cpu", weights_only=True)


# Output-count helpers live in a dependency-free module so they can be tested
# without nnU-Net installed.
from nnunet_output_checks import (  # noqa: E402
    CENTER_TO_PREFIX,
    PREFIX_TO_CENTER,
    check_prediction_outputs,
    expected_counts_by_prefix,
)


def prepare_inputs() -> tuple[dict[str, Path], dict[str, int], dict[str, set[str]]]:
    decision = json.loads((ROOT / "frozen_segmentation_decision.json").read_text())
    if not decision["development_reliable_for_frozen_inference"]:
        raise RuntimeError("Development reliability gate did not pass; external inference is prohibited")
    if sha256(MODEL) != decision["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint SHA-256 mismatch")
    manifest = pd.read_csv(ROOT / "segmentation_split_manifest.csv")
    selected = manifest[manifest.segmentation_role.isin([
        "frozen_inference_after_development_lock", "frozen_external_inference_only"
    ])]
    counts_by_center = selected.groupby("center").size().to_dict()
    if set(counts_by_center) != set(PREFIX_TO_CENTER.values()):
        raise RuntimeError(f"Locked frozen inference is missing a centre: {counts_by_center}")
    expected_by_prefix = expected_counts_by_prefix(counts_by_center)
    expected_ids_by_prefix = {
        prefix: set(selected.loc[selected.center == centre, "study_case_id"])
        for prefix, centre in PREFIX_TO_CENTER.items()
    }
    selected_ids = set(selected.study_case_id)
    audit_csv = Path(os.environ.get("GFR_NIFTI_AUDIT_CSV", PROJECT / "outputs/nifti_domain_audit/patient_level_audit.csv"))
    cohort = pd.read_csv(audit_csv, dtype=str)
    resolved = {}
    internal_rows = []
    for row in cohort.itertuples(index=False):
        if row.center not in ("Center_2", "Center_3"):
            continue
        sid = study_id(str(row.patient_id), row.center)
        if sid not in selected_ids:
            continue
        path = Path(row.nifti_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sid in resolved:
            raise RuntimeError(f"Duplicate resolved pseudonymized ID: {sid}")
        resolved[sid] = path
        internal_rows.append((sid, row.center, path))
    if set(resolved) != selected_ids:
        raise RuntimeError(f"Could not resolve {len(selected_ids - set(resolved))} locked cases")
    dirs = {}
    public_rows = []
    for prefix, center in PREFIX_TO_CENTER.items():
        directory = ROOT / "frozen_inference_inputs" / prefix
        directory.mkdir(parents=True, exist_ok=True)
        dirs[prefix] = directory
        for sid, row_center, source in internal_rows:
            if row_center != center:
                continue
            destination = directory / f"{sid}_0000.nii.gz"
            if not destination.exists():
                destination.symlink_to(source)
            public_rows.append({"study_case_id": sid, "center": center,
                                "locked_for_frozen_inference": True,
                                "input_file": str(destination.relative_to(ROOT))})
    pd.DataFrame(public_rows).sort_values(["center", "study_case_id"]).to_csv(
        ROOT / "frozen_inference_manifest.csv", index=False
    )
    return dirs, expected_by_prefix, expected_ids_by_prefix


def main() -> None:
    inputs, expected_by_prefix, expected_ids_by_prefix = prepare_inputs()
    ckpt = safe_load(MODEL)
    plans = ckpt["init_args"]["plans"]
    dataset_json = ckpt["init_args"]["dataset_json"]
    config_name = ckpt["init_args"]["configuration"]
    pm = PlansManager(plans)
    cm = pm.get_configuration(config_name)
    channels = determine_num_input_channels(pm, cm, dataset_json)
    heads = pm.get_label_manager(dataset_json).num_segmentation_heads
    network = nnUNetTrainer.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, channels, heads,
        enable_deep_supervision=False,
    )
    result = network.load_state_dict(ckpt["network_weights"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict frozen checkpoint load failed: {result}")
    if ckpt.get("inference_allowed_mirroring_axes") is not None:
        raise RuntimeError("Frozen checkpoint permits mirroring")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=torch.device("cuda", 0),
        verbose=False, verbose_preprocessing=False, allow_tqdm=True,
    )
    predictor.manual_initialization(
        network, pm, cm, [ckpt["network_weights"]], dataset_json,
        ckpt.get("trainer_name", "nnUNetTrainer_250epochs_NoMirroring"), None,
    )
    outputs = {}
    timings = {}
    reports = {}
    for prefix, directory in inputs.items():
        output = ROOT / "frozen_predictions" / prefix
        output.mkdir(parents=True, exist_ok=True)
        start = time.time()
        predictor.predict_from_files(
            str(directory), str(output), save_probabilities=False, overwrite=False,
            num_processes_preprocessing=2, num_processes_segmentation_export=2,
        )
        reports[prefix] = check_prediction_outputs(
            output, expected_ids_by_prefix[prefix], expected_by_prefix[prefix]
        )
        outputs[prefix], timings[prefix] = str(output), time.time() - start
    payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(MODEL), "checkpoint_sha256": sha256(MODEL),
        "state_dict_load": "strict=True complete network",
        "mirroring": False, "postprocessing": "none (frozen before GDPH)",
        "expected_counts_by_prefix": expected_by_prefix,
        "output_checks": reports,
        "timings_seconds": timings, "outputs": outputs,
        "GDPH_feedback_prohibited": True,
    }
    (ROOT / "frozen_inference_run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
