#!/usr/bin/env python3
"""Create continuous calibration and SRF<15% DCA source data without recalibration."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import classification_metrics, configure_logging, load_config, output_dir, write_environment_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    logger = configure_logging(config, "05_calibration_and_dca")
    case_path = output_dir(config, "case_level") / "locked_case_level_predictions.csv"
    data = pd.read_csv(case_path)
    source = output_dir(config, "source_data")
    calibration_rows, dca_rows = [], []
    start, stop, step = (config["dca_thresholds"][key] for key in ("start", "stop", "step"))
    thresholds = np.round(np.arange(start, stop + step/2, step), 6)
    for split in ("internal", "external"):
        cohort = data[data.split == split]
        for side in ("left", "right", "pooled"):
            block = cohort if side == "pooled" else cohort[cohort.side == side]
            block = block.dropna(subset=["true_side_gfr", "predicted_gfr_dfme", "true_srf_lt15", "predicted_probability_srf_lt15"])
            block = block.copy()
            try:
                block["calibration_bin"] = pd.qcut(block["predicted_gfr_dfme"], q=10, duplicates="drop")
            except ValueError:
                block["calibration_bin"] = "all"
            for label, bin_frame in block.groupby("calibration_bin", observed=True):
                calibration_rows.append({"outcome": "continuous_gfr", "cohort": split, "side": side, "bin": str(label), "n": len(bin_frame), "mean_predicted": bin_frame.predicted_gfr_dfme.mean(), "mean_observed": bin_frame.true_side_gfr.mean(), "observed_sd": bin_frame.true_side_gfr.std(ddof=1)})
            y = block.true_srf_lt15.astype(int).to_numpy(); p = block.predicted_probability_srf_lt15.to_numpy(float); n = len(y)
            prevalence = y.mean()
            for threshold in thresholds:
                pred = p >= threshold
                tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
                model_nb = tp/n - fp/n * threshold/(1-threshold)
                all_nb = prevalence - (1-prevalence)*threshold/(1-threshold)
                dca_rows.extend([
                    {"cohort": split, "side": side, "threshold": threshold, "strategy": "DFME", "net_benefit": model_nb, "n": n, "events": int(y.sum())},
                    {"cohort": split, "side": side, "threshold": threshold, "strategy": "treat-all", "net_benefit": all_nb, "n": n, "events": int(y.sum())},
                    {"cohort": split, "side": side, "threshold": threshold, "strategy": "treat-none", "net_benefit": 0.0, "n": n, "events": int(y.sum())},
                ])
    pd.DataFrame(calibration_rows).to_csv(source / "gfr_calibration_source_data.csv", index=False)
    pd.DataFrame(dca_rows).to_csv(source / "srf15_dca_source_data.csv", index=False)
    write_environment_manifest(config, [case_path], "05_calibration_and_dca")
    logger.info("Calibration and DCA source data written; no predictions changed")


if __name__ == "__main__":
    main()
