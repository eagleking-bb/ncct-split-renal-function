#!/usr/bin/env python3
"""Create an equal-weight ensemble from development-selected checkpoints."""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def calculate_ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_mean, pred_mean = np.mean(y_true), np.mean(y_pred)
    cov = np.cov(y_true, y_pred)[0, 1]
    return float(
        np.clip(
            (2 * cov)
            / (
                np.var(y_true, ddof=1)
                + np.var(y_pred, ddof=1)
                + (true_mean - pred_mean) ** 2
                + 1e-8
            ),
            -1.0,
            1.0,
        )
    )


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def load_prediction_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    raw = ckpt["plot_data"]
    n = len(raw["l_true"])
    case_id = raw.get("case_id", np.arange(n).astype(str))
    case_id = np.asarray(case_id).astype(str)
    cls_true = np.asarray(raw["cls_true"]).reshape(-1)
    cls_prob = np.asarray(raw["cls_prob"]).reshape(-1)
    df = pd.DataFrame(
        {
            "case_id": case_id,
            "left_true": np.asarray(raw["l_true"], dtype=float),
            "left_pred": np.asarray(raw["l_pred"], dtype=float),
            "right_true": np.asarray(raw["r_true"], dtype=float),
            "right_pred": np.asarray(raw["r_pred"], dtype=float),
            "left_cls_true": cls_true[:n].astype(int),
            "left_cls_prob": cls_prob[:n].astype(float),
            "right_cls_true": cls_true[n : 2 * n].astype(int),
            "right_cls_prob": cls_prob[n : 2 * n].astype(float),
        }
    )
    return ckpt, df.sort_values("case_id").reset_index(drop=True)


def evaluate_prediction_df(df):
    left_ccc = calculate_ccc(df["left_true"], df["left_pred"])
    right_ccc = calculate_ccc(df["right_true"], df["right_pred"])
    cls_true = np.concatenate([df["left_cls_true"].values, df["right_cls_true"].values])
    cls_prob = np.concatenate([df["left_cls_prob"].values, df["right_cls_prob"].values])
    auc = safe_auc(cls_true, cls_prob)
    return {
        "left_ccc": left_ccc,
        "right_ccc": right_ccc,
        "avg_ccc": (left_ccc + right_ccc) / 2.0,
        "min_ccc": min(left_ccc, right_ccc),
        "auc_srf_non_functioning": auc,
    }


def collect_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pth")))
        else:
            paths.extend(Path(x) for x in sorted(glob.glob(item)))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Checkpoint files, globs, or directories.")
    parser.add_argument("--top-n", type=int, default=0, help="Use only top N checkpoints by individual avg CCC.")
    parser.add_argument("--min-avg-ccc", type=float, default=-1.0)
    parser.add_argument("--outdir", type=str, default="ensemble_results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(args.paths)
    if not paths:
        raise FileNotFoundError("No checkpoint files matched.")

    records = []
    pred_frames = []
    reference_ids = None
    for path in paths:
        ckpt, df = load_prediction_checkpoint(path)
        metrics = evaluate_prediction_df(df)
        if metrics["avg_ccc"] < args.min_avg_ccc:
            continue
        if reference_ids is None:
            reference_ids = df["case_id"].tolist()
        elif reference_ids != df["case_id"].tolist():
            raise ValueError(f"Validation case order/set mismatch for {path}")
        records.append(
            {
                "path": str(path),
                "epoch": ckpt.get("epoch"),
                "run_seed": ckpt.get("run_seed"),
                "is_swa": ckpt.get("is_swa", False),
                **metrics,
            }
        )
        pred_frames.append((path, df))

    summary = pd.DataFrame(records).sort_values("avg_ccc", ascending=False).reset_index(drop=True)
    summary = summary.drop_duplicates(
        subset=["epoch", "run_seed", "avg_ccc", "left_ccc", "right_ccc"],
        keep="first",
    ).reset_index(drop=True)
    keep_unique_paths = set(summary["path"])
    pred_frames = [(p, df) for p, df in pred_frames if str(p) in keep_unique_paths]
    if args.top_n > 0:
        keep_paths = set(summary.head(args.top_n)["path"])
        pred_frames = [(p, df) for p, df in pred_frames if str(p) in keep_paths]
        summary = summary[summary["path"].isin(keep_paths)].reset_index(drop=True)

    if not pred_frames:
        raise RuntimeError("No checkpoints left after filtering.")

    base = pred_frames[0][1][
        ["case_id", "left_true", "right_true", "left_cls_true", "right_cls_true"]
    ].copy()
    base["left_pred"] = np.mean([df["left_pred"].values for _, df in pred_frames], axis=0)
    base["right_pred"] = np.mean([df["right_pred"].values for _, df in pred_frames], axis=0)
    base["left_cls_prob"] = np.mean([df["left_cls_prob"].values for _, df in pred_frames], axis=0)
    base["right_cls_prob"] = np.mean([df["right_cls_prob"].values for _, df in pred_frames], axis=0)
    ensemble_metrics = evaluate_prediction_df(base)
    base["ensemble_weight_per_checkpoint"] = 1.0 / len(pred_frames)

    summary.to_csv(outdir / "individual_checkpoint_metrics.csv", index=False)
    base.to_csv(outdir / "ensemble_predictions.csv", index=False)
    pd.DataFrame([{**ensemble_metrics, "n_checkpoints": len(pred_frames)}]).to_csv(
        outdir / "ensemble_metrics.csv",
        index=False,
    )

    print(summary[["path", "epoch", "run_seed", "is_swa", "avg_ccc", "left_ccc", "right_ccc", "auc_srf_non_functioning"]].to_string(index=False))
    print("\nEnsemble:")
    for key, value in ensemble_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  n_checkpoints: {len(pred_frames)}")
    print(f"\nSaved to: {outdir}")


if __name__ == "__main__":
    main()
