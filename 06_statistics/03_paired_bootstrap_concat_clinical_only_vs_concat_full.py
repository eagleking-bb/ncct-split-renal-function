#!/usr/bin/env python3
"""Paired bootstrap comparison: concat_clin_only vs concat_full.

The bootstrap samples validation cases with replacement. For SRF<15% AUC, each
sample pools the left and right kidneys from the sampled cases. Difference is
defined as concat_clin_only minus concat_full.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bootstrap_common import (
    P_VALUE_METHOD,
    bootstrap_share_above_zero,
    null_centred_bootstrap_p,
)


FULL_PRED_CSV: Path
CLIN_PRED_CSV: Path
OUT_DIR: Path

N_BOOTSTRAPS = 10000
RNG_SEED = 2026
MODEL_FULL = "concat_full"
MODEL_CLIN = "concat_clin_only"


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float((2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12))


def metric_bundle(df: pd.DataFrame, pred_prefix: str) -> dict[str, float]:
    left_ccc = concordance_ccc(df["left_true"].to_numpy(), df[f"{pred_prefix}_left_pred"].to_numpy())
    right_ccc = concordance_ccc(df["right_true"].to_numpy(), df[f"{pred_prefix}_right_pred"].to_numpy())
    y_cls = np.r_[df["left_cls_true"].to_numpy(), df["right_cls_true"].to_numpy()]
    p_cls = np.r_[
        df[f"{pred_prefix}_left_cls_prob"].to_numpy(),
        df[f"{pred_prefix}_right_cls_prob"].to_numpy(),
    ]
    auc = float(roc_auc_score(y_cls, p_cls))
    return {
        "left_ccc": left_ccc,
        "right_ccc": right_ccc,
        "avg_ccc": float((left_ccc + right_ccc) / 2.0),
        "auc_srf15": auc,
    }


def load_paired_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(FULL_PRED_CSV)
    clin = pd.read_csv(CLIN_PRED_CSV)

    full_ids = set(full["case_id"])
    clin_ids = set(clin["case_id"])
    mismatch = pd.concat(
        [
            pd.DataFrame({"case_id": sorted(full_ids - clin_ids), "present_in": "full_only"}),
            pd.DataFrame({"case_id": sorted(clin_ids - full_ids), "present_in": "clin_only"}),
        ],
        ignore_index=True,
    )

    common_ids = sorted(full_ids & clin_ids)
    full_common = full[full["case_id"].isin(common_ids)].copy()
    clin_common = clin[clin["case_id"].isin(common_ids)].copy()
    paired = full_common.merge(clin_common, on="case_id", suffixes=("_full", "_clin"))
    paired = paired.sort_values("case_id").reset_index(drop=True)

    truth_pairs = [
        ("left_true_full", "left_true_clin"),
        ("right_true_full", "right_true_clin"),
        ("left_cls_true_full", "left_cls_true_clin"),
        ("right_cls_true_full", "right_cls_true_clin"),
    ]
    for full_col, clin_col in truth_pairs:
        if not np.allclose(paired[full_col].to_numpy(), paired[clin_col].to_numpy()):
            bad = paired.loc[paired[full_col] != paired[clin_col], ["case_id", full_col, clin_col]]
            raise ValueError(f"Truth mismatch in {full_col}/{clin_col}:\n{bad.head(10)}")

    paired = paired.rename(
        columns={
            "left_true_full": "left_true",
            "right_true_full": "right_true",
            "left_cls_true_full": "left_cls_true",
            "right_cls_true_full": "right_cls_true",
            "left_pred_full": "full_left_pred",
            "right_pred_full": "full_right_pred",
            "left_cls_prob_full": "full_left_cls_prob",
            "right_cls_prob_full": "full_right_cls_prob",
            "left_pred_clin": "clin_left_pred",
            "right_pred_clin": "clin_right_pred",
            "left_cls_prob_clin": "clin_left_cls_prob",
            "right_cls_prob_clin": "clin_right_cls_prob",
        }
    )

    keep_cols = [
        "case_id",
        "left_true",
        "right_true",
        "left_cls_true",
        "right_cls_true",
        "full_left_pred",
        "full_right_pred",
        "full_left_cls_prob",
        "full_right_cls_prob",
        "clin_left_pred",
        "clin_right_pred",
        "clin_left_cls_prob",
        "clin_right_cls_prob",
    ]
    return paired[keep_cols], mismatch


def run_bootstrap(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_full = metric_bundle(paired, "full")
    observed_clin = metric_bundle(paired, "clin")
    observed_metrics = pd.DataFrame(
        [
            {"model": MODEL_FULL, **observed_full},
            {"model": MODEL_CLIN, **observed_clin},
        ]
    )

    rng = np.random.default_rng(RNG_SEED)
    n_cases = len(paired)
    boot_rows: list[dict[str, float | int]] = []

    for boot_idx in range(N_BOOTSTRAPS):
        idx = rng.integers(0, n_cases, n_cases)
        sample = paired.iloc[idx]
        y_cls = np.r_[sample["left_cls_true"].to_numpy(), sample["right_cls_true"].to_numpy()]
        if np.unique(y_cls).size < 2:
            continue

        full_metrics = metric_bundle(sample, "full")
        clin_metrics = metric_bundle(sample, "clin")
        boot_rows.append(
            {
                "bootstrap_index": boot_idx,
                "full_left_ccc": full_metrics["left_ccc"],
                "full_right_ccc": full_metrics["right_ccc"],
                "full_avg_ccc": full_metrics["avg_ccc"],
                "full_auc_srf15": full_metrics["auc_srf15"],
                "clin_left_ccc": clin_metrics["left_ccc"],
                "clin_right_ccc": clin_metrics["right_ccc"],
                "clin_avg_ccc": clin_metrics["avg_ccc"],
                "clin_auc_srf15": clin_metrics["auc_srf15"],
                "delta_left_ccc": clin_metrics["left_ccc"] - full_metrics["left_ccc"],
                "delta_right_ccc": clin_metrics["right_ccc"] - full_metrics["right_ccc"],
                "delta_avg_ccc": clin_metrics["avg_ccc"] - full_metrics["avg_ccc"],
                "delta_auc_srf15": clin_metrics["auc_srf15"] - full_metrics["auc_srf15"],
            }
        )

    return observed_metrics, pd.DataFrame(boot_rows)


def summarize_deltas(observed_metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    full = observed_metrics.loc[observed_metrics["model"] == MODEL_FULL].iloc[0]
    clin = observed_metrics.loc[observed_metrics["model"] == MODEL_CLIN].iloc[0]
    metrics = [
        ("left_ccc", "Left CCC", "delta_left_ccc"),
        ("right_ccc", "Right CCC", "delta_right_ccc"),
        ("avg_ccc", "Avg CCC", "delta_avg_ccc"),
        ("auc_srf15", "AUC", "delta_auc_srf15"),
    ]

    rows = []
    for metric, label, delta_col in metrics:
        delta_values = bootstrap[delta_col].to_numpy(dtype=float)
        ci_low, ci_high = np.percentile(delta_values, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "label": label,
                "concat_full_value": float(full[metric]),
                "concat_clin_only_value": float(clin[metric]),
                "observed_delta_clin_minus_full": float(clin[metric] - full[metric]),
                "bootstrap_mean_delta": float(np.mean(delta_values)),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                # Exploratory null-centred bootstrap test (see bootstrap_common).
                "p_two_sided_null_centred_bootstrap": null_centred_bootstrap_p(
                    delta_values, float(clin[metric] - full[metric])
                ),
                "p_value_method": P_VALUE_METHOD,
                "share_bootstrap_above_zero_descriptive": bootstrap_share_above_zero(delta_values),
                "n_bootstrap_valid": int(len(delta_values)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)


def plot_delta_forest(summary: pd.DataFrame) -> None:
    plot_df = summary.set_index("metric").loc[["left_ccc", "right_ccc", "avg_ccc", "auc_srf15"]]
    y = np.arange(len(plot_df))
    colors = ["#4c78a8", "#e15759", "#222222", "#59a14f"]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    deltas = plot_df["observed_delta_clin_minus_full"].to_numpy()
    ci_low = plot_df["ci_95_low"].to_numpy()
    ci_high = plot_df["ci_95_high"].to_numpy()
    xerr = np.vstack([deltas - ci_low, ci_high - deltas])

    ax.axvline(0.0, color="#8c8c8c", linestyle="--", linewidth=1.1)
    ax.errorbar(
        deltas,
        y,
        xerr=xerr,
        fmt="o",
        markersize=6.2,
        elinewidth=2.0,
        capsize=4,
        color="#222222",
        ecolor="#333333",
        zorder=3,
    )
    ax.scatter(deltas, y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=4)

    label_x = max(ci_high.max() + 0.004, 0.01)
    for row_idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            label_x,
            row_idx,
            f"Delta={row['observed_delta_clin_minus_full']:.3f} "
            f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel(f"Paired bootstrap delta ({MODEL_CLIN} - {MODEL_FULL})")
    ax.set_title("Paired Bootstrap Performance Difference", fontweight="bold", pad=10)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_xlim(min(-0.05, ci_low.min() - 0.015), max(0.04, label_x + 0.06))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "concat_clin_only_vs_concat_full_paired_bootstrap_delta_forest")
    plt.close(fig)


def plot_bootstrap_distributions(bootstrap: pd.DataFrame, summary: pd.DataFrame) -> None:
    panels = [
        ("delta_avg_ccc", "Avg CCC delta", "#222222"),
        ("delta_auc_srf15", "AUC delta", "#59a14f"),
    ]
    summary_by_metric = summary.set_index("metric")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, (col, title, color) in zip(axes, panels):
        values = bootstrap[col].to_numpy(dtype=float)
        metric = "avg_ccc" if col == "delta_avg_ccc" else "auc_srf15"
        row = summary_by_metric.loc[metric]
        ax.hist(values, bins=48, density=True, color=color, alpha=0.72, edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="#8c8c8c", linestyle="--", linewidth=1.0)
        ax.axvline(row["observed_delta_clin_minus_full"], color="#000000", linewidth=1.4)
        ax.axvspan(row["ci_95_low"], row["ci_95_high"], color=color, alpha=0.14)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"{MODEL_CLIN} - {MODEL_FULL}")
        ax.set_ylabel("Bootstrap density")
        ax.text(
            0.04,
            0.95,
            f"Delta={row['observed_delta_clin_minus_full']:.3f}\n"
            f"95% CI {row['ci_95_low']:.3f} to {row['ci_95_high']:.3f}\n"
            f"P={row['p_two_sided_null_centred_bootstrap']:.4f} (null-centred)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d9d9d9",
                "alpha": 0.94,
            },
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#ececec", linewidth=0.7)
    fig.suptitle("Bootstrap Distributions for Primary Endpoints", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "concat_clin_only_vs_concat_full_paired_bootstrap_delta_distributions")
    plt.close(fig)


def plot_metric_bars(observed_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    metric_order = [("avg_ccc", "Avg CCC"), ("auc_srf15", "AUC")]
    colors = {MODEL_FULL: "#2f5597", MODEL_CLIN: "#b0413e"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    summary_by_metric = summary.set_index("metric")
    for ax, (metric, title) in zip(axes, metric_order):
        values = [
            observed_metrics.loc[observed_metrics["model"] == MODEL_FULL, metric].iloc[0],
            observed_metrics.loc[observed_metrics["model"] == MODEL_CLIN, metric].iloc[0],
        ]
        ax.bar([0, 1], values, color=[colors[MODEL_FULL], colors[MODEL_CLIN]], width=0.58)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["concat_full", "clin_only"], rotation=0)
        ax.set_ylim(0, max(values) + 0.14)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        row = summary_by_metric.loc[metric]
        ax.text(
            0.5,
            max(values) + 0.07,
            f"Delta={row['observed_delta_clin_minus_full']:.3f}\n"
            f"95% CI {row['ci_95_low']:.3f} to {row['ci_95_high']:.3f}",
            ha="center",
            va="center",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("concat_clin_only vs concat_full", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "concat_clin_only_vs_concat_full_metric_bars")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, observed_metrics: pd.DataFrame, n_cases: int) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            formatted = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    formatted.append(f"{float(value):.6f}")
                else:
                    formatted.append(str(value))
            lines.append("| " + " | ".join(formatted) + " |")
        return "\n".join(lines)

    lines = [
        "# Paired bootstrap: concat_clin_only vs concat_full",
        "",
        "- Bootstrap unit: validation case, with left/right kidneys kept together.",
        f"- Paired cases used: {n_cases}.",
        f"- Bootstrap replicates: {N_BOOTSTRAPS}.",
        f"- Difference definition: {MODEL_CLIN} minus {MODEL_FULL}.",
        "",
        "## Observed metrics on paired cases",
        "",
        markdown_table(observed_metrics),
        "",
        "## Bootstrap differences",
        "",
        "Differences are reported as the 95% percentile CI of the paired patient-level bootstrap. P values, where shown, are null-centred bootstrap tests with add-one smoothing and are exploratory; the CI is the primary quantity.",
        "",
        markdown_table(
            summary[
                [
                    "label",
                    "observed_delta_clin_minus_full",
                    "ci_95_low",
                    "ci_95_high",
                    "p_two_sided_null_centred_bootstrap",
                ]
            ]
        ),
        "",
    ]
    (OUT_DIR / "paired_bootstrap_concat_clin_only_vs_concat_full_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    global FULL_PRED_CSV, CLIN_PRED_CSV, OUT_DIR, N_BOOTSTRAPS, RNG_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concat-full-predictions", type=Path, required=True)
    parser.add_argument("--concat-clin-only-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    FULL_PRED_CSV = args.concat_full_predictions
    CLIN_PRED_CSV = args.concat_clin_only_predictions
    OUT_DIR = args.output_dir
    N_BOOTSTRAPS = args.n_bootstraps
    RNG_SEED = args.seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    paired, mismatch = load_paired_predictions()
    observed_metrics, bootstrap = run_bootstrap(paired)
    summary = summarize_deltas(observed_metrics, bootstrap)

    paired.to_csv(OUT_DIR / "paired_validation_predictions.csv", index=False)
    mismatch.to_csv(OUT_DIR / "case_id_mismatch.csv", index=False)
    observed_metrics.to_csv(OUT_DIR / "paired_observed_metrics.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "paired_bootstrap_replicates.csv", index=False)
    summary.to_csv(OUT_DIR / "paired_bootstrap_delta_summary.csv", index=False)

    manifest = {
        "concat_full_predictions": str(FULL_PRED_CSV),
        "concat_clin_only_predictions": str(CLIN_PRED_CSV),
        "output_dir": str(OUT_DIR),
        "n_concat_full_cases": int(pd.read_csv(FULL_PRED_CSV).shape[0]),
        "n_concat_clin_only_cases": int(pd.read_csv(CLIN_PRED_CSV).shape[0]),
        "n_paired_cases": int(len(paired)),
        "n_case_id_mismatch": int(len(mismatch)),
        "n_bootstraps_requested": N_BOOTSTRAPS,
        "n_bootstraps_valid": int(len(bootstrap)),
        "rng_seed": RNG_SEED,
        "bootstrap_unit": "case_id (patient); both kidneys always stay together",
        "p_value_method": P_VALUE_METHOD,
        "p_value_status": "exploratory null-centred bootstrap test; not confirmatory and not a replacement for the 95% CI",
        "classification_auc_unit": "pooled left and right kidneys within sampled cases",
        "difference_definition": f"{MODEL_CLIN} - {MODEL_FULL}",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plot_delta_forest(summary)
    plot_bootstrap_distributions(bootstrap, summary)
    plot_metric_bars(observed_metrics, summary)
    write_markdown(summary, observed_metrics, len(paired))

    print(f"Saved paired bootstrap outputs to: {OUT_DIR}")
    print(observed_metrics.to_string(index=False))
    print(summary.to_string(index=False))
    if len(mismatch):
        print("\nCase IDs not shared by both prediction files:")
        print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
