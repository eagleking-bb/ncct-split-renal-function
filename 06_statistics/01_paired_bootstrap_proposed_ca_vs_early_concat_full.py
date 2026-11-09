#!/usr/bin/env python3
"""Paired patient-level bootstrap: final ensemble versus early concatenation.

The bootstrap samples validation cases with replacement. For SRF<15% AUC, each
sample then pools the left and right kidneys from the sampled cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bootstrap_common import (
    P_VALUE_METHOD,
    bootstrap_share_above_zero,
    null_centred_bootstrap_p,
)


MAIN_PRED_CSV: Path
CONCAT_PRED_CSV: Path
OUT_DIR: Path

N_BOOTSTRAPS = 10000
RNG_SEED = 2026
MODEL_MAIN = "Final equal-weight ensemble"
MODEL_CONCAT = "Full early concatenation"


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
    main = pd.read_csv(MAIN_PRED_CSV)
    concat = pd.read_csv(CONCAT_PRED_CSV)

    main_ids = set(main["case_id"])
    concat_ids = set(concat["case_id"])
    mismatch = pd.concat(
        [
            pd.DataFrame({"case_id": sorted(main_ids - concat_ids), "present_in": "main_only"}),
            pd.DataFrame({"case_id": sorted(concat_ids - main_ids), "present_in": "concat_only"}),
        ],
        ignore_index=True,
    )

    common_ids = sorted(main_ids & concat_ids)
    main_common = main[main["case_id"].isin(common_ids)].copy()
    concat_common = concat[concat["case_id"].isin(common_ids)].copy()

    paired = main_common.merge(concat_common, on="case_id", suffixes=("_main", "_concat"))
    paired = paired.sort_values("case_id").reset_index(drop=True)

    truth_pairs = [
        ("left_true_main", "left_true_concat"),
        ("right_true_main", "right_true_concat"),
        ("left_cls_true_main", "left_cls_true_concat"),
        ("right_cls_true_main", "right_cls_true_concat"),
    ]
    for main_col, concat_col in truth_pairs:
        if not np.allclose(paired[main_col].to_numpy(), paired[concat_col].to_numpy()):
            bad = paired.loc[paired[main_col] != paired[concat_col], ["case_id", main_col, concat_col]]
            raise ValueError(f"Truth mismatch in {main_col}/{concat_col}:\n{bad.head(10)}")

    paired = paired.rename(
        columns={
            "left_true_main": "left_true",
            "right_true_main": "right_true",
            "left_cls_true_main": "left_cls_true",
            "right_cls_true_main": "right_cls_true",
            "left_pred_main": "main_left_pred",
            "right_pred_main": "main_right_pred",
            "left_cls_prob_main": "main_left_cls_prob",
            "right_cls_prob_main": "main_right_cls_prob",
            "left_pred_concat": "concat_left_pred",
            "right_pred_concat": "concat_right_pred",
            "left_cls_prob_concat": "concat_left_cls_prob",
            "right_cls_prob_concat": "concat_right_cls_prob",
        }
    )

    keep_cols = [
        "case_id",
        "left_true",
        "right_true",
        "left_cls_true",
        "right_cls_true",
        "main_left_pred",
        "main_right_pred",
        "main_left_cls_prob",
        "main_right_cls_prob",
        "concat_left_pred",
        "concat_right_pred",
        "concat_left_cls_prob",
        "concat_right_cls_prob",
    ]
    return paired[keep_cols], mismatch


def run_bootstrap(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_main = metric_bundle(paired, "main")
    observed_concat = metric_bundle(paired, "concat")

    rows = [
        {"model": MODEL_MAIN, **observed_main},
        {"model": MODEL_CONCAT, **observed_concat},
    ]
    observed_metrics = pd.DataFrame(rows)

    rng = np.random.default_rng(RNG_SEED)
    n_cases = len(paired)
    boot_rows: list[dict[str, float | int]] = []

    for boot_idx in range(N_BOOTSTRAPS):
        idx = rng.integers(0, n_cases, n_cases)
        sample = paired.iloc[idx]
        y_cls = np.r_[sample["left_cls_true"].to_numpy(), sample["right_cls_true"].to_numpy()]
        if np.unique(y_cls).size < 2:
            continue

        main_metrics = metric_bundle(sample, "main")
        concat_metrics = metric_bundle(sample, "concat")
        boot_rows.append(
            {
                "bootstrap_index": boot_idx,
                "main_left_ccc": main_metrics["left_ccc"],
                "main_right_ccc": main_metrics["right_ccc"],
                "main_avg_ccc": main_metrics["avg_ccc"],
                "main_auc_srf15": main_metrics["auc_srf15"],
                "concat_left_ccc": concat_metrics["left_ccc"],
                "concat_right_ccc": concat_metrics["right_ccc"],
                "concat_avg_ccc": concat_metrics["avg_ccc"],
                "concat_auc_srf15": concat_metrics["auc_srf15"],
                "delta_left_ccc": concat_metrics["left_ccc"] - main_metrics["left_ccc"],
                "delta_right_ccc": concat_metrics["right_ccc"] - main_metrics["right_ccc"],
                "delta_avg_ccc": concat_metrics["avg_ccc"] - main_metrics["avg_ccc"],
                "delta_auc_srf15": concat_metrics["auc_srf15"] - main_metrics["auc_srf15"],
            }
        )

    bootstrap = pd.DataFrame(boot_rows)
    return observed_metrics, bootstrap


def summarize_deltas(observed_metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    main = observed_metrics.loc[observed_metrics["model"] == MODEL_MAIN].iloc[0]
    concat = observed_metrics.loc[observed_metrics["model"] == MODEL_CONCAT].iloc[0]
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
                "main_value": float(main[metric]),
                "concat_value": float(concat[metric]),
                "observed_delta_concat_minus_main": float(concat[metric] - main[metric]),
                "bootstrap_mean_delta": float(np.mean(delta_values)),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                # Exploratory null-centred bootstrap test (see bootstrap_common).
                "p_two_sided_null_centred_bootstrap": null_centred_bootstrap_p(
                    delta_values, float(concat[metric] - main[metric])
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

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    deltas = plot_df["observed_delta_concat_minus_main"].to_numpy()
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

    for row_idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["ci_95_high"] + 0.006,
            row_idx,
            f"Delta={row['observed_delta_concat_minus_main']:.3f} "
            f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel(f"Paired bootstrap delta ({MODEL_CONCAT} - {MODEL_MAIN})")
    ax.set_title("Paired Bootstrap Performance Difference", fontweight="bold", pad=10)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_xlim(min(-0.03, ci_low.min() - 0.02), max(0.09, ci_high.max() + 0.09))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "concat_full_vs_main_paired_bootstrap_delta_forest")
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
        ax.axvline(row["observed_delta_concat_minus_main"], color="#000000", linewidth=1.4)
        ax.axvspan(row["ci_95_low"], row["ci_95_high"], color=color, alpha=0.14)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"{MODEL_CONCAT} - {MODEL_MAIN}")
        ax.set_ylabel("Bootstrap density")
        ax.text(
            0.04,
            0.95,
            f"Delta={row['observed_delta_concat_minus_main']:.3f}\n"
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
    save_figure(fig, "concat_full_vs_main_paired_bootstrap_delta_distributions")
    plt.close(fig)


def plot_metric_bars(observed_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    metric_order = [("avg_ccc", "Avg CCC"), ("auc_srf15", "AUC")]
    colors = {MODEL_MAIN: "#2f5597", MODEL_CONCAT: "#b0413e"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    summary_by_metric = summary.set_index("metric")
    for ax, (metric, title) in zip(axes, metric_order):
        values = [
            observed_metrics.loc[observed_metrics["model"] == MODEL_MAIN, metric].iloc[0],
            observed_metrics.loc[observed_metrics["model"] == MODEL_CONCAT, metric].iloc[0],
        ]
        ax.bar([0, 1], values, color=[colors[MODEL_MAIN], colors[MODEL_CONCAT]], width=0.58)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Main", "concat_full"], rotation=0)
        ax.set_ylim(0, max(values) + 0.14)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        row = summary_by_metric.loc[metric]
        ax.text(
            0.5,
            max(values) + 0.07,
            f"Delta={row['observed_delta_concat_minus_main']:.3f}\n"
            f"95% CI {row['ci_95_low']:.3f} to {row['ci_95_high']:.3f}",
            ha="center",
            va="center",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("concat_full vs Final Cross-Attention Model", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "concat_full_vs_main_metric_bars")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, observed_metrics: pd.DataFrame, n_cases: int) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        rows = []
        for _, row in df.iterrows():
            formatted = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    formatted.append(f"{float(value):.6f}")
                else:
                    formatted.append(str(value))
            rows.append(formatted)

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = [
        "# Paired bootstrap: concat_full vs Main",
        "",
        f"- Bootstrap unit: validation case, with left/right kidneys kept together.",
        f"- Paired cases used: {n_cases}.",
        f"- Bootstrap replicates: {N_BOOTSTRAPS}.",
        f"- Difference definition: {MODEL_CONCAT} minus {MODEL_MAIN}.",
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
                "observed_delta_concat_minus_main",
                "ci_95_low",
                "ci_95_high",
                    "p_two_sided_null_centred_bootstrap",
                ]
            ]
        ),
        "",
    ]
    (OUT_DIR / "paired_bootstrap_concat_full_vs_main_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global MAIN_PRED_CSV, CONCAT_PRED_CSV, OUT_DIR, N_BOOTSTRAPS, RNG_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-predictions", type=Path, required=True)
    parser.add_argument("--concat-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    MAIN_PRED_CSV = args.main_predictions
    CONCAT_PRED_CSV = args.concat_predictions
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
        "main_predictions": str(MAIN_PRED_CSV),
        "concat_full_predictions": str(CONCAT_PRED_CSV),
        "output_dir": str(OUT_DIR),
        "n_main_cases": int(pd.read_csv(MAIN_PRED_CSV).shape[0]),
        "n_concat_cases": int(pd.read_csv(CONCAT_PRED_CSV).shape[0]),
        "n_paired_cases": int(len(paired)),
        "n_case_id_mismatch": int(len(mismatch)),
        "n_bootstraps_requested": N_BOOTSTRAPS,
        "n_bootstraps_valid": int(len(bootstrap)),
        "rng_seed": RNG_SEED,
        "bootstrap_unit": "case_id (patient); both kidneys always stay together",
        "p_value_method": P_VALUE_METHOD,
        "p_value_status": "exploratory null-centred bootstrap test; not confirmatory and not a replacement for the 95% CI",
        "classification_auc_unit": "pooled left and right kidneys within sampled cases",
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
