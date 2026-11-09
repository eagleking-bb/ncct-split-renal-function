#!/usr/bin/env python3
"""Create architecture comparison tables and figures on the fixed validation set."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_absolute_error, roc_curve


ROOT = Path(".")
ABLATION_DIR = ROOT
OUT_DIR = Path("outputs/architecture_comparison")

MODELS = [
    {
        "key": "final_ensemble",
        "name": "Final equal-weight ensemble",
        "architecture": "0.5 early concatenation + 0.5 dual-branch cross-attention",
        "modalities": "CT + clinical + radiomics",
        "color": "#1b6b4a",
        "pred_csv": ROOT.parent
        / "internal_development_predictions"
        / "Figure4_final_equal_weight_ensemble_predictions_internal175.csv",
        "metrics_csv": None,
    },
    {
        "key": "dual_attention_full",
        "name": "Dual-branch cross-attention",
        "architecture": "separate clinical/radiomics cross-attention",
        "modalities": "CT + clinical + radiomics",
        "color": "#c76b2a",
        "pred_csv": ABLATION_DIR
        / "ensemble_dual_attention_full_seed66_seed42_seed88_top18_formal"
        / "ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR
        / "ensemble_dual_attention_full_seed66_seed42_seed88_top18_formal"
        / "ensemble_metrics.csv",
    },
    {
        "key": "concat_full",
        "name": "concat_full",
        "architecture": "early concatenation",
        "modalities": "CT + clinical + radiomics",
        "color": "#2f5597",
        "pred_csv": ABLATION_DIR / "ensemble_concat_full_seed66_seed42_seed88_top18_formal_drop_last_false/ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR / "ensemble_concat_full_seed66_seed42_seed88_top18_formal_drop_last_false/ensemble_metrics.csv",
    },
    {
        "key": "concat_clin_only",
        "name": "concat_clin_only",
        "architecture": "early concatenation",
        "modalities": "CT + clinical",
        "color": "#4c78a8",
        "pred_csv": ABLATION_DIR
        / "ensemble_concat_clin_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR
        / "ensemble_concat_clin_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_metrics.csv",
    },
    {
        "key": "concat_rad_only",
        "name": "concat_rad_only",
        "architecture": "early concatenation",
        "modalities": "CT + radiomics",
        "color": "#2aa198",
        "pred_csv": ABLATION_DIR
        / "ensemble_concat_rad_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR
        / "ensemble_concat_rad_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_metrics.csv",
    },
    {
        "key": "vision_only",
        "name": "vision_only",
        "architecture": "vision-only",
        "modalities": "CT only",
        "color": "#f28e2b",
        "pred_csv": ABLATION_DIR / "ensemble_vision_only_seed66_seed42_seed88_top18_formal_drop_last_false/ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR / "ensemble_vision_only_seed66_seed42_seed88_top18_formal_drop_last_false/ensemble_metrics.csv",
    },
]


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float((2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12))


def regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ccc": concordance_ccc(y_true, y_pred),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_models() -> list[dict]:
    loaded = []
    reference = None
    for model in MODELS:
        df = pd.read_csv(model["pred_csv"]).sort_values("case_id").reset_index(drop=True)
        truth = df[["case_id", "left_true", "right_true", "left_cls_true", "right_cls_true"]].copy()
        if reference is None:
            reference = truth
        else:
            if not truth["case_id"].equals(reference["case_id"]):
                raise ValueError(f"Validation case IDs differ for {model['name']}")
            truth_cols = ["left_true", "right_true", "left_cls_true", "right_cls_true"]
            if not np.allclose(truth[truth_cols].to_numpy(), reference[truth_cols].to_numpy()):
                raise ValueError(f"Validation labels differ for {model['name']}")
        if model["metrics_csv"] is None:
            left_ccc = regression_stats(df["left_true"], df["left_pred"])["ccc"]
            right_ccc = regression_stats(df["right_true"], df["right_pred"])["ccc"]
            labels = np.r_[df["left_cls_true"], df["right_cls_true"]]
            probabilities = np.r_[df["left_cls_prob"], df["right_cls_prob"]]
            metrics = {
                "left_ccc": left_ccc,
                "right_ccc": right_ccc,
                "avg_ccc": 0.5 * (left_ccc + right_ccc),
                "auc_srf_non_functioning": float(auc(*roc_curve(labels, probabilities)[:2])),
                "n_checkpoints": 36,
            }
        else:
            metrics = pd.read_csv(model["metrics_csv"]).iloc[0].to_dict()
        loaded.append({**model, "df": df, "metrics": metrics})
    return loaded


def build_summary(models: list[dict]) -> pd.DataFrame:
    rows = []
    concat_metrics = next(model["metrics"] for model in models if model["key"] == "concat_full")
    for model in models:
        metrics = model["metrics"]
        rows.append(
            {
                "model_key": model["key"],
                "model": model["name"],
                "architecture": model["architecture"],
                "modalities": model["modalities"],
                "left_ccc": float(metrics["left_ccc"]),
                "right_ccc": float(metrics["right_ccc"]),
                "avg_ccc": float(metrics["avg_ccc"]),
                "auc_srf15": float(metrics["auc_srf_non_functioning"]),
                "n_checkpoints": int(metrics["n_checkpoints"]),
                "delta_avg_ccc_vs_concat_full": float(metrics["avg_ccc"] - concat_metrics["avg_ccc"]),
                "delta_auc_vs_concat_full": float(
                    metrics["auc_srf_non_functioning"] - concat_metrics["auc_srf_non_functioning"]
                ),
                "prediction_csv": str(model["pred_csv"]),
                "metrics_csv": str(model["metrics_csv"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank_avg_ccc"] = summary["avg_ccc"].rank(ascending=False, method="min").astype(int)
    summary["rank_auc"] = summary["auc_srf15"].rank(ascending=False, method="min").astype(int)
    return summary


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)


def plot_metric_bars(summary: pd.DataFrame, models: list[dict]) -> None:
    colors = {model["key"]: model["color"] for model in models}
    x = np.arange(len(summary))
    labels = summary["model"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharex=True)
    panels = [("avg_ccc", "Average CCC"), ("auc_srf15", "SRF<15% AUC")]
    for ax, (metric, title) in zip(axes, panels):
        values = summary[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=[colors[key] for key in summary["model_key"]], width=0.68)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(values) + 0.11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Architecture Ensemble Comparison on the Fixed Validation Set", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "architecture_comparison_metric_bars")
    plt.close(fig)


def plot_roc(models: list[dict]) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for model in models:
        df = model["df"]
        y_true = np.r_[df["left_cls_true"].to_numpy(), df["right_cls_true"].to_numpy()]
        y_prob = np.r_[df["left_cls_prob"].to_numpy(), df["right_cls_prob"].to_numpy()]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        rows.append({"model_key": model["key"], "model": model["name"], "auc": float(roc_auc)})
        ax.plot(
            fpr,
            tpr,
            color=model["color"],
            linewidth=2.2,
            solid_capstyle="round",
            label=f"{model['name']} (AUC {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity", fontweight="bold")
    ax.set_ylabel("Sensitivity", fontweight="bold")
    ax.set_title("SRF<15% ROC Across Architecture Ensembles", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, edgecolor="#222222", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "architecture_comparison_srf15_roc_auc")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_scatter_grid(models: list[dict]) -> pd.DataFrame:
    all_true = []
    all_pred = []
    rows = []
    for model in models:
        df = model["df"]
        all_true.extend(df["left_true"].to_numpy())
        all_true.extend(df["right_true"].to_numpy())
        all_pred.extend(df["left_pred"].to_numpy())
        all_pred.extend(df["right_pred"].to_numpy())

    lim_min = min(np.min(all_true), np.min(all_pred)) - 5
    lim_max = max(np.max(all_true), np.max(all_pred)) + 5
    lim_min = min(-10, np.floor(lim_min / 10) * 10)
    lim_max = max(90, np.ceil(lim_max / 10) * 10)
    line_x = np.array([lim_min, lim_max])

    fig, axes = plt.subplots(2, len(models), figsize=(21.5, 7.2), sharex=True, sharey=True)
    for col, model in enumerate(models):
        df = model["df"]
        panels = [
            ("left", "Left kidney", df["left_true"].to_numpy(), df["left_pred"].to_numpy()),
            ("right", "Right kidney", df["right_true"].to_numpy(), df["right_pred"].to_numpy()),
        ]
        for row_idx, (side, side_label, y_true, y_pred) in enumerate(panels):
            ax = axes[row_idx, col]
            stats = regression_stats(y_true, y_pred)
            rows.append({"model_key": model["key"], "model": model["name"], "side": side, **stats})
            ax.scatter(
                y_true,
                y_pred,
                s=22,
                color=model["color"],
                alpha=0.68,
                edgecolor="white",
                linewidth=0.35,
            )
            ax.plot(line_x, line_x, color="#444444", linestyle="--", linewidth=1.0)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.grid(True, color="#eeeeee", linewidth=0.7, linestyle=":")
            if row_idx == 0:
                ax.set_title(model["name"], pad=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{side_label}\nPredicted split GFR", fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Measured split GFR", fontweight="bold")
            ax.text(
                0.05,
                0.95,
                f"CCC {stats['ccc']:.3f}\nr {stats['r']:.3f}\nMAE {stats['mae']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.6,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cfd6dd",
                    "alpha": 0.94,
                },
            )
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Split GFR Agreement Across Architecture Ensembles", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "architecture_comparison_split_gfr_scatter_grid")
    plt.close(fig)
    return pd.DataFrame(rows)


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


def write_markdown(summary: pd.DataFrame) -> None:
    table_cols = [
        "model",
        "architecture",
        "modalities",
        "left_ccc",
        "right_ccc",
        "avg_ccc",
        "auc_srf15",
        "n_checkpoints",
        "delta_avg_ccc_vs_concat_full",
        "delta_auc_vs_concat_full",
        "rank_avg_ccc",
        "rank_auc",
    ]
    lines = [
        "# Architecture comparison",
        "",
        "- All rows use the same fixed validation set and top18 checkpoint ensemble.",
        "- The dual-branch cross-attention component uses four attention heads.",
        "- Delta columns are model minus concat_full.",
        "",
        markdown_table(summary[table_cols]),
        "",
        "## Generated files",
        "",
        "- architecture_comparison_metrics.csv",
        "- architecture_comparison_metric_bars.png/pdf/svg",
        "- architecture_comparison_srf15_roc_auc.png/pdf/svg",
        "- architecture_comparison_split_gfr_scatter_grid.png/pdf/svg",
        "- architecture_regression_plot_metrics.csv",
        "- architecture_roc_plot_metrics.csv",
        "",
    ]
    (OUT_DIR / "architecture_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-ensemble-predictions", type=Path, required=True)
    parser.add_argument("--dual-attention-dir", type=Path, required=True)
    parser.add_argument("--concat-full-dir", type=Path, required=True)
    parser.add_argument("--concat-clin-only-dir", type=Path, required=True)
    parser.add_argument("--concat-rad-only-dir", type=Path, required=True)
    parser.add_argument("--vision-only-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    input_dirs = {
        "dual_attention_full": args.dual_attention_dir,
        "concat_full": args.concat_full_dir,
        "concat_clin_only": args.concat_clin_only_dir,
        "concat_rad_only": args.concat_rad_only_dir,
        "vision_only": args.vision_only_dir,
    }
    for model in MODELS:
        if model["key"] == "final_ensemble":
            model["pred_csv"] = args.final_ensemble_predictions
        else:
            model_dir = input_dirs[model["key"]]
            model["pred_csv"] = model_dir / "ensemble_predictions.csv"
            model["metrics_csv"] = model_dir / "ensemble_metrics.csv"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()
    summary = build_summary(models)
    summary.to_csv(OUT_DIR / "architecture_comparison_metrics.csv", index=False)
    plot_metric_bars(summary, models)
    roc_metrics = plot_roc(models)
    reg_metrics = plot_scatter_grid(models)
    roc_metrics.to_csv(OUT_DIR / "architecture_roc_plot_metrics.csv", index=False)
    reg_metrics.to_csv(OUT_DIR / "architecture_regression_plot_metrics.csv", index=False)
    write_markdown(summary)
    print(f"Saved architecture comparison outputs to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
