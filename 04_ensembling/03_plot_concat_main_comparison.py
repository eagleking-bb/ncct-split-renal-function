#!/usr/bin/env python3
"""Plot concat-main comparison figures on the fixed validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_absolute_error, roc_auc_score, roc_curve


ROOT = Path(".")
ABLATION_DIR = ROOT
OUT_DIR = Path("outputs/concat_main_figures")

MODELS = [
    {
        "key": "concat_full",
        "name": "concat_full",
        "label": "CT + clinical + radiomics",
        "color": "#2f5597",
        "pred_csv": ABLATION_DIR
        / "ensemble_concat_full_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR
        / "ensemble_concat_full_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_metrics.csv",
    },
    {
        "key": "concat_clin_only",
        "name": "concat_clin_only",
        "label": "CT + clinical",
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
        "label": "CT + radiomics",
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
        "label": "CT only",
        "color": "#f28e2b",
        "pred_csv": ABLATION_DIR
        / "ensemble_vision_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_predictions.csv",
        "metrics_csv": ABLATION_DIR
        / "ensemble_vision_only_seed66_seed42_seed88_top18_formal_drop_last_false"
        / "ensemble_metrics.csv",
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


BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPLICATES = 2000


def patient_bootstrap_indices(
    case_ids: np.ndarray,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Draw patients with replacement, keeping both kidneys in the same block.

    The row order is assumed to be one patient per contiguous pair of kidneys
    (left and right).  We nevertheless group by ``case_id`` rather than relying
    on row positions, and refuse a case that does not contribute two rows.
    """
    case_ids = np.asarray(case_ids, dtype=str)
    unique_cases = np.unique(case_ids)
    blocks = {case: np.flatnonzero(case_ids == case) for case in unique_cases}
    bad = {case: len(rows) for case, rows in blocks.items() if len(rows) != 2}
    if bad:
        raise ValueError(
            "patient-level bootstrap requires exactly two kidney rows per case; "
            f"offending cases: {bad}"
        )

    rng = np.random.default_rng(seed)
    indices = []
    for _ in range(n_boot):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        indices.append(np.concatenate([blocks[case] for case in sampled_cases]))
    return indices


def bootstrap_auc_ci(
    case_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    indices: list[np.ndarray] | None = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """95% percentile AUC CI with patient-level resampling.

    ``indices`` can be shared across models so that every model is evaluated on
    the same bootstrap patients.  The third return value is the number of valid
    replicates (those containing both outcome classes).
    """
    case_ids = np.asarray(case_ids, dtype=str)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if not (len(case_ids) == len(y_true) == len(y_prob)):
        raise ValueError("case_ids, y_true and y_prob must have the same length")
    if indices is None:
        indices = patient_bootstrap_indices(case_ids, n_boot=n_boot, seed=seed)

    values = []
    for idx in indices:
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not values:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), len(values)


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
        metrics = pd.read_csv(model["metrics_csv"]).iloc[0].to_dict()
        loaded.append({**model, "df": df, "metrics": metrics})
    return loaded


def setup_style() -> None:
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)


def build_summary(models: list[dict]) -> pd.DataFrame:
    rows = []
    main_metrics = next(model["metrics"] for model in models if model["key"] == "concat_full")
    for model in models:
        metrics = model["metrics"]
        rows.append(
            {
                "model_key": model["key"],
                "model": model["name"],
                "modalities": model["label"],
                "left_ccc": float(metrics["left_ccc"]),
                "right_ccc": float(metrics["right_ccc"]),
                "avg_ccc": float(metrics["avg_ccc"]),
                "auc_srf15": float(metrics["auc_srf_non_functioning"]),
                "n_checkpoints": int(metrics["n_checkpoints"]),
                "delta_avg_ccc_vs_concat_full": float(metrics["avg_ccc"] - main_metrics["avg_ccc"]),
                "delta_auc_vs_concat_full": float(
                    metrics["auc_srf_non_functioning"] - main_metrics["auc_srf_non_functioning"]
                ),
                "prediction_csv": str(model["pred_csv"]),
                "metrics_csv": str(model["metrics_csv"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank_avg_ccc"] = summary["avg_ccc"].rank(ascending=False, method="min").astype(int)
    summary["rank_auc"] = summary["auc_srf15"].rank(ascending=False, method="min").astype(int)
    return summary


def plot_metric_bars(summary: pd.DataFrame, models: list[dict]) -> None:
    colors = {model["key"]: model["color"] for model in models}
    x = np.arange(len(summary))
    labels = summary["model"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    panels = [("avg_ccc", "Average CCC"), ("auc_srf15", "SRF<15% AUC")]
    for ax, (metric, title) in zip(axes, panels):
        values = summary[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=[colors[key] for key in summary["model_key"]], width=0.66)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(values) + 0.11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", color="#ececec", linewidth=0.8)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Concat Main Model Comparison", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "concat_main_metric_bars")
    plt.close(fig)


def plot_ccc_profile(summary: pd.DataFrame, models: list[dict]) -> None:
    colors = {model["key"]: model["color"] for model in models}
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    metric_specs = [
        ("left_ccc", "Left CCC", "o", (0, 12), "center", "bottom"),
        ("right_ccc", "Right CCC", "s", (0, -16), "center", "top"),
        ("avg_ccc", "Average CCC", "^", (18, 0), "left", "center"),
    ]
    for metric, label, marker, text_offset, ha, va in metric_specs:
        values = summary[metric].to_numpy(dtype=float)
        ax.plot(x, values, color="#303030", linewidth=1.6, alpha=0.45)
        ax.scatter(
            x,
            values,
            s=70,
            marker=marker,
            color=[colors[key] for key in summary["model_key"]],
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for idx, val in enumerate(values):
            ax.annotate(
                f"{val:.3f}",
                xy=(idx, val),
                xytext=text_offset,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Concordance correlation coefficient", fontweight="bold")
    ax.set_ylim(0.68, 0.86)
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    ax.set_title("Split GFR CCC Profile Across Concat Ablations", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "concat_main_ccc_profile")
    plt.close(fig)


def plot_roc(models: list[dict]) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    rows = []
    if not models:
        raise ValueError("at least one model is required")
    # One shared set of patient-level resampling indices for every model.
    shared_case_ids = models[0]["df"]["case_id"].to_numpy()
    shared_stacked_ids = np.r_[shared_case_ids, shared_case_ids]
    shared_indices = patient_bootstrap_indices(shared_stacked_ids)
    for model in models:
        df = model["df"]
        if not np.array_equal(df["case_id"].to_numpy(), shared_case_ids):
            raise ValueError("case IDs are not aligned across models")
        case_ids = df["case_id"].to_numpy()
        stacked_case_ids = np.r_[case_ids, case_ids]
        y_true = np.r_[df["left_cls_true"].to_numpy(), df["right_cls_true"].to_numpy()]
        y_prob = np.r_[df["left_cls_prob"].to_numpy(), df["right_cls_prob"].to_numpy()]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ci_low, ci_high, n_valid = bootstrap_auc_ci(
            stacked_case_ids, y_true, y_prob, indices=shared_indices
        )
        rows.append(
            {
                "model_key": model["key"],
                "model": model["name"],
                "auc": float(roc_auc),
                "auc_95ci_low": ci_low,
                "auc_95ci_high": ci_high,
                "bootstrap_unit": "patient",
                "n_patients": int(len(np.unique(case_ids))),
                "n_boot_requested": int(len(shared_indices)),
                "n_boot_valid": int(n_valid),
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
        ax.plot(
            fpr,
            tpr,
            color=model["color"],
            linewidth=2.5,
            solid_capstyle="round",
            label=f"{model['name']} (AUC {roc_auc:.3f}, 95% CI {ci_low:.3f}-{ci_high:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sensitivity", fontsize=14, fontweight="bold")
    ax.set_title("SRF<15% ROC Across Concat Ablations", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, edgecolor="#222222", fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "concat_main_srf15_roc_auc")
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

    fig, axes = plt.subplots(2, len(models), figsize=(15.0, 7.2), sharex=True, sharey=True)
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
                s=24,
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
                fontsize=8.8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cfd6dd",
                    "alpha": 0.94,
                },
            )
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Split GFR Agreement Across Concat Ablations", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "concat_main_split_gfr_scatter_grid")
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
        "# Concat main comparison",
        "",
        "- Main model: concat_full.",
        "- All rows use the same fixed validation set and top18 checkpoint ensemble.",
        "- Delta columns are model minus concat_full.",
        "",
        markdown_table(summary[table_cols]),
        "",
        "## Generated files",
        "",
        "- concat_main_comparison_metrics.csv",
        "- concat_main_metric_bars.png/pdf/svg",
        "- concat_main_ccc_profile.png/pdf/svg",
        "- concat_main_srf15_roc_auc.png/pdf/svg",
        "- concat_main_split_gfr_scatter_grid.png/pdf/svg",
        "- concat_main_regression_plot_metrics.csv",
        "- concat_main_roc_plot_metrics.csv",
        "",
    ]
    (OUT_DIR / "concat_main_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concat-full-dir", type=Path, required=True)
    parser.add_argument("--concat-clin-only-dir", type=Path, required=True)
    parser.add_argument("--concat-rad-only-dir", type=Path, required=True)
    parser.add_argument("--vision-only-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    input_dirs = {
        "concat_full": args.concat_full_dir,
        "concat_clin_only": args.concat_clin_only_dir,
        "concat_rad_only": args.concat_rad_only_dir,
        "vision_only": args.vision_only_dir,
    }
    for model in MODELS:
        model_dir = input_dirs[model["key"]]
        model["pred_csv"] = model_dir / "ensemble_predictions.csv"
        model["metrics_csv"] = model_dir / "ensemble_metrics.csv"

    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()
    summary = build_summary(models)
    summary.to_csv(OUT_DIR / "concat_main_comparison_metrics.csv", index=False)
    plot_metric_bars(summary, models)
    plot_ccc_profile(summary, models)
    roc_metrics = plot_roc(models)
    reg_metrics = plot_scatter_grid(models)
    roc_metrics.to_csv(OUT_DIR / "concat_main_roc_plot_metrics.csv", index=False)
    reg_metrics.to_csv(OUT_DIR / "concat_main_regression_plot_metrics.csv", index=False)
    plot_manifest = {
        "bootstrap": {
            "unit": "patient",
            "resampling": "patients drawn with replacement; both kidneys of a patient kept together",
            "n_patients": int(roc_metrics["n_patients"].iloc[0]),
            "n_boot_requested": int(roc_metrics["n_boot_requested"].iloc[0]),
            "n_boot_valid_by_model": {
                str(row.model_key): int(row.n_boot_valid) for row in roc_metrics.itertuples()
            },
            "seed": BOOTSTRAP_SEED,
            "confidence_interval": "95% percentile",
            "shared_indices_across_models": True,
        },
        "auc_point_estimate": "kidney-level ROC over both kidneys; dependence handled only in the CI via patient resampling",
        "models": roc_metrics["model_key"].tolist(),
    }
    (OUT_DIR / "concat_main_plot_manifest.json").write_text(
        json.dumps(plot_manifest, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    print(f"Saved concat-main comparison outputs to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
