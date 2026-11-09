#!/usr/bin/env python3
"""Publication-style plots for network-adaptive radiomics feature screening."""

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


RANKING_CSV: Path
SELECTED_CSV: Path
OUT_DIR: Path


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)


def wrap_feature(name: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(name, width=width, break_long_words=False))


def selected_flag(df: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    return df["Base_Name"].isin(set(selected["Base_Name"]))


def plot_score_landscape(rank: pd.DataFrame, selected: pd.DataFrame) -> None:
    df = rank.copy()
    df["Selected"] = selected_flag(df, selected)
    size = 30 + 300 * (df["SRF15Signal"] - df["SRF15Signal"].min()) / (
        df["SRF15Signal"].max() - df["SRF15Signal"].min() + 1e-12
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        df["Stability"],
        df["AbsCorrResidual"],
        c=df["RawScore"],
        s=size,
        cmap="viridis",
        alpha=0.55,
        linewidth=0.25,
        edgecolor="white",
    )
    sel = df[df["Selected"]].copy()
    ax.scatter(
        sel["Stability"],
        sel["AbsCorrResidual"],
        s=165,
        facecolor="none",
        edgecolor="#d62728",
        linewidth=1.8,
        label="Selected features",
        zorder=4,
    )
    for _, row in sel.iterrows():
        ax.annotate(
            str(int(selected.loc[selected["Base_Name"] == row["Base_Name"], "SelectionOrder"].iloc[0])),
            (row["Stability"], row["AbsCorrResidual"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Composite raw score")
    ax.set_xlabel("Bootstrap ElasticNet selection frequency")
    ax.set_ylabel("|Spearman r| with patient-grouped clinical OOF residual")
    ax.set_title("Radiomics Screening Landscape", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    save_figure(fig, "radiomics_screening_score_landscape")
    plt.close(fig)


def plot_top_ranked_features(rank: pd.DataFrame, selected: pd.DataFrame, top_n: int = 25) -> None:
    df = rank.head(top_n).copy()
    df["Selected"] = selected_flag(df, selected)
    df["FeatureLabel"] = df["Base_Name"].map(lambda x: wrap_feature(x, 36))
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    colors = np.where(df["Selected"], "#d55e5e", "#7f8fa6")
    ax.barh(df["FeatureLabel"], df["RawScore"], color=colors, alpha=0.92)
    ax.set_xlabel("Composite raw score")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Candidate Radiomics Features", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    legend_items = [
        Line2D([0], [0], color="#d55e5e", lw=8, label="Selected"),
        Line2D([0], [0], color="#7f8fa6", lw=8, label="Not selected"),
    ]
    ax.legend(handles=legend_items, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "radiomics_top_candidate_raw_scores")
    plt.close(fig)


def plot_selected_score_matrix(selected: pd.DataFrame) -> None:
    df = selected.copy().sort_values("SelectionOrder")
    df["FeatureLabel"] = df["SelectionOrder"].astype(str) + ". " + df["Base_Name"].map(lambda x: wrap_feature(x, 30))
    df["ClinicalIndependence"] = 1.0 - df["MaxAbsCorrClinical"]
    cols = [
        "Stability",
        "CoefRank",
        "AbsCorrResidual",
        "SRF15Signal",
        "CenterRobustCorr",
        "SideRobustCorr",
        "ClinicalIndependence",
        "RawScore",
        "AdjustedScore",
    ]
    heat = df.set_index("FeatureLabel")[cols]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    sns.heatmap(
        heat,
        cmap="mako",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Multicriteria Profile of Selected Radiomics Features", fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, "radiomics_selected_feature_score_matrix")
    plt.close(fig)


def plot_redundancy_penalty(selected: pd.DataFrame) -> None:
    df = selected.copy().sort_values("SelectionOrder")
    y = np.arange(len(df))
    labels = df["SelectionOrder"].astype(str) + ". " + df["Base_Name"].map(lambda x: wrap_feature(x, 32))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    for i, row in enumerate(df.itertuples(index=False)):
        ax.plot([row.AdjustedScore, row.RawScore], [i, i], color="#9aa4ad", linewidth=1.6, zorder=1)
    ax.scatter(df["RawScore"], y, color="#4c78a8", s=60, label="Raw score", zorder=3)
    ax.scatter(df["AdjustedScore"], y, color="#d55e5e", s=60, label="After redundancy penalty", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title("Redundancy-aware Score Adjustment", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, loc="lower right")

    ax2 = axes[1]
    ax2.barh(y, df["MaxCorrWithEarlierSelected"], color="#2aa198", alpha=0.88)
    ax2.axvline(0.85, color="#d62728", linestyle="--", linewidth=1.2, label="Redundancy threshold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Max |Spearman r| with earlier selected features")
    ax2.set_title("Residual Correlation with Selected Set", fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "radiomics_redundancy_penalty_profile")
    plt.close(fig)


def plot_metric_correlation(rank: pd.DataFrame) -> None:
    cols = [
        "Stability",
        "MeanAbsCoef",
        "AbsCorrResidual",
        "AbsCorrRawGFR",
        "SRF15Signal",
        "CenterRobustCorr",
        "SideRobustCorr",
        "MaxAbsCorrClinical",
        "AbsCorrSCr",
        "CoefRank",
        "RawScore",
    ]
    corr = rank[cols].corr(method="spearman")
    labels = [
        "Stability",
        "ElasticNet\ncoef.",
        "Residual\nassoc.",
        "Raw GFR\nassoc.",
        "SRF<15%\nsignal",
        "Center\nrobust",
        "Side\nrobust",
        "Clinical\ncorr.",
        "SCr\ncorr.",
        "Coef\nrank",
        "Raw\nscore",
    ]

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.4,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Structure of Screening Criteria", fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, "radiomics_screening_metric_correlation_heatmap")
    plt.close(fig)


def main() -> None:
    global RANKING_CSV, SELECTED_CSV, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    RANKING_CSV = args.ranking_csv
    SELECTED_CSV = args.selected_csv
    OUT_DIR = args.output_dir

    setup_style()
    rank = pd.read_csv(RANKING_CSV)
    selected = pd.read_csv(SELECTED_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rank.to_csv(OUT_DIR / "ranking_used_for_figures.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_used_for_figures.csv", index=False)

    plot_score_landscape(rank, selected)
    plot_top_ranked_features(rank, selected)
    plot_selected_score_matrix(selected)
    plot_redundancy_penalty(selected)
    plot_metric_correlation(rank)

    summary = pd.DataFrame(
        [
            {
                "n_ranked_features": len(rank),
                "n_selected_features": len(selected),
                "median_stability_all": rank["Stability"].median(),
                "median_raw_score_all": rank["RawScore"].median(),
                "mean_stability_selected": selected["Stability"].mean(),
                "mean_raw_score_selected": selected["RawScore"].mean(),
                "mean_adjusted_score_selected": selected["AdjustedScore"].mean(),
                "max_redundancy_selected": selected["MaxCorrWithEarlierSelected"].max(),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "screening_figure_summary.csv", index=False)
    print(f"Saved figures to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
