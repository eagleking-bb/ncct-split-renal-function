#!/usr/bin/env python3
"""Shared utilities for the final locked analysis; no model fitting or inference."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


MODEL_ORDER = [
    "clinical_only",
    "radiomics_only",
    "image_only",
    "early_concat",
    "cross_attention",
    "dfme",
]
MODEL_COLORS = {
    "clinical_only": "#8C8C8C",
    "radiomics_only": "#B0A18F",
    "image_only": "#7E9AA8",
    "early_concat": "#3F6C8E",
    "cross_attention": "#C17A4A",
    "dfme": "#2E6E5B",
}
COHORT_COLORS = {"internal": "#3F6C8E", "external": "#C17A4A"}
SIDE_COLORS = {"left": "#4C78A8", "right": "#E08B54", "pooled": "#5F6B6D"}


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = str(config_path)
    return config


def output_dir(config: dict, section: str | None = None) -> Path:
    root = Path(config["output_dir"])
    path = root / section if section else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(config: dict, stage: str) -> logging.Logger:
    log_dir = output_dir(config, "logs")
    logger = logging.getLogger(stage)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anonymize_id(value: object) -> str:
    token = f"DFME-final-locked-analysis::{str(value).strip()}".encode()
    return "DFME_" + hashlib.sha256(token).hexdigest()[:16]


def git_commit(project_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable (workspace has no usable Git metadata)"


def write_environment_manifest(config: dict, inputs: Iterable[str | Path], stage: str) -> Path:
    records = []
    for raw in inputs:
        path = Path(raw)
        records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            }
        )
    manifest = {
        "stage": stage,
        "config": config["_config_path"],
        "git_commit": git_commit(config["project_root"]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "random_seed": config["random_seed"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "inputs": records,
    }
    path = output_dir(config, "environment") / f"{stage}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def validate_prediction_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {
        "case_id",
        "left_true",
        "right_true",
        "left_pred",
        "right_pred",
        "left_cls_true",
        "right_cls_true",
        "left_cls_prob",
        "right_cls_prob",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing prediction columns {missing}")
    out = frame.copy()
    out["case_id"] = out["case_id"].astype(str).str.strip()
    if out["case_id"].isna().any() or out["case_id"].duplicated().any():
        raise ValueError(f"{source}: missing or duplicate patient ID")
    numeric = list(required - {"case_id"})
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    if out[numeric].isna().any().any() or not np.isfinite(out[numeric].to_numpy()).all():
        raise ValueError(f"{source}: non-finite truth/prediction/probability")
    if not set(out["left_cls_true"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{source}: nonbinary left label")
    if not set(out["right_cls_true"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{source}: nonbinary right label")
    for column in ("left_cls_prob", "right_cls_prob"):
        if not out[column].between(0, 1).all():
            raise ValueError(f"{source}: probability outside [0,1]")
    return out.sort_values("case_id").reset_index(drop=True)


def load_prediction(path: str | Path) -> pd.DataFrame:
    return validate_prediction_frame(pd.read_csv(path), str(path))


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(ddof=1), y_pred.var(ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    denominator = vt + vp + (mt - mp) ** 2
    return float(2 * cov / denominator) if denominator > 0 else float("nan")


def safe_corr(func: Callable, x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return float("nan")
    return float(func(x[valid], y[valid]).statistic)


def continuous_metrics(frame: pd.DataFrame, truth: str = "true_side_gfr", pred: str = "predicted_gfr_dfme") -> dict:
    data = frame[["anonymized_patient_id", truth, pred]].dropna()
    y = data[truth].to_numpy(float)
    p = data[pred].to_numpy(float)
    error = p - y
    if len(y) < 2:
        return {key: float("nan") for key in CONTINUOUS_METRICS}
    slope, intercept = np.polyfit(p, y, 1) if np.unique(p).size > 1 else (float("nan"), float("nan"))
    return {
        "n_patients": int(data["anonymized_patient_id"].nunique()),
        "n_kidneys": int(len(data)),
        "mae": float(np.mean(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, p)),
        "pearson_r": safe_corr(stats.pearsonr, y, p),
        "spearman_rho": safe_corr(stats.spearmanr, y, p),
        "ccc": ccc(y, p),
        "bland_altman_bias": float(np.mean(error)),
        "bland_altman_loa_lower": float(np.mean(error) - 1.96 * np.std(error, ddof=1)),
        "bland_altman_loa_upper": float(np.mean(error) + 1.96 * np.std(error, ddof=1)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


CONTINUOUS_METRICS = [
    "mae",
    "median_absolute_error",
    "rmse",
    "bias",
    "r2",
    "pearson_r",
    "spearman_rho",
    "ccc",
    "bland_altman_bias",
    "bland_altman_loa_lower",
    "bland_altman_loa_upper",
    "calibration_intercept",
    "calibration_slope",
]


def classification_metrics(frame: pd.DataFrame, threshold: float = 0.5) -> dict:
    data = frame[["anonymized_patient_id", "true_srf_lt15", "predicted_probability_srf_lt15"]].dropna()
    y = data["true_srf_lt15"].astype(int).to_numpy()
    p = data["predicted_probability_srf_lt15"].to_numpy(float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    calibration_intercept = calibration_slope = float("nan")
    if len(np.unique(y)) == 2 and np.std(logits) > 0:
        import statsmodels.api as sm

        try:
            fit = sm.GLM(y, sm.add_constant(logits), family=sm.families.Binomial()).fit(disp=0)
            calibration_intercept, calibration_slope = map(float, fit.params)
        except Exception:
            pass
    return {
        "n_patients": int(data["anonymized_patient_id"].nunique()),
        "n_kidneys": int(len(data)),
        "positive_kidneys": int(y.sum()),
        "positive_patients": int(data.groupby("anonymized_patient_id")["true_srf_lt15"].max().sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "threshold": float(threshold),
    }


CLASSIFICATION_METRICS = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "brier",
    "calibration_intercept",
    "calibration_slope",
]


def patient_bootstrap(
    frame: pd.DataFrame,
    metric_function: Callable[[pd.DataFrame], dict],
    metrics: Iterable[str],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    patients = frame["anonymized_patient_id"].drop_duplicates().to_numpy()
    groups = {pid: block.copy() for pid, block in frame.groupby("anonymized_patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        blocks = []
        for index, pid in enumerate(sampled):
            block = groups[pid].copy()
            block["anonymized_patient_id"] = f"boot_{index}"
            blocks.append(block)
        try:
            result = metric_function(pd.concat(blocks, ignore_index=True))
        except Exception:
            result = {metric: float("nan") for metric in metrics}
        rows.append({"replicate": replicate, **{metric: result.get(metric, float("nan")) for metric in metrics}})
    replicates = pd.DataFrame(rows)
    intervals = {}
    for metric in metrics:
        values = pd.to_numeric(replicates[metric], errors="coerce").dropna()
        intervals[metric] = (
            float(values.quantile(0.025)) if len(values) else float("nan"),
            float(values.quantile(0.975)) if len(values) else float("nan"),
        )
    return intervals, replicates


def paired_bootstrap_difference(
    frame: pd.DataFrame,
    pred_a: str,
    pred_b: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = ["anonymized_patient_id", "true_side_gfr", pred_a, pred_b]
    data = frame.dropna(subset=needed).copy()
    patients = data["anonymized_patient_id"].drop_duplicates().to_numpy()
    groups = {pid: block for pid, block in data.groupby("anonymized_patient_id", sort=False)}
    metric_names = ["mae", "rmse", "ccc"]

    def compute(block: pd.DataFrame, column: str) -> dict:
        return continuous_metrics(block.rename(columns={column: "_prediction"}), pred="_prediction")

    observed_a, observed_b = compute(data, pred_a), compute(data, pred_b)
    rows = []
    rng = np.random.default_rng(seed)
    for replicate in range(n_boot):
        sampled = rng.choice(patients, len(patients), replace=True)
        boot = pd.concat([groups[pid].assign(anonymized_patient_id=f"b{i}") for i, pid in enumerate(sampled)])
        a, b = compute(boot, pred_a), compute(boot, pred_b)
        rows.append({"replicate": replicate, **{f"delta_{m}": a[m] - b[m] for m in metric_names}})
    reps = pd.DataFrame(rows)
    summary = []
    for metric in metric_names:
        column = f"delta_{metric}"
        values = reps[column].dropna()
        observed = observed_a[metric] - observed_b[metric]
        summary.append(
            {
                "metric": metric,
                "estimate_model_a": observed_a[metric],
                "estimate_model_b": observed_b[metric],
                "delta_model_a_minus_model_b": observed,
                "ci_low": float(values.quantile(0.025)),
                "ci_high": float(values.quantile(0.975)),
                # Descriptive only: the bootstrap distribution is centred on the
                # observed difference, so this share is NOT a p value.
                "share_bootstrap_above_zero_descriptive": float((values > 0).mean()),
                "p_value": float("nan"),
                "p_value_method": "not computed (see 06_statistics for the null-centred test)",
                "n_patients": int(data["anonymized_patient_id"].nunique()),
                "n_kidneys": int(len(data)),
            }
        )
    return pd.DataFrame(summary), reps


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)):
        kwargs = {"dpi": dpi} if dpi else {}
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)


def write_xlsx(frame: pd.DataFrame, path: Path, sheet_name: str = "results") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def format_ci(estimate: float, low: float, high: float, digits: int = 3) -> str:
    if not all(np.isfinite([estimate, low, high])):
        return "NA"
    return f"{estimate:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    data = frame.head(max_rows) if max_rows else frame
    headers = list(data.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in data.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
