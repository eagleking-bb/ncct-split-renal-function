#!/usr/bin/env python3
"""Clinical and radiomics tabular baselines for split renal GFR prediction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.preprocessing import (
    TabularPreprocessor,
    coerce_numeric as coerce_numeric_shared,
    split_development_cohort,
    split_fingerprint,
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False


OUT_DIR: Path
CLINICAL_CSV_PATH: Path
RADIOMICS_CSV_PATH: Path
SELECTED_CSV: Path
CENTER_CONFIGS: list[dict]

CLINICAL_COLS = ["height", "weight", "sex", "age", "SCr"]
SEEDS = [66, 42, 88]
SRF_THRESHOLD = 0.15


def find_existing_nii(base_dir: Path, pid: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        path = base_dir / f"{pid}{suffix}"
        if path.exists():
            return path
    return None


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            codes, _ = pd.factorize(df[col])
            df[col] = codes.astype(float)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def selected_radiomics_features() -> list[str]:
    selected = pd.read_csv(SELECTED_CSV)
    bases = selected["Base_Name"].tolist()
    return [f"Left_{b}" for b in bases] + [f"Right_{b}" for b in bases]


def load_dataset():
    """Load the development cohort and fit preprocessing on the training split.

    Order is median imputation -> standardisation, with every statistic fitted
    on the training split only (per-centre radiomics, as in the deep-learning
    entrypoints).  The fitted preprocessor is returned so that it can be saved
    next to the estimators for external inference.
    """
    rad_features = selected_radiomics_features()
    df_clin = pd.read_csv(CLINICAL_CSV_PATH)
    df_rad = pd.read_csv(RADIOMICS_CSV_PATH)
    df = pd.merge(df_clin, df_rad, on="ID", how="inner")
    df = df[df["Center"].isin(["Center_1", "Center_2"])].copy()
    df.reset_index(drop=True, inplace=True)

    missing = [c for c in CLINICAL_COLS + rad_features + ["left_GFR", "right_GFR"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    path_map = {cfg["name"]: cfg for cfg in CENTER_CONFIGS}
    valid_indices = []
    for idx, row in df.iterrows():
        pid = str(row["ID"]).strip()
        center = row["Center"]
        ct_path = find_existing_nii(path_map[center]["ct_dir"], pid)
        mask_path = find_existing_nii(path_map[center]["mask_dir"], pid)
        if ct_path is not None and mask_path is not None:
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].copy().reset_index(drop=True)
    # Factorise categorical clinical variables once, before splitting, so that
    # train and validation share the same coding.
    df_valid = coerce_numeric(df_valid, CLINICAL_COLS + ["left_GFR", "right_GFR"])

    df_train, df_val = split_development_cohort(df_valid, test_size=0.2, random_state=42)

    preprocessor = TabularPreprocessor(
        clinical_features=CLINICAL_COLS,
        radiomics_features=rad_features,
        gfr_features=["left_GFR", "right_GFR"],
    ).fit(df_train)
    # Keep the raw GFR values alongside the transformed clinical/radiomics
    # columns.  ``left_GFR`` / ``right_GFR`` in the transformed frame are model
    # targets, not labels; they must never be used to derive SRF or report
    # true values.
    df_train_raw = df_train.reset_index(drop=True)
    df_val_raw = df_val.reset_index(drop=True)
    df_train = preprocessor.transform(df_train).reset_index(drop=True)
    df_val = preprocessor.transform(df_val).reset_index(drop=True)
    df_train["left_GFR_original"] = df_train_raw["left_GFR"].to_numpy(dtype=float)
    df_train["right_GFR_original"] = df_train_raw["right_GFR"].to_numpy(dtype=float)
    df_val["left_GFR_original"] = df_val_raw["left_GFR"].to_numpy(dtype=float)
    df_val["right_GFR_original"] = df_val_raw["right_GFR"].to_numpy(dtype=float)

    split = {
        "method": "train_test_split(test_size=0.2, random_state=42, stratify=Center); one row per patient",
        "unit": "patient",
        **split_fingerprint(df_train, df_val),
    }
    return df_train, df_val, rad_features, preprocessor, split


def labels_from_gfr(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw GFR targets, SRF<15% labels and total GFR.

    The ``*_original`` columns are mandatory.  ``left_GFR`` / ``right_GFR`` in
    a transformed training table are model targets and must never be used for
    labels or reported true values.
    """
    gfr_columns = ["left_GFR_original", "right_GFR_original"]
    if not set(gfr_columns) <= set(df.columns):
        raise ValueError(
            "labels_from_gfr requires raw left_GFR_original/right_GFR_original columns"
        )
    y_reg = df[gfr_columns].to_numpy(dtype=float)
    if not np.isfinite(y_reg).all():
        raise ValueError("GFR targets must be finite and expressed in mL/min/1.73 m2")
    total = y_reg.sum(axis=1) + 1e-8
    y_cls = np.column_stack([(y_reg[:, 0] / total) < SRF_THRESHOLD, (y_reg[:, 1] / total) < SRF_THRESHOLD]).astype(int)
    return y_reg, y_cls, total


def calculate_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    cov = np.cov(y_true, y_pred)[0, 1]
    denom = np.var(y_true, ddof=1) + np.var(y_pred, ddof=1) + (true_mean - pred_mean) ** 2 + 1e-8
    return float(np.clip(2 * cov / denom, -1.0, 1.0))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def evaluate_predictions(y_reg: np.ndarray, y_cls: np.ndarray, pred_reg: np.ndarray, pred_prob: np.ndarray) -> dict[str, float]:
    left_ccc = calculate_ccc(y_reg[:, 0], pred_reg[:, 0])
    right_ccc = calculate_ccc(y_reg[:, 1], pred_reg[:, 1])
    auc_all = safe_auc(y_cls.reshape(-1), pred_prob.reshape(-1))
    return {
        "left_ccc": left_ccc,
        "right_ccc": right_ccc,
        "avg_ccc": (left_ccc + right_ccc) / 2.0,
        "min_ccc": min(left_ccc, right_ccc),
        "auc_srf15": auc_all,
        "left_mae": float(mean_absolute_error(y_reg[:, 0], pred_reg[:, 0])),
        "right_mae": float(mean_absolute_error(y_reg[:, 1], pred_reg[:, 1])),
    }


def save_predictions(outdir: Path, model_name: str, df_val: pd.DataFrame, y_reg: np.ndarray, y_cls: np.ndarray, pred_reg: np.ndarray, pred_prob: np.ndarray) -> None:
    pred = pd.DataFrame(
        {
            "case_id": df_val["ID"].astype(str).values,
            "left_true": y_reg[:, 0],
            "left_pred": pred_reg[:, 0],
            "right_true": y_reg[:, 1],
            "right_pred": pred_reg[:, 1],
            "left_cls_true": y_cls[:, 0],
            "left_cls_prob": pred_prob[:, 0],
            "right_cls_true": y_cls[:, 1],
            "right_cls_prob": pred_prob[:, 1],
        }
    )
    pred.to_csv(outdir / f"{model_name}_predictions.csv", index=False)


def fit_sidewise_regression(regressor, x_left_tr, x_right_tr, y_tr, x_left_va, x_right_va) -> np.ndarray:
    model_l = clone(regressor)
    model_r = clone(regressor)
    model_l.fit(x_left_tr, y_tr[:, 0])
    model_r.fit(x_right_tr, y_tr[:, 1])
    return np.column_stack([model_l.predict(x_left_va), model_r.predict(x_right_va)])


def fit_sidewise_classifier_models(classifier, x_left_tr, x_right_tr, y_cls_tr, x_left_va, x_right_va):
    """Fit one classifier per side and return (probabilities, fitted models)."""
    probs, models = [], []
    for side, x_tr, x_va in ((0, x_left_tr, x_left_va), (1, x_right_tr, x_right_va)):
        if len(np.unique(y_cls_tr[:, side])) < 2:
            probs.append(np.full(len(x_va), float(y_cls_tr[:, side].mean())))
            models.append(None)
            continue
        model = clone(classifier)
        model.fit(x_tr, y_cls_tr[:, side])
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(x_va)[:, 1])
        else:
            probs.append(expit(model.decision_function(x_va)))
        models.append(model)
    return np.column_stack(probs), models


def fit_sidewise_classifier(classifier, x_left_tr, x_right_tr, y_cls_tr, x_left_va, x_right_va) -> np.ndarray:
    probs, _ = fit_sidewise_classifier_models(
        classifier, x_left_tr, x_right_tr, y_cls_tr, x_left_va, x_right_va
    )
    return probs


def fit_sidewise_regression_models(regressor, x_left_tr, x_right_tr, y_tr, x_left_va, x_right_va):
    model_l = clone(regressor)
    model_r = clone(regressor)
    model_l.fit(x_left_tr, y_tr[:, 0])
    model_r.fit(x_right_tr, y_tr[:, 1])
    preds = np.column_stack([model_l.predict(x_left_va), model_r.predict(x_right_va)])
    return preds, {"left": model_l, "right": model_r}


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden, 2)
        self.cls_head = nn.Linear(hidden, 2)

    def forward(self, x):
        z = self.backbone(x)
        return self.reg_head(z), self.cls_head(z)


def train_mlp_ensemble(model_name: str, x_train: np.ndarray, x_val: np.ndarray, y_reg_tr: np.ndarray, y_reg_va: np.ndarray, y_cls_tr: np.ndarray, y_cls_va: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_reg_tr)
    x_train_s = x_scaler.transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)
    y_train_s = y_scaler.transform(y_reg_tr).astype(np.float32)

    pos = y_cls_tr.sum(axis=0).astype(float)
    neg = len(y_cls_tr) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device)

    seed_preds_reg = []
    seed_preds_prob = []
    seed_records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MultiTaskMLP(x_train_s.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        reg_loss_fn = nn.SmoothL1Loss()
        cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        ds = TensorDataset(
            torch.tensor(x_train_s),
            torch.tensor(y_train_s),
            torch.tensor(y_cls_tr.astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(ds, batch_size=32, shuffle=True, generator=generator)
        x_val_t = torch.tensor(x_val_s, dtype=torch.float32, device=device)

        best_score = -np.inf
        best_reg = None
        best_prob = None
        best_epoch = 0
        patience = 100
        stale = 0
        for epoch in range(600):
            model.train()
            for xb, yb_reg, yb_cls in loader:
                xb = xb.to(device)
                yb_reg = yb_reg.to(device)
                yb_cls = yb_cls.to(device)
                optimizer.zero_grad(set_to_none=True)
                out_reg, out_cls = model(xb)
                loss = reg_loss_fn(out_reg, yb_reg) + 0.20 * cls_loss_fn(out_cls, yb_cls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_reg_s, val_logits = model(x_val_t)
                val_reg = y_scaler.inverse_transform(val_reg_s.cpu().numpy())
                val_prob = torch.sigmoid(val_logits).cpu().numpy()
            metrics = evaluate_predictions(y_reg_va, y_cls_va, val_reg, val_prob)
            score = metrics["avg_ccc"]
            if score > best_score + 1e-5:
                best_score = score
                best_reg = val_reg.copy()
                best_prob = val_prob.copy()
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        seed_preds_reg.append(best_reg)
        seed_preds_prob.append(best_prob)
        seed_records.append({"model": model_name, "seed": seed, "best_epoch": best_epoch, "best_avg_ccc": best_score})

    return np.mean(seed_preds_reg, axis=0), np.mean(seed_preds_prob, axis=0), seed_records


def save_estimator_bundle(
    outdir: Path,
    model_name: str,
    fitted: dict,
    clinical_features: list[str],
    radiomics_features: list[str],
    preprocessor: TabularPreprocessor,
    split: dict,
) -> None:
    """Persist a fitted tabular baseline together with its preprocessor.

    ``fitted`` documents how to call the estimators for external inference:

    ``regressor_layout="both_sides_same_features"``
        ``fitted["regressor"]`` predicts both sides from one feature matrix
        (column 0 = left, column 1 = right).
    ``regressor_layout="side_specific_features"``
        ``fitted["regressor_left"]`` / ``fitted["regressor_right"]`` each take
        that side's own radiomics vector.

    Classifiers always follow the side-specific layout.  The imputation /
    scaling statistics travel with the estimators, because re-fitting them on
    the external cohort is not permitted.
    """
    import joblib

    bundle = {
        "model_name": model_name,
        "estimators": fitted,
        "clinical_features": clinical_features,
        "radiomics_features": radiomics_features,
        "preprocessing": preprocessor.state(),
        "split": split,
        "gfr_target": {
            "unit": "mL/min/1.73 m2",
            "regression_estimators": "fitted on original GFR values; predictions are returned in original units",
            "mlp": "target standardised once inside train_mlp_ensemble, then inverse-transformed",
            "labels_and_original_values": "computed from raw GFR values",
        },
        "checkpoint_origin": "retrained",
    }
    path = outdir / f"{model_name}_estimator.joblib"
    joblib.dump(bundle, path, compress=3)
    manifest = {
        "model_name": model_name,
        "estimator_file": path.name,
        "clinical_features": clinical_features,
        "radiomics_features": radiomics_features,
        "preprocessing_order": preprocessor.state()["preprocessing_order"],
        "preprocessing_schema": preprocessor.state()["schema_version"],
        "gfr_target_unit": "mL/min/1.73 m2",
        "gfr_target_handling": "raw labels/original values; model target standardised at most once",
        "n_train": split["n_train"],
        "n_validation": split["n_validation"],
        "split_fingerprint_train": split["train_fingerprint"],
        "checkpoint_origin": "retrained",
    }
    (outdir / f"{model_name}_estimator_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved estimator bundle: {path}")


def main() -> None:
    global OUT_DIR, CLINICAL_CSV_PATH, RADIOMICS_CSV_PATH, SELECTED_CSV, CENTER_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-csv", type=Path, required=True)
    parser.add_argument("--radiomics-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--center1-ct-dir", type=Path, required=True)
    parser.add_argument("--center1-mask-dir", type=Path, required=True)
    parser.add_argument("--center2-ct-dir", type=Path, required=True)
    parser.add_argument("--center2-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    CLINICAL_CSV_PATH = args.clinical_csv
    RADIOMICS_CSV_PATH = args.radiomics_csv
    SELECTED_CSV = args.selected_csv
    OUT_DIR = args.output_dir
    CENTER_CONFIGS = [
        {"name": "Center_1", "ct_dir": args.center1_ct_dir, "mask_dir": args.center1_mask_dir},
        {"name": "Center_2", "ct_dir": args.center2_ct_dir, "mask_dir": args.center2_mask_dir},
    ]

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train, df_val, rad_features, preprocessor, split_info = load_dataset()
    y_reg_tr, y_cls_tr, _ = labels_from_gfr(df_train)
    y_reg_va, y_cls_va, _ = labels_from_gfr(df_val)

    left_rad = rad_features[:6]
    right_rad = rad_features[6:]
    x_clin_tr = df_train[CLINICAL_COLS].to_numpy(dtype=float)
    x_clin_va = df_val[CLINICAL_COLS].to_numpy(dtype=float)
    x_rad_l_tr = df_train[left_rad].to_numpy(dtype=float)
    x_rad_r_tr = df_train[right_rad].to_numpy(dtype=float)
    x_rad_l_va = df_val[left_rad].to_numpy(dtype=float)
    x_rad_r_va = df_val[right_rad].to_numpy(dtype=float)
    x_rad_tr = df_train[rad_features].to_numpy(dtype=float)
    x_rad_va = df_val[rad_features].to_numpy(dtype=float)
    x_clin_rad_tr = np.concatenate([x_clin_tr, x_rad_tr], axis=1)
    x_clin_rad_va = np.concatenate([x_clin_va, x_rad_va], axis=1)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clinical_features": CLINICAL_COLS,
        "radiomics_features": rad_features,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "seeds_for_mlp": SEEDS,
        "xgboost_available": HAS_XGBOOST,
        "hist_gradient_boosting_note": "HistGradientBoosting is retained as an additional sklearn gradient-boosted tree baseline.",
        "preprocessing": preprocessor.state(),
        "gfr_target": {
            "unit": "mL/min/1.73 m2",
            "labels_and_original_values": "computed from raw GFR values",
            "regression_estimators": "original GFR units",
            "mlp": "target standardised once and inverse-transformed before evaluation",
        },
        "split": split_info,
        # Imputation now precedes standardisation, so previously released
        # tabular outputs cannot be reproduced by this script.
        "checkpoint_origin": "retrained",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    df_train[["ID", "Center", "left_GFR_original", "right_GFR_original"]].rename(
        columns={"left_GFR_original": "left_GFR", "right_GFR_original": "right_GFR"}
    ).to_csv(OUT_DIR / "train_cases.csv", index=False)
    df_val[["ID", "Center", "left_GFR_original", "right_GFR_original"]].rename(
        columns={"left_GFR_original": "left_GFR", "right_GFR_original": "right_GFR"}
    ).to_csv(OUT_DIR / "validation_cases.csv", index=False)

    records = []
    seed_records = []

    def record(model_name: str, family: str, pred_reg: np.ndarray, pred_prob: np.ndarray):
        metrics = evaluate_predictions(y_reg_va, y_cls_va, pred_reg, pred_prob)
        records.append({"model": model_name, "family": family, **metrics})
        save_predictions(OUT_DIR, model_name, df_val, y_reg_va, y_cls_va, pred_reg, pred_prob)
        print(f"{model_name:36s} AvgCCC={metrics['avg_ccc']:.4f} AUC={metrics['auc_srf15']:.4f}")

    # 1. Statistical clinical baselines.
    ridge_reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 40)))
    ridge_reg.fit(x_clin_tr, y_reg_tr)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"))
    pred_prob, cls_models = fit_sidewise_classifier_models(
        logit, x_clin_tr, x_clin_tr, y_cls_tr, x_clin_va, x_clin_va
    )
    record("clinical_ridge_logistic", "statistical_clinical", ridge_reg.predict(x_clin_va), pred_prob)
    save_estimator_bundle(
        OUT_DIR,
        "clinical_ridge_logistic",
        {
            "regressor_layout": "both_sides_same_features",
            "regressor": ridge_reg,
            "classifier_left": cls_models[0],
            "classifier_right": cls_models[1],
        },
        CLINICAL_COLS,
        [],
        preprocessor,
        split_info,
    )

    enet_reg = MultiOutputRegressor(
        make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-4, 1, 60), cv=5, max_iter=20000),
        )
    )
    enet_reg.fit(x_clin_tr, y_reg_tr)
    record("clinical_elasticnet_logistic", "statistical_clinical", enet_reg.predict(x_clin_va), pred_prob)

    # 2. Machine-learning radiomics baselines.
    rad_enet = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-4, 1, 60), cv=5, max_iter=20000),
    )
    rad_logit = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=10,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.1, 0.5, 0.9],
            cv=5,
            max_iter=10000,
            class_weight="balanced",
            scoring="roc_auc",
        ),
    )
    pred_reg = fit_sidewise_regression(rad_enet, x_rad_l_tr, x_rad_r_tr, y_reg_tr, x_rad_l_va, x_rad_r_va)
    pred_prob = fit_sidewise_classifier(rad_logit, x_rad_l_tr, x_rad_r_tr, y_cls_tr, x_rad_l_va, x_rad_r_va)
    record("radiomics_elasticnet", "ml_radiomics", pred_reg, pred_prob)

    if HAS_XGBOOST:
        xgb_reg = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
        )
        xgb_cls = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
        )
        pred_reg = fit_sidewise_regression(xgb_reg, x_rad_l_tr, x_rad_r_tr, y_reg_tr, x_rad_l_va, x_rad_r_va)
        pred_prob = fit_sidewise_classifier(xgb_cls, x_rad_l_tr, x_rad_r_tr, y_cls_tr, x_rad_l_va, x_rad_r_va)
        record("radiomics_xgboost", "ml_radiomics", pred_reg, pred_prob)

    rf_reg = RandomForestRegressor(n_estimators=800, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf_cls = RandomForestClassifier(n_estimators=800, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    pred_reg, rf_reg_models = fit_sidewise_regression_models(
        rf_reg, x_rad_l_tr, x_rad_r_tr, y_reg_tr, x_rad_l_va, x_rad_r_va
    )
    pred_prob, rf_cls_models = fit_sidewise_classifier_models(
        rf_cls, x_rad_l_tr, x_rad_r_tr, y_cls_tr, x_rad_l_va, x_rad_r_va
    )
    record("radiomics_random_forest", "ml_radiomics", pred_reg, pred_prob)
    save_estimator_bundle(
        OUT_DIR,
        "radiomics_random_forest",
        {
            "regressor_layout": "side_specific_features",
            "regressor_left": rf_reg_models["left"],
            "regressor_right": rf_reg_models["right"],
            "classifier_left": rf_cls_models[0],
            "classifier_right": rf_cls_models[1],
        },
        CLINICAL_COLS,
        rad_features,
        preprocessor,
        split_info,
    )

    hgb_reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, l2_regularization=0.05, random_state=42)
    hgb_cls = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, l2_regularization=0.05, random_state=42)
    pred_reg = fit_sidewise_regression(hgb_reg, x_rad_l_tr, x_rad_r_tr, y_reg_tr, x_rad_l_va, x_rad_r_va)
    pred_prob = fit_sidewise_classifier(hgb_cls, x_rad_l_tr, x_rad_r_tr, y_cls_tr, x_rad_l_va, x_rad_r_va)
    record("radiomics_hist_gradient_boosting", "ml_radiomics", pred_reg, pred_prob)

    # 3. Tabular dual-task neural baselines.
    for model_name, xtr, xva in [
        ("mlp_clinical_only", x_clin_tr, x_clin_va),
        ("mlp_radiomics_only", x_rad_tr, x_rad_va),
        ("mlp_clinical_radiomics", x_clin_rad_tr, x_clin_rad_va),
    ]:
        pred_reg, pred_prob, sr = train_mlp_ensemble(model_name, xtr, xva, y_reg_tr, y_reg_va, y_cls_tr, y_cls_va)
        seed_records.extend(sr)
        record(model_name, "tabular_dual_task_mlp", pred_reg, pred_prob)

    summary = pd.DataFrame(records).sort_values(["family", "avg_ccc"], ascending=[True, False])
    summary.to_csv(OUT_DIR / "tabular_baseline_metrics.csv", index=False)
    pd.DataFrame(seed_records).to_csv(OUT_DIR / "mlp_seed_metrics.csv", index=False)
    print("\nSaved to:", OUT_DIR)
    print(summary.to_string(index=False))
    print(f"Elapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
