"""
Network-adaptive paired radiomics feature selection.

This selector is designed for the current CT + clinical/radiomics MTL networks:
- radiomics encoder expects 6 left features and 6 right features;
- left/right feature semantics should be paired and ordered identically;
- selection must use only the internal training split.

The scoring target is clinical-incremental split GFR signal, with SRF<15%
classification value as a secondary term. External centers are intentionally
not loaded or used.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


CSV_CLIN_COLS = ["height", "weight", "sex", "age", "SCr"]

INTERNAL_CENTERS = ["Center_1", "Center_2"]
SRF_THRESHOLD = 0.15
RANDOM_STATE = 42
DEFAULT_MAX_BASE_FEATURES = 6


def resolve_column(df, aliases, role):
    for col in aliases:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find {role}. Tried aliases: {aliases}")


def find_existing_image_pair(pid, center, path_map):
    ct_dir = path_map[center]["ct"]
    mask_dir = path_map[center]["mask"]

    ct_path = ct_dir / f"{pid}.nii.gz"
    if not ct_path.exists():
        ct_path = ct_dir / f"{pid}.nii"

    mask_path = mask_dir / f"{pid}.nii.gz"
    if not mask_path.exists():
        mask_path = mask_dir / f"{pid}.nii"

    if ct_path.exists() and mask_path.exists():
        return str(ct_path), str(mask_path)
    return None, None


def load_internal_train_dataframe(args, check_image_paths=True):
    df_clin = pd.read_csv(args.clinical_csv)
    df_rad = pd.read_csv(args.radiomics_csv)
    df = pd.merge(df_clin, df_rad, on="ID", how="inner")
    df = df[df["Center"].isin(INTERNAL_CENTERS)].copy().reset_index(drop=True)

    left_gfr_col = resolve_column(df, ["left_GFR", "Left_GFR", "left_gfr", "Left_gfr"], "left GFR")
    right_gfr_col = resolve_column(df, ["right_GFR", "Right_GFR", "right_gfr", "Right_gfr"], "right GFR")
    if left_gfr_col != "left_GFR":
        df["left_GFR"] = df[left_gfr_col]
    if right_gfr_col != "right_GFR":
        df["right_GFR"] = df[right_gfr_col]

    if check_image_paths:
        image_dirs = (
            args.center1_ct_dir,
            args.center1_mask_dir,
            args.center2_ct_dir,
            args.center2_mask_dir,
        )
        if any(path is None for path in image_dirs):
            raise ValueError(
                "CT and mask directories for both development centers are required unless "
                "--no-path-check is used."
            )
        path_map = {
            "Center_1": {"ct": args.center1_ct_dir, "mask": args.center1_mask_dir},
            "Center_2": {"ct": args.center2_ct_dir, "mask": args.center2_mask_dir},
        }
        valid_indices = []
        for idx, row in df.iterrows():
            pid = str(row["ID"]).strip()
            ct_path, mask_path = find_existing_image_pair(pid, row["Center"], path_map)
            if ct_path is not None and mask_path is not None:
                valid_indices.append(idx)
        df = df.loc[valid_indices].copy()

    df_train, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["Center"],
    )
    df_train = df_train.reset_index(drop=True)

    print(f"Internal valid cases: {len(df)}")
    print(f"Feature selection uses internal train only: {len(df_train)}")
    print("External centers are not loaded and not used.")
    return df_train


def is_valid_base_name(base):
    bad_exact = {"GFR", "gfr", "left_GFR", "right_GFR", "Left_GFR", "Right_GFR"}
    bad_fragments = ("diagnostic", "diagnostics")
    if base in bad_exact:
        return False
    lower = base.lower()
    return not any(fragment in lower for fragment in bad_fragments)


def get_paired_base_features(df):
    left_cols = [c for c in df.columns if c.startswith("Left_")]
    base_names = []
    for left_col in left_cols:
        base = left_col.replace("Left_", "", 1)
        right_col = f"Right_{base}"
        if right_col not in df.columns or not is_valid_base_name(base):
            continue
        base_names.append(base)
    return sorted(set(base_names))


def safe_spearman(x, y):
    try:
        rho = stats.spearmanr(x, y, nan_policy="omit")[0]
        return 0.0 if np.isnan(rho) else float(rho)
    except Exception:
        return 0.0


def safe_auc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        auc = roc_auc_score(y_true, scores)
        return float(abs(auc - 0.5) * 2.0)
    except Exception:
        return 0.0


def robust_signed_mean(corrs):
    corrs = np.asarray([c for c in corrs if np.isfinite(c)], dtype=np.float64)
    if corrs.size == 0:
        return 0.0, 0.0
    mean_abs = float(np.mean(np.abs(corrs)))
    sign_consistency = float(abs(np.mean(corrs)) / (mean_abs + 1e-8))
    return mean_abs, np.clip(sign_consistency, 0.0, 1.0)


def rank_to_unit_interval(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    if np.nanmax(values) <= np.nanmin(values):
        return np.ones_like(values) * 0.5
    ranks = pd.Series(values).rank(method="average", pct=True).values
    return np.asarray(ranks, dtype=np.float64)


def prepare_numeric_train_frame(df_train, base_names, max_missing_rate):
    needed_cols = (
        CSV_CLIN_COLS
        + ["left_GFR", "right_GFR"]
        + [f"Left_{base}" for base in base_names]
        + [f"Right_{base}" for base in base_names]
    )

    df = df_train.copy()
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    retained = []
    missing_rows = []
    for base in base_names:
        left_col, right_col = f"Left_{base}", f"Right_{base}"
        missing_rate = float(df[[left_col, right_col]].isna().mean().mean())
        if missing_rate > max_missing_rate:
            missing_rows.append((base, missing_rate, "high_missing"))
            continue

        left_var = float(df[left_col].var(skipna=True))
        right_var = float(df[right_col].var(skipna=True))
        if not np.isfinite(left_var) or not np.isfinite(right_var) or min(left_var, right_var) <= 1e-12:
            missing_rows.append((base, missing_rate, "near_zero_variance"))
            continue
        retained.append(base)

    feature_cols = [f"{side}_{base}" for side in ("Left", "Right") for base in retained]
    all_numeric_cols = CSV_CLIN_COLS + ["left_GFR", "right_GFR"] + feature_cols
    medians = df[all_numeric_cols].median(numeric_only=True)
    df[all_numeric_cols] = df[all_numeric_cols].fillna(medians)

    return df, retained, pd.DataFrame(missing_rows, columns=["Base_Name", "MissingRate", "FilterReason"])


def center_standardize_radiomics(df, base_names):
    df_scaled = df.copy()
    feature_cols = [f"{side}_{base}" for side in ("Left", "Right") for base in base_names]
    for center in INTERNAL_CENTERS:
        mask = df_scaled["Center"] == center
        if mask.sum() == 0:
            continue
        scaler = StandardScaler()
        df_scaled.loc[mask, feature_cols] = scaler.fit_transform(df_scaled.loc[mask, feature_cols])
    return df_scaled


def make_kidney_level_table(df, base_names):
    total_gfr = df["left_GFR"].values + df["right_GFR"].values + 1e-8
    left_srf = df["left_GFR"].values / total_gfr
    right_srf = df["right_GFR"].values / total_gfr

    left_rows = pd.DataFrame(
        {
            "PatientID": df["ID"].astype(str).values,
            "Center": df["Center"].values,
            "Side": "Left",
            "split_GFR": df["left_GFR"].values,
            "srf": left_srf,
            "srf_lt15": (left_srf < SRF_THRESHOLD).astype(int),
        }
    )
    right_rows = pd.DataFrame(
        {
            "PatientID": df["ID"].astype(str).values,
            "Center": df["Center"].values,
            "Side": "Right",
            "split_GFR": df["right_GFR"].values,
            "srf": right_srf,
            "srf_lt15": (right_srf < SRF_THRESHOLD).astype(int),
        }
    )

    for col in CSV_CLIN_COLS:
        left_rows[col] = df[col].values
        right_rows[col] = df[col].values

    for base in base_names:
        left_rows[base] = df[f"Left_{base}"].values
        right_rows[base] = df[f"Right_{base}"].values

    return pd.concat([left_rows, right_rows], axis=0, ignore_index=True)


def _clinical_design_matrix(kidney_df, columns=None):
    center_dummies = pd.get_dummies(kidney_df["Center"], prefix="Center", drop_first=True)
    side_dummies = pd.get_dummies(kidney_df["Side"], prefix="Side", drop_first=True)
    x = pd.concat(
        [
            kidney_df[CSV_CLIN_COLS].reset_index(drop=True),
            center_dummies.reset_index(drop=True),
            side_dummies.reset_index(drop=True),
        ],
        axis=1,
    ).astype(float)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def _fit_predict_clinical(train_x, train_y, test_x):
    """Median-impute + standardise on the training fold, then ridge-predict."""
    medians = train_x.median()
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    scaler = StandardScaler().fit(train_x)
    model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x))


def clinical_residualize(kidney_df, method="oof", n_splits=5, random_state=RANDOM_STATE):
    """Remove the clinical/centre/side signal from split GFR.

    ``method="oof"`` (default) produces genuine out-of-fold predictions with
    ``GroupKFold`` on the patient identifier, so both kidneys of a patient are
    always in the same fold and no row is predicted by a model that saw it.
    ``method="in_sample"`` keeps the previous behaviour for explicit diagnostic
    use only; it must not be described as out-of-fold.
    """
    columns = _clinical_design_matrix(kidney_df).columns
    x_all = _clinical_design_matrix(kidney_df, columns)
    y = kidney_df["split_GFR"].to_numpy(dtype=float)
    groups = kidney_df["PatientID"].astype(str).to_numpy()

    if method == "in_sample":
        medians = x_all.median()
        filled = x_all.fillna(medians)
        scaler = StandardScaler().fit(filled)
        model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler.transform(filled), y)
        pred = model.predict(scaler.transform(filled))
        return y - pred, float(model.score(scaler.transform(filled), y)), {
            "method": "in_sample",
            "note": "diagnostic only; NOT out-of-fold",
        }

    if method != "oof":
        raise ValueError(f"unsupported residualisation method: {method}")

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"GroupKFold needs at least {n_splits} patients, found {len(unique_groups)}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    pred = np.full(len(kidney_df), np.nan, dtype=float)
    fold_sizes = []
    row_folds = np.full(len(kidney_df), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_all, y, groups=groups)):
        pred[test_idx] = _fit_predict_clinical(x_all.iloc[train_idx], y[train_idx], x_all.iloc[test_idx])
        fold_sizes.append(int(len(test_idx)))
        row_folds[test_idx] = fold
    if not np.isfinite(pred).all():
        raise RuntimeError("out-of-fold clinical predictions contain missing rows")

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    info = {
        "method": "out_of_fold",
        "splitter": "GroupKFold",
        "grouping": "PatientID (both kidneys always in the same fold)",
        "n_splits": int(n_splits),
        "random_state": None,
        "seed_note": "GroupKFold is deterministic; no shuffling is requested.",
        "configured_seed": int(random_state),
        "fold_rows": fold_sizes,
        "row_folds": row_folds.tolist(),
    }
    return y - pred, float(r2), info


def bootstrap_elastic_net_stability(kidney_df, base_names, residual, n_bootstraps, random_state):
    x_all = kidney_df[base_names].values.astype(np.float64)
    y_all = np.asarray(residual, dtype=np.float64)
    patient_ids = kidney_df["PatientID"].values
    unique_patients = np.unique(patient_ids)
    rng = np.random.default_rng(random_state)

    hit_count = np.zeros(len(base_names), dtype=np.float64)
    coef_sum = np.zeros(len(base_names), dtype=np.float64)
    valid_runs = 0

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        row_indices = []
        for pid in sampled_patients:
            row_indices.extend(np.where(patient_ids == pid)[0].tolist())
        row_indices = np.asarray(row_indices, dtype=int)

        x_boot = x_all[row_indices]
        y_boot = y_all[row_indices]
        if np.nanstd(y_boot) <= 1e-8:
            continue

        try:
            model = ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.8, 1.0],
                alphas=None,
                cv=5,
                max_iter=10000,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(x_boot, y_boot)
        except Exception:
            continue

        coefs = np.abs(model.coef_)
        threshold = max(1e-6, np.nanmax(coefs) * 0.01)
        hit_count += (coefs > threshold).astype(float)
        coef_sum += coefs
        valid_runs += 1

    denom = max(valid_runs, 1)
    return hit_count / denom, coef_sum / denom, valid_runs


def score_features(kidney_df, base_names, residual, stability, mean_abs_coef):
    rows = []
    y_cls = kidney_df["srf_lt15"].values.astype(int)

    for i, base in enumerate(base_names):
        x = kidney_df[base].values.astype(float)
        corr_resid = safe_spearman(x, residual)
        corr_gfr = safe_spearman(x, kidney_df["split_GFR"].values)
        srf_auc_signal = safe_auc_score(y_cls, x)

        center_corrs = []
        for center in INTERNAL_CENTERS:
            mask = kidney_df["Center"].values == center
            center_corrs.append(safe_spearman(x[mask], residual[mask]))
        center_mean_abs, center_sign_consistency = robust_signed_mean(center_corrs)
        center_robust = center_mean_abs * center_sign_consistency

        side_corrs = []
        for side in ("Left", "Right"):
            mask = kidney_df["Side"].values == side
            side_corrs.append(safe_spearman(x[mask], residual[mask]))
        side_mean_abs, side_sign_consistency = robust_signed_mean(side_corrs)
        side_robust = side_mean_abs * side_sign_consistency

        clinical_corrs = [abs(safe_spearman(x, kidney_df[col].values)) for col in CSV_CLIN_COLS]
        max_clinical_corr = float(np.max(clinical_corrs)) if clinical_corrs else 0.0
        scr_corr = abs(safe_spearman(x, kidney_df["SCr"].values)) if "SCr" in kidney_df else 0.0

        rows.append(
            {
                "Base_Name": base,
                "Stability": float(stability[i]),
                "MeanAbsCoef": float(mean_abs_coef[i]),
                "AbsCorrResidual": abs(corr_resid),
                "SignedCorrResidual": corr_resid,
                "AbsCorrRawGFR": abs(corr_gfr),
                "SRF15Signal": srf_auc_signal,
                "CenterMeanAbsCorr": center_mean_abs,
                "CenterSignConsistency": center_sign_consistency,
                "CenterRobustCorr": center_robust,
                "SideMeanAbsCorr": side_mean_abs,
                "SideSignConsistency": side_sign_consistency,
                "SideRobustCorr": side_robust,
                "MaxAbsCorrClinical": max_clinical_corr,
                "AbsCorrSCr": scr_corr,
            }
        )

    scores = pd.DataFrame(rows)
    scores["CoefRank"] = rank_to_unit_interval(scores["MeanAbsCoef"].values)
    scores["RawScore"] = (
        0.30 * scores["Stability"]
        + 0.20 * scores["AbsCorrResidual"]
        + 0.15 * scores["CenterRobustCorr"]
        + 0.10 * scores["SideRobustCorr"]
        + 0.10 * scores["SRF15Signal"]
        + 0.10 * scores["CoefRank"]
        + 0.05 * scores["AbsCorrRawGFR"]
        - 0.10 * scores["MaxAbsCorrClinical"]
    )
    return scores.sort_values("RawScore", ascending=False).reset_index(drop=True)


def greedy_redundancy_prune(kidney_df, ranked_scores, max_features, corr_threshold, redundancy_weight):
    selected = []
    selected_rows = []

    while len(selected) < max_features:
        best_base = None
        best_adjusted_score = -np.inf
        best_max_corr = 0.0

        for _, row in ranked_scores.iterrows():
            base = row["Base_Name"]
            if base in selected:
                continue

            if selected:
                corr_to_selected = [
                    abs(safe_spearman(kidney_df[base].values, kidney_df[kept].values))
                    for kept in selected
                ]
                max_corr = float(np.max(corr_to_selected))
            else:
                max_corr = 0.0

            if max_corr >= corr_threshold:
                continue

            adjusted_score = float(row["RawScore"]) - redundancy_weight * max_corr
            if adjusted_score > best_adjusted_score:
                best_base = base
                best_adjusted_score = adjusted_score
                best_max_corr = max_corr

        if best_base is None:
            break

        selected.append(best_base)
        selected_row = ranked_scores[ranked_scores["Base_Name"] == best_base].iloc[0].to_dict()
        selected_row["AdjustedScore"] = best_adjusted_score
        selected_row["MaxCorrWithEarlierSelected"] = best_max_corr
        selected_row["SelectionOrder"] = len(selected)
        selected_rows.append(selected_row)

    if len(selected) < max_features:
        for _, row in ranked_scores.iterrows():
            base = row["Base_Name"]
            if base in selected:
                continue
            selected.append(base)
            selected_row = row.to_dict()
            selected_row["AdjustedScore"] = float(row["RawScore"])
            selected_row["MaxCorrWithEarlierSelected"] = np.nan
            selected_row["SelectionOrder"] = len(selected)
            selected_rows.append(selected_row)
            if len(selected) >= max_features:
                break

    return selected, pd.DataFrame(selected_rows)


def print_rad_features(selected_bases):
    selected_features = [f"Left_{base}" for base in selected_bases]
    selected_features += [f"Right_{base}" for base in selected_bases]

    print("\nSelected paired base features:")
    for i, base in enumerate(selected_bases, start=1):
        print(f"  {i}. {base}")

    print("\nRAD_FEATURES = [")
    for feat in selected_features:
        print(f'    "{feat}",')
    print("]")
    return selected_features


def run_selection(args):
    df_train = load_internal_train_dataframe(args, check_image_paths=not args.no_path_check)
    base_names = get_paired_base_features(df_train)
    print(f"Paired numeric radiomics candidates before filters: {len(base_names)}")

    df_train, base_names, filtered = prepare_numeric_train_frame(
        df_train,
        base_names,
        max_missing_rate=args.max_missing_rate,
    )
    print(f"Candidates after missingness/variance filters: {len(base_names)}")
    if len(base_names) < args.max_base_features:
        raise RuntimeError(f"Only {len(base_names)} paired candidates remain; cannot select {args.max_base_features}.")

    df_train = center_standardize_radiomics(df_train, base_names)
    kidney_df = make_kidney_level_table(df_train, base_names)
    residual, clinical_r2, residual_info = clinical_residualize(
        kidney_df,
        method=args.residual_mode,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    kidney_df["clinical_residual_split_GFR"] = residual

    print(f"Kidney-level rows for selection: {len(kidney_df)}")
    print(f"Clinical + center + side baseline R2 ({residual_info['method']}): {clinical_r2:.4f}")

    stability, mean_abs_coef, valid_bootstraps = bootstrap_elastic_net_stability(
        kidney_df,
        base_names,
        residual,
        n_bootstraps=args.n_bootstraps,
        random_state=args.seed,
    )
    print(f"Valid bootstrap ElasticNet runs: {valid_bootstraps}/{args.n_bootstraps}")

    scores = score_features(kidney_df, base_names, residual, stability, mean_abs_coef)
    selected_bases, selected_scores = greedy_redundancy_prune(
        kidney_df,
        scores,
        max_features=args.max_base_features,
        corr_threshold=args.corr_threshold,
        redundancy_weight=args.redundancy_weight,
    )

    timestamp = time.strftime("%Y%m%d_%H%M")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ranking_path = outdir / f"network_adaptive_radiomics_ranking_{timestamp}.csv"
    selected_path = outdir / f"network_adaptive_radiomics_selected_{timestamp}.csv"
    filtered_path = outdir / f"network_adaptive_radiomics_filtered_{timestamp}.csv"

    scores.to_csv(ranking_path, index=False)
    selected_scores.to_csv(selected_path, index=False)
    if not filtered.empty:
        filtered.to_csv(filtered_path, index=False)

    selected_features = print_rad_features(selected_bases)

    selection_manifest = {
        "selection_target": "split GFR residualised on clinical variables, centre and side",
        "residualisation": residual_info,
        "elastic_net_bootstrap": {
            "n_bootstraps": int(args.n_bootstraps),
            "valid_bootstraps": int(valid_bootstraps),
            "resampling_unit": "patient",
            "seed": int(args.seed),
        },
        "selected_base_features": selected_bases,
        "selected_features": selected_features,
        "n_train_patients": int(kidney_df["PatientID"].nunique()),
        "n_train_kidneys": int(len(kidney_df)),
        "external_centres_used": False,
    }
    (outdir / "radiomics_selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )

    print(f"\nRanking saved: {ranking_path}")
    print(f"Selected-feature audit saved: {selected_path}")
    if not filtered.empty:
        print(f"Filtered-feature audit saved: {filtered_path}")

    return selected_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select paired radiomics features adapted to the current GFR MTL networks."
    )
    parser.add_argument("--clinical-csv", type=Path, required=True)
    parser.add_argument("--radiomics-csv", type=Path, required=True)
    parser.add_argument("--center1-ct-dir", type=Path)
    parser.add_argument("--center1-mask-dir", type=Path)
    parser.add_argument("--center2-ct-dir", type=Path)
    parser.add_argument("--center2-mask-dir", type=Path)
    parser.add_argument("--max-base-features", type=int, default=DEFAULT_MAX_BASE_FEATURES)
    parser.add_argument("--n-bootstraps", type=int, default=150)
    parser.add_argument("--max-missing-rate", type=float, default=0.20)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--redundancy-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--residual-mode",
        choices=["oof", "in_sample"],
        default="oof",
        help="oof = patient-grouped out-of-fold clinical residual (default); "
             "in_sample is a diagnostic-only alternative that must not be labelled OOF.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold folds for the OOF residual.")
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="Skip CT/mask existence filtering. Use only for quick CSV-side debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_selection(parse_args())
