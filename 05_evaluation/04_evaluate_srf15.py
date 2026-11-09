#!/usr/bin/env python3
"""Evaluate locked DFME SRF<15% probabilities using patient-level bootstrap."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from utils import CLASSIFICATION_METRICS, classification_metrics, configure_logging, load_config, output_dir, write_environment_manifest, write_xlsx


def fast_metrics(y, p, patient_ids, threshold):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); eps=1e-6; logits=np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps))
    def objective(beta):
        q=expit(beta[0]+beta[1]*logits); return -np.sum(y*np.log(q+1e-12)+(1-y)*np.log(1-q+1e-12))
    fit=minimize(objective,[0.,1.],method="BFGS"); intercept,slope=(fit.x if fit.success else [np.nan,np.nan])
    return {"n_patients":len(np.unique(patient_ids)),"n_kidneys":len(y),"positive_kidneys":int(y.sum()),"positive_patients":int(pd.DataFrame({"id":patient_ids,"y":y}).groupby("id").y.max().sum()),"prevalence":y.mean(),"auroc":roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,"auprc":average_precision_score(y,p) if y.sum() else np.nan,"sensitivity":recall_score(y,pred,zero_division=0),"specificity":tn/(tn+fp) if tn+fp else np.nan,"ppv":precision_score(y,pred,zero_division=0),"npv":tn/(tn+fn) if tn+fn else np.nan,"f1":f1_score(y,pred,zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,pred),"brier":brier_score_loss(y,p),"calibration_intercept":intercept,"calibration_slope":slope,"threshold":threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    logger = configure_logging(config, "04_evaluate_srf15")
    case_path = output_dir(config, "case_level") / "locked_case_level_predictions.csv"
    data = pd.read_csv(case_path)
    tables, source = output_dir(config, "tables"), output_dir(config, "source_data")
    rows, replicate_parts = [], []
    threshold = float(config["classification_threshold"])
    for split in ("internal", "external"):
        cohort = data[data.split == split]
        for side_index, side in enumerate(("left", "right", "pooled")):
            block = cohort if side == "pooled" else cohort[cohort.side == side]
            block=block.dropna(subset=["true_srf_lt15","predicted_probability_srf_lt15"]).reset_index(drop=True); y=block.true_srf_lt15.astype(int).to_numpy(); p=block.predicted_probability_srf_lt15.to_numpy(float); ids=block.anonymized_patient_id.to_numpy(); estimate=fast_metrics(y,p,ids,threshold)
            unique=pd.unique(ids); groups=[np.flatnonzero(ids==patient) for patient in unique]; rng=np.random.default_rng(int(config["random_seed"]) + 70000 + side_index*1000 + (0 if split == "internal" else 10000)); rep_rows=[]
            for replicate in range(int(config["bootstrap_replicates"])):
                idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); result=fast_metrics(y[idx],p[idx],ids[idx],threshold); rep_rows.append({"replicate":replicate,**{m:result[m] for m in CLASSIFICATION_METRICS}})
            reps=pd.DataFrame(rep_rows); intervals={m:(reps[m].quantile(.025),reps[m].quantile(.975)) for m in CLASSIFICATION_METRICS}
            record = {"cohort": split, "model": "Dual-Fusion Multimodal Ensemble (DFME)", "side": side, **estimate}
            for metric in CLASSIFICATION_METRICS:
                record[f"{metric}_ci_low"], record[f"{metric}_ci_high"] = intervals[metric]
            rows.append(record)
            reps.insert(0, "cohort", split); reps.insert(1, "side", side); replicate_parts.append(reps)
    result = pd.DataFrame(rows)
    for split in ("internal", "external"):
        block = result[result.cohort == split]
        block.to_csv(tables / f"Table_SRF15_performance_{split}.csv", index=False)
        write_xlsx(block, tables / f"Table_SRF15_performance_{split}.xlsx")
    result.to_csv(tables / "Table_SRF15_performance_all.csv", index=False); write_xlsx(result, tables / "Table_SRF15_performance_all.xlsx")
    pd.concat(replicate_parts, ignore_index=True).to_csv(source / "srf15_bootstrap_replicates.csv.gz", index=False, compression="gzip")
    write_environment_manifest(config, [case_path], "04_evaluate_srf15")
    logger.info("SRF15 evaluation complete; threshold fixed at %.2f", threshold)


if __name__ == "__main__":
    main()
