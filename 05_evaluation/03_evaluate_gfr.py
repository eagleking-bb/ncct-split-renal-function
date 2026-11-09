#!/usr/bin/env python3
"""Evaluate continuous GFR using fast patient-index cluster bootstrap."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from utils import MODEL_ORDER, ccc, configure_logging, load_config, output_dir, write_environment_manifest, write_xlsx


METRICS=["mae","median_absolute_error","rmse","bias","r2","pearson_r","spearman_rho","ccc","bland_altman_bias","bland_altman_loa_lower","bland_altman_loa_upper","calibration_intercept","calibration_slope"]
PRED={"clinical_only":"predicted_gfr_clinical_only","radiomics_only":"predicted_gfr_radiomics_only","image_only":"predicted_gfr_image_only","early_concat":"predicted_gfr_early_concat","cross_attention":"predicted_gfr_cross_attention","dfme":"predicted_gfr_dfme"}


def metrics(y,p,n_patients):
    e=p-y; sd=e.std(ddof=1); slope,intercept=np.polyfit(p,y,1) if np.unique(p).size>1 else (np.nan,np.nan)
    return {"n_patients":n_patients,"n_kidneys":len(y),"mae":np.mean(np.abs(e)),"median_absolute_error":np.median(np.abs(e)),"rmse":np.sqrt(np.mean(e*e)),"bias":np.mean(e),"r2":r2_score(y,p),"pearson_r":stats.pearsonr(y,p).statistic,"spearman_rho":stats.spearmanr(y,p).statistic,"ccc":ccc(y,p),"bland_altman_bias":np.mean(e),"bland_altman_loa_lower":np.mean(e)-1.96*sd,"bland_altman_loa_upper":np.mean(e)+1.96*sd,"calibration_intercept":intercept,"calibration_slope":slope}


def cluster_indices(frame,n_boot,seed):
    patients=frame.anonymized_patient_id.drop_duplicates().to_numpy(); groups=[np.flatnonzero(frame.anonymized_patient_id.to_numpy()==pid) for pid in patients]
    rng=np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])


def perf_boot(frame,pred,n_boot,seed):
    block=frame.dropna(subset=["true_side_gfr",pred]).reset_index(drop=True); y=block.true_side_gfr.to_numpy(float); p=block[pred].to_numpy(float); n=block.anonymized_patient_id.nunique()
    estimate=metrics(y,p,n); rows=[]
    for replicate,idx in enumerate(cluster_indices(block,n_boot,seed)):
        result=metrics(y[idx],p[idx],n); rows.append({"replicate":replicate,**{m:result[m] for m in METRICS}})
    reps=pd.DataFrame(rows); ci={m:(reps[m].quantile(.025),reps[m].quantile(.975)) for m in METRICS}; return estimate,ci,reps


def paired_boot(frame,a,b,n_boot,seed):
    block=frame.dropna(subset=["true_side_gfr",a,b]).reset_index(drop=True); y=block.true_side_gfr.to_numpy(float); pa=block[a].to_numpy(float); pb=block[b].to_numpy(float); n=block.anonymized_patient_id.nunique()
    ma,mb=metrics(y,pa,n),metrics(y,pb,n); rows=[]
    for replicate,idx in enumerate(cluster_indices(block,n_boot,seed)):
        xa,xb=metrics(y[idx],pa[idx],n),metrics(y[idx],pb[idx],n); rows.append({"replicate":replicate,**{f"delta_{m}":xa[m]-xb[m] for m in ("mae","rmse","ccc")}})
    reps=pd.DataFrame(rows); summary=[]
    for m in ("mae","rmse","ccc"):
        vals=reps[f"delta_{m}"]; summary.append({"metric":m,"estimate_model_a":ma[m],"estimate_model_b":mb[m],"delta_model_a_minus_model_b":ma[m]-mb[m],"ci_low":vals.quantile(.025),"ci_high":vals.quantile(.975),"share_bootstrap_above_zero_descriptive":float((vals>0).mean()),"p_value":float("nan"),"p_value_method":"not computed (see 06_statistics for the null-centred test)","n_patients":n,"n_kidneys":len(block)})
    return pd.DataFrame(summary),reps


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); logger=configure_logging(config,"03_evaluate_gfr")
    case_path=output_dir(config,"case_level")/"locked_case_level_predictions.csv"; data=pd.read_csv(case_path); tables=output_dir(config,"tables"); source=output_dir(config,"source_data"); n_boot=int(config["bootstrap_replicates"]); seed=int(config["random_seed"])
    records=[]; all_reps=[]
    for split in ("internal","external"):
      cohort=data[data.split==split]
      for mi,model in enumerate(MODEL_ORDER):
       for si,side in enumerate(("left","right","pooled")):
        block=cohort if side=="pooled" else cohort[cohort.side==side]
        est,ci,reps=perf_boot(block,PRED[model],n_boot,seed+mi*1000+si*100+(0 if split=="internal" else 10000)); rec={"cohort":split,"model_key":model,"model":config["models"][model]["label"],"side":side,**est}
        for m in METRICS: rec[f"{m}_ci_low"],rec[f"{m}_ci_high"]=ci[m]
        records.append(rec); reps.insert(0,"cohort",split); reps.insert(1,"model_key",model); reps.insert(2,"side",side); all_reps.append(reps)
    performance=pd.DataFrame(records)
    for split in ("internal","external"):
      block=performance[performance.cohort==split]; block.to_csv(tables/f"Table_GFR_performance_{split}.csv",index=False); write_xlsx(block,tables/f"Table_GFR_performance_{split}.xlsx")
    performance.to_csv(tables/"Table_GFR_performance_all_models.csv",index=False); write_xlsx(performance,tables/"Table_GFR_performance_all_models.xlsx"); pd.concat(all_reps).to_csv(source/"gfr_performance_bootstrap_replicates.csv.gz",index=False,compression="gzip")
    comps=[]; creps=[]
    for split in ("internal","external"):
      cohort=data[data.split==split]
      for si,side in enumerate(("left","right","pooled")):
       block=cohort if side=="pooled" else cohort[cohort.side==side]
       for mi,model in enumerate(MODEL_ORDER[:-1]):
        summary,reps=paired_boot(block,PRED["dfme"],PRED[model],n_boot,seed+50000+mi*1000+si*100+(0 if split=="internal" else 10000)); summary.insert(0,"cohort",split); summary.insert(1,"side",side); summary.insert(2,"model_a","dfme"); summary.insert(3,"model_b",model); comps.append(summary); reps.insert(0,"cohort",split); reps.insert(1,"side",side); reps.insert(2,"model_b",model); creps.append(reps)
    comparisons=pd.concat(comps,ignore_index=True); comparisons.to_csv(tables/"Table_GFR_model_comparisons.csv",index=False); write_xlsx(comparisons,tables/"Table_GFR_model_comparisons.xlsx"); pd.concat(creps).to_csv(source/"gfr_paired_comparison_bootstrap_replicates.csv.gz",index=False,compression="gzip")
    write_environment_manifest(config,[case_path],"03_evaluate_gfr"); logger.info("Complete: %d performance and %d comparison rows",len(performance),len(comparisons))


if __name__=="__main__": main()
