#!/usr/bin/env python3
"""Analyze locked DFME error and center-level domain shift without model updating."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from utils import configure_logging, load_config, output_dir, patient_bootstrap, continuous_metrics, write_environment_manifest, write_xlsx


def rho_ci(frame: pd.DataFrame, variable: str, n_boot: int, seed: int) -> tuple[float, float, float, float, float, float]:
    block = frame[["anonymized_patient_id", variable, "error"]].dropna()
    estimate = float(stats.spearmanr(block[variable], block["error"]).statistic) if len(block) >= 3 else np.nan
    x=block[variable].to_numpy(float); e=block.error.to_numpy(float)
    slope = float(stats.linregress(x, e).slope) if len(block) >= 3 and np.unique(x).size > 1 else np.nan
    rng = np.random.default_rng(seed); patients = block.anonymized_patient_id.drop_duplicates().to_numpy(); ids=block.anonymized_patient_id.to_numpy(); groups=[np.flatnonzero(ids==p) for p in patients]; vals=[]; slopes=[]
    for _ in range(n_boot):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if np.unique(x[idx]).size>1 and np.unique(e[idx]).size>1:
            vals.append(stats.spearmanr(x[idx],e[idx]).statistic); slopes.append(stats.linregress(x[idx],e[idx]).slope)
    return estimate, float(np.quantile(vals,.025)) if vals else np.nan, float(np.quantile(vals,.975)) if vals else np.nan, slope, float(np.quantile(slopes,.025)) if slopes else np.nan, float(np.quantile(slopes,.975)) if slopes else np.nan


def smd(x: pd.Series, y: pd.Series) -> float:
    x=pd.to_numeric(x,errors="coerce").dropna(); y=pd.to_numeric(y,errors="coerce").dropna()
    pooled=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    return float((x.mean()-y.mean())/pooled) if pooled>0 else np.nan


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); args=parser.parse_args()
    config=load_config(args.config); logger=configure_logging(config,"06_external_error_analysis")
    case_path=output_dir(config,"case_level")/"locked_case_level_predictions.csv"; data=pd.read_csv(case_path)
    data["error"]=data.predicted_gfr_dfme-data.true_side_gfr
    tables=output_dir(config,"tables"); source=output_dir(config,"source_data")
    variables=["true_side_gfr","true_total_gfr","serum_creatinine","kidney_volume","kidney_HU_mean","z_spacing"]
    rows=[]
    for split in ("internal","external"):
        block=data[data.split==split]
        for index,var in enumerate(variables):
            estimate,low,high,slope,slope_low,slope_high=rho_ci(block,var,int(config["bootstrap_replicates"]),int(config["random_seed"])+90000+index+(0 if split=="internal" else 1000))
            valid=block[[var,"error"]].dropna()
            intercept=np.nan
            if len(valid)>=3 and valid[var].nunique()>1:
                fit=stats.linregress(valid[var],valid.error); intercept=float(fit.intercept)
            rows.append({"cohort":split,"analysis_type":"continuous_association","variable":var,"level":"continuous","n":len(valid),"missing_n":len(block)-len(valid),"spearman_rho":estimate,"rho_ci_low":low,"rho_ci_high":high,"linear_slope":slope,"slope_ci_low":slope_low,"slope_ci_high":slope_high,"linear_intercept":intercept,"group_bias":np.nan,"group_bias_ci_low":np.nan,"group_bias_ci_high":np.nan})
    for split, variables_cat in (("external", ["side", "same_side_disease_label"]), ("all_locked", ["center"])):
        block=data.copy() if split=="all_locked" else data[data.split==split].copy()
        for var_index,var in enumerate(variables_cat):
            for level_index,(level,group) in enumerate(block.groupby(var,dropna=False)):
                valid=group.dropna(subset=["error"]); ids=valid.anonymized_patient_id.to_numpy(); errors=valid.error.to_numpy(float); unique=pd.unique(ids); groups=[np.flatnonzero(ids==u) for u in unique]; rng=np.random.default_rng(int(config["random_seed"])+95000+var_index*1000+level_index); vals=[]
                for _ in range(int(config["bootstrap_replicates"])):
                    idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); vals.append(errors[idx].mean())
                rows.append({"cohort":split,"analysis_type":"categorical_group_bias","variable":var,"level":str(level),"n":len(valid),"missing_n":len(block)-len(valid),"spearman_rho":np.nan,"rho_ci_low":np.nan,"rho_ci_high":np.nan,"linear_slope":np.nan,"slope_ci_low":np.nan,"slope_ci_high":np.nan,"linear_intercept":np.nan,"group_bias":errors.mean(),"group_bias_ci_low":np.quantile(vals,.025),"group_bias_ci_high":np.quantile(vals,.975)})
    associations=pd.DataFrame(rows); associations.to_csv(tables/"error_association.csv",index=False); write_xlsx(associations,tables/"error_association.xlsx")

    bias_rows=[]
    for split in ("internal","external"):
        block=data[data.split==split]
        for side in ("left","right","pooled"):
            side_block=block if side=="pooled" else block[block.side==side]
            for bin_label,bin_block in [("all",side_block),*[(str(label),g) for label,g in side_block.groupby(pd.cut(side_block.true_side_gfr,[-np.inf,10,20,30,40,50,np.inf],right=False,labels=["<10","10-<20","20-<30","30-<40","40-<50",">=50"]))]]:
                if len(bin_block)==0: continue
                estimate=continuous_metrics(bin_block,pred="predicted_gfr_dfme"); ids=bin_block.anonymized_patient_id.to_numpy(); y=bin_block.true_side_gfr.to_numpy(float); p=bin_block.predicted_gfr_dfme.to_numpy(float); unique=pd.unique(ids); groups=[np.flatnonzero(ids==u) for u in unique]; rng=np.random.default_rng(int(config["random_seed"])+100000+len(bias_rows)); boot_bias=[]; boot_mae=[]
                for _ in range(int(config["bootstrap_replicates"])):
                    idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); error=p[idx]-y[idx]; boot_bias.append(error.mean()); boot_mae.append(np.abs(error).mean())
                intervals={"bias":(np.quantile(boot_bias,.025),np.quantile(boot_bias,.975)),"mae":(np.quantile(boot_mae,.025),np.quantile(boot_mae,.975))}
                bias_rows.append({"cohort":split,"side":side,"gfr_bin":bin_label,"n_patients":estimate["n_patients"],"n_kidneys":estimate["n_kidneys"],"bias":estimate["bias"],"bias_ci_low":intervals["bias"][0],"bias_ci_high":intervals["bias"][1],"mae":estimate["mae"],"mae_ci_low":intervals["mae"][0],"mae_ci_high":intervals["mae"][1]})
    bias=pd.DataFrame(bias_rows); bias.to_csv(tables/"Table_external_underestimation_by_GFR.csv",index=False); write_xlsx(bias,tables/"Table_external_underestimation_by_GFR.xlsx")

    audit=pd.read_csv(config["inputs"]["nifti_audit_csv"],dtype={"patient_id":str}); audit=audit[audit.qc_status=="pass"].copy()
    continuous=["total_GFR","left_GFR","right_GFR","SCr","age","height","weight","left_volume_ml","right_volume_ml","left_hu_mean","right_hu_mean","spacing_z"]
    domain=[]
    for var in continuous:
        for center in ("Center_1","Center_2","Center_3"):
            values=pd.to_numeric(audit.loc[audit.center==center,var],errors="coerce").dropna()
            domain.append({"variable":var,"type":"continuous","level":"overall","center":center,"n":len(values),"summary":f"{values.median():.2f} [{values.quantile(.25):.2f}, {values.quantile(.75):.2f}]" if len(values) else "NA","median":values.median(),"q1":values.quantile(.25),"q3":values.quantile(.75),"mean":values.mean(),"sd":values.std(ddof=1),"smd_vs_center3":smd(values,audit.loc[audit.center=="Center_3",var]) if center!="Center_3" else 0.0})
    disease=pd.concat([pd.read_csv(config["inputs"][k],dtype={"ID":str}).dropna(subset=["ID"]) for k in ("disease_center1_csv","disease_center2_csv","disease_center3_csv")],ignore_index=True).drop_duplicates("ID")
    disease=disease.melt(id_vars="ID",value_vars=["left_disease","right_disease"],var_name="side",value_name="disease").merge(audit[["patient_id","center"]],left_on="ID",right_on="patient_id")
    for variable, series in (("sex", audit[["patient_id","center","sex"]].rename(columns={"sex":"level"})), ("same_side_disease", disease[["ID","center","disease"]].rename(columns={"ID":"patient_id","disease":"level"}))):
      for level in sorted(series.level.dropna().astype(str).unique()):
        for center in ("Center_1","Center_2","Center_3"):
            block=series[series.center==center]; count=int((block.level.astype(str)==level).sum()); ref=series[series.center=="Center_3"]; p=count/len(block) if len(block) else np.nan; pref=(ref.level.astype(str)==level).mean() if len(ref) else np.nan; pooled=np.sqrt((p*(1-p)+pref*(1-pref))/2) if np.isfinite(p) and np.isfinite(pref) else np.nan; cat_smd=(p-pref)/pooled if pooled and pooled>0 else 0.0 if p==pref else np.nan
            domain.append({"variable":variable,"type":"categorical","level":level,"center":center,"n":len(block),"summary":f"{count} ({100*count/len(block):.1f}%)" if len(block) else "NA","median":np.nan,"q1":np.nan,"q3":np.nan,"mean":p,"sd":np.nan,"smd_vs_center3":cat_smd})
    domain=pd.DataFrame(domain); domain.to_csv(tables/"Table_domain_shift.csv",index=False); write_xlsx(domain,tables/"Table_domain_shift.xlsx")
    data.to_csv(source/"external_error_source_data.csv",index=False)
    write_environment_manifest(config,[case_path,config["inputs"]["nifti_audit_csv"]],"06_external_error_analysis")
    logger.info("External error and domain-shift analyses complete")


if __name__=="__main__": main()
