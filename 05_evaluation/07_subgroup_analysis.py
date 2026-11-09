#!/usr/bin/env python3
"""Conduct exploratory DFME subgroup analyses with patient-level bootstrap."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from utils import classification_metrics, configure_logging, continuous_metrics, load_config, output_dir, patient_bootstrap, write_environment_manifest, write_xlsx


def add_groups(data: pd.DataFrame) -> pd.DataFrame:
    out=data.copy()
    out["srf_group"]=np.where(out.true_srf_lt15==1,"SRF <15%","SRF ≥15%")
    out["total_gfr_group"]=pd.cut(out.true_total_gfr,[-np.inf,60,90,np.inf],labels=["<60","60-<90","≥90"],right=False).astype(str)
    out["age_group"]=pd.cut(out.age,[-np.inf,50,65,np.inf],labels=["<50","50-64","≥65"],right=False).astype(str)
    out["sex_group"]=out.sex.map({0:"Male",1:"Female"}).fillna("Unknown")
    out["scr_group"]=pd.qcut(out.serum_creatinine,3,labels=["Low","Middle","High"],duplicates="drop").astype(str)
    out["volume_group"]=pd.qcut(out.kidney_volume,3,labels=["Low","Middle","High"],duplicates="drop").astype(str)
    out["z_spacing_group"]=out.z_spacing.round(2).astype(str)+" mm"
    disease=out.same_side_disease_label.fillna("Unknown").astype(str)
    out["disease_group"]=disease
    out["obstructive_group"]=np.where(disease.str.contains("积水|梗阻",regex=True),"Hydronephrosis/obstructive","Non-obstructive")
    out["tumor_group"]=np.where(disease.str.contains("肿瘤|癌|瘤",regex=True),"Tumor","Non-tumor")
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); args=parser.parse_args()
    config=load_config(args.config); logger=configure_logging(config,"07_subgroup_analysis")
    case_path=output_dir(config,"case_level")/"locked_case_level_predictions.csv"; data=add_groups(pd.read_csv(case_path)); tables=output_dir(config,"tables")
    group_vars=["side","srf_group","total_gfr_group","age_group","sex_group","scr_group","volume_group","z_spacing_group","center","disease_group","obstructive_group","tumor_group"]
    rows=[]
    for split in ("internal","external"):
        cohort=data[data.split==split]
        for var_index,var in enumerate(group_vars):
            for level_index,(level,block) in enumerate(cohort.groupby(var,dropna=False)):
                cont=continuous_metrics(block,pred="predicted_gfr_dfme")
                valid=block.dropna(subset=["true_side_gfr","predicted_gfr_dfme"]).reset_index(drop=True); ids=valid.anonymized_patient_id.to_numpy(); y=valid.true_side_gfr.to_numpy(float); p=valid.predicted_gfr_dfme.to_numpy(float); unique=pd.unique(ids); groups=[np.flatnonzero(ids==u) for u in unique]; rng=np.random.default_rng(int(config["random_seed"])+120000+var_index*1000+level_index+(0 if split=="internal" else 20000)); vals={"mae":[],"ccc":[],"bias":[]}
                vals["auroc"]=[]; vals["auprc"]=[]; ycls=valid.true_srf_lt15.astype(int).to_numpy(); probs=valid.predicted_probability_srf_lt15.to_numpy(float)
                from sklearn.metrics import average_precision_score, roc_auc_score
                for _ in range(int(config["bootstrap_replicates"])):
                    idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); error=p[idx]-y[idx]; vals["mae"].append(np.abs(error).mean()); vals["bias"].append(error.mean());
                    from utils import ccc
                    vals["ccc"].append(ccc(y[idx],p[idx]))
                    if np.unique(ycls[idx]).size==2 and ycls[idx].sum()>=1: vals["auroc"].append(roc_auc_score(ycls[idx],probs[idx])); vals["auprc"].append(average_precision_score(ycls[idx],probs[idx]))
                ci={m:(np.nanquantile(v,.025),np.nanquantile(v,.975)) for m,v in vals.items()}
                cls=classification_metrics(block,float(config["classification_threshold"]))
                unstable=cont["n_patients"]<30 or cls["positive_kidneys"]<10
                rows.append({"cohort":split,"subgroup_variable":var,"subgroup_level":str(level),"n_patients":cont["n_patients"],"n_kidneys":cont["n_kidneys"],"positive_kidneys":cls["positive_kidneys"],"mae":cont["mae"],"mae_ci_low":ci["mae"][0],"mae_ci_high":ci["mae"][1],"ccc":cont["ccc"],"ccc_ci_low":ci["ccc"][0],"ccc_ci_high":ci["ccc"][1],"bias":cont["bias"],"bias_ci_low":ci["bias"][0],"bias_ci_high":ci["bias"][1],"auroc":cls["auroc"] if cls["positive_kidneys"]>=10 else np.nan,"auroc_ci_low":ci["auroc"][0] if cls["positive_kidneys"]>=10 else np.nan,"auroc_ci_high":ci["auroc"][1] if cls["positive_kidneys"]>=10 else np.nan,"auprc":cls["auprc"] if cls["positive_kidneys"]>=10 else np.nan,"auprc_ci_low":ci["auprc"][0] if cls["positive_kidneys"]>=10 else np.nan,"auprc_ci_high":ci["auprc"][1] if cls["positive_kidneys"]>=10 else np.nan,"stability_flag":"unstable/exploratory" if unstable else "adequate descriptive size"})
    result=pd.DataFrame(rows); result.to_csv(tables/"Table_DFME_subgroups.csv",index=False); write_xlsx(result,tables/"Table_DFME_subgroups.xlsx")
    write_environment_manifest(config,[case_path],"07_subgroup_analysis"); logger.info("Generated %d subgroup rows",len(result))


if __name__=="__main__": main()
