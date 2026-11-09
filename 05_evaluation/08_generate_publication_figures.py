#!/usr/bin/env python3
"""Generate publication figures and figure-specific source-data tables."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

from utils import COHORT_COLORS, MODEL_COLORS, SIDE_COLORS, configure_logging, load_config, output_dir, save_figure, set_publication_style, write_environment_manifest


def agreement(data, cohort, figures, source):
    block=data[data.split==cohort].copy(); block.to_csv(source/f"Figure_GFR_agreement_{cohort}_source.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,sharey=True)
    for ax,side in zip(axes,("left","right")):
        d=block[block.side==side]; ax.scatter(d.true_side_gfr,d.predicted_gfr_dfme,s=13,alpha=.55,color=SIDE_COLORS[side],edgecolor="white",linewidth=.25); lo=min(d.true_side_gfr.min(),d.predicted_gfr_dfme.min()); hi=max(d.true_side_gfr.max(),d.predicted_gfr_dfme.max()); ax.plot([lo,hi],[lo,hi],"--",color="#555",lw=1); slope,intercept=np.polyfit(d.true_side_gfr,d.predicted_gfr_dfme,1); x=np.array([lo,hi]); ax.plot(x,intercept+slope*x,color="#222",lw=1); err=d.predicted_gfr_dfme-d.true_side_gfr; from utils import ccc; ax.text(.04,.96,f"n={d.anonymized_patient_id.nunique()}\nCCC={ccc(d.true_side_gfr,d.predicted_gfr_dfme):.3f}\nMAE={np.abs(err).mean():.2f}",transform=ax.transAxes,va="top"); ax.set_title(side.capitalize()+" kidney"); ax.set_xlabel("Reference GFR (mL/min/1.73 m²)")
    axes[0].set_ylabel("Predicted GFR (mL/min/1.73 m²)"); fig.suptitle(f"DFME GFR agreement — {cohort} validation"); fig.tight_layout(); save_figure(fig,figures,f"Figure_GFR_agreement_{cohort}"); plt.close(fig)


def bland_altman(data,cohort,figures,source):
    block=data[data.split==cohort].copy(); block["mean_gfr"]=(block.true_side_gfr+block.predicted_gfr_dfme)/2; block["error"]=block.predicted_gfr_dfme-block.true_side_gfr; block.to_csv(source/f"Figure_GFR_BlandAltman_{cohort}_source.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,sharey=True)
    for ax,side in zip(axes,("left","right")):
        d=block[block.side==side]; bias=d.error.mean(); sd=d.error.std(ddof=1); ax.scatter(d.mean_gfr,d.error,s=13,alpha=.55,color=SIDE_COLORS[side],edgecolor="white",linewidth=.25); ax.axhline(bias,color="#222"); ax.axhline(bias-1.96*sd,color="#777",ls="--"); ax.axhline(bias+1.96*sd,color="#777",ls="--"); slope=stats.linregress(d.mean_gfr,d.error); x=np.array(ax.get_xlim()); ax.plot(x,slope.intercept+slope.slope*x,color="#A64B4B",lw=1); ax.text(.04,.96,f"Bias={bias:.2f}\nLoA {bias-1.96*sd:.2f} to {bias+1.96*sd:.2f}",transform=ax.transAxes,va="top"); ax.set_title(side.capitalize()+" kidney"); ax.set_xlabel("Mean of reference and prediction")
    axes[0].set_ylabel("Prediction − reference"); fig.suptitle(f"DFME Bland–Altman — {cohort} validation"); fig.tight_layout(); save_figure(fig,figures,f"Figure_GFR_BlandAltman_{cohort}"); plt.close(fig)


def gfr_calibration(calibration,cohort,figures):
    d=calibration[(calibration.cohort==cohort)&(calibration.side.isin(["left","right"]))]
    d.to_csv(output_dir(CURRENT_CONFIG,"source_data")/f"Figure_GFR_calibration_{cohort}_source.csv",index=False)
    fig,ax=plt.subplots(figsize=(3.6,3.3)); bounds=[d.mean_predicted.min(),d.mean_predicted.max()]; ax.plot(bounds,bounds,"--",color="#555",label="Ideal")
    for side in ("left","right"):
        x=d[d.side==side]; ax.plot(x.mean_predicted,x.mean_observed,marker="o",ms=3,color=SIDE_COLORS[side],label=side.capitalize())
    ax.set_xlabel("Mean predicted GFR"); ax.set_ylabel("Mean observed GFR"); ax.set_title(f"Continuous calibration — {cohort}"); ax.legend(frameon=False); fig.tight_layout(); save_figure(fig,figures,f"Figure_GFR_calibration_{cohort}"); plt.close(fig)


def classification_figures(data,figures,source):
    pooled=data.copy(); roc_rows=[]; pr_rows=[]; cal_rows=[]
    fig_roc,axr=plt.subplots(figsize=(3.6,3.3)); fig_pr,axp=plt.subplots(figsize=(3.6,3.3)); fig_cal,axc=plt.subplots(figsize=(3.6,3.3))
    for cohort in ("internal","external"):
        d=pooled[pooled.split==cohort]; y=d.true_srf_lt15.astype(int).to_numpy(); p=d.predicted_probability_srf_lt15.to_numpy(float); fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p); frac,mean=calibration_curve(y,p,n_bins=10,strategy="quantile")
        roc_rows += [{"cohort":cohort,"fpr":x,"tpr":z} for x,z in zip(fpr,tpr)]; pr_rows += [{"cohort":cohort,"recall":x,"precision":z} for x,z in zip(recall,precision)]; cal_rows += [{"cohort":cohort,"mean_predicted":x,"observed_fraction":z} for x,z in zip(mean,frac)]
        color=COHORT_COLORS[cohort]; axr.plot(fpr,tpr,color=color,lw=1.8,label=f"{cohort.capitalize()} (AUC {auc(fpr,tpr):.3f})"); axp.plot(recall,precision,color=color,lw=1.8,label=f"{cohort.capitalize()} (AP {average_precision_score(y,p):.3f})"); axc.plot(mean,frac,marker="o",ms=3,color=color,label=cohort.capitalize())
    axr.plot([0,1],[0,1],"--",color="#777"); axr.set(xlabel="1 − specificity",ylabel="Sensitivity",title="DFME SRF <15% ROC"); axr.legend(frameon=False)
    axp.set(xlabel="Recall",ylabel="Precision",title="DFME SRF <15% precision–recall"); axp.legend(frameon=False)
    axc.plot([0,1],[0,1],"--",color="#777"); axc.set(xlabel="Predicted probability",ylabel="Observed proportion",title="DFME SRF <15% calibration"); axc.legend(frameon=False)
    for fig,stem in ((fig_roc,"Figure_SRF15_ROC"),(fig_pr,"Figure_SRF15_PR"),(fig_cal,"Figure_SRF15_calibration")): fig.tight_layout(); save_figure(fig,figures,stem); plt.close(fig)
    pd.DataFrame(roc_rows).to_csv(source/"Figure_SRF15_ROC_source.csv",index=False); pd.DataFrame(pr_rows).to_csv(source/"Figure_SRF15_PR_source.csv",index=False); pd.DataFrame(cal_rows).to_csv(source/"Figure_SRF15_calibration_source.csv",index=False)


def dca_figure(dca,figures):
    d=dca[dca.side=="pooled"]; fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
    d.to_csv(output_dir(CURRENT_CONFIG,"source_data")/"Figure_SRF15_DCA_source.csv",index=False)
    for ax,cohort in zip(axes,("internal","external")):
        for strategy,color,ls in (("DFME",COHORT_COLORS[cohort],"-"),("treat-all","#888","--"),("treat-none","#222",":")):
            x=d[(d.cohort==cohort)&(d.strategy==strategy)]; ax.plot(x.threshold,x.net_benefit,color=color,ls=ls,label=strategy)
        ax.set_title(cohort.capitalize()); ax.set_xlabel("Threshold probability"); ax.legend(frameon=False)
    axes[0].set_ylabel("Net benefit"); fig.suptitle("Decision curve analysis for SRF <15%"); fig.tight_layout(); save_figure(fig,figures,"Figure_SRF15_DCA"); plt.close(fig)


def error_figures(data,associations,domain,figures,source):
    ext=data[data.split=="external"].copy(); ext["error"]=ext.predicted_gfr_dfme-ext.true_side_gfr; ext.to_csv(source/"Figure_external_error_vs_GFR_source.csv",index=False)
    fig,ax=plt.subplots(figsize=(3.8,3.3)); ax.scatter(ext.true_side_gfr,ext.error,s=12,alpha=.45,color=COHORT_COLORS["external"]); slope=stats.linregress(ext.true_side_gfr,ext.error); x=np.linspace(ext.true_side_gfr.min(),ext.true_side_gfr.max(),100); ax.plot(x,slope.intercept+slope.slope*x,color="#222"); ax.axhline(0,color="#777",ls="--"); ax.set(xlabel="Reference side-specific GFR",ylabel="Prediction − reference",title="External DFME error across GFR"); fig.tight_layout(); save_figure(fig,figures,"Figure_external_error_vs_GFR"); plt.close(fig)
    d=associations[(associations.cohort=="external")&(associations.analysis_type=="continuous_association")].sort_values("spearman_rho"); d.to_csv(source/"Figure_external_error_associations_source.csv",index=False); fig,ax=plt.subplots(figsize=(4.6,3.4)); y=np.arange(len(d)); ax.errorbar(d.spearman_rho,y,xerr=[d.spearman_rho-d.rho_ci_low,d.rho_ci_high-d.spearman_rho],fmt="o",color="#4C6A78"); ax.axvline(0,color="#777",ls="--"); ax.set_yticks(y,d.variable); ax.set_xlabel("Spearman ρ with error"); ax.set_title("External error associations"); fig.tight_layout(); save_figure(fig,figures,"Figure_external_error_associations"); plt.close(fig)
    z=domain[(domain.variable.isin(["total_GFR","SCr","spacing_z"]))&(domain.type=="continuous")]; z.to_csv(source/"Figure_center_distribution_shift_source.csv",index=False); fig,axes=plt.subplots(1,3,figsize=(7.2,2.8));
    for ax,var in zip(axes,("total_GFR","SCr","spacing_z")):
        x=z[z.variable==var]; ax.bar(x.center,x["median"],color=["#6D8EA0","#89A6B2","#C17A4A"]); ax.set_title(var); ax.tick_params(axis="x",rotation=25)
    fig.suptitle("Center-level distribution shift (median)"); fig.tight_layout(); save_figure(fig,figures,"Figure_center_distribution_shift"); plt.close(fig)


def subgroup_figure(subgroups,figures,source):
    d=subgroups[(subgroups.cohort=="external")&(subgroups.subgroup_variable.isin(["side","sex_group","total_gfr_group","z_spacing_group","obstructive_group","tumor_group"]))].copy(); d.to_csv(source/"Figure_DFME_subgroup_forest_source.csv",index=False); d=d.sort_values(["subgroup_variable","subgroup_level"]); labels=(d.subgroup_variable+": "+d.subgroup_level).tolist(); y=np.arange(len(d)); fig,ax=plt.subplots(figsize=(5.8,max(3.5,.28*len(d)))); ax.errorbar(d.mae,y,xerr=[d.mae-d.mae_ci_low,d.mae_ci_high-d.mae],fmt="o",color="#2E6E5B"); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("MAE (95% patient-bootstrap CI)"); ax.set_title("External DFME subgroup performance"); fig.tight_layout(); save_figure(fig,figures,"Figure_DFME_subgroup_forest"); plt.close(fig)


def main():
    global CURRENT_CONFIG
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); CURRENT_CONFIG=config; logger=configure_logging(config,"08_generate_publication_figures"); set_publication_style(); figures=output_dir(config,"figures"); source=output_dir(config,"source_data"); case_path=output_dir(config,"case_level")/"locked_case_level_predictions.csv"; data=pd.read_csv(case_path)
    calibration=pd.read_csv(source/"gfr_calibration_source_data.csv"); dca=pd.read_csv(source/"srf15_dca_source_data.csv"); associations=pd.read_csv(output_dir(config,"tables")/"error_association.csv"); domain=pd.read_csv(output_dir(config,"tables")/"Table_domain_shift.csv"); subgroups=pd.read_csv(output_dir(config,"tables")/"Table_DFME_subgroups.csv")
    for cohort in ("internal","external"): agreement(data,cohort,figures,source); bland_altman(data,cohort,figures,source); gfr_calibration(calibration,cohort,figures)
    classification_figures(data,figures,source); dca_figure(dca,figures); error_figures(data,associations,domain,figures,source); subgroup_figure(subgroups,figures,source)
    write_environment_manifest(config,[case_path],"08_generate_publication_figures"); logger.info("Publication figures complete")


if __name__=="__main__": main()
