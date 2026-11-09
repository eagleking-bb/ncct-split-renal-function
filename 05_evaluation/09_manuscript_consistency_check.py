#!/usr/bin/env python3
"""Perform final package consistency, privacy, and expected-output checks."""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd
from utils import configure_logging, load_config, output_dir, sha256_file, write_environment_manifest
from privacy_scan import (
    build_source_id_pattern,
    derive_center_prefixes,
    find_unanonymised_id_columns,
    scan_tree,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); logger=configure_logging(config,"10_manuscript_consistency_check"); root=output_dir(config); audit=output_dir(config,"audit")
    errors=[]; warnings=[]; master=pd.read_csv(root/"case_level/locked_case_level_predictions.csv")
    expected={"internal":175,"external":169}
    for split,n in expected.items():
        actual=master[master.split==split].anonymized_patient_id.nunique()
        if actual!=n: errors.append(f"{split} n={actual}, expected {n}")
    if master.duplicated(["anonymized_patient_id","side"]).any(): errors.append("duplicate patient-side rows")
    if set(master.side)!={"left","right"}: errors.append("unexpected/missing laterality")
    if not master.anonymized_patient_id.str.match(r"^DFME_[0-9a-f]{16}$").all(): errors.append("non-anonymized patient ID detected")
    derived_label=(master.true_srf_percent<15).astype(int)
    if not derived_label.equals(master.true_srf_lt15.astype(int)): errors.append("SRF<15 label differs from side-GFR formula")
    verification=pd.read_csv(root/"audit/dfme_equal_weight_verification.csv")
    if verification.max_abs_difference_dfme_vs_fixed_average.max()>1e-6: errors.append("DFME differs from fixed 0.5/0.5 average")
    gfr=pd.read_csv(root/"tables/Table_GFR_performance_all_models.csv"); srf=pd.read_csv(root/"tables/Table_SRF15_performance_all.csv")
    if len(gfr)!=36: errors.append(f"GFR performance rows={len(gfr)}, expected 36")
    if len(srf)!=6: errors.append(f"SRF performance rows={len(srf)}, expected 6")
    associations=pd.read_csv(root/"tables/error_association.csv")
    required_association_types={"continuous_association","categorical_group_bias"}
    if not required_association_types.issubset(set(associations.get("analysis_type",[]))): errors.append("error association table lacks continuous or categorical analyses")
    subgroups=pd.read_csv(root/"tables/Table_DFME_subgroups.csv")
    required_subgroup_ci={"auroc_ci_low","auroc_ci_high","auprc_ci_low","auprc_ci_high"}
    if not required_subgroup_ci.issubset(subgroups.columns): errors.append("subgroup classification CI columns absent")
    domain=pd.read_csv(root/"tables/Table_domain_shift.csv")
    if not {"sex","same_side_disease"}.issubset(set(domain.variable)): errors.append("domain table lacks sex or disease distributions")
    for pattern in ["Figure_GFR_agreement_internal","Figure_GFR_agreement_external","Figure_GFR_BlandAltman_internal","Figure_GFR_BlandAltman_external","Figure_GFR_calibration_internal","Figure_GFR_calibration_external","Figure_SRF15_ROC","Figure_SRF15_PR","Figure_SRF15_calibration","Figure_SRF15_DCA","Figure_external_error_vs_GFR","Figure_external_error_associations","Figure_center_distribution_shift","Figure_DFME_subgroup_forest"]:
        for ext in ("svg","pdf","tiff","png"):
            if not (root/f"figures/{pattern}.{ext}").is_file(): errors.append(f"missing figure {pattern}.{ext}")
    # Privacy scan: prefixes come from the configuration and the known centre
    # codes, so GZH_/NFH_/GDPH_ identifiers (any case) are all detected.
    prefixes=derive_center_prefixes(config); protected=build_source_id_pattern(prefixes)
    scan_roots=[root]
    package_root=config.get("package_root")
    if package_root and Path(str(package_root)).is_dir() and Path(str(package_root)).resolve()!=root.resolve():
        scan_roots.append(Path(str(package_root)))
    for scan_root in scan_roots:
        errors.extend(scan_tree(scan_root, protected))
    errors.extend(find_unanonymised_id_columns(master, "locked_case_level_predictions.csv"))
    logger.info("privacy scan prefixes: %s", prefixes)
    write_environment_manifest(config,[root/"case_level/locked_case_level_predictions.csv"],"10_manuscript_consistency_check")
    excluded_manifest_files={"audit/final_output_manifest.csv","audit/final_consistency_report.json"}
    manifest=[]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel=path.relative_to(root).as_posix()
        if rel not in excluded_manifest_files: manifest.append({"path":rel,"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
    pd.DataFrame(manifest).to_csv(audit/"final_output_manifest.csv",index=False)
    report={"status":"PASS" if not errors else "FAIL","errors":errors,"warnings":warnings,"n_manifested_files":len(manifest),"manifest_exclusions":sorted(excluded_manifest_files),"locked_internal_n":175,"locked_external_n":169,"privacy_prefixes":prefixes,"dfme_max_average_difference":verification.max_abs_difference_dfme_vs_fixed_average.max()}
    (audit/"final_consistency_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if errors: raise SystemExit("Final consistency audit failed:\n- "+"\n- ".join(errors))
    logger.info("Final consistency audit passed with %d files",len(manifest))


if __name__=="__main__": main()
