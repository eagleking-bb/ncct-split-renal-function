# Code and data availability

## Statement for the manuscript

**Code availability.** The analysis code that reproduces cohort definition,
kidney segmentation, radiomics extraction, model training, ensembling and
statistical evaluation is publicly available at
`https://github.com/eagleking-bb/ncct-split-renal-function` and archived at Zenodo
(DOI: 10.5281/zenodo.24080203).

**Data availability.** The datasets generated and/or analysed during the
current study are not publicly available because they contain clinical images
and linked clinical records that carry a residual re-identification risk. The
study protocol, the statistical analysis plan and the data dictionary are
available with the publication. De-identified individual participant data and
the derived feature tables are available from the corresponding author
(qwei@smu.edu.cn) on reasonable request for non-commercial research, after
approval of a study proposal and signature of a data-use agreement, from 12
months to 5 years after publication.

## What is in this repository

The code alone does not re-derive a published number: the locked prediction
tables it reads are withheld (see below) and are released only under the
data-use agreement. With those tables, every reported value is regenerated at
the precision printed in the manuscript.


| Included | Not included |
| --- | --- |
| Cohort definition and case-level inventory | DICOM or NIfTI images |
| nnU-Net fine-tuning and inference scripts | Trained model checkpoints |
| Radiomics extraction and network-adaptive selection | PyRadiomics feature matrices |
| Nine training entrypoints plus ensembling | Per-patient clinical records |
| External inference from checkpoints (`04_ensembling/05`) | Individual reference GFR values |
| External validation, GFR and SRF<15% evaluation | Individual reference GFR values |
| Calibration, decision-curve, error and subgroup analyses | Locked prediction tables (available on request) |
| Paired bootstrap comparisons | Publication figure artwork |
| Final package consistency audit (`05_evaluation/09`) | Internal development test harness |
| Publication figure code and figure source data | |

## Rebuilding external predictions

`04_ensembling/05_external_inference_from_checkpoints.py` is the supported route
from a trained model to `external_predictions.csv`; see
`config/external_inference.example.json` for the manifest layout. A checkpoint
must carry an `inference_state` block (imputation medians, means and scales,
per-centre radiomics statistics, patient split, image preprocessing and model
definition). Checkpoints written before this schema are rejected: their
preprocessing state cannot be recovered, and re-fitting it on the external
cohort would break the locked evaluation. Tabular baselines are distributed as
`joblib` estimator bundles that contain both the fitted estimators and their
preprocessing state.

Deep-learning checkpoints are available from the corresponding author under the
data-use agreement described above; they are not redistributed with the code.
Tabular estimator bundles declare `gfr_target.unit = mL/min/1.73 m2`; bundles
written before release 1.2.0 are rejected because their target standardisation
cannot be distinguished from the corrected procedure.

## Scope of this distribution

This is the minimal analysis package prepared for submission. It contains every
script that produces the cohort flow, the model results, the statistics and the
publication figures. The internal development test harness (synthetic
fixed-value regression suites) and the optional DICOM-header metadata utility are
not distributed; neither contributes to any reported number.

## Requirements for a full re-run

1. Institutional ethics approval and a data-use agreement covering the three
   participating centres.
2. The de-identified image archive and the linked clinical table described in
   the data dictionary.
3. A CUDA-capable GPU with at least 24 GB of memory for the deep-learning
   stages; the remaining stages run on CPU in under an hour.
4. The exact preprocessing order and augmentation schedule changed after the
   locked analysis (median imputation now precedes standardisation, and random
   augmentation is no longer cached). Re-trained models are `NEW_RUN_REQUIRED`
   and must not be presented as reproducing the released checkpoints.

## Versioning

- `1.0.0` — release accompanying the manuscript. Pipeline covered: cohort
  definition, segmentation, radiomics, training, ensembling, external
  validation on the locked 169 patients, statistics, figures.
- `1.1.0` — bug-fix and reproducibility release. Fixes the nnU-Net output-count
  check, ablation-environment guards, augmentation caching, NaN imputation
  order, pseudo-out-of-fold residualisation, bootstrap inference and
  manuscript-count generation, and adds checkpoint-to-external-prediction
  inference. Locked prediction tables and reported metrics are unchanged.
  Training runs built with this release use the corrected preprocessing order
  and augmentation schedule and are marked `NEW_RUN_REQUIRED`; they are not
  claimed to reproduce the released checkpoints.
- `1.2.0` — GFR target-unit correction. SRF labels and stored original GFR
  values are computed from raw GFR in mL/min/1.73 m²; the model target is
  standardised once with training statistics; tabular baseline predictions are
  returned in original units. Preprocessing states and estimator bundles from
  earlier releases are rejected. The unit contract is: raw GFR is preserved in
  `left_gfr_original` / `right_gfr_original`, SRF labels are derived from those
  raw values, the model target is standardised once with training statistics and
  inverse-transformed before reporting, and tabular external predictions stay in
  mL/min/1.73 m². Re-training is `NEW_RUN_REQUIRED`.

Committed alongside the submission so that the analysed version is
identifiable. Any later change is a new tag; published tags are not edited
after the archival DOI is minted.

## Licence

Released under the MIT License (see `LICENSE`). Copyright 2026 Guangdong
Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern
Medical University. The licence covers the source code only. Patient images, clinical records, model checkpoints and
derived feature tables are not distributed and remain subject to the
data-use agreement described above.

## Contact

Corresponding author: Prof Qiang Wei, Department of Urology, Guangdong
Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern
Medical University, Guangzhou 510080, China. qwei@smu.edu.cn
