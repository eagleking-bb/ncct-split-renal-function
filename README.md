# Code for "non-contrast CT radiomics and deep learning for predicting split renal function"

This repository contains the analysis code for the study *Deep learning and
radiomics from non-contrast CT to predict split renal function and identify
severe unilateral renal dysfunction* (submitted to eClinicalMedicine).

Release `1.2.0` (GFR target-unit fix; locked prediction tables and reported results are unchanged). This is the minimal distribution prepared for eClinicalMedicine: analysis, plotting and statistics code only. Source: https://github.com/eagleking-bb/ncct-split-renal-function. Archived at Zenodo: DOI `<TO BE INSERTED ON ACCEPTANCE>`.

## Reproducibility status (read first)

| Artefact | Status |
| --- | --- |
| Locked prediction tables, tables 2-3, figures 3-6, external evaluation | Reproducible from the frozen outputs shipped to reviewers |
| Preprocessing order (median imputation before standardisation) | **Changed** — the earlier order scaled before imputing and raised on NaN |
| Random augmentation schedule | **Changed** — augmentation used to be cached (`PersistentDataset`/`CacheDataset`) and is now re-sampled every epoch |
| Deep-learning checkpoints | **NEW_RUN_REQUIRED** — old checkpoints cannot be reproduced by this code |
| GFR labels and tabular outputs | **Fixed** — SRF labels and true/original GFR now come from raw mL/min/1.73 m² values; tabular predictions are returned in the same unit |

These training changes affect model fitting. Any model retrained with this release must
be treated as a new run: do not mix its predictions with the locked tables, and
do not describe the released checkpoints as reproducible by the current
training entrypoints. Every training entrypoint writes
`"rerun_required": "NEW_RUN_REQUIRED"` into its `run_manifest.json`, and every
new checkpoint carries an `inference_state` block with the preprocessing
statistics, split fingerprint and image pipeline that produced it.

### Provenance of the reported results

Every number in the manuscript comes from the locked analysis runs whose
checkpoints and prediction tables were frozen before this repository was
prepared. This release corrects the imputation order, the augmentation
caching and the GFR target unit. Those corrections change how a model is
fitted; they do not change the frozen predictions or any reported value.
Consequently, a model trained with this repository is a **new run** and must
never be pooled with the locked predictions. The Methods section should
describe the locked runs, and the differences listed in the table above should
be reported as implementation corrections rather than as an alternative
analysis.

### What this repository can and cannot recompute

Given the locked prediction tables, the stage-05 scripts regenerate every value
in Tables 2 and 3 and the source data behind Figures 4-6 at the reported
precision. Without those tables the pipeline still runs, but it cannot re-derive
a published number on its own: patient images, reference nuclear-medicine
measurements, derived radiomics matrices, model checkpoints and the locked
prediction tables are withheld because they carry individual-level clinical or
imaging information, and are released only under the data-use agreement in
`CODE_AVAILABILITY.md`. Re-segmenting, re-training or re-running external
inference therefore requires the full data-access procedure.

### Random-number use

`random_seed: 2026` is the base seed for every stage.

- Marginal confidence intervals in `05_evaluation` resample **patients** (both
  kidneys move together) with 5,000 replicates. Each model/side analysis draws
  from its own deterministic offset (`seed + 1000 x model + 100 x side`) so that
  one analysis cannot be perturbed by another.
- Paired comparisons in `06_statistics` use 10,000 replicates and one shared set
  of patient indices per replicate, so the difference between two models is
  computed on the same resample.
- `04_ensembling/04_external_validation_center3.py` deliberately keeps the
  legacy `numpy.random.RandomState(2026)` stream, because that is the stream
  used for the locked Table 2B intervals; every model is evaluated on the same
  patient indices within a replicate.

Changing the base seed moves percentile bounds by a few thousandths of a unit;
point estimates are unaffected.

## Study boundary

- The patient is the unit of sampling and of every bootstrap resampling step.
- Centres 1 and 2 form the development cohort (n = 873): 698 patients for
  training and 175 patients for internal development.
- The 175 internal patients were used for checkpoint selection, so internal
  performance is reported as an internal development estimate.
- Centre 3 forms the locked external validation cohort (n = 169 patients,
  338 kidneys, 28 kidneys with reference split renal function < 15%).
- Centre 3 was never used for feature selection, training, hyperparameter or
  threshold tuning, recalibration, ensemble reweighting or model replacement.
- Reference split GFR comes from nuclear-medicine imaging. It is not a model
  input and is not treated as an absolute gold standard.
- Side-specific disease fields are kidney-level labels and must not be read as
  a patient-level primary indication.

## Cohorts

| Stage | n |
| --- | --- |
| Clinical source records | 1,067 |
| Excluded before locking | 25 |
| Locked analysis population | 1,042 |
| Training | 698 |
| Internal development | 175 |
| **Locked external validation cohort** | **169 patients / 338 kidneys** |

The composition of the external cohort is described in the Supplementary
Materials. `05_evaluation/02_build_locked_prediction_table.py` accepts the
external case list with `--external-case-list` when the inference directory
holds more cases than the locked cohort, and it asserts 169 external patients.

## Pipeline

Run the stages in order. Every stage writes to `<OUTPUT_ROOT>` and reads only
paths given on the command line or in `config/config.yaml`.

| Stage | Script | Purpose |
| --- | --- | --- |
| 01 | `01_segmentation/01_nnunet_transfer_finetune.py` | Contrast-enhanced CT checkpoint to NCCT transfer fine-tuning |
| 01 | `01_segmentation/02_nnunet_predict_kidney_masks.py` | Frozen-checkpoint mask inference for every study scan |
| 02 | `02_radiomics/01_extract_radiomics_features.py` | Paired left/right radiomics from NCCT and kidney masks |
| 02 | `02_radiomics/02_select_network_adaptive_radiomics.py` | Network-adaptive selection of six paired feature families (development data only) |
| 02 | `02_radiomics/03_plot_radiomics_screening_figures.py` | Screening figure |
| 03 | `03_training/common/` | Shared preprocessing (train-only imputation then scaling), deterministic-cache/runtime-augmentation image pipeline, checkpoint inference-state schema |
| 03 | `03_training/*.py` | Nine model and ablation training entrypoints |
| 04 | `04_ensembling/01_ensemble_locked_checkpoints.py` | Equal-weight averaging of the prespecified checkpoint predictions |
| 04 | `04_ensembling/02_plot_internal_architecture_comparison.py` | Architecture comparison tables and figures on the fixed internal validation set |
| 04 | `04_ensembling/03_plot_concat_main_comparison.py` | Concat-ablation comparison figures; AUC CI uses patient-level bootstrap |
| 04 | `04_ensembling/04_external_validation_center3.py` | External validation on the locked 169 patients |
| 04 | `04_ensembling/05_external_inference_from_checkpoints.py` | Rebuilds `external_predictions.csv` from checkpoints and saved preprocessing state |
| 05 | `05_evaluation/01...#09...` | Case-level inventory, locked prediction table, GFR and SRF<15% evaluation, calibration, DCA, error analysis, subgroups, publication figures, final package consistency audit |
| 06 | `06_statistics/01...#04...` | Paired patient-level bootstrap comparisons |

## Reproducing the external validation

External validation runs in two stages.

**Stage 1 - inference from checkpoints (or saved tabular estimators).**

```bash
python 04_ensembling/05_external_inference_from_checkpoints.py \
    --manifest config/external_inference.json \
    --output-dir results/external_center3_eval
```

The manifest lists the clinical and radiomics tables, the Centre 3 image
directories, and one checkpoint per deep model plus the two `joblib` estimator
bundles for the clinical and radiomics baselines. Every checkpoint must contain
an `inference_state` block; a checkpoint without one is rejected rather than
re-fitted, because re-estimating the imputation or scaling statistics on Centre
3 would invalidate the locked evaluation. The script writes one directory per
model, each containing `external_predictions.csv` with `left_pred_raw` /
`right_pred_raw` (regression head output before the non-negative constraint) and
`left_pred` / `right_pred` (after `max(0, .)`). The DFME file is built by
averaging the two components' **raw** GFR values 0.5:0.5 and then clipping the
average at zero.

**Stage 2 - evaluation.**

```bash
python 04_ensembling/04_external_validation_center3.py \
    --predictions-dir results/external_center3_eval \
    --output-dir results/external169
```

`--predictions-dir` must contain one sub-directory per model, each holding
`external_predictions.csv` with the columns `case_id`, `left_true`,
`right_true`, `left_pred`, `right_pred`, `left_cls_true`, `right_cls_true`,
`left_cls_prob`, `right_cls_prob` and `left_pred_raw`, `right_pred_raw`.

The evaluator refuses to continue unless the cohort contains exactly 169
patients, 338 kidney observations and 28 positive kidneys, and unless the DFME
column equals the prespecified 0.5:0.5 average of its two components after the
non-negative constraint is applied to the averaged prediction.

### Verification against the published table

Running the released script on the locked outputs reproduces Table 2B:

| Model | CCC (95% CI) | MAE (95% CI) | RMSE (95% CI) | AUROC (95% CI) |
| --- | --- | --- | --- | --- |
| Clinical-only | 0.267 (0.187-0.347) | 14.09 (12.91-15.34) | 17.18 (15.82-18.61) | 0.524 (0.416-0.626) |
| Radiomics-only | 0.443 (0.367-0.512) | 12.78 (11.72-13.94) | 16.26 (14.77-17.86) | 0.865 (0.814-0.911) |
| NCCT-only | 0.555 (0.488-0.617) | 12.33 (11.39-13.36) | 15.21 (14.09-16.38) | 0.920 (0.874-0.957) |
| NCCT + clinical | 0.742 (0.693-0.785) | 9.49 (8.63-10.38) | 12.24 (11.16-13.34) | 0.930 (0.882-0.968) |
| NCCT + radiomics | 0.594 (0.536-0.648) | 11.88 (10.90-12.87) | 14.84 (13.78-15.90) | 0.935 (0.900-0.965) |
| Early concatenation | 0.741 (0.703-0.774) | 9.76 (9.02-10.51) | 12.14 (11.33-12.97) | 0.945 (0.911-0.972) |
| Dual-branch cross-attention | 0.802 (0.772-0.828) | 8.68 (8.05-9.36) | 10.62 (9.89-11.40) | 0.961 (0.938-0.980) |
| DFME | 0.781 (0.749-0.809) | 9.05 (8.40-9.76) | 11.10 (10.34-11.88) | 0.958 (0.933-0.978) |

Running the stage-05 pipeline (`05_evaluation/01` to `05_evaluation/04`) on the
locked tables reproduces the same external values: DFME CCC 0.781
(0.749-0.809), MAE 9.05, AUROC 0.958 (0.933-0.978) and AUPRC 0.695.

Point estimates reproduce the published table at the reported precision.
Bootstrap bounds agree to within 0.005 for CCC and AUROC and within 0.03
mL/min/1.73 m2 for MAE and RMSE. The released seed is 2026; percentile bounds
move by a few thousandths of a unit if the seed is changed.

## Model architecture wording

The image encoder in `03_training/01_train_proposed_cross_attention_full.py`
(and in the ablations) convolves the **whole bilateral CT volume**. The left and
right predictions come from side-specific pooling, cross-attention and
regression/classification heads that read the same bilateral feature map. The
implementation is therefore a *shared bilateral image feature map with
side-specific pooling and prediction heads*, not two independently encoded
same-side image branches. Manifests and manuscripts should use that wording.
A true same-side branch would need per-side cropping and a new experiment tag;
it is not implemented here and would be a separate, newly trained model.

## Environment

See `REQUIREMENTS.txt`. The locked analysis ran on Python 3.9 with PyTorch
2.4.0+cu124 on a single 24 GB GPU; nnU-Net v2 2.5.2 was used for segmentation
in its own environment.

| Environment | Needed for | Notes |
| --- | --- | --- |
| CPU evaluation | `05_evaluation`, `06_statistics`, `04_ensembling/04` | pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, openpyxl |
| Deep learning | `03_training`, `04_ensembling/05` | torch 2.4.0, monai 1.3.0, nibabel, tqdm, joblib |
| nnU-Net (separate) | `01_segmentation` | nnU-Net v2 2.5.2; installed in its own environment |

This distribution contains the analysis and figure code only. The internal test
harness used during development (synthetic fixed-value regression suites,
including the GFR label/unit contract) and the optional DICOM-header utility are
not redistributed; they do not affect any number in the manuscript. The GFR unit
contract they enforce is stated here: raw GFR is preserved in
`left_gfr_original` / `right_gfr_original`, SRF labels are computed from those
raw values, the model target is standardised once with training statistics,
inverse transformation restores mL/min/1.73 m², and tabular external predictions
remain in original units.

## What this repository deliberately does not do

- No script re-fits, re-weights, recalibrates or re-thresholds a model on
  Centre 3.
- No script reads Centre 3 outcomes during training or feature selection.
- No script writes to the raw image or clinical source directories.

## Licence

MIT (see `LICENSE`). The licence covers the code only; patient images,
clinical records, model checkpoints and derived feature tables are not
distributed and remain subject to the data-use agreement.
