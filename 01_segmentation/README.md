# Kidney segmentation (nnU-Net v2)

Bilateral kidney masks were produced with the public nnU-Net v2 framework
(citation 16 in the manuscript). This directory contains the two scripts that
define the locked segmentation recipe; nnU-Net itself is installed from its
public release and is not redistributed here.

## Locked recipe

| Item | Value |
| --- | --- |
| Dataset | `Dataset817_NFHNCCTKidney` (labels: 0 background, 1 left kidney, 2 right kidney) |
| Configuration | `3d_fullres` |
| Plans | `nnUNetPlans_StrictTransfer` (contrast-enhanced transfer) and `nnUNetPlans_StrictTransferNN` (nearest-neighbour mask geometry) |
| Trainer | `nnUNetTrainer_250epochs_NoMirroring` (all mirroring disabled, because left and right are distinct semantic labels) |
| Fold | 0 |
| Seed | 2026 |
| Initialisation | Kidney segmentation checkpoint trained on contrast-enhanced CT (see Methods) |
| Locked split | 332 manually annotated NCCT cases for fine-tuning, 83 held out for checkpoint selection |
| Manual annotation | Reference masks delineated in 3D Slicer, reviewed by a second senior radiologist, disagreements resolved by consensus |
| Software | nnunetv2 2.5.2, PyTorch 2.4.0+cu124, Python 3.9 |

The checkpoint used for every study scan is the fold-0 `checkpoint_best.pth`
selected on the fixed 83-case development set. No centre-3 scan influenced
training, checkpoint selection or post-processing.

## Scripts

1. `01_nnunet_transfer_finetune.py` — loads the contrast-enhanced CT
   checkpoint with a strict state-dict check (identical keys, identical shapes,
   fresh optimiser and schedule), verifies that the preprocessed cache, the
   nearest-neighbour resampling policy and the 332/83 split are intact, then
   fine-tunes for 250 epochs without mirroring.
2. `02_nnunet_predict_kidney_masks.py` — applies the frozen checkpoint to every
   scan of the locked cohort, writing one mask per study.

## Environment

```bash
export GFR_SEG_ROOT=/path/to/nnunet_work              # holds nnUNet_raw, nnUNet_preprocessed, nnUNet_results
export GFR_SEG_PRETRAINED_CHECKPOINT=/path/to/kidney_segmentation_checkpoint.pth
export GFR_PROJECT_ROOT=/path/to/project              # only used by the inference script
```

The fine-tuning script additionally expects `nnUNet_raw`, `nnUNet_preprocessed`
and `nnUNet_results` to resolve inside `GFR_SEG_ROOT`.
