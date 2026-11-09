#!/usr/bin/env python3
"""Train the clinical-query cross-attention ablation on Centers 1 and 2."""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from monai.transforms import (
    LoadImaged, EnsureChannelFirstD, ResizeD, Compose, MapTransform, ToTensorD,
    RandAffined, RandGaussianNoised, RandAdjustContrastd, Spacingd, CropForegroundd
)
from monai.data import CacheDataset, PersistentDataset
from monai.utils import set_determinism
# Shared preprocessing / image pipeline (single implementation for 01-05).
from common import monai_pipeline as _image_pipeline
from common.checkpoint import build_inference_state
from common.preprocessing import (
    DatasetBundle,
    FittedScaler,
    active_modalities_for_tag,
    build_development_bundle,
)

RUN_SEED = int(os.environ.get("RUN_SEED", "66"))
set_determinism(seed=RUN_SEED)

ABLATION_MODE = os.environ.get("EXPERIMENT_TAG", "ours_clin_only")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
TOP_K_CHECKPOINTS = int(os.environ.get("TOP_K_CHECKPOINTS", "5"))
TARGET_SHAPE = (16, 256, 256)
USE_PERSISTENT_PREPROCESS_CACHE = os.environ.get("USE_PERSISTENT_PREPROCESS_CACHE", "1") == "1"
PREPROCESSED_CACHE_ROOT = os.environ.get(
    "PREPROCESSED_CACHE_ROOT",
    "cache"
)
PREPROCESSED_CACHE_DIR = Path(PREPROCESSED_CACHE_ROOT) / f"srf15_{ABLATION_MODE}_{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}"
CCC_START_EPOCH = 15
CCC_WEIGHT = 0.35
SYM_WEIGHT = 0.05
TOTAL_GFR_WEIGHT = 0.03
SRF_REG_WEIGHT = 0.10
CLS_WEIGHT = 0.10
SRF_THRESHOLD = 0.15
BALANCED_CCC_MIN_WEIGHT = 0.35

RAD_FEATURES = [
    "Left_wavelet-LLL_firstorder_Skewness",
    "Left_original_firstorder_10Percentile",
    "Left_original_shape_Flatness",
    "Left_log-sigma-3-0-mm-3D_firstorder_Skewness",
    "Left_wavelet-LHL_firstorder_Median",
    "Left_log-sigma-3-0-mm-3D_glrlm_RunVariance",
    "Right_wavelet-LLL_firstorder_Skewness",
    "Right_original_firstorder_10Percentile",
    "Right_original_shape_Flatness",
    "Right_log-sigma-3-0-mm-3D_firstorder_Skewness",
    "Right_wavelet-LHL_firstorder_Median",
    "Right_log-sigma-3-0-mm-3D_glrlm_RunVariance",
]

class MovingWindowCCCLoss(nn.Module):
    def __init__(self, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.history_pred = None
        self.history_true = None

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        if self.history_pred is not None:
            all_pred = torch.cat([y_pred, self.history_pred])
            all_true = torch.cat([y_true, self.history_true])
        else:
            all_pred = y_pred
            all_true = y_true

        y_true_mean = torch.mean(all_true)
        y_pred_mean = torch.mean(all_pred)
        
        y_true_var = torch.var(all_true, unbiased=False)
        y_pred_var = torch.var(all_pred, unbiased=False)
        
        cov = torch.mean((all_pred - y_pred_mean) * (all_true - y_true_mean))
        
        ccc = (2.0 * cov) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean)**2 + 1e-8)
        
        if self.training:
            with torch.no_grad():
                new_pred = y_pred.detach()
                new_true = y_true.detach()
                
                if self.history_pred is None:
                    self.history_pred = new_pred
                    self.history_true = new_true
                else:
                    self.history_pred = torch.cat([new_pred, self.history_pred])[:self.window_size]
                    self.history_true = torch.cat([new_true, self.history_true])[:self.window_size]
                    
        return 1.0 - ccc

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels), nn.LeakyReLU(0.01, inplace=True), nn.Dropout3d(dropout),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD, diffH, diffW = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3), x2.size(4)-x1.size(4)
        x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2, diffD - diffD // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class AdaptivePoolingModule3D(nn.Module):
    def __init__(self, in_channels, out_channels=16, pool_size=4):
        super().__init__()
        mid_c = in_channels // 2 if in_channels > 1 else 1
        gn_groups = 1 if mid_c < 8 else 8 
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, mid_c, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, mid_c), nn.LeakyReLU(0.01, inplace=True), nn.Dropout3d(0.1),
            nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size)),
            nn.Conv3d(mid_c, out_channels, kernel_size=1), nn.LeakyReLU(0.01, inplace=True),
            nn.Flatten(), nn.Linear(out_channels * pool_size**3, 16), nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x): return self.features(x)

class UNet3D_MultiTask_Decoupled(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, bilinear=True, embed_dim=128, ablation_mode="ours"):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.embed_dim = embed_dim
        
        self.inc = DoubleConv3D(in_channels, 64)
        self.down1 = Down3D(64, 128); self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512); factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        self.up1 = Up3D(1024, 512 // factor, bilinear); self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear); self.up4 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

        self.f1_l = AdaptivePoolingModule3D(256, 16); self.f2_l = AdaptivePoolingModule3D(128, 16)
        self.f3_l = AdaptivePoolingModule3D(64, 16); self.f4_l = AdaptivePoolingModule3D(64, 16)
        self.f0_l = AdaptivePoolingModule3D(1, 16)
        self.f1_r, self.f2_r, self.f3_r, self.f4_r, self.f0_r = [AdaptivePoolingModule3D(c, 16) for c in [256, 128, 64, 64, 1]]

        self.prior_encoder_l = nn.Sequential(nn.Linear(11, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.prior_encoder_r = nn.Sequential(nn.Linear(11, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, embed_dim))
        
        self.img_kv_proj = nn.Sequential(nn.Linear(16, embed_dim), nn.LayerNorm(embed_dim))
        self.cross_attn_l = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.cross_attn_r = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.randn(1, 5, embed_dim))

        self.gate_l = nn.Sequential(nn.Linear(80 + embed_dim, embed_dim), nn.Sigmoid())
        self.gate_r = nn.Sequential(nn.Linear(80 + embed_dim, embed_dim), nn.Sigmoid())

        combined_dim = embed_dim + embed_dim

        # =========================================================================
        # =========================================================================
        self.router_reg_l = nn.Sequential(nn.Linear(combined_dim, 128), nn.LayerNorm(128), nn.Dropout(0.3), nn.GELU())
        self.router_reg_r = nn.Sequential(nn.Linear(combined_dim, 128), nn.LayerNorm(128), nn.Dropout(0.3), nn.GELU())
        self.head_reg_l = nn.Linear(128, 1)
        self.head_reg_r = nn.Linear(128, 1)
        
        self.router_cls_l = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.LayerNorm(128), nn.Dropout(0.5), nn.GELU(),
            nn.Linear(128, 32), nn.GELU()
        )
        self.router_cls_r = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.LayerNorm(128), nn.Dropout(0.5), nn.GELU(),
            nn.Linear(128, 32), nn.GELU()
        )
        self.head_cls_l = nn.Linear(32, 1)
        self.head_cls_r = nn.Linear(32, 1)

        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x, prior_vec_l, prior_vec_r):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x_up = self.up1(x5, x4); f1_l, f1_r = self.f1_l(x_up), self.f1_r(x_up)
        x_up = self.up2(x_up, x3); f2_l, f2_r = self.f2_l(x_up), self.f2_r(x_up)
        x_up = self.up3(x_up, x2); f3_l, f3_r = self.f3_l(x_up), self.f3_r(x_up)
        x_up = self.up4(x_up, x1); f4_l, f4_r = self.f4_l(x_up), self.f4_r(x_up)
        x_out = self.outc(x_up); f0_l, f0_r = self.f0_l(x_out), self.f0_r(x_out)

        flat_l_img = torch.cat([f1_l, f2_l, f3_l, f4_l, f0_l], dim=1)
        flat_r_img = torch.cat([f1_r, f2_r, f3_r, f4_r, f0_r], dim=1)

        latent_prior_l = self.prior_encoder_l(prior_vec_l)
        latent_prior_r = self.prior_encoder_r(prior_vec_r)

        img_kv_l = self.img_kv_proj(torch.stack([f1_l, f2_l, f3_l, f4_l, f0_l], dim=1)) + self.pos_embed 
        img_kv_r = self.img_kv_proj(torch.stack([f1_r, f2_r, f3_r, f4_r, f0_r], dim=1)) + self.pos_embed

        attn_l, attn_weight_l = self.cross_attn_l(latent_prior_l.unsqueeze(1), img_kv_l, img_kv_l)
        attn_r, attn_weight_r = self.cross_attn_r(latent_prior_r.unsqueeze(1), img_kv_r, img_kv_r)
        attn_l, attn_r = attn_l.squeeze(1), attn_r.squeeze(1)

        gated_visual_l = self.gate_l(torch.cat([flat_l_img, attn_l], dim=1)) * attn_l
        gated_visual_r = self.gate_r(torch.cat([flat_r_img, attn_r], dim=1)) * attn_r

        dropped_prior_l = F.dropout(latent_prior_l, p=0.4, training=self.training)
        dropped_prior_r = F.dropout(latent_prior_r, p=0.4, training=self.training)

        feat_l = torch.cat([gated_visual_l, dropped_prior_l], dim=1)
        feat_r = torch.cat([gated_visual_r, dropped_prior_r], dim=1)

        # =========================================================================
        # =========================================================================
        feat_reg_l = self.router_reg_l(feat_l)
        feat_reg_r = self.router_reg_r(feat_r)
        out_reg_l = self.head_reg_l(feat_reg_l).squeeze(1)
        out_reg_r = self.head_reg_r(feat_reg_r).squeeze(1)

        feat_cls_l = self.router_cls_l(feat_l.detach())
        feat_cls_r = self.router_cls_r(feat_r.detach())
        out_cls_l = self.head_cls_l(feat_cls_l).squeeze(1)
        out_cls_r = self.head_cls_r(feat_cls_r).squeeze(1)
        # =========================================================================

        if not self.training:
            return out_reg_l, out_reg_r, out_cls_l, out_cls_r, attn_weight_l, attn_weight_r
        return out_reg_l, out_reg_r, out_cls_l, out_cls_r


# ---------------------------------------------------------------------------
# Image transforms.
# The implementation lives in ``common/monai_pipeline.py``: deterministic
# preprocessing is cacheable, random augmentation runs per __getitem__ and is
# never cached.  See that module for the rationale.
# ---------------------------------------------------------------------------
PatientSpecificKidneyNormalizer = _image_pipeline.PatientSpecificKidneyNormalizer


def get_preprocess_transforms():
    """Deterministic, cacheable preprocessing (no randomness)."""
    return _image_pipeline.get_deterministic_transforms(TARGET_SHAPE)


def get_train_transforms():
    """Uncached training pipeline: deterministic + per-access augmentation."""
    return _image_pipeline.get_train_transforms(TARGET_SHAPE)


def get_val_transforms():
    """Validation: deterministic preprocessing + kidney normalisation only."""
    return _image_pipeline.get_val_transforms(TARGET_SHAPE)


def create_dataset():
    """Build datasets with the corrected cache policy.

    Only the deterministic part is persisted; random augmentation is applied on
    every access, so it is re-sampled for every epoch and every sample.
    """
    train_samples, val_samples = prepare_multicenter_dataset()
    print(f" Train split: n={len(train_samples)}")
    print(f" Val split: n={len(val_samples)}")
    if USE_PERSISTENT_PREPROCESS_CACHE:
        PREPROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for split_name, samples in (("train", train_samples), ("val", val_samples)):
            pd.DataFrame({
                "case_id": [s["case_id"] for s in samples],
                "left_gfr_original": [s["left_gfr_original"] for s in samples],
                "right_gfr_original": [s["right_gfr_original"] for s in samples],
                "label_cls_l": [float(s["label_cls_l"][0]) for s in samples],
                "label_cls_r": [float(s["label_cls_r"][0]) for s in samples],
            }).to_csv(PREPROCESSED_CACHE_DIR / f"{split_name}_manifest.csv", index=False)
        print(f" Persistent deterministic cache enabled: {PREPROCESSED_CACHE_DIR}")
        print(" Random augmentation is applied per epoch and is NOT cached.")
    else:
        print(" Persistent cache disabled: deterministic preprocessing recomputed per access.")
        print(" Random augmentation is applied per epoch and is NOT cached.")
    return _image_pipeline.build_datasets(
        train_samples,
        val_samples,
        target_shape=TARGET_SHAPE,
        cache_root=PREPROCESSED_CACHE_DIR,
        use_cache=USE_PERSISTENT_PREPROCESS_CACHE,
    )

CENTER_CONFIGS = [
    {"name": "Center_1", "ct_dir": os.environ.get("CENTER1_CT_DIR", "data/center1/ct"), "mask_dir": os.environ.get("CENTER1_MASK_DIR", "data/center1/mask")},
    {"name": "Center_2", "ct_dir": os.environ.get("CENTER2_CT_DIR", "data/center2/ct"), "mask_dir": os.environ.get("CENTER2_MASK_DIR", "data/center2/mask")}
]
CLINICAL_CSV_PATH = os.environ.get("CLINICAL_CSV_PATH", "data/clinical.csv")
RADIOMICS_CSV_PATH = os.environ.get("RADIOMICS_CSV_PATH", "data/radiomics.csv")

BATCH_SIZE = 4
ACCUMULATION_STEPS = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 120
INIT_LR, MIN_LR, WEIGHT_DECAY = 1e-4, 1e-6, 1e-2
GRAD_CLIP_NORM, EARLY_STOP_PATIENCE, WARMUP_EPOCHS = 0.5, 80, 10
CSV_CLIN_COLS = ["height", "weight", "sex", "age", "SCr"]

gfr_scaler_left, gfr_scaler_right = FittedScaler(), FittedScaler()

def inverse_gfr_tensors(left_z, right_z, device):
    dtype = torch.float32
    left_mean = torch.tensor(float(gfr_scaler_left.mean_[0]), dtype=dtype, device=device)
    left_scale = torch.tensor(float(gfr_scaler_left.scale_[0]), dtype=dtype, device=device)
    right_mean = torch.tensor(float(gfr_scaler_right.mean_[0]), dtype=dtype, device=device)
    right_scale = torch.tensor(float(gfr_scaler_right.scale_[0]), dtype=dtype, device=device)
    left_orig = left_z.float() * left_scale + left_mean
    right_orig = right_z.float() * right_scale + right_mean
    pooled_scale = torch.sqrt((left_scale.pow(2) + right_scale.pow(2)) / 2.0)
    total_scale = left_scale + right_scale
    return left_orig, right_orig, pooled_scale, total_scale

def apply_prior_ablation(norm_clin, norm_rad_l, norm_rad_r):
    norm_clin = norm_clin.astype(np.float32)
    norm_rad_l = norm_rad_l.astype(np.float32)
    norm_rad_r = norm_rad_r.astype(np.float32)

    if ABLATION_MODE in ["ours_clin_only", "concat_clin_only"]:
        norm_rad_l = np.zeros_like(norm_rad_l, dtype=np.float32)
        norm_rad_r = np.zeros_like(norm_rad_r, dtype=np.float32)
    elif ABLATION_MODE in ["ours_rad_only", "concat_rad_only"]:
        norm_clin = np.zeros_like(norm_clin, dtype=np.float32)
    elif ABLATION_MODE == "vision_only":
        norm_clin = np.zeros_like(norm_clin, dtype=np.float32)
        norm_rad_l = np.zeros_like(norm_rad_l, dtype=np.float32)
        norm_rad_r = np.zeros_like(norm_rad_r, dtype=np.float32)

    return norm_clin, norm_rad_l, norm_rad_r

DATASET_STATE: dict = {}


def prepare_multicenter_dataset():
    """Load the development cohort, split by patient and fit preprocessing on train.

    Medians / means / scales are estimated on the internal training split only
    and stored in ``DATASET_STATE`` so that every checkpoint can carry them.
    """
    global DATASET_STATE
    df_clin = pd.read_csv(CLINICAL_CSV_PATH)
    df_rad = pd.read_csv(RADIOMICS_CSV_PATH)
    df = pd.merge(df_clin, df_rad, on="ID", how='inner')
    df = df[df['Center'].isin(["Center_1", "Center_2"])].copy()
    df.reset_index(drop=True, inplace=True)

    path_map = {cfg["name"]: {"ct": Path(cfg["ct_dir"]), "mask": Path(cfg["mask_dir"])} for cfg in CENTER_CONFIGS}
    valid_indices, ct_paths, mask_paths = [], [], []
    for idx, row in df.iterrows():
        pid, center = str(row["ID"]).strip(), row["Center"]
        ct_dir, mask_dir = path_map[center]["ct"], path_map[center]["mask"]
        ct_path = ct_dir / f"{pid}.nii.gz" if (ct_dir / f"{pid}.nii.gz").exists() else ct_dir / f"{pid}.nii"
        mask_path = mask_dir / f"{pid}.nii.gz" if (mask_dir / f"{pid}.nii.gz").exists() else mask_dir / f"{pid}.nii"
        if ct_path.exists() and mask_path.exists():
            valid_indices.append(idx)
            ct_paths.append(str(ct_path))
            mask_paths.append(str(mask_path))

    df_valid = df.loc[valid_indices].copy()
    df_valid['ct_path'], df_valid['mask_path'] = ct_paths, mask_paths

    bundle = build_development_bundle(
        df_valid,
        radiomics_features=RAD_FEATURES,
        clinical_features=CSV_CLIN_COLS,
        target_shape=TARGET_SHAPE,
        srf_threshold=SRF_THRESHOLD,
        test_size=0.2,
        random_state=42,
        prior_ablation=apply_prior_ablation,
    )
    DATASET_STATE = bundle.state
    DATASET_STATE["image_preprocessing"] = _image_pipeline.image_preprocessing_spec(TARGET_SHAPE)
    gfr_stats = DATASET_STATE["preprocessing"]["gfr"]
    gfr_scaler_left.update_from_stats(gfr_stats["left_GFR"])
    gfr_scaler_right.update_from_stats(gfr_stats["right_GFR"])
    return bundle.train_samples, bundle.val_samples


def calculate_ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_mean, pred_mean = np.mean(y_true), np.mean(y_pred)
    cov = np.cov(y_true, y_pred)[0, 1]
    return np.clip((2 * cov) / (np.var(y_true, ddof=1) + np.var(y_pred, ddof=1) + (true_mean - pred_mean)**2 + 1e-8), -1.0, 1.0)

def safe_auc(y_true, y_pred):
    try:
        return 0.5 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_pred)
    except: return 0.5

def warmup_lr_scheduler(optimizer, epoch, warmup_epochs, init_lr, min_lr):
    lr = init_lr * (epoch + 1) / warmup_epochs if epoch < warmup_epochs else min_lr + (init_lr - min_lr) * 0.5 * (1 + np.cos((epoch - warmup_epochs) / (EPOCHS - warmup_epochs) * np.pi))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def train_one_epoch(model, dataloader, criterion_reg_base, criterion_reg_ccc_l, criterion_reg_ccc_r, criterion_cls, optimizer, device, epoch, scaler):
    model.train()
    criterion_reg_ccc_l.train()
    criterion_reg_ccc_r.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        ct = batch["ct"].to(device, non_blocking=True)
        prior_l = batch["prior_l"].float().to(device, non_blocking=True)
        prior_r = batch["prior_r"].float().to(device, non_blocking=True)
        left_gfr = batch["left_gfr"].squeeze(1).float().to(device, non_blocking=True)
        right_gfr = batch["right_gfr"].squeeze(1).float().to(device, non_blocking=True)
        lbl_cls_l = batch["label_cls_l"].squeeze(1).float().to(device, non_blocking=True)
        lbl_cls_r = batch["label_cls_r"].squeeze(1).float().to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            out_reg_l, out_reg_r, out_cls_l, out_cls_r = model(ct, prior_l, prior_r)
            
            loss_mse = (1.0 * criterion_reg_base(out_reg_l, left_gfr) + 
                        1.2 * criterion_reg_base(out_reg_r, right_gfr)) / 2.2
                        
            loss_ccc = (1.0 * criterion_reg_ccc_l(out_reg_l, left_gfr) + 
                        1.2 * criterion_reg_ccc_r(out_reg_r, right_gfr)) / 2.2
                        
            pred_l_orig, pred_r_orig, pooled_gfr_scale, total_gfr_scale = inverse_gfr_tensors(out_reg_l, out_reg_r, device)
            true_l_orig, true_r_orig, _, _ = inverse_gfr_tensors(left_gfr, right_gfr, device)

            diff_pred = torch.abs(pred_l_orig - pred_r_orig) / (pooled_gfr_scale + 1e-8)
            diff_true = torch.abs(true_l_orig - true_r_orig) / (pooled_gfr_scale + 1e-8)
            loss_sym = F.smooth_l1_loss(diff_pred, diff_true)

            total_pred = (pred_l_orig + pred_r_orig) / (total_gfr_scale + 1e-8)
            total_true = (true_l_orig + true_r_orig) / (total_gfr_scale + 1e-8)
            loss_total = F.smooth_l1_loss(total_pred, total_true)

            pred_l_pos = torch.clamp(pred_l_orig, min=0.0)
            pred_r_pos = torch.clamp(pred_r_orig, min=0.0)
            true_total = true_l_orig + true_r_orig + 1e-8
            pred_total = pred_l_pos + pred_r_pos + 1e-8
            pred_srf_l = pred_l_pos / pred_total
            pred_srf_r = pred_r_pos / pred_total
            true_srf_l = true_l_orig / true_total
            true_srf_r = true_r_orig / true_total
            loss_srf_reg = (
                F.smooth_l1_loss(pred_srf_l, true_srf_l) +
                F.smooth_l1_loss(pred_srf_r, true_srf_r)
            ) / 2.0
            
            loss_cls = (criterion_cls(out_cls_l, lbl_cls_l) + criterion_cls(out_cls_r, lbl_cls_r)) / 2.0
            
            reg_phase = min(1.0, max(0.0, (epoch + 1 - CCC_START_EPOCH) / 10.0))
            loss_reg = loss_mse + reg_phase * (
                CCC_WEIGHT * loss_ccc +
                SYM_WEIGHT * loss_sym +
                TOTAL_GFR_WEIGHT * loss_total +
                SRF_REG_WEIGHT * loss_srf_reg
            )
            loss = loss_reg + CLS_WEIGHT * loss_cls
        
        scaler.scale(loss / ACCUMULATION_STEPS).backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                        
        total_loss += loss.item() * ct.size(0)
        pbar.set_postfix({"L_Total": loss.item(), "L_SRF": loss_srf_reg.item(), "L_Sym": loss_sym.item()})
                
    if (step + 1) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion_reg_base, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    all_case_ids = []
    all_l_true_orig, all_r_true_orig, all_l_true_lbl, all_r_true_lbl = [], [], [], []
    all_l_pred_reg, all_r_pred_reg, all_l_pred_prob, all_r_pred_prob = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            ct = batch["ct"].to(device, non_blocking=True)
            prior_l = batch["prior_l"].float().to(device, non_blocking=True)
            prior_r = batch["prior_r"].float().to(device, non_blocking=True)
            left_gfr = batch["left_gfr"].squeeze(1).float().to(device, non_blocking=True)
            right_gfr = batch["right_gfr"].squeeze(1).float().to(device, non_blocking=True)
            lbl_cls_l = batch["label_cls_l"].squeeze(1).float().to(device, non_blocking=True)
            lbl_cls_r = batch["label_cls_r"].squeeze(1).float().to(device, non_blocking=True)

            out_reg_l, out_reg_r, out_cls_l, out_cls_r, _, _ = model(ct, prior_l, prior_r)
            
            loss_mse = (1.0 * criterion_reg_base(out_reg_l, left_gfr) + 1.2 * criterion_reg_base(out_reg_r, right_gfr)) / 2.2
            pred_l_orig_t, pred_r_orig_t, pooled_gfr_scale, total_gfr_scale = inverse_gfr_tensors(out_reg_l, out_reg_r, device)
            true_l_orig_t, true_r_orig_t, _, _ = inverse_gfr_tensors(left_gfr, right_gfr, device)
            loss_sym = F.smooth_l1_loss(
                torch.abs(pred_l_orig_t - pred_r_orig_t) / (pooled_gfr_scale + 1e-8),
                torch.abs(true_l_orig_t - true_r_orig_t) / (pooled_gfr_scale + 1e-8)
            )
            loss_total = F.smooth_l1_loss(
                (pred_l_orig_t + pred_r_orig_t) / (total_gfr_scale + 1e-8),
                (true_l_orig_t + true_r_orig_t) / (total_gfr_scale + 1e-8)
            )
            pred_l_pos = torch.clamp(pred_l_orig_t, min=0.0)
            pred_r_pos = torch.clamp(pred_r_orig_t, min=0.0)
            true_total = true_l_orig_t + true_r_orig_t + 1e-8
            pred_total = pred_l_pos + pred_r_pos + 1e-8
            loss_srf_reg = (
                F.smooth_l1_loss(pred_l_pos / pred_total, true_l_orig_t / true_total) +
                F.smooth_l1_loss(pred_r_pos / pred_total, true_r_orig_t / true_total)
            ) / 2.0
            loss_cls = (criterion_cls(out_cls_l, lbl_cls_l) + criterion_cls(out_cls_r, lbl_cls_r)) / 2.0
            
            total_loss += (
                loss_mse +
                SYM_WEIGHT * loss_sym +
                TOTAL_GFR_WEIGHT * loss_total +
                SRF_REG_WEIGHT * loss_srf_reg +
                CLS_WEIGHT * loss_cls
            ).item() * ct.size(0)
            
            pred_l = gfr_scaler_left.inverse_transform(out_reg_l.cpu().numpy().reshape(-1, 1)).reshape(-1)
            pred_r = gfr_scaler_right.inverse_transform(out_reg_r.cpu().numpy().reshape(-1, 1)).reshape(-1)
            prob_l = torch.sigmoid(out_cls_l).cpu().numpy()
            prob_r = torch.sigmoid(out_cls_r).cpu().numpy()
            
            all_case_ids.extend([str(x) for x in batch["case_id"]])
            all_l_true_orig.extend(np.array(batch["left_gfr_original"]).flatten())
            all_r_true_orig.extend(np.array(batch["right_gfr_original"]).flatten())
            all_l_true_lbl.extend(lbl_cls_l.cpu().numpy())
            all_r_true_lbl.extend(lbl_cls_r.cpu().numpy())
            
            all_l_pred_reg.extend(pred_l)
            all_r_pred_reg.extend(pred_r)
            all_l_pred_prob.extend(prob_l)
            all_r_pred_prob.extend(prob_r)

    lt_arr, lp_arr = np.array(all_l_true_orig), np.array(all_l_pred_reg)
    rt_arr, rp_arr = np.array(all_r_true_orig), np.array(all_r_pred_reg)
    
    both_true_lbl = np.concatenate([np.array(all_l_true_lbl), np.array(all_r_true_lbl)])
    both_pred_prob = np.concatenate([np.array(all_l_pred_prob), np.array(all_r_pred_prob)])
    auc_srf_non_functioning = safe_auc(both_true_lbl, both_pred_prob)
            
    return {
        "avg_loss": total_loss / len(dataloader.dataset),
        "left_ccc": calculate_ccc(lt_arr, lp_arr),
        "right_ccc": calculate_ccc(rt_arr, rp_arr),
        "auc_srf_non_functioning": auc_srf_non_functioning,
        "raw_data": {
            "case_id": np.array(all_case_ids),
            "l_true": lt_arr, "l_pred": lp_arr,
            "r_true": rt_arr, "r_pred": rp_arr,
            "cls_true": both_true_lbl, "cls_prob": both_pred_prob
        }
    }

def save_prediction_csv(raw_data, csv_path):
    cls_true = np.asarray(raw_data["cls_true"]).reshape(-1)
    cls_prob = np.asarray(raw_data["cls_prob"]).reshape(-1)
    n_cases = len(raw_data["case_id"])
    pred_df = pd.DataFrame({
        "case_id": raw_data["case_id"],
        "left_true": raw_data["l_true"],
        "left_pred": raw_data["l_pred"],
        "right_true": raw_data["r_true"],
        "right_pred": raw_data["r_pred"],
        "left_cls_true": cls_true[:n_cases],
        "left_cls_prob": cls_prob[:n_cases],
        "right_cls_true": cls_true[n_cases:],
        "right_cls_prob": cls_prob[n_cases:],
    })
    pred_df.to_csv(csv_path, index=False)

def save_topk_checkpoint(saved_items, metric, prefix, checkpoint, raw_data):
    if TOP_K_CHECKPOINTS <= 0:
        return saved_items
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch = int(checkpoint["epoch"]) + 1
    avg_ccc = float(checkpoint["best_avg_ccc"])
    balanced_ccc = float(checkpoint["balanced_ccc"])
    stem = (
        f"{prefix}_{ABLATION_MODE}_seed{RUN_SEED}_"
        f"epoch{epoch:03d}_avg{avg_ccc:.4f}_bal{balanced_ccc:.4f}"
    )
    pth_path = OUTPUT_DIR / f"{stem}.pth"
    csv_path = OUTPUT_DIR / f"{stem}_predictions.csv"
    torch.save(checkpoint, pth_path)
    save_prediction_csv(raw_data, csv_path)
    saved_items.append((float(metric), pth_path, csv_path))
    saved_items = sorted(saved_items, key=lambda x: x[0], reverse=True)
    while len(saved_items) > TOP_K_CHECKPOINTS:
        _, old_pth, old_csv = saved_items.pop()
        old_pth.unlink(missing_ok=True)
        old_csv.unlink(missing_ok=True)
    return saved_items

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "experiment_tag": ABLATION_MODE,
        "entrypoint": "03_training/02_train_internal_ablation_attention_clinical_only.py",
        "architecture": "clinical_query_cross_attention_clin_only",
        "run_seed": RUN_SEED,
        "target_shape": TARGET_SHAPE,
        "clinical_features": CSV_CLIN_COLS,
        "radiomics_features": RAD_FEATURES,
        "srf_threshold": SRF_THRESHOLD,
    }
    print("="*70)
    print("Clinical-query cross-attention ablation training")
    print(f"Experiment: {ABLATION_MODE} | seed: {RUN_SEED} | output: {OUTPUT_DIR}")
    print("="*70)
    
    train_dataset, val_dataset = create_dataset()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    model = UNet3D_MultiTask_Decoupled(embed_dim=128, ablation_mode=ABLATION_MODE).to(DEVICE)

    # Everything needed to rebuild predictions outside this run: preprocessing
    # statistics, patient split, image pipeline and model definition.
    inference_state = build_inference_state(
        DATASET_STATE,
        architecture="clinical_query_cross_attention_clin_only",
        experiment_tag=ABLATION_MODE,
        model_class="UNet3D_MultiTask_Decoupled",
        model_kwargs={"embed_dim": 128, "ablation_mode": ABLATION_MODE},
        code_entrypoint="03_training/02_train_internal_ablation_attention_clinical_only.py",
        rad_features=RAD_FEATURES,
        clinical_features=CSV_CLIN_COLS,
        target_shape=TARGET_SHAPE,
        srf_threshold=SRF_THRESHOLD,
        run_seed=RUN_SEED,
        active_tabular_modalities=active_modalities_for_tag(ABLATION_MODE),
    )
    run_manifest["architecture_note"] = (
        "The image encoder processes the whole bilateral CT volume; left and right "
        "predictions come from side-specific pooling and heads over a shared "
        "bilateral feature map, not from two independently encoded same-side "
        "image branches."
    )
    run_manifest["preprocessing"] = DATASET_STATE["preprocessing"]
    run_manifest["split"] = DATASET_STATE["split"]
    run_manifest["image_preprocessing"] = DATASET_STATE["image_preprocessing"]
    run_manifest["cached_preprocessing"] = {
        "cached": "deterministic preprocessing only (load, spacing, crop, resize, tensor)",
        "runtime": (
            "random affine, kidney normalisation, Gaussian noise and contrast are "
            "applied in __getitem__ on every access; random transforms are not cached"
        ),
    }
    run_manifest.setdefault("rerun_required", "NEW_RUN_REQUIRED")
    run_manifest["checkpoint_inference_state_schema"] = inference_state["schema_version"]
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    
    criterion_reg_base = nn.MSELoss().to(DEVICE)
    criterion_reg_ccc_l = MovingWindowCCCLoss(window_size=32).to(DEVICE)
    criterion_reg_ccc_r = MovingWindowCCCLoss(window_size=32).to(DEVICE)
    train_cls_labels = np.concatenate([
        np.array([sample["label_cls_l"][0], sample["label_cls_r"][0]], dtype=np.float32)
        for sample in train_dataset.data
    ])
    pos_count = float(train_cls_labels.sum())
    neg_count = float(len(train_cls_labels) - pos_count)
    cls_pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight).to(DEVICE)
    print(f"SRF<15% positive kidneys: {int(pos_count)}/{len(train_cls_labels)} | pos_weight={cls_pos_weight.item():.3f}")
    
    swa_model = AveragedModel(model)
    swa_start_epoch = 35 
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5) 
    
    best_val_metric, best_balanced_metric, early_stop_counter = -1.0, -1.0, 0
    saved_avg_checkpoints, saved_balanced_checkpoints = [], []
    
    for epoch in range(EPOCHS):
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model) 
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            current_lr = warmup_lr_scheduler(optimizer, epoch, WARMUP_EPOCHS, INIT_LR, MIN_LR)
            
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f}) =====")
        
        train_loss = train_one_epoch(model, train_loader, criterion_reg_base, criterion_reg_ccc_l, criterion_reg_ccc_r, criterion_cls, optimizer, DEVICE, epoch, scaler)
        val_metrics = validate(model, val_loader, criterion_reg_base, criterion_cls, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['avg_loss']:.4f}")
        print(f"[Regression] Left CCC: {val_metrics['left_ccc']:.4f} | Right CCC: {val_metrics['right_ccc']:.4f}")
        print(f"[Classification] SRF<15% AUC: {val_metrics['auc_srf_non_functioning']:.4f}")
        
        left_ccc = val_metrics["left_ccc"]
        right_ccc = val_metrics["right_ccc"]
        avg_ccc = (left_ccc + right_ccc) / 2
        min_ccc = min(left_ccc, right_ccc)
        balanced_ccc = (1.0 - BALANCED_CCC_MIN_WEIGHT) * avg_ccc + BALANCED_CCC_MIN_WEIGHT * min_ccc
        print(f"[Selection] Mean CCC: {avg_ccc:.4f} | Minimum CCC: {min_ccc:.4f} | Balanced CCC: {balanced_ccc:.4f}")

        epoch_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_avg_ccc": avg_ccc,
            "left_ccc": left_ccc,
            "right_ccc": right_ccc,
            "min_ccc": min_ccc,
            "balanced_ccc": balanced_ccc,
            "auc_srf_non_functioning": val_metrics["auc_srf_non_functioning"],
            "plot_data": val_metrics["raw_data"],
            "run_seed": RUN_SEED,
            "experiment_tag": ABLATION_MODE,
            "rad_features": RAD_FEATURES,
            "inference_state": inference_state,
        }
        saved_avg_checkpoints = save_topk_checkpoint(
            saved_avg_checkpoints, avg_ccc, "top_avg", epoch_checkpoint, val_metrics["raw_data"]
        )
        saved_balanced_checkpoints = save_topk_checkpoint(
            saved_balanced_checkpoints, balanced_ccc, "top_balanced", epoch_checkpoint, val_metrics["raw_data"]
        )

        improved = False
        if avg_ccc > best_val_metric:
            best_val_metric = avg_ccc
            improved = True
            best_path = OUTPUT_DIR / f"best_mtl_decoupled_{ABLATION_MODE}_seed{RUN_SEED}_srf15_ccc_anchor.pth"
            torch.save(epoch_checkpoint, best_path)
            print(f" Updated Best ATD-MTL Model (Avg CCC: {best_val_metric:.4f}) -> {best_path}")

        if balanced_ccc > best_balanced_metric:
            best_balanced_metric = balanced_ccc
            improved = True
            balanced_checkpoint = dict(epoch_checkpoint)
            balanced_checkpoint["balanced_ccc"] = best_balanced_metric
            balanced_path = OUTPUT_DIR / f"best_mtl_decoupled_{ABLATION_MODE}_seed{RUN_SEED}_srf15_balanced_ccc.pth"
            torch.save(balanced_checkpoint, balanced_path)
            print(f" Updated Balanced ATD-MTL Model (Balanced CCC: {best_balanced_metric:.4f}) -> {balanced_path}")

        if improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE and epoch < swa_start_epoch:
                print(f"\n Early Stop Triggered at Epoch {epoch+1}")
                break

    if epoch >= swa_start_epoch:
        print("\n" + "="*40)
        print("Evaluating the final SWA model.")
        swa_val_metrics = validate(swa_model, val_loader, criterion_reg_base, criterion_cls, DEVICE)
        print(f"SWA [Regression] Left CCC: {swa_val_metrics['left_ccc']:.4f} | Right CCC: {swa_val_metrics['right_ccc']:.4f}")
        print(f"SWA [Classification] SRF<15% AUC: {swa_val_metrics['auc_srf_non_functioning']:.4f}")
        swa_avg_ccc = (swa_val_metrics["left_ccc"] + swa_val_metrics["right_ccc"]) / 2
        swa_min_ccc = min(swa_val_metrics["left_ccc"], swa_val_metrics["right_ccc"])
        swa_balanced_ccc = (
            (1.0 - BALANCED_CCC_MIN_WEIGHT) * swa_avg_ccc
            + BALANCED_CCC_MIN_WEIGHT * swa_min_ccc
        )
        swa_checkpoint = {
            "epoch": EPOCHS,
            "model_state_dict": swa_model.state_dict(),
            "best_avg_ccc": swa_avg_ccc,
            "left_ccc": swa_val_metrics["left_ccc"],
            "right_ccc": swa_val_metrics["right_ccc"],
            "min_ccc": swa_min_ccc,
            "balanced_ccc": swa_balanced_ccc,
            "auc_srf_non_functioning": swa_val_metrics["auc_srf_non_functioning"],
            "plot_data": swa_val_metrics["raw_data"],
            "run_seed": RUN_SEED,
            "experiment_tag": ABLATION_MODE,
            "rad_features": RAD_FEATURES,
            "inference_state": inference_state,
            "is_swa": True,
        }
        swa_path = OUTPUT_DIR / f"swa_{ABLATION_MODE}_seed{RUN_SEED}_avg{swa_avg_ccc:.4f}_bal{swa_balanced_ccc:.4f}.pth"
        swa_csv = OUTPUT_DIR / f"swa_{ABLATION_MODE}_seed{RUN_SEED}_avg{swa_avg_ccc:.4f}_bal{swa_balanced_ccc:.4f}_predictions.csv"
        torch.save(swa_checkpoint, swa_path)
        save_prediction_csv(swa_val_metrics["raw_data"], swa_csv)
        print(f" Saved SWA checkpoint -> {swa_path}")


if __name__ == "__main__":
    main()
