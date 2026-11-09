#!/usr/bin/env python3
"""Extract paired renal radiomics features from NCCT images and kidney masks.

The input manifest must contain one row per patient and the columns
``patient_id``, ``center``, ``ct_path``, and ``mask_path``. This script does
not read, impute, or write clinical variables or reference measurements.
Mask labels 1 and 2 are interpreted as the prespecified left- and right-kidney
labels, respectively.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


REQUIRED_COLUMNS = ("patient_id", "center", "ct_path", "mask_path")
MASK_LABELS = {"Left": 1, "Right": 2}


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Create the fixed PyRadiomics extractor used in this study."""
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": [1.0, 1.0, 1.0],
        "interpolator": sitk.sitkBSpline,
        "resamplingFilter": sitk.sitkNearestNeighbor,
        "normalize": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
    extractor.disableAllFeatures()
    extractor.enableAllFeatures()
    return extractor


def validate_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the image manifest and validate identifiers and file paths."""
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "center": str})
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    manifest = manifest.loc[:, REQUIRED_COLUMNS].copy()
    manifest["patient_id"] = manifest["patient_id"].str.strip()
    manifest["center"] = manifest["center"].str.strip()
    if manifest["patient_id"].isna().any() or (manifest["patient_id"] == "").any():
        raise ValueError("Manifest contains an empty patient_id.")
    if manifest["patient_id"].duplicated().any():
        duplicated = manifest.loc[manifest["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Manifest contains duplicate patient identifiers: {duplicated[:5]}")

    missing_files: list[dict[str, str]] = []
    for row in manifest.itertuples(index=False):
        for field in ("ct_path", "mask_path"):
            if not Path(getattr(row, field)).is_file():
                missing_files.append(
                    {
                        "patient_id": row.patient_id,
                        "field": field,
                        "path": getattr(row, field),
                    }
                )
    if missing_files:
        preview = missing_files[:5]
        raise FileNotFoundError(f"Manifest references missing image files: {preview}")
    return manifest


def load_image_pair(ct_path: str, mask_path: str) -> tuple[sitk.Image, sitk.Image]:
    """Load an NCCT-mask pair while preserving mask labels and geometry."""
    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    if ct_image.GetSize() != mask_image.GetSize():
        raise ValueError("CT and mask dimensions do not match.")

    ct_array = np.clip(sitk.GetArrayFromImage(ct_image), -100, 300)
    mask_array = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    observed_labels = set(np.unique(mask_array).tolist())
    missing_labels = sorted(set(MASK_LABELS.values()) - observed_labels)
    if missing_labels:
        raise ValueError(f"Kidney mask is missing required label(s): {missing_labels}")

    clipped_ct = sitk.GetImageFromArray(ct_array)
    clipped_ct.CopyInformation(ct_image)
    integer_mask = sitk.GetImageFromArray(mask_array)
    integer_mask.CopyInformation(mask_image)
    return clipped_ct, integer_mask


def extract_patient(record: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Extract left- and right-kidney features for one patient."""
    result: dict[str, Any] = {
        "patient_id": record["patient_id"],
        "center": record["center"],
    }
    errors: list[dict[str, str]] = []
    try:
        ct_image, mask_image = load_image_pair(record["ct_path"], record["mask_path"])
        extractor = build_extractor()
        for side, label in MASK_LABELS.items():
            try:
                features = extractor.execute(ct_image, mask_image, label=label)
                for name, value in features.items():
                    if not name.startswith("diagnostics_"):
                        result[f"{side}_{name}"] = float(value)
            except Exception as exc:
                errors.append(
                    {
                        "patient_id": record["patient_id"],
                        "center": record["center"],
                        "stage": f"{side.lower()}_kidney_extraction",
                        "error": str(exc),
                    }
                )
    except Exception as exc:
        errors.append(
            {
                "patient_id": record["patient_id"],
                "center": record["center"],
                "stage": "image_loading_or_validation",
                "error": str(exc),
            }
        )
    return result, errors


def run_extraction(manifest: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run deterministic feature extraction and return features and errors."""
    records = manifest.to_dict(orient="records")
    extracted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if workers == 1:
        iterator = (extract_patient(record) for record in records)
        for patient_result, patient_errors in tqdm(iterator, total=len(records), desc="Extracting radiomics"):
            extracted.append(patient_result)
            errors.extend(patient_errors)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_patient, record): record["patient_id"] for record in records}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting radiomics"):
                patient_result, patient_errors = future.result()
                extracted.append(patient_result)
                errors.extend(patient_errors)

    feature_table = pd.DataFrame(extracted).sort_values(["center", "patient_id"]).reset_index(drop=True)
    error_table = pd.DataFrame(errors, columns=["patient_id", "center", "stage", "error"])
    return feature_table, error_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV containing de-identified image paths.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for newly generated outputs.")
    parser.add_argument("--workers", type=int, default=1, help="Number of extraction processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = validate_manifest(args.manifest)
    features, errors = run_extraction(manifest, args.workers)
    feature_path = args.output_dir / "paired_kidney_radiomics.csv"
    error_path = args.output_dir / "radiomics_extraction_errors.csv"
    manifest_path = args.output_dir / "radiomics_extraction_manifest.json"
    features.to_csv(feature_path, index=False)
    errors.to_csv(error_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "input_manifest": str(args.manifest.resolve()),
                "n_patients": int(len(manifest)),
                "n_patients_with_any_error": int(errors["patient_id"].nunique()) if not errors.empty else 0,
                "mask_labels": MASK_LABELS,
                "ct_intensity_window_hu": [-100, 300],
                "resampled_pixel_spacing_mm": [1.0, 1.0, 1.0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Radiomics table written to: {feature_path}")
    print(f"Extraction error log written to: {error_path}")


if __name__ == "__main__":
    main()
