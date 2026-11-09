"""Pure helpers for verifying frozen nnU-Net mask outputs.

Kept in a separate module so they can be unit-tested without nnU-Net, torch or
any imaging dependency.
"""

from __future__ import annotations

from pathlib import Path

#: on-disk pseudonym prefix -> analysis centre label
PREFIX_TO_CENTER = {"GZH": "Center_2", "GDPH": "Center_3"}
CENTER_TO_PREFIX = {center: prefix for prefix, center in PREFIX_TO_CENTER.items()}


def expected_counts_by_prefix(counts_by_center: dict) -> dict[str, int]:
    """Translate per-centre expected case counts into the on-disk prefix keys.

    The inference loop iterates over ``GZH``/``GDPH`` directories, while the
    locked manifest is keyed by ``Center_2``/``Center_3``.  Keying a lookup by
    the prefix against a centre-keyed dict returns ``None``; this function is the
    single translation point.
    """
    unknown = sorted(set(counts_by_center) - set(CENTER_TO_PREFIX))
    if unknown:
        raise ValueError(f"unexpected centre key(s) in the locked manifest: {unknown}")
    return {CENTER_TO_PREFIX[centre]: int(count) for centre, count in counts_by_center.items()}


def check_prediction_outputs(output_dir: Path, expected_ids, expected_count: int) -> dict:
    """Verify that a frozen-inference directory holds exactly the expected masks.

    ``overwrite=False`` means a stale mask from an earlier run survives in the
    directory, so a file count alone is not sufficient: filenames are compared
    with the locked identifiers as well.
    """
    output_dir = Path(output_dir)
    suffix = ".nii.gz"
    found = sorted(path.name for path in output_dir.glob(f"*{suffix}"))
    stems = {name[: -len(suffix)] for name in found}
    expected = set(expected_ids) if expected_ids is not None else stems
    missing = sorted(expected - stems)
    unexpected = sorted(stems - expected)
    report = {
        "directory": str(output_dir),
        "n_files": len(found),
        "n_expected": int(expected_count),
        "missing": missing,
        "unexpected": unexpected,
    }
    if missing:
        raise RuntimeError(f"{output_dir}: {len(missing)} locked cases have no prediction: {missing[:5]}")
    if unexpected:
        raise RuntimeError(
            f"{output_dir}: {len(unexpected)} prediction(s) do not belong to the locked cohort "
            f"(stale file from an earlier run?): {unexpected[:5]}"
        )
    if len(found) != expected_count:
        raise RuntimeError(f"{output_dir}: expected {expected_count} predictions, found {len(found)}")
    return report
