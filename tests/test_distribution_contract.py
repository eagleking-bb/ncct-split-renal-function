"""Guards for the two files a reader is most likely to trust blindly.

MANIFEST.csv is the map every reviewer uses to find a script; a path that no
longer exists is worse than no manifest at all.  data/cohort_flow.csv carries
the cohort counts quoted in the manuscript, so a silent change there would
propagate into Figure 1 and into the text.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_manifest_path_exists():
    rows = list(csv.DictReader((ROOT / "MANIFEST.csv").open(encoding="utf-8")))
    assert rows, "MANIFEST.csv is empty"
    missing = [r["file"] for r in rows if not (ROOT / r["file"]).is_file()]
    assert not missing, f"MANIFEST.csv references missing files: {missing}"
    assert [int(r["order"]) for r in rows] == list(range(1, len(rows) + 1))


def test_manifest_covers_every_shipped_file():
    listed = {
        r["file"]
        for r in csv.DictReader((ROOT / "MANIFEST.csv").open(encoding="utf-8"))
    }
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "MANIFEST.csv"
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and not p.name.endswith(".pyc")
    }
    assert on_disk - listed == set(), f"files not described in MANIFEST.csv: {sorted(on_disk - listed)}"


def test_locked_cohort_counts():
    rows = {
        r["stage"]: r
        for r in csv.DictReader((ROOT / "data/cohort_flow.csv").open(encoding="utf-8"))
    }
    assert rows["training"]["n_patients"] == "698"
    assert rows["internal_development"]["n_patients"] == "175"
    assert rows["centre_3_locked_external_test"]["n_patients"] == "169"
    note = rows["centre_3_locked_external_test"]["note"]
    assert "338 kidneys" in note and "28" in note
    assert rows["clinical_source_records"]["n_patients"] == "1067"
