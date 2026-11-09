"""Source-identifier detection for public-facing files.

The previous check hard-coded ``nfyy|gzyy|gdph``.  The project actually stores
development-centre identifiers with the ``GZH`` prefix, so those files were not
detected, and any new centre prefix would have been missed as well.

This module builds the pattern from

* an explicit list of known centre prefixes,
* the centre-specific input file names in the analysis configuration
  (``NFH_disease.csv`` -> ``NFH`` ...), and
* the ``*_centerN_csv`` / ``*_centerN_*`` keys of any configuration mapping,

then matches a prefix followed by a separator and at least two digits, e.g.
``GZH_000123``, ``gdhp-20240101`` (case-insensitive).  Anonymised tokens are
``DFME_`` + 16 hex characters, which never match because ``DFME`` is not a
centre prefix and the numeric part must be digits.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

KNOWN_CENTER_PREFIXES = ("NFH", "GZH", "GDPH")

#: Columns that must not survive de-identification with original values.
RAW_ID_COLUMNS = ("ID", "patient_id", "PatientID", "subject_id", "case_id")

#: Suffixes considered publishable text for the privacy scan.
SCANNED_SUFFIXES = {".md", ".csv", ".json", ".txt", ".yaml", ".yml"}

#: Directories that legitimately keep original identifiers.
AUDIT_PARTS = {"audit"}


def derive_center_prefixes(config: Mapping | None = None) -> list[str]:
    """Known prefixes plus any prefix implied by the configuration file names."""
    prefixes = {prefix.upper() for prefix in KNOWN_CENTER_PREFIXES}
    if config:
        inputs = config.get("inputs", {}) if isinstance(config, Mapping) else {}
        for key, value in inputs.items():
            if not isinstance(value, str):
                continue
            name = Path(value).name
            match = re.match(r"([A-Za-z]{2,6})[_-]", name)
            if match and ("center" in key or "disease" in key or "clinical" in key):
                prefixes.add(match.group(1).upper())
    return sorted(prefixes)


def build_source_id_pattern(prefixes: Iterable[str]) -> re.Pattern:
    ordered = sorted({p.upper() for p in prefixes if p}, key=len, reverse=True)
    if not ordered:
        raise ValueError("at least one centre prefix is required")
    alternation = "|".join(re.escape(prefix) for prefix in ordered)
    return re.compile(rf"\b(?:{alternation})[ _-]?\d{{2,}}", re.IGNORECASE)


def find_source_ids(text: str, pattern: re.Pattern) -> list[str]:
    return [match.group(0) for match in pattern.finditer(text)]


def anonymised_id_pattern() -> re.Pattern:
    return re.compile(r"^DFME_[0-9a-f]{16}$")


def find_unanonymised_id_columns(frame, name: str = "table") -> list[str]:
    """Report raw-ID columns whose values are not ``DFME_<16 hex>`` tokens."""
    import pandas as pd

    problems = []
    anonymised = anonymised_id_pattern()
    for column in RAW_ID_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        bad = values[~values.map(lambda value: bool(anonymised.match(value.strip())))]
        # A unique integer index column is not a patient identifier; only flag
        # values that look like identifiers (contain a letter or are long).
        suspicious = bad[bad.map(lambda value: bool(re.search(r"[A-Za-z]", value)) or len(value) >= 12)]
        if len(suspicious):
            problems.append(
                f"{name}: column {column!r} has {len(suspicious)} non-anonymised identifier(s), "
                f"e.g. {suspicious.iloc[0]!r}"
            )
    return problems


def scan_tree(
    root: Path,
    pattern: re.Pattern,
    skip_parts: Iterable[str] = AUDIT_PARTS,
    suffixes: Iterable[str] = SCANNED_SUFFIXES,
) -> list[str]:
    """Return one message per file that contains a source identifier."""
    skip_parts = set(skip_parts)
    suffixes = {suffix.lower() for suffix in suffixes}
    findings = []
    root = Path(root)
    if not root.exists():
        return [f"{root}: does not exist"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if skip_parts & set(path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        matches = find_source_ids(content, pattern)
        if matches:
            unique = sorted(set(matches))[:5]
            findings.append(
                f"potential source patient ID in public-facing file: "
                f"{path.relative_to(root) if root in path.parents else path} -> {unique}"
            )
    return findings
