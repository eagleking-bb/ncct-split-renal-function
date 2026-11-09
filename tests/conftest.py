"""Make the repository layout importable from the tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("03_training", "05_evaluation", "06_statistics", "01_segmentation", "02_radiomics"):
    path = ROOT / folder
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(relative_path: str, name: str):
    """Import a module whose filename is not a valid identifier (e.g. ``02_foo``)."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
