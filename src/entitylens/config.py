"""Tiny YAML config loader with base-merge."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


def load_config(model_config: str | Path, base: str | Path | None = None) -> Dict:
    cfg: Dict = {}
    if base is not None and Path(base).exists():
        cfg.update(yaml.safe_load(Path(base).read_text()) or {})
    cfg.update(yaml.safe_load(Path(model_config).read_text()) or {})
    return cfg
