from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ["raw_dir", "processed_dir", "eval_dir"]:
        Path(cfg["data"][key]).mkdir(parents=True, exist_ok=True)
