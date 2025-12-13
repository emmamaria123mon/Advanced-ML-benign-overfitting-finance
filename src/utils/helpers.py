from __future__ import annotations
from pathlib import Path
import yaml

def project_root() -> Path:
    # This file is at src/utils/helpers.py -> go up two levels to repo root
    return Path(__file__).resolve().parents[2]

def load_yaml(rel_path: str) -> dict:
    path = project_root() / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
