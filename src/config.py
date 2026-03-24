"""
Centralised configuration loader.

Usage:
    from src.config import load_config
    cfg = load_config("training")  # Loads configs/training.yaml
"""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env on import
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML file from the configs/ directory."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
