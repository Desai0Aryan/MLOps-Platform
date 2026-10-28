"""Centralised, typed-ish config loader (YAML + env overrides)."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.getenv("MLOPS_CONFIG", DEFAULT_CONFIG))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Environment overrides make the same code run locally, in Docker and in CI.
    if os.getenv("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    return cfg


def feature_columns(cfg: dict) -> tuple[list[str], list[str]]:
    return list(cfg["schema"]["numeric"]), list(cfg["schema"]["categorical"])
