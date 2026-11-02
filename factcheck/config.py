"""Locate the external data directory and point library caches at it.

The path comes from the FACTCHECK_DATA env var, else config.local.toml
(gitignored), else config.example.toml defaults for non-path settings.
"""
import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load() -> dict:
    cfg = {}
    for name in ("config.example.toml", "config.local.toml"):
        p = REPO / name
        if p.exists():
            with open(p, "rb") as f:
                for section, values in tomllib.load(f).items():
                    cfg.setdefault(section, {}).update(values)
    if os.environ.get("FACTCHECK_DATA"):
        cfg.setdefault("paths", {})["data_dir"] = os.environ["FACTCHECK_DATA"]
    return cfg


def data_dir() -> Path:
    d = Path(load()["paths"]["data_dir"])
    if not d.exists():
        raise FileNotFoundError(
            f"data dir {d} not found; set FACTCHECK_DATA or config.local.toml")
    return d


def setup_caches() -> Path:
    """Send HF and pip caches to the data dir so nothing large lands in the repo."""
    d = data_dir()
    os.environ.setdefault("HF_HOME", str(d / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(d / "hf" / "datasets"))
    os.environ.setdefault("PIP_CACHE_DIR", str(d / "pip"))
    return d
