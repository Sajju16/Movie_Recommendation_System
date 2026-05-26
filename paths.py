from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # src/movie_recom/paths.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return project_root() / "dataset"


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
