"""Shared utilities for RoCo dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPISODE_EXTENSIONS = {".h5", ".hdf5"}


@dataclass(frozen=True)
class EpisodeMetadata:
    """Metadata reported by the dataset loader for one HDF5 episode."""

    path: Path
    source_format: str
    length: int
    is_sim: bool | None
    task: str | None
    cameras: tuple[str, ...]
    has_depth: bool
    fields: tuple[str, ...]
    attrs: dict[str, Any]


def discover_episode_files(dataset_path: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted HDF5 episode files from a file or directory path."""

    path = Path(dataset_path).expanduser()
    if path.is_file():
        if path.suffix.lower() not in EPISODE_EXTENSIONS:
            raise ValueError(f"Expected an HDF5 episode file, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset path must be a file or directory: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in EPISODE_EXTENSIONS)


def to_python(value: Any) -> Any:
    """Convert common NumPy/HDF5 scalar values into JSON-friendly Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

