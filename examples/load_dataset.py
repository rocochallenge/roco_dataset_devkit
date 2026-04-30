#!/usr/bin/env python3
"""Minimal example for loading RoCo Gearbox Assembly HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import RoCoDataset


def summarize_value(value: Any) -> Any:
    if is_dataclass(value):
        result = asdict(value)
        result["path"] = str(result["path"])
        return result
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: summarize_value(item) for key, item in value.items()}
    if value is None:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect one RoCo dataset episode.")
    parser.add_argument("dataset", help="Episode file or directory containing .h5/.hdf5 files.")
    parser.add_argument("--index", type=int, default=0, help="Episode index to load from a directory.")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--no-images", action="store_true", help="Skip RGB image arrays.")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth arrays.")
    args = parser.parse_args()

    dataset = RoCoDataset(
        args.dataset,
        recursive=args.recursive,
        include_images=not args.no_images,
        include_depth=not args.no_depth,
    )
    print(json.dumps(dataset.summary(), indent=2))

    if len(dataset) == 0:
        print("No HDF5 episodes found.")
        return
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Episode index {args.index} is out of range for {len(dataset)} episodes.")

    episode = dataset[args.index]
    print(json.dumps(summarize_value(episode), indent=2))


if __name__ == "__main__":
    main()
