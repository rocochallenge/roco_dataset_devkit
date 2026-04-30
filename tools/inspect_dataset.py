#!/usr/bin/env python3
"""Inspect RoCo dataset files without loading large image arrays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import RoCoDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RoCo HDF5 dataset metadata.")
    parser.add_argument("dataset", help="Episode file or directory containing .h5/.hdf5 files.")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--episodes", action="store_true", help="Include per-episode metadata.")
    args = parser.parse_args()

    dataset = RoCoDataset(args.dataset, recursive=args.recursive, include_images=False, include_depth=False)
    result = {"summary": dataset.summary()}
    if args.episodes:
        episodes = []
        for index in range(len(dataset)):
            metadata = asdict(dataset.metadata(index))
            metadata["path"] = str(metadata["path"])
            episodes.append(metadata)
        result["episodes"] = episodes
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
