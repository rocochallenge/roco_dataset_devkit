#!/usr/bin/env python3
"""Summarize RoCo Gearbox Assembly episode score fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import RoCoDataset
from evaluation.metrics import aggregate_metrics, compute_episode_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RoCo HDF5 episode score fields.")
    parser.add_argument("dataset", help="Episode file or directory containing .h5/.hdf5 files.")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--success-score", type=float, help="Optional score threshold for success.")
    parser.add_argument("--per-episode", action="store_true", help="Print per-episode metrics.")
    args = parser.parse_args()

    dataset = RoCoDataset(args.dataset, recursive=args.recursive, include_images=False, include_depth=False)
    metrics = [compute_episode_metrics(dataset[i], success_score=args.success_score) for i in range(len(dataset))]

    result = {"summary": aggregate_metrics(metrics)}
    if args.per_episode:
        result["episodes"] = [item.to_dict() for item in metrics]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
