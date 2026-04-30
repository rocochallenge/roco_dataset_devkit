#!/usr/bin/env python3
"""Download the official RoCo dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_ID = "rocochallenge2025/rocochallenge2025"
DEFAULT_OUTPUT_DIR = "data/rocochallenge2025"
DATASET_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO_ID}"


def split_patterns(values: Sequence[str] | None) -> list[str] | None:
    """Parse comma-separated or repeated pattern arguments."""
    if not values:
        return None
    patterns: list[str] = []
    for value in values:
        patterns.extend(item.strip() for item in value.split(",") if item.strip())
    return patterns or None


def download_dataset(
    output_dir: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Download a Hugging Face dataset snapshot into a local directory."""
    output_dir = Path(output_dir).expanduser()
    include_patterns = split_patterns(include)
    exclude_patterns = split_patterns(exclude)

    print(f"Dataset repository: https://huggingface.co/datasets/{repo_id}")
    print(f"Output directory: {output_dir}")
    if include_patterns:
        print(f"Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"Exclude patterns: {exclude_patterns}")

    if dry_run:
        print("Dry run only. No files downloaded.")
        return output_dir

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for dataset download. Install it with "
            "`python -m pip install -e '.[download]'`."
        ) from exc

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=include_patterns,
        ignore_patterns=exclude_patterns,
        token=token,
    )
    print(f"Downloaded dataset snapshot to: {local_path}")
    return Path(local_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the official RoCo dataset from Hugging Face.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory for the dataset snapshot. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo ID. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision, branch, or commit hash.")
    parser.add_argument(
        "--include",
        action="append",
        help="Allow-list glob pattern. Can be repeated or comma-separated, e.g. '*.h5,*.hdf5'.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Ignore glob pattern. Can be repeated or comma-separated.",
    )
    parser.add_argument("--token", help="Hugging Face token if access requires authentication.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved download plan without downloading.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_dataset(
        args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        include=args.include,
        exclude=args.exclude,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
