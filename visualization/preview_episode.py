#!/usr/bin/env python3
"""Save first-frame RGB previews from a RoCo HDF5 episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import load_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Export first-frame RGB previews from one RoCo episode.")
    parser.add_argument("episode", help="Input .h5/.hdf5 episode.")
    parser.add_argument("--output-dir", default="previews", help="Directory for PNG previews.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to export.")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("preview_episode.py requires Pillow: python -m pip install pillow") from exc

    episode = load_episode(args.episode, include_images=True, include_depth=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for name, frames in episode["observations"]["images"].items():
        if args.frame < 0 or args.frame >= frames.shape[0]:
            raise IndexError(f"Frame {args.frame} is out of range for camera {name} with {frames.shape[0]} frames.")
        image = np.asarray(frames[args.frame])
        if image.shape[-1] == 4:
            image = image[..., :3]
        path = output_dir / f"{Path(args.episode).stem}_{name}_frame_{args.frame:06d}.png"
        Image.fromarray(image.astype(np.uint8)).save(path)
        exported.append(str(path))

    print(json.dumps({"exported": exported}, indent=2))


if __name__ == "__main__":
    main()
