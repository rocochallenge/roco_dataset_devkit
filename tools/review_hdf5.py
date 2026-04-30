#!/usr/bin/env python3
"""Interactively review arbitrary HDF5 datasets and camera streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_KEYWORDS = ("camera", "cam", "rgb", "image")


def collect_dataset_paths(h5_group: Any, prefix: str = "") -> list[str]:
    """Recursively collect dataset paths under a group."""
    h5py = require_h5py()
    paths: list[str] = []
    for name, item in h5_group.items():
        current_path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            paths.append(current_path)
        elif isinstance(item, h5py.Group):
            paths.extend(collect_dataset_paths(item, current_path))
    return paths


def read_dataset(dataset: Any, *, max_rows: int | None = None) -> Any:
    """Read scalar or array dataset data, optionally limiting the first axis."""
    if dataset.shape == ():
        return dataset[()]
    if max_rows is not None and dataset.shape and dataset.shape[0] > max_rows:
        return dataset[:max_rows]
    return dataset[:]


def is_camera_dataset(dataset_path: str) -> bool:
    """Return True when dataset path suggests camera/image frames."""
    lower_path = dataset_path.lower()
    return any(keyword in lower_path for keyword in CAMERA_KEYWORDS)


def to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """Convert frame to uint8 for OpenCV display."""
    frame = np.asarray(frame).squeeze()
    if frame.dtype == np.uint8:
        return frame

    frame = np.nan_to_num(frame)
    if frame.size == 0:
        return frame.astype(np.uint8)

    max_val = frame.max()
    if max_val <= 1.0:
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def print_dataset_table(file_path: Path, dataset_paths: list[str]) -> None:
    """Print HDF5 dataset paths with shape and dtype."""
    h5py = require_h5py()
    with h5py.File(file_path, "r") as root:
        print("\nAvailable datasets in HDF5 file:")
        if not dataset_paths:
            print("No datasets found in file.")
            return

        for index, dataset_path in enumerate(dataset_paths, 1):
            dataset = root[dataset_path]
            print(f"{index}. {dataset_path} - Shape: {dataset.shape}, Dtype: {dataset.dtype}")


def review_dataset(file_path: Path, dataset_path: str, *, max_rows: int | None, fps: float) -> None:
    """Read and display one dataset from an HDF5 file."""
    h5py = require_h5py()
    dataset_path = dataset_path.strip("/")
    with h5py.File(file_path, "r") as root:
        if dataset_path not in root:
            raise KeyError(f"Dataset '{dataset_path}' not found in {file_path}")

        dataset = root[dataset_path]
        data = read_dataset(dataset, max_rows=max_rows)

        print(f"\nDataset: {dataset_path}")
        print(f"Shape: {dataset.shape}, Dtype: {dataset.dtype}")
        if max_rows is not None and dataset.shape and dataset.shape[0] > max_rows:
            print(f"Loaded first {max_rows} row(s) for review.")

    if is_camera_dataset(dataset_path):
        play_camera_frames(data, window_name=f"Camera: {dataset_path}", fps=fps)
    elif looks_like_image_or_video(data):
        display_image_or_video(data, fps=fps)
    else:
        print_array(data)


def looks_like_image_or_video(data: Any) -> bool:
    """Infer image/video arrays from dimensionality."""
    if not isinstance(data, np.ndarray):
        return False
    return (data.ndim == 3 and data.shape[-1] in (1, 3, 4)) or (data.ndim == 4 and data.shape[-1] in (1, 3, 4))


def play_camera_frames(data: Any, *, window_name: str, fps: float) -> None:
    """Play camera frames in an OpenCV window."""
    if not isinstance(data, np.ndarray):
        print("Camera dataset is not an array. Cannot play frames.")
        return

    frames = normalize_video_frames(data)
    if frames is None:
        print(f"Unsupported camera data shape: {data.shape}")
        return

    cv2 = require_cv2()
    frame_interval_ms = max(1, int(round(1000 / fps)))
    print(f"Playing {len(frames)} frame(s) at {fps:g} Hz. Press 'q' to stop.")

    for frame in frames:
        frame = to_uint8_frame(frame)
        if frame.ndim == 2:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(frame_interval_ms) & 0xFF == ord("q"):
            break

    cv2.destroyWindow(window_name)


def display_image_or_video(data: np.ndarray, *, fps: float) -> None:
    """Display an image or video array."""
    frames = normalize_video_frames(data)
    if frames is None:
        print(f"Unsupported image/video shape: {data.shape}")
        return
    if len(frames) == 1:
        display_image(frames[0])
    else:
        play_camera_frames(frames, window_name="Video Playback", fps=fps)


def normalize_video_frames(data: np.ndarray) -> np.ndarray | None:
    """Normalize image-like arrays into an N,H,W,C or N,H,W sequence."""
    data = np.asarray(data)
    if data.ndim == 4:
        return data
    if data.ndim == 3:
        if data.shape[-1] in (1, 3, 4):
            return np.expand_dims(data, axis=0)
        return data
    if data.ndim == 2:
        return np.expand_dims(data, axis=0)
    return None


def display_image(image: np.ndarray) -> None:
    """Display a single image using matplotlib."""
    plt = require_matplotlib()
    image = to_uint8_frame(image)
    plt.figure(figsize=(8, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image[:, :, :3])
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_array(data: Any) -> None:
    """Print numeric or scalar data with compact numpy formatting."""
    if isinstance(data, np.ndarray):
        with np.printoptions(edgeitems=6, threshold=80, linewidth=120):
            print(f"Data:\n{data}")
    else:
        print(f"Data:\n{data}")


def require_cv2() -> Any:
    """Import OpenCV only when GUI playback is requested."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for video playback. Install visualization extras with "
            "`python -m pip install -e '.[visualization]'`."
        ) from exc
    return cv2


def require_h5py() -> Any:
    """Import h5py only when an HDF5 file is accessed."""
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required for HDF5 review. Install the devkit dependencies with "
            "`python -m pip install -e .`."
        ) from exc
    return h5py


def require_matplotlib() -> Any:
    """Import matplotlib only when still-image display is requested."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for image display. Install visualization extras with "
            "`python -m pip install -e '.[visualization]'`."
        ) from exc
    return plt


def interactive_review(file_path: Path, dataset_paths: list[str], *, max_rows: int | None, fps: float) -> None:
    """Prompt for dataset paths until the user exits."""
    while True:
        key_input = input("\nEnter dataset path to review (or 'quit' to exit): ").strip()
        if key_input.lower() in {"quit", "q", "exit"}:
            break

        dataset_path = key_input.strip("/")
        if dataset_path not in dataset_paths:
            print(f"Dataset '{key_input}' not found. Try again.")
            continue

        review_dataset(file_path, dataset_path, max_rows=max_rows, fps=fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review arbitrary HDF5 datasets and camera streams.")
    parser.add_argument("file", help="HDF5 file to inspect.")
    parser.add_argument("--dataset", help="Dataset path to review directly.")
    parser.add_argument("--list-only", action="store_true", help="Only list dataset paths, shapes, and dtypes.")
    parser.add_argument("--max-rows", type=int, help="Limit arrays to the first N rows when reading.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS for image sequences. Default: 30.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    file_path = Path(args.file).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive.")

    h5py = require_h5py()
    with h5py.File(file_path, "r") as root:
        dataset_paths = collect_dataset_paths(root)

    print_dataset_table(file_path, dataset_paths)
    if args.list_only:
        return
    if args.dataset:
        review_dataset(file_path, args.dataset, max_rows=args.max_rows, fps=args.fps)
        return
    if not sys.stdin.isatty():
        return
    interactive_review(file_path, dataset_paths, max_rows=args.max_rows, fps=args.fps)


if __name__ == "__main__":
    main()
