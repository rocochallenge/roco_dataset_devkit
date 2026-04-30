#!/usr/bin/env python3
"""Export RoCo raw or standard HDF5 episodes into the public standard schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - only used when h5py is absent.
    h5py = None

from data_loader import discover_episode_files, load_episode


SCHEMA_NAME = "roco.standard.v1"


def export_episode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    sim: bool | None = None,
    task: str | None = None,
    include_images: bool = True,
    include_depth: bool = True,
    compression: str | None = "gzip",
    compression_level: int | None = 4,
) -> Path:
    """Export one HDF5 episode into the standard RoCo schema."""

    _require_h5py()
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode = load_episode(input_path, include_images=include_images, include_depth=include_depth)
    metadata = episode["metadata"]
    _validate_episode(episode)

    with h5py.File(output_path, "w") as root:
        _write_attrs(root, metadata.attrs)
        root.attrs["schema"] = SCHEMA_NAME
        root.attrs["source_format"] = metadata.source_format
        root.attrs["source_file"] = str(metadata.path)
        sim_value = metadata.is_sim if sim is None else sim
        if sim_value is not None:
            root.attrs["sim"] = bool(sim_value)
        if task or metadata.task:
            root.attrs["task"] = task or metadata.task
        root.attrs["source_fields"] = json.dumps(list(metadata.fields))

        observations = root.create_group("observations")
        _write_optional_dataset(
            observations,
            "qpos",
            episode["observations"]["qpos"],
            compression=compression,
            compression_level=compression_level,
        )
        _write_optional_dataset(
            observations,
            "qvel",
            episode["observations"]["qvel"],
            compression=compression,
            compression_level=compression_level,
        )

        images = episode["observations"]["images"]
        if images:
            image_group = observations.create_group("images")
            for name, value in images.items():
                _write_optional_dataset(
                    image_group,
                    name,
                    value,
                    compression=compression,
                    compression_level=compression_level,
                )

        depth = episode["observations"]["depth"]
        if depth:
            depth_group = observations.create_group("depth")
            for name, value in depth.items():
                _write_optional_dataset(
                    depth_group,
                    name,
                    value,
                    compression=compression,
                    compression_level=compression_level,
                )

        _write_optional_dataset(
            root,
            "action",
            episode["actions"],
            compression=compression,
            compression_level=compression_level,
        )
        _write_optional_dataset(root, "score", episode["score"], compression=None, compression_level=None)
        _write_optional_dataset(root, "timestamp", episode["timestamp"], compression=None, compression_level=None)

    return output_path


def export_path(
    input_path: str | Path,
    output_path: str | Path,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    sim: bool | None = None,
    task: str | None = None,
    include_images: bool = True,
    include_depth: bool = True,
    compression: str | None = "gzip",
    compression_level: int | None = 4,
) -> list[Path]:
    """Export one file or all HDF5 episodes in a directory."""

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            target = output_path / f"{input_path.stem}.hdf5"
        elif output_path.suffix.lower() in {".h5", ".hdf5"}:
            target = output_path
        else:
            target = output_path / f"{input_path.stem}.hdf5"
        return [
            export_episode(
                input_path,
                target,
                overwrite=overwrite,
                sim=sim,
                task=task,
                include_images=include_images,
                include_depth=include_depth,
                compression=compression,
                compression_level=compression_level,
            )
        ]

    files = discover_episode_files(input_path, recursive=recursive)
    output_path.mkdir(parents=True, exist_ok=True)

    exported = []
    for index, episode_file in enumerate(files):
        target = output_path / f"episode_{index:06d}.hdf5"
        exported.append(
            export_episode(
                episode_file,
                target,
                overwrite=overwrite,
                sim=sim,
                task=task,
                include_images=include_images,
                include_depth=include_depth,
                compression=compression,
                compression_level=compression_level,
            )
        )
    return exported


def _write_attrs(root: h5py.File, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        try:
            root.attrs[key] = value
        except TypeError:
            root.attrs[key] = json.dumps(_json_safe(value))


def _validate_episode(episode: dict[str, Any]) -> None:
    lengths = {}
    observations = episode["observations"]
    for name, value in (
        ("action", episode["actions"]),
        ("qpos", observations["qpos"]),
        ("qvel", observations["qvel"]),
        ("score", episode["score"]),
        ("timestamp", episode["timestamp"]),
    ):
        length = _array_length(value)
        if length is not None:
            lengths[name] = length

    for group_name in ("images", "depth"):
        for name, value in observations[group_name].items():
            length = _array_length(value)
            if length is not None:
                lengths[f"{group_name}/{name}"] = length

    if not lengths:
        raise ValueError("No sequence arrays were found in the input episode.")
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Episode fields have inconsistent lengths: {lengths}")


def _array_length(value: np.ndarray | None) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return None
    return int(array.shape[0])


def _write_optional_dataset(
    group: h5py.Group | h5py.File,
    name: str,
    value: np.ndarray | None,
    *,
    compression: str | None,
    compression_level: int | None,
) -> None:
    if value is None:
        return
    array = np.asarray(value)
    kwargs = _compression_kwargs(array, compression=compression, compression_level=compression_level)
    if name in group:
        del group[name]
    group.create_dataset(name, data=array, **kwargs)


def _compression_kwargs(
    array: np.ndarray,
    *,
    compression: str | None,
    compression_level: int | None,
) -> dict[str, Any]:
    if compression is None or array.shape == ():
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": compression_level}
    return {"compression": compression}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _require_h5py() -> None:
    if h5py is None:
        raise ModuleNotFoundError(
            "scripts/export_episode.py requires h5py. Install it with `python -m pip install h5py`."
        )


def _parse_sim_flag(args: argparse.Namespace) -> bool | None:
    if args.sim:
        return True
    if args.real:
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RoCo raw or ACT/VLA-style HDF5 episodes into the standard public schema."
    )
    parser.add_argument("input", help="Input HDF5 file or directory.")
    parser.add_argument("output", help="Output HDF5 file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--task", help="Override the task name written to output attrs.")
    parser.add_argument("--sim", action="store_true", help="Mark exported episodes as simulation data.")
    parser.add_argument("--real", action="store_true", help="Mark exported episodes as real robot data.")
    parser.add_argument("--no-images", action="store_true", help="Skip RGB image export.")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth export.")
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="Dataset compression for array fields.",
    )
    parser.add_argument("--compression-level", type=int, default=4, help="Gzip compression level.")
    args = parser.parse_args()

    if args.sim and args.real:
        parser.error("--sim and --real are mutually exclusive.")

    compression = None if args.compression == "none" else args.compression
    exported = export_path(
        args.input,
        args.output,
        recursive=args.recursive,
        overwrite=args.overwrite,
        sim=_parse_sim_flag(args),
        task=args.task,
        include_images=not args.no_images,
        include_depth=not args.no_depth,
        compression=compression,
        compression_level=args.compression_level,
    )

    print(json.dumps({"exported": [str(path) for path in exported]}, indent=2))


if __name__ == "__main__":
    main()
