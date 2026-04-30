"""HDF5 loader for RoCo Gearbox Assembly real and simulation episodes.

The loader accepts both repository-native raw exports and normalized ACT/VLA-style
episodes. It returns a stable dictionary so downstream code does not need to know
which HDF5 schema produced the episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import EpisodeMetadata, discover_episode_files, to_python

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised only when h5py is absent.
    h5py = None


CAMERA_ALIASES = {
    "head_rgb": (
        "observations/images/head_rgb",
        "observations/images/rgb_head",
        "observations/head_rgb",
        "observations/rgb_head",
    ),
    "left_hand_rgb": (
        "observations/images/left_hand_rgb",
        "observations/images/rgb_left_hand",
        "observations/left_hand_rgb",
        "observations/rgb_left_hand",
    ),
    "right_hand_rgb": (
        "observations/images/right_hand_rgb",
        "observations/images/rgb_right_hand",
        "observations/right_hand_rgb",
        "observations/rgb_right_hand",
    ),
}

DEPTH_ALIASES = {
    "head_depth": ("observations/depth/head_depth", "observations/head_depth", "observations/images/head_depth"),
    "left_hand_depth": (
        "observations/depth/left_hand_depth",
        "observations/left_hand_depth",
        "observations/images/left_hand_depth",
    ),
    "right_hand_depth": (
        "observations/depth/right_hand_depth",
        "observations/right_hand_depth",
        "observations/images/right_hand_depth",
    ),
}

QPOS_PARTS = (
    ("left_arm", ("observations/left_arm_joint_pos", "observations/left_arm_joint_position")),
    ("left_gripper", ("observations/left_gripper_joint_pos", "observations/left_gripper_position")),
    ("right_arm", ("observations/right_arm_joint_pos", "observations/right_arm_joint_position")),
    ("right_gripper", ("observations/right_gripper_joint_pos", "observations/right_gripper_position")),
)

QVEL_PARTS = (
    ("left_arm", ("observations/left_arm_joint_vel", "observations/left_arm_joint_velocity")),
    ("left_gripper", ("observations/left_gripper_joint_vel", "observations/left_gripper_velocity")),
    ("right_arm", ("observations/right_arm_joint_vel", "observations/right_arm_joint_velocity")),
    ("right_gripper", ("observations/right_gripper_joint_vel", "observations/right_gripper_velocity")),
)

FOUR_PART_ACTIONS = (
    ("left_arm", ("actions/left_arm_action",)),
    ("left_gripper", ("actions/left_gripper_action",)),
    ("right_arm", ("actions/right_arm_action",)),
    ("right_gripper", ("actions/right_gripper_action",)),
)

TWO_PART_ACTIONS = (
    ("left", ("actions/left_arm_action",)),
    ("right", ("actions/right_arm_action",)),
)


def load_episode(
    path: str | Path,
    *,
    cameras: list[str] | tuple[str, ...] | None = None,
    include_images: bool = True,
    include_depth: bool = True,
    start: int | None = None,
    stop: int | None = None,
) -> dict[str, Any]:
    """Load one episode and normalize it into the public RoCo data dictionary.

    Returns:
        A dictionary with ``metadata``, ``observations``, ``actions``, ``score``,
        and ``timestamp`` keys. Missing optional fields are returned as ``None``
        or empty dictionaries.
    """

    path = Path(path).expanduser()
    row_slice = slice(start, stop)

    _require_h5py()
    with h5py.File(path, "r") as root:
        metadata = _read_metadata(root, path)
        observations: dict[str, Any] = {
            "images": _read_mapped_arrays(root, CAMERA_ALIASES, row_slice, cameras) if include_images else {},
            "depth": _read_mapped_arrays(root, DEPTH_ALIASES, row_slice) if include_depth else {},
            "qpos": _read_qpos(root, row_slice),
            "qvel": _read_qvel(root, row_slice),
        }

        episode = {
            "metadata": metadata,
            "observations": observations,
            "actions": _read_actions(root, row_slice),
            "score": _read_first_existing(root, ("score",), row_slice),
            "timestamp": _read_first_existing(root, ("timestamp", "current_time"), row_slice),
        }

    return episode


class RoCoDataset:
    """Small HDF5 dataset wrapper for RoCo real and simulation episodes."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        recursive: bool = False,
        cameras: list[str] | tuple[str, ...] | None = None,
        include_images: bool = True,
        include_depth: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser()
        self.files = discover_episode_files(self.dataset_path, recursive=recursive)
        self.cameras = tuple(_canonical_camera_name(name) for name in cameras) if cameras else None
        self.include_images = include_images
        self.include_depth = include_depth

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.load_episode(index)

    def load_episode(self, index: int) -> dict[str, Any]:
        return load_episode(
            self.files[index],
            cameras=self.cameras,
            include_images=self.include_images,
            include_depth=self.include_depth,
        )

    def metadata(self, index: int) -> EpisodeMetadata:
        _require_h5py()
        with h5py.File(self.files[index], "r") as root:
            return _read_metadata(root, self.files[index])

    def summary(self) -> dict[str, Any]:
        episodes = [self.metadata(i) for i in range(len(self))]
        return {
            "dataset_path": str(self.dataset_path),
            "num_episodes": len(episodes),
            "num_sim": sum(1 for item in episodes if item.is_sim is True),
            "num_real": sum(1 for item in episodes if item.is_sim is False),
            "formats": sorted({item.source_format for item in episodes}),
            "cameras": sorted({camera for item in episodes for camera in item.cameras}),
            "total_frames": sum(item.length for item in episodes),
        }


def _read_metadata(root: h5py.File, path: Path) -> EpisodeMetadata:
    source_format = _detect_source_format(root)
    length = _detect_length(root)
    cameras = tuple(name for name, aliases in CAMERA_ALIASES.items() if _first_existing_path(root, aliases))
    has_depth = any(_first_existing_path(root, aliases) for aliases in DEPTH_ALIASES.values())
    attrs = {key: to_python(value) for key, value in root.attrs.items()}
    task = attrs.get("task") or attrs.get("task_name")

    return EpisodeMetadata(
        path=path,
        source_format=source_format,
        length=length,
        is_sim=_read_bool_attr(attrs, "sim"),
        task=str(task) if task is not None else None,
        cameras=cameras,
        has_depth=has_depth,
        fields=tuple(_top_level_fields(root)),
        attrs=attrs,
    )


def _require_h5py() -> None:
    if h5py is None:
        raise ModuleNotFoundError(
            "The RoCo HDF5 loader requires h5py. Install it with `python -m pip install h5py` "
            "in the environment used for dataset inspection."
        )


def _detect_source_format(root: h5py.File) -> str:
    if "action" in root and "observations/qpos" in root:
        return "standard"
    if "actions" in root or any(path in root for _, aliases in QPOS_PARTS for path in aliases):
        return "raw"
    return "unknown"


def _detect_length(root: h5py.File) -> int:
    for path in (
        "action",
        "observations/qpos",
        "actions/left_arm_action",
        "observations/left_arm_joint_pos",
        "observations/left_arm_joint_position",
    ):
        if path in root:
            return int(root[path].shape[0])
    for aliases in CAMERA_ALIASES.values():
        path = _first_existing_path(root, aliases)
        if path is not None:
            return int(root[path].shape[0])
    return 0


def _read_actions(root: h5py.File, row_slice: slice) -> np.ndarray | None:
    if "action" in root:
        return _read_array(root, "action", row_slice).astype(np.float32, copy=False)
    action = _concat_parts(root, FOUR_PART_ACTIONS, row_slice)
    if action is not None:
        return action.astype(np.float32, copy=False)
    action = _concat_parts(root, TWO_PART_ACTIONS, row_slice)
    if action is not None:
        return action.astype(np.float32, copy=False)
    return None


def _read_qpos(root: h5py.File, row_slice: slice) -> np.ndarray | None:
    if "observations/qpos" in root:
        return _read_array(root, "observations/qpos", row_slice).astype(np.float32, copy=False)
    qpos = _concat_parts(root, QPOS_PARTS, row_slice)
    return qpos.astype(np.float32, copy=False) if qpos is not None else None


def _read_qvel(root: h5py.File, row_slice: slice) -> np.ndarray | None:
    if "observations/qvel" in root:
        return _read_array(root, "observations/qvel", row_slice).astype(np.float32, copy=False)
    qvel = _concat_parts(root, QVEL_PARTS, row_slice)
    return qvel.astype(np.float32, copy=False) if qvel is not None else None


def _concat_parts(root: h5py.File, parts: tuple[tuple[str, tuple[str, ...]], ...], row_slice: slice) -> np.ndarray | None:
    arrays = []
    for _, aliases in parts:
        path = _first_existing_path(root, aliases)
        if path is None:
            return None
        arrays.append(_ensure_2d(_read_array(root, path, row_slice)))
    return np.concatenate(arrays, axis=1)


def _read_mapped_arrays(
    root: h5py.File,
    mapping: dict[str, tuple[str, ...]],
    row_slice: slice,
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    requested = tuple(_canonical_camera_name(name) for name in names) if names else tuple(mapping)
    values = {}
    for name in requested:
        aliases = mapping.get(name)
        if aliases is None:
            raise KeyError(f"Unknown camera or data key: {name}")
        path = _first_existing_path(root, aliases)
        if path is not None:
            values[name] = _read_array(root, path, row_slice)
    return values


def _read_first_existing(root: h5py.File, aliases: tuple[str, ...], row_slice: slice) -> np.ndarray | None:
    path = _first_existing_path(root, aliases)
    return _read_array(root, path, row_slice) if path is not None else None


def _read_array(root: h5py.File, path: str, row_slice: slice) -> np.ndarray:
    dataset = root[path]
    if dataset.shape == ():
        return np.asarray(dataset[()])
    return np.asarray(dataset[row_slice])


def _first_existing_path(root: h5py.File, aliases: tuple[str, ...]) -> str | None:
    for path in aliases:
        if path in root:
            return path
    return None


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _canonical_camera_name(name: str) -> str:
    aliases = {
        "head": "head_rgb",
        "rgb_head": "head_rgb",
        "left": "left_hand_rgb",
        "left_hand": "left_hand_rgb",
        "rgb_left_hand": "left_hand_rgb",
        "right": "right_hand_rgb",
        "right_hand": "right_hand_rgb",
        "rgb_right_hand": "right_hand_rgb",
    }
    return aliases.get(name, name)


def _read_bool_attr(attrs: dict[str, Any], key: str) -> bool | None:
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _top_level_fields(root: h5py.File) -> list[str]:
    fields = []
    root.visititems(lambda name, obj: fields.append(name) if isinstance(obj, h5py.Dataset) else None)
    return fields


def _shape_summary(value: Any) -> Any:
    if isinstance(value, EpisodeMetadata):
        return {
            "path": str(value.path),
            "source_format": value.source_format,
            "length": value.length,
            "is_sim": value.is_sim,
            "task": value.task,
            "cameras": list(value.cameras),
            "has_depth": value.has_depth,
        }
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    if value is None:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or load RoCo Gearbox Assembly HDF5 episodes.")
    parser.add_argument("dataset", help="Episode file or directory containing .h5/.hdf5 episodes.")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--index", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument("--no-images", action="store_true", help="Skip image arrays when loading the episode.")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth arrays when loading the episode.")
    args = parser.parse_args()

    dataset = RoCoDataset(
        args.dataset,
        recursive=args.recursive,
        include_images=not args.no_images,
        include_depth=not args.no_depth,
    )
    print(json.dumps(dataset.summary(), indent=2))
    if len(dataset) == 0:
        return

    episode = dataset.load_episode(args.index)
    print(json.dumps(_shape_summary(episode), indent=2))


if __name__ == "__main__":
    main()
