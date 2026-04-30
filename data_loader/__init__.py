"""Public data-loading utilities for the RoCo Gearbox Assembly dataset."""

from .common import EpisodeMetadata, discover_episode_files

__all__ = [
    "EpisodeMetadata",
    "RoCoDataset",
    "discover_episode_files",
    "load_episode",
]


def __getattr__(name):
    if name in {"RoCoDataset", "load_episode"}:
        from .hdf5_dataset import RoCoDataset, load_episode

        return {"RoCoDataset": RoCoDataset, "load_episode": load_episode}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
