"""Core metrics for RoCo Gearbox Assembly benchmark episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics computed from one loaded RoCo episode."""

    path: str
    task: str | None
    is_sim: bool | None
    length: int
    final_score: float | None
    max_score: float | None
    success: bool | None
    duration_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_episode_metrics(episode: dict[str, Any], success_score: float | None = None) -> EpisodeMetrics:
    """Compute score summaries from a normalized episode dictionary."""

    metadata = episode["metadata"]
    score = _as_array_or_none(episode.get("score"))
    timestamp = _as_array_or_none(episode.get("timestamp"))
    action = _as_array_or_none(episode.get("actions"))
    qpos = _as_array_or_none(episode.get("observations", {}).get("qpos"))

    final_score = float(score[-1]) if score is not None and score.size else None
    max_score = float(np.max(score)) if score is not None and score.size else final_score
    success = final_score >= success_score if final_score is not None and success_score is not None else None

    if timestamp is not None and timestamp.size:
        duration_s = float(timestamp[-1] - timestamp[0]) if timestamp.size > 1 else float(timestamp[-1])
    else:
        duration_s = None

    length = int(metadata.length)
    if length == 0:
        for candidate in (action, qpos, score, timestamp):
            if candidate is not None and candidate.shape:
                length = int(candidate.shape[0])
                break

    return EpisodeMetrics(
        path=str(metadata.path),
        task=metadata.task,
        is_sim=metadata.is_sim,
        length=length,
        final_score=final_score,
        max_score=max_score,
        success=success,
        duration_s=duration_s,
    )


def aggregate_metrics(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate per-episode metrics into a dataset-level summary."""

    final_scores = [item.final_score for item in metrics if item.final_score is not None]
    max_scores = [item.max_score for item in metrics if item.max_score is not None]
    durations = [item.duration_s for item in metrics if item.duration_s is not None]
    successes = [item.success for item in metrics if item.success is not None]

    return {
        "num_episodes": len(metrics),
        "num_with_score": len(final_scores),
        "mean_final_score": _mean_or_none(final_scores),
        "mean_max_score": _mean_or_none(max_scores),
        "success_rate": _mean_or_none([float(item) for item in successes]),
        "mean_duration_s": _mean_or_none(durations),
        "total_frames": int(sum(item.length for item in metrics)),
    }


def _as_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    return array.reshape(-1) if array.ndim == 0 else array


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
