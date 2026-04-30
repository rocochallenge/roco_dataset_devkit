#!/usr/bin/env python3
"""Generate raw RoCo simulation episodes with Isaac Lab environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser(include_isaac_args: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate raw RoCo simulation episodes with Isaac Lab.")
    parser.add_argument("--config", help="Optional YAML config. Command-line values override config values.")
    parser.add_argument("--task", default="Template-Galaxea-Lab-External-Direct-v0", help="Isaac Lab task ID.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to collect.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments.")
    parser.add_argument("--output-dir", default="data/sim/raw", help="Directory where raw HDF5 files are written.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--record-freq", type=int, default=None, help="Override env cfg record_freq when available.")
    parser.add_argument("--no-action", action="store_true", help="Disable rule-based action application if supported.")
    parser.add_argument("--disable-fabric", action="store_true", help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--summary-file", help="Optional JSON file with generated episode paths.")

    if include_isaac_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headlessly.")
        parser.add_argument("--enable_cameras", action="store_true", help="Enable camera sensors.")
        parser.add_argument("--device", default=None, help="Simulation device.")
    return parser


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("YAML configs require PyYAML: python -m pip install pyyaml") from exc
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return data


def apply_config_defaults(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    cli_args = set()
    for index, token in enumerate(sys.argv[1:]):
        if token.startswith("--"):
            cli_args.add(token.lstrip("-").replace("-", "_"))
            if index + 1 < len(sys.argv[1:]) and not sys.argv[1:][index + 1].startswith("--"):
                continue

    for key, value in config.items():
        normalized = key.replace("-", "_")
        if hasattr(args, normalized) and normalized not in cli_args:
            setattr(args, normalized, value)
    return args


def _set_episode_output(env, output_dir: Path, episode_index: int) -> Path:
    path = output_dir / f"episode_{episode_index:06d}.hdf5"
    if hasattr(env.unwrapped, "save_hdf5_file_name"):
        env.unwrapped.save_hdf5_file_name = str(path)
    return path


def main() -> None:
    include_isaac_args = not any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    parser = build_parser(include_isaac_args=include_isaac_args)
    args = parser.parse_args()
    args = apply_config_defaults(args, load_config(args.config))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import Galaxea_Lab_External.tasks  # noqa: F401

    try:
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.record_freq is not None and hasattr(env_cfg, "record_freq"):
            env_cfg.record_freq = args.record_freq

        env = gym.make(args.task, cfg=env_cfg, use_action=not args.no_action)
        if hasattr(env.unwrapped, "cfg"):
            env.unwrapped.cfg.record_data = True

        generated = []
        current_output = _set_episode_output(env, output_dir, len(generated))

        with torch.inference_mode():
            env.reset()
            current_output = _set_episode_output(env, output_dir, len(generated))
            while simulation_app.is_running() and len(generated) < args.num_episodes:
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, truncated, _ = env.step(actions)
                done = bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())
                if done:
                    if current_output.exists():
                        generated.append(str(current_output))
                    env.reset()
                    current_output = _set_episode_output(env, output_dir, len(generated))

        env.close()

        result = {"output_dir": str(output_dir), "generated": generated[: args.num_episodes]}
        if args.summary_file:
            Path(args.summary_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
