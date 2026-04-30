# Webpage Assets for Dataset and Devkit Presentation

Use this page as copy material for a project website section focused on the dataset/devkit value of the repository.

## Repository

```text
https://github.com/rocochallenge/roco_dataset_devkit
```

## One-Line Description

RoCo Dataset Devkit provides HDF5 loading, standardization, inspection, and preview tools for real and simulated bimanual collaborative gearbox assembly episodes.

## Install Dataset Devkit

```bash
python -m pip install -e .
```

Install the download extra for Hugging Face access:

```bash
python -m pip install -e ".[download]"
```

Optional extras:

```bash
python -m pip install -e ".[visualization,generation]"
```

## Dataset Download

The official dataset is hosted only on Hugging Face:

```text
https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025
```

```bash
roco-download-dataset --output-dir data/rocochallenge2025
```

For partial downloads:

```bash
roco-download-dataset --output-dir data/rocochallenge2025 --include "*.h5,*.hdf5"
```

## Dataset Parts

- `real_assembly_r1lite`: real-robot data collected with R1 Lite.
- `gearbox_assembly_real_demos`: Task 1 real-robot demonstrations collected with R1.
- `gearbox_assembly_real_demos_task3`: Task 3 real-robot demonstrations collected with R1.
- `gearbox_assembly_demos_updated`: simulation data collected with R1 in the simulator.

## Optional Simulation Generation

The devkit includes a lightweight `roco-generate-dataset` entry point for users who already have the simulator stack available. Full benchmark setup, Isaac Lab task definitions, and baseline-agent instructions are maintained in the original repository:

```text
https://github.com/rocochallenge/gearboxAssembly
```

## Dataset Loading

```bash
roco-load-dataset data/rocochallenge2025 --recursive --no-images
```

## Raw HDF5 Review

```bash
roco-review-hdf5 path/to/episode.h5 --list-only
roco-review-hdf5 path/to/episode.h5 --dataset observations/head_rgb --max-rows 300
```

## Standardize Episodes

```bash
roco-export-episode path/to/raw_episode.hdf5 data/sim/episode_000000.hdf5 --sim
```

## Generate Simulation Data

```bash
roco-generate-dataset \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num-episodes 10 \
  --output-dir data/sim/raw \
  --enable_cameras \
  --headless
```

## Data Modalities

- RGBD observations from three views.
- Joint states and target joint-angle actions.
- Depth values are in millimeters.
- Timestamps are in seconds when present.
- 20 Hz sampling rate in the official release.

## Benchmark And Challenge Code

```text
https://github.com/rocochallenge/gearboxAssembly
```
