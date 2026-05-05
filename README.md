# RoCo Dataset Devkit

Dataset loading, inspection, conversion, generation, and preview tools for RoCo robotic collaborative gearbox assembly episodes.

RoCo is a cross-embodiment gearbox-assembly dataset: the public release includes demonstrations (in both simulation and real-world) collected across the Galaxea R1 and R1 Lite embodiments, with matching RGBD observations, joint states, and actions for learning policies that should transfer across robot morphologies.

This repository is scoped as a dataset/devkit package. It is intended for users who want to read RoCo HDF5 episodes, convert raw exports into a stable public schema, inspect dataset metadata, create quick visual previews without launching Isaac Sim, or collect additional simulator episodes with the included rule-based automation scripts.

For the original benchmark environments, challenge baselines, Isaac Lab task definitions, and competition execution instructions, use the original benchmark repository:

```text
https://github.com/rocochallenge/gearboxAssembly
```

The original RoCo Challenge page is available at [RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html).

## What This Repository Provides

- HDF5 data-loading utilities in `data_loader/` that work without Isaac Lab or Isaac Sim.
- A normalized episode interface for raw HDF5 files and devkit-standard HDF5 episodes.
- CLI examples for loading, exporting, inspecting, and previewing episode files.
- Documentation for the standard dataset schema and simulation data generation entry point.
- Cross-embodiment simulation support for both R1 and R1 Lite through robot bundles, robot-specific rule policies, and reusable task environments.
- Automatic rule-based data collection scripts that can keep generating high-quality additional episodes beyond the public dataset release.
- Python package metadata so the devkit can be installed in editable mode and used through `roco-*` commands.

## Repository Layout

```text
roco_dataset_devkit/
|-- README.md
|-- data_loader/
|   |-- __init__.py
|   |-- common.py
|   `-- hdf5_dataset.py
|-- docs/
|   |-- data_format.md
|   |-- data_generation.md
|   |-- webpage_assets.md
|   `-- images/
|-- examples/
|   `-- load_dataset.py
|-- scripts/
|   |-- download_dataset.py
|   |-- export_episode.py
|   `-- generate_dataset.py
|-- tools/
|   |-- inspect_dataset.py
|   `-- review_hdf5.py
`-- visualization/
    `-- preview_episode.py
```

`data_loader/` intentionally lives at the repository root. It is the public dataset/devkit layer and should stay importable without Isaac Lab or Isaac Sim.

## Installation

Dataset-only utilities require `numpy` and `h5py`.

Install the devkit in editable mode:

```bash
python -m pip install -e .
```

Install the download extra when fetching the official dataset from Hugging Face:

```bash
python -m pip install -e ".[download]"
```

Install optional extras when image preview, interactive HDF5 playback, or simulation-generation helpers are needed:

```bash
python -m pip install -e ".[visualization,generation]"
```

The editable install provides these command-line entry points:

```text
roco-load-dataset
roco-download-dataset
roco-export-episode
roco-inspect-dataset
roco-review-hdf5
roco-preview-episode
roco-generate-dataset
```

## Dataset Download

The official RoCo dataset is hosted only on Hugging Face:

```text
https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025
```

The Hugging Face dataset card describes the release as gearbox assembly demonstrations with RGBD observations from three views, joint states, and actions at 20 Hz. The full release is large, about 2.32 TB, so download it to external storage when needed.

The hosted files are the public release, not the limit of the dataset. RoCo is designed to be extensible: after installing the generation dependencies, users can collect additional high-quality simulation episodes automatically with the same rule-based agents used by the benchmark environments. The current devkit includes R1 Lite automation support in addition to the original R1 pipeline, so new data can be generated across embodiments and appended to local training sets without a fixed episode cap.

Dataset parts:

- `real_assembly_r1lite`: real-robot data collected with R1 Lite.
- `gearbox_assembly_real_demos`: Task 1 real-robot demonstrations collected with R1.
- `gearbox_assembly_real_demos_task3`: Task 3 real-robot demonstrations collected with R1.
- `gearbox_assembly_demos_updated`: simulation data collected with R1 in the simulator.

Download the full dataset snapshot:

```bash
roco-download-dataset --output-dir data/rocochallenge2025
```

Download only selected file patterns:

```bash
roco-download-dataset \
  --output-dir data/rocochallenge2025 \
  --include "*.h5,*.hdf5"
```

Preview the download plan without fetching files:

```bash
roco-download-dataset --output-dir data/rocochallenge2025 --dry-run
```

## Dataset Loading

The loader supports the public normalized schema and selected raw HDF5 layouts:

- **Normalized episodes**: root `/action`, `/observations/qpos`, `/observations/qvel`, and `/observations/images/<camera>`.
- **Raw HDF5 layouts**: split action fields such as `/actions/left_arm_action` and observation fields such as `/observations/head_rgb`, `/observations/left_arm_joint_pos`, and `/current_time`.

Inspect a dataset directory or one episode file:

```bash
python -m data_loader.hdf5_dataset data/rocochallenge2025 --recursive --no-images
```

Run the minimal loading example:

```bash
roco-load-dataset data/rocochallenge2025 --recursive --no-images
```

Load one episode in Python:

```python
from data_loader import RoCoDataset

dataset = RoCoDataset("data/rocochallenge2025", recursive=True, include_images=False)
episode = dataset[0]

print(episode["metadata"].source_format)
print(episode["observations"]["qpos"].shape)
print(episode["actions"].shape)
```

The returned episode dictionary has a stable structure:

```text
episode
|-- metadata
|-- observations
|   |-- images
|   |-- depth
|   |-- qpos
|   `-- qvel
|-- actions
`-- timestamp
```

Recommended local data organization after download:

```text
data/
`-- rocochallenge2025/
    |-- ...
    `-- episode files
```

Do not commit downloaded dataset files to this repository.

## Episode Export

Normalize raw HDF5 files into the public standard schema:

```bash
roco-export-episode \
  path/to/raw_episode.hdf5 \
  data/sim/episode_000000.hdf5 \
  --sim \
  --task Template-Galaxea-Lab-External-Direct-v0
```

Batch export a directory:

```bash
roco-export-episode \
  path/to/raw_episodes \
  data/sim_standard \
  --sim \
  --recursive
```

The exported standard schema uses `/action`, `/observations/qpos`, `/observations/qvel`, `/observations/images/<camera>`, optional `/observations/depth/<camera>`, and `/timestamp`.

## Automatic Data Generation

The devkit includes an Isaac Lab generation entry point for collecting new raw HDF5 episodes:

```bash
roco-generate-dataset \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num-episodes 100 \
  --output-dir data/sim/raw_r1lite \
  --enable_cameras \
  --headless
```

Generation uses the active robot bundle in `Galaxea_Lab_External.robots.robot_bundles`. The current bundle is R1 Lite, and the added `R1LiteRulePolicy` / `R1LiteRecoveryRulePolicy` paths allow rule-based scripted agents to collect and save R1 Lite episodes automatically. Switch the active bundle to R1 when collecting the same task family on the original R1 embodiment.

Large local collections can be produced by increasing `--num-episodes` and running multiple jobs as compute and storage allow. Convert generated raw files into the public schema with `roco-export-episode` before mixing them with downloaded release data.

## Inspection and Preview

Inspect metadata without loading image arrays:

```bash
roco-inspect-dataset data/sim/standard --episodes
```

Review the raw contents of one arbitrary HDF5 file:

```bash
roco-review-hdf5 path/to/episode.h5 --list-only
```

Interactively select a dataset path and preview camera/video datasets when visualization extras are installed:

```bash
roco-review-hdf5 path/to/episode.h5 --max-rows 300
```

Review one dataset path directly:

```bash
roco-review-hdf5 path/to/episode.h5 --dataset observations/head_rgb --max-rows 300
```

Save first-frame RGB previews when Pillow is installed:

```bash
roco-preview-episode data/sim/standard/episode_000000.hdf5 --output-dir previews
```

## Documentation

- [docs/data_format.md](docs/data_format.md): standard HDF5 schema and field conventions.
- [docs/data_generation.md](docs/data_generation.md): optional simulation data generation entry point.
- [docs/webpage_assets.md](docs/webpage_assets.md): concise copy material for a dataset/devkit project webpage.

## Benchmark And Challenge Code

This devkit does not document the full benchmark workflow. Use the original repository for benchmark environments, challenge baselines, Isaac Lab setup, and competition execution instructions:

```text
https://github.com/rocochallenge/gearboxAssembly
```
