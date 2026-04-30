# RoCo Gearbox Assembly Data Format

This repository provides tools for reading raw HDF5 files and exporting them into a devkit-normalized HDF5 schema.

## Official Dataset Access

The official RoCo dataset is hosted only on Hugging Face:

```text
https://huggingface.co/datasets/rocochallenge2025/rocochallenge2025
```

Download it with the devkit command:

```bash
roco-download-dataset --output-dir data/rocochallenge2025
```

The release contains RGBD observations from three views, joint states, and actions sampled at 20 Hz. The full dataset is large, so keep downloaded files outside git-tracked source directories.

Known dataset parts:

```text
real_assembly_r1lite                  real-robot data collected with R1 Lite
gearbox_assembly_real_demos           Task 1 real-robot demonstrations collected with R1
gearbox_assembly_real_demos_task3     Task 3 real-robot demonstrations collected with R1
gearbox_assembly_demos_updated        simulation data collected with R1 in the simulator
```

## Devkit-Normalized Episode Schema

The devkit can export raw HDF5 files into a normalized schema for downstream loading. This section documents the devkit convention, not extra physical semantics beyond the source field names.

```text
episode_000000.hdf5
|-- /action                         float32 [T, 14]
|-- /observations/qpos              float32 [T, 14]
|-- /observations/qvel              float32 [T, 14] optional
|-- /observations/images/head_rgb   uint8   [T, H, W, C]
|-- /observations/images/left_hand_rgb
|-- /observations/images/right_hand_rgb
|-- /observations/depth/head_depth  numeric [T, H, W] optional, millimeters
|-- /observations/depth/left_hand_depth      optional
|-- /observations/depth/right_hand_depth     optional
`-- /timestamp                      float32 [T] optional, seconds
```

Recommended HDF5 attributes:

```text
schema          roco.standard.v1
sim             true for simulation, false for real robot data
task            source task name when known
source_format   standard or raw; standard means the devkit-normalized schema
source_file     source episode path when exported from another file
```

## Joint State and Action Semantics

Raw HDF5 files store each robot side as a 6-dimensional arm vector plus a 1-dimensional gripper value. The devkit-normalized arrays concatenate the split fields in the order used by `data_loader/hdf5_dataset.py`:

```text
0:6    left arm
6      left gripper
7:13   right arm
13     right gripper
```

`/action` in the devkit-normalized schema is the concatenated target joint-angle action. When converting raw files, it is built from:

```text
/actions/left_arm_action
/actions/left_gripper_action
/actions/right_arm_action
/actions/right_gripper_action
```

`/observations/qpos` and `/observations/qvel` use the same left-arm, left-gripper, right-arm, right-gripper concatenation order when they are built from split raw fields.

## Raw HDF5 Layout Compatibility

The loader also accepts raw files with split fields:

```text
/actions/left_arm_action
/actions/left_gripper_action
/actions/right_arm_action
/actions/right_gripper_action
/observations/head_rgb
/observations/left_hand_rgb
/observations/right_hand_rgb
/observations/head_depth
/observations/left_hand_depth
/observations/right_hand_depth
/observations/left_arm_joint_pos
/observations/right_arm_joint_pos
/observations/left_gripper_joint_pos
/observations/right_gripper_joint_pos
/observations/left_arm_joint_vel
/observations/right_arm_joint_vel
/observations/left_gripper_joint_vel
/observations/right_gripper_joint_vel
/current_time
```

Convert raw files into the standard schema with:

```bash
roco-export-episode path/to/raw_episode.hdf5 data/sim/episode_000000.hdf5 --sim
```

## Real and Simulation Data

The dataset contains both real-robot and simulation parts. Use the directory names listed above as the source-domain reference for the official release. When exporting files into the devkit-normalized schema, optional fields that are not present in a source file should be omitted rather than filled with placeholders.
