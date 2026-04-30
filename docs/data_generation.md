# Automatic Simulation Data Generation

The Isaac Lab environments can generate new simulation rollouts for dataset extension, benchmark debugging, and policy development. The public entry point is:

```bash
roco-generate-dataset
```

## Minimal Example

```bash
roco-generate-dataset \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num-episodes 10 \
  --output-dir data/sim/raw \
  --enable_cameras \
  --headless
```

Recovery and partial-state examples:

```bash
roco-generate-dataset \
  --task Gearbox-Partial-Lackfourth \
  --num-episodes 10 \
  --output-dir data/sim/raw_partial \
  --enable_cameras \
  --headless

roco-generate-dataset \
  --task Gearbox-Recovery-Misplacedfourth \
  --num-episodes 10 \
  --output-dir data/sim/raw_recovery \
  --enable_cameras \
  --headless
```

## Config File

The default template is:

```text
configs/data_gen/default.yaml
```

Use it with:

```bash
roco-generate-dataset --config configs/data_gen/default.yaml
```

Command-line arguments override config values.

## Output

The generation script produces raw simulator HDF5 episodes. Convert them to the public standard schema with:

```bash
roco-export-episode \
  data/sim/raw \
  data/sim/standard \
  --sim \
  --recursive
```

The standardized files can then be loaded with:

```bash
roco-load-dataset data/sim/standard --no-images
```

## Notes

- Dataset-only tools do not require Isaac Sim.
- Data generation requires Isaac Sim, Isaac Lab, and the `Galaxea_Lab_External` extension installed in editable mode.
- The current rule-based environments record raw HDF5 during rollout. `scripts/generate_dataset.py` redirects the environment's episode output path into `--output-dir`.
- For large collections, generate raw files first, validate them, and then export into the standard schema.
