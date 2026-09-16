# CubiCasa5k-Next

Clean-room, paper-based reimplementation of the CubiCasa5k multi-task
floorplan parsing model with modern Python (3.13) and PyTorch (2.x).

> Derived solely from the papers
> *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan
> Image Analysis* and *Raster-to-Vector: Revisiting Floorplan Transformation*.
> No third-party implementation code was copied.

## Model

- Pre-activation hourglass with a single 44-channel head.
- 21 junction heatmaps, 12 room logits, 11 icon logits.
- Homoscedastic-uncertainty multi-task loss with per-channel scales.

## Layout

- `src/cubicasa5k_next/model/` — hourglass network
- `src/cubicasa5k_next/losses/` — uncertainty-weighted loss
- `src/cubicasa5k_next/data/` — SVG parsing, rasterization, heatmaps, dataset
- `src/cubicasa5k_next/postprocess/` — junction NMS, walls, rooms, icons, openings
- `src/cubicasa5k_next/training/` — trainer loop, segmentation metrics
- `src/cubicasa5k_next/inference/` — predictor with rotation TTA
- `src/cubicasa5k_next/cli/` — `train`, `infer`, `smoke`, `export onnx` commands
- `examples/` — standalone samples (e.g. ONNX inference)

## Setup

```bash
uv sync --extra cu132  # CUDA 13.2 builds (Linux/Windows)
uv sync --extra cpu    # CPU-only builds
```

## Usage

```bash
uv run --extra cu132 cubicasa5k-next smoke --device cuda
uv run --extra cu132 cubicasa5k-next train ./data/train ./data/val --epochs 400 --device cuda
uv run --extra cu132 cubicasa5k-next infer ./checkpoints/best.pt ./sample.png --output-dir ./outputs
```

## TensorBoard

The train CLI enables logging by default. Each invocation creates an isolated
`<checkpoint_dir>/tensorboard/<run_name>_<timestamp>_<unique_id>/` directory.
Python APIs use `tensorboard_dir=None` to disable logging; pass an explicit
directory to enable it. An empty string is not a disable flag.

Both projects expose the same options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--tensorboard-dir` | `<checkpoint_dir>/tensorboard` | Log root |
| `--tensorboard-run-name` | `cubicasa5k-next` | Run prefix; letters, digits, dots, underscores, hyphens |
| `--no-tensorboard` | false | Disable all event logging |
| `--tensorboard-image-every` | 5 | Images on epoch 1 and every N epochs; 0 disables |
| `--tensorboard-max-images` | 4 | Maximum images from the first validation batch; 0 disables |
| `--tensorboard-flush-secs` | 30 | Background flush interval; also flush after each epoch |

```bash
uv run --extra cu132 cubicasa5k-next train ./data/train ./data/val --tensorboard-run-name experiment-1
uv run --extra cu132 tensorboard --logdir checkpoints/tensorboard
uv run --extra cu132 cubicasa5k-next train ./data/train ./data/val --tensorboard-dir ./logs/tb --tensorboard-image-every 10
uv run --extra cu132 cubicasa5k-next train ./data/train ./data/val --no-tensorboard
```

Use `--extra cpu` instead for CPU environments.
Scalars use epoch steps and common `Loss/train`, `Loss/valid`, `Optimizer/lr`,
`Metrics/valid_*` tags. Validation images use `Samples/valid/*`.
Samples are collected during validation without another data-loader pass.
New invocations, including resumed or fine-tuned training, always create a new
run; previous event files are never purged. Resumed training keeps its checkpoint epoch numbering.

CubiCasa additionally records loss components, uncertainty weights, room/icon mIoU and accuracy, room label images and junction heatmaps. Validation metrics accumulate fixed-size confusion matrices; classes with false positives count in mIoU even if absent from the targets.
Model graph tracing is not performed during training.
See the [PyTorch TensorBoard documentation](https://docs.pytorch.org/docs/stable/tensorboard.html).

## Testing

```bash
uv run --extra cu132 pytest -q
uvx ruff check src tests
uvx ruff format --check src tests
uv run --extra cu132 mypy src
```

## Performance options

Training accepts `--target-cache-mb 256` and `--persistent-workers`.
The cache stores deterministic targets before random augmentation. Its limit is
per dataset instance per process, not a global RAM limit; multiply the limit by
the number of training and validation workers when budgeting memory.
Use `--target-cache-mb 0` to disable it. Workers start with empty caches.
With the default non-persistent workers, their caches are discarded each epoch.
For multi-epoch cache reuse, enable persistent workers or use `--num-workers 0`.

Persistent workers are opt-in because keeping worker RNG state changes the
augmentation sequence compared with recreating workers. They are automatically
disabled when the worker count is zero. The dataset files must remain stable
during a run. SVG target cache keys include file modification time, size, resolution and Gaussian radius.

Loss and confusion statistics accumulate on the compute device; floating loss
sums use float64 to retain the previous Python accumulator precision.
Training progress no longer transfers loss to the CPU for every batch.
Inference uses inference mode without changing the output format or precision.
TTA normalizes/uploads once and transfers the averaged result once. Optional `--tta-batch-size 2` or `4` uses more memory and may introduce small floating-point differences; the default is sequential `1`.
