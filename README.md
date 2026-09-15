# CubiCasa5k-Next

Clean-room, paper-based reimplementation of the CubiCasa5k multi-task
floorplan parsing model with modern Python (3.13) and PyTorch (2.x).

> Derived solely from the papers
> *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan
> Image Analysis* and *Raster-to-Vector: Revisiting Floorplan Transformation*.
> No third-party implementation code was copied.

## Model

- Shared ResNet-152 encoder + residual top-down decoder (hourglass style).
- Three parallel heads: 21 junction heatmaps, 12 room logits, 11 icon logits.
- Homoscedastic-uncertainty multi-task loss:
  `MSE/(2σ²)+log(1+σ)` for heatmaps, `CE/σ+log σ` for segmentation.

## Layout

- `src/cubicasa5k_next/model/` — encoder, decoder, multi-task network
- `src/cubicasa5k_next/losses/` — uncertainty-weighted loss
- `src/cubicasa5k_next/data/` — SVG parsing, rasterization, heatmaps, dataset
- `src/cubicasa5k_next/postprocess/` — junction NMS, walls, rooms, icons, openings
- `src/cubicasa5k_next/training/` — trainer loop, segmentation metrics
- `src/cubicasa5k_next/inference/` — predictor with rotation TTA
- `src/cubicasa5k_next/cli/` — `train`, `infer`, `smoke` commands

## Usage

```bash
uv sync
uv run cubicasa5k-next smoke
uv run cubicasa5k-next train ./data/train ./data/val --epochs 400
uv run cubicasa5k-next infer ./checkpoints/best.pt ./sample.png --output-dir ./outputs
```

## Testing

```bash
uv run pytest -q
uvx ruff check src tests
uvx ruff format --check src tests
uv run mypy src
```
