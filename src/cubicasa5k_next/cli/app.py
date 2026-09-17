"""Typer-based CLI for training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from cubicasa5k_next.config import AugmentationConfig, InferenceConfig, ModelConfig, TrainingConfig
from cubicasa5k_next.data.dataset import SvgFloorplanDataset, discover_pairs
from cubicasa5k_next.inference.predictor import FloorplanPredictor
from cubicasa5k_next.training.trainer import train_model
from cubicasa5k_next.utils.image import load_rgb_image, save_colored_mask
from cubicasa5k_next.utils.seed import set_random_seed

app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True, help="Model export commands.")
app.add_typer(export_app, name="export")


def _resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda to a concrete device with a clear CUDA error."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise typer.BadParameter("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise typer.BadParameter(f"Unknown device: {requested} (expected auto, cpu, or cuda)")


@app.command()
def train(
    train_root: Path = typer.Argument(..., help="Train split dir (images/ + annotations/)"),
    validation_root: Path = typer.Argument(
        ..., help="Validation split dir (images/ + annotations/)"
    ),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Where to store checkpoints"),
    epochs: int = typer.Option(400, help="Maximum training epochs"),
    batch_size: int = typer.Option(20, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Adam initial learning rate"),
    image_size: int = typer.Option(256, help="Square training resolution"),
    seed: int = typer.Option(42, help="Random seed"),
    num_workers: int = typer.Option(4, help="DataLoader workers (use 0 on Windows CPU)"),
    weights: Path | None = typer.Option(None, help="Resume from a checkpoint file"),
    device: str = typer.Option("auto", help="Compute device (auto, cpu, or cuda)"),
    labels_config: Path | None = typer.Option(
        None, help="Room/icon label config YAML (default: built-in 12/11)"
    ),
    tensorboard_dir: Path | None = typer.Option(
        None, help="TensorBoard log dir (default: <checkpoint_dir>/tensorboard)"
    ),
    persistent_workers: bool = typer.Option(
        False, help="Reuse workers across epochs; changes augmentation RNG sequence"
    ),
    target_cache_mb: int = typer.Option(
        256, min=0, help="Target cache MiB per dataset per worker; 0 disables"
    ),
    no_tensorboard: bool = typer.Option(False, help="Disable TensorBoard logging"),
    tensorboard_run_name: str = typer.Option(
        "cubicasa5k-next", help="Run name prefix (letters, digits, dots, underscores, hyphens)"
    ),
    tensorboard_image_every: int = typer.Option(
        5, min=0, help="Log images at epoch 1 and every N epochs; 0 disables images"
    ),
    tensorboard_max_images: int = typer.Option(
        4, min=0, help="Maximum sample images; 0 disables images"
    ),
    tensorboard_flush_secs: int = typer.Option(
        30, min=1, help="TensorBoard flush interval in seconds"
    ),
) -> None:
    """Train the hourglass model (supports resuming from a checkpoint)."""
    from cubicasa5k_next.labels_config import default_label_config, load_label_config

    active_device = _resolve_device(device)
    if active_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    set_random_seed(seed)
    label_config = (
        load_label_config(labels_config) if labels_config is not None else default_label_config()
    )
    augmentation = AugmentationConfig(image_size=image_size)
    resolved_tensorboard_dir: str | None
    if no_tensorboard:
        resolved_tensorboard_dir = None
    elif tensorboard_dir is not None:
        resolved_tensorboard_dir = str(tensorboard_dir)
    else:
        resolved_tensorboard_dir = str(checkpoint_dir / "tensorboard")
    config = TrainingConfig(
        batch_size=batch_size,
        max_epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        model=ModelConfig.for_label_counts(label_config.num_rooms, label_config.num_icons),
        augmentation=augmentation,
        num_workers=num_workers,
        tensorboard_dir=resolved_tensorboard_dir,
        persistent_workers=persistent_workers,
        target_cache_mb=target_cache_mb,
        tensorboard_run_name=tensorboard_run_name,
        tensorboard_image_every=tensorboard_image_every,
        tensorboard_max_images=tensorboard_max_images,
        tensorboard_flush_secs=tensorboard_flush_secs,
    )
    train_images, train_annotations = discover_pairs(train_root)
    validation_images, validation_annotations = discover_pairs(validation_root)
    if not train_images:
        raise typer.BadParameter(f"No (image, svg) pairs found under {train_root}")
    if not validation_images:
        raise typer.BadParameter(f"No (image, svg) pairs found under {validation_root}")

    train_dataset = SvgFloorplanDataset(
        train_images,
        train_annotations,
        augmentation,
        training=True,
        target_cache_mb=target_cache_mb,
        label_config=label_config,
    )
    validation_dataset = SvgFloorplanDataset(
        validation_images,
        validation_annotations,
        augmentation,
        training=False,
        target_cache_mb=target_cache_mb,
        label_config=label_config,
    )
    best_path = train_model(
        train_dataset, validation_dataset, config, device=active_device, resume_checkpoint=weights
    )
    typer.echo(f"Best checkpoint: {best_path} (device={active_device.type})")


@app.command()
def infer(
    checkpoint: Path = typer.Argument(..., help="Model checkpoint file"),
    image: Path = typer.Argument(..., help="Input floorplan image"),
    output_dir: Path = typer.Option(Path("outputs"), help="Where to write predictions"),
    use_tta: bool = typer.Option(False, help="Average predictions over 4 rotations"),
    tta_batch_size: int = typer.Option(1, min=1, max=4, help="TTA batch size: 1, 2 or 4"),
    device: str = typer.Option("auto", help="Compute device (auto, cpu, or cuda)"),
    labels_config: Path | None = typer.Option(
        None, help="Room/icon label config YAML (default: built-in 12/11)"
    ),
) -> None:
    """Run inference and save segmentation masks plus vectorization stats."""
    from cubicasa5k_next.labels_config import default_label_config, load_label_config

    output_dir.mkdir(parents=True, exist_ok=True)
    active_device = _resolve_device(device)
    if tta_batch_size not in (1, 2, 4):
        raise typer.BadParameter("TTA batch size must be 1, 2 or 4")
    inference_config = InferenceConfig(
        use_test_time_rotation=use_tta, tta_batch_size=tta_batch_size, device=active_device.type
    )
    label_config = (
        load_label_config(labels_config) if labels_config is not None else default_label_config()
    )
    # Single hourglass path handles checkpoint files with matching keys.
    predictor = FloorplanPredictor.from_checkpoint(
        checkpoint,
        inference_config,
        active_device,
        num_heatmap_channels=21,
        num_room_classes=label_config.num_rooms,
        num_icon_classes=label_config.num_icons,
    )
    rgb = load_rgb_image(image, target_size=inference_config.image_size)
    room_logits, icon_logits, heatmaps = predictor.predict_arrays(rgb)

    import numpy as np

    from cubicasa5k_next.postprocess.vectorize import vectorize_prediction

    wall_index = label_config.room_index.get("wall", 2)
    background_index = label_config.room_index.get("background", 0)
    window_index = label_config.icon_index.get("window", 1)
    door_index = label_config.icon_index.get("door", 2)
    empty_index = label_config.icon_index.get("empty", 0)
    vector_result = vectorize_prediction(
        room_logits,
        icon_logits,
        heatmaps,
        junction_threshold=inference_config.junction_threshold,
        alignment_tolerance=inference_config.alignment_tolerance_px,
        wall_class_index=int(wall_index),
        window_class_index=int(window_index),
        door_class_index=int(door_index),
        room_ignored_indices=(int(background_index), int(wall_index)),
        icon_empty_index=int(empty_index),
    )
    room_labels = np.argmax(room_logits, axis=0).astype("int64")
    icon_labels = np.argmax(icon_logits, axis=0).astype("int64")
    save_colored_mask(room_labels, output_dir / f"{image.stem}_rooms.png")
    save_colored_mask(icon_labels, output_dir / f"{image.stem}_icons.png")
    np.save(output_dir / f"{image.stem}_heatmaps.npy", heatmaps)
    typer.echo(
        f"walls={len(vector_result.walls)} rooms={len(vector_result.rooms)} "
        f"icons={len(vector_result.icons)} openings={len(vector_result.openings)} "
        f"junctions={len(vector_result.junctions)}"
    )


@export_app.command("onnx")
def export_onnx(
    checkpoint: Path = typer.Argument(..., help="Model checkpoint file"),
    output: Path = typer.Argument(..., help="Destination .onnx path"),
    image_size: int = typer.Option(256, help="Square input resolution"),
    opset: int = typer.Option(18, help="ONNX opset version"),
) -> None:
    """Export the hourglass model to ONNX and verify parity with torch."""
    import numpy as np

    from cubicasa5k_next.utils.checkpoint import load_checkpoint

    model, _ = load_checkpoint(checkpoint, device=torch.device("cpu"))
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_dim = torch.export.Dim("batch")
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["image"],
        output_names=["raw44"],
        dynamic_shapes={"image": {0: batch_dim}},
        opset_version=opset,
    )
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(dummy).numpy()
    actual = session.run(["raw44"], {"image": dummy.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    typer.echo(f"exported {output} (opset={opset}) max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-3:
        raise typer.BadParameter(f"ONNX parity check failed: diff={max_diff:.2e}")


@app.command()
def smoke(
    device: str = typer.Option("auto", help="Compute device (auto, cpu, or cuda)"),
) -> None:
    """Quick forward pass to verify installation."""
    active_device = _resolve_device(device)
    from cubicasa5k_next.model.hourglass import FloorplanHourglass

    model = FloorplanHourglass(44).to(active_device).eval()
    dummy = torch.randn(1, 3, 256, 256, device=active_device)
    with torch.no_grad():
        raw = model(dummy)
    typer.echo(
        f"device={active_device.type} cuda_available={torch.cuda.is_available()} "
        f"raw={tuple(raw.shape)} expected=(1, 44, 256, 256)"
    )


if __name__ == "__main__":
    app()
