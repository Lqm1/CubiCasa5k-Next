"""Training loop for the hourglass floorplan model (modern PyTorch)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cubicasa5k_next.config import TrainingConfig
from cubicasa5k_next.data.dataset import FloorplanSample
from cubicasa5k_next.losses.uncertainty import UncertaintyWeightedLoss
from cubicasa5k_next.model.hourglass import FloorplanHourglass
from cubicasa5k_next.tensorboard import create_writer, should_log_images
from cubicasa5k_next.utils.seed import set_random_seed


def _collate(samples: list[FloorplanSample]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([s.image for s in samples]),
        "room_labels": torch.stack([s.room_labels for s in samples]),
        "icon_labels": torch.stack([s.icon_labels for s in samples]),
        "heatmaps": torch.stack([s.heatmaps for s in samples]),
    }


def build_dataloaders(
    train_dataset, validation_dataset, config: TrainingConfig, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=use_pin_memory,
    )
    return train_loader, validation_loader


def _split_raw_output(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The model already applies sigmoid to the first 21 channels.
    heatmaps = raw[:, :21]
    room_logits = raw[:, 21:33]
    icon_logits = raw[:, 33:44]
    return room_logits, icon_logits, heatmaps


def resolve_tensorboard_dir(config: TrainingConfig) -> Path | None:
    """Return the explicit log root; None disables logging."""
    return None if config.tensorboard_dir is None else Path(config.tensorboard_dir)


@dataclass
class ValidationSnapshot:
    loss: float
    heatmap_term: float
    segmentation_term: float
    room_cross_entropy: float
    icon_cross_entropy: float
    heatmap_mse: float
    room_mean_iou: float
    room_accuracy: float
    icon_mean_iou: float
    icon_accuracy: float
    sample_images: torch.Tensor | None = None
    sample_room_targets: torch.Tensor | None = None
    sample_room_preds: torch.Tensor | None = None
    sample_heatmaps: torch.Tensor | None = None


def _denormalize_for_display(images: torch.Tensor) -> torch.Tensor:
    # Dataset maps uint8 [0, 255] to [-1, 1]; map back to [0, 1] for display.
    return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def _label_map_for_display(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Render integer label maps as single-channel float images in [0, 1].
    normalized = labels.float() / max(num_classes - 1, 1)
    return normalized.unsqueeze(1).clamp(0.0, 1.0)


def train_model(
    train_dataset,
    validation_dataset,
    config: TrainingConfig,
    device: torch.device | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Train (or continue training) and return the best checkpoint path."""
    set_random_seed(config.seed)
    active_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FloorplanHourglass(44).to(active_device)
    criterion = UncertaintyWeightedLoss(num_heatmap_channels=21).to(active_device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = torch.load(str(resume_checkpoint), map_location=active_device, weights_only=False)
        if isinstance(payload, dict):
            if "model_state" in payload:
                model.load_state_dict(payload["model_state"], strict=False)
            stored_criterion = payload.get("criterion_state")
            if isinstance(stored_criterion, dict):
                try:
                    criterion.load_state_dict(stored_criterion, strict=False)
                except RuntimeError:
                    pass
            if "optimizer_state" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer_state"])
                except (ValueError, RuntimeError):
                    pass
            start_epoch = int(payload.get("epoch", 0)) + 1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=active_device.type == "cuda")

    train_loader, validation_loader = build_dataloaders(
        train_dataset, validation_dataset, config, active_device
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    should_log_images(1, config.tensorboard_image_every, config.tensorboard_max_images)
    writer = create_writer(
        config.tensorboard_dir, config.tensorboard_run_name, config.tensorboard_flush_secs
    )

    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    try:
        if writer is not None:
            writer.add_text(
                "Config/training", json.dumps(asdict(config), indent=2), start_epoch - 1
            )
        for epoch in range(start_epoch, config.max_epochs + 1):
            model.train()
            running_total = 0.0
            running_heatmap = 0.0
            running_seg = 0.0
            running_room_ce = 0.0
            running_icon_ce = 0.0
            running_heat_mse = 0.0
            num_batches = 0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                images = batch["image"].to(active_device, non_blocking=True)
                room_labels = batch["room_labels"].to(active_device, non_blocking=True)
                icon_labels = batch["icon_labels"].to(active_device, non_blocking=True)
                heatmaps = batch["heatmaps"].to(active_device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=active_device.type == "cuda"):
                    raw = model(images)
                    room_logits, icon_logits, heat_pred = _split_raw_output(raw)
                    loss_output = criterion(
                        heat_pred, heatmaps, room_logits, room_labels, icon_logits, icon_labels
                    )
                scaler.scale(loss_output.total).backward()
                scaler.step(optimizer)
                scaler.update()
                running_total += float(loss_output.total.detach().cpu())
                running_heatmap += float(loss_output.heatmap_term.detach().cpu())
                running_seg += float(loss_output.segmentation_term.detach().cpu())
                running_room_ce += float(loss_output.room_cross_entropy.detach().cpu())
                running_icon_ce += float(loss_output.icon_cross_entropy.detach().cpu())
                running_heat_mse += float(loss_output.heatmap_mse.detach().cpu())
                num_batches += 1

            log_images = writer is not None and should_log_images(
                epoch, config.tensorboard_image_every, config.tensorboard_max_images
            )
            snapshot = evaluate_detailed(
                model,
                criterion,
                validation_loader,
                active_device,
                max_images=config.tensorboard_max_images if log_images else 0,
            )
            scheduler.step(snapshot.loss)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "validation_loss": snapshot.loss,
                    "config": config,
                },
                last_path,
            )
            if snapshot.loss < best_loss:
                best_loss = snapshot.loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "criterion_state": criterion.state_dict(),
                        "validation_loss": snapshot.loss,
                        "config": config,
                    },
                    best_path,
                )
            train_loss = running_total / max(num_batches, 1)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={snapshot.loss:.4f} best={best_loss:.4f}"
            )

            if writer is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/valid", snapshot.loss, epoch)
                writer.add_scalar(
                    "Loss/train_heatmap", running_heatmap / max(num_batches, 1), epoch
                )
                writer.add_scalar("Loss/train_seg", running_seg / max(num_batches, 1), epoch)
                writer.add_scalar(
                    "Loss/train_room_ce", running_room_ce / max(num_batches, 1), epoch
                )
                writer.add_scalar(
                    "Loss/train_icon_ce", running_icon_ce / max(num_batches, 1), epoch
                )
                writer.add_scalar(
                    "Loss/train_heatmap_mse", running_heat_mse / max(num_batches, 1), epoch
                )
                writer.add_scalar("Loss/valid_heatmap", snapshot.heatmap_term, epoch)
                writer.add_scalar("Loss/valid_seg", snapshot.segmentation_term, epoch)
                writer.add_scalar("Loss/valid_room_ce", snapshot.room_cross_entropy, epoch)
                writer.add_scalar("Loss/valid_icon_ce", snapshot.icon_cross_entropy, epoch)
                writer.add_scalar("Loss/valid_heatmap_mse", snapshot.heatmap_mse, epoch)
                writer.add_scalar("Optimizer/lr", current_lr, epoch)
                writer.add_scalar("Metrics/valid_room_miou", snapshot.room_mean_iou, epoch)
                writer.add_scalar("Metrics/valid_room_acc", snapshot.room_accuracy, epoch)
                writer.add_scalar("Metrics/valid_icon_miou", snapshot.icon_mean_iou, epoch)
                writer.add_scalar("Metrics/valid_icon_acc", snapshot.icon_accuracy, epoch)
                try:
                    log_vars = criterion.log_vars.detach().cpu()
                    writer.add_scalar("Uncertainty/log_var_room", float(log_vars[0]), epoch)
                    writer.add_scalar("Uncertainty/log_var_icon", float(log_vars[1]), epoch)
                    writer.add_scalar(
                        "Uncertainty/log_var_heatmap_mean",
                        float(criterion.log_vars_mse.detach().cpu().mean()),
                        epoch,
                    )
                except Exception:  # noqa: BLE001, S110 - auxiliary logging only
                    pass
                if (
                    snapshot.sample_images is not None
                    and snapshot.sample_room_targets is not None
                    and snapshot.sample_room_preds is not None
                    and snapshot.sample_heatmaps is not None
                ):
                    count = snapshot.sample_images.size(0)
                    writer.add_images(
                        "Samples/valid/input",
                        _denormalize_for_display(snapshot.sample_images[:count]),
                        epoch,
                    )
                    writer.add_images(
                        "Samples/valid/room_target",
                        snapshot.sample_room_targets[:count],
                        epoch,
                    )
                    writer.add_images(
                        "Samples/valid/room_pred",
                        snapshot.sample_room_preds[:count],
                        epoch,
                    )
                    writer.add_images(
                        "Samples/valid/heatmap_mean",
                        snapshot.sample_heatmaps[:count].mean(dim=1, keepdim=True),
                        epoch,
                    )
                writer.flush()
    finally:
        if writer is not None:
            writer.close()
    return best_path


@torch.no_grad()
def evaluate_detailed(
    model, criterion, loader: DataLoader, device: torch.device, max_images: int = 0
) -> ValidationSnapshot:
    """Evaluate the full validation split and keep samples for TensorBoard."""
    model.eval()
    total = 0.0
    heatmap_term = 0.0
    segmentation_term = 0.0
    room_ce = 0.0
    icon_ce = 0.0
    heat_mse = 0.0
    count = 0
    room_confusion = torch.zeros((12, 12), dtype=torch.int64, device=device)
    icon_confusion = torch.zeros((11, 11), dtype=torch.int64, device=device)
    sample_images: torch.Tensor | None = None
    sample_room_targets: torch.Tensor | None = None
    sample_room_preds: torch.Tensor | None = None
    sample_heatmaps: torch.Tensor | None = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        room_labels = batch["room_labels"].to(device, non_blocking=True)
        icon_labels = batch["icon_labels"].to(device, non_blocking=True)
        heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        raw = model(images)
        room_logits, icon_logits, heat_pred = _split_raw_output(raw)
        loss_output = criterion(
            heat_pred, heatmaps, room_logits, room_labels, icon_logits, icon_labels
        )
        batch_size = images.size(0)
        total += float(loss_output.total.cpu()) * batch_size
        heatmap_term += float(loss_output.heatmap_term.cpu()) * batch_size
        segmentation_term += float(loss_output.segmentation_term.cpu()) * batch_size
        room_ce += float(loss_output.room_cross_entropy.cpu()) * batch_size
        icon_ce += float(loss_output.icon_cross_entropy.cpu()) * batch_size
        heat_mse += float(loss_output.heatmap_mse.cpu()) * batch_size
        count += batch_size
        room_confusion += torch.bincount(
            (room_labels * 12 + room_logits.argmax(dim=1)).flatten(), minlength=144
        ).reshape(12, 12)
        icon_confusion += torch.bincount(
            (icon_labels * 11 + icon_logits.argmax(dim=1)).flatten(), minlength=121
        ).reshape(11, 11)
        if max_images > 0 and sample_images is None:
            keep = min(max_images, batch_size)
            sample_images = batch["image"][:keep].cpu().clone()
            sample_room_targets = _label_map_for_display(batch["room_labels"][:keep].cpu(), 12)
            sample_room_preds = _label_map_for_display(room_logits[:keep].cpu().argmax(dim=1), 12)
            sample_heatmaps = heat_pred[:keep].cpu().clamp(0.0, 1.0)
    denom = max(count, 1)
    room_miou, room_accuracy = _confusion_scores(room_confusion)
    icon_miou, icon_accuracy = _confusion_scores(icon_confusion)
    return ValidationSnapshot(
        loss=total / denom,
        heatmap_term=heatmap_term / denom,
        segmentation_term=segmentation_term / denom,
        room_cross_entropy=room_ce / denom,
        icon_cross_entropy=icon_ce / denom,
        heatmap_mse=heat_mse / denom,
        room_mean_iou=room_miou,
        room_accuracy=room_accuracy,
        icon_mean_iou=icon_miou,
        icon_accuracy=icon_accuracy,
        sample_images=sample_images,
        sample_room_targets=sample_room_targets,
        sample_room_preds=sample_room_preds,
        sample_heatmaps=sample_heatmaps,
    )


@torch.no_grad()
def evaluate_loss(model, criterion, loader: DataLoader, device: torch.device) -> float:
    return evaluate_detailed(model, criterion, loader, device).loss


def _confusion_scores(confusion: torch.Tensor) -> tuple[float, float]:
    """Compute dataset metrics from fixed-size counts, including false-positive-only classes."""
    counts = confusion.double()
    correct = counts.diag()
    union = counts.sum(0) + counts.sum(1) - correct
    present = union > 0
    miou = (correct[present] / union[present]).mean().item() if present.any() else 0.0
    accuracy = (correct.sum() / counts.sum().clamp_min(1)).item()
    return miou, accuracy
