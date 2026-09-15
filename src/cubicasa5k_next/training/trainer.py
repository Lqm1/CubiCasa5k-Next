"""Training loop for the hourglass floorplan model (modern PyTorch)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cubicasa5k_next.config import TrainingConfig
from cubicasa5k_next.data.dataset import FloorplanSample
from cubicasa5k_next.losses.uncertainty import UncertaintyWeightedLoss
from cubicasa5k_next.model.hourglass import FloorplanHourglass
from cubicasa5k_next.training.metrics import compute_segmentation_scores
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

    best_loss = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        running_total = 0.0
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

        validation_loss = evaluate_loss(model, criterion, validation_loader, active_device)
        scheduler.step(validation_loss)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "validation_loss": validation_loss,
                "config": config,
            },
            last_path,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "validation_loss": validation_loss,
                    "config": config,
                },
                best_path,
            )
        print(
            f"epoch={epoch} train_loss={running_total / max(len(train_loader), 1):.4f} "
            f"val_loss={validation_loss:.4f} best={best_loss:.4f}"
        )
    return best_path


@torch.no_grad()
def evaluate_loss(model, criterion, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
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
        total += float(loss_output.total.cpu()) * images.size(0)
        count += images.size(0)
        _ = compute_segmentation_scores(room_logits.cpu(), batch["room_labels"], 12)
    return total / max(count, 1)
