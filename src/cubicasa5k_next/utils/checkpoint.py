"""Checkpoint loading for hourglass weights."""

from __future__ import annotations

from pathlib import Path

import torch

from cubicasa5k_next.model.hourglass import FloorplanHourglass


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> tuple[FloorplanHourglass, dict]:
    active_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(checkpoint_path), map_location=active_device, weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else {}
    model = FloorplanHourglass(44).to(active_device)
    model.load_state_dict(state, strict=False)
    metadata = payload if isinstance(payload, dict) else {}
    return model, metadata


def load_criterion_state(checkpoint_path: str | Path) -> dict:
    """Return the loss parameter dict stored alongside model weights."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "criterion_state" not in payload:
        raise ValueError(f"No criterion state in {checkpoint_path}")
    return dict(payload["criterion_state"])
