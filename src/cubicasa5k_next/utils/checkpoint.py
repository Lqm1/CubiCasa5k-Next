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
    num_outputs = 44
    if isinstance(state, dict):
        for key in ("conv4_.weight", "upsample.weight", "upsample.bias"):
            tensor = state.get(key)
            if tensor is not None and hasattr(tensor, "shape"):
                if "conv4_" in key:
                    num_outputs = int(tensor.shape[0])
                elif "upsample" in key and tensor.dim() >= 1:
                    num_outputs = int(tensor.shape[0] if "weight" in key else tensor.shape[0])
                break
    model = FloorplanHourglass(num_outputs).to(active_device)
    model.load_state_dict(state, strict=False)
    metadata = payload if isinstance(payload, dict) else {}
    return model, metadata


def load_criterion_state(checkpoint_path: str | Path) -> dict:
    """Return the loss parameter dict stored alongside model weights."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "criterion_state" not in payload:
        raise ValueError(f"No criterion state in {checkpoint_path}")
    return dict(payload["criterion_state"])
