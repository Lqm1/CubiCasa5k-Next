"""Single-image inference with optional rotation test-time augmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as invoke_functional

from cubicasa5k_next.config import InferenceConfig
from cubicasa5k_next.data.transforms import to_normalized_tensor
from cubicasa5k_next.model.hourglass import FloorplanHourglass
from cubicasa5k_next.postprocess.vectorize import VectorFloorplan, vectorize_prediction
from cubicasa5k_next.utils.checkpoint import load_checkpoint


def split_raw_output(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 44 channels into room logits (12), icon logits (11), heatmaps (21)."""
    heatmaps = raw[:21]
    room_logits = raw[21:33]
    icon_logits = raw[33:44]
    return room_logits, icon_logits, heatmaps


class FloorplanPredictor:
    def __init__(
        self,
        model: FloorplanHourglass,
        config: InferenceConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: InferenceConfig | None = None,
        device: torch.device | None = None,
    ) -> FloorplanPredictor:
        model, _ = load_checkpoint(checkpoint_path, device=device)
        return cls(model, config, device)

    @torch.no_grad()
    def predict_arrays(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the network and return (room_logits, icon_logits, heatmaps) as numpy."""
        size = self.config.image_size
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        tensor = to_normalized_tensor(resized).unsqueeze(0).to(self.device)

        if not self.config.use_test_time_rotation:
            raw = self.model(tensor)[0].cpu().numpy()
            if raw.shape[-2:] != (size, size):
                raw_tensor = torch.from_numpy(raw).unsqueeze(0)
                raw = invoke_functional.interpolate(
                    raw_tensor, size=(size, size), mode="bilinear", align_corners=False
                )[0].numpy()
            return split_raw_output(raw)

        room_accumulator = None
        icon_accumulator = None
        heat_accumulator = None
        for turns in range(4):
            rotated = np.rot90(resized, turns).copy()
            rotated_tensor = to_normalized_tensor(rotated).unsqueeze(0).to(self.device)
            raw = self.model(rotated_tensor)[0].cpu().numpy()
            room, icon, heat = split_raw_output(raw)
            room = np.rot90(room, -turns, axes=(1, 2)).copy()
            icon = np.rot90(icon, -turns, axes=(1, 2)).copy()
            heat = np.rot90(heat, -turns, axes=(1, 2)).copy()
            room_accumulator = room if room_accumulator is None else room_accumulator + room
            icon_accumulator = icon if icon_accumulator is None else icon_accumulator + icon
            heat_accumulator = heat if heat_accumulator is None else heat_accumulator + heat
        assert (
            room_accumulator is not None
            and icon_accumulator is not None
            and heat_accumulator is not None
        )
        return room_accumulator / 4.0, icon_accumulator / 4.0, heat_accumulator / 4.0

    def predict_vector(self, image: np.ndarray) -> VectorFloorplan:
        room_logits, icon_logits, heatmaps = self.predict_arrays(image)
        return vectorize_prediction(
            room_logits,
            icon_logits,
            heatmaps,
            junction_threshold=self.config.junction_threshold,
            alignment_tolerance=self.config.alignment_tolerance_px,
        )
