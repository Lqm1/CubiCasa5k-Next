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

    @torch.inference_mode()
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

        # Normalize and upload once. Keep rotations sequential by default to limit VRAM.
        batch_size = self.config.tta_batch_size
        if batch_size not in (1, 2, 4):
            raise ValueError("tta_batch_size must be 1, 2 or 4")
        accumulator = None
        for start in range(0, 4, batch_size):
            turns = range(start, start + batch_size)
            rotated = torch.cat([torch.rot90(tensor, k, (-2, -1)) for k in turns])
            outputs = self.model(rotated)
            for offset, k in enumerate(turns):
                restored = torch.rot90(outputs[offset], -k, (-2, -1))
                accumulator = restored.clone() if accumulator is None else accumulator + restored
        assert accumulator is not None
        return split_raw_output((accumulator / 4.0).cpu().numpy())

    def predict_vector(self, image: np.ndarray) -> VectorFloorplan:
        room_logits, icon_logits, heatmaps = self.predict_arrays(image)
        return vectorize_prediction(
            room_logits,
            icon_logits,
            heatmaps,
            junction_threshold=self.config.junction_threshold,
            alignment_tolerance=self.config.alignment_tolerance_px,
        )
