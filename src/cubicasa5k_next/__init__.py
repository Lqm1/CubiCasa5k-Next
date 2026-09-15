"""CubiCasa5k-Next: clean-room multi-task floorplan parsing."""

from __future__ import annotations

__version__ = "0.1.0"

from cubicasa5k_next.config import InferenceConfig, ModelConfig, TrainingConfig
from cubicasa5k_next.labels import (
    ICON_CLASS_NAMES,
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CLASSES,
    NUM_ROOM_CLASSES,
    ROOM_CLASS_NAMES,
)

__all__ = [
    "__version__",
    "InferenceConfig",
    "ModelConfig",
    "TrainingConfig",
    "ROOM_CLASS_NAMES",
    "ICON_CLASS_NAMES",
    "NUM_ROOM_CLASSES",
    "NUM_ICON_CLASSES",
    "NUM_HEATMAP_CHANNELS",
]
