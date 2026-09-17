"""CubiCasa5k-Next: clean-room multi-task floorplan parsing."""

from __future__ import annotations

__version__ = "0.2.0"

from cubicasa5k_next.config import InferenceConfig, ModelConfig, TrainingConfig
from cubicasa5k_next.labels import (
    ICON_CLASS_NAMES,
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CLASSES,
    NUM_ROOM_CLASSES,
    ROOM_CLASS_NAMES,
)
from cubicasa5k_next.labels_config import LabelConfig, default_label_config, load_label_config

__all__ = [
    "__version__",
    "InferenceConfig",
    "ModelConfig",
    "TrainingConfig",
    "LabelConfig",
    "default_label_config",
    "load_label_config",
    "ROOM_CLASS_NAMES",
    "ICON_CLASS_NAMES",
    "NUM_ROOM_CLASSES",
    "NUM_ICON_CLASSES",
    "NUM_HEATMAP_CHANNELS",
]
