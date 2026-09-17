"""Default label constants derived from the default :class:`LabelConfig`.

New code should prefer :mod:`cubicasa5k_next.labels_config` with an explicit
``LabelConfig``; this module only exposes the default vocabulary constants.
"""

from __future__ import annotations

from cubicasa5k_next.labels_config import (
    ICON_CORNER_CHANNEL_RANGE,
    ICON_CORNER_NAMES,
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CORNER_TYPES,
    NUM_OPENING_ENDPOINT_TYPES,
    NUM_WALL_JUNCTION_TYPES,
    OPENING_CHANNEL_RANGE,
    WALL_CHANNEL_RANGE,
    default_label_config,
)

_DEFAULT = default_label_config()

ROOM_CLASS_NAMES: tuple[str, ...] = _DEFAULT.rooms
ICON_CLASS_NAMES: tuple[str, ...] = _DEFAULT.icons
NUM_ROOM_CLASSES = len(ROOM_CLASS_NAMES)
NUM_ICON_CLASSES = len(ICON_CLASS_NAMES)
ROOM_INDEX: dict[str, int] = {n: i for i, n in enumerate(ROOM_CLASS_NAMES)}
ICON_INDEX: dict[str, int] = {n: i for i, n in enumerate(ICON_CLASS_NAMES)}


__all__ = [
    "ROOM_CLASS_NAMES",
    "ICON_CLASS_NAMES",
    "NUM_ROOM_CLASSES",
    "NUM_ICON_CLASSES",
    "NUM_HEATMAP_CHANNELS",
    "NUM_WALL_JUNCTION_TYPES",
    "NUM_OPENING_ENDPOINT_TYPES",
    "NUM_ICON_CORNER_TYPES",
    "WALL_CHANNEL_RANGE",
    "OPENING_CHANNEL_RANGE",
    "ICON_CORNER_CHANNEL_RANGE",
    "ICON_CORNER_NAMES",
    "ROOM_INDEX",
    "ICON_INDEX",
]
