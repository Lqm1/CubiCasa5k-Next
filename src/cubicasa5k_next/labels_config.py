"""Config-driven room/icon label definitions.

Original CubiCasa5k hard-codes room/icon tables in Python. This module
replaces that with a serializable :class:`LabelConfig` (YAML-loadable) so
future classes can be added without code changes.

The default config reproduces the paper's 12 room / 11 icon layout and the
SVG second-token mappings from the CubiCasa5k loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Heatmap layout is fixed: 13 wall junctions + 4 opening endpoints + 4 icon corners.
NUM_HEATMAP_CHANNELS = 21
NUM_WALL_JUNCTION_TYPES = 13
NUM_OPENING_ENDPOINT_TYPES = 4
NUM_ICON_CORNER_TYPES = 4

WALL_CHANNEL_RANGE = (0, NUM_WALL_JUNCTION_TYPES)
OPENING_CHANNEL_RANGE = (
    NUM_WALL_JUNCTION_TYPES,
    NUM_WALL_JUNCTION_TYPES + NUM_OPENING_ENDPOINT_TYPES,
)
ICON_CORNER_CHANNEL_RANGE = (
    NUM_WALL_JUNCTION_TYPES + NUM_OPENING_ENDPOINT_TYPES,
    NUM_HEATMAP_CHANNELS,
)

ICON_CORNER_NAMES: tuple[str, ...] = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)

DEFAULT_ROOM_CLASSES: tuple[str, ...] = (
    "background",
    "outdoor",
    "wall",
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
    "hallway",
    "balcony_railing",
    "storage",
    "garage",
    "other_room",
)

DEFAULT_ICON_CLASSES: tuple[str, ...] = (
    "empty",
    "window",
    "door",
    "closet",
    "electrical_appliance",
    "toilet",
    "sink",
    "sauna_bench",
    "fireplace",
    "bathtub",
    "chimney",
)

# SVG second-token (class.split()[1]) -> room class name.
DEFAULT_ROOM_SVG_MAP: dict[str, str] = {
    "Background": "background",
    "Outdoor": "outdoor",
    "Wall": "wall",
    "Railing": "balcony_railing",
    "Kitchen": "kitchen",
    "LivingRoom": "living_room",
    "Dining": "living_room",
    "EatingArea": "living_room",
    "Lounge": "living_room",
    "Bedroom": "bedroom",
    "Bath": "bathroom",
    "Sauna": "bathroom",
    "Entry": "hallway",
    "DraughtLobby": "hallway",
    "HallWay": "hallway",
    "CarPort": "garage",
    "Garage": "garage",
    "Closet": "storage",
    "DressingRoom": "storage",
    "Storage": "storage",
}

# SVG second-token -> icon class name, or None to ignore the element.
# Mirrors icons_selected: cabinets/showers/jacuzzi/misc produce no supervision.
DEFAULT_ICON_SVG_MAP: dict[str, str | None] = {
    "Window": "window",
    "Door": "door",
    "Closet": "closet",
    "ClosetRound": "closet",
    "ClosetTriangle": "closet",
    "CoatCloset": "closet",
    "CoatRack": "closet",
    "CounterTop": "closet",
    "Housing": "closet",
    "ElectricalAppliance": "electrical_appliance",
    "WoodStove": "electrical_appliance",
    "GasStove": "electrical_appliance",
    "SaunaStove": "electrical_appliance",
    "IntegratedStove": "electrical_appliance",
    "Dishwasher": "electrical_appliance",
    "GeneralAppliance": "electrical_appliance",
    "Toilet": "toilet",
    "Urinal": "toilet",
    "Sink": "sink",
    "SideSink": "sink",
    "RoundSink": "sink",
    "CornerSink": "sink",
    "DoubleSink": "sink",
    "DoubleSinkRight": "sink",
    "WaterTap": "sink",
    "SaunaBench": "sauna_bench",
    "SaunaBenchHigh": "sauna_bench",
    "SaunaBenchLow": "sauna_bench",
    "SaunaBenchMid": "sauna_bench",
    "Fireplace": "fireplace",
    "FireplaceCorner": "fireplace",
    "FireplaceRound": "fireplace",
    "PlaceForFireplace": "fireplace",
    "PlaceForFireplaceCorner": "fireplace",
    "PlaceForFireplaceRound": "fireplace",
    "Bathtub": "bathtub",
    "BathtubRound": "bathtub",
    "Chimney": "chimney",
    # Ignored: present in drawings but excluded from supervision.
    "BaseCabinet": None,
    "BaseCabinetRound": None,
    "BaseCabinetTriangle": None,
    "WallCabinet": None,
    "Shower": None,
    "ShowerCab": None,
    "ShowerPlatform": None,
    "ShowerScreen": None,
    "ShowerScreenRoundLeft": None,
    "ShowerScreenRoundRight": None,
    "Jacuzzi": None,
    "Misc": None,
    "WashingMachine": None,
}


@dataclass(frozen=True)
class LabelConfig:
    """Ordered class lists plus SVG-token mappings.

    - ``rooms`` / ``icons`` define channel order (index 0 is background/empty).
    - ``room_svg_map`` maps an SVG second token to a room class name.
      Unknown tokens fall back to ``fallback_room``.
    - ``icon_svg_map`` maps an SVG second token to an icon class name or
      ``None`` (ignore element entirely).
    """

    rooms: tuple[str, ...] = DEFAULT_ROOM_CLASSES
    icons: tuple[str, ...] = DEFAULT_ICON_CLASSES
    room_svg_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROOM_SVG_MAP))
    icon_svg_map: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_ICON_SVG_MAP))
    fallback_room: str = "other_room"

    @property
    def num_rooms(self) -> int:
        return len(self.rooms)

    @property
    def num_icons(self) -> int:
        return len(self.icons)

    @property
    def room_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.rooms)}

    @property
    def icon_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.icons)}

    def resolve_room_token(self, token: str) -> int:
        """Map an SVG second token (e.g. ``Bedroom``) to a room index."""
        class_name = self.room_svg_map.get(token, self.fallback_room)
        idx = self.room_index.get(class_name)
        if idx is None:
            idx = self.room_index.get(self.fallback_room, 0)
        return int(idx if idx is not None else 0)

    def resolve_icon_token(self, token: str) -> int | None:
        """Map an SVG second token to an icon index, or ``None`` to ignore."""
        if token in self.icon_svg_map:
            class_name = self.icon_svg_map[token]
            if class_name is None:
                return None
            idx = self.icon_index.get(class_name)
            return int(idx) if idx is not None else None
        return None

    def to_dict(self) -> dict:
        return {
            "rooms": list(self.rooms),
            "icons": list(self.icons),
            "room_svg_map": dict(self.room_svg_map),
            "icon_svg_map": dict(self.icon_svg_map),
            "fallback_room": self.fallback_room,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LabelConfig:
        rooms = tuple(data.get("rooms", list(DEFAULT_ROOM_CLASSES)))
        icons = tuple(data.get("icons", list(DEFAULT_ICON_CLASSES)))
        room_svg_map = dict(data.get("room_svg_map", {}))
        icon_svg_map = dict(data.get("icon_svg_map", {}))
        fallback_room = str(data.get("fallback_room", "other_room"))
        # Validate class references early with clear errors.
        room_set = set(rooms)
        if fallback_room not in room_set:
            raise ValueError(f"fallback_room {fallback_room!r} not in rooms")
        for token, class_name in room_svg_map.items():
            if class_name not in room_set:
                raise ValueError(f"room_svg_map[{token!r}] -> unknown room {class_name!r}")
        icon_set = set(icons)
        for token, class_name in icon_svg_map.items():
            if class_name is not None and class_name not in icon_set:
                raise ValueError(f"icon_svg_map[{token!r}] -> unknown icon {class_name!r}")
        return cls(
            rooms=rooms,
            icons=icons,
            room_svg_map=room_svg_map,
            icon_svg_map=icon_svg_map,
            fallback_room=fallback_room,
        )


def default_label_config() -> LabelConfig:
    return LabelConfig()


def load_label_config(path: str | Path) -> LabelConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Label config {path} must contain a mapping")
    return LabelConfig.from_dict(data)


def save_label_config(config: LabelConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False, allow_unicode=True)
