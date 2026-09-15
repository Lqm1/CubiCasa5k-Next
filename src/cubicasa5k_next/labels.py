"""Canonical label definitions derived from the paper's evaluation tables.

The paper reports 12 room classes and 11 icon classes (including background).
Names below are paraphrased English descriptions, not copied from any
third-party codebase.
"""

from __future__ import annotations

ROOM_CLASS_NAMES: tuple[str, ...] = (
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

ICON_CLASS_NAMES: tuple[str, ...] = (
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

NUM_ROOM_CLASSES = len(ROOM_CLASS_NAMES)
NUM_ICON_CLASSES = len(ICON_CLASS_NAMES)

# Heatmap channel layout: 21 channels = 13 wall + 4 opening + 4 icon-corner.
# This follows the stratification described in the Raster-to-Vector paper
# (wall junctions by shape/orientation, opening endpoints, icon corners).
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

ROOM_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ROOM_CLASS_NAMES)}
ICON_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ICON_CLASS_NAMES)}


def room_name_to_index(name: str) -> int:
    """Map a normalized room name to its class index."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ROOM_INDEX:
        return ROOM_INDEX[key]
    # Alias table for the canonical 12-class layout.
    # Index 4 covers dining/living/lounge, index 7 covers entry-like spaces.
    aliases = {
        "livingroom": "living_room",
        "living": "living_room",
        "dining": "living_room",
        "lounge": "living_room",
        "eatingarea": "living_room",
        "bath": "bathroom",
        "sauna": "bathroom",
        "hall": "hallway",
        "hallway": "hallway",
        "entry": "hallway",
        "draughtlobby": "hallway",
        "draught_lobby": "hallway",
        "entrance": "hallway",
        "corridor": "hallway",
        "closet": "storage",
        "dressingroom": "storage",
        "dressing_room": "storage",
        "storage": "storage",
        "carport": "garage",
        "garage": "garage",
        "outdoor": "outdoor",
        "wall": "wall",
        "railing": "balcony_railing",
        "balcony": "balcony_railing",
        "balcony_railing": "balcony_railing",
        "kitchen": "kitchen",
        "bedroom": "bedroom",
        "background": "background",
        "other": "other_room",
        "other_rooms": "other_room",
        "room": "other_room",
    }
    mapped = aliases.get(key, "other_room")
    return ROOM_INDEX.get(mapped, ROOM_INDEX["other_room"])


def icon_name_to_index(name: str) -> int:
    """Map a normalized icon/opening name to its class index."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ICON_INDEX:
        return ICON_INDEX[key]
    aliases = {
        "electrical_appliance_unit": "electrical_appliance",
        "electr_appl": "electrical_appliance",
        "electrical": "electrical_appliance",
        "woodstove": "electrical_appliance",
        "gasstove": "electrical_appliance",
        "integratedstove": "electrical_appliance",
        "dishwasher": "electrical_appliance",
        "generalappliance": "electrical_appliance",
        "sauna": "sauna_bench",
        "saunabench": "sauna_bench",
        "saunabenchhigh": "sauna_bench",
        "saunabenchlow": "sauna_bench",
        "saunabenchmid": "sauna_bench",
        "fire_place": "fireplace",
        "fireplacecorner": "fireplace",
        "fireplaceround": "fireplace",
        "bath_tub": "bathtub",
        "bathtubround": "bathtub",
        "wash_basin": "sink",
        "basin": "sink",
        "sink": "sink",
        "roundsink": "sink",
        "cornersink": "sink",
        "doublesink": "sink",
        "sidesink": "sink",
        "watertap": "sink",
        "closet_unit": "closet",
        "closetround": "closet",
        "closettriangle": "closet",
        "coatcloset": "closet",
        "coatrack": "closet",
        "countertop": "closet",
        "housing": "closet",
        "toilet": "toilet",
        "urinal": "toilet",
        "chimney": "chimney",
        "window": "window",
        "door": "door",
        "empty": "empty",
        "background": "empty",
    }
    mapped = aliases.get(key, "empty")
    return ICON_INDEX.get(mapped, ICON_INDEX["empty"])
