from cubicasa5k_next.labels import (
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CLASSES,
    NUM_ROOM_CLASSES,
)
from cubicasa5k_next.labels_config import default_label_config


def test_class_counts_match_paper() -> None:
    assert NUM_ROOM_CLASSES == 12
    assert NUM_ICON_CLASSES == 11
    assert NUM_HEATMAP_CHANNELS == 21


def test_default_config_matches_constants() -> None:
    config = default_label_config()
    assert config.num_rooms == NUM_ROOM_CLASSES
    assert config.num_icons == NUM_ICON_CLASSES
    # Regression guards for previously broken mappings.
    assert config.resolve_room_token("Bedroom") == config.room_index["bedroom"]
    assert config.resolve_room_token("Outdoor") == config.room_index["outdoor"]
    assert config.resolve_room_token("Undefined") == config.room_index["other_room"]
    assert config.resolve_icon_token("Door") == config.icon_index["door"]
    assert config.resolve_icon_token("Window") == config.icon_index["window"]
    assert config.resolve_icon_token("BaseCabinet") is None
    assert config.resolve_icon_token("Shower") is None
    # Unknown tokens fall back safely without config changes.
    assert config.resolve_room_token("WesternRoom") == config.room_index["other_room"]
    assert config.resolve_icon_token("OtherFixture") is None
