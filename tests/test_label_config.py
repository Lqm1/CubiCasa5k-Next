"""Label-config extensibility: YAML round-trip and class extension."""

from __future__ import annotations

from cubicasa5k_next.config import ModelConfig
from cubicasa5k_next.labels_config import (
    LabelConfig,
    default_label_config,
    load_label_config,
    save_label_config,
)


def test_yaml_round_trip(tmp_path) -> None:
    config = default_label_config()
    path = tmp_path / "labels.yaml"
    save_label_config(config, path)
    reloaded = load_label_config(path)
    assert reloaded.rooms == config.rooms
    assert reloaded.icons == config.icons
    assert reloaded.room_svg_map == config.room_svg_map
    assert reloaded.icon_svg_map == config.icon_svg_map


def test_extending_classes_without_code_changes(tmp_path) -> None:
    base = default_label_config().to_dict()
    base["rooms"] = [*base["rooms"], "sauna_room"]
    base["room_svg_map"] = {**base["room_svg_map"], "SaunaRoom": "sauna_room"}
    base["icons"] = [*base["icons"], "stair_icon"]
    base["icon_svg_map"] = {**base["icon_svg_map"], "StairCase": "stair_icon"}
    extended = LabelConfig.from_dict(base)
    assert extended.num_rooms == 13
    assert extended.num_icons == 12
    assert extended.resolve_room_token("SaunaRoom") == 12
    assert extended.resolve_icon_token("StairCase") == 11
    # Model output dimensions follow the config.
    model = ModelConfig.for_label_counts(extended.num_rooms, extended.num_icons)
    assert model.num_outputs == 21 + 13 + 12
    # Unknown room tokens still fall back instead of raising.
    assert extended.resolve_room_token("NotARoom") == extended.room_index["other_room"]
    assert extended.resolve_icon_token("NotAnIcon") is None
