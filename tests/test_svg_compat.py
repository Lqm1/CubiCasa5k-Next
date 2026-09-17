"""SVG compatibility: group-structured CubiCasa5k annotations."""

from __future__ import annotations

from pathlib import Path

import pytest

from cubicasa5k_next.data.heatmaps import build_heatmap_targets_from_vector
from cubicasa5k_next.data.rasterizer import rasterize_vector
from cubicasa5k_next.data.svg_parser import parse_floorplan_svg
from cubicasa5k_next.data.transforms import apply_letterbox_to_points, letterbox_geometry
from cubicasa5k_next.labels_config import default_label_config

DATA_ROOT = Path(r"D:\Downloads\cubicasa5k-pairs")
SAMPLE = DATA_ROOT / "train" / "annotations" / "high_quality_10004.svg"


def _target_space(vector, image_hw=(1050, 1432), size=256):
    geom = letterbox_geometry(image_hw[0], image_hw[1], size)
    mapping: dict[int, list[tuple[float, float]]] = {}
    for shape in (*vector.rooms, *vector.walls, *vector.openings, *vector.icons):
        mapping[id(shape)] = apply_letterbox_to_points(list(shape.points), geom)
    return mapping


def test_wall_door_window_dispatch():
    if not SAMPLE.exists():
        pytest.skip("CubiCasa5k dataset not available")
    config = default_label_config()
    vector = parse_floorplan_svg(SAMPLE, config)
    assert len(vector.walls) > 0, "Wall/Railing groups must be detected"
    assert len(vector.rooms) > 0
    assert len(vector.doors) + len(vector.windows) > 0
    # Room classes must disperse instead of collapsing to other_room.
    room_indices = {room.label_index for room in vector.rooms}
    assert len(room_indices) > 1
    assert config.room_index["bedroom"] in room_indices
    # Doors/windows keep distinct icon labels.
    door_labels = {shape.label_index for shape in vector.doors}
    window_labels = {shape.label_index for shape in vector.windows}
    assert door_labels == {config.icon_index["door"]}
    assert window_labels == {config.icon_index["window"]}


def test_dimension_marks_are_ignored():
    if not SAMPLE.exists():
        pytest.skip("CubiCasa5k dataset not available")
    vector = parse_floorplan_svg(SAMPLE, default_label_config())
    # Old parser produced 100+ polygons on this file (mostly DimensionMark
    # triangles near the origin). Group parsing must stay an order of
    # magnitude smaller and contain no origin micro-triangles.
    total = (
        len(vector.rooms)
        + len(vector.walls)
        + len(vector.doors)
        + len(vector.windows)
        + len(vector.icons)
    )
    assert total < 80
    for shape in (*vector.rooms, *vector.walls, *vector.openings, *vector.icons):
        xs = [p[0] for p in shape.points]
        assert not (max(xs) < 20 and min(xs) > -10 and len(shape.points) == 3)


def test_furniture_transform_applied():
    if not SAMPLE.exists():
        pytest.skip("CubiCasa5k dataset not available")
    vector = parse_floorplan_svg(SAMPLE, default_label_config())
    assert len(vector.icons) > 0
    # Local-coords furniture (0..60) must be mapped into the floorplan area.
    assert any(min(p[0] for p in icon.points) > 100 for icon in vector.icons)
    # Ignored classes (cabinets/showers) must not appear.
    assert all(len(icon.points) == 4 for icon in vector.icons)


def test_rasterize_and_heatmaps_real_sample():
    if not SAMPLE.exists():
        pytest.skip("CubiCasa5k dataset not available")
    config = default_label_config()
    vector = parse_floorplan_svg(SAMPLE, config)
    mapping = _target_space(vector)
    room_mask, icon_mask = rasterize_vector(
        vector,
        target_size=(256, 256),
        num_room_classes=config.num_rooms,
        num_icon_classes=config.num_icons,
        geom_points=mapping,
    )
    assert room_mask.shape == (256, 256)
    assert icon_mask.shape == (256, 256)
    # Wall channel and an opening channel must be non-empty.
    assert (room_mask == config.room_index["wall"]).any()
    assert (
        (icon_mask == config.icon_index["door"]) | (icon_mask == config.icon_index["window"])
    ).any()
    # Room mask must contain a real room beyond background/other.
    assert (room_mask == config.room_index["bedroom"]).any()
    heatmaps = build_heatmap_targets_from_vector(
        vector, target_size=(256, 256), geom_points=mapping, radius=3
    )
    assert heatmaps.shape == (21, 256, 256)
    assert heatmaps[:13].max() > 0.3, "wall junctions must fire"
    assert heatmaps[13:17].max() > 0.3, "opening endpoints must fire"
    assert heatmaps[17:].max() > 0.3, "icon corners must fire"
