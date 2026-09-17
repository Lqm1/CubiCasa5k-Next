from cubicasa5k_next.data.heatmaps import build_heatmap_targets_from_vector
from cubicasa5k_next.data.rasterizer import rasterize_vector
from cubicasa5k_next.data.svg_parser import IconShape, OpeningShape, RoomShape
from cubicasa5k_next.labels_config import default_label_config


def _vector() -> object:
    config = default_label_config()
    from cubicasa5k_next.data.svg_parser import FloorplanVector

    return FloorplanVector(
        rooms=(
            RoomShape(
                points=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
                svg_token="LivingRoom",
                label_index=config.room_index["living_room"],
            ),
        ),
        walls=(),
        doors=(
            OpeningShape(
                points=((60.0, 60.0), (80.0, 60.0), (80.0, 80.0), (60.0, 80.0)),
                is_door=True,
                label_index=config.icon_index["door"],
            ),
        ),
        windows=(),
        icons=(
            IconShape(
                points=((60.0, 10.0), (80.0, 10.0), (80.0, 30.0), (60.0, 30.0)),
                svg_token="Sink",
                label_index=config.icon_index["sink"],
            ),
        ),
    )


def test_rasterize_and_heatmaps() -> None:
    config = default_label_config()
    vector = _vector()
    assert isinstance(vector.rooms[0], RoomShape)
    assert isinstance(vector.doors[0], OpeningShape)
    assert isinstance(vector.icons[0], IconShape)
    assert isinstance(vector.walls, tuple)
    room_mask, icon_mask = rasterize_vector(
        vector,
        target_size=(64, 64),
        num_room_classes=config.num_rooms,
        num_icon_classes=config.num_icons,
        geom_points=None,
    )
    assert room_mask.shape == (64, 64)
    assert icon_mask.shape == (64, 64)
    assert (room_mask == config.room_index["living_room"]).any()
    assert (icon_mask == config.icon_index["door"]).any()
    heatmaps = build_heatmap_targets_from_vector(
        vector, target_size=(64, 64), geom_points=None, radius=3
    )
    assert heatmaps.shape == (21, 64, 64)
    assert heatmaps.max() > 0.5
    assert heatmaps.min() >= 0.0
