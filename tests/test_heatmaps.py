from cubicasa5k_next.data.heatmaps import build_heatmap_targets
from cubicasa5k_next.data.rasterizer import rasterize_polygons
from cubicasa5k_next.data.svg_parser import AnnotatedPolygon


def _square(label: str, kind: str, offset: float = 10.0) -> AnnotatedPolygon:
    return AnnotatedPolygon(
        points=(
            (offset, offset),
            (offset + 20, offset),
            (offset + 20, offset + 20),
            (offset, offset + 20),
        ),
        raw_label=label,
        kind=kind,
    )


def test_rasterize_and_heatmaps() -> None:
    polygons = [_square("living_room", "room"), _square("door", "opening", offset=60.0)]
    room_mask, icon_mask = rasterize_polygons(polygons, (100.0, 100.0), (64, 64))
    assert room_mask.shape == (64, 64)
    assert icon_mask.shape == (64, 64)
    heatmaps = build_heatmap_targets(polygons, (100.0, 100.0), (64, 64), radius=3)
    assert heatmaps.shape == (21, 64, 64)
    assert heatmaps.max() > 0.5
    assert heatmaps.min() >= 0.0
