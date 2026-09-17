"""Convert CubiCasa5k vector shapes into dense room/icon masks.

Drawing order reproduces the original loader semantics: room areas first,
then walls/railings on top of the room layer, then doors/windows/icons on
the icon layer. Ignored icon classes never reach this module (the parser
drops them via ``LabelConfig``).
"""

from __future__ import annotations

import cv2
import numpy as np

from cubicasa5k_next.data.svg_parser import FloorplanVector


def _to_contour(points: list[tuple[float, float]]) -> np.ndarray:
    array = np.array(points, dtype=np.float32)
    return np.round(array).astype(np.int32).reshape(-1, 1, 2)


def rasterize_vector(
    vector: FloorplanVector,
    target_size: tuple[int, int],
    num_room_classes: int = 12,
    num_icon_classes: int = 11,
    geom_points: dict[int, list[tuple[float, float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize pre-transformed (target-space) vector shapes.

    Args:
        vector: parsed floorplan (only label indices are used here).
        target_size: ``(H, W)`` output resolution.
        geom_points: optional precomputed target-space point lists keyed by
            ``id(shape)``; when omitted, raw SVG coordinates are used
            directly (caller must ensure they are already in target space).
    """
    height, width = target_size
    room_canvas = np.zeros((height, width), dtype=np.uint8)
    icon_canvas = np.zeros((height, width), dtype=np.uint8)

    def points_of(shape) -> list[tuple[float, float]] | None:
        if geom_points is not None:
            mapped = geom_points.get(id(shape))
            if mapped is not None:
                return mapped
        return list(shape.points)

    for room in vector.rooms:
        pts = points_of(room)
        if pts is None or len(pts) < 3:
            continue
        idx = int(room.label_index)
        if 0 <= idx < num_room_classes:
            cv2.fillPoly(room_canvas, [_to_contour(pts)], idx)
    for wall in vector.walls:
        pts = points_of(wall)
        if pts is None or len(pts) < 3:
            continue
        idx = int(wall.label_index)
        if 0 <= idx < num_room_classes:
            cv2.fillPoly(room_canvas, [_to_contour(pts)], idx)
    for opening in vector.openings:
        pts = points_of(opening)
        if pts is None or len(pts) < 3:
            continue
        idx = int(opening.label_index)
        if 0 <= idx < num_icon_classes:
            cv2.fillPoly(icon_canvas, [_to_contour(pts)], idx)
    for icon in vector.icons:
        pts = points_of(icon)
        if pts is None or len(pts) < 3:
            continue
        idx = int(icon.label_index)
        if 0 <= idx < num_icon_classes:
            cv2.fillPoly(icon_canvas, [_to_contour(pts)], idx)
    return room_canvas.astype(np.int64), icon_canvas.astype(np.int64)
