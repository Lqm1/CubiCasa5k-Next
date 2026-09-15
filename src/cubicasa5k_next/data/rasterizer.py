"""Convert vector polygons into dense room/icon segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from cubicasa5k_next.data.svg_parser import AnnotatedPolygon
from cubicasa5k_next.labels import icon_name_to_index, room_name_to_index


def _scale_points(
    points: tuple[tuple[float, float], ...],
    source_size: tuple[float, float],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_height, target_width = target_size
    scale_x = target_width / max(source_width, 1e-6)
    scale_y = target_height / max(source_height, 1e-6)
    scaled = np.array([(x * scale_x, y * scale_y) for x, y in points], dtype=np.int32)
    return scaled.reshape(-1, 1, 2)


def rasterize_polygons(
    polygons: list[AnnotatedPolygon],
    source_size: tuple[float, float],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize room and icon layers.

    Returns:
        room_mask: (H, W) int64 with values in [0, 12).
        icon_mask: (H, W) int64 with values in [0, 11).
    """
    height, width = target_size
    room_canvas = np.zeros((height, width), dtype=np.uint8)
    icon_canvas = np.zeros((height, width), dtype=np.uint8)

    # Draw rooms first, then walls on top, then icons/openings on their own layer.
    def draw_order(polygon: AnnotatedPolygon) -> int:
        if polygon.kind == "room":
            return 0
        if polygon.kind == "wall":
            return 1
        return 2

    for polygon in sorted(polygons, key=draw_order):
        contour = _scale_points(polygon.points, source_size, target_size)
        label = polygon.raw_label
        if polygon.kind == "wall":
            cv2.fillPoly(room_canvas, [contour], 2)  # wall index
        elif polygon.kind == "room":
            room_index = room_name_to_index(label)
            cv2.fillPoly(room_canvas, [contour], int(room_index))
        elif polygon.kind in {"icon", "opening"}:
            icon_index = icon_name_to_index(label)
            cv2.fillPoly(icon_canvas, [contour], int(icon_index))
        else:
            # Unknown kinds default to room layer to avoid losing geometry.
            room_index = room_name_to_index(label)
            cv2.fillPoly(room_canvas, [contour], int(room_index))
    return room_canvas.astype(np.int64), icon_canvas.astype(np.int64)
