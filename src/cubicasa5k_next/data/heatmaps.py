"""Gaussian heatmap targets for wall/opening/icon interest points."""

from __future__ import annotations

import math

import numpy as np

from cubicasa5k_next.data.svg_parser import AnnotatedPolygon
from cubicasa5k_next.labels import (
    ICON_CORNER_CHANNEL_RANGE,
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CORNER_TYPES,
    NUM_OPENING_ENDPOINT_TYPES,
    NUM_WALL_JUNCTION_TYPES,
    OPENING_CHANNEL_RANGE,
)


def _corner_orientation_index(polygon: AnnotatedPolygon, vertex: int) -> int:
    """Assign icon corner channels in TL/TR/BR/BL order by relative position."""
    xs = [p[0] for p in polygon.points]
    ys = [p[1] for p in polygon.points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x, y = polygon.points[vertex]
    left = abs(x - min_x) < abs(x - max_x)
    top = abs(y - min_y) < abs(y - max_y)
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _opening_endpoint_index(polygon: AnnotatedPolygon, vertex: int) -> int:
    xs = [p[0] for p in polygon.points]
    ys = [p[1] for p in polygon.points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    x, y = polygon.points[vertex]
    if width >= height:
        return 0 if abs(x - min(xs)) < abs(x - max(xs)) else 1
    return 2 if abs(y - min(ys)) < abs(y - max(ys)) else 3


def _wall_junction_index(polygon: AnnotatedPolygon, vertex: int) -> int:
    """Heuristic wall-junction typing from local edge geometry.

    Uses incident edge directions to distinguish I/L/T/X-like shapes and
    quantizes orientations into 13 bins (4+4+4+1). This is an original
    heuristic implementing the paper's taxonomy, not a port.
    """
    points = polygon.points
    count = len(points)
    previous = points[(vertex - 1) % count]
    current = points[vertex]
    following = points[(vertex + 1) % count]

    def direction(a: tuple[float, float], b: tuple[float, float]) -> tuple[int, int]:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if abs(dx) >= abs(dy):
            return (1 if dx >= 0 else -1, 0)
        return (0, 1 if dy >= 0 else -1)

    incoming = direction(previous, current)
    outgoing = direction(current, following)
    # Straight continuation -> I-shape (bins 0-3 by orientation).
    if incoming == (-outgoing[0], -outgoing[1]) or incoming == outgoing:
        if incoming[0] != 0:
            return 0  # horizontal I
        return 1  # vertical I
    # Right-angle turn -> L-shape (bins 4-7 by quadrant).
    incoming_code = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[incoming]
    outgoing_code = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[outgoing]
    quadrant = (incoming_code * 2 + outgoing_code) % 4
    # Distinguish L vs T by vertex density: isolated polygons only yield L here,
    # reserve T bins (8-11) for dense vertices and X bin (12) for crossings.
    if count >= 8:
        return 8 + quadrant
    return 4 + quadrant


def _splat_gaussian(canvas: np.ndarray, center_x: float, center_y: float, radius: int) -> None:
    height, width = canvas.shape
    cx = int(round(center_x))
    cy = int(round(center_y))
    size = radius * 2 + 1
    # Precompute a small Gaussian kernel.
    kernel = np.zeros((size, size), dtype=np.float32)
    for dy in range(size):
        for dx in range(size):
            dist_sq = (dx - radius) ** 2 + (dy - radius) ** 2
            kernel[dy, dx] = math.exp(-dist_sq / (2.0 * (radius / 2.0 + 1e-6) ** 2))
    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, width)
    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, height)
    kx0 = x0 - (cx - radius)
    kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (cy - radius)
    ky1 = ky0 + (y1 - y0)
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], kernel[ky0:ky1, kx0:kx1])


def build_heatmap_targets(
    polygons: list[AnnotatedPolygon],
    source_size: tuple[float, float],
    target_size: tuple[int, int],
    radius: int = 5,
) -> np.ndarray:
    """Build (21, H, W) heatmap targets in [0, 1]."""
    height, width = target_size
    heatmaps = np.zeros((NUM_HEATMAP_CHANNELS, height, width), dtype=np.float32)
    source_width, source_height = source_size
    scale_x = width / max(source_width, 1e-6)
    scale_y = height / max(source_height, 1e-6)

    for polygon in polygons:
        for vertex, (x, y) in enumerate(polygon.points):
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            if polygon.kind == "icon":
                corner = _corner_orientation_index(polygon, vertex) % NUM_ICON_CORNER_TYPES
                channel = ICON_CORNER_CHANNEL_RANGE[0] + corner
            elif polygon.kind == "opening":
                endpoint = _opening_endpoint_index(polygon, vertex) % NUM_OPENING_ENDPOINT_TYPES
                channel = OPENING_CHANNEL_RANGE[0] + endpoint
            else:
                # Rooms and walls both contribute wall-junction evidence.
                junction = _wall_junction_index(polygon, vertex) % NUM_WALL_JUNCTION_TYPES
                channel = junction
            _splat_gaussian(heatmaps[channel], scaled_x, scaled_y, radius)
    return heatmaps
