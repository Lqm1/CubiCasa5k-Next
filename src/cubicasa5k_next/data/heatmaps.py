"""Gaussian heatmap targets with CubiCasa5k-compatible semantics.

Channel layout (21)::

    0-12  wall junctions from the merged wall centerline graph
          (I/L/T/X groups with orientation subtypes)
    13-16 opening endpoints (door/window quads: left/right/up/down)
    17-20 icon corners (furniture quads: TL/TR/BR/BL by position)

Wall junctions are derived from wall *centerlines*, not polygon vertices:
each wall quad yields a direction (H/V), end points (short-edge midpoints)
and width; colinear walls are merged; pairwise H/V interactions produce
I/L/T/X junctions which are then proximity-merged with the standard
13-entry lookup. This mirrors the behaviour of the reference loaders with
an independent implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from cubicasa5k_next.data.svg_parser import FloorplanVector
from cubicasa5k_next.labels_config import (
    ICON_CORNER_CHANNEL_RANGE,
    NUM_HEATMAP_CHANNELS,
    NUM_ICON_CORNER_TYPES,
    NUM_OPENING_ENDPOINT_TYPES,
    NUM_WALL_JUNCTION_TYPES,
    OPENING_CHANNEL_RANGE,
)

# Proximity-merge lookup: [first_id][second_id] -> merged_id (0..12) or None.
# Row/column ids are (group-1)*4 + (type-1); ids encode I/L/T/X groups.
_MERGE_LOOKUP: tuple[tuple[int | None, ...], ...] = (
    (0, 7, None, 6, 9, 11, 6, 7, 8, 9, 12, 11, 12),
    (7, 1, 4, None, 4, 10, 8, 7, 8, 9, 10, 12, 12),
    (None, 4, 2, 5, 4, 5, 11, 9, 12, 9, 10, 11, 12),
    (6, None, 5, 3, 10, 5, 6, 8, 8, 12, 10, 11, 12),
    (9, 4, 4, 10, 4, 10, 12, 9, 12, 9, 10, 12, 12),
    (11, 10, 5, 5, 10, 5, 11, 12, 12, 12, 10, 11, 12),
    (6, 8, 11, 6, 12, 11, 6, 8, 8, 12, 12, 11, 12),
    (7, 7, 9, 8, 9, 12, 8, 7, 8, 9, 12, 12, 12),
    (8, 8, 12, 8, 12, 12, 8, 8, 8, 12, 12, 12, 12),
    (9, 9, 9, 12, 9, 12, 12, 9, 12, 9, 12, 12, 12),
    (12, 10, 10, 10, 10, 10, 12, 12, 12, 12, 10, 12, 12),
    (11, 12, 11, 11, 12, 11, 11, 12, 12, 12, 12, 11, 12),
    (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
)


@dataclass
class _WallLine:
    """Centerline of one wall quad (internal to heatmap construction)."""

    end_points: np.ndarray  # (2, 2) float
    direction: str  # "H" | "V"
    max_width: float
    min_width: float
    cross_min: list[float]
    cross_max: list[float]


def _short_edge_midpoints(quad: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.array(quad, dtype=np.float64)
    assert len(pts) == 4
    best: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        dist = float(np.linalg.norm(a - b))
        best.append((dist, a, b))
    best.sort(key=lambda item: item[0])
    (_, a1, b1), (_, a2, b2) = best[0], best[1]
    mid1 = (a1 + b1) / 2.0
    mid2 = (a2 + b2) / 2.0
    width = float((best[0][0] + best[1][0]) / 2.0)
    return mid1, mid2, width


def _wall_line_from_quad(quad: list[tuple[float, float]]) -> _WallLine | None:
    if len(quad) != 4:
        boxed = _bbox_quad_fallback(quad)
        if boxed is None:
            return None
        quad = boxed
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if max(xs) - min(xs) < 4 or max(ys) - min(ys) < 4:
        return None  # tiny wall, ignored like the reference loader
    direction = "H" if (max(xs) - min(xs)) > (max(ys) - min(ys)) else "V"
    mid1, mid2, width = _short_edge_midpoints(quad)
    if direction == "V":
        ys_sorted = sorted([float(mid1[1]), float(mid2[1])])
        x_mean = float((mid1[0] + mid2[0]) / 2.0)
        ends = np.array([[x_mean, ys_sorted[0]], [x_mean, ys_sorted[1]]])
        cross = sorted([float(mid1[0]), float(mid2[0])])
    else:
        xs_sorted = sorted([float(mid1[0]), float(mid2[0])])
        y_mean = float((mid1[1] + mid2[1]) / 2.0)
        ends = np.array([[xs_sorted[0], y_mean], [xs_sorted[1], y_mean]])
        cross = sorted([float(mid1[1]), float(mid2[1])])
    return _WallLine(
        end_points=ends,
        direction=direction,
        max_width=width,
        min_width=width,
        cross_min=cross,
        cross_max=cross,
    )


def _bbox_quad_fallback(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x < 1e-6 or max_y - min_y < 1e-6:
        return None
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def _overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _merge_pair(base: _WallLine, other: _WallLine) -> _WallLine | None:
    if base.direction != other.direction:
        return None
    if abs(base.max_width - other.max_width) > other.max_width:
        return None
    max_dist = max(base.max_width, other.max_width)
    dist_head_to_tail = float(np.linalg.norm(base.end_points[0] - other.end_points[1]))
    dist_tail_to_head = float(np.linalg.norm(base.end_points[1] - other.end_points[0]))
    if dist_head_to_tail <= max_dist * 1.5:
        if _overlap(base.cross_min, other.cross_min) <= 0:
            return None
        merged_ends = np.array([other.end_points[0], base.end_points[1]])
    elif dist_tail_to_head <= max_dist * 1.5:
        if _overlap(base.cross_min, other.cross_min) <= 0:
            return None
        merged_ends = np.array([base.end_points[0], other.end_points[1]])
    else:
        return None
    cross_min = [
        max(base.cross_min[0], other.cross_min[0]),
        min(base.cross_min[1], other.cross_min[1]),
    ]
    cross_max = (
        base.cross_max
        if (base.cross_max[1] - base.cross_max[0]) > (other.cross_max[1] - other.cross_max[0])
        else other.cross_max
    )
    return _WallLine(
        end_points=merged_ends,
        direction=base.direction,
        max_width=max(base.max_width, other.max_width),
        min_width=min(base.min_width, other.min_width),
        cross_min=cross_min,
        cross_max=cross_max,
    )


def _merge_walls(lines: list[_WallLine]) -> list[_WallLine]:
    remaining = list(lines)
    merged: list[_WallLine] = []
    while remaining:
        base = remaining.pop(0)
        progressed = True
        while progressed:
            progressed = False
            for i, other in enumerate(remaining):
                candidate = _merge_pair(base, other)
                if candidate is not None:
                    base = candidate
                    remaining.pop(i)
                    progressed = True
                    break
        # Recenter the cross coordinate on the narrow overlap mean.
        mean_cross = float(np.mean(base.cross_min))
        if base.direction == "V":
            base.end_points[0][0] = mean_cross
            base.end_points[1][0] = mean_cross
        else:
            base.end_points[0][1] = mean_cross
            base.end_points[1][1] = mean_cross
        merged.append(base)
    return merged


def _line_dim(line: np.ndarray, tolerance: float = 1.0) -> int:
    dx = abs(float(line[0][0] - line[1][0]))
    dy = abs(float(line[0][1] - line[1][1]))
    if dx > dy and dy <= tolerance:
        return 0
    if dy > dx and dx <= tolerance:
        return 1
    return -1


def _junctions_from_lines(
    lines: list[_WallLine], avg_width: float
) -> list[tuple[float, float, int]]:
    """Return ``(x, y, junction_id)`` with ``junction_id`` in 0..12."""
    endpoints = [line.end_points for line in lines]
    used: list[list[bool]] = [[False, False] for _ in lines]
    points: list[tuple[list[float], int, int]] = []  # (xy, group, type)

    for i, line_1 in enumerate(endpoints):
        dim_1 = _line_dim(line_1)
        if dim_1 < 0:
            continue
        fixed_1 = float((line_1[0][1 - dim_1] + line_1[1][1 - dim_1]) / 2.0)
        for j, line_2 in enumerate(endpoints):
            if j <= i:
                continue
            dim_2 = _line_dim(line_2)
            if dim_1 + dim_2 != 1:
                continue
            fixed_2 = float((line_2[0][1 - dim_2] + line_2[1][1 - dim_2]) / 2.0)
            gap = max(lines[i].max_width, lines[j].max_width)
            nearest, min_dist = _nearest_pair(line_1, line_2, gap, fixed_1, fixed_2, dim_1, dim_2)
            if min_dist > gap:
                # Crossing of two interiors -> X junction.
                if (
                    min(line_1[0][dim_1], line_1[1][dim_1]) < fixed_2
                    and max(line_1[0][dim_1], line_1[1][dim_1]) > fixed_2
                    and min(line_2[0][dim_2], line_2[1][dim_2]) < fixed_1
                    and max(line_2[0][dim_2], line_2[1][dim_2]) > fixed_1
                ):
                    point = [0.0, 0.0]
                    point[dim_1] = fixed_2
                    point[dim_2] = fixed_1
                    points.append((point, 4, 1))
                continue
            idx_1, idx_2 = nearest
            if idx_1 >= 0 and idx_2 >= 0:
                point = [0.0, 0.0]
                point[dim_1] = fixed_2
                point[dim_2] = fixed_1
                side = [0.0, 0.0]
                side[dim_1] = float(line_1[1 - idx_1][dim_1] - fixed_2)
                side[dim_2] = float(line_2[1 - idx_2][dim_2] - fixed_1)
                if side[0] < 0 and side[1] < 0:
                    points.append((point, 2, 1))
                elif side[0] > 0 and side[1] < 0:
                    points.append((point, 2, 2))
                elif side[0] > 0 and side[1] > 0:
                    points.append((point, 2, 3))
                else:
                    points.append((point, 2, 4))
                used[i][idx_1] = True
                used[j][idx_2] = True
            elif (idx_1 >= 0) != (idx_2 >= 0):
                if idx_1 >= 0:
                    dim, endpoint_idx, fixed, value = (
                        dim_1,
                        idx_1,
                        fixed_2,
                        line_1[idx_1][1 - dim_1],
                    )
                    used[i][idx_1] = True
                else:
                    dim, endpoint_idx, fixed, value = (
                        dim_2,
                        idx_2,
                        fixed_1,
                        line_2[idx_2][1 - dim_2],
                    )
                    used[j][idx_2] = True
                point = [0.0, 0.0]
                point[dim] = fixed
                point[1 - dim] = float(value)
                if endpoint_idx == 0:
                    points.append((point, 3, 4 if dim == 0 else 1))
                else:
                    points.append((point, 3, 2 if dim == 0 else 3))

    for line_idx, mask in enumerate(used):
        dim = _line_dim(endpoints[line_idx])
        for end_idx in range(2):
            if mask[end_idx]:
                continue
            point = [float(endpoints[line_idx][end_idx][0]), float(endpoints[line_idx][end_idx][1])]
            if end_idx == 0:
                points.append((point, 1, 4 if dim == 0 else 1))
            else:
                points.append((point, 1, 2 if dim == 0 else 3))

    merged = _merge_close_junctions(points, avg_width)
    return [(float(x), float(y), _point_id_to_channel(g, t)) for (x, y), g, t in merged]


def _nearest_pair(
    line_1: np.ndarray,
    line_2: np.ndarray,
    gap: float,
    fixed_1: float,
    fixed_2: float,
    dim_1: int,
    dim_2: int,
) -> tuple[tuple[int, int], float]:
    nearest = (0, 0)
    min_dist: float | None = None
    for a in range(2):
        for b in range(2):
            dist = float(np.linalg.norm(line_1[a] - line_2[b]))
            if min_dist is None or dist < min_dist:
                nearest = (a, b)
                min_dist = dist
    assert min_dist is not None
    if min_dist > gap:
        for index in range(2):
            dist = abs(float(line_1[index][dim_1]) - fixed_2)
            if dist < min_dist:
                nearest = (index, -1)
                min_dist = dist
        for index in range(2):
            dist = abs(float(line_2[index][dim_2]) - fixed_1)
            if dist < min_dist:
                nearest = (-1, index)
                min_dist = dist
    return nearest, min_dist


def _point_id_to_channel(group: int, kind: int) -> int:
    return (group - 1) * 4 + kind - 1


def _merge_close_junctions(
    points: list[tuple[list[float], int, int]], width: float
) -> list[tuple[tuple[float, float], int, int]]:
    def dist(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    merged_flags = [False] * len(points)
    output: list[tuple[tuple[float, float], int, int]] = []
    for i, first in enumerate(points):
        if merged_flags[i]:
            continue
        pool = [first]
        for j, second in enumerate(points):
            if j != i and not merged_flags[j] and dist(first[0], second[0]) < width:
                merged_flags[j] = True
                pool.append(second)
        if len(pool) == 1:
            merged_flags[i] = True
            output.append(((float(first[0][0]), float(first[0][1])), first[1], first[2]))
            continue
        current = pool[0]
        for other in pool[1:]:
            key = (current[1] - 1) * 4 + (current[2] - 1)
            key_other = (other[1] - 1) * 4 + (other[2] - 1)
            merged_id = _MERGE_LOOKUP[key][key_other]
            if merged_id is None:
                continue
            current = (current[0], merged_id // 4 + 1, merged_id % 4 + 1)
        output.append(((float(current[0][0]), float(current[0][1])), current[1], current[2]))
        merged_flags[i] = True
    return output


def _opening_endpoints(
    quad: list[tuple[float, float]],
) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width >= height:  # horizontal opening: left/right
        ordered = sorted(quad, key=lambda p: p[0])
        left = np.mean(ordered[:2], axis=0)
        right = np.mean(ordered[2:], axis=0)
        return ((float(left[0]), float(left[1]), 0), (float(right[0]), float(right[1]), 1))
    ordered = sorted(quad, key=lambda p: p[1])
    up = np.mean(ordered[:2], axis=0)
    down = np.mean(ordered[2:], axis=0)
    return ((float(up[0]), float(up[1]), 2), (float(down[0]), float(down[1]), 3))


def _icon_corners(
    quad: list[tuple[float, float]],
) -> list[tuple[float, float, int]]:
    """Return ``(x, y, corner_id)`` with ids TL=0, TR=1, BR=2, BL=3."""
    remaining = [tuple(map(float, p)) for p in quad]
    sums = [x + y for x, y in remaining]
    upper_left = remaining.pop(int(np.argmin(sums)))
    sums = [x + y for x, y in remaining]
    bottom_right = remaining.pop(int(np.argmax(sums)))
    # Of the last two, the higher one is top-right.
    if remaining[0][1] <= remaining[1][1]:
        upper_right, bottom_left = remaining[0], remaining[1]
    else:
        upper_right, bottom_left = remaining[1], remaining[0]
    return [
        (upper_left[0], upper_left[1], 0),
        (upper_right[0], upper_right[1], 1),
        (bottom_right[0], bottom_right[1], 2),
        (bottom_left[0], bottom_left[1], 3),
    ]


@lru_cache(maxsize=32)
def _gaussian_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    sigma = radius / 2.0 + 1e-6
    for dy in range(size):
        for dx in range(size):
            dist_sq = (dx - radius) ** 2 + (dy - radius) ** 2
            kernel[dy, dx] = math.exp(-dist_sq / (2.0 * sigma**2))
    kernel.flags.writeable = False
    return kernel


def _splat_gaussian(canvas: np.ndarray, center_x: float, center_y: float, radius: int) -> None:
    height, width = canvas.shape
    cx = int(round(center_x))
    cy = int(round(center_y))
    kernel = _gaussian_kernel(radius)
    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, width)
    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, height)
    if x0 >= x1 or y0 >= y1:
        return  # Center too far off-canvas (e.g. wall lines meeting outside).
    kx0 = x0 - (cx - radius)
    kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (cy - radius)
    ky1 = ky0 + (y1 - y0)
    kx0 = max(kx0, 0)
    ky0 = max(ky0, 0)
    kx1 = min(kx1, kernel.shape[1])
    ky1 = min(ky1, kernel.shape[0])
    w = min(x1 - x0, kx1 - kx0)
    h = min(y1 - y0, ky1 - ky0)
    if w <= 0 or h <= 0:
        return
    canvas[y0 : y0 + h, x0 : x0 + w] = np.maximum(
        canvas[y0 : y0 + h, x0 : x0 + w], kernel[ky0 : ky0 + h, kx0 : kx0 + w]
    )


def build_heatmap_targets_from_vector(
    vector: FloorplanVector,
    target_size: tuple[int, int],
    geom_points: dict[int, list[tuple[float, float]]] | None = None,
    radius: int = 5,
) -> np.ndarray:
    """Build ``(21, H, W)`` heatmaps from target-space vector shapes."""
    height, width = target_size
    heatmaps = np.zeros((NUM_HEATMAP_CHANNELS, height, width), dtype=np.float32)

    def points_of(shape) -> list[tuple[float, float]] | None:
        if geom_points is not None:
            mapped = geom_points.get(id(shape))
            if mapped is not None:
                return mapped
        return list(shape.points)

    wall_quads: list[list[tuple[float, float]]] = []
    for wall in vector.walls:
        pts = points_of(wall)
        if pts is not None and len(pts) >= 3:
            wall_quads.append(pts if len(pts) == 4 else _bbox_quad_fallback(pts) or pts)

    lines: list[_WallLine] = []
    for quad in wall_quads:
        line = _wall_line_from_quad(quad)
        if line is not None:
            lines.append(line)
    if lines:
        avg_width = float(np.mean([line.max_width for line in lines]))
        merged = _merge_walls(lines)
        for x, y, channel in _junctions_from_lines(merged, avg_width):
            if 0 <= channel < NUM_WALL_JUNCTION_TYPES:
                _splat_gaussian(heatmaps[channel], x, y, radius)

    for opening in vector.openings:
        pts = points_of(opening)
        if pts is None or len(pts) < 3:
            continue
        opening_quad = pts if len(pts) == 4 else _bbox_quad_fallback(pts)
        if opening_quad is None:
            continue
        for x, y, endpoint in _opening_endpoints(opening_quad):
            channel = OPENING_CHANNEL_RANGE[0] + (endpoint % NUM_OPENING_ENDPOINT_TYPES)
            _splat_gaussian(heatmaps[channel], x, y, radius)

    for icon in vector.icons:
        pts = points_of(icon)
        if pts is None or len(pts) < 3:
            continue
        icon_quad = pts if len(pts) == 4 else _bbox_quad_fallback(pts)
        if icon_quad is None:
            continue
        for x, y, corner in _icon_corners(icon_quad):
            channel = ICON_CORNER_CHANNEL_RANGE[0] + (corner % NUM_ICON_CORNER_TYPES)
            _splat_gaussian(heatmaps[channel], x, y, radius)
    return heatmaps
