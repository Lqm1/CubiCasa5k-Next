"""Wall skeleton inference from wall junctions and wall segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubicasa5k_next.postprocess.junctions import DetectedJunction


@dataclass(frozen=True)
class WallSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int


def _are_compatible(first: DetectedJunction, second: DetectedJunction, tolerance: int) -> bool:
    horizontal = abs(first.y - second.y) <= tolerance
    vertical = abs(first.x - second.x) <= tolerance
    return horizontal != vertical  # exactly one axis aligned


def _wall_support(
    first: DetectedJunction,
    second: DetectedJunction,
    wall_probability: np.ndarray,
    samples: int = 20,
) -> float:
    xs = np.linspace(first.x, second.x, samples).astype(int)
    ys = np.linspace(first.y, second.y, samples).astype(int)
    height, width = wall_probability.shape
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    return float(wall_probability[ys, xs].mean())


def infer_wall_segments(
    junctions: list[DetectedJunction],
    room_labels: np.ndarray,
    wall_class_index: int = 2,
    tolerance: int = 10,
    support_threshold: float = 0.3,
) -> list[WallSegment]:
    """Pairwise-connect compatible wall junctions, prune by wall segmentation."""
    wall_points = [j for j in junctions if j.group == "wall"]
    wall_probability = (room_labels == wall_class_index).astype(np.float32)
    height, width = room_labels.shape
    segments: list[WallSegment] = []
    for i in range(len(wall_points)):
        for j in range(i + 1, len(wall_points)):
            first = wall_points[i]
            second = wall_points[j]
            if not _are_compatible(first, second, tolerance):
                continue
            support = _wall_support(first, second, wall_probability)
            if support < support_threshold:
                continue
            thickness = _estimate_thickness(first, second, wall_probability, width, height)
            segments.append(
                WallSegment(x1=first.x, y1=first.y, x2=second.x, y2=second.y, thickness=thickness)
            )
    return _remove_duplicates(segments)


def _estimate_thickness(
    first: DetectedJunction,
    second: DetectedJunction,
    wall_probability: np.ndarray,
    width: int,
    height: int,
    max_half_width: int = 8,
) -> int:
    horizontal = abs(first.y - second.y) <= abs(first.x - second.x)
    mid_x = (first.x + second.x) // 2
    mid_y = (first.y + second.y) // 2
    thickness = 1
    for offset in range(1, max_half_width + 1):
        if horizontal:
            top = mid_y - offset
            bottom = mid_y + offset
            if top < 0 or bottom >= height:
                break
            if wall_probability[top, mid_x] < 0.5 and wall_probability[bottom, mid_x] < 0.5:
                break
        else:
            left = mid_x - offset
            right = mid_x + offset
            if left < 0 or right >= width:
                break
            if wall_probability[mid_y, left] < 0.5 and wall_probability[mid_y, right] < 0.5:
                break
        thickness = offset * 2 + 1
    return thickness


def _remove_duplicates(segments: list[WallSegment]) -> list[WallSegment]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[WallSegment] = []
    for segment in segments:
        endpoints = tuple(sorted([(segment.x1, segment.y1), (segment.x2, segment.y2)]))
        key = (endpoints[0][0], endpoints[0][1], endpoints[1][0], endpoints[1][1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(segment)
    return unique
