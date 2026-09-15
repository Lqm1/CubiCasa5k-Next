"""Door/window line recovery from opening endpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from cubicasa5k_next.postprocess.junctions import DetectedJunction
from cubicasa5k_next.postprocess.walls import WallSegment


@dataclass(frozen=True)
class OpeningSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    label_index: int


def recover_opening_segments(
    junctions: list[DetectedJunction],
    icon_labels: np.ndarray,
    wall_segments: list[WallSegment],
    tolerance: int = 10,
    window_class_index: int = 1,
    door_class_index: int = 2,
) -> list[OpeningSegment]:
    endpoints = [j for j in junctions if j.group == "opening"]
    segments: list[OpeningSegment] = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            first = endpoints[i]
            second = endpoints[j]
            horizontal = abs(first.y - second.y) <= tolerance
            vertical = abs(first.x - second.x) <= tolerance
            if horizontal == vertical:
                continue
            if not _lies_on_wall(first, second, wall_segments, tolerance):
                continue
            label = _vote_opening_label(
                first, second, icon_labels, window_class_index, door_class_index
            )
            segments.append(
                OpeningSegment(x1=first.x, y1=first.y, x2=second.x, y2=second.y, label_index=label)
            )
    return segments


def _lies_on_wall(
    first: DetectedJunction,
    second: DetectedJunction,
    wall_segments: list[WallSegment],
    tolerance: int,
) -> bool:
    # Openings must sit on (or very near) a wall primitive.
    for wall in wall_segments:
        if abs(first.y - second.y) <= tolerance:  # horizontal opening
            if abs(wall.y1 - wall.y2) > tolerance:
                continue
            if abs(wall.y1 - first.y) > tolerance:
                continue
            wall_x0, wall_x1 = sorted([wall.x1, wall.x2])
            opening_x0, opening_x1 = sorted([first.x, second.x])
            if opening_x0 >= wall_x0 - tolerance and opening_x1 <= wall_x1 + tolerance:
                return True
        else:  # vertical opening
            if abs(wall.x1 - wall.x2) > tolerance:
                continue
            if abs(wall.x1 - first.x) > tolerance:
                continue
            wall_y0, wall_y1 = sorted([wall.y1, wall.y2])
            opening_y0, opening_y1 = sorted([first.y, second.y])
            if opening_y0 >= wall_y0 - tolerance and opening_y1 <= wall_y1 + tolerance:
                return True
    # Fall back to wall segmentation proximity when no wall primitive matched.
    return len(wall_segments) == 0


def _vote_opening_label(
    first: DetectedJunction,
    second: DetectedJunction,
    icon_labels: np.ndarray,
    window_class_index: int,
    door_class_index: int,
) -> int:
    height, width = icon_labels.shape
    samples = 10
    xs = np.linspace(first.x, second.x, samples).astype(int)
    ys = np.linspace(first.y, second.y, samples).astype(int)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    values = icon_labels[ys, xs].tolist()
    votes = Counter(values)
    window_votes = votes.get(window_class_index, 0)
    door_votes = votes.get(door_class_index, 0)
    if window_votes == 0 and door_votes == 0:
        return door_class_index
    return window_class_index if window_votes >= door_votes else door_class_index
