"""Icon box recovery from icon-corner junctions and icon segmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from cubicasa5k_next.labels import ICON_CORNER_CHANNEL_RANGE
from cubicasa5k_next.postprocess.junctions import DetectedJunction


@dataclass(frozen=True)
class IconBox:
    x0: int
    y0: int
    x1: int
    y1: int
    label_index: int
    confidence: float


def _corner_kind(channel: int) -> int:
    return channel - ICON_CORNER_CHANNEL_RANGE[0]


def recover_icon_boxes(
    junctions: list[DetectedJunction],
    icon_labels: np.ndarray,
    tolerance: int = 10,
) -> list[IconBox]:
    icon_points = [j for j in junctions if j.group == "icon"]
    by_kind: dict[int, list[DetectedJunction]] = {0: [], 1: [], 2: [], 3: []}
    for point in icon_points:
        by_kind[_corner_kind(point.channel) % 4].append(point)

    boxes: list[IconBox] = []
    for top_left in by_kind[0]:
        for top_right in by_kind[1]:
            if abs(top_left.y - top_right.y) > tolerance:
                continue
            for bottom_right in by_kind[2]:
                if abs(top_right.x - bottom_right.x) > tolerance:
                    continue
                for bottom_left in by_kind[3]:
                    xs = [top_left.x, top_right.x, bottom_right.x, bottom_left.x]
                    ys = [top_left.y, top_right.y, bottom_right.y, bottom_left.y]
                    if max(xs) - min(xs) < 4 or max(ys) - min(ys) < 4:
                        continue
                    if abs(top_left.y - top_right.y) > tolerance:
                        continue
                    if abs(bottom_left.y - bottom_right.y) > tolerance:
                        continue
                    if abs(top_left.x - bottom_left.x) > tolerance:
                        continue
                    if abs(top_right.x - bottom_right.x) > tolerance:
                        continue
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    patch = icon_labels[y0:y1, x0:x1]
                    if patch.size == 0:
                        continue
                    votes = Counter(patch.reshape(-1).tolist())
                    if 0 in votes and len(votes) > 1:
                        del votes[0]
                    if not votes:
                        continue
                    label, count = votes.most_common(1)[0]
                    if label == 0:
                        continue
                    confidence = count / patch.size
                    boxes.append(
                        IconBox(
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            label_index=int(label),
                            confidence=float(confidence),
                        )
                    )
    return _suppress_overlaps(boxes)


def _suppress_overlaps(boxes: list[IconBox], iou_threshold: float = 0.5) -> list[IconBox]:
    ordered = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept: list[IconBox] = []
    for candidate in ordered:
        overlaps = False
        for existing in kept:
            if _iou(candidate, existing) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)
    return kept


def _iou(first: IconBox, second: IconBox) -> float:
    inter_x0 = max(first.x0, second.x0)
    inter_y0 = max(first.y0, second.y0)
    inter_x1 = min(first.x1, second.x1)
    inter_y1 = min(first.y1, second.y1)
    inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    if inter <= 0:
        return 0.0
    first_area = (first.x1 - first.x0) * (first.y1 - first.y0)
    second_area = (second.x1 - second.x0) * (second.y1 - second.y0)
    union = first_area + second_area - inter
    return inter / max(union, 1)
