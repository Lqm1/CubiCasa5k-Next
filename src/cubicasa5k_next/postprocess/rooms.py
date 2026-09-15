"""Room polygon recovery from wall junctions and room segmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from cubicasa5k_next.postprocess.junctions import DetectedJunction


@dataclass(frozen=True)
class RoomPolygon:
    contour: np.ndarray  # (N, 2) int
    label_index: int
    confidence: float


def recover_room_polygons(
    junctions: list[DetectedJunction],
    room_labels: np.ndarray,
    tolerance: int = 10,
    min_area: int = 100,
) -> list[RoomPolygon]:
    """Grid the interior with junction triplets, vote labels, merge neighbors.

    This is a heuristic analogue of the paper's cell-gridding + voting +
    merging step, implemented without integer programming.
    """
    wall_points = [(j.x, j.y) for j in junctions if j.group == "wall"]
    if len(wall_points) < 4:
        return _fallback_from_segmentation(room_labels, min_area)

    xs = sorted({x for x, _ in wall_points})
    ys = sorted({y for _, y in wall_points})
    # Merge coordinates closer than tolerance to form grid lines.
    grid_x = _merge_coordinates(xs, tolerance, room_labels.shape[1])
    grid_y = _merge_coordinates(ys, tolerance, room_labels.shape[0])
    if len(grid_x) < 2 or len(grid_y) < 2:
        return _fallback_from_segmentation(room_labels, min_area)

    cells: list[tuple[int, int, int, int, int]] = []  # x0,y0,x1,y1,label
    for ix in range(len(grid_x) - 1):
        for iy in range(len(grid_y) - 1):
            x0, x1 = grid_x[ix], grid_x[ix + 1]
            y0, y1 = grid_y[iy], grid_y[iy + 1]
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            patch = room_labels[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            votes = Counter(patch.reshape(-1).tolist())
            # Ignore background/wall votes when a room label exists.
            for background_index in (0, 2):
                if len(votes) > 1 and background_index in votes:
                    del votes[background_index]
            if not votes:
                continue
            label, count = votes.most_common(1)[0]
            confidence = count / patch.size
            cells.append((x0, y0, x1, y1, int(label)))

    merged = _merge_cells(cells)
    polygons: list[RoomPolygon] = []
    for x0, y0, x1, y1, label in merged:
        area = (x1 - x0) * (y1 - y0)
        if area < min_area or label in (0, 2):
            continue
        contour = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        patch = room_labels[y0:y1, x0:x1]
        confidence = float((patch == label).mean()) if patch.size else 0.0
        polygons.append(RoomPolygon(contour=contour, label_index=label, confidence=confidence))
    if not polygons:
        return _fallback_from_segmentation(room_labels, min_area)
    return polygons


def _merge_coordinates(values: list[int], tolerance: int, limit: int) -> list[int]:
    merged: list[int] = []
    for value in sorted(values):
        value = min(max(value, 0), limit - 1)
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
    if merged[0] != 0:
        merged = [0, *merged]
    if merged[-1] != limit - 1:
        merged = [*merged, limit - 1]
    return merged


def _merge_cells(
    cells: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    # Horizontal merge for cells sharing label and row band.
    changed = True
    current = cells
    while changed:
        changed = False
        used = [False] * len(current)
        next_cells: list[tuple[int, int, int, int, int]] = []
        for i, first in enumerate(current):
            if used[i]:
                continue
            merged_cell = first
            used[i] = True
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                second = current[j]
                if (
                    merged_cell[4] == second[4]
                    and merged_cell[1] == second[1]
                    and merged_cell[3] == second[3]
                    and merged_cell[2] == second[0]
                ):
                    merged_cell = (
                        merged_cell[0],
                        merged_cell[1],
                        second[2],
                        merged_cell[3],
                        merged_cell[4],
                    )
                    used[j] = True
                    changed = True
            next_cells.append(merged_cell)
        current = next_cells
    return current


def _fallback_from_segmentation(room_labels: np.ndarray, min_area: int) -> list[RoomPolygon]:
    polygons: list[RoomPolygon] = []
    for label in sorted(int(v) for v in np.unique(room_labels).tolist()):
        if label in (0, 2):
            continue
        binary = ((room_labels == label).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            polygons.append(RoomPolygon(contour=approximated, label_index=label, confidence=0.5))
    return polygons
