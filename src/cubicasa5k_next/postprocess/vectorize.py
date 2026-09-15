"""End-to-end heuristic vectorization entry point."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cubicasa5k_next.postprocess.icons import IconBox, recover_icon_boxes
from cubicasa5k_next.postprocess.junctions import DetectedJunction, extract_junctions
from cubicasa5k_next.postprocess.openings import OpeningSegment, recover_opening_segments
from cubicasa5k_next.postprocess.rooms import RoomPolygon, recover_room_polygons
from cubicasa5k_next.postprocess.walls import WallSegment, infer_wall_segments


@dataclass
class VectorFloorplan:
    walls: list[WallSegment] = field(default_factory=list)
    rooms: list[RoomPolygon] = field(default_factory=list)
    icons: list[IconBox] = field(default_factory=list)
    openings: list[OpeningSegment] = field(default_factory=list)
    junctions: list[DetectedJunction] = field(default_factory=list)


def vectorize_prediction(
    room_logits: np.ndarray,  # (num_room, H, W)
    icon_logits: np.ndarray,  # (num_icon, H, W)
    heatmaps: np.ndarray,  # (21, H, W)
    junction_threshold: float = 0.4,
    alignment_tolerance: int = 10,
) -> VectorFloorplan:
    room_labels = np.argmax(room_logits, axis=0).astype(np.int64)
    icon_labels = np.argmax(icon_logits, axis=0).astype(np.int64)

    junctions = extract_junctions(heatmaps, threshold=junction_threshold)
    return vectorize_from_junctions(
        room_labels,
        icon_labels,
        junctions,
        alignment_tolerance=alignment_tolerance,
    )


def vectorize_from_junctions(
    room_labels: np.ndarray,
    icon_labels: np.ndarray,
    junctions: list[DetectedJunction],
    alignment_tolerance: int = 10,
) -> VectorFloorplan:
    """Vectorize from an existing junction list and label maps."""

    walls = infer_wall_segments(junctions, room_labels, tolerance=alignment_tolerance)
    rooms = recover_room_polygons(junctions, room_labels, tolerance=alignment_tolerance)
    icons = recover_icon_boxes(junctions, icon_labels, tolerance=alignment_tolerance)
    openings = recover_opening_segments(
        junctions, icon_labels, walls, tolerance=alignment_tolerance
    )

    return VectorFloorplan(
        walls=walls, rooms=rooms, icons=icons, openings=openings, junctions=junctions
    )
