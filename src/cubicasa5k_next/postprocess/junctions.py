"""Junction detection via thresholding + non-maximum suppression."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cubicasa5k_next.labels_config import (
    ICON_CORNER_CHANNEL_RANGE,
    NUM_HEATMAP_CHANNELS,
    OPENING_CHANNEL_RANGE,
)


@dataclass(frozen=True)
class DetectedJunction:
    x: int
    y: int
    channel: int
    score: float
    group: str  # "wall" | "opening" | "icon"


def _channel_group(channel: int) -> str:
    if OPENING_CHANNEL_RANGE[0] <= channel < OPENING_CHANNEL_RANGE[1]:
        return "opening"
    if ICON_CORNER_CHANNEL_RANGE[0] <= channel < ICON_CORNER_CHANNEL_RANGE[1]:
        return "icon"
    return "wall"


def extract_junctions(
    heatmaps: np.ndarray,
    threshold: float = 0.4,
    suppression_radius: int = 5,
) -> list[DetectedJunction]:
    """Extract junctions from (C, H, W) heatmaps.

    Applies per-channel thresholding, dilation-based NMS, and connected
    component centroiding. This mirrors the paper's threshold + NMS step
    with an original implementation.
    """
    if heatmaps.ndim != 3 or heatmaps.shape[0] != NUM_HEATMAP_CHANNELS:
        raise ValueError(f"Expected heatmaps with shape (21, H, W), got {heatmaps.shape}")
    _, height, width = heatmaps.shape
    detections: list[DetectedJunction] = []
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (suppression_radius * 2 + 1, suppression_radius * 2 + 1)
    )
    for channel in range(NUM_HEATMAP_CHANNELS):
        response = heatmaps[channel]
        dilated = cv2.dilate(response, kernel)
        peaks = (response >= threshold) & (response >= dilated - 1e-6)
        if not np.any(peaks):
            continue
        peak_map = (peaks.astype(np.uint8)) * 255
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(peak_map, connectivity=8)
        for label_id in range(1, count):
            cx, cy = centroids[label_id]
            x = int(round(cx))
            y = int(round(cy))
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            detections.append(
                DetectedJunction(
                    x=x,
                    y=y,
                    channel=channel,
                    score=float(response[y, x]),
                    group=_channel_group(channel),
                )
            )
    # Strongest first so greedy linking prefers confident junctions.
    detections.sort(key=lambda item: item.score, reverse=True)
    _ = height
    return detections
