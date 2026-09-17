"""Image preprocessing matching the paper's augmentation recipe."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class LetterboxGeom:
    """Shared image/vector geometry: scale long side, then center-pad."""

    scale: float
    top: int
    left: int
    new_height: int
    new_width: int
    target_size: int


def letterbox_geometry(height: int, width: int, target_size: int) -> LetterboxGeom:
    scale = target_size / max(height, width)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    return LetterboxGeom(
        scale=scale,
        top=top,
        left=left,
        new_height=new_height,
        new_width=new_width,
        target_size=target_size,
    )


def apply_letterbox_to_points(
    points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    geom: LetterboxGeom,
) -> list[tuple[float, float]]:
    return [(x * geom.scale + geom.left, y * geom.scale + geom.top) for x, y in points]


def resize_with_padding(
    image: np.ndarray, target_size: int, fill: int = 0
) -> tuple[np.ndarray, float, int, int]:
    """Scale the long side to ``target_size`` and zero-pad to a square."""
    height, width = image.shape[:2]
    geom = letterbox_geometry(height, width, target_size)
    resized = cv2.resize(image, (geom.new_width, geom.new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), fill, dtype=resized.dtype)
    canvas[geom.top : geom.top + geom.new_height, geom.left : geom.left + geom.new_width] = resized
    return canvas, geom.scale, geom.top, geom.left


def resize_mask_letterbox(mask: np.ndarray, geom: LetterboxGeom, fill: int = 0) -> np.ndarray:
    """Resize a (H, W) label mask with the same geometry (nearest-neighbour)."""
    resized = cv2.resize(
        mask,
        (geom.new_width, geom.new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((geom.target_size, geom.target_size), fill, dtype=resized.dtype)
    canvas[geom.top : geom.top + geom.new_height, geom.left : geom.left + geom.new_width] = resized
    return canvas


def random_right_angle_rotation(
    image: np.ndarray,
    room_mask: np.ndarray | None = None,
    icon_mask: np.ndarray | None = None,
    heatmaps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Apply a random 0/90/180/270-degree rotation (paper augmentation)."""
    choice = random.choice([0, 1, 2, 3])
    if choice == 0:
        return image, room_mask, icon_mask, heatmaps
    rotated_image = np.rot90(image, choice).copy()
    rotated_room = np.rot90(room_mask, choice).copy() if room_mask is not None else None
    rotated_icon = np.rot90(icon_mask, choice).copy() if icon_mask is not None else None
    rotated_heat = np.rot90(heatmaps, choice, axes=(1, 2)).copy() if heatmaps is not None else None
    return rotated_image, rotated_room, rotated_icon, rotated_heat


def apply_color_jitter(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """Lightweight brightness/contrast jitter without extra dependencies."""
    if strength <= 0:
        return image
    brightness = random.uniform(1.0 - strength, 1.0 + strength)
    contrast = random.uniform(1.0 - strength, 1.0 + strength)
    jittered = image.astype(np.float32) * brightness
    mean = jittered.mean(axis=(0, 1), keepdims=True)
    jittered = (jittered - mean) * contrast + mean
    return np.clip(jittered, 0, 255).astype(np.uint8)


def to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    """Scale RGB uint8 images to [-1, 1] tensors."""
    array = 2.0 * (image.astype(np.float32) / 255.0) - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)
