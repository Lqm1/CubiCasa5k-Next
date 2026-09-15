"""Image preprocessing matching the paper's augmentation recipe."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def resize_with_padding(
    image: np.ndarray, target_size: int, fill: int = 0
) -> tuple[np.ndarray, float, int, int]:
    """Scale the long side to ``target_size`` and zero-pad to a square."""
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), fill, dtype=resized.dtype)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, top, left


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
