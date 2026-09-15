"""Small image I/O helpers built on OpenCV/Pillow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb_image(path: str | Path, target_size: int | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


def save_colored_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a label mask with a deterministic pseudo-color palette."""
    height, width = mask.shape
    palette = np.zeros((256, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    palette[1:] = rng.integers(0, 255, size=(255, 3), dtype=np.uint8)
    colored = palette[mask.clip(0, 255).reshape(-1)].reshape(height, width, 3)
    cv2.imwrite(str(path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
