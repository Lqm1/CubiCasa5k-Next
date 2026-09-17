"""PyTorch dataset binding images to SVG-derived training targets.

Dataset layout per split directory::

    <split>/images/<stem>.png
    <split>/annotations/<stem>.svg

Pairs are discovered by :func:`discover_pairs`. Image and vector masks share
one letterbox geometry (scale long side, then center-pad), so supervision
stays aligned with the network input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cubicasa5k_next.config import AugmentationConfig
from cubicasa5k_next.data.heatmaps import build_heatmap_targets_from_vector
from cubicasa5k_next.data.rasterizer import rasterize_vector
from cubicasa5k_next.data.svg_parser import FloorplanVector, parse_floorplan_svg
from cubicasa5k_next.data.transforms import (
    LetterboxGeom,
    apply_color_jitter,
    apply_letterbox_to_points,
    letterbox_geometry,
    random_right_angle_rotation,
    resize_with_padding,
    to_normalized_tensor,
)
from cubicasa5k_next.labels_config import LabelConfig, default_label_config
from cubicasa5k_next.target_cache import TargetCache, file_version


@dataclass
class FloorplanSample:
    image: torch.Tensor  # (3, S, S) normalized
    room_labels: torch.Tensor  # (S, S) long
    icon_labels: torch.Tensor  # (S, S) long
    heatmaps: torch.Tensor  # (21, S, S) float
    source_path: str


class SvgFloorplanDataset(Dataset[FloorplanSample]):
    """Pairs raster images with SVG annotations rasterized on the fly."""

    def __init__(
        self,
        image_paths: list[Path],
        annotation_paths: list[Path],
        augmentation: AugmentationConfig | None = None,
        training: bool = True,
        target_cache_mb: int = 256,
        label_config: LabelConfig | None = None,
    ) -> None:
        if len(image_paths) != len(annotation_paths):
            raise ValueError("Image and annotation lists must have equal length")
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.augmentation = augmentation or AugmentationConfig()
        self.training = training
        self.target_cache = TargetCache(target_cache_mb)
        self.label_config = label_config or default_label_config()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _build_targets(
        self, annotation_path: Path, image_hw: tuple[int, int], target_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector: FloorplanVector = parse_floorplan_svg(annotation_path, self.label_config)
        height, width = image_hw
        geom = letterbox_geometry(height, width, target_size)
        geom_points: dict[int, list[tuple[float, float]]] = {}
        shapes: list[object] = [*vector.rooms, *vector.walls, *vector.openings, *vector.icons]
        for shape in shapes:
            points = getattr(shape, "points", ())
            geom_points[id(shape)] = apply_letterbox_to_points(list(points), geom)
        room_mask, icon_mask = rasterize_vector(
            vector,
            target_size=(target_size, target_size),
            num_room_classes=self.label_config.num_rooms,
            num_icon_classes=self.label_config.num_icons,
            geom_points=geom_points,
        )
        heatmaps = build_heatmap_targets_from_vector(
            vector,
            target_size=(target_size, target_size),
            geom_points=geom_points,
            radius=self.augmentation.gaussian_radius,
        )
        return room_mask, icon_mask, heatmaps

    def __getitem__(self, index: int) -> FloorplanSample:
        image_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_size = self.augmentation.image_size
        key = (
            file_version(annotation_path),
            image.shape[:2],
            target_size,
            self.augmentation.gaussian_radius,
            self.label_config.num_rooms,
            self.label_config.num_icons,
            len(self.label_config.room_svg_map),
            len(self.label_config.icon_svg_map),
        )
        targets = self.target_cache.get(key)
        if targets is None:
            room_mask, icon_mask, heatmaps = self._build_targets(
                annotation_path, (image.shape[0], image.shape[1]), target_size
            )
            self.target_cache.put(key, (room_mask, icon_mask, heatmaps))
        else:
            room_mask, icon_mask, heatmaps = targets

        resized_image, _, _, _ = resize_with_padding(image, target_size)
        if self.training:
            if self.augmentation.use_color_jitter:
                resized_image = apply_color_jitter(resized_image)
            if self.augmentation.use_rotation_augmentation:
                rotated = random_right_angle_rotation(resized_image, room_mask, icon_mask, heatmaps)
                resized_image = rotated[0]
                assert rotated[1] is not None and rotated[2] is not None and rotated[3] is not None
                room_mask, icon_mask, heatmaps = rotated[1], rotated[2], rotated[3]

        image_tensor = to_normalized_tensor(resized_image)
        return FloorplanSample(
            image=image_tensor,
            room_labels=torch.from_numpy(room_mask).long(),
            icon_labels=torch.from_numpy(icon_mask).long(),
            heatmaps=torch.from_numpy(heatmaps).float(),
            source_path=str(image_path),
        )


def discover_pairs(root: Path, image_extension: str = "*.png") -> tuple[list[Path], list[Path]]:
    """Discover (image, svg) pairs under ``root`` by matching file stems."""
    images = sorted(root.rglob(image_extension))
    paired_images: list[Path] = []
    paired_annotations: list[Path] = []
    for image_path in images:
        candidates: list[Path] = []
        # Separated layout only: images/<stem>.png + annotations/<stem>.svg.
        if image_path.parent.name == "images":
            candidates.append(image_path.parent.parent / "annotations" / f"{image_path.stem}.svg")
        candidates.append(root / "annotations" / f"{image_path.stem}.svg")
        for candidate in candidates:
            if candidate.exists():
                paired_images.append(image_path)
                paired_annotations.append(candidate)
                break
    return paired_images, paired_annotations


__all__ = [
    "FloorplanSample",
    "SvgFloorplanDataset",
    "discover_pairs",
    "LetterboxGeom",
]
