"""Segmentation metrics used in the paper's CubiCasa5k experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentationScores:
    overall_accuracy: float
    mean_accuracy: float
    mean_iou: float
    per_class_iou: list[float]
    per_class_accuracy: list[float]


def compute_segmentation_scores(
    predicted: torch.Tensor, target: torch.Tensor, num_classes: int
) -> SegmentationScores:
    """Compute overall accuracy, mean accuracy, and mean IoU."""
    if predicted.dim() == 4:
        predicted_labels = predicted.argmax(dim=1)
    else:
        predicted_labels = predicted
    predicted_flat = predicted_labels.reshape(-1)
    target_flat = target.reshape(-1)

    overall = (predicted_flat == target_flat).float().mean().item()

    per_class_accuracy: list[float] = []
    per_class_iou: list[float] = []
    for class_id in range(num_classes):
        target_mask = target_flat == class_id
        predicted_mask = predicted_flat == class_id
        target_count = int(target_mask.sum().item())
        if target_count == 0:
            per_class_accuracy.append(float("nan"))
            per_class_iou.append(float("nan"))
            continue
        correct = int((predicted_mask & target_mask).sum().item())
        per_class_accuracy.append(correct / max(target_count, 1))
        union = int((predicted_mask | target_mask).sum().item())
        per_class_iou.append(correct / max(union, 1))

    valid_accuracy = [v for v in per_class_accuracy if v == v]
    valid_iou = [v for v in per_class_iou if v == v]
    mean_accuracy = sum(valid_accuracy) / max(len(valid_accuracy), 1)
    mean_iou = sum(valid_iou) / max(len(valid_iou), 1)
    return SegmentationScores(
        overall_accuracy=overall,
        mean_accuracy=mean_accuracy,
        mean_iou=mean_iou,
        per_class_iou=per_class_iou,
        per_class_accuracy=per_class_accuracy,
    )
