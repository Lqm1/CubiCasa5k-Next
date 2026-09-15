"""Training loop and metrics."""

from __future__ import annotations

from cubicasa5k_next.training.metrics import SegmentationScores, compute_segmentation_scores

__all__ = ["SegmentationScores", "compute_segmentation_scores"]
