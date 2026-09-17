"""Dataset utilities for SVG-annotated floorplans."""

from __future__ import annotations

from cubicasa5k_next.data.dataset import (
    FloorplanSample,
    SvgFloorplanDataset,
    discover_pairs,
)

__all__ = ["FloorplanSample", "SvgFloorplanDataset", "discover_pairs"]
