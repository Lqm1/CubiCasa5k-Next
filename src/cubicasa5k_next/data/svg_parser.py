"""Minimal SVG annotation reader for polygon-based floorplan labels.

Supports the dataset's vector format: an SVG document whose shapes carry a
semantic label in attributes such as ``class``, ``label``, ``category`` or
``data-type``. Only axis-aligned geometry is required downstream, but this
parser preserves arbitrary polygons and lets the rasterizer handle them.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnnotatedPolygon:
    points: tuple[tuple[float, float], ...]
    raw_label: str
    kind: str  # "room" | "icon" | "wall" | "opening" | "unknown"


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_points_attribute(text: str) -> tuple[tuple[float, float], ...]:
    numbers = [float(v) for v in _NUMBER_PATTERN.findall(text)]
    paired = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]
    return tuple(paired)


def _parse_rect(element: ET.Element) -> tuple[tuple[float, float], ...] | None:
    try:
        x = float(element.get("x", "0"))
        y = float(element.get("y", "0"))
        width = float(element.get("width", "0"))
        height = float(element.get("height", "0"))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))


def _element_label(element: ET.Element) -> str:
    for key in ("label", "class", "category", "data-type", "data-label", "id"):
        value = element.get(key)
        if value:
            return value
    # Look at child title/desc elements if present.
    for child in element:
        tag = child.tag.split("}")[-1].lower()
        if tag in {"title", "desc"} and child.text:
            return child.text
    return "unknown"


def _element_kind(element: ET.Element, label: str) -> str:
    kind_hint = (element.get("data-kind") or element.get("kind") or "").lower()
    if kind_hint in {"room", "icon", "wall", "opening"}:
        return kind_hint
    normalized = label.lower()
    if any(token in normalized for token in ("wall", "railing")) and "room" not in normalized:
        # Walls are often separate, but railing rooms stay rooms; handled by caller.
        pass
    if any(token in normalized for token in ("window", "door", "opening")):
        return "opening"
    if any(
        token in normalized
        for token in (
            "closet",
            "toilet",
            "sink",
            "bathtub",
            "chimney",
            "fireplace",
            "sauna",
            "electr",
            "appliance",
            "icon",
        )
    ):
        return "icon"
    if normalized in {"wall"}:
        return "wall"
    return "room"


def _group_context(element: ET.Element, parent_map: dict[int, ET.Element]) -> str:
    """Collect ancestor <g> identifiers for group-structured annotations."""
    parts: list[str] = []
    current = parent_map.get(id(element))
    while current is not None:
        tag = current.tag.split("}")[-1].lower()
        if tag == "g":
            group_id = current.get("id", "")
            group_class = current.get("class", "")
            if group_id:
                parts.append(group_id)
            if group_class:
                parts.append(group_class)
        current = parent_map.get(id(current))
    return " ".join(reversed(parts))


def parse_svg_polygons(svg_path: str | Path) -> list[AnnotatedPolygon]:
    """Parse polygons/rects from an SVG annotation file."""
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    return _polygons_from_root(root)


def parse_svg_annotation(
    svg_path: str | Path,
) -> tuple[list[AnnotatedPolygon], tuple[float, float] | None]:
    """Read geometry and dimensions with one XML parse."""
    root = ET.parse(str(svg_path)).getroot()
    return _polygons_from_root(root), _size_from_root(root)


def _polygons_from_root(root: ET.Element) -> list[AnnotatedPolygon]:
    parent_map: dict[int, ET.Element] = {
        id(child): parent for parent in root.iter() for child in parent
    }
    results: list[AnnotatedPolygon] = []
    for element in root.iter():
        tag = element.tag.split("}")[-1].lower()
        points: tuple[tuple[float, float], ...] | None = None
        if tag == "polygon" or tag == "polyline":
            raw = element.get("points", "")
            if raw:
                points = _parse_points_attribute(raw)
        elif tag == "rect":
            points = _parse_rect(element)
        elif tag == "path":
            # Only support simple rectangular paths (M ... H/V ... Z).
            raw = element.get("d", "")
            if raw:
                numbers = [float(v) for v in _NUMBER_PATTERN.findall(raw)]
                if len(numbers) >= 8:
                    points = tuple(
                        (numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)
                    )
        if not points or len(points) < 3:
            continue
        label = _element_label(element)
        context = _group_context(element, parent_map)
        combined = f"{context} {label}".strip() if context else label
        # Prefer explicit group semantics when shapes carry no label.
        if context and (label == "unknown" or tag in {"polygon", "path"}):
            label = combined
        kind = _element_kind(element, label)
        results.append(AnnotatedPolygon(points=points, raw_label=label, kind=kind))
    return results


def read_svg_size(svg_path: str | Path) -> tuple[float, float] | None:
    """Best-effort extraction of the SVG canvas size (width, height)."""
    try:
        tree = ET.parse(str(svg_path))
    except ET.ParseError:
        return None
    root = tree.getroot()
    return _size_from_root(root)


def _size_from_root(root: ET.Element) -> tuple[float, float] | None:
    width_raw = root.get("width")
    height_raw = root.get("height")
    view_box = root.get("viewBox") or root.get("viewbox")
    try:
        if width_raw and height_raw:
            width = float(_NUMBER_PATTERN.findall(width_raw)[0])
            height = float(_NUMBER_PATTERN.findall(height_raw)[0])
            return (width, height)
        if view_box:
            numbers = [float(v) for v in _NUMBER_PATTERN.findall(view_box)]
            if len(numbers) == 4:
                return (numbers[2], numbers[3])
    except (IndexError, ValueError):
        return None
    return None
