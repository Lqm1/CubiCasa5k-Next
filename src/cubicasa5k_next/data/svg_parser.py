"""CubiCasa5k SVG annotation reader (group-structured, transform-aware).

Real floorplans encode semantics per ``<g>`` element, not per shape:

- ``<g id="Wall">`` / ``<g id="Railing">``: direct child ``<polygon>`` is the shape.
- ``<g id="Door">`` / ``<g id="Window">``: direct child ``<polygon>`` is the shape.
  Nested ``Threshold`` / ``Panel`` / ``PanelArea`` / ``Glass`` groups are
  duplicates/decorations and are ignored.
- ``<g class="Space Bedroom">``: second class token is the room token;
  direct child ``<polygon>`` is the shape. ``Dimension`` subtrees are ignored.
- ``<g class="FixedFurniture Toilet" transform="matrix(...)">``: geometry is
  stored in local coordinates and must be mapped through the ``matrix()``
  chain (including an optional ``FixedFurnitureSet`` parent). The
  ``BoundaryPolygon`` child is preferred; otherwise a bounding quad over the
  inner ``polygon`` / ``rect`` / ``path`` shapes is used. One furniture
  element yields at most one quad.

Only the direct ``<polygon>`` child of a semantic group is read, so
dimension marks, glass panes, and panel decorations never become labels.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from cubicasa5k_next.labels_config import LabelConfig, default_label_config

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_MATRIX_PATTERN = re.compile(r"matrix\(\s*([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^\)]+)\)")


def _tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1].lower()


@dataclass(frozen=True)
class RoomShape:
    points: tuple[tuple[float, float], ...]
    svg_token: str
    label_index: int


@dataclass(frozen=True)
class WallShape:
    points: tuple[tuple[float, float], ...]
    is_railing: bool
    label_index: int


@dataclass(frozen=True)
class OpeningShape:
    points: tuple[tuple[float, float], ...]
    is_door: bool  # False -> window
    label_index: int


@dataclass(frozen=True)
class IconShape:
    points: tuple[tuple[float, float], ...]  # quad in SVG coordinates
    svg_token: str
    label_index: int


@dataclass(frozen=True)
class FloorplanVector:
    rooms: tuple[RoomShape, ...] = ()
    walls: tuple[WallShape, ...] = ()
    doors: tuple[OpeningShape, ...] = ()
    windows: tuple[OpeningShape, ...] = ()
    icons: tuple[IconShape, ...] = ()
    canvas_size: tuple[float, float] | None = None

    @property
    def openings(self) -> tuple[OpeningShape, ...]:
        return (*self.doors, *self.windows)


def _parse_points_text(text: str) -> list[tuple[float, float]]:
    numbers = [float(v) for v in _NUMBER_PATTERN.findall(text)]
    return [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]


def _direct_polygon_points(group: ET.Element) -> list[tuple[float, float]] | None:
    """Return the first direct-child ``<polygon>`` points, if usable."""
    for child in group:
        if _tag(child) == "polygon":
            raw = child.get("points", "")
            if not raw:
                continue
            points = _parse_points_text(raw)
            if len(points) >= 3:
                return points
    return None


def _parse_matrix(transform: str | None) -> tuple[float, float, float, float, float, float] | None:
    if not transform:
        return None
    match = _MATRIX_PATTERN.search(transform.replace(" ", ""))
    # Fall back to a whitespace-tolerant parse.
    if match is None:
        match = _MATRIX_PATTERN.search(transform)
    if match is None:
        return None
    try:
        return (
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
            float(match.group(5)),
            float(match.group(6)),
        )
    except ValueError:
        return None


def _compose_matrices(
    outer: tuple[float, float, float, float, float, float] | None,
    inner: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float, float, float] | None:
    if outer is None:
        return inner
    if inner is None:
        return outer
    a0, b0, c0, d0, e0, f0 = outer
    a1, b1, c1, d1, e1, f1 = inner
    return (
        a0 * a1 + c0 * b1,
        b0 * a1 + d0 * b1,
        a0 * c1 + c0 * d1,
        b0 * c1 + d0 * d1,
        a0 * e1 + c0 * f1 + e0,
        b0 * e1 + d0 * f1 + f0,
    )


def _apply_matrix(
    points: list[tuple[float, float]],
    matrix: tuple[float, float, float, float, float, float] | None,
) -> list[tuple[float, float]]:
    if matrix is None:
        return points
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


def _rect_corners(element: ET.Element) -> list[tuple[float, float]] | None:
    try:
        x = float(element.get("x", "0") or "0")
        y = float(element.get("y", "0") or "0")
        w = float(element.get("width", "0") or "0")
        h = float(element.get("height", "0") or "0")
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _path_bbox_corners(element: ET.Element) -> list[tuple[float, float]] | None:
    numbers = [float(v) for v in _NUMBER_PATTERN.findall(element.get("d", ""))]
    if len(numbers) < 4:
        return None
    xs = numbers[0::2]
    ys = numbers[1::2]
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x < 1e-6 or max_y - min_y < 1e-6:
        return None
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def _bbox_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x < 1e-6 or max_y - min_y < 1e-6:
        return None
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def _furniture_local_quad(group: ET.Element) -> list[tuple[float, float]] | None:
    """Resolve one furniture element to a local-coords quad (pre-transform)."""
    # Preferred: BoundaryPolygon > polygon.
    for child in group:
        if _tag(child) == "g" and child.get("class") == "BoundaryPolygon":
            for node in child:
                if _tag(node) == "polygon":
                    points = _parse_points_text(node.get("points", ""))
                    if len(points) >= 3:
                        if len(points) == 4:
                            return points
                        boxed = _bbox_quad(points)
                        if boxed is not None:
                            return boxed
            # BoundaryPolygon without a usable polygon -> fall through to bbox.
            break
    # Fallback: bounding quad over inner polygon/rect/path shapes.
    gathered: list[tuple[float, float]] = []
    for node in group.iter():
        if node is group:
            continue
        tag = _tag(node)
        if tag == "polygon":
            gathered.extend(_parse_points_text(node.get("points", "")))
        elif tag == "rect":
            corners = _rect_corners(node)
            if corners is not None:
                gathered.extend(corners)
        elif tag == "path":
            corners = _path_bbox_corners(node)
            if corners is not None:
                gathered.extend(corners)
        elif tag == "circle":
            try:
                cx = float(node.get("cx", "0") or "0")
                cy = float(node.get("cy", "0") or "0")
                r = float(node.get("r", "0") or "0")
            except ValueError:
                continue
            if r > 0:
                gathered.extend(
                    [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
                )
    return _bbox_quad(gathered)


def _second_token(class_attr: str) -> str:
    parts = class_attr.split()
    return parts[1] if len(parts) >= 2 else ""


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


def parse_floorplan_svg(
    svg_path: str | Path,
    label_config: LabelConfig | None = None,
) -> FloorplanVector:
    """Parse a CubiCasa5k ``model.svg`` into typed vector shapes."""
    config = label_config or default_label_config()
    root = ET.parse(str(svg_path)).getroot()
    parent_map: dict[int, ET.Element] = {id(c): p for p in root.iter() for c in p}

    rooms: list[RoomShape] = []
    walls: list[WallShape] = []
    doors: list[OpeningShape] = []
    windows: list[OpeningShape] = []
    icons: list[IconShape] = []

    for element in root.iter():
        if _tag(element) != "g":
            continue
        element_id = (element.get("id") or "").strip()
        class_attr = (element.get("class") or "").strip()

        if element_id in ("Wall", "Railing"):
            points = _direct_polygon_points(element)
            if points is None:
                continue
            is_railing = element_id == "Railing"
            token = "Railing" if is_railing else "Wall"
            walls.append(
                WallShape(
                    points=tuple(points),
                    is_railing=is_railing,
                    label_index=config.resolve_room_token(token),
                )
            )
        elif element_id in ("Door", "Window"):
            points = _direct_polygon_points(element)
            if points is None or len(points) < 3:
                continue
            is_door = element_id == "Door"
            token = "Door" if is_door else "Window"
            resolved = config.resolve_icon_token(token)
            if resolved is None:
                continue
            shape = OpeningShape(points=tuple(points), is_door=is_door, label_index=int(resolved))
            (doors if is_door else windows).append(shape)
        elif "FixedFurniture " in f"{class_attr} " or class_attr.startswith("FixedFurniture"):
            token = _second_token(class_attr)
            if not token:
                continue
            resolved = config.resolve_icon_token(token)
            if resolved is None:
                continue  # Ignored class (cabinet/shower/...) by config.
            local = _furniture_local_quad(element)
            if local is None:
                continue
            # Compose the ancestor matrix chain (outermost first).
            chain: list[ET.Element] = []
            current: ET.Element | None = element
            while current is not None:
                chain.append(current)
                current = parent_map.get(id(current))
            matrix = None
            for node in reversed(chain):
                matrix = _compose_matrices(matrix, _parse_matrix(node.get("transform")))
            mapped = _apply_matrix(local, matrix)
            rounded = tuple((round(x), round(y)) for x, y in mapped)
            if len(set(rounded)) < 4:
                boxed = _bbox_quad(mapped)
                if boxed is None:
                    continue
                mapped = boxed
            else:
                # Keep quad structure; round to int-like floats.
                mapped = [(float(x), float(y)) for x, y in rounded]
            if len(mapped) != 4:
                boxed = _bbox_quad(mapped)
                if boxed is None:
                    continue
                mapped = boxed
            icons.append(
                IconShape(points=tuple(mapped), svg_token=token, label_index=int(resolved))
            )
        elif "Space " in f"{class_attr} " or class_attr.startswith("Space"):
            token = _second_token(class_attr)
            if not token:
                continue
            points = _direct_polygon_points(element)
            if points is None:
                continue
            rooms.append(
                RoomShape(
                    points=tuple(points),
                    svg_token=token,
                    label_index=config.resolve_room_token(token),
                )
            )

    return FloorplanVector(
        rooms=tuple(rooms),
        walls=tuple(walls),
        doors=tuple(doors),
        windows=tuple(windows),
        icons=tuple(icons),
        canvas_size=_size_from_root(root),
    )
