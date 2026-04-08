from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from dxf_renderer import render_graph_plot

SMALL_MODEL_SCALE = 5000.0
DEFAULT_DPI = 100
GRAPHICAL_TAGS = {
    "Drawing",
    "ShapeCatalogue",
    "Presentation",
    "Position",
    "Label",
    "Text",
    "CenterLine",
    "PolyLine",
    "Circle",
    "PipeFlowArrow",
    "PipeSlopeSymbol",
}
PRIMITIVE_TAGS = {"CenterLine", "PolyLine", "Shape", "Circle", "Ellipse", "Text"}


@dataclass(frozen=True)
class ViewBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(self.max_x - self.min_x, 0.1)

    @property
    def height(self) -> float:
        return max(self.max_y - self.min_y, 0.1)


@dataclass(frozen=True)
class Transform:
    tx: float = 0.0
    ty: float = 0.0
    rotation_deg: float = 0.0
    sx: float = 1.0
    sy: float = 1.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        px = x * self.sx
        py = y * self.sy
        angle = math.radians(self.rotation_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        return (self.tx + rx, self.ty + ry)


def render_dexpi_plot(path_xml: str, path_graph: str, path_plot_stem: str) -> None:
    xml_path = Path(path_xml)
    stem = _normalized_output_stem(path_plot_stem)
    if _is_graphical_dexpi(xml_path):
        _render_graphical_dexpi(xml_path, stem)
        return
    render_graph_plot(path_graph, str(stem))


def _normalized_output_stem(path_plot_stem: str) -> Path:
    path = Path(path_plot_stem)
    if path.suffix:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_graphical_dexpi(xml_path: Path) -> bool:
    root = ET.parse(xml_path).getroot()
    hits = 0
    for tag in GRAPHICAL_TAGS:
        if root.find(f".//{tag}") is not None:
            hits += 1
    return hits >= 4 and root.find(".//Extent") is not None


def _render_graphical_dexpi(xml_path: Path, output_stem: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    view_box = _extract_view_box(root)
    pixel_scale = _pixel_scale(view_box)
    figure, axis = _build_figure(view_box, pixel_scale)
    axis.set_facecolor("white")
    shape_catalogue = root.find(".//ShapeCatalogue")
    component_map = _build_component_map(shape_catalogue)

    background = root.find(".//Drawing/Presentation")
    if background is not None:
        axis.set_facecolor(_color_from_presentation(background))

    for element in _iter_render_elements(root):
        definition = component_map.get(element.get("ComponentName", ""))
        if definition is not None and element is not definition:
            _draw_catalogue_definition(axis, definition, _transform_from_element(element), view_box, pixel_scale)
        for child in list(element):
            if child.tag in PRIMITIVE_TAGS:
                _draw_primitive(axis, child, view_box, pixel_scale, transform=None)

    axis.set_aspect("equal")
    axis.axis("off")
    with mpl.rc_context({"svg.fonttype": "none"}):
        figure.savefig(output_stem.with_suffix(".png"), dpi=DEFAULT_DPI, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(output_stem.with_suffix(".svg"), dpi=DEFAULT_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _build_figure(view_box: ViewBox, pixel_scale: float):
    width_px = view_box.width * pixel_scale
    height_px = view_box.height * pixel_scale
    figure = plt.figure(
        figsize=(max(8.0, width_px / DEFAULT_DPI), max(6.0, height_px / DEFAULT_DPI)),
        dpi=DEFAULT_DPI,
    )
    axis = figure.add_subplot(111)
    axis.set_xlim(0, width_px)
    axis.set_ylim(height_px, 0)
    return figure, axis


def _extract_view_box(root) -> ViewBox:
    extent = root.find(".//Drawing/Extent")
    if extent is not None:
        min_node = extent.find("Min")
        max_node = extent.find("Max")
        if min_node is not None and max_node is not None:
            return ViewBox(
                _float_attr(min_node, "X"),
                _float_attr(min_node, "Y"),
                _float_attr(max_node, "X"),
                _float_attr(max_node, "Y"),
            )

    coords = []
    for coordinate in root.findall(".//Coordinate"):
        coords.append((_float_attr(coordinate, "X"), _float_attr(coordinate, "Y")))
    for position in root.findall(".//Position/Location"):
        coords.append((_float_attr(position, "X"), _float_attr(position, "Y")))
    if not coords:
        return ViewBox(0.0, 0.0, 1.0, 1.0)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return ViewBox(min(xs), min(ys), max(xs), max(ys))


def _pixel_scale(view_box: ViewBox) -> float:
    if view_box.width <= 10.0 and view_box.height <= 10.0:
        return SMALL_MODEL_SCALE
    return 1.0


def _build_component_map(shape_catalogue):
    component_map = {}
    if shape_catalogue is None:
        return component_map
    for element in list(shape_catalogue):
        component_name = element.get("ComponentName")
        if component_name:
            component_map[component_name] = element
    return component_map


def _iter_render_elements(root):
    stack = [root]
    while stack:
        element = stack.pop()
        if element.tag == "ShapeCatalogue":
            continue
        if element.tag not in PRIMITIVE_TAGS:
            yield element
        children = list(element)
        for child in reversed(children):
            if child.tag != "ShapeCatalogue":
                stack.append(child)


def _draw_catalogue_definition(axis, definition, transform: Transform, view_box: ViewBox, pixel_scale: float) -> None:
    for child in list(definition):
        if child.tag in PRIMITIVE_TAGS:
            _draw_primitive(axis, child, view_box, pixel_scale, transform=transform)


def _draw_primitive(axis, primitive, view_box: ViewBox, pixel_scale: float, transform: Transform | None) -> None:
    if primitive.tag in {"PolyLine", "CenterLine", "Shape"}:
        points = _coordinates_from_primitive(primitive, transform)
        if len(points) < 2:
            return
        style = _style_from_presentation(primitive.find("Presentation"), pixel_scale)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if primitive.tag == "Shape" and primitive.get("Filled") == "Solid":
            axis.add_patch(
                Polygon(
                    [_to_canvas(point[0], point[1], view_box, pixel_scale) for point in points],
                    closed=True,
                    fill=True,
                    facecolor=style["color"],
                    edgecolor=style["color"],
                    linewidth=style["linewidth"],
                    joinstyle="round",
                    zorder=3,
                )
            )
            return
        axis.plot(
            [_to_canvas_x(x, view_box, pixel_scale) for x in xs],
            [_to_canvas_y(y, view_box, pixel_scale) for y in ys],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            solid_capstyle="round",
            zorder=2,
        )
        return

    if primitive.tag == "Circle":
        style = _style_from_presentation(primitive.find("Presentation"), pixel_scale)
        center = _circle_center(primitive, transform)
        radius = _float_attr(primitive, "Radius") * pixel_scale
        axis.add_patch(
            Circle(
                _to_canvas(center[0], center[1], view_box, pixel_scale),
                radius=radius,
                fill=primitive.get("Filled") == "Solid",
                facecolor=style["color"] if primitive.get("Filled") == "Solid" else "none",
                edgecolor=style["color"],
                linewidth=style["linewidth"],
                zorder=3,
            )
        )
        return

    if primitive.tag == "Ellipse":
        return

    if primitive.tag == "Text":
        position = primitive.find("Position/Location")
        if position is None:
            return
        x = _float_attr(position, "X")
        y = _float_attr(position, "Y")
        text_style = _style_from_presentation(primitive.find("Presentation"), pixel_scale)
        justification = primitive.get("Justification", "CenterCenter")
        ha, va = _text_alignment(justification)
        rotation = float(primitive.get("TextAngle", "0") or "0")
        axis.text(
            _to_canvas_x(x, view_box, pixel_scale),
            _to_canvas_y(y, view_box, pixel_scale),
            primitive.get("String", ""),
            fontsize=max(6.0, _float_attr(primitive, "Height", 0.002) * pixel_scale),
            color=text_style["color"],
            ha=ha,
            va=va,
            rotation=-rotation,
            family=_font_family(primitive.get("Font", "")),
            zorder=4,
        )


def _coordinates_from_primitive(primitive, transform: Transform | None):
    points = []
    for coordinate in primitive.findall("Coordinate"):
        x = _float_attr(coordinate, "X")
        y = _float_attr(coordinate, "Y")
        if transform is not None:
            x, y = transform.apply(x, y)
        points.append((x, y))
    return points


def _circle_center(primitive, transform: Transform | None):
    location = primitive.find("Position/Location")
    if location is None:
        x = 0.0
        y = 0.0
    else:
        x = _float_attr(location, "X")
        y = _float_attr(location, "Y")
    if transform is not None:
        x, y = transform.apply(x, y)
    return (x, y)


def _transform_from_element(element) -> Transform:
    position = element.find("Position")
    scale = element.find("Scale")
    tx = 0.0
    ty = 0.0
    rotation_deg = 0.0
    sx = 1.0
    sy = 1.0

    if position is not None:
        location = position.find("Location")
        reference = position.find("Reference")
        if location is not None:
            tx = _float_attr(location, "X")
            ty = _float_attr(location, "Y")
        if reference is not None:
            ref_x = _float_attr(reference, "X", 1.0)
            ref_y = _float_attr(reference, "Y", 0.0)
            rotation_deg = math.degrees(math.atan2(ref_y, ref_x))

    if scale is not None:
        sx = _float_attr(scale, "X", 1.0)
        sy = _float_attr(scale, "Y", 1.0)

    return Transform(tx=tx, ty=ty, rotation_deg=rotation_deg, sx=sx, sy=sy)


def _style_from_presentation(presentation, pixel_scale: float):
    color = _color_from_presentation(presentation)
    line_weight = 0.0002
    line_type = "0"
    if presentation is not None:
        line_weight = _float_attr(presentation, "LineWeight", 0.0002)
        line_type = presentation.get("LineType", "0")
    linewidth = max(0.8, line_weight * pixel_scale)
    linestyle = "-" if line_type in {"", "0"} else "--"
    return {"color": color, "linewidth": linewidth, "linestyle": linestyle}


def _color_from_presentation(presentation):
    if presentation is None:
        return "#000000"
    r = int(max(0.0, min(1.0, _float_attr(presentation, "R", 0.0))) * 255)
    g = int(max(0.0, min(1.0, _float_attr(presentation, "G", 0.0))) * 255)
    b = int(max(0.0, min(1.0, _float_attr(presentation, "B", 0.0))) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _text_alignment(justification: str) -> tuple[str, str]:
    mapping = {
        "CenterCenter": ("center", "center"),
        "RightTop": ("right", "top"),
        "RightCenter": ("right", "center"),
        "LeftCenter": ("left", "center"),
        "LeftTop": ("left", "top"),
        "CenterTop": ("center", "top"),
        "CenterBottom": ("center", "bottom"),
        "RightBottom": ("right", "bottom"),
        "LeftBottom": ("left", "bottom"),
    }
    return mapping.get(justification, ("center", "center"))


def _to_canvas(point_x: float, point_y: float, view_box: ViewBox, pixel_scale: float) -> tuple[float, float]:
    return (_to_canvas_x(point_x, view_box, pixel_scale), _to_canvas_y(point_y, view_box, pixel_scale))


def _to_canvas_x(point_x: float, view_box: ViewBox, pixel_scale: float) -> float:
    return (point_x - view_box.min_x) * pixel_scale


def _to_canvas_y(point_y: float, view_box: ViewBox, pixel_scale: float) -> float:
    return (view_box.max_y - point_y) * pixel_scale


def _float_attr(element, key: str, default: float = 0.0) -> float:
    value = element.get(key)
    if value in (None, ""):
        return default
    return float(str(value).replace(",", "."))


def _font_family(font_name: str) -> str:
    normalized = (font_name or "").strip().lower()
    if normalized in {"calibri", ""}:
        return "DejaVu Sans"
    return font_name
