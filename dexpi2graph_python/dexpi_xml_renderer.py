from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from dxf_renderer import ASSET_DIR, BBox, VisualSpec, draw_visual_spec, render_graph_plot

SMALL_MODEL_SCALE = 5000.0
DEFAULT_DPI = 100
STUB_LENGTH_PX = 5.0
MIN_RENDER_SIZE = 1e-6
MIN_SYMBOL_PX = 24.0
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
PROTEUS_SYMBOL_ALIASES = {
    "off_page_connector": "inside_plant_sheet_connector",
    "offpageconnector": "inside_plant_sheet_connector",
    "inside_plant_sheet_connector": "inside_plant_sheet_connector",
    "check_valve": "check_valve",
    "check_valve_with_flange": "check_valve_with_flange",
    "flanged_joint": "blind_flange",
    "flangedjoint": "blind_flange",
}


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
    root = ET.parse(xml_path).getroot()

    if _is_proteusxml_root(root):
        _render_proteusxml(root, stem)
        return
    if _is_xmplant_bbox_root(root):
        _render_xmplant_bbox(root, stem)
        return
    if _is_graphical_dexpi_root(root):
        _render_graphical_dexpi(root, stem)
        return
    render_graph_plot(path_graph, str(stem))


def _normalized_output_stem(path_plot_stem: str) -> Path:
    path = Path(path_plot_stem)
    if path.suffix.lower() in {".png", ".svg"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _output_file(stem: Path, extension: str) -> Path:
    return Path(f"{stem}{extension}")


def _is_proteusxml_root(root) -> bool:
    return _local_name(root.tag) == "ProteusXML" and any(
        _local_name(elem.tag) == "Component" for elem in root.iter()
    )


def _is_graphical_dexpi_root(root) -> bool:
    if _is_proteusxml_root(root):
        return False
    if _is_xmplant_bbox_root(root):
        return False
    available = {_local_name(elem.tag) for elem in root.iter()}
    hits = sum(1 for tag in GRAPHICAL_TAGS if tag in available)
    return hits >= 4 and "Extent" in available


def _is_xmplant_bbox_root(root) -> bool:
    if _local_name(root.tag) != "PlantModel":
        return False
    tags = {_local_name(elem.tag) for elem in root.iter()}
    return (
        "PipingComponent" in tags
        and "PipingNetworkSegment" in tags
        and "GenericAttribute" in tags
        and "Position" in tags
        and "Extent" in tags
    )


def _render_graphical_dexpi(root, output_stem: Path) -> None:
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
            _draw_catalogue_definition(
                axis,
                definition,
                _transform_from_element(element),
                view_box,
                pixel_scale,
            )
        for child in list(element):
            if child.tag in PRIMITIVE_TAGS:
                _draw_primitive(
                    axis,
                    child,
                    view_box,
                    pixel_scale,
                    transform=None,
                )

    _finalize_figure(figure, axis, output_stem)


def _render_proteusxml(root, output_stem: Path) -> None:
    view_box = _extract_proteus_view_box(root)
    pixel_scale = _pixel_scale(view_box)
    figure, axis = _build_figure(view_box, pixel_scale)
    axis.set_facecolor("white")

    components = {component.get("id", ""): component for component in _iter_local(root, "Component")}
    bbox_map = {
        component_id: _bbox_from_proteus_element(component, view_box, pixel_scale)
        for component_id, component in components.items()
        if component_id
    }
    port_map = _build_port_map(components, view_box, pixel_scale)

    for segment in _iter_local(root, "PipeSegment"):
        _draw_proteus_segment(axis, segment, view_box, pixel_scale, bbox_map, port_map)

    for component in components.values():
        _draw_proteus_component(axis, component, view_box, pixel_scale)

    _finalize_figure(figure, axis, output_stem)


def _render_xmplant_bbox(root, output_stem: Path) -> None:
    view_box = _extract_xmplant_view_box(root)
    pixel_scale = _pixel_scale(view_box)
    figure, axis = _build_figure(view_box, pixel_scale)
    axis.set_facecolor("white")

    components = {component.get("ID", ""): component for component in _iter_local(root, "PipingComponent")}
    bbox_map = {
        component_id: _bbox_from_xmplant_component(component, view_box, pixel_scale)
        for component_id, component in components.items()
        if component_id
    }

    for segment in _iter_local(root, "PipingNetworkSegment"):
        _draw_xmplant_segment(axis, segment, bbox_map)

    for component in components.values():
        _draw_xmplant_component(axis, component, view_box, pixel_scale)

    _finalize_figure(figure, axis, output_stem)


def _finalize_figure(figure, axis, output_stem: Path) -> None:
    axis.set_aspect("equal")
    axis.axis("off")
    with mpl.rc_context({"svg.fonttype": "none"}):
        figure.savefig(
            _output_file(output_stem, ".png"),
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        figure.savefig(
            _output_file(output_stem, ".svg"),
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )
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


def _extract_proteus_view_box(root) -> ViewBox:
    bounds = []
    for bbox in _iter_local(root, "GraphicBounds"):
        bounds.append(
            (
                _float_attr(bbox, "min_x"),
                _float_attr(bbox, "min_y"),
                _float_attr(bbox, "max_x"),
                _float_attr(bbox, "max_y"),
            )
        )
    for pos in _iter_local(root, "Position"):
        x = _float_attr(pos, "x")
        y = _float_attr(pos, "y")
        if x or y:
            bounds.append((x, y, x, y))
    if not bounds:
        return ViewBox(0.0, 0.0, 1.0, 1.0)
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    pw = max((max_x - min_x) * 0.12, 1e-3)
    ph = max((max_y - min_y) * 0.12, 1e-3)
    return ViewBox(min_x - pw, min_y - ph, max_x + pw, max_y + ph)


def _extract_xmplant_view_box(root) -> ViewBox:
    bounds = []
    for comp in _iter_local(root, "PipingComponent"):
        extent = _find_child_local(comp, "Extent")
        if extent is not None:
            min_node = _find_child_local(extent, "Min")
            max_node = _find_child_local(extent, "Max")
            if min_node is not None and max_node is not None:
                bounds.append((
                    _float_attr(min_node, "X"),
                    _float_attr(min_node, "Y"),
                    _float_attr(max_node, "X"),
                    _float_attr(max_node, "Y"),
                ))
        else:
            pos = _find_child_local(comp, "Position")
            if pos is not None:
                loc = _find_child_local(pos, "Location")
                if loc is not None:
                    x = _float_attr(loc, "X")
                    y = _float_attr(loc, "Y")
                    bounds.append((x, y, x, y))
    if bounds:
        min_x = min(item[0] for item in bounds)
        min_y = min(item[1] for item in bounds)
        max_x = max(item[2] for item in bounds)
        max_y = max(item[3] for item in bounds)
        pw = max((max_x - min_x) * 0.12, 1e-5)
        ph = max((max_y - min_y) * 0.12, 1e-5)
        return ViewBox(min_x - pw, min_y - ph, max_x + pw, max_y + ph)
    return _extract_view_box(root)


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


def _compute_definition_natural_size(
    definition, transform: Transform, pixel_scale: float
) -> tuple[float, float]:
    xs, ys = [], []
    for child in list(definition):
        if child.tag in {"PolyLine", "CenterLine", "Shape"}:
            for coord in child.findall("Coordinate"):
                xs.append(_float_attr(coord, "X") * transform.sx * pixel_scale)
                ys.append(_float_attr(coord, "Y") * transform.sy * pixel_scale)
    if not xs:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def _draw_catalogue_definition(
    axis,
    definition,
    transform: Transform,
    view_box: ViewBox,
    pixel_scale: float,
) -> None:
    w, h = _compute_definition_natural_size(definition, transform, pixel_scale)
    max_dim = max(w, h)
    if 0.0 < max_dim < MIN_SYMBOL_PX:
        scale_factor = MIN_SYMBOL_PX / max_dim
        transform = Transform(
            tx=transform.tx,
            ty=transform.ty,
            rotation_deg=transform.rotation_deg,
            sx=transform.sx * scale_factor,
            sy=transform.sy * scale_factor,
        )
    for child in list(definition):
        if child.tag in PRIMITIVE_TAGS:
            _draw_primitive(axis, child, view_box, pixel_scale, transform=transform)


def _draw_primitive(
    axis,
    primitive,
    view_box: ViewBox,
    pixel_scale: float,
    transform: Transform | None,
) -> None:
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
                    [
                        _to_canvas(point[0], point[1], view_box, pixel_scale)
                        for point in points
                    ],
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
        scale = math.sqrt(transform.sx * transform.sy) if transform is not None else 1.0
        radius = _float_attr(primitive, "Radius") * pixel_scale * scale
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
        text_style = _style_from_presentation(
            primitive.find("Presentation"),
            pixel_scale,
        )
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


def _draw_proteus_segment(
    axis, segment, view_box: ViewBox, pixel_scale: float, bbox_map, port_map=None
) -> None:
    start_xml = _find_child_local(segment, "Start")
    end_xml = _find_child_local(segment, "End")
    if start_xml is None or end_xml is None:
        return

    start_id = segment.get("startNode", "")
    end_id = segment.get("endNode", "")
    start_box = bbox_map.get(start_id)
    end_box = bbox_map.get(end_id)

    seg_sx = _to_canvas_x(_float_attr(start_xml, "x"), view_box, pixel_scale)
    seg_sy = _to_canvas_y(_float_attr(start_xml, "y"), view_box, pixel_scale)
    seg_ex = _to_canvas_x(_float_attr(end_xml, "x"), view_box, pixel_scale)
    seg_ey = _to_canvas_y(_float_attr(end_xml, "y"), view_box, pixel_scale)

    style = _proteus_segment_style(segment.get("layer", ""))

    # Use the opposite endpoint as the "facing" reference so we pick the correct port
    start_point = _resolve_port_anchor(
        axis, start_id, start_box, seg_ex, seg_ey, port_map, style
    )
    end_point = _resolve_port_anchor(
        axis, end_id, end_box, seg_sx, seg_sy, port_map, style
    )

    points = _orthogonal_route(start_point, end_point)
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        solid_capstyle="round",
        zorder=1,
    )

    nominal_diameter = segment.get("nominalDiameter", "")
    if nominal_diameter:
        mid_x = (start_point[0] + end_point[0]) / 2.0
        mid_y = (start_point[1] + end_point[1]) / 2.0
        axis.text(
            mid_x, mid_y + 6,
            nominal_diameter,
            fontsize=7,
            color=style["color"],
            ha="center",
            va="top",
            zorder=3,
        )


def _resolve_port_anchor(
    axis,
    component_id: str,
    bbox,
    target_x: float,
    target_y: float,
    port_map,
    style: dict,
) -> tuple[float, float]:
    """Return the connection anchor for a component, drawing a stub when a port is found."""
    if port_map is not None:
        ports = port_map.get(component_id, [])
        if ports and bbox is not None:
            port = _find_closest_port(ports, target_x, target_y)
            stub_tip = _stub_endpoint(port[0], port[1], bbox)
            axis.plot(
                [port[0], stub_tip[0]],
                [port[1], stub_tip[1]],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle="-",
                solid_capstyle="round",
                zorder=2,
            )
            return stub_tip

    if bbox is not None:
        cx = (bbox.left + bbox.right) / 2.0
        cy = (bbox.top + bbox.bottom) / 2.0
        dx = target_x - cx
        dy = target_y - cy
        if abs(dx) >= abs(dy):
            return (bbox.right if dx >= 0 else bbox.left, cy)
        return (cx, bbox.bottom if dy >= 0 else bbox.top)

    return (target_x, target_y)


def _draw_proteus_component(axis, component, view_box: ViewBox, pixel_scale: float) -> None:
    position = _find_child_local(component, "Position")
    bbox_element = _find_child_local(component, "GraphicBounds")
    if position is None and bbox_element is None:
        return

    bbox = _bbox_from_proteus_element(component, view_box, pixel_scale)
    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    rotation = -_float_attr(position, "rotation") if position is not None else 0.0

    spec = _resolve_proteus_visual_spec(component, bbox)
    if spec is not None:
        drawn_bbox = draw_visual_spec(axis, center_x, center_y, spec, rotation, flip_y=True)
    else:
        drawn_bbox = _draw_rotated_bbox_fallback(axis, bbox, rotation)

    labels = _proteus_labels(component)
    if labels:
        _draw_component_label(axis, drawn_bbox, labels)


def _draw_xmplant_component(axis, component, view_box: ViewBox, pixel_scale: float) -> None:
    bbox = _bbox_from_xmplant_component(component, view_box, pixel_scale)
    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    rotation = -_xmplant_rotation(component)

    spec = _resolve_xmplant_visual_spec(component, bbox)
    if spec is not None:
        drawn_bbox = draw_visual_spec(axis, center_x, center_y, spec, rotation, flip_y=True)
    else:
        drawn_bbox = _draw_rotated_bbox_fallback(axis, bbox, rotation)

    labels = _xmplant_labels(component)
    if labels:
        _draw_component_label(axis, drawn_bbox, labels)


def _bbox_from_proteus_element(component, view_box: ViewBox, pixel_scale: float) -> BBox:
    bbox = _find_child_local(component, "GraphicBounds")
    if bbox is not None:
        left = _to_canvas_x(_float_attr(bbox, "min_x"), view_box, pixel_scale)
        right = _to_canvas_x(_float_attr(bbox, "max_x"), view_box, pixel_scale)
        top = _to_canvas_y(_float_attr(bbox, "max_y"), view_box, pixel_scale)
        bottom = _to_canvas_y(_float_attr(bbox, "min_y"), view_box, pixel_scale)
        return BBox(min(left, right), max(left, right), min(top, bottom), max(top, bottom))

    position = _find_child_local(component, "Position")
    x = _to_canvas_x(_float_attr(position, "x"), view_box, pixel_scale)
    y = _to_canvas_y(_float_attr(position, "y"), view_box, pixel_scale)
    return BBox(x - 6.0, x + 6.0, y - 6.0, y + 6.0)


def _bbox_from_xmplant_component(component, view_box: ViewBox, pixel_scale: float) -> BBox:
    extent = _find_child_local(component, "Extent")
    if extent is None:
        position = _find_child_local(component, "Position")
        location = _find_child_local(position, "Location") if position is not None else None
        if location is None:
            return BBox(0.0, 12.0, 0.0, 12.0)
        x = _to_canvas_x(_float_attr(location, "X"), view_box, pixel_scale)
        y = _to_canvas_y(_float_attr(location, "Y"), view_box, pixel_scale)
        return BBox(x - 6.0, x + 6.0, y - 6.0, y + 6.0)

    min_node = _find_child_local(extent, "Min")
    max_node = _find_child_local(extent, "Max")
    if min_node is None or max_node is None:
        return BBox(0.0, 12.0, 0.0, 12.0)

    left = _to_canvas_x(_float_attr(min_node, "X"), view_box, pixel_scale)
    right = _to_canvas_x(_float_attr(max_node, "X"), view_box, pixel_scale)
    top = _to_canvas_y(_float_attr(max_node, "Y"), view_box, pixel_scale)
    bottom = _to_canvas_y(_float_attr(min_node, "Y"), view_box, pixel_scale)
    return BBox(min(left, right), max(left, right), min(top, bottom), max(top, bottom))


def _resolve_proteus_visual_spec(component, bbox: BBox) -> VisualSpec | None:
    symbol_key = _proteus_generic_attribute(component, "SOURCE_SYMBOL")
    raw_symbol_key = _proteus_generic_attribute(component, "SOURCE_SYMBOL_RAW")
    candidates = [
        symbol_key,
        raw_symbol_key,
        component.get("blockName", ""),
        component.get("componentName", ""),
        component.get("componentSubType", ""),
        component.get("componentClass", ""),
    ]

    file_name = ""
    for candidate in candidates:
        file_name = _resolve_asset_file(candidate)
        if file_name:
            break

    if not file_name:
        return None

    return VisualSpec(
        "dxf",
        max(bbox.width, MIN_SYMBOL_PX),
        max(bbox.height, MIN_SYMBOL_PX),
        file_name,
    )


def _resolve_xmplant_visual_spec(component, bbox: BBox) -> VisualSpec | None:
    candidates = [
        _xmplant_generic_attribute(component, "SOURCE_SYMBOL"),
        _xmplant_generic_attribute(component, "SOURCE_SYMBOL_RAW"),
        _xmplant_generic_attribute(component, "DXF_BLOCK_NAME"),
        component.get("ComponentName", ""),
        component.get("TagName", ""),
        component.get("ComponentClass", ""),
        _xmplant_generic_attribute(component, "SUB_CLASS"),
    ]

    file_name = ""
    for candidate in candidates:
        file_name = _resolve_asset_file(candidate)
        if file_name:
            break

    if not file_name:
        return None

    return VisualSpec(
        "dxf",
        max(bbox.width, MIN_SYMBOL_PX),
        max(bbox.height, MIN_SYMBOL_PX),
        file_name,
    )


def _resolve_asset_file(symbol_key: str) -> str:
    normalized = _normalize_key(symbol_key)
    if not normalized:
        return ""

    candidates = [normalized]
    aliased = PROTEUS_SYMBOL_ALIASES.get(normalized, "")
    if aliased:
        candidates.insert(0, aliased)

    exact_map = _asset_name_map()
    for candidate in candidates:
        if candidate in exact_map:
            return exact_map[candidate]

    best_name = ""
    best_score = 0
    for candidate in candidates:
        candidate_tokens = {
            token for token in candidate.split("_") if len(token) >= 2
        }
        if not candidate_tokens:
            continue
        for asset_key, asset_name in exact_map.items():
            asset_tokens = {token for token in asset_key.split("_") if len(token) >= 2}
            overlap = len(candidate_tokens & asset_tokens)
            if overlap > best_score:
                best_score = overlap
                best_name = asset_name
            elif overlap == best_score and overlap > 0 and best_name:
                if len(asset_name) < len(best_name):
                    best_name = asset_name

    if best_score > 0:
        return best_name
    return ""


def _draw_rotated_bbox_fallback(axis, bbox: BBox, rotation: float) -> BBox:
    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    half_w = bbox.width / 2.0
    half_h = bbox.height / 2.0
    corners = [
        (center_x - half_w, center_y - half_h),
        (center_x + half_w, center_y - half_h),
        (center_x + half_w, center_y + half_h),
        (center_x - half_w, center_y + half_h),
    ]
    rotated = [_rotate_point(point, (center_x, center_y), rotation) for point in corners]
    axis.add_patch(
        Polygon(
            rotated,
            closed=True,
            fill=False,
            edgecolor="#222222",
            linewidth=1.2,
            zorder=3,
        )
    )
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    return BBox(min(xs), max(xs), min(ys), max(ys))


def _draw_component_label(axis, bbox: BBox, labels: list[str]) -> None:
    text = "\n".join(labels[:3])
    axis.text(
        bbox.right + 4.0,
        (bbox.top + bbox.bottom) / 2.0,
        text,
        fontsize=8,
        color="#111111",
        ha="left",
        va="center",
        zorder=5,
    )


def _proteus_segment_style(layer_name: str):
    normalized = (layer_name or "").strip().lower()
    if "signal" in normalized:
        return {"color": "#444444", "linewidth": 1.0, "linestyle": "--"}
    if "instrument" in normalized:
        return {"color": "#666666", "linewidth": 1.0, "linestyle": "--"}
    if "off_page" in normalized:
        return {"color": "#222222", "linewidth": 1.4, "linestyle": "-"}
    return {"color": "#222222", "linewidth": 1.8, "linestyle": "-"}


def _proteus_labels(component) -> list[str]:
    labels = []
    tag_name = (component.get("tagName") or "").strip()
    if tag_name:
        labels.append(tag_name)

    nearby = _find_child_local(component, "NearbyLabels")
    if nearby is not None:
        for label in _find_children_local(nearby, "Label"):
            text = (label.text or "").strip()
            if text and text not in labels:
                labels.append(text)
    return labels


def _xmplant_labels(component) -> list[str]:
    labels = []
    tag_name = (component.get("TagName") or "").strip()
    if tag_name:
        labels.append(tag_name)
    short_tag = _xmplant_generic_attribute(component, "TAG").strip()
    if short_tag and short_tag not in labels:
        labels.append(short_tag)
    return labels


def _proteus_generic_attribute(component, name: str) -> str:
    generic_attributes = _find_child_local(component, "GenericAttributes")
    if generic_attributes is None:
        return ""
    for attribute in _find_children_local(generic_attributes, "GenericAttribute"):
        if attribute.get("Name") == name:
            return attribute.get("Value", "")
    return ""


def _xmplant_generic_attribute(component, name: str) -> str:
    generic_attributes = _find_child_local(component, "GenericAttributes")
    if generic_attributes is None:
        return ""
    for attribute in _find_children_local(generic_attributes, "GenericAttribute"):
        if attribute.get("Name") == name:
            return attribute.get("Value", "")
    return ""


def _xmplant_rotation(component) -> float:
    position = _find_child_local(component, "Position")
    if position is None:
        return 0.0
    reference = _find_child_local(position, "Reference")
    if reference is not None:
        ref_x = _float_attr(reference, "X", 1.0)
        ref_y = _float_attr(reference, "Y", 0.0)
        return math.degrees(math.atan2(ref_y, ref_x))
    return 0.0


def _draw_xmplant_segment(axis, segment, bbox_map) -> None:
    connection = _find_child_local(segment, "Connection")
    if connection is None:
        return

    from_id = connection.get("FromID", "")
    to_id = connection.get("ToID", "")
    from_box = bbox_map.get(from_id)
    to_box = bbox_map.get(to_id)
    if from_box is None or to_box is None:
        return

    start = _bbox_anchor(from_box, to_box)
    end = _bbox_anchor(to_box, from_box)
    points = _orthogonal_route(start, end)
    style = _xmplant_segment_style(segment)
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        solid_capstyle="round",
        zorder=1,
    )

    nominal_diameter = _xmplant_segment_attr(segment, "NOMINAL_DIAMETER")
    if nominal_diameter:
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        axis.text(
            mid_x, mid_y + 6,
            nominal_diameter,
            fontsize=7,
            color=style["color"],
            ha="center",
            va="top",
            zorder=3,
        )


def _build_port_map(components: dict, view_box: ViewBox, pixel_scale: float) -> dict:
    port_map = {}
    for component_id, component in components.items():
        ports_elem = _find_child_local(component, "Ports")
        if ports_elem is None:
            port_map[component_id] = []
            continue
        ports = []
        for port in _find_children_local(ports_elem, "Port"):
            px = _to_canvas_x(_float_attr(port, "x"), view_box, pixel_scale)
            py = _to_canvas_y(_float_attr(port, "y"), view_box, pixel_scale)
            ports.append((px, py))
        port_map[component_id] = ports
    return port_map


def _find_closest_port(ports: list, target_x: float, target_y: float) -> tuple[float, float]:
    return min(ports, key=lambda p: (p[0] - target_x) ** 2 + (p[1] - target_y) ** 2)


def _stub_endpoint(port_x: float, port_y: float, bbox: BBox) -> tuple[float, float]:
    cx = (bbox.left + bbox.right) / 2.0
    cy = (bbox.top + bbox.bottom) / 2.0
    dx = port_x - cx
    dy = port_y - cy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        return port_x + STUB_LENGTH_PX, port_y
    # Use max(STUB_LENGTH_PX, 30% of component half-size) so stubs are visible on larger symbols
    component_half = max(bbox.width, bbox.height) / 2.0
    length = max(STUB_LENGTH_PX, component_half * 0.3)
    return port_x + dx / dist * length, port_y + dy / dist * length


def _bbox_anchor(box: BBox, other: BBox) -> tuple[float, float]:
    cx = (box.left + box.right) / 2.0
    cy = (box.top + box.bottom) / 2.0
    ox = (other.left + other.right) / 2.0
    oy = (other.top + other.bottom) / 2.0
    dx = ox - cx
    dy = oy - cy
    if abs(dx) >= abs(dy):
        return (box.right, cy) if dx >= 0 else (box.left, cy)
    return (cx, box.bottom) if dy >= 0 else (cx, box.top)


def _orthogonal_route(
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    sx, sy = start
    ex, ey = end
    if math.isclose(sx, ex, abs_tol=0.1) or math.isclose(sy, ey, abs_tol=0.1):
        return [start, end]

    bend = (ex, sy)
    return [start, bend, end]


def _xmplant_segment_style(segment):
    connection_type = _xmplant_segment_attr(segment, "CONNECTION_TYPE").lower()
    sub_class = _xmplant_segment_attr(segment, "SUB_CLASS").lower()
    normalized = f"{connection_type} {sub_class}"
    if "signal" in normalized:
        return {"color": "#444444", "linewidth": 1.0, "linestyle": "--"}
    if "instrument" in normalized:
        return {"color": "#666666", "linewidth": 1.0, "linestyle": "--"}
    if "off-page" in normalized or "off_page" in normalized:
        return {"color": "#222222", "linewidth": 1.4, "linestyle": "-"}
    return {"color": "#222222", "linewidth": 1.8, "linestyle": "-"}


def _xmplant_segment_attr(segment, name: str) -> str:
    generic_attributes = _find_child_local(segment, "GenericAttributes")
    if generic_attributes is None:
        return ""
    for attribute in _find_children_local(generic_attributes, "GenericAttribute"):
        if attribute.get("Name") == name:
            return attribute.get("Value", "")
    return ""


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


def _to_canvas(
    point_x: float,
    point_y: float,
    view_box: ViewBox,
    pixel_scale: float,
) -> tuple[float, float]:
    return (
        _to_canvas_x(point_x, view_box, pixel_scale),
        _to_canvas_y(point_y, view_box, pixel_scale),
    )


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


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _iter_local(root, name: str):
    for element in root.iter():
        if _local_name(element.tag) == name:
            yield element


def _find_child_local(element, name: str):
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _find_children_local(element, name: str):
    return [child for child in list(element) if _local_name(child.tag) == name]


def _normalize_key(value: str) -> str:
    value = _camel_to_snake(value or "")
    cleaned = []
    for char in value.strip().lower():
        cleaned.append(char if char.isalnum() else "_")
    normalized = "".join(cleaned)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _camel_to_snake(value: str) -> str:
    chars = []
    previous = ""
    for char in value:
        if previous and previous.islower() and char.isupper():
            chars.append("_")
        chars.append(char)
        previous = char
    return "".join(chars)


def _rotate_point(
    point: tuple[float, float],
    center: tuple[float, float],
    rotation_deg: float,
) -> tuple[float, float]:
    angle = math.radians(rotation_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    px = point[0] - center[0]
    py = point[1] - center[1]
    return (
        center[0] + px * cos_a - py * sin_a,
        center[1] + px * sin_a + py * cos_a,
    )


def _asset_name_map():
    return _asset_name_map_cached()


@lru_cache(maxsize=1)
def _asset_name_map_cached():
    return {
        _normalize_key(path.stem): path.name
        for path in ASSET_DIR.glob("*.dxf")
    }
