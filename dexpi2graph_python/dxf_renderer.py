from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import ezdxf
from ezdxf import recover
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = BASE_DIR / "assets" / "dxf_components"
MANIFEST_PATH = ASSET_DIR / "component_map.json"
MIN_GEOMETRY_SIZE = 1e-6


@dataclass(frozen=True)
class VisualSpec:
    kind: str
    width: float
    height: float
    file_name: str = ""


@dataclass(frozen=True)
class BBox:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom


@dataclass(frozen=True)
class SymbolGeometry:
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    bbox: BBox
    center: tuple[float, float]


def render_graph_plot(path_graph: str, path_plot_stem: str) -> None:
    graph = nx.read_graphml(path_graph)
    stem = _normalized_output_stem(path_plot_stem)
    figure, axis = _build_figure(graph)
    placed_nodes = _draw_nodes(axis, graph)
    _draw_edges(axis, graph, placed_nodes)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(_output_file(stem, ".png"), bbox_inches="tight", pad_inches=0.1)
    figure.savefig(_output_file(stem, ".svg"), bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def draw_visual_spec(
    axis,
    x: float,
    y: float,
    spec: VisualSpec,
    rotation: float = 0.0,
) -> BBox:
    return _draw_symbol(axis, x, y, spec, rotation)


def _normalized_output_stem(path_plot_stem: str) -> Path:
    path = Path(path_plot_stem)
    if path.suffix.lower() in {".png", ".svg"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _output_file(stem: Path, extension: str) -> Path:
    return Path(f"{stem}{extension}")


def _build_figure(graph: nx.Graph):
    xs = []
    ys = []
    for _, data in graph.nodes(data=True):
        xs.append(float(data.get("node_x", 0)))
        ys.append(float(data.get("node_y", 0)))
    if not xs:
        xs = [0.0]
        ys = [0.0]
    padding = 35.0
    width = max(xs) - min(xs) + 2 * padding
    height = max(ys) - min(ys) + 2 * padding
    figure = plt.figure(figsize=(max(8.0, width / 18.0), max(6.0, height / 18.0)))
    axis = figure.add_subplot(111)
    axis.set_xlim(min(xs) - padding, max(xs) + padding)
    axis.set_ylim(min(ys) - padding, max(ys) + padding)
    return figure, axis


def _draw_nodes(axis, graph: nx.Graph):
    placed_nodes = {}
    manifest = _load_manifest()
    positions = {
        node: (
            float(data.get("node_x", 0.0)),
            float(data.get("node_y", 0.0)),
        )
        for node, data in graph.nodes(data=True)
    }

    for node, data in graph.nodes(data=True):
        spec = _resolve_visual_spec(data, manifest)
        x, y = positions[node]
        rotation = _resolve_rotation(node, graph, positions, spec)
        bbox = _draw_symbol(axis, x, y, spec, rotation)
        label = data.get("node_name") or node
        axis.text(
            bbox.right + 1.5,
            bbox.top + 1.5,
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        placed_nodes[node] = {
            "bbox": bbox,
            "center": (x, y),
        }
    return placed_nodes


def _draw_edges(axis, graph: nx.Graph, placed_nodes):
    for from_id, to_id, data in graph.edges(data=True):
        start_box = placed_nodes[from_id]["bbox"]
        end_box = placed_nodes[to_id]["bbox"]
        start = placed_nodes[from_id]["center"]
        end = placed_nodes[to_id]["center"]
        points = _route_edge(start, end, start_box, end_box)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        edge_class = data.get("edge_class", "")
        edge_sub_class = data.get("edge_sub_class", "")
        style = _edge_style(edge_class, edge_sub_class)
        axis.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            solid_capstyle="round",
        )


def _resolve_rotation(node: str, graph: nx.Graph, positions, spec: VisualSpec) -> float:
    if spec.kind not in {"dxf", "valve", "tee"}:
        return 0.0

    x0, y0 = positions[node]
    dx_total = 0.0
    dy_total = 0.0
    neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
    for other in neighbors:
        x1, y1 = positions[other]
        dx_total += abs(x1 - x0)
        dy_total += abs(y1 - y0)
    if dy_total > dx_total:
        return 90.0
    return 0.0


def _draw_symbol(axis, x: float, y: float, spec: VisualSpec, rotation: float) -> BBox:
    if spec.kind == "dxf" and spec.file_name:
        geometry = _load_symbol_geometry(spec.file_name)
        if geometry is not None:
            return _draw_dxf_symbol(axis, geometry, x, y, spec.width, spec.height, rotation)
    return _draw_primitive(axis, spec, x, y, rotation)


def _draw_dxf_symbol(
    axis,
    geometry: SymbolGeometry,
    x: float,
    y: float,
    target_width: float,
    target_height: float,
    rotation: float,
) -> BBox:
    source_width = max(geometry.bbox.width, MIN_GEOMETRY_SIZE)
    source_height = max(geometry.bbox.height, MIN_GEOMETRY_SIZE)
    line_segments = []
    cos_r = math.cos(math.radians(rotation))
    sin_r = math.sin(math.radians(rotation))
    effective_width = max(
        source_width * abs(cos_r) + source_height * abs(sin_r),
        MIN_GEOMETRY_SIZE,
    )
    effective_height = max(
        source_width * abs(sin_r) + source_height * abs(cos_r),
        MIN_GEOMETRY_SIZE,
    )
    scale = min(target_width / effective_width, target_height / effective_height)

    transformed_points = []
    for start, end in geometry.segments:
        transformed_segment = []
        for px, py in (start, end):
            local_x = (px - geometry.center[0]) * scale
            local_y = (py - geometry.center[1]) * scale
            rot_x = local_x * cos_r - local_y * sin_r
            rot_y = local_x * sin_r + local_y * cos_r
            world_point = (x + rot_x, y + rot_y)
            transformed_segment.append(world_point)
            transformed_points.append(world_point)
        line_segments.append(tuple(transformed_segment))

    if line_segments:
        axis.add_collection(
            LineCollection(line_segments, colors="black", linewidths=1.2, zorder=3)
        )

    if transformed_points:
        xs = [point[0] for point in transformed_points]
        ys = [point[1] for point in transformed_points]
        return BBox(min(xs), max(xs), min(ys), max(ys))

    return BBox(
        x - target_width / 2.0,
        x + target_width / 2.0,
        y - target_height / 2.0,
        y + target_height / 2.0,
    )


def _draw_primitive(axis, spec: VisualSpec, x: float, y: float, rotation: float) -> BBox:
    half_w = spec.width / 2.0
    half_h = spec.height / 2.0

    if spec.kind == "instrument":
        axis.add_patch(Circle((x, y), radius=half_w, fill=False, linewidth=1.3, zorder=3))
    elif spec.kind == "connector":
        points = [(x - half_w, y - half_h), (x - half_w, y + half_h), (x + half_w, y)]
        axis.add_patch(Polygon(points, fill=False, linewidth=1.3, zorder=3))
    elif spec.kind == "tee":
        axis.add_patch(Rectangle((x - 1.2, y - 1.2), 2.4, 2.4, fill=True, color="black", zorder=3))
        return BBox(x - 1.2, x + 1.2, y - 1.2, y + 1.2)
    elif spec.kind == "pump":
        axis.add_patch(Circle((x, y), radius=min(half_w, half_h), fill=False, linewidth=1.3, zorder=3))
        axis.plot([x - half_w, x + half_w], [y, y], color="black", linewidth=1.0, zorder=3)
        axis.plot([x + half_w * 0.2, x + half_w], [y, y + half_h * 0.45], color="black", linewidth=1.0, zorder=3)
    elif spec.kind == "heat_exchanger":
        axis.add_patch(Rectangle((x - half_w, y - half_h), spec.width, spec.height, fill=False, linewidth=1.3, zorder=3))
        axis.plot([x - half_w, x + half_w], [y - half_h, y + half_h], color="black", linewidth=1.0, zorder=3)
        axis.plot([x - half_w, x + half_w], [y + half_h, y - half_h], color="black", linewidth=1.0, zorder=3)
    elif spec.kind == "column":
        axis.add_patch(Rectangle((x - half_w, y - half_h), spec.width, spec.height, fill=False, linewidth=1.3, zorder=3))
        axis.add_patch(Circle((x, y + half_h), radius=half_w, fill=False, linewidth=1.0, zorder=3))
        axis.add_patch(Circle((x, y - half_h), radius=half_w, fill=False, linewidth=1.0, zorder=3))
    else:
        axis.add_patch(Rectangle((x - half_w, y - half_h), spec.width, spec.height, fill=False, linewidth=1.3, zorder=3))

    return BBox(x - half_w, x + half_w, y - half_h, y + half_h)


def _edge_style(edge_class: str, edge_sub_class: str):
    normalized = (edge_class or "").lower()
    if "signal" in normalized:
        return {"color": "#444444", "linewidth": 1.0, "linestyle": "--"}
    if "process connection" in normalized:
        return {"color": "#666666", "linewidth": 1.0, "linestyle": "--"}
    if edge_class == "Piping" and edge_sub_class == "Secondary pipe":
        return {"color": "#4f4f4f", "linewidth": 1.4, "linestyle": "-"}
    return {"color": "#222222", "linewidth": 1.8, "linestyle": "-"}


def _route_edge(start, end, start_box: BBox, end_box: BBox):
    sx, sy = start
    tx, ty = end
    start_anchor = _choose_anchor(start_box, sx, sy, tx, ty)
    end_anchor = _choose_anchor(end_box, tx, ty, sx, sy)

    if math.isclose(start_anchor[0], end_anchor[0], abs_tol=0.1) or math.isclose(
        start_anchor[1], end_anchor[1], abs_tol=0.1
    ):
        return [start_anchor, end_anchor]

    bend_hv = (end_anchor[0], start_anchor[1])
    bend_vh = (start_anchor[0], end_anchor[1])
    valid_hv = not (_inside_bbox(bend_hv, start_box) or _inside_bbox(bend_hv, end_box))
    valid_vh = not (_inside_bbox(bend_vh, start_box) or _inside_bbox(bend_vh, end_box))

    if valid_hv and not valid_vh:
        return [start_anchor, bend_hv, end_anchor]
    if valid_vh and not valid_hv:
        return [start_anchor, bend_vh, end_anchor]

    length_hv = abs(end_anchor[0] - start_anchor[0]) + abs(end_anchor[1] - start_anchor[1])
    length_vh = length_hv
    if valid_hv and valid_vh:
        if length_hv <= length_vh:
            return [start_anchor, bend_hv, end_anchor]
        return [start_anchor, bend_vh, end_anchor]

    if abs(end_anchor[0] - start_anchor[0]) >= abs(end_anchor[1] - start_anchor[1]):
        mid_x = (start_anchor[0] + end_anchor[0]) / 2.0
        return [start_anchor, (mid_x, start_anchor[1]), (mid_x, end_anchor[1]), end_anchor]
    mid_y = (start_anchor[1] + end_anchor[1]) / 2.0
    return [start_anchor, (start_anchor[0], mid_y), (end_anchor[0], mid_y), end_anchor]


def _choose_anchor(box: BBox, x0: float, y0: float, x1: float, y1: float):
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) >= abs(dy):
        return (box.right, y0) if dx >= 0 else (box.left, y0)
    return (x0, box.top) if dy >= 0 else (x0, box.bottom)


def _inside_bbox(point, box: BBox, padding: float = 1.0):
    px, py = point
    return (
        box.left - padding < px < box.right + padding
        and box.bottom - padding < py < box.top + padding
    )


def _resolve_visual_spec(data, manifest) -> VisualSpec:
    request = (data.get("node_request") or "").strip()
    node_class = (data.get("node_class") or "").strip()
    node_group = (data.get("node_group") or "").strip()

    for group_name, mapping in [
        ("node_request", request),
        ("node_class", node_class),
        ("node_group", node_group),
    ]:
        if mapping:
            spec = manifest.get(group_name, {}).get(mapping)
            if spec:
                return _spec_from_manifest(spec)

    if node_group == "MSR":
        return VisualSpec("instrument", 10.0, 10.0)
    if node_group == "Valves/Fittings":
        return VisualSpec("valve", 10.0, 10.0)
    if node_group == "Vessel":
        return VisualSpec("equipment", 24.0, 18.0)
    if node_group == "Column":
        return VisualSpec("column", 18.0, 34.0)
    if node_group == "Heat exchanger":
        return VisualSpec("heat_exchanger", 22.0, 12.0)
    if node_group == "Pump":
        return VisualSpec("pump", 16.0, 12.0)
    if node_group == "Connector":
        return VisualSpec("connector", 10.0, 8.0)
    return VisualSpec("equipment", 12.0, 10.0)


def _spec_from_manifest(spec):
    size = spec.get("size", [12.0, 10.0])
    return VisualSpec(
        kind=spec.get("kind", "dxf"),
        width=float(size[0]),
        height=float(size[1]),
        file_name=spec.get("file", ""),
    )


@lru_cache(maxsize=1)
def _load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


@lru_cache(maxsize=256)
def _load_symbol_geometry(file_name: str):
    path = ASSET_DIR / file_name
    if not path.exists():
        return None
    doc = _load_dxf(path)
    modelspace = doc.modelspace()
    segments = []
    for entity in modelspace:
        _collect_segments(entity, segments)
    if not segments:
        return None
    points = [point for segment in segments for point in segment]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = BBox(min(xs), max(xs), min(ys), max(ys))
    center = ((bbox.left + bbox.right) / 2.0, (bbox.bottom + bbox.top) / 2.0)
    return SymbolGeometry(tuple(segments), bbox, center)


def _load_dxf(path: Path):
    try:
        return ezdxf.readfile(str(path))
    except ezdxf.DXFError:
        doc, _ = recover.readfile(str(path), errors="ignore")
        return doc


def _collect_segments(entity, segments):
    entity_type = entity.dxftype()
    if entity_type == "INSERT":
        try:
            for virtual in entity.virtual_entities():
                _collect_segments(virtual, segments)
        except Exception:
            return
        return

    if entity_type == "LINE":
        segments.append(
            (
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            )
        )
        return

    if entity_type == "LWPOLYLINE":
        points = [(float(point[0]), float(point[1])) for point in entity.get_points("xy")]
        _points_to_segments(points, bool(entity.closed), segments)
        return

    if entity_type == "POLYLINE":
        points = []
        try:
            points = [
                (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
                for vertex in entity.vertices
            ]
        except Exception:
            points = []
        _points_to_segments(points, bool(entity.is_closed), segments)
        return

    if entity_type == "ARC":
        center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        radius = float(entity.dxf.radius)
        start_angle = float(entity.dxf.start_angle)
        end_angle = float(entity.dxf.end_angle)
        points = _sample_arc(center, radius, start_angle, end_angle)
        _points_to_segments(points, False, segments)
        return

    if entity_type == "CIRCLE":
        center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        radius = float(entity.dxf.radius)
        points = _sample_arc(center, radius, 0.0, 360.0)
        _points_to_segments(points, True, segments)
        return

    if entity_type == "ELLIPSE":
        try:
            points = [(float(point.x), float(point.y)) for point in entity.flattening(0.5)]
            _points_to_segments(points, False, segments)
        except Exception:
            return
        return

    if entity_type == "SPLINE":
        try:
            points = [(float(point.x), float(point.y)) for point in entity.flattening(0.5)]
            _points_to_segments(points, False, segments)
        except Exception:
            return


def _sample_arc(center, radius, start_angle, end_angle, step_degrees: float = 12.0):
    start = start_angle
    end = end_angle
    if end <= start:
        end += 360.0
    steps = max(8, int(math.ceil((end - start) / step_degrees)))
    points = []
    for index in range(steps + 1):
        angle = math.radians(start + (end - start) * (index / steps))
        points.append(
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
        )
    return points


def _points_to_segments(points, closed: bool, segments):
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        segments.append((start, end))
    if closed and len(points) > 2:
        segments.append((points[-1], points[0]))
