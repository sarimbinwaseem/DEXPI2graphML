---
name: arrowhead-connection-fix
description: TODO reminder — connections to horizontal arrowheads must arrive at the horizontal midpoint (left/right side, center Y), not the top/bottom
metadata:
  type: project
---

Connections routing to arrowhead (PipeFlowArrow) components are incorrect: when the opposite endpoint is more vertical than horizontal, `_resolve_port_anchor` in `dexpi_xml_renderer.py:550` returns `(cx, top/bottom)` instead of `(left/right, cy)`. This causes the pipe line to enter the arrowhead from its top or bottom instead of its left or right side at center Y.

**Why:** Arrowheads are horizontal symbols — they should always accept connections from the left or right side (horizontal midpoint). The current `_bbox_anchor` / `_resolve_port_anchor` logic picks the anchor side purely by dominant direction to the other endpoint, which fails when the other component is above/below.

**How to apply:** When fixing this, two things are needed:
1. Detect arrowhead components (by `componentClass`, `blockName`, or resolved visual spec containing "arrow_head"). Force `(left_or_right, cy)` anchor in `_resolve_port_anchor` for those.
2. Fix `_orthogonal_route` to use `bend = (sx, ey)` (vertical-first, then horizontal) when the end anchor is on a left/right edge — so the pipe arrives horizontally, not vertically, at the arrowhead.
