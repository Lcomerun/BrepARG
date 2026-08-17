"""Fail-closed repair for a diagnosed self-intersecting face wire.

The helper deliberately works on a copied face. OCC's topology-changing wire
fixer can mutate shared edge topology and, for malformed inputs, may terminate
the native process instead of raising a Python exception. Callers therefore
use this helper from one-CAD worker processes and accept a candidate only when
the repaired face has no remaining strict wire intersection and passes the OCC
native analyzer. A rejected candidate returns the original face unchanged.
"""

from __future__ import annotations

import itertools
import math
from typing import Any


# A copied-face candidate may absorb tolerance noise, but it may not materially
# reshape a CAD face.  Area and perimeter are limited to 0.5 percent change;
# axis-aligned bounds are limited to 0.1 percent of the coordinate scale.
MAX_AREA_RELATIVE_DELTA = 0.005
MAX_BOUNDARY_LENGTH_RELATIVE_DELTA = 0.005
MAX_BBOX_RELATIVE_DELTA = 0.001
MAX_PCURVE_BRANCH_SHIFT = 2
PCURVE_GAP_TOLERANCE = 1e-7
CURVE_SAMPLE_TOLERANCE = 1e-10


def _uv_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _shift_uv(
    point: tuple[float, float],
    offset: tuple[int, int],
    periods: tuple[float | None, float | None],
) -> tuple[float, float]:
    return (
        point[0] + offset[0] * (periods[0] or 0.0),
        point[1] + offset[1] * (periods[1] or 0.0),
    )


def select_periodic_pcurve_branches(
    edge_endpoints: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    u_period: float | None,
    v_period: float | None,
    movable: list[bool] | None = None,
    max_period_shift: int = MAX_PCURVE_BRANCH_SHIFT,
) -> dict[str, Any]:
    """Choose integer-period translations that close an ordered UV wire.

    Each endpoint pair is already oriented in wire traversal order.  The
    dynamic program minimizes the squared gaps between consecutive edges and
    the closing gap.  Translation magnitude is the deterministic tie-breaker,
    which avoids an arbitrary whole-wire period shift when zero is equivalent.
    """
    count = len(edge_endpoints)
    periods = (
        float(u_period) if u_period is not None else None,
        float(v_period) if v_period is not None else None,
    )
    if count == 0:
        return {
            "solved": False,
            "reason": "empty_wire",
            "offsets": [],
            "before_max_gap": None,
            "after_max_gap": None,
        }
    if any(period is not None and (not math.isfinite(period) or period <= 0.0) for period in periods):
        raise ValueError("periods must be finite and positive")
    if periods == (None, None):
        return {
            "solved": False,
            "reason": "surface_not_periodic",
            "offsets": [(0, 0)] * count,
            "before_max_gap": None,
            "after_max_gap": None,
        }
    if max_period_shift < 0:
        raise ValueError("max_period_shift must be non-negative")
    if movable is None:
        movable = [True] * count
    if len(movable) != count:
        raise ValueError("movable must match edge_endpoints")
    normalized: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for pair in edge_endpoints:
        if len(pair) != 2 or any(len(point) != 2 for point in pair):
            raise ValueError("every edge must have two UV endpoints")
        converted = tuple(
            (float(point[0]), float(point[1])) for point in pair
        )
        if not all(math.isfinite(value) for point in converted for value in point):
            raise ValueError("UV endpoints must be finite")
        normalized.append(converted)

    axis_values = [
        range(-max_period_shift, max_period_shift + 1) if period is not None else (0,)
        for period in periods
    ]
    all_offsets = [tuple(values) for values in itertools.product(*axis_values)]
    edge_states = [all_offsets if can_move else [(0, 0)] for can_move in movable]
    best: tuple[tuple[Any, ...], list[tuple[int, int]], list[float]] | None = None
    for initial in edge_states[0]:
        # state -> (open-chain squared cost, L1 shift cost, path, gaps)
        states: dict[tuple[int, int], tuple[float, int, list[tuple[int, int]], list[float]]] = {
            initial: (0.0, abs(initial[0]) + abs(initial[1]), [initial], [])
        }
        for edge_index in range(1, count):
            next_states: dict[tuple[int, int], tuple[float, int, list[tuple[int, int]], list[float]]] = {}
            current_start = normalized[edge_index][0]
            for offset in edge_states[edge_index]:
                shifted_start = _shift_uv(current_start, offset, periods)
                candidate: tuple[float, int, list[tuple[int, int]], list[float]] | None = None
                for previous_offset, (cost, shift_cost, path, gaps) in states.items():
                    previous_end = _shift_uv(
                        normalized[edge_index - 1][1], previous_offset, periods
                    )
                    gap = _uv_distance(previous_end, shifted_start)
                    row = (
                        cost + gap * gap,
                        shift_cost + abs(offset[0]) + abs(offset[1]),
                        path + [offset],
                        gaps + [gap],
                    )
                    row_key = (row[0], row[1], tuple(row[2]))
                    if candidate is None or row_key < (
                        candidate[0], candidate[1], tuple(candidate[2])
                    ):
                        candidate = row
                assert candidate is not None
                next_states[offset] = candidate
            states = next_states
        for final_offset, (cost, shift_cost, path, gaps) in states.items():
            final_end = _shift_uv(normalized[-1][1], final_offset, periods)
            initial_start = _shift_uv(normalized[0][0], initial, periods)
            closing_gap = _uv_distance(final_end, initial_start)
            all_gaps = gaps + [closing_gap]
            total_cost = cost + closing_gap * closing_gap
            key = (total_cost, max(all_gaps), shift_cost, tuple(path))
            if best is None or key < best[0]:
                best = (key, path, all_gaps)
    assert best is not None
    before_gaps = [
        _uv_distance(normalized[index - 1][1], normalized[index][0])
        for index in range(1, count)
    ]
    before_gaps.append(_uv_distance(normalized[-1][1], normalized[0][0]))
    offsets = best[1]
    return {
        "solved": True,
        "reason": "optimized",
        "offsets": offsets,
        "changed_edge_indices": [
            index for index, offset in enumerate(offsets) if offset != (0, 0)
        ],
        "before_gaps": before_gaps,
        "after_gaps": best[2],
        "before_max_gap": max(before_gaps),
        "after_max_gap": max(best[2]),
        "objective_squared_gap": best[0][0],
        "periods": [periods[0], periods[1]],
        "max_period_shift": int(max_period_shift),
    }


def face_geometry_signature(face: Any) -> dict[str, Any]:
    """Measure topology and 3D geometry that a local pcurve fix must preserve."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Extend.TopologyUtils import TopologyExplorer

    topology = TopologyExplorer(face)
    wire_count = sum(1 for _ in topology.wires())
    edge_count = sum(1 for _ in TopologyExplorer(face).edges())
    surface = GProp_GProps()
    boundary = GProp_GProps()
    brepgprop.SurfaceProperties(face, surface)
    brepgprop.LinearProperties(face, boundary)
    box = Bnd_Box()
    brepbndlib.Add(face, box)
    if box.IsVoid():
        raise ValueError("face has a void bounding box")
    bounds = tuple(float(value) for value in box.Get())
    return {
        "wire_count": int(wire_count),
        "edge_count": int(edge_count),
        "area": abs(float(surface.Mass())),
        "boundary_length": abs(float(boundary.Mass())),
        "bbox": list(bounds),
    }


def _relative_delta(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), 1e-12
    )


def geometry_preservation_gate(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when a topology repair changes the represented boundary."""
    area_relative_delta = _relative_delta(before["area"], after["area"])
    boundary_relative_delta = _relative_delta(
        before["boundary_length"], after["boundary_length"]
    )
    before_bbox = [float(value) for value in before["bbox"]]
    after_bbox = [float(value) for value in after["bbox"]]
    bbox_scale = max(
        max(abs(value) for value in before_bbox + after_bbox),
        1.0,
    )
    bbox_relative_delta = max(
        abs(first - second) for first, second in zip(before_bbox, after_bbox)
    ) / bbox_scale
    checks = {
        "wire_count_equal": before["wire_count"] == after["wire_count"],
        "edge_count_equal": before["edge_count"] == after["edge_count"],
        "positive_area": before["area"] > 0.0 and after["area"] > 0.0,
        "area_within_tolerance": area_relative_delta <= MAX_AREA_RELATIVE_DELTA,
        "boundary_length_within_tolerance": (
            boundary_relative_delta <= MAX_BOUNDARY_LENGTH_RELATIVE_DELTA
        ),
        "bbox_within_tolerance": bbox_relative_delta <= MAX_BBOX_RELATIVE_DELTA,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "area_relative_delta": area_relative_delta,
        "boundary_length_relative_delta": boundary_relative_delta,
        "bbox_relative_delta": bbox_relative_delta,
        "thresholds": {
            "max_area_relative_delta": MAX_AREA_RELATIVE_DELTA,
            "max_boundary_length_relative_delta": MAX_BOUNDARY_LENGTH_RELATIVE_DELTA,
            "max_bbox_relative_delta": MAX_BBOX_RELATIVE_DELTA,
        },
    }


def wire_self_intersection_state(face: Any) -> dict[str, Any]:
    """Return strict-style self-intersection observations for every face wire."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Extend.TopologyUtils import TopologyExplorer

    bad_wire_indices: list[int] = []
    checked = 0
    for wire_index, wire in enumerate(TopologyExplorer(face).wires()):
        fixer = ShapeFix_Wire(wire, face, 0.01)
        fixer.Load(wire)
        fixer.SetFace(face)
        fixer.SetPrecision(0.01)
        fixer.SetMaxTolerance(1.0)
        fixer.SetMinTolerance(1e-4)
        fixer.Perform()
        analysis = ShapeAnalysis_Wire(fixer.Wire(), face, 0.01)
        analysis.Load(fixer.Wire())
        analysis.SetPrecision(0.01)
        analysis.SetSurface(BRep_Tool.Surface(face))
        checked += 1
        if bool(analysis.CheckSelfIntersection()):
            bad_wire_indices.append(int(wire_index))
    return {
        "checked_wires": checked,
        "bad_wire_indices": bad_wire_indices,
        "self_intersection_count": len(bad_wire_indices),
    }


def face_topology_incidence_signature(face: Any) -> dict[str, Any]:
    """Return orientation-independent discrete face incidence evidence."""
    from OCC.Extend.TopologyUtils import TopologyExplorer

    wires = list(TopologyExplorer(face).wires())
    edges = list(TopologyExplorer(face).edges())
    vertices: list[Any] = []
    vertex_degrees: list[int] = []
    for edge in edges:
        for vertex in TopologyExplorer(edge).vertices():
            existing = next(
                (index for index, known in enumerate(vertices) if vertex.IsSame(known)),
                None,
            )
            if existing is None:
                vertices.append(vertex)
                vertex_degrees.append(1)
            else:
                vertex_degrees[existing] += 1
    return {
        "wire_count": len(wires),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "wire_edge_counts": sorted(
            sum(1 for _ in TopologyExplorer(wire).edges()) for wire in wires
        ),
        "edge_vertex_counts": sorted(
            sum(1 for _ in TopologyExplorer(edge).vertices()) for edge in edges
        ),
        "vertex_edge_degrees": sorted(vertex_degrees),
    }


def _edge_curve_samples(edge: Any, *, sample_count: int = 9) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool

    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        return {"available": False}
    values = []
    for index in range(sample_count):
        parameter = first + (last - first) * index / max(sample_count - 1, 1)
        point = curve.Value(parameter)
        values.append((float(point.X()), float(point.Y()), float(point.Z())))
    return {
        "available": True,
        "curve_type": str(curve.DynamicType().Name()),
        "range": (float(first), float(last)),
        "samples": values,
    }


def corresponding_3d_curve_gate(
    original_edges: list[Any], copied_edges: list[Any]
) -> dict[str, Any]:
    """Prove that pcurve surgery did not alter corresponding 3D curves."""
    if len(original_edges) != len(copied_edges):
        return {
            "accepted": False,
            "reason": "edge_count_mismatch",
            "edge_count_before": len(original_edges),
            "edge_count_after": len(copied_edges),
        }
    maximum_delta = 0.0
    rows = []
    accepted = True
    for index, (original, copied) in enumerate(zip(original_edges, copied_edges)):
        before = _edge_curve_samples(original)
        after = _edge_curve_samples(copied)
        row = {
            "edge_index": index,
            "availability_equal": before["available"] == after["available"],
            "curve_type_equal": True,
            "parameter_range_equal": True,
            "max_sample_delta": 0.0,
        }
        if not row["availability_equal"]:
            accepted = False
        elif before["available"]:
            row["curve_type_equal"] = (
                before["curve_type"] == after["curve_type"]
            )
            row["parameter_range_equal"] = before["range"] == after["range"]
            deltas = [
                math.dist(first, second)
                for first, second in zip(before["samples"], after["samples"])
            ]
            row["max_sample_delta"] = max(deltas, default=0.0)
            maximum_delta = max(maximum_delta, row["max_sample_delta"])
            accepted = bool(
                accepted
                and row["curve_type_equal"]
                and row["parameter_range_equal"]
                and row["max_sample_delta"] <= CURVE_SAMPLE_TOLERANCE
            )
        rows.append(row)
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "curve_changed",
        "max_sample_delta": maximum_delta,
        "sample_tolerance": CURVE_SAMPLE_TOLERANCE,
        "edges": rows,
    }


def _periodic_pcurve_continuity_data(face: Any) -> dict[str, Any]:
    """Collect UV continuity plus private OCC edge handles."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Extend.TopologyUtils import TopologyExplorer

    surface = BRep_Tool.Surface(face)
    u_period = float(surface.UPeriod()) if surface.IsUPeriodic() else None
    v_period = float(surface.VPeriod()) if surface.IsVPeriodic() else None
    wires = []
    for wire_index, wire in enumerate(TopologyExplorer(face).wires()):
        explorer = BRepTools_WireExplorer(wire, face)
        endpoints = []
        movable = []
        edges = []
        while explorer.More():
            edge = explorer.Current()
            pcurve, first, last = BRep_Tool.CurveOnSurface(edge, face)
            if pcurve is None:
                return {
                    "available": False,
                    "reason": "missing_pcurve",
                    "wire_index": wire_index,
                }
            first_point = pcurve.Value(first)
            last_point = pcurve.Value(last)
            pair = (
                (float(first_point.X()), float(first_point.Y())),
                (float(last_point.X()), float(last_point.Y())),
            )
            if edge.Orientation() == TopAbs_REVERSED:
                pair = (pair[1], pair[0])
            endpoints.append(pair)
            movable.append(not bool(BRep_Tool.IsClosed(edge, face)))
            edges.append(edge)
            explorer.Next()
        plan = select_periodic_pcurve_branches(
            endpoints,
            u_period=u_period,
            v_period=v_period,
            movable=movable,
        )
        wires.append(
            {
                "wire_index": wire_index,
                "edge_count": len(edges),
                "edges": edges,
                "endpoints": endpoints,
                "movable": movable,
                "plan": plan,
            }
        )
    return {
        "available": True,
        "reason": "measured",
        "periods": [u_period, v_period],
        "wires": wires,
        "max_gap": max(
            (wire["plan"]["before_max_gap"] or 0.0 for wire in wires),
            default=0.0,
        ),
    }


def _public_periodic_pcurve_state(data: dict[str, Any]) -> dict[str, Any]:
    if not data.get("available"):
        return dict(data)
    return {
        **{key: value for key, value in data.items() if key != "wires"},
        "wires": [
            {key: value for key, value in wire.items() if key != "edges"}
            for wire in data["wires"]
        ],
    }


def periodic_pcurve_continuity_state(face: Any) -> dict[str, Any]:
    """Measure oriented UV closure gaps with JSON-safe evidence."""
    return _public_periodic_pcurve_state(_periodic_pcurve_continuity_data(face))


def repair_face_periodic_pcurve_branches(face: Any) -> tuple[Any, dict[str, Any]]:
    """Translate copied-edge pcurves by surface periods, never 3D curves.

    Seam edges carry two pcurves and are deliberately fixed in place in this
    prototype.  Only wire indices diagnosed on a disposable copy are eligible
    for translation.  A candidate is returned only if an actual branch
    discontinuity closes, all 3D edge curves remain sample-identical, and
    discrete incidence, conservative geometry, native validity, and strict
    wire checks all pass.
    """
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.Geom2d import Geom2d_Curve
    from OCC.Core.TopAbs import TopAbs_FORWARD
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Vec2d
    from OCC.Extend.TopologyUtils import TopologyExplorer

    diagnostic_copy = BRepBuilderAPI_Copy(face, True, False)
    diagnostic_face = topods.Face(diagnostic_copy.Shape())
    diagnosis = wire_self_intersection_state(diagnostic_face)
    target_wire_indices = sorted(
        {int(index) for index in diagnosis["bad_wire_indices"]}
    )
    diagnosis_source = "strict_wire_check_on_copy"
    if not target_wire_indices:
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "no_diagnosed_self_intersection",
            "diagnosis": diagnosis,
            "diagnosis_source": diagnosis_source,
        }

    before_data = _periodic_pcurve_continuity_data(face)
    before = _public_periodic_pcurve_state(before_data)
    if not before_data["available"]:
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": before["reason"],
            "before": before,
            "diagnosis": diagnosis,
            "diagnosis_source": diagnosis_source,
        }
    if before_data["periods"] == [None, None]:
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "surface_not_periodic",
            "before": before,
            "diagnosis": diagnosis,
            "diagnosis_source": diagnosis_source,
        }
    wire_count = len(before_data["wires"])
    if any(index < 0 or index >= wire_count for index in target_wire_indices):
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "diagnosed_wire_index_out_of_range",
            "before": before,
            "diagnosis": diagnosis,
            "diagnosis_source": diagnosis_source,
        }
    plans = [before_data["wires"][index]["plan"] for index in target_wire_indices]
    if not any(
        plan["changed_edge_indices"]
        and plan["before_max_gap"] > PCURVE_GAP_TOLERANCE
        and plan["after_max_gap"] <= PCURVE_GAP_TOLERANCE
        for plan in plans
    ):
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "no_repairable_periodic_branch_gap",
            "before": before,
            "diagnosis": diagnosis,
            "diagnosis_source": diagnosis_source,
        }

    original_edges = list(TopologyExplorer(face).edges())
    before_geometry = face_geometry_signature(face)
    before_topology = face_topology_incidence_signature(face)
    copier = BRepBuilderAPI_Copy(face, True, False)
    candidate = topods.Face(copier.Shape())
    copied_edges = [topods.Edge(copier.ModifiedShape(edge)) for edge in original_edges]
    builder = BRep_Builder()
    translations = []
    for wire_row in before_data["wires"]:
        if wire_row["wire_index"] not in target_wire_indices:
            continue
        for edge_index, offset in enumerate(wire_row["plan"]["offsets"]):
            if offset == (0, 0):
                continue
            copied_edge = topods.Edge(
                copier.ModifiedShape(wire_row["edges"][edge_index])
            )
            if BRep_Tool.IsClosed(copied_edge, candidate):
                return face, {
                    "attempted": True,
                    "accepted": False,
                    "reason": "seam_edge_translation_forbidden",
                    "before": before,
                    "diagnosis": diagnosis,
                    "diagnosis_source": diagnosis_source,
                }
            pcurve, first, last = BRep_Tool.CurveOnSurface(copied_edge, candidate)
            if pcurve is None:
                return face, {
                    "attempted": True,
                    "accepted": False,
                    "reason": "copied_edge_missing_pcurve",
                    "before": before,
                    "diagnosis": diagnosis,
                    "diagnosis_source": diagnosis_source,
                }
            translated = Geom2d_Curve.DownCast(pcurve.Copy())
            translated.Translate(
                gp_Vec2d(
                    offset[0] * (before_data["periods"][0] or 0.0),
                    offset[1] * (before_data["periods"][1] or 0.0),
                )
            )
            forward_edge = topods.Edge(copied_edge.Oriented(TopAbs_FORWARD))
            builder.UpdateEdge(
                forward_edge, translated, candidate, BRep_Tool.Tolerance(copied_edge)
            )
            builder.Range(forward_edge, candidate, first, last)
            translations.append(
                {
                    "wire_index": wire_row["wire_index"],
                    "edge_index": edge_index,
                    "offset": list(offset),
                }
            )

    after_data = _periodic_pcurve_continuity_data(candidate)
    after = _public_periodic_pcurve_state(after_data)
    after_geometry = face_geometry_signature(candidate)
    after_topology = face_topology_incidence_signature(candidate)
    geometry_gate = geometry_preservation_gate(before_geometry, after_geometry)
    curve_gate = corresponding_3d_curve_gate(original_edges, copied_edges)
    topology_equal = before_topology == after_topology
    native_valid = bool(BRepCheck_Analyzer(candidate, True).IsValid())
    verification_copy = BRepBuilderAPI_Copy(candidate, True, False)
    verification_face = topods.Face(verification_copy.Shape())
    strict_state = wire_self_intersection_state(verification_face)
    uv_closed = bool(
        after_data["available"]
        and all(
            after_data["wires"][index]["plan"]["before_max_gap"]
            <= PCURVE_GAP_TOLERANCE
            for index in target_wire_indices
        )
    )
    accepted = bool(
        translations
        and uv_closed
        and topology_equal
        and curve_gate["accepted"]
        and geometry_gate["accepted"]
        and native_valid
        and not strict_state["bad_wire_indices"]
    )
    diagnostics = {
        "attempted": True,
        "accepted": accepted,
        "reason": "accepted" if accepted else "candidate_rejected",
        "strategy": "periodic_pcurve_branch_translation",
        "diagnosis": diagnosis,
        "diagnosis_source": diagnosis_source,
        "target_wire_indices": target_wire_indices,
        "translations": translations,
        "before": before,
        "after": after,
        "before_topology": before_topology,
        "after_topology": after_topology,
        "topology_incidence_equal": topology_equal,
        "curve_3d_preservation": curve_gate,
        "geometry_preservation": geometry_gate,
        "candidate_native_brep_valid": native_valid,
        "strict_wire_state": strict_state,
        "uv_closed": uv_closed,
    }
    return (candidate if accepted else face), diagnostics


def repair_face_local_topology(face: Any) -> tuple[Any, dict[str, Any]]:
    """Repair one already-constructed face, or return it unchanged on failure."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.ShapeFix import ShapeFix_Face
    from OCC.Core.TopoDS import topods

    before = wire_self_intersection_state(face)
    if not before["bad_wire_indices"]:
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "no_diagnosed_self_intersection",
            "before": before,
        }
    before_geometry = face_geometry_signature(face)

    copied = BRepBuilderAPI_Copy(face, True, False)
    copied_face = topods.Face(copied.Shape())
    fixer = ShapeFix_Face(copied_face)
    fixer.SetPrecision(0.01)
    fixer.SetMinTolerance(1e-4)
    fixer.SetMaxTolerance(0.1)
    fixer.SetFixWireMode(True)
    fixer.SetFixIntersectingWiresMode(True)
    fixer.SetFixLoopWiresMode(False)
    wire_tool = fixer.FixWireTool()
    wire_tool.SetPrecision(0.01)
    wire_tool.SetMinTolerance(1e-4)
    wire_tool.SetMaxTolerance(0.1)
    wire_tool.SetClosedWireMode(False)
    wire_tool.SetFixReorderMode(False)
    wire_tool.SetFixConnectedMode(False)
    wire_tool.SetFixEdgeCurvesMode(False)
    wire_tool.SetFixAddPCurveMode(False)
    wire_tool.SetFixReversed2dMode(False)
    wire_tool.SetFixSameParameterMode(False)
    wire_tool.SetFixSelfIntersectionMode(True)
    wire_tool.SetFixSelfIntersectingEdgeMode(True)
    wire_tool.SetFixIntersectingEdgesMode(True)
    wire_tool.SetFixNonAdjacentIntersectingEdgesMode(True)
    wire_tool.SetModifyGeometryMode(False)
    wire_tool.SetModifyTopologyMode(True)
    wire_tool.SetModifyRemoveLoopMode(False)

    performed = bool(fixer.Perform())
    fixer.FixIntersectingWires()
    fixer.FixOrientation()
    candidate = fixer.Face()
    after = wire_self_intersection_state(candidate)
    native_valid = bool(BRepCheck_Analyzer(candidate, True).IsValid())
    after_geometry = face_geometry_signature(candidate)
    preservation = geometry_preservation_gate(before_geometry, after_geometry)
    accepted = bool(
        performed
        and not after["bad_wire_indices"]
        and native_valid
        and preservation["accepted"]
    )
    if accepted:
        reason = "accepted"
    elif not preservation["accepted"]:
        reason = "geometry_preservation_failed"
    else:
        reason = "candidate_rejected"
    diagnostics = {
        "attempted": True,
        "accepted": accepted,
        "performed": performed,
        "before": before,
        "after": after,
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "geometry_preservation": preservation,
        "candidate_native_brep_valid": native_valid,
        "reason": reason,
    }
    return (candidate if accepted else face), diagnostics


def repair_face_local_pcurve(face: Any) -> tuple[Any, dict[str, Any]]:
    """Repair only pcurve continuity on a copied, diagnosed face wire.

    This strategy leaves OCC topology unchanged and enables the 2D gap,
    reversed-pcurve, and same-parameter fixes that the topology strategy
    deliberately disables.  A candidate is accepted only when the diagnosed
    self-intersection disappears, OCC accepts the face, and its 3D geometry
    remains within the same conservative preservation gate.
    """
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.ShapeFix import ShapeFix_Face
    from OCC.Core.TopoDS import topods

    before = wire_self_intersection_state(face)
    if not before["bad_wire_indices"]:
        return face, {
            "attempted": False,
            "accepted": False,
            "reason": "no_diagnosed_self_intersection",
            "before": before,
        }
    before_geometry = face_geometry_signature(face)

    copied = BRepBuilderAPI_Copy(face, True, False)
    copied_face = topods.Face(copied.Shape())
    fixer = ShapeFix_Face(copied_face)
    fixer.SetPrecision(0.01)
    fixer.SetMinTolerance(1e-4)
    fixer.SetMaxTolerance(0.1)
    fixer.SetFixWireMode(True)
    fixer.SetFixIntersectingWiresMode(False)
    fixer.SetFixLoopWiresMode(False)
    wire_tool = fixer.FixWireTool()
    wire_tool.SetPrecision(0.01)
    wire_tool.SetMinTolerance(1e-4)
    wire_tool.SetMaxTolerance(0.1)
    wire_tool.SetClosedWireMode(False)
    wire_tool.SetFixGaps2dMode(True)
    wire_tool.SetFixGapsByRangesMode(True)
    wire_tool.SetFixReversed2dMode(True)
    wire_tool.SetFixSameParameterMode(True)
    wire_tool.SetFixSelfIntersectionMode(True)
    wire_tool.SetFixSelfIntersectingEdgeMode(True)
    wire_tool.SetFixIntersectingEdgesMode(True)
    wire_tool.SetFixNonAdjacentIntersectingEdgesMode(True)
    wire_tool.SetFixEdgeCurvesMode(True)
    wire_tool.SetFixAddPCurveMode(True)
    wire_tool.SetModifyGeometryMode(True)
    wire_tool.SetModifyTopologyMode(False)
    wire_tool.SetModifyRemoveLoopMode(False)
    wire_tool.SetPreferencePCurveMode(True)

    performed = bool(fixer.Perform())
    candidate = fixer.Face()
    after = wire_self_intersection_state(candidate)
    native_valid = bool(BRepCheck_Analyzer(candidate, True).IsValid())
    after_geometry = face_geometry_signature(candidate)
    preservation = geometry_preservation_gate(before_geometry, after_geometry)
    accepted = bool(
        performed
        and not after["bad_wire_indices"]
        and native_valid
        and preservation["accepted"]
    )
    if accepted:
        reason = "accepted"
    elif not preservation["accepted"]:
        reason = "geometry_preservation_failed"
    else:
        reason = "candidate_rejected"
    diagnostics = {
        "attempted": True,
        "accepted": accepted,
        "performed": performed,
        "before": before,
        "after": after,
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "geometry_preservation": preservation,
        "candidate_native_brep_valid": native_valid,
        "strategy": "pcurve_continuity",
        "reason": reason,
    }
    return (candidate if accepted else face), diagnostics


__all__ = [
    "corresponding_3d_curve_gate",
    "face_geometry_signature",
    "face_topology_incidence_signature",
    "geometry_preservation_gate",
    "periodic_pcurve_continuity_state",
    "repair_face_periodic_pcurve_branches",
    "repair_face_local_topology",
    "repair_face_local_pcurve",
    "select_periodic_pcurve_branches",
    "wire_self_intersection_state",
]
