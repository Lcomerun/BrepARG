"""Fail-closed repair for a diagnosed self-intersecting face wire.

The helper deliberately works on a copied face. OCC's topology-changing wire
fixer can mutate shared edge topology and, for malformed inputs, may terminate
the native process instead of raising a Python exception. Callers therefore
use this helper from one-CAD worker processes and accept a candidate only when
the repaired face has no remaining strict wire intersection and passes the OCC
native analyzer. A rejected candidate returns the original face unchanged.
"""

from __future__ import annotations

from typing import Any


# A copied-face candidate may absorb tolerance noise, but it may not materially
# reshape a CAD face.  Area and perimeter are limited to 0.5 percent change;
# axis-aligned bounds are limited to 0.1 percent of the coordinate scale.
MAX_AREA_RELATIVE_DELTA = 0.005
MAX_BOUNDARY_LENGTH_RELATIVE_DELTA = 0.005
MAX_BBOX_RELATIVE_DELTA = 0.001


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
    "face_geometry_signature",
    "geometry_preservation_gate",
    "repair_face_local_topology",
    "repair_face_local_pcurve",
    "wire_self_intersection_state",
]
