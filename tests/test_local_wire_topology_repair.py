import json

from tools.local_wire_topology_repair import (
    face_geometry_signature,
    geometry_preservation_gate,
    periodic_pcurve_continuity_state,
    repair_face_periodic_pcurve_branches,
    repair_face_local_pcurve,
    repair_face_local_topology,
    select_periodic_pcurve_branches,
)


def _planar_face(points):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon()
    for xyz in points:
        polygon.Add(gp_Pnt(*xyz))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def test_clean_planar_face_is_returned_without_topology_mutation():
    face = _planar_face(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))

    result, diagnostics = repair_face_local_topology(face)

    assert result.IsSame(face)
    assert diagnostics["attempted"] is False
    assert diagnostics["reason"] == "no_diagnosed_self_intersection"
    assert diagnostics["before"]["self_intersection_count"] == 0


def test_geometry_preservation_gate_accepts_same_face_signature():
    face = _planar_face(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    signature = face_geometry_signature(face)

    result = geometry_preservation_gate(signature, dict(signature))

    assert result["accepted"] is True
    assert all(result["checks"].values())
    assert result["area_relative_delta"] == 0.0
    assert result["boundary_length_relative_delta"] == 0.0


def test_geometry_preservation_gate_rejects_boundary_or_topology_change():
    before = {
        "wire_count": 1,
        "edge_count": 4,
        "area": 1.0,
        "boundary_length": 4.0,
        "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    }
    after = {
        "wire_count": 1,
        "edge_count": 3,
        "area": 0.4,
        "boundary_length": 3.0,
        "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    }

    result = geometry_preservation_gate(before, after)

    assert result["accepted"] is False
    assert result["checks"]["edge_count_equal"] is False
    assert result["checks"]["area_within_tolerance"] is False


def test_geometry_preservation_gate_allows_only_bounded_tolerance_noise():
    before = {
        "wire_count": 1,
        "edge_count": 8,
        "area": 10.0,
        "boundary_length": 20.0,
        "bbox": [0.0, 0.0, 0.0, 2.0, 2.0, 1.0],
    }
    within = {
        **before,
        "area": 10.01,
        "boundary_length": 20.04,
        "bbox": [0.0, 0.0, 0.0, 2.001, 2.0, 1.0],
    }
    outside = {**within, "boundary_length": 20.2}

    assert geometry_preservation_gate(before, within)["accepted"] is True
    assert geometry_preservation_gate(before, outside)["accepted"] is False


def test_self_intersecting_candidate_is_rejected_and_original_face_is_returned():
    face = _planar_face(((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)))

    result, diagnostics = repair_face_local_topology(face)

    assert result.IsSame(face)
    assert diagnostics["attempted"] is True
    assert diagnostics["accepted"] is False
    assert diagnostics["before"]["self_intersection_count"] == 1


def test_local_pcurve_candidate_fails_closed_on_unrepairable_crossing():
    face = _planar_face(((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)))

    result, diagnostics = repair_face_local_pcurve(face)

    assert result.IsSame(face)
    assert diagnostics["attempted"] is True
    assert diagnostics["accepted"] is False
    assert diagnostics["strategy"] == "pcurve_continuity"
    assert diagnostics["before"]["self_intersection_count"] == 1


def test_periodic_branch_selector_closes_integer_period_gap():
    period = 6.0
    endpoints = [
        ((12.0, 1.0), (6.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0), (6.0, 0.0)),
        ((6.0, 0.0), (6.0, 1.0)),
    ]

    result = select_periodic_pcurve_branches(
        endpoints,
        u_period=period,
        v_period=None,
        movable=[True, False, True, False],
    )

    assert result["solved"] is True
    assert result["before_max_gap"] == period
    assert result["after_max_gap"] == 0.0
    assert result["offsets"] == [(-1, 0), (0, 0), (0, 0), (0, 0)]
    assert result["changed_edge_indices"] == [0]


def test_periodic_branch_selector_rejects_nonperiodic_surface():
    result = select_periodic_pcurve_branches(
        [((0.0, 0.0), (1.0, 0.0))],
        u_period=None,
        v_period=None,
    )

    assert result["solved"] is False
    assert result["reason"] == "surface_not_periodic"


def _cylinder_face_with_shifted_pcurve():
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.Geom2d import Geom2d_Curve
    from OCC.Core.TopAbs import TopAbs_FORWARD
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Vec2d
    from OCC.Extend.TopologyUtils import TopologyExplorer

    cylinder = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    source_face = next(
        face
        for face in TopologyExplorer(cylinder).faces()
        if BRep_Tool.Surface(face).IsUPeriodic()
    )
    copier = BRepBuilderAPI_Copy(source_face, True, False)
    face = topods.Face(copier.Shape())
    wire = next(TopologyExplorer(face).wires())
    explorer = BRepTools_WireExplorer(wire, face)
    edge = explorer.Current()
    pcurve, first, last = BRep_Tool.CurveOnSurface(edge, face)
    translated = Geom2d_Curve.DownCast(pcurve.Copy())
    period = float(BRep_Tool.Surface(face).UPeriod())
    translated.Translate(gp_Vec2d(period, 0.0))
    forward_edge = topods.Edge(edge.Oriented(TopAbs_FORWARD))
    builder = BRep_Builder()
    builder.UpdateEdge(forward_edge, translated, face, BRep_Tool.Tolerance(edge))
    builder.Range(forward_edge, face, first, last)
    return face


def test_periodic_pcurve_repair_changes_only_copied_uv_branch():
    face = _cylinder_face_with_shifted_pcurve()
    input_state = periodic_pcurve_continuity_state(face)

    result, diagnostics = repair_face_periodic_pcurve_branches(face)

    assert diagnostics["attempted"] is True
    assert diagnostics["accepted"] is True
    assert diagnostics["strategy"] == "periodic_pcurve_branch_translation"
    assert diagnostics["uv_closed"] is True
    assert diagnostics["topology_incidence_equal"] is True
    assert diagnostics["curve_3d_preservation"]["accepted"] is True
    assert diagnostics["curve_3d_preservation"]["max_sample_delta"] == 0.0
    assert diagnostics["geometry_preservation"]["accepted"] is True
    assert diagnostics["translations"] == [
        {"wire_index": 0, "edge_index": 0, "offset": [-1, 0]}
    ]
    json.dumps(diagnostics, allow_nan=False)
    assert not result.IsSame(face)
    assert input_state["max_gap"] > 6.0
    assert periodic_pcurve_continuity_state(face)["max_gap"] == input_state["max_gap"]
    assert periodic_pcurve_continuity_state(result)["max_gap"] <= 1e-7


def test_periodic_pcurve_repair_leaves_nonperiodic_face_unchanged():
    face = _planar_face(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))

    result, diagnostics = repair_face_periodic_pcurve_branches(face)

    assert result.IsSame(face)
    assert diagnostics["attempted"] is False
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "surface_not_periodic"
