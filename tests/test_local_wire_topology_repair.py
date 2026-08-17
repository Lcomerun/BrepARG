from tools.local_wire_topology_repair import (
    face_geometry_signature,
    geometry_preservation_gate,
    repair_face_local_pcurve,
    repair_face_local_topology,
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
