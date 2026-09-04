import json

import pytest

import tools.post_sewing_graph_repair as repair
from tools.post_sewing_graph_repair import (
    attempt_post_sewing_face_pcurve_reprojection,
    combine_graph_gates,
    copied_identity_graph_gate,
    exact_curve_sample_gate,
    exact_identity_graph_gate,
    evaluate_graph_preservation_gate,
    select_exact_reversed_pair_targets,
    shape_topology_incidence_signature,
    source_edge_identity_gate,
    topology_incidence_gate,
)


def _diagnosis(*occurrences):
    return {
        "status": "diagnosed",
        "edge_position_basis": "occ_1_based",
        "occurrences": list(occurrences),
    }


def _occurrence(kind, source_ids, *, face=5, wire=0, positions=(1, 2)):
    return {
        "kind": kind,
        "status": "detected",
        "source_mapping_status": "mapped",
        "source_face_index": face,
        "wire_index": wire,
        "edge_positions": list(positions),
        "source_edge_ids": list(source_ids),
    }


def _reversed_pair_diagnosis(pair=(9, 23)):
    return _diagnosis(
        _occurrence("closure", pair),
        _occurrence("adjacent", tuple(reversed(pair))),
    )


def _box():
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()


def _planar_face():
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon()
    for xyz in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)):
        polygon.Add(gp_Pnt(*xyz))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def _face_bindings(face, *, status="exact_sewing_history"):
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    wires = []
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    next_id = 0
    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        candidates = []
        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        while edge_explorer.More():
            candidates.append(
                {
                    "source_edge_id": next_id,
                    "observed_edge": topods.Edge(edge_explorer.Current()),
                    "proof_method": "identity",
                }
            )
            next_id += 1
            edge_explorer.Next()
        wires.append(
            {"observed_wire": wire, "source_edge_candidates": candidates}
        )
        wire_explorer.Next()
    return [
        {
            "source_face_index": 0,
            "expected_source_face_count": 1,
            "expected_source_edge_count": next_id,
            "face": face,
            "source_mapping": {
                "status": status,
                "failures": [],
                "wire_rows": wires,
            },
        }
    ]


def _curve_row(source_id, value=0.0):
    return {
        "source_edge_id": source_id,
        "available": True,
        "curve_type": 0,
        "parameter_range": [0.0, 1.0],
        "samples": [[float(index), value, 0.0] for index in range(repair.CURVE_SAMPLE_COUNT)],
    }


def test_selector_accepts_only_complete_reversed_adjacent_closure_pair():
    result = select_exact_reversed_pair_targets(
        _reversed_pair_diagnosis(),
        source_face_index=5,
        expected_source_edge_pairs=((23, 9),),
    )

    assert result["accepted"] is True
    assert result["source_edge_pairs"] == [[9, 23]]
    assert result["target_source_edge_ids"] == [9, 23]


@pytest.mark.parametrize(
    ("diagnosis", "reason"),
    [
        (_diagnosis(_occurrence("adjacent", (23, 9))), "exact_reversed_pair_representation_missing"),
        (
            _diagnosis(
                _occurrence("closure", (9, 23)),
                _occurrence("adjacent", (9, 23)),
            ),
            "exact_reversed_pair_representation_missing",
        ),
        (
            _diagnosis(
                _occurrence("closure", (9, 23)),
                _occurrence("adjacent", (23, 9)),
                _occurrence("non_adjacent", (1, 2)),
            ),
            "additional_defect_kind_present",
        ),
        (
            _diagnosis(
                _occurrence("closure", (9, 23)),
                {**_occurrence("adjacent", (23, 9)), "source_mapping_status": "ambiguous"},
            ),
            "defect_source_mapping_not_exact",
        ),
    ],
)
def test_selector_fails_closed_on_incomplete_or_ambiguous_evidence(diagnosis, reason):
    assert select_exact_reversed_pair_targets(
        diagnosis, source_face_index=5
    )["reason"] == reason


def test_selector_expected_pairs_are_exact_set_not_subset():
    result = select_exact_reversed_pair_targets(
        _reversed_pair_diagnosis(),
        source_face_index=5,
        expected_source_edge_pairs=((9, 23), (10, 11)),
    )

    assert result["accepted"] is False
    assert result["reason"] == "expected_source_edge_pairs_mismatch"


def test_selector_rejects_repeated_occurrence_position():
    diagnosis = _diagnosis(
        _occurrence("closure", (9, 23), positions=(2, 2)),
        _occurrence("adjacent", (23, 9)),
    )

    result = select_exact_reversed_pair_targets(
        diagnosis,
        source_face_index=5,
        expected_source_edge_pairs=((9, 23),),
    )

    assert result["accepted"] is False
    assert result["reason"] == "defect_edge_positions_invalid"


def test_real_box_topology_and_copy_history_gates_are_exact():
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy

    original = _box()
    copier = BRepBuilderAPI_Copy(original, True, False)
    candidate = copier.Shape()

    count_gate = topology_incidence_gate(
        shape_topology_incidence_signature(original),
        shape_topology_incidence_signature(candidate),
    )
    history_gate = copied_identity_graph_gate(original, candidate, copier)

    assert count_gate["accepted"] is True
    assert history_gate["accepted"] is True
    assert all(history_gate["checks"].values())


def test_pre_make_solid_shell_gets_disposable_single_solid_validation_view():
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    box = _box()
    explorer = TopExp_Explorer(box, TopAbs_SHELL)
    assert explorer.More()
    sewn_shell = topods.Shell(explorer.Current())
    assert repair.strict_shape_state(sewn_shell)["solid_count"] == 0

    validation_shape, evidence = repair._single_solid_validation_shape(sewn_shell)

    assert evidence == {
        "mode": "wrapped_single_shell",
        "input_solid_count": 0,
        "input_shell_count": 1,
        "validation_solid_count": 1,
    }
    assert repair.strict_shape_state(validation_shape)["solid_count"] == 1
    assert repair.strict_shape_state(validation_shape)["accepted"] is True
    assert repair._native_valid(validation_shape) is True
    # The helper creates only a validation wrapper.  The hook value remains a
    # shell and is still the graph whose copy/history gates are authoritative.
    assert not TopExp_Explorer(sewn_shell, TopAbs_SOLID).More()


def test_identity_graph_gate_detects_a_distinct_same_count_shape():
    before = repair._graph_inventory(_box())
    another_box = _box()

    result = exact_identity_graph_gate(before, another_box)

    assert result["accepted"] is False
    assert result["checks"]["faces_identity_bijection"] is False


def test_pure_gates_reject_missing_or_changed_evidence():
    signature = shape_topology_incidence_signature(_box())
    changed = dict(signature)
    changed["vertex_count"] += 1
    topology = topology_incidence_gate(signature, changed)
    assert topology["accepted"] is False
    assert "vertex_count_equal" in topology["rejection_reasons"]

    before_identity = {
        "schema": repair.SOURCE_IDENTITY_SCHEMA,
        "complete": True,
        "source_to_global_edge_bijection": True,
        "split_source_edge_ids": [],
        "merged_source_edge_pairs": [],
        "source_edge_ids": [0],
        "occurrence_counts": [{"source_edge_id": 0, "count": 2}],
        "face_source_edge_occurrences": [
            {"source_face_index": 0, "source_edge_ids": [0, 0]}
        ],
    }
    after_identity = dict(before_identity)
    after_identity["occurrence_counts"] = [{"source_edge_id": 0, "count": 1}]
    assert source_edge_identity_gate(before_identity, after_identity)["accepted"] is False

    curves = exact_curve_sample_gate([_curve_row(0)], [_curve_row(0, 1e-12)])
    assert curves["accepted"] is False
    assert curves["max_sample_delta"] == pytest.approx(1e-12)


def test_combined_and_final_gate_require_explicit_true_for_every_component():
    accepted = {"accepted": True}
    rejected = {"accepted": False}
    assert combine_graph_gates(accepted, rejected)["accepted"] is False

    kwargs = dict(
        topology_gate=accepted,
        source_identity_gate=accepted,
        curve_3d_gate=accepted,
        pcurve_operations_complete=True,
        original_shape_unchanged=True,
        target_face_was_invalid=True,
        target_face_is_clean=True,
        native_valid=True,
        strict_valid=True,
    )
    assert evaluate_graph_preservation_gate(**kwargs)["accepted"] is True
    kwargs["source_identity_gate"] = {}
    assert evaluate_graph_preservation_gate(**kwargs)["accepted"] is False


def test_pcurve_reprojection_disables_3d_curve_build_and_proves_state(monkeypatch):
    class Surface:
        def __init__(self, _face):
            pass

        def IsUPeriodic(self):
            return False

        def IsVPeriodic(self):
            return False

    class Projector:
        def SetBuildCurveMode(self, value):
            assert value is False

        def SetPrecision(self, value):
            assert value == pytest.approx(1e-7)

    class Tool:
        def __init__(self):
            self.projector = Projector()

        def FixRemovePCurve(self, edge, face):
            assert (edge, face) == ("edge", "face")
            return True

        def Projector(self):
            return self.projector

        def FixAddPCurve(self, edge, face, seam, precision):
            assert (edge, face, seam) == ("edge", "face", False)
            assert precision == pytest.approx(1e-7)
            return True

    import OCC.Core.BRep
    import OCC.Core.BRepAdaptor
    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.BRepAdaptor, "BRepAdaptor_Surface", Surface)
    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)
    monkeypatch.setattr(OCC.Core.BRep.BRep_Tool, "IsClosed", lambda *_args: False)
    monkeypatch.setattr(repair, "_target_edges", lambda *_args: [(9, "edge")])
    states = iter(
        [
            {"available": True, "samples": [[0.0, 0.0]]},
            {"available": False, "reason": "pcurve_missing", "samples": []},
            {"available": True, "samples": [[0.0, 0.0]]},
        ]
    )
    monkeypatch.setattr(repair, "_pcurve_sample", lambda *_args: next(states))

    result = repair._reproject_target_pcurves(
        {"face": "face"}, [9], precision=1e-7
    )

    assert result["accepted"] is True
    operation = result["operations"][0]
    assert operation["pcurve_absent_after_remove"] is True
    assert operation["pcurve_present_after_add"] is True
    assert operation["projector_build_curve_mode"] is False


def test_reprojection_fails_when_remove_return_does_not_match_observed_state(monkeypatch):
    class Surface:
        def __init__(self, _face):
            pass

        def IsUPeriodic(self):
            return False

        def IsVPeriodic(self):
            return False

    class Tool:
        def FixRemovePCurve(self, *_args):
            return True

    import OCC.Core.BRep
    import OCC.Core.BRepAdaptor
    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.BRepAdaptor, "BRepAdaptor_Surface", Surface)
    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)
    monkeypatch.setattr(OCC.Core.BRep.BRep_Tool, "IsClosed", lambda *_args: False)
    monkeypatch.setattr(repair, "_target_edges", lambda *_args: [(9, "edge")])
    monkeypatch.setattr(
        repair, "_pcurve_sample", lambda *_args: {"available": True, "samples": []}
    )

    result = repair._reproject_target_pcurves(
        {"face": "face"}, [9], precision=1e-7
    )

    assert result["accepted"] is False
    assert result["reason"] == "pcurve_remove_not_proven"


def test_reprojection_does_not_treat_nonfinite_pcurve_as_removed(monkeypatch):
    class Surface:
        def __init__(self, _face):
            pass

        def IsUPeriodic(self):
            return False

        def IsVPeriodic(self):
            return False

    class Tool:
        def FixRemovePCurve(self, *_args):
            return True

    import OCC.Core.BRep
    import OCC.Core.BRepAdaptor
    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.BRepAdaptor, "BRepAdaptor_Surface", Surface)
    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)
    monkeypatch.setattr(OCC.Core.BRep.BRep_Tool, "IsClosed", lambda *_args: False)
    monkeypatch.setattr(repair, "_target_edges", lambda *_args: [(9, "edge")])
    states = iter(
        [
            {"available": True, "samples": [[0.0, 0.0]]},
            {
                "available": False,
                "reason": "nonfinite_pcurve_sample",
                "samples": [],
            },
        ]
    )
    monkeypatch.setattr(repair, "_pcurve_sample", lambda *_args: next(states))

    result = repair._reproject_target_pcurves(
        {"face": "face"}, [9], precision=1e-7
    )

    assert result["accepted"] is False
    assert result["reason"] == "pcurve_remove_not_proven"
    assert result["operations"][0]["pcurve_absent_after_remove"] is False
    assert (
        result["operations"][0]["pcurve_state_after_remove"]
        == "nonfinite_pcurve_sample"
    )


def test_clean_real_face_rejects_without_mutating_original(monkeypatch):
    face = _planar_face()
    bindings = _face_bindings(face)
    before = shape_topology_incidence_signature(face)
    monkeypatch.setattr(
        repair,
        "diagnose_face_wires_v2" if hasattr(repair, "diagnose_face_wires_v2") else "_unused",
        None,
        raising=False,
    )

    candidate, diagnostics = attempt_post_sewing_face_pcurve_reprojection(
        face,
        source_face_bindings=bindings,
        target_source_face_index=0,
        target_source_edge_ids=(0, 1),
        expected_source_edge_pairs=((0, 1),),
    )

    assert candidate.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["attempted"] is False
    assert diagnostics["reason"] in {
        "diagnosis_occurrences_missing",
        "source_edge_identity_not_exact",
    }
    assert shape_topology_incidence_signature(face) == before
    json.dumps(diagnostics, allow_nan=False)


def test_bad_mapping_status_rejects_before_any_copy(monkeypatch):
    face = _planar_face()
    bindings = _face_bindings(face, status="unmapped")
    monkeypatch.setattr(
        repair,
        "_deep_copy",
        lambda *_args: pytest.fail("invalid source binding must reject before copy"),
    )

    candidate, diagnostics = attempt_post_sewing_face_pcurve_reprojection(
        face,
        source_face_bindings=bindings,
        target_source_face_index=0,
        target_source_edge_ids=(0, 1),
    )

    assert candidate.IsSame(face)
    assert diagnostics["reason"] == "source_edge_identity_not_exact"
