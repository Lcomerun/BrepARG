import json

import pytest

import tools.targeted_nonperiodic_pcurve_repair as targeted
from tools.targeted_nonperiodic_pcurve_repair import (
    repair_face_targeted_nonperiodic_pcurves,
    select_exact_adjacent_targets,
)


def _diagnosis(*occurrences):
    return {
        "status": "diagnosed",
        "edge_position_basis": "occ_1_based",
        "occurrences": list(occurrences),
    }


def _adjacent(
    source_ids=(101, 102),
    *,
    face_index=7,
    wire_index=0,
    edge_positions=(4, 5),
    status="detected",
    mapping_status="mapped",
):
    return {
        "kind": "adjacent",
        "status": status,
        "wire_index": wire_index,
        "edge_positions": list(edge_positions),
        "source_face_index": face_index,
        "source_edge_ids": list(source_ids),
        "source_mapping_status": mapping_status,
    }


def _planar_face(points=None):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon()
    for xyz in points or ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)):
        polygon.Add(gp_Pnt(*xyz))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def _identity_mapping(face, *, source_ids=None):
    from OCC.Extend.TopologyUtils import TopologyExplorer

    wires = list(TopologyExplorer(face, ignore_orientation=False).wires())
    assert len(wires) == 1
    edges = list(TopologyExplorer(wires[0], ignore_orientation=False).edges())
    ids = list(range(len(edges))) if source_ids is None else list(source_ids)
    assert len(ids) == len(edges)
    return (
        {
            "status": "exact_identity",
            "wire_rows": [
                {
                    "observed_wire": wires[0],
                    "source_edge_candidates": [
                        {
                            "source_edge_id": source_id,
                            "observed_edge": edge,
                            "proof_method": "identity",
                        }
                        for source_id, edge in zip(ids, edges)
                    ],
                }
            ],
            "failures": [],
            "edge_proof_methods": ["identity"] * len(edges),
        },
        list(zip(ids, edges)),
    )


def test_selector_accepts_complete_unordered_exact_adjacent_pairs():
    result = select_exact_adjacent_targets(
        _diagnosis(
            _adjacent((102, 101)),
            _adjacent((201, 202), wire_index=1, edge_positions=(1, 2)),
        ),
        source_face_index=7,
        expected_source_edge_pairs=((101, 102), (202, 201)),
    )

    assert result["accepted"] is True
    assert result["source_edge_pairs"] == [[101, 102], [201, 202]]
    assert result["target_source_edge_ids"] == [101, 102, 201, 202]


@pytest.mark.parametrize(
    ("occurrence", "reason"),
    [
        ({**_adjacent(), "kind": "closure"}, "non_adjacent_or_additional_defect_present"),
        ({**_adjacent(), "kind": "non_adjacent"}, "non_adjacent_or_additional_defect_present"),
        (_adjacent(status="occ_fail"), "adjacent_occurrence_not_detected"),
        (_adjacent(mapping_status="ambiguous"), "adjacent_source_mapping_not_exact"),
        (_adjacent(source_ids=(101, 101)), "adjacent_source_edge_pair_invalid"),
        (_adjacent(edge_positions=(0, 1)), "adjacent_edge_positions_invalid"),
        (_adjacent(face_index=8), "occurrence_source_face_mismatch"),
    ],
)
def test_selector_fails_closed_on_any_non_exact_occurrence(occurrence, reason):
    result = select_exact_adjacent_targets(
        _diagnosis(occurrence), source_face_index=7
    )

    assert result == {"accepted": False, "reason": reason}


def test_selector_requires_expected_pair_set_to_match_not_merely_overlap():
    result = select_exact_adjacent_targets(
        _diagnosis(_adjacent((102, 101))),
        source_face_index=7,
        expected_source_edge_pairs=((101, 102), (201, 202)),
    )

    assert result["accepted"] is False
    assert result["reason"] == "expected_source_edge_pairs_mismatch"
    assert result["observed_source_edge_pairs"] == [[101, 102]]


def test_selector_rejects_overlapping_target_pairs():
    result = select_exact_adjacent_targets(
        _diagnosis(
            _adjacent((101, 102)),
            _adjacent((102, 201), wire_index=1, edge_positions=(1, 2)),
        ),
        source_face_index=7,
    )

    assert result == {
        "accepted": False,
        "reason": "adjacent_source_pairs_overlap",
    }


def test_selector_rejects_repeated_occ_edge_position():
    result = select_exact_adjacent_targets(
        _diagnosis(_adjacent((101, 102), edge_positions=(2, 2))),
        source_face_index=7,
    )

    assert result == {
        "accepted": False,
        "reason": "adjacent_edge_positions_invalid",
    }


def test_clean_real_face_is_a_safe_noop_and_returns_original_face():
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["attempted"] is False
    assert diagnostics["reason"] == "no_exact_adjacent_targets"
    assert diagnostics["source_mapping_gate"]["accepted"] is True
    assert diagnostics["surface_gate"]["accepted"] is True
    json.dumps(diagnostics, allow_nan=False)


def test_periodic_real_face_is_rejected_before_diagnosis(monkeypatch):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Extend.TopologyUtils import TopologyExplorer

    face = next(
        value
        for value in TopologyExplorer(
            BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
        ).faces()
        if BRep_Tool.Surface(value).IsUPeriodic()
    )
    mapping, occurrences = _identity_mapping(face)
    monkeypatch.setattr(
        targeted,
        "_mapping_context",
        lambda *_args, **_kwargs: (
            {"accepted": True, "reason": "test_exact_mapping"},
            {"test_only": True},
        ),
    )
    monkeypatch.setattr(
        targeted,
        "diagnose_face_wires_v2",
        lambda *_args, **_kwargs: pytest.fail("periodic face must fail before diagnosis"),
    )

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "periodic_surface_forbidden"


def test_incomplete_mapping_is_rejected_before_diagnosis(monkeypatch):
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    mapping["wire_rows"][0]["source_edge_candidates"].pop()
    monkeypatch.setattr(
        targeted,
        "diagnose_face_wires_v2",
        lambda *_args, **_kwargs: pytest.fail("invalid mapping must fail first"),
    )

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "source_mapping_wire_edges_not_unique_complete"


def test_source_occurrence_multiset_is_independently_verified(monkeypatch):
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    occurrences[-1] = (999, occurrences[-1][1])
    monkeypatch.setattr(
        targeted,
        "diagnose_face_wires_v2",
        lambda *_args, **_kwargs: pytest.fail("bad source inventory must fail first"),
    )

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["reason"] == "source_mapping_occurrence_multiset_mismatch"


def test_target_seam_preflight_rejects_without_surgery(monkeypatch):
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    monkeypatch.setattr(
        targeted,
        "diagnose_face_wires_v2",
        lambda *_args, **_kwargs: _diagnosis(
            _adjacent((0, 1), face_index=0, edge_positions=(1, 2))
        ),
    )
    monkeypatch.setattr(
        targeted,
        "_edge_preflight",
        lambda *_args, **_kwargs: (
            {"accepted": False, "reason": "target_seam_edge_forbidden"},
            None,
        ),
    )
    monkeypatch.setattr(
        targeted,
        "_copy_face_and_mapping",
        lambda *_args, **_kwargs: pytest.fail("preflight must precede copy/surgery"),
    )

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "target_seam_edge_forbidden"


def test_rebuild_one_pcurve_requires_remove_and_add_state_transitions(monkeypatch):
    class Projector:
        def SetBuildCurveMode(self, value):
            assert value is False

        def SetPrecision(self, value):
            assert value == pytest.approx(1e-7)

    class Tool:
        instances = []

        def __init__(self):
            self.index = len(self.instances)
            self.instances.append(self)
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

        def Status(self, _status):
            return False

    states = iter(
        [
            {"available": False, "reason": "pcurve_missing"},
            {"available": True},
        ]
    )
    monkeypatch.setattr(targeted, "_pcurve_fingerprint", lambda *_args: next(states))
    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)

    result = targeted._rebuild_one_pcurve(
        "edge", "face", source_edge_id=7, projection_precision=1e-7
    )

    assert result["accepted"] is True
    assert result["pcurve_absent_after_remove"] is True
    assert result["pcurve_present_after_add"] is True
    assert result["projector_build_curve_mode"] is False


def test_rebuild_one_pcurve_fails_if_remove_return_lies_about_state(monkeypatch):
    class Tool:
        def FixRemovePCurve(self, _edge, _face):
            return True

        def Status(self, _status):
            return False

    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)
    monkeypatch.setattr(
        targeted, "_pcurve_fingerprint", lambda *_args: {"available": True}
    )

    result = targeted._rebuild_one_pcurve(
        "edge", "face", source_edge_id=7, projection_precision=1e-7
    )

    assert result["accepted"] is False
    assert result["reason"] == "pcurve_remove_not_proven"
    assert result["pcurve_absent_after_remove"] is False


def test_rebuild_one_pcurve_does_not_treat_nonfinite_state_as_absent(monkeypatch):
    class Tool:
        def FixRemovePCurve(self, _edge, _face):
            return True

        def Status(self, _status):
            return False

    import OCC.Core.ShapeFix

    monkeypatch.setattr(OCC.Core.ShapeFix, "ShapeFix_Edge", Tool)
    monkeypatch.setattr(
        targeted,
        "_pcurve_fingerprint",
        lambda *_args: {
            "available": False,
            "reason": "nonfinite_parameter_range",
        },
    )

    result = targeted._rebuild_one_pcurve(
        "edge", "face", source_edge_id=7, projection_precision=1e-7
    )

    assert result["accepted"] is False
    assert result["reason"] == "pcurve_remove_not_proven"
    assert result["pcurve_state_after_remove"] == "nonfinite_parameter_range"


def test_real_occ_copy_is_distinct_and_complete_mapping_is_reproved():
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    gate, context = targeted._mapping_context(face, mapping, occurrences)
    assert gate["accepted"] is True

    copied, copied_mapping, copy_gate = targeted._copy_face_and_mapping(face, context)

    assert copy_gate["accepted"] is True
    assert copy_gate["mapping_status"] == "exact_copy_history"
    assert copied is not None and not copied.IsSame(face)
    assert copied_mapping["status"] == "exact_copy_history"
    assert copied_mapping["upstream_mapping_status"] == "exact_identity"
    copied_gate, copied_context = targeted._mapping_context(
        copied, copied_mapping, occurrences
    )
    assert copied_gate["accepted"] is True
    assert copied_context is not None
    assert copied_gate["source_edge_ids"] == gate["source_edge_ids"]


@pytest.mark.parametrize("maximum_delta", [float("nan"), 1.01e-10])
def test_strict_curve_gate_rejects_nonfinite_or_over_limit_delta(
    monkeypatch, maximum_delta
):
    monkeypatch.setattr(
        targeted,
        "corresponding_3d_curve_gate",
        lambda *_args: {"accepted": True, "max_sample_delta": maximum_delta},
    )
    monkeypatch.setattr(
        targeted,
        "_curve_3d_fingerprint",
        lambda _edge: {"available": True},
    )

    gate = targeted._strict_corresponding_3d_curve_gate(["a"], ["b"])

    assert gate["accepted"] is False
    assert gate["explicit_sample_tolerance"] == pytest.approx(1e-10)
    json.dumps(gate, allow_nan=False)
    if not targeted.math.isfinite(maximum_delta):
        assert gate["max_sample_delta"] is None


def test_target_preflight_rejects_nonfinite_3d_curve_samples(monkeypatch):
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    gate, context = targeted._mapping_context(face, mapping, occurrences)
    assert gate["accepted"] is True
    monkeypatch.setattr(
        targeted,
        "_curve_3d_fingerprint",
        lambda _edge: {
            "available": False,
            "reason": "nonfinite_curve_3d_sample",
        },
    )

    preflight, rows = targeted._edge_preflight(face, context, [occurrences[0][0]])

    assert rows is None
    assert preflight["reason"] == "target_3d_curve_missing_or_nonfinite"


def test_any_internal_occ_exception_returns_original_face(monkeypatch):
    face = _planar_face()
    mapping, occurrences = _identity_mapping(face)
    monkeypatch.setattr(
        targeted,
        "_mapping_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native wrapper")),
    )

    result, diagnostics = repair_face_targeted_nonperiodic_pcurves(
        face,
        source_face_index=0,
        source_mapping=mapping,
        source_edge_occurrences=occurrences,
    )

    assert result.IsSame(face)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "occ_or_evidence_exception"
    assert diagnostics["error_type"] == "RuntimeError"
