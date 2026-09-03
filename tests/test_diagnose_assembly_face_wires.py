import pickle

import numpy as np

from tools.diagnose_assembly_face_wires import (
    _edge_walk_issue,
    build_case_row_v2,
    collect_wire_occurrences,
    crossing_pair_candidates,
    diagnose_face_wires_v2,
    diagnose_step_face_wires_v2,
    enrich_wire_occurrences_with_source_edges,
    frozen_p0a_baseline_rows,
    source_topology_summary,
    summarize,
    summarize_v2,
)


class CrossingAnalysis:
    def __init__(self, detected):
        self.detected = {tuple(values) for values in detected}

    def CheckSelfIntersectingEdge(self, _position):
        return False

    def CheckIntersectingEdges(self, *positions):
        return tuple(positions) in self.detected

    def CheckGap2d(self, _position):
        return False

    def CheckConnected(self, _position):
        return False

    def CheckSeam(self, _position):
        return False


def test_source_topology_reports_only_concrete_local_issues():
    parsed = {
        "faceEdge_adj": [[0, 1, 2], [3, 3, 4]],
        "edgeCorner_adj": np.asarray([[0, 1], [1, 2], [2, 0], [3, 4], [4, 3]]),
        "edge_ncs": np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]),
    }
    result = source_topology_summary(parsed)
    assert result["suspicious_faces"] == [{"face_index": 1, "reason": "duplicate_incident_edge_id"}]
    assert result["degenerate_edge_indices"] == [3]


def test_edge_walk_identifies_open_vertex_incidence():
    adjacency = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    assert _edge_walk_issue([0, 1, 2], adjacency) == "open_or_branching_vertex_incidence"


def test_summary_keeps_missing_steps_explicit():
    rows = [
        {"cad_id": "observed", "step_diagnosis_available": True, "source_topology": {"suspicious_faces": []}, "step_diagnosis": {"self_intersection_faces": [2], "order_failure_faces": []}},
        {"cad_id": "unavailable", "step_diagnosis_available": False, "source_topology": {"suspicious_faces": [{"face_index": 0, "reason": "x"}]}, "step_diagnosis": {"faces": [], "wires": []}},
    ]
    result = summarize(rows)
    assert result["step_diagnosis_available"] == 1
    assert result["step_diagnosis_unavailable"] == 1
    assert result["self_intersection_faces_by_cad"] == [{"cad_id": "observed", "face_indices": [2]}]
    assert result["unavailable_step_cad_ids"] == ["unavailable"]


def test_p0a_baseline_selector_uses_stage_aware_baseline(tmp_path):
    rows = []
    for index in range(16):
        rows.append({"cad_id": f"cad-{index}", "joint_iterations": 200, "sewing_tolerance": 1e-3})
    rows.append({"cad_id": "wrong-joint", "joint_iterations": 0, "sewing_tolerance": 1e-3})
    rows.append({"cad_id": "wrong-tol", "joint_iterations": 200, "sewing_tolerance": 1e-2})
    path = tmp_path / "attempts.jsonl"
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = frozen_p0a_baseline_rows(path)
    assert [row["cad_id"] for row in selected] == sorted(f"cad-{index}" for index in range(16))


def test_v2_occurrences_classify_adjacent_pair_with_one_based_positions():
    rows = collect_wire_occurrences(
        CrossingAnalysis({(3,)}), edge_count=5
    )
    assert [row for row in rows if row["kind"] == "adjacent"] == [{
        "kind": "adjacent",
        "edge_positions": [2, 3],
        "status": "detected",
        "occ_method": "CheckIntersectingEdges",
    }]


def test_v2_occurrences_classify_only_n_to_one_as_closure():
    rows = collect_wire_occurrences(
        CrossingAnalysis({(1,)}), edge_count=5
    )
    assert [row for row in rows if row["kind"] == "closure"] == [{
        "kind": "closure",
        "edge_positions": [5, 1],
        "status": "detected",
        "occ_method": "CheckIntersectingEdges",
    }]
    assert crossing_pair_candidates(5)[0]["edge_positions"] == [5, 1]


def test_v2_occurrences_classify_only_cyclic_distance_gt_one_as_non_adjacent():
    rows = collect_wire_occurrences(
        CrossingAnalysis({(1, 3), (1, 5)}), edge_count=5
    )
    assert [row["edge_positions"] for row in rows if row["kind"] == "non_adjacent"] == [[1, 3]]
    assert not any(
        row["kind"] == "non_adjacent" and row["edge_positions"] == [1, 5]
        for row in rows
    )


class IdentityShape:
    def __init__(self, identity):
        self.identity = identity

    def IsSame(self, other):
        return self.identity == other.identity


def identity_mapping(status="exact_identity"):
    wire = IdentityShape("wire")
    return wire, {
        "status": status,
        "wire_rows": [{
            "observed_wire": wire,
            "source_edge_candidates": [
                {"source_edge_id": 101, "observed_edge": IdentityShape("edge-a")},
                {"source_edge_id": 102, "observed_edge": IdentityShape("edge-b")},
                {"source_edge_id": 103, "observed_edge": IdentityShape("edge-c")},
            ],
        }],
    }


def test_source_edge_enrichment_maps_fixed_occurrences_by_identity_not_position():
    occurrences = [{
        "kind": "non_adjacent",
        "edge_positions": [1, 3],
        "status": "detected",
    }]
    wire, mapping = identity_mapping()
    # The fixed wire order deliberately differs from candidate order.
    result = enrich_wire_occurrences_with_source_edges(
        occurrences,
        observed_wire=wire,
        occurrence_edges={1: IdentityShape("edge-c"), 3: IdentityShape("edge-a")},
        source_mapping=mapping,
    )
    assert result == [{
        "kind": "non_adjacent",
        "edge_positions": [1, 3],
        "status": "detected",
        "source_edge_ids": [103, 101],
        "source_mapping_status": "mapped",
    }]
    assert "source_edge_ids" not in occurrences[0]


def test_source_edge_enrichment_accepts_exact_sewing_history():
    wire, mapping = identity_mapping("exact_sewing_history")
    result = enrich_wire_occurrences_with_source_edges(
        [{"kind": "adjacent", "edge_positions": [2, 3], "status": "detected"}],
        observed_wire=wire,
        occurrence_edges={2: IdentityShape("edge-b"), 3: IdentityShape("edge-c")},
        source_mapping=mapping,
    )

    assert result[0]["source_mapping_status"] == "mapped"
    assert result[0]["source_edge_ids"] == [102, 103]


def test_source_edge_enrichment_accepts_exact_sewing_face_local_geometry():
    wire, mapping = identity_mapping("exact_sewing_face_local_geometry")
    result = enrich_wire_occurrences_with_source_edges(
        [{"kind": "closure", "edge_positions": [1, 3], "status": "detected"}],
        observed_wire=wire,
        occurrence_edges={1: IdentityShape("edge-a"), 3: IdentityShape("edge-c")},
        source_mapping=mapping,
    )

    assert result[0]["source_mapping_status"] == "mapped"
    assert result[0]["source_edge_ids"] == [101, 103]


def test_source_edge_enrichment_marks_zero_wire_matches_unavailable():
    wire, mapping = identity_mapping()
    result = enrich_wire_occurrences_with_source_edges(
        [{"kind": "self_only", "edge_positions": [1], "status": "detected"}],
        observed_wire=IdentityShape("other-wire"),
        occurrence_edges={1: IdentityShape("edge-a")},
        source_mapping=mapping,
    )
    assert result[0]["source_mapping_status"] == "unavailable"
    assert result[0]["source_mapping_reason"] == "source_wire_mapping_not_found"
    assert "source_edge_ids" not in result[0]


def test_source_edge_enrichment_marks_multiple_wire_matches_ambiguous():
    wire, mapping = identity_mapping()
    mapping["wire_rows"].append({
        "observed_wire": IdentityShape("wire"),
        "source_edge_candidates": [
            {"source_edge_id": 8, "observed_edge": IdentityShape("edge-a")},
        ],
    })
    result = enrich_wire_occurrences_with_source_edges(
        [{"kind": "self_only", "edge_positions": [1], "status": "detected"}],
        observed_wire=wire,
        occurrence_edges={1: IdentityShape("edge-a")},
        source_mapping=mapping,
    )
    assert result[0]["source_mapping_status"] == "ambiguous"
    assert result[0]["source_mapping_reason"] == "source_wire_mapping_not_unique"
    assert "source_edge_ids" not in result[0]


def test_source_edge_enrichment_rejects_unproven_fixed_edge_and_clears_stale_ids():
    wire, mapping = identity_mapping()
    result = enrich_wire_occurrences_with_source_edges(
        [{
            "kind": "non_adjacent",
            "edge_positions": [1, 2],
            "status": "detected",
            "source_edge_ids": [999],
            "source_mapping_status": "mapped",
            "source_mapping_reason": "stale",
        }],
        observed_wire=wire,
        occurrence_edges={1: IdentityShape("edge-a"), 2: IdentityShape("unknown")},
        source_mapping=mapping,
    )
    assert result[0]["source_mapping_status"] == "unavailable"
    assert result[0]["source_mapping_reason"] == "source_edge_identity_not_found"
    assert "source_edge_ids" not in result[0]


def test_real_occ_shapefix_reordered_wire_edges_map_by_identity():
    """Regression: ShapeFix order differs, but source IDs remain correct."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face, topods_Wire
    from OCC.Extend.TopologyUtils import TopologyExplorer

    shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face = topods_Face(face_explorer.Current())
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    wire = topods_Wire(wire_explorer.Current())
    original_edges = list(
        TopologyExplorer(wire, ignore_orientation=False).edges()
    )

    fixer = ShapeFix_Wire(wire, face, 0.01)
    fixer.Load(wire)
    fixer.SetFace(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(1.0)
    fixer.SetMinTolerance(1e-4)
    fixer.Perform()
    fixed_wire = fixer.Wire()
    analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
    analysis.Load(fixed_wire)
    wire_data = analysis.WireData()
    occurrence_edges = {
        position: wire_data.Edge(position)
        for position in range(1, int(analysis.NbEdges()) + 1)
    }
    fixed_source_order = [
        next(
            index
            for index, original_edge in enumerate(original_edges)
            if fixed_edge.IsSame(original_edge)
        )
        for fixed_edge in occurrence_edges.values()
    ]
    assert fixed_source_order != list(range(len(original_edges)))

    mapping = {
        "status": "exact_identity",
        "wire_rows": [{
            "observed_wire": wire,
            "source_edge_candidates": [
                {"source_edge_id": index, "observed_edge": edge}
                for index, edge in enumerate(original_edges)
            ],
        }],
    }
    result = enrich_wire_occurrences_with_source_edges(
        [{
            "kind": "test",
            "edge_positions": list(occurrence_edges),
            "status": "detected",
        }],
        observed_wire=wire,
        occurrence_edges=occurrence_edges,
        source_mapping=mapping,
    )
    assert result[0]["source_mapping_status"] == "mapped"
    assert result[0]["source_edge_ids"] == fixed_source_order


def test_face_v2_accepts_source_face_index_and_enriches_exact_occurrence(
    monkeypatch
):
    from OCC.Core import TopExp, TopoDS
    import tools.diagnose_assembly_face_wires as diagnosis_module

    wire = IdentityShape("wire")
    first_fixed_edge = IdentityShape("fixed-edge-a")
    second_fixed_edge = IdentityShape("fixed-edge-b")

    class WireExplorer:
        def __init__(self, _face, _kind):
            self.available = True

        def More(self):
            return self.available

        def Current(self):
            return wire

        def Next(self):
            self.available = False

    def wire_row(*, face_index, wire_index, wire, face):
        assert (face_index, wire_index) == (17, 0)
        return {
            "face_index": face_index,
            "wire_index": wire_index,
            "edge_count": 2,
            "edge_position_basis": "occ_1_based",
            "aggregate_self_intersection": True,
            "crossing_detail_status": "diagnosed",
            "pcurve_edge_positions": [1, 2],
            "seam_edge_positions": [],
            "occurrences": [{
                "kind": "adjacent",
                "edge_positions": [1, 2],
                "status": "detected",
            }],
            "occurrence_kinds": ["adjacent"],
            "_observed_wire": wire,
            "_occurrence_edges": {
                1: first_fixed_edge,
                2: second_fixed_edge,
            },
        }

    monkeypatch.setattr(TopExp, "TopExp_Explorer", WireExplorer)
    monkeypatch.setattr(TopoDS, "topods_Wire", lambda value: value)
    monkeypatch.setattr(diagnosis_module, "_wire_row_v2", wire_row)

    result = diagnose_face_wires_v2(
        object(),
        source_face_index=17,
        source_mapping={
            "status": "exact_identity",
            "wire_rows": [{
                "observed_wire": IdentityShape("wire"),
                "source_edge_candidates": [
                    {"source_edge_id": 41, "observed_edge": first_fixed_edge},
                    {"source_edge_id": 43, "observed_edge": second_fixed_edge},
                ],
            }],
        },
    )
    assert result["faces"][0]["face_index"] == 17
    assert result["faces"][0]["source_face_index"] == 17
    assert result["wires"][0]["source_face_index"] == 17
    assert result["occurrences"] == [{
        "face_index": 17,
        "wire_index": 0,
        "source_face_index": 17,
        "kind": "adjacent",
        "edge_positions": [1, 2],
        "status": "detected",
        "source_edge_ids": [41, 43],
        "source_mapping_status": "mapped",
    }]


def test_step_v2_preserves_existing_face_and_flattened_occurrence_semantics(
    monkeypatch, tmp_path
):
    from OCC.Core import STEPControl, TopExp, TopoDS
    import tools.diagnose_assembly_face_wires as diagnosis_module

    faces = [object(), object()]

    class Reader:
        def ReadFile(self, _path):
            return 1

        def TransferRoots(self):
            return None

        def OneShape(self):
            return object()

    class FaceExplorer:
        def __init__(self, _shape, _kind):
            self.position = 0

        def More(self):
            return self.position < len(faces)

        def Current(self):
            return faces[self.position]

        def Next(self):
            self.position += 1

    calls = []

    def diagnose_face(face, *, face_index, source_face_index=None, source_mapping=None):
        calls.append((face, face_index, source_face_index, source_mapping))
        occurrence = {
            "kind": "self_only",
            "edge_positions": [1],
            "status": "detected",
        }
        wire = {
            "face_index": face_index,
            "wire_index": 0,
            "occurrences": [occurrence],
        }
        return {
            "status": "diagnosed",
            "edge_position_basis": "occ_1_based",
            "faces": [{
                "face_index": face_index,
                "wire_count": 1,
                "wires_with_occurrences": [0],
                "occurrence_kinds": ["self_only"],
            }],
            "wires": [wire],
            "occurrences": [{
                "face_index": face_index,
                "wire_index": 0,
                **occurrence,
            }],
            "occurrence_kinds": ["self_only"],
        }

    monkeypatch.setattr(STEPControl, "STEPControl_Reader", Reader)
    monkeypatch.setattr(TopExp, "TopExp_Explorer", FaceExplorer)
    monkeypatch.setattr(TopoDS, "topods_Face", lambda face: face)
    monkeypatch.setattr(diagnosis_module, "diagnose_face_wires_v2", diagnose_face)

    result = diagnose_step_face_wires_v2(
        tmp_path / "ignored.step", breparg_root=tmp_path
    )
    assert calls == [
        (faces[0], 0, None, None),
        (faces[1], 1, None, None),
    ]
    assert result == {
        "status": "diagnosed",
        "edge_position_basis": "occ_1_based",
        "faces": [
            {
                "face_index": 0,
                "wire_count": 1,
                "wires_with_occurrences": [0],
                "occurrence_kinds": ["self_only"],
            },
            {
                "face_index": 1,
                "wire_count": 1,
                "wires_with_occurrences": [0],
                "occurrence_kinds": ["self_only"],
            },
        ],
        "wires": [
            {
                "face_index": 0,
                "wire_index": 0,
                "occurrences": [{
                    "kind": "self_only",
                    "edge_positions": [1],
                    "status": "detected",
                }],
            },
            {
                "face_index": 1,
                "wire_index": 0,
                "occurrences": [{
                    "kind": "self_only",
                    "edge_positions": [1],
                    "status": "detected",
                }],
            },
        ],
        "occurrences": [
            {
                "face_index": 0,
                "wire_index": 0,
                "kind": "self_only",
                "edge_positions": [1],
                "status": "detected",
            },
            {
                "face_index": 1,
                "wire_index": 0,
                "kind": "self_only",
                "edge_positions": [1],
                "status": "detected",
            },
        ],
        "occurrence_kinds": ["self_only"],
    }


def test_v2_no_step_case_is_explicitly_unavailable(tmp_path):
    source = tmp_path / "source.pkl"
    parsed = {
        "faceEdge_adj": [[0, 1, 2]],
        "edgeCorner_adj": np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        "edge_ncs": np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]),
    }
    with source.open("wb") as handle:
        pickle.dump(parsed, handle)
    row = build_case_row_v2(
        {
            "cad_id": "no-step",
            "source_path": str(source),
            "step_saved": False,
            "status": "assembly_error",
        },
        breparg_root=tmp_path / "unused",
    )
    assert row["step_diagnosis_available"] is False
    assert row["step_diagnosis"]["occurrences"] == [{
        "kind": "unavailable",
        "edge_positions": [],
        "status": "unavailable_no_saved_step",
    }]


def test_v2_summary_keeps_zero_count_occurrence_categories_explicit():
    summary = summarize_v2([{
        "cad_id": "no-step",
        "historical_step_saved": False,
        "step_diagnosis_available": False,
        "source_topology": {"suspicious_faces": []},
        "step_diagnosis": {
            "wires": [],
            "occurrences": [{
                "kind": "unavailable",
                "edge_positions": [],
                "status": "unavailable_no_saved_step",
            }],
        },
    }])
    assert summary["occurrence_counts"] == {
        "adjacent": 0,
        "closure": 0,
        "non_adjacent": 0,
        "self_only": 0,
        "pcurve_gap": 0,
        "seam": 0,
        "disconnected": 0,
        "unavailable": 1,
    }
