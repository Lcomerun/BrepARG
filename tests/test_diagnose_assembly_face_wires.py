import pickle

import numpy as np

from tools.diagnose_assembly_face_wires import (
    _edge_walk_issue,
    build_case_row_v2,
    collect_wire_occurrences,
    crossing_pair_candidates,
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
