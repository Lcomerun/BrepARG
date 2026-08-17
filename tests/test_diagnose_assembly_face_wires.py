import numpy as np

from tools.diagnose_assembly_face_wires import (
    _edge_walk_issue,
    frozen_p0a_baseline_rows,
    source_topology_summary,
    summarize,
)


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
