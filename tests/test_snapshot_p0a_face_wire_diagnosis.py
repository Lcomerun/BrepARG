import json

from tools.snapshot_p0a_face_wire_diagnosis import snapshot


def test_snapshot_removes_local_paths_and_forbidden_source_bytes(tmp_path):
    run, report = tmp_path / "run", tmp_path / "report"
    run.mkdir()
    row = {
        "cad_id": "cad", "source_path": "private.pkl", "step_path": "private.step",
        "source_pickle_sha256": "pickle-hash", "step_sha256": "step-hash",
        "step_diagnosis_available": True, "step_diagnosis": {"faces": [], "wires": []},
        "source_topology": {}, "validity_components": {},
    }
    (run / "face_wire_cases.jsonl").write_text("".join(json.dumps({**row, "cad_id": f"cad-{index}"}) + "\n" for index in range(16)), encoding="utf-8")
    (run / "face_wire_summary.json").write_text(json.dumps({
        "cases": 16, "input_path": "D:/private/attempts.jsonl", "self_intersection_cases": 1,
        "order_failure_cases": 0, "step_diagnosis_available": 11, "step_diagnosis_unavailable": 5,
        "self_intersection_faces_by_cad": [{"cad_id": "cad-0", "face_indices": [2]}],
        "order_failure_faces_by_cad": [], "schema": "test",
    }), encoding="utf-8")

    result = snapshot(run, report)

    assert result["valid"] is True
    archived = json.loads((report / "face_wire_cases.jsonl").read_text().splitlines()[0])
    assert "source_path" not in archived
    assert "step_path" not in archived
    summary = json.loads((report / "face_wire_summary.json").read_text())
    assert "input_path" not in summary
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))
