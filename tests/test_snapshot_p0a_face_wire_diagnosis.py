import json

from tools.snapshot_p0a_face_wire_diagnosis import V1_SCHEMA, V2_SCHEMA, snapshot


def test_snapshot_removes_local_paths_and_forbidden_source_bytes(tmp_path):
    run, report = tmp_path / "run", tmp_path / "report"
    run.mkdir()
    row = {
        "schema": V1_SCHEMA,
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
        "order_failure_faces_by_cad": [], "schema": V1_SCHEMA,
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


def test_v2_snapshot_validates_schema_population_and_sanitizes_paths(tmp_path):
    run, report = tmp_path / "run", tmp_path / "report"
    run.mkdir()
    rows = []
    for index in range(16):
        available = index < 11
        rows.append({
            "schema": V2_SCHEMA,
            "cad_id": f"cad-{index}",
            "source_path": f"D:/private/cad-{index}.pkl",
            "step_path": f"D:/private/cad-{index}.step" if available else None,
            "historical_step_saved": available,
            "source_pickle_sha256": f"source-{index}",
            "step_sha256": f"step-{index}" if available else None,
            "step_diagnosis_available": available,
            "step_diagnosis": {
                "status": "diagnosed" if available else "unavailable_no_saved_step",
                "faces": [],
                "wires": [],
                "occurrences": [{
                    "kind": "adjacent" if available else "unavailable",
                    "edge_positions": [1, 2] if available else [],
                    "status": "detected" if available else "unavailable_no_saved_step",
                }],
            },
            "source_topology": {},
            "validity_components": {} if available else None,
        })
    (run / "face_wire_cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run / "face_wire_summary.json").write_text(json.dumps({
        "schema": V2_SCHEMA,
        "cases": 16,
        "historical_step_saved": 11,
        "step_diagnosis_available": 11,
        "step_diagnosis_unavailable": 5,
        "edge_position_basis": "occ_1_based",
        "occurrence_counts": {"adjacent": 11, "unavailable": 5},
        "occurrence_case_counts": {"adjacent": 11, "unavailable": 5},
        "self_intersection_wire_count": 11,
        "self_intersection_wires_with_classified_occurrences": 11,
        "input_path": "D:/private/attempts.jsonl",
    }), encoding="utf-8")

    result = snapshot(run, report)

    assert result == {
        "valid": True,
        "schema": V2_SCHEMA,
        "cases": 16,
        "step_diagnosis_available": 11,
        "step_diagnosis_unavailable": 5,
        "forbidden_artifacts": [],
        "source_bytes_archived": False,
        "step_bytes_archived": False,
    }
    archived = json.loads((report / "face_wire_cases.jsonl").read_text().splitlines()[0])
    assert "source_path" not in archived
    assert "step_path" not in archived
    assert archived["step_diagnosis"]["occurrences"][0]["edge_positions"] == [1, 2]
    summary = json.loads((report / "face_wire_summary.json").read_text())
    assert "input_path" not in summary
    assert "crossing diagnosis v2" in (report / "README.md").read_text(encoding="utf-8")
