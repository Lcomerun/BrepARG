import json
from pathlib import Path

import pytest

from tools.snapshot_assembly_repair import compact_selection, sha256_file, snapshot


def write_completed_run_manifest(run: Path, *, attempts: int = 1) -> dict:
    summary_path = run / "assembly_repair_summary.json"
    manifest = {
        "schema": "assembly-repair-run-v2",
        "signature": "signed-run",
        "status": "COMPLETED",
        "attempts": attempts,
        "summary_sha256": sha256_file(summary_path),
        "payload": {"selected_cohort_count": attempts},
    }
    (run / "assembly_repair_run.json").write_text(json.dumps(manifest))
    return manifest


def test_snapshot_excludes_source_and_step_paths(tmp_path):
    run = tmp_path / "run"
    report = tmp_path / "report"
    run.mkdir()
    row = {
        "schema": "assembly-repair-matrix-v1", "cad_id": "cad",
        "parent_id": "parent", "profile": "directed_trim",
        "switches": ["directed_trim"], "historical_strict_valid": False,
        "status": "both_valid", "step_saved": True,
        "native_brep_valid": True, "strict_brep_valid": True, "both_valid": True,
        "source_path": "secret.pkl", "step_path": "cad.step",
        "step_bytes": 123, "step_sha256": "abc", "validity_components": {},
        "assembly_diagnostics": {
            "directed_trim_loop_policies": [
                {"face_index": 0, "mode": "regrouped_directed"},
                {
                    "face_index": 1,
                    "mode": "historical_fallback_unproven_topology",
                },
            ],
            "local_intersection_topology": [
                {"face_index": 0, "attempted": True, "accepted": True, "reason": "accepted"}
            ],
            "solid_topology_repair": {
                "candidate_pair_count": 8,
                "mutual_pair_count": 8,
                "merged_vertex_count": 8,
                "applied": True,
            },
        },
    }
    (run / "assembly_repair_matrix.jsonl").write_text(json.dumps(row) + "\n")
    summary = {
        "gate_passed": False,
        "profiles": [{
            "profile": "directed_trim", "attempts": 1, "step_readable": 1,
            "native_valid": 1, "strict_valid": 1, "both_valid": 1,
            "restored_cad_ids": ["cad"], "regressed_cad_ids": [],
        }],
    }
    (run / "assembly_repair_summary.json").write_text(json.dumps(summary))
    manifest = write_completed_run_manifest(run)

    result = snapshot(run, report, label="pilot")

    assert result["valid"] is True
    archived = json.loads((report / "assembly_repair_attempts.jsonl").read_text())
    assert "source_path" not in archived
    assert "step_path" not in archived
    assert archived["step_sha256"] == "abc"
    archived_run = json.loads((report / "assembly_repair_run.json").read_text())
    assert archived_run == manifest
    assert result["run_signature"] == "signed-run"
    assert result["summary_sha256"] == manifest["summary_sha256"]
    diagnostics = json.loads(
        (report / "repair_diagnostics_summary.json").read_text()
    )
    assert diagnostics["directed_trim_loop_policies"]["mode_counts"] == {
        "historical_fallback_unproven_topology": 1,
        "regrouped_directed": 1,
    }
    assert diagnostics["local_intersection_topology"]["accepted_cad_ids"] == [
        "cad"
    ]
    assert diagnostics["solid_topology_repair"] == {
        "candidate_pair_count": 8,
        "mutual_pair_count": 8,
        "merged_vertex_count": 8,
        "applied_cad_ids": ["cad"],
    }
    assert result["repair_diagnostics_present"] is True
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))
    artifact_manifest = json.loads((report / "artifact_manifest.json").read_text())
    for artifact in artifact_manifest["artifacts"]:
        path = report / artifact["path"]
        assert b"\r" not in path.read_bytes()
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == sha256_file(path)


def test_snapshot_rejects_summary_not_bound_to_run_manifest(tmp_path):
    run = tmp_path / "run"
    report = tmp_path / "report"
    run.mkdir()
    row = {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": "cad",
        "profile": "baseline",
        "historical_strict_valid": True,
        "strict_brep_valid": True,
    }
    (run / "assembly_repair_matrix.jsonl").write_text(json.dumps(row) + "\n")
    (run / "assembly_repair_summary.json").write_text(
        json.dumps({"gate_passed": False, "profiles": []})
    )
    write_completed_run_manifest(run)
    (run / "assembly_repair_summary.json").write_text(
        json.dumps({"gate_passed": True, "profiles": []})
    )

    with pytest.raises(RuntimeError, match="summary hash"):
        snapshot(run, report, label="tampered")


def test_snapshot_drops_transient_error_text_from_archived_manifest(tmp_path):
    run = _single_solid_run(tmp_path)
    manifest_path = run / "assembly_repair_run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["error_type"] = "FileNotFoundError"
    manifest["error"] = r"D:\private\input.pkl"
    manifest_path.write_text(json.dumps(manifest))

    snapshot(run, tmp_path / "report", label="sanitized")

    archived = json.loads(
        (tmp_path / "report" / "assembly_repair_run.json").read_text()
    )
    assert "error" not in archived
    assert "error_type" not in archived
    assert "D:\\private" not in (
        tmp_path / "report" / "assembly_repair_run.json"
    ).read_text()


def test_selector_snapshot_rejects_unbound_candidate_ledger(tmp_path):
    run = _single_solid_run(tmp_path)
    matrix_path = run / "assembly_repair_matrix.jsonl"
    candidate_path = run / "assembly_selector_candidates.jsonl"
    candidate_path.write_text("{}\n")
    manifest_path = run / "assembly_repair_run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload"] = {"run_kind": "assembly-repair-selector-v1"}
    manifest["final_matrix_sha256"] = sha256_file(matrix_path)
    manifest["candidate_manifest_sha256"] = "not-the-ledger-hash"
    manifest["candidate_attempts"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="candidate ledger hash"):
        snapshot(run, tmp_path / "report", label="tampered-selector")


def _single_solid_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "schema": "assembly-repair-matrix-v1", "cad_id": "cad",
        "parent_id": "parent", "profile": "single_solid",
        "switches": ["single_solid"], "historical_strict_valid": False,
        "status": "both_valid", "step_saved": True,
        "native_brep_valid": True, "strict_brep_valid": True, "both_valid": True,
    }
    (run / "assembly_repair_matrix.jsonl").write_text(json.dumps(row) + "\n")
    (run / "assembly_repair_summary.json").write_text(json.dumps({
        "gate_passed": False,
        "profiles": [{
            "profile": "single_solid", "attempts": 1, "step_readable": 1,
            "native_valid": 1, "strict_valid": 1, "both_valid": 1,
            "restored_cad_ids": ["cad"], "regressed_cad_ids": [],
        }],
    }))
    write_completed_run_manifest(run)
    return run


def test_snapshot_archives_only_path_free_solid_topology_diagnosis(tmp_path):
    run = _single_solid_run(tmp_path)
    report = tmp_path / "report"
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({
        "schema": "solid-topology-diagnosis-v1", "cad_id": "cad",
        "source_pickle": {"archived": False, "sha256": "source"},
    }))

    result = snapshot(
        run,
        report,
        label="solid",
        solid_topology_diagnosis=diagnosis,
    )

    assert result["solid_topology_diagnosis_archived"] is True
    archived = json.loads((report / "solid_topology_diagnosis.json").read_text())
    assert archived["cad_id"] == "cad"


def test_snapshot_rejects_absolute_path_in_solid_topology_diagnosis(tmp_path):
    run = _single_solid_run(tmp_path)
    diagnosis = tmp_path / "diagnosis.json"
    diagnosis.write_text(json.dumps({
        "schema": "solid-topology-diagnosis-v1", "cad_id": "cad",
        "source_path": "D:\\private\\source.pkl",
    }))

    with pytest.raises(RuntimeError, match="path field"):
        snapshot(
            run,
            tmp_path / "report",
            label="solid",
            solid_topology_diagnosis=diagnosis,
        )


def test_snapshot_binds_equivalent_reference_cohort_without_paths(tmp_path):
    run = _single_solid_run(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "assembly_repair_attempts.jsonl").write_text(
        json.dumps(
            {
                "cad_id": "cad",
                "parent_id": "parent",
                "historical_strict_valid": False,
            }
        )
        + "\n"
    )

    result = snapshot(
        run,
        tmp_path / "report",
        label="equivalent",
        reference_report_dir=reference,
    )

    assert result["cohort_equivalence_valid"] is True
    binding = json.loads(
        (tmp_path / "report" / "cohort_equivalence.json").read_text()
    )
    assert binding["valid"] is True
    assert binding["same_cad_set"] is True
    assert binding["same_parent_and_historical_strict_map"] is True
    assert str(tmp_path) not in json.dumps(binding, sort_keys=True)


def test_snapshot_rejects_non_equivalent_reference_cohort(tmp_path):
    run = _single_solid_run(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "assembly_repair_attempts.jsonl").write_text(
        json.dumps(
            {
                "cad_id": "other",
                "parent_id": "parent",
                "historical_strict_valid": False,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="does not match"):
        snapshot(
            run,
            tmp_path / "report",
            label="mismatch",
            reference_report_dir=reference,
        )


def test_selector_snapshot_whitelists_candidate_evidence_and_rejects_paths():
    selection = {
        "schema": "assembly-repair-selector-v1",
        "primary_profile": "primary",
        "fallback_order": ["fallback"],
        "attempted_profiles": ["primary", "fallback"],
        "selected_profile": "fallback",
        "selected_reason": "fallback_native_strict_geometry_passed",
        "fallback_accepted": True,
        "candidates": [
            {
                "profile": "primary",
                "switches": ["primary"],
                "status": "step_invalid",
                "step_saved": True,
                "native_brep_valid": False,
                "strict_brep_valid": False,
                "both_valid": False,
                "step_path": r"D:\private\candidate.step",
                "worker_stdout_log": r"D:\private\worker.log",
                "error": r"D:\private\raw error",
                "rejection_reasons": ["candidate_strict_invalid"],
            }
        ],
    }
    with pytest.raises(RuntimeError, match="path"):
        compact_selection(selection)

    safe = json.loads(json.dumps(selection))
    safe["candidates"][0].pop("step_path")
    safe["candidates"][0].pop("worker_stdout_log")
    safe["candidates"][0].pop("error")
    safe["candidates"][0]["geometry_topology_gate"] = {
        "schema": "assembly-selector-geometry-gate-v2",
        "accepted": True,
        "input_vertex_edge_incidence_counts": [1, 1, 2],
        "candidate_vertex_edge_incidence_counts": [1, 1, 2],
        "input_projection_sample_count": 16,
        "input_to_candidate_sample_count": 16,
        "input_to_candidate_projected_sample_count": 16,
        "candidate_to_input_sample_count": 32,
        "candidate_to_input_projected_sample_count": 32,
    }
    compact = compact_selection(safe)
    archived_candidate = compact["candidates"][0]
    assert "step_path" not in archived_candidate
    assert "worker_stdout_log" not in archived_candidate
    assert "error" not in archived_candidate
    archived_gate = archived_candidate["geometry_topology_gate"]
    assert archived_gate["input_vertex_edge_incidence_counts"] == [1, 1, 2]
    assert archived_gate["candidate_vertex_edge_incidence_counts"] == [1, 1, 2]
    assert archived_gate["input_projection_sample_count"] == 16
    assert archived_gate["candidate_to_input_projected_sample_count"] == 32
