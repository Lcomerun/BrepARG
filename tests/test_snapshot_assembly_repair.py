import json
from pathlib import Path

import pytest

from tools.snapshot_assembly_repair import sha256_file, snapshot


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
    assert result["repair_diagnostics_present"] is True
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))


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
