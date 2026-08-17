import json
from pathlib import Path

from tools.snapshot_assembly_repair import snapshot


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

    result = snapshot(run, report, label="pilot")

    assert result["valid"] is True
    archived = json.loads((report / "assembly_repair_attempts.jsonl").read_text())
    assert "source_path" not in archived
    assert "step_path" not in archived
    assert archived["step_sha256"] == "abc"
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))
