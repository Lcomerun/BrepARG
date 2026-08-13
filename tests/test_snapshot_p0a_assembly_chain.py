import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.snapshot_p0a_assembly_chain import snapshot, validate_evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    cases = []
    attempts = []
    for case_index in range(16):
        cad_id = f"cad-{case_index:02d}"
        cases.append(
            {
                "cad_id": cad_id,
                "attempts": 6,
                "attributed": True,
                "primary_cause": "wire_self_intersection",
                "joint_sensitive": case_index < 4,
                "tolerance_sensitive": case_index == 4,
                "any_variant_both_valid": case_index == 0,
            }
        )
        for joint in (200, 0):
            for tolerance in (1e-4, 1e-3, 1e-2):
                step = run / "steps" / f"joint{joint}_sew{tolerance:.0e}" / f"{cad_id}.step"
                step.parent.mkdir(parents=True, exist_ok=True)
                step.write_bytes(f"{cad_id}-{joint}-{tolerance}".encode())
                both = case_index == 0 and joint == 0
                attempts.append(
                    {
                        "cad_id": cad_id,
                        "joint_iterations": joint,
                        "sewing_tolerance": tolerance,
                        "status": "both_valid" if both else "step_invalid",
                        "step_saved": True,
                        "step_path": str(step),
                        "step_bytes": step.stat().st_size,
                        "step_sha256": _sha256(step),
                        "native_brep_valid": True,
                        "strict_brep_valid": both,
                        "both_valid": both,
                        "validity_components": {
                            "native_brep_valid": True,
                            "wire_count": 1,
                            "wire_order_failures": 0,
                            "wire_self_intersections": 0 if both else 1,
                            "shell_count": 1,
                            "shells_with_bad_edges": 0,
                            "free_edges": 0,
                            "solid_count": 1,
                        },
                    }
                )
    summary = {
        "cases": 16,
        "expected_cases": 16,
        "attempts": 96,
        "expected_attempts": 96,
        "complete_cases": 16,
        "attributed_cases": 16,
        "attribution_rate": 1.0,
        "matrix_complete": True,
        "gate_passed": True,
        "primary_cause_counts": {"wire_self_intersection": 16},
        "joint_sensitive_cases": 4,
        "tolerance_sensitive_cases": 1,
        "cases_with_any_both_valid_variant": 1,
        "advance_to_boundary_consistency": False,
        "source_manifest": str(tmp_path / "repo" / "reports" / "source" / "manifest.jsonl"),
        "source_manifest_sha256": "a" * 64,
    }
    (run / "assembly_chain_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "assembly_chain_cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (run / "assembly_chain_attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8"
    )
    (run / "repair_checklist.md").write_text("# Repairs\n", encoding="utf-8")
    return run


def test_snapshot_archives_only_lightweight_normalized_evidence(tmp_path):
    run = _fixture_run(tmp_path)
    report = tmp_path / "repo" / "reports" / "p0a"

    result = snapshot(run, report, repo_root=tmp_path / "repo")

    assert result["attempts"] == 96
    assert result["saved_steps_bound_by_sha256"] == 96
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))
    normalized = json.loads((report / "assembly_chain_summary.json").read_text())
    assert normalized["source_manifest"] == "reports/source/manifest.jsonl"
    assert normalized["step_bytes_archived"] is False
    with (report / "attempts_compact.csv").open(encoding="utf-8", newline="") as handle:
        compact = list(csv.DictReader(handle))
    assert len(compact) == 96
    assert all(not Path(row["step_relative_path"]).is_absolute() for row in compact)
    with (report / "step_sha256.csv").open(encoding="utf-8", newline="") as handle:
        steps = list(csv.DictReader(handle))
    assert len(steps) == 96
    assert all(row["step_bytes_archived"] == "False" for row in steps)
    readme = (report / "README.md").read_text(encoding="utf-8")
    assert "签名敏感”不等于“修复成功" in readme
    assert "16/16 明确归因" in readme
    manifest = json.loads((report / "artifact_manifest.json").read_text())
    archived = {item["path"] for item in manifest["artifacts"]}
    assert "README.md" in archived
    assert "step_sha256.csv" in archived


def test_snapshot_rejects_tampered_step(tmp_path):
    run = _fixture_run(tmp_path)
    step = next((run / "steps").rglob("*.step"))
    step.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="STEP byte count mismatch|STEP SHA-256 mismatch"):
        snapshot(run, tmp_path / "report", repo_root=tmp_path)


def test_validate_evidence_rejects_incomplete_matrix():
    summary = {
        "cases": 16,
        "expected_cases": 16,
        "attempts": 95,
        "expected_attempts": 96,
        "attributed_cases": 16,
        "primary_cause_counts": {"curve_fit": 16},
        "matrix_complete": False,
        "gate_passed": False,
    }
    cases = [
        {"cad_id": f"cad-{index}", "attributed": True} for index in range(16)
    ]

    with pytest.raises(RuntimeError, match="96 attempts"):
        validate_evidence(summary, cases, [])
