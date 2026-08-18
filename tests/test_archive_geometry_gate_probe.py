import json
from pathlib import Path

import pytest

from tools.archive_geometry_gate_probe import archive, compact_row, failure_family


def _run(tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "assembly_repair_matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run / "assembly_repair_run.json").write_text(
        json.dumps(
            {
                "schema": "assembly-repair-run-v2",
                "signature": "signed",
                "status": "COMPLETED_PARTIAL",
                "attempts": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return run


def _row(cad_id: str, *, gate=None, status="both_valid") -> dict:
    return {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": cad_id,
        "parent_id": "parent",
        "profile": "probe",
        "switches": ["directed_trim"],
        "historical_strict_valid": False,
        "status": status,
        "step_saved": status == "both_valid",
        "native_brep_valid": status == "both_valid",
        "strict_brep_valid": status == "both_valid",
        "both_valid": status == "both_valid",
        "step_bytes": 12,
        "step_sha256": "0" * 64,
        "validity_components": {
            "wire_self_intersections": 0,
            "shell_count": 1,
            "solid_count": 1,
        },
        "selector_geometry_topology_gate": gate,
    }


def test_compact_row_drops_paths_and_keeps_gate():
    gate = {
        "schema": "assembly-selector-geometry-gate-v2",
        "accepted": True,
        "checks": {"face_count_equal": True},
        "rejection_reasons": [],
        "thresholds": {"max_bbox_relative_delta": 0.02},
        "unlisted_measurement": 1.0,
    }
    row = _row("cad", gate=gate)
    compact = compact_row(row)
    assert compact["selector_geometry_topology_gate"]["accepted"] is True
    assert "unlisted_measurement" not in compact["selector_geometry_topology_gate"]
    assert "source_path" not in compact


def test_compact_row_rejects_path_bearing_gate():
    with pytest.raises(ValueError, match="path"):
        compact_row(_row("cad", gate={"accepted": False, "source_path": "D:\\x"}))


def test_failure_family_classifies_curve_and_wire():
    assert failure_family(_row("curve", status="assembly_error") | {"error": "curve_fit_not_done"}) == "curve_fit"
    wire = _row("wire")
    wire["status"] = "step_invalid"
    wire["both_valid"] = False
    wire["validity_components"]["wire_self_intersections"] = 1
    assert failure_family(wire) == "closure_or_self_intersection"


def test_archive_records_gate_counts_and_forbids_binary_artifacts(tmp_path):
    rows = [
        _row("accepted", gate={
            "schema": "assembly-selector-geometry-gate-v2",
            "accepted": True,
            "checks": {},
            "rejection_reasons": [],
        }),
        _row("missing", gate=None, status="step_invalid"),
    ]
    run = _run(tmp_path, rows)
    report = tmp_path / "report"
    summary = archive(run, report, label="test")
    assert summary["attempts"] == 2
    assert summary["gate_present"] == 1
    assert summary["gate_accepted"] == 1
    assert summary["gate_missing"] == 1
    assert not list(report.rglob("*.step"))
    assert not list(report.rglob("*.pkl"))
    assert (report / "geometry_gate_attempts.jsonl").is_file()
