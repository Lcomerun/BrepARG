import copy
import json
from dataclasses import asdict

import pytest

from tools.assembly_selector_geometry import GEOMETRY_GATE_SCHEMA
from tools.probe_periodic_pcurve_applicability import canonical_sha256, sha256_file
from tools.run_exact_cad_repair_feasibility import (
    RUN_NAME,
    RUN_SCHEMA,
    ROWS_NAME,
    SUMMARY_NAME,
    TARGET_CAD_IDS,
    VARIANTS,
    _base_row,
    summarize,
)
from tools.snapshot_exact_cad_repair_feasibility import (
    EXPECTED_REPORT_FILES,
    REQUIRED_SOURCE_HASHES,
    snapshot,
)


SOURCE_SHA = "a" * 64


def _binding(index):
    return {"bytes": 100 + index, "sha256": str(index + 1) * 64}


def _geometry_gate():
    return {
        "schema": GEOMETRY_GATE_SCHEMA,
        "accepted": False,
        "checks": {
            "measurement_completed": True,
            "no_wire_self_intersections": False,
        },
        "rejection_reasons": ["geometry_gate:no_wire_self_intersections"],
        "thresholds": {"max_bbox_relative_delta": 0.02},
        "bbox_relative_delta": 0.01,
        "input_face_count": 2,
        "candidate_face_count": 2,
    }


def _candidate_diagnostics(variant):
    if variant == VARIANTS[1]:
        return {
            "face_mutations": [
                {
                    "source_face_index": 10,
                    "attempted": True,
                    "accepted": False,
                    "reason": "pcurve_remove_not_proven",
                    "strategy": "targeted_nonperiodic_pcurve_reprojection",
                    "target_selection": {
                        "accepted": True,
                        "reason": "exact_adjacent_targets_selected",
                        "source_edge_pairs": [[13, 20]],
                        "target_source_edge_ids": [13, 20],
                        "targets": [],
                    },
                    "surgery": [
                        {
                            "source_edge_id": 13,
                            "accepted": False,
                            "reason": "pcurve_remove_not_proven",
                            "remove_returned": False,
                            "pcurve_absent_after_remove": False,
                        }
                    ],
                }
            ],
            "local_face_gates": {
                "mapping_exact": False,
                "non_target_pcurves_preserved": False,
            },
            "whole_cad_step_gate": {
                "accepted": False,
                "lineage_status": "exact_geometry_incidence",
                "mapping_exact": True,
                "mapping_failures": [],
                "target_definition_complete": True,
                "occurrences_complete": True,
                "malformed_occurrence_count": 0,
                "final_occurrence_count": 3,
                "target_defects_removed": False,
                "no_new_non_target_defects": False,
                "target_residuals": [
                    {
                        "source_face_index": 10,
                        "kind": "adjacent",
                        "status": "detected",
                        "source_edge_ids": [13, 20],
                    }
                ],
                "non_target_defects": [],
                "geometry_incidence_proof": {
                    "status": "exact_geometry_incidence",
                    "failure_codes": [],
                    "mapped_face_count": 44,
                    "mapped_edge_occurrence_count": 250,
                },
            },
        }
    return {
        "post_sewing_mutation": {
            "attempted": True,
            "accepted": False,
            "reason": "pcurve_remove_not_proven",
            "strategy": "post_sewing_exact_face_pcurve_reprojection_feasibility",
            "projection_precision": 1e-4,
            "target_source_face_index": 5,
            "target_source_edge_ids": [9, 23],
            "target_face_before": {"wire_self_intersections": 1},
            "target_selection": {
                "accepted": True,
                "reason": "exact_reversed_adjacent_closure_pairs_selected",
                "source_edge_pairs": [[9, 23]],
                "target_source_edge_ids": [9, 23],
                "targets": [],
            },
            "copy_topology_gate": {
                "accepted": True,
                "checks": {"gate_0_accepted": True},
                "rejection_reasons": [],
            },
            "copy_source_edge_identity_gate": {
                "accepted": True,
                "checks": {"no_split_after": True},
                "rejection_reasons": [],
            },
            "pcurve_reprojection": {
                "accepted": False,
                "reason": "pcurve_remove_not_proven",
                "operations": [
                    {
                        "source_edge_id": 9,
                        "remove_reported": False,
                        "pcurve_absent_after_remove": False,
                        "add_reported": False,
                        "pcurve_present_after_add": False,
                    }
                ],
            },
        }
    }


def _payload():
    bindings = [_binding(index) for index in range(2)]
    return {
        "schema": RUN_SCHEMA,
        "calibration_manifest_sha256": "3" * 64,
        "selector_matrix_sha256": "4" * 64,
        "selector_run": {
            "bytes": 200,
            "sha256": "5" * 64,
            "signature": "6" * 64,
            "status": "COMPLETED",
        },
        "lineage": {
            "cases_sha256": "7" * 64,
            "run_sha256": "8" * 64,
            "run_signature": "9" * 64,
            "source_bindings": [
                {"cad_id": cad_id, **binding}
                for cad_id, binding in zip(TARGET_CAD_IDS, bindings)
            ],
        },
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "ordered_task_ids": [variant.task_id for variant in VARIANTS],
        "variants": json.loads(json.dumps([asdict(value) for value in VARIANTS])),
        "sources": [
            {
                "cad_id": cad_id,
                "parent_id": f"parent-{index}",
                "historical_strict_valid": False,
                "selector_strict_valid": False,
                "binding": binding,
            }
            for index, (cad_id, binding) in enumerate(zip(TARGET_CAD_IDS, bindings))
        ],
        "joint_iterations": 200,
        "worker_timeout_seconds": 600.0,
        "repository": {
            "commit": "b" * 40,
            "dirty": False,
            "status_sha256": "c" * 64,
            "source_sha256": {
                name: "d" * 64 for name in sorted(REQUIRED_SOURCE_HASHES)
            },
        },
        "breparg_runtime": {"utils_sha256": "e" * 64},
    }


def _write_run(root):
    payload = _payload()
    signature = canonical_sha256(payload)
    bindings = {
        cad_id: source["binding"]
        for cad_id, source in zip(TARGET_CAD_IDS, payload["sources"])
    }
    rows = []
    for variant in VARIANTS:
        source_payload = next(
            item for item in payload["sources"] if item["cad_id"] == variant.cad_id
        )
        source = {
            "cad_id": variant.cad_id,
            "parent_id": source_payload["parent_id"],
            "brep_valid": False,
        }
        binding = bindings[variant.cad_id]
        row = _base_row(
            source, variant, run_signature=signature, expected_binding=binding
        )
        row.update(
            status="candidate_rejected" if variant.is_candidate else "control_reproduced",
            callback_completed=True,
            step_saved=False,
            step_readable=False,
            native_brep_valid=False,
            strict_brep_valid=False,
            both_valid=False,
            worker_returncode=0,
            source_binding_before=binding,
            source_binding_loaded_bytes=binding,
            source_binding_after_load=binding,
            source_binding_after_attempt=binding,
            geometry_topology_gate=_geometry_gate(),
            validity_components={
                "status": "diagnosed",
                "wire_self_intersections": variant.expected_control_wire_self_intersections,
            },
        )
        if variant.is_candidate:
            row["candidate_application"] = {
                "attempted": True,
                "applied": False,
                "status": "rejected_by_local_helper",
                "diagnostics": _candidate_diagnostics(variant),
            }
            row["defect_gate"]["source_binding_preserved"] = True
            row["defect_gate"]["rejection_reasons"] = [
                "candidate_not_applied",
                "target_defects_removed",
            ]
        else:
            row["candidate_application"] = {
                "attempted": False,
                "applied": False,
                "status": "control",
            }
            row["defect_gate"]["rejection_reasons"] = [
                "control_registered_failure"
            ]
            row["control_expectation"] = {
                "expected_native_brep_valid": variant.expected_control_native_valid,
                "expected_strict_brep_valid": False,
                "expected_wire_self_intersections": (
                    variant.expected_control_wire_self_intersections
                ),
                "observed_wire_self_intersections": (
                    variant.expected_control_wire_self_intersections
                ),
                "reproduced": True,
            }
        rows.append(row)

    rows_path = root / ROWS_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows)
    summary_path = root / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    run = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETED",
        "attempts": 4,
        "signature": signature,
        "payload": payload,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    (root / RUN_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")
    return rows, summary, run


def _rewrite_rows(root, rows, run):
    rows_path = root / ROWS_NAME
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run["rows_sha256"] = sha256_file(rows_path)
    (root / RUN_NAME).write_text(json.dumps(run) + "\n")


def test_snapshot_archives_compact_negative_four_cell_evidence(tmp_path):
    run_root = tmp_path / "run"
    report = tmp_path / "report"
    run_root.mkdir()
    _write_run(run_root)
    (run_root / "private.step").write_bytes(b"STEP")
    (run_root / "source.pkl").write_bytes(b"pickle")
    logs = run_root / "worker_logs"
    logs.mkdir()
    (logs / "worker.stdout.log").write_text("raw OCC output")

    result = snapshot(run_root, report)

    assert result["valid"] is True
    assert result["controls_reproduced"] == 2
    assert result["candidate_attempted"] == 2
    assert result["candidate_applied"] == 0
    assert result["candidate_rejected"] == 2
    assert result["worker_or_protocol_failures"] == 0
    assert result["nonfinite_count"] == 0
    assert result["does_not_generalize_to_all_pcurve_mechanisms"] is True
    assert {path.name for path in report.iterdir()} == EXPECTED_REPORT_FILES
    archived_text = "\n".join(path.read_text() for path in report.iterdir())
    assert "raw OCC output" not in archived_text
    assert "step_relative_path" not in archived_text
    assert "worker_stdout_log" not in archived_text
    assert "FixRemovePCurve" in (report / "README.md").read_text()
    rows = [json.loads(line) for line in (report / ROWS_NAME).read_text().splitlines()]
    candidates = [row for row in rows if row["arm"] == "candidate"]
    assert candidates[0]["candidate_application"]["causal_evidence"][
        "whole_cad_step_gate"
    ]["target_defects_removed"] is False
    assert candidates[1]["candidate_application"]["causal_evidence"][
        "post_sewing_mutation"
    ]["pcurve_reprojection"]["reason"] == "pcurve_remove_not_proven"
    manifest = json.loads((report / "artifact_manifest.json").read_text())
    assert {item["path"] for item in manifest["artifacts"]} == (
        EXPECTED_REPORT_FILES - {"artifact_manifest.json"}
    )
    for item in manifest["artifacts"]:
        path = report / item["path"]
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == sha256_file(path)


def test_snapshot_rejects_terminal_hash_or_summary_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _write_run(root)
    summary = json.loads((root / SUMMARY_NAME).read_text())
    summary["controls_reproduced"] = 1
    (root / SUMMARY_NAME).write_text(json.dumps(summary))

    with pytest.raises(RuntimeError, match="summary hash"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_source_binding_chain_drift_even_when_rehashed(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_run(root)
    rows[0]["source_binding_after_attempt"] = {"bytes": 1, "sha256": SOURCE_SHA}
    _rewrite_rows(root, rows, run)

    with pytest.raises(RuntimeError, match="attempt validation|source binding"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_candidate_rejection_without_actual_attempt(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_run(root)
    rows[1]["candidate_application"] = {
        "attempted": False,
        "applied": False,
        "status": "not_attempted",
    }
    _rewrite_rows(root, rows, run)

    with pytest.raises(RuntimeError, match="candidate rejection"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_source_hash_population_drift(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_run(root)
    run["payload"]["repository"]["source_sha256"].pop(
        "tools/directed_trim_assembly.py"
    )
    run["signature"] = canonical_sha256(run["payload"])
    for row in rows:
        row["run_signature"] = run["signature"]
    _rewrite_rows(root, rows, run)

    with pytest.raises(RuntimeError, match="source hash population"):
        snapshot(root, tmp_path / "report")
