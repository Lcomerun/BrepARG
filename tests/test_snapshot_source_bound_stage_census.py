from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from tools import probe_source_bound_stage_census as census
from tools.assembly_stage_lineage import STAGE_NAMES, STAGE_ORDER
from tools.assembly_stage_lineage import (
    GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
    STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
    STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
    STEP_VERTEX_IDENTITY_PROOF_METHOD,
)
from tools.snapshot_source_bound_stage_census import (
    ARCHIVE_SCHEMA,
    EMPTY_GIT_STATUS_SHA256,
    EXPECTED_REPORT_FILES,
    FROZEN_BREPARG_UTILS_SHA256,
    FROZEN_RUNTIME_IDENTITY,
    FROZEN_CALIBRATION_MANIFEST_SHA256,
    FROZEN_SELECTOR_MATRIX_SHA256,
    FROZEN_SELECTOR_RUN_SHA256,
    REQUIRED_SOURCE_HASHES,
    _compact_source_vertex_lineage,
    snapshot,
    validate_artifact_manifest,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_snapshot_global_vertex_proof_accepts_registered_preassignment_failure():
    proof = _source_vertex_lineage(constraint_occurrence_count=7)
    proof.update(
        status="ambiguous",
        solution_count=None,
        mapped_source_vertex_count=0,
        mapped_observed_vertex_count=0,
        max_observed_per_source=0,
        max_source_per_observed=0,
        failure_codes=["source_vertex_constraint_occurrence_coverage_incomplete"],
    )

    assert _compact_source_vertex_lineage(proof, label="proof") == proof


@pytest.mark.parametrize(
    ("failure", "solution_count"),
    [
        ("source_vertex_assignment_missing", 2),
        ("source_vertex_assignment_nonunique", 0),
        ("source_vertex_assignment_constraint_replay_failed", None),
    ],
)
def test_snapshot_global_vertex_proof_binds_failure_to_solution_count(
    failure, solution_count
):
    proof = _source_vertex_lineage(constraint_occurrence_count=8)
    proof.update(
        status="ambiguous",
        solution_count=solution_count,
        mapped_source_vertex_count=0,
        mapped_observed_vertex_count=0,
        max_observed_per_source=0,
        max_source_per_observed=0,
        failure_codes=[failure],
    )

    with pytest.raises(RuntimeError, match="contradicts|lacks solution_count"):
        _compact_source_vertex_lineage(proof, label="proof")


def _binding(index: int) -> dict[str, object]:
    return {"bytes": 100 + index, "sha256": _digest(f"source-{index}")}


def _source_topology() -> dict[str, object]:
    return {
        "face_count": 2,
        "edge_count": 4,
        "vertex_count": 4,
        "face_edge_occurrence_count": 8,
        "face_edge_incidence_counts": [4, 4],
        "edge_face_incidence_counts": [2, 2, 2, 2],
        "vertex_edge_incidence_counts": [2, 2, 2, 2],
        "face_edge_source_ids": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "edge_face_source_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "edge_vertex_source_ids": [[0, 1], [1, 2], [2, 3], [0, 3]],
    }


def _lineage(stage: str) -> dict[str, object]:
    value: dict[str, object] = {
        "status": "exact_identity",
        "proof_method": "fixture_source_identity",
        "solution_count": 1,
        "failure_codes": [],
        "entities": {},
    }
    if stage == "S1":
        value.update(source_face_ids=[0, 1], source_edge_ids=[0, 1, 2, 3])
    elif stage == "S2":
        value.update(source_edge_ids=[0, 1, 2, 3])
    else:
        value.update(
            source_face_ids=[0, 1],
            source_edge_ids=[0, 1, 2, 3],
            source_edge_occurrence_keys=[
                [face_id, edge_id, 0]
                for face_id in range(2)
                for edge_id in range(4)
            ],
        )
    return value


def _source_vertex_lineage(*, constraint_occurrence_count: int) -> dict[str, object]:
    return {
        "status": "exact_identity",
        "proof_method": GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": 4,
        "observed_vertex_count": 4,
        "mapped_source_vertex_count": 4,
        "mapped_observed_vertex_count": 4,
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": constraint_occurrence_count,
        "failure_codes": [],
    }


def _stage_local_proof(
    stage: str, *, scope_count: int, edge_count: int, constraint_count: int
) -> dict[str, object]:
    return {
        "status": "exact_stage_local_topology",
        "proof_method": STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
        "scope_kind": "source_edge" if stage == "S2" else "source_face",
        "scope_count": scope_count,
        "source_edge_count": edge_count,
        "constraint_occurrence_count": constraint_count,
        "max_observed_per_source_within_scope": 1,
        "max_source_per_observed_within_scope": 1,
        "failure_codes": [],
    }


def _step_geometry_incidence_proof() -> dict[str, object]:
    return {
        "status": "exact_geometry_incidence",
        "failure_codes": [],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_candidate_degree_counts": {"1": 2},
        "face_matching_count_capped": 1,
        "vertex_proof_required": True,
        "mapped_face_count": 2,
        "mapped_edge_occurrence_count": 8,
        "vertex_proof_status": "exact",
        "vertex_proof_method": STEP_VERTEX_IDENTITY_PROOF_METHOD,
        "vertex_tolerance_normalized": (
            STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ),
        "vertex_candidate_degree_counts": {"1": 4},
        "vertex_matching_count_capped": 1,
        "source_vertex_count": 4,
        "step_vertex_count": 4,
        "mapped_source_edge_count": 4,
        "edge_endpoint_pair_expected_count": 4,
        "edge_endpoint_pair_proof_count": 4,
        "edge_endpoint_occurrence_expected_count": 8,
        "edge_endpoint_occurrence_proof_count": 8,
        "self_loop_endpoint_pair_expected_count": 0,
        "self_loop_endpoint_pair_proof_count": 0,
    }


def _stages() -> list[dict[str, object]]:
    source = _source_topology()
    result: list[dict[str, object]] = []
    for stage in STAGE_ORDER:
        if stage == "S1":
            topology = {"face_count": 2}
        elif stage == "S2":
            topology = {
                "edge_count": 4,
                "vertex_count": 4,
                "vertex_edge_incidence_counts": [2, 2, 2, 2],
            }
        else:
            topology = source
        record: dict[str, object] = {
            "stage": stage,
            "phase": STAGE_NAMES[stage],
            "status": "observed",
            "lineage": _lineage(stage),
            "topology": topology,
            "defects": [{"code": "wire_adjacent"}] if stage == "S4" else [],
        }
        if stage == "S1":
            record["evidence"] = {
                "observation_granularity": "per_source_edge",
                "observed_source_edge_count": 4,
                "unique_source_edge_count": 4,
                "complete_order_independent_source_edge_coverage": True,
            }
        if stage == "S2":
            record["evidence"] = {
                "stage_local_occ_topology_proof": _stage_local_proof(
                    stage, scope_count=4, edge_count=4, constraint_count=4
                )
            }
        if stage in {"S3", "S4"}:
            record["evidence"] = {
                "stage_local_occ_topology_proof": _stage_local_proof(
                    stage, scope_count=2, edge_count=4, constraint_count=8
                ),
            }
        if stage in {"S5", "S6"}:
            record["evidence"] = {
                "source_vertex_lineage": _source_vertex_lineage(
                    constraint_occurrence_count=8
                )
            }
        if stage == "S6":
            record["construction_native_valid"] = True
        if stage == "S7":
            record.update(
                reimport_native_valid=True,
                strict_valid=False,
                evidence={
                    "step_bytes": 321,
                    "step_sha256": _digest("step"),
                    "validity_components": {
                        "status": "diagnosed",
                        "wire_self_intersections": 1,
                    },
                    "step_geometry_incidence_proof": (
                        _step_geometry_incidence_proof()
                    ),
                },
            )
        result.append(record)
    return result


def _payload() -> dict[str, object]:
    sources = [
        {
            "cad_id": cad_id,
            "parent_id": f"parent-{index}",
            "historical_strict_valid": False,
            "selector_strict_valid": False,
            "binding": _binding(index),
        }
        for index, cad_id in enumerate(census.TARGET_CAD_IDS)
    ]
    return {
        "schema": census.RUN_SCHEMA,
        "run_kind": "formal",
        "calibration_manifest_sha256": FROZEN_CALIBRATION_MANIFEST_SHA256,
        "selector_matrix_sha256": FROZEN_SELECTOR_MATRIX_SHA256,
        "selector_run": {
            "bytes": 456,
            "sha256": FROZEN_SELECTOR_RUN_SHA256,
            "signature": _digest("selector-signature"),
            "status": "COMPLETED",
        },
        "selector": {
            "cohort_count": 100,
            "strict_valid": 91,
            "historical_valid_preserved": 84,
            "regressions": 0,
            "residual_cad_ids": sorted(
                set(census.TARGET_CAD_IDS)
                | set(census.EXCLUDED_EXACT_NEGATIVE_CAD_IDS)
            ),
        },
        "exact_negative_evidence": dict(census.EXACT_NEGATIVE_EVIDENCE),
        "excluded_exact_negative_cad_ids": sorted(
            census.EXCLUDED_EXACT_NEGATIVE_CAD_IDS
        ),
        "ordered_target_cad_ids": list(census.TARGET_CAD_IDS),
        "ordered_tasks": json.loads(
            json.dumps(
                [asdict(task) | {"task_id": task.task_id} for task in census.TASKS]
            )
        ),
        "sources": sources,
        "stages": [
            {"stage": stage, "phase": STAGE_NAMES[stage]} for stage in STAGE_ORDER
        ],
        "stage_record_schema": census.STAGE_RECORD_SCHEMA,
        "stage_assessment_schema": census.ASSESSMENT_SCHEMA,
        "schema_v2": {
            "identity": "assembly-selector-geometry-gate-v2",
            "max_bbox_relative_delta": 0.02,
            "max_edge_length_relative_delta": 0.05,
            "max_edge_sample_rms_normalized": 0.01,
            "max_edge_sample_max_normalized": 0.05,
            "unchanged": True,
        },
        "joint_iterations": 200,
        "worker_timeout_seconds": 600.0,
        "python": copy.deepcopy(FROZEN_RUNTIME_IDENTITY["python"]),
        "native_runtime": {
            key: copy.deepcopy(FROZEN_RUNTIME_IDENTITY[key])
            for key in (
                "schema", "scope", "process_isolation", "numpy", "pythonocc",
                "occt",
            )
        },
        "repository": {
            "commit": "b" * 40,
            "upstream_commit": "b" * 40,
            "head_matches_upstream": True,
            "dirty": False,
            "formal": True,
            "status_sha256": EMPTY_GIT_STATUS_SHA256,
            "source_sha256": {
                name: _digest(name) for name in sorted(REQUIRED_SOURCE_HASHES)
            },
        },
        "breparg_runtime": {"utils_sha256": FROZEN_BREPARG_UTILS_SHA256},
        "authorization_ceiling": "exact_candidate_design_only",
    }


def _write_formal_run(root):
    payload = _payload()
    signature = census.canonical_sha256(payload)
    source_by_id = {item["cad_id"]: item for item in payload["sources"]}
    rows = []
    for task in census.TASKS:
        source_payload = source_by_id[task.cad_id]
        source = {
            "cad_id": task.cad_id,
            "parent_id": source_payload["parent_id"],
            "brep_valid": False,
        }
        binding = source_payload["binding"]
        assessment = census.assess_stage_lineage(
            _stages(), source_topology=_source_topology()
        )
        row = census._base_row(
            source,
            task,
            run_signature=signature,
            expected_binding=binding,
        )
        row.update(
            status="completed",
            worker_runtime_abi_sentinel=copy.deepcopy(
                FROZEN_RUNTIME_IDENTITY
            ),
            stage_records=assessment["stages"],
            assessment=assessment,
            source_binding_before_load=binding,
            source_binding_loaded_bytes=binding,
            source_binding_after_load=binding,
            source_binding_after_measurement=binding,
            source_binding_parent_after_child=binding,
            step_roundtrip={
                "saved_to_persistent_output": True,
                "artifact_id": f"attempt-{task.ordinal:02d}-opaque",
                "bytes": 321,
                "sha256": _digest("step"),
            },
            elapsed_seconds=1.25,
            worker_returncode=0,
            worker_stdout_log=f"worker_logs/{task.ordinal}.stdout.log",
            worker_stderr_log=f"worker_logs/{task.ordinal}.stderr.log",
        )
        rows.append(row)

    rows_path = root / census.ROWS_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = census.summarize(rows)
    summary_path = root / census.SUMMARY_NAME
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    run = {
        "schema": census.RUN_SCHEMA,
        "signature": signature,
        "payload": payload,
        "status": "COMPLETED",
        "attempts": 10,
        "rows_sha256": census.sha256_file(rows_path),
        "summary_sha256": census.sha256_file(summary_path),
    }
    (root / census.RUN_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")
    return rows, summary, run


def _rewrite_rows(root, rows, run, *, rewrite_summary=True):
    rows_path = root / census.ROWS_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    run["rows_sha256"] = census.sha256_file(rows_path)
    if rewrite_summary:
        summary = census.summarize(rows)
        summary_path = root / census.SUMMARY_NAME
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        run["summary_sha256"] = census.sha256_file(summary_path)
    (root / census.RUN_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")


def _resign_payload(root, rows, run):
    run["signature"] = census.canonical_sha256(run["payload"])
    for row in rows:
        row["run_signature"] = run["signature"]
    _rewrite_rows(root, rows, run, rewrite_summary=True)


def test_snapshot_archives_exact_compact_git_safe_evidence(tmp_path):
    root = tmp_path / "run"
    report = tmp_path / "report"
    root.mkdir()
    _write_formal_run(root)
    (root / "private.step").write_bytes(b"STEP")
    (root / "source.pkl").write_bytes(b"pickle")
    logs = root / "worker_logs"
    logs.mkdir()
    (logs / "1.stdout.log").write_text("raw OCC output D:\\private\\shape.step")

    result = snapshot(root, report)

    assert result["valid"] is True
    assert result["attempts"] == 10
    assert result["scientifically_conclusive_assessments"] == 10
    assert result["bridge_repairs_counted"] == 0
    assert result["selector_strict_valid_after"] == 91
    assert result["authorizes_training"] is False
    assert {path.name for path in report.iterdir()} == EXPECTED_REPORT_FILES
    validate_artifact_manifest(report)

    archived_text = "\n".join(path.read_text() for path in report.iterdir())
    assert "raw OCC output" not in archived_text
    assert "worker_stdout_log" not in archived_text
    assert "worker_stderr_log" not in archived_text
    assert "D:\\private" not in archived_text

    compact_rows = [
        json.loads(line)
        for line in (report / census.ROWS_NAME).read_text().splitlines()
    ]
    assert len(compact_rows) == 10
    assert [row["task_id"] for row in compact_rows] == [
        task.task_id for task in census.TASKS
    ]
    assert all(row["source_binding_chain"]["all_equal"] for row in compact_rows)
    assert all(
        row["source_binding_chain"]["parent_after_child_equal"]
        for row in compact_rows
    )
    assert all(row["step_roundtrip"]["bytes_archived"] is False for row in compact_rows)
    assert all("artifact_id" not in row["step_roundtrip"] for row in compact_rows)
    assert all(
        row["worker_runtime_abi_sentinel"] == FROZEN_RUNTIME_IDENTITY
        for row in compact_rows
    )
    assert "attempt-01-opaque" not in archived_text
    for row in compact_rows:
        stages = row["assessment"]["stages"]
        assert stages[1]["evidence"]["stage_local_occ_topology_proof"][
            "constraint_occurrence_count"
        ] == 4
        assert stages[2]["evidence"]["stage_local_occ_topology_proof"][
            "proof_method"
        ] == STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD
        proof = stages[6]["evidence"]["step_geometry_incidence_proof"]
        assert proof["vertex_proof_required"] is True
        assert proof["vertex_matching_count_capped"] == 1
        assert proof["edge_endpoint_occurrence_proof_count"] == 8
    manifest = json.loads((report / "artifact_manifest.json").read_text())
    assert manifest["schema"] == ARCHIVE_SCHEMA
    assert {item["path"] for item in manifest["artifacts"]} == (
        EXPECTED_REPORT_FILES - {"artifact_manifest.json"}
    )


def test_snapshot_rejects_signature_and_terminal_hash_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _rows, _summary, run = _write_formal_run(root)
    run["payload"]["joint_iterations"] = 201
    (root / census.RUN_NAME).write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RuntimeError, match="signature"):
        snapshot(root, tmp_path / "report-signature")

    root2 = tmp_path / "run2"
    root2.mkdir()
    _write_formal_run(root2)
    summary = json.loads((root2 / census.SUMMARY_NAME).read_text())
    summary["selector_strict_valid_after"] = 92
    (root2 / census.SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary hash"):
        snapshot(root2, tmp_path / "report-hash")


@pytest.mark.parametrize(
    ("component", "field", "value"),
    [
        ("python", "executable_sha256", "0" * 64),
        ("numpy", "version", "0.0.0"),
        ("pythonocc", "wrapper_binary_sha256", "1" * 64),
        ("occt", "kernel_binary_sha256", "2" * 64),
        ("occt", "file_version", "7.7.3.0"),
    ],
)
def test_snapshot_rejects_resigned_runtime_identity_drift(
    tmp_path, component, field, value
):
    root = tmp_path / f"runtime-{component}-{field}"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    target = (
        run["payload"]["python"]
        if component == "python"
        else run["payload"]["native_runtime"][component]
    )
    target[field] = value
    _resign_payload(root, rows, run)

    with pytest.raises(RuntimeError, match="runtime differs from the frozen"):
        snapshot(root, tmp_path / f"report-{component}-{field}")


@pytest.mark.parametrize(
    ("payload_field", "match"),
    [
        ("calibration_manifest_sha256", "calibration_manifest_sha256.*frozen"),
        ("selector_matrix_sha256", "selector_matrix_sha256.*frozen"),
        ("selector_run.sha256", "selector run.*frozen"),
    ],
)
def test_snapshot_rejects_resigned_frozen_input_digest_tampering(
    tmp_path, payload_field, match
):
    root = tmp_path / payload_field.replace(".", "-")
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    if payload_field == "selector_run.sha256":
        run["payload"]["selector_run"]["sha256"] = _digest("tampered-selector-run")
    else:
        run["payload"][payload_field] = _digest(f"tampered-{payload_field}")
    # Recompute every outer binding.  The immutable input-value gate must be
    # the reason the archive still rejects this internally consistent forgery.
    _resign_payload(root, rows, run)

    with pytest.raises(RuntimeError, match=match):
        snapshot(root, tmp_path / f"report-{payload_field.replace('.', '-')}")


@pytest.mark.parametrize(
    "artifact_id",
    ["../escape", "subdir/attempt", r"subdir\attempt", "C:/attempt", ".", ""],
)
def test_snapshot_rejects_unsafe_step_artifact_id_without_reading_step(
    tmp_path, artifact_id
):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0]["step_roundtrip"]["artifact_id"] = artifact_id
    _rewrite_rows(root, rows, run, rewrite_summary=False)

    with pytest.raises(
        RuntimeError, match="artifact_id|attempt validation|path/native/finite safe"
    ):
        snapshot(root, tmp_path / "report")


def test_snapshot_accepts_runner_compatible_dotted_step_artifact_id(tmp_path):
    root = tmp_path / "run"
    report = tmp_path / "report"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    for row in rows:
        row["step_roundtrip"]["artifact_id"] += ".retry-1"
    _rewrite_rows(root, rows, run, rewrite_summary=False)

    result = snapshot(root, report)

    assert result["valid"] is True
    archived_text = "\n".join(path.read_text() for path in report.iterdir())
    assert ".retry-1" not in archived_text


def test_snapshot_rejects_task_order_and_summary_derivation_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0], rows[1] = rows[1], rows[0]
    _rewrite_rows(root, rows, run, rewrite_summary=False)
    with pytest.raises(RuntimeError, match="ordered ten-task"):
        snapshot(root, tmp_path / "report-order")

    root2 = tmp_path / "run2"
    root2.mkdir()
    _rows, summary2, run2 = _write_formal_run(root2)
    summary2["selector_strict_valid_after"] = 90
    summary_path = root2 / census.SUMMARY_NAME
    summary_path.write_text(json.dumps(summary2) + "\n", encoding="utf-8")
    run2["summary_sha256"] = census.sha256_file(summary_path)
    (root2 / census.RUN_NAME).write_text(json.dumps(run2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="derivable"):
        snapshot(root2, tmp_path / "report-summary")


@pytest.mark.parametrize(
    "field",
    ["source_binding_after_measurement", "source_binding_parent_after_child"],
)
def test_snapshot_rejects_child_or_parent_source_binding_drift(tmp_path, field):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0][field] = {"bytes": 1, "sha256": _digest("drift")}
    _rewrite_rows(root, rows, run, rewrite_summary=False)
    with pytest.raises(RuntimeError, match="attempt validation|source binding"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_stage_assessment_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0]["assessment"]["first_bad_stage"] = "S5"
    _rewrite_rows(root, rows, run, rewrite_summary=False)
    with pytest.raises(RuntimeError, match="attempt validation|assessment|derivable"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_dirty_or_source_hash_population_drift(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    run["payload"]["repository"]["dirty"] = True
    run["signature"] = census.canonical_sha256(run["payload"])
    for row in rows:
        row["run_signature"] = run["signature"]
    _rewrite_rows(root, rows, run, rewrite_summary=False)
    with pytest.raises(RuntimeError, match="clean formal"):
        snapshot(root, tmp_path / "report-dirty")

    root2 = tmp_path / "run2"
    root2.mkdir()
    rows2, _summary2, run2 = _write_formal_run(root2)
    run2["payload"]["repository"]["source_sha256"].pop(
        "tools/directed_trim_assembly.py"
    )
    run2["signature"] = census.canonical_sha256(run2["payload"])
    for row in rows2:
        row["run_signature"] = run2["signature"]
    _rewrite_rows(root2, rows2, run2, rewrite_summary=False)
    with pytest.raises(RuntimeError, match="source hash population"):
        snapshot(root2, tmp_path / "report-hashes")


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("error_type", "failed at D:\\secret\\source.pkl", "path"),
        ("error_type", "<TopoDS_Shape object at 0x1234abcd>", "native"),
    ],
)
def test_snapshot_rejects_raw_paths_or_native_handles(tmp_path, key, value, match):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0][key] = value
    _rewrite_rows(root, rows, run, rewrite_summary=False)
    with pytest.raises(RuntimeError, match=match):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_unknown_stage_evidence_and_nonempty_report(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0]["stage_records"][0]["evidence"]["new_unreviewed_metric"] = 1
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    _rewrite_rows(root, rows, run, rewrite_summary=True)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        snapshot(root, tmp_path / "report-unknown")

    root2 = tmp_path / "run2"
    root2.mkdir()
    _write_formal_run(root2)
    report = tmp_path / "report-nonempty"
    report.mkdir()
    (report / "existing.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="must be empty"):
        snapshot(root2, report)


@pytest.mark.parametrize(
    ("stage_index", "nested_key", "mutation", "match"),
    [
        (
            2,
            "stage_local_occ_topology_proof",
            lambda proof: proof.pop("proof_method"),
            "key set drifted",
        ),
        (
            3,
            "stage_local_occ_topology_proof",
            lambda proof: proof.__setitem__("private_assignment", [0, 1, 2, 3]),
            "key set drifted",
        ),
        (
            4,
            "source_vertex_lineage",
            lambda proof: proof.__setitem__("source_to_observed_assignment", [0, 1, 2, 3]),
            "key set drifted",
        ),
        (
            5,
            "source_vertex_lineage",
            lambda proof: proof.__setitem__("constraint_occurrence_count", 7),
            "exact proof drifted|assessment|census_conclusive drifted",
        ),
        (
            6,
            "step_geometry_incidence_proof",
            lambda proof: proof.pop("vertex_proof_required"),
            "key set drifted",
        ),
        (
            6,
            "step_geometry_incidence_proof",
            lambda proof: proof.__setitem__("source_vertex_points", [[0.0, 0.0, 0.0]]),
            "key set drifted",
        ),
        (
            6,
            "step_geometry_incidence_proof",
            lambda proof: proof.__setitem__("vertex_matching_count_capped", 2),
            "vertex_matching_count_capped.*drifted|assessment|census_conclusive drifted",
        ),
    ],
)
def test_snapshot_rejects_missing_extra_or_tampered_nested_proof(
    tmp_path, stage_index, nested_key, mutation, match
):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    proof = rows[0]["stage_records"][stage_index]["evidence"][nested_key]
    mutation(proof)
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    rows[0]["status"] = (
        "completed"
        if rows[0]["assessment"]["conclusive"]
        else "scientific_inconclusive"
    )
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    with pytest.raises(RuntimeError, match=match):
        snapshot(root, tmp_path / "report")


def test_snapshot_accepts_real_downstream_unavailable_s7_short_circuit(tmp_path):
    root = tmp_path / "run"
    report = tmp_path / "report"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    for stage in rows[0]["stage_records"]:
        stage["defects"] = []
    rows[0]["stage_records"][5]["construction_native_valid"] = False
    s7 = rows[0]["stage_records"][6]
    s7["lineage"] = {
        "status": "missing",
        "failure_codes": ["face_assignment_has_no_perfect_matching"],
    }
    s7["evidence"]["step_geometry_incidence_proof"] = {
        "status": "unavailable",
        "failure_codes": ["face_assignment_has_no_perfect_matching"],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_candidate_degree_counts": {"0": 1, "1": 1},
        "face_matching_count_capped": 0,
        "vertex_proof_required": True,
        "vertex_proof_status": "not_evaluated",
    }
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    assert rows[0]["assessment"]["first_bad_stage"] == "S6"
    assert rows[0]["assessment"]["conclusive"] is True
    rows[0]["status"] = "completed"
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    result = snapshot(root, report)

    assert result["valid"] is True
    archived = [
        json.loads(line)
        for line in (report / census.ROWS_NAME).read_text().splitlines()
    ]
    proof = archived[0]["assessment"]["stages"][6]["evidence"][
        "step_geometry_incidence_proof"
    ]
    assert proof == s7["evidence"]["step_geometry_incidence_proof"]


def test_snapshot_rejects_unregistered_unavailable_s7_proof_shape(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    proof = rows[0]["stage_records"][6]["evidence"][
        "step_geometry_incidence_proof"
    ]
    proof.update(
        status="unavailable",
        failure_codes=["face_assignment_has_no_perfect_matching"],
        face_matching_count_capped=0,
        vertex_proof_status="not_evaluated",
    )
    # An unavailable face-assignment short circuit must not pretend that the
    # exact-only mapped populations and global vertex proof were measured.
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    rows[0]["status"] = "completed"
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    with pytest.raises(RuntimeError, match="key set drifted for non-exact"):
        snapshot(root, tmp_path / "report")


@pytest.mark.parametrize(
    ("status", "failure", "matching_count", "degrees", "match"),
    [
        (
            "ambiguous",
            "face_assignment_has_no_perfect_matching",
            2,
            {"1": 2},
            "face no-match semantics drifted",
        ),
        (
            "unavailable",
            "face_assignment_not_unique",
            0,
            {"2": 2},
            "face ambiguity semantics drifted",
        ),
        (
            "unavailable",
            "face_assignment_has_no_perfect_matching",
            0,
            {"0": 1},
            "face candidate population drifted",
        ),
    ],
)
def test_snapshot_rejects_nonexact_s7_face_semantic_contradictions(
    tmp_path, status, failure, matching_count, degrees, match
):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    s7 = rows[0]["stage_records"][6]
    s7["lineage"] = {"status": "missing", "failure_codes": [failure]}
    s7["evidence"]["step_geometry_incidence_proof"] = {
        "status": status,
        "failure_codes": [failure],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_candidate_degree_counts": degrees,
        "face_matching_count_capped": matching_count,
        "vertex_proof_required": True,
        "vertex_proof_status": "not_evaluated",
    }
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    rows[0]["status"] = "completed"
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    with pytest.raises(RuntimeError, match=match):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_zero_population_candidate_bucket(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0]["stage_records"][6]["evidence"][
        "step_geometry_incidence_proof"
    ]["face_candidate_degree_counts"] = {"0": 0, "1": 2}
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    with pytest.raises(RuntimeError, match="positive integer"):
        snapshot(root, tmp_path / "report")


@pytest.mark.parametrize(
    ("nested_key", "field", "value", "match"),
    [
        (
            "source_vertex_lineage",
            "native_handle",
            "<TopoDS_Vertex object at 0x1234abcd>",
            "native",
        ),
        (
            "step_geometry_incidence_proof",
            "debug_path",
            "D:\\secret\\roundtrip.step",
            "path",
        ),
        (
            "step_geometry_incidence_proof",
            "absolute_vertex_points",
            [[10.0, 20.0, 30.0]],
            "unexpected keys|key set drifted|attempt validation failed",
        ),
    ],
)
def test_snapshot_rejects_private_nested_proof_payloads(
    tmp_path, nested_key, field, value, match
):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    stage_index = 4 if nested_key == "source_vertex_lineage" else 6
    rows[0]["stage_records"][stage_index]["evidence"][nested_key][field] = value
    # Keep the previously signed assessment so the archive boundary itself
    # must inspect and reject the hostile raw row.  Native/path evidence is
    # intentionally invalid even before independent assessment recomputation.
    _rewrite_rows(root, rows, run, rewrite_summary=False)

    with pytest.raises(RuntimeError, match=match):
        snapshot(root, tmp_path / "report")


@pytest.mark.parametrize(
    ("stage_index", "field", "value"),
    [
        (2, "constraint_occurrence_count", 7),
        (2, "source_edge_count", 3),
        (2, "proof_method", "unregistered_method"),
    ],
)
def test_snapshot_rejects_tampered_s3_stage_local_proof_scalars(
    tmp_path, stage_index, field, value
):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_formal_run(root)
    rows[0]["stage_records"][stage_index]["evidence"][
        "stage_local_occ_topology_proof"
    ][field] = value
    rows[0]["assessment"] = census.assess_stage_lineage(
        rows[0]["stage_records"], source_topology=_source_topology()
    )
    rows[0]["status"] = "scientific_inconclusive"
    _rewrite_rows(root, rows, run, rewrite_summary=True)

    with pytest.raises(
        RuntimeError,
        match="census_conclusive drifted|assessment|attempt validation failed",
    ):
        snapshot(root, tmp_path / "report")


def test_artifact_manifest_detects_post_archive_mutation(tmp_path):
    root = tmp_path / "run"
    report = tmp_path / "report"
    root.mkdir()
    _write_formal_run(root)
    snapshot(root, report)
    with (report / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="size drifted|hash drifted"):
        validate_artifact_manifest(report)
