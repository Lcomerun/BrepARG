import copy
import json

import pytest

from tools.probe_downstream_bad_wire_lineage import (
    MEMORY_PHASES,
    PROFILE,
    RUN_MANIFEST_NAME,
    RUN_SCHEMA,
    ROWS_NAME,
    SCHEMA,
    STEP_PHASE,
    SUMMARY_NAME,
    TARGET_CAD_IDS,
    canonical_sha256,
    sha256_file,
    summarize,
)
from tools.snapshot_downstream_bad_wire_lineage import (
    EXPECTED_REPORT_FILES,
    REQUIRED_SOURCE_HASHES,
    snapshot,
)


SOURCE_SHA = "b" * 64


def _diagnosis(*occurrences, proof=None):
    result = {
        "status": "diagnosed",
        "faces": [],
        "wires": [],
        "occurrences": list(occurrences),
        "occurrence_kinds": sorted(
            {str(value.get("kind")) for value in occurrences}
        ),
    }
    if proof is not None:
        result["geometry_incidence_proof"] = proof
    return result


def _mapped_occurrence(face_index=0):
    return {
        "status": "detected",
        "kind": "non_adjacent",
        "wire_index": 0,
        "edge_positions": [1, 2],
        "source_face_index": face_index,
        "source_mapping_status": "mapped",
        "source_edge_ids": [0, 1],
    }


def _observations(*, downstream_bad=False):
    values = []
    for phase in MEMORY_PHASES:
        for face_index in range(2):
            occurrences = []
            if face_index == 0 and (
                phase == MEMORY_PHASES[0]
                or (downstream_bad and phase == MEMORY_PHASES[1])
            ):
                occurrences = [_mapped_occurrence(face_index)]
            values.append(
                {
                    "phase": phase,
                    "source_face_index": face_index,
                    "lineage_status": "exact_identity",
                    "mapping_failures": [],
                    "diagnosis": _diagnosis(*occurrences),
                }
            )
    values.append(
        {
            "phase": STEP_PHASE,
            "entity_kind": "step_shape",
            "lineage_status": "exact_geometry_incidence",
            "mapping_failures": [],
            "diagnosis": _diagnosis(
                proof={
                    "status": "exact_geometry_incidence",
                    "failure_codes": [],
                    "tolerance_normalized": 1e-4,
                    "face_candidate_degree_counts": {"1": 2},
                    "face_matching_count_capped": 1,
                    "mapped_face_count": 2,
                    "mapped_edge_occurrence_count": 4,
                }
            ),
        }
    )
    return values


def _case(cad_id, parent_id, signature, *, downstream_bad=False):
    observations = _observations(downstream_bad=downstream_bad)
    first_phase = MEMORY_PHASES[0]
    return {
        "schema": SCHEMA,
        "cad_id": cad_id,
        "parent_id": parent_id,
        "profile": PROFILE,
        "run_signature": signature,
        "source_binding": {"bytes": 123, "sha256": SOURCE_SHA},
        "source_binding_loaded_bytes": {"bytes": 123, "sha256": SOURCE_SHA},
        "source_binding_after_load": {"bytes": 123, "sha256": SOURCE_SHA},
        "status": "completed",
        "assembly_status": "completed",
        "step_roundtrip_status": "diagnosed",
        "source_face_count": 2,
        "source_edge_count": 4,
        "observations": observations,
        "phase_counts": {
            **{phase: 2 for phase in MEMORY_PHASES},
            STEP_PHASE: 1,
        },
        "all_stages_observed": True,
        "coverage_failure_count": 0,
        "observation_failure_count": 0,
        "mapping_failure_count": 0,
        "mapped_defect_count": 1 + int(downstream_bad),
        "first_bad_phase": first_phase,
        "first_bad_occurrences": [
            {
                "source_face_index": 0,
                "wire_index": 0,
                "kind": "non_adjacent",
                "source_edge_ids": [0, 1],
            }
        ],
    }


def _payload():
    return {
        "schema": RUN_SCHEMA,
        "calibration_manifest_sha256": "1" * 64,
        "selector_matrix_sha256": "2" * 64,
        "selector_run": {
            "bytes": 1,
            "sha256": "3" * 64,
            "signature": "4" * 64,
            "status": "COMPLETED",
        },
        "selector_cohort_signature": "5" * 64,
        "selector_strict_valid": 91,
        "selector_historical_strict_valid": 84,
        "selector_residual_ids": ["residual"],
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "source_bindings": [
            {"cad_id": cad_id, "bytes": 123, "sha256": SOURCE_SHA}
            for cad_id in TARGET_CAD_IDS
        ],
        "profile": PROFILE,
        "memory_phases": list(MEMORY_PHASES),
        "step_phase": STEP_PHASE,
        "joint_iterations": 200,
        "worker_timeout_seconds": 600.0,
        "repository": {
            "commit": "a" * 40,
            "dirty": False,
            "status_sha256": "6" * 64,
            "source_sha256": {
                name: "7" * 64 for name in sorted(REQUIRED_SOURCE_HASHES)
            },
        },
        "breparg_runtime": {"utils_sha256": "8" * 64},
    }


def _write_run(root):
    payload = _payload()
    signature = canonical_sha256(payload)
    rows = [
        _case(cad_id, f"parent-{index}", signature)
        for index, cad_id in enumerate(TARGET_CAD_IDS)
    ]
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
        "attempts": 2,
        "signature": signature,
        "payload": payload,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    (root / RUN_MANIFEST_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")
    return rows, summary, run


def _rewrite_run(root, run):
    (root / RUN_MANIFEST_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")


def test_snapshot_archives_only_compact_git_safe_evidence(tmp_path):
    run_root = tmp_path / "run"
    report = tmp_path / "report"
    run_root.mkdir()
    _write_run(run_root)
    logs = run_root / "worker_logs"
    logs.mkdir()
    (logs / "worker.stdout.log").write_text("private raw output")
    (run_root / "candidate.step").write_bytes(b"STEP bytes")
    (run_root / "source.pkl").write_bytes(b"pickle bytes")

    result = snapshot(run_root, report)

    assert result["valid"] is True
    assert result["cases"] == result["completed_cases"] == 2
    assert set(path.name for path in report.iterdir()) == EXPECTED_REPORT_FILES
    assert not any(path.suffix in {".step", ".pkl", ".log"} for path in report.iterdir())
    archived_text = "\n".join(
        path.read_text(encoding="utf-8") for path in report.iterdir()
    )
    assert "private raw output" not in archived_text
    assert "observed_wire" not in archived_text
    assert "observed_edge" not in archived_text
    archived_cases = [
        json.loads(line)
        for line in (report / ROWS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert all("observations" not in row for row in archived_cases)
    assert archived_cases[0]["phase_evidence"][-1]["geometry_incidence_proofs"][0][
        "status"
    ] == "exact_geometry_incidence"
    manifest = json.loads((report / "artifact_manifest.json").read_text())
    assert {item["path"] for item in manifest["artifacts"]} == (
        EXPECTED_REPORT_FILES - {"artifact_manifest.json"}
    )
    for item in manifest["artifacts"]:
        artifact = report / item["path"]
        assert artifact.stat().st_size == item["bytes"]
        assert sha256_file(artifact) == item["sha256"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows, summary, run: run.__setitem__("status", "INCONCLUSIVE"), "completed"),
        (lambda rows, summary, run: run.__setitem__("attempts", 1), "two-case"),
        (
            lambda rows, summary, run: summary.__setitem__("mapping_failures", 1),
            "hash differs",
        ),
        (
            lambda rows, summary, run: run["payload"]["repository"].__setitem__(
                "dirty", True
            ),
            "signature",
        ),
    ],
)
def test_snapshot_rejects_incomplete_or_tampered_run(tmp_path, mutation, message):
    root = tmp_path / "run"
    root.mkdir()
    rows, summary, run = _write_run(root)
    mutation(rows, summary, run)
    if summary != json.loads((root / SUMMARY_NAME).read_text()):
        (root / SUMMARY_NAME).write_text(json.dumps(summary) + "\n")
    _rewrite_run(root, run)

    with pytest.raises(RuntimeError, match=message):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_path_even_when_field_would_be_compacted_away(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_run(root)
    rows[0]["observations"][0]["step_path"] = "D:/private/candidate.step"
    rows_path = root / ROWS_NAME
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run["rows_sha256"] = sha256_file(rows_path)
    _rewrite_run(root, run)

    with pytest.raises(RuntimeError, match="path-free"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_serialized_native_handle_trace(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rows, _summary, run = _write_run(root)
    rows[0]["observations"][0]["observed_wire"] = "TopoDS_Wire object"
    rows_path = root / ROWS_NAME
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run["rows_sha256"] = sha256_file(rows_path)
    _rewrite_run(root, run)

    with pytest.raises((RuntimeError, ValueError), match="native|private"):
        snapshot(root, tmp_path / "report")


def test_snapshot_rejects_source_hash_population_drift_even_if_resigned(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _rows, _summary, run = _write_run(root)
    run["payload"]["repository"]["source_sha256"].pop(
        "tools/directed_trim_assembly.py"
    )
    run["signature"] = canonical_sha256(run["payload"])
    # Case signatures deliberately remain old: repository validation must fail
    # before an archive could accept the weakened source binding.
    _rewrite_run(root, run)

    with pytest.raises(RuntimeError, match="source hash population"):
        snapshot(root, tmp_path / "report")
