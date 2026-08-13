import json
from pathlib import Path

import pytest

from tools.diagnose_assembly_chain import (
    RUNNER_VERSION,
    StageFailure,
    build_variants,
    classify_case,
    run_attempt,
    select_frozen_failures,
    summarize_cases,
    write_repair_checklist,
)


def _manifest_row(tmp_path: Path, cad_id: str, *, valid: bool, arm: str = "original"):
    source = tmp_path / f"{cad_id}.pkl"
    source.write_bytes(b"source")
    return {
        "cad_id": cad_id,
        "parent_id": "a" * 24,
        "arm": arm,
        "brep_valid": valid,
        "status": "saved" if valid else "brep_invalid",
        "source_path": str(source),
    }


def _write_manifest(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _attempt(cad_id="cad-1", *, joint=200, tolerance=1e-3, **updates):
    row = {
        "runner_version": RUNNER_VERSION,
        "cad_id": cad_id,
        "joint_iterations": joint,
        "sewing_tolerance": tolerance,
        "status": "step_invalid",
        "step_saved": True,
        "native_brep_valid": True,
        "strict_brep_valid": False,
        "both_valid": False,
        "validity_components": {
            "native_brep_valid": True,
            "wire_order_failures": 0,
            "wire_self_intersections": 0,
            "shells_with_bad_edges": 0,
            "free_edges": 0,
            "solid_count": 1,
        },
    }
    row.update(updates)
    return row


def test_select_frozen_failures_binds_hash_and_rejects_wrong_count(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _manifest_row(tmp_path, "bad-2", valid=False),
        _manifest_row(tmp_path, "good", valid=True),
        _manifest_row(tmp_path, "bad-1", valid=False),
        _manifest_row(tmp_path, "other-arm", valid=False, arm="bypass"),
    ]
    _write_manifest(manifest, rows)

    selected = select_frozen_failures(manifest, expected_count=2)

    assert [row["cad_id"] for row in selected] == ["bad-1", "bad-2"]
    assert len({row["source_manifest_sha256"] for row in selected}) == 1
    assert len(selected[0]["source_manifest_sha256"]) == 64
    with pytest.raises(RuntimeError, match="expected 3"):
        select_frozen_failures(manifest, expected_count=3)


def test_variant_matrix_is_exact_cross_product():
    variants = build_variants((200, 0), (1e-4, 1e-3, 1e-2))
    assert len(variants) == 6
    assert variants[0] == {"joint_iterations": 200, "sewing_tolerance": 1e-4}
    assert variants[-1] == {"joint_iterations": 0, "sewing_tolerance": 1e-2}


def test_run_attempt_captures_stage_entity_without_raising(tmp_path):
    def failing_pipeline(*args, **kwargs):
        raise StageFailure(
            "curve_fit",
            "degenerate edge",
            entity_kind="edge",
            entity_index=7,
            cause_type="StdFail_NotDone",
            details={
                "curve_fit_attempts": [
                    {"edge_index": 7, "tolerance": 0.005, "status": "failed"}
                ]
            },
        )

    row = run_attempt(
        {},
        {
            "cad_id": "cad-1",
            "parent_id": "a" * 24,
            "source_path": "cad.pkl",
            "source_manifest": "manifest.jsonl",
            "source_manifest_sha256": "hash",
        },
        200,
        1e-3,
        tmp_path,
        tmp_path,
        pipeline_runner=failing_pipeline,
    )

    assert row["status"] == "stage_error"
    assert row["failure_stage"] == "curve_fit"
    assert row["failure_entity_kind"] == "edge"
    assert row["failure_entity_index"] == 7
    assert row["error_type"] == "StdFail_NotDone"
    assert row["curve_fit_attempts"][0]["tolerance"] == 0.005
    assert row["step_saved"] is False


def test_run_attempt_preserves_native_strict_and_both_valid(tmp_path):
    def successful_pipeline(*args, **kwargs):
        return {
            "status": "step_invalid",
            "step_saved": True,
            "strict_brep_valid": False,
            "validity_components": {"native_brep_valid": True},
        }

    row = run_attempt(
        {},
        {
            "cad_id": "cad-1",
            "source_path": "cad.pkl",
            "source_manifest": "manifest.jsonl",
            "source_manifest_sha256": "hash",
        },
        200,
        1e-3,
        tmp_path,
        tmp_path,
        pipeline_runner=successful_pipeline,
    )
    assert row["native_brep_valid"] is True
    assert row["strict_brep_valid"] is False
    assert row["both_valid"] is False


def test_run_attempt_preserves_saved_step_on_post_export_failure(tmp_path):
    def failing_after_export(*args, **kwargs):
        raise StageFailure(
            "strict_check",
            "checker crashed",
            cause_type="RuntimeError",
            details={
                "step_saved": True,
                "step_path": "cad.step",
                "step_bytes": 123,
                "step_sha256": "abc",
                "native_brep_valid": True,
            },
        )

    row = run_attempt(
        {},
        {
            "cad_id": "cad-1",
            "source_path": "cad.pkl",
            "source_manifest": "manifest.jsonl",
            "source_manifest_sha256": "hash",
        },
        200,
        1e-3,
        tmp_path,
        tmp_path,
        pipeline_runner=failing_after_export,
    )

    assert row["status"] == "stage_error"
    assert row["failure_stage"] == "strict_check"
    assert row["step_saved"] is True
    assert row["native_brep_valid"] is True
    result = classify_case(
        [
            row,
            _attempt(joint=0, tolerance=1e-3),
            _attempt(joint=200, tolerance=1e-4),
            _attempt(joint=200, tolerance=1e-2),
            _attempt(joint=0, tolerance=1e-4),
            _attempt(joint=0, tolerance=1e-2),
        ]
    )
    assert result["primary_cause"] == "post_step:strict_check"


def test_classify_case_names_pre_step_failure_and_sensitivities():
    rows = [
        _attempt(
            joint=200,
            tolerance=1e-3,
            status="stage_error",
            step_saved=False,
            failure_stage="wire_build",
            validity_components={},
        ),
        _attempt(joint=0, tolerance=1e-3, status="both_valid", strict_brep_valid=True, both_valid=True),
        _attempt(joint=200, tolerance=1e-4, status="stage_error", step_saved=False, failure_stage="wire_build", validity_components={}),
        _attempt(joint=200, tolerance=1e-2, status="both_valid", strict_brep_valid=True, both_valid=True),
        _attempt(joint=0, tolerance=1e-4),
        _attempt(joint=0, tolerance=1e-2),
    ]

    result = classify_case(rows)

    assert result["primary_cause"] == "pre_step:wire_build"
    assert result["attributed"] is True
    assert result["joint_sensitive"] is True
    assert result["tolerance_sensitive"] is True


def test_classify_case_decomposes_strict_component_failure():
    rows = [
        _attempt(
            joint=200,
            tolerance=1e-3,
            validity_components={
                "native_brep_valid": True,
                "wire_order_failures": 0,
                "wire_self_intersections": 2,
                "shells_with_bad_edges": 0,
                "free_edges": 0,
                "solid_count": 1,
            },
        ),
        _attempt(joint=0, tolerance=1e-3),
        _attempt(joint=200, tolerance=1e-4),
        _attempt(joint=200, tolerance=1e-2),
        _attempt(joint=0, tolerance=1e-4),
        _attempt(joint=0, tolerance=1e-2),
    ]

    result = classify_case(rows)

    assert result["primary_cause"] == "wire_self_intersection"
    assert result["baseline_strict_failures"] == ["wire_self_intersection"]
    assert result["attributed"] is True


def test_classify_case_detects_component_sensitivity_while_still_invalid():
    baseline_components = {
        "native_brep_valid": True,
        "wire_order_failures": 0,
        "wire_self_intersections": 1,
        "shells_with_bad_edges": 0,
        "free_edges": 0,
        "shell_count": 1,
        "solid_count": 1,
    }
    changed_components = {**baseline_components, "wire_self_intersections": 2}
    rows = [
        _attempt(joint=200, tolerance=1e-3, validity_components=baseline_components),
        _attempt(joint=0, tolerance=1e-3, validity_components=baseline_components),
        _attempt(joint=200, tolerance=1e-4, validity_components=changed_components),
        _attempt(joint=200, tolerance=1e-2, validity_components=baseline_components),
        _attempt(joint=0, tolerance=1e-4, validity_components=baseline_components),
        _attempt(joint=0, tolerance=1e-2, validity_components=baseline_components),
    ]

    result = classify_case(rows)

    assert result["primary_cause"] == "wire_self_intersection"
    assert result["joint_sensitive"] is False
    assert result["tolerance_sensitive"] is True


def test_strict_checker_disagreement_does_not_count_as_attribution():
    rows = [
        _attempt(joint=200, tolerance=1e-3),
        _attempt(joint=0, tolerance=1e-3),
        _attempt(joint=200, tolerance=1e-4),
        _attempt(joint=200, tolerance=1e-2),
        _attempt(joint=0, tolerance=1e-4),
        _attempt(joint=0, tolerance=1e-2),
    ]

    result = classify_case(rows)

    assert result["primary_cause"] == "strict_checker_disagreement"
    assert result["attributed"] is False


def test_strict_disagreement_with_joint_sensitivity_is_attributed():
    rows = [
        _attempt(joint=200, tolerance=1e-3),
        _attempt(
            joint=0,
            tolerance=1e-3,
            status="both_valid",
            native_brep_valid=True,
            strict_brep_valid=True,
            both_valid=True,
        ),
        _attempt(joint=200, tolerance=1e-4),
        _attempt(joint=200, tolerance=1e-2),
        _attempt(joint=0, tolerance=1e-4),
        _attempt(joint=0, tolerance=1e-2),
    ]

    result = classify_case(rows)

    assert result["primary_cause"] == "joint_optimize_sensitivity"
    assert result["attributed"] is True


def test_summary_gate_requires_complete_matrix_and_80_percent_attribution():
    passing_cases = [
        {"cad_id": f"cad-{index}", "attempts": 6, "attributed": index < 13, "primary_cause": "curve_fit" if index < 13 else "unknown"}
        for index in range(16)
    ]
    attempts = [
        _attempt(cad_id=f"cad-{case_index}", joint=variant_index, tolerance=1e-3)
        for case_index in range(16)
        for variant_index in range(6)
    ]

    summary = summarize_cases(passing_cases, attempts)

    assert summary["matrix_complete"] is True
    assert summary["attributed_cases"] == 13
    assert summary["attribution_rate"] == pytest.approx(13 / 16)
    assert summary["gate_passed"] is True
    incomplete = summarize_cases(passing_cases[:-1], attempts[:-6])
    assert incomplete["gate_passed"] is False


def test_repair_checklist_reconciles_case_counts(tmp_path):
    output = tmp_path / "repair.md"
    write_repair_checklist(
        output,
        [
            {"primary_cause": "pre_step:curve_fit"},
            {"primary_cause": "pre_step:curve_fit"},
            {"primary_cause": "wire_self_intersection"},
        ],
    )
    text = output.read_text(encoding="utf-8")
    assert "`pre_step:curve_fit` (2 case(s))" in text
    assert "`wire_self_intersection` (1 case(s))" in text
