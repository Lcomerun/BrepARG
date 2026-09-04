import argparse
import json
import subprocess

import pytest

from tools.assembly_selector_geometry import GEOMETRY_GATE_SCHEMA
from tools.probe_periodic_pcurve_applicability import canonical_sha256, sha256_file
from tools.run_exact_cad_repair_feasibility import (
    CAD_47472,
    CAD_63055,
    LINEAGE_RUN_SCHEMA,
    LINEAGE_SCHEMA,
    RUN_NAME,
    SCHEMA,
    VARIANTS,
    WORKER_MARKER,
    _base_row,
    _whole_cad_step_defect_evidence,
    append_row,
    bind_run_manifest,
    failure_row,
    parse_worker_result,
    read_rows,
    run_47472_candidate_variant,
    run_63055_candidate_variant,
    run_isolated,
    run_worker,
    source_binding,
    summarize,
    validate_attempt_row,
    validate_lineage_evidence,
    validate_terminal_artifact_hashes,
)


SIGNATURE = "a" * 64


def _source(tmp_path, cad_id, parent_id):
    path = tmp_path / f"{cad_id}.pkl"
    path.write_bytes(b"not-needed-by-this-test")
    return {
        "cad_id": cad_id,
        "parent_id": parent_id,
        "source_path": str(path),
        "brep_valid": False,
    }


def _rejected_geometry():
    return {
        "schema": GEOMETRY_GATE_SCHEMA,
        "accepted": False,
        "checks": {"measurement_completed": False},
        "rejection_reasons": ["test_rejection"],
        "thresholds": {},
    }


def _completed_row(tmp_path, variant, *, status):
    source = _source(tmp_path, variant.cad_id, f"parent-{variant.cad_id}")
    binding = source_binding(source["source_path"])
    row = _base_row(
        source, variant, run_signature=SIGNATURE, expected_binding=binding
    )
    row.update(
        status=status,
        callback_completed=True,
        source_binding_before=binding,
        source_binding_loaded_bytes=binding,
        source_binding_after_load=binding,
        source_binding_after_attempt=binding,
    )
    if variant.is_candidate:
        row["candidate_application"] = {
            "attempted": True,
            "applied": False,
            "status": "rejected",
        }
        row["defect_gate"]["rejection_reasons"] = ["candidate_rejected"]
    else:
        row["candidate_application"] = {
            "attempted": False,
            "applied": False,
            "status": "control",
        }
        row["control_expectation"] = {"reproduced": status == "control_reproduced"}
    return source, binding, row


def test_registered_matrix_is_exactly_two_controls_and_two_candidates():
    assert [(variant.cad_id, variant.arm) for variant in VARIANTS] == [
        (CAD_47472, "control"),
        (CAD_47472, "candidate"),
        (CAD_63055, "control"),
        (CAD_63055, "candidate"),
    ]
    assert len({variant.task_id for variant in VARIANTS}) == 4


def test_worker_sentinel_must_be_unique_and_final():
    payload = '{"status":"candidate_rejected"}'
    assert parse_worker_result("OCC noise\n" + WORKER_MARKER + payload) == {
        "status": "candidate_rejected"
    }
    assert parse_worker_result(WORKER_MARKER + payload + "\nlate noise") is None
    assert parse_worker_result(
        WORKER_MARKER + payload + "\n" + WORKER_MARKER + payload
    ) is None
    assert parse_worker_result(WORKER_MARKER + "[1]") is None
    assert parse_worker_result(WORKER_MARKER + "{broken") is None


def test_failure_row_is_an_explicit_false_denominator(tmp_path):
    variant = VARIANTS[1]
    source = _source(tmp_path, variant.cad_id, "parent")
    binding = source_binding(source["source_path"])

    row = failure_row(
        source,
        variant,
        run_signature=SIGNATURE,
        expected_binding=binding,
        status="worker_timeout",
        error_type="TimeoutExpired",
    )

    assert row["denominator"] is True
    assert row["step_saved"] is False
    assert row["native_brep_valid"] is False
    assert row["strict_brep_valid"] is False
    assert row["both_valid"] is False
    validate_attempt_row(
        row,
        source=source,
        variant=variant,
        run_signature=SIGNATURE,
        expected_binding=binding,
    )


def test_missing_candidate_hook_fails_closed_without_unpickling_geometry(
    tmp_path, monkeypatch
):
    variant = VARIANTS[1]
    source = _source(tmp_path, variant.cad_id, "parent")
    # run_worker must deserialize before importing the callback; use a valid,
    # tiny pickle and deliberately never enter geometry/OCC work.
    import pickle

    source_path = tmp_path / "valid.pkl"
    source_path.write_bytes(pickle.dumps({"sentinel": True}))
    source["source_path"] = str(source_path)
    binding = source_binding(source_path)

    def unavailable(_reference):
        raise AttributeError("adapter not implemented")

    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.resolve_callback", unavailable
    )
    row = run_worker(
        source,
        variant,
        output_dir=tmp_path / "worker",
        breparg_root=tmp_path,
        joint_iterations=200,
        expected_binding=binding,
        run_signature=SIGNATURE,
    )

    assert row["status"] == "candidate_hook_missing"
    assert row["callback_completed"] is False
    assert row["candidate_application"]["applied"] is False
    assert row["both_valid"] is False
    assert row["source_binding_after_attempt"] == binding
    validate_attempt_row(
        row,
        source=source,
        variant=variant,
        run_signature=SIGNATURE,
        expected_binding=binding,
    )


def test_incomplete_candidate_callback_becomes_worker_error(tmp_path, monkeypatch):
    variant = VARIANTS[1]
    source = _source(tmp_path, variant.cad_id, "parent")
    import pickle

    source_path = tmp_path / "valid.pkl"
    source_path.write_bytes(pickle.dumps({"sentinel": True}))
    source["source_path"] = str(source_path)
    binding = source_binding(source_path)
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.resolve_callback",
        lambda _reference: lambda **_kwargs: {},
    )

    row = run_worker(
        source,
        variant,
        output_dir=tmp_path / "worker",
        breparg_root=tmp_path,
        joint_iterations=200,
        expected_binding=binding,
        run_signature=SIGNATURE,
    )

    assert row["status"] == "worker_error"
    assert row["error_type"] == "ValueError"
    assert row["both_valid"] is False


def test_candidate_accepted_cannot_overclaim_rejected_geometry(tmp_path):
    variant = VARIANTS[1]
    source = _source(tmp_path, variant.cad_id, "parent")
    binding = source_binding(source["source_path"])
    row = _base_row(
        source, variant, run_signature=SIGNATURE, expected_binding=binding
    )
    row.update(
        status="candidate_accepted",
        callback_completed=True,
        step_saved=True,
        step_readable=True,
        native_brep_valid=True,
        strict_brep_valid=True,
        both_valid=True,
        source_binding_before=binding,
        source_binding_loaded_bytes=binding,
        source_binding_after_load=binding,
        source_binding_after_attempt=binding,
        candidate_application={"attempted": True, "applied": True, "status": "applied"},
        defect_gate={
            "accepted": True,
            "target_defects_removed": True,
            "no_new_non_target_defects": True,
            "mapping_exact": True,
            "source_topology_preserved": True,
            "shared_edge_correspondence_preserved": True,
            "curves_3d_preserved": True,
            "source_binding_preserved": True,
            "nonfinite_count": 0,
            "rejection_reasons": [],
        },
        geometry_topology_gate=_rejected_geometry(),
    )

    with pytest.raises(ValueError, match="overclaims"):
        validate_attempt_row(
            row,
            source=source,
            variant=variant,
            run_signature=SIGNATURE,
            expected_binding=binding,
        )


def test_isolated_timeout_and_malformed_sentinel_remain_denominator_rows(
    tmp_path, monkeypatch
):
    variant = VARIANTS[1]
    source = _source(tmp_path, variant.cad_id, "parent")
    binding = source_binding(source["source_path"])
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        calibration_manifest=tmp_path / "calibration.jsonl",
        selector_matrix=tmp_path / "selector.jsonl",
        selector_run=tmp_path / "selector-run.json",
        lineage_cases=tmp_path / "lineage.jsonl",
        lineage_run=tmp_path / "lineage-run.json",
        breparg_root=tmp_path / "upstream",
        joint_iterations=200,
        worker_timeout_seconds=1.0,
    )

    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("worker", 1.0, output="partial")
        ),
    )
    timed_out = run_isolated(
        source,
        variant,
        args=args,
        run_signature=SIGNATURE,
        expected_binding=binding,
    )
    assert timed_out["status"] == "worker_timeout"
    assert timed_out["denominator"] is True

    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OCC noise only", stderr=""
        ),
    )
    malformed = run_isolated(
        source,
        variant,
        args=args,
        run_signature=SIGNATURE,
        expected_binding=binding,
    )
    assert malformed["status"] == "worker_protocol_error"
    assert malformed["denominator"] is True


def test_manifest_resume_is_exact_and_jsonl_recovers_only_torn_tail(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    payload = {"schema": "contract", "ordered_task_ids": ["a", "b"]}
    first = bind_run_manifest(root, payload)
    second = bind_run_manifest(root, payload)
    assert first == second
    assert json.loads((root / RUN_NAME).read_text())["payload"] == payload
    with pytest.raises(RuntimeError, match="different signed run"):
        bind_run_manifest(root, {**payload, "ordered_task_ids": ["b", "a"]})

    rows_path = tmp_path / "rows.jsonl"
    append_row(rows_path, {"task_id": "a"})
    with rows_path.open("ab") as handle:
        handle.write(b'{"task_id":"torn"')
    assert read_rows(rows_path, recover_truncated_tail=True) == [{"task_id": "a"}]
    assert rows_path.read_text().endswith("\n")


def test_terminal_resume_rejects_tampered_ledger_or_summary(tmp_path):
    import tools.run_exact_cad_repair_feasibility as module

    root = tmp_path / "terminal"
    root.mkdir()
    rows_path = root / module.ROWS_NAME
    summary_path = root / module.SUMMARY_NAME
    rows_path.write_text('{"row":1}\n', encoding="utf-8")
    summary_path.write_text('{"summary":1}\n', encoding="utf-8")
    payload = {"schema": "contract"}
    record = {
        "schema": module.RUN_SCHEMA,
        "signature": canonical_sha256(payload),
        "payload": payload,
        "status": "COMPLETED",
        "attempts": 4,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    (root / RUN_NAME).write_text(json.dumps(record), encoding="utf-8")

    rows_path.write_text('{"row":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger hash"):
        validate_terminal_artifact_hashes(
            record, rows_path=rows_path, summary_path=summary_path
        )
    rows_path.write_text('{"row":1}\n', encoding="utf-8")
    summary_path.write_text('{"summary":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary hash"):
        validate_terminal_artifact_hashes(
            record, rows_path=rows_path, summary_path=summary_path
        )


def test_lineage_binding_rejects_stage_or_source_drift(tmp_path):
    sources = [
        _source(tmp_path, CAD_47472, "parent-47472"),
        _source(tmp_path, CAD_63055, "parent-63055"),
    ]
    bindings = [source_binding(source["source_path"]) for source in sources]
    rows = [
        {
            "schema": LINEAGE_SCHEMA,
            "cad_id": CAD_47472,
            "status": "completed",
            "source_binding": bindings[0],
            "first_bad_phase": "post_add_pcurves_pre_repair",
            "first_bad_occurrences": [
                {"kind": "adjacent", "source_face_index": 10, "source_edge_ids": [20, 13]},
                {"kind": "adjacent", "source_face_index": 43, "source_edge_ids": [16, 24]},
            ],
        },
        {
            "schema": LINEAGE_SCHEMA,
            "cad_id": CAD_63055,
            "status": "completed",
            "source_binding": bindings[1],
            "first_bad_phase": "post_sewing_pre_step",
            "first_bad_occurrences": [
                {"kind": "closure", "source_face_index": 5, "source_edge_ids": [9, 23]},
                {"kind": "adjacent", "source_face_index": 5, "source_edge_ids": [23, 9]},
            ],
        },
    ]
    cases = tmp_path / "lineage.jsonl"
    for row in rows:
        append_row(cases, row)
    payload = {
        "ordered_cad_ids": [CAD_47472, CAD_63055],
        "source_bindings": [
            {"cad_id": source["cad_id"], **binding}
            for source, binding in zip(sources, bindings)
        ],
    }
    run = tmp_path / "lineage-run.json"
    run.write_text(
        json.dumps(
            {
                "schema": LINEAGE_RUN_SCHEMA,
                "status": "COMPLETED",
                "attempts": 2,
                "payload": payload,
                "signature": canonical_sha256(payload),
                "rows_sha256": sha256_file(cases),
            }
        ),
        encoding="utf-8",
    )

    bound = validate_lineage_evidence(
        rows, sources=sources, lineage_cases=cases, lineage_run=run
    )
    assert bound["run_signature"] == canonical_sha256(payload)

    drifted = [dict(row) for row in rows]
    drifted[1] = {**drifted[1], "first_bad_phase": "post_step_roundtrip"}
    with pytest.raises(ValueError, match="phase drifted"):
        validate_lineage_evidence(
            drifted, sources=sources, lineage_cases=cases, lineage_run=run
        )


def test_summary_is_inconclusive_when_hook_is_missing_and_never_authorizes_100cad(
    tmp_path,
):
    rows = []
    for variant in VARIANTS:
        source, binding, row = _completed_row(
            tmp_path, variant,
            status="candidate_rejected" if variant.is_candidate else "control_reproduced",
        )
        if variant == VARIANTS[1]:
            row = failure_row(
                source,
                variant,
                run_signature=SIGNATURE,
                expected_binding=binding,
                status="candidate_hook_missing",
                error_type="CandidateCallbackUnavailable",
            )
        rows.append(row)

    result = summarize(rows)

    assert result["attempts"] == 4
    assert result["denominator_rows"] == 4
    assert result["conclusive"] is False
    assert result["candidate_hooks_unavailable"] == 1
    assert result["authorizes_full_100cad"] is False
    assert result["authorizes_training_or_ar"] is False


def _parsed_geometry():
    return {
        "faceEdge_adj": [[0]],
        "edgeCorner_adj": [[0, 1]],
        "surf_ncs": [[[0.0, 0.0, 0.0]]],
        "edge_ncs": [[[0.0, 0.0, 0.0]]],
        "surf_bbox_wcs": [[0.0, 0.0, 0.0, 1.0]],
        "corner_unique": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }


def test_47472_adapter_targets_only_registered_faces_and_pairs(tmp_path, monkeypatch):
    variant = VARIANTS[1]
    calls = []

    def fake_face_helper(face, **kwargs):
        calls.append(kwargs)
        return object(), {
            "attempted": True,
            "accepted": True,
            "strict_face_gate": {"accepted": True},
            "post_repair_mapping_gate": {"accepted": True},
            "topology_incidence_gate": {"accepted": True},
            "curve_3d_preservation": {"accepted": True},
            "non_target_pcurve_gate": {"accepted": True},
        }

    def fake_construct(
        *_args,
        post_pcurve_face_mutator,
        assembly_stage_face_observer,
        **kwargs,
    ):
        untouched = object()
        same, diagnostic = post_pcurve_face_mutator(
            0, untouched, {"source_mapping": {}, "source_edge_occurrences": ()}
        )
        assert same is untouched
        assert diagnostic["reason"] == "face_not_targeted"
        for face_index in (10, 43):
            post_pcurve_face_mutator(
                face_index,
                object(),
                {"source_mapping": {"status": "exact"},
                 "source_edge_occurrences": ((1, object()),)},
            )
        assembly_stage_face_observer(
            0,
            object(),
            {"phase": "post_sewing_pre_step", "source_mapping": {}},
        )
        assert kwargs["directed_trim"] is True
        assert kwargs["local_intersection_topology"] is True
        return object(), {"adapter": "47472"}

    monkeypatch.setattr(
        "tools.targeted_nonperiodic_pcurve_repair.repair_face_targeted_nonperiodic_pcurves",
        fake_face_helper,
    )
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility._joint_optimized_inputs",
        lambda *_args: (object(), object(), [[0]], object()),
    )
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.construct_brep_directed",
        fake_construct,
    )
    step = tmp_path / "candidate.step"
    step.write_bytes(b"step")
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility._write_candidate_step",
        lambda *_args: step,
    )
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility._step_observation",
        lambda *_args, **_kwargs: {
            "lineage_status": "exact_geometry_incidence",
            "mapping_failures": [],
            "diagnosis": {
                "status": "diagnosed",
                "occurrences": [],
                "geometry_incidence_proof": {"status": "exact_geometry_incidence"},
            },
        },
    )

    result = run_47472_candidate_variant(
        parsed=_parsed_geometry(), joint_iterations=200, variant=variant,
        breparg_root=tmp_path, output_dir=tmp_path,
    )

    assert [call["source_face_index"] for call in calls] == [10, 43]
    assert [call["expected_source_edge_pairs"] for call in calls] == [
        ((13, 20),),
        ((16, 24),),
    ]
    assert result["candidate_application"]["applied"] is True
    assert result["defect_gate"]["accepted"] is True


def _final_step_observation(*occurrences, lineage="exact_geometry_incidence"):
    return {
        "lineage_status": lineage,
        "mapping_failures": [] if lineage == "exact_geometry_incidence" else ["ambiguous"],
        "diagnosis": {
            "status": "diagnosed",
            "occurrences": list(occurrences),
            "geometry_incidence_proof": {"status": lineage},
        },
    }


def _final_occurrence(face, pair, *, kind="adjacent"):
    return {
        "kind": kind,
        "status": "detected",
        "source_face_index": face,
        "source_edge_ids": list(pair),
    }


def test_47472_whole_cad_gate_rejects_step_only_face_one_defect():
    gate = _whole_cad_step_defect_evidence(
        _final_step_observation(_final_occurrence(1, (10, 12))),
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["target_defects_removed"] is True
    assert gate["no_new_non_target_defects"] is False
    assert gate["non_target_defects"][0]["source_face_index"] == 1
    assert gate["accepted"] is False


def test_47472_whole_cad_gate_rejects_target_residual():
    gate = _whole_cad_step_defect_evidence(
        _final_step_observation(_final_occurrence(10, (20, 13))),
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["target_defects_removed"] is False
    assert gate["no_new_non_target_defects"] is True
    assert gate["target_residuals"][0]["source_face_index"] == 10
    assert gate["accepted"] is False


def test_47472_whole_cad_gate_accepts_only_exact_clean_step():
    gate = _whole_cad_step_defect_evidence(
        _final_step_observation(),
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["mapping_exact"] is True
    assert gate["target_definition_complete"] is True
    assert gate["target_defects_removed"] is True
    assert gate["no_new_non_target_defects"] is True
    assert gate["accepted"] is True


def test_47472_whole_cad_gate_rejects_other_non_target_defect():
    gate = _whole_cad_step_defect_evidence(
        _final_step_observation(_final_occurrence(27, (30, 31), kind="closure")),
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["target_defects_removed"] is True
    assert gate["no_new_non_target_defects"] is False
    assert gate["accepted"] is False


@pytest.mark.parametrize("lineage", ["ambiguous", "unavailable"])
def test_47472_whole_cad_gate_rejects_nonexact_step_mapping(lineage):
    gate = _whole_cad_step_defect_evidence(
        _final_step_observation(lineage=lineage),
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["mapping_exact"] is False
    assert gate["target_defects_removed"] is False
    assert gate["no_new_non_target_defects"] is False
    assert gate["accepted"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "source_edge_20_split_after_step",
        "source_edges_9_23_merged_after_step",
    ],
)
def test_47472_whole_cad_gate_rejects_shared_edge_split_or_merge(failure):
    observation = _final_step_observation(lineage="unavailable")
    observation["mapping_failures"] = [failure]

    gate = _whole_cad_step_defect_evidence(
        observation,
        target_source_face_indices=(10, 43),
        target_source_edge_pairs=((13, 20), (16, 24)),
    )

    assert gate["mapping_exact"] is False
    assert gate["mapping_failures"] == [failure]
    assert gate["accepted"] is False


def test_63055_adapter_enriches_complete_bindings_and_freezes_tolerance(
    tmp_path, monkeypatch
):
    variant = VARIANTS[3]
    observed = {}

    def fake_post_helper(shape, **kwargs):
        observed.update(kwargs)
        return object(), {
            "attempted": True,
            "accepted": True,
            "graph_preservation_gate": {"accepted": True},
            "topology_incidence_gate": {"accepted": True},
            "source_edge_identity_gate": {"accepted": True},
            "curve_3d_preservation": {"accepted": True},
            "target_face_after": {"accepted": True},
        }

    def fake_construct(*_args, post_sewing_shape_mutator, **kwargs):
        bindings = [
            {"source_face_index": index, "face": object(), "source_mapping": {}}
            for index in range(2)
        ]
        post_sewing_shape_mutator(
            object(), bindings,
            {"expected_source_face_count": 2, "expected_source_edge_count": 3},
        )
        assert kwargs["sewing_tolerance"] == pytest.approx(1e-4)
        assert kwargs["directed_trim"] is True
        assert kwargs["local_intersection_topology"] is True
        return object(), {"adapter": "63055"}

    monkeypatch.setattr(
        "tools.post_sewing_graph_repair.attempt_post_sewing_face_pcurve_reprojection",
        fake_post_helper,
    )
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility._joint_optimized_inputs",
        lambda *_args: (object(), object(), [[0]], object()),
    )
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility.construct_brep_directed",
        fake_construct,
    )
    step = tmp_path / "candidate.step"
    step.write_bytes(b"step")
    monkeypatch.setattr(
        "tools.run_exact_cad_repair_feasibility._write_candidate_step",
        lambda *_args: step,
    )

    result = run_63055_candidate_variant(
        parsed=_parsed_geometry(), joint_iterations=200, variant=variant,
        breparg_root=tmp_path, output_dir=tmp_path,
    )

    assert observed["target_source_face_index"] == 5
    assert observed["target_source_edge_ids"] == (9, 23)
    assert observed["expected_source_edge_pairs"] == ((9, 23),)
    assert observed["projection_precision"] == pytest.approx(1e-4)
    assert all(row["expected_source_face_count"] == 2
               and row["expected_source_edge_count"] == 3
               for row in observed["source_face_bindings"])
    assert result["candidate_application"]["applied"] is True
    assert result["defect_gate"]["accepted"] is True
