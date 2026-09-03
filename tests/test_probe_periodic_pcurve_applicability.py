import copy
import json
import math

import pytest

from tools.probe_periodic_pcurve_applicability import (
    PROFILE,
    SCHEMA,
    TARGET_CAD_IDS,
    WORKER_MARKER,
    build_run_payload,
    classify_wire_applicability,
    parse_worker_result,
    select_census_sources,
    summarize,
    validate_case_row,
)


RUN_SIGNATURE = "a" * 64
SOURCE_BINDING = {"bytes": 123, "sha256": "b" * 64}
PHASE = "post_add_pcurves_pre_repair"


def _frozen_cohort():
    """Return a 100-CAD, 84-baseline-valid, 91-selector-valid cohort."""
    calibration = []
    selector = []
    for index in range(100):
        cad_id = f"cad-{index:03d}"
        parent_id = f"parent-{index:03d}"
        historical = index < 84
        # Seven of the historical-invalid CADs are restored, leaving nine.
        selected_strict = historical or 84 <= index < 91
        calibration.append(
            {
                "arm": "original",
                "cad_id": cad_id,
                "parent_id": parent_id,
                "brep_valid": historical,
                "source_path": f"source-{index:03d}.pkl",
            }
        )
        selector.append(
            {
                "cad_id": cad_id,
                "parent_id": parent_id,
                "historical_strict_valid": historical,
                "strict_brep_valid": selected_strict,
            }
        )
    return calibration, selector


def _repairable_wire(*, movable=(True, True), before=6.0, after=0.0):
    return {
        "edge_count": 2,
        "movable": list(movable),
        "plan": {
            "solved": True,
            "reason": "optimized",
            "changed_edge_indices": [0],
            "offsets": [[-1, 0], [0, 0]],
            "before_max_gap": before,
            "after_max_gap": after,
        },
    }


def _nonrepairable_wire():
    return {
        "edge_count": 2,
        "movable": [True, True],
        "plan": {
            "solved": True,
            "reason": "optimized",
            "changed_edge_indices": [],
            "offsets": [[0, 0], [0, 0]],
            "before_max_gap": 0.0,
            "after_max_gap": 0.0,
        },
    }


def _periodic_state(*wires, periods=(6.0, None)):
    return {
        "available": True,
        "reason": "measured",
        "periods": list(periods),
        "wires": list(wires),
    }


def _face(face_index, *, applicable=False, reason="surface_not_periodic"):
    return {
        "face_index": face_index,
        "phase": PHASE,
        "applicable": applicable,
        "periodic_gap_candidate": applicable,
        "partial_only": False,
        "reason": reason,
        "diagnosis": {"bad_wire_indices": []},
        "is_u_periodic": False,
        "is_v_periodic": False,
    }


def _case_row(
    cad_id,
    *,
    parent_id="parent",
    status="completed",
    source_face_count=1,
    faces=None,
):
    observed = (
        [_face(index) for index in range(source_face_count)]
        if faces is None and status == "completed"
        else list(faces or [])
    )
    return {
        "schema": SCHEMA,
        "cad_id": cad_id,
        "parent_id": parent_id,
        "profile": PROFILE,
        "run_signature": RUN_SIGNATURE,
        "source_binding": dict(SOURCE_BINDING),
        "source_binding_loaded_bytes": dict(SOURCE_BINDING),
        "source_binding_after_load": dict(SOURCE_BINDING),
        "status": status,
        "assembly_status": "completed",
        "source_face_count": source_face_count,
        "face_count": len(observed),
        "all_faces_observed": (
            status == "completed" and len(observed) == source_face_count
        ),
        "faces": observed,
        "bad_face_indices": [],
        "periodic_bad_face_indices": [],
        "repairable_face_indices": [
            face["face_index"]
            for face in observed
            if face.get("applicable") is True
        ],
    }


def test_select_census_sources_proves_100_to_9_to_ordered_5_contract():
    calibration, selector = _frozen_cohort()
    targets = ("cad-095", "cad-091", "cad-098", "cad-092", "cad-097")

    selected = select_census_sources(
        calibration,
        selector,
        target_ids=targets,
    )

    assert [row["cad_id"] for row in selected] == list(targets)
    assert all(row["brep_valid"] is False for row in selected)
    selector_by_id = {row["cad_id"]: row for row in selector}
    assert all(
        selector_by_id[row["cad_id"]]["strict_brep_valid"] is False
        for row in selected
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda calibration, selector, targets: calibration.pop(),
        lambda calibration, selector, targets: calibration.__setitem__(
            83, {**calibration[83], "brep_valid": False}
        ),
        lambda calibration, selector, targets: selector.__setitem__(
            99, dict(selector[0])
        ),
        lambda calibration, selector, targets: selector.__setitem__(
            91, {**selector[91], "parent_id": "drifted-parent"}
        ),
        # Reducing the current residual set from nine to eight must not silently
        # turn the five hard-coded targets into an apparently valid cohort.
        lambda calibration, selector, targets: selector.__setitem__(
            99, {**selector[99], "strict_brep_valid": True}
        ),
        lambda calibration, selector, targets: targets.__setitem__(0, "cad-090"),
    ],
    ids=(
        "not-100-calibration",
        "not-84-historical-valid",
        "duplicate-selector-cad",
        "parent-map-drift",
        "not-nine-current-residuals",
        "target-is-not-current-residual",
    ),
)
def test_select_census_sources_fails_closed_on_cohort_drift(mutate):
    calibration, selector = _frozen_cohort()
    targets = ["cad-091", "cad-092", "cad-093", "cad-094", "cad-095"]
    mutate(calibration, selector, targets)

    with pytest.raises((TypeError, ValueError)):
        select_census_sources(calibration, selector, target_ids=targets)


def test_applicability_requires_every_diagnosed_bad_wire_to_be_repairable():
    result = classify_wire_applicability(
        periodic_state=_periodic_state(
            _repairable_wire(),
            _repairable_wire(before=12.0),
        ),
        bad_wire_indices=[0, 1],
    )

    assert result["periodic_gap_candidate"] is True
    assert result["applicable"] is True
    assert result["partial_only"] is False
    assert result["repairable_wire_indices"] == [0, 1]


def test_applicability_does_not_promote_a_partially_repairable_face():
    result = classify_wire_applicability(
        periodic_state=_periodic_state(
            _repairable_wire(),
            _nonrepairable_wire(),
        ),
        bad_wire_indices=[0, 1],
    )

    assert result["periodic_gap_candidate"] is True
    assert result["applicable"] is False
    assert result["partial_only"] is True
    assert result["repairable_wire_indices"] == [0]


def test_nonperiodic_surface_is_not_applicable():
    result = classify_wire_applicability(
        periodic_state=_periodic_state(
            _repairable_wire(),
            periods=(None, None),
        ),
        bad_wire_indices=[0],
    )

    assert result["periodic_gap_candidate"] is False
    assert result["applicable"] is False
    assert result["partial_only"] is False
    assert result["repairable_wire_indices"] == []
    assert result["reason"] == "surface_not_periodic"


def test_branch_shift_on_an_immovable_seam_edge_fails_closed():
    result = classify_wire_applicability(
        periodic_state=_periodic_state(
            _repairable_wire(movable=(False, True)),
        ),
        bad_wire_indices=[0],
    )

    assert result["periodic_gap_candidate"] is False
    assert result["applicable"] is False
    assert result["partial_only"] is False
    assert result["repairable_wire_indices"] == []


@pytest.mark.parametrize(
    ("periods", "before", "after"),
    [
        ((math.inf, None), 6.0, 0.0),
        ((6.0, None), math.inf, 0.0),
        ((6.0, None), 6.0, math.nan),
    ],
    ids=("infinite-period", "infinite-before-gap", "nan-after-gap"),
)
def test_nonfinite_period_or_gap_never_becomes_applicable(periods, before, after):
    result = classify_wire_applicability(
        periodic_state=_periodic_state(
            _repairable_wire(before=before, after=after),
            periods=periods,
        ),
        bad_wire_indices=[0],
    )

    assert result["periodic_gap_candidate"] is False
    assert result["applicable"] is False
    assert result["repairable_wire_indices"] == []
    # The classifier itself must not propagate non-standard JSON numbers into
    # the signed census even when its observation contained one.
    json.dumps(result, allow_nan=False)


def test_worker_parser_accepts_one_final_nonempty_sentinel():
    payload = {"schema": SCHEMA, "cad_id": "cad-001", "status": "completed"}
    stdout = (
        "ordinary OCC diagnostic\n"
        + WORKER_MARKER
        + json.dumps(payload)
        + "\n\n"
    )

    assert parse_worker_result(stdout) == payload


@pytest.mark.parametrize(
    "stdout",
    [
        "ordinary OCC diagnostic only\n",
        WORKER_MARKER + "{broken\n",
        WORKER_MARKER + "[]\n",
        WORKER_MARKER + '{"status":"completed"}\ntrailing diagnostic\n',
        (
            WORKER_MARKER
            + '{"status":"first"}\n'
            + WORKER_MARKER
            + '{"status":"second"}\n'
        ),
    ],
    ids=("missing", "bad-json", "not-object", "trailing-noise", "double-marker"),
)
def test_worker_parser_rejects_ambiguous_or_nonfinal_sentinel(stdout):
    assert parse_worker_result(stdout) is None


@pytest.mark.parametrize(
    "failure_status",
    ("worker_timeout", "worker_process_exit", "worker_protocol_error", "probe_error"),
)
def test_summary_is_inconclusive_when_any_case_has_an_error(failure_status):
    rows = [_case_row(cad_id) for cad_id in TARGET_CAD_IDS]
    rows[2] = _case_row(
        TARGET_CAD_IDS[2],
        status=failure_status,
        source_face_count=0,
        faces=[],
    )

    result = summarize(rows)

    assert result["conclusive"] is False
    assert result["decision"] == "INCONCLUSIVE_REQUIRES_RERUN"


def test_summary_is_inconclusive_for_embedded_face_measurement_error():
    rows = [_case_row(cad_id) for cad_id in TARGET_CAD_IDS]
    failed_face = _face(0, reason="occ_probe_error")
    failed_face["error_type"] = "Standard_Failure"
    rows[0] = _case_row(TARGET_CAD_IDS[0], faces=[failed_face])

    result = summarize(rows)

    assert result["conclusive"] is False
    assert result["decision"] == "INCONCLUSIVE_REQUIRES_RERUN"


def test_validate_case_row_accepts_exact_complete_face_coverage():
    source = {"cad_id": "cad-complete", "parent_id": "parent"}
    row = _case_row("cad-complete", source_face_count=3)

    validate_case_row(row, source=source, run_signature=RUN_SIGNATURE)


@pytest.mark.parametrize(
    "indices",
    ([0, 2], [0, 0, 2], [0, 1, 3]),
    ids=("missing-face", "duplicate-face", "out-of-range-face"),
)
def test_validate_case_row_rejects_incomplete_or_ambiguous_face_coverage(indices):
    source = {"cad_id": "cad-incomplete", "parent_id": "parent"}
    row = _case_row(
        "cad-incomplete",
        source_face_count=3,
        faces=[_face(index) for index in indices],
    )

    with pytest.raises(ValueError, match="face|coverage"):
        validate_case_row(row, source=source, run_signature=RUN_SIGNATURE)


def test_validate_case_row_rejects_parent_identity_drift():
    source = {"cad_id": "cad-parent", "parent_id": "expected-parent"}
    row = _case_row("cad-parent", parent_id="different-parent")

    with pytest.raises(ValueError, match="parent"):
        validate_case_row(row, source=source, run_signature=RUN_SIGNATURE)


def test_validate_case_row_rejects_derived_repairable_face_drift():
    source = {"cad_id": "cad-derived", "parent_id": "parent"}
    row = _case_row("cad-derived")
    row["repairable_face_indices"] = [0]

    with pytest.raises(ValueError, match="repairable|derived"):
        validate_case_row(row, source=source, run_signature=RUN_SIGNATURE)


def test_formal_run_payload_rejects_a_dirty_repository(monkeypatch, tmp_path):
    source_path = tmp_path / "source.pkl"
    source_path.write_bytes(b"bound-source")
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "utils.py").write_text("# bound runtime\n", encoding="utf-8")
    calibration = tmp_path / "calibration.jsonl"
    selector = tmp_path / "selector.jsonl"
    selector_run = tmp_path / "selector_run.json"
    calibration.write_text("{}\n", encoding="utf-8")
    selector.write_text("{}\n", encoding="utf-8")
    selector_run.write_text("{}\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "calibration_manifest": calibration,
            "selector_matrix": selector,
            "selector_run": selector_run,
            "breparg_root": runtime_root,
            "joint_iterations": 200,
            "worker_timeout_seconds": 600.0,
        },
    )()
    monkeypatch.setattr(
        "tools.probe_periodic_pcurve_applicability.git_identity",
        lambda _root: {"commit": "c" * 40, "dirty": True, "status_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        "tools.probe_periodic_pcurve_applicability.source_hashes",
        lambda _root: {"tools/probe_periodic_pcurve_applicability.py": "e" * 64},
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        build_run_payload(
            args,
            [{"cad_id": "cad", "source_path": str(source_path)}],
            [],
        )
