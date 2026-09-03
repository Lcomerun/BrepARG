import copy
import json
import math

import numpy as np
import pytest

from tools.probe_downstream_bad_wire_lineage import (
    ALL_PHASES,
    MEMORY_PHASES,
    PROFILE,
    SCHEMA,
    STEP_PHASE,
    TARGET_CAD_IDS,
    WORKER_MARKER,
    _cyclic_curve_sample_distance,
    _edge_fingerprint_metrics,
    _edge_fingerprints_compatible,
    _matching_count_capped,
    _match_step_geometry_incidence,
    _validate_global_edge_incidence,
    _compact_metadata,
    assert_path_free_evidence,
    assess_observations,
    output_writer_lock,
    parse_lineage_worker_result,
    select_lineage_sources,
    summarize,
    validate_case_row,
)


RUN_SIGNATURE = "a" * 64
SOURCE_BINDING = {"bytes": 123, "sha256": "b" * 64}


class _Shape:
    def __init__(self, identity):
        self.identity = identity

    def IsSame(self, other):
        return isinstance(other, _Shape) and self.identity == other.identity


def _fingerprint(points, *, curve_type="bspline", closed=False, seam=False):
    values = np.asarray(points, dtype=np.float64)
    segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return {
        "curve_type": curve_type,
        "length": float(np.sum(segments)),
        "bbox": np.concatenate((np.min(values, axis=0), np.max(values, axis=0))),
        "endpoints": np.asarray((values[0], values[-1])),
        "samples": values,
        "closed": bool(closed),
        "degenerated": False,
        "seam": bool(seam),
    }


def _synthetic_face(source_face_index, fingerprints, identities, *, step=False):
    edges = []
    for edge_index, (fingerprint, identity) in enumerate(zip(fingerprints, identities)):
        row = {"fingerprint": fingerprint, "observed_edge": _Shape(identity)}
        if not step:
            row["source_edge_id"] = int(identity)
        edges.append(row)
    return {
        **({} if step else {"source_face_index": source_face_index}),
        "face": _Shape(f"face-{source_face_index}"),
        "surface_type": 6,
        "surface_u_periodic": False,
        "surface_v_periodic": False,
        "wire_pattern": [(True, len(edges))],
        "wires": [
            {
                "observed_wire": _Shape(f"wire-{source_face_index}"),
                "outer": True,
                "edges": edges,
            }
        ],
    }


def test_capped_matching_distinguishes_unique_zero_and_multiple_solutions():
    assert _matching_count_capped([[1], [0]], 2) == (1, [1, 0])
    assert _matching_count_capped([[0], [0]], 2) == (0, None)
    count, first = _matching_count_capped([[0, 1], [0, 1]], 2)
    assert count == 2
    assert first in ([0, 1], [1, 0])


def test_closed_curve_distance_accepts_cyclic_shift_and_reverse():
    square = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    core = np.roll(square[:-1][::-1], 2, axis=0)
    shifted_reversed = np.concatenate((core, core[:1]), axis=0)

    assert _cyclic_curve_sample_distance(square, shifted_reversed, closed=True) == 0


def test_fingerprint_curve_type_is_advisory_but_geometry_is_strict():
    points = np.stack(
        (np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1
    )
    first = _fingerprint(points, curve_type="bspline")
    wrapped = _fingerprint(points[::-1], curve_type="trimmed_curve")

    metrics = _edge_fingerprint_metrics(first, wrapped, scale=1.0)
    assert metrics["curve_type_equal"] is False
    assert _edge_fingerprints_compatible(first, wrapped, scale=1.0)


def test_nonfinite_fingerprint_fails_closed():
    points = np.stack(
        (np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1
    )
    clean = _fingerprint(points)
    invalid = copy.deepcopy(clean)
    invalid["samples"][3, 1] = math.nan

    metrics = _edge_fingerprint_metrics(invalid, clean, scale=1.0)
    assert metrics["available"] is False
    assert not _edge_fingerprints_compatible(invalid, clean, scale=1.0)


def test_geometry_incidence_matcher_accepts_only_unique_global_assignment():
    horizontal = _fingerprint(
        np.stack((np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1)
    )
    vertical = _fingerprint(
        np.stack((np.zeros(17), np.linspace(0, 2, 17), np.zeros(17)), axis=1)
    )
    sources = [
        _synthetic_face(10, [horizontal], [7]),
        _synthetic_face(20, [vertical], [9]),
    ]
    # STEP explorer order is deliberately reversed; it is only a local key.
    steps = [
        _synthetic_face(0, [vertical], [9], step=True),
        _synthetic_face(1, [horizontal], [7], step=True),
    ]

    result = _match_step_geometry_incidence(sources, steps, scale=2.0)

    assert result["status"] == "exact_geometry_incidence"
    assert result["face_matching_count_capped"] == 1
    assert [row["source_face_index"] for row in result["face_rows"]] == [10, 20]


def test_geometry_incidence_matcher_reports_zero_and_multiple_face_solutions():
    horizontal = _fingerprint(
        np.stack((np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1)
    )
    vertical = _fingerprint(
        np.stack((np.zeros(17), np.linspace(0, 1, 17), np.zeros(17)), axis=1)
    )
    source = [_synthetic_face(0, [horizontal], [1])]
    zero = _match_step_geometry_incidence(
        source, [_synthetic_face(0, [vertical], [2], step=True)], scale=1.0
    )
    assert zero["status"] == "unavailable"
    assert zero["failure_codes"] == ["face_assignment_has_no_perfect_matching"]

    sources = [
        _synthetic_face(0, [horizontal], [1]),
        _synthetic_face(1, [horizontal], [2]),
    ]
    steps = [
        _synthetic_face(0, [horizontal], [10], step=True),
        _synthetic_face(1, [horizontal], [11], step=True),
    ]
    multiple = _match_step_geometry_incidence(sources, steps, scale=1.0)
    assert multiple["status"] == "ambiguous"
    assert multiple["failure_codes"] == ["face_assignment_not_unique"]


def test_geometry_incidence_matcher_rejects_ambiguous_edges_inside_unique_face():
    identical = _fingerprint(
        np.stack((np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1)
    )
    source = [_synthetic_face(8, [identical, identical], [3, 4])]
    step = [_synthetic_face(0, [identical, identical], [30, 31], step=True)]

    result = _match_step_geometry_incidence(source, step, scale=1.0)

    # Face compatibility requires only existence, but the final occurrence
    # proof requires exactly one perfect matching and therefore fails closed.
    assert result["status"] == "ambiguous"
    assert result["failure_codes"] == [
        "source_face_8_edge_assignment_not_unique"
    ]


def test_global_source_edge_split_and_merge_are_rejected():
    split = _validate_global_edge_incidence(
        [
            {"source_edge_id": 4, "observed_edge": _Shape("step-a")},
            {"source_edge_id": 4, "observed_edge": _Shape("step-b")},
        ]
    )
    assert split == ["source_edge_4_split_after_step"]

    merge = _validate_global_edge_incidence(
        [
            {"source_edge_id": 4, "observed_edge": _Shape("step-a")},
            {"source_edge_id": 8, "observed_edge": _Shape("step-a")},
        ]
    )
    assert merge == ["source_edges_4_8_merged_after_step"]


def _frozen_cohort():
    calibration = []
    selector = []
    for index in range(100):
        cad_id = f"cad-{index:03d}"
        historical = index < 84
        selected = historical or 84 <= index < 91
        calibration.append(
            {
                "arm": "original",
                "cad_id": cad_id,
                "parent_id": f"parent-{index:03d}",
                "brep_valid": historical,
                "source_path": f"source-{index:03d}.pkl",
            }
        )
        selector.append(
            {
                "cad_id": cad_id,
                "parent_id": f"parent-{index:03d}",
                "historical_strict_valid": historical,
                "strict_brep_valid": selected,
                "selection": {
                    "primary_profile": PROFILE,
                    "selected_profile": PROFILE,
                    "candidates": [{"profile": PROFILE}],
                },
            }
        )
    return calibration, selector


def _diagnosis(*occurrences):
    return {
        "status": "diagnosed",
        "faces": [],
        "wires": [],
        "occurrences": list(occurrences),
        "occurrence_kinds": sorted(
            {str(row.get("kind")) for row in occurrences}
        ),
    }


def _observation(phase, face_index=None, *, occurrences=(), exact=True):
    row = {
        "phase": phase,
        "entity_kind": "step_shape" if phase == STEP_PHASE else "face",
        "lineage_status": "exact_geometry_incidence" if phase == STEP_PHASE else "exact_identity",
        "mapping_failures": [],
        "diagnosis": _diagnosis(*occurrences),
    }
    if face_index is not None:
        row["source_face_index"] = face_index
    if not exact:
        row["lineage_status"] = "unavailable"
        row["mapping_failures"] = ["not_proven"]
    return row


def _mapped_occurrence(face_index=0, *, edge_ids=(0, 1), kind="non_adjacent"):
    return {
        "status": "detected",
        "kind": kind,
        "edge_positions": [1, 2],
        "source_face_index": face_index,
        "wire_index": 0,
        "source_mapping_status": "mapped",
        "source_edge_ids": list(edge_ids),
    }


def _observations(*, source_face_count=2, step_exact=True):
    rows = [
        _observation(phase, face_index)
        for phase in MEMORY_PHASES
        for face_index in range(source_face_count)
    ]
    rows.append(_observation(STEP_PHASE, exact=step_exact))
    return rows


def _case(cad_id, parent_id, *, observations=None, status="completed"):
    rows = _observations() if observations is None else observations
    assessment = assess_observations(
        rows, source_face_count=2, source_edge_count=4
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
        "step_roundtrip_status": "diagnosed",
        "source_face_count": 2,
        "source_edge_count": 4,
        "observations": rows,
        **{
            key: assessment[key]
            for key in (
                "phase_counts",
                "all_stages_observed",
                "coverage_failure_count",
                "observation_failure_count",
                "mapping_failure_count",
                "mapped_defect_count",
                "first_bad_phase",
                "first_bad_occurrences",
            )
        },
    }


def test_select_lineage_sources_proves_ordered_two_of_nine_contract():
    calibration, selector = _frozen_cohort()
    targets = ("cad-098", "cad-092")

    selected = select_lineage_sources(calibration, selector, target_ids=targets)

    assert [row["cad_id"] for row in selected] == list(targets)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda calibration, selector, targets: calibration.pop(),
        lambda calibration, selector, targets: calibration.__setitem__(
            83, {**calibration[83], "brep_valid": False}
        ),
        lambda calibration, selector, targets: selector.__setitem__(
            99, dict(selector[0])
        ),
        lambda calibration, selector, targets: selector.__setitem__(
            92, {**selector[92], "parent_id": "drift"}
        ),
        lambda calibration, selector, targets: selector.__setitem__(
            99, {**selector[99], "strict_brep_valid": True}
        ),
        lambda calibration, selector, targets: targets.__setitem__(0, "cad-090"),
        lambda calibration, selector, targets: targets.__setitem__(1, targets[0]),
    ],
)
def test_select_lineage_sources_fails_closed_on_any_cohort_drift(mutation):
    calibration, selector = _frozen_cohort()
    targets = ["cad-091", "cad-092"]
    mutation(calibration, selector, targets)

    with pytest.raises((TypeError, ValueError)):
        select_lineage_sources(calibration, selector, target_ids=targets)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda selection: selection.__setitem__("primary_profile", "other"),
        lambda selection: selection.__setitem__("selected_profile", "other"),
        lambda selection: selection.__setitem__("candidates", []),
        lambda selection: selection.__setitem__(
            "candidates", [{"profile": PROFILE}, {"profile": PROFILE}]
        ),
    ],
)
def test_select_lineage_sources_binds_target_primary_candidate(mutation):
    calibration, selector = _frozen_cohort()
    mutation(selector[91]["selection"])

    with pytest.raises(ValueError):
        select_lineage_sources(
            calibration, selector, target_ids=("cad-091", "cad-092")
        )


def test_writer_lock_uses_only_lineage_protocol_name(tmp_path):
    with output_writer_lock(tmp_path):
        assert (tmp_path / ".downstream_bad_wire_lineage_writer.lock").is_file()

    assert not (tmp_path / ".periodic_pcurve_writer.lock").exists()


@pytest.mark.parametrize(
    "value",
    [
        {"source_path": "redacted"},
        {"nested": {"step_path": "redacted"}},
        {"message": "D:/machine/private/candidate.step"},
        {"message": "/machine/private/candidate.step"},
    ],
)
def test_path_free_evidence_rejects_path_keys_and_absolute_values(value):
    with pytest.raises(ValueError, match="path"):
        assert_path_free_evidence(value)


def test_worker_sentinel_must_be_unique_and_final():
    payload = '{"status":"completed"}'
    assert parse_lineage_worker_result("noise\n" + WORKER_MARKER + payload) == {
        "status": "completed"
    }
    assert parse_lineage_worker_result(WORKER_MARKER + payload + "\nnoise") is None
    assert (
        parse_lineage_worker_result(
            WORKER_MARKER + payload + "\n" + WORKER_MARKER + payload
        )
        is None
    )


def test_compact_metadata_drops_native_handles_and_is_json_serializable():
    class FakeOccHandle:
        pass

    handle = FakeOccHandle()
    metadata = {
        "source_face_index": 3,
        "source_loop_edge_uses": [[{"source_edge_id": 7}]],
        "sewing_lineage": {
            "status": "mapped",
            "failure_codes": [],
            "shape": handle,
            "_private_shape": handle,
        },
        "source_mapping": {
            "status": "exact_sewing_history",
            "failures": [],
            "wire_rows": [
                {
                    "observed_wire": handle,
                    "source_edge_candidates": [
                        {"source_edge_id": 7, "observed_edge": handle}
                    ],
                }
            ],
        },
    }

    compact = _compact_metadata(metadata)

    encoded = json.dumps(compact, allow_nan=False, sort_keys=True)
    assert "FakeOccHandle" not in encoded
    assert "observed_wire" not in encoded
    assert "observed_edge" not in encoded
    assert "shape" not in encoded
    assert "source_mapping" not in compact
    assert compact["source_mapping_summary"] == {
        "status": "exact_sewing_history",
        "failure_codes": [],
        "wire_count": 1,
        "edge_occurrence_count": 1,
    }


def test_complete_stage_population_is_accepted_and_derived():
    assessment = assess_observations(
        _observations(), source_face_count=2, source_edge_count=4
    )

    assert assessment["all_stages_observed"] is True
    assert assessment["phase_counts"] == {
        **{phase: 2 for phase in MEMORY_PHASES},
        STEP_PHASE: 1,
    }
    assert assessment["mapping_failure_count"] == 0


@pytest.mark.parametrize(
    "mutation, failure_key",
    [
        (lambda rows: rows.pop(0), "coverage_failure_count"),
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "coverage_failure_count"),
        (
            lambda rows: rows.__setitem__(
                -1, _observation(STEP_PHASE, exact=False)
            ),
            "mapping_failure_count",
        ),
    ],
)
def test_missing_duplicate_or_unmapped_stage_is_explicit(mutation, failure_key):
    rows = _observations()
    mutation(rows)

    assessment = assess_observations(
        rows, source_face_count=2, source_edge_count=4
    )

    assert assessment[failure_key] > 0


def test_nonfinite_metadata_is_rejected_before_it_can_be_signed():
    rows = _observations()
    rows[0]["face_3d_endpoint_max_gap"] = math.nan

    with pytest.raises(ValueError, match="non-finite"):
        assess_observations(rows, source_face_count=2, source_edge_count=4)


@pytest.mark.parametrize(
    "change",
    [
        {"source_mapping_status": "ambiguous"},
        {"source_mapping_status": "unavailable", "source_edge_ids": []},
        {"source_edge_ids": [99]},
        {"source_edge_ids": [True]},
    ],
)
def test_bad_occurrence_requires_exact_source_edge_mapping(change):
    occurrence = {**_mapped_occurrence(), **change}
    rows = _observations()
    rows[0] = _observation(MEMORY_PHASES[0], 0, occurrences=[occurrence])

    assessment = assess_observations(
        rows, source_face_count=2, source_edge_count=4
    )

    assert assessment["mapping_failure_count"] > 0


def test_first_bad_phase_and_exact_source_occurrence_are_derived():
    rows = _observations()
    rows[2] = _observation(
        MEMORY_PHASES[1], 0, occurrences=[_mapped_occurrence()]
    )

    assessment = assess_observations(
        rows, source_face_count=2, source_edge_count=4
    )

    assert assessment["first_bad_phase"] == MEMORY_PHASES[1]
    assert assessment["first_bad_occurrences"][0]["source_edge_ids"] == [0, 1]


def test_validate_case_rejects_claimed_completion_with_unmapped_step():
    rows = _observations(step_exact=False)
    row = _case("cad", "parent", observations=rows, status="completed")

    with pytest.raises(ValueError, match="completed worker"):
        validate_case_row(
            row,
            source={"cad_id": "cad", "parent_id": "parent"},
            run_signature=RUN_SIGNATURE,
            expected_binding=SOURCE_BINDING,
        )


def test_validate_case_rejects_derived_field_tampering():
    row = _case("cad", "parent")
    row["mapping_failure_count"] = 1

    with pytest.raises(ValueError, match="derived mapping_failure_count"):
        validate_case_row(
            row,
            source={"cad_id": "cad", "parent_id": "parent"},
            run_signature=RUN_SIGNATURE,
            expected_binding=SOURCE_BINDING,
        )


def test_summary_promotes_only_two_fully_conclusive_rows(monkeypatch):
    monkeypatch.setattr(
        "tools.probe_downstream_bad_wire_lineage.TARGET_CAD_IDS",
        ("cad-a", "cad-b"),
    )
    rows = _observations()
    rows[2] = _observation(
        MEMORY_PHASES[1], 0, occurrences=[_mapped_occurrence()]
    )
    first = _case("cad-a", "parent-a", observations=rows)
    second = _case("cad-b", "parent-b")

    result = summarize([first, second])

    assert result["conclusive"] is True
    assert result["decision"] == "PROMOTE_TARGETED_NONPERIODIC_REPAIR_PROBE"


def test_summary_does_not_call_a_pre_repair_defect_downstream(monkeypatch):
    monkeypatch.setattr(
        "tools.probe_downstream_bad_wire_lineage.TARGET_CAD_IDS",
        ("cad-a", "cad-b"),
    )
    rows = _observations()
    rows[0] = _observation(
        MEMORY_PHASES[0], 0, occurrences=[_mapped_occurrence()]
    )

    result = summarize(
        [_case("cad-a", "parent-a", observations=rows), _case("cad-b", "parent-b")]
    )

    assert result["conclusive"] is True
    assert result["decision"] == "CLOSE_DOWNSTREAM_BAD_WIRE_ROUTE"


def test_summary_never_promotes_when_step_mapping_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "tools.probe_downstream_bad_wire_lineage.TARGET_CAD_IDS",
        ("cad-a", "cad-b"),
    )
    first = _case(
        "cad-a",
        "parent-a",
        observations=_observations(step_exact=False),
        status="measurement_incomplete",
    )
    second = _case("cad-b", "parent-b")

    result = summarize([first, second])

    assert result["conclusive"] is False
    assert result["decision"] == "INCONCLUSIVE_REQUIRES_RERUN"
    assert result["mapping_failures"] > 0


def test_registered_phase_order_is_stable():
    assert ALL_PHASES == (*MEMORY_PHASES, STEP_PHASE)
