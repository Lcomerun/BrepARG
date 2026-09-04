import copy
import json
import math

import pytest

from tools.assembly_stage_lineage import (
    ASSESSMENT_SCHEMA,
    EDGE_ENDPOINT_IDENTITY_PROOF_METHOD,
    GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
    STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
    STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
    STEP_VERTEX_IDENTITY_PROOF_METHOD,
    STAGE_NAMES,
    STAGE_ORDER,
    assert_path_free_finite,
    assess_stage_lineage,
    infer_first_bad_stage,
    make_not_reached_stage,
    normalize_stage_record,
    redact_path_and_native_text,
    validate_stage_sequence,
    validate_topology_census,
)


def _source_topology():
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


def _exact_lineage():
    return {
        "status": "exact_geometry_incidence",
        "proof_method": "source_bound_test_fixture",
        "solution_count": 1,
        "source_face_ids": [0, 1],
        "source_edge_ids": [0, 1, 2, 3],
        "source_edge_occurrence_keys": [
            [face_id, edge_id, 0]
            for face_id in range(2)
            for edge_id in range(4)
        ],
        "failure_codes": [],
        "entities": {
            "faces": {
                "source_count": 2,
                "observed_count": 2,
                "mapped_source_count": 2,
                "mapped_observed_count": 2,
                "max_observed_per_source": 1,
                "max_source_per_observed": 1,
                "solution_count": 1,
            },
            "edges": {
                "source_count": 4,
                "observed_count": 4,
                "mapped_source_count": 4,
                "mapped_observed_count": 4,
                "max_observed_per_source": 1,
                "max_source_per_observed": 1,
                "solution_count": 1,
            },
        },
    }


def _global_vertex_proof(vertex_count, constraint_count):
    return {
        "status": "exact_identity",
        "proof_method": GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": vertex_count,
        "observed_vertex_count": vertex_count,
        "mapped_source_vertex_count": vertex_count,
        "mapped_observed_vertex_count": vertex_count,
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": constraint_count,
        "failure_codes": [],
    }


def _step_geometry_incidence_proof(source):
    self_loop_count = sum(
        endpoints[0] == endpoints[1]
        for endpoints in source["edge_vertex_source_ids"]
    )
    return {
        "status": "exact_geometry_incidence",
        "failure_codes": [],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_candidate_degree_counts": {"1": source["face_count"]},
        "face_matching_count_capped": 1,
        "vertex_proof_required": True,
        "vertex_proof_method": STEP_VERTEX_IDENTITY_PROOF_METHOD,
        "vertex_tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "vertex_candidate_degree_counts": {"1": source["vertex_count"]},
        "vertex_matching_count_capped": 1,
        "source_vertex_count": source["vertex_count"],
        "step_vertex_count": source["vertex_count"],
        "mapped_source_edge_count": source["edge_count"],
        "edge_endpoint_pair_expected_count": source["edge_count"],
        "edge_endpoint_pair_proof_count": source["edge_count"],
        "edge_endpoint_occurrence_expected_count": 2 * source["edge_count"],
        "edge_endpoint_occurrence_proof_count": 2 * source["edge_count"],
        "self_loop_endpoint_pair_expected_count": self_loop_count,
        "self_loop_endpoint_pair_proof_count": self_loop_count,
        "mapped_face_count": source["face_count"],
        "mapped_edge_occurrence_count": source["face_edge_occurrence_count"],
        "vertex_proof_status": "exact",
    }


def _stage_local_proof(stage, scope_count, edge_count, constraint_count):
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


def _stage(stage, **changes):
    source = _source_topology()
    if stage == "S1":
        topology = {"face_count": source["face_count"]}
    elif stage == "S2":
        topology = {
            "edge_count": source["edge_count"],
            "vertex_count": source["vertex_count"],
            "vertex_edge_incidence_counts": source["vertex_edge_incidence_counts"],
        }
    else:
        topology = source
    value = {
        "stage": stage,
        "phase": STAGE_NAMES[stage],
        "status": "observed",
        "lineage": _exact_lineage(),
        "topology": topology,
        "defects": [],
    }
    if stage == "S2":
        value["evidence"] = {
            "stage_local_occ_topology_proof": _stage_local_proof(
                stage, source["edge_count"], source["edge_count"], source["edge_count"]
            )
        }
    if stage in {"S3", "S4"}:
        value["evidence"] = {
            "stage_local_occ_topology_proof": _stage_local_proof(
                stage,
                source["face_count"],
                source["edge_count"],
                source["face_edge_occurrence_count"],
            ),
        }
    if stage in {"S5", "S6"}:
        value["evidence"] = {
            "source_vertex_lineage": _global_vertex_proof(
                source["vertex_count"], source["face_edge_occurrence_count"]
            )
        }
    if stage == "S6":
        value["construction_native_valid"] = True
    if stage == "S7":
        value.update(
            reimport_native_valid=True,
            strict_valid=True,
            evidence={
                "step_geometry_incidence_proof": _step_geometry_incidence_proof(source)
            },
        )
    value.update(changes)
    return value


def _valid_records():
    return [_stage(stage) for stage in STAGE_ORDER]


def test_registered_stage_order_and_names_are_stable():
    assert STAGE_ORDER == ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
    assert tuple(STAGE_NAMES) == STAGE_ORDER
    assert STAGE_NAMES["S6"] == "post_solid_pre_step"
    assert STAGE_NAMES["S7"] == "post_step_roundtrip_strict"


def test_valid_chain_is_conclusive_and_keeps_s6_s7_validity_distinct():
    result = assess_stage_lineage(
        _valid_records(), source_topology=_source_topology()
    )

    assert result["schema"] == ASSESSMENT_SCHEMA
    assert result["conclusive"] is True
    assert result["valid_chain"] is True
    assert result["first_bad_stage"] is None
    assert result["first_bad_inference"]["status"] == "no_bad_stage"
    assert result["protocol_failure_count"] == 0
    assert result["inconclusive_reason_count"] == 0
    assert result["stages"][5]["construction_native_valid"] is True
    assert "reimport_native_valid" not in result["stages"][5]
    assert result["stages"][6]["reimport_native_valid"] is True
    assert result["stages"][6]["strict_valid"] is True
    assert "construction_native_valid" not in result["stages"][6]
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_s6_and_s7_reject_ambiguous_native_valid_field():
    with pytest.raises(ValueError, match="native_valid is ambiguous"):
        normalize_stage_record(
            {**_stage("S6"), "native_valid": True},
            source_topology=_source_topology(),
        )

    with pytest.raises(ValueError, match="requires separate"):
        normalize_stage_record(
            {
                key: value
                for key, value in _stage("S7").items()
                if key != "strict_valid"
            },
            source_topology=_source_topology(),
        )


def test_stage_sequence_reports_missing_duplicate_and_out_of_order():
    missing = validate_stage_sequence(_valid_records()[:-1])
    assert missing["all_stages_accounted"] is False
    assert missing["protocol_failures"] == ["missing_stage:S7"]

    duplicate_records = _valid_records()
    duplicate_records[-1] = copy.deepcopy(duplicate_records[-2])
    duplicate = validate_stage_sequence(duplicate_records)
    assert duplicate["duplicate_stages"] == ["S6"]
    assert duplicate["missing_stages"] == ["S7"]

    reversed_records = list(reversed(_valid_records()))
    out_of_order = assess_stage_lineage(
        reversed_records, source_topology=_source_topology()
    )
    assert out_of_order["protocol_failures"] == ["stage_order_mismatch"]
    assert out_of_order["conclusive"] is False


def test_not_reached_placeholders_preserve_denominator_after_known_failure():
    records = _valid_records()
    records[2] = _stage(
        "S3",
        failure={"kind": "wire_build", "reason": "wire constructor rejected input"},
        topology=None,
    )
    records[3:] = [
        make_not_reached_stage(stage, "blocked by S3", blocked_by_stage="S3")
        for stage in STAGE_ORDER[3:]
    ]

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["coverage"]["all_stages_accounted"] is True
    assert len(result["stages"]) == 7
    assert result["first_bad_stage"] == "S3"
    assert result["first_bad_reasons"] == ["failure:wire_build"]
    assert result["conclusive"] is True
    assert result["protocol_failure_count"] == 0
    assert result["inconclusive_reason_count"] == 0


def test_not_reached_without_a_proven_bad_stage_is_inconclusive():
    records = _valid_records()
    records[3:] = [
        make_not_reached_stage(stage, "worker had no target", blocked_by_stage="S3")
        for stage in STAGE_ORDER[3:]
    ]

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["conclusive"] is False
    assert result["first_bad_inference"]["status"] == "inconclusive"
    assert "S4:not_reached_without_prior_bad_stage" in result["inconclusive_reasons"]


@pytest.mark.parametrize("status", ["ambiguous", "missing", "nonunique", "split", "merge"])
def test_explicit_nonexact_lineage_is_scientific_inconclusive(status):
    records = _valid_records()
    records[2]["lineage"] = {"status": status, "failure_codes": []}
    records[3]["defects"] = ["wire_self_intersection"]

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["protocol_failure_count"] == 0
    assert result["conclusive"] is False
    assert result["first_bad_stage"] is None
    assert result["first_bad_inference"]["blocked_at_stage"] == "S3"
    assert any(status in reason for reason in result["inconclusive_reasons"])


@pytest.mark.parametrize(
    "entity_change, expected_status",
    [
        ({"max_observed_per_source": 2}, "split"),
        ({"max_source_per_observed": 2}, "merge"),
        ({"solution_count": 2}, "nonunique"),
        ({"mapped_source_count": 3}, "missing"),
    ],
)
def test_cardinality_evidence_overrides_false_exact_claim(entity_change, expected_status):
    record = _stage("S3")
    record["lineage"]["entities"]["edges"].update(entity_change)

    normalized = normalize_stage_record(
        record, source_topology=_source_topology()
    )

    assert normalized["lineage"]["exact"] is False
    assert normalized["lineage"]["classification"] == "inconclusive"
    assert normalized["lineage"]["status"] == expected_status
    assert f"lineage_{expected_status}" in normalized["lineage"][
        "inconclusive_reasons"
    ]


def test_topology_drift_is_a_scientific_first_bad_observation_not_protocol_failure():
    records = _valid_records()
    drifted = _source_topology()
    drifted.update(
        vertex_count=5,
        vertex_edge_incidence_counts=[1, 1, 2, 2, 2],
        edge_vertex_source_ids=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    records[3]["topology"] = drifted

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["protocol_failure_count"] == 0
    assert result["inconclusive_reason_count"] == 0
    assert result["conclusive"] is True
    assert result["first_bad_stage"] == "S4"
    assert set(result["first_bad_reasons"]) == {
        "topology_drift:vertex_count",
        "topology_drift:vertex_edge_incidence_counts",
        "topology_drift:edge_vertex_source_ids",
    }


def test_downstream_ambiguity_does_not_erase_exact_earlier_native_failure():
    records = _valid_records()
    records[5]["construction_native_valid"] = False
    records[6]["lineage"] = {
        "status": "missing",
        "failure_codes": ["face_assignment_has_no_perfect_matching"],
    }
    records[6]["topology"] = {
        "face_count": 2,
        "edge_count": 5,
        "vertex_count": 4,
        "face_edge_occurrence_count": 10,
        "face_edge_incidence_counts": [5, 5],
        "edge_face_incidence_counts": [2, 2, 2, 2, 2],
        "vertex_edge_incidence_counts": [2, 2, 3, 3],
    }
    records[6]["evidence"] = {
        "step_geometry_incidence_proof": {
            **_step_geometry_incidence_proof(_source_topology()),
            "status": "unavailable",
            "failure_codes": ["face_assignment_has_no_perfect_matching"],
            "face_candidate_degree_counts": {"0": 1, "1": 1},
            "face_matching_count_capped": 0,
            "vertex_proof_status": "not_evaluated",
        }
    }

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["first_bad_stage"] == "S6"
    assert result["first_bad_reasons"] == ["construction_native_invalid"]
    assert result["conclusive"] is True
    assert result["valid_chain"] is False
    assert result["inconclusive_reason_count"] == 0
    assert "S7" in result["observed_bad_stages"]
    assert any(
        item["stage"] == "S7" for item in result["topology_drift_observations"]
    )
    assert result["topology_drift_observations"] == [
        {
            "stage": "S7",
            "phase": STAGE_NAMES["S7"],
            "drifted_fields": [
                "edge_count",
                "face_edge_occurrence_count",
                "face_edge_incidence_counts",
                "edge_face_incidence_counts",
                "vertex_edge_incidence_counts",
            ],
        }
    ]


def test_first_bad_inference_uses_registered_order_not_defect_count():
    records = [
        normalize_stage_record(record, source_topology=_source_topology())
        for record in _valid_records()
    ]
    records[4]["scientifically_bad"] = True
    records[4]["bad_reasons"] = ["defect:first"]
    records[6]["scientifically_bad"] = True
    records[6]["bad_reasons"] = ["defect:second", "defect:third"]

    result = infer_first_bad_stage(records)

    assert result == {
        "status": "identified",
        "stage": "S5",
        "phase": STAGE_NAMES["S5"],
        "reasons": ["defect:first"],
        "blocked_at_stage": None,
    }


def test_malformed_topology_is_rejected_but_well_formed_drift_is_returned():
    malformed = _source_topology()
    malformed["face_edge_occurrence_count"] = 7
    with pytest.raises(ValueError, match="face incidence sum"):
        validate_topology_census(malformed, source_topology=_source_topology())

    drifted = _source_topology()
    drifted.update(
        face_edge_incidence_counts=[3, 5],
        face_edge_source_ids=[[0, 1, 2], [0, 1, 2, 3, 3]],
        edge_face_source_ids=[[0, 1], [0, 1], [0, 1], [1, 1]],
    )
    normalized = validate_topology_census(
        drifted, source_topology=_source_topology()
    )
    assert normalized["matches_source"] is False
    assert normalized["drifted_fields"] == [
        "face_edge_incidence_counts",
        "face_edge_source_ids",
        "edge_face_source_ids",
    ]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value[2].update(evidence={"metric": math.nan}), "non-finite"),
        (
            lambda value: value[2].update(evidence={"source_path": "redacted"}),
            "path",
        ),
        (
            lambda value: value[2].update(evidence={"note": "D:/private/cad.pkl"}),
            "absolute path",
        ),
    ],
)
def test_nonfinite_and_path_bearing_evidence_fail_before_archival(mutation, message):
    records = _valid_records()
    mutation(records)

    with pytest.raises(ValueError, match=message):
        assess_stage_lineage(records, source_topology=_source_topology())


def test_assert_path_free_finite_rejects_native_objects():
    class FakeOccHandle:
        pass

    with pytest.raises(TypeError, match="non-JSON"):
        assert_path_free_finite({"shape": FakeOccHandle()})


@pytest.mark.parametrize(
    "text",
    [
        "failed while reading D:/private/cad.pkl at stage",
        r"failed while reading \\server\share\cad.pkl",
        "failed while reading /mnt/private/cad.pkl",
        "pythonocc value=<TopoDS_Face; proxy of <Swig Object of type 'TopoDS_Face *' at 0x12345678>>",
        "native pointer 0x1234abcd leaked",
        "Handle_Geom_Surface",
    ],
)
def test_path_and_native_repr_are_rejected_anywhere_in_text(text):
    with pytest.raises(ValueError, match="absolute path|native handle"):
        assert_path_free_finite({"reason": text})
    redacted = redact_path_and_native_text(text)
    assert "redacted" in redacted
    assert_path_free_finite({"reason": redacted})


def test_bare_mapped_and_unproved_exact_claims_are_inconclusive():
    for lineage in (
        {"status": "mapped"},
        {"status": "exact_identity"},
        {
            "status": "exact_identity",
            "proof_method": "identity",
            "solution_count": 2,
            "source_face_ids": [0, 1],
            "source_edge_ids": [0, 1, 2, 3],
        },
    ):
        record = _stage("S1", lineage=lineage)
        normalized = normalize_stage_record(
            record, source_topology=_source_topology()
        )
        assert normalized["lineage"]["exact"] is False
        assert normalized["lineage"]["classification"] == "inconclusive"


def test_full_exact_claim_requires_stage_specific_source_populations():
    record = _stage("S3")
    del record["lineage"]["source_edge_occurrence_keys"]
    normalized = normalize_stage_record(record, source_topology=_source_topology())
    assert normalized["lineage"]["exact"] is False
    assert "lineage_source_occurrence_coverage_incomplete" in normalized[
        "lineage"
    ]["inconclusive_reasons"]

    record = _stage("S1")
    record["lineage"]["source_edge_ids"] = [0, 1, 3]
    normalized = normalize_stage_record(record, source_topology=_source_topology())
    assert normalized["lineage"]["exact"] is False
    assert "lineage_source_edge_coverage_incomplete" in normalized["lineage"][
        "inconclusive_reasons"
    ]


def test_canonical_topology_relations_must_be_mutual_and_obey_graph_sums():
    source = _source_topology()
    normalized = validate_topology_census(source, require_complete=True)
    assert normalized["complete"] is True

    broken_inverse = copy.deepcopy(source)
    broken_inverse["edge_face_source_ids"][0] = [0, 0]
    with pytest.raises(ValueError, match="not inverses"):
        validate_topology_census(broken_inverse, require_complete=True)

    broken_vertex = copy.deepcopy(source)
    broken_vertex["edge_vertex_source_ids"][0] = [0, 2]
    with pytest.raises(ValueError, match="vertex incidence"):
        validate_topology_census(broken_vertex, require_complete=True)

    broken_sum = copy.deepcopy(source)
    broken_sum["vertex_edge_incidence_counts"] = [1, 2, 2, 2]
    with pytest.raises(ValueError, match="two per edge"):
        validate_topology_census(broken_sum, require_complete=True)


def test_canonical_topology_counts_self_loop_twice():
    topology = {
        "face_count": 1,
        "edge_count": 1,
        "vertex_count": 1,
        "face_edge_occurrence_count": 1,
        "face_edge_incidence_counts": [1],
        "edge_face_incidence_counts": [1],
        "vertex_edge_incidence_counts": [2],
        "face_edge_source_ids": [[0]],
        "edge_face_source_ids": [[0]],
        "edge_vertex_source_ids": [[0, 0]],
    }
    assert validate_topology_census(topology, require_complete=True)["complete"]


def test_local_exact_failure_requires_historical_distributed_event_sequence():
    record = _stage(
        "S2",
        lineage={
            "status": "local_exact_failure",
            "proof_method": "borrowed_occ_identity",
            "solution_count": 1,
            "source_edge_ids": [0, 1],
            "distributed_scope": {
                "entity_kind": "source_edge",
                "expected_ids": [0, 1, 2, 3],
                "completed_ids": [0, 1],
                "terminal_failure_entity_id": 2,
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        {"entity_id": 0, "event": "post_boundary_ok"},
                        {"entity_id": 1, "event": "post_boundary_ok"},
                        {"entity_id": 2, "event": "terminal_failure"},
                    ]
                },
            },
        },
        topology={
            "edge_count": 2,
            "vertex_count": 3,
            "vertex_edge_incidence_counts": [1, 2, 1],
        },
        evidence={"stage_local_occ_topology_proof": _stage_local_proof("S2", 2, 2, 2)},
    )
    normalized = normalize_stage_record(record, source_topology=_source_topology())
    assert normalized["lineage"]["exact"] is False
    assert normalized["lineage"]["local_failure_exact"] is True
    assert normalized["scientifically_bad"] is True
    inference = infer_first_bad_stage([normalized])
    assert inference["status"] == "identified"
    assert inference["stage"] == "S2"

    broken = copy.deepcopy(record)
    broken["lineage"]["distributed_scope"]["event_sequence_proof"]["events"][1][
        "event"
    ] = "terminal_failure"
    normalized = normalize_stage_record(broken, source_topology=_source_topology())
    assert normalized["lineage"]["local_failure_exact"] is False


def test_exact_prefix_pass_is_not_bad_and_allows_next_boundary_failure():
    s1 = _stage(
        "S1",
        lineage={
            "status": "exact_prefix",
            "proof_method": "borrowed_occ_identity",
            "solution_count": 1,
            "distributed_scope": {
                "entity_kind": "source_edge",
                "expected_ids": [0, 1, 2, 3],
                "completed_ids": [0, 1, 2],
                "terminal_failure_entity_id": None,
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        {"entity_id": 0, "event": "pre_boundary_ok"},
                        {"entity_id": 1, "event": "pre_boundary_ok"},
                        {"entity_id": 2, "event": "pre_boundary_ok"},
                    ]
                },
            },
        },
    )
    s2 = _stage(
        "S2",
        lineage={
            "status": "local_exact_failure",
            "proof_method": "borrowed_occ_identity",
            "solution_count": 1,
            "source_edge_ids": [0, 1],
            "distributed_scope": {
                "entity_kind": "source_edge",
                "expected_ids": [0, 1, 2, 3],
                "completed_ids": [0, 1],
                "terminal_failure_entity_id": 2,
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        {"entity_id": 0, "event": "post_boundary_ok"},
                        {"entity_id": 1, "event": "post_boundary_ok"},
                        {"entity_id": 2, "event": "terminal_failure"},
                    ]
                },
            },
        },
        topology={
            "edge_count": 2,
            "vertex_count": 3,
            "vertex_edge_incidence_counts": [1, 2, 1],
        },
        evidence={"stage_local_occ_topology_proof": _stage_local_proof("S2", 2, 2, 2)},
    )
    normalized_s1 = normalize_stage_record(s1, source_topology=_source_topology())
    normalized_s2 = normalize_stage_record(s2, source_topology=_source_topology())

    assert normalized_s1["lineage"]["prefix_pass_exact"] is True
    assert normalized_s1["lineage"]["local_failure_exact"] is False
    assert normalized_s1["scientifically_bad"] is False
    inference = infer_first_bad_stage([normalized_s1, normalized_s2])
    assert inference["status"] == "identified"
    assert inference["stage"] == "S2"


@pytest.mark.parametrize(
    "stage, wrong_entity_kind, expected_ids, completed_event",
    [
        ("S1", "source_face", [0, 1], "pre_boundary_ok"),
        ("S2", "source_face", [0, 1], "post_boundary_ok"),
        ("S3", "source_edge", [0, 1, 2, 3], "post_boundary_ok"),
        ("S4", "source_edge", [0, 1, 2, 3], "post_boundary_ok"),
    ],
)
def test_distributed_scope_entity_kind_is_hard_bound_to_stage(
    stage, wrong_entity_kind, expected_ids, completed_event
):
    record = _stage(stage)
    record["lineage"]["distributed_scope"] = {
        "entity_kind": wrong_entity_kind,
        "expected_ids": expected_ids,
        "completed_ids": expected_ids,
        "terminal_failure_entity_id": None,
        "preceding_stage_prefix_verified": False,
        "event_sequence_proof": {
            "events": [
                {"entity_id": entity_id, "event": completed_event}
                for entity_id in expected_ids
            ]
        },
    }

    normalized = normalize_stage_record(record, source_topology=_source_topology())

    assert normalized["lineage"]["exact"] is False
    assert normalized["lineage"]["prefix_pass_exact"] is False
    assert normalized["lineage"]["local_failure_exact"] is False
    assert normalized["lineage"]["classification"] == "inconclusive"
    assert (
        "lineage_distributed_entity_kind_mismatch_stage"
        in normalized["lineage"]["inconclusive_reasons"]
    )
    assert normalized["scientifically_bad"] is False


@pytest.mark.parametrize(
    "stage, wrong_entity_kind, expected_ids, completed_event",
    [
        ("S1", "source_face", [0, 1], "pre_boundary_ok"),
        ("S2", "source_face", [0, 1], "post_boundary_ok"),
        ("S3", "source_edge", [0, 1, 2, 3], "post_boundary_ok"),
        ("S4", "source_edge", [0, 1, 2, 3], "post_boundary_ok"),
    ],
)
def test_wrong_distributed_entity_kind_blocks_census_inference_without_protocol_error(
    stage, wrong_entity_kind, expected_ids, completed_event
):
    records = _valid_records()
    record = records[STAGE_ORDER.index(stage)]
    record["lineage"]["distributed_scope"] = {
        "entity_kind": wrong_entity_kind,
        "expected_ids": expected_ids,
        "completed_ids": expected_ids,
        "terminal_failure_entity_id": None,
        "preceding_stage_prefix_verified": False,
        "event_sequence_proof": {
            "events": [
                {"entity_id": entity_id, "event": completed_event}
                for entity_id in expected_ids
            ]
        },
    }

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["protocol_failure_count"] == 0
    assert result["conclusive"] is False
    assert result["valid_chain"] is False
    assert result["first_bad_stage"] is None
    assert result["first_bad_inference"]["status"] == "inconclusive"
    assert result["first_bad_inference"]["blocked_at_stage"] == stage
    assert (
        f"{stage}:lineage_distributed_entity_kind_mismatch_stage"
        in result["inconclusive_reasons"]
    )


@pytest.mark.parametrize(
    "stage, wrong_entity_kind, expected_ids",
    [
        ("S1", "source_face", [0, 1]),
        ("S2", "source_face", [0, 1]),
        ("S3", "source_edge", [0, 1, 2, 3]),
        ("S4", "source_edge", [0, 1, 2, 3]),
    ],
)
@pytest.mark.parametrize("claim", ["exact_prefix", "local_exact_failure"])
def test_wrong_distributed_entity_kind_cannot_prove_prefix_or_terminal(
    stage, wrong_entity_kind, expected_ids, claim
):
    lineage = {
        "status": claim,
        "proof_method": "source_bound_test_fixture",
        "solution_count": 1,
        "failure_codes": [],
        "distributed_scope": {
            "entity_kind": wrong_entity_kind,
            "expected_ids": expected_ids,
            "completed_ids": [],
            "terminal_failure_entity_id": (
                0 if claim == "local_exact_failure" else None
            ),
            "preceding_stage_prefix_verified": True,
            "event_sequence_proof": {
                "events": (
                    [{"entity_id": 0, "event": "terminal_failure"}]
                    if claim == "local_exact_failure"
                    else []
                )
            },
        },
    }
    if stage == "S2":
        lineage["source_edge_ids"] = []
    elif stage in {"S3", "S4"}:
        lineage.update(
            source_face_ids=[],
            source_edge_ids=[],
            source_edge_occurrence_keys=[],
        )

    normalized = normalize_stage_record(
        _stage(stage, lineage=lineage), source_topology=_source_topology()
    )

    assert normalized["lineage"]["exact"] is False
    assert normalized["lineage"]["prefix_pass_exact"] is False
    assert normalized["lineage"]["local_failure_exact"] is False
    assert normalized["lineage"]["classification"] == "inconclusive"
    assert (
        "lineage_distributed_entity_kind_mismatch_stage"
        in normalized["lineage"]["inconclusive_reasons"]
    )


@pytest.mark.parametrize("stage", ["S5", "S6", "S7"])
def test_whole_shape_stages_do_not_require_distributed_entity_kind(stage):
    normalized = normalize_stage_record(
        _stage(stage), source_topology=_source_topology()
    )

    assert normalized["lineage"]["distributed_scope"] is None
    assert normalized["lineage"]["exact"] is True
    assert normalized["lineage"]["classification"] == "exact"


def test_face_prefix_can_localize_next_face_failure_without_global_topology():
    records = _valid_records()
    records[2] = _stage(
        "S3",
        lineage={
            "status": "exact_prefix",
            "proof_method": "source_identity_or_authoritative_history",
            "solution_count": 1,
            "source_face_ids": [0],
            "source_edge_ids": [0, 1, 2, 3],
            "source_edge_occurrence_keys": [
                [0, edge_id, 0] for edge_id in range(4)
            ],
            "failure_codes": [],
            "distributed_scope": {
                "entity_kind": "source_face",
                "expected_ids": [0, 1],
                "completed_ids": [0],
                "terminal_failure_entity_id": None,
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        {"entity_id": 0, "event": "post_boundary_ok"}
                    ]
                },
            },
        },
        topology=None,
        evidence={
            "stage_local_occ_topology_proof": _stage_local_proof("S3", 1, 4, 4),
        },
    )
    records[3] = _stage(
        "S4",
        lineage={
            "status": "local_exact_failure",
            "proof_method": "source_identity_or_authoritative_history",
            "solution_count": 1,
            "source_face_ids": [],
            "source_edge_ids": [],
            "source_edge_occurrence_keys": [],
            "failure_codes": [],
            "distributed_scope": {
                "entity_kind": "source_face",
                "expected_ids": [0, 1],
                "completed_ids": [],
                "terminal_failure_entity_id": 0,
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        {"entity_id": 0, "event": "terminal_failure"}
                    ]
                },
            },
        },
        topology=None,
        failure={"kind": "face_repair_failed", "reason": "face_repair_failed"},
        evidence={
            "stage_local_occ_topology_proof": _stage_local_proof("S4", 0, 0, 0),
        },
    )
    records[4:] = [
        make_not_reached_stage(stage, "blocked", blocked_by_stage="S4")
        for stage in STAGE_ORDER[4:]
    ]

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["protocol_failure_count"] == 0
    assert result["conclusive"] is True
    assert result["first_bad_stage"] == "S4"
    assert not any(
        reason.startswith("S3:topology_evidence_missing")
        for reason in result["inconclusive_reasons"]
    )


@pytest.mark.parametrize("stage", ["S3", "S4", "S5", "S6"])
def test_full_exact_claim_requires_native_endpoint_identity_evidence(stage):
    record = _stage(stage)
    record.pop("evidence")

    normalized = normalize_stage_record(record, source_topology=_source_topology())

    assert normalized["lineage"]["exact"] is False
    assert "lineage_endpoint_identity_evidence_missing" in normalized[
        "lineage"
    ]["inconclusive_reasons"]


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda proof: proof.update(solution_count=2),
            "lineage_global_source_vertex_solution_count_not_one",
        ),
        (
            lambda proof: proof.update(observed_vertex_count=5),
            "lineage_global_source_vertex_observed_count_mismatch",
        ),
        (
            lambda proof: proof.update(max_observed_per_source=2),
            "lineage_global_source_vertex_split",
        ),
        (
            lambda proof: proof.update(max_source_per_observed=2),
            "lineage_global_source_vertex_merge",
        ),
    ],
)
def test_s5_s6_global_vertex_assignment_is_a_hard_exactness_gate(mutation, reason):
    for stage in ("S5", "S6"):
        record = _stage(stage)
        mutation(record["evidence"]["source_vertex_lineage"])

        normalized = normalize_stage_record(
            record, source_topology=_source_topology()
        )

        assert normalized["lineage"]["exact"] is False
        assert reason in normalized["lineage"]["inconclusive_reasons"]


def test_global_vertex_evidence_rejects_native_handle_or_assignment_payloads():
    record = _stage("S5")
    record["evidence"]["source_vertex_lineage"]["source_to_observed_assignment"] = [
        3, 2, 1, 0
    ]

    normalized = normalize_stage_record(record, source_topology=_source_topology())

    assert normalized["lineage"]["exact"] is False
    assert (
        "lineage_global_source_vertex_evidence_contains_unknown_fields"
        in normalized["lineage"]["inconclusive_reasons"]
    )


@pytest.mark.parametrize("stage", ["S5", "S6"])
def test_sewn_stages_use_global_vertex_proof_without_fake_s2_identity(stage):
    record = _stage(stage)
    assert "endpoint_identity_proof_method" not in record["evidence"]

    normalized = normalize_stage_record(record, source_topology=_source_topology())

    assert normalized["lineage"]["exact"] is True


def test_normalized_lineage_is_idempotent_when_optional_populations_are_none():
    record = _stage(
        "S2",
        lineage={"status": "mapped", "failure_codes": []},
    )
    once = normalize_stage_record(record, source_topology=_source_topology())
    twice = normalize_stage_record(once, source_topology=_source_topology())
    assert twice == once


def test_phase_only_record_is_supported_but_stage_phase_drift_is_rejected():
    record = _stage("S1")
    del record["stage"]
    normalized = normalize_stage_record(
        record, expected_stage="S1", source_topology=_source_topology()
    )
    assert normalized["stage"] == "S1"

    bad = _stage("S1")
    bad["phase"] = STAGE_NAMES["S2"]
    with pytest.raises(ValueError, match="disagree"):
        normalize_stage_record(bad, source_topology=_source_topology())


def test_incomplete_observed_topology_is_inconclusive_not_silently_complete():
    records = _valid_records()
    records[4]["topology"] = {"face_count": 2, "edge_count": 4}

    result = assess_stage_lineage(records, source_topology=_source_topology())

    assert result["protocol_failure_count"] == 0
    assert result["conclusive"] is False
    assert "S5:topology_evidence_missing:vertex_count" in result[
        "inconclusive_reasons"
    ]


@pytest.mark.parametrize("stage, prerequisite", [("S5", "S4"), ("S6", "S5")])
def test_whole_shape_terminal_localizes_ordinary_construction_exception(
    stage, prerequisite
):
    record = _stage(
        stage,
        lineage={
            "status": "local_exact_failure",
            "proof_method": "whole_shape_boundary_exception_v1",
            "solution_count": 1,
            "whole_stage_terminal": {
                "scope_kind": "whole_shape_boundary_failure",
                "boundary_stage": stage,
                "prerequisite_stage": prerequisite,
                "prerequisite_exact": True,
                "construction_exception_observed": True,
            },
        },
        topology=None,
        evidence=None,
        failure={"kind": "construction_exception", "reason": "construction_exception"},
    )
    normalized = normalize_stage_record(record, source_topology=_source_topology())
    assert normalized["lineage"]["local_failure_exact"] is True
    assert normalized["lineage"]["distributed_scope"] is None
    assert normalized["scientifically_bad"] is True
    inference = infer_first_bad_stage([normalized])
    assert inference["stage"] == stage
    assert f"whole_stage_terminal_failure:{stage}" in inference["reasons"]
    assert all("None" not in reason for reason in inference["reasons"])

    broken = copy.deepcopy(record)
    broken["lineage"]["whole_stage_terminal"]["prerequisite_exact"] = False
    normalized = normalize_stage_record(broken, source_topology=_source_topology())
    assert normalized["lineage"]["local_failure_exact"] is False
    assert normalized["lineage"]["classification"] == "inconclusive"
