from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import pickle
import sys
import types
from pathlib import Path

import pytest

from tools import probe_source_bound_stage_census as census
from tools.assembly_stage_lineage import (
    EDGE_ENDPOINT_IDENTITY_PROOF_METHOD,
    STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
    STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
    STEP_VERTEX_IDENTITY_PROOF_METHOD,
    STAGE_NAMES,
    STAGE_ORDER,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding(label: str = "source") -> dict[str, object]:
    return {"bytes": 123, "sha256": _digest(label)}


def _original(cad_id: str, index: int, *, valid: bool) -> dict[str, object]:
    return {
        "arm": "original",
        "cad_id": cad_id,
        "parent_id": f"parent-{index}",
        "source_path": f"relative-{index}.pkl",
        "brep_valid": bool(valid),
    }


def _cohort_inputs():
    other = [f"cad-{index:03d}" for index in range(91)]
    cad_ids = [*other, *sorted(census.SELECTOR_RESIDUAL_CAD_IDS)]
    originals = [
        _original(cad_id, index, valid=index < 84)
        for index, cad_id in enumerate(cad_ids)
    ]
    calibration = []
    for arm in ("original", "continuous_bypass_64d", "fsq_8192_4d"):
        for source in originals:
            row = dict(source)
            row["arm"] = arm
            calibration.append(row)
    selector = []
    residual = census.SELECTOR_RESIDUAL_CAD_IDS
    for source in originals:
        selector.append(
            {
                "schema": "assembly-repair-selector-v1",
                "cad_id": source["cad_id"],
                "parent_id": source["parent_id"],
                "historical_strict_valid": source["brep_valid"],
                "strict_brep_valid": source["cad_id"] not in residual,
                "status": (
                    "both_valid"
                    if source["cad_id"] not in residual
                    else "step_invalid"
                ),
                "error_type": None,
                "selection": {
                    "primary_profile": census.PRIMARY_PROFILE_NAME,
                    "selected_profile": census.PRIMARY_PROFILE_NAME,
                    "attempted_profiles": [census.PRIMARY_PROFILE_NAME],
                    "candidates": [{
                        "profile": census.PRIMARY_PROFILE_NAME,
                        "status": (
                            "both_valid"
                            if source["cad_id"] not in residual
                            else "step_invalid"
                        ),
                        "worker_returncode": 0,
                        "error_type": None,
                        "rejection_reasons": [],
                    }],
                },
            }
        )
    return calibration, selector


def _source(cad_id: str) -> dict[str, object]:
    return {
        "cad_id": cad_id,
        "parent_id": "parent",
        "source_path": "relative.pkl",
        "brep_valid": False,
    }


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


def _exact_lineage(stage="S3"):
    value = {
        "status": "exact_identity",
        "proof_method": "test",
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
                for face_id in range(2) for edge_id in range(4)
            ],
        )
    return value


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


def _stages(first_bad: str = "S4"):
    topology = _source_topology()
    rows = []
    for stage in STAGE_ORDER:
        if stage == "S1":
            stage_topology = {"face_count": topology["face_count"]}
        elif stage == "S2":
            stage_topology = {
                "edge_count": topology["edge_count"],
                "vertex_count": topology["vertex_count"],
                "vertex_edge_incidence_counts": topology["vertex_edge_incidence_counts"],
            }
        else:
            stage_topology = topology
        row = {
            "stage": stage,
            "phase": STAGE_NAMES[stage],
            "status": "observed",
            "lineage": _exact_lineage(stage),
            "topology": stage_topology,
            "defects": [{"code": "wire_adjacent"}] if stage == first_bad else [],
        }
        if stage == "S2":
            row["evidence"] = {
                "stage_local_occ_topology_proof": _stage_local_proof(
                    stage, topology["edge_count"], topology["edge_count"],
                    topology["edge_count"],
                )
            }
        if stage in {"S3", "S4"}:
            row["evidence"] = {
                "stage_local_occ_topology_proof": _stage_local_proof(
                    stage, topology["face_count"], topology["edge_count"],
                    topology["face_edge_occurrence_count"],
                ),
            }
        if stage in {"S5", "S6"}:
            row["evidence"] = {
                "source_vertex_lineage": _exact_global_vertex_lineage(
                    topology,
                    constraint_occurrence_count=topology[
                        "face_edge_occurrence_count"
                    ],
                )
            }
        if stage == "S6":
            row["construction_native_valid"] = True
        if stage == "S7":
            row.update(
                reimport_native_valid=True,
                strict_valid=False,
                evidence={
                    "step_geometry_incidence_proof": (
                        _exact_step_geometry_incidence_proof(topology)
                    )
                },
            )
        rows.append(row)
    return rows


def _scientific_row(task: census.TaskSpec, *, first_bad: str = "S4"):
    source = _source(task.cad_id)
    binding = _binding(task.cad_id)
    assessment = census.assess_stage_lineage(
        _stages(first_bad), source_topology=_source_topology()
    )
    row = census._base_row(
        source, task, run_signature=_digest("run"), expected_binding=binding
    )
    row.update(
        status="completed" if assessment["conclusive"] else "scientific_inconclusive",
        worker_runtime_abi_sentinel=copy.deepcopy(
            census.FROZEN_RUNTIME_IDENTITY
        ),
        stage_records=assessment["stages"],
        assessment=assessment,
        source_binding_before_load=binding,
        source_binding_loaded_bytes=binding,
        source_binding_after_load=binding,
        source_binding_after_measurement=binding,
        source_binding_parent_after_child=binding,
    )
    return source, binding, row


def test_fixed_cohort_and_exact_ten_task_order():
    assert len(census.SELECTOR_RESIDUAL_CAD_IDS) == 9
    assert census.EXCLUDED_EXACT_NEGATIVE_CAD_IDS == {
        census.CAD_47472,
        census.CAD_63055,
    }
    assert census.TARGET_CAD_IDS == (
        census.CAD_51602,
        census.CAD_61931,
        census.CAD_67160,
        census.CAD_87341,
        census.CAD_76198,
        census.CAD_95733,
        census.CAD_32101,
    )
    assert [(task.ordinal, task.cad_id, task.arm) for task in census.TASKS] == [
        *((index, cad_id, census.PRIMARY_ARM)
          for index, cad_id in enumerate(census.TARGET_CAD_IDS, 1)),
        *((index, cad_id, census.BRIDGE_ARM)
          for index, cad_id in enumerate(census.BRIDGE_CAD_IDS, 8)),
    ]
    assert len(census.TASKS_BY_ID) == 10


def test_select_census_sources_proves_100_to_9_to_7_in_frozen_order():
    calibration, selector = _cohort_inputs()
    selected = census.select_census_sources(calibration, selector)

    assert [row["cad_id"] for row in selected] == list(census.TARGET_CAD_IDS)
    assert len(selected) == 7

    broken = [dict(row) for row in selector]
    target = next(row for row in broken if row["cad_id"] == census.CAD_47472)
    target["strict_brep_valid"] = True
    with pytest.raises(ValueError, match="91 strict-valid|residual"):
        census.select_census_sources(calibration, broken)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.__setitem__("status", "worker_timeout"), "worker/protocol"),
        (
            lambda row: row["selection"]["candidates"][0].__setitem__(
                "worker_returncode", 7
            ),
            "return code",
        ),
        (lambda row: row.__setitem__("nonfinite_count", 1), "nonfinite_count"),
        (
            lambda row: row.__setitem__("protocol_failures", ["bad"]),
            "protocol failures",
        ),
    ],
)
def test_selector_health_gate_rejects_operational_failures(mutation, match):
    calibration, selector = _cohort_inputs()
    mutation(selector[0])
    with pytest.raises(ValueError, match=match):
        census.select_census_sources(calibration, selector)


def test_selector_health_gate_retains_scientific_assembly_error():
    calibration, selector = _cohort_inputs()
    target = selector[-1]
    target["status"] = "assembly_error"
    target["error_type"] = "RuntimeError"
    candidate = target["selection"]["candidates"][0]
    candidate.update(status="assembly_error", error_type="RuntimeError")
    assert len(census.select_census_sources(calibration, selector)) == 7


def test_bridge_is_directed_interpolation_reachability_only():
    bridges = [task for task in census.TASKS if task.is_reachability_bridge]
    assert [task.cad_id for task in bridges] == list(census.BRIDGE_CAD_IDS)
    assert all(task.profile_name == "directed_trim_curve_interpolate" for task in bridges)
    assert all(task.switches == ("directed_trim", "curve_interpolate") for task in bridges)
    assert all("curve_fit_rescue" not in task.switches for task in bridges)
    assert all("local_intersection_topology" not in task.switches for task in bridges)


@pytest.mark.parametrize("stage", ["S1", "S2"])
def test_s1_s2_aggregate_per_source_edge_and_reject_duplicates(stage, monkeypatch):
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    entries = [
        {
            "target": edge_handles[edge_id] if stage == "S2" else object(),
            "metadata": {
                "source_edge_id": edge_id,
                "observation_scope": "distributed_source_edge_event",
                "event_sequence_position": 2 * edge_id + (1 if stage == "S2" else 0),
                "fitted_curve_prefix_count": edge_id + 1,
                "built_edge_prefix_count": edge_id + (1 if stage == "S2" else 0),
                "effective_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
            },
        }
        for edge_id in range(topology["edge_count"])
    ]
    record = census._aggregate_edge_stage(stage, entries, topology)
    assert record["lineage"]["status"] == "exact_identity"
    expected_evidence = {
        "observation_granularity": "per_source_edge",
        "observed_source_edge_count": 4,
        "unique_source_edge_count": 4,
        "complete_order_independent_source_edge_coverage": True,
        "terminal_failure_source_edge_id": None,
        "paired_stage_terminal_failure_source_edge_id": None,
    }
    assert {
        key: record["evidence"][key] for key in expected_evidence
    } == expected_evidence
    if stage == "S2":
        proof = record["evidence"]["stage_local_occ_topology_proof"]
        assert proof["status"] == "exact_stage_local_topology"
        assert proof["scope_count"] == topology["edge_count"]
        assert proof["constraint_occurrence_count"] == topology["edge_count"]
    else:
        assert "source_vertex_lineage" not in record["evidence"]

    duplicate = entries[:-1] + [entries[0]]
    record = census._aggregate_edge_stage(stage, duplicate, topology)
    assert record["lineage"]["status"] == "ambiguous"
    assert "source_edge_coverage_incomplete_or_duplicate" in record["lineage"]["failure_codes"]


def test_source_topology_contains_canonical_source_bound_relations():
    parsed = {
        "faceEdge_adj": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "edgeCorner_adj": [[0, 1], [1, 2], [2, 3], [3, 0]],
    }
    assert census._source_topology(parsed) == _source_topology()


def test_face_stage_topology_is_derived_from_mapping_assignments(monkeypatch):
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    entries = []
    for face_id, edge_ids in enumerate(topology["face_edge_source_ids"]):
        entries.append(
            {
                "source_face_index": face_id,
                "face": object(),
                "source_mapping": {
                    "status": "exact_identity",
                    "wire_rows": [
                        {
                            "source_edge_candidates": [
                                {"source_edge_id": edge_id, "observed_edge": object()}
                                for edge_id in reversed(edge_ids)
                            ]
                        }
                    ],
                },
            }
        )
    endpoint_references = {
        edge_id: {
            "endpoints": edge_handles[edge_id],
            "source_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
        }
        for edge_id in range(topology["edge_count"])
    }
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    monkeypatch.setattr(census, "_endpoint_pair_is_same", lambda _a, _b: True)
    for entry in entries:
        for wire_row in entry["source_mapping"]["wire_rows"]:
            for candidate in wire_row["source_edge_candidates"]:
                candidate["observed_edge"] = tuple(
                    reversed(edge_handles[candidate["source_edge_id"]])
                )
    record, _references = census._aggregate_face_stage(
        "S3",
        entries,
        topology,
        s2_endpoint_references=endpoint_references,
    )
    assert record["lineage"]["status"] == "exact_identity"
    assert record["topology"]["face_edge_source_ids"] == [
        list(reversed(row)) for row in topology["face_edge_source_ids"]
    ]
    assert record["topology"] is not topology
    proof = record["evidence"]["stage_local_occ_topology_proof"]
    assert proof["status"] == "exact_stage_local_topology"
    assert proof["scope_kind"] == "source_face"
    assert proof["scope_count"] == topology["face_count"]
    assert proof["source_edge_count"] == topology["edge_count"]
    assert proof["constraint_occurrence_count"] == topology[
        "face_edge_occurrence_count"
    ]
    assert "endpoint_identity_source_edge_count" not in record["evidence"]


def test_global_edge_stream_requires_exact_s1_s2_interleave():
    topology = _source_topology()
    stream = []
    for edge_id in range(4):
        for stage in ("S1", "S2"):
            stream.append(
                {
                    "metadata": {
                        "stage": stage,
                        "source_edge_id": edge_id,
                        "event_sequence_position": len(stream),
                        "boundary_event": "completed",
                    }
                }
            )
    assert census._validate_edge_event_stream(stream, topology) == []
    stream[1], stream[2] = stream[2], stream[1]
    assert "global_edge_stage_interleave_mismatch" in census._validate_edge_event_stream(
        stream, topology
    )


def _face_event(stage, face_id, position, *, boundary="completed"):
    return {
        "metadata": {
            "stage": stage,
            "source_face_index": face_id,
            "event_sequence_position": position,
            "boundary_event": boundary,
        }
    }


def test_face_failure_localization_uses_only_exact_callback_prefix():
    topology = _source_topology()

    # Before any S3 callback, the only exact next boundary is S3(face 0).
    assert census._localize_face_failure_from_event_prefix(
        [], topology, RuntimeError("opaque native wrapper failure")
    ) == ([], "S3", 0)

    # After S3(face 0), the only exact next boundary is S4(face 0).
    s3_prefix = [_face_event("S3", 0, 0)]
    assert census._localize_face_failure_from_event_prefix(
        s3_prefix, topology, RuntimeError("do not parse face=999")
    ) == ([], "S4", 0)

    # After a complete S3/S4 pair, the next source-bound boundary advances.
    pair = [*s3_prefix, _face_event("S4", 0, 1)]
    assert census._localize_face_failure_from_event_prefix(
        pair, topology, RuntimeError("opaque")
    ) == ([], "S3", 1)


def test_face_failure_localization_is_fail_closed_for_observer_and_bad_streams():
    topology = _source_topology()
    observer_error = RuntimeError(
        "assembly_stage_observer_failed stage=S3 error_type=ValueError"
    )
    assert census._localize_face_failure_from_event_prefix(
        [_face_event("S3", 0, 0)], topology, observer_error
    ) == ([], None, None)

    malformed_streams = [
        # Out of order: S4 cannot precede S3.
        [_face_event("S4", 0, 0)],
        # Duplicate S3 callback at the next S4 position.
        [_face_event("S3", 0, 0), _face_event("S3", 0, 1)],
        # Missing the face-0 pair before face 1.
        [_face_event("S3", 1, 0)],
    ]
    for stream in malformed_streams:
        failures, stage, face_id = census._localize_face_failure_from_event_prefix(
            stream, topology, RuntimeError("opaque")
        )
        assert failures
        assert stage is None
        assert face_id is None


@pytest.mark.parametrize(("stage", "prerequisite"), [("S5", "S4"), ("S6", "S5")])
def test_whole_stage_exception_state_machine_requires_exact_prerequisite(
    stage, prerequisite
):
    topology = _source_topology()
    target_index = STAGE_ORDER.index(stage)
    preceding = _stages(first_bad="never")[:target_index]
    raw = {name: [] for name in STAGE_ORDER[:-1]}
    error = RuntimeError("opaque construction exception")
    assert census._whole_stage_failure_is_localizable(
        preceding,
        stage=stage,
        construction_error=error,
        raw=raw,
        source_topology=topology,
    ) is True
    terminal = census._whole_stage_terminal_record(stage, topology, error)
    normalized = census.normalize_stage_record(
        terminal, expected_stage=stage, source_topology=topology
    )
    assert normalized["lineage"]["local_failure_exact"] is True
    assert normalized["lineage"]["whole_stage_terminal"] == {
        "scope_kind": "whole_shape_boundary_failure",
        "boundary_stage": stage,
        "prerequisite_stage": prerequisite,
        "prerequisite_exact": True,
        "construction_exception_observed": True,
    }

    ambiguous = [dict(row) for row in preceding]
    ambiguous[-1] = {
        **ambiguous[-1],
        "lineage": {**ambiguous[-1]["lineage"], "status": "ambiguous"},
    }
    assert census._whole_stage_failure_is_localizable(
        ambiguous,
        stage=stage,
        construction_error=error,
        raw=raw,
        source_topology=topology,
    ) is False
    assert census._whole_stage_failure_is_localizable(
        preceding,
        stage=stage,
        construction_error=RuntimeError("assembly_stage_observer_failed S5"),
        raw=raw,
        source_topology=topology,
    ) is False
    malformed = dict(raw)
    malformed[stage] = [{"target": object(), "metadata": {}}]
    assert census._whole_stage_failure_is_localizable(
        preceding,
        stage=stage,
        construction_error=error,
        raw=malformed,
        source_topology=topology,
    ) is False


def test_edge_build_terminal_keeps_s1_as_prefix_and_s2_as_local_failure(monkeypatch):
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    s1_entries, s2_entries = [], []
    for edge_id in range(3):
        s1_entries.append(
            {
                "target": object(),
                "metadata": {
                    "stage": "S1", "source_edge_id": edge_id,
                    "observation_scope": "distributed_source_edge_event",
                    "event_sequence_position": 2 * edge_id,
                    "fitted_curve_prefix_count": edge_id + 1,
                    "built_edge_prefix_count": edge_id,
                    "effective_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
                    "boundary_event": "completed",
                },
            }
        )
        s2_entries.append(
            {
                "target": None if edge_id == 2 else edge_handles[edge_id],
                "metadata": {
                    "stage": "S2", "source_edge_id": edge_id,
                    "observation_scope": "distributed_source_edge_event",
                    "event_sequence_position": 2 * edge_id + 1,
                    "fitted_curve_prefix_count": edge_id + 1,
                    "built_edge_prefix_count": edge_id if edge_id == 2 else edge_id + 1,
                    "effective_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
                    "boundary_event": "terminal_failure" if edge_id == 2 else "completed",
                },
            }
        )
    s1 = census._aggregate_edge_stage("S1", s1_entries, topology, paired_entries=s2_entries)
    s2 = census._aggregate_edge_stage("S2", s2_entries, topology, paired_entries=s1_entries)
    assert s1["lineage"]["status"] == "exact_prefix"
    assert s2["lineage"]["status"] == "local_exact_failure"
    assert s1["lineage"]["entities"] == {}
    assert s2["lineage"]["entities"] == {}
    assert s1["lineage"]["distributed_scope"]["completed_ids"] == [0, 1, 2]
    assert s2["lineage"]["distributed_scope"]["completed_ids"] == [0, 1]
    assert s1["lineage"]["distributed_scope"]["event_sequence_proof"]["events"] == [
        {"entity_id": 0, "event": "pre_boundary_ok"},
        {"entity_id": 1, "event": "pre_boundary_ok"},
        {"entity_id": 2, "event": "pre_boundary_ok"},
    ]
    assert s2["lineage"]["distributed_scope"]["event_sequence_proof"]["events"] == [
        {"entity_id": 0, "event": "post_boundary_ok"},
        {"entity_id": 1, "event": "post_boundary_ok"},
        {"entity_id": 2, "event": "terminal_failure"},
    ]
    assessed = census.assess_stage_lineage(
        [
            s1,
            s2,
            *(
                census.make_not_reached_stage(stage, "blocked", blocked_by_stage="S2")
                for stage in STAGE_ORDER[2:]
            ),
        ],
        source_topology=topology,
    )
    assert assessed["first_bad_stage"] == "S2"
    assert assessed["stages"][0]["lineage"]["prefix_pass_exact"] is True
    assert assessed["stages"][1]["lineage"]["local_failure_exact"] is True


def test_curve_fit_terminal_is_s1_local_failure_without_endpoint_metadata():
    topology = _source_topology()
    entries = []
    for edge_id in range(3):
        terminal = edge_id == 2
        metadata = {
            "stage": "S1",
            "source_edge_id": edge_id,
            "observation_scope": "distributed_source_edge_event",
            "event_sequence_position": 2 * edge_id,
            "fitted_curve_prefix_count": edge_id if terminal else edge_id + 1,
            "built_edge_prefix_count": edge_id,
            "boundary_event": "terminal_failure" if terminal else "completed",
        }
        if not terminal:
            metadata["effective_vertex_ids"] = topology["edge_vertex_source_ids"][edge_id]
        entries.append({"target": None if terminal else object(), "metadata": metadata})
    record = census._aggregate_edge_stage("S1", entries, topology)
    assert record["lineage"]["status"] == "local_exact_failure"
    assert record["lineage"]["entities"] == {}
    assert record["lineage"]["distributed_scope"]["completed_ids"] == [0, 1]
    assert record["lineage"]["distributed_scope"]["event_sequence_proof"]["events"] == [
        {"entity_id": 0, "event": "pre_boundary_ok"},
        {"entity_id": 1, "event": "pre_boundary_ok"},
        {"entity_id": 2, "event": "terminal_failure"},
    ]
    assessed = census.assess_stage_lineage(
        [
            record,
            *(
                census.make_not_reached_stage(stage, "blocked", blocked_by_stage="S1")
                for stage in STAGE_ORDER[1:]
            ),
        ],
        source_topology=topology,
    )
    assert assessed["first_bad_stage"] == "S1"


def test_compact_failure_redacts_embedded_paths_and_native_handles():
    failure = census._compact_failure(
        RuntimeError(
            r"wrapper failed at D:\private\cad\source.pkl for "
            r"<OCC.Core.TopoDS.TopoDS_Shape; proxy of <Swig Object of type 'TopoDS_Shape *'>>"
        )
    )
    assert failure["kind"] == "RuntimeError"
    assert "D:" not in failure["reason"]
    assert "source.pkl" not in failure["reason"]
    assert "Swig Object" not in failure["reason"]
    census.assert_path_free_finite(failure)


class _FakeOcc:
    def __init__(self, identity):
        self.identity = identity

    def IsSame(self, other):
        return isinstance(other, _FakeOcc) and self.identity == other.identity


class _OwnedFakeOcc(_FakeOcc):
    def __init__(self, owner, identity):
        super().__init__(identity)
        self.owner = owner

    def IsSame(self, other):
        if isinstance(other, _OwnedFakeOcc) and self.owner != other.owner:
            raise AssertionError("stage-local proof compared different OCC owners")
        return (
            isinstance(other, _OwnedFakeOcc)
            and self.owner == other.owner
            and self.identity == other.identity
        )


def _fake_edge_handles(topology):
    vertices = [_FakeOcc(index) for index in range(topology["vertex_count"])]
    return [
        tuple(vertices[vertex_id] for vertex_id in endpoint_ids)
        for endpoint_ids in topology["edge_vertex_source_ids"]
    ]


def test_endpoint_pair_identity_is_direction_independent_and_preserves_self_loop():
    first, second = _FakeOcc("v0"), _FakeOcc("v1")
    assert census._endpoint_pair_is_same((first, second), (second, first)) is True
    assert census._endpoint_pair_is_same((first, first), (first, first)) is True
    assert census._endpoint_pair_is_same((first, first), (first, second)) is False


def test_global_vertex_proof_rejects_cross_edge_split_and_merge():
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    exact = census._prove_endpoint_occurrence_vertex_lineage(
        [
            {"source_edge_id": edge_id, "endpoints": endpoints}
            for edge_id, endpoints in enumerate(edge_handles)
        ],
        topology,
        expected_occurrence_edge_ids=list(range(topology["edge_count"])),
    )
    assert exact["status"] == "exact_identity"
    assert exact["solution_count"] == 1

    split_handles = list(edge_handles)
    split_handles[1] = (_FakeOcc("split-v1"), split_handles[1][1])
    split = census._prove_endpoint_occurrence_vertex_lineage(
        [
            {"source_edge_id": edge_id, "endpoints": endpoints}
            for edge_id, endpoints in enumerate(split_handles)
        ],
        topology,
        expected_occurrence_edge_ids=list(range(topology["edge_count"])),
    )
    assert split["status"] == "ambiguous"
    assert "source_vertex_split_or_extra_observed_vertex" in split["failure_codes"]

    merged_handles = list(edge_handles)
    v0 = merged_handles[0][0]
    merged_handles[1] = (merged_handles[1][0], v0)
    merged_handles[2] = (v0, merged_handles[2][1])
    merge = census._prove_endpoint_occurrence_vertex_lineage(
        [
            {"source_edge_id": edge_id, "endpoints": endpoints}
            for edge_id, endpoints in enumerate(merged_handles)
        ],
        topology,
        expected_occurrence_edge_ids=list(range(topology["edge_count"])),
    )
    assert merge["status"] == "ambiguous"
    assert "source_vertex_merge_or_missing_observed_vertex" in merge["failure_codes"]


def test_face_stage_uses_only_current_face_handles_and_ignores_s2_references(
    monkeypatch,
):
    topology = _source_topology()
    entries = []
    for face_id, edge_ids in enumerate(topology["face_edge_source_ids"]):
        local_vertices = {
            vertex_id: _OwnedFakeOcc(face_id, vertex_id)
            for vertex_id in range(topology["vertex_count"])
        }
        candidates = []
        for occurrence_index, edge_id in enumerate(edge_ids):
            first, second = topology["edge_vertex_source_ids"][edge_id]
            endpoints = (local_vertices[first], local_vertices[second])
            if occurrence_index % 2:
                endpoints = tuple(reversed(endpoints))
            candidates.append(
                {"source_edge_id": edge_id, "observed_edge": endpoints}
            )
        entries.append(
            {
                "source_face_index": face_id,
                "face": object(),
                "source_mapping": {
                    "status": "exact_identity",
                    "wire_rows": [{"source_edge_candidates": candidates}],
                },
            }
        )

    class _PoisonedReferences(dict):
        def __bool__(self):
            raise AssertionError("S3/S4 inspected S2 endpoint references")

        def __iter__(self):
            raise AssertionError("S3/S4 iterated S2 endpoint references")

        def __getitem__(self, key):
            raise AssertionError(f"S3/S4 read S2 endpoint reference {key}")

        def get(self, key, default=None):
            raise AssertionError(f"S3/S4 read S2 endpoint reference {key}")

    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    record, _ = census._aggregate_face_stage(
        "S4",
        entries,
        topology,
        s2_endpoint_references=_PoisonedReferences(),
    )
    assert record["lineage"]["status"] == "exact_identity"
    proof = record["evidence"]["stage_local_occ_topology_proof"]
    assert proof["status"] == "exact_stage_local_topology"
    assert proof["scope_count"] == 2
    assert proof["constraint_occurrence_count"] == 8


def test_stage_local_proof_handles_self_loops_and_endpoint_extraction_errors(
    monkeypatch,
):
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    shared = _FakeOcc("loop")
    exact = census._prove_stage_local_occ_topology(
        [[(0, (0, 0), (shared, shared))]], scope_kind="source_face"
    )
    assert exact["status"] == "exact_stage_local_topology"

    split = census._prove_stage_local_occ_topology(
        [[(0, (0, 0), (_FakeOcc("left"), _FakeOcc("right")))]],
        scope_kind="source_face",
    )
    assert split["status"] == "ambiguous"
    assert "scope_0_source_vertex_split_or_merge" in split["failure_codes"]

    def raising_endpoints(_edge):
        raise RuntimeError("native endpoint extraction failed")

    monkeypatch.setattr(census, "_edge_endpoint_handles", raising_endpoints)
    failed = census._prove_stage_local_occ_topology(
        [[(0, (0, 1), object())]], scope_kind="source_face"
    )
    assert failed["status"] == "ambiguous"
    assert any(
        code.startswith("scope_0_occurrence_0_malformed:RuntimeError")
        for code in failed["failure_codes"]
    )


def test_stage_local_identity_measurement_error_is_ambiguous(monkeypatch):
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)

    class _RaisingOcc(_FakeOcc):
        def IsSame(self, other):
            raise RuntimeError("native identity measurement failed")

    failed = census._prove_stage_local_occ_topology(
        [[
            (0, (0, 1), (_FakeOcc("v0"), _FakeOcc("v1"))),
            (1, (1, 2), (_RaisingOcc("v1-copy"), _FakeOcc("v2"))),
        ]],
        scope_kind="source_face",
    )
    assert failed["status"] == "ambiguous"
    assert any(
        code.startswith("scope_0_occurrence_1_identity_measurement_failed:")
        for code in failed["failure_codes"]
    )


def test_s2_stage_local_topology_proof_is_required_for_completed_prefix(monkeypatch):
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    entries = []
    for edge_id in range(3):
        terminal = edge_id == 2
        entries.append(
            {
                "target": None if terminal else edge_handles[edge_id],
                "metadata": {
                    "source_edge_id": edge_id,
                    "observation_scope": "distributed_source_edge_event",
                    "event_sequence_position": 2 * edge_id + 1,
                    "fitted_curve_prefix_count": edge_id + 1,
                    "built_edge_prefix_count": edge_id if terminal else edge_id + 1,
                    "effective_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
                    "boundary_event": "terminal_failure" if terminal else "completed",
                },
            }
        )
    record = census._aggregate_edge_stage("S2", entries, topology)
    assert record["lineage"]["status"] == "local_exact_failure"
    assert record["evidence"]["stage_local_occ_topology_proof"][
        "constraint_occurrence_count"
    ] == 2

    split = _FakeOcc("split-v1")
    entries[1]["target"] = (split, edge_handles[1][1])
    record = census._aggregate_edge_stage("S2", entries, topology)
    assert record["lineage"]["status"] == "local_exact_failure"
    assert record["evidence"]["stage_local_occ_topology_proof"]["status"] == (
        "exact_stage_local_topology"
    )


def test_face_stage_endpoint_drift_is_fail_closed(monkeypatch):
    topology = _source_topology()
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    v0, v1, v2, v3 = (_FakeOcc(index) for index in range(4))
    endpoint_by_edge = {
        0: (v0, v1), 1: (v1, v2), 2: (v2, v3), 3: (v3, v0),
    }
    references = {
        edge_id: {
            "endpoints": endpoints,
            "source_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
        }
        for edge_id, endpoints in endpoint_by_edge.items()
    }
    observed = dict(endpoint_by_edge)
    observed[2] = (v0, v3)
    entries = [
        {
            "source_face_index": face_id,
            "face": object(),
            "source_mapping": {
                "status": "exact_identity",
                "wire_rows": [
                    {
                        "source_edge_candidates": [
                            {"source_edge_id": edge_id, "observed_edge": observed[edge_id]}
                            for edge_id in edge_ids
                        ]
                    }
                ],
            },
        }
        for face_id, edge_ids in enumerate(topology["face_edge_source_ids"])
    ]
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    record, _ = census._aggregate_face_stage(
        "S3", entries, topology, s2_endpoint_references=references
    )
    assert record["lineage"]["status"] == "ambiguous"
    assert "scope_0_source_topology_not_bijective" in record["lineage"]["failure_codes"]
    assert "scope_1_source_topology_not_bijective" in record["lineage"]["failure_codes"]
    assert record["topology"] is None


def test_s3_prefix_before_s4_failure_does_not_swallow_endpoint_drift(monkeypatch):
    topology = _source_topology()
    v0, v1, v2, v3 = (_FakeOcc(index) for index in range(4))
    references = {
        0: {"endpoints": (v0, v1), "source_vertex_ids": [0, 1]},
        1: {"endpoints": (v1, v2), "source_vertex_ids": [1, 2]},
        2: {"endpoints": (v2, v3), "source_vertex_ids": [2, 3]},
        3: {"endpoints": (v0, v3), "source_vertex_ids": [0, 3]},
    }
    observed = {
        0: (v0, v1),
        1: (v0, v2),  # source edge 1 drifted away from its S2 endpoint pair
        2: (v2, v3),
        3: (v0, v3),
    }
    entry = {
        "target": object(),
        "metadata": {
            "source_face_index": 0,
            "source_mapping": {
                "status": "exact_identity",
                "wire_rows": [
                    {
                        "source_edge_candidates": [
                            {
                                "source_edge_id": edge_id,
                                "observed_edge": observed[edge_id],
                            }
                            for edge_id in topology["face_edge_source_ids"][0]
                        ]
                    }
                ],
            },
        },
    }
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    record, _ = census._face_prefix_record(
        "S3",
        topology,
        [entry],
        next_stage_terminal_face_id=0,
        s2_endpoint_references=references,
    )
    assert record["lineage"]["status"] == "ambiguous"
    assert "scope_0_source_topology_not_bijective" in record["lineage"]["failure_codes"]
    assert record["evidence"]["prefix_pass_before_next_stage_failure"] is False
    assert record["topology"] is None


def test_s4_terminal_audits_completed_face_prefix_before_localizing(monkeypatch):
    topology = _source_topology()
    edge_handles = _fake_edge_handles(topology)
    references = {
        edge_id: {
            "endpoints": endpoints,
            "source_vertex_ids": topology["edge_vertex_source_ids"][edge_id],
        }
        for edge_id, endpoints in enumerate(edge_handles)
    }
    candidates = [
        {
            "source_edge_id": edge_id,
            "observed_edge": tuple(reversed(edge_handles[edge_id])),
        }
        for edge_id in topology["face_edge_source_ids"][0]
    ]
    completed_entry = {
        "target": object(),
        "metadata": {
            "source_face_index": 0,
            "source_mapping": {
                "status": "exact_identity",
                "wire_rows": [{"source_edge_candidates": candidates}],
            },
        },
    }
    monkeypatch.setattr(census, "_edge_endpoint_handles", lambda edge: edge)
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    record = census._face_terminal_record(
        "S4",
        topology,
        [completed_entry],
        1,
        failure_code="face_repair_failed",
        s2_endpoint_references=references,
    )
    assert record["lineage"]["status"] == "local_exact_failure"
    assert record["lineage"]["distributed_scope"]["completed_ids"] == [0]
    proof = record["evidence"]["stage_local_occ_topology_proof"]
    assert proof["status"] == "exact_stage_local_topology"
    assert proof["constraint_occurrence_count"] == 4

    # A split of a shared source label inside this one owning face is a real
    # local-topology failure, even though the altered S2 fixture is made to
    # match the individual edge pair.
    candidates[1]["observed_edge"] = (
        _FakeOcc("split-v1"),
        candidates[1]["observed_edge"][1],
    )
    references[1]["endpoints"] = candidates[1]["observed_edge"]
    record = census._face_terminal_record(
        "S4",
        topology,
        [completed_entry],
        1,
        failure_code="face_repair_failed",
        s2_endpoint_references=references,
    )
    assert record["lineage"]["status"] == "ambiguous"
    assert "scope_0_source_vertex_split_or_merge" in record["lineage"][
        "failure_codes"
    ]


def _exact_global_vertex_lineage(
    topology, *, constraint_occurrence_count=None
):
    if constraint_occurrence_count is None:
        constraint_occurrence_count = topology["face_edge_occurrence_count"]
    return {
        "status": "exact_identity",
        "proof_method": (
            "source_edge_endpoint_constraints_plus_target_vertex_IsSame_"
            "unique_perfect_assignment"
        ),
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": topology["vertex_count"],
        "observed_vertex_count": topology["vertex_count"],
        "mapped_source_vertex_count": topology["vertex_count"],
        "mapped_observed_vertex_count": topology["vertex_count"],
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": constraint_occurrence_count,
        "failure_codes": [],
    }


def _exact_step_geometry_incidence_proof(topology):
    self_loop_count = sum(
        endpoints[0] == endpoints[1]
        for endpoints in topology["edge_vertex_source_ids"]
    )
    return {
        "status": "exact_geometry_incidence",
        "failure_codes": [],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_candidate_degree_counts": {"1": topology["face_count"]},
        "face_matching_count_capped": 1,
        "vertex_proof_required": True,
        "vertex_proof_method": STEP_VERTEX_IDENTITY_PROOF_METHOD,
        "vertex_tolerance_normalized": (
            STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ),
        "vertex_candidate_degree_counts": {"1": topology["vertex_count"]},
        "vertex_matching_count_capped": 1,
        "source_vertex_count": topology["vertex_count"],
        "step_vertex_count": topology["vertex_count"],
        "mapped_source_edge_count": topology["edge_count"],
        "edge_endpoint_pair_expected_count": topology["edge_count"],
        "edge_endpoint_pair_proof_count": topology["edge_count"],
        "edge_endpoint_occurrence_expected_count": 2 * topology["edge_count"],
        "edge_endpoint_occurrence_proof_count": 2 * topology["edge_count"],
        "self_loop_endpoint_pair_expected_count": self_loop_count,
        "self_loop_endpoint_pair_proof_count": self_loop_count,
        "mapped_face_count": topology["face_count"],
        "mapped_edge_occurrence_count": topology["face_edge_occurrence_count"],
        "vertex_proof_status": "exact",
    }


@pytest.mark.parametrize(
    "mapping_status",
    ["exact_sewing_history", "exact_sewing_face_local_geometry"],
)
def test_s5_s6_accept_sewing_mapping_with_global_vertex_proof(
    monkeypatch, mapping_status
):
    topology = _source_topology()
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    # These handles intentionally have no S2 IsSame relation.  After sewing,
    # the authoritative proof is the global endpoint-constraint assignment.
    entries = [
        {
            "source_face_index": face_id,
            "face": object(),
            "source_mapping": {
                "status": mapping_status,
                "wire_rows": [
                    {
                        "source_edge_candidates": [
                            {"source_edge_id": edge_id, "observed_edge": object()}
                            for edge_id in edge_ids
                        ]
                    }
                ],
            },
        }
        for face_id, edge_ids in enumerate(topology["face_edge_source_ids"])
    ]
    proof = _exact_global_vertex_lineage(topology)
    for stage in ("S5", "S6"):
        record, references = census._aggregate_face_stage(
            stage, entries, topology, source_vertex_lineage=proof
        )
        assert record["lineage"]["status"] == "exact_identity"
        assert record["lineage"]["failure_codes"] == []
        assert record["topology"]["edge_vertex_source_ids"] == topology[
            "edge_vertex_source_ids"
        ]
        assert record["evidence"] == {"source_vertex_lineage": proof}
        assert sorted(references) == [0, 1]


def test_s5_s6_fail_closed_on_nonexact_global_vertex_proof(monkeypatch):
    topology = _source_topology()
    monkeypatch.setattr(census, "_diagnose_face_defects", lambda *_a: [])
    entries = [
        {
            "source_face_index": face_id,
            "face": object(),
            "source_mapping": {
                "status": "exact_sewing_history",
                "wire_rows": [
                    {
                        "source_edge_candidates": [
                            {"source_edge_id": edge_id, "observed_edge": object()}
                            for edge_id in edge_ids
                        ]
                    }
                ],
            },
        }
        for face_id, edge_ids in enumerate(topology["face_edge_source_ids"])
    ]
    broken = _exact_global_vertex_lineage(topology)
    broken.update(
        status="ambiguous",
        solution_count=2,
        failure_codes=["source_vertex_assignment_nonunique"],
    )
    record, _ = census._aggregate_face_stage(
        "S5", entries, topology, source_vertex_lineage=broken
    )
    assert record["lineage"]["status"] == "ambiguous"
    assert "global_source_vertex_lineage_not_exact" in record["lineage"][
        "failure_codes"
    ]
    assert record["topology"] is None
    assert record["evidence"] == {"source_vertex_lineage": broken}


def test_parent_module_has_no_occ_or_native_assembly_imports_at_module_scope():
    source = Path(census.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "OCC",
        "tools.directed_trim_assembly",
        "directed_trim_assembly",
        "tools.run_assembly_repair_matrix",
        "run_assembly_repair_matrix",
    }
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(
        value == name or value.startswith(name + ".")
        for value in imported for name in forbidden
    )


def test_s7_runner_opts_into_vertex_proof_and_archives_public_proof():
    source = inspect.getsource(census._s7_record)
    assert "edge_vertex_adj=edge_vertex_adj" in source
    assert "source_edge_wcs=edge_wcs" in source
    assert "require_vertex_proof=True" in source
    assert "fail_on_matching_exception=True" in source
    assert "step_geometry_incidence_proof" in source
    # The finalized API derives optimized WCS vertex points from edge_wcs at
    # the same STEP shape scale; raw corner_unique must not be a hidden oracle.
    assert "source_vertex_points=" not in source


def test_worker_escalates_internal_s7_matching_failure_instead_of_downgrading():
    source = inspect.getsource(census.run_worker)
    assert "StepGeometryIncidenceMatchingError" in source
    assert "except StepGeometryIncidenceMatchingError:" in source
    assert "raise" in source


def test_parser_has_no_formal_denominator_or_generic_arm_override(tmp_path):
    args = census.parse_args(
        [
            "--calibration-manifest", str(tmp_path / "cal.jsonl"),
            "--selector-matrix", str(tmp_path / "selector.jsonl"),
            "--selector-run", str(tmp_path / "run.json"),
            "--breparg-root", str(tmp_path / "runtime"),
            "--output-dir", str(tmp_path / "output"),
        ]
    )
    assert not hasattr(args, "max_cads")
    assert not hasattr(args, "arm")
    assert not hasattr(args, "task_ids")


def test_worker_sentinel_is_unique_and_must_be_the_last_nonempty_line():
    good = {"status": "completed"}
    line = census.WORKER_MARKER + json.dumps(good)
    assert census.parse_worker_result("occ output\n" + line + "\n") == good
    assert census.parse_worker_result(line + "\nafter") is None
    assert census.parse_worker_result(line + "\n" + line) is None
    assert census.parse_worker_result(census.WORKER_MARKER + "{") is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"completed","status":"worker_error"}',
        '{"status":"completed","value":NaN}',
        '{"status":"completed","value":Infinity}',
        '{"status":"completed","value":-Infinity}',
        '{"status":"completed","value":1e9999}',
    ],
)
def test_worker_sentinel_rejects_ambiguous_or_nonfinite_json(payload):
    assert census.parse_worker_result(census.WORKER_MARKER + payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e9999}',
    ],
)
def test_strict_json_loader_rejects_duplicates_and_nonfinite_numbers(payload):
    with pytest.raises(ValueError):
        census.strict_json_loads(payload, label="attack")


def test_torn_tail_recovery_removes_only_unterminated_last_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"task_id":"first"}\n{"task_id":')
    assert census.read_rows(path, recover_truncated_tail=True) == [{"task_id": "first"}]
    assert path.read_bytes() == b'{"task_id":"first"}\n'

    path.write_bytes(b'{"task_id":}\n')
    with pytest.raises(json.JSONDecodeError):
        census.read_rows(path, recover_truncated_tail=True)

    path.write_bytes(b'{"task_id":"first"}\n{"task_id":"complete-but-uncommitted"}')
    assert census.read_rows(path, recover_truncated_tail=True) == [
        {"task_id": "first"}
    ]
    assert path.read_bytes() == b'{"task_id":"first"}\n'
    census.append_row(path, {"task_id": "replacement"})
    assert census.read_rows(path) == [
        {"task_id": "first"},
        {"task_id": "replacement"},
    ]


def test_torn_tail_is_never_mutated_without_explicit_recovery(tmp_path):
    path = tmp_path / "terminal-rows.jsonl"
    original = b'{"task_id":"first"}\n{"task_id":'
    path.write_bytes(original)
    with pytest.raises(json.JSONDecodeError):
        census.read_rows(path, recover_truncated_tail=False)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        b'{"task_id":"first","task_id":"second"}\n',
        b'{"task_id":"first","metric":NaN}\n',
        b'{"task_id":"first","metric":Infinity}\n',
        b'{"task_id":"first","metric":1e9999}\n',
    ],
)
def test_jsonl_reader_rejects_ambiguous_or_nonfinite_rows(tmp_path, payload):
    path = tmp_path / "attacked.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        census.read_rows(path)


def test_bind_manifest_is_immutable_and_rejects_unsigned_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    payload = {
        "schema": census.RUN_SCHEMA,
        "run_kind": "formal",
        "ordered_tasks": ["a"],
    }
    first = census.bind_run_manifest(root, payload)
    assert first["signature"] == census.canonical_sha256(payload)
    assert census.bind_run_manifest(root, payload) == first
    with pytest.raises(RuntimeError, match="another census signature"):
        census.bind_run_manifest(root, {**payload, "ordered_tasks": ["b"]})

    other = tmp_path / "other"
    other.mkdir()
    (other / "orphan.txt").write_text("x")
    with pytest.raises(RuntimeError, match="not empty"):
        census.bind_run_manifest(other, payload)


def test_manifest_resume_recomputes_stored_signature_and_uses_exact_types(tmp_path):
    payload = {
        "schema": census.RUN_SCHEMA,
        "run_kind": "formal",
        "joint_iterations": 200,
    }

    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    stored = census.bind_run_manifest(tampered_root, payload)
    attacked = copy.deepcopy(stored)
    attacked["payload"]["joint_iterations"] = 201
    # Keep the old signature: comparing only with today's expected signature
    # must not accept a stored payload whose own signature is now false.
    (tampered_root / census.RUN_NAME).write_text(
        json.dumps(attacked), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="another census signature"):
        census.bind_run_manifest(tampered_root, payload)

    typed_root = tmp_path / "typed"
    typed_root.mkdir()
    census.bind_run_manifest(typed_root, payload)
    manifest_path = typed_root / census.RUN_NAME
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace('"joint_iterations": 200', '"joint_iterations": 200.0')
    manifest_path.write_text(raw, encoding="utf-8")
    with pytest.raises(RuntimeError, match="another census signature"):
        census.bind_run_manifest(typed_root, payload)


@pytest.mark.parametrize(
    "attacked",
    [
        '{"schema":"source-bound-stage-census-run-v1",'
        '"signature":"%s","payload":{},"status":"RUNNING",'
        '"status":"COMPLETED"}',
        '{"schema":"source-bound-stage-census-run-v1",'
        '"signature":"%s","payload":{"value":NaN},"status":"RUNNING"}',
    ],
)
def test_manifest_resume_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path, attacked
):
    root = tmp_path / "run"
    root.mkdir()
    payload = {"run_kind": "formal"}
    signature = census.canonical_sha256(payload)
    (root / census.RUN_NAME).write_text(attacked % signature, encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        census.bind_run_manifest(root, payload)


def test_dirty_development_manifest_cannot_be_resumed(tmp_path):
    root = tmp_path / "development"
    root.mkdir()
    payload = {
        "schema": census.RUN_SCHEMA,
        "run_kind": "development_dirty_nonformal_nonresumable",
    }
    census.bind_run_manifest(root, payload)
    with pytest.raises(RuntimeError, match="non-resumable"):
        census.bind_run_manifest(root, payload)


def test_runtime_identity_uses_isolated_probe_and_matches_frozen_runtime(monkeypatch, tmp_path):
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python-binary")
    expected = copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)
    expected["python"].update(
        executable_name=executable.name,
        executable_bytes=executable.stat().st_size,
        executable_sha256=census.sha256_file(executable),
    )
    monkeypatch.setattr(census, "FROZEN_RUNTIME_IDENTITY", expected)
    monkeypatch.setattr(census.sys, "executable", str(executable))
    monkeypatch.setattr(
        census.platform, "python_implementation",
        lambda: expected["python"]["implementation"],
    )
    monkeypatch.setattr(
        census.platform, "python_version", lambda: expected["python"]["version"]
    )
    calls = []

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(expected, sort_keys=True) + "\n"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(census.subprocess, "run", fake_run)

    assert census._runtime_identity() == expected
    assert calls[0][0] == [str(executable), "-I", "-c", census.RUNTIME_PROBE_SOURCE]
    assert calls[0][1]["timeout"] == 60.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra="unexpected"),
        lambda value: value["python"].update(executable_bytes=96256.0),
        lambda value: value["python"].update(executable_sha256="A" * 64),
        lambda value: value["pythonocc"].update(wrapper_binary_name="C:/x/_Standard.pyd"),
        lambda value: value["occt"].update(kernel_binary_bytes=True),
        lambda value: value["numpy"].update(extra="unexpected"),
    ],
)
def test_runtime_identity_schema_rejects_extra_fields_bad_types_and_paths(mutation):
    identity = copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)
    mutation(identity)
    with pytest.raises(ValueError):
        census.normalize_runtime_identity(identity)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema":"source-bound-runtime-identity-v1",'
        '"schema":"source-bound-runtime-identity-v1"}',
        '{"schema":"source-bound-runtime-identity-v1","value":NaN}',
        '{"schema":"source-bound-runtime-identity-v1","value":Infinity}',
    ],
)
def test_runtime_probe_rejects_duplicate_keys_and_nonfinite_json(monkeypatch, payload):
    class Completed:
        returncode = 0
        stderr = ""
        stdout = payload + "\n"

    monkeypatch.setattr(census.subprocess, "run", lambda *_a, **_k: Completed())
    with pytest.raises(RuntimeError, match="invalid JSON"):
        census._runtime_identity()


@pytest.mark.parametrize(
    ("returncode", "stderr", "stdout", "match"),
    [
        (1, "", "", "failed or was noisy"),
        (0, "native warning", "{}\n", "failed or was noisy"),
        (0, "", "{}\n{}\n", "failed or was noisy"),
        (0, "", "not-json\n", "invalid JSON"),
    ],
)
def test_runtime_identity_fails_closed_on_probe_protocol_errors(
    monkeypatch, returncode, stderr, stdout, match
):
    class Completed:
        pass

    completed = Completed()
    completed.returncode = returncode
    completed.stderr = stderr
    completed.stdout = stdout
    monkeypatch.setattr(census.subprocess, "run", lambda *_a, **_k: completed)

    with pytest.raises(RuntimeError, match=match):
        census._runtime_identity()


def test_runtime_identity_rejects_interpreter_or_native_drift(monkeypatch):
    identity = copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)

    class Completed:
        returncode = 0
        stderr = ""

    def probe(*_args, **_kwargs):
        completed = Completed()
        completed.stdout = json.dumps(identity) + "\n"
        return completed

    monkeypatch.setattr(census.subprocess, "run", probe)
    identity["python"]["executable_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="another interpreter"):
        census._runtime_identity()

    identity["python"] = copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY["python"])
    identity["numpy"]["version"] = "0.0.0"
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        census._runtime_identity()


def test_run_isolated_uses_isolated_runpy_bootstrap(monkeypatch, tmp_path):
    task = census.TASKS[0]
    source_path = tmp_path / "source.pkl"
    source_path.write_bytes(b"source")
    source = {
        "cad_id": task.cad_id,
        "parent_id": "parent",
        "source_path": str(source_path),
        "brep_valid": False,
    }
    binding = census.source_binding(source_path)
    args = argparse.Namespace(
        output_dir=tmp_path / "run",
        calibration_manifest=tmp_path / "calibration.jsonl",
        selector_matrix=tmp_path / "selector.jsonl",
        selector_run=tmp_path / "selector-run.json",
        breparg_root=tmp_path / "breparg",
        joint_iterations=200,
        worker_timeout_seconds=30.0,
        development_allow_dirty=False,
    )
    calls = []

    class Completed:
        returncode = 9
        stdout = ""
        stderr = "synthetic exit"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(census.subprocess, "run", fake_run)
    row = census.run_isolated(
        source,
        task,
        args=args,
        run_signature=_digest("run"),
        expected_binding=binding,
        expected_runtime_abi_sentinel=copy.deepcopy(
            census.FROZEN_RUNTIME_IDENTITY
        ),
    )

    command = calls[0][0]
    assert command[:5] == [
        sys.executable,
        "-I",
        "-c",
        census.WORKER_BOOTSTRAP_SOURCE,
        str(Path(census.__file__).resolve().parents[1]),
    ]
    assert "--worker-task-id" in command
    assert task.task_id in command
    assert row["status"] == "worker_process_exit"
    assert row["worker_runtime_abi_sentinel"] is None


def test_worker_measures_same_process_sentinel_before_scientific_work(
    monkeypatch, tmp_path
):
    task = census.TASKS[0]
    parsed = {
        "faceEdge_adj": [[0]],
        "edgeCorner_adj": [[0, 1]],
        "surf_ncs": [[[0.0]]],
        "edge_ncs": [[[0.0]]],
        "surf_bbox_wcs": [[[0.0]]],
        "corner_unique": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    source_path = tmp_path / "source.pkl"
    source_path.write_bytes(pickle.dumps(parsed))
    source = {
        "cad_id": task.cad_id,
        "parent_id": "parent",
        "source_path": str(source_path),
        "brep_valid": False,
    }
    binding = census.source_binding(source_path)
    events = []

    def measure():
        events.append("measure")
        return copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)

    def optimize(*_args, **_kwargs):
        events.append("optimize")
        return [], []

    calibration = types.ModuleType("tools.run_assembly_calibration_oracle")
    calibration.cpu_joint_optimize = optimize
    monkeypatch.setitem(
        sys.modules, "tools.run_assembly_calibration_oracle", calibration
    )
    monkeypatch.setattr(
        census, "_measure_runtime_abi_sentinel_current_process", measure
    )
    monkeypatch.setattr(
        census,
        "_stage_records_from_native",
        lambda *_args, **_kwargs: ([], None, [], {}),
    )
    monkeypatch.setattr(
        census,
        "assess_stage_lineage",
        lambda *_args, **_kwargs: {"conclusive": True, "stages": []},
    )

    row = census.run_worker(
        source,
        task,
        output_dir=tmp_path / "attempt",
        breparg_root=tmp_path / "breparg",
        joint_iterations=0,
        expected_binding=binding,
        run_signature=_digest("run"),
        expected_runtime_abi_sentinel=copy.deepcopy(
            census.FROZEN_RUNTIME_IDENTITY
        ),
        worker_runtime_abi_sentinel=measure(),
    )
    assert events == ["measure", "optimize"]
    assert row["status"] == "completed"
    assert row["worker_runtime_abi_sentinel"] == census.FROZEN_RUNTIME_IDENTITY

    drifted = copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)
    drifted["numpy"]["version"] = "0.0.0"
    events.clear()
    row = census.run_worker(
        source,
        task,
        output_dir=tmp_path / "attempt-drift",
        breparg_root=tmp_path / "breparg",
        joint_iterations=0,
        expected_binding=binding,
        run_signature=_digest("run"),
        expected_runtime_abi_sentinel=copy.deepcopy(
            census.FROZEN_RUNTIME_IDENTITY
        ),
        worker_runtime_abi_sentinel=drifted,
    )
    assert events == []
    assert row["status"] == "worker_error"
    assert row["worker_runtime_abi_sentinel"] is None


def test_worker_main_measures_runtime_before_loading_any_inputs(monkeypatch):
    events = []
    args = argparse.Namespace(worker_task_id="synthetic-worker")
    monkeypatch.setattr(census, "parse_args", lambda _argv=None: args)

    def measure():
        events.append("measure")
        return copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY)

    def load(_args):
        events.append("load-inputs")
        raise RuntimeError("stop after ordering observation")

    monkeypatch.setattr(
        census, "_measure_runtime_abi_sentinel_current_process", measure
    )
    monkeypatch.setattr(census, "_load_inputs", load)
    with pytest.raises(RuntimeError, match="ordering observation"):
        census.main([])
    assert events == ["measure", "load-inputs"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("worker_runtime_abi_sentinel", None),
        lambda value: value["worker_runtime_abi_sentinel"]["python"].update(
            executable_bytes=96256.0
        ),
        lambda value: value["worker_runtime_abi_sentinel"]["numpy"].update(
            version="0.0.0"
        ),
        lambda value: value["worker_runtime_abi_sentinel"].update(
            scope="complete_native_runtime_inventory"
        ),
        lambda value: value["worker_runtime_abi_sentinel"].update(
            extra="unexpected"
        ),
    ],
)
def test_scientific_row_requires_exact_worker_runtime_abi_sentinel(mutation):
    task = census.TASKS[0]
    source, binding, row = _scientific_row(task)
    mutation(row)
    with pytest.raises(ValueError, match="runtime ABI sentinel"):
        census.validate_attempt_row(
            row,
            source=source,
            task=task,
            run_signature=_digest("run"),
            expected_binding=binding,
            expected_runtime_abi_sentinel=copy.deepcopy(
                census.FROZEN_RUNTIME_IDENTITY
            ),
        )


def test_worker_failure_must_keep_runtime_abi_sentinel_null():
    task = census.TASKS[0]
    source, binding = _source(task.cad_id), _binding(task.cad_id)
    row = census.failure_row(
        source,
        task,
        run_signature=_digest("run"),
        expected_binding=binding,
        status="worker_timeout",
        error_type="TimeoutExpired",
        parent_after_child=binding,
    )
    census.validate_attempt_row(
        row,
        source=source,
        task=task,
        run_signature=_digest("run"),
        expected_binding=binding,
    )
    row["worker_runtime_abi_sentinel"] = copy.deepcopy(
        census.FROZEN_RUNTIME_IDENTITY
    )
    with pytest.raises(ValueError, match="failure cannot claim"):
        census.validate_attempt_row(
            row,
            source=source,
            task=task,
            run_signature=_digest("run"),
            expected_binding=binding,
        )

def test_git_identity_requires_verified_full_commit_object(monkeypatch, tmp_path):
    calls = []

    class Completed:
        def __init__(self, stdout="", returncode=0):
            self.stdout = stdout
            self.returncode = returncode

    def valid_run(command, **_kwargs):
        calls.append(command)
        if command[:3] == ["git", "rev-parse", "--verify"]:
            return Completed("A" * 40 + "\n")
        return Completed("")

    monkeypatch.setattr(census.subprocess, "run", valid_run)
    identity = census._git_identity(tmp_path, allow_dirty_development=False)
    assert calls[0] == ["git", "rev-parse", "--verify", "HEAD^{commit}"]
    assert identity["commit"] == "a" * 40
    assert identity["upstream_commit"] == "a" * 40
    assert identity["head_matches_upstream"] is True
    assert identity["formal"] is True

    def abbreviated_run(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "--verify"]:
            return Completed("abc1234\n")
        return Completed("")

    monkeypatch.setattr(census.subprocess, "run", abbreviated_run)
    with pytest.raises(RuntimeError, match="40-hex commit object"):
        census._git_identity(tmp_path, allow_dirty_development=False)

    def diverged_run(command, **_kwargs):
        if command[-1] == "HEAD^{commit}":
            return Completed("a" * 40 + "\n")
        if command[-1] == "@{upstream}^{commit}":
            return Completed("b" * 40 + "\n")
        return Completed("")

    monkeypatch.setattr(census.subprocess, "run", diverged_run)
    with pytest.raises(RuntimeError, match="HEAD to match its upstream"):
        census._git_identity(tmp_path, allow_dirty_development=False)

    def missing_upstream_run(command, **_kwargs):
        if command[-1] == "HEAD^{commit}":
            return Completed("a" * 40 + "\n")
        if command[-1] == "@{upstream}^{commit}":
            return Completed("", returncode=128)
        return Completed("")

    monkeypatch.setattr(census.subprocess, "run", missing_upstream_run)
    with pytest.raises(RuntimeError, match="could not bind repository identity"):
        census._git_identity(tmp_path, allow_dirty_development=False)


def test_validate_attempt_row_recomputes_stage_assessment_and_five_bindings():
    task = census.TASKS[0]
    source, binding, row = _scientific_row(task)
    census.validate_attempt_row(
        row, source=source, task=task, run_signature=_digest("run"),
        expected_binding=binding,
    )

    row["source_binding_after_measurement"] = _binding("changed")
    with pytest.raises(ValueError, match="five exact"):
        census.validate_attempt_row(
            row, source=source, task=task, run_signature=_digest("run"),
            expected_binding=binding,
        )


def test_local_step_artifact_is_attempt_unique_and_rehashed(tmp_path):
    task = census.TASKS[0]
    _source_value, _binding_value, row = _scientific_row(task)
    artifact_id = "0123456789abcdef-attempt"
    step_path = tmp_path / ".attempts" / artifact_id / "roundtrip.step"
    step_path.parent.mkdir(parents=True)
    step_path.write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;\n")
    row["step_roundtrip"] = {
        "saved_to_persistent_output": True,
        "artifact_id": artifact_id,
        "bytes": step_path.stat().st_size,
        "sha256": census.sha256_file(step_path),
    }
    assert census.validate_local_step_artifact(row, tmp_path) == step_path.resolve()

    step_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding mismatch"):
        census.validate_local_step_artifact(row, tmp_path)

    row["step_roundtrip"]["artifact_id"] = "../escape"
    with pytest.raises(ValueError, match="unsafe"):
        census.validate_local_step_artifact(row, tmp_path)


def test_unsaved_step_contract_requires_null_identity(tmp_path):
    task = census.TASKS[0]
    _source_value, _binding_value, row = _scientific_row(task)
    assert census.validate_local_step_artifact(row, tmp_path) is None
    row["step_roundtrip"]["artifact_id"] = "orphan-attempt"
    with pytest.raises(ValueError, match="must be null"):
        census.validate_local_step_artifact(row, tmp_path)


@pytest.mark.parametrize(
    "status",
    ["worker_timeout", "worker_process_exit", "worker_spawn_error", "worker_protocol_error"],
)
def test_worker_failures_are_valid_denominator_rows(status):
    task = census.TASKS[0]
    source, binding = _source(task.cad_id), _binding(task.cad_id)
    row = census.failure_row(
        source, task, run_signature=_digest("run"), expected_binding=binding,
        status=status, error_type="SyntheticFailure",
        parent_after_child=binding,
    )
    census.validate_attempt_row(
        row, source=source, task=task, run_signature=_digest("run"),
        expected_binding=binding,
    )
    assert row["denominator"] is True
    assert row["stage_records"] == []


def test_summary_never_counts_bridge_as_repair_or_authorizes_later_stages():
    rows = []
    for task in census.TASKS:
        _source_value, _binding_value, row = _scientific_row(task)
        rows.append(row)
    summary = census.summarize(rows)

    assert summary["protocol_conclusive"] is True
    assert summary["denominator_rows"] == 10
    assert summary["primary_controls"] == 7
    assert summary["curve_interpolate_bridges"] == 3
    assert summary["bridge_repairs_counted"] == 0
    assert summary["selector_strict_valid_before"] == 91
    assert summary["selector_strict_valid_after"] == 91
    assert summary["authorizes_exact_candidate_design"] is True
    for key in (
        "authorizes_repair", "authorizes_residual_expansion",
        "authorizes_full_100cad", "authorizes_selector_score_change",
        "authorizes_schema_v2_relaxation", "authorizes_training",
        "authorizes_sequence_generation", "authorizes_ar",
    ):
        assert summary[key] is False


def test_summary_requires_exact_task_order_and_worker_failure_is_retained():
    rows = [_scientific_row(task)[2] for task in census.TASKS]
    with pytest.raises(ValueError, match="exact ordered"):
        census.summarize(list(reversed(rows)))

    task = census.TASKS[4]
    source, binding = _source(task.cad_id), _binding(task.cad_id)
    rows[4] = census.failure_row(
        source, task, run_signature=_digest("run"), expected_binding=binding,
        status="worker_timeout", error_type="TimeoutExpired",
        parent_after_child=binding,
    )
    summary = census.summarize(rows)
    assert summary["attempts"] == 10
    assert summary["denominator_rows"] == 10
    assert summary["worker_or_protocol_failures"] == 1
    assert summary["protocol_conclusive"] is False
    assert summary["authorizes_exact_candidate_design"] is False


def test_construct_call_passes_both_mutators_explicitly_none_and_observer():
    source = inspect.getsource(census._stage_records_from_native)
    assert "assembly_stage_observer=observer" in source
    assert "post_pcurve_face_mutator=None" in source
    assert "post_sewing_shape_mutator=None" in source
    assert 'constructor_kwargs.get("solid_topology_repair") is not False' in source


def test_static_guard_forbids_closed_exact_pcurve_route():
    source = Path(census.__file__).read_text(encoding="utf-8")
    assert "FixRemovePCurve" not in source
    assert "targeted_nonperiodic_pcurve_repair" not in source
    assert "post_sewing_graph_repair" not in source


def test_terminal_hash_validation_rebinds_both_artifacts(tmp_path):
    rows = tmp_path / census.ROWS_NAME
    summary = tmp_path / census.SUMMARY_NAME
    rows.write_text('{"a":1}\n', encoding="utf-8")
    summary.write_text('{"b":2}\n', encoding="utf-8")
    run = {
        "rows_sha256": census.sha256_file(rows),
        "summary_sha256": census.sha256_file(summary),
    }
    assert census.validate_terminal_artifact_hashes(
        run, rows_path=rows, summary_path=summary
    ) == {"b": 2}
    summary.write_text('{"b":3}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary hash"):
        census.validate_terminal_artifact_hashes(
            run, rows_path=rows, summary_path=summary
        )


def test_current_source_binding_failures_rehashes_changed_and_missing_pickles(
    tmp_path,
):
    sources = []
    bindings = {}
    for index, cad_id in enumerate(census.TARGET_CAD_IDS[:3]):
        source_path = tmp_path / f"source-{index}.pkl"
        source_path.write_bytes(f"source-{index}".encode("ascii"))
        sources.append({"cad_id": cad_id, "source_path": str(source_path)})
        bindings[cad_id] = census.source_binding(source_path)

    assert census.current_source_binding_failures(sources, bindings) == []

    Path(sources[1]["source_path"]).write_bytes(b"changed")
    Path(sources[2]["source_path"]).unlink()
    assert census.current_source_binding_failures(sources, bindings) == [
        census.TARGET_CAD_IDS[1],
        census.TARGET_CAD_IDS[2],
    ]


def test_prefinalization_source_drift_never_writes_nonderivable_terminal(
    monkeypatch, tmp_path
):
    sources = [
        {"cad_id": cad_id, "source_path": str(tmp_path / f"{index}.pkl")}
        for index, cad_id in enumerate(census.TARGET_CAD_IDS)
    ]
    bindings = {cad_id: _binding(cad_id) for cad_id in census.TARGET_CAD_IDS}
    writes = []
    monkeypatch.setattr(
        census,
        "current_source_binding_failures",
        lambda _sources, _bindings: [census.TARGET_CAD_IDS[0]],
    )
    monkeypatch.setattr(
        census,
        "atomic_json",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    # Exercise the policy directly at the helper boundary used by main: drift
    # has no ledger representation, hence callers must raise and leave a
    # RUNNING manifest rather than sign an unrecomputable custom summary.
    failures = census.current_source_binding_failures(sources, bindings)
    with pytest.raises(RuntimeError, match="before finalization"):
        if failures:
            raise RuntimeError(
                "terminal source binding drifted before finalization: "
                + ",".join(failures)
            )
    assert writes == []


def test_terminal_reopen_rehashes_sources_before_accepting_archive(
    monkeypatch, tmp_path
):
    sources = []
    bindings = {}
    for index, cad_id in enumerate(census.TARGET_CAD_IDS):
        source_path = tmp_path / f"source-{index}.pkl"
        source_path.write_bytes(f"source-{index}".encode("ascii"))
        sources.append(
            {
                "cad_id": cad_id,
                "parent_id": f"parent-{index}",
                "source_path": str(source_path),
                "brep_valid": False,
            }
        )
        bindings[cad_id] = census.source_binding(source_path)

        payload = {
            "schema": census.RUN_SCHEMA,
            "run_kind": "formal",
            "python": copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY["python"]),
            "native_runtime": {
                key: copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY[key])
                for key in (
                    "schema", "scope", "process_isolation", "numpy",
                    "pythonocc", "occt",
                )
            },
            "sources": [
            {"cad_id": source["cad_id"], "binding": bindings[source["cad_id"]]}
            for source in sources
        ],
    }
    signature = census.canonical_sha256(payload)
    rows = []
    for task in census.TASKS:
        source = next(item for item in sources if item["cad_id"] == task.cad_id)
        binding = bindings[task.cad_id]
        row = census.failure_row(
            source,
            task,
            run_signature=signature,
            expected_binding=binding,
            status="worker_timeout",
            error_type="TimeoutExpired",
            parent_after_child=binding,
        )
        rows.append(row)
    summary = census.summarize(rows)
    output = tmp_path / "terminal"
    output.mkdir()
    rows_path = output / census.ROWS_NAME
    summary_path = output / census.SUMMARY_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema": census.RUN_SCHEMA,
        "signature": signature,
        "payload": payload,
        "status": "INCONCLUSIVE",
        "attempts": len(census.TASKS),
        "rows_sha256": census.sha256_file(rows_path),
        "summary_sha256": census.sha256_file(summary_path),
    }

    args = argparse.Namespace(
        output_dir=output,
        worker_task_id=None,
    )
    monkeypatch.setattr(census, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(census, "_load_inputs", lambda _args: (sources, []))
    monkeypatch.setattr(
        census,
        "build_run_payload",
        lambda _args, *, sources, selector_rows: payload,
    )
    monkeypatch.setattr(census, "bind_run_manifest", lambda _root, _payload: run)

    assert census.main([]) == 2

    valid_rows_bytes = rows_path.read_bytes()
    torn_rows_bytes = valid_rows_bytes + b'{"torn":'
    rows_path.write_bytes(torn_rows_bytes)
    with pytest.raises(json.JSONDecodeError):
        census.main([])
    assert rows_path.read_bytes() == torn_rows_bytes
    rows_path.write_bytes(valid_rows_bytes)

    Path(sources[0]["source_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="terminal census source binding drifted"):
        census.main([])

    Path(sources[0]["source_path"]).unlink()
    with pytest.raises(RuntimeError, match="terminal census source binding drifted"):
        census.main([])


def test_payload_names_unchanged_schema_v2_and_exact_negative_archive(monkeypatch, tmp_path):
    sources = [_source(cad_id) for cad_id in census.TARGET_CAD_IDS]
    selector = [
        {"cad_id": source["cad_id"], "strict_brep_valid": False}
        for source in sources
    ]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "utils.py").write_text("# runtime\n", encoding="utf-8")
    calibration = tmp_path / "cal.jsonl"
    selector_path = tmp_path / "selector.jsonl"
    selector_run = tmp_path / "selector-run.json"
    calibration.write_text("x", encoding="utf-8")
    selector_path.write_text("x", encoding="utf-8")
    selector_run.write_text("{}", encoding="utf-8")
    for index, source in enumerate(sources):
        path = tmp_path / f"source-{index}.pkl"
        path.write_bytes(b"x")
        source["source_path"] = str(path)
    args = argparse.Namespace(
        calibration_manifest=calibration, selector_matrix=selector_path,
        selector_run=selector_run, breparg_root=runtime,
        joint_iterations=200, worker_timeout_seconds=600.0,
        development_allow_dirty=True,
    )
    monkeypatch.setattr(census, "_git_identity", lambda *_a, **_k: {
        "commit": "a" * 40, "dirty": True, "formal": False,
        "status_sha256": _digest("status"),
    })
    monkeypatch.setattr(census, "_source_hashes", lambda _root: {"runner": _digest("runner")})
    monkeypatch.setattr(
        census, "_runtime_identity",
        lambda: copy.deepcopy(census.FROZEN_RUNTIME_IDENTITY),
    )
    original_sha256_file = census.sha256_file
    monkeypatch.setattr(
        census,
        "sha256_file",
        lambda path: (
            census.FROZEN_BREPARG_UTILS_SHA256
            if Path(path) == runtime / "utils.py"
            else census.FROZEN_CALIBRATION_MANIFEST_SHA256
            if Path(path) == calibration
            else census.FROZEN_SELECTOR_MATRIX_SHA256
            if Path(path) == selector_path
            else census.FROZEN_SELECTOR_RUN_SHA256
            if Path(path) == selector_run
            else original_sha256_file(path)
        ),
    )
    monkeypatch.setattr(census, "_selector_run_binding", lambda *_a, **_k: {
        "bytes": 1, "sha256": census.FROZEN_SELECTOR_RUN_SHA256,
        "signature": _digest("signature"),
        "status": "COMPLETED",
    })
    payload = census.build_run_payload(args, sources=sources, selector_rows=selector)
    assert payload["schema"] == census.RUN_SCHEMA
    assert payload["ordered_target_cad_ids"] == list(census.TARGET_CAD_IDS)
    assert [item["task_id"] for item in payload["ordered_tasks"]] == [
        task.task_id for task in census.TASKS
    ]
    assert payload["schema_v2"]["identity"] == "assembly-selector-geometry-gate-v2"
    assert payload["schema_v2"]["unchanged"] is True
    assert payload["exact_negative_evidence"] == census.EXACT_NEGATIVE_EVIDENCE
    assert payload["python"] == census.FROZEN_RUNTIME_IDENTITY["python"]
    assert payload["native_runtime"] == {
        key: census.FROZEN_RUNTIME_IDENTITY[key]
        for key in (
            "schema", "scope", "process_isolation", "numpy", "pythonocc",
            "occt",
        )
    }
