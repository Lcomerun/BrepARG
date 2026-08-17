import copy

import numpy as np
import pytest

from tools.assembly_selector_geometry import (
    GEOMETRY_GATE_CHECK_NAMES,
    MAX_EDGE_SAMPLE_MAX_NORMALIZED,
    geometry_topology_gate,
    input_geometry_signature,
    sample_input_edge_points,
    validate_accepted_geometry_gate,
)


def _input_arrays():
    surfaces = np.asarray(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        ]
    )
    edges = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    return surfaces, edges, [[0, 1]], np.asarray([[0, 1], [1, 2]])


def _candidate(signature):
    input_projection = {
        "sample_count": signature["projection_sample_count"],
        "projected_sample_count": signature["projection_sample_count"],
        "projection_failure_count": 0,
        "rms_normalized": 0.001,
        "max_normalized": 0.002,
    }
    candidate_projection = {
        "sample_count": signature["edge_count"] * 8,
        "projected_sample_count": signature["edge_count"] * 8,
        "projection_failure_count": 0,
        "rms_normalized": 0.001,
        "max_normalized": 0.002,
    }
    return {
        "face_count": signature["face_count"],
        "edge_count": signature["edge_count"],
        "vertex_count": signature["vertex_count"],
        "face_edge_occurrences": signature["face_edge_occurrences"],
        "face_edge_incidence_counts": list(signature["face_edge_incidence_counts"]),
        "edge_face_incidence_counts": list(signature["edge_face_incidence_counts"]),
        "vertex_edge_incidence_counts": list(
            signature["vertex_edge_incidence_counts"]
        ),
        "solid_count": 1,
        "free_edges": 0,
        "wire_order_failures": 0,
        "wire_self_intersections": 0,
        "bbox": list(signature["bbox"]),
        "edge_length": signature["edge_polyline_length"],
        "projectable_edge_count": signature["edge_count"],
        "unprojectable_edge_count": 0,
        "input_to_candidate_projection": input_projection,
        "candidate_to_input_projection": candidate_projection,
        "candidate_curve_sampling": {
            "requested_sample_count": signature["edge_count"] * 8,
            "successful_sample_count": signature["edge_count"] * 8,
            "sampling_failure_count": 0,
        },
    }


def test_input_signature_and_samples_are_deterministic():
    surfaces, edges, face_edges, corners = _input_arrays()

    first = input_geometry_signature(surfaces, edges, face_edges, corners)
    second = input_geometry_signature(surfaces, edges, face_edges, corners)

    assert first == second
    assert first["face_count"] == 1
    assert first["edge_count"] == 2
    assert first["vertex_count"] == 3
    assert first["projection_sample_count"] == 4
    assert sample_input_edge_points(edges).tolist() == edges.reshape(-1, 3).tolist()


@pytest.mark.parametrize(
    "mutate, expected_check",
    [
        (
            lambda candidate: candidate.update(
                face_count=2, face_edge_incidence_counts=[1, 1]
            ),
            "face_count_equal",
        ),
        (
            lambda candidate: candidate.update(
                edge_count=1,
                edge_face_incidence_counts=[2],
                projectable_edge_count=1,
                candidate_to_input_projection={
                    "sample_count": 8,
                    "projected_sample_count": 8,
                    "projection_failure_count": 0,
                    "rms_normalized": 0.001,
                    "max_normalized": 0.002,
                },
                candidate_curve_sampling={
                    "requested_sample_count": 8,
                    "successful_sample_count": 8,
                    "sampling_failure_count": 0,
                },
            ),
            "edge_count_equal",
        ),
        (
            lambda candidate: candidate.update(
                vertex_count=2, vertex_edge_incidence_counts=[1, 3]
            ),
            "vertex_count_equal",
        ),
        (
            lambda candidate: candidate.__setitem__("face_edge_occurrences", 1),
            "face_edge_occurrences_equal",
        ),
        (
            lambda candidate: candidate.__setitem__(
                "face_edge_incidence_counts", [1]
            ),
            "face_edge_incidence_equal",
        ),
        (
            lambda candidate: candidate.__setitem__(
                "edge_face_incidence_counts", [1, 2]
            ),
            "edge_face_incidence_equal",
        ),
        (
            lambda candidate: candidate.__setitem__(
                "vertex_edge_incidence_counts", [0, 2, 2]
            ),
            "vertex_edge_incidence_equal",
        ),
        (lambda candidate: candidate.__setitem__("solid_count", 0), "single_solid"),
        (lambda candidate: candidate.__setitem__("free_edges", 1), "no_free_edges"),
        (
            lambda candidate: candidate.__setitem__("wire_order_failures", 1),
            "no_wire_order_failures",
        ),
        (
            lambda candidate: candidate.__setitem__(
                "wire_self_intersections", 1
            ),
            "no_wire_self_intersections",
        ),
        (
            lambda candidate: candidate["input_to_candidate_projection"].__setitem__(
                "max_normalized", MAX_EDGE_SAMPLE_MAX_NORMALIZED * 1.01
            ),
            "input_to_candidate_max_within_tolerance",
        ),
        (
            lambda candidate: candidate["candidate_to_input_projection"].__setitem__(
                "projection_failure_count", 1
            ),
            "all_candidate_edge_samples_projected",
        ),
        (
            lambda candidate: candidate.__setitem__("unprojectable_edge_count", 1),
            "all_candidate_edges_projectable",
        ),
        (
            lambda candidate: candidate["candidate_curve_sampling"].__setitem__(
                "sampling_failure_count", 1
            ),
            "all_candidate_curve_samples_evaluated",
        ),
    ],
)
def test_geometry_gate_rejects_each_required_invariant(mutate, expected_check):
    signature = input_geometry_signature(*_input_arrays())
    candidate = _candidate(signature)
    mutate(candidate)

    result = geometry_topology_gate(signature, candidate)

    assert result["accepted"] is False
    assert result["checks"][expected_check] is False
    assert f"geometry_gate:{expected_check}" in result["rejection_reasons"]


def test_geometry_gate_accepts_matching_finite_signature():
    signature = input_geometry_signature(*_input_arrays())

    result = geometry_topology_gate(signature, _candidate(signature))

    assert result["accepted"] is True
    assert tuple(result["checks"]) == GEOMETRY_GATE_CHECK_NAMES
    assert all(result["checks"].values())
    assert validate_accepted_geometry_gate(result) == (True, [])


def test_geometry_gate_fails_closed_for_missing_or_nonfinite_data():
    signature = input_geometry_signature(*_input_arrays())
    missing = _candidate(signature)
    del missing["candidate_to_input_projection"]
    missing_result = geometry_topology_gate(signature, missing)
    assert missing_result["accepted"] is False
    assert "missing_signature_field:candidate:candidate_to_input_projection" in (
        missing_result["rejection_reasons"]
    )

    nonfinite = copy.deepcopy(_candidate(signature))
    nonfinite["edge_length"] = float("nan")
    nonfinite_result = geometry_topology_gate(signature, nonfinite)
    assert nonfinite_result["accepted"] is False
    assert nonfinite_result["checks"] == {"signature_well_formed": False}


def test_geometry_gate_rejects_same_global_counts_with_different_vertex_incidence():
    signature = input_geometry_signature(*_input_arrays())
    candidate = _candidate(signature)
    # Face, edge, and vertex counts all remain unchanged.  This captures the
    # topology hole that existed before vertex-edge incidence became a gate.
    candidate["vertex_edge_incidence_counts"] = [0, 2, 2]

    result = geometry_topology_gate(signature, candidate)

    assert result["accepted"] is False
    assert result["checks"]["vertex_edge_incidence_equal"] is False
    assert "geometry_gate:vertex_edge_incidence_equal" in result[
        "rejection_reasons"
    ]


def test_input_signature_rejects_bad_adjacency_and_nonfinite_geometry():
    surfaces, edges, face_edges, corners = _input_arrays()
    with pytest.raises(ValueError, match="invalid edge index"):
        input_geometry_signature(surfaces, edges, [[2]], corners)

    bad_edges = edges.copy()
    bad_edges[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        input_geometry_signature(surfaces, bad_edges, face_edges, corners)
