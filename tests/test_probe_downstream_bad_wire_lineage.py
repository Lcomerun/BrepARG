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
    StepGeometryIncidenceMatchingError,
    TARGET_CAD_IDS,
    WORKER_MARKER,
    _cyclic_curve_sample_distance,
    _edge_fingerprint_metrics,
    _edge_fingerprints_compatible,
    _matching_count_capped,
    _match_step_geometry_incidence,
    _match_step_vertex_incidence,
    _source_vertex_points_from_edge_endpoints,
    _step_observation,
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


def _vertex(identity, point):
    return {"observed_vertex": _Shape(f"vertex-{identity}"), "point": point}


def _mapped_edge(source_edge_id, source_vertex_ids, step_vertices):
    return {
        "source_edge_id": source_edge_id,
        "source_vertex_ids": source_vertex_ids,
        "observed_edge": _Shape(f"edge-{source_edge_id}"),
        "step_vertex_endpoints": step_vertices,
    }


def _triangle_vertex_proof_inputs():
    step_vertices = [
        _vertex("b", [1.0, 0.0, 0.0]),
        _vertex("c", [0.0, 1.0, 0.0]),
        _vertex("a", [0.0, 0.0, 0.0]),
    ]
    by_name = {row["observed_vertex"].identity[-1]: row for row in step_vertices}
    edges = [
        _mapped_edge(0, (0, 1), [by_name["a"], by_name["b"]]),
        _mapped_edge(1, (1, 2), [by_name["c"], by_name["b"]]),
        _mapped_edge(2, (2, 0), [by_name["a"], by_name["c"]]),
    ]
    source_points = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
    }
    return edges, step_vertices, source_points


def _line(start, end, sample_count=17):
    weights = np.linspace(0.0, 1.0, sample_count)[:, None]
    return (1.0 - weights) * np.asarray(start) + weights * np.asarray(end)


def _triangle_geometry_incidence_inputs():
    points = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 2.0, 0.0],
    }
    curves = [
        _line(points[0], points[1]),
        _line(points[1], points[2]),
        _line(points[2], points[0]),
    ]
    fingerprints = [_fingerprint(curve) for curve in curves]
    source = _synthetic_face(4, fingerprints, [0, 1, 2])
    for edge, labels in zip(
        source["wires"][0]["edges"], ((0, 1), (1, 2), (2, 0))
    ):
        edge["source_vertex_ids"] = labels

    step_vertices = [
        _vertex("b", points[1]),
        _vertex("c", points[2]),
        _vertex("a", points[0]),
    ]
    by_name = {row["observed_vertex"].identity[-1]: row for row in step_vertices}
    # STEP face and endpoint traversal orders are both deliberately unrelated
    # to the source order.  Fingerprints bind edges; the vertex proof must use
    # native identity and unordered endpoint pairs rather than those orders.
    step = _synthetic_face(
        0,
        [
            _fingerprint(curves[2][::-1]),
            _fingerprint(curves[0][::-1]),
            _fingerprint(curves[1][::-1]),
        ],
        ["step-2", "step-0", "step-1"],
        step=True,
    )
    for edge, endpoints in zip(
        step["wires"][0]["edges"],
        (
            [by_name["a"], by_name["c"]],
            [by_name["b"], by_name["a"]],
            [by_name["c"], by_name["b"]],
        ),
    ):
        edge["step_vertex_endpoints"] = endpoints
    return [source], [step], step_vertices, points


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


def test_source_vertex_points_use_ordered_edge_endpoints_and_json_scalars():
    curves = [
        _line([0, 0, 0], [1, 0, 0]),
        _line([1, 0, 0], [0, 2, 0]),
        _line([0, 2, 0], [0, 0, 0]),
    ]

    points = _source_vertex_points_from_edge_endpoints(
        curves,
        np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64),
        scale=3.0,
    )

    assert points == {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 2.0, 0.0],
    }
    assert json.loads(json.dumps(points, allow_nan=False)) == {
        "0": [0.0, 0.0, 0.0],
        "1": [1.0, 0.0, 0.0],
        "2": [0.0, 2.0, 0.0],
    }


def test_source_vertex_points_accept_full_diameter_at_tolerance():
    curves = [
        _line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _line([1.0001, 0.0, 0.0], [2.0, 0.0, 0.0]),
    ]

    points = _source_vertex_points_from_edge_endpoints(
        curves, ((0, 1), (1, 2)), scale=1.0
    )

    assert points[1] == pytest.approx([1.00005, 0.0, 0.0])


def test_source_vertex_points_reject_endpoint_spread_above_tolerance():
    curves = [
        _line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _line([1.0001001, 0.0, 0.0], [2.0, 0.0, 0.0]),
    ]

    with pytest.raises(ValueError, match="endpoint occurrences exceed tolerance"):
        _source_vertex_points_from_edge_endpoints(
            curves, ((0, 1), (1, 2)), scale=1.0
        )


def test_source_vertex_points_preserve_both_self_loop_endpoint_occurrences():
    points = _source_vertex_points_from_edge_endpoints(
        [_line([0.0, 0.0, 0.0], [0.00008, 0.0, 0.0])],
        ((0, 0),),
        scale=1.0,
    )

    assert points == {0: [0.00004, 0.0, 0.0]}


@pytest.mark.parametrize(
    "curves, adjacency",
    [
        ([], []),
        ([_line([0, 0, 0], [1, 0, 0])], []),
        ([np.asarray([0.0, 1.0, 2.0])], ((0, 1),)),
        ([np.zeros((1, 3))], ((0, 1),)),
        ([np.zeros((2, 2))], ((0, 1),)),
        ([np.asarray([[0.0, 0.0, 0.0], [math.nan, 0.0, 0.0]])], ((0, 1),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((0,),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((0, 1, 2),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((False, 1),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((0.0, 1),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((-1, 0),)),
        ([_line([0, 0, 0], [1, 0, 0])], ((1, 1),)),
    ],
)
def test_source_vertex_points_reject_malformed_populations(curves, adjacency):
    with pytest.raises(ValueError):
        _source_vertex_points_from_edge_endpoints(
            curves, adjacency, scale=1.0
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf])
def test_source_vertex_points_reject_invalid_scale(scale):
    with pytest.raises(ValueError):
        _source_vertex_points_from_edge_endpoints(
            [_line([0, 0, 0], [1, 0, 0])], ((0, 1),), scale=scale
        )


@pytest.mark.parametrize("tolerance", [-1.0, math.nan, math.inf])
def test_source_vertex_points_reject_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        _source_vertex_points_from_edge_endpoints(
            [_line([0, 0, 0], [1, 0, 0])],
            ((0, 1),),
            scale=1.0,
            tolerance=tolerance,
        )


def test_step_vertex_proof_accepts_one_global_direction_free_assignment():
    edges, step_vertices, source_points = _triangle_vertex_proof_inputs()

    proof = _match_step_vertex_incidence(
        edges,
        step_vertices,
        source_vertex_points=source_points,
        expected_source_edge_ids=(0, 1, 2),
        scale=2.0,
    )

    assert proof["status"] == "exact"
    assert proof["vertex_tolerance_normalized"] == 1e-4
    assert proof["vertex_candidate_degree_counts"] == {"1": 3}
    assert proof["vertex_matching_count_capped"] == 1
    assert proof["edge_endpoint_pair_expected_count"] == 3
    assert proof["edge_endpoint_pair_proof_count"] == 3
    assert proof["edge_endpoint_occurrence_expected_count"] == 6
    assert proof["edge_endpoint_occurrence_proof_count"] == 6
    assert proof["self_loop_endpoint_pair_expected_count"] == 0
    assert proof["self_loop_endpoint_pair_proof_count"] == 0
    assert "observed_vertex" not in json.dumps(proof, sort_keys=True)


def test_step_vertex_proof_checks_repeated_edge_endpoint_pairs_direction_free():
    edges, step_vertices, source_points = _triangle_vertex_proof_inputs()
    reverse_occurrence = copy.deepcopy(edges[0])
    reverse_occurrence["observed_edge"] = edges[0]["observed_edge"]
    reverse_occurrence["source_vertex_ids"] = (1, 0)
    reverse_occurrence["step_vertex_endpoints"] = list(
        reversed(edges[0]["step_vertex_endpoints"])
    )

    proof = _match_step_vertex_incidence(
        [*edges, reverse_occurrence],
        step_vertices,
        source_vertex_points=source_points,
        expected_source_edge_ids=(0, 1, 2),
        scale=2.0,
    )

    assert proof["status"] == "exact"
    assert proof["mapped_source_edge_count"] == 3
    assert proof["edge_endpoint_occurrence_proof_count"] == 6


def test_step_vertex_proof_rejects_inconsistent_repeated_edge_endpoint_pair():
    edges, step_vertices, source_points = _triangle_vertex_proof_inputs()
    duplicate = copy.deepcopy(edges[0])
    duplicate["observed_edge"] = edges[0]["observed_edge"]
    duplicate["step_vertex_endpoints"] = [
        edges[0]["step_vertex_endpoints"][0],
        edges[2]["step_vertex_endpoints"][1],
    ]

    proof = _match_step_vertex_incidence(
        [*edges, duplicate],
        step_vertices,
        source_vertex_points=source_points,
        expected_source_edge_ids=(0, 1, 2),
        scale=2.0,
    )

    assert proof["status"] == "unavailable"
    assert proof["failure_codes"] == [
        "source_edge_0_step_endpoints_inconsistent"
    ]


def test_step_vertex_proof_preserves_self_loop_endpoint_multiplicity_two():
    vertex = _vertex("a", [0.0, 0.0, 0.0])
    edge = _mapped_edge(4, (7, 7), [vertex, vertex])

    proof = _match_step_vertex_incidence(
        [edge],
        [vertex],
        source_vertex_points={7: [0.0, 0.0, 0.0]},
        expected_source_edge_ids=(4,),
        scale=1.0,
    )

    assert proof["status"] == "exact"
    assert proof["self_loop_endpoint_pair_expected_count"] == 1
    assert proof["self_loop_endpoint_pair_proof_count"] == 1
    assert proof["edge_endpoint_occurrence_proof_count"] == 2


@pytest.mark.parametrize(
    "mutate, failure_code",
    [
        (
            lambda edges, vertices, points: edges.pop(),
            "mapped_source_edge_census_incomplete",
        ),
        (
            lambda edges, vertices, points: edges[0].__setitem__(
                "step_vertex_endpoints",
                [edges[0]["step_vertex_endpoints"][0], vertices[1]],
            ),
            "vertex_assignment_has_no_perfect_matching",
        ),
        (
            lambda edges, vertices, points: points[0].__setitem__(0, math.nan),
            "source_vertex_points_missing_or_malformed",
        ),
        (
            lambda edges, vertices, points: vertices[0].__setitem__(
                "point", [math.inf, 0.0, 0.0]
            ),
            "step_vertices_missing_or_malformed",
        ),
        (
            lambda edges, vertices, points: points[0].__setitem__(
                0, 2.0e-4
            ),
            "vertex_assignment_has_no_perfect_matching",
        ),
    ],
)
def test_step_vertex_proof_rejects_missing_reconnected_nonfinite_or_distant(
    mutate, failure_code
):
    edges, step_vertices, source_points = _triangle_vertex_proof_inputs()
    mutate(edges, step_vertices, source_points)

    proof = _match_step_vertex_incidence(
        edges,
        step_vertices,
        source_vertex_points=source_points,
        expected_source_edge_ids=(0, 1, 2),
        scale=1.0,
    )

    assert proof["status"] == "unavailable"
    assert proof["failure_codes"] == [failure_code]


def test_step_vertex_proof_rejects_multiple_global_matchings():
    first = _vertex("a", [0.0, 0.0, 0.0])
    second = _vertex("b", [0.0, 0.0, 0.0])
    # Two parallel mapped edges give both source and STEP vertices identical
    # incidence multisets. Coincident points deliberately leave two bijections.
    edges = [
        _mapped_edge(0, (0, 1), [first, second]),
        _mapped_edge(1, (0, 1), [second, first]),
    ]

    proof = _match_step_vertex_incidence(
        edges,
        [first, second],
        source_vertex_points={0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]},
        expected_source_edge_ids=(0, 1),
        scale=1.0,
    )

    assert proof["status"] == "ambiguous"
    assert proof["failure_codes"] == ["vertex_assignment_not_unique"]
    assert proof["vertex_candidate_degree_counts"] == {"2": 2}
    assert proof["vertex_matching_count_capped"] == 2


def test_geometry_incidence_exact_requires_vertex_proof_when_opted_in():
    horizontal = _fingerprint(
        np.stack((np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1)
    )
    source = [_synthetic_face(0, [horizontal], [0])]
    step = [_synthetic_face(0, [horizontal], [10], step=True)]

    result = _match_step_geometry_incidence(
        source,
        step,
        scale=1.0,
        require_vertex_proof=True,
    )

    assert result["status"] == "unavailable"
    assert result["failure_codes"] == ["source_edge_0_endpoint_labels_missing"]
    assert result["vertex_proof_required"] is True


def test_geometry_incidence_opt_in_face_count_failure_still_declares_vertex_gate():
    result = _match_step_geometry_incidence(
        [], [], scale=1.0, require_vertex_proof=True
    )

    assert result["status"] == "unavailable"
    assert result["failure_codes"] == ["source_step_face_count_mismatch"]
    assert result["face_rows"] == []
    assert result["vertex_proof_required"] is True
    assert result["vertex_proof_status"] == "not_evaluated"


def test_geometry_incidence_opt_in_exact_requires_complete_global_vertex_proof():
    source, step, step_vertices, source_points = (
        _triangle_geometry_incidence_inputs()
    )

    result = _match_step_geometry_incidence(
        source,
        step,
        scale=3.0,
        step_vertices=step_vertices,
        source_vertex_points=source_points,
        require_vertex_proof=True,
    )

    assert result["status"] == "exact_geometry_incidence"
    assert result["failure_codes"] == []
    assert result["face_matching_count_capped"] == 1
    assert result["mapped_face_count"] == 1
    assert result["mapped_edge_occurrence_count"] == 3
    assert result["vertex_proof_required"] is True
    assert result["vertex_proof_status"] == "exact"
    assert result["vertex_matching_count_capped"] == 1
    assert result["source_vertex_count"] == result["step_vertex_count"] == 3
    assert result["mapped_source_edge_count"] == 3
    assert result["edge_endpoint_pair_proof_count"] == 3
    assert (
        result["edge_endpoint_pair_proof_count"]
        == result["edge_endpoint_pair_expected_count"]
    )
    assert result["edge_endpoint_occurrence_proof_count"] == 6
    assert (
        result["edge_endpoint_occurrence_proof_count"]
        == result["edge_endpoint_occurrence_expected_count"]
    )
    public_proof = {
        key: value for key, value in result.items() if key != "face_rows"
    }
    encoded = json.dumps(public_proof, allow_nan=False, sort_keys=True)
    assert "observed_vertex" not in encoded
    assert not {
        "observed_edge",
        "observed_vertex",
        "point",
        "source_vertex_points",
        "step_vertex_endpoints",
        "step_vertices",
    } & set(public_proof)
    assert "\\\\" not in encoded
    assert ":\\" not in encoded
    assert_path_free_evidence(public_proof)


def test_geometry_incidence_legacy_result_shape_remains_backward_compatible():
    horizontal = _fingerprint(
        np.stack((np.linspace(0, 1, 17), np.zeros(17), np.zeros(17)), axis=1)
    )

    result = _match_step_geometry_incidence(
        [_synthetic_face(0, [horizontal], [0])],
        [_synthetic_face(0, [horizontal], [10], step=True)],
        scale=1.0,
    )

    assert result["status"] == "exact_geometry_incidence"
    assert result["vertex_proof_required"] is False
    assert result["vertex_proof_status"] == "legacy_not_required"
    assert "vertex_proof_method" not in result
    assert "vertex_matching_count_capped" not in result


@pytest.mark.parametrize("missing", ["step_vertices", "source_vertex_points"])
def test_geometry_incidence_vertex_proof_inputs_fail_closed_independently(missing):
    source, step, step_vertices, source_points = (
        _triangle_geometry_incidence_inputs()
    )
    kwargs = {
        "step_vertices": step_vertices,
        "source_vertex_points": source_points,
        "require_vertex_proof": True,
    }
    kwargs[missing] = None

    result = _match_step_geometry_incidence(
        source, step, scale=3.0, **kwargs
    )

    assert result["status"] == "unavailable"
    assert result["vertex_proof_required"] is True
    assert result["failure_codes"] == [
        (
            "step_vertices_missing_or_malformed"
            if missing == "step_vertices"
            else "source_vertex_points_missing_or_malformed"
        )
    ]


def test_step_observation_derives_source_points_with_reimport_scale(
    monkeypatch, tmp_path
):
    import tools.probe_downstream_bad_wire_lineage as lineage
    import tools.diagnose_assembly_face_wires as diagnosis_module

    source, step, step_vertices, _source_points = (
        _triangle_geometry_incidence_inputs()
    )
    source_face = source[0]
    step_face = step[0]
    curves = [
        _line([0, 0, 0], [1, 0, 0]),
        _line([1, 0, 0], [0, 2, 0]),
        _line([0, 2, 0], [0, 0, 0]),
    ]
    captured = {}

    monkeypatch.setattr(
        diagnosis_module,
        "diagnose_step_face_wires_v2",
        lambda *_args, **_kwargs: {
            "status": "diagnosed",
            "edge_position_basis": "occ_1_based",
            "faces": [],
            "wires": [],
            "occurrences": [],
            "occurrence_kinds": [],
        },
    )
    monkeypatch.setattr(
        diagnosis_module,
        "diagnose_face_wires_v2",
        lambda *_args, **_kwargs: {
            "faces": [], "wires": [], "occurrences": []
        },
    )
    monkeypatch.setattr(lineage, "_read_step_faces", lambda _path: ("shape", ["face"]))
    monkeypatch.setattr(
        lineage,
        "_occ_shape_bbox",
        lambda _shape: np.asarray([0, 0, 0, 3, 4, 0], dtype=np.float64),
    )
    monkeypatch.setattr(
        lineage, "_occ_step_vertex_signatures", lambda _shape: step_vertices
    )
    monkeypatch.setattr(
        lineage, "_occ_step_face_signature", lambda *_args, **_kwargs: step_face
    )
    monkeypatch.setattr(
        lineage,
        "_source_face_signature_from_observer",
        lambda *_args, **_kwargs: source_face,
    )
    original = lineage._source_vertex_points_from_edge_endpoints

    def capture_source_points(edge_wcs, edge_vertex_adj, *, scale, tolerance=1e-4):
        captured["scale"] = scale
        return original(
            edge_wcs, edge_vertex_adj, scale=scale, tolerance=tolerance
        )

    monkeypatch.setattr(
        lineage,
        "_source_vertex_points_from_edge_endpoints",
        capture_source_points,
    )

    observation = _step_observation(
        tmp_path / "roundtrip.step",
        breparg_root=tmp_path,
        source_face_references={
            0: {"face": object(), "source_mapping": {}}
        },
        face_edge_adj=[[0, 1, 2]],
        edge_vertex_adj=((0, 1), (1, 2), (2, 0)),
        source_edge_wcs=curves,
        require_vertex_proof=True,
    )

    assert captured["scale"] == 5.0
    assert observation["lineage_status"] == "exact_geometry_incidence"
    proof = observation["diagnosis"]["geometry_incidence_proof"]
    assert proof["vertex_proof_status"] == "exact"
    assert proof["vertex_proof_required"] is True


def test_step_observation_rejects_conflicting_source_point_inputs(
    monkeypatch, tmp_path
):
    import tools.probe_downstream_bad_wire_lineage as lineage
    import tools.diagnose_assembly_face_wires as diagnosis_module

    monkeypatch.setattr(
        diagnosis_module,
        "diagnose_step_face_wires_v2",
        lambda *_args, **_kwargs: {
            "status": "diagnosed", "occurrences": []
        },
    )
    monkeypatch.setattr(
        lineage, "_read_step_faces", lambda _path: ("shape", ["face"])
    )
    monkeypatch.setattr(
        lineage,
        "_occ_shape_bbox",
        lambda _shape: np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float64),
    )

    observation = _step_observation(
        tmp_path / "roundtrip.step",
        breparg_root=tmp_path,
        source_face_references={0: {"face": object(), "source_mapping": {}}},
        face_edge_adj=[[0]],
        edge_vertex_adj=((0, 1),),
        source_edge_wcs=[_line([0, 0, 0], [1, 0, 0])],
        source_vertex_points={0: [0, 0, 0], 1: [1, 0, 0]},
        require_vertex_proof=True,
    )

    assert observation["lineage_status"] == "unavailable"
    assert observation["mapping_failures"] == [
        "step_geometry_incidence_matching_failed:ValueError"
    ]


def test_formal_step_observation_escalates_internal_matching_exception(
    monkeypatch, tmp_path
):
    """A formal caller must not archive an internal exception as geometry."""

    import tools.probe_downstream_bad_wire_lineage as lineage
    import tools.diagnose_assembly_face_wires as diagnosis_module

    monkeypatch.setattr(
        diagnosis_module,
        "diagnose_step_face_wires_v2",
        lambda *_args, **_kwargs: {"status": "diagnosed", "occurrences": []},
    )
    monkeypatch.setattr(
        lineage, "_read_step_faces", lambda _path: ("shape", ["face"])
    )
    monkeypatch.setattr(
        lineage,
        "_occ_shape_bbox",
        lambda _shape: np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float64),
    )

    with pytest.raises(
        StepGeometryIncidenceMatchingError,
        match="step_geometry_incidence_matching_failed:ValueError",
    ):
        _step_observation(
            tmp_path / "roundtrip.step",
            breparg_root=tmp_path,
            source_face_references={0: {"face": object(), "source_mapping": {}}},
            face_edge_adj=[[0]],
            edge_vertex_adj=((0, 1),),
            source_edge_wcs=[_line([0, 0, 0], [1, 0, 0])],
            source_vertex_points={0: [0, 0, 0], 1: [1, 0, 0]},
            require_vertex_proof=True,
            fail_on_matching_exception=True,
        )


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
