import numpy as np
import pytest

from tools.assembly_repair import (
    DEFAULT_PROFILES,
    RepairProfile,
    curve_fit_attempts,
    directed_face_loops,
    guarded_directed_face_loops,
    historical_face_loops,
    orient_ordered_loop,
    parse_profiles,
    sanitize_curve_points,
    validate_directed_loop,
)


def test_profiles_are_independent_and_combined():
    assert [profile.name for profile in DEFAULT_PROFILES] == [
        "baseline",
        "directed_trim",
        "curve_fit_fallback",
        "curve_fit_rescue",
        "wire_continuity",
        "single_solid",
        "pcurve_self_intersection",
        "local_intersection_topology",
        "local_pcurve_continuity",
        "directed_trim_curve_fit",
        "directed_trim_pcurve",
        "directed_trim_local_intersection_topology",
        "directed_trim_curve_rescue_local_intersection_topology",
        "directed_trim_local_pcurve_continuity",
        "combined",
    ]
    assert parse_profiles(["directed_trim"])[0].enabled("directed_trim")
    assert parse_profiles(["local_intersection_topology"])[0].enabled(
        "local_intersection_topology"
    )
    directed_local = parse_profiles(["directed_trim_local_intersection_topology"])[0]
    assert directed_local.enabled("directed_trim") is True
    assert directed_local.enabled("local_intersection_topology") is True
    directed_pcurve = parse_profiles(
        ["directed_trim_local_pcurve_continuity"]
    )[0]
    assert directed_pcurve.enabled("directed_trim") is True
    assert directed_pcurve.enabled("local_pcurve_continuity") is True
    combined = parse_profiles(["combined"])[0]
    assert combined.enabled("local_intersection_topology") is False
    assert combined.enabled("pcurve_self_intersection") is False
    with pytest.raises(ValueError, match="alternative OCC repair strategies"):
        RepairProfile(
            "ambiguous",
            ("local_intersection_topology", "local_pcurve_continuity"),
        )
    with pytest.raises(ValueError, match="alternative curve repair strategies"):
        RepairProfile(
            "ambiguous_curve_fit",
            ("curve_fit_fallback", "curve_fit_rescue"),
        )
    with pytest.raises(ValueError, match="unknown assembly repair"):
        RepairProfile("bad", ("magic",))


def test_directed_loops_orient_regular_edges():
    adjacency = np.asarray([[10, 11], [12, 11], [10, 12]], dtype=np.int64)
    loops = directed_face_loops([0, 1, 2], adjacency)
    assert loops == [[(0, False), (1, True), (2, True)]]
    validate_directed_loop(loops[0], adjacency)


def test_degenerate_closed_edges_are_separate_loops():
    adjacency = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    loops = directed_face_loops([0, 1, 2], adjacency)
    assert [(0, False)] in loops
    assert [(1, False), (2, False)] in loops
    for loop in loops:
        validate_directed_loop(loop, adjacency)


def test_historical_grouping_can_be_oriented_without_regrouping():
    adjacency = np.asarray([[0, 1], [2, 1], [2, 0]], dtype=np.int64)
    loops = historical_face_loops([0, 1, 2], adjacency)
    assert [[edge for edge, _ in loop] for loop in loops] == [[0, 1, 2]]
    oriented = orient_ordered_loop(loops[0], adjacency)
    assert oriented == [(0, False), (1, True), (2, False)]
    validate_directed_loop(oriented, adjacency)


def test_open_or_branching_topology_is_rejected():
    adjacency = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="open or branching"):
        directed_face_loops([0, 1], adjacency)


def test_guarded_directed_loop_falls_back_to_historical_on_unproven_topology():
    adjacency = np.asarray([[0, 1], [2, 3], [3, 4]], dtype=np.int64)
    historical = historical_face_loops([0, 1, 2], adjacency)

    loops, diagnostics = guarded_directed_face_loops([0, 1, 2], adjacency)

    assert loops == historical
    assert diagnostics["mode"] == "historical_fallback_unproven_topology"
    assert "cannot be oriented" in diagnostics["historical_orientation_error"]
    assert "open or branching" in diagnostics["regroup_error"]


def test_guarded_directed_loop_prefers_closed_orientation():
    adjacency = np.asarray([[0, 1], [2, 1], [2, 0]], dtype=np.int64)

    loops, diagnostics = guarded_directed_face_loops([0, 1, 2], adjacency)

    assert diagnostics == {"mode": "historical_order_oriented"}
    assert loops == [[(0, False), (1, True), (2, False)]]


def test_curve_sanitizer_removes_only_consecutive_duplicates():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    cleaned, stats = sanitize_curve_points(points)
    assert cleaned.tolist() == [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    ]
    assert stats == {"input_points": 4, "retained_points": 3}


def test_curve_fallback_is_bounded_and_prefers_lower_degree():
    attempts = curve_fit_attempts()
    assert attempts[:3] == ((3, 8, 5e-3), (2, 8, 5e-3), (1, 8, 5e-3))
    assert len(attempts) == 9
    assert max(tolerance for _, _, tolerance in attempts) == 5e-2
