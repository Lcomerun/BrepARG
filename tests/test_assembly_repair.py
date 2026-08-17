import numpy as np
import pytest

from tools.assembly_repair import (
    DEFAULT_PROFILES,
    RepairProfile,
    curve_fit_attempts,
    directed_face_loops,
    parse_profiles,
    sanitize_curve_points,
    validate_directed_loop,
)


def test_profiles_are_independent_and_combined():
    assert [profile.name for profile in DEFAULT_PROFILES] == [
        "baseline",
        "directed_trim",
        "curve_fit_fallback",
        "wire_continuity",
        "single_solid",
        "combined",
    ]
    assert parse_profiles(["directed_trim"])[0].enabled("directed_trim")
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


def test_open_or_branching_topology_is_rejected():
    adjacency = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="open or branching"):
        directed_face_loops([0, 1], adjacency)


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
