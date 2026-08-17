"""Conservative, independently switchable repairs for the local CAD assembler.

This module contains no upstream BrepARG source.  It provides topology and
geometry helpers used by the fixed 100-CAD repair matrix.  Every helper fails
closed on ambiguous input so a repair cannot silently change unrelated CADs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


REPAIR_SWITCHES = (
    "directed_trim",
    "curve_fit_fallback",
    "wire_continuity",
    "single_solid",
)


@dataclass(frozen=True)
class RepairProfile:
    name: str
    switches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.switches) - set(REPAIR_SWITCHES))
        if unknown:
            raise ValueError(f"unknown assembly repair switches: {unknown}")
        if len(set(self.switches)) != len(self.switches):
            raise ValueError("assembly repair switches must be unique")

    def enabled(self, name: str) -> bool:
        return name in self.switches


BASELINE_PROFILE = RepairProfile("baseline")
INDIVIDUAL_PROFILES = tuple(RepairProfile(name, (name,)) for name in REPAIR_SWITCHES)
COMBINED_PROFILE = RepairProfile("combined", REPAIR_SWITCHES)
DEFAULT_PROFILES = (BASELINE_PROFILE, *INDIVIDUAL_PROFILES, COMBINED_PROFILE)


def parse_profiles(values: Iterable[str] | None = None) -> tuple[RepairProfile, ...]:
    if values is None:
        return DEFAULT_PROFILES
    by_name = {profile.name: profile for profile in DEFAULT_PROFILES}
    selected = []
    for value in values:
        name = str(value).strip()
        if name not in by_name:
            raise ValueError(f"unknown repair profile {name!r}; expected {sorted(by_name)}")
        selected.append(by_name[name])
    if not selected or len({profile.name for profile in selected}) != len(selected):
        raise ValueError("repair profiles must be non-empty and unique")
    return tuple(selected)


def _edge_endpoints(edge_vertex_adj: np.ndarray, edge_id: int) -> tuple[int, int]:
    row = np.asarray(edge_vertex_adj, dtype=np.int64)[int(edge_id)]
    if row.shape != (2,):
        raise ValueError(f"edge {edge_id} does not have exactly two endpoint ids")
    return int(row[0]), int(row[1])


def directed_face_loops(
    face_edge_ids: Sequence[int], edge_vertex_adj: np.ndarray
) -> list[list[tuple[int, bool]]]:
    """Return deterministic closed walks as ``(global edge id, reversed)``.

    Non-degenerate edges are walked by vertex incidence.  Degenerate closed
    edges (same topological start and end vertex) are retained as one-edge
    loops instead of making the incidence graph branch.
    """
    edge_ids = [int(value) for value in face_edge_ids]
    if not edge_ids:
        raise ValueError("face has no incident edge")
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("face contains duplicate edge ids")

    loops: list[list[tuple[int, bool]]] = []
    regular: set[int] = set()
    for edge_id in edge_ids:
        start, end = _edge_endpoints(edge_vertex_adj, edge_id)
        if start == end:
            loops.append([(edge_id, False)])
        else:
            regular.add(edge_id)

    while regular:
        first = min(regular)
        start_vertex, current_vertex = _edge_endpoints(edge_vertex_adj, first)
        loop = [(first, False)]
        regular.remove(first)
        while current_vertex != start_vertex:
            candidates = [
                edge_id
                for edge_id in sorted(regular)
                if current_vertex in _edge_endpoints(edge_vertex_adj, edge_id)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"trim loop is open or branching at vertex {current_vertex}: {candidates}"
                )
            edge_id = candidates[0]
            vertex_a, vertex_b = _edge_endpoints(edge_vertex_adj, edge_id)
            reverse = vertex_b == current_vertex
            current_vertex = vertex_a if reverse else vertex_b
            loop.append((edge_id, reverse))
            regular.remove(edge_id)
            if len(loop) > len(edge_ids):
                raise ValueError("trim loop traversal exceeded incident edge count")
        loops.append(loop)
    return loops


def validate_directed_loop(
    loop: Sequence[tuple[int, bool]], edge_vertex_adj: np.ndarray
) -> None:
    if not loop:
        raise ValueError("wire loop is empty")
    oriented = []
    for edge_id, reverse in loop:
        start, end = _edge_endpoints(edge_vertex_adj, int(edge_id))
        oriented.append((end, start) if reverse else (start, end))
    for index, (_, end) in enumerate(oriented):
        next_start = oriented[(index + 1) % len(oriented)][0]
        if end != next_start:
            raise ValueError(
                f"wire endpoint discontinuity between positions {index} and {(index + 1) % len(oriented)}: "
                f"{end} != {next_start}"
            )


def sanitize_curve_points(
    points: np.ndarray, *, duplicate_tolerance: float = 1e-9
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove only consecutive duplicate points and retain endpoint order."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"curve points must have shape [N,3], got {values.shape}")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("curve points are insufficient or non-finite")
    keep = np.ones(len(values), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1) > duplicate_tolerance
    cleaned = values[keep]
    if len(cleaned) < 2:
        raise ValueError("curve collapses after duplicate-point removal")
    span = float(np.linalg.norm(np.max(cleaned, axis=0) - np.min(cleaned, axis=0)))
    if span <= duplicate_tolerance:
        raise ValueError("curve has negligible geometric span")
    return cleaned, {"input_points": len(values), "retained_points": len(cleaned)}


def curve_fit_attempts() -> tuple[tuple[int, int, float], ...]:
    """Bounded fallback order: lower degree before looser tolerance."""
    return (
        (3, 8, 5e-3),
        (2, 8, 5e-3),
        (1, 8, 5e-3),
        (3, 8, 8e-3),
        (2, 8, 8e-3),
        (1, 8, 8e-3),
        (3, 8, 5e-2),
        (2, 8, 5e-2),
        (1, 8, 5e-2),
    )


def loop_bbox_diagonal(
    loop: Sequence[tuple[int, bool]], edge_wcs: np.ndarray
) -> float:
    points = np.concatenate(
        [np.asarray(edge_wcs[int(edge_id)]).reshape(-1, 3) for edge_id, _ in loop]
    )
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
