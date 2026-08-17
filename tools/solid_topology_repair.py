"""Fail-closed near-vertex reconciliation for the non-unit-solid failure.

The P0-A non-unit-solid case is not fixed by wrapping the returned shell in a
different ``TopoDS_Solid``.  Its source topology contains endpoint ids that
are geometrically coincident within the assembly tolerance but are represented
as separate vertices.  This module only reconciles an unambiguous one-to-one
pair when both ids occur on the same face.  Every other input is returned
unchanged so the caller can keep the historical construction path.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_NEAR_VERTEX_TOLERANCE = 2e-4


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(int(size)))

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _vertex_representatives(
    edge_wcs: np.ndarray, edge_vertex_adj: np.ndarray
) -> dict[int, np.ndarray]:
    """Estimate one representative point for every source vertex id."""
    points: dict[int, list[np.ndarray]] = {}
    for edge_index, row in enumerate(np.asarray(edge_vertex_adj, dtype=np.int64)):
        if row.shape != (2,):
            raise ValueError(
                f"edge_vertex_adj row {edge_index} must have shape (2), got {row.shape}"
            )
        values = np.asarray(edge_wcs[int(edge_index)], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
            raise ValueError(f"edge_wcs row {edge_index} is not a [N,3] curve")
        if not np.isfinite(values).all():
            raise ValueError(f"edge_wcs row {edge_index} contains non-finite points")
        points.setdefault(int(row[0]), []).append(values[0])
        points.setdefault(int(row[1]), []).append(values[-1])
    return {vertex: np.median(values, axis=0) for vertex, values in points.items()}


def _face_vertex_ids(
    face_edge_adj: Sequence[Sequence[int]], edge_vertex_adj: np.ndarray
) -> list[set[int]]:
    result: list[set[int]] = []
    edges = np.asarray(edge_vertex_adj, dtype=np.int64)
    for face_index, incident in enumerate(face_edge_adj):
        ids: set[int] = set()
        for edge_id in incident:
            edge_id = int(edge_id)
            if edge_id < 0 or edge_id >= len(edges):
                raise ValueError(f"face {face_index} references invalid edge {edge_id}")
            ids.update(map(int, edges[edge_id]))
        result.append(ids)
    return result


def reconcile_near_vertices(
    edge_wcs: np.ndarray,
    edge_vertex_adj: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
    *,
    tolerance: float = DEFAULT_NEAR_VERTEX_TOLERANCE,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Return remapped adjacency and shared vertices for safe one-to-one gaps.

    A candidate pair is accepted only if it is the unique nearest candidate for
    each endpoint, the two ids occur on at least one common face, and their
    distance is strictly positive but no larger than ``tolerance``.  Clusters
    larger than two ids are rejected.  The original adjacency and an empty
    vertex map are returned when no pair is proven; this makes the operation a
    no-op for ordinary CADs.
    """
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("near-vertex tolerance must be finite and positive")
    edge_values = np.asarray(edge_wcs, dtype=np.float64)
    adjacency = np.asarray(edge_vertex_adj, dtype=np.int64)
    representatives = _vertex_representatives(edge_values, adjacency)
    face_vertices = _face_vertex_ids(face_edge_adj, adjacency)
    vertex_ids = sorted(representatives)
    common_face_pairs: set[tuple[int, int]] = set()
    for ids in face_vertices:
        ordered = sorted(ids)
        common_face_pairs.update(
            (left, right)
            for index, left in enumerate(ordered)
            for right in ordered[index + 1 :]
        )

    candidates: dict[int, list[tuple[float, int]]] = {vertex: [] for vertex in vertex_ids}
    for index, left in enumerate(vertex_ids):
        for right in vertex_ids[index + 1 :]:
            if (left, right) not in common_face_pairs:
                continue
            distance = float(np.linalg.norm(representatives[left] - representatives[right]))
            if 0.0 < distance <= float(tolerance):
                candidates[left].append((distance, right))
                candidates[right].append((distance, left))
    for values in candidates.values():
        values.sort(key=lambda item: (item[0], item[1]))

    mutual_pairs: list[tuple[int, int, float]] = []
    for left in vertex_ids:
        if len(candidates[left]) != 1:
            continue
        distance, right = candidates[left][0]
        if len(candidates[right]) == 1 and candidates[right][0][1] == left:
            mutual_pairs.append((min(left, right), max(left, right), distance))
    mutual_pairs = sorted(set(mutual_pairs))
    diagnostics: dict[str, Any] = {
        "schema": "solid-near-vertex-repair-v1",
        "tolerance": float(tolerance),
        "candidate_pair_count": int(sum(len(values) for values in candidates.values()) // 2),
        "candidate_pairs": [
            {"left": left, "right": right, "distance": distance}
            for left, right, distance in sorted(
                {
                    (min(left, right), max(left, right), distance)
                    for left, values in candidates.items()
                    for distance, right in values
                }
            )
        ],
        "mutual_pair_count": len(mutual_pairs),
        "merged_pairs": [
            {"left": left, "right": right, "distance": distance}
            for left, right, distance in mutual_pairs
        ],
        "applied": bool(mutual_pairs),
        "reason": "unambiguous_mutual_pairs" if mutual_pairs else "no_unambiguous_pair",
    }
    if not mutual_pairs:
        return adjacency.copy(), {}, diagnostics

    union_find = _UnionFind(max(vertex_ids) + 1)
    for left, right, _ in mutual_pairs:
        union_find.union(left, right)
    groups: dict[int, list[int]] = {}
    for vertex in vertex_ids:
        groups.setdefault(union_find.find(vertex), []).append(vertex)
    if any(len(group) > 2 for group in groups.values()):
        diagnostics.update(applied=False, reason="cluster_would_exceed_two_vertices", merged_pairs=[])
        return adjacency.copy(), {}, diagnostics

    remapped = adjacency.copy()
    shared_vertices: dict[int, np.ndarray] = {}
    for group in groups.values():
        root = min(group)
        point = np.mean([representatives[vertex] for vertex in group], axis=0)
        shared_vertices[root] = point
        for vertex in group:
            remapped[adjacency == vertex] = root
    diagnostics["clusters"] = [
        {"root": min(group), "members": sorted(group)}
        for group in groups.values()
        if len(group) > 1
    ]
    diagnostics["merged_vertex_count"] = int(sum(len(group) - 1 for group in groups.values()))
    return remapped, shared_vertices, diagnostics


__all__ = [
    "DEFAULT_NEAR_VERTEX_TOLERANCE",
    "reconcile_near_vertices",
]
