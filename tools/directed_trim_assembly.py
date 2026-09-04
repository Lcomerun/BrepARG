"""BRep construction pilot with topology-directed trim loops.

This module intentionally lives outside the upstream ``BrepARG`` tree.  It
keeps the original fitting and tolerance choices while fixing two trim-loop
semantics: local face-edge positions are mapped to global edge ids for outer
loop selection, and edges are oriented along the vertex walk before insertion
into an OCC wire.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        guarded_directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        sanitize_curve_points,
        validate_directed_loop,
    )
    from .solid_topology_repair import reconcile_near_vertices
except ImportError:  # direct script execution
    from assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        guarded_directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        sanitize_curve_points,
        validate_directed_loop,
    )
    from solid_topology_repair import reconcile_near_vertices


def effective_topology_summary(edge_vertex_adj: Any) -> dict[str, Any]:
    """Return path-free counts for the adjacency actually used to build edges.

    Near-vertex reconciliation can remap endpoint ids before OCC edge creation.
    The selector must compare the candidate STEP with that effective topology,
    not with the source pickle's pre-reconciliation vertex ids.
    """
    adjacency = np.asarray(edge_vertex_adj, dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape[1:] != (2,):
        raise ValueError("effective edge_vertex_adj must have shape (edge_count, 2)")
    if len(adjacency) == 0 or np.any(adjacency < 0):
        raise ValueError("effective edge_vertex_adj must contain nonnegative endpoints")
    _, counts = np.unique(adjacency.reshape(-1), return_counts=True)
    return {
        "edge_count": int(len(adjacency)),
        "vertex_count": int(len(counts)),
        "vertex_edge_incidence_counts": sorted(int(value) for value in counts),
    }


def _unique_sewing_history_target(
    sewing: Any,
    source_shape: Any,
    output_shapes: Sequence[Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Return one history-proven output shape, or an explicit failure.

    Output sequence positions are used only to group identity matches inside
    this call.  They are never treated as source/output correspondence.
    """
    attempts = []
    proven_indices: dict[str, int] = {}
    for method_name in ("ModifiedSubShape", "Modified"):
        method = getattr(sewing, method_name, None)
        if method is None:
            attempts.append(
                {"method": method_name, "status": "method_unavailable"}
            )
            continue
        try:
            candidate = method(source_shape)
            if bool(candidate.IsNull()):
                attempts.append(
                    {"method": method_name, "status": "null_candidate"}
                )
                continue
            matches = [
                index
                for index, output_shape in enumerate(output_shapes)
                if bool(candidate.IsSame(output_shape))
            ]
        except Exception as exc:
            attempts.append(
                {
                    "method": method_name,
                    "status": "history_measurement_failed",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        attempts.append(
            {
                "method": method_name,
                "status": (
                    "unique_output_identity"
                    if len(matches) == 1
                    else "output_identity_not_unique"
                ),
                "identity_match_count": len(matches),
            }
        )
        if len(matches) == 1:
            proven_indices[method_name] = matches[0]

    required_methods = {"ModifiedSubShape", "Modified"}
    if set(proven_indices) != required_methods:
        return None, {
            "status": "mapping_failed",
            "failure_codes": ["sewing_history_methods_not_both_unique"],
            "attempts": attempts,
        }
    target_indices = set(proven_indices.values())
    if len(target_indices) != 1:
        return None, {
            "status": "mapping_failed",
            "failure_codes": ["sewing_history_methods_disagree"],
            "attempts": attempts,
        }
    target_index = next(iter(target_indices))
    return output_shapes[target_index], {
        "status": "mapped",
        "failure_codes": [],
        "mapping_methods": sorted(required_methods),
        "attempts": attempts,
    }


def _invalidate_colliding_sewing_targets(
    lineage_rows: list[dict[str, Any]],
) -> None:
    """Fail closed on merged/unmeasurable targets without cascading failure."""
    collided_indices: set[int] = set()
    collision_failure_codes: dict[int, set[str]] = {
        index: set() for index in range(len(lineage_rows))
    }
    for row_index, row in enumerate(lineage_rows):
        if row["status"] != "mapped":
            continue
        for other_index, other_row in enumerate(
            lineage_rows[row_index + 1 :], row_index + 1
        ):
            if other_row["status"] != "mapped":
                continue
            try:
                same_target = bool(row["shape"].IsSame(other_row["shape"]))
            except Exception:
                failure_code = "cross_source_identity_measurement_failed"
                collided_indices.update((row_index, other_index))
                collision_failure_codes[row_index].add(failure_code)
                collision_failure_codes[other_index].add(failure_code)
                continue
            if same_target:
                failure_code = "distinct_source_faces_merged"
                collided_indices.update((row_index, other_index))
                collision_failure_codes[row_index].add(failure_code)
                collision_failure_codes[other_index].add(failure_code)

    # Mutate only after every comparison. Otherwise replacing one collided
    # shape with None could poison later, unrelated identity comparisons.
    for row_index in sorted(collided_indices):
        collided_row = lineage_rows[row_index]
        collided_row["status"] = "mapping_failed"
        collided_row["shape"] = None
        collided_row["failure_codes"].extend(
            sorted(collision_failure_codes[row_index])
        )


def _edge_curve_fingerprint(edge: Any, *, sample_count: int = 11) -> dict[str, Any]:
    """Sample an orientation-invariant 3D fingerprint for one OCC edge."""
    from OCC.Core.BRep import BRep_Tool

    try:
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is None:
            return {"available": False, "reason": "curve_unavailable"}
        parameters = np.linspace(
            float(first), float(last), int(sample_count), dtype=np.float64
        )
        samples = np.asarray(
            [
                (
                    float(curve.Value(float(parameter)).X()),
                    float(curve.Value(float(parameter)).Y()),
                    float(curve.Value(float(parameter)).Z()),
                )
                for parameter in parameters
            ],
            dtype=np.float64,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"curve_sampling_failed:{type(exc).__name__}",
        }
    if samples.shape != (sample_count, 3) or not np.isfinite(samples).all():
        return {"available": False, "reason": "curve_samples_nonfinite"}
    segment_lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    return {
        "available": True,
        "curve_type": str(curve.DynamicType().Name()),
        "samples": samples,
        "bbox_min": np.min(samples, axis=0),
        "bbox_max": np.max(samples, axis=0),
        "sampled_length": float(np.sum(segment_lengths)),
    }


def _curve_fingerprints_match(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Return true only for a tight forward/reverse 3D curve match."""
    if not first.get("available") or not second.get("available"):
        return False
    if first.get("curve_type") != second.get("curve_type"):
        return False
    first_samples = np.asarray(first["samples"], dtype=np.float64)
    second_samples = np.asarray(second["samples"], dtype=np.float64)
    if first_samples.shape != second_samples.shape:
        return False
    scale = max(
        float(first.get("sampled_length", 0.0)),
        float(second.get("sampled_length", 0.0)),
        float(np.linalg.norm(np.ptp(first_samples, axis=0))),
        float(np.linalg.norm(np.ptp(second_samples, axis=0))),
        1.0,
    )
    tolerance = 1e-7 * scale + 1e-10
    if abs(
        float(first.get("sampled_length", 0.0))
        - float(second.get("sampled_length", 0.0))
    ) > tolerance:
        return False
    for key in ("bbox_min", "bbox_max"):
        if float(
            np.max(
                np.abs(
                    np.asarray(first[key], dtype=np.float64)
                    - np.asarray(second[key], dtype=np.float64)
                )
            )
        ) > tolerance:
            return False
    forward = float(np.max(np.linalg.norm(first_samples - second_samples, axis=1)))
    reverse = float(
        np.max(np.linalg.norm(first_samples - second_samples[::-1], axis=1))
    )
    return min(forward, reverse) <= tolerance


def _unique_perfect_assignment(
    compatibility: Sequence[Sequence[int]], candidate_count: int
) -> list[int] | None:
    """Return the sole perfect assignment, or None for zero/multiple solutions."""
    solutions = _perfect_assignments_up_to_two(compatibility, candidate_count)
    return solutions[0] if len(solutions) == 1 else None


def _perfect_assignments_up_to_two(
    compatibility: Sequence[Sequence[int]], candidate_count: int
) -> list[list[int]]:
    """Return at most two perfect assignments for fail-closed uniqueness tests.

    Two solutions are enough to reject an identity claim.  Capping the search
    avoids enumerating every automorphism of a symmetric topology while still
    distinguishing no solution, one unique solution, and more than one.
    """
    if len(compatibility) != int(candidate_count):
        return []
    normalized = [sorted(set(int(value) for value in row)) for row in compatibility]
    if any(not row for row in normalized):
        return []
    if any(value < 0 or value >= candidate_count for row in normalized for value in row):
        return []
    order = sorted(range(len(normalized)), key=lambda index: len(normalized[index]))
    solutions: list[list[int]] = []
    assignment = [-1] * len(normalized)

    def search(depth: int, used: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if depth == len(order):
            solutions.append(list(assignment))
            return
        observed_index = order[depth]
        for candidate_index in normalized[observed_index]:
            if candidate_index in used:
                continue
            assignment[observed_index] = candidate_index
            search(depth + 1, {*used, candidate_index})
            assignment[observed_index] = -1

    search(0, set())
    return solutions


def _edge_endpoint_handles(edge: Any) -> tuple[Any, Any]:
    """Return two borrowed OCC endpoint occurrences without serializing them."""

    from OCC.Extend.TopologyUtils import TopologyExplorer

    endpoints = list(
        TopologyExplorer(edge, ignore_orientation=False).vertices()
    )
    if len(endpoints) != 2:
        raise ValueError("edge_endpoint_occurrence_count_not_two")
    return endpoints[0], endpoints[1]


def _shape_vertex_handles(shape: Any) -> list[Any]:
    """Return borrowed target vertices; identity reduction happens separately."""

    from OCC.Extend.TopologyUtils import TopologyExplorer

    return list(TopologyExplorer(shape, ignore_orientation=True).vertices())


def _same_occ_identity(first: Any, second: Any) -> bool:
    """Measure OCC topological identity and label native failures uniformly."""

    try:
        return bool(first.IsSame(second))
    except Exception as exc:
        raise RuntimeError("occ_vertex_identity_measurement_failed") from exc


def _unique_identity_representatives(values: Sequence[Any]) -> list[Any]:
    """Collapse native handles into direction-independent ``IsSame`` classes."""

    representatives: list[Any] = []
    for value in values:
        matches = [
            index
            for index, representative in enumerate(representatives)
            if _same_occ_identity(value, representative)
        ]
        if len(matches) > 1:
            # Representatives are already pairwise distinct, so reaching two
            # of them means IsSame itself is not behaving as an equivalence
            # relation for this measurement.  Do not pick an explorer index.
            raise RuntimeError("occ_vertex_identity_class_not_unique")
        if not matches:
            representatives.append(value)
    return representatives


def _identity_class_index(value: Any, representatives: Sequence[Any]) -> int:
    """Return the sole target identity class for one borrowed vertex handle."""

    matches = [
        index
        for index, representative in enumerate(representatives)
        if _same_occ_identity(value, representative)
    ]
    if len(matches) != 1:
        raise RuntimeError("endpoint_target_vertex_identity_not_unique")
    return int(matches[0])


def _source_vertex_lineage_failure(
    *,
    proof_method: str,
    source_vertex_count: int,
    observed_vertex_count: int,
    constraint_occurrence_count: int,
    failure_codes: Sequence[str],
    solution_count: int | None = None,
) -> dict[str, Any]:
    """Create one path-free scalar failure result for an OCC child process."""

    return {
        "status": "ambiguous",
        "proof_method": str(proof_method),
        "solution_count": solution_count,
        "solution_count_capped_at_two": True,
        "source_vertex_count": int(source_vertex_count),
        "observed_vertex_count": int(observed_vertex_count),
        "mapped_source_vertex_count": 0,
        "mapped_observed_vertex_count": 0,
        "max_observed_per_source": 0,
        "max_source_per_observed": 0,
        "constraint_occurrence_count": int(constraint_occurrence_count),
        "failure_codes": list(dict.fromkeys(map(str, failure_codes))),
    }


def _prove_stage_local_occ_topology(
    scopes: Sequence[Sequence[tuple[int, Sequence[int], Any]]],
    *,
    scope_kind: str,
) -> dict[str, Any]:
    """Prove endpoint-label consistency only within each owning OCC scope.

    A scope is one independently built edge at S2 or one face at S3/S4.  OCC
    may legally copy vertex handles between such scopes, so this function does
    not compare identities across scopes or against an earlier stage.  Inside
    a scope, however, equal source labels must resolve to one OCC identity and
    distinct labels must not merge.  This detects a real local split or merge
    without inventing a CAD-global source-vertex bijection.
    """

    proof_method = "source_edge_endpoint_labels_to_stage_local_occ_identity_classes_v1"
    failures: list[str] = []
    source_edge_ids: set[int] = set()
    constraint_count = 0
    for scope_index, occurrences in enumerate(scopes):
        representatives: list[Any] = []
        label_to_classes: dict[int, set[int]] = {}
        class_to_labels: dict[int, set[int]] = {}
        for occurrence_index, occurrence in enumerate(occurrences):
            try:
                edge_id, labels, edge = occurrence
                edge_id = int(edge_id)
                labels = tuple(map(int, labels))
                if len(labels) != 2:
                    raise ValueError
                endpoints = _edge_endpoint_handles(edge)
            except Exception as exc:
                failures.append(
                    f"scope_{scope_index}_occurrence_{occurrence_index}_malformed:"
                    f"{type(exc).__name__}"
                )
                continue
            source_edge_ids.add(edge_id)
            constraint_count += 1
            for label, endpoint in zip(labels, endpoints):
                matches = [
                    index for index, representative in enumerate(representatives)
                    if _same_occ_identity(endpoint, representative)
                ]
                if len(matches) > 1:
                    failures.append(f"scope_{scope_index}_identity_class_nonunique")
                    continue
                if matches:
                    class_id = matches[0]
                else:
                    class_id = len(representatives)
                    representatives.append(endpoint)
                label_to_classes.setdefault(label, set()).add(class_id)
                class_to_labels.setdefault(class_id, set()).add(label)
        if any(len(values) > 1 for values in label_to_classes.values()):
            failures.append(f"scope_{scope_index}_source_vertex_split")
        if any(len(values) > 1 for values in class_to_labels.values()):
            failures.append(f"scope_{scope_index}_source_vertices_merged")
    return {
        "status": (
            "exact_stage_local_topology" if not failures else "ambiguous"
        ),
        "proof_method": proof_method,
        "scope_kind": str(scope_kind),
        "scope_count": int(len(scopes)),
        "source_edge_count": int(len(source_edge_ids)),
        "constraint_occurrence_count": int(constraint_count),
        "max_observed_per_source_within_scope": 1 if not failures else 0,
        "max_source_per_observed_within_scope": 1 if not failures else 0,
        "failure_codes": list(dict.fromkeys(failures)),
    }


def _prove_global_source_vertex_lineage(
    observed_shape: Any,
    source_face_bindings: Sequence[Mapping[str, Any]],
    source_face_edge_ids: Sequence[Sequence[int]],
    source_edge_vertex_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Prove a unique source-vertex bijection after sewing or solid creation.

    Each source edge is already identity-bound by the face lineage.  Its two
    source endpoint labels and its two observed OCC endpoint handles are both
    treated as unordered pairs, so reversing an edge cannot change the proof.
    Intersections across every incidence produce a bipartite compatibility
    graph from source vertices to vertices actually present in
    ``observed_shape``.  Only one perfect assignment, followed by a replay of
    every endpoint-pair constraint, establishes that no source vertex split,
    merged, disappeared, or reconnected.

    Native handles remain local variables and never appear in the returned
    mapping.  The result contains only JSON-safe counts, a stable proof name,
    and bounded failure codes.
    """

    proof_method = (
        "source_edge_endpoint_constraints_plus_target_vertex_IsSame_"
        "unique_perfect_assignment"
    )
    try:
        edge_rows = [tuple(map(int, row)) for row in source_edge_vertex_ids]
    except Exception:
        edge_rows = []
        input_edge_relation_malformed = True
    else:
        input_edge_relation_malformed = False
    try:
        face_rows = [list(map(int, row)) for row in source_face_edge_ids]
    except Exception:
        face_rows = []
        input_face_relation_malformed = True
    else:
        input_face_relation_malformed = False
    source_vertex_ids = sorted({value for row in edge_rows for value in row})
    source_vertex_count = len(source_vertex_ids)
    source_index = {value: index for index, value in enumerate(source_vertex_ids)}
    failures: list[str] = []
    constraints: list[tuple[int, int, int, int]] = []

    if input_edge_relation_malformed or not edge_rows or any(
        len(row) != 2 for row in edge_rows
    ):
        failures.append("source_edge_vertex_relation_malformed")
    if input_face_relation_malformed or not face_rows:
        failures.append("source_face_edge_relation_malformed")
    if sorted(source_index) != list(range(source_vertex_count)):
        failures.append("source_vertex_ids_not_contiguous")

    try:
        target_vertices = _unique_identity_representatives(
            _shape_vertex_handles(observed_shape)
        )
    except Exception as exc:
        failures.append(
            "target_vertex_identity_census_failed:" + type(exc).__name__
        )
        target_vertices = []
    observed_vertex_count = len(target_vertices)

    bindings_by_face: dict[int, Mapping[str, Any]] = {}
    for binding in source_face_bindings:
        if not isinstance(binding, Mapping):
            failures.append("source_face_binding_not_mapping")
            continue
        face_id = binding.get("source_face_index")
        if type(face_id) is not int or not 0 <= face_id < len(face_rows):
            failures.append("source_face_binding_id_invalid")
            continue
        if face_id in bindings_by_face:
            failures.append(f"source_face_{face_id}_binding_duplicate")
            continue
        bindings_by_face[int(face_id)] = binding
    if sorted(bindings_by_face) != list(range(len(face_rows))):
        failures.append("source_face_binding_coverage_incomplete")

    for face_id, expected_edge_ids in enumerate(face_rows):
        binding = bindings_by_face.get(face_id)
        if binding is None:
            continue
        sewing_lineage = binding.get("sewing_lineage")
        source_mapping = binding.get("source_mapping")
        if not isinstance(sewing_lineage, Mapping):
            failures.append(f"source_face_{face_id}_sewing_lineage_missing")
        elif sewing_lineage.get("status") != "mapped":
            failures.append(f"source_face_{face_id}_sewing_lineage_not_mapped")
        if not isinstance(source_mapping, Mapping):
            failures.append(f"source_face_{face_id}_edge_mapping_missing")
            continue
        if source_mapping.get("status") not in {
            "exact_sewing_history",
            "exact_sewing_face_local_geometry",
        }:
            failures.append(f"source_face_{face_id}_edge_mapping_not_exact")

        candidates: list[Mapping[str, Any]] = []
        wire_rows = source_mapping.get("wire_rows")
        if not isinstance(wire_rows, Sequence) or isinstance(
            wire_rows, (str, bytes, bytearray)
        ):
            failures.append(f"source_face_{face_id}_wire_rows_malformed")
            continue
        for wire_row in wire_rows:
            if not isinstance(wire_row, Mapping):
                failures.append(f"source_face_{face_id}_wire_row_malformed")
                continue
            values = wire_row.get("source_edge_candidates")
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                failures.append(
                    f"source_face_{face_id}_edge_candidates_malformed"
                )
                continue
            candidates.extend(
                candidate for candidate in values if isinstance(candidate, Mapping)
            )
            if any(not isinstance(candidate, Mapping) for candidate in values):
                failures.append(
                    f"source_face_{face_id}_edge_candidate_not_mapping"
                )

        candidate_ids = [candidate.get("source_edge_id") for candidate in candidates]
        if any(type(edge_id) is not int for edge_id in candidate_ids):
            failures.append(f"source_face_{face_id}_edge_candidate_id_invalid")
            continue
        if sorted(map(int, candidate_ids)) != sorted(expected_edge_ids):
            failures.append(
                f"source_face_{face_id}_edge_occurrence_relation_mismatch"
            )

        for candidate in candidates:
            if type(candidate.get("source_edge_id")) is not int:
                continue
            edge_id = int(candidate["source_edge_id"])
            if not 0 <= edge_id < len(edge_rows):
                failures.append(f"source_face_{face_id}_source_edge_id_out_of_range")
                continue
            observed_edge = candidate.get("observed_edge")
            if observed_edge is None:
                failures.append(
                    f"source_face_{face_id}_source_edge_{edge_id}_handle_missing"
                )
                continue
            if len(edge_rows[edge_id]) != 2:
                failures.append(
                    f"source_edge_{edge_id}_source_endpoints_malformed"
                )
                continue
            try:
                first, second = _edge_endpoint_handles(observed_edge)
                observed_first = _identity_class_index(first, target_vertices)
                observed_second = _identity_class_index(second, target_vertices)
            except Exception as exc:
                failures.append(
                    f"source_face_{face_id}_source_edge_{edge_id}_"
                    f"endpoint_identity_failed:{type(exc).__name__}"
                )
                continue
            source_first, source_second = edge_rows[edge_id]
            if source_first == source_second and observed_first != observed_second:
                failures.append(
                    f"source_edge_{edge_id}_self_loop_split_after_sewing"
                )
            if source_first != source_second and observed_first == observed_second:
                failures.append(
                    f"source_edge_{edge_id}_distinct_endpoints_merged_after_sewing"
                )
            constraints.append(
                (
                    source_index[source_first],
                    source_index[source_second],
                    observed_first,
                    observed_second,
                )
            )

    expected_occurrence_count = sum(map(len, face_rows))
    if len(constraints) != expected_occurrence_count:
        failures.append("source_vertex_constraint_occurrence_coverage_incomplete")
    if observed_vertex_count > source_vertex_count:
        failures.append("source_vertex_split_or_extra_observed_vertex")
    elif observed_vertex_count < source_vertex_count:
        failures.append("source_vertex_merge_or_missing_observed_vertex")
    if failures:
        return _source_vertex_lineage_failure(
            proof_method=proof_method,
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            failure_codes=failures,
        )

    # An undirected edge permits exactly two pairings, not the Cartesian
    # product produced by intersecting endpoint sets independently.  Search
    # source vertices in constrained order, and require every completed map
    # to replay every unordered endpoint pair exactly.  Two valid solutions
    # are enough to prove non-uniqueness.
    constraints_by_source: list[list[tuple[int, int, int]]] = [
        [] for _ in source_vertex_ids
    ]
    for source_first, source_second, observed_first, observed_second in constraints:
        constraints_by_source[source_first].append(
            (source_second, observed_first, observed_second)
        )
        if source_second != source_first:
            constraints_by_source[source_second].append(
                (source_first, observed_first, observed_second)
            )
    compatibility = [
        sorted(
            set(range(observed_vertex_count)).intersection(
                *(
                    {observed_first, observed_second}
                    for _other, observed_first, observed_second in vertex_constraints
                )
            )
        )
        if vertex_constraints
        else list(range(observed_vertex_count))
        for vertex_constraints in constraints_by_source
    ]
    order = sorted(range(source_vertex_count), key=lambda value: len(compatibility[value]))
    solutions: list[list[int]] = []
    assignment = [-1] * source_vertex_count

    def search_vertex_assignment(depth: int, used: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if depth == len(order):
            if all(
                sorted((assignment[source_first], assignment[source_second]))
                == sorted((observed_first, observed_second))
                for source_first, source_second, observed_first, observed_second in constraints
            ):
                solutions.append(list(assignment))
            return
        source_id = order[depth]
        for observed_id in compatibility[source_id]:
            if observed_id in used:
                continue
            consistent = True
            for other_source, observed_first, observed_second in constraints_by_source[source_id]:
                other_observed = assignment[other_source]
                if other_observed >= 0 and sorted((observed_id, other_observed)) != sorted(
                    (observed_first, observed_second)
                ):
                    consistent = False
                    break
            if not consistent:
                continue
            assignment[source_id] = observed_id
            search_vertex_assignment(depth + 1, {*used, observed_id})
            assignment[source_id] = -1

    search_vertex_assignment(0, set())
    if len(solutions) != 1:
        return _source_vertex_lineage_failure(
            proof_method=proof_method,
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            solution_count=len(solutions),
            failure_codes=[
                "source_vertex_assignment_missing"
                if not solutions
                else "source_vertex_assignment_nonunique"
            ],
        )

    assignment = solutions[0]
    if any(
        sorted((assignment[source_first], assignment[source_second]))
        != sorted((observed_first, observed_second))
        for source_first, source_second, observed_first, observed_second in constraints
    ):
        return _source_vertex_lineage_failure(
            proof_method=proof_method,
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            solution_count=1,
            failure_codes=["source_vertex_assignment_constraint_replay_failed"],
        )
    return {
        "status": "exact_identity",
        "proof_method": proof_method,
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": int(source_vertex_count),
        "observed_vertex_count": int(observed_vertex_count),
        "mapped_source_vertex_count": int(source_vertex_count),
        "mapped_observed_vertex_count": int(observed_vertex_count),
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": int(len(constraints)),
        "failure_codes": [],
    }


def _bindings_with_vertex_lineage_gate(
    source_face_bindings: Sequence[Mapping[str, Any]],
    vertex_lineage: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Clone borrowed bindings and fail every clone closed on a global defect."""

    exact = (
        vertex_lineage.get("status") == "exact_identity"
        and vertex_lineage.get("solution_count") == 1
        and not vertex_lineage.get("failure_codes")
    )
    result: list[dict[str, Any]] = []
    for binding in source_face_bindings:
        cloned = dict(binding)
        sewing = dict(cloned.get("sewing_lineage") or {})
        if not exact:
            sewing["status"] = "mapping_failed"
            failures = list(sewing.get("failure_codes") or [])
            failures.append("global_source_vertex_lineage_not_exact")
            failures.extend(
                str(value) for value in vertex_lineage.get("failure_codes") or []
            )
            sewing["failure_codes"] = list(dict.fromkeys(failures))
        cloned["sewing_lineage"] = sewing
        result.append(cloned)
    return tuple(result)


def _identity_or_geometry_edge_assignment(
    observed_edges: Sequence[Any], source_candidates: Sequence[Mapping[str, Any]]
) -> tuple[list[int] | None, list[str], list[str]]:
    """Uniquely assign observed edges to source occurrences without ordinals."""
    failures: list[str] = []
    source_fingerprints = []
    for candidate in source_candidates:
        candidate_edge = candidate.get("observed_edge")
        if candidate_edge is None:
            failures.append("source_candidate_edge_missing")
            source_fingerprints.append({"available": False})
        else:
            source_fingerprints.append(_edge_curve_fingerprint(candidate_edge))
    observed_fingerprints = [_edge_curve_fingerprint(edge) for edge in observed_edges]
    compatibility: list[list[int]] = []
    proof_methods: dict[tuple[int, int], str] = {}
    for observed_index, (observed_edge, observed_fingerprint) in enumerate(
        zip(observed_edges, observed_fingerprints)
    ):
        candidates = []
        for source_index, (candidate, source_fingerprint) in enumerate(
            zip(source_candidates, source_fingerprints)
        ):
            candidate_edge = candidate.get("observed_edge")
            try:
                identity = bool(
                    candidate_edge is not None
                    and observed_edge.IsSame(candidate_edge)
                )
            except Exception:
                identity = False
            geometry = _curve_fingerprints_match(
                observed_fingerprint, source_fingerprint
            )
            if identity or geometry:
                candidates.append(source_index)
                proof_methods[(observed_index, source_index)] = (
                    "identity" if identity else "geometry_fingerprint"
                )
        compatibility.append(candidates)
    assignment = _unique_perfect_assignment(
        compatibility, len(source_candidates)
    )
    if assignment is None:
        failures.append("edge_unique_perfect_assignment_failed")
        return None, [], failures
    methods = [
        proof_methods[(observed_index, source_index)]
        for observed_index, source_index in enumerate(assignment)
    ]
    return assignment, methods, failures


def _unique_face_local_geometry_target(
    source_edge_occurrences: Sequence[tuple[int, Any]],
    output_faces: Sequence[Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Prove one output face from its complete edge multiset, without ordinals."""
    source_candidates = [
        {"source_edge_id": int(source_edge_id), "observed_edge": edge}
        for source_edge_id, edge in source_edge_occurrences
    ]
    matches: list[tuple[Any, list[str]]] = []
    attempts: list[dict[str, Any]] = []
    for output_face in output_faces:
        try:
            output_edges = list(
                TopologyExplorer(output_face, ignore_orientation=False).edges()
            )
        except Exception as exc:
            attempts.append({
                "status": "edge_exploration_failed",
                "error_type": type(exc).__name__,
            })
            continue
        assignment, methods, failures = _identity_or_geometry_edge_assignment(
            output_edges, source_candidates
        )
        succeeded = assignment is not None and not failures
        attempts.append({
            "status": "unique_edge_multiset" if succeeded else "not_a_unique_edge_multiset",
            "edge_count": len(output_edges),
            "failure_codes": list(failures),
        })
        if succeeded:
            matches.append((output_face, methods))
    if len(matches) != 1:
        return None, {
            "status": "mapping_failed",
            "failure_codes": [
                "sewn_face_geometry_match_not_unique"
                if len(matches) > 1
                else "sewn_face_geometry_match_not_found"
            ],
            "attempts": attempts,
        }
    target, methods = matches[0]
    return target, {
        "status": "mapped",
        "failure_codes": [],
        "mapping_methods": ["face_local_edge_multiset_geometry"],
        "edge_proof_methods": methods,
        "attempts": attempts,
    }


ASSEMBLY_STAGE_PHASES: tuple[tuple[str, str], ...] = (
    ("S1", "post_surface_curve_fit_pre_edge_build"),
    ("S2", "post_edge_build_pre_face_build"),
    ("S3", "post_add_pcurves_pre_optional_face_repair"),
    ("S4", "post_optional_face_repair_pre_sewing"),
    ("S5", "post_sewing_pre_solid"),
    ("S6", "post_solid_pre_step"),
)


def _emit_assembly_stage_failure(
    observer: Callable[[Any, Mapping[str, Any]], None] | None,
    *,
    stage: str,
    phase: str,
    failure_code: str,
    failure: BaseException,
    metadata: Mapping[str, Any],
) -> None:
    """Emit a path-free terminal boundary event without masking the cause."""
    if observer is None:
        return
    _emit_assembly_stage_observation(
        observer,
        None,
        stage=stage,
        phase=phase,
        metadata={
            **dict(metadata),
            "boundary_event": "terminal_failure",
            "failure_code": str(failure_code),
            "failure_type": type(failure).__name__,
        },
    )


def _emit_assembly_stage_observation(
    observer: Callable[[Any, Mapping[str, Any]], None] | None,
    target: Any,
    *,
    stage: str,
    phase: str,
    metadata: Mapping[str, Any],
) -> None:
    """Invoke the borrowed-target observer without accepting a replacement.

    The constructor deliberately ignores the callback return value.  The
    native target and any handles in ``metadata`` are borrowed for immediate
    measurement only; an observer has no mutation or replacement contract.
    Callback failures stop construction and retain both the exact stage and
    the original exception text for the isolated worker's failure record.
    """
    if observer is None:
        return
    expected_phases = dict(ASSEMBLY_STAGE_PHASES)
    if expected_phases.get(stage) != phase:
        raise RuntimeError(
            "invalid assembly observation boundary "
            f"stage={stage} phase={phase}"
        )
    payload = {**dict(metadata), "stage": stage, "phase": phase}
    try:
        observer(target, payload)
    except Exception as exc:
        raise RuntimeError(
            "assembly_stage_observer_failed "
            f"stage={stage} phase={phase} "
            f"error_type={type(exc).__name__}: {exc}"
        ) from exc


def construct_brep_directed(
    surf_wcs: np.ndarray,
    edge_wcs: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
    edge_vertex_adj: np.ndarray,
    *,
    breparg_root: Path,
    directed_trim: bool = True,
    curve_fit_fallback: bool = True,
    wire_continuity: bool = True,
    single_solid: bool = True,
    solid_topology_repair: bool = False,
    pcurve_self_intersection: bool = False,
    local_intersection_topology: bool = False,
    curve_fit_rescue: bool = False,
    curve_interpolate: bool = False,
    local_pcurve_continuity: bool = False,
    surface_fit_precision: bool = False,
    sewing_tolerance: float = 1e-3,
    post_pcurve_face_observer: (
        Callable[[int, Any, Mapping[str, Any]], None] | None
    ) = None,
    assembly_stage_face_observer: (
        Callable[[int, Any, Mapping[str, Any]], None] | None
    ) = None,
    assembly_stage_observer: (
        Callable[[Any, Mapping[str, Any]], None] | None
    ) = None,
    post_pcurve_face_mutator: (
        Callable[[int, Any, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]] | None
    ) = None,
    post_sewing_shape_mutator: (
        Callable[[Any, Sequence[Mapping[str, Any]], Mapping[str, Any]],
                 tuple[Any, Mapping[str, Any]]] | None
    ) = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct one solid using directed trim loops and fail-closed OCC checks.

    ``solid_topology_repair`` is deliberately separate from ``single_solid``.
    The latter is a validation guard retained for historical profile parity;
    the former enables the narrowly scoped near-vertex reconciliation needed
    by the P0-A non-unit-solid case.

    ``assembly_stage_observer`` is a default-off, observation-only six-stage
    construction hook.  Its target is borrowed for immediate measurement, its
    return value is ignored, and STEP roundtrip stage S7 belongs to the runner.

    The two mutators are experimental, default-off stage boundaries.  They may
    return a replacement only with ``diagnostics["accepted"] is True``.  A
    rejected result leaves the current face or sewn shape unchanged.  Native
    OCC work behind either hook must still run in a one-CAD child process.
    """
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.GeomAPI import (
        GeomAPI_Interpolate,
        GeomAPI_PointsToBSpline,
        GeomAPI_PointsToBSplineSurface,
    )
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Shell
    from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
    from OCC.Extend.TopologyUtils import TopologyExplorer

    surface_fit_tolerance = 1e-4 if surface_fit_precision else 5e-2
    surf_wcs = np.asarray(surf_wcs, dtype=np.float64)
    edge_wcs = np.asarray(edge_wcs, dtype=np.float64)
    edge_vertex_adj = np.asarray(edge_vertex_adj, dtype=np.int64)
    if sum(
        bool(value)
        for value in (curve_fit_fallback, curve_fit_rescue, curve_interpolate)
    ) > 1:
        raise ValueError(
            "curve_fit_fallback, curve_fit_rescue, and curve_interpolate are "
            "mutually exclusive"
        )
    diagnostics: dict[str, Any] = {
        "faces": len(surf_wcs), "edges": len(edge_wcs), "loop_count": 0,
        "reversed_edge_uses": 0, "multi_loop_faces": 0,
        "curve_fit_attempts": [], "directed_trim_loop_policies": [],
    }
    topology_edge_vertex_adj = edge_vertex_adj
    shared_vertex_points: dict[int, np.ndarray] = {}
    if solid_topology_repair:
        (
            topology_edge_vertex_adj,
            shared_vertex_points,
            near_vertex_diagnostics,
        ) = reconcile_near_vertices(edge_wcs, edge_vertex_adj, face_edge_adj)
        diagnostics["solid_topology_repair"] = near_vertex_diagnostics
    diagnostics["effective_input_topology"] = effective_topology_summary(
        topology_edge_vertex_adj
    )

    surfaces = []
    for face_index, points in enumerate(surf_wcs):
        grid = TColgp_Array2OfPnt(1, 32, 1, 32)
        for u_index in range(1, 33):
            for v_index in range(1, 33):
                point = points[u_index - 1, v_index - 1]
                grid.SetValue(u_index, v_index, gp_Pnt(*map(float, point)))
        fitter = GeomAPI_PointsToBSplineSurface(
            grid, 3, 8, GeomAbs_C2, surface_fit_tolerance
        )
        if not fitter.IsDone():
            raise RuntimeError(f"surface_fit_not_done face={face_index}")
        surfaces.append(fitter.Surface())
    diagnostics["surface_fit"] = {
        "precision_enabled": bool(surface_fit_precision),
        "tolerance": surface_fit_tolerance,
    }
    shared_vertices: dict[int, Any] = {}
    if shared_vertex_points:
        vertex_builder = BRep_Builder()
        for vertex_id, point in shared_vertex_points.items():
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
            # The representative is the mean of endpoints that were already
            # no farther than 2e-4 apart.  Its tolerance must cover both
            # original curve endpoints without allowing a global snap.
            vertex_builder.UpdateVertex(vertex, 2e-4)
            shared_vertices[int(vertex_id)] = vertex

    edges = []
    curve_tolerances = []
    historical_fit_attempts = ((0, 8, 5e-3), (0, 8, 8e-3), (0, 8, 5e-2))

    for edge_index, points in enumerate(edge_wcs):
        raw_points = np.asarray(points, dtype=np.float64)
        fit_passes = []
        if curve_interpolate:
            cleaned, point_stats = sanitize_curve_points(raw_points)
            fit_passes.append(("interpolate", cleaned, point_stats, ()))
        elif curve_fit_fallback:
            cleaned, point_stats = sanitize_curve_points(raw_points)
            fit_passes.append(
                ("fallback_sanitized", cleaned, point_stats, curve_fit_attempts())
            )
        else:
            fit_passes.append(
                (
                    "historical",
                    raw_points,
                    {"input_points": len(raw_points), "retained_points": len(raw_points)},
                    historical_fit_attempts,
                )
            )
        curve = None
        for fit_mode, candidate_points, point_stats, fit_attempts in fit_passes:
            values = TColgp_Array1OfPnt(1, len(candidate_points))
            for point_index, point in enumerate(candidate_points, 1):
                values.SetValue(point_index, gp_Pnt(*map(float, point)))
            if fit_mode == "interpolate":
                from OCC.Core.TColgp import TColgp_HArray1OfPnt

                point_array = TColgp_HArray1OfPnt(1, len(candidate_points))
                for point_index, point in enumerate(candidate_points, 1):
                    point_array.SetValue(point_index, gp_Pnt(*map(float, point)))
                periodic = bool(
                    len(candidate_points) >= 3
                    and np.linalg.norm(candidate_points[0] - candidate_points[-1])
                    <= 1e-7
                )
                attempt = {
                    "edge_index": edge_index,
                    "fit_mode": fit_mode,
                    **point_stats,
                    "periodic": periodic,
                    "status": "pending",
                }
                try:
                    fitter = GeomAPI_Interpolate(point_array, periodic, 1e-7)
                    fitter.Perform()
                    if fitter.IsDone():
                        curve = fitter.Curve()
                        attempt["status"] = "succeeded"
                    else:
                        attempt["status"] = "not_done"
                except Exception as exc:
                    attempt.update(
                        status="failed", error_type=type(exc).__name__, error=str(exc)
                    )
                diagnostics["curve_fit_attempts"].append(attempt)
                if curve is not None:
                    break
                continue
            for min_degree, max_degree, tolerance in fit_attempts:
                attempt = {
                    "edge_index": edge_index, "fit_mode": fit_mode,
                    "min_degree": min_degree,
                    "max_degree": max_degree, "tolerance": tolerance,
                    **point_stats, "status": "pending",
                }
                try:
                    fitter = GeomAPI_PointsToBSpline(
                        values, min_degree, max_degree, GeomAbs_C2, tolerance
                    )
                    if fitter.IsDone():
                        curve = fitter.Curve()
                        curve_tolerances.append(tolerance)
                        attempt["status"] = "succeeded"
                        diagnostics["curve_fit_attempts"].append(attempt)
                        break
                    attempt["status"] = "not_done"
                except Exception as exc:
                    attempt.update(
                        status="failed", error_type=type(exc).__name__, error=str(exc)
                    )
                diagnostics["curve_fit_attempts"].append(attempt)
            if curve is not None:
                break
        if curve is None and curve_fit_rescue and not curve_fit_fallback:
            try:
                cleaned, point_stats = sanitize_curve_points(raw_points)
                values = TColgp_Array1OfPnt(1, len(cleaned))
                for point_index, point in enumerate(cleaned, 1):
                    values.SetValue(point_index, gp_Pnt(*map(float, point)))
                for min_degree, max_degree, tolerance in curve_fit_attempts():
                    attempt = {
                        "edge_index": edge_index, "fit_mode": "rescue_sanitized",
                        "min_degree": min_degree,
                        "max_degree": max_degree, "tolerance": tolerance,
                        **point_stats, "status": "pending",
                    }
                    try:
                        fitter = GeomAPI_PointsToBSpline(
                            values, min_degree, max_degree, GeomAbs_C2, tolerance
                        )
                        if fitter.IsDone():
                            curve = fitter.Curve()
                            curve_tolerances.append(tolerance)
                            attempt["status"] = "succeeded"
                            diagnostics["curve_fit_attempts"].append(attempt)
                            break
                        attempt["status"] = "not_done"
                    except Exception as exc:
                        attempt.update(
                            status="failed", error_type=type(exc).__name__, error=str(exc)
                        )
                    diagnostics["curve_fit_attempts"].append(attempt)
            except ValueError as exc:
                diagnostics["curve_fit_attempts"].append(
                    {
                        "edge_index": edge_index,
                        "fit_mode": "rescue_sanitized",
                        "status": "sanitize_failed",
                        "input_points": len(raw_points),
                        "retained_points": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if curve is None:
            failure = RuntimeError(f"curve_fit_not_done edge={edge_index}")
            _emit_assembly_stage_failure(
                assembly_stage_observer,
                stage="S1",
                phase="post_surface_curve_fit_pre_edge_build",
                failure_code="curve_fit_not_done",
                failure=failure,
                metadata={
                    "entity_kind": "curve",
                    "observation_scope": "distributed_source_edge_event",
                    "source_edge_id": int(edge_index),
                    "event_sequence_position": int(2 * edge_index),
                    "expected_source_face_count": len(surfaces),
                    "expected_source_edge_count": len(edge_wcs),
                    "fitted_curve_prefix_count": int(edge_index),
                    "built_edge_prefix_count": len(edges),
                },
            )
            raise failure
        # S1 and S2 are distributed, source-edge-bound observations.  Keeping
        # both callbacks inside this historical fit -> MakeEdge loop is
        # essential: enabling the census must not fit a later curve after an
        # earlier edge builder would have stopped the unchanged constructor.
        if assembly_stage_observer is not None:
            _emit_assembly_stage_observation(
                assembly_stage_observer,
                curve,
                stage="S1",
                phase="post_surface_curve_fit_pre_edge_build",
                metadata={
                    "entity_kind": "curve",
                    "boundary_event": "completed",
                    "observation_scope": "distributed_source_edge_event",
                    "source_edge_id": int(edge_index),
                    "event_sequence_position": int(2 * edge_index),
                    "expected_source_face_count": len(surfaces),
                    "expected_source_edge_count": len(edge_wcs),
                    "fitted_surface_count": len(surfaces),
                    "fitted_curve_prefix_count": int(edge_index + 1),
                    "built_edge_prefix_count": len(edges),
                    "surface_fit_tolerance": float(surface_fit_tolerance),
                    "source_vertex_ids": tuple(
                        int(value) for value in edge_vertex_adj[edge_index]
                    ),
                    "effective_vertex_ids": tuple(
                        int(value) for value in topology_edge_vertex_adj[edge_index]
                    ),
                    "source_surface_bindings": tuple(
                        {
                            "source_face_index": int(source_face_index),
                            "surface": surface,
                        }
                        for source_face_index, surface in enumerate(surfaces)
                    ),
                },
            )
        try:
            if shared_vertices:
                start_vertex, end_vertex = map(
                    int, topology_edge_vertex_adj[edge_index]
                )
                if start_vertex != end_vertex:
                    builder = BRepBuilderAPI_MakeEdge(
                        curve, shared_vertices[start_vertex], shared_vertices[end_vertex]
                    )
                else:
                    # A closed edge cannot use one explicit vertex twice through
                    # this OCC overload.  Preserve the original curve-only path.
                    builder = BRepBuilderAPI_MakeEdge(curve)
            else:
                builder = BRepBuilderAPI_MakeEdge(curve)
            if not builder.IsDone():
                raise RuntimeError(f"edge_builder_not_done edge={edge_index}")
            edge = builder.Edge()
        except Exception as failure:
            _emit_assembly_stage_failure(
                assembly_stage_observer,
                stage="S2",
                phase="post_edge_build_pre_face_build",
                failure_code="edge_builder_not_done",
                failure=failure,
                metadata={
                    "entity_kind": "edge",
                    "observation_scope": "distributed_source_edge_event",
                    "source_edge_id": int(edge_index),
                    "event_sequence_position": int(2 * edge_index + 1),
                    "expected_source_face_count": len(surfaces),
                    "expected_source_edge_count": len(edge_wcs),
                    "fitted_curve_prefix_count": int(edge_index + 1),
                    "built_edge_prefix_count": len(edges),
                    "source_vertex_ids": tuple(
                        int(value) for value in edge_vertex_adj[edge_index]
                    ),
                    "effective_vertex_ids": tuple(
                        int(value) for value in topology_edge_vertex_adj[edge_index]
                    ),
                },
            )
            raise
        edges.append(edge)
        if assembly_stage_observer is not None:
            _emit_assembly_stage_observation(
                assembly_stage_observer,
                edge,
                stage="S2",
                phase="post_edge_build_pre_face_build",
                metadata={
                    "entity_kind": "edge",
                    "boundary_event": "completed",
                    "observation_scope": "distributed_source_edge_event",
                    "source_edge_id": int(edge_index),
                    "event_sequence_position": int(2 * edge_index + 1),
                    "expected_source_face_count": len(surfaces),
                    "expected_source_edge_count": len(edge_wcs),
                    "fitted_curve_prefix_count": int(edge_index + 1),
                    "built_edge_prefix_count": len(edges),
                    "source_vertex_ids": tuple(
                        int(value) for value in edge_vertex_adj[edge_index]
                    ),
                    "effective_vertex_ids": tuple(
                        int(value) for value in topology_edge_vertex_adj[edge_index]
                    ),
                },
            )

    def exact_source_edge_mapping(
        observed_face: Any,
        *,
        source_edge_occurrences: Sequence[tuple[int, Any]],
        sewing: Any | None = None,
    ) -> dict[str, Any]:
        """Build proof-bearing wire and edge identity candidates.

        Explorer positions are never exported as correspondence.  A consumer
        must first identify its actual wire by ``IsSame`` and then re-prove
        each edge occurrence against the handles retained in that wire row.
        """
        if observed_face is None:
            return {
                "status": "unmapped",
                "wire_rows": [],
                "failures": ["observation_target_unavailable"],
            }
        wire_rows = []
        failures = []
        try:
            observed_wires = list(
                TopologyExplorer(observed_face, ignore_orientation=False).wires()
            )
        except Exception as exc:
            return {
                "status": "unmapped",
                "wire_rows": [],
                "failures": [f"wire_exploration_failed:{type(exc).__name__}"],
            }
        source_candidates: list[dict[str, Any]] = [
            {
                "source_edge_id": int(source_edge_id),
                "observed_edge": source_edge,
            }
            for source_edge_id, source_edge in source_edge_occurrences
        ]
        sewing_failures = []
        if sewing is not None:
            observed_edges = list(
                TopologyExplorer(
                    observed_face, ignore_orientation=False
                ).edges()
            )
            sewing_projected_candidates: list[dict[str, Any]] = []
            for occurrence_index, (
                source_edge_id,
                pre_sewing_edge,
            ) in enumerate(source_edge_occurrences):
                sewn_edge, sewn_lineage = _unique_sewing_history_target(
                    sewing, pre_sewing_edge, observed_edges
                )
                if sewn_edge is None:
                    sewing_failures.extend(
                        f"source_occurrence_{occurrence_index}_{code}"
                        for code in sewn_lineage["failure_codes"]
                    )
                    continue
                sewing_projected_candidates.append(
                    {
                        "source_edge_id": int(source_edge_id),
                        "observed_edge": sewn_edge,
                    }
                )
            # Both history APIs are a preferred proof only when they jointly
            # cover every occurrence.  Otherwise keep the pre-sewing handles
            # and require the independent face-local geometry assignment.
            if len(sewing_projected_candidates) == len(source_candidates):
                source_candidates = sewing_projected_candidates
                sewing_failures = []

        observed_edges_by_wire: list[list[Any]] = []
        for observed_wire in observed_wires:
            try:
                observed_edges = list(
                    TopologyExplorer(
                        observed_wire, ignore_orientation=False
                    ).edges()
                )
            except Exception as exc:
                failures.append(
                    "wire_edge_exploration_failed:"
                    f"{type(exc).__name__}"
                )
                observed_edges = []
            observed_edges_by_wire.append(observed_edges)

        flattened_observed_edges = [
            edge for wire_edges in observed_edges_by_wire for edge in wire_edges
        ]
        assignment, proof_methods, assignment_failures = (
            _identity_or_geometry_edge_assignment(
                flattened_observed_edges, source_candidates
            )
        )
        failures.extend(assignment_failures)
        assigned_offset = 0
        for observed_wire, observed_edges in zip(
            observed_wires, observed_edges_by_wire
        ):
            edge_candidates = []
            if assignment is not None:
                for local_index, observed_edge in enumerate(observed_edges):
                    observed_index = assigned_offset + local_index
                    source_candidate = source_candidates[
                        assignment[observed_index]
                    ]
                    edge_candidates.append(
                        {
                            "source_edge_id": int(
                                source_candidate["source_edge_id"]
                            ),
                            "observed_edge": observed_edge,
                            "proof_method": proof_methods[observed_index],
                        }
                    )
            wire_rows.append(
                {
                    "observed_wire": observed_wire,
                    "source_edge_candidates": edge_candidates,
                }
            )
            assigned_offset += len(observed_edges)

        expected_source_ids = sorted(
            int(source_edge_id)
            for source_edge_id, _source_edge in source_edge_occurrences
        )
        observed_source_ids = sorted(
            int(candidate["source_edge_id"])
            for row in wire_rows
            for candidate in row["source_edge_candidates"]
        )
        if observed_source_ids != expected_source_ids:
            failures.append("source_edge_occurrence_multiset_mismatch")

        for row_index, row in enumerate(wire_rows):
            for other_row in wire_rows[row_index + 1 :]:
                try:
                    same_wire = bool(
                        row["observed_wire"].IsSame(other_row["observed_wire"])
                    )
                except Exception as exc:
                    failures.append(
                        "wire_identity_failed:" f"{type(exc).__name__}"
                    )
                    continue
                if same_wire:
                    failures.append("observed_wire_identity_not_unique")
        if failures:
            status = "unmapped"
        elif sewing is not None:
            status = (
                "exact_sewing_history"
                if not sewing_failures
                and all(method == "identity" for method in proof_methods)
                else "exact_sewing_face_local_geometry"
            )
        elif all(method == "identity" for method in proof_methods):
            status = "exact_identity"
        else:
            status = "exact_face_local_geometry"
        result = {
            "status": status,
            "wire_rows": wire_rows,
            "failures": failures,
            "edge_proof_methods": proof_methods,
        }
        if sewing_failures:
            result["sewing_history_attempt_notes"] = sewing_failures
        return result

    faces = []
    face_observation_metadata: list[dict[str, Any]] = []
    face_source_edge_occurrences: list[list[tuple[int, Any]]] = []
    for face_index, (surface, incident) in enumerate(zip(surfaces, face_edge_adj)):
        if directed_trim:
            loops, loop_policy = guarded_directed_face_loops(
                incident, topology_edge_vertex_adj
            )
            diagnostics["directed_trim_loop_policies"].append(
                {"face_index": face_index, **loop_policy}
            )
        else:
            loops = historical_face_loops(incident, topology_edge_vertex_adj)
        diagnostics["loop_count"] += len(loops)
        diagnostics["multi_loop_faces"] += int(len(loops) > 1)
        loop_endpoint_gaps: list[list[float]] = []
        loop_endpoint_max_gaps: list[float] = []
        if (
            post_pcurve_face_observer is not None
            or assembly_stage_face_observer is not None
            or assembly_stage_observer is not None
            or post_pcurve_face_mutator is not None
            or post_sewing_shape_mutator is not None
        ):
            for loop in loops:
                endpoint_gaps = []
                for loop_position, (edge_id, reverse) in enumerate(loop):
                    next_edge_id, next_reverse = loop[
                        (loop_position + 1) % len(loop)
                    ]
                    points = np.asarray(edge_wcs[edge_id], dtype=np.float64)
                    next_points = np.asarray(
                        edge_wcs[next_edge_id], dtype=np.float64
                    )
                    edge_end = points[0] if reverse else points[-1]
                    next_start = (
                        next_points[-1] if next_reverse else next_points[0]
                    )
                    endpoint_gaps.append(
                        float(np.linalg.norm(edge_end - next_start))
                    )
                loop_endpoint_gaps.append(endpoint_gaps)
                loop_endpoint_max_gaps.append(max(endpoint_gaps, default=0.0))
        spans = [loop_bbox_diagonal(loop, edge_wcs) for loop in loops]
        outer_index = int(np.argmax(np.asarray(spans)))
        wires = []
        source_edge_occurrences: list[tuple[int, Any]] = []
        for loop_index, loop in enumerate(loops):
            if wire_continuity:
                validate_directed_loop(loop, topology_edge_vertex_adj)
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_id, reverse in loop:
                edge = edges[edge_id].Reversed() if reverse else edges[edge_id]
                source_edge_occurrences.append((int(edge_id), edge))
                diagnostics["reversed_edge_uses"] += int(reverse)
                wire_builder.Add(edge)
            if not wire_builder.IsDone():
                raise RuntimeError(
                    f"wire_builder_not_done face={face_index} loop={loop_index} "
                    f"error={wire_builder.Error()}"
                )
            wires.append(wire_builder.Wire())
        face_builder = BRepBuilderAPI_MakeFace(surface, wires[outer_index])
        for loop_index, wire in enumerate(wires):
            if loop_index != outer_index:
                face_builder.Add(wire)
        if not face_builder.IsDone():
            raise RuntimeError(f"face_builder_not_done face={face_index}")
        face = face_builder.Shape()
        brep_utils.fix_wires(face)
        brep_utils.add_pcurves_to_edges(face)
        post_pcurve_mapping = None
        if (
            assembly_stage_face_observer is not None
            or assembly_stage_observer is not None
            or post_pcurve_face_mutator is not None
        ):
            post_pcurve_mapping = exact_source_edge_mapping(
                face,
                source_edge_occurrences=source_edge_occurrences,
            )
        if (
            assembly_stage_face_observer is not None
            or assembly_stage_observer is not None
            or post_sewing_shape_mutator is not None
        ):
            source_loop_edge_uses = [
                [
                    {
                        "loop_index": int(loop_index),
                        "loop_position": int(loop_position),
                        "source_edge_id": int(edge_id),
                        "reversed": bool(reverse),
                        "endpoint_gap_to_next_3d": float(
                            loop_endpoint_gaps[loop_index][loop_position]
                        ),
                    }
                    for loop_position, (edge_id, reverse) in enumerate(loop)
                ]
                for loop_index, loop in enumerate(loops)
            ]
            source_face_observation = {
                "entity_kind": "face",
                "source_face_index": int(face_index),
                "source_loop_edge_uses": source_loop_edge_uses,
                "outer_loop_index": int(outer_index),
                "loop_3d_endpoint_gaps": loop_endpoint_gaps,
                "loop_3d_endpoint_max_gaps": loop_endpoint_max_gaps,
                "face_3d_endpoint_max_gap": max(
                    loop_endpoint_max_gaps, default=0.0
                ),
            }
            face_observation_metadata.append(source_face_observation)
            if assembly_stage_observer is not None:
                _emit_assembly_stage_observation(
                    assembly_stage_observer,
                    face,
                    stage="S3",
                    phase="post_add_pcurves_pre_optional_face_repair",
                    metadata={
                        **source_face_observation,
                        "boundary_event": "completed",
                        "observation_scope": "distributed_source_face_event",
                        "event_sequence_position": int(2 * face_index),
                        "constructed_face_prefix_count": int(face_index + 1),
                        "post_repair_face_prefix_count": int(face_index),
                        "expected_source_face_count": len(surfaces),
                        "expected_source_edge_count": len(edges),
                        "source_mapping": post_pcurve_mapping,
                    },
                )
        if post_pcurve_face_mutator is not None:
            mutated_face, mutation_diagnostics = post_pcurve_face_mutator(
                int(face_index),
                face,
                {
                    "phase": "post_add_pcurves_pre_repair",
                    "source_face_index": int(face_index),
                    "source_mapping": post_pcurve_mapping,
                    "source_edge_occurrences": tuple(source_edge_occurrences),
                },
            )
            if not isinstance(mutation_diagnostics, Mapping):
                raise RuntimeError("post-pcurve face mutator diagnostics are not a mapping")
            diagnostics.setdefault("post_pcurve_face_mutations", []).append(
                {
                    "source_face_index": int(face_index),
                    **dict(mutation_diagnostics),
                }
            )
            if mutation_diagnostics.get("accepted") is True:
                if mutated_face is None or mutated_face.IsNull():
                    raise RuntimeError("accepted post-pcurve face mutator returned null face")
                face = mutated_face
                post_pcurve_mapping = exact_source_edge_mapping(
                    face,
                    source_edge_occurrences=source_edge_occurrences,
                )
                if post_pcurve_mapping.get("status") not in {
                    "exact_identity",
                    "exact_face_local_geometry",
                }:
                    raise RuntimeError(
                        "accepted post-pcurve face mutator lost exact source mapping"
                    )
        if post_pcurve_face_observer is not None:
            # This hook intentionally sits after pcurve construction and before
            # every local/global face repair. It is observation-only; callers
            # isolate native failures in one-CAD worker processes.
            post_pcurve_face_observer(
                int(face_index),
                face,
                {
                    "phase": "post_add_pcurves_pre_repair",
                    "loop_count": len(loops),
                    "outer_loop_index": outer_index,
                    "loop_3d_endpoint_max_gaps": loop_endpoint_max_gaps,
                    "face_3d_endpoint_max_gap": max(
                        loop_endpoint_max_gaps, default=0.0
                    ),
                },
            )
        if assembly_stage_face_observer is not None:
            assembly_stage_face_observer(
                int(face_index),
                face,
                {
                    "phase": "post_add_pcurves_pre_repair",
                    **source_face_observation,
                    "source_mapping": post_pcurve_mapping,
                },
            )
        if local_pcurve_continuity:
            try:
                from .local_wire_topology_repair import repair_face_local_pcurve
            except ImportError:  # direct script execution
                from local_wire_topology_repair import repair_face_local_pcurve

            repaired_face, repair_diagnostics = repair_face_local_pcurve(face)
            diagnostics.setdefault("local_pcurve_continuity", []).append(
                {"face_index": face_index, **repair_diagnostics}
            )
            if repair_diagnostics["accepted"]:
                face = repaired_face
            else:
                brep_utils.fix_wires(face)
                face = brep_utils.fix_face(face)
        elif local_intersection_topology:
            try:
                from .local_wire_topology_repair import repair_face_local_topology
            except ImportError:  # direct script execution
                from local_wire_topology_repair import repair_face_local_topology

            repaired_face, repair_diagnostics = repair_face_local_topology(face)
            diagnostics.setdefault("local_intersection_topology", []).append(
                {"face_index": face_index, **repair_diagnostics}
            )
            if repair_diagnostics["accepted"]:
                face = repaired_face
            else:
                brep_utils.fix_wires(face)
                face = brep_utils.fix_face(face)
        elif pcurve_self_intersection:
            face_fixer = ShapeFix_Face(face)
            face_fixer.SetPrecision(0.01)
            face_fixer.SetMaxTolerance(0.1)
            face_fixer.SetFixWireMode(True)
            face_fixer.SetFixIntersectingWiresMode(True)
            face_fixer.SetFixLoopWiresMode(True)
            wire_tool = face_fixer.FixWireTool()
            wire_tool.SetPrecision(0.01)
            wire_tool.SetMinTolerance(1e-4)
            wire_tool.SetMaxTolerance(0.1)
            wire_tool.SetClosedWireMode(True)
            wire_tool.SetFixReorderMode(True)
            wire_tool.SetFixConnectedMode(True)
            wire_tool.SetFixEdgeCurvesMode(True)
            wire_tool.SetFixAddPCurveMode(True)
            wire_tool.SetFixReversed2dMode(True)
            wire_tool.SetFixSameParameterMode(True)
            wire_tool.SetFixSelfIntersectionMode(True)
            wire_tool.SetFixSelfIntersectingEdgeMode(True)
            wire_tool.SetFixIntersectingEdgesMode(True)
            wire_tool.SetFixNonAdjacentIntersectingEdgesMode(True)
            wire_tool.SetModifyGeometryMode(True)
            wire_tool.SetModifyTopologyMode(False)
            wire_tool.SetModifyRemoveLoopMode(False)
            fixed_wires = sum(1 for _ in TopologyExplorer(face).wires())
            face_fixer.Perform()
            face_fixer.FixIntersectingWires()
            face_fixer.FixOrientation()
            face = face_fixer.Face()
            diagnostics.setdefault("pcurve_repair", {"faces": 0, "wires": 0})
            diagnostics["pcurve_repair"]["faces"] += 1
            diagnostics["pcurve_repair"]["wires"] += fixed_wires
        else:
            brep_utils.fix_wires(face)
            face = brep_utils.fix_face(face)
        if (
            assembly_stage_face_observer is not None
            or assembly_stage_observer is not None
            or post_sewing_shape_mutator is not None
        ):
            pre_sewing_mapping = exact_source_edge_mapping(
                face,
                source_edge_occurrences=source_edge_occurrences,
            )
            face_source_edge_occurrences.append(
                [
                    (
                        int(candidate["source_edge_id"]),
                        candidate["observed_edge"],
                    )
                    for wire_row in pre_sewing_mapping.get("wire_rows", [])
                    for candidate in wire_row.get("source_edge_candidates", [])
                ]
            )
            if assembly_stage_face_observer is not None:
                assembly_stage_face_observer(
                    int(face_index),
                    face,
                    {
                        "phase": "post_optional_face_repair_pre_sewing",
                        **source_face_observation,
                        "source_mapping": pre_sewing_mapping,
                    },
                )
            if assembly_stage_observer is not None:
                _emit_assembly_stage_observation(
                    assembly_stage_observer,
                    face,
                    stage="S4",
                    phase="post_optional_face_repair_pre_sewing",
                    metadata={
                        **source_face_observation,
                        "boundary_event": "completed",
                        "observation_scope": "distributed_source_face_event",
                        "event_sequence_position": int(2 * face_index + 1),
                        "constructed_face_prefix_count": int(face_index + 1),
                        "post_repair_face_prefix_count": int(face_index + 1),
                        "expected_source_face_count": len(surfaces),
                        "expected_source_edge_count": len(edges),
                        "source_mapping": pre_sewing_mapping,
                    },
                )
        faces.append(face)

    sewing_tolerance_value = float(sewing_tolerance)
    if not np.isfinite(sewing_tolerance_value) or sewing_tolerance_value <= 0.0:
        raise ValueError("sewing_tolerance must be finite and positive")
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(sewing_tolerance_value)
    diagnostics["sewing_tolerance"] = sewing_tolerance_value
    for face in faces:
        sewing.Add(face)
    sewing.Perform()
    sewn = sewing.SewedShape()
    post_sewing_bindings: list[dict[str, Any]] = []
    if (
        assembly_stage_face_observer is not None
        or assembly_stage_observer is not None
        or post_sewing_shape_mutator is not None
    ):
        sewn_faces = list(
            TopologyExplorer(sewn, ignore_orientation=False).faces()
        )
        lineage_rows: list[dict[str, Any]] = []
        for source_face_index, source_face in enumerate(faces):
            target, lineage = _unique_sewing_history_target(
                sewing, source_face, sewn_faces
            )
            if target is None:
                history_lineage = lineage
                target, lineage = _unique_face_local_geometry_target(
                    face_source_edge_occurrences[source_face_index], sewn_faces
                )
                lineage["sewing_history_attempt"] = history_lineage
            lineage_rows.append(
                {
                    "source_face_index": int(source_face_index),
                    "shape": target,
                    **lineage,
                }
            )

        # Sewing may map two distinct input faces onto the same output face.
        # Such a merge is not a proven one-to-one lineage and must not be
        # resolved by comparing explorer ordinals.
        _invalidate_colliding_sewing_targets(lineage_rows)

        for lineage_row in lineage_rows:
            source_face_index = int(lineage_row["source_face_index"])
            observation_target = lineage_row.pop("shape")
            source_mapping = exact_source_edge_mapping(
                observation_target,
                source_edge_occurrences=face_source_edge_occurrences[
                    source_face_index
                ],
                sewing=sewing,
            )
            if (
                lineage_row["status"] == "mapped"
                and source_mapping["status"] not in {
                    "exact_sewing_history",
                    "exact_sewing_face_local_geometry",
                }
            ):
                lineage_row["status"] = "mapping_failed"
                lineage_row["failure_codes"].append(
                    "sewn_edge_occurrences_not_exact_source_identity"
                )
            post_sewing_bindings.append(
                {
                    "source_face_index": source_face_index,
                    "face": observation_target,
                    "source_mapping": source_mapping,
                    "sewing_lineage": dict(lineage_row),
                }
            )
            if assembly_stage_face_observer is not None:
                assembly_stage_face_observer(
                    source_face_index,
                    observation_target,
                    {
                        "phase": "post_sewing_pre_step",
                        **face_observation_metadata[source_face_index],
                        "expected_source_face_count": len(faces),
                        "expected_source_edge_count": len(edges),
                        "sewn_face_count": len(sewn_faces),
                        "sewing_lineage": lineage_row,
                        "source_mapping": source_mapping,
                    },
                )
    if assembly_stage_observer is not None:
        s5_vertex_lineage = _prove_global_source_vertex_lineage(
            sewn,
            post_sewing_bindings,
            face_edge_adj,
            topology_edge_vertex_adj,
        )
        s5_source_face_bindings = _bindings_with_vertex_lineage_gate(
            post_sewing_bindings, s5_vertex_lineage
        )
        _emit_assembly_stage_observation(
            assembly_stage_observer,
            sewn,
            stage="S5",
            phase="post_sewing_pre_solid",
            metadata={
                "entity_kind": "shape",
                "expected_source_face_count": len(faces),
                "expected_source_edge_count": len(edges),
                "sewn_face_count": len(sewn_faces),
                "sewing_tolerance": sewing_tolerance_value,
                "source_face_bindings": s5_source_face_bindings,
                "source_vertex_lineage": s5_vertex_lineage,
            },
        )
    if post_sewing_shape_mutator is not None:
        mutated_sewn, mutation_diagnostics = post_sewing_shape_mutator(
            sewn,
            tuple(post_sewing_bindings),
            {
                "phase": "post_sewing_pre_step",
                "expected_source_face_count": len(faces),
                "expected_source_edge_count": len(edges),
                "sewn_face_count": len(sewn_faces),
            },
        )
        if not isinstance(mutation_diagnostics, Mapping):
            raise RuntimeError("post-sewing shape mutator diagnostics are not a mapping")
        diagnostics["post_sewing_shape_mutation"] = dict(mutation_diagnostics)
        if mutation_diagnostics.get("accepted") is True:
            if mutated_sewn is None or mutated_sewn.IsNull():
                raise RuntimeError("accepted post-sewing shape mutator returned null shape")
            sewn = mutated_sewn
    shell_explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    shells = []
    while shell_explorer.More():
        shells.append(topods_Shell(shell_explorer.Current()))
        shell_explorer.Next()
    diagnostics["shell_count"] = len(shells)
    diagnostics["curve_tolerance_counts"] = {
        str(value): curve_tolerances.count(value) for value in sorted(set(curve_tolerances))
    }
    if single_solid and len(shells) != 1:
        raise RuntimeError(f"sewing_produced_shell_count={len(shells)}")
    if not shells:
        raise RuntimeError("sewing_produced_no_shell")
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(shells[0])
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("solid_builder_not_done")
    solid = maker.Solid()
    solid_explorer = TopExp_Explorer(solid, TopAbs_SOLID)
    solid_count = 0
    while solid_explorer.More():
        solid_count += 1
        solid_explorer.Next()
    diagnostics["solid_count"] = solid_count
    if assembly_stage_observer is not None:
        s6_vertex_lineage = _prove_global_source_vertex_lineage(
            solid,
            post_sewing_bindings,
            face_edge_adj,
            topology_edge_vertex_adj,
        )
        s6_source_face_bindings = _bindings_with_vertex_lineage_gate(
            post_sewing_bindings, s6_vertex_lineage
        )
        _emit_assembly_stage_observation(
            assembly_stage_observer,
            solid,
            stage="S6",
            phase="post_solid_pre_step",
            metadata={
                "entity_kind": "shape",
                "expected_source_face_count": len(faces),
                "expected_source_edge_count": len(edges),
                "shell_count": len(shells),
                "solid_count": solid_count,
                "sewing_tolerance": sewing_tolerance_value,
                "post_sewing_mutation_enabled": (
                    post_sewing_shape_mutator is not None
                ),
                "effective_input_topology": dict(
                    diagnostics["effective_input_topology"]
                ),
                "source_face_bindings": s6_source_face_bindings,
                "source_vertex_lineage": s6_vertex_lineage,
            },
        )
    if single_solid and solid_count != 1:
        raise RuntimeError(f"solid_builder_produced_solid_count={solid_count}")
    return solid, diagnostics
