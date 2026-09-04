"""Fail-closed feasibility helper for a post-sewing pcurve repair.

This module is deliberately not connected to an assembly profile.  It answers
one narrow question: can pcurves on an *exactly mapped* sewn face be
reprojected on a disposable whole-shape copy without changing the BRep graph
or any three-dimensional edge curve?

The caller supplies the private in-memory face/edge bindings emitted by the
``post_sewing_pre_step`` observer in :mod:`tools.directed_trim_assembly`.
Explorer ordinals are never treated as correspondence.  A candidate is
returned only when every source face and source edge has an exact identity
binding, the copied graph is bijective, the requested target is non-periodic
and non-seam, every requested remove/add operation succeeds, all topology and
incidence signatures are unchanged, every sampled 3D curve is bit-identical,
and both native and project-strict-style checks pass.  Any uncertainty returns
the original shape and JSON-safe diagnostics.

``ShapeFix_Edge.FixRemovePCurve`` and ``FixAddPCurve`` are the only mutation
operators.  In particular, this experiment does not retry the already-rejected
``BRepBuilderAPI_Sewing.SetSameParameterMode(False)`` ablation and does not use
the SWIG-ambiguous ``BRep_Builder.UpdateEdge(..., None, ...)`` deletion path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


GRAPH_GATE_SCHEMA = "post_sewing_graph_repair_gate/v1"
SOURCE_IDENTITY_SCHEMA = "post_sewing_source_edge_identity/v1"
CURVE_GATE_SCHEMA = "post_sewing_curve_3d_preservation/v1"
TOPOLOGY_SCHEMA = "post_sewing_topology_incidence/v1"
CURVE_SAMPLE_COUNT = 17

EXACT_MAPPING_STATUSES = frozenset(
    {
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
    }
)
ALLOWED_EDGE_PROOF_METHODS = frozenset(
    {"identity", "geometry_fingerprint", "copy_modified_shape"}
)
EXACT_DEFECT_KINDS = ("adjacent", "closure")

TOPOLOGY_FIELDS = (
    "face_count",
    "edge_count",
    "vertex_count",
    "wire_count",
    "shell_count",
    "solid_count",
    "face_edge_occurrences",
    "face_edge_incidence_counts",
    "face_wire_incidence_counts",
    "wire_edge_incidence_counts",
    "edge_face_incidence_counts",
    "edge_vertex_incidence_counts",
    "vertex_edge_incidence_counts",
    "shell_face_incidence_counts",
    "solid_shell_incidence_counts",
)

GRAPH_KINDS = (
    "faces",
    "edges",
    "vertices",
    "wires",
    "shells",
    "solids",
)

GRAPH_RELATIONS = (
    ("face_edge_occurrences", "faces", "edges"),
    ("face_wires", "faces", "wires"),
    ("wire_edge_occurrences", "wires", "edges"),
    ("edge_vertex_occurrences", "edges", "vertices"),
    ("shell_faces", "shells", "faces"),
    ("solid_shells", "solids", "shells"),
)


def _same(first: Any, second: Any) -> bool:
    try:
        return bool(first.IsSame(second))
    except Exception as exc:  # native identity uncertainty must fail closed
        raise RuntimeError("occ_identity_measurement_failed") from exc


def _unique_shapes(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if not any(_same(value, known) for known in result):
            result.append(value)
    return result


def _subshapes(shape: Any, kind: Any, *, unique: bool = True) -> list[Any]:
    from OCC.Core.TopExp import TopExp_Explorer

    result: list[Any] = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return _unique_shapes(result) if unique else result


def _identity_match(value: Any, candidates: Sequence[Any]) -> list[int]:
    return [index for index, candidate in enumerate(candidates) if _same(value, candidate)]


def _identity_multiset_equal(first: Sequence[Any], second: Sequence[Any]) -> bool:
    """Compare occurrence multiplicities without trusting explorer order.

    One OCC edge can legitimately occur twice in a seam wire.  A one-to-one
    positional comparison would therefore be both order-sensitive and unable
    to distinguish a duplicated candidate from a missing one.  Grouping by
    ``IsSame`` and comparing each group's multiplicity proves the multiset
    while remaining independent of explorer ordinals.
    """

    if len(first) != len(second):
        return False
    representatives = _unique_shapes([*first, *second])
    return all(
        sum(_same(value, representative) for value in first)
        == sum(_same(value, representative) for value in second)
        for representative in representatives
    )


def shape_topology_incidence_signature(shape: Any) -> dict[str, Any]:
    """Measure whole-shape counts and incidence multisets without ordinals."""
    from OCC.Core.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )

    faces = _subshapes(shape, TopAbs_FACE)
    edges = _subshapes(shape, TopAbs_EDGE)
    vertices = _subshapes(shape, TopAbs_VERTEX)
    wires = _subshapes(shape, TopAbs_WIRE)
    shells = _subshapes(shape, TopAbs_SHELL)
    solids = _subshapes(shape, TopAbs_SOLID)

    face_edge_occurrences = [
        _subshapes(face, TopAbs_EDGE, unique=False) for face in faces
    ]
    face_edges = [_unique_shapes(values) for values in face_edge_occurrences]
    face_wires = [_subshapes(face, TopAbs_WIRE) for face in faces]
    wire_edges = [_subshapes(wire, TopAbs_EDGE) for wire in wires]
    edge_vertices = [
        _subshapes(edge, TopAbs_VERTEX, unique=False) for edge in edges
    ]
    shell_faces = [_subshapes(shell, TopAbs_FACE) for shell in shells]
    solid_shells = [_subshapes(solid, TopAbs_SHELL) for solid in solids]

    edge_face_counts = [
        sum(any(_same(edge, value) for value in per_face) for per_face in face_edges)
        for edge in edges
    ]
    vertex_edge_counts = [0 for _ in vertices]
    for endpoints in edge_vertices:
        for endpoint in endpoints:
            matches = _identity_match(endpoint, vertices)
            if len(matches) != 1:
                raise RuntimeError("edge_endpoint_global_identity_not_unique")
            vertex_edge_counts[matches[0]] += 1

    result = {
        "schema": TOPOLOGY_SCHEMA,
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "wire_count": len(wires),
        "shell_count": len(shells),
        "solid_count": len(solids),
        "face_edge_occurrences": sum(len(values) for values in face_edge_occurrences),
        "face_edge_incidence_counts": sorted(len(values) for values in face_edges),
        "face_wire_incidence_counts": sorted(len(values) for values in face_wires),
        "wire_edge_incidence_counts": sorted(len(values) for values in wire_edges),
        "edge_face_incidence_counts": sorted(edge_face_counts),
        "edge_vertex_incidence_counts": sorted(len(values) for values in edge_vertices),
        "vertex_edge_incidence_counts": sorted(vertex_edge_counts),
        "shell_face_incidence_counts": sorted(len(values) for values in shell_faces),
        "solid_shell_incidence_counts": sorted(len(values) for values in solid_shells),
    }
    if any(type(value) is not int or value < 0 for key in TOPOLOGY_FIELDS for value in (
        result[key] if isinstance(result[key], list) else [result[key]]
    )):
        raise RuntimeError("topology_signature_contains_invalid_count")
    return result


def _graph_inventory(shape: Any) -> dict[str, Any]:
    """Retain a private identity-labelled topology graph for an in-process gate."""

    from OCC.Core.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )

    kinds = {
        "faces": _subshapes(shape, TopAbs_FACE),
        "edges": _subshapes(shape, TopAbs_EDGE),
        "vertices": _subshapes(shape, TopAbs_VERTEX),
        "wires": _subshapes(shape, TopAbs_WIRE),
        "shells": _subshapes(shape, TopAbs_SHELL),
        "solids": _subshapes(shape, TopAbs_SOLID),
    }

    def relation(parent_kind: str, child_kind: str, child_type: Any) -> list[list[int]]:
        children = kinds[child_kind]
        rows: list[list[int]] = []
        for parent in kinds[parent_kind]:
            child_indices: list[int] = []
            for child in _subshapes(parent, child_type, unique=False):
                matches = _identity_match(child, children)
                if len(matches) != 1:
                    raise RuntimeError(
                        f"{parent_kind}_{child_kind}_incidence_identity_not_unique"
                    )
                child_indices.append(matches[0])
            rows.append(sorted(child_indices))
        return rows

    relations = {
        "face_edge_occurrences": relation("faces", "edges", TopAbs_EDGE),
        "face_wires": relation("faces", "wires", TopAbs_WIRE),
        "wire_edge_occurrences": relation("wires", "edges", TopAbs_EDGE),
        "edge_vertex_occurrences": relation("edges", "vertices", TopAbs_VERTEX),
        "shell_faces": relation("shells", "faces", TopAbs_FACE),
        "solid_shells": relation("solids", "shells", TopAbs_SHELL),
    }
    return {"kinds": kinds, "relations": relations}


def _bijection_by_identity(
    expected: Sequence[Any], observed: Sequence[Any]
) -> tuple[dict[int, int], bool]:
    mapping: dict[int, int] = {}
    for expected_index, value in enumerate(expected):
        matches = _identity_match(value, observed)
        if len(matches) != 1:
            return {}, False
        mapping[expected_index] = matches[0]
    return mapping, bool(
        len(expected) == len(observed)
        and sorted(mapping.values()) == list(range(len(observed)))
    )


def _relations_equal_under_mapping(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    mappings: Mapping[str, Mapping[int, int]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for relation_name, parent_kind, child_kind in GRAPH_RELATIONS:
        before_rows = before["relations"][relation_name]
        after_rows = after["relations"][relation_name]
        parent_mapping = mappings[parent_kind]
        child_mapping = mappings[child_kind]
        relation_ok = len(before_rows) == len(after_rows)
        if relation_ok:
            for before_parent_index, before_children in enumerate(before_rows):
                after_parent_index = parent_mapping.get(before_parent_index)
                expected_children = sorted(
                    child_mapping.get(child_index, -1) for child_index in before_children
                )
                if (
                    after_parent_index is None
                    or -1 in expected_children
                    or expected_children != after_rows[after_parent_index]
                ):
                    relation_ok = False
                    break
        checks[f"{relation_name}_equal"] = relation_ok
    return checks


def exact_identity_graph_gate(before: Mapping[str, Any], after_shape: Any) -> dict[str, Any]:
    """Prove that an in-place mutation retained every identity and incidence."""

    try:
        after = _graph_inventory(after_shape)
        mappings: dict[str, dict[int, int]] = {}
        checks: dict[str, bool] = {}
        for kind in GRAPH_KINDS:
            mapping, accepted = _bijection_by_identity(
                before["kinds"][kind], after["kinds"][kind]
            )
            mappings[kind] = mapping
            checks[f"{kind}_identity_bijection"] = accepted
        if all(checks.values()):
            checks.update(_relations_equal_under_mapping(before, after, mappings))
        else:
            checks.update(
                {f"{name}_equal": False for name, _parent, _child in GRAPH_RELATIONS}
            )
    except Exception as exc:
        return {
            "accepted": False,
            "checks": {"identity_graph_measurement_completed": False},
            "rejection_reasons": [
                f"identity_graph_measurement_failed:{type(exc).__name__}"
            ],
        }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, value in checks.items() if not value],
    }


def copied_identity_graph_gate(
    before_shape: Any, after_shape: Any, copier: Any
) -> dict[str, Any]:
    """Prove a deep copy is a full bijective graph copy using OCC history."""

    try:
        before = _graph_inventory(before_shape)
        after = _graph_inventory(after_shape)
        mappings: dict[str, dict[int, int]] = {}
        checks: dict[str, bool] = {}
        for kind in GRAPH_KINDS:
            copied_handles = []
            for value in before["kinds"][kind]:
                copied = copier.ModifiedShape(value)
                if copied.IsNull():
                    raise RuntimeError(f"copied_{kind}_history_missing")
                copied_handles.append(copied)
            mapping, accepted = _bijection_by_identity(
                copied_handles, after["kinds"][kind]
            )
            mappings[kind] = mapping
            checks[f"{kind}_copy_history_bijection"] = accepted
        if all(checks.values()):
            checks.update(_relations_equal_under_mapping(before, after, mappings))
        else:
            checks.update(
                {f"{name}_equal": False for name, _parent, _child in GRAPH_RELATIONS}
            )
    except Exception as exc:
        return {
            "accepted": False,
            "checks": {"copy_graph_measurement_completed": False},
            "rejection_reasons": [f"copy_graph_measurement_failed:{type(exc).__name__}"],
        }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, value in checks.items() if not value],
    }


def combine_graph_gates(*gates: Mapping[str, Any]) -> dict[str, Any]:
    """Combine independently measured graph gates without weakening any one."""

    checks = {
        f"gate_{index}_accepted": gate.get("accepted") is True
        for index, gate in enumerate(gates)
    }
    return {
        "accepted": bool(gates) and all(checks.values()),
        "checks": checks,
        "components": list(gates),
        "rejection_reasons": [name for name, value in checks.items() if not value],
    }


def topology_incidence_gate(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Pure exact-equality gate for all registered topology invariants."""
    checks: dict[str, bool] = {}
    for field in TOPOLOGY_FIELDS:
        checks[f"{field}_equal"] = bool(
            field in before and field in after and before[field] == after[field]
        )
    schema_valid = bool(
        before.get("schema") == TOPOLOGY_SCHEMA
        and after.get("schema") == TOPOLOGY_SCHEMA
    )
    checks = {"schema_valid": schema_valid, **checks}
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, accepted in checks.items() if not accepted],
    }


def _binding_rows(
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(source_face_bindings, Mapping):
        rows = list(source_face_bindings.values())
    elif isinstance(source_face_bindings, Sequence) and not isinstance(
        source_face_bindings, (str, bytes)
    ):
        rows = list(source_face_bindings)
    else:
        raise TypeError("source_face_bindings must be a sequence or mapping")
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("source_face_bindings must contain mapping rows")
    return rows


def _extract_binding_occurrences(
    shape: Any,
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Validate complete exact observer bindings and retain private handles."""
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE

    rows = _binding_rows(source_face_bindings)
    global_faces = _subshapes(shape, TopAbs_FACE)
    global_edges = _subshapes(shape, TopAbs_EDGE)
    failures: list[str] = []
    normalized: list[dict[str, Any]] = []
    all_occurrences: list[dict[str, Any]] = []
    seen_source_faces: set[int] = set()
    seen_global_faces: set[int] = set()
    expected_source_face_count: int | None = None
    expected_source_edge_count: int | None = None

    for row_position, row in enumerate(rows):
        source_face_index = row.get("source_face_index")
        face = row.get("face")
        source_mapping = row.get("source_mapping")
        row_expected_faces = row.get("expected_source_face_count")
        row_expected_edges = row.get("expected_source_edge_count")
        if row_expected_faces is None and isinstance(row.get("metadata"), Mapping):
            row_expected_faces = row["metadata"].get("expected_source_face_count")
        if row_expected_edges is None and isinstance(row.get("metadata"), Mapping):
            row_expected_edges = row["metadata"].get("expected_source_edge_count")
        if type(row_expected_faces) is not int or row_expected_faces < 1:
            failures.append(f"source_face_{source_face_index}_expected_face_count_invalid")
        elif expected_source_face_count is None:
            expected_source_face_count = int(row_expected_faces)
        elif row_expected_faces != expected_source_face_count:
            failures.append("expected_source_face_count_inconsistent")
        if type(row_expected_edges) is not int or row_expected_edges < 1:
            failures.append(f"source_face_{source_face_index}_expected_edge_count_invalid")
        elif expected_source_edge_count is None:
            expected_source_edge_count = int(row_expected_edges)
        elif row_expected_edges != expected_source_edge_count:
            failures.append("expected_source_edge_count_inconsistent")
        if type(source_face_index) is not int or source_face_index < 0:
            failures.append(f"binding_{row_position}_source_face_index_invalid")
            continue
        if source_face_index in seen_source_faces:
            failures.append(f"source_face_{source_face_index}_binding_duplicated")
            continue
        seen_source_faces.add(source_face_index)
        if face is None:
            failures.append(f"source_face_{source_face_index}_face_missing")
            continue
        face_matches = _identity_match(face, global_faces)
        if len(face_matches) != 1:
            failures.append(f"source_face_{source_face_index}_global_face_identity_not_unique")
            continue
        if face_matches[0] in seen_global_faces:
            failures.append(f"source_face_{source_face_index}_global_face_reused")
            continue
        seen_global_faces.add(face_matches[0])
        if not isinstance(source_mapping, Mapping):
            failures.append(f"source_face_{source_face_index}_mapping_missing")
            continue
        if source_mapping.get("status") not in EXACT_MAPPING_STATUSES:
            failures.append(f"source_face_{source_face_index}_mapping_not_exact")
            continue
        if source_mapping.get("failures"):
            failures.append(f"source_face_{source_face_index}_mapping_has_failures")
            continue
        wire_rows = source_mapping.get("wire_rows")
        if not isinstance(wire_rows, Sequence) or isinstance(wire_rows, (str, bytes)):
            failures.append(f"source_face_{source_face_index}_wire_rows_missing")
            continue
        face_wires = _subshapes(face, TopAbs_WIRE)
        seen_wires: set[int] = set()
        normalized_wires: list[dict[str, Any]] = []
        for wire_position, wire_row in enumerate(wire_rows):
            if not isinstance(wire_row, Mapping):
                failures.append(f"source_face_{source_face_index}_wire_row_invalid")
                continue
            wire = wire_row.get("observed_wire")
            wire_matches = _identity_match(wire, face_wires) if wire is not None else []
            if len(wire_matches) != 1 or wire_matches[0] in seen_wires:
                failures.append(
                    f"source_face_{source_face_index}_wire_{wire_position}_identity_not_unique"
                )
                continue
            seen_wires.add(wire_matches[0])
            candidates = wire_row.get("source_edge_candidates")
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                failures.append(
                    f"source_face_{source_face_index}_wire_{wire_position}_candidates_missing"
                )
                continue
            wire_edge_occurrences = _subshapes(wire, TopAbs_EDGE, unique=False)
            normalized_candidates: list[dict[str, Any]] = []
            candidate_edges: list[Any] = []
            for occurrence_position, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    failures.append(
                        f"source_face_{source_face_index}_wire_{wire_position}_candidate_invalid"
                    )
                    continue
                source_edge_id = candidate.get("source_edge_id")
                edge = candidate.get("observed_edge")
                proof_method = candidate.get("proof_method")
                if type(source_edge_id) is not int or source_edge_id < 0 or edge is None:
                    failures.append(
                        f"source_face_{source_face_index}_wire_{wire_position}_candidate_invalid"
                    )
                    continue
                if proof_method not in ALLOWED_EDGE_PROOF_METHODS:
                    failures.append(
                        f"source_face_{source_face_index}_source_edge_{source_edge_id}_proof_method_invalid"
                    )
                    continue
                if not _identity_match(edge, wire_edge_occurrences):
                    failures.append(
                        f"source_face_{source_face_index}_source_edge_{source_edge_id}_not_in_wire"
                    )
                    continue
                if len(_identity_match(edge, global_edges)) != 1:
                    failures.append(
                        f"source_face_{source_face_index}_source_edge_{source_edge_id}_not_global"
                    )
                    continue
                occurrence = {
                    "source_face_index": source_face_index,
                    "source_edge_id": source_edge_id,
                    "face": face,
                    "wire": wire,
                    "edge": edge,
                    "wire_position": wire_position,
                    "occurrence_position": occurrence_position,
                    "proof_method": str(proof_method),
                }
                normalized_candidates.append(occurrence)
                candidate_edges.append(edge)
                all_occurrences.append(occurrence)
            if len(normalized_candidates) != len(wire_edge_occurrences):
                failures.append(
                    f"source_face_{source_face_index}_wire_{wire_position}_edge_occurrence_count_mismatch"
                )
            elif not _identity_multiset_equal(candidate_edges, wire_edge_occurrences):
                failures.append(
                    f"source_face_{source_face_index}_wire_{wire_position}_edge_occurrence_identity_multiset_mismatch"
                )
            normalized_wires.append(
                {"wire": wire, "occurrences": normalized_candidates}
            )
        if len(seen_wires) != len(face_wires):
            failures.append(f"source_face_{source_face_index}_wire_coverage_incomplete")
        normalized.append(
            {
                "source_face_index": source_face_index,
                "face": face,
                "wires": normalized_wires,
            }
        )

    if len(rows) != len(global_faces) or seen_global_faces != set(range(len(global_faces))):
        failures.append("source_face_binding_coverage_incomplete")
    if seen_source_faces and seen_source_faces != set(range(len(global_faces))):
        failures.append("source_face_ids_not_contiguous")
    if expected_source_face_count != len(global_faces):
        failures.append("expected_source_face_count_mismatch")
    if expected_source_edge_count != len(global_edges):
        failures.append("expected_source_edge_count_mismatch")
    return normalized, all_occurrences, sorted(set(failures))


def measure_source_edge_identity(
    shape: Any,
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove source-edge incidence and reject every split/merge ambiguity."""
    from OCC.Core.TopAbs import TopAbs_EDGE

    try:
        bindings, occurrences, failures = _extract_binding_occurrences(
            shape, source_face_bindings
        )
        global_edges = _subshapes(shape, TopAbs_EDGE)
        by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for occurrence in occurrences:
            by_source[int(occurrence["source_edge_id"])].append(occurrence)
        split_ids: list[int] = []
        representatives: dict[int, Any] = {}
        for source_edge_id, values in sorted(by_source.items()):
            representative = values[0]["edge"]
            representatives[source_edge_id] = representative
            if any(not _same(representative, value["edge"]) for value in values[1:]):
                split_ids.append(source_edge_id)
        merged_pairs: list[list[int]] = []
        source_ids = sorted(representatives)
        for offset, source_edge_id in enumerate(source_ids):
            for other_id in source_ids[offset + 1 :]:
                if _same(representatives[source_edge_id], representatives[other_id]):
                    merged_pairs.append([source_edge_id, other_id])
        representative_global_indices: list[int] = []
        for source_edge_id in source_ids:
            matches = _identity_match(representatives[source_edge_id], global_edges)
            if len(matches) != 1:
                failures.append(f"source_edge_{source_edge_id}_global_identity_not_unique")
            else:
                representative_global_indices.append(matches[0])
        source_edge_bijection = bool(
            source_ids == list(range(len(global_edges)))
            and sorted(representative_global_indices) == list(range(len(global_edges)))
            and not split_ids
            and not merged_pairs
        )
        if not source_edge_bijection:
            failures.append("source_to_global_edge_bijection_failed")
        face_source_edges = [
            {
                "source_face_index": int(binding["source_face_index"]),
                "source_edge_ids": sorted(
                    int(occurrence["source_edge_id"])
                    for wire in binding["wires"]
                    for occurrence in wire["occurrences"]
                ),
            }
            for binding in sorted(bindings, key=lambda value: value["source_face_index"])
        ]
        occurrence_counts = [
            {"source_edge_id": source_edge_id, "count": len(by_source[source_edge_id])}
            for source_edge_id in source_ids
        ]
    except Exception as exc:
        return {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "complete": False,
            "source_edge_ids": [],
            "occurrence_counts": [],
            "face_source_edge_occurrences": [],
            "split_source_edge_ids": [],
            "merged_source_edge_pairs": [],
            "source_to_global_edge_bijection": False,
            "failure_codes": [f"identity_measurement_failed:{type(exc).__name__}"],
        }
    failures = sorted(set(failures))
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "complete": not failures,
        "source_edge_ids": source_ids,
        "occurrence_counts": occurrence_counts,
        "face_source_edge_occurrences": face_source_edges,
        "split_source_edge_ids": split_ids,
        "merged_source_edge_pairs": merged_pairs,
        "source_to_global_edge_bijection": source_edge_bijection,
        "failure_codes": failures,
    }


def source_edge_identity_gate(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Pure gate for complete source-edge incidence and split/merge freedom."""
    checks = {
        "schema_valid": bool(
            before.get("schema") == SOURCE_IDENTITY_SCHEMA
            and after.get("schema") == SOURCE_IDENTITY_SCHEMA
        ),
        "before_complete": before.get("complete") is True,
        "after_complete": after.get("complete") is True,
        "before_bijective": before.get("source_to_global_edge_bijection") is True,
        "after_bijective": after.get("source_to_global_edge_bijection") is True,
        "no_split_before": before.get("split_source_edge_ids") == [],
        "no_split_after": after.get("split_source_edge_ids") == [],
        "no_merge_before": before.get("merged_source_edge_pairs") == [],
        "no_merge_after": after.get("merged_source_edge_pairs") == [],
        "source_edge_ids_equal": bool(
            "source_edge_ids" in before
            and before.get("source_edge_ids") == after.get("source_edge_ids")
        ),
        "occurrence_counts_equal": bool(
            "occurrence_counts" in before
            and before.get("occurrence_counts") == after.get("occurrence_counts")
        ),
        "face_source_edge_occurrences_equal": bool(
            "face_source_edge_occurrences" in before
            and before.get("face_source_edge_occurrences")
            == after.get("face_source_edge_occurrences")
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, accepted in checks.items() if not accepted],
    }


def _edge_curve_sample(source_edge_id: int, edge: Any) -> dict[str, Any]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

    try:
        adaptor = BRepAdaptor_Curve(edge)
        first = float(adaptor.FirstParameter())
        last = float(adaptor.LastParameter())
        if not math.isfinite(first) or not math.isfinite(last):
            raise ValueError("curve parameter range is non-finite")
        samples = []
        for position in range(CURVE_SAMPLE_COUNT):
            parameter = first + (last - first) * position / (CURVE_SAMPLE_COUNT - 1)
            point = adaptor.Value(parameter)
            samples.append([float(point.X()), float(point.Y()), float(point.Z())])
        if any(not math.isfinite(value) for sample in samples for value in sample):
            raise ValueError("curve samples are non-finite")
        return {
            "source_edge_id": int(source_edge_id),
            "available": True,
            "curve_type": int(adaptor.GetType()),
            "parameter_range": [first, last],
            "samples": samples,
        }
    except Exception as exc:
        return {
            "source_edge_id": int(source_edge_id),
            "available": False,
            "error_type": type(exc).__name__,
            "samples": [],
        }


def _representative_edges(
    shape: Any,
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> dict[int, Any]:
    _bindings, occurrences, failures = _extract_binding_occurrences(
        shape, source_face_bindings
    )
    if failures:
        raise RuntimeError("source_edge_binding_not_exact")
    result: dict[int, Any] = {}
    for occurrence in occurrences:
        source_edge_id = int(occurrence["source_edge_id"])
        edge = occurrence["edge"]
        if source_edge_id in result and not _same(result[source_edge_id], edge):
            raise RuntimeError("source_edge_split")
        result[source_edge_id] = edge
    return result


def sample_source_edge_curves(
    shape: Any,
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _edge_curve_sample(source_edge_id, edge)
        for source_edge_id, edge in sorted(
            _representative_edges(shape, source_face_bindings).items()
        )
    ]


def exact_curve_sample_gate(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pure, orientation-independent, exact 3D curve preservation gate."""
    before_by_id = {row.get("source_edge_id"): row for row in before}
    after_by_id = {row.get("source_edge_id"): row for row in after}
    source_ids_equal = bool(
        len(before_by_id) == len(before)
        and len(after_by_id) == len(after)
        and set(before_by_id) == set(after_by_id)
    )
    rows: list[dict[str, Any]] = []
    maximum: float | None = 0.0
    if source_ids_equal:
        for source_edge_id in sorted(before_by_id):
            left = before_by_id[source_edge_id]
            right = after_by_id[source_edge_id]
            available = bool(left.get("available") and right.get("available"))
            left_samples = left.get("samples") or []
            right_samples = right.get("samples") or []
            samples_finite = bool(
                all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for point in [*left_samples, *right_samples]
                    if isinstance(point, Sequence) and not isinstance(point, (str, bytes))
                    for value in point
                )
            )
            sample_shapes_equal = bool(
                len(left_samples) == CURVE_SAMPLE_COUNT
                and len(right_samples) == CURVE_SAMPLE_COUNT
                and all(len(point) == 3 for point in [*left_samples, *right_samples])
                and samples_finite
            )
            forward = reverse = math.inf
            if sample_shapes_equal:
                forward = max(
                    math.dist(tuple(first), tuple(second))
                    for first, second in zip(left_samples, right_samples)
                )
                reverse = max(
                    math.dist(tuple(first), tuple(second))
                    for first, second in zip(left_samples, reversed(right_samples))
                )
            edge_delta = min(forward, reverse)
            if not math.isfinite(edge_delta):
                maximum = None
            elif maximum is not None:
                maximum = max(maximum, edge_delta)
            row_accepted = bool(
                available
                and sample_shapes_equal
                and left.get("curve_type") == right.get("curve_type")
                and left.get("parameter_range") == right.get("parameter_range")
                and edge_delta == 0.0
            )
            rows.append(
                {
                    "source_edge_id": source_edge_id,
                    "available": available,
                    "sample_count_equal": sample_shapes_equal,
                    "curve_type_equal": left.get("curve_type") == right.get("curve_type"),
                    "parameter_range_equal": left.get("parameter_range")
                    == right.get("parameter_range"),
                    "max_sample_delta": edge_delta if math.isfinite(edge_delta) else None,
                    "accepted": row_accepted,
                }
            )
    checks = {
        "source_edge_ids_equal": source_ids_equal,
        "nonempty": bool(before),
        "all_edges_available_and_exact": bool(rows) and all(row["accepted"] for row in rows),
        "max_sample_delta_zero": maximum == 0.0,
    }
    return {
        "schema": CURVE_GATE_SCHEMA,
        "accepted": all(checks.values()),
        "checks": checks,
        "max_sample_delta": maximum,
        "required_max_sample_delta": 0.0,
        "edges": rows,
        "rejection_reasons": [name for name, accepted in checks.items() if not accepted],
    }


def evaluate_graph_preservation_gate(
    *,
    topology_gate: Mapping[str, Any],
    source_identity_gate: Mapping[str, Any],
    curve_3d_gate: Mapping[str, Any],
    pcurve_operations_complete: bool,
    original_shape_unchanged: bool,
    target_face_was_invalid: bool,
    target_face_is_clean: bool,
    native_valid: bool,
    strict_valid: bool,
) -> dict[str, Any]:
    """Pure final feasibility decision; absent or non-boolean evidence fails."""
    checks = {
        "topology_incidence_preserved": topology_gate.get("accepted") is True,
        "source_edge_graph_preserved": source_identity_gate.get("accepted") is True,
        "three_dimensional_curves_preserved": curve_3d_gate.get("accepted") is True,
        "pcurve_operations_complete": pcurve_operations_complete is True,
        "original_shape_unchanged": original_shape_unchanged is True,
        "target_face_was_invalid": target_face_was_invalid is True,
        "target_face_is_clean": target_face_is_clean is True,
        "native_valid": native_valid is True,
        "strict_valid": strict_valid is True,
    }
    return {
        "schema": GRAPH_GATE_SCHEMA,
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, accepted in checks.items() if not accepted],
    }


def _deep_copy(shape: Any) -> tuple[Any, Any]:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy

    copier = BRepBuilderAPI_Copy(shape, True, False)
    if not copier.IsDone() or copier.Shape().IsNull():
        raise RuntimeError("deep_copy_not_done")
    return copier, copier.Shape()


def _copy_bindings(
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    copier: Any,
) -> list[dict[str, Any]]:
    """Map every private observer handle through one whole-shape copy history."""
    from OCC.Core.TopoDS import topods

    result = []
    for row in _binding_rows(source_face_bindings):
        face = row.get("face")
        source_mapping = row.get("source_mapping")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        expected_source_face_count = row.get(
            "expected_source_face_count", metadata.get("expected_source_face_count")
        )
        expected_source_edge_count = row.get(
            "expected_source_edge_count", metadata.get("expected_source_edge_count")
        )
        if face is None or not isinstance(source_mapping, Mapping):
            raise RuntimeError("binding_missing_private_handles")
        copied_face_shape = copier.ModifiedShape(face)
        if copied_face_shape.IsNull():
            raise RuntimeError("copied_face_mapping_missing")
        copied_wire_rows = []
        for wire_row in source_mapping.get("wire_rows") or []:
            copied_wire_shape = copier.ModifiedShape(wire_row.get("observed_wire"))
            if copied_wire_shape.IsNull():
                raise RuntimeError("copied_wire_mapping_missing")
            copied_candidates = []
            for candidate in wire_row.get("source_edge_candidates") or []:
                copied_edge_shape = copier.ModifiedShape(candidate.get("observed_edge"))
                if copied_edge_shape.IsNull():
                    raise RuntimeError("copied_edge_mapping_missing")
                copied_candidates.append(
                    {
                        "source_edge_id": int(candidate["source_edge_id"]),
                        "observed_edge": topods.Edge(copied_edge_shape),
                        "proof_method": "copy_modified_shape",
                    }
                )
            copied_wire_rows.append(
                {
                    "observed_wire": topods.Wire(copied_wire_shape),
                    "source_edge_candidates": copied_candidates,
                }
            )
        result.append(
            {
                "source_face_index": int(row["source_face_index"]),
                "expected_source_face_count": int(expected_source_face_count),
                "expected_source_edge_count": int(expected_source_edge_count),
                "face": topods.Face(copied_face_shape),
                "source_mapping": {
                    "status": source_mapping.get("status"),
                    "failures": list(source_mapping.get("failures") or []),
                    "wire_rows": copied_wire_rows,
                },
            }
        )
    return result


def _target_binding(
    bindings: Sequence[Mapping[str, Any]], source_face_index: int
) -> Mapping[str, Any]:
    matches = [row for row in bindings if row.get("source_face_index") == source_face_index]
    if len(matches) != 1:
        raise RuntimeError("target_source_face_binding_not_unique")
    return matches[0]


def _target_edges(
    binding: Mapping[str, Any], source_edge_ids: Sequence[int]
) -> list[tuple[int, Any]]:
    requested = [int(value) for value in source_edge_ids]
    if not requested or len(set(requested)) != len(requested) or any(value < 0 for value in requested):
        raise ValueError("target_source_edge_ids must be unique nonnegative ids")
    found: dict[int, list[Any]] = defaultdict(list)
    for wire_row in binding["source_mapping"]["wire_rows"]:
        for candidate in wire_row["source_edge_candidates"]:
            source_edge_id = int(candidate["source_edge_id"])
            if source_edge_id in requested:
                found[source_edge_id].append(candidate["observed_edge"])
    if any(len(found[source_edge_id]) != 1 for source_edge_id in requested):
        raise RuntimeError("target_source_edge_occurrence_not_unique_on_face")
    return [(source_edge_id, found[source_edge_id][0]) for source_edge_id in requested]


def _normalize_pair(value: Any) -> tuple[int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or value[0] == value[1]
    ):
        return None
    return tuple(sorted((int(value[0]), int(value[1]))))


def select_exact_reversed_pair_targets(
    diagnosis: Mapping[str, Any],
    *,
    source_face_index: int,
    expected_source_edge_pairs: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Select the exact adjacent+closure representation seen after sewing.

    The 63055 lineage records one physical two-edge defect twice: once as an
    ``adjacent`` occurrence and once as a ``closure`` occurrence with reversed
    source-edge order.  This selector accepts only that complete two-row
    representation.  Any extra occurrence, unavailable OCC result, ambiguous
    source mapping, or merely overlapping expected pair fails closed.
    """

    if type(source_face_index) is not int or source_face_index < 0:
        return {"accepted": False, "reason": "source_face_index_invalid"}
    if not isinstance(diagnosis, Mapping):
        return {"accepted": False, "reason": "diagnosis_not_mapping"}
    if diagnosis.get("status") != "diagnosed":
        return {"accepted": False, "reason": "diagnosis_not_completed"}
    if diagnosis.get("edge_position_basis") != "occ_1_based":
        return {"accepted": False, "reason": "diagnosis_position_basis_invalid"}
    occurrences = diagnosis.get("occurrences")
    if (
        not isinstance(occurrences, Sequence)
        or isinstance(occurrences, (str, bytes))
        or not occurrences
    ):
        return {"accepted": False, "reason": "diagnosis_occurrences_missing"}

    expected: set[tuple[int, int]] | None = None
    if expected_source_edge_pairs is not None:
        if not isinstance(expected_source_edge_pairs, Sequence) or isinstance(
            expected_source_edge_pairs, (str, bytes)
        ):
            return {"accepted": False, "reason": "expected_source_edge_pairs_invalid"}
        normalized_expected = [_normalize_pair(value) for value in expected_source_edge_pairs]
        if (
            not normalized_expected
            or any(value is None for value in normalized_expected)
            or len(set(normalized_expected)) != len(normalized_expected)
        ):
            return {"accepted": False, "reason": "expected_source_edge_pairs_invalid"}
        expected = set(normalized_expected)  # type: ignore[arg-type]

    rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, Mapping):
            return {"accepted": False, "reason": "diagnosis_occurrence_invalid"}
        if occurrence.get("kind") not in EXACT_DEFECT_KINDS:
            return {"accepted": False, "reason": "additional_defect_kind_present"}
        if occurrence.get("status") != "detected":
            return {"accepted": False, "reason": "defect_occurrence_not_detected"}
        if occurrence.get("source_mapping_status") != "mapped":
            return {"accepted": False, "reason": "defect_source_mapping_not_exact"}
        if occurrence.get("source_face_index") != source_face_index:
            return {"accepted": False, "reason": "occurrence_source_face_mismatch"}
        pair = _normalize_pair(occurrence.get("source_edge_ids"))
        if pair is None:
            return {"accepted": False, "reason": "defect_source_edge_pair_invalid"}
        positions = occurrence.get("edge_positions")
        if (
            not isinstance(positions, Sequence)
            or isinstance(positions, (str, bytes))
            or len(positions) != 2
            or any(type(value) is not int or value < 1 for value in positions)
            or positions[0] == positions[1]
        ):
            return {"accepted": False, "reason": "defect_edge_positions_invalid"}
        wire_index = occurrence.get("wire_index")
        if type(wire_index) is not int or wire_index < 0:
            return {"accepted": False, "reason": "defect_wire_index_invalid"}
        rows.append(
            {
                "kind": str(occurrence["kind"]),
                "wire_index": int(wire_index),
                "source_edge_ids": [int(value) for value in occurrence["source_edge_ids"]],
                "source_edge_pair": list(pair),
                "edge_positions": [int(value) for value in positions],
            }
        )

    grouped: dict[tuple[int, tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["wire_index"], tuple(row["source_edge_pair"]))].append(row)
    observed_pairs = {pair for (_wire, pair) in grouped}
    complete = bool(grouped) and all(
        len(values) == 2
        and Counter(row["kind"] for row in values) == Counter(EXACT_DEFECT_KINDS)
        and values[0]["source_edge_ids"] == list(reversed(values[1]["source_edge_ids"]))
        for values in grouped.values()
    )
    if not complete or sum(map(len, grouped.values())) != len(rows):
        return {"accepted": False, "reason": "exact_reversed_pair_representation_missing"}
    if expected is not None and observed_pairs != expected:
        return {
            "accepted": False,
            "reason": "expected_source_edge_pairs_mismatch",
            "observed_source_edge_pairs": [list(value) for value in sorted(observed_pairs)],
            "expected_source_edge_pairs": [list(value) for value in sorted(expected)],
        }
    target_ids = sorted({source_id for pair in observed_pairs for source_id in pair})
    return {
        "accepted": True,
        "reason": "exact_reversed_adjacent_closure_pairs_selected",
        "source_edge_pairs": [list(value) for value in sorted(observed_pairs)],
        "target_source_edge_ids": target_ids,
        "targets": rows,
    }


def _pcurve_sample(edge: Any, face: Any) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool

    result = BRep_Tool.CurveOnSurface(edge, face)
    if not isinstance(result, tuple) or len(result) != 3 or result[0] is None:
        return {"available": False, "reason": "pcurve_missing", "samples": []}
    curve, first, last = result
    first = float(first)
    last = float(last)
    if not math.isfinite(first) or not math.isfinite(last):
        return {
            "available": False,
            "reason": "nonfinite_parameter_range",
            "samples": [],
        }
    samples = []
    for position in range(CURVE_SAMPLE_COUNT):
        parameter = first + (last - first) * position / (CURVE_SAMPLE_COUNT - 1)
        point = curve.Value(parameter)
        samples.append([float(point.X()), float(point.Y())])
    if any(not math.isfinite(value) for point in samples for value in point):
        return {
            "available": False,
            "reason": "nonfinite_pcurve_sample",
            "samples": [],
        }
    return {
        "available": True,
        "curve_type": str(curve.DynamicType().Name()),
        "parameter_range": [first, last],
        "samples": samples,
    }


def _target_pcurve_snapshot(
    binding: Mapping[str, Any], source_edge_ids: Sequence[int]
) -> list[dict[str, Any]]:
    face = binding["face"]
    return [
        {"source_edge_id": source_edge_id, **_pcurve_sample(edge, face)}
        for source_edge_id, edge in _target_edges(binding, source_edge_ids)
    ]


def _face_wire_state(face: Any) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_WIRE
    from OCC.Core.TopoDS import topods

    rows = []
    failures = []
    for wire_index, wire_shape in enumerate(_subshapes(face, TopAbs_WIRE)):
        try:
            wire = topods.Wire(wire_shape)
            fixer = ShapeFix_Wire(wire, face, 0.01)
            fixer.Load(wire)
            fixer.SetFace(face)
            fixer.SetPrecision(0.01)
            fixer.SetMaxTolerance(1.0)
            fixer.SetMinTolerance(1e-4)
            fixer.Perform()
            fixed = fixer.Wire()
            analysis = ShapeAnalysis_Wire(fixed, face, 0.01)
            analysis.Load(fixed)
            analysis.SetPrecision(0.01)
            analysis.SetSurface(BRep_Tool.Surface(face))
            rows.append(
                {
                    "wire_index": wire_index,
                    "order_failure": int(analysis.CheckOrder()) != 0,
                    "self_intersection": bool(analysis.CheckSelfIntersection()),
                }
            )
        except Exception as exc:
            failures.append(
                {"wire_index": wire_index, "error_type": type(exc).__name__}
            )
    return {
        "wire_count": len(rows),
        "wire_order_failures": sum(row["order_failure"] for row in rows),
        "wire_self_intersections": sum(row["self_intersection"] for row in rows),
        "diagnostic_failures": failures,
        "accepted": bool(
            rows
            and not failures
            and not any(row["order_failure"] for row in rows)
            and not any(row["self_intersection"] for row in rows)
        ),
    }


def strict_shape_state(shape: Any) -> dict[str, Any]:
    """Measure the in-memory analogue of the project's strict STEP checks."""
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Shell
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopoDS import topods

    failures: list[str] = []
    wire_order_failures = 0
    wire_self_intersections = 0
    wire_count = 0
    try:
        for face_shape in _subshapes(shape, TopAbs_FACE):
            state = _face_wire_state(topods.Face(face_shape))
            wire_count += int(state["wire_count"])
            wire_order_failures += int(state["wire_order_failures"])
            wire_self_intersections += int(state["wire_self_intersections"])
            if state["diagnostic_failures"]:
                failures.append("wire_diagnosis_failed")
        shells_with_bad_edges = 0
        for shell_shape in _subshapes(shape, TopAbs_SHELL):
            analysis = ShapeAnalysis_Shell()
            analysis.LoadShells(topods.Shell(shell_shape))
            shells_with_bad_edges += int(bool(analysis.HasBadEdges()))
        free_bounds = ShapeAnalysis_FreeBounds(shape)
        free_edges = len(_subshapes(free_bounds.GetOpenWires(), TopAbs_EDGE))
        solid_count = len(_subshapes(shape, TopAbs_SOLID))
    except Exception as exc:
        failures.append(f"strict_measurement_failed:{type(exc).__name__}")
        shells_with_bad_edges = -1
        free_edges = -1
        solid_count = -1
    accepted = bool(
        wire_count > 0
        and not failures
        and wire_order_failures == 0
        and wire_self_intersections == 0
        and shells_with_bad_edges == 0
        and free_edges == 0
        and solid_count == 1
    )
    return {
        "accepted": accepted,
        "wire_count": wire_count,
        "wire_order_failures": wire_order_failures,
        "wire_self_intersections": wire_self_intersections,
        "shells_with_bad_edges": shells_with_bad_edges,
        "free_edges": free_edges,
        "solid_count": solid_count,
        "failure_codes": sorted(set(failures)),
    }


def _native_valid(shape: Any) -> bool:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    return bool(BRepCheck_Analyzer(shape, True).IsValid())


def _single_solid_validation_shape(shape: Any) -> tuple[Any, dict[str, Any]]:
    """Return a disposable one-solid view for the pre-``MakeSolid`` hook.

    ``construct_brep_directed`` invokes the post-sewing mutator immediately
    after ``BRepBuilderAPI_Sewing.SewedShape()`` and before its historical
    ``BRepBuilderAPI_MakeSolid`` call.  A successful sewing result is therefore
    normally a shell, not yet a solid.  Requiring ``strict_shape_state`` to see
    one solid directly on that hook value makes every otherwise viable
    candidate fail by construction.

    This helper mirrors the constructor's next stage on a disposable validation
    copy.  It never replaces or mutates the graph-preserved shell returned by
    the repair helper.  Ambiguous/multiple shells and malformed existing solids
    still fail closed.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopoDS import topods

    solids = _subshapes(shape, TopAbs_SOLID)
    shells = _subshapes(shape, TopAbs_SHELL)
    if len(solids) == 1:
        if len(shells) != 1:
            raise RuntimeError("existing_single_solid_shell_count_not_one")
        return shape, {
            "mode": "existing_single_solid",
            "input_solid_count": 1,
            "input_shell_count": 1,
            "validation_solid_count": 1,
        }
    if solids:
        raise RuntimeError("validation_input_solid_count_not_zero_or_one")
    if len(shells) != 1:
        raise RuntimeError("validation_input_shell_count_not_one")

    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(topods.Shell(shells[0]))
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("validation_solid_builder_not_done")
    solid = maker.Solid()
    if solid.IsNull():
        raise RuntimeError("validation_solid_builder_returned_null")
    validation_solids = _subshapes(solid, TopAbs_SOLID)
    validation_shells = _subshapes(solid, TopAbs_SHELL)
    if len(validation_solids) != 1 or len(validation_shells) != 1:
        raise RuntimeError("validation_solid_topology_not_single")
    return solid, {
        "mode": "wrapped_single_shell",
        "input_solid_count": 0,
        "input_shell_count": 1,
        "validation_solid_count": 1,
    }


def _reproject_target_pcurves(
    binding: Mapping[str, Any],
    source_edge_ids: Sequence[int],
    *,
    precision: float,
) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.ShapeFix import ShapeFix_Edge

    face = binding["face"]
    surface = BRepAdaptor_Surface(face)
    if bool(surface.IsUPeriodic()) or bool(surface.IsVPeriodic()):
        return {
            "accepted": False,
            "reason": "target_face_periodic_forbidden",
            "operations": [],
        }
    operations = []
    for source_edge_id, edge in _target_edges(binding, source_edge_ids):
        operation = {
            "source_edge_id": source_edge_id,
            "seam": bool(BRep_Tool.IsClosed(edge, face)),
            "pcurve_before": _pcurve_sample(edge, face),
            "remove_reported": False,
            "pcurve_absent_after_remove": False,
            "add_reported": False,
            "pcurve_present_after_add": False,
        }
        operations.append(operation)
        if operation["seam"]:
            return {"accepted": False, "reason": "target_edge_seam_forbidden", "operations": operations}
        if not operation["pcurve_before"]["available"]:
            return {"accepted": False, "reason": "target_edge_pcurve_missing", "operations": operations}
        remover = ShapeFix_Edge()
        operation["remove_reported"] = bool(remover.FixRemovePCurve(edge, face))
        after_remove = _pcurve_sample(edge, face)
        operation["pcurve_state_after_remove"] = after_remove.get("reason")
        operation["pcurve_absent_after_remove"] = (
            after_remove.get("available") is False
            and after_remove.get("reason") == "pcurve_missing"
        )
        if not operation["remove_reported"] or not operation["pcurve_absent_after_remove"]:
            return {"accepted": False, "reason": "pcurve_remove_not_proven", "operations": operations}
        adder = ShapeFix_Edge()
        projector = adder.Projector()
        projector.SetBuildCurveMode(False)
        projector.SetPrecision(float(precision))
        operation["projector_build_curve_mode"] = False
        operation["projector_precision"] = float(precision)
        operation["add_reported"] = bool(
            adder.FixAddPCurve(edge, face, False, float(precision))
        )
        operation["pcurve_after"] = _pcurve_sample(edge, face)
        operation["pcurve_present_after_add"] = operation["pcurve_after"]["available"]
        if not operation["add_reported"] or not operation["pcurve_present_after_add"]:
            return {"accepted": False, "reason": "pcurve_add_not_proven", "operations": operations}
    return {
        "accepted": bool(operations) and len(operations) == len(source_edge_ids),
        "reason": "reprojected" if operations else "no_target_operations",
        "operations": operations,
    }


def _reject(
    original_shape: Any, diagnostics: dict[str, Any], reason: str
) -> tuple[Any, dict[str, Any]]:
    diagnostics["accepted"] = False
    diagnostics["reason"] = str(reason)
    return original_shape, diagnostics


def attempt_post_sewing_face_pcurve_reprojection(
    sewn_shape: Any,
    *,
    source_face_bindings: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    target_source_face_index: int,
    target_source_edge_ids: Sequence[int],
    expected_source_edge_pairs: Sequence[Sequence[int]] | None = None,
    projection_precision: float = 1e-4,
    strict_validator: Callable[[Any], bool] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Try one copy-only face-local reprojection and otherwise return input.

    ``strict_validator``, when supplied, is called with a disposable
    single-solid validation view of the candidate.  It may strengthen, but
    never weaken, the built-in project-strict-style check.  The returned
    accepted candidate has therefore never been passed through a potentially
    mutating wire checker.
    """
    diagnostics: dict[str, Any] = {
        "strategy": "post_sewing_exact_face_pcurve_reprojection_feasibility",
        "target_source_face_index": target_source_face_index,
        "target_source_edge_ids": list(target_source_edge_ids),
        "projection_precision": projection_precision,
        "attempted": False,
        "accepted": False,
    }
    if type(target_source_face_index) is not int or target_source_face_index < 0:
        return _reject(sewn_shape, diagnostics, "target_source_face_index_invalid")
    try:
        requested_edge_ids = [int(value) for value in target_source_edge_ids]
    except Exception:
        return _reject(sewn_shape, diagnostics, "target_source_edge_ids_invalid")
    if (
        not requested_edge_ids
        or len(set(requested_edge_ids)) != len(requested_edge_ids)
        or any(value < 0 for value in requested_edge_ids)
    ):
        return _reject(sewn_shape, diagnostics, "target_source_edge_ids_invalid")
    if (
        not isinstance(projection_precision, (int, float))
        or not math.isfinite(float(projection_precision))
        or float(projection_precision) <= 0.0
    ):
        return _reject(sewn_shape, diagnostics, "projection_precision_invalid")
    if strict_validator is not None and not callable(strict_validator):
        return _reject(sewn_shape, diagnostics, "strict_validator_not_callable")

    try:
        before_topology = shape_topology_incidence_signature(sewn_shape)
        before_identity = measure_source_edge_identity(sewn_shape, source_face_bindings)
        diagnostics.update(
            before_topology=before_topology,
            before_source_edge_identity=before_identity,
        )
        if not before_identity["complete"]:
            return _reject(sewn_shape, diagnostics, "source_edge_identity_not_exact")
        before_curves = sample_source_edge_curves(sewn_shape, source_face_bindings)
        original_target = _target_binding(
            _binding_rows(source_face_bindings), target_source_face_index
        )
        _target_edges(original_target, requested_edge_ids)
        original_pcurves_before = _target_pcurve_snapshot(
            original_target, requested_edge_ids
        )

        diagnostic_copier, diagnostic_shape = _deep_copy(sewn_shape)
        diagnostic_bindings = _copy_bindings(source_face_bindings, diagnostic_copier)
        diagnostic_target = _target_binding(
            diagnostic_bindings, target_source_face_index
        )
        try:
            from .diagnose_assembly_face_wires import diagnose_face_wires_v2
            from .local_wire_topology_repair import (
                face_geometry_signature,
                geometry_preservation_gate,
            )
        except ImportError:  # direct script execution
            from diagnose_assembly_face_wires import diagnose_face_wires_v2
            from local_wire_topology_repair import (
                face_geometry_signature,
                geometry_preservation_gate,
            )

        target_diagnosis = diagnose_face_wires_v2(
            diagnostic_target["face"],
            face_index=target_source_face_index,
            source_face_index=target_source_face_index,
            source_mapping=diagnostic_target["source_mapping"],
        )
        target_selection = select_exact_reversed_pair_targets(
            target_diagnosis,
            source_face_index=target_source_face_index,
            expected_source_edge_pairs=expected_source_edge_pairs,
        )
        diagnostics["target_diagnosis"] = target_diagnosis
        diagnostics["target_selection"] = target_selection
        if not target_selection["accepted"]:
            return _reject(sewn_shape, diagnostics, str(target_selection["reason"]))
        if set(requested_edge_ids) != set(target_selection["target_source_edge_ids"]):
            return _reject(
                sewn_shape, diagnostics, "requested_edges_do_not_equal_diagnosed_targets"
            )
        target_before = _face_wire_state(diagnostic_target["face"])
        diagnostics["target_face_before"] = target_before
        if target_before["diagnostic_failures"]:
            return _reject(sewn_shape, diagnostics, "target_face_before_diagnosis_failed")
        if target_before["wire_self_intersections"] <= 0:
            return _reject(sewn_shape, diagnostics, "target_face_not_strict_invalid")
        target_geometry_before = face_geometry_signature(diagnostic_target["face"])

        copier, candidate = _deep_copy(sewn_shape)
        candidate_bindings = _copy_bindings(source_face_bindings, copier)
        candidate_topology_before = shape_topology_incidence_signature(candidate)
        candidate_identity_graph_before = _graph_inventory(candidate)
        copy_count_gate = topology_incidence_gate(
            before_topology, candidate_topology_before
        )
        copy_history_gate = copied_identity_graph_gate(sewn_shape, candidate, copier)
        copy_gate = combine_graph_gates(copy_count_gate, copy_history_gate)
        candidate_identity_before = measure_source_edge_identity(
            candidate, candidate_bindings
        )
        copy_identity_gate = source_edge_identity_gate(
            before_identity, candidate_identity_before
        )
        diagnostics["copy_topology_gate"] = copy_gate
        diagnostics["copy_source_edge_identity_gate"] = copy_identity_gate
        if not copy_gate["accepted"] or not copy_identity_gate["accepted"]:
            return _reject(sewn_shape, diagnostics, "deep_copy_graph_not_exact")

        diagnostics["attempted"] = True
        candidate_target = _target_binding(
            candidate_bindings, target_source_face_index
        )
        mutation = _reproject_target_pcurves(
            candidate_target,
            requested_edge_ids,
            precision=float(projection_precision),
        )
        diagnostics["pcurve_reprojection"] = mutation
        if not mutation["accepted"]:
            return _reject(sewn_shape, diagnostics, str(mutation["reason"]))

        after_topology = shape_topology_incidence_signature(candidate)
        after_identity = measure_source_edge_identity(candidate, candidate_bindings)
        after_curves = sample_source_edge_curves(candidate, candidate_bindings)
        topology_count_gate = topology_incidence_gate(before_topology, after_topology)
        topology_identity_gate = exact_identity_graph_gate(
            candidate_identity_graph_before, candidate
        )
        original_to_candidate_history_gate = copied_identity_graph_gate(
            sewn_shape, candidate, copier
        )
        topology_gate_result = combine_graph_gates(
            topology_count_gate,
            topology_identity_gate,
            original_to_candidate_history_gate,
        )
        identity_gate_result = source_edge_identity_gate(before_identity, after_identity)
        curve_gate_result = exact_curve_sample_gate(before_curves, after_curves)

        original_pcurves_after = _target_pcurve_snapshot(
            original_target, requested_edge_ids
        )
        original_unchanged = bool(
            original_pcurves_before
            and all(row.get("available") is True for row in original_pcurves_before)
            and original_pcurves_after == original_pcurves_before
        )

        validation_copier, validation_shape = _deep_copy(candidate)
        validation_bindings = _copy_bindings(candidate_bindings, validation_copier)
        validation_target = _target_binding(
            validation_bindings, target_source_face_index
        )
        target_after = _face_wire_state(validation_target["face"])
        target_geometry_after = face_geometry_signature(validation_target["face"])
        target_geometry_gate = geometry_preservation_gate(
            target_geometry_before, target_geometry_after
        )
        validation_solid, validation_solid_gate = _single_solid_validation_shape(
            validation_shape
        )
        strict_state = strict_shape_state(validation_solid)
        native_valid = _native_valid(validation_solid)
        external_strict_valid = True
        external_strict_error_type = None
        if strict_validator is not None:
            try:
                external_strict_valid = strict_validator(validation_solid) is True
            except Exception as exc:
                external_strict_valid = False
                external_strict_error_type = type(exc).__name__
        strict_valid = bool(strict_state["accepted"] and external_strict_valid)
        graph_gate = evaluate_graph_preservation_gate(
            topology_gate=topology_gate_result,
            source_identity_gate=identity_gate_result,
            curve_3d_gate=curve_gate_result,
            pcurve_operations_complete=mutation["accepted"],
            original_shape_unchanged=original_unchanged,
            target_face_was_invalid=target_before["wire_self_intersections"] > 0,
            target_face_is_clean=target_after["accepted"],
            native_valid=native_valid,
            strict_valid=bool(strict_valid and target_geometry_gate["accepted"]),
        )
        diagnostics.update(
            after_topology=after_topology,
            after_source_edge_identity=after_identity,
            topology_incidence_gate=topology_gate_result,
            source_edge_identity_gate=identity_gate_result,
            curve_3d_preservation=curve_gate_result,
            original_shape_unchanged=original_unchanged,
            target_face_after=target_after,
            target_geometry_before=target_geometry_before,
            target_geometry_after=target_geometry_after,
            target_geometry_preservation=target_geometry_gate,
            validation_solid_gate=validation_solid_gate,
            native_brep_valid=native_valid,
            strict_state=strict_state,
            external_strict_valid=external_strict_valid,
            external_strict_error_type=external_strict_error_type,
            graph_preservation_gate=graph_gate,
        )
        if not graph_gate["accepted"]:
            return _reject(sewn_shape, diagnostics, "graph_preservation_gate_rejected")
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        return _reject(sewn_shape, diagnostics, "feasibility_exception")

    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted_copy_only_candidate"
    return candidate, diagnostics


__all__ = [
    "CURVE_SAMPLE_COUNT",
    "attempt_post_sewing_face_pcurve_reprojection",
    "combine_graph_gates",
    "copied_identity_graph_gate",
    "evaluate_graph_preservation_gate",
    "exact_curve_sample_gate",
    "exact_identity_graph_gate",
    "measure_source_edge_identity",
    "sample_source_edge_curves",
    "select_exact_reversed_pair_targets",
    "shape_topology_incidence_signature",
    "source_edge_identity_gate",
    "strict_shape_state",
    "topology_incidence_gate",
]
