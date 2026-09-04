"""Fail-closed repair of diagnosis-selected pcurves on a non-periodic face.

This module is intentionally a face-local building block, not an assembly
profile.  A caller supplies the complete source-edge occurrence inventory and
the proof-bearing mapping made for the *same* face.  The helper independently
checks that evidence, runs the strict v2 wire diagnosis, deep-copies the face,
and rebuilds only the two pcurves named by each exact adjacent occurrence.

No CAD, face, wire, or edge identifier is special-cased.  A rejected
candidate returns the input face.  An accepted candidate is always a distinct
copy and has passed mapping, incidence, 3-D curve, non-target pcurve,
conservative geometry, native OCC, and strict wire gates.  OCC work can still
terminate a native process, so callers must retain the project's one-CAD per
subprocess policy.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .diagnose_assembly_face_wires import diagnose_face_wires_v2
from .local_wire_topology_repair import (
    corresponding_3d_curve_gate,
    face_geometry_signature,
    face_topology_incidence_signature,
    geometry_preservation_gate,
)


EXACT_MAPPING_STATUSES = frozenset(
    {
        "exact_identity",
        "exact_face_local_geometry",
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
        # Local-only provenance emitted after BRepBuilderAPI_Copy has mapped
        # every observed wire/edge through ModifiedShape. This must not be
        # called ``exact_identity``: copied shapes are deliberately distinct
        # from the input shapes even though the copy history is exact.
        "exact_copy_history",
    }
)
ALLOWED_EDGE_PROOF_METHODS = frozenset(
    {"identity", "geometry_fingerprint", "copy_modified_shape"}
)
DEFAULT_PROJECTION_PRECISION = 1e-7
PCURVE_SAMPLE_COUNT = 11
PCURVE_SAMPLE_TOLERANCE = 1e-12
CURVE_3D_SAMPLE_COUNT = 11
CURVE_3D_SAMPLE_TOLERANCE = 1e-10


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _source_pair(value: Any) -> tuple[int, int] | None:
    if not _is_sequence(value) or len(value) != 2:
        return None
    if any(type(item) is not int or item < 0 for item in value):
        return None
    first, second = map(int, value)
    if first == second:
        return None
    return tuple(sorted((first, second)))


def _expected_pair_set(
    values: Sequence[Sequence[int]] | None,
) -> tuple[set[tuple[int, int]] | None, str | None]:
    if values is None:
        return None, None
    if not _is_sequence(values):
        return None, "expected_source_edge_pairs_not_sequence"
    pairs: list[tuple[int, int]] = []
    for value in values:
        pair = _source_pair(value)
        if pair is None:
            return None, "expected_source_edge_pair_invalid"
        pairs.append(pair)
    if not pairs:
        return None, "expected_source_edge_pairs_empty"
    if len(set(pairs)) != len(pairs):
        return None, "expected_source_edge_pairs_duplicate"
    return set(pairs), None


def select_exact_adjacent_targets(
    diagnosis: Mapping[str, Any],
    *,
    source_face_index: int,
    expected_source_edge_pairs: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Select only complete, exact, mapped adjacent defect occurrences.

    The selector is deliberately strict: an adjacent repair is not attempted
    when the same face has an additional closure, non-adjacent, pcurve-gap,
    disconnected, seam, self-only, unavailable, or failed occurrence.  Pilot
    callers may bind the complete unordered pair set through
    ``expected_source_edge_pairs``; it is never interpreted as a subset.
    """

    if type(source_face_index) is not int or source_face_index < 0:
        return {"accepted": False, "reason": "source_face_index_invalid"}
    expected, expected_error = _expected_pair_set(expected_source_edge_pairs)
    if expected_error is not None:
        return {"accepted": False, "reason": expected_error}
    if not isinstance(diagnosis, Mapping):
        return {"accepted": False, "reason": "diagnosis_not_mapping"}
    if diagnosis.get("status") != "diagnosed":
        return {"accepted": False, "reason": "diagnosis_not_completed"}
    if diagnosis.get("edge_position_basis") != "occ_1_based":
        return {"accepted": False, "reason": "diagnosis_position_basis_invalid"}
    occurrences = diagnosis.get("occurrences")
    if not _is_sequence(occurrences):
        return {"accepted": False, "reason": "diagnosis_occurrences_missing"}
    if not occurrences:
        return {"accepted": False, "reason": "no_exact_adjacent_targets"}

    targets: list[dict[str, Any]] = []
    pair_keys: list[tuple[int, int]] = []
    used_source_edges: set[int] = set()
    for occurrence in occurrences:
        if not isinstance(occurrence, Mapping):
            return {"accepted": False, "reason": "diagnosis_occurrence_invalid"}
        if occurrence.get("kind") != "adjacent":
            return {
                "accepted": False,
                "reason": "non_adjacent_or_additional_defect_present",
            }
        if occurrence.get("status") != "detected":
            return {"accepted": False, "reason": "adjacent_occurrence_not_detected"}
        if occurrence.get("source_mapping_status") != "mapped":
            return {"accepted": False, "reason": "adjacent_source_mapping_not_exact"}
        if occurrence.get("source_face_index") != source_face_index:
            return {"accepted": False, "reason": "occurrence_source_face_mismatch"}
        pair = _source_pair(occurrence.get("source_edge_ids"))
        if pair is None:
            return {"accepted": False, "reason": "adjacent_source_edge_pair_invalid"}
        positions = occurrence.get("edge_positions")
        if (
            not _is_sequence(positions)
            or len(positions) != 2
            or any(type(value) is not int or value < 1 for value in positions)
            or positions[0] == positions[1]
        ):
            return {"accepted": False, "reason": "adjacent_edge_positions_invalid"}
        wire_index = occurrence.get("wire_index")
        if type(wire_index) is not int or wire_index < 0:
            return {"accepted": False, "reason": "adjacent_wire_index_invalid"}
        if pair in pair_keys:
            return {"accepted": False, "reason": "adjacent_source_pair_duplicate"}
        if used_source_edges.intersection(pair):
            return {"accepted": False, "reason": "adjacent_source_pairs_overlap"}
        used_source_edges.update(pair)
        pair_keys.append(pair)
        targets.append(
            {
                "wire_index": int(wire_index),
                "edge_positions": [int(value) for value in positions],
                "source_edge_ids": [int(value) for value in occurrence["source_edge_ids"]],
                "source_edge_pair": list(pair),
            }
        )

    observed = set(pair_keys)
    if expected is not None and observed != expected:
        return {
            "accepted": False,
            "reason": "expected_source_edge_pairs_mismatch",
            "observed_source_edge_pairs": [list(pair) for pair in sorted(observed)],
            "expected_source_edge_pairs": [list(pair) for pair in sorted(expected)],
        }
    return {
        "accepted": True,
        "reason": "exact_adjacent_targets_selected",
        "target_count": len(targets),
        "target_source_edge_ids": sorted(used_source_edges),
        "source_edge_pairs": [list(pair) for pair in sorted(observed)],
        "targets": targets,
    }


def _same(first: Any, second: Any) -> bool:
    try:
        return bool(first.IsSame(second))
    except Exception:
        return False


def _unique_identity_assignment(
    observed: Sequence[Any], candidates: Sequence[Any]
) -> list[int] | None:
    """Return a unique identity-only perfect assignment, otherwise None."""

    if len(observed) != len(candidates):
        return None
    graph = [
        [index for index, candidate in enumerate(candidates) if _same(item, candidate)]
        for item in observed
    ]
    if any(not row for row in graph):
        return None
    order = sorted(range(len(graph)), key=lambda index: len(graph[index]))
    assignment = [-1] * len(graph)
    solutions: list[list[int]] = []

    def search(depth: int, used: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if depth == len(order):
            solutions.append(list(assignment))
            return
        observed_index = order[depth]
        for candidate_index in graph[observed_index]:
            if candidate_index in used:
                continue
            assignment[observed_index] = candidate_index
            search(depth + 1, {*used, candidate_index})
            assignment[observed_index] = -1

    search(0, set())
    return solutions[0] if len(solutions) == 1 else None


def _parse_source_occurrences(
    source_edge_occurrences: Sequence[tuple[int, Any]],
) -> tuple[list[tuple[int, Any]] | None, str | None]:
    if not _is_sequence(source_edge_occurrences) or not source_edge_occurrences:
        return None, "source_edge_occurrences_missing"
    parsed: list[tuple[int, Any]] = []
    for row in source_edge_occurrences:
        if not _is_sequence(row) or len(row) != 2:
            return None, "source_edge_occurrence_invalid"
        source_id, edge = row
        if type(source_id) is not int or source_id < 0 or edge is None:
            return None, "source_edge_occurrence_invalid"
        parsed.append((int(source_id), edge))
    # A repeated source edge is a seam-like/ambiguous occurrence for this
    # narrow prototype.  It cannot safely be targeted by one pcurve rebuild.
    if len({source_id for source_id, _edge in parsed}) != len(parsed):
        return None, "source_edge_occurrence_ids_not_unique"
    return parsed, None


def _mapping_context(
    face: Any,
    source_mapping: Mapping[str, Any],
    source_edge_occurrences: Sequence[tuple[int, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Independently prove that a mapping completely describes this face."""

    from OCC.Extend.TopologyUtils import TopologyExplorer

    parsed, parse_error = _parse_source_occurrences(source_edge_occurrences)
    if parse_error is not None:
        return {"accepted": False, "reason": parse_error}, None
    assert parsed is not None
    if not isinstance(source_mapping, Mapping):
        return {"accepted": False, "reason": "source_mapping_not_mapping"}, None
    status = source_mapping.get("status")
    if status not in EXACT_MAPPING_STATUSES:
        return {
            "accepted": False,
            "reason": "source_mapping_status_not_exact",
            "mapping_status": status if isinstance(status, str) else None,
        }, None
    failures = source_mapping.get("failures")
    if not _is_sequence(failures) or failures:
        return {"accepted": False, "reason": "source_mapping_has_failures"}, None
    wire_rows = source_mapping.get("wire_rows")
    if not _is_sequence(wire_rows) or not wire_rows:
        return {"accepted": False, "reason": "source_mapping_wire_rows_missing"}, None

    actual_wires = list(TopologyExplorer(face, ignore_orientation=False).wires())
    mapped_wires: list[Any] = []
    mapped_candidates: list[dict[str, Any]] = []
    public_wire_source_ids: list[list[int]] = []
    for wire_row in wire_rows:
        if not isinstance(wire_row, Mapping) or wire_row.get("observed_wire") is None:
            return {"accepted": False, "reason": "source_mapping_wire_row_invalid"}, None
        candidates = wire_row.get("source_edge_candidates")
        if not _is_sequence(candidates) or not candidates:
            return {
                "accepted": False,
                "reason": "source_mapping_edge_candidates_missing",
            }, None
        wire_candidates: list[dict[str, Any]] = []
        wire_ids: list[int] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                return {
                    "accepted": False,
                    "reason": "source_mapping_edge_candidate_invalid",
                }, None
            source_id = candidate.get("source_edge_id")
            edge = candidate.get("observed_edge")
            proof = candidate.get("proof_method")
            if (
                type(source_id) is not int
                or source_id < 0
                or edge is None
                or proof not in ALLOWED_EDGE_PROOF_METHODS
            ):
                return {
                    "accepted": False,
                    "reason": "source_mapping_edge_candidate_invalid",
                }, None
            value = {
                "source_edge_id": int(source_id),
                "observed_edge": edge,
                "proof_method": str(proof),
            }
            mapped_candidates.append(value)
            wire_candidates.append(value)
            wire_ids.append(int(source_id))
        observed_wire = wire_row["observed_wire"]
        actual_edges = list(
            TopologyExplorer(observed_wire, ignore_orientation=False).edges()
        )
        if _unique_identity_assignment(
            actual_edges, [row["observed_edge"] for row in wire_candidates]
        ) is None:
            return {
                "accepted": False,
                "reason": "source_mapping_wire_edges_not_unique_complete",
            }, None
        mapped_wires.append(observed_wire)
        public_wire_source_ids.append(wire_ids)

    if _unique_identity_assignment(actual_wires, mapped_wires) is None:
        return {
            "accepted": False,
            "reason": "source_mapping_wires_not_unique_complete",
        }, None
    source_ids = [source_id for source_id, _edge in parsed]
    mapped_ids = [row["source_edge_id"] for row in mapped_candidates]
    if Counter(source_ids) != Counter(mapped_ids):
        return {
            "accepted": False,
            "reason": "source_mapping_occurrence_multiset_mismatch",
        }, None
    proof_methods = source_mapping.get("edge_proof_methods")
    if (
        not _is_sequence(proof_methods)
        or list(map(str, proof_methods))
        != [row["proof_method"] for row in mapped_candidates]
    ):
        return {
            "accepted": False,
            "reason": "source_mapping_proof_method_sequence_mismatch",
        }, None

    source_by_id = {source_id: edge for source_id, edge in parsed}
    mapped_by_id = {row["source_edge_id"]: row["observed_edge"] for row in mapped_candidates}
    curve_rows = []
    for source_id in sorted(source_by_id):
        gate = _strict_corresponding_3d_curve_gate(
            [source_by_id[source_id]], [mapped_by_id[source_id]]
        )
        curve_rows.append(
            {
                "source_edge_id": int(source_id),
                "accepted": bool(gate.get("accepted")),
                "max_sample_delta": gate.get("max_sample_delta"),
            }
        )
        if not gate.get("accepted"):
            return {
                "accepted": False,
                "reason": "source_mapping_3d_curve_binding_failed",
                "curve_bindings": curve_rows,
            }, None
    context = {
        "status": str(status),
        "wire_rows": list(wire_rows),
        "mapped_candidates": mapped_candidates,
        "mapped_by_id": mapped_by_id,
        "source_by_id": source_by_id,
        "parsed_occurrences": parsed,
    }
    return {
        "accepted": True,
        "reason": "source_mapping_complete",
        "mapping_status": str(status),
        "wire_count": len(mapped_wires),
        "edge_occurrence_count": len(mapped_candidates),
        "source_edge_ids": sorted(source_ids),
        "wire_source_edge_ids": public_wire_source_ids,
        "curve_bindings": curve_rows,
    }, context


def _surface_nonperiodic_gate(face: Any) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool

    surface = BRep_Tool.Surface(face)
    if surface is None or (hasattr(surface, "IsNull") and surface.IsNull()):
        return {"accepted": False, "reason": "face_surface_missing"}
    u_periodic = bool(surface.IsUPeriodic())
    v_periodic = bool(surface.IsVPeriodic())
    return {
        "accepted": not (u_periodic or v_periodic),
        "reason": (
            "surface_nonperiodic"
            if not (u_periodic or v_periodic)
            else "periodic_surface_forbidden"
        ),
        "u_periodic": u_periodic,
        "v_periodic": v_periodic,
    }


def _pcurve_fingerprint(edge: Any, face: Any) -> dict[str, Any]:
    from OCC.Core.BRep import BRep_Tool

    pcurve, first, last = BRep_Tool.CurveOnSurface(edge, face)
    if pcurve is None or (hasattr(pcurve, "IsNull") and pcurve.IsNull()):
        return {"available": False, "reason": "pcurve_missing"}
    first_value = float(first)
    last_value = float(last)
    if not (math.isfinite(first_value) and math.isfinite(last_value)):
        return {"available": False, "reason": "nonfinite_parameter_range"}
    samples: list[list[float]] = []
    for index in range(PCURVE_SAMPLE_COUNT):
        parameter = first_value + (last_value - first_value) * index / (
            PCURVE_SAMPLE_COUNT - 1
        )
        point = pcurve.Value(parameter)
        uv = [float(point.X()), float(point.Y())]
        if not all(math.isfinite(value) for value in uv):
            return {"available": False, "reason": "nonfinite_pcurve_sample"}
        samples.append(uv)
    return {
        "available": True,
        "range": [first_value, last_value],
        "samples": samples,
    }


def _pcurve_fingerprints_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if first.get("available") is not True or second.get("available") is not True:
        # Two unavailable observations are not proof that a non-target
        # pcurve was preserved. In particular, a nonfinite pcurve and an
        # absent pcurve used to compare equal here.
        return False
    try:
        first_range = [float(value) for value in first["range"]]
        second_range = [float(value) for value in second["range"]]
        first_samples = first["samples"]
        second_samples = second["samples"]
        deltas = [
            math.dist([float(value) for value in left], [float(value) for value in right])
            for left, right in zip(first_samples, second_samples)
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(first_range) == len(second_range) == 2
        and len(first_samples) == len(second_samples) == PCURVE_SAMPLE_COUNT
        and max(
            [abs(left - right) for left, right in zip(first_range, second_range)]
            + deltas,
            default=float("inf"),
        )
        <= PCURVE_SAMPLE_TOLERANCE
    )


def _curve_3d_fingerprint(edge: Any) -> dict[str, Any]:
    """Return finite 3-D curve evidence, failing closed on every sample."""

    from OCC.Core.BRep import BRep_Tool

    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None or (hasattr(curve, "IsNull") and curve.IsNull()):
        return {"available": False, "reason": "curve_3d_missing"}
    first_value = float(first)
    last_value = float(last)
    if not (math.isfinite(first_value) and math.isfinite(last_value)):
        return {"available": False, "reason": "nonfinite_parameter_range"}
    samples: list[list[float]] = []
    for index in range(CURVE_3D_SAMPLE_COUNT):
        parameter = first_value + (last_value - first_value) * index / (
            CURVE_3D_SAMPLE_COUNT - 1
        )
        point = curve.Value(parameter)
        xyz = [float(point.X()), float(point.Y()), float(point.Z())]
        if not all(math.isfinite(value) for value in xyz):
            return {"available": False, "reason": "nonfinite_curve_3d_sample"}
        samples.append(xyz)
    return {
        "available": True,
        "curve_type": str(curve.DynamicType().Name()),
        "range": [first_value, last_value],
        "samples": samples,
    }


def _strict_corresponding_3d_curve_gate(
    original_edges: Sequence[Any], copied_edges: Sequence[Any]
) -> dict[str, Any]:
    """Strengthen the shared curve gate with explicit finite evidence."""

    gate = corresponding_3d_curve_gate(list(original_edges), list(copied_edges))
    before = [_curve_3d_fingerprint(edge) for edge in original_edges]
    after = [_curve_3d_fingerprint(edge) for edge in copied_edges]
    all_finite = bool(
        len(before) == len(after)
        and before
        and all(row.get("available") is True for row in before + after)
    )
    try:
        maximum_delta = float(gate.get("max_sample_delta"))
    except (TypeError, ValueError):
        maximum_delta = float("nan")
    maximum_delta_finite = math.isfinite(maximum_delta)
    within_explicit_tolerance = bool(
        maximum_delta_finite and maximum_delta <= CURVE_3D_SAMPLE_TOLERANCE
    )
    accepted = bool(
        gate.get("accepted") and all_finite and within_explicit_tolerance
    )
    public_gate = dict(gate)
    public_gate["max_sample_delta"] = (
        maximum_delta if maximum_delta_finite else None
    )
    if _is_sequence(public_gate.get("edges")):
        public_edges = []
        for edge_row in public_gate["edges"]:
            row = dict(edge_row) if isinstance(edge_row, Mapping) else {}
            try:
                row_delta = float(row.get("max_sample_delta"))
            except (TypeError, ValueError):
                row_delta = float("nan")
            row["max_sample_delta"] = row_delta if math.isfinite(row_delta) else None
            public_edges.append(row)
        public_gate["edges"] = public_edges
    return {
        **public_gate,
        "accepted": accepted,
        "reason": "accepted" if accepted else "curve_changed_or_not_finite",
        "all_curves_available_and_finite": all_finite,
        "max_sample_delta_finite": maximum_delta_finite,
        "max_sample_delta_within_explicit_tolerance": within_explicit_tolerance,
        "explicit_sample_tolerance": CURVE_3D_SAMPLE_TOLERANCE,
    }


def _copy_face_and_mapping(
    face: Any, context: Mapping[str, Any]
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any]]:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.TopoDS import topods

    copier = BRepBuilderAPI_Copy(face, True, False)
    candidate = topods.Face(copier.Shape())
    if candidate.IsNull() or candidate.IsSame(face):
        return None, None, {"accepted": False, "reason": "face_copy_not_distinct"}
    copied_rows = []
    proof_methods: list[str] = []
    for wire_row in context["wire_rows"]:
        copied_wire_shape = copier.ModifiedShape(wire_row["observed_wire"])
        if copied_wire_shape.IsNull():
            return None, None, {
                "accepted": False,
                "reason": "copied_wire_mapping_missing",
            }
        copied_candidates = []
        for row in wire_row["source_edge_candidates"]:
            copied_edge_shape = copier.ModifiedShape(row["observed_edge"])
            if copied_edge_shape.IsNull():
                return None, None, {
                    "accepted": False,
                    "reason": "copied_edge_mapping_missing",
                }
            copied_candidates.append(
                {
                    "source_edge_id": int(row["source_edge_id"]),
                    "observed_edge": topods.Edge(copied_edge_shape),
                    "proof_method": "copy_modified_shape",
                }
            )
            proof_methods.append("copy_modified_shape")
        copied_rows.append(
            {
                "observed_wire": topods.Wire(copied_wire_shape),
                "source_edge_candidates": copied_candidates,
            }
        )
    copied_mapping = {
        "status": "exact_copy_history",
        "upstream_mapping_status": str(context["status"]),
        "wire_rows": copied_rows,
        "failures": [],
        "edge_proof_methods": proof_methods,
    }
    return candidate, copied_mapping, {
        "accepted": True,
        "reason": "distinct_face_copy_with_complete_mapping",
        "mapping_status": "exact_copy_history",
        "upstream_mapping_status": str(context["status"]),
    }


def _diagnosis_identity_view(source_mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Make a same-copied-face identity view for the existing v2 diagnoser."""

    wire_rows = []
    proof_methods: list[str] = []
    for wire_row in source_mapping["wire_rows"]:
        candidates = []
        for row in wire_row["source_edge_candidates"]:
            candidates.append(
                {
                    "source_edge_id": int(row["source_edge_id"]),
                    "observed_edge": row["observed_edge"],
                    "proof_method": "identity",
                }
            )
            proof_methods.append("identity")
        wire_rows.append(
            {
                "observed_wire": wire_row["observed_wire"],
                "source_edge_candidates": candidates,
            }
        )
    return {
        "status": "exact_identity",
        "wire_rows": wire_rows,
        "failures": [],
        "edge_proof_methods": proof_methods,
        "authoritative_mapping_status": source_mapping.get("status"),
    }


def _edge_preflight(
    face: Any,
    context: Mapping[str, Any],
    target_source_edge_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]] | None]:
    from OCC.Core.BRep import BRep_Tool

    target_ids = set(map(int, target_source_edge_ids))
    rows: dict[int, dict[str, Any]] = {}
    for source_id, edge in context["mapped_by_id"].items():
        fingerprint = _pcurve_fingerprint(edge, face)
        curve_3d_fingerprint = _curve_3d_fingerprint(edge)
        target = source_id in target_ids
        row = {
            "source_edge_id": int(source_id),
            "target": target,
            "pcurve_available": bool(fingerprint.get("available")),
            "seam": bool(BRep_Tool.IsClosed(edge, face)),
            "pcurve_fingerprint": fingerprint,
            "curve_3d_fingerprint": curve_3d_fingerprint,
        }
        if target:
            row["curve_3d_available"] = bool(
                curve_3d_fingerprint.get("available") is True
            )
            if row["seam"]:
                return {
                    "accepted": False,
                    "reason": "target_seam_edge_forbidden",
                    "source_edge_id": int(source_id),
                }, None
            if not row["pcurve_available"]:
                return {
                    "accepted": False,
                    "reason": "target_pcurve_missing_or_nonfinite",
                    "source_edge_id": int(source_id),
                }, None
            if not row["curve_3d_available"]:
                return {
                    "accepted": False,
                    "reason": "target_3d_curve_missing_or_nonfinite",
                    "source_edge_id": int(source_id),
                }, None
        rows[int(source_id)] = row
    if set(rows).intersection(target_ids) != target_ids:
        return {"accepted": False, "reason": "target_source_edge_not_mapped"}, None
    return {
        "accepted": True,
        "reason": "target_edges_nonseam_with_pcurves_and_3d_curves",
        "target_source_edge_ids": sorted(target_ids),
        "checked_edge_count": len(rows),
    }, rows


def _shape_fix_status(tool: Any) -> dict[str, bool]:
    from OCC.Core.ShapeExtend import (
        ShapeExtend_DONE,
        ShapeExtend_DONE1,
        ShapeExtend_DONE2,
        ShapeExtend_FAIL,
        ShapeExtend_FAIL1,
        ShapeExtend_FAIL2,
        ShapeExtend_OK,
    )

    return {
        "ok": bool(tool.Status(ShapeExtend_OK)),
        "done": bool(tool.Status(ShapeExtend_DONE)),
        "done1": bool(tool.Status(ShapeExtend_DONE1)),
        "done2": bool(tool.Status(ShapeExtend_DONE2)),
        "fail": bool(tool.Status(ShapeExtend_FAIL)),
        "fail1": bool(tool.Status(ShapeExtend_FAIL1)),
        "fail2": bool(tool.Status(ShapeExtend_FAIL2)),
    }


def _rebuild_one_pcurve(
    edge: Any,
    face: Any,
    *,
    source_edge_id: int,
    projection_precision: float,
) -> dict[str, Any]:
    from OCC.Core.ShapeFix import ShapeFix_Edge

    remove_tool = ShapeFix_Edge()
    removed = bool(remove_tool.FixRemovePCurve(edge, face))
    remove_status = _shape_fix_status(remove_tool)
    after_remove = _pcurve_fingerprint(edge, face)
    if (
        not removed
        or remove_status["fail"]
        or after_remove.get("available") is not False
        or after_remove.get("reason") != "pcurve_missing"
    ):
        return {
            "accepted": False,
            "reason": "pcurve_remove_not_proven",
            "source_edge_id": int(source_edge_id),
            "remove_returned": removed,
            "remove_status": remove_status,
            "pcurve_absent_after_remove": after_remove.get("available") is False,
            "pcurve_state_after_remove": after_remove.get("reason"),
        }

    add_tool = ShapeFix_Edge()
    projector = add_tool.Projector()
    projector.SetBuildCurveMode(False)
    projector.SetPrecision(float(projection_precision))
    added = bool(
        add_tool.FixAddPCurve(edge, face, False, float(projection_precision))
    )
    add_status = _shape_fix_status(add_tool)
    after_add = _pcurve_fingerprint(edge, face)
    accepted = bool(
        added
        and not add_status["fail"]
        and after_add.get("available") is True
    )
    return {
        "accepted": accepted,
        "reason": "pcurve_rebuilt" if accepted else "pcurve_add_not_proven",
        "source_edge_id": int(source_edge_id),
        "remove_returned": removed,
        "remove_status": remove_status,
        "pcurve_absent_after_remove": True,
        "pcurve_state_after_remove": "pcurve_missing",
        "add_returned": added,
        "add_status": add_status,
        "pcurve_present_after_add": after_add.get("available") is True,
        "projector_build_curve_mode": False,
        "projection_precision": float(projection_precision),
    }


def _strict_clean_gate(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    if diagnosis.get("status") != "diagnosed":
        return {"accepted": False, "reason": "post_repair_diagnosis_not_completed"}
    wires = diagnosis.get("wires")
    occurrences = diagnosis.get("occurrences")
    if not _is_sequence(wires) or not _is_sequence(occurrences):
        return {"accepted": False, "reason": "post_repair_diagnosis_incomplete"}
    aggregate = [row.get("aggregate_self_intersection") for row in wires]
    accepted = bool(
        wires
        and not occurrences
        and all(value is False for value in aggregate)
    )
    return {
        "accepted": accepted,
        "reason": "strict_face_clean" if accepted else "strict_face_defect_remains",
        "checked_wire_count": len(wires),
        "occurrence_count": len(occurrences),
        "aggregate_self_intersections": aggregate,
    }


def _non_target_pcurve_gate(
    before_rows: Mapping[int, Mapping[str, Any]],
    candidate_context: Mapping[str, Any],
    candidate_face: Any,
    target_source_edge_ids: Sequence[int],
) -> dict[str, Any]:
    targets = set(map(int, target_source_edge_ids))
    changed = []
    checked = 0
    for source_id, edge in candidate_context["mapped_by_id"].items():
        if source_id in targets:
            continue
        checked += 1
        before = before_rows[source_id]["pcurve_fingerprint"]
        after = _pcurve_fingerprint(edge, candidate_face)
        if not _pcurve_fingerprints_equal(before, after):
            changed.append(int(source_id))
    return {
        "accepted": not changed,
        "reason": "non_target_pcurves_preserved" if not changed else "non_target_pcurve_changed",
        "checked_edge_count": checked,
        "changed_source_edge_ids": changed,
        "sample_tolerance": PCURVE_SAMPLE_TOLERANCE,
    }


def _reject(reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "attempted": bool(evidence.get("surgery")),
        "accepted": False,
        "reason": str(reason),
        "strategy": "targeted_nonperiodic_pcurve_reprojection",
        **evidence,
    }


def repair_face_targeted_nonperiodic_pcurves(
    face: Any,
    *,
    source_face_index: int,
    source_mapping: Mapping[str, Any],
    source_edge_occurrences: Sequence[tuple[int, Any]],
    expected_source_edge_pairs: Sequence[Sequence[int]] | None = None,
    projection_precision: float = DEFAULT_PROJECTION_PRECISION,
) -> tuple[Any, dict[str, Any]]:
    """Reproject only exact adjacent, non-seam pcurves on a copied face.

    ``expected_source_edge_pairs`` is an optional pilot binding.  When given,
    its unordered pair set must equal the complete diagnosis-derived set.  It
    is not a production eligibility rule and never narrows the diagnosis.
    """

    try:
        if type(source_face_index) is not int or source_face_index < 0:
            return face, _reject("source_face_index_invalid")
        precision = float(projection_precision)
        if not math.isfinite(precision) or precision <= 0.0:
            return face, _reject("projection_precision_invalid")

        mapping_gate, context = _mapping_context(
            face, source_mapping, source_edge_occurrences
        )
        if context is None:
            return face, _reject(mapping_gate["reason"], source_mapping_gate=mapping_gate)

        surface_gate = _surface_nonperiodic_gate(face)
        if not surface_gate["accepted"]:
            return face, _reject(
                surface_gate["reason"],
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
            )

        diagnosis = diagnose_face_wires_v2(
            face,
            face_index=source_face_index,
            source_face_index=source_face_index,
            source_mapping=source_mapping,
        )
        selection = select_exact_adjacent_targets(
            diagnosis,
            source_face_index=source_face_index,
            expected_source_edge_pairs=expected_source_edge_pairs,
        )
        if not selection["accepted"]:
            return face, _reject(
                selection["reason"],
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
                target_selection=selection,
            )

        target_ids = selection["target_source_edge_ids"]
        preflight, before_rows = _edge_preflight(face, context, target_ids)
        if before_rows is None:
            return face, _reject(
                preflight["reason"],
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
                target_selection=selection,
                edge_preflight=preflight,
            )

        before_geometry = face_geometry_signature(face)
        before_topology = face_topology_incidence_signature(face)
        original_edges = [
            context["mapped_by_id"][source_id]
            for source_id in sorted(context["mapped_by_id"])
        ]
        candidate, copied_mapping, copy_gate = _copy_face_and_mapping(face, context)
        if candidate is None or copied_mapping is None:
            return face, _reject(
                copy_gate["reason"],
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
                target_selection=selection,
                edge_preflight=preflight,
                copy_gate=copy_gate,
            )

        candidate_mapping_gate, candidate_context = _mapping_context(
            candidate, copied_mapping, source_edge_occurrences
        )
        if candidate_context is None:
            return face, _reject(
                "copied_source_mapping_incomplete",
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
                target_selection=selection,
                edge_preflight=preflight,
                copy_gate=copy_gate,
                candidate_mapping_gate=candidate_mapping_gate,
            )

        surgery = []
        for source_id in target_ids:
            row = _rebuild_one_pcurve(
                candidate_context["mapped_by_id"][source_id],
                candidate,
                source_edge_id=source_id,
                projection_precision=precision,
            )
            surgery.append(row)
            if not row["accepted"]:
                return face, _reject(
                    row["reason"],
                    source_mapping_gate=mapping_gate,
                    surface_gate=surface_gate,
                    target_selection=selection,
                    edge_preflight=preflight,
                    copy_gate=copy_gate,
                    candidate_mapping_gate=candidate_mapping_gate,
                    surgery=surgery,
                )

        # Re-prove the complete mapping after mutation; do not assume that an
        # in-place OCC method retained identity merely because it usually does.
        post_mapping_gate, post_context = _mapping_context(
            candidate, copied_mapping, source_edge_occurrences
        )
        if post_context is None:
            return face, _reject(
                "post_repair_source_mapping_incomplete",
                source_mapping_gate=mapping_gate,
                surface_gate=surface_gate,
                target_selection=selection,
                edge_preflight=preflight,
                copy_gate=copy_gate,
                candidate_mapping_gate=candidate_mapping_gate,
                surgery=surgery,
                post_repair_mapping_gate=post_mapping_gate,
            )

        diagnosis_mapping = _diagnosis_identity_view(copied_mapping)
        after_diagnosis = diagnose_face_wires_v2(
            candidate,
            face_index=source_face_index,
            source_face_index=source_face_index,
            source_mapping=diagnosis_mapping,
        )
        strict_gate = _strict_clean_gate(after_diagnosis)
        non_target_gate = _non_target_pcurve_gate(
            before_rows, post_context, candidate, target_ids
        )
        copied_edges = [
            post_context["mapped_by_id"][source_id]
            for source_id in sorted(post_context["mapped_by_id"])
        ]
        curve_gate = _strict_corresponding_3d_curve_gate(
            original_edges, copied_edges
        )
        after_topology = face_topology_incidence_signature(candidate)
        topology_gate = {
            "accepted": before_topology == after_topology,
            "reason": (
                "topology_incidence_preserved"
                if before_topology == after_topology
                else "topology_incidence_changed"
            ),
            "before": before_topology,
            "after": after_topology,
        }
        after_geometry = face_geometry_signature(candidate)
        geometry_gate = geometry_preservation_gate(before_geometry, after_geometry)

        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        native_valid = bool(BRepCheck_Analyzer(candidate, True).IsValid())
        copy_distinct = not candidate.IsSame(face)
        accepted = bool(
            copy_distinct
            and post_mapping_gate["accepted"]
            and strict_gate["accepted"]
            and non_target_gate["accepted"]
            and curve_gate["accepted"]
            and topology_gate["accepted"]
            and geometry_gate["accepted"]
            and native_valid
        )
        diagnostics = {
            "attempted": True,
            "accepted": accepted,
            "reason": "accepted" if accepted else "post_repair_acceptance_gate_failed",
            "strategy": "targeted_nonperiodic_pcurve_reprojection",
            "source_mapping_gate": mapping_gate,
            "surface_gate": surface_gate,
            "target_selection": selection,
            "edge_preflight": preflight,
            "copy_gate": copy_gate,
            "candidate_mapping_gate": candidate_mapping_gate,
            "surgery": surgery,
            "post_repair_mapping_gate": post_mapping_gate,
            "strict_face_gate": strict_gate,
            "non_target_pcurve_gate": non_target_gate,
            "curve_3d_preservation": curve_gate,
            "topology_incidence_gate": topology_gate,
            "geometry_preservation": geometry_gate,
            "candidate_native_brep_valid": native_valid,
            "candidate_copy_distinct": copy_distinct,
        }
        return (candidate if accepted else face), diagnostics
    except Exception as exc:
        return face, _reject(
            "occ_or_evidence_exception",
            error_type=type(exc).__name__,
        )


__all__ = [
    "DEFAULT_PROJECTION_PRECISION",
    "repair_face_targeted_nonperiodic_pcurves",
    "select_exact_adjacent_targets",
]
