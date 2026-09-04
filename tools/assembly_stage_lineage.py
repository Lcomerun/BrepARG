"""Pure, Git-safe evidence contract for the seven assembly census stages.

The assembly census needs to distinguish two very different outcomes.  A
malformed worker record is a protocol failure, while a well-formed record that
shows a topology change or cannot prove a unique source mapping is scientific
evidence.  This module keeps that distinction explicit without importing Open
CASCADE or any assembly mutation code.

Callers first reduce native objects to JSON-like stage records, then call
``assess_stage_lineage``.  The returned value contains only finite, path-free
Python scalars and containers and can therefore be placed in a Git-safe report.
Every stage must have exactly one record.  A stage that could not run because
an earlier stage failed is represented by ``make_not_reached_stage`` rather
than being omitted from the denominator.
"""

from __future__ import annotations

import math
import numbers
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


STAGE_RECORD_SCHEMA = "assembly-stage-lineage-record-v1"
TOPOLOGY_CENSUS_SCHEMA = "assembly-stage-topology-census-v1"
ASSESSMENT_SCHEMA = "assembly-stage-lineage-assessment-v1"

STAGE_ORDER = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
STAGE_NAMES = {
    "S1": "post_surface_curve_fit_pre_edge_build",
    "S2": "post_edge_build_pre_face_build",
    "S3": "post_add_pcurves_pre_optional_face_repair",
    "S4": "post_optional_face_repair_pre_sewing",
    "S5": "post_sewing_pre_solid",
    "S6": "post_solid_pre_step",
    "S7": "post_step_roundtrip_strict",
}
STAGE_BY_NAME = {name: stage for stage, name in STAGE_NAMES.items()}

# S1-S4 are distributed observations bound to the source entity iterated by
# the unchanged constructor.  The edge loop owns S1/S2 and the face loop owns
# S3/S4.  A structurally valid scope naming the other entity population must
# therefore remain scientific ambiguity instead of being allowed to support an
# exact prefix or terminal claim.  S5-S7 are whole-shape boundaries and retain
# their existing (non-distributed) protocol behavior.
_DISTRIBUTED_ENTITY_KIND_BY_STAGE = {
    "S1": "source_edge",
    "S2": "source_edge",
    "S3": "source_face",
    "S4": "source_face",
}

EXACT_LINEAGE_STATUSES = frozenset(
    {
        "exact",
        "exact_identity",
        "exact_face_local_geometry",
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
        "exact_geometry_incidence",
    }
)
# Distributed construction boundaries can stop at two semantically different
# places.  ``exact_prefix`` means every entity reached so far passed this
# boundary, but a later interleaved boundary stopped full traversal.  It is
# useful preceding evidence, not itself a bad stage.  By contrast,
# ``local_exact_failure`` says this boundary failed at its terminal entity.
PREFIX_PASS_LINEAGE_STATUSES = frozenset({"exact_prefix", "exact_prefix_pass"})
LOCAL_FAILURE_LINEAGE_STATUSES = frozenset({"local_exact_failure"})
PREFIX_LINEAGE_STATUSES = frozenset(
    {*PREFIX_PASS_LINEAGE_STATUSES, *LOCAL_FAILURE_LINEAGE_STATUSES}
)
INCONCLUSIVE_LINEAGE_STATUSES = frozenset(
    {
        "ambiguous",
        "mapped",
        "missing",
        "unavailable",
        "nonunique",
        "split",
        "merge",
    }
)

TOPOLOGY_COUNT_FIELDS = (
    "face_count",
    "edge_count",
    "vertex_count",
    "face_edge_occurrence_count",
)
TOPOLOGY_INCIDENCE_FIELDS = (
    "face_edge_incidence_counts",
    "edge_face_incidence_counts",
    "vertex_edge_incidence_counts",
)
TOPOLOGY_RELATION_FIELDS = (
    "face_edge_source_ids",
    "edge_face_source_ids",
    "edge_vertex_source_ids",
)
TOPOLOGY_FIELDS = (
    *TOPOLOGY_COUNT_FIELDS,
    *TOPOLOGY_INCIDENCE_FIELDS,
    *TOPOLOGY_RELATION_FIELDS,
)

# Before faces exist, only the corresponding source-bound entity census can be
# observed.  Once faces exist, the full incidence census is expected unless a
# scientific construction failure prevented it from being measured.
STAGE_REQUIRED_TOPOLOGY_FIELDS = {
    "S1": ("face_count",),
    "S2": ("edge_count", "vertex_count", "vertex_edge_incidence_counts"),
    "S3": TOPOLOGY_FIELDS,
    "S4": TOPOLOGY_FIELDS,
    "S5": TOPOLOGY_FIELDS,
    "S6": TOPOLOGY_FIELDS,
    "S7": TOPOLOGY_FIELDS,
}

# A full-stage exact claim must enumerate the source-bound populations that
# actually exist at that construction boundary.  S1 has fitted surfaces and
# curves, S2 has curves materialized as edges, and S3 onward has face/edge
# occurrences.  The occurrence key is the canonical triple
# ``(source_face_id, source_edge_id, duplicate_ordinal)``.  The ordinal only
# distinguishes a seam or other repeated use of one edge on the same face; it
# is not an OCC explorer position.
STAGE_REQUIRED_LINEAGE_POPULATIONS = {
    "S1": ("source_face_ids", "source_edge_ids"),
    "S2": ("source_edge_ids",),
    "S3": (
        "source_face_ids",
        "source_edge_ids",
        "source_edge_occurrence_keys",
    ),
    "S4": (
        "source_face_ids",
        "source_edge_ids",
        "source_edge_occurrence_keys",
    ),
    "S5": (
        "source_face_ids",
        "source_edge_ids",
        "source_edge_occurrence_keys",
    ),
    "S6": (
        "source_face_ids",
        "source_edge_ids",
        "source_edge_occurrence_keys",
    ),
    "S7": (
        "source_face_ids",
        "source_edge_ids",
        "source_edge_occurrence_keys",
    ),
}

# These scalar measurements are the JSON-safe residue of borrowed OCC handle
# comparisons performed inside the isolated child.  From S3 onward, copying a
# source edge's endpoint labels into the topology relation is insufficient:
# exact lineage also requires proof that the observed edge endpoints are the
# same unordered OCC endpoint pair seen at S2.  Sewing may additionally split
# or merge shared vertices, so S5/S6 require one globally unique source-to-
# observed vertex assignment across every edge occurrence.
EDGE_ENDPOINT_IDENTITY_PROOF_METHOD = "s2_to_stage_occ_unordered_pair_IsSame"
STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD = (
    "source_edge_endpoint_labels_to_stage_local_occ_identity_classes_v1"
)
GLOBAL_VERTEX_IDENTITY_PROOF_METHOD = (
    "source_edge_endpoint_constraints_plus_target_vertex_IsSame_"
    "unique_perfect_assignment"
)
STEP_VERTEX_IDENTITY_PROOF_METHOD = (
    "unique_global_incident_mapped_edge_multiset_and_3d_point_v1"
)
STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED = 1e-4
GLOBAL_VERTEX_IDENTITY_PROOF_KEYS = frozenset(
    {
        "status",
        "proof_method",
        "solution_count",
        "solution_count_capped_at_two",
        "source_vertex_count",
        "observed_vertex_count",
        "mapped_source_vertex_count",
        "mapped_observed_vertex_count",
        "max_observed_per_source",
        "max_source_per_observed",
        "constraint_occurrence_count",
        "failure_codes",
    }
)
STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS = frozenset(
    {
        "status",
        "proof_method",
        "scope_kind",
        "scope_count",
        "source_edge_count",
        "constraint_occurrence_count",
        "max_observed_per_source_within_scope",
        "max_source_per_observed_within_scope",
        "failure_codes",
    }
)
STEP_GEOMETRY_INCIDENCE_PROOF_KEYS = frozenset(
    {
        "status",
        "failure_codes",
        "tolerance_normalized",
        "face_candidate_degree_counts",
        "face_matching_count_capped",
        "vertex_proof_required",
        "vertex_proof_method",
        "vertex_tolerance_normalized",
        "vertex_candidate_degree_counts",
        "vertex_matching_count_capped",
        "source_vertex_count",
        "step_vertex_count",
        "mapped_source_edge_count",
        "edge_endpoint_pair_expected_count",
        "edge_endpoint_pair_proof_count",
        "edge_endpoint_occurrence_expected_count",
        "edge_endpoint_occurrence_proof_count",
        "self_loop_endpoint_pair_expected_count",
        "self_loop_endpoint_pair_proof_count",
        "mapped_face_count",
        "mapped_edge_occurrence_count",
        "vertex_proof_status",
    }
)
STAGE_ENDPOINT_EVIDENCE_REQUIREMENTS = {
    "S2": "stage_local_occ_topology",
    "S3": "stage_local_occ_topology",
    "S4": "stage_local_occ_topology",
    "S5": "global_vertex_identity",
    "S6": "global_vertex_identity",
    "S7": "step_geometry_incidence_identity",
}

_TOPOLOGY_ALIASES = {
    "face_edge_occurrences": "face_edge_occurrence_count",
    "edge_vertex_incidence_counts": "vertex_edge_incidence_counts",
}
_LINEAGE_STATUS_ALIASES = {
    "multiple": "nonunique",
    "not_unique": "nonunique",
    "not-unique": "nonunique",
    "not_found": "missing",
    "unmapped": "missing",
}
_PATH_KEYS = {
    "path",
    "paths",
    "source_path",
    "step_path",
    "pickle_path",
    "output_dir",
    "checkpoint_path",
}
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")
_PATH_IN_TEXT = re.compile(
    r"(?i)(?:file:///?[A-Z]:/[^\s'\"<>|]*|[A-Z]:[\\/][^\s'\"<>|]*|"
    r"\\\\[^\\/\s]+[\\/][^\s'\"<>|]*|"
    r"(?<![A-Za-z0-9_.-])/(?!/)(?:[^/\s'\"<>|]+/)*[^/\s'\"<>|]+)"
)
_NATIVE_REPR_IN_TEXT = re.compile(
    r"(?i)(?:\b0x[0-9a-f]{6,}\b|SwigPyObject|\bthisown\b|"
    r"<[^>]*(?:TopoDS|Geom_|BRep|Handle_)[^>]*>|\bHandle_[A-Za-z0-9_]+)"
)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any, *, name: str) -> int:
    if not _finite_number(value) or int(value) != float(value):
        raise ValueError(f"{name} must be a finite integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _path_like_string(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return bool(
        normalized.startswith("/")
        or normalized.startswith("//")
        or _WINDOWS_ABSOLUTE.match(value)
        or _PATH_IN_TEXT.search(value)
    )


def _native_repr_string(value: str) -> bool:
    """Return true for common serialized pythonocc/native handle forms."""

    return bool(_NATIVE_REPR_IN_TEXT.search(value))


def redact_path_and_native_text(value: str) -> str:
    """Return bounded diagnostic text with path and native repr tokens removed.

    Archival code should prefer stable error codes.  This helper exists for the
    few human-readable exception reasons that remain useful: it redacts tokens
    wherever they occur in the string, rather than checking only its prefix.
    The result is suitable for :func:`assert_path_free_finite`.
    """

    if not isinstance(value, str):
        raise TypeError("diagnostic text must be a string")
    result = _PATH_IN_TEXT.sub("<redacted-path>", value)
    result = _NATIVE_REPR_IN_TEXT.sub("<redacted-native>", result)
    return result[:240]


def assert_path_free_finite(value: Any, *, label: str = "evidence") -> None:
    """Reject paths, non-finite numbers, native handles, and non-JSON values.

    This validator intentionally accepts tuples as input because observers
    often construct tuples, but normalization converts every sequence to a
    list.  Mapping keys must be strings so no lossy key conversion can occur.
    """

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, numbers.Integral):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, str):
        if _path_like_string(value):
            raise ValueError(f"{label} contains an absolute path")
        if _native_repr_string(value):
            raise ValueError(f"{label} contains a native handle representation")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string mapping key")
            key_lower = key.lower()
            if (
                key_lower in _PATH_KEYS
                or key_lower.endswith("_path")
                or key_lower.endswith("_paths")
            ):
                raise ValueError(f"{label} contains forbidden path key {key!r}")
            assert_path_free_finite(child, label=f"{label}.{key}")
        return
    if _is_sequence(value):
        for index, child in enumerate(value):
            assert_path_free_finite(child, label=f"{label}[{index}]")
        return
    raise TypeError(f"{label} contains non-JSON value {type(value).__name__}")


def _json_value(value: Any) -> Any:
    """Copy a validated JSON-like value while converting numeric scalars."""

    assert_path_free_finite(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if _is_sequence(value):
        return [_json_value(child) for child in value]
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _string_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not _is_sequence(value):
        raise TypeError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain non-empty strings")
        if item not in result:
            result.append(item)
    return result


def _normalize_stage_code(stage: Any, phase: Any = None) -> str:
    stage_value = str(stage) if stage is not None else None
    phase_value = str(phase) if phase is not None else None
    if stage_value is None:
        stage_value = STAGE_BY_NAME.get(phase_value or "")
    if stage_value not in STAGE_ORDER:
        raise ValueError("stage is not registered in the seven-stage census")
    if phase_value is not None and phase_value != STAGE_NAMES[stage_value]:
        raise ValueError("stage and phase disagree")
    return stage_value


def _topology_input_counts(census: Mapping[str, Any]) -> Mapping[str, Any]:
    counts = census.get("counts")
    if counts is None:
        return census
    if not isinstance(counts, Mapping):
        raise TypeError("topology counts must be a mapping")
    return counts


def _normalize_incidence(
    value: Any, *, name: str, expected_length: int | None
) -> list[int]:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be an integer sequence")
    result = [_integer(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(f"{name} length does not match its entity count")
    # Explorer order is not identity.  The census records the incidence
    # multiset, so sorting makes comparison independent of traversal order.
    return sorted(result)


def _normalize_id_rows(
    value: Any,
    *,
    name: str,
    row_count: int,
    member_upper_bound: int,
    row_width: int | None = None,
    preserve_member_duplicates: bool = False,
) -> list[list[int]]:
    """Normalize an indexed relation while preserving its entity identities."""

    if not _is_sequence(value) or len(value) != row_count:
        raise ValueError(f"{name} must contain exactly {row_count} rows")
    rows: list[list[int]] = []
    for row_index, raw_row in enumerate(value):
        if not _is_sequence(raw_row):
            raise TypeError(f"{name}[{row_index}] must be an integer sequence")
        row = [
            _integer(item, name=f"{name}[{row_index}][{item_index}]")
            for item_index, item in enumerate(raw_row)
        ]
        if row_width is not None and len(row) != row_width:
            raise ValueError(f"{name}[{row_index}] must contain {row_width} ids")
        if any(item >= member_upper_bound for item in row):
            raise ValueError(f"{name}[{row_index}] contains an out-of-range id")
        if not preserve_member_duplicates and len(row) != len(set(row)):
            raise ValueError(f"{name}[{row_index}] contains duplicate ids")
        rows.append(sorted(row))
    return rows


def _inverse_face_edge_relation(
    face_edge_source_ids: Sequence[Sequence[int]], edge_count: int
) -> list[list[int]]:
    inverse = [[] for _ in range(edge_count)]
    for face_id, edge_ids in enumerate(face_edge_source_ids):
        for edge_id in edge_ids:
            inverse[int(edge_id)].append(int(face_id))
    return [sorted(row) for row in inverse]


def validate_topology_census(
    census: Mapping[str, Any],
    *,
    source_topology: Mapping[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Normalize and compare a topology census without treating drift as error.

    Malformed counts or internally inconsistent incidence sums raise an error:
    they are protocol defects.  A well-formed difference from
    ``source_topology`` is returned in ``drifted_fields`` and is a scientific
    observation.  It never appears as a protocol failure.
    """

    if not isinstance(census, Mapping):
        raise TypeError("topology census must be a mapping")
    assert_path_free_finite(census, label="topology census")
    raw = _topology_input_counts(census)
    aliased: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _TOPOLOGY_ALIASES.get(str(key), str(key))
        if canonical in TOPOLOGY_FIELDS:
            if canonical in aliased and aliased[canonical] != value:
                raise ValueError(f"conflicting topology field {canonical}")
            aliased[canonical] = value

    counts: dict[str, Any] = {}
    for key in TOPOLOGY_COUNT_FIELDS:
        if key in aliased:
            counts[key] = _integer(aliased[key], name=key)

    length_by_incidence = {
        "face_edge_incidence_counts": counts.get("face_count"),
        "edge_face_incidence_counts": counts.get("edge_count"),
        "vertex_edge_incidence_counts": counts.get("vertex_count"),
    }
    for key in TOPOLOGY_INCIDENCE_FIELDS:
        if key in aliased:
            expected = length_by_incidence[key]
            if expected is None:
                raise ValueError(f"{key} requires its corresponding entity count")
            counts[key] = _normalize_incidence(
                aliased[key], name=key, expected_length=expected
            )

    relation_required = require_complete or any(
        field in aliased for field in TOPOLOGY_RELATION_FIELDS
    )
    if relation_required:
        missing_relations = [
            field for field in TOPOLOGY_RELATION_FIELDS if field not in aliased
        ]
        if missing_relations:
            raise ValueError(
                "topology canonical relations are incomplete: "
                + ", ".join(missing_relations)
            )

    face_count = counts.get("face_count")
    edge_count = counts.get("edge_count")
    vertex_count = counts.get("vertex_count")
    if "face_edge_source_ids" in aliased:
        if face_count is None or edge_count is None:
            raise ValueError("face_edge_source_ids requires face_count and edge_count")
        counts["face_edge_source_ids"] = _normalize_id_rows(
            aliased["face_edge_source_ids"],
            name="face_edge_source_ids",
            row_count=face_count,
            member_upper_bound=edge_count,
            # A seam can use the same source edge twice on one source face.
            preserve_member_duplicates=True,
        )
    if "edge_face_source_ids" in aliased:
        if face_count is None or edge_count is None:
            raise ValueError("edge_face_source_ids requires face_count and edge_count")
        counts["edge_face_source_ids"] = _normalize_id_rows(
            aliased["edge_face_source_ids"],
            name="edge_face_source_ids",
            row_count=edge_count,
            member_upper_bound=face_count,
            # Match repeated occurrences in face_edge_source_ids exactly.
            preserve_member_duplicates=True,
        )
    if "edge_vertex_source_ids" in aliased:
        if edge_count is None or vertex_count is None:
            raise ValueError("edge_vertex_source_ids requires edge_count and vertex_count")
        counts["edge_vertex_source_ids"] = _normalize_id_rows(
            aliased["edge_vertex_source_ids"],
            name="edge_vertex_source_ids",
            row_count=edge_count,
            member_upper_bound=vertex_count,
            row_width=2,
            # A closed edge legitimately has the same endpoint twice.
            preserve_member_duplicates=True,
        )

    present_fields = [field for field in TOPOLOGY_FIELDS if field in counts]
    missing_fields = [field for field in TOPOLOGY_FIELDS if field not in counts]
    if require_complete and missing_fields:
        raise ValueError(
            "topology census is incomplete: " + ", ".join(missing_fields)
        )

    occurrence_count = counts.get("face_edge_occurrence_count")
    face_incidence = counts.get("face_edge_incidence_counts")
    edge_incidence = counts.get("edge_face_incidence_counts")
    if occurrence_count is not None and face_incidence is not None:
        if sum(face_incidence) != occurrence_count:
            raise ValueError("face incidence sum disagrees with occurrence count")
    if occurrence_count is not None and edge_incidence is not None:
        if sum(edge_incidence) != occurrence_count:
            raise ValueError("edge incidence sum disagrees with occurrence count")
    vertex_incidence = counts.get("vertex_edge_incidence_counts")
    if edge_count is not None and vertex_incidence is not None:
        if sum(vertex_incidence) != 2 * edge_count:
            raise ValueError("vertex incidence sum must equal two per edge")

    face_edge_ids = counts.get("face_edge_source_ids")
    edge_face_ids = counts.get("edge_face_source_ids")
    edge_vertex_ids = counts.get("edge_vertex_source_ids")
    if face_edge_ids is not None:
        relation_occurrences = sum(len(row) for row in face_edge_ids)
        if occurrence_count is not None and relation_occurrences != occurrence_count:
            raise ValueError("face-edge relation disagrees with occurrence count")
        derived_face_incidence = sorted(len(row) for row in face_edge_ids)
        if face_incidence is not None and derived_face_incidence != face_incidence:
            raise ValueError("face-edge relation disagrees with face incidence")
        derived_inverse = _inverse_face_edge_relation(face_edge_ids, edge_count)
        if edge_face_ids is not None and derived_inverse != edge_face_ids:
            raise ValueError("face-edge and edge-face relations are not inverses")
        derived_edge_incidence = sorted(len(row) for row in derived_inverse)
        if edge_incidence is not None and derived_edge_incidence != edge_incidence:
            raise ValueError("face-edge relation disagrees with edge incidence")
    if edge_face_ids is not None:
        relation_occurrences = sum(len(row) for row in edge_face_ids)
        if occurrence_count is not None and relation_occurrences != occurrence_count:
            raise ValueError("edge-face relation disagrees with occurrence count")
        derived_edge_incidence = sorted(len(row) for row in edge_face_ids)
        if edge_incidence is not None and derived_edge_incidence != edge_incidence:
            raise ValueError("edge-face relation disagrees with edge incidence")
    if edge_vertex_ids is not None:
        derived_vertex_incidence = [0] * vertex_count
        for endpoints in edge_vertex_ids:
            # Iterating both entries counts a self-loop twice, as required by
            # the graph degree identity sum(degree) == 2 * edge_count.
            for vertex_id in endpoints:
                derived_vertex_incidence[vertex_id] += 1
        if vertex_incidence is not None and sorted(derived_vertex_incidence) != vertex_incidence:
            raise ValueError("edge-vertex relation disagrees with vertex incidence")

    source_counts: Mapping[str, Any] | None = None
    if source_topology is not None:
        if not isinstance(source_topology, Mapping):
            raise TypeError("source topology must be a mapping")
        source_normalized = validate_topology_census(
            source_topology, require_complete=True
        )
        source_counts = source_normalized["counts"]
    comparable_fields = [
        field for field in present_fields if source_counts is not None and field in source_counts
    ]
    drifted_fields = [
        field for field in comparable_fields if counts[field] != source_counts[field]
    ]
    result = {
        "schema": TOPOLOGY_CENSUS_SCHEMA,
        "counts": counts,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "complete": not missing_fields,
        "comparable_fields": comparable_fields,
        "drifted_fields": drifted_fields,
        "compared_to_source": source_counts is not None,
        "matches_source": None if source_counts is None else not drifted_fields,
    }
    assert_path_free_finite(result, label="normalized topology census")
    return result


def _lineage_entity(value: Any, *, name: str) -> tuple[dict[str, int], list[str]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"lineage entity {name} must be a mapping")
    allowed = (
        "source_count",
        "observed_count",
        "mapped_source_count",
        "mapped_observed_count",
        "max_observed_per_source",
        "max_source_per_observed",
        "solution_count",
    )
    result = {
        key: _integer(value[key], name=f"lineage.entities.{name}.{key}")
        for key in allowed
        if key in value
    }
    reasons: list[str] = []
    if result.get("max_observed_per_source", 0) > 1:
        reasons.append("split")
    if result.get("max_source_per_observed", 0) > 1:
        reasons.append("merge")
    if "solution_count" in result:
        if result["solution_count"] == 0:
            reasons.append("missing")
        elif result["solution_count"] > 1:
            reasons.append("nonunique")
    if (
        "mapped_source_count" in result
        and "source_count" in result
        and result["mapped_source_count"] < result["source_count"]
    ):
        reasons.append("missing")
    if (
        "mapped_observed_count" in result
        and "observed_count" in result
        and result["mapped_observed_count"] < result["observed_count"]
    ):
        reasons.append("missing")
    return result, list(dict.fromkeys(reasons))


def _normalize_id_set(value: Any, *, name: str) -> list[int]:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be an integer sequence")
    result = [_integer(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate ids")
    if result != sorted(result):
        raise ValueError(f"{name} must be in canonical sorted order")
    return result


def _normalize_occurrence_keys(value: Any, *, name: str) -> list[list[int]]:
    if not _is_sequence(value):
        raise TypeError(f"{name} must be a sequence")
    result: list[list[int]] = []
    for index, raw in enumerate(value):
        if not _is_sequence(raw) or len(raw) != 3:
            raise ValueError(
                f"{name}[{index}] must be [source_face_id, source_edge_id, duplicate_ordinal]"
            )
        row = [
            _integer(item, name=f"{name}[{index}][{offset}]")
            for offset, item in enumerate(raw)
        ]
        result.append(row)
    if len(result) != len({tuple(row) for row in result}):
        raise ValueError(f"{name} contains duplicate occurrence keys")
    if result != sorted(result):
        raise ValueError(f"{name} must be in canonical sorted order")
    return result


def _expected_occurrence_keys(source_topology: Mapping[str, Any]) -> list[list[int]]:
    counts = source_topology.get("counts", source_topology)
    face_rows = counts.get("face_edge_source_ids")
    if not _is_sequence(face_rows):
        raise ValueError("source topology lacks canonical face-edge relations")
    result: list[list[int]] = []
    for face_id, edge_ids in enumerate(face_rows):
        per_edge: Counter[int] = Counter()
        for edge_id in edge_ids:
            ordinal = per_edge[int(edge_id)]
            result.append([int(face_id), int(edge_id), int(ordinal)])
            per_edge[int(edge_id)] += 1
    return sorted(result)


def _normalize_distributed_scope(
    raw: Mapping[str, Any], *, stage: str, source_topology: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str]]:
    value = raw.get("distributed_scope")
    if value is None:
        return None, []
    if not isinstance(value, Mapping):
        raise TypeError("distributed_scope must be a mapping")
    assert_path_free_finite(value, label="distributed_scope")
    entity_kind = value.get("entity_kind")
    if entity_kind not in {"source_edge", "source_face"}:
        raise ValueError("distributed_scope entity_kind must be source_edge or source_face")
    expected_ids = _normalize_id_set(value.get("expected_ids"), name="expected_ids")
    completed_value = value.get("completed_ids", value.get("observed_ids"))
    completed_ids = _normalize_id_set(completed_value, name="completed_ids")
    terminal = value.get("terminal_failure_entity_id")
    if terminal is not None:
        terminal = _integer(terminal, name="terminal_failure_entity_id")
    preceding_prefix = value.get("preceding_stage_prefix_verified", False)
    if type(preceding_prefix) is not bool:
        raise TypeError("preceding_stage_prefix_verified must be a boolean")
    event_sequence = value.get("event_sequence_proof")
    if event_sequence is None:
        raw_events = []
    elif not isinstance(event_sequence, Mapping):
        raise TypeError("distributed_scope event_sequence_proof must be a mapping")
    else:
        raw_events = event_sequence.get("events")
        if not _is_sequence(raw_events):
            raise TypeError("event_sequence_proof.events must be a sequence")
    events: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise TypeError(f"event_sequence_proof.events[{index}] must be a mapping")
        entity_id = _integer(
            raw_event.get("entity_id"),
            name=f"event_sequence_proof.events[{index}].entity_id",
        )
        event = raw_event.get("event")
        if event not in {"pre_boundary_ok", "post_boundary_ok", "terminal_failure"}:
            raise ValueError("event_sequence_proof.events contains an invalid event")
        events.append({"entity_id": entity_id, "event": event})

    # S1 is explicitly the pre-MakeEdge boundary; all other distributed
    # source-entity observations are post-boundary.  Treating S1 as a generic
    # post event would erase the distinction between curve fit and edge build
    # in the very failure interval this census is designed to localize.
    completed_event = "pre_boundary_ok" if stage == "S1" else "post_boundary_ok"

    reasons: list[str] = []
    expected_entity_kind = _DISTRIBUTED_ENTITY_KIND_BY_STAGE.get(stage)
    if (
        expected_entity_kind is not None
        and entity_kind != expected_entity_kind
    ):
        reasons.append("distributed_entity_kind_mismatch_stage")
    if source_topology is not None:
        counts = source_topology.get("counts", source_topology)
        expected_population = (
            int(counts["edge_count"])
            if entity_kind == "source_edge"
            else int(counts["face_count"])
        )
        if expected_ids != list(range(expected_population)):
            reasons.append("distributed_expected_ids_mismatch_source")
    full_coverage = completed_ids == expected_ids and terminal is None
    prefix_pass_exact = False
    local_failure_exact = False
    if terminal is not None:
        if terminal not in expected_ids:
            reasons.append("terminal_failure_entity_out_of_range")
        else:
            position = expected_ids.index(terminal)
            if completed_ids != expected_ids[:position]:
                reasons.append("distributed_failure_prefix_not_historical_order")
            if not preceding_prefix:
                reasons.append("distributed_preceding_stage_prefix_not_verified")
            if events:
                expected_events = [
                    *(
                        {"entity_id": entity_id, "event": completed_event}
                        for entity_id in completed_ids
                    ),
                    {"entity_id": terminal, "event": "terminal_failure"},
                ]
                if events != expected_events:
                    reasons.append("distributed_failure_event_sequence_not_exact")
            local_failure_exact = not reasons
    else:
        if completed_ids != expected_ids[: len(completed_ids)]:
            reasons.append("distributed_completed_ids_not_historical_prefix")
        if events:
            expected_events = [
                {"entity_id": entity_id, "event": completed_event}
                for entity_id in completed_ids
            ]
            if events != expected_events:
                reasons.append("distributed_pass_event_sequence_not_exact")
        if not full_coverage:
            if not preceding_prefix:
                reasons.append("distributed_prefix_cause_not_verified")
            prefix_pass_exact = not reasons
    return {
        "entity_kind": entity_kind,
        "expected_ids": expected_ids,
        "completed_ids": completed_ids,
        "terminal_failure_entity_id": terminal,
        "preceding_stage_prefix_verified": preceding_prefix,
        "event_sequence_proof": {"events": events},
        "full_coverage": full_coverage,
        "prefix_pass_exact": prefix_pass_exact,
        "local_failure_exact": local_failure_exact,
    }, reasons


def _normalize_lineage(
    value: Any,
    *,
    stage: str,
    source_topology: Mapping[str, Any] | None,
    fallback_failures: Any = None,
) -> dict[str, Any]:
    if isinstance(value, str):
        raw: Mapping[str, Any] = {"status": value}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError("observed stage lineage must be a mapping or status string")
    assert_path_free_finite(raw, label="lineage")
    # Normalized records retain the original claim separately from the
    # effective status.  Prefer it on a second pass so normalization is
    # idempotent instead of turning e.g. ``mapped -> ambiguous`` into a new
    # ``ambiguous`` claim with different reasons.
    reported = str(raw.get("reported_status", raw.get("status")) or "").strip().lower()
    reported = _LINEAGE_STATUS_ALIASES.get(reported, reported)
    if reported in EXACT_LINEAGE_STATUSES:
        canonical_reported = "exact"
    elif reported in PREFIX_LINEAGE_STATUSES:
        canonical_reported = reported
    elif reported in INCONCLUSIVE_LINEAGE_STATUSES:
        canonical_reported = reported
    else:
        raise ValueError(f"unregistered lineage status {reported!r}")

    failures = _string_list(
        raw.get("failure_codes", raw.get("failures", fallback_failures)),
        name="lineage failure codes",
    )
    proof_method = raw.get("proof_method", raw.get("method"))
    if proof_method is not None and (
        not isinstance(proof_method, str) or not proof_method.strip()
    ):
        raise ValueError("lineage proof_method must be a non-empty string")
    solution_count_value = raw.get("solution_count")
    solution_count = (
        None
        if solution_count_value is None
        else _integer(solution_count_value, name="lineage.solution_count")
    )
    source_face_value = raw.get("source_face_ids")
    source_face_ids = (
        None
        if source_face_value is None
        else _normalize_id_set(source_face_value, name="lineage.source_face_ids")
    )
    source_edge_value = raw.get("source_edge_ids")
    source_edge_ids = (
        None
        if source_edge_value is None
        else _normalize_id_set(source_edge_value, name="lineage.source_edge_ids")
    )
    occurrence_value = raw.get("source_edge_occurrence_keys")
    occurrence_keys = (
        None
        if occurrence_value is None
        else _normalize_occurrence_keys(
            occurrence_value,
            name="lineage.source_edge_occurrence_keys",
        )
    )
    distributed_scope, scope_reasons = _normalize_distributed_scope(
        raw, stage=stage, source_topology=source_topology
    )
    whole_stage_terminal = raw.get("whole_stage_terminal")
    whole_stage_reasons: list[str] = []
    if whole_stage_terminal is not None:
        if not isinstance(whole_stage_terminal, Mapping):
            raise TypeError("whole_stage_terminal must be a mapping")
        assert_path_free_finite(whole_stage_terminal, label="whole_stage_terminal")
        expected_keys = {
            "scope_kind", "boundary_stage", "prerequisite_stage",
            "prerequisite_exact", "construction_exception_observed",
        }
        if set(whole_stage_terminal) != expected_keys:
            whole_stage_reasons.append("whole_stage_terminal_fields_mismatch")
        expected_prerequisite = {"S5": "S4", "S6": "S5"}.get(stage)
        if stage not in {"S5", "S6"}:
            whole_stage_reasons.append("whole_stage_terminal_stage_not_supported")
        if whole_stage_terminal.get("scope_kind") != "whole_shape_boundary_failure":
            whole_stage_reasons.append("whole_stage_terminal_scope_kind_mismatch")
        if whole_stage_terminal.get("boundary_stage") != stage:
            whole_stage_reasons.append("whole_stage_terminal_boundary_mismatch")
        if whole_stage_terminal.get("prerequisite_stage") != expected_prerequisite:
            whole_stage_reasons.append("whole_stage_terminal_prerequisite_mismatch")
        if whole_stage_terminal.get("prerequisite_exact") is not True:
            whole_stage_reasons.append("whole_stage_terminal_prerequisite_not_exact")
        if whole_stage_terminal.get("construction_exception_observed") is not True:
            whole_stage_reasons.append("whole_stage_terminal_exception_not_observed")
        whole_stage_terminal = dict(whole_stage_terminal)
    entities_value = raw.get("entities") or {}
    if not isinstance(entities_value, Mapping):
        raise TypeError("lineage entities must be a mapping")
    entities: dict[str, dict[str, int]] = {}
    cardinality_reasons: list[str] = []
    for entity_name, entity_value in entities_value.items():
        if not isinstance(entity_name, str) or not entity_name:
            raise ValueError("lineage entity names must be non-empty strings")
        normalized_entity, reasons = _lineage_entity(entity_value, name=entity_name)
        entities[entity_name] = normalized_entity
        cardinality_reasons.extend(reasons)

    reasons = []
    if canonical_reported not in {"exact", *PREFIX_LINEAGE_STATUSES}:
        reasons.append(f"lineage_{canonical_reported}")
    reasons.extend(f"lineage_{reason}" for reason in cardinality_reasons)
    reasons.extend(f"lineage_failure:{code}" for code in failures)
    reasons.extend(f"lineage_{reason}" for reason in scope_reasons)
    reasons.extend(f"lineage_{reason}" for reason in whole_stage_reasons)

    full_claim = canonical_reported == "exact"
    if full_claim:
        if proof_method is None:
            reasons.append("lineage_proof_method_missing")
        if solution_count != 1:
            reasons.append("lineage_solution_count_not_one")
        if source_topology is None:
            reasons.append("lineage_source_topology_missing")
        else:
            counts = source_topology.get("counts", source_topology)
            expected_faces = list(range(int(counts["face_count"])))
            expected_edges = list(range(int(counts["edge_count"])))
            required = STAGE_REQUIRED_LINEAGE_POPULATIONS[stage]
            if "source_face_ids" in required and source_face_ids != expected_faces:
                reasons.append("lineage_source_face_coverage_incomplete")
            if "source_edge_ids" in required and source_edge_ids != expected_edges:
                reasons.append("lineage_source_edge_coverage_incomplete")
            if "source_edge_occurrence_keys" in required:
                try:
                    expected_occurrences = _expected_occurrence_keys(source_topology)
                except (TypeError, ValueError):
                    reasons.append("lineage_source_occurrence_contract_unavailable")
                else:
                    if occurrence_keys != expected_occurrences:
                        reasons.append("lineage_source_occurrence_coverage_incomplete")
        if distributed_scope is not None and not distributed_scope["full_coverage"]:
            reasons.append("lineage_full_claim_has_partial_distributed_scope")

    prefix_pass_claim = canonical_reported in PREFIX_PASS_LINEAGE_STATUSES
    local_failure_claim = canonical_reported in LOCAL_FAILURE_LINEAGE_STATUSES
    if prefix_pass_claim or local_failure_claim:
        if proof_method is None:
            reasons.append("lineage_proof_method_missing")
        if solution_count != 1:
            reasons.append("lineage_solution_count_not_one")
        if source_topology is None:
            reasons.append("lineage_source_topology_missing")
        elif stage in {"S2", "S3", "S4"} and distributed_scope is not None:
            counts = source_topology.get("counts", source_topology)
            completed_ids = distributed_scope["completed_ids"]
            if stage == "S2":
                if source_edge_ids != completed_ids:
                    reasons.append("lineage_source_edge_prefix_census_mismatch")
            else:
                expected_face_ids = completed_ids
                expected_occurrences: list[list[int]] = []
                for face_id in expected_face_ids:
                    duplicate_ordinals: Counter[int] = Counter()
                    for raw_edge_id in counts["face_edge_source_ids"][face_id]:
                        edge_id = int(raw_edge_id)
                        expected_occurrences.append(
                            [face_id, edge_id, int(duplicate_ordinals[edge_id])]
                        )
                        duplicate_ordinals[edge_id] += 1
                expected_occurrences.sort()
                expected_edge_ids = sorted(
                    {row[1] for row in expected_occurrences}
                )
                if source_face_ids != expected_face_ids:
                    reasons.append("lineage_source_face_prefix_census_mismatch")
                if source_edge_ids != expected_edge_ids:
                    reasons.append("lineage_source_edge_prefix_census_mismatch")
                if occurrence_keys != expected_occurrences:
                    reasons.append("lineage_source_occurrence_prefix_census_mismatch")
    if prefix_pass_claim and (
        distributed_scope is None or not distributed_scope["prefix_pass_exact"]
    ):
        reasons.append("lineage_exact_prefix_scope_not_proven")
    if local_failure_claim:
        whole_stage_exact = whole_stage_terminal is not None and not whole_stage_reasons
        distributed_exact = (
            distributed_scope is not None
            and distributed_scope["local_failure_exact"]
        )
        if whole_stage_terminal is not None and distributed_scope is not None:
            reasons.append("lineage_terminal_scope_conflict")
        if not whole_stage_exact and not distributed_exact:
            reasons.append("lineage_local_failure_scope_not_proven")
    reasons = list(dict.fromkeys(reasons))

    effective_status = canonical_reported
    if cardinality_reasons:
        unique = set(cardinality_reasons)
        if {"split", "merge"}.issubset(unique):
            effective_status = "nonunique"
        else:
            effective_status = next(
                status
                for status in ("split", "merge", "nonunique", "missing")
                if status in unique
            )
    elif reasons:
        effective_status = "ambiguous"
    elif failures and effective_status == "exact":
        effective_status = "ambiguous"
    exact = effective_status == "exact" and not reasons and not failures
    prefix_pass_exact = (
        effective_status in PREFIX_PASS_LINEAGE_STATUSES
        and not reasons
        and not failures
    )
    local_failure_exact = (
        effective_status in LOCAL_FAILURE_LINEAGE_STATUSES
        and not reasons
        and not failures
    )
    return {
        "reported_status": reported,
        "status": effective_status,
        "classification": (
            "exact"
            if exact
            else "exact_prefix"
            if prefix_pass_exact
            else "local_exact_failure"
            if local_failure_exact
            else "inconclusive"
        ),
        "exact": exact,
        "prefix_pass_exact": prefix_pass_exact,
        "local_failure_exact": local_failure_exact,
        "proof_method": proof_method,
        "method": proof_method,
        "solution_count": solution_count,
        "source_face_ids": source_face_ids,
        "source_edge_ids": source_edge_ids,
        "source_edge_occurrence_keys": occurrence_keys,
        "distributed_scope": distributed_scope,
        "whole_stage_terminal": whole_stage_terminal,
        "failure_codes": failures,
        "entities": entities,
        "inconclusive_reasons": reasons,
    }


def _normalize_failure(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError("stage failure reason must be non-empty")
        return {"kind": "stage_failure", "reason": value}
    if not isinstance(value, Mapping):
        raise TypeError("stage failure must be a string or mapping")
    normalized = _json_value(value)
    kind = normalized.get("kind", normalized.get("code", "stage_failure"))
    reason = normalized.get("reason", normalized.get("message", kind))
    if not isinstance(kind, str) or not kind:
        raise ValueError("stage failure kind must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise ValueError("stage failure reason must be a non-empty string")
    normalized["kind"] = kind
    normalized["reason"] = reason
    return normalized


def _normalize_defects(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not _is_sequence(value):
        raise TypeError("defects must be a sequence")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            if not item:
                raise ValueError("defect codes must be non-empty")
            result.append({"code": item})
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"defects[{index}] must be a string or mapping")
        normalized = _json_value(item)
        code = normalized.get("code", normalized.get("kind"))
        if not isinstance(code, str) or not code:
            raise ValueError(f"defects[{index}] needs a non-empty code")
        normalized["code"] = code
        result.append(normalized)
    return result


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _source_identity_scope(
    stage: str,
    lineage: Mapping[str, Any],
    source_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact source population reached by one lineage claim.

    A distributed prefix is not compared with whole-CAD cardinalities.  Its
    completed source faces determine the edge occurrences that actually
    crossed the boundary, and those occurrences in turn determine the unique
    source edges and shared source vertices for which native identity proof is
    required.  S2 is edge-distributed, so its completed edge IDs are the
    corresponding constraint population directly.
    """

    counts = source_topology.get("counts", source_topology)
    face_rows = counts["face_edge_source_ids"]
    edge_rows = counts["edge_vertex_source_ids"]
    distributed = lineage.get("distributed_scope")
    partial_claim = bool(
        lineage.get("prefix_pass_exact") is True
        or lineage.get("local_failure_exact") is True
    )
    if partial_claim:
        if not isinstance(distributed, Mapping):
            raise ValueError("distributed lineage scope is unavailable")
        completed_ids = [int(value) for value in distributed.get("completed_ids") or []]
    else:
        completed_ids = []

    if stage == "S2":
        edge_ids = (
            completed_ids
            if partial_claim
            else list(range(int(counts["edge_count"])))
        )
        face_ids: list[int] = []
        occurrence_keys: list[list[int]] = []
        constraint_count = len(edge_ids)
    else:
        face_ids = (
            completed_ids
            if partial_claim and stage in {"S3", "S4"}
            else list(range(int(counts["face_count"])))
        )
        occurrence_keys = []
        for face_id in face_ids:
            duplicate_ordinals: Counter[int] = Counter()
            for raw_edge_id in face_rows[face_id]:
                edge_id = int(raw_edge_id)
                occurrence_keys.append(
                    [face_id, edge_id, int(duplicate_ordinals[edge_id])]
                )
                duplicate_ordinals[edge_id] += 1
        occurrence_keys.sort()
        edge_ids = sorted({row[1] for row in occurrence_keys})
        constraint_count = len(occurrence_keys)

    vertex_ids = sorted(
        {
            int(vertex_id)
            for edge_id in edge_ids
            for vertex_id in edge_rows[edge_id]
        }
    )
    return {
        "face_ids": face_ids,
        "edge_ids": edge_ids,
        "occurrence_keys": occurrence_keys,
        "constraint_count": constraint_count,
        "vertex_ids": vertex_ids,
    }


def _global_vertex_identity_reasons(
    vertex: Any,
    *,
    expected_vertex_count: int,
    expected_constraint_count: int,
) -> list[str]:
    """Validate the public scalar residue of one native shared-vertex proof."""

    if not isinstance(vertex, Mapping):
        return ["global_source_vertex_lineage_missing"]
    reasons: list[str] = []
    if set(vertex) - GLOBAL_VERTEX_IDENTITY_PROOF_KEYS:
        reasons.append("global_source_vertex_evidence_contains_unknown_fields")
    if vertex.get("proof_method") != GLOBAL_VERTEX_IDENTITY_PROOF_METHOD:
        reasons.append("global_source_vertex_proof_method_mismatch")
    if vertex.get("status") != "exact_identity":
        reasons.append("global_source_vertex_status_not_exact")
    if vertex.get("solution_count_capped_at_two") is not True:
        reasons.append("global_source_vertex_solution_cap_not_proven")
    try:
        solution_count = _integer(
            vertex.get("solution_count"), name="source_vertex_lineage.solution_count"
        )
        observed_count = _integer(
            vertex.get("observed_vertex_count"),
            name="source_vertex_lineage.observed_vertex_count",
        )
        reported_source_count = _integer(
            vertex.get("source_vertex_count"),
            name="source_vertex_lineage.source_vertex_count",
        )
        mapped_source_count = _integer(
            vertex.get("mapped_source_vertex_count"),
            name="source_vertex_lineage.mapped_source_vertex_count",
        )
        mapped_observed_count = _integer(
            vertex.get("mapped_observed_vertex_count"),
            name="source_vertex_lineage.mapped_observed_vertex_count",
        )
        max_observed_per_source = _integer(
            vertex.get("max_observed_per_source"),
            name="source_vertex_lineage.max_observed_per_source",
        )
        max_source_per_observed = _integer(
            vertex.get("max_source_per_observed"),
            name="source_vertex_lineage.max_source_per_observed",
        )
        constraint_occurrences = _integer(
            vertex.get("constraint_occurrence_count"),
            name="source_vertex_lineage.constraint_occurrence_count",
        )
    except (TypeError, ValueError):
        reasons.append("global_source_vertex_counts_malformed")
    else:
        if solution_count != 1:
            reasons.append("global_source_vertex_solution_count_not_one")
        if reported_source_count != expected_vertex_count:
            reasons.append("global_source_vertex_source_count_mismatch")
        if observed_count != expected_vertex_count:
            reasons.append("global_source_vertex_observed_count_mismatch")
        if mapped_source_count != expected_vertex_count:
            reasons.append("global_source_vertex_source_coverage_incomplete")
        if mapped_observed_count != observed_count:
            reasons.append("global_source_vertex_observed_coverage_incomplete")
        if max_observed_per_source != 1:
            reasons.append("global_source_vertex_split")
        if max_source_per_observed != 1:
            reasons.append("global_source_vertex_merge")
        if constraint_occurrences != expected_constraint_count:
            reasons.append("global_source_vertex_constraint_coverage_incomplete")
    failure_codes = vertex.get("failure_codes")
    if not _is_sequence(failure_codes):
        reasons.append("global_source_vertex_failure_codes_malformed")
    elif failure_codes:
        reasons.append("global_source_vertex_has_failure_codes")
    return list(dict.fromkeys(reasons))


def _stage_local_occ_topology_reasons(
    proof: Any,
    *,
    stage: str,
    expected_scope_count: int,
    expected_source_edge_count: int,
    expected_constraint_count: int,
) -> list[str]:
    """Validate identity only inside the OCC scope that owns each handle.

    MakeEdge may create a fresh endpoint handle for every independent edge,
    and building or repairing a face may copy handles again.  Those legal OCC
    ownership boundaries cannot prove one CAD-global source-vertex bijection.
    This residue therefore proves only that source endpoint labels retain a
    one-to-one relation inside each built edge (S2) or each built face (S3/S4).
    Cross-scope identity is deliberately neither claimed nor required.
    """

    if not isinstance(proof, Mapping):
        return ["stage_local_occ_topology_proof_missing"]
    reasons: list[str] = []
    if set(proof) != STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS:
        reasons.append("stage_local_occ_topology_proof_fields_mismatch")
    if proof.get("status") != "exact_stage_local_topology":
        reasons.append("stage_local_occ_topology_status_not_exact")
    if proof.get("proof_method") != STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD:
        reasons.append("stage_local_occ_topology_proof_method_mismatch")
    expected_scope_kind = "source_edge" if stage == "S2" else "source_face"
    if proof.get("scope_kind") != expected_scope_kind:
        reasons.append("stage_local_occ_topology_scope_kind_mismatch")
    expected = {
        "scope_count": expected_scope_count,
        "source_edge_count": expected_source_edge_count,
        "constraint_occurrence_count": expected_constraint_count,
        "max_observed_per_source_within_scope": 1,
        "max_source_per_observed_within_scope": 1,
    }
    for field, wanted in expected.items():
        try:
            observed = _integer(proof.get(field), name=f"stage local proof {field}")
        except (TypeError, ValueError):
            reasons.append(f"stage_local_occ_topology_{field}_malformed")
        else:
            if observed != wanted:
                reasons.append(f"stage_local_occ_topology_{field}_mismatch")
    failures = proof.get("failure_codes")
    if not _is_sequence(failures):
        reasons.append("stage_local_occ_topology_failure_codes_malformed")
    elif failures:
        reasons.append("stage_local_occ_topology_has_failure_codes")
    return list(dict.fromkeys(reasons))


def _candidate_degree_census_reasons(
    value: Any, *, expected_population: int, label: str
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label}_malformed"]
    population = 0
    try:
        for raw_degree, raw_count in value.items():
            if not isinstance(raw_degree, str) or not raw_degree.isdigit():
                raise ValueError
            degree = int(raw_degree)
            count = _integer(raw_count, name=f"{label}[{raw_degree}]")
            if degree < 1 or count < 1:
                raise ValueError
            population += count
    except (TypeError, ValueError):
        return [f"{label}_malformed"]
    if population != expected_population:
        return [f"{label}_population_mismatch"]
    return []


def _step_geometry_incidence_reasons(
    evidence: Mapping[str, Any], source_topology: Mapping[str, Any]
) -> list[str]:
    proof = evidence.get("step_geometry_incidence_proof")
    if not isinstance(proof, Mapping):
        return ["step_geometry_incidence_proof_missing"]
    reasons: list[str] = []
    if set(proof) != STEP_GEOMETRY_INCIDENCE_PROOF_KEYS:
        reasons.append("step_geometry_incidence_proof_fields_mismatch")
    counts = source_topology.get("counts", source_topology)
    face_count = int(counts["face_count"])
    edge_count = int(counts["edge_count"])
    vertex_count = int(counts["vertex_count"])
    occurrence_count = int(counts["face_edge_occurrence_count"])
    self_loop_count = sum(
        int(row[0]) == int(row[1]) for row in counts["edge_vertex_source_ids"]
    )

    if proof.get("status") != "exact_geometry_incidence":
        reasons.append("step_geometry_incidence_status_not_exact")
    failure_codes = proof.get("failure_codes")
    if not _is_sequence(failure_codes):
        reasons.append("step_geometry_incidence_failure_codes_malformed")
    elif failure_codes:
        reasons.append("step_geometry_incidence_has_failure_codes")
    if proof.get("vertex_proof_required") is not True:
        reasons.append("step_vertex_proof_not_required")
    if proof.get("vertex_proof_status") != "exact":
        reasons.append("step_vertex_proof_status_not_exact")
    if proof.get("vertex_proof_method") != STEP_VERTEX_IDENTITY_PROOF_METHOD:
        reasons.append("step_vertex_proof_method_mismatch")
    for field in ("tolerance_normalized", "vertex_tolerance_normalized"):
        value = proof.get(field)
        if (
            not _finite_number(value)
            or float(value) != STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ):
            reasons.append(f"step_geometry_{field}_mismatch")
    reasons.extend(
        _candidate_degree_census_reasons(
            proof.get("face_candidate_degree_counts"),
            expected_population=face_count,
            label="step_face_candidate_degree_counts",
        )
    )
    reasons.extend(
        _candidate_degree_census_reasons(
            proof.get("vertex_candidate_degree_counts"),
            expected_population=vertex_count,
            label="step_vertex_candidate_degree_counts",
        )
    )
    expected_values = {
        "face_matching_count_capped": 1,
        "vertex_matching_count_capped": 1,
        "source_vertex_count": vertex_count,
        "step_vertex_count": vertex_count,
        "mapped_source_edge_count": edge_count,
        "edge_endpoint_pair_expected_count": edge_count,
        "edge_endpoint_pair_proof_count": edge_count,
        "edge_endpoint_occurrence_expected_count": 2 * edge_count,
        "edge_endpoint_occurrence_proof_count": 2 * edge_count,
        "self_loop_endpoint_pair_expected_count": self_loop_count,
        "self_loop_endpoint_pair_proof_count": self_loop_count,
        "mapped_face_count": face_count,
        "mapped_edge_occurrence_count": occurrence_count,
    }
    for field, expected in expected_values.items():
        try:
            observed = _integer(proof.get(field), name=f"step proof {field}")
        except (TypeError, ValueError):
            reasons.append(f"step_geometry_{field}_malformed")
        else:
            if observed != expected:
                reasons.append(f"step_geometry_{field}_mismatch")
    return list(dict.fromkeys(reasons))


def _endpoint_identity_evidence_reasons(
    stage: str,
    evidence: Mapping[str, Any] | None,
    source_topology: Mapping[str, Any] | None,
    lineage: Mapping[str, Any],
) -> list[str]:
    """Return stable reasons when an exact S3-S6 endpoint claim lacks proof.

    The caller has already reduced native objects to path-free scalars.  This
    pure check prevents a record from becoming exact merely because it copied
    expected ``edge_vertex_source_ids`` into its topology census.
    """

    requirement = STAGE_ENDPOINT_EVIDENCE_REQUIREMENTS.get(stage)
    if requirement is None:
        return []
    if evidence is None:
        return ["endpoint_identity_evidence_missing"]
    if not isinstance(evidence, Mapping):
        raise TypeError("stage evidence must be a mapping")
    if source_topology is None:
        return ["endpoint_identity_source_topology_missing"]
    if requirement == "step_geometry_incidence_identity":
        return _step_geometry_incidence_reasons(evidence, source_topology)
    scope = _source_identity_scope(stage, lineage, source_topology)
    source_edge_count = len(scope["edge_ids"])
    source_occurrence_count = int(scope["constraint_count"])
    source_vertex_count = len(scope["vertex_ids"])
    reasons: list[str] = []

    if requirement == "stage_local_occ_topology":
        scope_count = (
            len(scope["edge_ids"]) if stage == "S2" else len(scope["face_ids"])
        )
        return _stage_local_occ_topology_reasons(
            evidence.get("stage_local_occ_topology_proof"),
            stage=stage,
            expected_scope_count=scope_count,
            expected_source_edge_count=source_edge_count,
            expected_constraint_count=source_occurrence_count,
        )

    if requirement == "edge_and_global_vertex_identity":
        edge_method = evidence.get("endpoint_identity_proof_method")
        if edge_method != EDGE_ENDPOINT_IDENTITY_PROOF_METHOD:
            reasons.append("endpoint_identity_proof_method_mismatch")
        try:
            edge_count = _integer(
                evidence.get("endpoint_identity_source_edge_count"),
                name="endpoint_identity_source_edge_count",
            )
            occurrence_count = _integer(
                evidence.get("endpoint_identity_occurrence_count"),
                name="endpoint_identity_occurrence_count",
            )
        except (TypeError, ValueError):
            reasons.append("endpoint_identity_counts_malformed")
        else:
            if edge_count != source_edge_count:
                reasons.append("endpoint_identity_source_edge_coverage_incomplete")
            if occurrence_count != source_occurrence_count:
                reasons.append("endpoint_identity_occurrence_coverage_incomplete")
        # A terminal at the first face has no completed target on which a
        # shared-vertex assertion could be measured.  The zero endpoint census
        # remains mandatory, but a fabricated non-empty global proof is not.
        if source_occurrence_count == 0:
            if "source_vertex_lineage" in evidence:
                reasons.append("global_source_vertex_lineage_unexpected_for_empty_scope")
            return list(dict.fromkeys(reasons))

    # Sewing is allowed to rebuild edge and vertex handles.  S5/S6 therefore
    # must not pretend that direct S2-to-stage IsSame survived.  Their stronger
    # replacement proof is the global source-vertex assignment constrained by
    # every already source-bound sewn edge occurrence.
    if source_vertex_count == 0:
        if "source_vertex_lineage" in evidence:
            reasons.append("global_source_vertex_lineage_unexpected_for_empty_scope")
        return list(dict.fromkeys(reasons))
    reasons.extend(
        _global_vertex_identity_reasons(
            evidence.get("source_vertex_lineage"),
            expected_vertex_count=source_vertex_count,
            expected_constraint_count=source_occurrence_count,
        )
    )
    return list(dict.fromkeys(reasons))


def make_not_reached_stage(
    stage: str,
    reason: str,
    *,
    blocked_by_stage: str | None = None,
) -> dict[str, Any]:
    """Return a denominator-preserving placeholder for a downstream stage."""

    stage_code = _normalize_stage_code(stage)
    if not isinstance(reason, str) or not reason:
        raise ValueError("not_reached reason must be a non-empty string")
    if blocked_by_stage is not None:
        blocked_by_stage = _normalize_stage_code(blocked_by_stage)
        if STAGE_ORDER.index(blocked_by_stage) >= STAGE_ORDER.index(stage_code):
            raise ValueError("blocked_by_stage must precede the not-reached stage")
    result = {
        "schema": STAGE_RECORD_SCHEMA,
        "stage": stage_code,
        "phase": STAGE_NAMES[stage_code],
        "status": "not_reached",
        "reached": False,
        "reason": reason,
        "blocked_by_stage": blocked_by_stage,
    }
    assert_path_free_finite(result)
    return result


def normalize_stage_record(
    record: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
    source_topology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one stage observation to the versioned evidence contract."""

    if not isinstance(record, Mapping):
        raise TypeError("stage record must be a mapping")
    assert_path_free_finite(record, label="stage record")
    if "native_valid" in record:
        raise ValueError(
            "native_valid is ambiguous; use construction_native_valid at S6 "
            "or reimport_native_valid at S7"
        )
    schema = record.get("schema")
    if schema is not None and schema != STAGE_RECORD_SCHEMA:
        raise ValueError("stage record schema mismatch")
    stage = _normalize_stage_code(record.get("stage"), record.get("phase"))
    if expected_stage is not None and stage != _normalize_stage_code(expected_stage):
        raise ValueError("stage record is not the expected stage")

    reached_value = record.get("reached")
    status = record.get("status")
    if status is None:
        status = "not_reached" if reached_value is False else "observed"
    if status not in {"observed", "not_reached"}:
        raise ValueError("stage status must be observed or not_reached")
    reached = status == "observed"
    if reached_value is not None and type(reached_value) is not bool:
        raise TypeError("stage reached must be a boolean")
    if reached_value is not None and reached_value != reached:
        raise ValueError("stage status and reached flag disagree")

    if not reached:
        reason = record.get("reason")
        blocked_by = record.get("blocked_by_stage")
        result = make_not_reached_stage(
            stage,
            reason,
            blocked_by_stage=blocked_by,
        )
        return result

    evidence_value = record.get("evidence")
    if evidence_value is not None and not isinstance(evidence_value, Mapping):
        raise TypeError("stage evidence must be a mapping")
    fallback_lineage = record.get("lineage_status", record.get("mapping_status"))
    lineage_value = record.get("lineage", fallback_lineage)
    lineage = _normalize_lineage(
        lineage_value,
        stage=stage,
        source_topology=source_topology,
        fallback_failures=record.get("mapping_failures"),
    )
    if (
        lineage.get("exact") is True
        or lineage.get("prefix_pass_exact") is True
        or lineage.get("local_failure_exact") is True
    ):
        whole_stage_failure = (
            lineage.get("local_failure_exact") is True
            and lineage.get("whole_stage_terminal") is not None
        )
        endpoint_reasons = [] if whole_stage_failure else _endpoint_identity_evidence_reasons(
            stage, evidence_value, source_topology, lineage
        )
        if endpoint_reasons:
            existing_reasons = list(lineage.get("inconclusive_reasons") or [])
            existing_reasons.extend(
                f"lineage_{reason}" for reason in endpoint_reasons
            )
            lineage = {
                **lineage,
                "status": "ambiguous",
                "classification": "inconclusive",
                "exact": False,
                "prefix_pass_exact": False,
                "local_failure_exact": False,
                "inconclusive_reasons": list(dict.fromkeys(existing_reasons)),
            }
    failure = _normalize_failure(record.get("failure"))
    defects_value = record.get("defects", record.get("defect_codes"))
    defects = _normalize_defects(defects_value)

    topology_value = record.get("topology", record.get("topology_census"))
    topology = None
    if topology_value is not None:
        topology = validate_topology_census(
            topology_value, source_topology=source_topology
        )

    construction_native = _optional_bool(
        record.get("construction_native_valid"), name="construction_native_valid"
    )
    reimport_native = _optional_bool(
        record.get("reimport_native_valid"), name="reimport_native_valid"
    )
    strict_valid = _optional_bool(record.get("strict_valid"), name="strict_valid")
    if stage != "S6" and construction_native is not None:
        raise ValueError("construction_native_valid belongs only to S6")
    if stage != "S7" and (reimport_native is not None or strict_valid is not None):
        raise ValueError("reimport_native_valid and strict_valid belong only to S7")
    if stage == "S6" and construction_native is None and failure is None:
        raise ValueError("observed S6 requires construction_native_valid")
    if stage == "S7" and failure is None:
        if reimport_native is None or strict_valid is None:
            raise ValueError(
                "observed S7 requires separate reimport_native_valid and strict_valid"
            )
        if strict_valid and not reimport_native:
            raise ValueError("strict_valid cannot be true when reimport native is false")

    bad_reasons: list[str] = []
    if failure is not None:
        bad_reasons.append(f"failure:{failure['kind']}")
    if lineage.get("local_failure_exact") is True:
        distributed = lineage.get("distributed_scope") or {}
        whole_stage = lineage.get("whole_stage_terminal") or {}
        if whole_stage:
            bad_reasons.append("whole_stage_terminal_failure:" + stage)
        else:
            bad_reasons.append(
                "local_terminal_failure:"
                f"{distributed.get('entity_kind')}:"
                f"{distributed.get('terminal_failure_entity_id')}"
            )
    bad_reasons.extend(f"defect:{defect['code']}" for defect in defects)
    if topology is not None:
        bad_reasons.extend(
            f"topology_drift:{field}" for field in topology["drifted_fields"]
        )
    if stage == "S6" and construction_native is False:
        bad_reasons.append("construction_native_invalid")
    if stage == "S7":
        if reimport_native is False:
            bad_reasons.append("reimport_native_invalid")
        if strict_valid is False:
            bad_reasons.append("strict_invalid")
    bad_reasons = list(dict.fromkeys(bad_reasons))

    result: dict[str, Any] = {
        "schema": STAGE_RECORD_SCHEMA,
        "stage": stage,
        "phase": STAGE_NAMES[stage],
        "status": "observed",
        "reached": True,
        "lineage": lineage,
        "topology": topology,
        "defects": defects,
        "failure": failure,
        "scientifically_bad": bool(bad_reasons),
        "bad_reasons": bad_reasons,
    }
    if stage == "S6":
        result["construction_native_valid"] = construction_native
    if stage == "S7":
        result["reimport_native_valid"] = reimport_native
        result["strict_valid"] = strict_valid
    if evidence_value is not None:
        result["evidence"] = _json_value(evidence_value)
    assert_path_free_finite(result, label="normalized stage record")
    return result


def validate_stage_sequence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe seven-stage coverage; omissions and ordering are protocol errors."""

    if not _is_sequence(records):
        raise TypeError("stage records must be a sequence")
    received: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"stage record {index} must be a mapping")
        received.append(_normalize_stage_code(record.get("stage"), record.get("phase")))
    counts = Counter(received)
    missing = [stage for stage in STAGE_ORDER if counts[stage] == 0]
    duplicates = [stage for stage in STAGE_ORDER if counts[stage] > 1]
    canonical_received = [stage for stage in STAGE_ORDER if counts[stage] == 1]
    out_of_order = not duplicates and not missing and received != list(STAGE_ORDER)
    failures = [f"missing_stage:{stage}" for stage in missing]
    failures.extend(f"duplicate_stage:{stage}" for stage in duplicates)
    if out_of_order:
        failures.append("stage_order_mismatch")
    result = {
        "expected_stages": list(STAGE_ORDER),
        "received_stages": received,
        "canonical_unique_stages": canonical_received,
        "stage_counts": {stage: int(counts[stage]) for stage in STAGE_ORDER},
        "missing_stages": missing,
        "duplicate_stages": duplicates,
        "out_of_order": out_of_order,
        "all_stages_accounted": not missing and not duplicates,
        "protocol_failures": failures,
    }
    assert_path_free_finite(result)
    return result


def _bad_stage_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons = record.get("bad_reasons") or []
    return [str(value) for value in reasons]


def infer_first_bad_stage(
    normalized_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Infer the earliest bad stage only across an exact source-bound prefix."""

    if not _is_sequence(normalized_records):
        raise TypeError("normalized stage records must be a sequence")
    ambiguity_stage: str | None = None
    ambiguity_reasons: list[str] = []
    for record in normalized_records:
        if not isinstance(record, Mapping):
            raise TypeError("normalized stage record must be a mapping")
        stage = _normalize_stage_code(record.get("stage"), record.get("phase"))
        status = record.get("status")
        if status == "not_reached":
            if ambiguity_stage is not None:
                return {
                    "status": "inconclusive",
                    "stage": None,
                    "phase": None,
                    "reasons": ambiguity_reasons,
                    "blocked_at_stage": ambiguity_stage,
                }
            return {
                "status": "inconclusive",
                "stage": None,
                "phase": None,
                "reasons": [f"{stage}:not_reached_without_identified_bad_stage"],
                "blocked_at_stage": stage,
            }
        lineage = record.get("lineage")
        lineage_exact = isinstance(lineage, Mapping) and lineage.get("exact") is True
        local_failure_exact = (
            isinstance(lineage, Mapping)
            and lineage.get("local_failure_exact") is True
        )
        prefix_pass_exact = (
            isinstance(lineage, Mapping)
            and lineage.get("prefix_pass_exact") is True
        )
        if not lineage_exact and not prefix_pass_exact and not local_failure_exact:
            if ambiguity_stage is None:
                ambiguity_stage = stage
                raw_reasons = (
                    lineage.get("inconclusive_reasons", [])
                    if isinstance(lineage, Mapping)
                    else ["lineage_missing"]
                )
                ambiguity_reasons = [f"{stage}:{reason}" for reason in raw_reasons]
                if not ambiguity_reasons:
                    ambiguity_reasons = [f"{stage}:lineage_not_exact"]
        reasons = _bad_stage_reasons(record)
        if reasons:
            if ambiguity_stage is not None:
                return {
                    "status": "inconclusive",
                    "stage": None,
                    "phase": None,
                    "reasons": ambiguity_reasons,
                    "blocked_at_stage": ambiguity_stage,
                }
            return {
                "status": "identified",
                "stage": stage,
                "phase": STAGE_NAMES[stage],
                "reasons": reasons,
                "blocked_at_stage": None,
            }
        if local_failure_exact:
            whole_stage = lineage.get("whole_stage_terminal") or {}
            if whole_stage:
                reasons = [f"whole_stage_terminal_failure:{stage}"]
            else:
                distributed = lineage.get("distributed_scope") or {}
                terminal = distributed.get("terminal_failure_entity_id")
                reasons = [
                    f"local_terminal_failure:{distributed.get('entity_kind')}:{terminal}"
                ]
            return {
                "status": "identified",
                "stage": stage,
                "phase": STAGE_NAMES[stage],
                "reasons": reasons,
                "blocked_at_stage": None,
            }
    if ambiguity_stage is not None:
        return {
            "status": "inconclusive",
            "stage": None,
            "phase": None,
            "reasons": ambiguity_reasons,
            "blocked_at_stage": ambiguity_stage,
        }
    return {
        "status": "no_bad_stage",
        "stage": None,
        "phase": None,
        "reasons": [],
        "blocked_at_stage": None,
    }


def assess_stage_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    source_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate, normalize, and derive one seven-stage census assessment."""

    source = validate_topology_census(source_topology, require_complete=True)
    coverage = validate_stage_sequence(records)
    normalized = [
        normalize_stage_record(record, source_topology=source)
        for record in records
    ]
    by_stage: dict[str, Mapping[str, Any]] = {}
    for record in normalized:
        by_stage.setdefault(str(record["stage"]), record)
    ordered = [by_stage[stage] for stage in STAGE_ORDER if stage in by_stage]

    protocol_failures = list(coverage["protocol_failures"])
    first_not_reached: str | None = None
    for record in ordered:
        stage = str(record["stage"])
        if record["status"] == "not_reached":
            if first_not_reached is None:
                first_not_reached = stage
        elif first_not_reached is not None:
            protocol_failures.append(f"observed_after_not_reached:{stage}")

    inference = infer_first_bad_stage(ordered)
    inconclusive_reasons: list[str] = []
    topology_drift: list[dict[str, Any]] = []
    observed_bad_stages: list[str] = []
    last_identified_bad: str | None = None
    for record in ordered:
        stage = str(record["stage"])
        if record["status"] == "not_reached":
            blocked_by = record.get("blocked_by_stage")
            if last_identified_bad is None:
                inconclusive_reasons.append(
                    f"{stage}:not_reached_without_prior_bad_stage"
                )
            elif blocked_by is not None and blocked_by != last_identified_bad:
                inconclusive_reasons.append(
                    f"{stage}:blocked_by_stage_does_not_match_first_bad"
                )
            continue

        lineage = record["lineage"]
        # Once an exact prefix has already identified a scientifically bad
        # boundary, later ambiguity is downstream evidence and cannot erase
        # that first-bad conclusion.  We still retain every later stage,
        # topology drift, and validity value in the report.  Before (and at)
        # the first bad boundary, lineage and required topology remain strict.
        before_first_identified_bad = last_identified_bad is None
        if before_first_identified_bad and (
            lineage["exact"] is not True
            and lineage.get("prefix_pass_exact") is not True
            and lineage.get("local_failure_exact") is not True
        ):
            reasons = lineage.get("inconclusive_reasons") or ["lineage_not_exact"]
            inconclusive_reasons.extend(f"{stage}:{reason}" for reason in reasons)

        # A distributed prefix is deliberately not a global topology
        # snapshot.  Its exactness comes from the source-ordered event scope
        # plus the same mapping/endpoint proof used by completed events.  A
        # later paired boundary can therefore be localized without inventing
        # missing faces, edges, or vertices at this earlier prefix.
        requires_full_topology = not (
            lineage.get("prefix_pass_exact") is True
        )
        if (
            before_first_identified_bad
            and record.get("failure") is None
            and requires_full_topology
        ):
            topology = record.get("topology")
            if topology is None:
                inconclusive_reasons.append(f"{stage}:topology_evidence_missing")
            else:
                required = set(STAGE_REQUIRED_TOPOLOGY_FIELDS[stage])
                present = set(topology["present_fields"])
                for field in sorted(required - present):
                    inconclusive_reasons.append(
                        f"{stage}:topology_evidence_missing:{field}"
                    )
        topology = record.get("topology")
        if topology is not None and topology["drifted_fields"]:
            topology_drift.append(
                {
                    "stage": stage,
                    "phase": STAGE_NAMES[stage],
                    "drifted_fields": list(topology["drifted_fields"]),
                }
            )
        if record.get("scientifically_bad"):
            observed_bad_stages.append(stage)
            if last_identified_bad is None:
                last_identified_bad = stage

    protocol_failures = list(dict.fromkeys(protocol_failures))
    inconclusive_reasons = list(dict.fromkeys(inconclusive_reasons))
    all_observed = len(ordered) == len(STAGE_ORDER) and all(
        record.get("status") == "observed" for record in ordered
    )
    s6 = by_stage.get("S6", {})
    s7 = by_stage.get("S7", {})
    valid_chain = bool(
        all_observed
        and not protocol_failures
        and not inconclusive_reasons
        and not observed_bad_stages
        and s6.get("construction_native_valid") is True
        and s7.get("reimport_native_valid") is True
        and s7.get("strict_valid") is True
    )
    conclusive = bool(
        not protocol_failures
        and not inconclusive_reasons
        and inference["status"] in {"identified", "no_bad_stage"}
        and (inference["status"] != "no_bad_stage" or valid_chain)
    )
    result = {
        "schema": ASSESSMENT_SCHEMA,
        "stage_order": list(STAGE_ORDER),
        "source_topology": source,
        "stages": ordered,
        "coverage": coverage,
        "protocol_failures": protocol_failures,
        "protocol_failure_count": len(protocol_failures),
        "inconclusive_reasons": inconclusive_reasons,
        "inconclusive_reason_count": len(inconclusive_reasons),
        "topology_drift_observations": topology_drift,
        "topology_drift_count": len(topology_drift),
        "observed_bad_stages": observed_bad_stages,
        "first_bad_inference": inference,
        "first_bad_stage": inference["stage"],
        "first_bad_phase": inference["phase"],
        "first_bad_reasons": inference["reasons"],
        "valid_chain": valid_chain,
        "conclusive": conclusive,
    }
    assert_path_free_finite(result, label="stage assessment")
    return result


__all__ = [
    "ASSESSMENT_SCHEMA",
    "EDGE_ENDPOINT_IDENTITY_PROOF_METHOD",
    "EXACT_LINEAGE_STATUSES",
    "GLOBAL_VERTEX_IDENTITY_PROOF_KEYS",
    "GLOBAL_VERTEX_IDENTITY_PROOF_METHOD",
    "INCONCLUSIVE_LINEAGE_STATUSES",
    "LOCAL_FAILURE_LINEAGE_STATUSES",
    "PREFIX_PASS_LINEAGE_STATUSES",
    "PREFIX_LINEAGE_STATUSES",
    "STAGE_BY_NAME",
    "STAGE_NAMES",
    "STAGE_ORDER",
    "STAGE_RECORD_SCHEMA",
    "STAGE_REQUIRED_TOPOLOGY_FIELDS",
    "STAGE_REQUIRED_LINEAGE_POPULATIONS",
    "STAGE_ENDPOINT_EVIDENCE_REQUIREMENTS",
    "STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS",
    "STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD",
    "STEP_GEOMETRY_INCIDENCE_PROOF_KEYS",
    "STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED",
    "STEP_VERTEX_IDENTITY_PROOF_METHOD",
    "TOPOLOGY_CENSUS_SCHEMA",
    "TOPOLOGY_FIELDS",
    "TOPOLOGY_RELATION_FIELDS",
    "assert_path_free_finite",
    "assess_stage_lineage",
    "infer_first_bad_stage",
    "make_not_reached_stage",
    "normalize_stage_record",
    "redact_path_and_native_text",
    "validate_stage_sequence",
    "validate_topology_census",
]
