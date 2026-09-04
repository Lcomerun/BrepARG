"""Archive a completed source-bound assembly census as Git-safe evidence.

The native census runner intentionally keeps source pickles, temporary STEP
files, and worker logs outside version control.  This module is the independent
archive boundary.  It accepts only a signed, clean-worktree, terminal formal
run; revalidates the ten-task ledger and every source/stage binding; and writes
only a compact allowlisted report.

This module is deliberately Open-CASCADE free.  It imports only the runner's
pure protocol API and the pure stage-lineage normalizer.  In particular, it
never opens a source pickle or a STEP file and never copies a worker log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .assembly_stage_lineage import (
        ASSESSMENT_SCHEMA,
        GLOBAL_VERTEX_IDENTITY_PROOF_KEYS,
        GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        STAGE_ORDER,
        STAGE_RECORD_SCHEMA,
        STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS,
        STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
        STEP_GEOMETRY_INCIDENCE_PROOF_KEYS,
        STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        STEP_VERTEX_IDENTITY_PROOF_METHOD,
        TOPOLOGY_CENSUS_SCHEMA,
        assert_path_free_finite,
        assess_stage_lineage,
    )
    from .probe_source_bound_stage_census import (
        BRIDGE_ARM,
        BRIDGE_CAD_IDS,
        EXACT_NEGATIVE_EVIDENCE,
        EXCLUDED_EXACT_NEGATIVE_CAD_IDS,
        FROZEN_CALIBRATION_MANIFEST_SHA256,
        FROZEN_SELECTOR_MATRIX_SHA256,
        FROZEN_SELECTOR_RUN_SHA256,
        FROZEN_RUNTIME_IDENTITY,
        RUNTIME_ABI_SENTINEL_SCOPE,
        RUNTIME_IDENTITY_SCHEMA,
        PRIMARY_ARM,
        ROWS_NAME,
        RUN_NAME,
        RUN_SCHEMA,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        TASKS,
        canonical_sha256,
        normalize_binding,
        normalize_runtime_identity,
        runtime_abi_sentinel_from_payload,
        sha256_file,
        summarize,
        validate_attempt_row,
        validate_terminal_artifact_hashes,
    )
except ImportError:  # pragma: no cover - direct script execution
    from assembly_stage_lineage import (
        ASSESSMENT_SCHEMA,
        GLOBAL_VERTEX_IDENTITY_PROOF_KEYS,
        GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        STAGE_ORDER,
        STAGE_RECORD_SCHEMA,
        STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS,
        STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
        STEP_GEOMETRY_INCIDENCE_PROOF_KEYS,
        STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        STEP_VERTEX_IDENTITY_PROOF_METHOD,
        TOPOLOGY_CENSUS_SCHEMA,
        assert_path_free_finite,
        assess_stage_lineage,
    )
    from probe_source_bound_stage_census import (
        BRIDGE_ARM,
        BRIDGE_CAD_IDS,
        EXACT_NEGATIVE_EVIDENCE,
        EXCLUDED_EXACT_NEGATIVE_CAD_IDS,
        FROZEN_CALIBRATION_MANIFEST_SHA256,
        FROZEN_SELECTOR_MATRIX_SHA256,
        FROZEN_SELECTOR_RUN_SHA256,
        FROZEN_RUNTIME_IDENTITY,
        RUNTIME_ABI_SENTINEL_SCOPE,
        RUNTIME_IDENTITY_SCHEMA,
        PRIMARY_ARM,
        ROWS_NAME,
        RUN_NAME,
        RUN_SCHEMA,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        TASKS,
        canonical_sha256,
        normalize_binding,
        normalize_runtime_identity,
        runtime_abi_sentinel_from_payload,
        sha256_file,
        summarize,
        validate_attempt_row,
        validate_terminal_artifact_hashes,
    )


ARCHIVE_SCHEMA = "source-bound-stage-census-archive-v1"
ARCHIVE_ATTEMPT_SCHEMA = "source-bound-stage-census-archive-attempt-v1"
ARCHIVE_RUN_SCHEMA = "source-bound-stage-census-archive-run-v1"

EXPECTED_REPORT_FILES = {
    "README.md",
    "archive_validation.json",
    "artifact_manifest.json",
    RUN_NAME,
    ROWS_NAME,
    SUMMARY_NAME,
}
FORBIDDEN_SUFFIXES = {
    ".step",
    ".stp",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".log",
}
REQUIRED_SOURCE_HASHES = {
    "tools/probe_source_bound_stage_census.py",
    "tools/assembly_stage_lineage.py",
    "tools/directed_trim_assembly.py",
    "tools/assembly_repair.py",
    "tools/assembly_selector_geometry.py",
    "tools/diagnose_assembly_face_wires.py",
    "tools/diagnose_step_validity_components.py",
    "tools/probe_downstream_bad_wire_lineage.py",
    "tools/run_assembly_calibration_oracle.py",
    "tools/run_assembly_repair_matrix.py",
}
FROZEN_BREPARG_UTILS_SHA256 = (
    "e2509a844db0a9e0f8eaf670fffb9d4ad9e240af755155d25891d37b4468d521"
)
EMPTY_GIT_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
NATIVE_HANDLE_PATTERN = re.compile(
    r"(?:OCC\.Core|TopoDS_|SwigPyObject|\bthisown\b|"
    r"<[^>]*(?:TopoDS|Geom_|BRep|Handle_)[^>]*>|"
    r"\bHandle_[A-Za-z0-9_]+|\b0x[0-9a-f]{6,}\b)",
    re.IGNORECASE,
)
PRIVATE_NATIVE_KEYS = {
    "shape",
    "face",
    "edge",
    "wire",
    "surface",
    "curve",
    "target",
    "observed_shape",
    "observed_face",
    "observed_edge",
    "observed_wire",
    "native_handle",
}

RUN_KEYS = {
    "schema",
    "signature",
    "payload",
    "status",
    "attempts",
    "rows_sha256",
    "summary_sha256",
}
PAYLOAD_KEYS = {
    "schema",
    "run_kind",
    "calibration_manifest_sha256",
    "selector_matrix_sha256",
    "selector_run",
    "selector",
    "exact_negative_evidence",
    "excluded_exact_negative_cad_ids",
    "ordered_target_cad_ids",
    "ordered_tasks",
    "sources",
    "stages",
    "stage_record_schema",
    "stage_assessment_schema",
    "schema_v2",
    "joint_iterations",
    "worker_timeout_seconds",
    "python",
    "native_runtime",
    "repository",
    "breparg_runtime",
    "authorization_ceiling",
}
ROW_KEYS = {
    "schema",
    "task_id",
    "task_ordinal",
    "cad_id",
    "parent_id",
    "arm",
    "profile_name",
    "switches",
    "is_reachability_bridge",
    "counts_as_repair",
    "denominator",
    "historical_strict_valid",
    "run_signature",
    "source_binding_expected",
    "source_binding_before_load",
    "source_binding_loaded_bytes",
    "source_binding_after_load",
    "source_binding_after_measurement",
    "source_binding_parent_after_child",
    "worker_runtime_abi_sentinel",
    "status",
    "stage_records",
    "assessment",
    "step_roundtrip",
    "nonfinite_count",
    "elapsed_seconds",
    "worker_returncode",
    "worker_stdout_log",
    "worker_stderr_log",
    "error_type",
}
SUMMARY_KEYS = {
    "schema",
    "attempts",
    "denominator_rows",
    "primary_controls",
    "curve_interpolate_bridges",
    "primary_completed_or_scientific",
    "bridge_completed_or_scientific",
    "status_counts",
    "worker_or_protocol_failures",
    "source_binding_failures",
    "nonfinite_count",
    "first_bad",
    "protocol_health",
    "census_conclusive",
    "protocol_conclusive",
    "decision",
    "bridge_results_are_reachability_only",
    "bridge_repairs_counted",
    "selector_strict_valid_before",
    "selector_strict_valid_after",
    "authorizes_exact_candidate_design",
    "authorizes_repair",
    "authorizes_residual_expansion",
    "authorizes_full_100cad",
    "authorizes_selector_score_change",
    "authorizes_schema_v2_relaxation",
    "authorizes_training",
    "authorizes_sequence_generation",
    "authorizes_ar",
}
ASSESSMENT_KEYS = {
    "schema",
    "stage_order",
    "source_topology",
    "stages",
    "coverage",
    "protocol_failures",
    "protocol_failure_count",
    "inconclusive_reasons",
    "inconclusive_reason_count",
    "topology_drift_observations",
    "topology_drift_count",
    "observed_bad_stages",
    "first_bad_inference",
    "first_bad_stage",
    "first_bad_phase",
    "first_bad_reasons",
    "valid_chain",
    "conclusive",
}
STAGE_KEYS = {
    "schema",
    "stage",
    "phase",
    "status",
    "reached",
    "reason",
    "blocked_by_stage",
    "lineage",
    "topology",
    "defects",
    "failure",
    "scientifically_bad",
    "bad_reasons",
    "construction_native_valid",
    "reimport_native_valid",
    "strict_valid",
    "evidence",
}
LINEAGE_KEYS = {
    "reported_status",
    "status",
    "classification",
    "exact",
    "prefix_pass_exact",
    "local_failure_exact",
    "proof_method",
    "method",
    "solution_count",
    "source_face_ids",
    "source_edge_ids",
    "source_edge_occurrence_keys",
    "distributed_scope",
    "whole_stage_terminal",
    "failure_codes",
    "entities",
    "inconclusive_reasons",
}
TOPOLOGY_KEYS = {
    "schema",
    "counts",
    "present_fields",
    "missing_fields",
    "complete",
    "comparable_fields",
    "drifted_fields",
    "compared_to_source",
    "matches_source",
}
TOPOLOGY_SCALAR_FIELDS = {
    "face_count",
    "edge_count",
    "vertex_count",
    "face_edge_occurrence_count",
}
TOPOLOGY_VECTOR_FIELDS = {
    "face_edge_incidence_counts",
    "edge_face_incidence_counts",
    "vertex_edge_incidence_counts",
}
TOPOLOGY_RELATION_FIELDS = {
    "face_edge_source_ids",
    "edge_face_source_ids",
    "edge_vertex_source_ids",
}
ENTITY_KEYS = {
    "source_count",
    "observed_count",
    "mapped_source_count",
    "mapped_observed_count",
    "max_observed_per_source",
    "max_source_per_observed",
    "solution_count",
}
ALLOWED_ENTITY_NAMES = {"faces", "edges"}
DEFECT_KEYS = {
    "code",
    "kind",
    "source_face_index",
    "wire_index",
    "source_edge_ids",
    "mapping_status",
}
FAILURE_KEYS = {"kind", "reason", "code"}
EVIDENCE_SCALAR_KEYS = {
    "fitted_surface_count",
    "fitted_curve_count",
    "built_edge_count",
    "observed_source_face_count",
    "exact_source_face_count",
    "source_edge_occurrence_count",
    "observed_source_edge_count",
    "unique_source_edge_count",
    "complete_order_independent_source_edge_coverage",
    "observation_granularity",
    "failure_localized_by_strict_event_state_machine",
    "terminal_failure_source_face_id",
    "prefix_pass_before_next_stage_failure",
    "terminal_failure_source_edge_id",
    "paired_stage_terminal_failure_source_edge_id",
    "endpoint_identity_proof_method",
    "endpoint_identity_source_edge_count",
    "endpoint_identity_occurrence_count",
    "step_bytes",
    "step_sha256",
}
EVIDENCE_NESTED_KEYS = {
    "stage_local_occ_topology_proof",
    "source_vertex_lineage",
    "step_geometry_incidence_proof",
}

# ``_match_step_geometry_incidence`` returns progressively richer public
# evidence.  A face-count short circuit cannot honestly populate face-match
# statistics, a face/edge short circuit cannot populate the global vertex
# proof, and only an exact result may carry the two final mapped-population
# totals.  Keep those three non-exact shapes explicit so the archive accepts
# genuine downstream observations without treating absent exact-only fields as
# optional in an exact claim.
STEP_GEOMETRY_INCIDENCE_COMMON_KEYS = frozenset(
    {
        "status",
        "failure_codes",
        "vertex_proof_required",
        "vertex_proof_status",
    }
)
STEP_GEOMETRY_INCIDENCE_FACE_KEYS = STEP_GEOMETRY_INCIDENCE_COMMON_KEYS | {
    "tolerance_normalized",
    "face_candidate_degree_counts",
    "face_matching_count_capped",
}
STEP_GEOMETRY_INCIDENCE_VERTEX_KEYS = (
    set(STEP_GEOMETRY_INCIDENCE_PROOF_KEYS)
    - set(STEP_GEOMETRY_INCIDENCE_FACE_KEYS)
    - {"mapped_face_count", "mapped_edge_occurrence_count"}
)
STEP_GEOMETRY_INCIDENCE_NONEXACT_KEYSETS = frozenset(
    {
        frozenset(STEP_GEOMETRY_INCIDENCE_COMMON_KEYS),
        frozenset(STEP_GEOMETRY_INCIDENCE_FACE_KEYS),
        frozenset(
            set(STEP_GEOMETRY_INCIDENCE_FACE_KEYS)
            | STEP_GEOMETRY_INCIDENCE_VERTEX_KEYS
        ),
    }
)
VALIDITY_COMPONENT_KEYS = {
    "status",
    "native_brep_valid",
    "wire_count",
    "wire_order_failures",
    "wire_self_intersections",
    "free_edges",
    "shell_count",
    "shells_with_bad_edges",
    "solid_count",
}
SCHEMA_V2_MEASUREMENT_KEYS = {
    "schema",
    "accepted",
    "checks",
    "rejection_reasons",
    "thresholds",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read strict JSON {label}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read JSON artifact {target.name}") from exc
    value = _strict_json_loads(text, label=target.name)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {target.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read JSONL artifact {target.name}") from exc
    values = [
        _strict_json_loads(line, label=f"{target.name}:{index}")
        for index, line in enumerate(lines, 1)
        if line.strip()
    ]
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"JSONL rows must be objects in {target.name}")
    return values


def _write_text_lf(path: Path, value: str) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, value: Any) -> None:
    _write_text_lf(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_lf(
        path,
        "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise RuntimeError(
            f"{label} key set drifted; missing={missing}, unexpected={unexpected}"
        )


def _require_allowed_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise RuntimeError(f"{label} contains unexpected keys: {unexpected}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"{label} must be a list of strings")
    return list(value)


def _assert_no_native_handles(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{label} contains a non-string key")
            key_lower = key.lower()
            if key.startswith("_") or key_lower in PRIVATE_NATIVE_KEYS:
                raise RuntimeError(f"{label} contains private/native field {key!r}")
            _assert_no_native_handles(child, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_native_handles(child, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and NATIVE_HANDLE_PATTERN.search(value):
        raise RuntimeError(f"{label} contains a serialized native handle")
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")


def _assert_git_safe_json(value: Any, *, label: str) -> None:
    try:
        assert_path_free_finite(value, label=label)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not path/native/finite safe") from exc
    _assert_no_native_handles(value, label=label)
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{label} must be a {qualifier} integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _compact_int_vector(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise RuntimeError(f"{label} must be an integer vector")
    return {
        "kind": "integer_vector_digest",
        "length": len(value),
        "sum": sum(value),
        "min": min(value) if value else None,
        "max": max(value) if value else None,
        "sha256": canonical_sha256(value),
    }


def _compact_int_rows(value: Any, *, label: str, width: int | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be integer rows")
    flattened: list[int] = []
    lengths: list[int] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or any(type(item) is not int for item in row):
            raise RuntimeError(f"{label}[{index}] must be an integer row")
        if width is not None and len(row) != width:
            raise RuntimeError(f"{label}[{index}] has the wrong width")
        lengths.append(len(row))
        flattened.extend(row)
    return {
        "kind": "integer_rows_digest",
        "row_count": len(value),
        "entry_count": len(flattened),
        "row_length_min": min(lengths) if lengths else None,
        "row_length_max": max(lengths) if lengths else None,
        "value_min": min(flattened) if flattened else None,
        "value_max": max(flattened) if flattened else None,
        "sha256": canonical_sha256(value),
    }


def _compact_topology(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a topology mapping")
    _require_exact_keys(value, TOPOLOGY_KEYS, label)
    if value.get("schema") != TOPOLOGY_CENSUS_SCHEMA:
        raise RuntimeError(f"{label} schema drifted")
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        raise RuntimeError(f"{label}.counts must be a mapping")
    allowed_counts = (
        TOPOLOGY_SCALAR_FIELDS | TOPOLOGY_VECTOR_FIELDS | TOPOLOGY_RELATION_FIELDS
    )
    _require_allowed_keys(counts, allowed_counts, f"{label}.counts")
    compact_counts: dict[str, Any] = {}
    for key in sorted(TOPOLOGY_SCALAR_FIELDS):
        if key in counts:
            compact_counts[key] = _integer(counts[key], f"{label}.counts.{key}")
    for key in sorted(TOPOLOGY_VECTOR_FIELDS):
        if key in counts:
            compact_counts[key] = _compact_int_vector(
                counts[key], label=f"{label}.counts.{key}"
            )
    for key in sorted(TOPOLOGY_RELATION_FIELDS):
        if key in counts:
            compact_counts[key] = _compact_int_rows(
                counts[key],
                label=f"{label}.counts.{key}",
                width=2 if key == "edge_vertex_source_ids" else None,
            )
    result = {
        "schema": value["schema"],
        "counts": compact_counts,
        "present_fields": _require_string_list(
            value.get("present_fields"), f"{label}.present_fields"
        ),
        "missing_fields": _require_string_list(
            value.get("missing_fields"), f"{label}.missing_fields"
        ),
        "complete": value.get("complete"),
        "comparable_fields": _require_string_list(
            value.get("comparable_fields"), f"{label}.comparable_fields"
        ),
        "drifted_fields": _require_string_list(
            value.get("drifted_fields"), f"{label}.drifted_fields"
        ),
        "compared_to_source": value.get("compared_to_source"),
        "matches_source": value.get("matches_source"),
        "source_counts_sha256": canonical_sha256(counts),
    }
    for key in ("complete", "compared_to_source"):
        if type(result[key]) is not bool:
            raise RuntimeError(f"{label}.{key} must be boolean")
    if result["matches_source"] is not None and type(result["matches_source"]) is not bool:
        raise RuntimeError(f"{label}.matches_source must be boolean or null")
    return result


def _compact_entities(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    unexpected_names = sorted(set(value) - ALLOWED_ENTITY_NAMES)
    if unexpected_names:
        raise RuntimeError(f"{label} contains unknown entity kinds: {unexpected_names}")
    result: dict[str, Any] = {}
    for name, entity in sorted(value.items()):
        if not isinstance(entity, Mapping):
            raise RuntimeError(f"{label}.{name} must be a mapping")
        _require_allowed_keys(entity, ENTITY_KEYS, f"{label}.{name}")
        result[name] = {
            key: _integer(item, f"{label}.{name}.{key}")
            for key, item in sorted(entity.items())
        }
    return result


def _compact_distributed_scope(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    expected = {
        "entity_kind",
        "expected_ids",
        "completed_ids",
        "terminal_failure_entity_id",
        "preceding_stage_prefix_verified",
        "event_sequence_proof",
        "full_coverage",
        "prefix_pass_exact",
        "local_failure_exact",
    }
    _require_exact_keys(value, expected, label)
    events_container = value.get("event_sequence_proof")
    if not isinstance(events_container, Mapping):
        raise RuntimeError(f"{label}.event_sequence_proof must be a mapping")
    _require_exact_keys(
        events_container, {"events"}, f"{label}.event_sequence_proof"
    )
    events = events_container.get("events")
    if not isinstance(events, list):
        raise RuntimeError(f"{label}.event_sequence_proof.events must be a list")
    event_counts: Counter[str] = Counter()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise RuntimeError(f"{label}.events[{index}] must be a mapping")
        _require_exact_keys(event, {"entity_id", "event"}, f"{label}.events[{index}]")
        _integer(event.get("entity_id"), f"{label}.events[{index}].entity_id")
        if not isinstance(event.get("event"), str):
            raise RuntimeError(f"{label}.events[{index}].event must be a string")
        event_counts[str(event["event"])] += 1
    terminal = value.get("terminal_failure_entity_id")
    if terminal is not None:
        terminal = _integer(terminal, f"{label}.terminal_failure_entity_id")
    for key in (
        "preceding_stage_prefix_verified",
        "full_coverage",
        "prefix_pass_exact",
        "local_failure_exact",
    ):
        if type(value.get(key)) is not bool:
            raise RuntimeError(f"{label}.{key} must be boolean")
    if value.get("entity_kind") not in {"source_edge", "source_face"}:
        raise RuntimeError(f"{label}.entity_kind is not registered")
    return {
        "entity_kind": value["entity_kind"],
        "expected_ids": _compact_int_vector(
            value["expected_ids"], label=f"{label}.expected_ids"
        ),
        "completed_ids": _compact_int_vector(
            value["completed_ids"], label=f"{label}.completed_ids"
        ),
        "terminal_failure_entity_id": terminal,
        "preceding_stage_prefix_verified": value[
            "preceding_stage_prefix_verified"
        ],
        "event_sequence_proof": {
            "event_count": len(events),
            "event_counts": dict(sorted(event_counts.items())),
            "sha256": canonical_sha256(events),
        },
        "full_coverage": value["full_coverage"],
        "prefix_pass_exact": value["prefix_pass_exact"],
        "local_failure_exact": value["local_failure_exact"],
    }


def _compact_lineage(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_allowed_keys(value, LINEAGE_KEYS, label)
    required = {
        "reported_status",
        "status",
        "classification",
        "exact",
        "failure_codes",
        "entities",
        "inconclusive_reasons",
    }
    if not required <= set(value):
        raise RuntimeError(f"{label} lacks normalized lineage fields")
    result: dict[str, Any] = {
        "reported_status": value.get("reported_status"),
        "status": value.get("status"),
        "classification": value.get("classification"),
        "exact": value.get("exact"),
        "failure_codes": _require_string_list(
            value.get("failure_codes"), f"{label}.failure_codes"
        ),
        "entities": _compact_entities(value.get("entities"), label=f"{label}.entities"),
        "inconclusive_reasons": _require_string_list(
            value.get("inconclusive_reasons"), f"{label}.inconclusive_reasons"
        ),
    }
    for key in ("reported_status", "status", "classification"):
        if not isinstance(result[key], str) or not result[key]:
            raise RuntimeError(f"{label}.{key} must be a non-empty string")
    if type(result["exact"]) is not bool:
        raise RuntimeError(f"{label}.exact must be boolean")
    for key in (
        "prefix_pass_exact",
        "local_failure_exact",
        "proof_method",
        "method",
        "solution_count",
    ):
        if key in value:
            result[key] = value.get(key)
    for key in ("prefix_pass_exact", "local_failure_exact"):
        if key in result and type(result[key]) is not bool:
            raise RuntimeError(f"{label}.{key} must be boolean")
    for key in ("proof_method", "method"):
        if key in result and result[key] is not None and not isinstance(result[key], str):
            raise RuntimeError(f"{label}.{key} must be a string or null")
    if "solution_count" in result and result["solution_count"] is not None:
        result["solution_count"] = _integer(
            result["solution_count"], f"{label}.solution_count"
        )
    for key in ("source_face_ids", "source_edge_ids"):
        if key in value:
            result[key] = _compact_int_vector(value.get(key), label=f"{label}.{key}")
    if "source_edge_occurrence_keys" in value:
        result["source_edge_occurrence_keys"] = _compact_int_rows(
            value.get("source_edge_occurrence_keys"),
            label=f"{label}.source_edge_occurrence_keys",
            width=3,
        )
    if "distributed_scope" in value:
        result["distributed_scope"] = _compact_distributed_scope(
            value.get("distributed_scope"), label=f"{label}.distributed_scope"
        )
    if "whole_stage_terminal" in value:
        whole_stage = value.get("whole_stage_terminal")
        if whole_stage is None:
            result["whole_stage_terminal"] = None
        else:
            if not isinstance(whole_stage, Mapping):
                raise RuntimeError(f"{label}.whole_stage_terminal must be a mapping")
            expected = {
                "scope_kind",
                "boundary_stage",
                "prerequisite_stage",
                "prerequisite_exact",
                "construction_exception_observed",
            }
            _require_exact_keys(
                whole_stage, expected, f"{label}.whole_stage_terminal"
            )
            if whole_stage.get("scope_kind") != "whole_shape_boundary_failure":
                raise RuntimeError(
                    f"{label}.whole_stage_terminal scope kind drifted"
                )
            boundary = whole_stage.get("boundary_stage")
            prerequisite = whole_stage.get("prerequisite_stage")
            expected_prerequisite = {"S5": "S4", "S6": "S5"}.get(boundary)
            if expected_prerequisite is None or prerequisite != expected_prerequisite:
                raise RuntimeError(
                    f"{label}.whole_stage_terminal stage binding drifted"
                )
            for key in (
                "prerequisite_exact",
                "construction_exception_observed",
            ):
                if whole_stage.get(key) is not True:
                    raise RuntimeError(
                        f"{label}.whole_stage_terminal.{key} must be true"
                    )
            result["whole_stage_terminal"] = dict(whole_stage)
    return result


def _compact_defects(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list")
    result: list[dict[str, Any]] = []
    for index, defect in enumerate(value):
        if not isinstance(defect, Mapping):
            raise RuntimeError(f"{label}[{index}] must be a mapping")
        _require_allowed_keys(defect, DEFECT_KEYS, f"{label}[{index}]")
        code = defect.get("code")
        if not isinstance(code, str) or not code:
            raise RuntimeError(f"{label}[{index}].code must be non-empty")
        compact = {
            key: defect.get(key)
            for key in (
                "code",
                "kind",
                "source_face_index",
                "wire_index",
                "source_edge_ids",
                "mapping_status",
            )
            if key in defect
        }
        if "source_edge_ids" in compact:
            edge_ids = compact["source_edge_ids"]
            if not isinstance(edge_ids, list) or any(type(item) is not int for item in edge_ids):
                raise RuntimeError(f"{label}[{index}].source_edge_ids is malformed")
        result.append(compact)
    return result


def _compact_failure(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping or null")
    _require_allowed_keys(value, FAILURE_KEYS, label)
    result = {key: value.get(key) for key in ("kind", "reason", "code") if key in value}
    if not isinstance(result.get("kind"), str) or not isinstance(result.get("reason"), str):
        raise RuntimeError(f"{label} lacks kind/reason strings")
    return result


def _compact_metric(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{label} is non-finite")
        return value
    if isinstance(value, list):
        return {
            "kind": "sequence_digest",
            "item_count": len(value),
            "sha256": canonical_sha256(value),
        }
    raise RuntimeError(f"{label} is not a compactable metric")


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{label} must be boolean")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _compact_nonnegative_int_counter(
    value: Any, *, label: str
) -> dict[str, int]:
    """Validate a JSON object whose integer-like keys count integer buckets."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, (str, int)) or isinstance(raw_key, bool):
            raise RuntimeError(f"{label} contains a non-integer bucket key")
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} contains a non-integer bucket key") from exc
        if key < 0 or str(key) != str(raw_key):
            raise RuntimeError(f"{label} contains a noncanonical bucket key")
        canonical = str(key)
        if canonical in result:
            raise RuntimeError(f"{label} contains duplicate canonical bucket keys")
        result[canonical] = _integer(
            raw_count, f"{label}.{canonical}", positive=True
        )
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _compact_source_vertex_lineage(value: Any, *, label: str) -> dict[str, Any]:
    """Archive only the registered scalar global source-vertex proof.

    The assignment itself, OCC handles, endpoint coordinates, and candidate
    arrays are deliberately outside this schema.  Exact-key validation makes
    any future proof-field addition an explicit archive protocol review.
    """

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_exact_keys(value, set(GLOBAL_VERTEX_IDENTITY_PROOF_KEYS), label)
    status = _nonempty_string(value.get("status"), f"{label}.status")
    raw_solution_count = value.get("solution_count")
    solution_count = (
        None
        if raw_solution_count is None
        else _integer(raw_solution_count, f"{label}.solution_count")
    )
    result = {
        "status": status,
        "proof_method": _nonempty_string(
            value.get("proof_method"), f"{label}.proof_method"
        ),
        "solution_count": solution_count,
        "solution_count_capped_at_two": _boolean(
            value.get("solution_count_capped_at_two"),
            f"{label}.solution_count_capped_at_two",
        ),
        "failure_codes": _require_string_list(
            value.get("failure_codes"), f"{label}.failure_codes"
        ),
    }
    for key in (
        "source_vertex_count",
        "observed_vertex_count",
        "mapped_source_vertex_count",
        "mapped_observed_vertex_count",
        "max_observed_per_source",
        "max_source_per_observed",
        "constraint_occurrence_count",
    ):
        result[key] = _integer(value.get(key), f"{label}.{key}")
    if result["proof_method"] != GLOBAL_VERTEX_IDENTITY_PROOF_METHOD:
        raise RuntimeError(f"{label}.proof_method drifted")
    if result["solution_count_capped_at_two"] is not True:
        raise RuntimeError(f"{label} did not cap matching solutions at two")
    failures = result["failure_codes"]
    if status == "exact_identity":
        if solution_count != 1 or failures:
            raise RuntimeError(f"{label} exact proof semantics drifted")
    elif status == "ambiguous":
        if not failures:
            raise RuntimeError(f"{label} non-exact proof lacks failure codes")
        if solution_count not in {None, 0, 1, 2}:
            raise RuntimeError(f"{label} non-exact solution count drifted")
        assignment_failures = {
            "source_vertex_assignment_missing": 0,
            "source_vertex_assignment_nonunique": 2,
            "source_vertex_assignment_constraint_replay_failed": 1,
        }
        for failure, expected_count in assignment_failures.items():
            if failure in failures and solution_count != expected_count:
                raise RuntimeError(
                    f"{label} {failure} contradicts solution_count"
                )
        if solution_count is None and any(
            failure in assignment_failures for failure in failures
        ):
            raise RuntimeError(f"{label} measured assignment lacks solution_count")
    else:
        raise RuntimeError(f"{label}.status is not registered")
    return result


def _compact_stage_local_occ_topology_proof(
    value: Any, *, label: str
) -> dict[str, Any]:
    """Validate and retain the coordinate-free S2--S4 local proof."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_exact_keys(value, set(STAGE_LOCAL_OCC_TOPOLOGY_PROOF_KEYS), label)
    result: dict[str, Any] = {
        "status": _nonempty_string(value.get("status"), f"{label}.status"),
        "proof_method": _nonempty_string(
            value.get("proof_method"), f"{label}.proof_method"
        ),
        "scope_kind": _nonempty_string(
            value.get("scope_kind"), f"{label}.scope_kind"
        ),
        "failure_codes": _require_string_list(
            value.get("failure_codes"), f"{label}.failure_codes"
        ),
    }
    for key in (
        "scope_count",
        "source_edge_count",
        "constraint_occurrence_count",
        "max_observed_per_source_within_scope",
        "max_source_per_observed_within_scope",
    ):
        result[key] = _integer(value.get(key), f"{label}.{key}")
    if result["status"] != "exact_stage_local_topology":
        raise RuntimeError(f"{label}.status drifted")
    if result["proof_method"] != STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD:
        raise RuntimeError(f"{label}.proof_method drifted")
    if result["scope_kind"] not in {"source_edge", "source_face"}:
        raise RuntimeError(f"{label}.scope_kind drifted")
    if result["failure_codes"]:
        raise RuntimeError(f"{label} claims exactness with failure codes")
    return result


def _compact_step_geometry_incidence_proof(
    value: Any, *, label: str
) -> dict[str, Any]:
    """Validate and retain one status-specific, coordinate-free S7 proof."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    status = _nonempty_string(value.get("status"), f"{label}.status")
    keys = frozenset(value)
    if status == "exact_geometry_incidence":
        _require_exact_keys(value, STEP_GEOMETRY_INCIDENCE_PROOF_KEYS, label)
    elif status in {"unavailable", "ambiguous"}:
        if keys not in STEP_GEOMETRY_INCIDENCE_NONEXACT_KEYSETS:
            raise RuntimeError(
                f"{label} key set drifted for non-exact status: "
                f"{sorted(keys)}"
            )
    else:
        raise RuntimeError(f"{label}.status is not registered")
    result: dict[str, Any] = {
        "status": status,
        "failure_codes": _require_string_list(
            value.get("failure_codes"), f"{label}.failure_codes"
        ),
        "vertex_proof_required": _boolean(
            value.get("vertex_proof_required"), f"{label}.vertex_proof_required"
        ),
        "vertex_proof_status": _nonempty_string(
            value.get("vertex_proof_status"), f"{label}.vertex_proof_status"
        ),
    }
    if result["vertex_proof_required"] is not True:
        raise RuntimeError(f"{label} did not require the frozen S7 vertex proof")
    if STEP_GEOMETRY_INCIDENCE_FACE_KEYS <= keys:
        result.update(
            tolerance_normalized=_finite_number(
                value.get("tolerance_normalized"),
                f"{label}.tolerance_normalized",
            ),
            face_candidate_degree_counts=_compact_nonnegative_int_counter(
                value.get("face_candidate_degree_counts"),
                label=f"{label}.face_candidate_degree_counts",
            ),
            face_matching_count_capped=_integer(
                value.get("face_matching_count_capped"),
                f"{label}.face_matching_count_capped",
            ),
        )
        if (
            result["tolerance_normalized"]
            != STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ):
            raise RuntimeError(
                f"{label}.tolerance_normalized drifted from the frozen threshold"
            )
        if result["face_matching_count_capped"] > 2:
            raise RuntimeError(
                f"{label}.face_matching_count_capped exceeds its cap of two"
            )
    if STEP_GEOMETRY_INCIDENCE_VERTEX_KEYS <= keys:
        result.update(
            vertex_proof_method=_nonempty_string(
                value.get("vertex_proof_method"), f"{label}.vertex_proof_method"
            ),
            vertex_tolerance_normalized=_finite_number(
                value.get("vertex_tolerance_normalized"),
                f"{label}.vertex_tolerance_normalized",
            ),
            vertex_candidate_degree_counts=_compact_nonnegative_int_counter(
                value.get("vertex_candidate_degree_counts"),
                label=f"{label}.vertex_candidate_degree_counts",
            ),
        )
        if result["vertex_proof_method"] != STEP_VERTEX_IDENTITY_PROOF_METHOD:
            raise RuntimeError(f"{label}.vertex_proof_method drifted")
        if (
            result["vertex_tolerance_normalized"]
            != STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ):
            raise RuntimeError(
                f"{label}.vertex_tolerance_normalized drifted from the frozen threshold"
            )
    for key in (
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
    ):
        if key in value:
            result[key] = _integer(value.get(key), f"{label}.{key}")
    for key in ("mapped_face_count", "mapped_edge_occurrence_count"):
        if key in value:
            result[key] = _integer(value.get(key), f"{label}.{key}")
    if result.get("vertex_matching_count_capped", 0) > 2:
        raise RuntimeError(f"{label}.vertex_matching_count_capped exceeds its cap of two")
    if status == "exact_geometry_incidence":
        if result["failure_codes"]:
            raise RuntimeError(f"{label} claims exactness with failure codes")
        if result["vertex_proof_status"] != "exact":
            raise RuntimeError(f"{label} exact claim lacks an exact vertex proof")
    else:
        if not result["failure_codes"]:
            raise RuntimeError(f"{label} non-exact status lacks failure codes")
        if keys == STEP_GEOMETRY_INCIDENCE_COMMON_KEYS:
            if (
                status != "unavailable"
                or result["failure_codes"] != ["source_step_face_count_mismatch"]
                or result["vertex_proof_status"] != "not_evaluated"
            ):
                raise RuntimeError(f"{label} face-count short circuit drifted")
        elif not (STEP_GEOMETRY_INCIDENCE_VERTEX_KEYS <= keys):
            if result["vertex_proof_status"] != "not_evaluated":
                raise RuntimeError(f"{label} pre-vertex short circuit drifted")
        elif result["vertex_proof_status"] not in {"unavailable", "ambiguous"}:
            raise RuntimeError(f"{label} non-exact vertex proof status drifted")
    return result


def _compact_evidence(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_allowed_keys(
        value,
        EVIDENCE_SCALAR_KEYS
        | EVIDENCE_NESTED_KEYS
        | {"validity_components", "schema_v2"},
        label,
    )
    # Copy only the finite logical evidence registered by the census protocol.
    # Native or newly invented diagnostics require an explicit archive-schema
    # review; silently guessing their meaning would weaken the evidence gate.
    result: dict[str, Any] = {}
    for key in sorted(EVIDENCE_SCALAR_KEYS):
        if key not in value:
            continue
        item = value[key]
        if key == "step_sha256" and item is not None:
            item = _require_sha256(item, f"{label}.{key}")
        elif key == "step_bytes" and item is not None:
            item = _integer(item, f"{label}.{key}", positive=True)
        else:
            item = _compact_metric(item, label=f"{label}.{key}")
        result[key] = item
    if "source_vertex_lineage" in value:
        result["source_vertex_lineage"] = _compact_source_vertex_lineage(
            value.get("source_vertex_lineage"),
            label=f"{label}.source_vertex_lineage",
        )
    if "stage_local_occ_topology_proof" in value:
        result["stage_local_occ_topology_proof"] = (
            _compact_stage_local_occ_topology_proof(
                value.get("stage_local_occ_topology_proof"),
                label=f"{label}.stage_local_occ_topology_proof",
            )
        )
    if "step_geometry_incidence_proof" in value:
        result["step_geometry_incidence_proof"] = (
            _compact_step_geometry_incidence_proof(
                value.get("step_geometry_incidence_proof"),
                label=f"{label}.step_geometry_incidence_proof",
            )
        )
    components = value.get("validity_components")
    if components is not None:
        if not isinstance(components, Mapping):
            raise RuntimeError(f"{label}.validity_components must be a mapping")
        result["validity_components"] = {
            key: _compact_metric(
                components[key], label=f"{label}.validity_components.{key}"
            )
            for key in sorted(VALIDITY_COMPONENT_KEYS)
            if key in components
        }
    schema_v2 = value.get("schema_v2")
    if schema_v2 is not None:
        if not isinstance(schema_v2, Mapping):
            raise RuntimeError(f"{label}.schema_v2 must be a mapping")
        _require_exact_keys(
            schema_v2,
            {"applicable_to_census_authorization", "measurement"},
            f"{label}.schema_v2",
        )
        if schema_v2.get("applicable_to_census_authorization") is not False:
            raise RuntimeError(
                f"{label}.schema_v2 cannot authorize a census repair"
            )
        measurement = schema_v2.get("measurement")
        if not isinstance(measurement, Mapping):
            raise RuntimeError(f"{label}.schema_v2.measurement must be a mapping")
        _assert_git_safe_json(measurement, label=f"{label}.schema_v2.measurement")
        safe_measurement = {
            key: measurement.get(key)
            for key in sorted(SCHEMA_V2_MEASUREMENT_KEYS)
            if key in measurement
        }
        if set(safe_measurement) != SCHEMA_V2_MEASUREMENT_KEYS:
            raise RuntimeError(
                f"{label}.schema_v2.measurement lacks the registered decision fields"
            )
        result["schema_v2"] = {
            "applicable_to_census_authorization": False,
            "measurement": safe_measurement,
            "measurement_sha256": canonical_sha256(measurement),
        }
    return result or None


def _source_counts_for_evidence_contract(
    source_topology: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    counts = source_topology.get("counts", source_topology)
    if not isinstance(counts, Mapping):
        raise RuntimeError(f"{label} source counts must be a mapping")
    required = {
        "face_count",
        "edge_count",
        "vertex_count",
        "face_edge_occurrence_count",
        "face_edge_source_ids",
        "edge_vertex_source_ids",
    }
    if not required <= set(counts):
        raise RuntimeError(f"{label} source topology lacks proof populations")
    return counts


def _completed_population(
    stage: Mapping[str, Any],
    *,
    full_count: int,
    expected_entity_kind: str,
    label: str,
) -> list[int] | None:
    lineage = stage.get("lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError(f"{label}.lineage must be a mapping")
    if lineage.get("exact") is True:
        return list(range(full_count))
    if not (
        lineage.get("prefix_pass_exact") is True
        or lineage.get("local_failure_exact") is True
    ):
        return None
    scope = lineage.get("distributed_scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError(f"{label} exact prefix lacks distributed scope")
    if scope.get("entity_kind") != expected_entity_kind:
        raise RuntimeError(f"{label} distributed entity kind drifted")
    completed = scope.get("completed_ids")
    if not isinstance(completed, list) or any(type(item) is not int for item in completed):
        raise RuntimeError(f"{label} completed population is malformed")
    if completed != list(range(len(completed))) or len(completed) > full_count:
        raise RuntimeError(f"{label} completed population is not a canonical prefix")
    return completed


def _require_exact_source_vertex_lineage(
    evidence: Mapping[str, Any],
    *,
    expected_vertex_count: int,
    expected_constraint_occurrences: int,
    required: bool,
    label: str,
) -> None:
    raw = evidence.get("source_vertex_lineage")
    if raw is None:
        if required:
            raise RuntimeError(f"{label}.source_vertex_lineage is required")
        return
    proof = _compact_source_vertex_lineage(
        raw, label=f"{label}.source_vertex_lineage"
    )
    expected = {
        "status": "exact_identity",
        "proof_method": GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": expected_vertex_count,
        "observed_vertex_count": expected_vertex_count,
        "mapped_source_vertex_count": expected_vertex_count,
        "mapped_observed_vertex_count": expected_vertex_count,
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": expected_constraint_occurrences,
        "failure_codes": [],
    }
    if proof != expected:
        raise RuntimeError(f"{label}.source_vertex_lineage exact proof drifted")


def _require_exact_stage_local_occ_topology_proof(
    evidence: Mapping[str, Any],
    *,
    stage: str,
    expected_scope_count: int,
    expected_source_edge_count: int,
    expected_constraint_occurrences: int,
    label: str,
) -> None:
    raw = evidence.get("stage_local_occ_topology_proof")
    if raw is None:
        raise RuntimeError(
            f"{label}.stage_local_occ_topology_proof is required"
        )
    proof = _compact_stage_local_occ_topology_proof(
        raw, label=f"{label}.stage_local_occ_topology_proof"
    )
    expected = {
        "status": "exact_stage_local_topology",
        "proof_method": STAGE_LOCAL_OCC_TOPOLOGY_PROOF_METHOD,
        "scope_kind": "source_edge" if stage == "S2" else "source_face",
        "scope_count": expected_scope_count,
        "source_edge_count": expected_source_edge_count,
        "constraint_occurrence_count": expected_constraint_occurrences,
        "max_observed_per_source_within_scope": 1,
        "max_source_per_observed_within_scope": 1,
        "failure_codes": [],
    }
    if proof != expected:
        raise RuntimeError(
            f"{label}.stage_local_occ_topology_proof exact proof drifted"
        )


def _validate_exact_step_geometry_proof(
    evidence: Mapping[str, Any],
    *,
    counts: Mapping[str, Any],
    label: str,
) -> None:
    raw = evidence.get("step_geometry_incidence_proof")
    if raw is None:
        raise RuntimeError(f"{label}.step_geometry_incidence_proof is required")
    proof = _compact_step_geometry_incidence_proof(
        raw, label=f"{label}.step_geometry_incidence_proof"
    )
    face_count = int(counts["face_count"])
    edge_count = int(counts["edge_count"])
    vertex_count = int(counts["vertex_count"])
    occurrence_count = int(counts["face_edge_occurrence_count"])
    self_loop_count = sum(
        row[0] == row[1] for row in counts["edge_vertex_source_ids"]
    )
    required = {
        "status": "exact_geometry_incidence",
        "failure_codes": [],
        "tolerance_normalized": STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED,
        "face_matching_count_capped": 1,
        "vertex_proof_required": True,
        "mapped_face_count": face_count,
        "mapped_edge_occurrence_count": occurrence_count,
        "vertex_proof_status": "exact",
        "vertex_proof_method": STEP_VERTEX_IDENTITY_PROOF_METHOD,
        "vertex_tolerance_normalized": (
            STEP_GEOMETRY_INCIDENCE_TOLERANCE_NORMALIZED
        ),
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
    }
    for key, expected in required.items():
        if proof.get(key) != expected:
            raise RuntimeError(
                f"{label}.step_geometry_incidence_proof.{key} drifted"
            )
    for key, expected_population in (
        ("face_candidate_degree_counts", face_count),
        ("vertex_candidate_degree_counts", vertex_count),
    ):
        distribution = proof[key]
        if sum(distribution.values()) != expected_population or any(
            int(degree) < 1 and population > 0
            for degree, population in distribution.items()
        ):
            raise RuntimeError(
                f"{label}.step_geometry_incidence_proof.{key} coverage drifted"
            )


def _validate_nonexact_step_geometry_proof(
    proof: Mapping[str, Any],
    *,
    counts: Mapping[str, Any],
    label: str,
) -> None:
    """Bind a short-circuit S7 observation to its measured failure family."""

    status = proof.get("status")
    failures = proof.get("failure_codes")
    if status not in {"unavailable", "ambiguous"} or not isinstance(failures, list):
        raise RuntimeError(f"{label} non-exact status is malformed")
    keys = set(proof)
    if keys == set(STEP_GEOMETRY_INCIDENCE_COMMON_KEYS):
        if (
            status != "unavailable"
            or failures != ["source_step_face_count_mismatch"]
            or proof.get("vertex_proof_status") != "not_evaluated"
        ):
            raise RuntimeError(f"{label} face-count short circuit drifted")
        return

    face_count = int(counts["face_count"])
    face_distribution = proof.get("face_candidate_degree_counts")
    if (
        not isinstance(face_distribution, Mapping)
        or sum(face_distribution.values()) != face_count
    ):
        raise RuntimeError(f"{label} face candidate population drifted")
    face_matching_count = proof.get("face_matching_count_capped")
    failure = failures[0] if len(failures) == 1 else None
    if failure == "face_assignment_has_no_perfect_matching":
        if (
            status != "unavailable"
            or face_matching_count != 0
            or int(face_distribution.get("0", 0)) <= 0
            or proof.get("vertex_proof_status") != "not_evaluated"
        ):
            raise RuntimeError(f"{label} face no-match semantics drifted")
        return
    if failure == "face_assignment_not_unique":
        if (
            status != "ambiguous"
            or face_matching_count != 2
            or proof.get("vertex_proof_status") != "not_evaluated"
        ):
            raise RuntimeError(f"{label} face ambiguity semantics drifted")
        return

    if face_matching_count != 1:
        raise RuntimeError(f"{label} downstream failure lacks a unique face match")
    has_vertex_fields = STEP_GEOMETRY_INCIDENCE_VERTEX_KEYS <= keys
    if not has_vertex_fields:
        if proof.get("vertex_proof_status") != "not_evaluated":
            raise RuntimeError(f"{label} pre-vertex short circuit drifted")
        # Edge/global-incidence failures are emitted only after a unique face
        # assignment.  Their stable producer families are deliberately narrow.
        registered = bool(
            isinstance(failure, str)
            and (
                re.fullmatch(
                    r"source_face_[0-9]+_edge_assignment_"
                    r"(?:has_no_perfect_matching|not_unique)",
                    failure,
                )
                or failure == "global_edge_occurrence_invalid"
                or re.fullmatch(
                    r"source_edge_[0-9]+_split_after_step", failure
                )
                or re.fullmatch(
                    r"source_edges_[0-9]+_[0-9]+_merged_after_step", failure
                )
            )
        )
        if not registered:
            raise RuntimeError(f"{label} pre-vertex failure code is not registered")
        expected_status = "ambiguous" if failure.endswith("_not_unique") else "unavailable"
        if status != expected_status:
            raise RuntimeError(f"{label} pre-vertex failure status drifted")
        return

    vertex_status = proof.get("vertex_proof_status")
    if vertex_status != status:
        raise RuntimeError(f"{label} vertex and outer statuses disagree")
    vertex_matching_count = proof.get("vertex_matching_count_capped")
    if failure == "vertex_assignment_has_no_perfect_matching":
        if status != "unavailable" or vertex_matching_count != 0:
            raise RuntimeError(f"{label} vertex no-match semantics drifted")
    elif failure == "vertex_assignment_not_unique":
        if status != "ambiguous" or vertex_matching_count != 2:
            raise RuntimeError(f"{label} vertex ambiguity semantics drifted")
    elif not (
        isinstance(failure, str)
        and (
            failure.startswith("vertex_proof_")
            or failure.startswith("mapped_source_edge_")
            or failure.startswith("source_vertex_")
            or failure.startswith("step_vertex_")
            or re.fullmatch(r"source_edge_[0-9]+_.+", failure)
            or failure == "source_step_vertex_count_mismatch"
        )
    ):
        raise RuntimeError(f"{label} vertex failure code is not registered")


def _validate_stage_evidence_contract(
    stage: Mapping[str, Any],
    *,
    source_topology: Mapping[str, Any],
    label: str,
) -> None:
    """Independently bind exact endpoint/vertex evidence to source populations."""

    if stage.get("status") != "observed":
        return
    stage_name = stage.get("stage")
    lineage = stage.get("lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError(f"{label}.lineage must be a mapping")
    is_exact_claim = any(
        lineage.get(key) is True
        for key in ("exact", "prefix_pass_exact", "local_failure_exact")
    )
    evidence = stage.get("evidence")
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise RuntimeError(f"{label}.evidence must be a mapping")
    counts = _source_counts_for_evidence_contract(
        source_topology, label=label
    )
    face_rows = counts["face_edge_source_ids"]

    if stage_name == "S2" and is_exact_claim:
        completed_edges = _completed_population(
            stage,
            full_count=int(counts["edge_count"]),
            expected_entity_kind="source_edge",
            label=label,
        )
        if completed_edges is None:
            raise RuntimeError(f"{label} exact S2 population is unavailable")
        _require_exact_stage_local_occ_topology_proof(
            evidence,
            stage="S2",
            expected_scope_count=len(completed_edges),
            expected_source_edge_count=len(completed_edges),
            expected_constraint_occurrences=len(completed_edges),
            label=label,
        )

    if stage_name in {"S3", "S4"} and is_exact_claim:
        completed_faces = _completed_population(
            stage,
            full_count=int(counts["face_count"]),
            expected_entity_kind="source_face",
            label=label,
        )
        if completed_faces is None:
            raise RuntimeError(f"{label} exact face population is unavailable")
        occurrence_count = sum(len(face_rows[face_id]) for face_id in completed_faces)
        edge_ids = sorted(
            {
                int(edge_id)
                for face_id in completed_faces
                for edge_id in face_rows[face_id]
            }
        )
        _require_exact_stage_local_occ_topology_proof(
            evidence,
            stage=str(stage_name),
            expected_scope_count=len(completed_faces),
            expected_source_edge_count=len(edge_ids),
            expected_constraint_occurrences=occurrence_count,
            label=label,
        )

    source_vertex_proof = evidence.get("source_vertex_lineage")
    source_vertex_exact_claim = (
        isinstance(source_vertex_proof, Mapping)
        and source_vertex_proof.get("status") == "exact_identity"
    )
    if stage_name in {"S5", "S6"} and (
        lineage.get("exact") is True or source_vertex_exact_claim
    ):
        _require_exact_source_vertex_lineage(
            evidence,
            expected_vertex_count=int(counts["vertex_count"]),
            expected_constraint_occurrences=int(
                counts["face_edge_occurrence_count"]
            ),
            required=True,
            label=label,
        )

    step_proof = evidence.get("step_geometry_incidence_proof")
    step_exact_claim = (
        isinstance(step_proof, Mapping)
        and step_proof.get("status") == "exact_geometry_incidence"
    )
    if stage_name == "S7" and isinstance(step_proof, Mapping):
        if lineage.get("exact") is True or step_exact_claim:
            _validate_exact_step_geometry_proof(
                evidence, counts=counts, label=label
            )
        else:
            compact_step_proof = _compact_step_geometry_incidence_proof(
                step_proof,
                label=f"{label}.step_geometry_incidence_proof",
            )
            _validate_nonexact_step_geometry_proof(
                compact_step_proof, counts=counts,
                label=f"{label}.step_geometry_incidence_proof",
            )


def _compact_stage(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_allowed_keys(value, STAGE_KEYS, label)
    if value.get("schema") != STAGE_RECORD_SCHEMA:
        raise RuntimeError(f"{label} schema drifted")
    stage = value.get("stage")
    if stage not in STAGE_ORDER:
        raise RuntimeError(f"{label}.stage is not registered")
    result: dict[str, Any] = {
        "schema": value["schema"],
        "stage": stage,
        "phase": value.get("phase"),
        "status": value.get("status"),
        "reached": value.get("reached"),
    }
    if result["status"] == "not_reached":
        if result["reached"] is not False:
            raise RuntimeError(f"{label} not-reached contract drifted")
        result["reason"] = value.get("reason")
        result["blocked_by_stage"] = value.get("blocked_by_stage")
        return result
    if result["status"] != "observed" or result["reached"] is not True:
        raise RuntimeError(f"{label} observed contract drifted")
    required = {
        "lineage",
        "topology",
        "defects",
        "failure",
        "scientifically_bad",
        "bad_reasons",
    }
    if not required <= set(value):
        raise RuntimeError(f"{label} lacks normalized observed fields")
    result.update(
        lineage=_compact_lineage(value["lineage"], label=f"{label}.lineage"),
        topology=_compact_topology(value["topology"], label=f"{label}.topology"),
        defects=_compact_defects(value["defects"], label=f"{label}.defects"),
        failure=_compact_failure(value["failure"], label=f"{label}.failure"),
        scientifically_bad=value["scientifically_bad"],
        bad_reasons=_require_string_list(value["bad_reasons"], f"{label}.bad_reasons"),
    )
    if type(result["scientifically_bad"]) is not bool:
        raise RuntimeError(f"{label}.scientifically_bad must be boolean")
    for key in (
        "construction_native_valid",
        "reimport_native_valid",
        "strict_valid",
    ):
        if key in value:
            if value[key] is not None and type(value[key]) is not bool:
                raise RuntimeError(f"{label}.{key} must be boolean or null")
            result[key] = value[key]
    if "evidence" in value:
        compact_evidence = _compact_evidence(value.get("evidence"), label=f"{label}.evidence")
        if compact_evidence is not None:
            result["evidence"] = compact_evidence
    return result


def _compact_coverage(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    expected = {
        "expected_stages",
        "received_stages",
        "canonical_unique_stages",
        "stage_counts",
        "missing_stages",
        "duplicate_stages",
        "out_of_order",
        "all_stages_accounted",
        "protocol_failures",
    }
    _require_exact_keys(value, expected, label)
    stage_counts = value.get("stage_counts")
    if not isinstance(stage_counts, Mapping) or set(stage_counts) != set(STAGE_ORDER):
        raise RuntimeError(f"{label}.stage_counts drifted")
    return {
        "expected_stages": _require_string_list(value["expected_stages"], f"{label}.expected_stages"),
        "received_stages": _require_string_list(value["received_stages"], f"{label}.received_stages"),
        "canonical_unique_stages": _require_string_list(
            value["canonical_unique_stages"], f"{label}.canonical_unique_stages"
        ),
        "stage_counts": {
            stage: _integer(stage_counts[stage], f"{label}.stage_counts.{stage}")
            for stage in STAGE_ORDER
        },
        "missing_stages": _require_string_list(value["missing_stages"], f"{label}.missing_stages"),
        "duplicate_stages": _require_string_list(value["duplicate_stages"], f"{label}.duplicate_stages"),
        "out_of_order": value["out_of_order"],
        "all_stages_accounted": value["all_stages_accounted"],
        "protocol_failures": _require_string_list(
            value["protocol_failures"], f"{label}.protocol_failures"
        ),
    }


def _compact_first_bad(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    expected = {"status", "stage", "phase", "reasons", "blocked_at_stage"}
    _require_exact_keys(value, expected, label)
    return {
        "status": value.get("status"),
        "stage": value.get("stage"),
        "phase": value.get("phase"),
        "reasons": _require_string_list(value.get("reasons"), f"{label}.reasons"),
        "blocked_at_stage": value.get("blocked_at_stage"),
    }


def _compact_assessment(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    _require_exact_keys(value, ASSESSMENT_KEYS, label)
    if value.get("schema") != ASSESSMENT_SCHEMA:
        raise RuntimeError(f"{label} schema drifted")
    stages = value.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError(f"{label}.stages must be a list")
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise RuntimeError(f"{label}.stages[{index}] must be a mapping")
        _validate_stage_evidence_contract(
            stage,
            source_topology=value["source_topology"],
            label=f"{label}.stages[{index}]",
        )
    drift = value.get("topology_drift_observations")
    if not isinstance(drift, list):
        raise RuntimeError(f"{label}.topology_drift_observations must be a list")
    compact_drift = []
    for index, item in enumerate(drift):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"{label}.topology_drift_observations[{index}] malformed")
        _require_exact_keys(
            item,
            {"stage", "phase", "drifted_fields"},
            f"{label}.topology_drift_observations[{index}]",
        )
        compact_drift.append(
            {
                "stage": item.get("stage"),
                "phase": item.get("phase"),
                "drifted_fields": _require_string_list(
                    item.get("drifted_fields"),
                    f"{label}.topology_drift_observations[{index}].drifted_fields",
                ),
            }
        )
    result = {
        "schema": value["schema"],
        "stage_order": _require_string_list(value["stage_order"], f"{label}.stage_order"),
        "source_topology": _compact_topology(
            value["source_topology"], label=f"{label}.source_topology"
        ),
        "stages": [
            _compact_stage(stage, label=f"{label}.stages[{index}]")
            for index, stage in enumerate(stages)
        ],
        "coverage": _compact_coverage(value["coverage"], label=f"{label}.coverage"),
        "protocol_failures": _require_string_list(
            value["protocol_failures"], f"{label}.protocol_failures"
        ),
        "protocol_failure_count": _integer(
            value["protocol_failure_count"], f"{label}.protocol_failure_count"
        ),
        "inconclusive_reasons": _require_string_list(
            value["inconclusive_reasons"], f"{label}.inconclusive_reasons"
        ),
        "inconclusive_reason_count": _integer(
            value["inconclusive_reason_count"], f"{label}.inconclusive_reason_count"
        ),
        "topology_drift_observations": compact_drift,
        "topology_drift_count": _integer(
            value["topology_drift_count"], f"{label}.topology_drift_count"
        ),
        "observed_bad_stages": _require_string_list(
            value["observed_bad_stages"], f"{label}.observed_bad_stages"
        ),
        "first_bad_inference": _compact_first_bad(
            value["first_bad_inference"], label=f"{label}.first_bad_inference"
        ),
        "first_bad_stage": value.get("first_bad_stage"),
        "first_bad_phase": value.get("first_bad_phase"),
        "first_bad_reasons": _require_string_list(
            value["first_bad_reasons"], f"{label}.first_bad_reasons"
        ),
        "valid_chain": value["valid_chain"],
        "conclusive": value["conclusive"],
    }
    if result["stage_order"] != list(STAGE_ORDER):
        raise RuntimeError(f"{label}.stage_order drifted")
    if [stage.get("stage") for stage in result["stages"]] != list(STAGE_ORDER):
        raise RuntimeError(f"{label}.stages order drifted")
    for key in ("valid_chain", "conclusive"):
        if type(result[key]) is not bool:
            raise RuntimeError(f"{label}.{key} must be boolean")
    return result


def compact_attempt(
    row: Mapping[str, Any], *, canonical_assessment: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the allowlisted archival form of one already-validated row."""

    _require_allowed_keys(row, ROW_KEYS, "attempt")
    step = row.get("step_roundtrip")
    if not isinstance(step, Mapping):
        raise RuntimeError("attempt step_roundtrip must be a mapping")
    _require_exact_keys(
        step,
        {"saved_to_persistent_output", "artifact_id", "bytes", "sha256"},
        "attempt.step_roundtrip",
    )
    saved = step.get("saved_to_persistent_output")
    artifact_id = step.get("artifact_id")
    if type(saved) is not bool:
        raise RuntimeError(
            "attempt.step_roundtrip.saved_to_persistent_output must be boolean"
        )
    if saved:
        if (
            not isinstance(artifact_id, str)
            or artifact_id in {".", ".."}
            or not ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
            or Path(artifact_id).is_absolute()
            or len(Path(artifact_id).parts) != 1
        ):
            raise RuntimeError("attempt.step_roundtrip artifact_id is unsafe")
        step_bytes = _integer(
            step.get("bytes"), "attempt.step_roundtrip.bytes", positive=True
        )
        step_sha256 = _require_sha256(
            step.get("sha256"), "attempt.step_roundtrip.sha256"
        )
    else:
        if artifact_id is not None or step.get("bytes") is not None or step.get("sha256") is not None:
            raise RuntimeError(
                "attempt.step_roundtrip unsaved identity must be entirely null"
            )
        step_bytes = None
        step_sha256 = None
    bindings = {
        "expected": row.get("source_binding_expected"),
        "before_load": row.get("source_binding_before_load"),
        "loaded_bytes": row.get("source_binding_loaded_bytes"),
        "after_load": row.get("source_binding_after_load"),
        "after_measurement": row.get("source_binding_after_measurement"),
        "parent_after_child": row.get("source_binding_parent_after_child"),
    }
    if len({json.dumps(value, sort_keys=True) for value in bindings.values()}) != 1:
        raise RuntimeError("attempt source-binding chain is not exactly equal")
    for name, binding in bindings.items():
        try:
            bindings[name] = normalize_binding(binding)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"attempt source binding {name} is malformed") from exc

    compact_assessment = _compact_assessment(
        canonical_assessment, label="attempt.assessment"
    )
    s7 = compact_assessment["stages"][-1]
    s7_evidence = s7.get("evidence") or {}
    if step_bytes is not None:
        if (
            s7_evidence.get("step_bytes") != step_bytes
            or s7_evidence.get("step_sha256") != step_sha256
        ):
            raise RuntimeError("attempt STEP identity disagrees with S7 evidence")
    elif "step_bytes" in s7_evidence or "step_sha256" in s7_evidence:
        raise RuntimeError("S7 evidence claims STEP identity absent from attempt")

    result: dict[str, Any] = {
        "schema": ARCHIVE_ATTEMPT_SCHEMA,
        "source_schema": row.get("schema"),
        "task_id": row.get("task_id"),
        "task_ordinal": row.get("task_ordinal"),
        "cad_id": row.get("cad_id"),
        "parent_id": row.get("parent_id"),
        "arm": row.get("arm"),
        "profile_name": row.get("profile_name"),
        "switches": row.get("switches"),
        "is_reachability_bridge": row.get("is_reachability_bridge"),
        "counts_as_repair": row.get("counts_as_repair"),
        "denominator": row.get("denominator"),
        "historical_strict_valid": row.get("historical_strict_valid"),
        "run_signature": row.get("run_signature"),
        "status": row.get("status"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "worker_returncode": row.get("worker_returncode"),
        "nonfinite_count": row.get("nonfinite_count"),
        "worker_runtime_abi_sentinel": normalize_runtime_identity(
            row.get("worker_runtime_abi_sentinel")
        ),
        "source_binding_chain": {
            **bindings,
            "all_equal": True,
            "parent_after_child_equal": bindings["parent_after_child"] == bindings["expected"],
            "pickle_bytes_archived": False,
            "path_archived": False,
        },
        "step_roundtrip": {
            "saved_to_persistent_output": saved,
            "bytes": step_bytes,
            "sha256": step_sha256,
            "bytes_archived": False,
            "path_archived": False,
        },
        "source_stage_records_sha256": canonical_sha256(row.get("stage_records")),
        "source_assessment_sha256": canonical_sha256(row.get("assessment")),
        "stage_assessment_recomputed": True,
        "assessment": compact_assessment,
        "raw_worker_logs_archived": False,
        "native_handles_archived": False,
        "original_arrays_archived": False,
    }
    if row.get("error_type") is not None:
        result["error_type"] = row.get("error_type")
    _assert_git_safe_json(result, label="compact_attempt")
    return result


def _validate_repository(payload: Mapping[str, Any]) -> dict[str, Any]:
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise RuntimeError("signed repository binding is missing")
    _require_exact_keys(
        repository,
        {
            "commit", "upstream_commit", "head_matches_upstream",
            "dirty", "formal", "status_sha256", "source_sha256",
        },
        "payload.repository",
    )
    commit = repository.get("commit")
    upstream_commit = repository.get("upstream_commit")
    if (
        not isinstance(commit, str)
        or not COMMIT_PATTERN.fullmatch(commit)
        or not isinstance(upstream_commit, str)
        or not COMMIT_PATTERN.fullmatch(upstream_commit)
    ):
        raise RuntimeError("signed repository commit is invalid")
    if (
        repository.get("head_matches_upstream") is not True
        or upstream_commit != commit
    ):
        raise RuntimeError("formal census commit did not match its upstream")
    if repository.get("dirty") is not False or repository.get("formal") is not True:
        raise RuntimeError("formal census did not use a clean formal worktree")
    status_sha = _require_sha256(
        repository.get("status_sha256"), "repository status binding"
    )
    if status_sha != EMPTY_GIT_STATUS_SHA256:
        raise RuntimeError("clean repository status binding is not SHA-256(empty)")
    source_hashes = repository.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != REQUIRED_SOURCE_HASHES:
        raise RuntimeError("signed source hash population is incomplete or unexpected")
    normalized: dict[str, str] = {}
    for name, digest in sorted(source_hashes.items()):
        path = Path(str(name))
        if path.is_absolute() or ".." in path.parts or "\\" in str(name):
            raise RuntimeError("signed source hash has an unsafe repository path")
        normalized[str(name)] = _require_sha256(digest, f"source hash {name}")
    return {
        "commit": commit,
        "upstream_commit": upstream_commit,
        "head_matches_upstream": True,
        "dirty": False,
        "formal": True,
        "status_sha256": status_sha,
        "source_sha256": normalized,
    }


def _safe_runtime_binary_name(value: Any, *, expected: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != expected
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
        or "\\" in value
    ):
        raise RuntimeError(f"{label} drifted or contains a path")
    return value


def _validate_runtime_identity(payload: Mapping[str, Any]) -> None:
    python = payload.get("python")
    if not isinstance(python, Mapping):
        raise RuntimeError("signed Python runtime identity is missing")
    _require_exact_keys(
        python,
        {
            "implementation", "version", "executable_name",
            "executable_bytes", "executable_sha256",
        },
        "payload.python",
    )
    if python != FROZEN_RUNTIME_IDENTITY["python"]:
        raise RuntimeError("signed Python runtime differs from the frozen runtime")
    _safe_runtime_binary_name(
        python.get("executable_name"), expected="python.exe",
        label="payload.python.executable_name",
    )
    _integer(python.get("executable_bytes"), "Python executable bytes", positive=True)
    _require_sha256(python.get("executable_sha256"), "Python executable binding")

    native = payload.get("native_runtime")
    if not isinstance(native, Mapping):
        raise RuntimeError("signed native runtime identity is missing")
    _require_exact_keys(
        native,
        {"schema", "scope", "process_isolation", "numpy", "pythonocc", "occt"},
        "payload.native_runtime",
    )
    expected_native = {
        key: FROZEN_RUNTIME_IDENTITY[key]
        for key in (
            "schema", "scope", "process_isolation", "numpy", "pythonocc", "occt"
        )
    }
    if native != expected_native:
        raise RuntimeError("signed native runtime differs from the frozen runtime")
    if native.get("schema") != RUNTIME_IDENTITY_SCHEMA:
        raise RuntimeError("signed native runtime schema drifted")
    if native.get("scope") != RUNTIME_ABI_SENTINEL_SCOPE:
        raise RuntimeError("signed native runtime ABI sentinel scope drifted")
    isolation = native.get("process_isolation")
    expected_isolation = FROZEN_RUNTIME_IDENTITY["process_isolation"]
    if type(isolation) is not dict or isolation != expected_isolation:
        raise RuntimeError("signed native runtime process isolation drifted")
    try:
        reconstructed = runtime_abi_sentinel_from_payload(payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("signed runtime ABI sentinel is malformed") from exc
    if reconstructed != FROZEN_RUNTIME_IDENTITY:
        raise RuntimeError("signed runtime ABI sentinel differs from frozen")
    numpy = native.get("numpy")
    pythonocc = native.get("pythonocc")
    occt = native.get("occt")
    if not all(isinstance(value, Mapping) for value in (numpy, pythonocc, occt)):
        raise RuntimeError("signed native runtime components are malformed")
    _require_exact_keys(numpy, {"version"}, "payload.native_runtime.numpy")
    _require_exact_keys(
        pythonocc,
        {
            "version", "wrapper_binary_name", "wrapper_binary_bytes",
            "wrapper_binary_sha256",
        },
        "payload.native_runtime.pythonocc",
    )
    _require_exact_keys(
        occt,
        {
            "version_source", "file_version", "product_version",
            "kernel_binary_name", "kernel_binary_bytes",
            "kernel_binary_sha256",
        },
        "payload.native_runtime.occt",
    )
    for value, label in (
        (numpy.get("version"), "NumPy version"),
        (pythonocc.get("version"), "pythonocc version"),
        (occt.get("file_version"), "OCCT file version"),
        (occt.get("product_version"), "OCCT product version"),
    ):
        _nonempty_string(value, label)
    _safe_runtime_binary_name(
        pythonocc.get("wrapper_binary_name"), expected="_Standard.pyd",
        label="payload.native_runtime.pythonocc.wrapper_binary_name",
    )
    _safe_runtime_binary_name(
        occt.get("kernel_binary_name"), expected="TKernel.dll",
        label="payload.native_runtime.occt.kernel_binary_name",
    )
    _integer(
        pythonocc.get("wrapper_binary_bytes"), "pythonocc wrapper bytes",
        positive=True,
    )
    _integer(occt.get("kernel_binary_bytes"), "OCCT kernel bytes", positive=True)
    _require_sha256(
        pythonocc.get("wrapper_binary_sha256"), "pythonocc wrapper binding"
    )
    _require_sha256(occt.get("kernel_binary_sha256"), "OCCT kernel binding")
    if occt.get("version_source") != "TKernel.dll PE VS_FIXEDFILEINFO":
        raise RuntimeError("OCCT version source drifted")


def _validate_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require_exact_keys(payload, PAYLOAD_KEYS, "signed payload")
    if payload.get("schema") != RUN_SCHEMA or payload.get("run_kind") != "formal":
        raise RuntimeError("signed payload is not the formal census protocol")
    if payload.get("authorization_ceiling") != "exact_candidate_design_only":
        raise RuntimeError("signed authorization ceiling drifted")
    frozen_inputs = {
        "calibration_manifest_sha256": FROZEN_CALIBRATION_MANIFEST_SHA256,
        "selector_matrix_sha256": FROZEN_SELECTOR_MATRIX_SHA256,
    }
    for key, expected_digest in frozen_inputs.items():
        observed_digest = _require_sha256(payload.get(key), f"signed input {key}")
        if observed_digest != expected_digest:
            raise RuntimeError(f"signed input {key} differs from its frozen SHA-256")

    selector_run = payload.get("selector_run")
    if not isinstance(selector_run, Mapping):
        raise RuntimeError("signed selector run binding is missing")
    _require_exact_keys(
        selector_run, {"bytes", "sha256", "signature", "status"}, "payload.selector_run"
    )
    if selector_run.get("status") != "COMPLETED":
        raise RuntimeError("signed selector run is not completed")
    _integer(selector_run.get("bytes"), "selector run bytes", positive=True)
    selector_run_sha256 = _require_sha256(
        selector_run.get("sha256"), "selector run sha256"
    )
    if selector_run_sha256 != FROZEN_SELECTOR_RUN_SHA256:
        raise RuntimeError("signed selector run differs from its frozen SHA-256")
    _require_sha256(selector_run.get("signature"), "selector run signature")

    expected_residual = sorted(
        set(TARGET_CAD_IDS) | set(EXCLUDED_EXACT_NEGATIVE_CAD_IDS)
    )
    selector = payload.get("selector")
    if not isinstance(selector, Mapping):
        raise RuntimeError("signed selector summary is missing")
    _require_exact_keys(
        selector,
        {
            "cohort_count",
            "strict_valid",
            "historical_valid_preserved",
            "regressions",
            "residual_cad_ids",
        },
        "payload.selector",
    )
    if selector != {
        "cohort_count": 100,
        "strict_valid": 91,
        "historical_valid_preserved": 84,
        "regressions": 0,
        "residual_cad_ids": expected_residual,
    }:
        raise RuntimeError("signed selector 100/91/84/0 cohort contract drifted")

    if payload.get("exact_negative_evidence") != dict(EXACT_NEGATIVE_EVIDENCE):
        raise RuntimeError("signed exact-negative evidence drifted")
    if payload.get("excluded_exact_negative_cad_ids") != sorted(
        EXCLUDED_EXACT_NEGATIVE_CAD_IDS
    ):
        raise RuntimeError("signed exact-negative exclusion set drifted")
    if payload.get("ordered_target_cad_ids") != list(TARGET_CAD_IDS):
        raise RuntimeError("signed seven-CAD order drifted")
    expected_tasks = json.loads(
        json.dumps([asdict(task) | {"task_id": task.task_id} for task in TASKS])
    )
    if payload.get("ordered_tasks") != expected_tasks:
        raise RuntimeError("signed ten-task order or arm contract drifted")

    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != len(TARGET_CAD_IDS):
        raise RuntimeError("signed source inventory must contain seven CADs")
    source_by_id: dict[str, dict[str, Any]] = {}
    parent_ids: set[str] = set()
    for expected_cad, source in zip(TARGET_CAD_IDS, sources):
        if not isinstance(source, Mapping):
            raise RuntimeError("signed source entry is not a mapping")
        _require_exact_keys(
            source,
            {
                "cad_id",
                "parent_id",
                "historical_strict_valid",
                "selector_strict_valid",
                "binding",
            },
            f"payload.sources[{expected_cad}]",
        )
        if source.get("cad_id") != expected_cad:
            raise RuntimeError("signed source inventory order or identity drifted")
        parent = source.get("parent_id")
        if not isinstance(parent, str) or not parent or parent in parent_ids:
            raise RuntimeError("signed source parent identities are missing or duplicate")
        parent_ids.add(parent)
        if source.get("historical_strict_valid") is not False:
            raise RuntimeError("census target is not a historical invalid residual")
        if source.get("selector_strict_valid") is not False:
            raise RuntimeError("census target is not a selector residual")
        try:
            binding = normalize_binding(source.get("binding"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("signed source binding is malformed") from exc
        source_by_id[expected_cad] = {
            "cad_id": expected_cad,
            "parent_id": parent,
            "brep_valid": False,
            "binding": binding,
        }

    expected_stages = [
        {"stage": stage, "phase": TASK_STAGE_NAMES[stage]}
        for stage in STAGE_ORDER
    ]
    if payload.get("stages") != expected_stages:
        raise RuntimeError("signed S1-S7 registry drifted")
    if payload.get("stage_record_schema") != STAGE_RECORD_SCHEMA:
        raise RuntimeError("signed stage record schema drifted")
    if payload.get("stage_assessment_schema") != ASSESSMENT_SCHEMA:
        raise RuntimeError("signed stage assessment schema drifted")

    expected_schema_v2 = {
        "identity": "assembly-selector-geometry-gate-v2",
        "max_bbox_relative_delta": 0.02,
        "max_edge_length_relative_delta": 0.05,
        "max_edge_sample_rms_normalized": 0.01,
        "max_edge_sample_max_normalized": 0.05,
        "unchanged": True,
    }
    if payload.get("schema_v2") != expected_schema_v2:
        raise RuntimeError("signed schema-v2 thresholds were changed or relaxed")
    if payload.get("joint_iterations") != 200:
        raise RuntimeError("signed joint-optimization iteration count drifted")
    timeout = _finite_number(
        payload.get("worker_timeout_seconds"), "worker timeout"
    )
    if timeout <= 0:
        raise RuntimeError("signed worker timeout must be positive")

    _validate_runtime_identity(payload)

    runtime = payload.get("breparg_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("signed BrepARG runtime binding is missing")
    _require_exact_keys(runtime, {"utils_sha256"}, "payload.breparg_runtime")
    runtime_digest = _require_sha256(
        runtime.get("utils_sha256"), "BrepARG utils binding"
    )
    if runtime_digest != FROZEN_BREPARG_UTILS_SHA256:
        raise RuntimeError("BrepARG runtime differs from the frozen census runtime")

    repository = _validate_repository(payload)
    _assert_git_safe_json(payload, label="signed_payload")
    return repository, source_by_id


# Imported lazily into a plain mapping so this archive module stays coupled to
# the public stage registry rather than native assembly implementation details.
try:
    from .assembly_stage_lineage import STAGE_NAMES as TASK_STAGE_NAMES
except ImportError:  # pragma: no cover
    from assembly_stage_lineage import STAGE_NAMES as TASK_STAGE_NAMES


def _validate_summary(summary: Mapping[str, Any]) -> None:
    _require_exact_keys(summary, SUMMARY_KEYS, "summary")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise RuntimeError("summary schema drifted")
    expected_scalars = {
        "attempts": 10,
        "denominator_rows": 10,
        "primary_controls": 7,
        "curve_interpolate_bridges": 3,
        "primary_completed_or_scientific": 7,
        "bridge_completed_or_scientific": 3,
        "worker_or_protocol_failures": 0,
        "source_binding_failures": 0,
        "nonfinite_count": 0,
        "protocol_health": True,
        "census_conclusive": True,
        "protocol_conclusive": True,
        "bridge_results_are_reachability_only": True,
        "bridge_repairs_counted": 0,
        "selector_strict_valid_before": 91,
        "selector_strict_valid_after": 91,
        "authorizes_repair": False,
        "authorizes_residual_expansion": False,
        "authorizes_full_100cad": False,
        "authorizes_selector_score_change": False,
        "authorizes_schema_v2_relaxation": False,
        "authorizes_training": False,
        "authorizes_sequence_generation": False,
        "authorizes_ar": False,
    }
    for key, expected in expected_scalars.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"formal census summary gate {key} drifted")
    design = summary.get("authorizes_exact_candidate_design")
    if type(design) is not bool:
        raise RuntimeError("summary exact-candidate authorization is not boolean")
    expected_decision = (
        "AUTHORIZE_EXACT_CANDIDATE_DESIGN"
        if design
        else "INCONCLUSIVE_NO_CANDIDATE_AUTHORIZATION"
    )
    if summary.get("decision") != expected_decision:
        raise RuntimeError("summary decision contradicts candidate authorization")
    first_bad = summary.get("first_bad")
    if not isinstance(first_bad, list) or len(first_bad) != len(TASKS):
        raise RuntimeError("summary lacks the exact ten first-bad rows")
    if [item.get("task_id") for item in first_bad if isinstance(item, Mapping)] != [
        task.task_id for task in TASKS
    ]:
        raise RuntimeError("summary first-bad task order drifted")
    _assert_git_safe_json(summary, label="summary")


def _readme(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for row in rows:
        assessment = row.get("assessment") or {}
        lines.append(
            f"- `{row['task_id']}`: status `{row['status']}`; first bad "
            f"`{assessment.get('first_bad_stage')}`; conclusive "
            f"`{str(bool(assessment.get('conclusive'))).lower()}`."
        )
    return f"""# Source-bound seven-CAD assembly-stage census

This directory is the compact Git-safe snapshot of the signed ten-cell census:
seven unchanged primary controls and three curve-interpolation reachability
bridges.  The archive was emitted only after independently checking the run
signature, terminal row and summary hashes, exact task order, all six source
bindings per row (including the parent re-hash after the child exited), every
stage assessment, and an exact `summarize(rows)` recomputation.

## Result

- Decision: `{summary['decision']}`
- Protocol conclusive: `{str(summary['protocol_conclusive']).lower()}`
- Denominator: `{summary['denominator_rows']}/10`
- Primary controls: `{summary['primary_controls']}/7`
- Reachability bridges: `{summary['curve_interpolate_bridges']}/3`
- Worker/protocol failures: `{summary['worker_or_protocol_failures']}`
- Source-binding failures: `{summary['source_binding_failures']}`
- Non-finite observations: `{summary['nonfinite_count']}`
- Selector score remains: `{summary['selector_strict_valid_after']}/100`

{chr(10).join(lines)}

The three bridge rows establish reachability only.  They do not count as
repairs, do not increase the selector score, and do not relax schema-v2.  This
census can authorize only the design of one exact-CAD candidate; it does not
authorize a repair, residual-family or 100-CAD expansion, training, sequence
generation, or AR.

No STEP or pickle bytes, worker stdout/stderr, machine-local path, upstream
source tree, checkpoint, original NumPy array, or OCC/native handle is
archived.  Source, STEP, input, runtime, code, stage, and assessment identities
are retained only as compact logical fields, byte counts, and cryptographic
hashes.
"""


def _artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(Path(report_dir).iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"archive text is not canonical LF: {path.name}")
        values.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return values


def _validate_report_inventory(report_dir: Path) -> None:
    inventory = list(Path(report_dir).iterdir())
    if {path.name for path in inventory} != EXPECTED_REPORT_FILES:
        raise RuntimeError("archive report inventory differs from its allowlist")
    if any(not path.is_file() for path in inventory):
        raise RuntimeError("archive report contains a directory")
    forbidden = [
        path.name
        for path in inventory
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
        or "worker" in path.name.lower()
        or "stdout" in path.name.lower()
        or "stderr" in path.name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"archive report contains forbidden artifacts: {forbidden}")
    for path in inventory:
        if b"\r" in path.read_bytes():
            raise RuntimeError(f"archive text is not canonical LF: {path.name}")


def validate_artifact_manifest(report_dir: Path) -> None:
    """Recheck the exact six-file inventory and every non-self manifest hash."""

    root = Path(report_dir)
    _validate_report_inventory(root)
    manifest = _read_json(root / "artifact_manifest.json")
    _require_exact_keys(manifest, {"schema", "artifacts"}, "artifact manifest")
    if manifest.get("schema") != ARCHIVE_SCHEMA:
        raise RuntimeError("artifact manifest schema drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("artifact manifest entries must be a list")
    expected_names = EXPECTED_REPORT_FILES - {"artifact_manifest.json"}
    if {
        item.get("path") for item in artifacts if isinstance(item, Mapping)
    } != expected_names or len(artifacts) != len(expected_names):
        raise RuntimeError("artifact manifest inventory drifted")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise RuntimeError("artifact manifest entry is not a mapping")
        _require_exact_keys(item, {"path", "bytes", "sha256"}, "artifact manifest entry")
        name = item.get("path")
        if not isinstance(name, str) or name not in expected_names:
            raise RuntimeError("artifact manifest path is not allowlisted")
        path = root / name
        if item.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"artifact size drifted: {name}")
        if item.get("sha256") != sha256_file(path):
            raise RuntimeError(f"artifact hash drifted: {name}")


def snapshot(run_root: Path, report_dir: Path) -> dict[str, Any]:
    """Validate one completed formal run and emit its compact six-file report."""

    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    sources = {
        ROWS_NAME: run_root / ROWS_NAME,
        SUMMARY_NAME: run_root / SUMMARY_NAME,
        RUN_NAME: run_root / RUN_NAME,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise RuntimeError(f"completed census artifact is missing: {name}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise RuntimeError("report directory must be empty")

    rows = _read_jsonl(sources[ROWS_NAME])
    summary = _read_json(sources[SUMMARY_NAME])
    run = _read_json(sources[RUN_NAME])
    _require_exact_keys(run, RUN_KEYS, "run")
    payload = run.get("payload")
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "COMPLETED"
        or run.get("attempts") != len(TASKS)
        or not isinstance(payload, Mapping)
    ):
        raise RuntimeError("archive requires a completed formal ten-task census")
    signature = _require_sha256(run.get("signature"), "run signature")
    if signature != canonical_sha256(payload):
        raise RuntimeError("census run signature does not bind its payload")
    terminal_summary = validate_terminal_artifact_hashes(
        run,
        rows_path=sources[ROWS_NAME],
        summary_path=sources[SUMMARY_NAME],
    )
    if terminal_summary != summary:
        raise RuntimeError("terminal summary reload is inconsistent")

    repository, source_by_id = _validate_payload(payload)
    _assert_git_safe_json(run, label="source_run")
    _assert_git_safe_json(rows, label="source_rows")
    _assert_git_safe_json(summary, label="source_summary")

    expected_task_ids = [task.task_id for task in TASKS]
    if len(rows) != len(TASKS) or [row.get("task_id") for row in rows] != expected_task_ids:
        raise RuntimeError("archive requires the exact ordered ten-task ledger")

    compact_rows: list[dict[str, Any]] = []
    conclusive_assessments = 0
    for row, task in zip(rows, TASKS):
        _require_allowed_keys(row, ROW_KEYS, f"attempt {task.task_id}")
        source = source_by_id[task.cad_id]
        binding = source["binding"]
        try:
            validate_attempt_row(
                row,
                source=source,
                task=task,
                run_signature=signature,
                expected_binding=binding,
                expected_runtime_abi_sentinel=runtime_abi_sentinel_from_payload(
                    payload
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"attempt validation failed for {task.task_id}") from exc
        for name in (
            "source_binding_expected",
            "source_binding_before_load",
            "source_binding_loaded_bytes",
            "source_binding_after_load",
            "source_binding_after_measurement",
            "source_binding_parent_after_child",
        ):
            if row.get(name) != binding:
                raise RuntimeError(f"source binding chain failed for {task.task_id}")

        assessment = row.get("assessment")
        if not isinstance(assessment, Mapping):
            raise RuntimeError(f"attempt {task.task_id} lacks a stage assessment")
        try:
            independently_recomputed = assess_stage_lineage(
                row.get("stage_records") or [],
                source_topology=assessment.get("source_topology") or {},
            )
            canonical_stored = assess_stage_lineage(
                assessment.get("stages") or [],
                source_topology=assessment.get("source_topology") or {},
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"stage assessment recomputation failed for {task.task_id}"
            ) from exc
        if independently_recomputed != canonical_stored:
            raise RuntimeError(f"stage assessment drifted for {task.task_id}")
        conclusive_assessments += int(bool(canonical_stored.get("conclusive")))
        compact_rows.append(
            compact_attempt(row, canonical_assessment=canonical_stored)
        )

    recomputed_summary = summarize(rows)
    if recomputed_summary != summary:
        raise RuntimeError("census summary is not exactly derivable from its rows")
    _validate_summary(summary)

    compact_summary = dict(summary)
    _assert_git_safe_json(compact_summary, label="compact_summary")
    compact_payload = dict(payload)
    compact_payload["repository"] = repository
    _assert_git_safe_json(compact_payload, label="compact_payload")
    source_artifact_bindings = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sources.items()
    }
    compact_run = {
        "schema": ARCHIVE_RUN_SCHEMA,
        "source_schema": RUN_SCHEMA,
        "source_status": "COMPLETED",
        "attempts": len(TASKS),
        "signature": signature,
        "payload": compact_payload,
        "terminal_bindings": {
            "rows_sha256": run.get("rows_sha256"),
            "summary_sha256": run.get("summary_sha256"),
        },
        "source_artifact_bindings": source_artifact_bindings,
        "step_bytes_archived": False,
        "source_pickle_bytes_archived": False,
        "raw_worker_logs_archived": False,
        "checkpoint_bytes_archived": False,
        "native_handles_archived": False,
        "original_arrays_archived": False,
    }
    _assert_git_safe_json(compact_run, label="compact_run")

    validation = {
        "schema": ARCHIVE_SCHEMA,
        "valid": True,
        "archive_integrity_valid_not_repair_success": True,
        "attempts": len(TASKS),
        "denominator_rows": summary["denominator_rows"],
        "primary_controls": summary["primary_controls"],
        "curve_interpolate_bridges": summary["curve_interpolate_bridges"],
        "protocol_conclusive": summary["protocol_conclusive"],
        "scientifically_conclusive_assessments": conclusive_assessments,
        "decision": summary["decision"],
        "run_signature": signature,
        "signature_valid": True,
        "terminal_rows_hash_valid": True,
        "terminal_summary_hash_valid": True,
        "summary_recomputed_equal": True,
        "task_order_valid": True,
        "source_binding_chains_valid": True,
        "parent_after_child_bindings_valid": True,
        "stage_assessments_recomputed_equal": True,
        "repository_binding_valid": True,
        "repository_commit": repository["commit"],
        "repository_source_hash_count": len(repository["source_sha256"]),
        "schema_v2_unchanged": True,
        "bridge_results_are_reachability_only": True,
        "bridge_repairs_counted": 0,
        "selector_strict_valid_before": 91,
        "selector_strict_valid_after": 91,
        "authorizes_training": False,
        "authorizes_sequence_generation": False,
        "authorizes_ar": False,
        "path_free": True,
        "step_bytes_archived": False,
        "source_pickle_bytes_archived": False,
        "raw_worker_logs_archived": False,
        "checkpoint_bytes_archived": False,
        "native_handles_archived": False,
        "original_arrays_archived": False,
        "source_artifact_bindings": source_artifact_bindings,
    }
    _assert_git_safe_json(validation, label="archive_validation")

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(report_dir / ROWS_NAME, compact_rows)
    _write_json(report_dir / SUMMARY_NAME, compact_summary)
    _write_json(report_dir / RUN_NAME, compact_run)
    _write_text_lf(report_dir / "README.md", _readme(summary, rows))
    _write_json(report_dir / "archive_validation.json", validation)
    _write_json(
        report_dir / "artifact_manifest.json",
        {"schema": ARCHIVE_SCHEMA, "artifacts": _artifact_manifest(report_dir)},
    )

    validate_artifact_manifest(report_dir)
    for path in report_dir.iterdir():
        if path.suffix.lower() == ".jsonl":
            archived: Any = _read_jsonl(path)
        elif path.suffix.lower() == ".json":
            archived = _read_json(path)
        else:
            continue
        if path.name != "artifact_manifest.json":
            _assert_git_safe_json(archived, label=f"archive.{path.name}")
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            snapshot(args.run_root, args.report_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
