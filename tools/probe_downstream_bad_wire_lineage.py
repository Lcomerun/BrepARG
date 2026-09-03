"""Trace strict bad-wire evidence through assembly without guessing identity.

The coordinator is deliberately free of Open CASCADE imports.  It binds the
fixed two-CAD cohort, source bytes, selector result, repository revision, and
runtime helper, then starts one isolated child per CAD.  A child observes the
three in-memory face phases exposed by ``construct_brep_directed`` and one
STEP-roundtrip phase.  Missing or ambiguous source lineage is evidence of an
inconclusive experiment; explorer ordinals are never treated as identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

try:
    from .probe_periodic_pcurve_applicability import (
        append_jsonl,
        atomic_json,
        canonical_sha256,
        normalize_binding,
        payload_binding,
        read_jsonl,
        sha256_file,
        source_binding,
    )
except ImportError:  # direct script execution
    from probe_periodic_pcurve_applicability import (
        append_jsonl,
        atomic_json,
        canonical_sha256,
        normalize_binding,
        payload_binding,
        read_jsonl,
        sha256_file,
        source_binding,
    )


SCHEMA = "downstream-bad-wire-lineage-case-v1"
RUN_SCHEMA = "downstream-bad-wire-lineage-run-v1"
SUMMARY_SCHEMA = "downstream-bad-wire-lineage-summary-v1"
WORKER_MARKER = "__DOWNSTREAM_BAD_WIRE_LINEAGE_RESULT__="
PROFILE = "directed_trim_local_intersection_topology"
TARGET_CAD_IDS = (
    "00047472_197769bbdd814278b715d88a_step_000",
    "00063055_e309c689b9b44f0686f47966_step_000",
)
MEMORY_PHASES = (
    "post_add_pcurves_pre_repair",
    "post_optional_face_repair_pre_sewing",
    "post_sewing_pre_step",
)
STEP_PHASE = "post_step_roundtrip"
ALL_PHASES = (*MEMORY_PHASES, STEP_PHASE)
EXACT_LINEAGE_STATUSES = {
    "exact_identity",
    "exact_face_local_geometry",
    "exact_sewing_history",
    "exact_sewing_face_local_geometry",
    "exact_geometry_incidence",
}
STEP_EDGE_TOLERANCE_NORMALIZED = 1e-4
STEP_CURVE_SAMPLE_COUNT = 17
FAILURE_STATUSES = {
    "worker_timeout",
    "worker_process_exit",
    "worker_protocol_error",
    "worker_error",
    "source_binding_mismatch",
    "measurement_incomplete",
}
RUN_MANIFEST_NAME = "downstream_bad_wire_lineage_run.json"
ROWS_NAME = "downstream_bad_wire_lineage_cases.jsonl"
SUMMARY_NAME = "downstream_bad_wire_lineage_summary.json"
LOCK_NAME = ".downstream_bad_wire_lineage_writer.lock"


@contextmanager
def output_writer_lock(output_dir: Path) -> Iterator[None]:
    """Hold this protocol's nonblocking lock while mutating its JSON files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError(
                "downstream-lineage output already has an active writer"
            ) from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_lineage_worker_result(stdout: str) -> dict[str, Any] | None:
    """Accept one sentinel only, and require it to be the final nonempty line."""
    lines = [line for line in str(stdout).splitlines() if line.strip()]
    indices = [
        index for index, line in enumerate(lines) if line.startswith(WORKER_MARKER)
    ]
    if not lines or indices != [len(lines) - 1]:
        return None
    try:
        value = json.loads(lines[-1][len(WORKER_MARKER) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} rows must be mappings")
        cad_id = row.get("cad_id")
        if not isinstance(cad_id, str) or not cad_id:
            raise ValueError(f"{name} CAD ids must be non-empty strings")
        if cad_id in by_id:
            raise ValueError(f"duplicate {name} CAD id: {cad_id}")
        by_id[cad_id] = row
    return by_id


def select_lineage_sources(
    calibration_rows: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
    *,
    target_ids: Sequence[str] = TARGET_CAD_IDS,
) -> list[dict[str, Any]]:
    """Prove the frozen 100 -> current nine -> registered two identity chain."""
    if len(calibration_rows) not in (100, 300):
        raise ValueError("calibration must contain 100 originals or the formal 3x100 arms")
    originals = [row for row in calibration_rows if row.get("arm") == "original"]
    if len(originals) != 100:
        raise ValueError("calibration must contain exactly 100 original rows")
    calibration_by_id = _unique_rows(originals, name="calibration original")
    if any(type(row.get("brep_valid")) is not bool for row in originals):
        raise TypeError("calibration brep_valid values must be booleans")
    if sum(row["brep_valid"] for row in originals) != 84:
        raise ValueError("calibration must contain exactly 84 historically valid CADs")

    arms = sorted({str(row.get("arm")) for row in calibration_rows})
    if len(calibration_rows) == 300 and set(arms) != {
        "original",
        "continuous_bypass_64d",
        "fsq_8192_4d",
    }:
        raise ValueError("formal calibration must contain the registered three arms")
    original_order = [str(row["cad_id"]) for row in originals]
    for arm in arms:
        arm_rows = [row for row in calibration_rows if str(row.get("arm")) == arm]
        if len(arm_rows) != 100:
            raise ValueError(f"calibration arm {arm!r} does not contain 100 rows")
        arm_by_id = _unique_rows(arm_rows, name=f"calibration {arm}")
        if set(arm_by_id) != set(calibration_by_id):
            raise ValueError(f"calibration arm {arm!r} changes the CAD cohort")
        if [str(row["cad_id"]) for row in arm_rows] != original_order:
            raise ValueError(f"calibration arm {arm!r} changes the CAD order")
        for cad_id, original in calibration_by_id.items():
            for field in ("parent_id", "source_path"):
                if arm_by_id[cad_id].get(field) != original.get(field):
                    raise ValueError(
                        f"calibration arm {arm!r} changes {field} for {cad_id}"
                    )

    if len(selector_rows) != 100:
        raise ValueError("selector matrix must contain exactly 100 rows")
    selector_by_id = _unique_rows(selector_rows, name="selector")
    if set(selector_by_id) != set(calibration_by_id):
        raise ValueError("selector and calibration CAD cohorts differ")
    if [str(row["cad_id"]) for row in selector_rows] != original_order:
        raise ValueError("selector and calibration CAD order differ")
    schemas = {row.get("schema") for row in selector_rows if "schema" in row}
    if schemas and schemas != {"assembly-repair-selector-v1"}:
        raise ValueError("selector matrix schema mismatch")
    for cad_id, original in calibration_by_id.items():
        selected = selector_by_id[cad_id]
        if selected.get("parent_id") != original.get("parent_id"):
            raise ValueError(f"selector parent identity drift for {cad_id}")
        if (
            "source_path" in selected
            and selected.get("source_path") != original.get("source_path")
        ):
            raise ValueError(f"selector source identity drift for {cad_id}")
        if type(selected.get("historical_strict_valid")) is not bool:
            raise TypeError("selector historical validity must be boolean")
        if selected["historical_strict_valid"] is not original["brep_valid"]:
            raise ValueError(f"selector historical validity drift for {cad_id}")
        if type(selected.get("strict_brep_valid")) is not bool:
            raise TypeError("selector strict validity must be boolean")
    if sum(row["strict_brep_valid"] for row in selector_rows) != 91:
        raise ValueError("selector must contain exactly 91 strict-valid CADs")
    if any(
        row["historical_strict_valid"] and not row["strict_brep_valid"]
        for row in selector_rows
    ):
        raise ValueError("selector regresses a historically valid control")
    residual_ids = {
        str(row["cad_id"])
        for row in selector_rows
        if row["strict_brep_valid"] is False
    }
    if len(residual_ids) != 9:
        raise ValueError("selector must contain exactly nine strict residuals")

    ordered = [str(value) for value in target_ids]
    if len(ordered) != 2 or len(set(ordered)) != 2:
        raise ValueError("lineage target list must contain two unique CAD ids")
    if not set(ordered).issubset(residual_ids):
        raise ValueError("every lineage target must be a current strict residual")
    for cad_id in ordered:
        selected = selector_by_id[cad_id]
        selection = selected.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError(f"selector target {cad_id} lacks selection evidence")
        if selection.get("primary_profile") != PROFILE:
            raise ValueError(f"selector target {cad_id} primary profile drifted")
        if selection.get("selected_profile") != PROFILE:
            raise ValueError(f"selector target {cad_id} selected profile drifted")
        candidates = selection.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"selector target {cad_id} lacks candidate evidence")
        primary = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and candidate.get("profile") == PROFILE
        ]
        if len(primary) != 1:
            raise ValueError(
                f"selector target {cad_id} must contain one primary candidate"
            )
    return [dict(calibration_by_id[cad_id]) for cad_id in ordered]


def git_identity(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if revision.returncode or status.returncode:
        raise RuntimeError("could not bind repository identity")
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "status_sha256": hashlib.sha256(status.stdout.encode("utf-8")).hexdigest(),
    }


def source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "tools/probe_downstream_bad_wire_lineage.py",
        "tools/probe_periodic_pcurve_applicability.py",
        "tools/directed_trim_assembly.py",
        "tools/diagnose_assembly_face_wires.py",
        "tools/local_wire_topology_repair.py",
        "tools/assembly_repair.py",
        "tools/run_assembly_calibration_oracle.py",
        "tools/run_assembly_repair_matrix.py",
    )
    return {path: sha256_file(Path(repo_root) / path) for path in paths}


def _validate_selector_run(
    selector_run: Path,
    *,
    calibration_manifest: Path,
    selector_matrix: Path,
    source_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = json.loads(Path(selector_run).read_text(encoding="utf-8"))
    payload = value.get("payload") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "assembly-repair-run-v2"
        or value.get("status") != "COMPLETED"
        or value.get("attempts") != 100
        or not isinstance(payload, Mapping)
    ):
        raise ValueError("selector run is not a completed formal 100-CAD run")
    signature = value.get("signature")
    if not isinstance(signature, str) or canonical_sha256(payload) != signature:
        raise ValueError("selector run signature mismatch")
    if payload.get("calibration_manifest_sha256") != sha256_file(calibration_manifest):
        raise ValueError("selector run calibration binding mismatch")
    if value.get("final_matrix_sha256") != sha256_file(selector_matrix):
        raise ValueError("selector run matrix binding mismatch")
    if payload.get("full_cohort_count") != 100 or payload.get(
        "selected_cohort_count"
    ) != 100:
        raise ValueError("selector run cohort count mismatch")
    if (
        payload.get("run_kind") != "assembly-repair-selector-v1"
        or payload.get("matrix_schema") != "assembly-repair-selector-v1"
        or payload.get("candidate_schema") != "assembly-selector-candidate-v1"
        or payload.get("historical_invalid_only") is not False
    ):
        raise ValueError("selector run protocol identity mismatch")
    registered = payload.get("selected_source_pickles")
    if not isinstance(registered, Mapping):
        raise ValueError("selector run lacks source pickle bindings")
    for binding in source_bindings:
        cad_id = str(binding["cad_id"])
        if normalize_binding(registered.get(cad_id) or {}) != normalize_binding(
            {key: binding[key] for key in ("bytes", "sha256")}
        ):
            raise ValueError(f"selector run source binding mismatch for {cad_id}")
    return {
        "bytes": Path(selector_run).stat().st_size,
        "sha256": sha256_file(selector_run),
        "signature": signature,
        "status": "COMPLETED",
    }


def _cohort_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cad_id": str(row["cad_id"]),
            "parent_id": str(row["parent_id"]),
            "historical_strict_valid": bool(row["historical_strict_valid"]),
            "strict_brep_valid": bool(row["strict_brep_valid"]),
        }
        for row in rows
    ]


def build_run_payload(
    args: argparse.Namespace,
    sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_utils = Path(args.breparg_root).resolve() / "utils.py"
    if not runtime_utils.is_file():
        raise FileNotFoundError(runtime_utils)
    bindings = []
    for row in sources:
        source_path = Path(str(row["source_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        bindings.append({"cad_id": str(row["cad_id"]), **source_binding(source_path)})
    repository = {**git_identity(repo_root), "source_sha256": source_hashes(repo_root)}
    if repository["dirty"]:
        raise RuntimeError("formal downstream-lineage probe requires a clean Git worktree")
    return {
        "schema": RUN_SCHEMA,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "selector_matrix_sha256": sha256_file(args.selector_matrix),
        "selector_run": _validate_selector_run(
            args.selector_run,
            calibration_manifest=args.calibration_manifest,
            selector_matrix=args.selector_matrix,
            source_bindings=bindings,
        ),
        "selector_cohort_signature": canonical_sha256(_cohort_identity(selector_rows)),
        "selector_strict_valid": sum(
            row.get("strict_brep_valid") is True for row in selector_rows
        ),
        "selector_historical_strict_valid": sum(
            row.get("historical_strict_valid") is True for row in selector_rows
        ),
        "selector_residual_ids": [
            str(row["cad_id"])
            for row in selector_rows
            if row.get("strict_brep_valid") is False
        ],
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "source_bindings": bindings,
        "profile": PROFILE,
        "memory_phases": list(MEMORY_PHASES),
        "step_phase": STEP_PHASE,
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "repository": repository,
        "breparg_runtime": {"utils_sha256": sha256_file(runtime_utils)},
    }


def validate_bound_inputs(
    args: argparse.Namespace,
    *,
    payload: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> None:
    if build_run_payload(args, sources, selector_rows) != payload:
        raise RuntimeError("signed lineage inputs, runtime, or source code drifted")


def _assert_finite_json(value: Any, *, label: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_json(child, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_finite_json(child, label=f"{label}[{index}]")


def assert_path_free_evidence(value: Any, *, label: str = "evidence") -> None:
    """Reject machine paths or path-bearing keys from Git-safe evidence."""
    forbidden_keys = {"source_path", "step_path", "pickle_path", "output_dir"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in forbidden_keys or key_text.endswith("_path"):
                raise ValueError(f"{label} contains forbidden path key {key!r}")
            assert_path_free_evidence(child, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            assert_path_free_evidence(child, label=f"{label}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if (
            normalized.startswith("/")
            or (len(normalized) >= 3 and normalized[1:3] == ":/")
            or normalized.startswith("//")
        ):
            raise ValueError(f"{label} contains an absolute path")


def assess_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    source_face_count: int,
    source_edge_count: int,
) -> dict[str, Any]:
    """Validate stage population and derive fail-closed evidence counts."""
    if type(source_face_count) is not int or source_face_count <= 0:
        raise ValueError("source_face_count must be a positive integer")
    if type(source_edge_count) is not int or source_edge_count <= 0:
        raise ValueError("source_edge_count must be a positive integer")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence")

    coverage_failures: list[str] = []
    observation_failures: list[str] = []
    mapping_failures: list[str] = []
    phase_counts = Counter()
    memory_indices: dict[str, list[int]] = {phase: [] for phase in MEMORY_PHASES}
    mapped_defects: dict[str, list[dict[str, Any]]] = {phase: [] for phase in ALL_PHASES}

    for observation_index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise TypeError("observation rows must be mappings")
        _assert_finite_json(observation, label=f"observation[{observation_index}]")
        phase = observation.get("phase")
        if phase not in ALL_PHASES:
            raise ValueError("observation phase is not registered")
        phase_counts[str(phase)] += 1
        lineage_status = observation.get("lineage_status")
        row_mapping_failures = observation.get("mapping_failures") or []
        if not isinstance(row_mapping_failures, list) or any(
            not isinstance(value, str) for value in row_mapping_failures
        ):
            raise TypeError("mapping_failures must be a list of strings")
        if lineage_status not in EXACT_LINEAGE_STATUSES or row_mapping_failures:
            mapping_failures.append(f"observation_{observation_index}_lineage_not_exact")

        if phase in MEMORY_PHASES:
            source_face_index = observation.get("source_face_index")
            if (
                type(source_face_index) is not int
                or not 0 <= source_face_index < source_face_count
            ):
                coverage_failures.append(f"{phase}_source_face_index_invalid")
            else:
                memory_indices[str(phase)].append(source_face_index)
        elif observation.get("entity_kind") != "step_shape":
            coverage_failures.append("step_entity_kind_invalid")

        diagnosis = observation.get("diagnosis")
        if not isinstance(diagnosis, Mapping) or diagnosis.get("status") != "diagnosed":
            observation_failures.append(f"observation_{observation_index}_not_diagnosed")
            continue
        occurrences = diagnosis.get("occurrences")
        if not isinstance(occurrences, list):
            observation_failures.append(f"observation_{observation_index}_occurrences_missing")
            continue
        for occurrence_index, occurrence in enumerate(occurrences):
            if not isinstance(occurrence, Mapping):
                observation_failures.append(
                    f"observation_{observation_index}_occurrence_{occurrence_index}_invalid"
                )
                continue
            if occurrence.get("status") != "detected":
                observation_failures.append(
                    f"observation_{observation_index}_occurrence_{occurrence_index}_unavailable"
                )
            mapping_status = occurrence.get("source_mapping_status")
            source_ids = occurrence.get("source_edge_ids")
            source_face_index = occurrence.get(
                "source_face_index", observation.get("source_face_index")
            )
            valid_ids = bool(
                isinstance(source_ids, list)
                and source_ids
                and all(
                    type(edge_id) is int and 0 <= edge_id < source_edge_count
                    for edge_id in source_ids
                )
            )
            valid_face = bool(
                type(source_face_index) is int
                and 0 <= source_face_index < source_face_count
            )
            if mapping_status != "mapped" or not valid_ids or not valid_face:
                mapping_failures.append(
                    f"observation_{observation_index}_occurrence_{occurrence_index}_unmapped"
                )
                continue
            mapped_defects[str(phase)].append(
                {
                    "source_face_index": int(source_face_index),
                    "wire_index": occurrence.get("wire_index"),
                    "kind": occurrence.get("kind"),
                    "source_edge_ids": [int(value) for value in source_ids],
                }
            )

    expected_indices = list(range(source_face_count))
    for phase in MEMORY_PHASES:
        indices = memory_indices[phase]
        if sorted(indices) != expected_indices or len(indices) != len(set(indices)):
            coverage_failures.append(f"{phase}_coverage_incomplete_or_duplicate")
    if phase_counts[STEP_PHASE] != 1:
        coverage_failures.append("step_phase_count_not_one")

    first_bad_phase = None
    first_bad_occurrences: list[dict[str, Any]] = []
    for phase in ALL_PHASES:
        if mapped_defects[phase]:
            first_bad_phase = phase
            first_bad_occurrences = mapped_defects[phase]
            break
    return {
        "phase_counts": {phase: int(phase_counts[phase]) for phase in ALL_PHASES},
        "all_stages_observed": not coverage_failures,
        "coverage_failures": coverage_failures,
        "coverage_failure_count": len(coverage_failures),
        "observation_failures": observation_failures,
        "observation_failure_count": len(observation_failures),
        "mapping_failures": mapping_failures,
        "mapping_failure_count": len(mapping_failures),
        "mapped_defect_count": sum(len(rows) for rows in mapped_defects.values()),
        "first_bad_phase": first_bad_phase,
        "first_bad_occurrences": first_bad_occurrences,
    }


def validate_case_row(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    run_signature: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> None:
    assert_path_free_evidence(row)
    if row.get("schema") != SCHEMA:
        raise ValueError("worker schema mismatch")
    if str(row.get("cad_id")) != str(source["cad_id"]):
        raise ValueError("worker CAD identity mismatch")
    if row.get("parent_id") != source.get("parent_id"):
        raise ValueError("worker parent identity mismatch")
    if row.get("profile") != PROFILE:
        raise ValueError("worker profile mismatch")
    if row.get("run_signature") != run_signature:
        raise ValueError("worker run signature mismatch")
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("worker status must be a non-empty string")
    bound = normalize_binding(row.get("source_binding") or {})
    pre_measurement_failure = status in FAILURE_STATUSES - {"measurement_incomplete"}
    loaded_value = row.get("source_binding_loaded_bytes")
    after_value = row.get("source_binding_after_load")
    if pre_measurement_failure and loaded_value is None and after_value is None:
        loaded = after = bound
    else:
        loaded = normalize_binding(loaded_value or {})
        after = normalize_binding(after_value or {})
    if loaded != bound or after != bound:
        raise ValueError("worker source binding stages differ")
    if expected_binding is not None and bound != normalize_binding(expected_binding):
        raise ValueError("worker source binding mismatches signed input")

    if pre_measurement_failure:
        if row.get("observations") not in ([], None):
            raise ValueError("pre-measurement failure must not claim observations")
        return
    source_face_count = row.get("source_face_count")
    source_edge_count = row.get("source_edge_count")
    observations = row.get("observations")
    assessment = assess_observations(
        observations,
        source_face_count=source_face_count,
        source_edge_count=source_edge_count,
    )
    for key in (
        "phase_counts",
        "all_stages_observed",
        "coverage_failure_count",
        "observation_failure_count",
        "mapping_failure_count",
        "mapped_defect_count",
        "first_bad_phase",
        "first_bad_occurrences",
    ):
        if row.get(key) != assessment[key]:
            raise ValueError(f"derived {key} differs from observation evidence")
    complete = bool(
        assessment["all_stages_observed"]
        and assessment["observation_failure_count"] == 0
        and assessment["mapping_failure_count"] == 0
        and row.get("assembly_status") == "completed"
        and row.get("step_roundtrip_status") == "diagnosed"
    )
    if status == "completed" and not complete:
        raise ValueError("completed worker contains incomplete evidence")
    if status == "measurement_incomplete" and complete:
        raise ValueError("measurement_incomplete worker contains complete evidence")
    if status not in {"completed", "measurement_incomplete"}:
        raise ValueError("unknown post-measurement worker status")


def worker_failure_row(
    source: Mapping[str, Any],
    *,
    run_signature: str,
    status: str,
    returncode: int | None,
    error_type: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": PROFILE,
        "run_signature": run_signature,
        "source_binding": normalize_binding(expected_binding),
        "source_binding_loaded_bytes": None,
        "source_binding_after_load": None,
        "status": status,
        "worker_returncode": returncode,
        "error_type": str(error_type),
        "observations": [],
    }


_OMIT_METADATA_VALUE = object()
_PRIVATE_METADATA_KEYS = {
    "shape",
    "observed_wire",
    "observed_edge",
    "occurrence_edges",
}


def _public_json_metadata_value(value: Any) -> Any:
    """Copy metadata while dropping native OCC handles and private fields.

    Observer metadata is an in-process proof object: ``source_mapping`` in
    particular carries ``TopoDS_Wire``/``TopoDS_Edge`` handles needed by the
    diagnosis call.  Those handles must never cross the JSON evidence
    boundary.  This copier admits only JSON primitives and recursively drops
    explicitly private fields or values of any other type.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, np.generic):
        return _public_json_metadata_value(value.item())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            if key.startswith("_") or key in _PRIVATE_METADATA_KEYS:
                continue
            public_child = _public_json_metadata_value(child)
            if public_child is not _OMIT_METADATA_VALUE:
                result[key] = public_child
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for child in value:
            public_child = _public_json_metadata_value(child)
            if public_child is not _OMIT_METADATA_VALUE:
                result.append(public_child)
        return result
    return _OMIT_METADATA_VALUE


def _source_mapping_summary(value: Any) -> dict[str, Any]:
    """Return only path-free scalar evidence from a private source mapping."""
    if not isinstance(value, Mapping):
        return {
            "status": "unavailable",
            "failure_codes": ["source_mapping_not_mapping"],
            "wire_count": 0,
            "edge_occurrence_count": 0,
        }
    wire_rows = value.get("wire_rows")
    rows = (
        list(wire_rows)
        if isinstance(wire_rows, Sequence) and not isinstance(wire_rows, (str, bytes))
        else []
    )
    edge_occurrence_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidates = row.get("source_edge_candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            edge_occurrence_count += len(candidates)
    failures = value.get("failures")
    failure_codes = (
        [item for item in failures if isinstance(item, str)]
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes))
        else []
    )
    status = value.get("status")
    return {
        "status": status if isinstance(status, str) else "unavailable",
        "failure_codes": failure_codes,
        "wire_count": len(rows),
        "edge_occurrence_count": edge_occurrence_count,
    }


def _compact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "entity_kind",
        "source_face_index",
        "source_loop_edge_uses",
        "outer_loop_index",
        "loop_3d_endpoint_gaps",
        "loop_3d_endpoint_max_gaps",
        "face_3d_endpoint_max_gap",
        "expected_source_face_count",
        "expected_source_edge_count",
        "sewn_face_count",
        "sewing_lineage",
    )
    result: dict[str, Any] = {}
    for key in keys:
        if key not in metadata:
            continue
        public_value = _public_json_metadata_value(metadata[key])
        if public_value is not _OMIT_METADATA_VALUE:
            result[key] = public_value
    if "source_mapping" in metadata:
        result["source_mapping_summary"] = _source_mapping_summary(
            metadata["source_mapping"]
        )
    return result


def _normalize_diagnosis_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt independently proven sewing history to the diagnosis API."""
    value = dict(mapping)
    if value.get("status") in {
        "exact_face_local_geometry",
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
        "exact_geometry_incidence",
    }:
        value["status"] = "exact_identity"
    return value


def _unavailable_diagnosis(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": str(reason),
        "faces": [],
        "wires": [],
        "occurrences": [],
        "occurrence_kinds": [],
    }


def _matching_count_capped(
    compatibility: Sequence[Sequence[int]], candidate_count: int, *, cap: int = 2
) -> tuple[int, list[int] | None]:
    """Count perfect bipartite assignments up to ``cap`` and retain the first.

    The left-side order is only an implementation key.  The search always
    branches on the currently most constrained row and therefore makes no
    correspondence claim from either source or STEP explorer ordinals.
    """
    count_right = int(candidate_count)
    if count_right < 0 or len(compatibility) != count_right or cap < 1:
        return 0, None
    graph = [sorted(set(int(value) for value in row)) for row in compatibility]
    if any(
        not row or any(value < 0 or value >= count_right for value in row)
        for row in graph
    ):
        return 0, None
    assignment = [-1] * len(graph)
    solution_count = 0
    first: list[int] | None = None

    def search(remaining: tuple[int, ...], used: set[int]) -> None:
        nonlocal solution_count, first
        if solution_count >= cap:
            return
        if not remaining:
            solution_count += 1
            if first is None:
                first = list(assignment)
            return
        row_index = min(
            remaining,
            key=lambda index: sum(candidate not in used for candidate in graph[index]),
        )
        options = [candidate for candidate in graph[row_index] if candidate not in used]
        for candidate in options:
            assignment[row_index] = candidate
            search(
                tuple(index for index in remaining if index != row_index),
                {*used, candidate},
            )
            assignment[row_index] = -1

    search(tuple(range(len(graph))), set())
    return solution_count, first


def _cyclic_curve_sample_distance(first: Any, second: Any, *, closed: bool) -> float:
    """Return an orientation-invariant max 3-D sample distance.

    Closed curves additionally admit cyclic phase shifts.  A repeated terminal
    sample is removed before shifting so it cannot privilege one seam choice.
    Non-finite or differently shaped samples fail closed with infinity.
    """
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 2
        or left.shape[1:] != (3,)
        or left.shape != right.shape
        or len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        return float("inf")
    if not closed:
        return min(
            float(np.max(np.linalg.norm(left - right, axis=1))),
            float(np.max(np.linalg.norm(left - right[::-1], axis=1))),
        )
    if len(left) > 2:
        left_repeated = float(np.linalg.norm(left[0] - left[-1])) <= 1e-9
        right_repeated = float(np.linalg.norm(right[0] - right[-1])) <= 1e-9
        if left_repeated != right_repeated:
            return float("inf")
        if left_repeated:
            left = left[:-1]
            right = right[:-1]
    errors: list[float] = []
    for candidate in (right, right[::-1]):
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, shift, axis=0)
            errors.append(
                float(np.max(np.linalg.norm(left - shifted, axis=1)))
            )
    return min(errors, default=float("inf"))


def _edge_fingerprint_metrics(
    first: Mapping[str, Any], second: Mapping[str, Any], *, scale: float
) -> dict[str, Any]:
    """Compare two JSON-like edge fingerprints in normalized WCS units."""
    shape_scale = float(scale)
    if not math.isfinite(shape_scale) or shape_scale <= 0:
        raise ValueError("fingerprint scale must be finite and positive")
    required = ("length", "bbox", "endpoints", "samples", "closed", "degenerated")
    if any(key not in first or key not in second for key in required):
        return {"available": False, "reason": "fingerprint_field_missing"}
    try:
        left_bbox = np.asarray(first["bbox"], dtype=np.float64)
        right_bbox = np.asarray(second["bbox"], dtype=np.float64)
        left_endpoints = np.asarray(first["endpoints"], dtype=np.float64)
        right_endpoints = np.asarray(second["endpoints"], dtype=np.float64)
        left_length = float(first["length"])
        right_length = float(second["length"])
    except (TypeError, ValueError):
        return {"available": False, "reason": "fingerprint_value_invalid"}
    arrays = (left_bbox, right_bbox, left_endpoints, right_endpoints)
    if (
        left_bbox.shape != (6,)
        or right_bbox.shape != (6,)
        or left_endpoints.shape != (2, 3)
        or right_endpoints.shape != (2, 3)
        or not all(np.isfinite(value).all() for value in arrays)
        or not math.isfinite(left_length)
        or not math.isfinite(right_length)
    ):
        return {"available": False, "reason": "fingerprint_nonfinite_or_malformed"}
    direct = max(
        float(np.linalg.norm(left_endpoints[0] - right_endpoints[0])),
        float(np.linalg.norm(left_endpoints[1] - right_endpoints[1])),
    )
    reverse = max(
        float(np.linalg.norm(left_endpoints[0] - right_endpoints[1])),
        float(np.linalg.norm(left_endpoints[1] - right_endpoints[0])),
    )
    closed = bool(first["closed"] and second["closed"])
    sample_distance = _cyclic_curve_sample_distance(
        first["samples"], second["samples"], closed=closed
    )
    normalized = {
        "length": abs(left_length - right_length) / shape_scale,
        "bbox": float(np.max(np.abs(left_bbox - right_bbox))) / shape_scale,
        "endpoints": min(direct, reverse) / shape_scale,
        "samples": sample_distance / shape_scale,
    }
    if not all(math.isfinite(value) for value in normalized.values()):
        return {"available": False, "reason": "fingerprint_metric_nonfinite"}
    return {
        "available": True,
        **normalized,
        # Curve type is evidence only. STEP may wrap the same geometry in a
        # different adaptor type, so it is deliberately not a hard gate.
        "curve_type_equal": first.get("curve_type") == second.get("curve_type"),
        "closed_equal": bool(first["closed"]) == bool(second["closed"]),
        "degenerated_equal": bool(first["degenerated"])
        == bool(second["degenerated"]),
        "seam_equal": first.get("seam") == second.get("seam"),
    }


def _edge_fingerprints_compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    scale: float,
    tolerance: float = STEP_EDGE_TOLERANCE_NORMALIZED,
) -> bool:
    metrics = _edge_fingerprint_metrics(first, second, scale=scale)
    tol = float(tolerance)
    return bool(
        metrics.get("available")
        and math.isfinite(tol)
        and tol >= 0
        and metrics["closed_equal"]
        and metrics["degenerated_equal"]
        and metrics["seam_equal"]
        and max(
            metrics["length"],
            metrics["bbox"],
            metrics["endpoints"],
            metrics["samples"],
        )
        <= tol
    )


def _occ_shape_bbox(shape: Any) -> np.ndarray:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib.AddOptimal(shape, box, False, False)
    if box.IsVoid():
        raise RuntimeError("occ_shape_bbox_void")
    value = np.asarray(box.Get(), dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise RuntimeError("occ_shape_bbox_nonfinite")
    return value


def _occ_edge_fingerprint(
    edge: Any, *, face: Any | None = None, sample_count: int = STEP_CURVE_SAMPLE_COUNT
) -> dict[str, Any]:
    """Measure an OCC edge without retaining parameterization direction."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    from OCC.Core.GProp import GProp_GProps

    adapter = BRepAdaptor_Curve(edge)
    first = float(adapter.FirstParameter())
    last = float(adapter.LastParameter())
    if not (math.isfinite(first) and math.isfinite(last) and first < last):
        raise RuntimeError("occ_curve_parameter_interval_invalid")
    sampler = GCPnts_UniformAbscissa(
        adapter, int(sample_count), first, last, 1e-10
    )
    if sampler.IsDone() and int(sampler.NbPoints()) == int(sample_count):
        parameters = [
            float(sampler.Parameter(index)) for index in range(1, sample_count + 1)
        ]
        sample_mode = "uniform_abscissa"
    else:
        parameters = np.linspace(first, last, int(sample_count)).tolist()
        sample_mode = "uniform_parameter_fallback"
    samples = np.asarray(
        [
            (
                float(adapter.Value(parameter).X()),
                float(adapter.Value(parameter).Y()),
                float(adapter.Value(parameter).Z()),
            )
            for parameter in parameters
        ],
        dtype=np.float64,
    )
    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)
    seam = None
    if face is not None:
        seam = bool(BRep_Tool.IsClosed(edge, face))
    fingerprint = {
        "curve_type": int(adapter.GetType()),
        "length": float(properties.Mass()),
        "bbox": _occ_shape_bbox(edge),
        "endpoints": np.asarray((samples[0], samples[-1]), dtype=np.float64),
        "samples": samples,
        "sample_mode": sample_mode,
        "closed": bool(adapter.IsClosed()),
        "degenerated": bool(BRep_Tool.Degenerated(edge)),
        "seam": seam,
    }
    if not _edge_fingerprint_metrics(fingerprint, fingerprint, scale=1.0).get(
        "available"
    ):
        raise RuntimeError("occ_edge_fingerprint_invalid")
    return fingerprint


def _occ_topology_list(shape: Any, kind: Any) -> list[Any]:
    from OCC.Core.TopExp import TopExp_Explorer

    values = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        values.append(explorer.Current())
        explorer.Next()
    return values


def _fixed_diagnosis_edges(wire: Any, face: Any) -> list[Any]:
    """Return the exact post-ShapeFix edge handles used by wire diagnosis."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire

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
    data = analysis.WireData()
    return [data.Edge(position) for position in range(1, int(analysis.NbEdges()) + 1)]


def _occ_step_face_signature(face: Any) -> dict[str, Any]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_WIRE
    from OCC.Core.TopoDS import topods_Wire

    surface = BRepAdaptor_Surface(face)
    outer = breptools.OuterWire(face)
    wires = []
    for wire_shape in _occ_topology_list(face, TopAbs_WIRE):
        wire = topods_Wire(wire_shape)
        edges = _fixed_diagnosis_edges(wire, face)
        wires.append(
            {
                "observed_wire": wire,
                "outer": bool(not outer.IsNull() and wire.IsSame(outer)),
                "edges": [
                    {"observed_edge": edge, "fingerprint": _occ_edge_fingerprint(edge, face=face)}
                    for edge in edges
                ],
            }
        )
    return {
        "face": face,
        "surface_type": int(surface.GetType()),
        "surface_u_periodic": bool(surface.IsUPeriodic()),
        "surface_v_periodic": bool(surface.IsVPeriodic()),
        "wire_pattern": sorted((row["outer"], len(row["edges"])) for row in wires),
        "wires": wires,
    }


def _source_face_signature_from_observer(
    source_face_index: int,
    face: Any,
    source_mapping: Mapping[str, Any],
    *,
    expected_edge_ids: Sequence[int],
) -> dict[str, Any]:
    """Turn private post-sewing proof handles into a source face signature."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools

    status = source_mapping.get("status")
    failures = source_mapping.get("failures") or []
    if status not in {
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
        "exact_face_local_geometry",
    } or failures:
        raise RuntimeError(f"source_face_{source_face_index}_mapping_not_exact")
    wire_rows = source_mapping.get("wire_rows")
    if not isinstance(wire_rows, Sequence) or isinstance(wire_rows, (str, bytes)):
        raise RuntimeError(f"source_face_{source_face_index}_wire_rows_missing")
    outer = breptools.OuterWire(face)
    wires = []
    observed_ids: list[int] = []
    for row in wire_rows:
        if not isinstance(row, Mapping) or row.get("observed_wire") is None:
            raise RuntimeError(f"source_face_{source_face_index}_wire_row_invalid")
        wire = row["observed_wire"]
        candidates = row.get("source_edge_candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise RuntimeError(f"source_face_{source_face_index}_edge_candidates_missing")
        edges = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"source_face_{source_face_index}_edge_candidate_invalid")
            source_edge_id = candidate.get("source_edge_id")
            observed_edge = candidate.get("observed_edge")
            if type(source_edge_id) is not int or observed_edge is None:
                raise RuntimeError(f"source_face_{source_face_index}_edge_candidate_invalid")
            observed_ids.append(source_edge_id)
            edges.append(
                {
                    "source_edge_id": source_edge_id,
                    "observed_edge": observed_edge,
                    "fingerprint": _occ_edge_fingerprint(observed_edge, face=face),
                }
            )
        wires.append(
            {
                "observed_wire": wire,
                "outer": bool(not outer.IsNull() and wire.IsSame(outer)),
                "edges": edges,
            }
        )
    if Counter(observed_ids) != Counter(int(value) for value in expected_edge_ids):
        raise RuntimeError(f"source_face_{source_face_index}_edge_multiset_mismatch")
    surface = BRepAdaptor_Surface(face)
    return {
        "source_face_index": int(source_face_index),
        "face": face,
        "surface_type": int(surface.GetType()),
        "surface_u_periodic": bool(surface.IsUPeriodic()),
        "surface_v_periodic": bool(surface.IsVPeriodic()),
        "wire_pattern": sorted((row["outer"], len(row["edges"])) for row in wires),
        "wires": wires,
    }


def _flat_face_edges(face_signature: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [edge for wire in face_signature["wires"] for edge in wire["edges"]]


def _edge_assignment_graph(
    source_face: Mapping[str, Any],
    step_face: Mapping[str, Any],
    *,
    scale: float,
    tolerance: float,
) -> list[list[int]]:
    source_edges = _flat_face_edges(source_face)
    step_edges = _flat_face_edges(step_face)
    if len(source_edges) != len(step_edges):
        return []
    return [
        [
            step_index
            for step_index, step_edge in enumerate(step_edges)
            if _edge_fingerprints_compatible(
                source_edge["fingerprint"],
                step_edge["fingerprint"],
                scale=scale,
                tolerance=tolerance,
            )
        ]
        for source_edge in source_edges
    ]


def _face_pair_compatible(
    source_face: Mapping[str, Any],
    step_face: Mapping[str, Any],
    *,
    scale: float,
    tolerance: float,
) -> bool:
    if source_face["wire_pattern"] != step_face["wire_pattern"]:
        return False
    if source_face["surface_type"] != step_face["surface_type"]:
        return False
    if (
        source_face["surface_u_periodic"],
        source_face["surface_v_periodic"],
    ) != (step_face["surface_u_periodic"], step_face["surface_v_periodic"]):
        return False
    graph = _edge_assignment_graph(
        source_face, step_face, scale=scale, tolerance=tolerance
    )
    count, _assignment = _matching_count_capped(
        graph, len(_flat_face_edges(step_face))
    )
    return count > 0


def _same_occ_shape(first: Any, second: Any) -> bool:
    try:
        return bool(first.IsSame(second))
    except Exception as exc:
        raise RuntimeError("occ_identity_measurement_failed") from exc


def _validate_global_edge_incidence(
    occurrences: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject a STEP edge split across faces or merged across source ids."""
    by_source: dict[int, list[Any]] = defaultdict(list)
    for occurrence in occurrences:
        source_edge_id = occurrence.get("source_edge_id")
        step_edge = occurrence.get("observed_edge")
        if type(source_edge_id) is not int or step_edge is None:
            return ["global_edge_occurrence_invalid"]
        by_source[source_edge_id].append(step_edge)
    failures: list[str] = []
    representatives: dict[int, Any] = {}
    for source_edge_id, edges in sorted(by_source.items()):
        representatives[source_edge_id] = edges[0]
        if any(not _same_occ_shape(edges[0], edge) for edge in edges[1:]):
            failures.append(f"source_edge_{source_edge_id}_split_after_step")
    source_ids = sorted(representatives)
    for offset, source_edge_id in enumerate(source_ids):
        for other_id in source_ids[offset + 1 :]:
            if _same_occ_shape(
                representatives[source_edge_id], representatives[other_id]
            ):
                failures.append(
                    f"source_edges_{source_edge_id}_{other_id}_merged_after_step"
                )
    return failures


def _match_step_geometry_incidence(
    source_faces: Sequence[Mapping[str, Any]],
    step_faces: Sequence[Mapping[str, Any]],
    *,
    scale: float,
    tolerance: float = STEP_EDGE_TOLERANCE_NORMALIZED,
) -> dict[str, Any]:
    """Prove unique face, edge-occurrence, and global-edge STEP lineage."""
    if len(source_faces) != len(step_faces) or not source_faces:
        return {
            "status": "unavailable",
            "failure_codes": ["source_step_face_count_mismatch"],
            "face_rows": [],
        }
    face_graph = [
        [
            step_index
            for step_index, step_face in enumerate(step_faces)
            if _face_pair_compatible(
                source_face,
                step_face,
                scale=scale,
                tolerance=tolerance,
            )
        ]
        for source_face in source_faces
    ]
    face_count, face_assignment = _matching_count_capped(face_graph, len(step_faces))
    public = {
        "tolerance_normalized": float(tolerance),
        "face_candidate_degree_counts": dict(
            sorted(Counter(len(row) for row in face_graph).items())
        ),
        "face_matching_count_capped": int(face_count),
    }
    if face_count != 1 or face_assignment is None:
        reason = (
            "face_assignment_has_no_perfect_matching"
            if face_count == 0
            else "face_assignment_not_unique"
        )
        return {
            "status": "unavailable" if face_count == 0 else "ambiguous",
            "failure_codes": [reason],
            "face_rows": [],
            **public,
        }
    face_rows = []
    global_occurrences = []
    for source_position, step_position in enumerate(face_assignment):
        source_face = source_faces[source_position]
        step_face = step_faces[step_position]
        source_edges = _flat_face_edges(source_face)
        step_edges = _flat_face_edges(step_face)
        graph = _edge_assignment_graph(
            source_face, step_face, scale=scale, tolerance=tolerance
        )
        edge_count, edge_assignment = _matching_count_capped(graph, len(step_edges))
        source_face_index = int(source_face["source_face_index"])
        if edge_count != 1 or edge_assignment is None:
            reason = (
                "edge_assignment_has_no_perfect_matching"
                if edge_count == 0
                else "edge_assignment_not_unique"
            )
            return {
                "status": "unavailable" if edge_count == 0 else "ambiguous",
                "failure_codes": [f"source_face_{source_face_index}_{reason}"],
                "face_rows": [],
                **public,
            }
        # Assignment is source occurrence -> flattened STEP occurrence.
        source_id_by_step_position = {
            int(step_edge_position): int(source_edges[source_position]["source_edge_id"])
            for source_position, step_edge_position in enumerate(edge_assignment)
        }
        wire_rows = []
        flat_position = 0
        for step_wire in step_face["wires"]:
            candidates = []
            for edge_row in step_wire["edges"]:
                source_edge_id = source_id_by_step_position[flat_position]
                candidates.append(
                    {
                        "source_edge_id": source_edge_id,
                        "observed_edge": edge_row["observed_edge"],
                    }
                )
                global_occurrences.append(
                    {
                        "source_edge_id": source_edge_id,
                        "observed_edge": edge_row["observed_edge"],
                    }
                )
                flat_position += 1
            wire_rows.append(
                {
                    "observed_wire": step_wire["observed_wire"],
                    "source_edge_candidates": candidates,
                }
            )
        face_rows.append(
            {
                "source_face_index": source_face_index,
                "step_face_internal": int(step_position),
                "edge_matching_count_capped": int(edge_count),
                "source_mapping": {
                    "status": "exact_geometry_incidence",
                    "failures": [],
                    "wire_rows": wire_rows,
                },
                "face": step_face["face"],
            }
        )
    incidence_failures = _validate_global_edge_incidence(global_occurrences)
    if incidence_failures:
        return {
            "status": "unavailable",
            "failure_codes": incidence_failures,
            "face_rows": [],
            **public,
        }
    return {
        "status": "exact_geometry_incidence",
        "failure_codes": [],
        "face_rows": face_rows,
        "mapped_face_count": len(face_rows),
        "mapped_edge_occurrence_count": len(global_occurrences),
        **public,
    }


def _read_step_faces(step_path: Path) -> tuple[Any, list[Any]]:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods_Face

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError("step_read_failed")
    reader.TransferRoots()
    shape = reader.OneShape()
    faces = [topods_Face(value) for value in _occ_topology_list(shape, TopAbs_FACE)]
    return shape, faces


def _step_observation(
    step_path: Path,
    *,
    breparg_root: Path,
    source_face_references: Mapping[int, Mapping[str, Any]] | None = None,
    face_edge_adj: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    try:
        from .diagnose_assembly_face_wires import (
            diagnose_face_wires_v2,
            diagnose_step_face_wires_v2,
        )
    except ImportError:  # direct script execution
        from diagnose_assembly_face_wires import (
            diagnose_face_wires_v2,
            diagnose_step_face_wires_v2,
        )

    diagnosis = diagnose_step_face_wires_v2(step_path, breparg_root=breparg_root)
    matching_proof: dict[str, Any] | None = None
    if source_face_references is not None and face_edge_adj is not None:
        try:
            step_shape, step_face_handles = _read_step_faces(step_path)
            source_faces = []
            for source_face_index in range(len(face_edge_adj)):
                reference = source_face_references.get(source_face_index)
                if not isinstance(reference, Mapping):
                    raise RuntimeError(
                        f"source_face_{source_face_index}_reference_missing"
                    )
                source_faces.append(
                    _source_face_signature_from_observer(
                        source_face_index,
                        reference["face"],
                        reference["source_mapping"],
                        expected_edge_ids=face_edge_adj[source_face_index],
                    )
                )
            step_faces = [_occ_step_face_signature(face) for face in step_face_handles]
            bounds = _occ_shape_bbox(step_shape)
            scale = max(float(np.linalg.norm(bounds[3:] - bounds[:3])), 1e-12)
            matching = _match_step_geometry_incidence(
                source_faces, step_faces, scale=scale
            )
            matching_proof = {
                key: value for key, value in matching.items() if key != "face_rows"
            }
            if matching["status"] == "exact_geometry_incidence":
                faces = []
                wires = []
                occurrences = []
                for face_row in sorted(
                    matching["face_rows"], key=lambda row: row["source_face_index"]
                ):
                    face_diagnosis = diagnose_face_wires_v2(
                        face_row["face"],
                        face_index=int(face_row["step_face_internal"]),
                        source_face_index=int(face_row["source_face_index"]),
                        source_mapping=_normalize_diagnosis_mapping(
                            face_row["source_mapping"]
                        ),
                    )
                    faces.extend(face_diagnosis["faces"])
                    wires.extend(face_diagnosis["wires"])
                    occurrences.extend(face_diagnosis["occurrences"])
                exact_diagnosis = {
                    "status": "diagnosed",
                    "edge_position_basis": "occ_1_based",
                    "faces": faces,
                    "wires": wires,
                    "occurrences": occurrences,
                    "occurrence_kinds": sorted(
                        {str(row.get("kind")) for row in occurrences}
                    ),
                    "geometry_incidence_proof": matching_proof,
                }
                return {
                    "phase": STEP_PHASE,
                    "entity_kind": "step_shape",
                    "lineage_status": "exact_geometry_incidence",
                    "mapping_failures": [],
                    "diagnosis": exact_diagnosis,
                }
            failure_codes = [str(value) for value in matching["failure_codes"]]
        except Exception as exc:
            failure_codes = [
                f"step_geometry_incidence_matching_failed:{type(exc).__name__}"
            ]
            matching_proof = {
                "status": "unavailable",
                "failure_codes": list(failure_codes),
            }
    else:
        failure_codes = ["step_source_reference_missing"]
    compact_occurrences = []
    for occurrence in diagnosis.get("occurrences") or []:
        compact_occurrences.append(
            {
                **dict(occurrence),
                "source_mapping_status": "unavailable",
                "source_mapping_reason": failure_codes[0],
            }
        )
    compact = {
        "status": diagnosis.get("status"),
        "edge_position_basis": diagnosis.get("edge_position_basis"),
        "faces": diagnosis.get("faces") or [],
        "wires": diagnosis.get("wires") or [],
        "occurrences": compact_occurrences,
        "occurrence_kinds": diagnosis.get("occurrence_kinds") or [],
    }
    if matching_proof is not None:
        compact["geometry_incidence_proof"] = matching_proof
    return {
        "phase": STEP_PHASE,
        "entity_kind": "step_shape",
        "lineage_status": "unavailable",
        "mapping_failures": failure_codes,
        "diagnosis": compact,
    }


def run_worker(
    source: Mapping[str, Any],
    *,
    breparg_root: Path,
    joint_iterations: int,
    expected_binding: Mapping[str, Any],
    run_signature: str,
) -> dict[str, Any]:
    try:
        from .assembly_repair import DIRECTED_LOCAL_TOPOLOGY_PROFILE
        from .diagnose_assembly_face_wires import diagnose_face_wires_v2
        from .directed_trim_assembly import construct_brep_directed
        from .run_assembly_calibration_oracle import cpu_joint_optimize
        from .run_assembly_repair_matrix import profile_kwargs
    except ImportError:  # direct script execution
        from assembly_repair import DIRECTED_LOCAL_TOPOLOGY_PROFILE
        from diagnose_assembly_face_wires import diagnose_face_wires_v2
        from directed_trim_assembly import construct_brep_directed
        from run_assembly_calibration_oracle import cpu_joint_optimize
        from run_assembly_repair_matrix import profile_kwargs

    started = time.perf_counter()
    source_path = Path(str(source["source_path"]))
    expected = normalize_binding(expected_binding)
    before = source_binding(source_path)
    if before != expected:
        raise RuntimeError("source_binding_mismatch_before_load")
    payload = source_path.read_bytes()
    loaded_binding = normalize_binding(payload_binding(payload))
    if loaded_binding != expected:
        raise RuntimeError("source_binding_mismatch_loaded_bytes")
    parsed = pickle.loads(payload)
    after_binding = source_binding(source_path)
    if after_binding != expected:
        raise RuntimeError("source_binding_mismatch_after_load")

    face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    source_face_count = len(face_edge_adj)
    source_edge_count = len(parsed["edge_ncs"])
    if source_face_count <= 0 or len(parsed["surf_ncs"]) != source_face_count:
        raise ValueError("source face adjacency and surface counts differ")
    surf_wcs, edge_wcs = cpu_joint_optimize(
        np.asarray(parsed["surf_ncs"], dtype=np.float32),
        np.asarray(parsed["edge_ncs"], dtype=np.float32),
        np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
        np.asarray(parsed["corner_unique"], dtype=np.float32),
        edge_vertex_adj,
        face_edge_adj,
        iterations=int(joint_iterations),
    )

    observations: list[dict[str, Any]] = []
    # Native handles are process-private proof material. They are retained only
    # until STEP lineage is established and are never copied into JSON rows.
    post_sewing_source_faces: dict[int, dict[str, Any]] = {}

    def observe(
        source_face_index: int, face: Any | None, metadata: Mapping[str, Any]
    ) -> None:
        phase = str(metadata.get("phase"))
        mapping = metadata.get("source_mapping")
        mapping = dict(mapping) if isinstance(mapping, Mapping) else {}
        lineage_status = str(mapping.get("status") or "unavailable")
        mapping_failures = [str(value) for value in mapping.get("failures") or []]
        sewing_lineage = metadata.get("sewing_lineage")
        if isinstance(sewing_lineage, Mapping) and sewing_lineage.get("status") != "mapped":
            mapping_failures.extend(
                str(value) for value in sewing_lineage.get("failure_codes") or []
            )
        if face is None:
            diagnosis = _unavailable_diagnosis("stage_face_unavailable")
        else:
            try:
                diagnosis = diagnose_face_wires_v2(
                    face,
                    face_index=int(source_face_index),
                    source_face_index=int(source_face_index),
                    source_mapping=_normalize_diagnosis_mapping(mapping),
                )
            except Exception as exc:
                diagnosis = _unavailable_diagnosis(
                    f"face_diagnosis_failed:{type(exc).__name__}"
                )
        if phase == "post_sewing_pre_step" and face is not None:
            post_sewing_source_faces[int(source_face_index)] = {
                "face": face,
                "source_mapping": mapping,
            }
        observations.append(
            {
                "phase": phase,
                "source_face_index": int(source_face_index),
                **_compact_metadata(metadata),
                "lineage_status": lineage_status,
                "mapping_failures": mapping_failures,
                "diagnosis": diagnosis,
            }
        )

    assembly_status = "completed"
    assembly_error_type = None
    solid = None
    try:
        solid, _diagnostics = construct_brep_directed(
            surf_wcs,
            edge_wcs,
            face_edge_adj,
            edge_vertex_adj,
            breparg_root=breparg_root,
            **profile_kwargs(DIRECTED_LOCAL_TOPOLOGY_PROFILE),
            assembly_stage_face_observer=observe,
        )
    except Exception as exc:
        assembly_status = "assembly_error"
        assembly_error_type = type(exc).__name__

    step_roundtrip_status = "not_attempted"
    if solid is not None:
        try:
            from OCC.Extend.DataExchange import write_step_file

            with tempfile.TemporaryDirectory(prefix="downstream-lineage-") as temp_root:
                step_path = Path(temp_root) / "candidate.step"
                write_step_file(solid, str(step_path))
                if not step_path.is_file() or step_path.stat().st_size <= 0:
                    raise RuntimeError("step_writer_produced_no_file")
                step_observation = _step_observation(
                    step_path,
                    breparg_root=breparg_root,
                    source_face_references=post_sewing_source_faces,
                    face_edge_adj=face_edge_adj,
                )
                observations.append(step_observation)
                step_roundtrip_status = str(
                    step_observation["diagnosis"].get("status")
                )
        except Exception as exc:
            step_roundtrip_status = "step_error"
            observations.append(
                {
                    "phase": STEP_PHASE,
                    "entity_kind": "step_shape",
                    "lineage_status": "unavailable",
                    "mapping_failures": ["step_roundtrip_failed"],
                    "diagnosis": _unavailable_diagnosis(
                        f"step_roundtrip_failed:{type(exc).__name__}"
                    ),
                }
            )

    assessment = assess_observations(
        observations,
        source_face_count=source_face_count,
        source_edge_count=source_edge_count,
    )
    complete = bool(
        assessment["all_stages_observed"]
        and assessment["observation_failure_count"] == 0
        and assessment["mapping_failure_count"] == 0
        and assembly_status == "completed"
        and step_roundtrip_status == "diagnosed"
    )
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": PROFILE,
        "run_signature": run_signature,
        "source_binding": expected,
        "source_binding_loaded_bytes": loaded_binding,
        "source_binding_after_load": after_binding,
        "status": "completed" if complete else "measurement_incomplete",
        "assembly_status": assembly_status,
        "assembly_error_type": assembly_error_type,
        "step_roundtrip_status": step_roundtrip_status,
        "source_face_count": source_face_count,
        "source_edge_count": source_edge_count,
        "observations": observations,
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
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_isolated(
    source: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    run_signature: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    cad_id = str(source["cad_id"])
    log_dir = Path(args.output_dir) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--calibration-manifest",
        str(Path(args.calibration_manifest).resolve()),
        "--selector-matrix",
        str(Path(args.selector_matrix).resolve()),
        "--selector-run",
        str(Path(args.selector_run).resolve()),
        "--breparg-root",
        str(Path(args.breparg_root).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--joint-iterations",
        str(int(args.joint_iterations)),
        "--worker-timeout-seconds",
        str(float(args.worker_timeout_seconds)),
        "--worker-cad-id",
        cad_id,
        "--worker-run-signature",
        run_signature,
        "--worker-source-binding-json",
        json.dumps(dict(expected_binding), sort_keys=True, separators=(",", ":")),
        "--worker-source-path",
        str(Path(str(source["source_path"])).resolve()),
        "--worker-parent-id",
        str(source["parent_id"]),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(args.worker_timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        (log_dir / f"{cad_id}.stdout.log").write_text(stdout, encoding="utf-8")
        (log_dir / f"{cad_id}.stderr.log").write_text(stderr, encoding="utf-8")
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_timeout",
            returncode=None,
            error_type="TimeoutExpired",
            expected_binding=expected_binding,
        )
    except OSError as exc:
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_process_exit",
            returncode=None,
            error_type=type(exc).__name__,
            expected_binding=expected_binding,
        )
    (log_dir / f"{cad_id}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{cad_id}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    row = parse_lineage_worker_result(completed.stdout)
    if completed.returncode != 0:
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_process_exit",
            returncode=int(completed.returncode),
            error_type="NonzeroWorkerExit",
            expected_binding=expected_binding,
        )
    if row is None:
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_protocol_error",
            returncode=int(completed.returncode),
            error_type="InvalidWorkerSentinel",
            expected_binding=expected_binding,
        )
    try:
        validate_case_row(
            row,
            source=source,
            run_signature=run_signature,
            expected_binding=expected_binding,
        )
        if source_binding(Path(str(source["source_path"]))) != normalize_binding(
            expected_binding
        ):
            raise ValueError("source binding changed after worker completion")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_protocol_error",
            returncode=int(completed.returncode),
            error_type=type(exc).__name__,
            expected_binding=expected_binding,
        )
    row["worker_returncode"] = int(completed.returncode)
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("cad_id")) for row in rows]
    if len(rows) != 2 or ids != list(TARGET_CAD_IDS):
        raise ValueError("lineage summary requires the ordered two target rows")
    worker_failures = sum(str(row.get("status", "")).startswith("worker_") for row in rows)
    binding_failures = sum(row.get("status") == "source_binding_mismatch" for row in rows)
    coverage_failures = sum(int(row.get("coverage_failure_count") or 0) for row in rows)
    observation_failures = sum(int(row.get("observation_failure_count") or 0) for row in rows)
    mapping_failures = sum(int(row.get("mapping_failure_count") or 0) for row in rows)
    completed = sum(row.get("status") == "completed" for row in rows)
    explicit_failures = sum(row.get("status") in FAILURE_STATUSES for row in rows)
    conclusive = bool(
        completed == 2
        and explicit_failures == 0
        and worker_failures == 0
        and binding_failures == 0
        and coverage_failures == 0
        and observation_failures == 0
        and mapping_failures == 0
    )
    targets = [
        {
            "cad_id": str(row["cad_id"]),
            "first_bad_phase": row.get("first_bad_phase"),
            "first_bad_occurrences": row.get("first_bad_occurrences") or [],
        }
        for row in rows
    ]
    first_phases = {
        str(item["first_bad_phase"])
        for item in targets
        if item["first_bad_phase"] is not None
    }
    if not conclusive:
        decision = "INCONCLUSIVE_REQUIRES_RERUN"
    elif first_phases - {MEMORY_PHASES[0]}:
        decision = "PROMOTE_TARGETED_NONPERIODIC_REPAIR_PROBE"
    else:
        # A defect already present at the first observable phase was not
        # introduced downstream.  This probe cannot authorize a downstream
        # mutation for it, even though its source mapping may be exact.
        decision = "CLOSE_DOWNSTREAM_BAD_WIRE_ROUTE"
    result = {
        "schema": SUMMARY_SCHEMA,
        "cases": 2,
        "completed_cases": completed,
        "worker_or_protocol_failures": worker_failures,
        "source_binding_failures": binding_failures,
        "coverage_failures": coverage_failures,
        "observation_failures": observation_failures,
        "mapping_failures": mapping_failures,
        "explicit_failures": explicit_failures,
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "targets": targets,
        "conclusive": conclusive,
        "decision": decision,
        "assembly_release_gate_before": {"strict_valid": 91, "required": 95},
        "authorizes_full_100cad": False,
        "authorizes_boundary_or_ar": False,
    }
    assert_path_free_evidence(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--selector-matrix", type=Path, required=True)
    parser.add_argument("--selector-run", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-cad-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-signature", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-binding-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-parent-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.joint_iterations < 0:
        parser.error("--joint-iterations must be non-negative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibration_rows = read_jsonl(args.calibration_manifest)
    selector_rows = read_jsonl(args.selector_matrix)
    sources = select_lineage_sources(calibration_rows, selector_rows)
    if args.worker_cad_id is not None:
        if (
            not args.worker_run_signature
            or not args.worker_source_binding_json
            or args.worker_source_path is None
            or not args.worker_parent_id
        ):
            raise SystemExit(
                "worker mode requires run signature, source binding, source path, and parent id"
            )
        matches = [row for row in sources if str(row["cad_id"]) == args.worker_cad_id]
        if len(matches) != 1:
            raise SystemExit("worker CAD id is not unique in target cohort")
        worker_source = dict(matches[0])
        if (
            Path(str(worker_source["source_path"])).resolve()
            != args.worker_source_path.resolve()
            or worker_source.get("parent_id") != args.worker_parent_id
        ):
            raise SystemExit("worker source identity mismatches signed parent arguments")
        try:
            expected = normalize_binding(json.loads(args.worker_source_binding_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"worker source binding argument is invalid: {type(exc).__name__}"
            ) from exc
        try:
            row = run_worker(
                worker_source,
                breparg_root=args.breparg_root,
                joint_iterations=args.joint_iterations,
                expected_binding=expected,
                run_signature=args.worker_run_signature,
            )
        except Exception as exc:
            row = worker_failure_row(
                worker_source,
                run_signature=args.worker_run_signature,
                status="worker_error",
                returncode=0,
                error_type=type(exc).__name__,
                expected_binding=expected,
            )
        print(
            WORKER_MARKER
            + json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False),
            flush=True,
        )
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The shared lock helper intentionally uses only this directory and lock
    # file; the lineage-specific name prevents cross-protocol writers.
    with output_writer_lock(output_dir):
        payload = build_run_payload(args, sources, selector_rows)
        signature = canonical_sha256(payload)
        run_path = output_dir / RUN_MANIFEST_NAME
        if run_path.is_file():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if (
                existing.get("schema") != RUN_SCHEMA
                or existing.get("signature") != signature
                or existing.get("payload") != payload
            ):
                raise RuntimeError("output directory belongs to a different signed lineage run")
        else:
            unexpected = [
                path
                for path in output_dir.iterdir()
                if path.name not in {RUN_MANIFEST_NAME, LOCK_NAME}
            ]
            if unexpected:
                raise RuntimeError("unsigned lineage output directory is not empty")
            atomic_json(
                run_path,
                {
                    "schema": RUN_SCHEMA,
                    "signature": signature,
                    "payload": payload,
                    "status": "RUNNING",
                },
            )
        rows_path = output_dir / ROWS_NAME
        rows = read_jsonl(rows_path)
        done = {str(row.get("cad_id")) for row in rows}
        if len(done) != len(rows) or not done.issubset(set(TARGET_CAD_IDS)):
            raise RuntimeError("existing lineage rows do not match the signed cohort")
        bindings = {str(item["cad_id"]): item for item in payload["source_bindings"]}
        sources_by_id = {str(source["cad_id"]): source for source in sources}
        for row in rows:
            cad_id = str(row["cad_id"])
            validate_case_row(
                row,
                source=sources_by_id[cad_id],
                run_signature=signature,
                expected_binding={key: bindings[cad_id][key] for key in ("bytes", "sha256")},
            )
        validate_bound_inputs(
            args, payload=payload, sources=sources, selector_rows=selector_rows
        )
        for source in sources:
            cad_id = str(source["cad_id"])
            if cad_id in done:
                continue
            row = run_isolated(
                source,
                args=args,
                run_signature=signature,
                expected_binding={key: bindings[cad_id][key] for key in ("bytes", "sha256")},
            )
            append_jsonl(rows_path, row)
            rows.append(row)
            done.add(cad_id)
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "cad_id",
                            "status",
                            "first_bad_phase",
                            "mapping_failure_count",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        by_id = {str(row["cad_id"]): row for row in rows}
        ordered = [by_id[cad_id] for cad_id in TARGET_CAD_IDS]
        validate_bound_inputs(
            args, payload=payload, sources=sources, selector_rows=selector_rows
        )
        for row in ordered:
            cad_id = str(row["cad_id"])
            validate_case_row(
                row,
                source=sources_by_id[cad_id],
                run_signature=signature,
                expected_binding={key: bindings[cad_id][key] for key in ("bytes", "sha256")},
            )
        summary = summarize(ordered)
        atomic_json(output_dir / SUMMARY_NAME, summary)
        record = json.loads(run_path.read_text(encoding="utf-8"))
        record.update(
            status="COMPLETED" if summary["conclusive"] else "INCONCLUSIVE",
            attempts=2,
            rows_sha256=sha256_file(rows_path),
            summary_sha256=sha256_file(output_dir / SUMMARY_NAME),
        )
        atomic_json(run_path, record)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["conclusive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
