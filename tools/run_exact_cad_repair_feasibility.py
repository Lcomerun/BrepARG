"""Run two exact-CAD repair probes as an isolated control/candidate matrix.

This is a feasibility coordinator, not a production assembly profile. It
binds the frozen 100-CAD calibration/selector evidence and signed lineage
evidence, selects exactly two registered residual CADs, and runs exactly four
attempts: one control and one candidate for each CAD. Every attempt runs in a
fresh child process. A timeout, native crash, malformed sentinel, source-byte
drift, missing candidate callback, or incomplete proof remains an explicit
failed denominator row.

Candidate callback contract
---------------------------
A callback named by :class:`VariantSpec.callback_ref` is invoked with keyword
arguments ``source``, ``source_bytes``, ``parsed``, ``output_dir``,
``breparg_root``, ``joint_iterations``, ``variant``,
``expected_source_binding``, and ``run_signature``. It must return a mapping
that identifies a non-empty STEP using ``step_path`` or
``step_relative_path`` and provides ``candidate_application`` and
``defect_gate`` proof objects. The worker independently re-reads the STEP,
computes native and project-strict validity, and applies the unchanged
``assembly-selector-geometry-gate-v2``. Missing or incomplete evidence fails
closed; a face-local helper cannot masquerade as a whole-CAD success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import DIRECTED_LOCAL_TOPOLOGY_PROFILE
    from .assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        candidate_step_signature,
        geometry_gate_thresholds,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
        validate_accepted_geometry_gate,
    )
    from .probe_downstream_bad_wire_lineage import (
        _step_observation,
        _validate_selector_run,
        git_identity,
        select_lineage_sources,
    )
    from .probe_periodic_pcurve_applicability import (
        atomic_json,
        canonical_sha256,
        normalize_binding,
        sha256_file,
        source_binding,
    )
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .run_assembly_repair_matrix import (
        profile_kwargs,
        run_one,
        source_pickle_binding_from_bytes,
        strict_validate_step,
    )
    from .directed_trim_assembly import construct_brep_directed
except ImportError:  # pragma: no cover - direct script execution
    from assembly_repair import DIRECTED_LOCAL_TOPOLOGY_PROFILE
    from assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        candidate_step_signature,
        geometry_gate_thresholds,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
        validate_accepted_geometry_gate,
    )
    from probe_downstream_bad_wire_lineage import (
        _step_observation,
        _validate_selector_run,
        git_identity,
        select_lineage_sources,
    )
    from probe_periodic_pcurve_applicability import (
        atomic_json,
        canonical_sha256,
        normalize_binding,
        sha256_file,
        source_binding,
    )
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from run_assembly_repair_matrix import (
        profile_kwargs,
        run_one,
        source_pickle_binding_from_bytes,
        strict_validate_step,
    )
    from directed_trim_assembly import construct_brep_directed


SCHEMA = "exact-cad-repair-feasibility-attempt-v1"
RUN_SCHEMA = "exact-cad-repair-feasibility-run-v1"
SUMMARY_SCHEMA = "exact-cad-repair-feasibility-summary-v1"
LINEAGE_SCHEMA = "downstream-bad-wire-lineage-case-v1"
LINEAGE_RUN_SCHEMA = "downstream-bad-wire-lineage-run-v1"
RUN_NAME = "exact_cad_repair_feasibility_run.json"
ROWS_NAME = "exact_cad_repair_feasibility_attempts.jsonl"
SUMMARY_NAME = "exact_cad_repair_feasibility_summary.json"
LOCK_NAME = ".exact_cad_repair_feasibility_writer.lock"
WORKER_MARKER = "__EXACT_CAD_REPAIR_FEASIBILITY_RESULT__="

CAD_47472 = "00047472_197769bbdd814278b715d88a_step_000"
CAD_63055 = "00063055_e309c689b9b44f0686f47966_step_000"
TARGET_CAD_IDS = (CAD_47472, CAD_63055)

DEFECT_BOOLEAN_FIELDS = (
    "accepted",
    "target_defects_removed",
    "no_new_non_target_defects",
    "mapping_exact",
    "source_topology_preserved",
    "shared_edge_correspondence_preserved",
    "curves_3d_preserved",
    "source_binding_preserved",
)
WORKER_FAILURE_STATUSES = frozenset(
    {
        "worker_timeout",
        "worker_process_exit",
        "worker_spawn_error",
        "worker_protocol_error",
        "worker_error",
        "source_binding_mismatch",
    }
)
ATTEMPT_STATUSES = WORKER_FAILURE_STATUSES | frozenset(
    {
        "candidate_hook_missing",
        "candidate_accepted",
        "candidate_rejected",
        "control_reproduced",
        "control_drift",
    }
)


@dataclass(frozen=True)
class VariantSpec:
    """One immutable cell of the registered exact-CAD 2x2 matrix."""

    task_id: str
    cad_id: str
    arm: str
    callback_ref: str
    control_sewing_tolerance: float
    candidate_sewing_tolerance: float | None
    target_source_face_indices: tuple[int, ...]
    target_source_edge_pairs: tuple[tuple[int, int], ...]
    expected_control_native_valid: bool
    expected_control_strict_valid: bool
    expected_control_wire_self_intersections: int

    @property
    def is_candidate(self) -> bool:
        return self.arm == "candidate"


def _variant(
    *, cad_id: str, arm: str, callback_ref: str,
    control_sewing_tolerance: float,
    candidate_sewing_tolerance: float | None,
    faces: Sequence[int], pairs: Sequence[Sequence[int]],
    control_native: bool, control_strict: bool, control_bad_wires: int,
) -> VariantSpec:
    if arm not in {"control", "candidate"}:
        raise ValueError("variant arm must be control or candidate")
    return VariantSpec(
        task_id=f"{cad_id}::{arm}", cad_id=cad_id, arm=arm,
        callback_ref=callback_ref,
        control_sewing_tolerance=float(control_sewing_tolerance),
        candidate_sewing_tolerance=(None if candidate_sewing_tolerance is None
                                    else float(candidate_sewing_tolerance)),
        target_source_face_indices=tuple(map(int, faces)),
        target_source_edge_pairs=tuple(
            tuple(sorted((int(pair[0]), int(pair[1])))) for pair in pairs
        ),
        expected_control_native_valid=bool(control_native),
        expected_control_strict_valid=bool(control_strict),
        expected_control_wire_self_intersections=int(control_bad_wires),
    )


VARIANTS: tuple[VariantSpec, ...] = (
    _variant(
        cad_id=CAD_47472, arm="control",
        callback_ref="tools.run_exact_cad_repair_feasibility:run_control_variant",
        control_sewing_tolerance=1e-3, candidate_sewing_tolerance=None,
        faces=(10, 43), pairs=((13, 20), (16, 24)), control_native=True,
        control_strict=False, control_bad_wires=3,
    ),
    _variant(
        cad_id=CAD_47472, arm="candidate",
        callback_ref=("tools.run_exact_cad_repair_feasibility:"
                      "run_47472_candidate_variant"),
        control_sewing_tolerance=1e-3, candidate_sewing_tolerance=1e-3,
        faces=(10, 43), pairs=((13, 20), (16, 24)), control_native=True,
        control_strict=False, control_bad_wires=3,
    ),
    _variant(
        cad_id=CAD_63055, arm="control",
        callback_ref="tools.run_exact_cad_repair_feasibility:run_control_variant",
        control_sewing_tolerance=1e-3, candidate_sewing_tolerance=None,
        faces=(5,), pairs=((9, 23),), control_native=False,
        control_strict=False, control_bad_wires=1,
    ),
    _variant(
        cad_id=CAD_63055, arm="candidate",
        callback_ref=("tools.run_exact_cad_repair_feasibility:"
                      "run_63055_candidate_variant"),
        control_sewing_tolerance=1e-3, candidate_sewing_tolerance=1e-4,
        faces=(5,), pairs=((9, 23),), control_native=False,
        control_strict=False, control_bad_wires=1,
    ),
)
VARIANTS_BY_ID = {variant.task_id: variant for variant in VARIANTS}

EXPECTED_LINEAGE: dict[str, dict[str, Any]] = {
    CAD_47472: {
        "first_bad_phase": "post_add_pcurves_pre_repair",
        "occurrences": {
            ("adjacent", 10, (13, 20)), ("adjacent", 43, (16, 24)),
        },
    },
    CAD_63055: {
        "first_bad_phase": "post_sewing_pre_step",
        "occurrences": {
            ("closure", 5, (9, 23)), ("adjacent", 5, (9, 23)),
        },
    },
}


def _task_slug(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


@contextmanager
def output_writer_lock(output_dir: Path) -> Iterator[None]:
    """Prevent two coordinators from interleaving one output root."""
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
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("exact-CAD output already has an active writer") from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_worker_result(stdout: str) -> dict[str, Any] | None:
    """Accept exactly one sentinel, as the final nonempty stdout line."""
    lines = [line for line in str(stdout).splitlines() if line.strip()]
    indices = [i for i, line in enumerate(lines) if line.startswith(WORKER_MARKER)]
    if not lines or indices != [len(lines) - 1]:
        return None
    try:
        value = json.loads(lines[-1][len(WORKER_MARKER):])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _assert_json_safe(value: Any, *, label: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_json_safe(child, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_json_safe(child, label=f"{label}[{index}]")
    json.dumps(value, allow_nan=False, sort_keys=True)


def read_rows(path: Path, *, recover_truncated_tail: bool = False) -> list[dict[str, Any]]:
    """Read JSONL and optionally remove only an unterminated torn tail."""
    target = Path(path)
    if not target.is_file():
        return []
    payload = target.read_bytes()
    rows: list[dict[str, Any]] = []
    offset = 0
    lines = payload.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            offset += len(raw)
            continue
        try:
            value = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            torn = index == len(lines) - 1 and not raw.endswith((b"\n", b"\r"))
            if recover_truncated_tail and torn:
                with target.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                return rows
            raise
        if not isinstance(value, dict):
            raise ValueError("attempt JSONL must contain objects")
        rows.append(value)
        offset += len(raw)
    return rows


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    _assert_json_safe(row, label="attempt")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _occurrence_key(value: Mapping[str, Any]) -> tuple[str, int, tuple[int, int]]:
    kind, face, edge_ids = (value.get("kind"), value.get("source_face_index"),
                            value.get("source_edge_ids"))
    if (not isinstance(kind, str) or type(face) is not int
            or not isinstance(edge_ids, Sequence)
            or isinstance(edge_ids, (str, bytes)) or len(edge_ids) != 2
            or any(type(item) is not int or item < 0 for item in edge_ids)
            or edge_ids[0] == edge_ids[1]):
        raise ValueError("lineage occurrence is malformed")
    return kind, face, tuple(sorted(map(int, edge_ids)))


def validate_lineage_evidence(
    rows: Sequence[Mapping[str, Any]], *, sources: Sequence[Mapping[str, Any]],
    lineage_cases: Path, lineage_run: Path,
) -> dict[str, Any]:
    """Bind the exact two registered first-bad-stage observations."""
    if len(rows) != 2 or [row.get("cad_id") for row in rows] != list(TARGET_CAD_IDS):
        raise ValueError("lineage must contain the ordered two target CADs")
    sources_by_id = {str(source["cad_id"]): source for source in sources}
    bindings = []
    for row in rows:
        cad_id = str(row.get("cad_id"))
        if row.get("schema") != LINEAGE_SCHEMA or row.get("status") != "completed":
            raise ValueError(f"lineage case is not completed: {cad_id}")
        expected = EXPECTED_LINEAGE[cad_id]
        if row.get("first_bad_phase") != expected["first_bad_phase"]:
            raise ValueError(f"lineage first-bad phase drifted: {cad_id}")
        occurrences = row.get("first_bad_occurrences")
        if not isinstance(occurrences, Sequence) or isinstance(occurrences, (str, bytes)):
            raise ValueError(f"lineage occurrences missing: {cad_id}")
        observed = {_occurrence_key(item) for item in occurrences
                    if isinstance(item, Mapping)}
        if len(observed) != len(occurrences) or observed != expected["occurrences"]:
            raise ValueError(f"lineage occurrences drifted: {cad_id}")
        actual = normalize_binding(row.get("source_binding") or {})
        if actual != source_binding(Path(str(sources_by_id[cad_id]["source_path"]))):
            raise ValueError(f"lineage/source binding mismatch: {cad_id}")
        bindings.append({"cad_id": cad_id, **actual})
    run = json.loads(Path(lineage_run).read_text(encoding="utf-8"))
    payload = run.get("payload") if isinstance(run, Mapping) else None
    if (not isinstance(run, Mapping) or run.get("schema") != LINEAGE_RUN_SCHEMA
            or run.get("status") != "COMPLETED" or run.get("attempts") != 2
            or not isinstance(payload, Mapping)
            or canonical_sha256(payload) != run.get("signature")
            or run.get("rows_sha256") != sha256_file(lineage_cases)
            or payload.get("ordered_cad_ids") != list(TARGET_CAD_IDS)):
        raise ValueError("lineage run is not a signed completed two-CAD run")
    registered = payload.get("source_bindings")
    if not isinstance(registered, Sequence) or isinstance(registered, (str, bytes)):
        raise ValueError("lineage run lacks source bindings")
    normalized = []
    for item in registered:
        if not isinstance(item, Mapping):
            raise ValueError("lineage source binding is malformed")
        normalized.append({"cad_id": str(item.get("cad_id")), **normalize_binding(
            {key: item.get(key) for key in ("bytes", "sha256")})})
    if normalized != bindings:
        raise ValueError("lineage source inventory drifted")
    return {"cases_sha256": sha256_file(lineage_cases),
            "run_sha256": sha256_file(lineage_run),
            "run_signature": str(run["signature"]), "source_bindings": bindings}


def source_hashes(repo_root: Path) -> dict[str, str | None]:
    paths = (
        "tools/run_exact_cad_repair_feasibility.py",
        "tools/targeted_nonperiodic_pcurve_repair.py",
        "tools/post_sewing_graph_repair.py",
        "tools/run_assembly_repair_matrix.py",
        "tools/assembly_selector_geometry.py",
        "tools/probe_downstream_bad_wire_lineage.py",
        "tools/directed_trim_assembly.py",
    )
    return {rel: sha256_file(Path(repo_root) / rel)
            if (Path(repo_root) / rel).is_file() else None for rel in paths}


def build_run_payload(
    args: argparse.Namespace, *, sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]], lineage_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the path-free signed contract shared by parent and children."""
    repo_root = Path(__file__).resolve().parents[1]
    runtime_utils = Path(args.breparg_root).resolve() / "utils.py"
    if not runtime_utils.is_file():
        raise FileNotFoundError(runtime_utils)
    bindings = [{"cad_id": str(source["cad_id"]),
                 **source_binding(Path(str(source["source_path"]))) }
                for source in sources]
    selector_run = _validate_selector_run(
        args.selector_run, calibration_manifest=args.calibration_manifest,
        selector_matrix=args.selector_matrix, source_bindings=bindings)
    repository = {**git_identity(repo_root), "source_sha256": source_hashes(repo_root)}
    if repository["dirty"]:
        raise RuntimeError("formal exact-CAD run requires a clean Git worktree")
    selector_by_id = {str(row["cad_id"]): row for row in selector_rows}
    return {
        "schema": RUN_SCHEMA,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "selector_matrix_sha256": sha256_file(args.selector_matrix),
        "selector_run": selector_run, "lineage": dict(lineage_binding),
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "ordered_task_ids": [variant.task_id for variant in VARIANTS],
        # Round-trip through JSON here so tuple-valued dataclass fields do not
        # become lists only after the manifest is written. Exact resume then
        # compares the same in-memory shape that was signed on disk.
        "variants": json.loads(json.dumps([asdict(variant) for variant in VARIANTS])),
        "sources": [{
            "cad_id": str(source["cad_id"]), "parent_id": str(source["parent_id"]),
            "historical_strict_valid": bool(source.get("brep_valid")),
            "selector_strict_valid": bool(selector_by_id[str(source["cad_id"])]["strict_brep_valid"]),
            "binding": {key: binding[key] for key in ("bytes", "sha256")},
        } for source, binding in zip(sources, bindings)],
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "repository": repository,
        "breparg_runtime": {"utils_sha256": sha256_file(runtime_utils)},
    }


def bind_run_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create or validate an immutable run identity for safe resume."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path, signature = root / RUN_NAME, canonical_sha256(payload)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(current, Mapping) or current.get("schema") != RUN_SCHEMA
                or current.get("signature") != signature
                or current.get("payload") != dict(payload)):
            raise RuntimeError("output directory belongs to a different signed run")
        return dict(current)
    unexpected = [item for item in root.iterdir()
                  if item.name not in {LOCK_NAME, RUN_NAME}]
    if unexpected:
        raise RuntimeError("unsigned exact-CAD output directory is not empty")
    record = {"schema": RUN_SCHEMA, "signature": signature,
              "payload": dict(payload), "status": "RUNNING"}
    atomic_json(path, record)
    return record


def _rejected_geometry_gate(reason: str) -> dict[str, Any]:
    return {"schema": GEOMETRY_GATE_SCHEMA, "accepted": False,
            "checks": {"measurement_completed": False},
            "rejection_reasons": [str(reason)],
            "thresholds": geometry_gate_thresholds()}


def _base_row(source: Mapping[str, Any], variant: VariantSpec, *,
              run_signature: str, expected_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "task_id": variant.task_id, "cad_id": variant.cad_id,
        "parent_id": source.get("parent_id"), "arm": variant.arm,
        "callback_ref": variant.callback_ref, "run_signature": run_signature,
        "denominator": True,
        "historical_strict_valid": bool(source.get("brep_valid")),
        "status": "running", "callback_completed": False, "step_saved": False,
        "step_readable": False, "native_brep_valid": False,
        "strict_brep_valid": False, "both_valid": False,
        "source_binding_expected": dict(expected_binding),
        "source_binding_before": None, "source_binding_loaded_bytes": None,
        "source_binding_after_load": None, "source_binding_after_attempt": None,
        "candidate_application": {"attempted": False, "applied": False,
                                  "status": "not_started"},
        "defect_gate": {**{name: False for name in DEFECT_BOOLEAN_FIELDS},
                        "nonfinite_count": 0,
                        "rejection_reasons": ["attempt_not_completed"]},
        "geometry_topology_gate": _rejected_geometry_gate(
            "geometry_gate:attempt_not_completed"),
        "control_expectation": None,
    }


def failure_row(source: Mapping[str, Any], variant: VariantSpec, *,
                run_signature: str, expected_binding: Mapping[str, Any],
                status: str, error_type: str,
                returncode: int | None = None) -> dict[str, Any]:
    row = _base_row(source, variant, run_signature=run_signature,
                    expected_binding=expected_binding)
    row.update(status=str(status), error_type=str(error_type),
               worker_returncode=returncode)
    row["candidate_application"]["status"] = str(status)
    row["defect_gate"]["rejection_reasons"] = [str(status)]
    row["geometry_topology_gate"] = _rejected_geometry_gate(str(status))
    return row


def resolve_callback(reference: str) -> Callable[..., Mapping[str, Any]]:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("callback reference must use module:attribute")
    module_name, attribute = reference.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("callback reference must use module:attribute")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError("candidate callback is not callable")
    return callback


def run_control_variant(**kwargs: Any) -> Mapping[str, Any]:
    """Run the unchanged directed-local-topology control."""
    return run_one(
        kwargs["source"], DIRECTED_LOCAL_TOPOLOGY_PROFILE,
        output_dir=Path(kwargs["output_dir"]),
        breparg_root=Path(kwargs["breparg_root"]),
        joint_iterations=int(kwargs["joint_iterations"]),
        assembly_backend="directed", selector_geometry_gate=False,
        expected_source_binding=kwargs["expected_source_binding"])


def _joint_optimized_inputs(
    parsed: Mapping[str, Any], joint_iterations: int
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    face_edge_adj = [list(map(int, row)) for row in parsed["faceEdge_adj"]]
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    surf_wcs, edge_wcs = cpu_joint_optimize(
        np.asarray(parsed["surf_ncs"], dtype=np.float32),
        np.asarray(parsed["edge_ncs"], dtype=np.float32),
        np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
        np.asarray(parsed["corner_unique"], dtype=np.float32),
        edge_vertex_adj,
        face_edge_adj,
        iterations=int(joint_iterations),
    )
    return surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj


def _write_candidate_step(shape: Any, output_dir: Path, filename: str) -> Path:
    from OCC.Extend.DataExchange import write_step_file

    path = Path(output_dir) / "steps" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    write_step_file(shape, str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("candidate STEP writer produced no non-empty file")
    return path


def _count_nonfinite(value: Any) -> int:
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, Mapping):
        return sum(_count_nonfinite(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(_count_nonfinite(child) for child in value)
    return 0


def _compact_diagnostics(value: Any) -> Any:
    """Drop native handles and private callback state from protocol evidence."""
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            key_text = str(key)
            if (
                key_text.startswith("_")
                or key_text in {
                    "shape",
                    "face",
                    "observed_edge",
                    "observed_wire",
                    "source_mapping",
                    "source_edge_occurrences",
                }
            ):
                continue
            compact = _compact_diagnostics(child)
            if compact is not None:
                result[key_text] = compact
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [child for item in value if (child := _compact_diagnostics(item)) is not None]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return None


def _candidate_contract(
    *, step_path: Path, diagnostics: Mapping[str, Any],
    attempted: bool, applied: bool, target_defects_removed: bool,
    no_new_non_target_defects: bool, mapping_exact: bool,
    source_topology_preserved: bool, shared_edges_preserved: bool,
    curves_preserved: bool,
) -> dict[str, Any]:
    compact = _compact_diagnostics(diagnostics)
    nonfinite = _count_nonfinite(compact)
    reasons = []
    gates = {
        "target_defects_removed": bool(target_defects_removed),
        "no_new_non_target_defects": bool(no_new_non_target_defects),
        "mapping_exact": bool(mapping_exact),
        "source_topology_preserved": bool(source_topology_preserved),
        "shared_edge_correspondence_preserved": bool(shared_edges_preserved),
        "curves_3d_preserved": bool(curves_preserved),
        "source_binding_preserved": True,
    }
    if not attempted:
        reasons.append("candidate_not_attempted")
    if not applied:
        reasons.append("candidate_not_applied")
    reasons.extend(name for name, passed in gates.items() if not passed)
    if nonfinite:
        reasons.append("nonfinite_evidence")
    accepted = bool(attempted and applied and all(gates.values()) and not nonfinite)
    return {
        "step_path": str(step_path),
        "candidate_application": {
            "attempted": bool(attempted),
            "applied": bool(applied),
            "status": "applied" if applied else "rejected_by_local_helper",
            "diagnostics": compact,
        },
        "defect_gate": {
            "accepted": accepted,
            **gates,
            "nonfinite_count": nonfinite,
            "rejection_reasons": reasons,
        },
    }


def _whole_cad_step_defect_evidence(
    step_observation: Mapping[str, Any],
    *,
    target_source_face_indices: Sequence[int],
    target_source_edge_pairs: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Authorize defect booleans only from exact, source-indexed STEP evidence."""

    target_definition_complete = bool(
        target_source_face_indices
        and len(target_source_face_indices) == len(target_source_edge_pairs)
    )
    target_keys: set[tuple[int, tuple[int, int]]] = set()
    if target_definition_complete:
        for face_index, value in zip(
            target_source_face_indices, target_source_edge_pairs
        ):
            if (
                type(face_index) is not int
                or not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
                or any(type(item) is not int or item < 0 for item in value)
                or value[0] == value[1]
            ):
                target_definition_complete = False
                target_keys.clear()
                break
            target_keys.add(
                (int(face_index), tuple(sorted((int(value[0]), int(value[1])))))
            )
        if len(target_keys) != len(target_source_face_indices):
            target_definition_complete = False
    diagnosis = step_observation.get("diagnosis")
    mapping_failures = step_observation.get("mapping_failures")
    exact = bool(
        target_definition_complete
        and step_observation.get("lineage_status") == "exact_geometry_incidence"
        and isinstance(mapping_failures, Sequence)
        and not isinstance(mapping_failures, (str, bytes))
        and not mapping_failures
        and isinstance(diagnosis, Mapping)
        and diagnosis.get("status") == "diagnosed"
    )
    raw_occurrences = diagnosis.get("occurrences") if isinstance(diagnosis, Mapping) else None
    occurrences_complete = bool(
        exact
        and isinstance(raw_occurrences, Sequence)
        and not isinstance(raw_occurrences, (str, bytes))
    )
    occurrence_rows: list[Mapping[str, Any]] = (
        [row for row in raw_occurrences if isinstance(row, Mapping)]
        if occurrences_complete
        else []
    )
    if occurrences_complete and len(occurrence_rows) != len(raw_occurrences):
        occurrences_complete = False

    malformed = False
    target_residuals: list[dict[str, Any]] = []
    non_target_defects: list[dict[str, Any]] = []
    for row in occurrence_rows:
        face_index = row.get("source_face_index")
        edge_ids = row.get("source_edge_ids")
        if (
            type(face_index) is not int
            or not isinstance(edge_ids, Sequence)
            or isinstance(edge_ids, (str, bytes))
            or len(edge_ids) != 2
            or any(type(value) is not int or value < 0 for value in edge_ids)
            or edge_ids[0] == edge_ids[1]
        ):
            malformed = True
            continue
        pair = tuple(sorted(map(int, edge_ids)))
        public = {
            "kind": str(row.get("kind")),
            "source_face_index": int(face_index),
            "source_edge_ids": list(pair),
            "status": str(row.get("status")),
        }
        if (int(face_index), pair) in target_keys:
            target_residuals.append(public)
        else:
            non_target_defects.append(public)

    target_defects_removed = bool(
        exact and occurrences_complete and not malformed and not target_residuals
    )
    no_new_non_target_defects = bool(
        exact and occurrences_complete and not malformed and not non_target_defects
    )
    return {
        "accepted": bool(target_defects_removed and no_new_non_target_defects),
        "mapping_exact": exact,
        "target_definition_complete": target_definition_complete,
        "occurrences_complete": occurrences_complete,
        "malformed_occurrence_count": int(malformed),
        "target_defects_removed": target_defects_removed,
        "no_new_non_target_defects": no_new_non_target_defects,
        "target_residuals": target_residuals,
        "non_target_defects": non_target_defects,
        "final_occurrence_count": len(occurrence_rows),
        "geometry_incidence_proof": (
            diagnosis.get("geometry_incidence_proof")
            if isinstance(diagnosis, Mapping)
            else None
        ),
        "lineage_status": step_observation.get("lineage_status"),
        "mapping_failures": (
            list(mapping_failures)
            if isinstance(mapping_failures, Sequence)
            and not isinstance(mapping_failures, (str, bytes))
            else ["mapping_failures_malformed"]
        ),
    }


def run_47472_candidate_variant(**kwargs: Any) -> Mapping[str, Any]:
    """Integrate the face-local repair into one exact whole-CAD construction."""
    try:
        from .targeted_nonperiodic_pcurve_repair import (
            repair_face_targeted_nonperiodic_pcurves,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from targeted_nonperiodic_pcurve_repair import (
            repair_face_targeted_nonperiodic_pcurves,
        )

    variant: VariantSpec = kwargs["variant"]
    pairs_by_face = {
        face_index: tuple(
            pair
            for pair in variant.target_source_edge_pairs
            if face_index in {10, 43}
            and pair == ({10: (13, 20), 43: (16, 24)}[face_index])
        )
        for face_index in variant.target_source_face_indices
    }
    mutation_rows: list[dict[str, Any]] = []
    post_sewing_source_faces: dict[int, dict[str, Any]] = {}

    def observe(face_index: int, face: Any, metadata: Mapping[str, Any]) -> None:
        if metadata.get("phase") != "post_sewing_pre_step" or face is None:
            return
        mapping = metadata.get("source_mapping")
        if not isinstance(mapping, Mapping):
            mapping = {}
        post_sewing_source_faces[int(face_index)] = {
            "face": face,
            "source_mapping": dict(mapping),
        }

    def mutate(face_index: int, face: Any, metadata: Mapping[str, Any]):
        if face_index not in variant.target_source_face_indices:
            return face, {"accepted": False, "attempted": False,
                          "reason": "face_not_targeted"}
        repaired, diagnostics = repair_face_targeted_nonperiodic_pcurves(
            face,
            source_face_index=int(face_index),
            source_mapping=metadata["source_mapping"],
            source_edge_occurrences=metadata["source_edge_occurrences"],
            expected_source_edge_pairs=pairs_by_face[int(face_index)],
        )
        mutation_rows.append({"source_face_index": int(face_index), **dict(diagnostics)})
        return repaired, diagnostics

    surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj = _joint_optimized_inputs(
        kwargs["parsed"], kwargs["joint_iterations"]
    )
    shape, assembly = construct_brep_directed(
        surf_wcs,
        edge_wcs,
        face_edge_adj,
        edge_vertex_adj,
        breparg_root=Path(kwargs["breparg_root"]),
        **profile_kwargs(DIRECTED_LOCAL_TOPOLOGY_PROFILE),
        post_pcurve_face_mutator=mutate,
        assembly_stage_face_observer=observe,
    )
    step_path = _write_candidate_step(shape, kwargs["output_dir"], "47472_candidate.step")
    targeted = [row for row in mutation_rows if row["source_face_index"] in {10, 43}]
    applied = len(targeted) == 2 and all(row.get("accepted") is True for row in targeted)
    local_mapping_exact = applied and all(
        (row.get("post_repair_mapping_gate") or {}).get("accepted") is True
        for row in targeted
    )
    topology = applied and all(
        (row.get("topology_incidence_gate") or {}).get("accepted") is True
        for row in targeted
    )
    curves = applied and all(
        (row.get("curve_3d_preservation") or {}).get("accepted") is True
        for row in targeted
    )
    local_non_target_pcurves = applied and all(
        (row.get("non_target_pcurve_gate") or {}).get("accepted") is True
        for row in targeted
    )
    step_observation = _step_observation(
        step_path,
        breparg_root=Path(kwargs["breparg_root"]),
        source_face_references=post_sewing_source_faces,
        face_edge_adj=face_edge_adj,
    )
    whole_cad = _whole_cad_step_defect_evidence(
        step_observation,
        target_source_face_indices=variant.target_source_face_indices,
        target_source_edge_pairs=variant.target_source_edge_pairs,
    )
    complete_source_face_census = (
        len(post_sewing_source_faces) == len(face_edge_adj)
        and set(post_sewing_source_faces) == set(range(len(face_edge_adj)))
    )
    whole_cad_mapping_exact = bool(
        whole_cad["mapping_exact"] and complete_source_face_census
    )
    return _candidate_contract(
        step_path=step_path,
        diagnostics={
            "face_mutations": targeted,
            "local_face_gates": {
                "mapping_exact": local_mapping_exact,
                "non_target_pcurves_preserved": local_non_target_pcurves,
            },
            "whole_cad_step_gate": whole_cad,
            "complete_post_sewing_source_face_census": complete_source_face_census,
            "assembly": assembly,
        },
        attempted=bool(targeted),
        applied=applied,
        target_defects_removed=bool(
            applied and whole_cad["target_defects_removed"]
        ),
        no_new_non_target_defects=bool(
            applied and whole_cad["no_new_non_target_defects"]
        ),
        mapping_exact=bool(applied and whole_cad_mapping_exact),
        source_topology_preserved=topology,
        shared_edges_preserved=bool(applied and whole_cad_mapping_exact),
        curves_preserved=curves,
    )


def run_63055_candidate_variant(**kwargs: Any) -> Mapping[str, Any]:
    """Integrate the post-sewing graph repair with the preregistered 1e-4 arm."""
    try:
        from .post_sewing_graph_repair import (
            attempt_post_sewing_face_pcurve_reprojection,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from post_sewing_graph_repair import (
            attempt_post_sewing_face_pcurve_reprojection,
        )

    variant: VariantSpec = kwargs["variant"]
    mutation: dict[str, Any] = {}

    def mutate(sewn: Any, bindings: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]):
        # The low-level gate deliberately requires complete census counts on
        # every face binding. The constructor gives those counts once at the
        # shape hook boundary, so the adapter copies them into each private
        # row rather than weakening the helper's completeness checks.
        enriched_bindings = [
            {
                **dict(binding),
                "expected_source_face_count": metadata[
                    "expected_source_face_count"
                ],
                "expected_source_edge_count": metadata[
                    "expected_source_edge_count"
                ],
            }
            for binding in bindings
        ]
        repaired, diagnostics = attempt_post_sewing_face_pcurve_reprojection(
            sewn,
            source_face_bindings=enriched_bindings,
            target_source_face_index=5,
            target_source_edge_ids=(9, 23),
            expected_source_edge_pairs=variant.target_source_edge_pairs,
            projection_precision=float(variant.candidate_sewing_tolerance or 1e-4),
        )
        mutation.update(diagnostics)
        return repaired, diagnostics

    surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj = _joint_optimized_inputs(
        kwargs["parsed"], kwargs["joint_iterations"]
    )
    shape, assembly = construct_brep_directed(
        surf_wcs,
        edge_wcs,
        face_edge_adj,
        edge_vertex_adj,
        breparg_root=Path(kwargs["breparg_root"]),
        **profile_kwargs(DIRECTED_LOCAL_TOPOLOGY_PROFILE),
        sewing_tolerance=float(variant.candidate_sewing_tolerance or 1e-4),
        post_sewing_shape_mutator=mutate,
    )
    step_path = _write_candidate_step(shape, kwargs["output_dir"], "63055_candidate.step")
    graph = mutation.get("graph_preservation_gate") or {}
    topology = mutation.get("topology_incidence_gate") or {}
    identity = mutation.get("source_edge_identity_gate") or {}
    curves = mutation.get("curve_3d_preservation") or {}
    target_after = mutation.get("target_face_after") or {}
    applied = mutation.get("accepted") is True
    return _candidate_contract(
        step_path=step_path,
        diagnostics={"post_sewing_mutation": mutation, "assembly": assembly},
        attempted=mutation.get("attempted") is True,
        applied=applied,
        target_defects_removed=target_after.get("accepted") is True,
        no_new_non_target_defects=graph.get("accepted") is True,
        mapping_exact=identity.get("accepted") is True,
        source_topology_preserved=topology.get("accepted") is True,
        shared_edges_preserved=identity.get("accepted") is True,
        curves_preserved=curves.get("accepted") is True,
    )


def _public_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): child for key, child in value.items()
            if str(key) not in {"shape", "step_path", "output_dir"}
            and not str(key).startswith("_")}


def _assert_path_free_evidence(value: Any, *, label: str) -> None:
    """Keep callback proof objects portable and safe to archive later."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text.endswith("_path") or key_text in {"path", "output_dir"}:
                raise ValueError(f"{label} contains path-bearing key {key!r}")
            _assert_path_free_evidence(child, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_path_free_evidence(child, label=f"{label}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if (normalized.startswith("/") or normalized.startswith("//")
                or (len(normalized) >= 3 and normalized[1:3] == ":/")):
            raise ValueError(f"{label} contains an absolute path")


def _normalize_candidate_application(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("candidate_application is missing")
    if type(value.get("attempted")) is not bool or type(value.get("applied")) is not bool:
        raise ValueError("candidate_application booleans are malformed")
    if not isinstance(value.get("status"), str) or not value["status"]:
        raise ValueError("candidate_application status is malformed")
    public = _public_mapping(value)
    _assert_json_safe(public, label="candidate_application")
    _assert_path_free_evidence(public, label="candidate_application")
    if value["applied"] and not value["attempted"]:
        raise ValueError("candidate cannot be applied without being attempted")
    return public


def _normalize_defect_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("defect_gate is missing")
    for name in DEFECT_BOOLEAN_FIELDS:
        if type(value.get(name)) is not bool:
            raise ValueError(f"defect_gate {name} must be boolean")
    nonfinite = value.get("nonfinite_count")
    reasons = value.get("rejection_reasons")
    if type(nonfinite) is not int or nonfinite < 0:
        raise ValueError("defect_gate nonfinite_count is malformed")
    if (not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes))
            or any(not isinstance(reason, str) for reason in reasons)):
        raise ValueError("defect_gate rejection_reasons is malformed")
    logically_accepted = (all(value[name] is True for name in DEFECT_BOOLEAN_FIELDS[1:])
                          and nonfinite == 0 and not reasons)
    if value["accepted"] is not logically_accepted:
        raise ValueError("defect_gate accepted is inconsistent")
    public = _public_mapping(value)
    _assert_json_safe(public, label="defect_gate")
    _assert_path_free_evidence(public, label="defect_gate")
    return public


def _resolve_step_path(result: Mapping[str, Any], output_dir: Path) -> Path | None:
    raw = result.get("step_path", result.get("step_relative_path"))
    if raw is None:
        return None
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = Path(output_dir) / candidate
    root, resolved = Path(output_dir).resolve(), candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("callback STEP escapes isolated output") from exc
    return resolved


def _input_geometry(parsed: Mapping[str, Any], joint_iterations: int) -> dict[str, Any]:
    face_edge_adj = [list(map(int, row)) for row in parsed["faceEdge_adj"]]
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    surf_wcs, edge_wcs = cpu_joint_optimize(
        np.asarray(parsed["surf_ncs"], dtype=np.float32),
        np.asarray(parsed["edge_ncs"], dtype=np.float32),
        np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
        np.asarray(parsed["corner_unique"], dtype=np.float32),
        edge_vertex_adj, face_edge_adj, iterations=int(joint_iterations))
    signature = input_geometry_signature(surf_wcs, edge_wcs, face_edge_adj,
                                         edge_vertex_adj)
    return {"signature": signature, "edge_samples": sample_input_edge_points(edge_wcs),
            "edge_polylines": edge_wcs}


def _measure_step(step_path: Path, *, breparg_root: Path,
                  input_geometry: Mapping[str, Any]) -> dict[str, Any]:
    if not step_path.is_file() or step_path.stat().st_size <= 0:
        raise ValueError("callback did not produce a non-empty STEP")
    validity = strict_validate_step(step_path, breparg_root=breparg_root)
    candidate = candidate_step_signature(
        step_path, input_edge_samples=input_geometry["edge_samples"],
        input_edge_polylines=input_geometry["edge_polylines"],
        input_signature=input_geometry["signature"],
        validity_components=validity["validity_components"])
    gate = geometry_topology_gate(input_geometry["signature"], candidate)
    return {
        "step_saved": True, "step_readable": True,
        "step_bytes": step_path.stat().st_size, "step_sha256": sha256_file(step_path),
        "validity_components": validity["validity_components"],
        "native_brep_valid": bool(validity["native_brep_valid"]),
        "strict_brep_valid": bool(validity["strict_brep_valid"]),
        "both_valid": bool(validity["both_valid"]), "geometry_topology_gate": gate,
    }


def run_worker(
    source: Mapping[str, Any], variant: VariantSpec, *, output_dir: Path,
    breparg_root: Path, joint_iterations: int,
    expected_binding: Mapping[str, Any], run_signature: str,
) -> dict[str, Any]:
    """Execute and independently measure one matrix cell inside a child."""
    started = time.perf_counter()
    expected = normalize_binding(expected_binding)
    row = _base_row(source, variant, run_signature=run_signature,
                    expected_binding=expected)
    try:
        source_path = Path(str(source["source_path"]))
        before = source_binding(source_path)
        row["source_binding_before"] = before
        if before != expected:
            raise RuntimeError("source_binding_mismatch_before_load")
        source_bytes = source_path.read_bytes()
        loaded = source_pickle_binding_from_bytes(source_bytes)
        row["source_binding_loaded_bytes"] = loaded
        if loaded != expected:
            raise RuntimeError("source_binding_mismatch_loaded_bytes")
        parsed = pickle.loads(source_bytes)
        after_load = source_binding(source_path)
        row["source_binding_after_load"] = after_load
        if after_load != expected:
            raise RuntimeError("source_binding_mismatch_after_load")
        try:
            callback = resolve_callback(variant.callback_ref)
        except (AttributeError, ImportError, ModuleNotFoundError, TypeError, ValueError):
            if not variant.is_candidate:
                raise
            row.update(status="candidate_hook_missing",
                       error_type="CandidateCallbackUnavailable")
            row["candidate_application"]["status"] = "candidate_hook_missing"
            row["defect_gate"]["rejection_reasons"] = ["candidate_hook_missing"]
            row["geometry_topology_gate"] = _rejected_geometry_gate("candidate_hook_missing")
            row["source_binding_after_attempt"] = source_binding(source_path)
            return row
        result = callback(
            source=source, source_bytes=source_bytes, parsed=parsed,
            output_dir=Path(output_dir), breparg_root=Path(breparg_root),
            joint_iterations=int(joint_iterations), variant=variant,
            expected_source_binding=expected, run_signature=run_signature)
        if not isinstance(result, Mapping):
            raise ValueError("callback result is not a mapping")
        row["callback_completed"] = True
        row["source_binding_after_attempt"] = source_binding(source_path)
        if row["source_binding_after_attempt"] != expected:
            raise RuntimeError("source_binding_mismatch_after_attempt")
        if variant.is_candidate:
            row["candidate_application"] = _normalize_candidate_application(
                result.get("candidate_application"))
            row["defect_gate"] = _normalize_defect_gate(result.get("defect_gate"))
        else:
            row["candidate_application"] = {"attempted": False, "applied": False,
                                            "status": "control"}
        step_path = _resolve_step_path(result, Path(output_dir))
        if step_path is not None:
            row.update(_measure_step(step_path, breparg_root=Path(breparg_root),
                                     input_geometry=_input_geometry(parsed, joint_iterations)))
            row["step_relative_path"] = step_path.relative_to(
                Path(output_dir).resolve()).as_posix()
        else:
            row["geometry_topology_gate"] = _rejected_geometry_gate(
                "geometry_gate:step_missing")
        if not variant.is_candidate:
            bad_wires = (row.get("validity_components") or {}).get(
                "wire_self_intersections")
            reproduced = bool(
                row["step_readable"]
                and row["native_brep_valid"] is variant.expected_control_native_valid
                and row["strict_brep_valid"] is variant.expected_control_strict_valid
                and bad_wires == variant.expected_control_wire_self_intersections)
            row["control_expectation"] = {
                "reproduced": reproduced,
                "expected_native_brep_valid": variant.expected_control_native_valid,
                "expected_strict_brep_valid": variant.expected_control_strict_valid,
                "expected_wire_self_intersections": variant.expected_control_wire_self_intersections,
                "observed_wire_self_intersections": bad_wires,
            }
            row["defect_gate"] = {**{name: False for name in DEFECT_BOOLEAN_FIELDS},
                                  "nonfinite_count": 0,
                                  "rejection_reasons": ["control_registered_failure"]}
            row["status"] = "control_reproduced" if reproduced else "control_drift"
        else:
            gate_valid, gate_reasons = validate_accepted_geometry_gate(
                row["geometry_topology_gate"])
            accepted = bool(row["candidate_application"].get("attempted") is True
                            and row["candidate_application"].get("applied") is True
                            and row["defect_gate"].get("accepted") is True
                            and row["both_valid"] and gate_valid and not gate_reasons)
            row["status"] = "candidate_accepted" if accepted else "candidate_rejected"
    except RuntimeError as exc:
        code = str(exc)
        row.update(status=("source_binding_mismatch" if code.startswith(
            "source_binding_mismatch") else "worker_error"), error_type=code)
    except Exception as exc:
        row.update(status="worker_error", error_type=type(exc).__name__)
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
    return row


def validate_attempt_row(
    row: Mapping[str, Any], *, source: Mapping[str, Any], variant: VariantSpec,
    run_signature: str, expected_binding: Mapping[str, Any],
) -> None:
    """Reject a row that escapes or overclaims its signed matrix cell."""
    required = {
        "schema": SCHEMA, "task_id": variant.task_id, "cad_id": variant.cad_id,
        "parent_id": source.get("parent_id"), "arm": variant.arm,
        "callback_ref": variant.callback_ref, "run_signature": run_signature,
        "denominator": True,
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_binding_expected": normalize_binding(expected_binding),
    }
    for name, expected in required.items():
        if row.get(name) != expected:
            raise ValueError(f"attempt {name} mismatches signed cell")
    if not isinstance(row.get("status"), str) or not row["status"]:
        raise ValueError("attempt status must be non-empty")
    if row["status"] not in ATTEMPT_STATUSES:
        raise ValueError("attempt status is not registered")
    if variant.is_candidate and row["status"].startswith("control_"):
        raise ValueError("candidate cell cannot report a control status")
    if not variant.is_candidate and row["status"].startswith("candidate_"):
        raise ValueError("control cell cannot report a candidate status")
    for name in ("callback_completed", "step_saved", "step_readable",
                 "native_brep_valid", "strict_brep_valid", "both_valid"):
        if type(row.get(name)) is not bool:
            raise ValueError(f"attempt {name} must be boolean")
    if row["both_valid"] is not bool(row["native_brep_valid"] and
                                     row["strict_brep_valid"]):
        raise ValueError("attempt both_valid is inconsistent")
    if row["step_readable"] and not row["step_saved"]:
        raise ValueError("readable STEP must be saved")
    if (row["native_brep_valid"] or row["strict_brep_valid"]) and not row["step_readable"]:
        raise ValueError("valid attempt must have readable STEP")
    binding_fields = ("source_binding_before", "source_binding_loaded_bytes",
                      "source_binding_after_load", "source_binding_after_attempt")
    for name in binding_fields:
        if row.get(name) is not None:
            normalize_binding(row[name])
    if row["status"] not in WORKER_FAILURE_STATUSES | {"candidate_hook_missing"}:
        expected = normalize_binding(expected_binding)
        if any(row.get(name) != expected for name in binding_fields):
            raise ValueError("completed attempt lacks four exact source bindings")
    application, defect, geometry = (row.get("candidate_application"),
                                     row.get("defect_gate"),
                                     row.get("geometry_topology_gate"))
    if not isinstance(application, Mapping):
        raise ValueError("candidate_application must be a mapping")
    if type(application.get("attempted")) is not bool or type(application.get("applied")) is not bool:
        raise ValueError("candidate application booleans are malformed")
    if not isinstance(defect, Mapping):
        raise ValueError("defect_gate must be a mapping")
    if not isinstance(geometry, Mapping) or geometry.get("schema") != GEOMETRY_GATE_SCHEMA:
        raise ValueError("attempt lacks schema-v2 geometry evidence")
    if row["status"] == "control_reproduced":
        control = row.get("control_expectation")
        if variant.is_candidate or not isinstance(control, Mapping) or control.get("reproduced") is not True:
            raise ValueError("control success lacks evidence")
    if row["status"] == "candidate_accepted":
        gate_valid, reasons = validate_accepted_geometry_gate(geometry)
        if (not variant.is_candidate or row["callback_completed"] is not True
                or _normalize_candidate_application(application)["attempted"] is not True
                or application["applied"] is not True
                or _normalize_defect_gate(defect)["accepted"] is not True
                or row["both_valid"] is not True or not gate_valid or reasons):
            raise ValueError("candidate_accepted overclaims incomplete evidence")
    elif variant.is_candidate and row["callback_completed"] is True:
        _normalize_candidate_application(application)
        _normalize_defect_gate(defect)
    _assert_json_safe(row, label="attempt")


def validate_saved_step(row: Mapping[str, Any], *, output_dir: Path) -> None:
    """Rebind a resumed row to its immutable promoted STEP artifact."""
    if row.get("step_saved") is not True:
        return
    relative = Path(str(row.get("step_relative_path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("saved STEP relative path is invalid")
    root = Path(output_dir).resolve()
    step_path = (root / relative).resolve()
    step_path.relative_to(root)
    if (not step_path.is_file() or step_path.stat().st_size != row.get("step_bytes")
            or sha256_file(step_path) != row.get("step_sha256")):
        raise ValueError("saved STEP artifact is missing or hash-mismatched")


def validate_terminal_artifact_hashes(
    record: Mapping[str, Any], *, rows_path: Path, summary_path: Path
) -> dict[str, Any]:
    """Rebind a terminal run record to its immutable ledger and summary."""
    if record.get("rows_sha256") != sha256_file(rows_path):
        raise RuntimeError("terminal attempt ledger hash mismatched")
    if record.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("terminal summary hash mismatched")
    value = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError("terminal summary is not a JSON object")
    return dict(value)


def _subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def run_isolated(
    source: Mapping[str, Any], variant: VariantSpec, *, args: argparse.Namespace,
    run_signature: str, expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one cell in a fresh process and retain every failure as one row."""
    root = Path(args.output_dir).resolve()
    slug = _task_slug(variant.task_id)
    work_root = root / ".attempts"
    work_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(tempfile.mkdtemp(prefix=f"{slug}-", dir=work_root))
    # Include the unique child directory name in logs and promoted STEP paths.
    # If the parent is interrupted after promotion but before JSONL append, a
    # retry cannot overwrite the orphaned native artifact.
    attempt_name = attempt_dir.name
    log_dir = root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = (
        log_dir / f"{attempt_name}.stdout.log",
        log_dir / f"{attempt_name}.stderr.log",
    )
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--calibration-manifest", str(Path(args.calibration_manifest).resolve()),
        "--selector-matrix", str(Path(args.selector_matrix).resolve()),
        "--selector-run", str(Path(args.selector_run).resolve()),
        "--lineage-cases", str(Path(args.lineage_cases).resolve()),
        "--lineage-run", str(Path(args.lineage_run).resolve()),
        "--breparg-root", str(Path(args.breparg_root).resolve()),
        "--output-dir", str(attempt_dir),
        "--joint-iterations", str(int(args.joint_iterations)),
        "--worker-timeout-seconds", str(float(args.worker_timeout_seconds)),
        "--worker-task-id", variant.task_id,
        "--worker-run-signature", run_signature,
        "--worker-source-binding-json", json.dumps(
            dict(expected_binding), sort_keys=True, separators=(",", ":")),
        "--worker-source-path", str(Path(str(source["source_path"])).resolve()),
        "--worker-parent-id", str(source["parent_id"]),
    ]
    try:
        completed = subprocess.run(
            command, cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=float(args.worker_timeout_seconds), check=False)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_subprocess_output(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_subprocess_output(exc.stderr), encoding="utf-8")
        return failure_row(source, variant, run_signature=run_signature,
            expected_binding=expected_binding, status="worker_timeout",
            error_type="TimeoutExpired")
    except OSError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(type(exc).__name__, encoding="utf-8")
        return failure_row(source, variant, run_signature=run_signature,
            expected_binding=expected_binding, status="worker_spawn_error",
            error_type=type(exc).__name__)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return failure_row(source, variant, run_signature=run_signature,
            expected_binding=expected_binding, status="worker_process_exit",
            error_type="NonzeroWorkerExit", returncode=int(completed.returncode))
    row = parse_worker_result(completed.stdout)
    if row is None:
        return failure_row(source, variant, run_signature=run_signature,
            expected_binding=expected_binding, status="worker_protocol_error",
            error_type="InvalidWorkerSentinel", returncode=int(completed.returncode))
    try:
        validate_attempt_row(row, source=source, variant=variant,
                             run_signature=run_signature,
                             expected_binding=expected_binding)
        if source_binding(Path(str(source["source_path"]))) != normalize_binding(expected_binding):
            raise ValueError("source binding changed after child")
        if row["step_saved"]:
            relative = Path(str(row.get("step_relative_path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("worker STEP relative path is invalid")
            staged = (attempt_dir / relative).resolve()
            staged.relative_to(attempt_dir.resolve())
            if (not staged.is_file() or staged.stat().st_size != row.get("step_bytes")
                    or sha256_file(staged) != row.get("step_sha256")):
                raise ValueError("worker STEP is missing or hash-mismatched")
            final_relative = Path("steps") / slug / f"{attempt_name}.step"
            final = root / final_relative
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise ValueError("final STEP exists before row append")
            os.replace(staged, final)
            row["step_relative_path"] = final_relative.as_posix()
        row["worker_returncode"] = int(completed.returncode)
        row["worker_stdout_log"] = f"worker_logs/{stdout_path.name}"
        row["worker_stderr_log"] = f"worker_logs/{stderr_path.name}"
        validate_attempt_row(row, source=source, variant=variant,
                             run_signature=run_signature,
                             expected_binding=expected_binding)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return failure_row(source, variant, run_signature=run_signature,
            expected_binding=expected_binding, status="worker_protocol_error",
            error_type=type(exc).__name__, returncode=int(completed.returncode))
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the pilot decision without authorizing broader cohorts."""
    expected_ids = [variant.task_id for variant in VARIANTS]
    if len(rows) != 4 or [row.get("task_id") for row in rows] != expected_ids:
        raise ValueError("summary requires four ordered attempts")
    controls = [row for row in rows if row.get("arm") == "control"]
    candidates = [row for row in rows if row.get("arm") == "candidate"]
    failures = sum(row.get("status") in WORKER_FAILURE_STATUSES for row in rows)
    unavailable = sum(row.get("status") == "candidate_hook_missing" for row in rows)
    drift = sum(row.get("status") != "control_reproduced" for row in controls)
    accepted = [str(row["cad_id"]) for row in candidates
                if row.get("status") == "candidate_accepted"]
    rejected = [str(row["cad_id"]) for row in candidates
                if row.get("status") == "candidate_rejected"]
    complete = all(row.get("callback_completed") is True for row in candidates)
    conclusive = failures == unavailable == drift == 0 and complete
    decision = ("INCONCLUSIVE_REQUIRES_IMPLEMENTATION_OR_RERUN" if not conclusive
                else "EXPAND_ACCEPTED_CANDIDATES_TO_RELEVANT_RESIDUAL_FAMILIES"
                if accepted else "CLOSE_EXACT_CAD_CANDIDATES")
    return {
        "schema": SUMMARY_SCHEMA, "attempts": len(rows),
        "denominator_rows": sum(row.get("denominator") is True for row in rows),
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "controls_reproduced": len(controls) - drift, "control_drift": drift,
        "candidate_callbacks_complete": sum(row.get("callback_completed") is True
                                              for row in candidates),
        "candidate_hooks_unavailable": unavailable,
        "candidate_accepted_cad_ids": accepted,
        "candidate_rejected_cad_ids": rejected,
        "worker_or_protocol_failures": failures,
        "nonfinite_count": sum(int((row.get("defect_gate") or {}).get(
            "nonfinite_count") or 0) for row in candidates),
        "conclusive": conclusive, "decision": decision,
        "assembly_release_gate_before": {"strict_valid": 91, "required": 95},
        "authorizes_relevant_residual_expansion": bool(conclusive and accepted),
        "authorizes_full_100cad": False, "authorizes_training_or_ar": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration-manifest", "selector-matrix", "selector-run",
                 "lineage-cases", "lineage-run", "breparg-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-task-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-signature", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-binding-json", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-path", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-parent-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.joint_iterations < 0:
        parser.error("--joint-iterations must be non-negative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    return args


def _load_bound_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]],
                                                           list[dict[str, Any]],
                                                           dict[str, Any]]:
    calibration_rows, selector_rows = (read_rows(args.calibration_manifest),
                                       read_rows(args.selector_matrix))
    sources = select_lineage_sources(calibration_rows, selector_rows,
                                     target_ids=TARGET_CAD_IDS)
    lineage = validate_lineage_evidence(
        read_rows(args.lineage_cases), sources=sources,
        lineage_cases=args.lineage_cases, lineage_run=args.lineage_run)
    return sources, selector_rows, lineage


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources, selector_rows, lineage = _load_bound_inputs(args)
    payload = build_run_payload(args, sources=sources, selector_rows=selector_rows,
                                lineage_binding=lineage)
    signature = canonical_sha256(payload)
    sources_by_id = {str(source["cad_id"]): source for source in sources}
    bindings = {str(item["cad_id"]): dict(item["binding"]) for item in payload["sources"]}
    if args.worker_task_id is not None:
        variant = VARIANTS_BY_ID.get(str(args.worker_task_id))
        if (variant is None or not args.worker_run_signature
                or not args.worker_source_binding_json or args.worker_source_path is None
                or not args.worker_parent_id):
            raise SystemExit("worker mode requires one exact signed task and source")
        source = sources_by_id[variant.cad_id]
        try:
            supplied = normalize_binding(json.loads(args.worker_source_binding_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit("worker source binding is malformed") from exc
        if (args.worker_run_signature != signature or supplied != bindings[variant.cad_id]
                or args.worker_source_path.resolve() != Path(str(source["source_path"])).resolve()
                or args.worker_parent_id != str(source["parent_id"])):
            raise SystemExit("worker arguments mismatch signed task")
        try:
            row = run_worker(
                source, variant, output_dir=args.output_dir,
                breparg_root=args.breparg_root, joint_iterations=args.joint_iterations,
                expected_binding=supplied, run_signature=signature)
        except Exception as exc:
            row = failure_row(source, variant, run_signature=signature,
                              expected_binding=supplied, status="worker_error",
                              error_type=type(exc).__name__, returncode=0)
        print(WORKER_MARKER + json.dumps(row, sort_keys=True, ensure_ascii=True,
                                         allow_nan=False), flush=True)
        return 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_writer_lock(output_dir):
        record = bind_run_manifest(output_dir, payload)
        rows_path = output_dir / ROWS_NAME
        rows = read_rows(rows_path, recover_truncated_tail=True)
        seen: set[str] = set()
        for row in rows:
            task_id = str(row.get("task_id"))
            if task_id in seen or task_id not in VARIANTS_BY_ID:
                raise RuntimeError("existing rows escape or duplicate matrix")
            seen.add(task_id)
            variant = VARIANTS_BY_ID[task_id]
            validate_attempt_row(row, source=sources_by_id[variant.cad_id],
                variant=variant, run_signature=signature,
                expected_binding=bindings[variant.cad_id])
            validate_saved_step(row, output_dir=output_dir)
        if record.get("status") in {"COMPLETED", "INCONCLUSIVE"}:
            if len(rows) != 4 or not (output_dir / SUMMARY_NAME).is_file():
                raise RuntimeError("terminal run manifest has incomplete artifacts")
            archived_summary = validate_terminal_artifact_hashes(
                record,
                rows_path=rows_path,
                summary_path=output_dir / SUMMARY_NAME,
            )
            by_task = {str(row["task_id"]): row for row in rows}
            summary = summarize([by_task[variant.task_id] for variant in VARIANTS])
            if archived_summary != summary:
                raise RuntimeError("terminal summary content drifted")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary["conclusive"] else 2
        for variant in VARIANTS:
            if variant.task_id in seen:
                continue
            row = run_isolated(
                sources_by_id[variant.cad_id], variant, args=args,
                run_signature=signature, expected_binding=bindings[variant.cad_id])
            append_row(rows_path, row)
            rows.append(row)
            seen.add(variant.task_id)
            print(json.dumps({"task_id": variant.task_id, "status": row.get("status"),
                              "both_valid": row.get("both_valid")}, sort_keys=True),
                  flush=True)
        # Rebind every source and every promoted STEP immediately before
        # signing terminal output. This catches source mutation or artifact
        # deletion that happened between an earlier attempt and completion.
        for variant in VARIANTS:
            source = sources_by_id[variant.cad_id]
            if source_binding(Path(str(source["source_path"]))) != bindings[variant.cad_id]:
                raise RuntimeError("source binding drifted before terminal summary")
        for row in rows:
            validate_saved_step(row, output_dir=output_dir)
        by_task = {str(row["task_id"]): row for row in rows}
        ordered = [by_task[variant.task_id] for variant in VARIANTS]
        summary = summarize(ordered)
        atomic_json(output_dir / SUMMARY_NAME, summary)
        final_record = dict(record)
        final_record.update(status="COMPLETED" if summary["conclusive"] else "INCONCLUSIVE",
                            attempts=4, rows_sha256=sha256_file(rows_path),
                            summary_sha256=sha256_file(output_dir / SUMMARY_NAME))
        atomic_json(output_dir / RUN_NAME, final_record)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["conclusive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
