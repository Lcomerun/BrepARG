"""Run independent assembly-repair profiles on the frozen 100-CAD cohort.

The runner keeps every attempt in the denominator, writes STEP files only to a
local output root, and emits a machine-readable restored/regressed map.  A
profile passes only when it preserves all historically strict-valid controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import RepairProfile, parse_profiles
    from .assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from .diagnose_step_validity_components import diagnose_step
    from .directed_trim_assembly import construct_brep_directed
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .run_p0b_stability_retest import (
        atomic_json,
        canonical_signature,
        output_root_writer_lock,
    )
    from .solid_topology_repair import reconcile_near_vertices
except ImportError:  # direct script execution
    from assembly_repair import RepairProfile, parse_profiles
    from assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from diagnose_step_validity_components import diagnose_step
    from directed_trim_assembly import construct_brep_directed
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from run_p0b_stability_retest import (
        atomic_json,
        canonical_signature,
        output_root_writer_lock,
    )
    from solid_topology_repair import reconcile_near_vertices


SCHEMA = "assembly-repair-matrix-v1"
RUN_SCHEMA = "assembly-repair-run-v2"
RUN_MANIFEST_NAME = "assembly_repair_run.json"
EXPECTED_CADS = 100
EXPECTED_BASELINE_VALID = 84
WORKER_MARKER = "__ASSEMBLY_REPAIR_WORKER_RESULT__="
ISOLATED_WORKER_SWITCHES = frozenset(
    {
        "local_intersection_topology",
        "local_pcurve_continuity",
        # Near-vertex reconciliation changes low-level OCC edge vertices.
        # Contain it just like the face-level repair profiles so one native
        # failure cannot compromise the matrix denominator.
        "single_solid",
        "near_vertex_reconciliation",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_PICKLE_BINDING_FIELDS = frozenset({"bytes", "sha256"})


class SourcePickleBindingMismatch(RuntimeError):
    """The source bytes changed or differ from the signed worker input."""


def source_pickle_binding_from_bytes(payload: bytes) -> dict[str, Any]:
    """Return a path-free identity for the exact bytes given to ``pickle.loads``."""
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def normalize_source_pickle_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Reject malformed or path-bearing source bindings before an OCC attempt."""
    if not isinstance(binding, Mapping) or set(binding) != SOURCE_PICKLE_BINDING_FIELDS:
        raise ValueError("source pickle binding must contain exactly bytes and sha256")
    byte_count = binding.get("bytes")
    digest = binding.get("sha256")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError("source pickle binding bytes must be a non-negative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("source pickle binding sha256 must be a lowercase hex digest")
    return {"bytes": byte_count, "sha256": digest}


def source_pickle_binding(path: Path) -> dict[str, Any]:
    """Hash a pickle without putting its host path into the binding payload."""
    return source_pickle_binding_from_bytes(Path(path).read_bytes())


def read_jsonl(
    path: Path, *, recover_truncated_tail: bool = False
) -> list[dict[str, Any]]:
    """Read JSONL, optionally removing only an unterminated final torn write."""
    target = Path(path)
    payload = target.read_bytes()
    rows: list[dict[str, Any]] = []
    offset = 0
    lines = payload.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            offset += len(raw_line)
            continue
        try:
            row = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            torn_final_line = (
                index == len(lines) - 1
                and not raw_line.endswith((b"\n", b"\r"))
            )
            if recover_truncated_tail and torn_final_line:
                with target.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                return rows
            raise
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in JSONL: {target}")
        rows.append(row)
        offset += len(raw_line)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def requires_isolated_worker(profile: RepairProfile) -> bool:
    """Return whether an OCC repair must be contained in a child process."""
    return any(profile.enabled(name) for name in ISOLATED_WORKER_SWITCHES)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def git_identity(repo_root: Path) -> dict[str, Any]:
    """Return a reproducible revision binding for the local repair sources."""
    root = Path(repo_root).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise RuntimeError(
            "cannot bind assembly repair source revision: "
            f"rev={revision.stderr.strip()!r} status={status.stderr.strip()!r}"
        )
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "status_sha256": hashlib.sha256(status.stdout.encode("utf-8")).hexdigest(),
    }


def assembly_source_hashes(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    relative_paths = (
        "tools/assembly_repair.py",
        "tools/directed_trim_assembly.py",
        "tools/local_wire_topology_repair.py",
        "tools/solid_topology_repair.py",
        "tools/run_assembly_calibration_oracle.py",
        "tools/run_assembly_repair_matrix.py",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        result[relative] = sha256_file(source)
    return result


def cohort_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_signature(
        {
            "ordered_rows": [
                {
                    "cad_id": str(row["cad_id"]),
                    "parent_id": row.get("parent_id"),
                    "source_path": str(row["source_path"]),
                    "historical_strict_valid": bool(row.get("brep_valid")),
                }
                for row in rows
            ]
        }
    )


def build_run_payload(
    *,
    args: argparse.Namespace,
    full_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[RepairProfile],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    breparg_root = Path(args.breparg_root).resolve()
    breparg_utils = breparg_root / "utils.py"
    if not breparg_utils.is_file():
        raise FileNotFoundError(breparg_utils)
    return {
        "schema": RUN_SCHEMA,
        "matrix_schema": SCHEMA,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "full_cohort_count": len(full_rows),
        "full_cohort_signature": cohort_signature(full_rows),
        "selected_cohort_count": len(selected_rows),
        "selected_cohort_signature": cohort_signature(selected_rows),
        "profiles": [
            {"name": profile.name, "switches": list(profile.switches)}
            for profile in profiles
        ],
        "joint_iterations": int(args.joint_iterations),
        "assembly_backend": str(args.assembly_backend),
        "historical_invalid_only": bool(args.historical_invalid_only),
        "max_cads": args.max_cads,
        "isolate_cad_workers": bool(args.isolate_cad_workers),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "repository": {
            **git_identity(repo_root),
            "source_sha256": assembly_source_hashes(repo_root),
        },
        "breparg_runtime": {
            "utils_sha256": sha256_file(breparg_utils),
        },
    }


def bind_run_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create or validate an immutable run contract for safe resume."""
    root = Path(output_dir).resolve()
    path = root / RUN_MANIFEST_NAME
    signature = canonical_signature(payload)
    expected = {
        "schema": RUN_SCHEMA,
        "signature": signature,
        "payload": dict(payload),
        "status": "RUNNING",
    }
    if path.is_file():
        current = read_json_object(path)
        if (
            current.get("schema") != RUN_SCHEMA
            or current.get("signature") != signature
            or current.get("payload") != dict(payload)
        ):
            raise RuntimeError(
                "assembly repair output root belongs to a different run signature: "
                f"{path}"
            )
        current["status"] = "RUNNING"
        # A prior interrupted attempt may retain an exception string containing
        # a host path.  It is failure-only state and must not survive a clean
        # resumed completion or enter a Git-safe report.
        current.pop("error", None)
        current.pop("error_type", None)
        atomic_json(path, current)
        return current
    unexpected = [
        candidate
        for candidate in root.iterdir()
        if candidate.name not in {".p0b_writer.lock", RUN_MANIFEST_NAME}
    ]
    if unexpected:
        raise RuntimeError(
            "assembly repair output root has artifacts but no signed run manifest: "
            f"{root}"
        )
    atomic_json(path, expected)
    return expected


def frozen_original_rows(calibration_manifest: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(calibration_manifest) if row.get("arm") == "original"]
    if len(rows) != EXPECTED_CADS or len({str(row["cad_id"]) for row in rows}) != EXPECTED_CADS:
        raise ValueError(f"expected {EXPECTED_CADS} unique original CADs, found {len(rows)}")
    for row in rows:
        source = Path(str(row.get("source_path")))
        if not source.is_file():
            raise FileNotFoundError(source)
    return rows


def historical_strict_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    result = {str(row["cad_id"]): bool(row.get("brep_valid")) for row in rows}
    if sum(result.values()) != EXPECTED_BASELINE_VALID:
        raise ValueError(
            f"historical strict baseline must be {EXPECTED_BASELINE_VALID}, got {sum(result.values())}"
        )
    return result


def profile_kwargs(profile: RepairProfile) -> dict[str, bool]:
    if profile.name == "baseline":
        return {
            "directed_trim": False, "curve_fit_fallback": False,
            "curve_fit_rescue": False, "curve_interpolate": False,
            "wire_continuity": False, "single_solid": False,
            "solid_topology_repair": False,
            "pcurve_self_intersection": False,
            "local_intersection_topology": False,
            "local_pcurve_continuity": False,
        }
    result = {name: profile.enabled(name) for name in (
        "directed_trim", "curve_fit_fallback", "curve_fit_rescue",
        "curve_interpolate",
        "wire_continuity", "single_solid",
        "pcurve_self_intersection", "local_intersection_topology",
        "local_pcurve_continuity",
    )}
    # The legacy switch name remains the external profile name.  It used to
    # enforce only the output count; it now additionally enables the narrow,
    # separately documented topology reconciliation before construction.
    near_vertex = profile.enabled("near_vertex_reconciliation")
    result["single_solid"] = bool(result["single_solid"] or near_vertex)
    result["solid_topology_repair"] = bool(
        profile.enabled("single_solid") or near_vertex
    )
    return result


def production_profile_topology_inputs(
    profile: RepairProfile,
    edge_wcs: np.ndarray,
    edge_vertex_adj: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply profile-controlled topology inputs before production construction.

    The production ``utils.construct_brep`` API has no explicit shared-vertex
    argument.  For the narrowly scoped near-vertex repair, remap only proven
    endpoint-id pairs and move the affected curve endpoints to the shared
    representative so the production wire builder sees matching endpoints.
    """
    edge_values = np.asarray(edge_wcs, dtype=np.float64)
    adjacency = np.asarray(edge_vertex_adj, dtype=np.int64)
    diagnostics: dict[str, Any] = {}
    if not profile.enabled("single_solid"):
        return edge_values, adjacency, diagnostics

    remapped, shared_vertices, near_vertex_diagnostics = reconcile_near_vertices(
        edge_values,
        adjacency,
        face_edge_adj,
    )
    repaired_edges = edge_values.copy()
    merged_roots = {
        int(cluster["root"])
        for cluster in near_vertex_diagnostics.get("clusters", [])
        if isinstance(cluster, Mapping)
    }
    endpoint_adjustments: list[float] = []
    if near_vertex_diagnostics.get("applied"):
        for edge_index, (start_vertex, end_vertex) in enumerate(remapped):
            for point_index, vertex_id in ((0, int(start_vertex)), (-1, int(end_vertex))):
                if vertex_id not in merged_roots:
                    continue
                replacement = np.asarray(shared_vertices[vertex_id], dtype=np.float64)
                endpoint_adjustments.append(
                    float(np.linalg.norm(repaired_edges[edge_index, point_index] - replacement))
                )
                repaired_edges[edge_index, point_index] = replacement
    diagnostics["solid_topology_repair"] = {
        **near_vertex_diagnostics,
        "production_endpoint_adjustment_count": len(endpoint_adjustments),
        "production_endpoint_adjustment_max": (
            max(endpoint_adjustments) if endpoint_adjustments else 0.0
        ),
    }
    return repaired_edges, remapped, diagnostics


def strict_validate_step(path: Path, *, breparg_root: Path) -> dict[str, Any]:
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils

    components = diagnose_step(path, breparg_root=root)
    native = bool(components.get("native_brep_valid"))
    strict = bool(brep_utils.check_brep_validity(str(path)))
    return {
        "native_brep_valid": native, "strict_brep_valid": strict,
        "both_valid": bool(native and strict), "validity_components": components,
    }


def run_one(
    source: Mapping[str, Any], profile: RepairProfile, *, output_dir: Path,
    breparg_root: Path, joint_iterations: int,
    assembly_backend: str = "directed",
    selector_geometry_gate: bool = False,
    expected_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cad_id = str(source["cad_id"])
    row: dict[str, Any] = {
        "schema": SCHEMA, "cad_id": cad_id, "parent_id": source.get("parent_id"),
        "profile": profile.name, "switches": list(profile.switches),
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_path": str(source["source_path"]), "step_saved": False,
        "native_brep_valid": False, "strict_brep_valid": False,
        "both_valid": False, "status": "running",
    }
    started = time.perf_counter()
    try:
        expected_binding = (
            normalize_source_pickle_binding(expected_source_binding)
            if expected_source_binding is not None
            else None
        )
        source_path = Path(str(source["source_path"]))
        source_bytes = source_path.read_bytes()
        actual_binding = source_pickle_binding_from_bytes(source_bytes)
        row["source_pickle_binding"] = actual_binding
        if expected_binding is not None and actual_binding != expected_binding:
            raise SourcePickleBindingMismatch(
                "source pickle binding mismatched before load"
            )
        # Deserialize the bytes that were just hashed rather than reopening the
        # path, so the reported binding names the exact input that OCC receives.
        parsed = pickle.loads(source_bytes)
        post_load_binding = source_pickle_binding(source_path)
        row["source_pickle_binding_after"] = post_load_binding
        if post_load_binding != actual_binding:
            raise SourcePickleBindingMismatch(
                "source pickle binding changed during load"
            )
        face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
        edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
        if assembly_backend == "production":
            root = Path(breparg_root).resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import utils as brep_utils

            if not callable(getattr(brep_utils, "resolve_edge_scale", None)):
                raise RuntimeError(
                    "production backend requires utils.resolve_edge_scale"
                )
            surf_wcs, edge_wcs = cpu_joint_optimize(
                np.asarray(parsed["surf_ncs"], dtype=np.float32),
                np.asarray(parsed["edge_ncs"], dtype=np.float32),
                np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
                np.asarray(parsed["corner_unique"], dtype=np.float32),
                edge_vertex_adj,
                face_edge_adj,
                iterations=joint_iterations,
                edge_bboxes=np.asarray(parsed["edge_bbox_wcs"], dtype=np.float32),
                edge_scale_resolver=brep_utils.resolve_edge_scale,
            )
            edge_wcs, production_edge_vertex_adj, topology_diagnostics = (
                production_profile_topology_inputs(
                    profile, edge_wcs, edge_vertex_adj, face_edge_adj
                )
            )
            solid = brep_utils.construct_brep(
                surf_wcs, edge_wcs, face_edge_adj, production_edge_vertex_adj
            )
            diagnostics = {
                "assembly_backend": "production",
                "utils_sha256": sha256_file(root / "utils.py"),
                **topology_diagnostics,
            }
        elif assembly_backend == "directed":
            surf_wcs, edge_wcs = cpu_joint_optimize(
                np.asarray(parsed["surf_ncs"], dtype=np.float32),
                np.asarray(parsed["edge_ncs"], dtype=np.float32),
                np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
                np.asarray(parsed["corner_unique"], dtype=np.float32),
                edge_vertex_adj, face_edge_adj, iterations=joint_iterations,
            )
            solid, diagnostics = construct_brep_directed(
                surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj,
                breparg_root=breparg_root, **profile_kwargs(profile),
            )
        else:
            raise ValueError(f"unknown assembly backend: {assembly_backend!r}")
        from OCC.Extend.DataExchange import write_step_file

        step_path = Path(output_dir) / "steps" / profile.name / f"{cad_id}.step"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        write_step_file(solid, str(step_path))
        if not step_path.is_file() or step_path.stat().st_size <= 0:
            raise RuntimeError("STEP writer produced no non-empty file")
        validity = strict_validate_step(step_path, breparg_root=breparg_root)
        row.update(
            status="both_valid" if validity["both_valid"] else "step_invalid",
            step_saved=True, step_path=str(step_path), step_bytes=step_path.stat().st_size,
            step_sha256=sha256_file(step_path), assembly_diagnostics=diagnostics, **validity,
        )
        if selector_geometry_gate and validity["both_valid"]:
            try:
                effective = diagnostics.get("effective_input_topology") or {}
                input_signature = input_geometry_signature(
                    surf_wcs,
                    edge_wcs,
                    face_edge_adj,
                    edge_vertex_adj,
                    effective_vertex_count=effective.get("vertex_count"),
                    effective_vertex_edge_incidence_counts=effective.get(
                        "vertex_edge_incidence_counts"
                    ),
                )
                candidate_signature = candidate_step_signature(
                    step_path,
                    input_edge_samples=sample_input_edge_points(edge_wcs),
                    input_edge_polylines=edge_wcs,
                    input_signature=input_signature,
                    validity_components=validity["validity_components"],
                )
                row["selector_geometry_topology_gate"] = geometry_topology_gate(
                    input_signature, candidate_signature
                )
            except Exception as exc:
                # This native OCC work deliberately happens in this child worker.
                # Its failure rejects only this candidate rather than losing the
                # entire selector denominator in the parent process.
                row["selector_geometry_topology_gate"] = {
                    "schema": GEOMETRY_GATE_SCHEMA,
                    "accepted": False,
                    "checks": {"geometry_measurement_completed": False},
                    "rejection_reasons": [
                        f"geometry_measurement_error:{type(exc).__name__}"
                    ],
                }
    except SourcePickleBindingMismatch as exc:
        row.update(
            status="source_binding_mismatch",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    except Exception as exc:
        row.update(
            status="assembly_error", error_type=type(exc).__name__, error=str(exc),
        )
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def parse_worker_result(stdout: str) -> dict[str, Any] | None:
    """Return the final sentinel row without trusting other OCC stdout."""
    for line in reversed(str(stdout).splitlines()):
        if line.startswith(WORKER_MARKER):
            try:
                row = json.loads(line[len(WORKER_MARKER):])
            except json.JSONDecodeError:
                return None
            return row if isinstance(row, dict) else None
    return None


def subprocess_output_text(value: str | bytes | None) -> str:
    """Normalize subprocess output, including TimeoutExpired byte payloads."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def validate_attempt_row(
    row: Mapping[str, Any], source: Mapping[str, Any], profile: RepairProfile
) -> None:
    """Bind one result to the requested CAD/profile and enforce validity logic."""
    required_values = {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_path": str(source["source_path"]),
    }
    for key, expected in required_values.items():
        if row.get(key) != expected:
            raise ValueError(
                f"worker result {key} mismatch: expected {expected!r}, got {row.get(key)!r}"
            )
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("worker result status must be a non-empty string")
    for key in (
        "step_saved",
        "native_brep_valid",
        "strict_brep_valid",
        "both_valid",
    ):
        if type(row.get(key)) is not bool:
            raise ValueError(f"worker result {key} must be a boolean")
    expected_both = bool(row["native_brep_valid"] and row["strict_brep_valid"])
    if row["both_valid"] is not expected_both:
        raise ValueError("worker result both_valid is inconsistent with native/strict validity")
    if (row["native_brep_valid"] or row["strict_brep_valid"]) and not row["step_saved"]:
        raise ValueError("worker result cannot be valid without a saved STEP")


def validate_existing_rows(
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    profiles: Sequence[RepairProfile],
) -> None:
    by_cad = {str(source["cad_id"]): source for source in sources}
    by_profile = {profile.name: profile for profile in profiles}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("profile")), str(row.get("cad_id")))
        if key in seen:
            raise RuntimeError(f"duplicate assembly repair attempt in output root: {key}")
        seen.add(key)
        if key[0] not in by_profile or key[1] not in by_cad:
            raise RuntimeError(f"assembly repair attempt escapes signed matrix: {key}")
        validate_attempt_row(row, by_cad[key[1]], by_profile[key[0]])


def worker_failure_row(
    source: Mapping[str, Any],
    profile: RepairProfile,
    *,
    status: str,
    returncode: int | None,
    stdout_log: Path,
    stderr_log: Path,
    error: str,
) -> dict[str, Any]:
    """Retain a native exit or timeout as one explicit denominator row."""
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_path": str(source["source_path"]),
        "status": str(status),
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
        "worker_returncode": returncode,
        "worker_stdout_log": str(stdout_log),
        "worker_stderr_log": str(stderr_log),
        "error_type": "WorkerProcessFailure",
        "error": str(error),
    }


def run_one_isolated(
    source: Mapping[str, Any],
    profile: RepairProfile,
    *,
    calibration_manifest: Path,
    output_dir: Path,
    breparg_root: Path,
    joint_iterations: int,
    timeout_seconds: float,
    assembly_backend: str = "directed",
    selector_geometry_gate: bool = False,
    expected_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one OCC attempt in a child so a native exit cannot lose the matrix."""
    cad_id = str(source["cad_id"])
    log_dir = Path(output_dir) / "worker_logs" / profile.name
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{cad_id}.stdout.log"
    stderr_log = log_dir / f"{cad_id}.stderr.log"
    attempt_dir = (
        Path(output_dir)
        / ".w"
        / uuid.uuid4().hex[:12]
    )
    expected_binding: dict[str, Any] | None = None
    if expected_source_binding is not None:
        try:
            expected_binding = normalize_source_pickle_binding(expected_source_binding)
            if source_pickle_binding(Path(str(source["source_path"]))) != expected_binding:
                raise SourcePickleBindingMismatch(
                    "parent source pickle binding mismatched before worker launch"
                )
        except (OSError, TypeError, ValueError, SourcePickleBindingMismatch) as exc:
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
            return worker_failure_row(
                source, profile, status="worker_protocol_error", returncode=None,
                stdout_log=stdout_log, stderr_log=stderr_log,
                error=f"source binding preflight failed: {type(exc).__name__}",
            )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--calibration-manifest", str(Path(calibration_manifest).resolve()),
        "--breparg-root", str(Path(breparg_root).resolve()),
        "--output-dir", str(attempt_dir.resolve()),
        "--joint-iterations", str(int(joint_iterations)),
        "--isolate-cad-workers",
        "--assembly-backend", str(assembly_backend),
        "--worker-profile", profile.name,
        "--worker-cad-id", cad_id,
    ]
    if expected_binding is not None:
        command.extend(
            [
                "--worker-source-binding-json",
                json.dumps(expected_binding, sort_keys=True, separators=(",", ":")),
            ]
        )
    if selector_geometry_gate:
        command.append("--selector-geometry-gate")
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = subprocess_output_text(exc.stdout)
        stderr = subprocess_output_text(exc.stderr)
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return worker_failure_row(
            source, profile, status="worker_timeout", returncode=None,
            stdout_log=stdout_log, stderr_log=stderr_log,
            error=f"worker exceeded {float(timeout_seconds):g} seconds",
        )
    except OSError as exc:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(str(exc) + "\n", encoding="utf-8")
        return worker_failure_row(
            source, profile, status="worker_spawn_error", returncode=None,
            stdout_log=stdout_log, stderr_log=stderr_log,
            error=f"worker could not be started: {type(exc).__name__}: {exc}",
        )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    row = parse_worker_result(completed.stdout)
    if completed.returncode != 0 or row is None:
        return worker_failure_row(
            source, profile, status="worker_process_exit",
            returncode=int(completed.returncode), stdout_log=stdout_log,
            stderr_log=stderr_log,
            error=(
                f"worker exited {completed.returncode} without a complete result"
                if row is None else f"worker exited {completed.returncode}"
            ),
        )
    if row.get("status") == "source_binding_mismatch":
        return worker_failure_row(
            source, profile, status="worker_protocol_error",
            returncode=int(completed.returncode), stdout_log=stdout_log,
            stderr_log=stderr_log,
            error="worker detected a source pickle binding mismatch",
        )
    try:
        validate_attempt_row(row, source, profile)
    except (TypeError, ValueError) as exc:
        return worker_failure_row(
            source, profile, status="worker_protocol_error",
            returncode=int(completed.returncode), stdout_log=stdout_log,
            stderr_log=stderr_log, error=str(exc),
        )
    if expected_binding is not None:
        try:
            worker_binding = normalize_source_pickle_binding(
                row["source_pickle_binding"]
            )
            worker_post_load_binding = normalize_source_pickle_binding(
                row["source_pickle_binding_after"]
            )
            parent_post_worker_binding = source_pickle_binding(
                Path(str(source["source_path"]))
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return worker_failure_row(
                source, profile, status="worker_protocol_error",
                returncode=int(completed.returncode), stdout_log=stdout_log,
                stderr_log=stderr_log,
                error=f"worker source binding evidence invalid: {type(exc).__name__}",
            )
        if (
            worker_binding != expected_binding
            or worker_post_load_binding != expected_binding
            or parent_post_worker_binding != expected_binding
        ):
            return worker_failure_row(
                source, profile, status="worker_protocol_error",
                returncode=int(completed.returncode), stdout_log=stdout_log,
                stderr_log=stderr_log,
                error="worker source pickle binding mismatched expected bytes",
            )
    if row["step_saved"]:
        staged_step = attempt_dir / "steps" / profile.name / f"{cad_id}.step"
        if (
            not staged_step.is_file()
            or staged_step.stat().st_size <= 0
            or row.get("step_sha256") != sha256_file(staged_step)
        ):
            return worker_failure_row(
                source, profile, status="worker_protocol_error",
                returncode=int(completed.returncode), stdout_log=stdout_log,
                stderr_log=stderr_log,
                error="worker STEP artifact is missing, empty, or hash-mismatched",
            )
        final_step = Path(output_dir) / "steps" / profile.name / f"{cad_id}.step"
        final_step.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_step, final_step)
        row["step_path"] = str(final_step)
        row["step_bytes"] = final_step.stat().st_size
    row["worker_returncode"] = int(completed.returncode)
    row["worker_stdout_log"] = str(stdout_log)
    row["worker_stderr_log"] = str(stderr_log)
    return row


def summarize_profile(
    rows: Sequence[Mapping[str, Any]], historical: Mapping[str, bool]
) -> dict[str, Any]:
    if len(rows) != len(historical):
        raise ValueError("profile does not cover the full frozen cohort")
    observed = {str(row["cad_id"]): bool(row.get("strict_brep_valid")) for row in rows}
    if set(observed) != set(historical):
        raise ValueError("profile CAD identities differ from the frozen cohort")
    restored = sorted(cad for cad, old in historical.items() if not old and observed[cad])
    regressed = sorted(cad for cad, old in historical.items() if old and not observed[cad])
    unchanged = sorted(cad for cad in historical if historical[cad] == observed[cad])
    strict_count = sum(observed.values())
    return {
        "profile": str(rows[0]["profile"]), "attempts": len(rows),
        "strict_valid": strict_count,
        "native_valid": sum(bool(row.get("native_brep_valid")) for row in rows),
        "both_valid": sum(bool(row.get("both_valid")) for row in rows),
        "step_readable": sum(bool(row.get("step_saved")) for row in rows),
        "restored_cad_ids": restored, "regressed_cad_ids": regressed,
        "unchanged_cad_ids": unchanged,
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "preserves_original_84": not regressed,
        "meets_95_gate": bool(strict_count >= 95 and not regressed),
    }


def summarize_matrix(
    rows: Sequence[Mapping[str, Any]], profiles: Sequence[RepairProfile],
    historical: Mapping[str, bool],
) -> dict[str, Any]:
    summaries = []
    for profile in profiles:
        profile_rows = [row for row in rows if row.get("profile") == profile.name]
        summaries.append(summarize_profile(profile_rows, historical))
    accepted = [item["profile"] for item in summaries if item["meets_95_gate"]]
    return {
        "schema": SCHEMA, "cohort_size": len(historical),
        "historical_strict_valid": sum(historical.values()),
        "profiles": summaries, "accepted_profiles": accepted,
        "gate_passed": bool(accepted), "advance_to_boundary_consistency": False,
        "advance_to_sequence_or_ar": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", default=None)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument(
        "--assembly-backend",
        choices=("directed", "production"),
        default="directed",
        help="Use the experimental directed assembler or the isolated production utils.py path.",
    )
    parser.add_argument("--max-cads", type=int, default=None)
    parser.add_argument("--isolate-cad-workers", action="store_true")
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-profile", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-cad-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-source-binding-json", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--selector-geometry-gate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--historical-invalid-only", action="store_true",
        help="Development-only pilot on the 16 historical failures; cannot pass the formal gate.",
    )
    args = parser.parse_args(argv)

    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if args.assembly_backend == "production" and not args.isolate_cad_workers:
        parser.error(
            "production backend requires --isolate-cad-workers to contain OCC native exits"
        )
    full_source_rows = frozen_original_rows(args.calibration_manifest)
    if bool(args.worker_profile) != bool(args.worker_cad_id):
        parser.error("--worker-profile and --worker-cad-id must be provided together")
    if args.worker_source_binding_json is not None and not args.worker_profile:
        parser.error("--worker-source-binding-json is valid only for a worker")
    if args.worker_profile:
        expected_source_binding = None
        if args.worker_source_binding_json is not None:
            try:
                decoded_source_binding = json.loads(args.worker_source_binding_json)
                expected_source_binding = normalize_source_pickle_binding(
                    decoded_source_binding
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                parser.error("--worker-source-binding-json must be a valid source binding")
        profile = parse_profiles([args.worker_profile])[0]
        selected = [
            row for row in full_source_rows
            if str(row["cad_id"]) == args.worker_cad_id
        ]
        if len(selected) != 1:
            parser.error(f"worker CAD id is not unique in frozen cohort: {args.worker_cad_id}")
        row = run_one(
            selected[0], profile, output_dir=args.output_dir,
            breparg_root=args.breparg_root,
            joint_iterations=args.joint_iterations,
            assembly_backend=args.assembly_backend,
            selector_geometry_gate=bool(args.selector_geometry_gate),
            expected_source_binding=expected_source_binding,
        )
        print(WORKER_MARKER + json.dumps(row, sort_keys=True), flush=True)
        return 0

    historical = historical_strict_map(full_source_rows)
    profiles = parse_profiles(args.profile)
    if any(requires_isolated_worker(profile) for profile in profiles):
        if not args.isolate_cad_workers:
            parser.error(
                "local face repair profiles require --isolate-cad-workers"
            )
    source_rows = list(full_source_rows)
    if args.historical_invalid_only:
        source_rows = [row for row in source_rows if not row.get("brep_valid")]
        historical = {str(row["cad_id"]): False for row in source_rows}
    if args.max_cads is not None:
        source_rows = source_rows[: args.max_cads]
        historical = {str(row["cad_id"]): bool(row.get("brep_valid")) for row in source_rows}
    run_payload = build_run_payload(
        args=args,
        full_rows=full_source_rows,
        selected_rows=source_rows,
        profiles=profiles,
    )
    lock_command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    with output_root_writer_lock(args.output_dir, command=lock_command):
        run_record = bind_run_manifest(args.output_dir, run_payload)
        try:
            manifest_path = args.output_dir / "assembly_repair_matrix.jsonl"
            rows = read_jsonl(manifest_path) if manifest_path.is_file() else []
            validate_existing_rows(rows, source_rows, profiles)
            done = {(str(row["profile"]), str(row["cad_id"])) for row in rows}
            for profile in profiles:
                for source in source_rows:
                    key = (profile.name, str(source["cad_id"]))
                    if key in done:
                        continue
                    if (
                        args.assembly_backend == "production"
                        or requires_isolated_worker(profile)
                    ):
                        row = run_one_isolated(
                            source, profile,
                            calibration_manifest=args.calibration_manifest,
                            output_dir=args.output_dir,
                            breparg_root=args.breparg_root,
                            joint_iterations=args.joint_iterations,
                            timeout_seconds=args.worker_timeout_seconds,
                            assembly_backend=args.assembly_backend,
                        )
                    else:
                        row = run_one(
                            source, profile, output_dir=args.output_dir,
                            breparg_root=args.breparg_root,
                            joint_iterations=args.joint_iterations,
                            assembly_backend=args.assembly_backend,
                        )
                    validate_attempt_row(row, source, profile)
                    append_jsonl(manifest_path, row)
                    rows.append(row)
                    done.add(key)
                    print(
                        json.dumps(
                            {
                                name: row.get(name)
                                for name in (
                                    "profile", "cad_id", "status",
                                    "strict_brep_valid", "error",
                                )
                            }
                        ),
                        flush=True,
                    )
            summary = summarize_matrix(rows, profiles, historical)
            summary_path = args.output_dir / "assembly_repair_summary.json"
            atomic_json(summary_path, summary)
            run_record.update(
                status=(
                    "COMPLETED_PARTIAL"
                    if args.max_cads is not None or args.historical_invalid_only
                    else "COMPLETED"
                ),
                attempts=len(rows),
                summary_sha256=sha256_file(summary_path),
            )
            atomic_json(args.output_dir / RUN_MANIFEST_NAME, run_record)
        except Exception as exc:
            run_record.update(
                status="FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            atomic_json(args.output_dir / RUN_MANIFEST_NAME, run_record)
            raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    partial = args.max_cads is not None or args.historical_invalid_only
    return 0 if summary["gate_passed"] or partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
