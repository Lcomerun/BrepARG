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
    from .diagnose_step_validity_components import diagnose_step
    from .directed_trim_assembly import construct_brep_directed
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .run_p0b_stability_retest import (
        atomic_json,
        canonical_signature,
        output_root_writer_lock,
    )
except ImportError:  # direct script execution
    from assembly_repair import RepairProfile, parse_profiles
    from diagnose_step_validity_components import diagnose_step
    from directed_trim_assembly import construct_brep_directed
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from run_p0b_stability_retest import (
        atomic_json,
        canonical_signature,
        output_root_writer_lock,
    )


SCHEMA = "assembly-repair-matrix-v1"
RUN_SCHEMA = "assembly-repair-run-v2"
RUN_MANIFEST_NAME = "assembly_repair_run.json"
EXPECTED_CADS = 100
EXPECTED_BASELINE_VALID = 84
WORKER_MARKER = "__ASSEMBLY_REPAIR_WORKER_RESULT__="
ISOLATED_WORKER_SWITCHES = frozenset(
    {"local_intersection_topology", "local_pcurve_continuity"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()


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
            "curve_fit_rescue": False,
            "wire_continuity": False, "single_solid": False,
            "pcurve_self_intersection": False,
            "local_intersection_topology": False,
            "local_pcurve_continuity": False,
        }
    return {name: profile.enabled(name) for name in (
        "directed_trim", "curve_fit_fallback", "curve_fit_rescue",
        "wire_continuity", "single_solid", "pcurve_self_intersection",
        "local_intersection_topology",
        "local_pcurve_continuity",
    )}


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
        with Path(str(source["source_path"])).open("rb") as handle:
            parsed = pickle.load(handle)
        face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
        edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
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
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--calibration-manifest", str(Path(calibration_manifest).resolve()),
        "--breparg-root", str(Path(breparg_root).resolve()),
        "--output-dir", str(attempt_dir.resolve()),
        "--joint-iterations", str(int(joint_iterations)),
        "--worker-profile", profile.name,
        "--worker-cad-id", cad_id,
    ]
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
    try:
        validate_attempt_row(row, source, profile)
    except (TypeError, ValueError) as exc:
        return worker_failure_row(
            source, profile, status="worker_protocol_error",
            returncode=int(completed.returncode), stdout_log=stdout_log,
            stderr_log=stderr_log, error=str(exc),
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
    parser.add_argument("--max-cads", type=int, default=None)
    parser.add_argument("--isolate-cad-workers", action="store_true")
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-profile", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-cad-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--historical-invalid-only", action="store_true",
        help="Development-only pilot on the 16 historical failures; cannot pass the formal gate.",
    )
    args = parser.parse_args(argv)

    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    full_source_rows = frozen_original_rows(args.calibration_manifest)
    if bool(args.worker_profile) != bool(args.worker_cad_id):
        parser.error("--worker-profile and --worker-cad-id must be provided together")
    if args.worker_profile:
        profile = parse_profiles([args.worker_profile])[0]
        selected = [
            row for row in full_source_rows
            if str(row["cad_id"]) == args.worker_cad_id
        ]
        if len(selected) != 1:
            parser.error(f"worker CAD id is not unique in frozen cohort: {args.worker_cad_id}")
        row = run_one(
            selected[0], profile, output_dir=args.output_dir,
            breparg_root=args.breparg_root, joint_iterations=args.joint_iterations,
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
                    if requires_isolated_worker(profile):
                        row = run_one_isolated(
                            source, profile,
                            calibration_manifest=args.calibration_manifest,
                            output_dir=args.output_dir,
                            breparg_root=args.breparg_root,
                            joint_iterations=args.joint_iterations,
                            timeout_seconds=args.worker_timeout_seconds,
                        )
                    else:
                        row = run_one(
                            source, profile, output_dir=args.output_dir,
                            breparg_root=args.breparg_root,
                            joint_iterations=args.joint_iterations,
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
