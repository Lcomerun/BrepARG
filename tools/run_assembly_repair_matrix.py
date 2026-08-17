"""Run independent assembly-repair profiles on the frozen 100-CAD cohort.

The runner keeps every attempt in the denominator, writes STEP files only to a
local output root, and emits a machine-readable restored/regressed map.  A
profile passes only when it preserves all historically strict-valid controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import RepairProfile, parse_profiles
    from .diagnose_step_validity_components import diagnose_step
    from .directed_trim_assembly import construct_brep_directed
    from .run_assembly_calibration_oracle import cpu_joint_optimize
except ImportError:  # direct script execution
    from assembly_repair import RepairProfile, parse_profiles
    from diagnose_step_validity_components import diagnose_step
    from directed_trim_assembly import construct_brep_directed
    from run_assembly_calibration_oracle import cpu_joint_optimize


SCHEMA = "assembly-repair-matrix-v1"
EXPECTED_CADS = 100
EXPECTED_BASELINE_VALID = 84


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
            "wire_continuity": False, "single_solid": False,
            "pcurve_self_intersection": False,
        }
    return {name: profile.enabled(name) for name in (
        "directed_trim", "curve_fit_fallback", "wire_continuity", "single_solid",
        "pcurve_self_intersection",
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
    parser.add_argument(
        "--historical-invalid-only", action="store_true",
        help="Development-only pilot on the 16 historical failures; cannot pass the formal gate.",
    )
    args = parser.parse_args(argv)

    source_rows = frozen_original_rows(args.calibration_manifest)
    historical = historical_strict_map(source_rows)
    profiles = parse_profiles(args.profile)
    if args.historical_invalid_only:
        source_rows = [row for row in source_rows if not row.get("brep_valid")]
        historical = {str(row["cad_id"]): False for row in source_rows}
    if args.max_cads is not None:
        source_rows = source_rows[: args.max_cads]
        historical = {str(row["cad_id"]): bool(row.get("brep_valid")) for row in source_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "assembly_repair_matrix.jsonl"
    rows = read_jsonl(manifest_path) if manifest_path.is_file() else []
    done = {(str(row.get("profile")), str(row.get("cad_id"))) for row in rows}
    for profile in profiles:
        for source in source_rows:
            key = (profile.name, str(source["cad_id"]))
            if key in done:
                continue
            row = run_one(
                source, profile, output_dir=args.output_dir, breparg_root=args.breparg_root,
                joint_iterations=args.joint_iterations,
            )
            append_jsonl(manifest_path, row)
            rows.append(row)
            done.add(key)
            print(json.dumps({key: row.get(key) for key in ("profile", "cad_id", "status", "strict_brep_valid", "error")}), flush=True)
    summary = summarize_matrix(rows, profiles, historical)
    (args.output_dir / "assembly_repair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    partial = args.max_cads is not None or args.historical_invalid_only
    return 0 if summary["gate_passed"] or partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
