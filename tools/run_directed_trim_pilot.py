"""Reassemble a frozen calibration subset with topology-directed trim loops."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .audit_assembly_step_validity import summarize_validity_rows
    from .directed_trim_assembly import construct_brep_directed
    from .run_assembly_calibration_oracle import (
        cpu_joint_optimize,
        load_checkpoint_model,
        reconstruct_patch_batch,
        sha256_file,
    )
except ImportError:  # direct script execution
    from audit_assembly_step_validity import summarize_validity_rows
    from directed_trim_assembly import construct_brep_directed
    from run_assembly_calibration_oracle import (
        cpu_joint_optimize,
        load_checkpoint_model,
        reconstruct_patch_batch,
        sha256_file,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_cad_ids(audit_rows: Sequence[dict[str, Any]], *, reference_arm: str) -> list[str]:
    return sorted({
        str(row["cad_id"])
        for row in audit_rows
        if str(row.get("arm")) == reference_arm
        and row.get("native_brep_valid") is not None
        and row.get("strict_brep_valid") is False
    })


def parse_checkpoint(value: str) -> tuple[str, Path]:
    arm, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("checkpoint must use ARM=PATH")
    return arm.strip(), Path(path.strip())


def validate_written_step(path: Path, *, breparg_root: Path) -> tuple[bool, bool]:
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        return False, False
    reader.TransferRoots()
    native = bool(BRepCheck_Analyzer(reader.OneShape(), True).IsValid())
    return native, bool(brep_utils.check_brep_validity(str(path)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-arm", default="original")
    parser.add_argument("--arm", action="append", choices=("original", "continuous_bypass_64d", "fsq_8192_4d"), required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--joint-iterations", type=int, default=200)
    args = parser.parse_args(argv)

    import torch
    from OCC.Extend.DataExchange import write_step_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "directed_trim_manifest.jsonl"
    audit_rows = read_jsonl(args.audit_manifest)
    cad_ids = selected_cad_ids(audit_rows, reference_arm=args.reference_arm)
    calibration_rows = read_jsonl(args.calibration_manifest)
    source_by_id = {
        str(row["cad_id"]): row for row in calibration_rows if row.get("arm") == "original"
    }
    missing = sorted(set(cad_ids) - set(source_by_id))
    if missing:
        raise RuntimeError(f"selected CADs missing source rows: {missing}")
    checkpoints = dict(args.checkpoint)
    models = {}
    metadata = {}
    for arm in args.arm:
        if arm == "original":
            continue
        if arm not in checkpoints:
            raise RuntimeError(f"checkpoint missing for arm {arm}")
        models[arm], metadata[arm] = load_checkpoint_model(
            arm, checkpoints[arm], device=args.device,
            import_output_root=args.output_dir / "import_state",
        )
    existing = read_jsonl(manifest_path) if manifest_path.is_file() else []
    done = {(str(row.get("cad_id")), str(row.get("arm"))) for row in existing}
    for cad_id in cad_ids:
        source = source_by_id[cad_id]
        with Path(source["source_path"]).open("rb") as handle:
            parsed = pickle.load(handle)
        for arm in args.arm:
            if (cad_id, arm) in done:
                continue
            started = time.time()
            row: dict[str, Any] = {
                "cad_id": cad_id, "parent_id": source.get("parent_id"), "arm": arm,
                "source_path": source.get("source_path"), "status": "pending",
                "step_saved": False, "native_brep_valid": None, "strict_brep_valid": False,
                "checkpoint_sha256": metadata.get(arm, {}).get("sha256"),
            }
            try:
                surfaces = np.asarray(parsed["surf_ncs"], dtype=np.float32)
                edges = np.asarray(parsed["edge_ncs"], dtype=np.float32)
                if arm != "original":
                    surfaces = reconstruct_patch_batch(
                        models[arm], surfaces, kind="surface", device=args.device,
                        batch_size=args.batch_size,
                    )
                    edges = reconstruct_patch_batch(
                        models[arm], edges, kind="edge", device=args.device,
                        batch_size=args.batch_size,
                    )
                face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
                edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
                surf_wcs, edge_wcs = cpu_joint_optimize(
                    surfaces, edges, np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
                    np.asarray(parsed["corner_unique"], dtype=np.float32), edge_vertex_adj,
                    face_edge_adj, iterations=args.joint_iterations,
                )
                solid, diagnostics = construct_brep_directed(
                    surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj,
                    breparg_root=args.breparg_root,
                )
                step_path = args.output_dir / "steps" / arm / f"{cad_id}.step"
                step_path.parent.mkdir(parents=True, exist_ok=True)
                write_step_file(solid, str(step_path))
                saved = step_path.is_file() and step_path.stat().st_size > 0
                native, strict = validate_written_step(step_path, breparg_root=args.breparg_root) if saved else (False, False)
                row.update({
                    "status": "saved" if strict else ("brep_invalid" if saved else "step_save_failed"),
                    "step_saved": saved, "native_brep_valid": native if saved else None,
                    "strict_brep_valid": strict, "step_path": str(step_path) if saved else None,
                    "step_bytes": step_path.stat().st_size if saved else 0,
                    "step_sha256": sha256_file(step_path) if saved else None,
                    "assembly_diagnostics": diagnostics,
                })
            except Exception as exc:
                row.update(status="assembly_error", error_type=type(exc).__name__, error=str(exc))
            row["elapsed_seconds"] = time.time() - started
            with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            done.add((cad_id, arm))
            print(json.dumps({key: row.get(key) for key in ("cad_id", "arm", "status", "native_brep_valid", "strict_brep_valid", "error")}), flush=True)
    rows = read_jsonl(manifest_path)
    summary = summarize_validity_rows(rows)
    summary.update({
        "selected_cad_ids": cad_ids, "reference_arm": args.reference_arm,
        "advance_to_vq_300k": False, "advance_to_ar": False,
    })
    (args.output_dir / "directed_trim_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
