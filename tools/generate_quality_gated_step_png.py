"""Generate V13 candidates and retain only quality-gated STEP/PNG outputs."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"
TOOLS_DIR = REPO_ROOT / "tools"
for item in (REPO_ROOT, IMPROVEMENTS_DIR, BREPARG_DIR, TOOLS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_reconstruction_v13 import (
    append_jsonl,
    infer_device,
    load_ar_model,
    load_fsq_vqvae,
    normalize_vocab_info,
    read_pickle,
    reconstruct_one,
    write_json,
)
from generation_quality_gate import quality_gate_decision
from run_deep_root_cause_experiment import ArItem, generate_ar_records_batch, make_contact_sheet


def run_quality_check(row: dict[str, Any], output_dir: Path, timeout_sec: int) -> dict[str, Any]:
    if not row.get("step_saved") or not row.get("step_path"):
        return {"png_saved": False, "brep_valid": False, "solid_closed_no_open_shell": False}
    step_path = Path(row["step_path"])
    stl_path = output_dir / "stl" / f"{step_path.stem}.stl"
    png_path = output_dir / "png" / f"{step_path.stem}.png"
    title = f"f={row.get('grammar_faces', '?')} e={row.get('grammar_edges', '?')} len={row.get('sequence_length', '?')}"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "validate_step_quality_once.py"),
        "--step",
        str(step_path),
        "--stl",
        str(stl_path),
        "--png",
        str(png_path),
        "--title",
        title,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        return {
            "brep_valid": False,
            "solid_closed_no_open_shell": False,
            "png_saved": png_path.exists() and png_path.stat().st_size > 0,
            "png_path": str(png_path),
            "stl_saved": stl_path.exists() and stl_path.stat().st_size > 0,
            "stl_path": str(stl_path),
            "quality_error": f"quality_timeout_{timeout_sec}s",
        }

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {}
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not payload:
        payload = {
            "brep_valid": False,
            "solid_closed_no_open_shell": False,
            "png_saved": False,
            "quality_error": (completed.stderr or completed.stdout or f"quality_exit_{completed.returncode}")[-500:],
        }
    return payload


def move_candidate_files(row: dict[str, Any], quality: dict[str, Any], accepted_dir: Path, accepted_index: int) -> dict[str, Any]:
    stem = f"accepted_{accepted_index:04d}_f{int(row.get('grammar_faces', 0)):02d}_e{int(row.get('grammar_edges', 0)):03d}"
    for key, subdir, suffix in (
        ("step_path", "steps", ".step"),
        ("stl_path", "stl", ".stl"),
        ("png_path", "png", ".png"),
    ):
        raw = quality.get(key) or row.get(key)
        if not raw:
            continue
        src = Path(raw)
        if not src.exists():
            continue
        dst = accepted_dir / subdir / f"{stem}{suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row[key] = str(dst)
        if key in quality:
            quality[key] = str(dst)
    return row


def cleanup_rejected_candidate(row: dict[str, Any], quality: dict[str, Any]) -> None:
    for raw in (row.get("step_path"), quality.get("stl_path"), quality.get("png_path")):
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            path.unlink()


def summarize(rows: list[dict[str, Any]], accepted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in row.get("gate_reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "attempted": len(rows),
        "accepted": len(accepted_rows),
        "rejected": len(rows) - len(accepted_rows),
        "step_saved": sum(1 for row in rows if row.get("step_saved")),
        "png_saved": sum(1 for row in rows if row.get("png_saved")),
        "brep_valid": sum(1 for row in rows if row.get("brep_valid")),
        "gate_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2047)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--constraint-min-faces", type=int, default=12)
    parser.add_argument("--use-bbox-monotonic", action="store_true")
    parser.add_argument("--no-face-unique", action="store_true")
    parser.add_argument("--gate-min-faces", type=int, default=12)
    parser.add_argument("--gate-min-edges", type=int, default=20)
    parser.add_argument("--gate-max-faces", type=int, default=45)
    parser.add_argument("--gate-max-edges", type=int, default=120)
    parser.add_argument(
        "--require-both-min-topology",
        action="store_true",
        help="Reject a candidate unless both grammar face count and edge count meet the configured minima.",
    )
    parser.add_argument(
        "--allow-primitive-like",
        action="store_true",
        help="Do not reject simple primitive-like topology when the face/edge gate allows it.",
    )
    parser.add_argument("--quality-timeout-sec", type=int, default=30)
    parser.add_argument("--keep-rejected-files", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    package = read_pickle(args.sequence)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    output_dir = args.output_dir
    work_dir = output_dir / "_work"
    accepted_dir = output_dir / "accepted"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "quality_gated_manifest.jsonl"
    accepted_manifest_path = output_dir / "accepted_manifest.jsonl"

    model = load_ar_model(args.ar_checkpoint, vocab_info, device)
    vqvae_model = load_fsq_vqvae(args.vqvae_checkpoint, device)
    item = ArItem(
        name="quality_gated",
        constrained=True,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        use_bbox_monotonic=bool(args.use_bbox_monotonic),
        enforce_face_unique=not bool(args.no_face_unique),
        min_faces=int(args.constraint_min_faces),
    )

    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    attempts = 0
    started = time.time()
    while len(accepted_rows) < args.target_count and attempts < args.max_attempts:
        batch = min(args.batch_size, args.max_attempts - attempts)
        records = generate_ar_records_batch(
            model=model,
            vocab_info=vocab_info,
            item=item,
            batch_size=batch,
            max_new_tokens=args.max_new_tokens,
            device=device,
            start_index=attempts,
        )
        for record in records:
            attempts += 1
            record["index"] = attempts - 1
            row = reconstruct_one(
                record,
                vocab_info=vocab_info,
                vqvae_model=vqvae_model,
                device=device,
                output_dir=work_dir,
                write_step=True,
                write_stl=False,
                validate_step=False,
                scale_factor=1.0,
            )
            quality = run_quality_check(row, work_dir, timeout_sec=args.quality_timeout_sec)
            row.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
                "constraint_min_faces": args.constraint_min_faces,
                "brep_valid": bool(quality.get("brep_valid")),
                "solid_closed_no_open_shell": bool(quality.get("solid_closed_no_open_shell")),
                "advanced_faces": int(quality.get("advanced_faces", 0) or 0),
                "edge_curves": int(quality.get("edge_curves", 0) or 0),
                "stl_saved": bool(quality.get("stl_saved")),
                "stl_path": quality.get("stl_path"),
                "png_saved": bool(quality.get("png_saved")),
                "png_path": quality.get("png_path"),
            })
            decision = quality_gate_decision(
                row,
                quality,
                min_faces=args.gate_min_faces,
                min_edges=args.gate_min_edges,
                max_faces=args.gate_max_faces,
                max_edges=args.gate_max_edges,
                reject_primitive_like=not bool(args.allow_primitive_like),
                require_both_min_topology=bool(args.require_both_min_topology),
            )
            row["accepted"] = bool(decision["accept"])
            row["gate_reasons"] = decision["reasons"]
            if row["accepted"]:
                row = move_candidate_files(row, quality, accepted_dir, len(accepted_rows))
                accepted_rows.append(row)
                append_jsonl(accepted_manifest_path, row)
            elif not args.keep_rejected_files:
                cleanup_rejected_candidate(row, quality)
            rows.append(row)
            append_jsonl(manifest_path, row)
            print(
                f"[{attempts:04d}] accept={int(row['accepted'])} "
                f"accepted={len(accepted_rows)}/{args.target_count} "
                f"status={row.get('status')} f={row.get('grammar_faces')} e={row.get('grammar_edges')} "
                f"valid={int(row.get('brep_valid', False))} reasons={','.join(row['gate_reasons']) or '-'}",
                flush=True,
            )
            if len(accepted_rows) >= args.target_count or attempts >= args.max_attempts:
                break

    contact_sheet = make_contact_sheet(accepted_dir, accepted_rows, "quality_gated", args.target_count)
    report = {
        "status": "VERIFIED" if len(accepted_rows) >= args.target_count else "INCOMPLETE",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
        "sequence": str(args.sequence),
        "ar_checkpoint": str(args.ar_checkpoint),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "output_dir": str(output_dir),
        "accepted_dir": str(accepted_dir),
        "manifest": str(manifest_path),
        "accepted_manifest": str(accepted_manifest_path),
        "contact_sheet": contact_sheet,
        "config": jsonable_config(args),
        "summary": summarize(rows, accepted_rows),
    }
    write_json(output_dir / "quality_gated_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
    return 0 if report["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
