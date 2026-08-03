"""Generate STEP/PNG files with the original BrepARG sampling logic.

This is a comparison runner: it uses BrepARG.generate_brep.generate_sequence
without the newer constrained decoding or quality gate, then reconstructs with
the current FSQ VQ-VAE checkpoint.
"""

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
BREPARG_DIR = REPO_ROOT / "BrepARG"
TOOLS_DIR = REPO_ROOT / "tools"
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
for item in (BREPARG_DIR, TOOLS_DIR, IMPROVEMENTS_DIR, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_reconstruction_v13 import (  # noqa: E402
    append_jsonl,
    infer_device,
    load_ar_model,
    load_fsq_vqvae,
    normalize_vocab_info,
    read_pickle,
    reconstruct_one,
    write_json,
)
from generate_brep import generate_sequence, load_config, load_model_config  # noqa: E402
from run_deep_root_cause_experiment import enrich_preview_subprocess, make_contact_sheet  # noqa: E402
from generation_quality_gate import quality_gate_decision  # noqa: E402


def summarize_rows(rows: list[dict[str, Any]], target_count: int) -> dict[str, Any]:
    saved = [row for row in rows if row.get("step_saved") and row.get("png_saved")]
    accepted = [row for row in rows if row.get("accepted")]
    grammar_ok = [row for row in rows if row.get("grammar_ok")]
    reasons: dict[str, int] = {}
    statuses: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    for row in rows:
        statuses[str(row.get("status", "unknown"))] = statuses.get(str(row.get("status", "unknown")), 0) + 1
        if not row.get("grammar_ok"):
            reason = str(row.get("grammar_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason in row.get("gate_reasons") or []:
            gate_reasons[str(reason)] = gate_reasons.get(str(reason), 0) + 1

    def stats(key: str) -> dict[str, int | None]:
        values = [int(row.get(key, 0) or 0) for row in grammar_ok]
        if not values:
            return {"min": None, "median": None, "max": None}
        ordered = sorted(values)
        return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2], "max": ordered[-1]}

    return {
        "target_count": int(target_count),
        "attempted": len(rows),
        "grammar_ok": len(grammar_ok),
        "step_saved": sum(1 for row in rows if row.get("step_saved")),
        "png_saved": sum(1 for row in rows if row.get("png_saved")),
        "accepted_visual": len(saved),
        "quality_accepted": len(accepted),
        "brep_valid": sum(1 for row in rows if row.get("brep_valid")),
        "solid_closed_no_open_shell": sum(1 for row in rows if row.get("solid_closed_no_open_shell")),
        "complex_grammar": sum(
            1 for row in grammar_ok if int(row.get("grammar_faces", 0) or 0) >= 12 or int(row.get("grammar_edges", 0) or 0) >= 20
        ),
        "faces": stats("grammar_faces"),
        "edges": stats("grammar_edges"),
        "length": stats("sequence_length"),
        "status_counts": dict(sorted(statuses.items())),
        "grammar_fail_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "gate_reasons": dict(sorted(gate_reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


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

    payload: dict[str, Any] = {}
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
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


def copy_accepted_files(row: dict[str, Any], accepted_dir: Path, accepted_index: int) -> dict[str, Any]:
    stem = f"accepted_{accepted_index:04d}_f{int(row.get('grammar_faces', 0) or 0):02d}_e{int(row.get('grammar_edges', 0) or 0):03d}"
    for key, subdir, suffix in (
        ("step_path", "steps", ".step"),
        ("stl_path", "stl", ".stl"),
        ("png_path", "png", ".png"),
    ):
        raw = row.get(key)
        if not raw:
            continue
        src = Path(raw)
        if not src.exists():
            continue
        dst = accepted_dir / subdir / f"{stem}{suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row[f"accepted_{key}"] = str(dst)
    return row


def cleanup_rejected_files(row: dict[str, Any]) -> None:
    for raw in (row.get("step_path"), row.get("stl_path"), row.get("png_path")):
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=BREPARG_DIR / "config.json")
    parser.add_argument("--dataset-type", choices=["abc", "deepcad"], default="abc")
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--preview-timeout-sec", type=int, default=10)
    parser.add_argument("--quality-gate", action="store_true", help="Validate each STEP and copy only accepted samples into accepted/.")
    parser.add_argument("--quality-timeout-sec", type=int, default=30)
    parser.add_argument("--gate-min-faces", type=int, default=0)
    parser.add_argument("--gate-min-edges", type=int, default=0)
    parser.add_argument("--gate-max-faces", type=int, default=50)
    parser.add_argument("--gate-max-edges", type=int, default=150)
    parser.add_argument(
        "--require-both-min-topology",
        action="store_true",
        help="Reject a candidate unless both grammar face count and edge count meet the configured minima.",
    )
    parser.add_argument("--reject-primitive-like", action="store_true")
    parser.add_argument("--keep-rejected-files", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    package = read_pickle(args.sequence)
    package_vocab = normalize_vocab_info(package)
    config_vocab = load_model_config(load_config(str(args.config)), args.dataset_type)
    vocab_info = dict(config_vocab)
    vocab_info["se_tokens_per_element"] = int(package.get("se_tokens_per_element", package_vocab.get("se_tokens_per_element", 4)))
    vocab_info["bbox_tokens_per_element"] = int(package.get("bbox_tokens_per_element", package_vocab.get("bbox_tokens_per_element", 6)))

    device = infer_device(args.device)
    ar_model = load_ar_model(args.ar_checkpoint, vocab_info, device)
    vqvae_model = load_fsq_vqvae(args.vqvae_checkpoint, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = output_dir / "accepted"
    if args.quality_gate:
        accepted_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "breparg_logic_manifest.jsonl"
    accepted_manifest_path = output_dir / "accepted_manifest.jsonl"
    report_path = output_dir / "breparg_logic_report.json"

    rows: list[dict[str, Any]] = []
    saved_count = 0
    accepted_count = 0
    started = time.time()
    while (accepted_count if args.quality_gate else saved_count) < args.target_count and len(rows) < args.max_attempts:
        attempt = len(rows)
        sequence = generate_sequence(
            model=ar_model,
            vocab_info=vocab_info,
            device=str(device),
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        record = {
            "source": "breparg_logic",
            "split": "generated",
            "index": attempt,
            "length": len(sequence),
            "sequence": sequence,
        }
        row = reconstruct_one(
            record,
            vocab_info=vocab_info,
            vqvae_model=vqvae_model,
            device=device,
            output_dir=output_dir,
            write_step=True,
            write_stl=False,
            validate_step=False,
            scale_factor=1.0,
        )
        row.update(
            {
                "attempt": attempt,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "sampling_logic": "BrepARG.generate_brep.generate_sequence",
            }
        )
        if args.quality_gate:
            quality = run_quality_check(row, output_dir, timeout_sec=args.quality_timeout_sec)
            row.update(
                {
                    "step_read_ok": bool(quality.get("step_read_ok")),
                    "brep_valid": bool(quality.get("brep_valid")),
                    "solid_closed_no_open_shell": bool(quality.get("solid_closed_no_open_shell")),
                    "advanced_faces": int(quality.get("advanced_faces", 0) or 0),
                    "edge_curves": int(quality.get("edge_curves", 0) or 0),
                    "stl_saved": bool(quality.get("stl_saved")),
                    "stl_path": quality.get("stl_path"),
                    "png_saved": bool(quality.get("png_saved")),
                    "png_path": quality.get("png_path"),
                    "quality_error": quality.get("quality_error"),
                }
            )
            decision = quality_gate_decision(
                row,
                quality,
                min_faces=args.gate_min_faces,
                min_edges=args.gate_min_edges,
                max_faces=args.gate_max_faces,
                max_edges=args.gate_max_edges,
                reject_primitive_like=bool(args.reject_primitive_like),
                require_both_min_topology=bool(args.require_both_min_topology),
            )
            row["accepted"] = bool(decision["accept"])
            row["gate_reasons"] = decision["reasons"]
            if row["accepted"]:
                row = copy_accepted_files(row, accepted_dir, accepted_count)
                accepted_count += 1
                append_jsonl(accepted_manifest_path, row)
            elif not args.keep_rejected_files:
                cleanup_rejected_files(row)
        else:
            row = enrich_preview_subprocess(row, output_dir, timeout_sec=args.preview_timeout_sec)
            row["accepted"] = bool(row.get("step_saved") and row.get("png_saved"))
            row["gate_reasons"] = []
        rows.append(row)
        append_jsonl(manifest_path, row)
        if row.get("step_saved") and row.get("png_saved"):
            saved_count += 1
        print(
            f"[{attempt + 1:04d}] saved={saved_count} accepted={accepted_count if args.quality_gate else saved_count}/{args.target_count} "
            f"status={row.get('status')} grammar={int(bool(row.get('grammar_ok')))} "
            f"f={row.get('grammar_faces')} e={row.get('grammar_edges')} len={row.get('sequence_length')} "
            f"reasons={','.join(row.get('gate_reasons') or []) or '-'}",
            flush=True,
        )

    contact_rows = rows
    contact_dir = output_dir
    if args.quality_gate:
        contact_rows = [
            {
                **row,
                "png_saved": bool(row.get("accepted_png_path")),
                "png_path": row.get("accepted_png_path"),
            }
            for row in rows
            if row.get("accepted_png_path")
        ]
        contact_dir = accepted_dir
    contact_sheet = make_contact_sheet(contact_dir, contact_rows, "breparg_logic", args.target_count)
    report = {
        "status": "VERIFIED" if (accepted_count if args.quality_gate else saved_count) >= args.target_count else "INCOMPLETE",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
        "sequence": str(args.sequence),
        "ar_checkpoint": str(args.ar_checkpoint),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "output_dir": str(output_dir),
        "accepted_dir": str(accepted_dir) if args.quality_gate else None,
        "manifest": str(manifest_path),
        "accepted_manifest": str(accepted_manifest_path) if args.quality_gate else None,
        "contact_sheet": contact_sheet,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "summary": summarize_rows(rows, args.target_count),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
    return 0 if report["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
