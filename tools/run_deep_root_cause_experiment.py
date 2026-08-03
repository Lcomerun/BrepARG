"""Run STEP/PNG root-cause comparisons for V13 AR generation.

The experiment has two families:
1. AR sampling settings, where each item keeps sampling until it saves the
   requested number of STEP+PNG examples or reaches an attempt cap.
2. Known validation sequences, where each item attempts a fixed number of real
   samples so reconstruction failures remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import LogitsProcessorList


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"
TOOLS_DIR = REPO_ROOT / "tools"
for item in (REPO_ROOT, IMPROVEMENTS_DIR, BREPARG_DIR, TOOLS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from constrained_decoding import BrepVocab, TopologyConstrainedLogitsProcessor
from evaluate_reconstruction_v13 import (
    append_jsonl,
    grammar_validation,
    infer_device,
    load_ar_model,
    load_fsq_vqvae,
    normalize_vocab_info,
    read_pickle,
    reconstruct_one,
    write_json,
)
@dataclass(frozen=True)
class ArItem:
    name: str
    constrained: bool
    temperature: float
    top_p: float
    use_bbox_monotonic: bool = True
    enforce_face_unique: bool = True
    min_faces: int = 1


@dataclass(frozen=True)
class RealItem:
    name: str
    split: str
    min_faces: int
    max_faces: int
    min_edges: int
    max_edges: int
    max_seq_len: int = 4096
    order: str = "random"


def default_ar_items() -> list[ArItem]:
    return [
        ArItem("ar_strict_t07_min1", True, 0.70, 0.90, True, True, 1),
        ArItem("ar_strict_t09_min1", True, 0.90, 0.95, True, True, 1),
        ArItem("ar_no_bbox_t09_min1", True, 0.90, 0.95, False, True, 1),
        ArItem("ar_min_faces8_t09", True, 0.90, 0.95, True, True, 8),
        ArItem("ar_min_faces12_t095_no_bbox", True, 0.95, 0.97, False, True, 12),
    ]


def default_real_items() -> list[RealItem]:
    return [
        RealItem("vq_val_simple_f1_8_e0_20", "val", 1, 8, 0, 20),
        RealItem("vq_val_mid_f12_24_e20_80", "val", 12, 24, 20, 80),
        RealItem("vq_val_boundary_f50_e0_150", "val", 50, 50, 0, 150),
    ]


def quantile(values: list[int | float], p: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * p)]


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": min(values),
        "median": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "max": max(values),
    }


def input_ids(group: dict[str, Any]) -> list[int]:
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in original.get("input_ids") or []]
    return [int(item) for item in group.get("input_ids") or []]


def source_path_of(group: dict[str, Any]) -> str | None:
    keys = ("source_path", "path", "file_path", "pkl_path")
    for key in keys:
        value = group.get(key)
        if value:
            return str(value)
    original = group.get("original")
    if isinstance(original, dict):
        for key in keys:
            value = original.get(key)
            if value:
                return str(value)
    return None


def select_real_records(
    package: dict[str, Any],
    vocab_info: dict[str, Any],
    item: RealItem,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, group in enumerate(package.get(item.split, [])):
        ids = input_ids(group)
        if not ids or len(ids) > item.max_seq_len:
            continue
        grammar = grammar_validation(ids, vocab_info)
        if not grammar["ok"]:
            continue
        faces = int(grammar["n_faces"])
        edges = int(grammar["n_edges"])
        if not (item.min_faces <= faces <= item.max_faces):
            continue
        if not (item.min_edges <= edges <= item.max_edges):
            continue
        record: dict[str, Any] = {
            "source": item.name,
            "split": item.split,
            "index": index,
            "length": len(ids),
            "sequence": ids,
            "selected_faces": faces,
            "selected_edges": edges,
        }
        source_path = source_path_of(group)
        if source_path:
            record["source_path"] = source_path
        candidates.append(record)

    if item.order == "random":
        rng = random.Random(seed)
        rng.shuffle(candidates)
    elif item.order == "most_faces":
        candidates.sort(key=lambda row: (-int(row["selected_faces"]), -int(row["selected_edges"]), -int(row["length"])))
    elif item.order == "shortest":
        candidates.sort(key=lambda row: (int(row["length"]), int(row["index"])))
    else:
        candidates.sort(key=lambda row: int(row["index"]))
    return candidates[:target_count]


def ar_logits_processor(vocab_info: dict[str, Any], item: ArItem) -> LogitsProcessorList | None:
    if not item.constrained:
        return None
    vocab = BrepVocab(
        face_index_size=int(vocab_info["face_index_size"]),
        se_codebook_size=int(vocab_info["se_codebook_size"]),
        bbox_index_size=int(vocab_info["bbox_index_size"]),
    )
    processor = TopologyConstrainedLogitsProcessor(
        vocab,
        prompt_len=1,
        use_bbox_monotonic=bool(item.use_bbox_monotonic),
        enforce_face_unique=bool(item.enforce_face_unique),
        min_faces=int(item.min_faces),
    )
    return LogitsProcessorList([processor])


def generate_ar_records_batch(
    model: torch.nn.Module,
    vocab_info: dict[str, Any],
    item: ArItem,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    start_index: int,
) -> list[dict[str, Any]]:
    prompt = torch.full((batch_size, 1), int(vocab_info["START_TOKEN"]), dtype=torch.long, device=device)
    attention = torch.ones_like(prompt)
    kwargs: dict[str, Any] = {
        "input_ids": prompt,
        "attention_mask": attention,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": True,
        "temperature": float(item.temperature),
        "top_p": float(item.top_p),
        "top_k": 0,
        "pad_token_id": int(vocab_info["PAD_TOKEN"]),
        "eos_token_id": int(vocab_info["END_TOKEN"]),
    }
    processor = ar_logits_processor(vocab_info, item)
    if processor is not None:
        kwargs["logits_processor"] = processor
    with torch.no_grad():
        generated = model.generate(**kwargs)

    records: list[dict[str, Any]] = []
    for offset, row in enumerate(generated.detach().cpu().tolist()):
        records.append(
            {
                "source": item.name,
                "split": "generated",
                "index": start_index + offset,
                "length": len(row),
                "sequence": row,
            }
        )
    return records


def make_contact_sheet(output_dir: Path, rows: list[dict[str, Any]], name: str, limit: int) -> str | None:
    png_paths = [Path(row["png_path"]) for row in rows if row.get("png_saved") and row.get("png_path")]
    png_paths = [path for path in png_paths if path.exists()][:limit]
    if not png_paths:
        return None

    thumbs: list[Image.Image] = []
    labels: list[str] = []
    thumb_size = 180
    label_h = 26
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size + label_h), "white")
        canvas.paste(image, ((thumb_size - image.width) // 2, 0))
        thumbs.append(canvas)
        labels.append(path.stem[-28:])

    cols = min(10, max(1, len(thumbs)))
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows_count * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_size
        y = (idx // cols) * (thumb_size + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_size + 5), labels[idx], fill=(30, 35, 40))
    sheet_path = output_dir / f"{name}_contact_sheet.png"
    sheet.save(sheet_path)
    return str(sheet_path)


def enrich_preview_subprocess(row: dict[str, Any], output_dir: Path, timeout_sec: int) -> dict[str, Any]:
    if not row.get("step_saved") or not row.get("step_path"):
        row["stl_saved"] = False
        row["png_saved"] = False
        return row
    step_path = Path(row["step_path"])
    stl_path = output_dir / "stl" / f"{step_path.stem}.stl"
    png_path = output_dir / "png" / f"{step_path.stem}.png"
    title = f"f={row.get('grammar_faces', '?')} e={row.get('grammar_edges', '?')} len={row.get('sequence_length', '?')}"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "render_step_preview_once.py"),
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
        row["stl_saved"] = stl_path.exists() and stl_path.stat().st_size > 0
        row["stl_path"] = str(stl_path)
        row["png_saved"] = png_path.exists() and png_path.stat().st_size > 0
        row["png_path"] = str(png_path)
        row["png_error"] = f"preview_timeout_{timeout_sec}s"
        return row

    row["stl_saved"] = stl_path.exists() and stl_path.stat().st_size > 0
    row["stl_path"] = str(stl_path)
    row["png_saved"] = png_path.exists() and png_path.stat().st_size > 0
    row["png_path"] = str(png_path)
    if completed.returncode != 0 or not row["png_saved"]:
        err = (completed.stderr or completed.stdout or "").strip()
        row["png_error"] = err[-500:] if err else f"preview_exit_{completed.returncode}"
    return row


def summarize_rows(rows: list[dict[str, Any]], target_count: int) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    grammar_reasons: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if not row.get("grammar_ok"):
            reason = str(row.get("grammar_reason", "unknown"))
            grammar_reasons[reason] = grammar_reasons.get(reason, 0) + 1

    grammar_ok = [row for row in rows if row.get("grammar_ok")]
    step_rows = [row for row in rows if row.get("step_saved")]
    png_rows = [row for row in rows if row.get("png_saved")]
    brep_valid_rows = [row for row in rows if row.get("brep_valid")]
    faces = [int(row["grammar_faces"]) for row in grammar_ok if row.get("grammar_faces") is not None]
    edges = [int(row["grammar_edges"]) for row in grammar_ok if row.get("grammar_edges") is not None]
    lengths = [int(row["sequence_length"]) for row in rows if row.get("sequence_length") is not None]
    complex_rows = [row for row in grammar_ok if int(row.get("grammar_faces", 0)) >= 12 or int(row.get("grammar_edges", 0)) >= 20]
    return {
        "target_count": int(target_count),
        "attempted": len(rows),
        "grammar_ok": len(grammar_ok),
        "step_saved": len(step_rows),
        "png_saved": len(png_rows),
        "brep_valid": len(brep_valid_rows),
        "complex_grammar": len(complex_rows),
        "grammar_ok_fraction": round(len(grammar_ok) / max(1, len(rows)), 4),
        "step_saved_fraction": round(len(step_rows) / max(1, len(rows)), 4),
        "png_saved_fraction": round(len(png_rows) / max(1, len(rows)), 4),
        "brep_valid_fraction": round(len(brep_valid_rows) / max(1, len(rows)), 4),
        "complex_fraction_of_grammar_ok": round(len(complex_rows) / max(1, len(grammar_ok)), 4),
        "faces": numeric_summary(faces),
        "edges": numeric_summary(edges),
        "length": numeric_summary(lengths),
        "status_counts": dict(sorted(status_counts.items())),
        "grammar_fail_reasons": dict(sorted(grammar_reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def write_summary_csv(path: Path, item_reports: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "family",
        "attempted",
        "grammar_ok",
        "step_saved",
        "png_saved",
        "brep_valid",
        "complex_grammar",
        "faces_median",
        "faces_max",
        "edges_median",
        "edges_max",
        "length_median",
        "length_max",
        "contact_sheet",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in item_reports:
            summary = report["summary"]
            writer.writerow(
                {
                    "name": report["name"],
                    "family": report["family"],
                    "attempted": summary["attempted"],
                    "grammar_ok": summary["grammar_ok"],
                    "step_saved": summary["step_saved"],
                    "png_saved": summary["png_saved"],
                    "brep_valid": summary["brep_valid"],
                    "complex_grammar": summary["complex_grammar"],
                    "faces_median": summary["faces"]["median"],
                    "faces_max": summary["faces"]["max"],
                    "edges_median": summary["edges"]["median"],
                    "edges_max": summary["edges"]["max"],
                    "length_median": summary["length"]["median"],
                    "length_max": summary["length"]["max"],
                    "contact_sheet": report.get("contact_sheet"),
                }
            )


def write_summary_markdown(path: Path, item_reports: list[dict[str, Any]]) -> None:
    lines = [
        "# Deep Root Cause Experiment",
        "",
        "| item | family | attempted | grammar ok | STEP | PNG | BREP valid | complex | faces med/max | edges med/max | contact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for report in item_reports:
        summary = report["summary"]
        contact = report.get("contact_sheet") or ""
        contact_name = Path(contact).name if contact else "-"
        lines.append(
            "| {name} | {family} | {attempted} | {grammar_ok} | {step_saved} | {png_saved} | "
            "{brep_valid} | {complex_grammar} | {fmed}/{fmax} | {emed}/{emax} | {contact} |".format(
                name=report["name"],
                family=report["family"],
                attempted=summary["attempted"],
                grammar_ok=summary["grammar_ok"],
                step_saved=summary["step_saved"],
                png_saved=summary["png_saved"],
                brep_valid=summary["brep_valid"],
                complex_grammar=summary["complex_grammar"],
                fmed=summary["faces"]["median"],
                fmax=summary["faces"]["max"],
                emed=summary["edges"]["median"],
                emax=summary["edges"]["max"],
                contact=contact_name,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_records(
    records: list[dict[str, Any]],
    vocab_info: dict[str, Any],
    vqvae_model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    validate_step: bool,
    preview_timeout_sec: int,
    manifest_path: Path,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = reconstruct_one(
            record,
            vocab_info=vocab_info,
            vqvae_model=vqvae_model,
            device=device,
            output_dir=output_dir,
            write_step=True,
            write_stl=False,
            validate_step=validate_step,
            scale_factor=1.0,
        )
        row.update(extra)
        row = enrich_preview_subprocess(row, output_dir, timeout_sec=preview_timeout_sec)
        rows.append(row)
        append_jsonl(manifest_path, row)
        ok = bool(row.get("step_saved") and row.get("png_saved"))
        print(
            f"  [{len(rows):03d}] ok={int(ok)} status={row.get('status')} "
            f"faces={row.get('grammar_faces')} edges={row.get('grammar_edges')} "
            f"len={row.get('sequence_length')}",
            flush=True,
        )
    return rows


def run_ar_item(
    item: ArItem,
    model: torch.nn.Module,
    vocab_info: dict[str, Any],
    vqvae_model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    target_count: int,
    max_attempts: int,
    batch_size: int,
    max_new_tokens: int,
    validate_step: bool,
    preview_timeout_sec: int,
) -> dict[str, Any]:
    item_dir = output_dir / item.name
    item_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = item_dir / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    attempts = 0
    t0 = time.time()
    while sum(1 for row in rows if row.get("step_saved") and row.get("png_saved")) < target_count and attempts < max_attempts:
        success_count = sum(1 for row in rows if row.get("step_saved") and row.get("png_saved"))
        remaining_success = max(1, target_count - success_count)
        batch = min(batch_size, remaining_success, max_attempts - attempts)
        records = generate_ar_records_batch(
            model=model,
            vocab_info=vocab_info,
            item=item,
            batch_size=batch,
            max_new_tokens=max_new_tokens,
            device=device,
            start_index=attempts,
        )
        attempts += len(records)
        batch_rows = reconstruct_records(
            records,
            vocab_info=vocab_info,
            vqvae_model=vqvae_model,
            device=device,
            output_dir=item_dir,
            validate_step=validate_step,
            preview_timeout_sec=preview_timeout_sec,
            manifest_path=manifest_path,
            extra={
                "family": "ar_generated",
                "temperature": item.temperature,
                "top_p": item.top_p,
                "constrained": item.constrained,
                "use_bbox_monotonic": item.use_bbox_monotonic,
                "enforce_face_unique": item.enforce_face_unique,
                "min_faces": item.min_faces,
            },
        )
        rows.extend(batch_rows)

    contact_sheet = make_contact_sheet(item_dir, rows, item.name, target_count)
    summary = summarize_rows(rows, target_count)
    report = {
        "name": item.name,
        "family": "ar_generated",
        "output_dir": str(item_dir),
        "manifest": str(manifest_path),
        "contact_sheet": contact_sheet,
        "elapsed_min": round((time.time() - t0) / 60.0, 3),
        "config": item.__dict__,
        "summary": summary,
    }
    write_json(item_dir / "summary.json", report)
    return report


def run_real_item(
    item: RealItem,
    package: dict[str, Any],
    vocab_info: dict[str, Any],
    vqvae_model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    target_count: int,
    seed: int,
    validate_step: bool,
    preview_timeout_sec: int,
) -> dict[str, Any]:
    item_dir = output_dir / item.name
    item_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = item_dir / "manifest.jsonl"
    t0 = time.time()
    records = select_real_records(package, vocab_info, item, target_count, seed)
    print(f"[real] {item.name} selected={len(records)}/{target_count}", flush=True)
    rows = reconstruct_records(
        records,
        vocab_info=vocab_info,
        vqvae_model=vqvae_model,
        device=device,
        output_dir=item_dir,
        validate_step=validate_step,
        preview_timeout_sec=preview_timeout_sec,
        manifest_path=manifest_path,
        extra={
            "family": "real_sequence",
            "split": item.split,
            "min_faces_filter": item.min_faces,
            "max_faces_filter": item.max_faces,
            "min_edges_filter": item.min_edges,
            "max_edges_filter": item.max_edges,
        },
    )
    contact_sheet = make_contact_sheet(item_dir, rows, item.name, target_count)
    summary = summarize_rows(rows, target_count)
    report = {
        "name": item.name,
        "family": "real_sequence",
        "output_dir": str(item_dir),
        "manifest": str(manifest_path),
        "contact_sheet": contact_sheet,
        "elapsed_min": round((time.time() - t0) / 60.0, 3),
        "selected_records": len(records),
        "config": item.__dict__,
        "summary": summary,
    }
    write_json(item_dir / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--ar-max-attempts", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2047)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--validate-step", action="store_true")
    parser.add_argument("--preview-timeout-sec", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or f"deep_root_cause_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    package = read_pickle(args.sequence)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] device={device} target_count={args.target_count}", flush=True)

    model = load_ar_model(args.ar_checkpoint, vocab_info, device)
    vqvae_model = load_fsq_vqvae(args.vqvae_checkpoint, device)

    item_reports: list[dict[str, Any]] = []
    started = time.time()
    for item in default_ar_items():
        print(f"[ar] {item.name}", flush=True)
        item_reports.append(
            run_ar_item(
                item=item,
                model=model,
                vocab_info=vocab_info,
                vqvae_model=vqvae_model,
                device=device,
                output_dir=output_dir,
                target_count=args.target_count,
                max_attempts=args.ar_max_attempts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                validate_step=args.validate_step,
                preview_timeout_sec=args.preview_timeout_sec,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for item in default_real_items():
        item_reports.append(
            run_real_item(
                item=item,
                package=package,
                vocab_info=vocab_info,
                vqvae_model=vqvae_model,
                device=device,
                output_dir=output_dir,
                target_count=args.target_count,
                seed=args.seed,
                validate_step=args.validate_step,
                preview_timeout_sec=args.preview_timeout_sec,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "deep_root_cause_summary.csv"
    md_path = output_dir / "deep_root_cause_summary.md"
    write_summary_csv(csv_path, item_reports)
    write_summary_markdown(md_path, item_reports)
    report = {
        "status": "VERIFIED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
        "sequence": str(args.sequence),
        "ar_checkpoint": str(args.ar_checkpoint),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "output_dir": str(output_dir),
        "target_count": int(args.target_count),
        "ar_max_attempts": int(args.ar_max_attempts),
        "validate_step": bool(args.validate_step),
        "preview_timeout_sec": int(args.preview_timeout_sec),
        "summary_csv": str(csv_path),
        "summary_markdown": str(md_path),
        "items": item_reports,
    }
    report_path = output_dir / "deep_root_cause_report.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
