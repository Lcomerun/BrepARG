"""Run AR decoding ablations and summarize generated sequence complexity."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
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

from constrained_decoding import BrepVocab, TopologyConstrainedLogitsProcessor
from evaluate_reconstruction_v13 import (
    grammar_validation,
    infer_device,
    load_ar_model,
    normalize_vocab_info,
    read_pickle,
    write_json,
)
from transformers import LogitsProcessorList


@dataclass(frozen=True)
class AblationConfig:
    name: str
    constrained: bool
    temperature: float
    top_p: float
    use_bbox_monotonic: bool = True
    enforce_face_unique: bool = True
    min_faces: int = 1


def default_configs() -> list[AblationConfig]:
    return [
        AblationConfig("strict_t07", True, 0.70, 0.90, True, True, 1),
        AblationConfig("strict_t09", True, 0.90, 0.95, True, True, 1),
        AblationConfig("no_bbox_t09", True, 0.90, 0.95, False, True, 1),
        AblationConfig("min_faces_8_t09", True, 0.90, 0.95, True, True, 8),
        AblationConfig("no_bbox_min_faces_8_t09", True, 0.90, 0.95, False, True, 8),
        AblationConfig("min_faces_12_t095", True, 0.95, 0.97, False, True, 12),
        AblationConfig("no_face_unique_t09", True, 0.90, 0.95, False, False, 1),
        AblationConfig("unconstrained_t09", False, 0.90, 0.95, False, True, 1),
    ]


def quantile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return int(ordered[int((len(ordered) - 1) * p)])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["grammar_ok"]]
    faces = [int(row["faces"]) for row in valid]
    edges = [int(row["edges"]) for row in valid]
    lengths = [int(row["length"]) for row in valid]
    complex_rows = [row for row in valid if int(row["faces"]) >= 12 or int(row["edges"]) >= 20]
    reasons: dict[str, int] = {}
    for row in rows:
        if not row["grammar_ok"]:
            reasons[row["grammar_reason"]] = reasons.get(row["grammar_reason"], 0) + 1
    return {
        "n": len(rows),
        "valid": len(valid),
        "valid_fraction": round(len(valid) / max(1, len(rows)), 4),
        "complex": len(complex_rows),
        "complex_fraction_all": round(len(complex_rows) / max(1, len(rows)), 4),
        "complex_fraction_valid": round(len(complex_rows) / max(1, len(valid)), 4),
        "faces": {
            "median": quantile(faces, 0.50),
            "p90": quantile(faces, 0.90),
            "p95": quantile(faces, 0.95),
            "max": max(faces) if faces else None,
        },
        "edges": {
            "median": quantile(edges, 0.50),
            "p90": quantile(edges, 0.90),
            "p95": quantile(edges, 0.95),
            "max": max(edges) if edges else None,
        },
        "length": {
            "median": quantile(lengths, 0.50),
            "p90": quantile(lengths, 0.90),
            "p95": quantile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
        },
        "fail_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def sample_config(
    model: torch.nn.Module,
    vocab_info: dict[str, Any],
    config: AblationConfig,
    count: int,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    V = BrepVocab(
        face_index_size=int(vocab_info["face_index_size"]),
        se_codebook_size=int(vocab_info["se_codebook_size"]),
        bbox_index_size=int(vocab_info["bbox_index_size"]),
    )
    processor = None
    if config.constrained:
        processor = TopologyConstrainedLogitsProcessor(
            V,
            prompt_len=1,
            use_bbox_monotonic=config.use_bbox_monotonic,
            enforce_face_unique=config.enforce_face_unique,
            min_faces=config.min_faces,
        )
    for start in range(0, count, batch_size):
        batch = min(batch_size, count - start)
        prompt = torch.full((batch, 1), int(vocab_info["START_TOKEN"]), dtype=torch.long, device=device)
        attention = torch.ones_like(prompt)
        kwargs = {
            "input_ids": prompt,
            "attention_mask": attention,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "top_k": 0,
            "pad_token_id": int(vocab_info["PAD_TOKEN"]),
            "eos_token_id": int(vocab_info["END_TOKEN"]),
        }
        if processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([processor])
        with torch.no_grad():
            generated = model.generate(**kwargs)
        for offset, seq in enumerate(generated.detach().cpu().tolist()):
            grammar = grammar_validation(seq, vocab_info)
            rows.append(
                {
                    "config": config.name,
                    "index": start + offset,
                    "length": len(seq),
                    "grammar_ok": bool(grammar["ok"]),
                    "grammar_reason": grammar["reason"],
                    "faces": int(grammar["n_faces"]),
                    "edges": int(grammar["n_edges"]),
                }
            )
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AR Sampling Ablation",
        "",
        "| config | valid | complex | faces med/p90/max | edges med/p90/max | length med/p90/max | fail reasons |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for name, summary in report["summaries"].items():
        fail = ", ".join(f"{key}:{value}" for key, value in summary["fail_reasons"].items()) or "-"
        lines.append(
            "| {name} | {valid}/{n} ({vf:.1%}) | {complex}/{n} ({cf:.1%}) | "
            "{fm}/{fp90}/{fmax} | {em}/{ep90}/{emax} | {lm}/{lp90}/{lmax} | {fail} |".format(
                name=name,
                valid=summary["valid"],
                n=summary["n"],
                vf=summary["valid_fraction"],
                complex=summary["complex"],
                cf=summary["complex_fraction_all"],
                fm=summary["faces"]["median"],
                fp90=summary["faces"]["p90"],
                fmax=summary["faces"]["max"],
                em=summary["edges"]["median"],
                ep90=summary["edges"]["p90"],
                emax=summary["edges"]["max"],
                lm=summary["length"]["median"],
                lp90=summary["length"]["p90"],
                lmax=summary["length"]["max"],
                fail=fail,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    package = read_pickle(args.sequence)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    model = load_ar_model(args.ar_checkpoint, vocab_info, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    started = time.time()
    for config in default_configs():
        print(f"[ablation] {config.name}", flush=True)
        rows = sample_config(
            model=model,
            vocab_info=vocab_info,
            config=config,
            count=args.count,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        all_rows.extend(rows)
        summaries[config.name] = summarize_rows(rows)

    csv_path = args.output_dir / "ar_sampling_ablation_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["config", "index", "length", "grammar_ok", "grammar_reason", "faces", "edges"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "status": "VERIFIED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence": str(args.sequence),
        "ar_checkpoint": str(args.ar_checkpoint),
        "count_per_config": int(args.count),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "device": str(device),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
        "summaries": summaries,
        "rows_csv": str(csv_path),
    }
    report_path = args.output_dir / "ar_sampling_ablation_report.json"
    markdown_path = args.output_dir / "ar_sampling_ablation_report.md"
    write_json(report_path, report)
    write_markdown(markdown_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
