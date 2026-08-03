"""Summarize AR sequence coverage at multiple context lengths.

The current V13 AR model was trained with a 1024-token limit. This utility
quantifies how many geometry sequences, especially complex ones, would become
usable at longer limits such as 1536 or 2048 before launching an expensive
server run.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"

for item in (str(REPO_ROOT), str(IMPROVEMENTS_DIR), str(BREPARG_DIR), str(REPO_ROOT / "tools")):
    if item not in sys.path:
        sys.path.insert(0, item)

from evaluate_reconstruction_v13 import grammar_validation, normalize_vocab_info


DEFAULT_LIMITS = (1024, 1536, 2048)
DEFAULT_COMPLEX_MIN_FACES = 12
DEFAULT_COMPLEX_MIN_EDGES = 20


def input_ids_of(group: Any) -> list[int]:
    if not isinstance(group, dict):
        return []
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in (original.get("input_ids") or [])]
    return [int(item) for item in (group.get("input_ids") or [])]


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * float(q)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def distribution_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": int(min(values)),
        "p25": round(float(percentile(values, 0.25)), 2),
        "median": float(statistics.median(values)),
        "mean": round(float(statistics.mean(values)), 2),
        "p75": round(float(percentile(values, 0.75)), 2),
        "p90": round(float(percentile(values, 0.90)), 2),
        "p95": round(float(percentile(values, 0.95)), 2),
        "p99": round(float(percentile(values, 0.99)), 2),
        "max": int(max(values)),
    }


def length_stats(lengths: list[int]) -> dict[str, Any]:
    return distribution_stats(lengths)


def sequence_record(
    ids: list[int],
    vocab_info: dict[str, Any],
    complex_min_faces: int,
    complex_min_edges: int,
) -> dict[str, Any]:
    grammar = grammar_validation(ids, vocab_info)
    faces = int(grammar.get("n_faces", 0))
    edges = int(grammar.get("n_edges", 0))
    grammar_ok = bool(grammar.get("ok"))
    is_complex = grammar_ok and (faces >= int(complex_min_faces) or edges >= int(complex_min_edges))
    return {
        "length": int(len(ids)),
        "grammar_ok": grammar_ok,
        "grammar_reason": str(grammar.get("reason", "")),
        "faces": faces,
        "edges": edges,
        "complex": bool(is_complex),
    }


def summarize_records(records: list[dict[str, Any]], limits: Iterable[int]) -> dict[str, Any]:
    lengths = [int(record["length"]) for record in records]
    grammar_ok_records = [record for record in records if record["grammar_ok"]]
    complex_records = [record for record in records if record["complex"]]
    face_counts = [int(record["faces"]) for record in grammar_ok_records]
    edge_counts = [int(record["edges"]) for record in grammar_ok_records]
    split_summary: dict[str, Any] = {
        "nonempty_sequences": len(records),
        "lengths": length_stats(lengths),
        "faces": distribution_stats(face_counts),
        "edges": distribution_stats(edge_counts),
        "grammar_ok": len(grammar_ok_records),
        "grammar_failed": len(records) - len(grammar_ok_records),
        "complex_total": len(complex_records),
        "complex_fraction_of_grammar_ok": round(len(complex_records) / len(grammar_ok_records), 6)
        if grammar_ok_records
        else 0.0,
        "faces_ge_12_total": sum(1 for record in grammar_ok_records if int(record["faces"]) >= 12),
        "faces_ge_20_total": sum(1 for record in grammar_ok_records if int(record["faces"]) >= 20),
        "edges_ge_20_total": sum(1 for record in grammar_ok_records if int(record["edges"]) >= 20),
        "by_limit": {},
    }
    for limit in limits:
        limit_int = int(limit)
        allowed = [record for record in records if int(record["length"]) <= limit_int]
        allowed_complex = [record for record in complex_records if int(record["length"]) <= limit_int]
        split_summary["by_limit"][str(limit_int)] = {
            "allowed": len(allowed),
            "excluded": len(records) - len(allowed),
            "allowed_fraction": round(len(allowed) / len(records), 4) if records else 0.0,
            "complex_allowed": len(allowed_complex),
            "complex_excluded": len(complex_records) - len(allowed_complex),
            "complex_allowed_fraction": round(len(allowed_complex) / len(complex_records), 4)
            if complex_records
            else 0.0,
            "faces_ge_12_allowed": sum(
                1 for record in allowed if record["grammar_ok"] and int(record["faces"]) >= 12
            ),
            "faces_ge_20_allowed": sum(
                1 for record in allowed if record["grammar_ok"] and int(record["faces"]) >= 20
            ),
            "edges_ge_20_allowed": sum(
                1 for record in allowed if record["grammar_ok"] and int(record["edges"]) >= 20
            ),
        }
    return split_summary


def summarize_split(
    groups: list[Any],
    vocab_info: dict[str, Any],
    limits: Iterable[int],
    complex_min_faces: int,
    complex_min_edges: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    empty = 0
    for group in groups:
        ids = input_ids_of(group)
        if not ids:
            empty += 1
            continue
        records.append(sequence_record(ids, vocab_info, complex_min_faces, complex_min_edges))
    summary = summarize_records(records, limits)
    summary.update({
        "total_groups": len(groups),
        "empty_sequences": empty,
    })
    return summary, records


def build_recommendation(summary: dict[str, Any], limits: list[int]) -> dict[str, Any]:
    if not limits:
        return {"action": "inspect_sequence_package", "preferred_max_seq_len": None, "reason": "no limits supplied"}
    base = int(limits[0])
    overall = summary["overall"]["by_limit"]
    best_limit = max(
        limits,
        key=lambda item: (
            overall[str(int(item))]["complex_allowed"],
            overall[str(int(item))]["allowed"],
            -int(item),
        ),
    )
    base_complex = int(overall[str(base)]["complex_allowed"])
    best_complex = int(overall[str(int(best_limit))]["complex_allowed"])
    if best_limit > base and best_complex > base_complex:
        return {
            "action": "train_long_context_ar",
            "preferred_max_seq_len": int(best_limit),
            "baseline_max_seq_len": base,
            "baseline_complex_allowed": base_complex,
            "preferred_complex_allowed": best_complex,
            "additional_complex_sequences": best_complex - base_complex,
            "reason": "longer context admits more grammar-valid complex sequences into AR training/evaluation",
        }
    if int(summary["overall"]["complex_total"]) == 0:
        return {
            "action": "rebuild_sequences_or_relax_complex_threshold",
            "preferred_max_seq_len": base,
            "reason": "no grammar-valid complex sequences were detected",
        }
    return {
        "action": "focus_vqvae_recovery_before_ar_length_change",
        "preferred_max_seq_len": base,
        "reason": "longer context did not increase complex-sequence coverage",
    }


def summarize_length_coverage(
    package: dict[str, Any],
    limits: Iterable[int] = DEFAULT_LIMITS,
    complex_min_faces: int = DEFAULT_COMPLEX_MIN_FACES,
    complex_min_edges: int = DEFAULT_COMPLEX_MIN_EDGES,
) -> dict[str, Any]:
    limits_list = sorted({int(item) for item in limits})
    vocab_info = normalize_vocab_info(package)
    split_summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_summary, records = summarize_split(
            list(package.get(split, [])),
            vocab_info,
            limits_list,
            complex_min_faces,
            complex_min_edges,
        )
        split_summaries[split] = split_summary
        all_records.extend(records)

    overall = summarize_records(all_records, limits_list)
    overall["total_groups"] = sum(split_summaries[split]["total_groups"] for split in split_summaries)
    overall["empty_sequences"] = sum(split_summaries[split]["empty_sequences"] for split in split_summaries)
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limits": limits_list,
        "thresholds": {
            "complex_min_faces": int(complex_min_faces),
            "complex_min_edges": int(complex_min_edges),
        },
        "splits": split_summaries,
        "overall": overall,
    }
    summary["recommendation"] = build_recommendation(summary, limits_list)
    return summary


def load_package(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any], sequence_path: Path | None = None) -> str:
    def display(value: Any) -> str:
        return "NA" if value is None else str(value)

    lines = [
        "# V13 AR Length Coverage",
        "",
        f"- Created: {summary['created']}",
    ]
    if sequence_path is not None:
        lines.append(f"- Sequence package: `{sequence_path}`")
    lines.extend([
        f"- Limits: {', '.join(str(item) for item in summary['limits'])}",
        "- Complex threshold: "
        f"faces >= {summary['thresholds']['complex_min_faces']} or "
        f"edges >= {summary['thresholds']['complex_min_edges']}",
        f"- Recommendation: `{summary['recommendation']['action']}` "
        f"(preferred max_seq_len={summary['recommendation'].get('preferred_max_seq_len')})",
        "",
        "## Overall",
        "",
        "| Limit | Allowed | Excluded | Complex allowed | Complex excluded | Complex allowed frac |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for limit in summary["limits"]:
        row = summary["overall"]["by_limit"][str(limit)]
        lines.append(
            f"| {limit} | {row['allowed']} | {row['excluded']} | "
            f"{row['complex_allowed']} | {row['complex_excluded']} | {row['complex_allowed_fraction']:.4f} |"
        )
    lines.extend([
        "",
        "## Splits",
        "",
        "| Split | Groups | Empty | Grammar ok | Grammar failed | Complex total | Max len |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for split, split_summary in summary["splits"].items():
        max_len = split_summary["lengths"]["max"]
        lines.append(
            f"| {split} | {split_summary['total_groups']} | {split_summary['empty_sequences']} | "
            f"{split_summary['grammar_ok']} | {split_summary['grammar_failed']} | "
            f"{split_summary['complex_total']} | {max_len if max_len is not None else 'NA'} |"
        )
    lines.extend([
        "",
        "## Split Distributions",
        "",
        "| Split | Metric | Count | Min | P25 | Median | P75 | P95 | P99 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split, split_summary in summary["splits"].items():
        for metric in ("lengths", "faces", "edges"):
            row = split_summary[metric]
            lines.append(
                f"| {split} | {metric} | {row['count']} | {display(row['min'])} | "
                f"{display(row['p25'])} | {display(row['median'])} | {display(row['p75'])} | "
                f"{display(row['p95'])} | {display(row['p99'])} | {display(row['max'])} |"
            )
    lines.extend([
        "",
        "| Split | Grammar valid | Complex | Complex fraction |",
        "|---|---:|---:|---:|",
    ])
    for split, split_summary in summary["splits"].items():
        lines.append(
            f"| {split} | {split_summary['grammar_ok']} | {split_summary['complex_total']} | "
            f"{split_summary['complex_fraction_of_grammar_ok']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        summary["recommendation"]["reason"],
    ])
    return "\n".join(lines) + "\n"


def parse_limits(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AR sequence coverage across context lengths.")
    parser.add_argument("sequence", type=Path, help="Path to sequences_fsq_rcm.pkl or equivalent sequence package.")
    parser.add_argument("--limits", default="1024,1536,2048", help="Comma-separated max_seq_len values.")
    parser.add_argument("--complex-min-faces", type=int, default=DEFAULT_COMPLEX_MIN_FACES)
    parser.add_argument("--complex-min-edges", type=int, default=DEFAULT_COMPLEX_MIN_EDGES)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = load_package(args.sequence)
    summary = summarize_length_coverage(
        package,
        limits=parse_limits(args.limits),
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
    )
    summary["sequence"] = str(args.sequence)
    if args.output:
        write_json(args.output, summary)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary, args.sequence), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
