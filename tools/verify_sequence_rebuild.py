"""Verify a source-path-aware sequence rebuild before long-context AR training.

This gate runs after a promoted VQ-VAE checkpoint has been used to rebuild the
sequence package. It prevents AR1536/AR2048 training from starting when the
rebuilt package is missing, source_path metadata is incomplete, or length
coverage has not been refreshed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def sequence_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    size = path.stat().st_size if exists and is_file else 0
    issues: list[str] = []
    if not exists:
        issues.append("missing_sequence_package")
    elif not is_file:
        issues.append("sequence_path_is_not_file")
    elif size <= 0:
        issues.append("empty_sequence_package")
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "bytes": int(size),
        "ok": not issues,
        "issues": issues,
    }


def source_path_status(path: Path, min_source_path_coverage: float) -> dict[str, Any]:
    payload = read_json(path)
    issues: list[str] = []
    if payload is None:
        return {
            "path": str(path),
            "exists": path.exists(),
            "ready": False,
            "issues": ["missing_or_invalid_source_path_audit"],
        }

    coverage = float(payload.get("source_path_coverage") or 0.0)
    all_ready = bool(payload.get("all_splits_source_path_ready"))
    validation_ready = bool(payload.get("validation_most_curved_ready"))
    missing = int(payload.get("groups_missing_source_path") or 0)
    if not all_ready or not validation_ready or coverage < float(min_source_path_coverage) or missing:
        issues.append("source_path_audit_not_ready")
    return {
        "path": str(path),
        "exists": True,
        "ready": not issues,
        "all_splits_source_path_ready": all_ready,
        "validation_most_curved_ready": validation_ready,
        "source_path_coverage": coverage,
        "groups_missing_source_path": missing,
        "issues": issues,
    }


def length_coverage_status(path: Path | None, min_preferred_max_seq_len: int) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "ready": False,
            "preferred_max_seq_len": None,
            "issues": ["missing_length_coverage_report"],
        }
    payload = read_json(path)
    if payload is None:
        return {
            "path": str(path),
            "exists": path.exists(),
            "ready": False,
            "preferred_max_seq_len": None,
            "issues": ["missing_or_invalid_length_coverage_report"],
        }

    recommendation = payload.get("recommendation") or {}
    preferred = recommendation.get("preferred_max_seq_len")
    action = recommendation.get("action")
    complex_total = int((payload.get("overall") or {}).get("complex_total") or 0)
    issues: list[str] = []
    if action != "train_long_context_ar":
        issues.append("length_coverage_does_not_recommend_long_context_ar")
    if preferred is None or int(preferred) < int(min_preferred_max_seq_len):
        issues.append("preferred_max_seq_len_too_small")
    if complex_total <= 0:
        issues.append("no_complex_sequences_detected")
    return {
        "path": str(path),
        "exists": True,
        "ready": not issues,
        "recommendation": action,
        "preferred_max_seq_len": int(preferred) if preferred is not None else None,
        "complex_total": complex_total,
        "issues": issues,
    }


def verify_sequence_rebuild(
    *,
    sequence: str | Path,
    source_path_audit: str | Path,
    length_coverage: str | Path | None = None,
    min_source_path_coverage: float = 1.0,
    min_preferred_max_seq_len: int = 1536,
) -> dict[str, Any]:
    sequence_report = sequence_status(Path(sequence))
    source_report = source_path_status(Path(source_path_audit), min_source_path_coverage=min_source_path_coverage)
    length_report = length_coverage_status(
        Path(length_coverage) if length_coverage is not None else None,
        min_preferred_max_seq_len=min_preferred_max_seq_len,
    )
    blocking_reasons: list[str] = []
    if not sequence_report["ok"]:
        blocking_reasons.extend(sequence_report["issues"])
    if not source_report["ready"]:
        blocking_reasons.extend(source_report["issues"])
    if not length_report["ready"]:
        blocking_reasons.extend(length_report["issues"])

    ready = not blocking_reasons
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY_FOR_AR_LONG_CONTEXT" if ready else "SEQUENCE_REBUILD_NOT_READY",
        "sequence_rebuild_ready": ready,
        "next_action": "train_ar1536_then_ar2048_if_needed" if ready else "fix_sequence_rebuild_before_ar_training",
        "blocking_reasons": blocking_reasons,
        "recommended_max_seq_len": length_report.get("preferred_max_seq_len") if ready else None,
        "sequence_package": sequence_report,
        "source_path_audit": source_report,
        "length_coverage": length_report,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V13 Sequence Rebuild Verification",
        "",
        f"- Created: {report['created']}",
        f"- Status: `{report['status']}`",
        f"- Sequence rebuild ready: {report['sequence_rebuild_ready']}",
        f"- Next action: `{report['next_action']}`",
        f"- Recommended max seq len: {report.get('recommended_max_seq_len')}",
        "",
        "## Blocking Reasons",
        "",
    ]
    if report["blocking_reasons"]:
        lines.extend(f"- `{reason}`" for reason in report["blocking_reasons"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Sequence package: `{report['sequence_package']['path']}`",
            f"- Source-path audit: `{report['source_path_audit']['path']}`",
            f"- Length coverage: `{report['length_coverage']['path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify source-path sequence rebuild readiness before AR training.")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--source-path-audit", type=Path, required=True)
    parser.add_argument("--length-coverage", type=Path)
    parser.add_argument("--min-source-path-coverage", type=float, default=1.0)
    parser.add_argument("--min-preferred-max-seq-len", type=int, default=1536)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_sequence_rebuild(
        sequence=args.sequence,
        source_path_audit=args.source_path_audit,
        length_coverage=args.length_coverage,
        min_source_path_coverage=args.min_source_path_coverage,
        min_preferred_max_seq_len=args.min_preferred_max_seq_len,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["sequence_rebuild_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
