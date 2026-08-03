from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MIN_ATTEMPTS = 100
DEFAULT_MIN_STRICT_VALID = 50
DEFAULT_MAX_TOP_TWO_FRACTION = 0.60
DEFAULT_MIN_COMPLEX_STRICT_VALID = 10
DEFAULT_MIN_UNIQUE_STEP_RATE = 0.95
DEFAULT_COMPLEX_MIN_FACES = 12
DEFAULT_COMPLEX_MIN_EDGES = 20
DEFAULT_MIN_NONPRIMITIVE_STRICT_VALID = 20
DEFAULT_MAX_PRIMITIVE_STRICT_VALID_FRACTION = 0.70


PRIMITIVE_LIKE_TOPOLOGIES = {
    (2, 2),
    (3, 3),
    (4, 6),
    (4, 8),
    (5, 8),
    (5, 9),
    (6, 12),
    (7, 12),
    (8, 18),
    (9, 18),
    (9, 20),
    (10, 22),
    (10, 24),
    (11, 24),
    (12, 20),
    (12, 22),
    (12, 24),
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def topology_key(row: dict[str, Any]) -> str:
    return f"{int(row.get('grammar_faces', 0) or 0)}F/{int(row.get('grammar_edges', 0) or 0)}E"


def is_complex(row: dict[str, Any], *, min_faces: int, min_edges: int) -> bool:
    faces = int(row.get("grammar_faces", 0) or 0)
    edges = int(row.get("grammar_edges", 0) or 0)
    return faces >= int(min_faces) or edges >= int(min_edges)


def is_primitive_like(row: dict[str, Any]) -> bool:
    """Heuristic for low-detail shapes that direct CAD scripts can often create."""
    faces = int(row.get("grammar_faces", 0) or 0)
    edges = int(row.get("grammar_edges", 0) or 0)
    return (faces, edges) in PRIMITIVE_LIKE_TOPOLOGIES or (faces <= 12 and edges <= 24)


def resolve_step_path(row: dict[str, Any], run_dir: Path) -> Path | None:
    raw = row.get("step_path")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
        fallback = run_dir / "steps" / path.name
        if fallback.exists():
            return fallback
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_hash_summary(saved_rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    hashes = []
    missing = 0
    for row in saved_rows:
        path = resolve_step_path(row, run_dir)
        if path is None:
            missing += 1
            continue
        hashes.append(sha256_file(path))
    unique = len(set(hashes))
    total = len(hashes)
    return {
        "hashed_step_files": total,
        "missing_step_files": missing,
        "unique_step_hashes": unique,
        "unique_step_rate": (unique / total) if total else None,
    }


def build_paper_gate(
    *,
    attempted: int,
    strict_valid: int,
    top_two_fraction: float | None,
    strict_valid_complex: int,
    unique_step_rate: float | None,
    contact_sheet_found: bool,
    min_attempts: int,
    min_strict_valid: int,
    max_top_two_fraction: float,
    min_complex_strict_valid: int,
    min_unique_step_rate: float,
    nonprimitive_strict_valid: int,
    primitive_strict_valid_fraction: float | None,
    min_nonprimitive_strict_valid: int,
    max_primitive_strict_valid_fraction: float,
    require_render: bool,
) -> dict[str, Any]:
    requirements = {
        "attempts_enough": attempted >= min_attempts,
        "strict_valid_enough": strict_valid >= min_strict_valid,
        "topology_not_collapsed": top_two_fraction is not None and top_two_fraction <= max_top_two_fraction,
        "complex_strict_valid_enough": strict_valid_complex >= min_complex_strict_valid,
        "nonprimitive_strict_valid_enough": nonprimitive_strict_valid >= min_nonprimitive_strict_valid,
        "primitive_fraction_not_dominant": (
            primitive_strict_valid_fraction is not None
            and primitive_strict_valid_fraction <= max_primitive_strict_valid_fraction
        ),
        "unique_step_hashes_enough": unique_step_rate is not None and unique_step_rate >= min_unique_step_rate,
        "contact_sheet_present": (not require_render) or contact_sheet_found,
    }
    reasons = []
    if not requirements["attempts_enough"]:
        reasons.append("generated attempts are below the paper-candidate minimum")
    if not requirements["strict_valid_enough"]:
        reasons.append("strict-valid STEP count is below the paper-candidate minimum")
    if not requirements["topology_not_collapsed"]:
        reasons.append("top two topology buckets dominate saved outputs")
    if not requirements["complex_strict_valid_enough"]:
        reasons.append("too few complex strict-valid generated outputs")
    if not requirements["nonprimitive_strict_valid_enough"]:
        reasons.append("too few non-primitive strict-valid generated outputs")
    if not requirements["primitive_fraction_not_dominant"]:
        reasons.append("primitive-like strict-valid outputs dominate the retained set")
    if not requirements["unique_step_hashes_enough"]:
        reasons.append("unique STEP hash rate is below the paper-candidate minimum")
    if not requirements["contact_sheet_present"]:
        reasons.append("rendered contact sheet is missing")

    promote = all(requirements.values())
    return {
        "promote": promote,
        "decision": "promote_as_paper_candidates" if promote else "hold_for_failure_analysis",
        "requirements": requirements,
        "reasons": reasons,
        "thresholds": {
            "min_attempts": min_attempts,
            "min_strict_valid": min_strict_valid,
            "max_top_two_fraction": max_top_two_fraction,
            "min_complex_strict_valid": min_complex_strict_valid,
            "min_nonprimitive_strict_valid": min_nonprimitive_strict_valid,
            "max_primitive_strict_valid_fraction": max_primitive_strict_valid_fraction,
            "min_unique_step_rate": min_unique_step_rate,
            "require_render": require_render,
        },
    }


def summarize_generated_run(
    run_dir: Path,
    *,
    min_attempts: int = DEFAULT_MIN_ATTEMPTS,
    min_strict_valid: int = DEFAULT_MIN_STRICT_VALID,
    max_top_two_fraction: float = DEFAULT_MAX_TOP_TWO_FRACTION,
    min_complex_strict_valid: int = DEFAULT_MIN_COMPLEX_STRICT_VALID,
    min_unique_step_rate: float = DEFAULT_MIN_UNIQUE_STEP_RATE,
    complex_min_faces: int = DEFAULT_COMPLEX_MIN_FACES,
    complex_min_edges: int = DEFAULT_COMPLEX_MIN_EDGES,
    min_nonprimitive_strict_valid: int = DEFAULT_MIN_NONPRIMITIVE_STRICT_VALID,
    max_primitive_strict_valid_fraction: float = DEFAULT_MAX_PRIMITIVE_STRICT_VALID_FRACTION,
    require_render: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    report = read_json(run_dir / "reconstruction_report.json")
    rows = read_jsonl(run_dir / "reconstruction_manifest.jsonl")
    saved_rows = [row for row in rows if row.get("step_saved")]
    strict_valid_rows = [row for row in rows if row.get("brep_valid")]
    complex_rows = [row for row in rows if is_complex(row, min_faces=complex_min_faces, min_edges=complex_min_edges)]
    strict_valid_complex_rows = [
        row for row in strict_valid_rows if is_complex(row, min_faces=complex_min_faces, min_edges=complex_min_edges)
    ]
    primitive_like_strict_valid_rows = [row for row in strict_valid_rows if is_primitive_like(row)]
    nonprimitive_strict_valid_rows = [row for row in strict_valid_rows if not is_primitive_like(row)]
    primitive_like_saved_rows = [row for row in saved_rows if is_primitive_like(row)]

    topology_counts = Counter(topology_key(row) for row in saved_rows)
    saved_count = len(saved_rows)
    topologies = [
        {"topology": key, "count": count, "fraction": (count / saved_count) if saved_count else None}
        for key, count in topology_counts.most_common()
    ]
    top_two_count = sum(item["count"] for item in topologies[:2])
    top_two_fraction = (top_two_count / saved_count) if saved_count else None
    hash_summary = step_hash_summary(saved_rows, run_dir)
    contact_sheet = run_dir / "renders" / "contact_sheet.png"

    attempted = int(report.get("summary", {}).get("attempted", len(rows)) or 0)
    strict_valid = int(report.get("summary", {}).get("brep_valid", len(strict_valid_rows)) or 0)
    step_saved = int(report.get("summary", {}).get("step_saved", saved_count) or 0)
    errors = int(report.get("summary", {}).get("errors", max(0, attempted - len(strict_valid_rows))) or 0)

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "report_path": str(run_dir / "reconstruction_report.json"),
        "manifest_path": str(run_dir / "reconstruction_manifest.jsonl"),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_found": contact_sheet.exists(),
        "summary": {
            "attempted": attempted,
            "manifest_rows": len(rows),
            "step_saved": step_saved,
            "strict_valid": strict_valid,
            "errors": errors,
        },
        "topology": {
            "unique_topology_count": len(topology_counts),
            "top_two_count": top_two_count,
            "top_two_fraction": top_two_fraction,
            "topologies": topologies,
        },
        "complexity": {
            "complex_min_faces": complex_min_faces,
            "complex_min_edges": complex_min_edges,
            "saved_complex": len([row for row in saved_rows if is_complex(row, min_faces=complex_min_faces, min_edges=complex_min_edges)]),
            "strict_valid_complex": len(strict_valid_complex_rows),
            "manifest_complex": len(complex_rows),
        },
        "semantic_complexity": {
            "primitive_like_rule": "primitive-like when topology is in a known simple bucket or faces<=12 and edges<=24",
            "primitive_like_saved": len(primitive_like_saved_rows),
            "primitive_like_strict_valid": len(primitive_like_strict_valid_rows),
            "nonprimitive_strict_valid": len(nonprimitive_strict_valid_rows),
            "primitive_like_strict_valid_fraction": (
                len(primitive_like_strict_valid_rows) / len(strict_valid_rows) if strict_valid_rows else None
            ),
        },
        "step_hashes": hash_summary,
        "paper_gate": build_paper_gate(
            attempted=attempted,
            strict_valid=strict_valid,
            top_two_fraction=top_two_fraction,
            strict_valid_complex=len(strict_valid_complex_rows),
            unique_step_rate=hash_summary["unique_step_rate"],
            nonprimitive_strict_valid=len(nonprimitive_strict_valid_rows),
            primitive_strict_valid_fraction=(
                len(primitive_like_strict_valid_rows) / len(strict_valid_rows) if strict_valid_rows else None
            ),
            contact_sheet_found=contact_sheet.exists(),
            min_attempts=min_attempts,
            min_strict_valid=min_strict_valid,
            max_top_two_fraction=max_top_two_fraction,
            min_complex_strict_valid=min_complex_strict_valid,
            min_unique_step_rate=min_unique_step_rate,
            min_nonprimitive_strict_valid=min_nonprimitive_strict_valid,
            max_primitive_strict_valid_fraction=max_primitive_strict_valid_fraction,
            require_render=require_render,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize whether a generated reconstruction run is paper-quality.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-attempts", type=int, default=DEFAULT_MIN_ATTEMPTS)
    parser.add_argument("--min-strict-valid", type=int, default=DEFAULT_MIN_STRICT_VALID)
    parser.add_argument("--max-top-two-fraction", type=float, default=DEFAULT_MAX_TOP_TWO_FRACTION)
    parser.add_argument("--min-complex-strict-valid", type=int, default=DEFAULT_MIN_COMPLEX_STRICT_VALID)
    parser.add_argument("--min-nonprimitive-strict-valid", type=int, default=DEFAULT_MIN_NONPRIMITIVE_STRICT_VALID)
    parser.add_argument(
        "--max-primitive-strict-valid-fraction",
        type=float,
        default=DEFAULT_MAX_PRIMITIVE_STRICT_VALID_FRACTION,
    )
    parser.add_argument("--min-unique-step-rate", type=float, default=DEFAULT_MIN_UNIQUE_STEP_RATE)
    parser.add_argument("--complex-min-faces", type=int, default=DEFAULT_COMPLEX_MIN_FACES)
    parser.add_argument("--complex-min-edges", type=int, default=DEFAULT_COMPLEX_MIN_EDGES)
    parser.add_argument("--no-require-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_generated_run(
        args.run_dir,
        min_attempts=args.min_attempts,
        min_strict_valid=args.min_strict_valid,
        max_top_two_fraction=args.max_top_two_fraction,
        min_complex_strict_valid=args.min_complex_strict_valid,
        min_unique_step_rate=args.min_unique_step_rate,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
        min_nonprimitive_strict_valid=args.min_nonprimitive_strict_valid,
        max_primitive_strict_valid_fraction=args.max_primitive_strict_valid_fraction,
        require_render=not args.no_require_render,
    )
    output = args.output or (args.run_dir / "generated_quality_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated quality summary written to {output}")
    print(f"Paper gate decision: {summary['paper_gate']['decision']}")
    for reason in summary["paper_gate"]["reasons"]:
        print(f"- {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
