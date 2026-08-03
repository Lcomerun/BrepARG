"""Audit generated outputs from upstream BrepARG baseline runs.

Upstream ``BrepARG/generate_brep.py`` writes STEP/STL files directly into the
chosen output directory, while the V13 reconstruction tools usually write into
``steps/``, ``stl/``, ``png/`` and JSONL manifests. This adapter normalizes both
layouts into one summary protocol so official or same-data BrepARG baselines
can be compared against the current method.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
import sys

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_step_geometry_entities import audit_step_file  # noqa: E402
from generation_quality_gate import quality_gate_decision  # noqa: E402


DEFAULT_MIN_FACES = 12
DEFAULT_MIN_EDGES = 20
DEFAULT_MAX_FACES = 45
DEFAULT_MAX_EDGES = 120


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def find_files(run_dir: Path, suffix: str) -> list[Path]:
    paths = []
    for path in run_dir.rglob(f"*{suffix}"):
        if path.is_file() and "quality_check" not in path.parts:
            paths.append(path)
    return sorted(paths)


def lower_median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    return ordered[(len(ordered) - 1) // 2]


def int_stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(int(value) for value in values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "min": ordered[0],
        "median": lower_median(ordered),
        "mean": float(statistics.fmean(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def quality_manifest_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "quality_check" / "step_quality_manifest.jsonl",
        run_dir / "step_quality_manifest.jsonl",
    ]


def load_quality_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    quality_by_name: dict[str, dict[str, Any]] = {}
    for path in quality_manifest_candidates(run_dir):
        for row in read_jsonl(path):
            raw = row.get("step") or row.get("quality_step") or row.get("step_path") or row.get("path")
            if raw:
                quality_by_name[Path(str(raw)).name] = row
    return quality_by_name


def quality_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    prefixed = f"quality_{key}"
    if prefixed in row:
        return row[prefixed]
    if key in row:
        return row[key]
    return default


def companion_file(step_path: Path, run_dir: Path, suffix: str) -> Path | None:
    candidates = [
        step_path.with_suffix(suffix),
        run_dir / suffix.lstrip(".") / f"{step_path.stem}{suffix}",
        run_dir / f"{step_path.stem}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(run_dir.rglob(f"{step_path.stem}*{suffix}"))
    matches = [path for path in matches if "quality_check" not in path.parts]
    return matches[0] if matches else None


def entry_from_step(
    step_path: Path,
    run_dir: Path,
    quality_row: dict[str, Any] | None,
    *,
    min_faces: int,
    min_edges: int,
    max_faces: int,
    max_edges: int,
    require_quality: bool,
) -> dict[str, Any]:
    file_audit = audit_step_file(step_path)
    entity_counts = file_audit["entity_counts"]
    advanced_faces = int(file_audit["advanced_faces"])
    edge_curves = int(entity_counts.get("EDGE_CURVE", 0))
    png_path = companion_file(step_path, run_dir, ".png")
    stl_path = companion_file(step_path, run_dir, ".stl")
    png_saved = bool(png_path and png_path.exists())

    if quality_row:
        step_read_ok = bool(quality_value(quality_row, "step_read_ok", True))
        brep_valid = bool(quality_value(quality_row, "brep_valid", False))
        solid_closed = bool(quality_value(quality_row, "solid_closed_no_open_shell", file_audit["solid_closed_no_open_shell"]))
        png_saved = bool(quality_value(quality_row, "png_saved", quality_value(quality_row, "png_existing", png_saved)))
        advanced_faces = int(quality_value(quality_row, "advanced_faces", advanced_faces) or 0)
        edge_curves = int(quality_value(quality_row, "edge_curves", edge_curves) or 0)
    else:
        step_read_ok = True
        brep_valid = False
        solid_closed = bool(file_audit["solid_closed_no_open_shell"])

    row = {
        "grammar_ok": True,
        "grammar_faces": advanced_faces,
        "grammar_edges": edge_curves,
        "step_saved": True,
    }
    quality_for_gate = {
        "brep_valid": brep_valid,
        "solid_closed_no_open_shell": solid_closed,
        "png_saved": png_saved,
        "advanced_faces": advanced_faces,
        "edge_curves": edge_curves,
    }
    decision = quality_gate_decision(
        row,
        quality_for_gate,
        min_faces=min_faces,
        min_edges=min_edges,
        max_faces=max_faces,
        max_edges=max_edges,
        require_brep_valid=require_quality,
        require_closed_solid=True,
        require_preview=False,
        reject_primitive_like=True,
    )

    return {
        "step_path": str(step_path),
        "stem": step_path.stem,
        "bytes": int(step_path.stat().st_size),
        "png_path": str(png_path) if png_path else None,
        "png_existing": bool(png_path and png_path.exists()),
        "stl_path": str(stl_path) if stl_path else None,
        "stl_existing": bool(stl_path and stl_path.exists()),
        "quality_manifest_present": bool(quality_row),
        "step_read_ok": step_read_ok,
        "brep_valid": brep_valid,
        "solid_closed_no_open_shell": solid_closed,
        "advanced_faces": advanced_faces,
        "edge_curves": edge_curves,
        "nonplanar_surfaces": int(file_audit["nonplanar_surfaces"]),
        "planar_surfaces": int(file_audit["planar_surfaces"]),
        "has_nonplanar_surfaces": bool(file_audit["has_nonplanar_surfaces"]),
        "has_manifold_solid_brep": bool(file_audit["has_manifold_solid_brep"]),
        "has_closed_shell": bool(file_audit["has_closed_shell"]),
        "has_open_shell": bool(file_audit["has_open_shell"]),
        "complex_by_step_entities": bool(advanced_faces >= min_faces or edge_curves >= min_edges),
        "quality_gate_accept": bool(decision["accept"]),
        "quality_gate_reasons": list(decision["reasons"]),
        "entity_counts": entity_counts,
    }


def audit_breparg_baseline_outputs(
    run_dir: Path,
    *,
    min_faces: int = DEFAULT_MIN_FACES,
    min_edges: int = DEFAULT_MIN_EDGES,
    max_faces: int = DEFAULT_MAX_FACES,
    max_edges: int = DEFAULT_MAX_EDGES,
    require_quality: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    step_files = find_files(run_dir, ".step")
    png_files = find_files(run_dir, ".png")
    stl_files = find_files(run_dir, ".stl")
    quality_by_name = load_quality_rows(run_dir)

    entries = [
        entry_from_step(
            path,
            run_dir,
            quality_by_name.get(path.name),
            min_faces=min_faces,
            min_edges=min_edges,
            max_faces=max_faces,
            max_edges=max_edges,
            require_quality=require_quality,
        )
        for path in step_files
    ]

    reason_counts: Counter[str] = Counter()
    for entry in entries:
        reason_counts.update(entry["quality_gate_reasons"])

    advanced_faces = [int(entry["advanced_faces"]) for entry in entries]
    edge_curves = [int(entry["edge_curves"]) for entry in entries]
    lengths = [int(entry["bytes"]) for entry in entries]
    accepted = [entry for entry in entries if entry["quality_gate_accept"]]
    complex_entries = [entry for entry in entries if entry["complex_by_step_entities"]]
    complex_and_closed = [entry for entry in complex_entries if entry["solid_closed_no_open_shell"]]
    complex_and_brep_valid = [entry for entry in complex_entries if entry["brep_valid"]]
    complex_and_brep_valid_closed = [
        entry for entry in complex_entries if entry["brep_valid"] and entry["solid_closed_no_open_shell"]
    ]

    summary = {
        "step_files": len(step_files),
        "png_files": len(png_files),
        "stl_files": len(stl_files),
        "quality_manifest_rows": len(quality_by_name),
        "entries": len(entries),
        "step_read_ok": sum(1 for entry in entries if entry["step_read_ok"]),
        "brep_valid": sum(1 for entry in entries if entry["brep_valid"]),
        "files_solid_closed_no_open_shell": sum(1 for entry in entries if entry["solid_closed_no_open_shell"]),
        "files_with_nonplanar_surfaces": sum(1 for entry in entries if entry["has_nonplanar_surfaces"]),
        "complex_by_step_entities_12faces_or_20edges": len(complex_entries),
        "complex_and_closed": len(complex_and_closed),
        "complex_and_brep_valid": len(complex_and_brep_valid),
        "complex_and_brep_valid_closed": len(complex_and_brep_valid_closed),
        "strict_quality_accepted": len(accepted),
        "simple_or_rejected": len(entries) - len(accepted),
        "accepted_fraction": len(accepted) / len(entries) if entries else None,
        "complex_fraction": len(complex_entries) / len(entries) if entries else None,
    }

    warnings = []
    if not entries:
        warnings.append("no STEP files found")
    if require_quality and quality_by_name and len(quality_by_name) < len(entries):
        warnings.append("quality manifest does not cover every STEP file")
    if require_quality and not quality_by_name:
        warnings.append("no quality manifest found; strict BRep validity is unknown")
    if complex_entries and not accepted:
        warnings.append("complex STEP entities exist but none pass the strict quality gate")
    if entries and len(complex_entries) == 0:
        warnings.append("no baseline outputs satisfy the complex face/edge threshold")

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "protocol": {
            "min_faces": int(min_faces),
            "min_edges": int(min_edges),
            "max_faces": int(max_faces),
            "max_edges": int(max_edges),
            "require_quality_manifest_for_brep_valid": bool(require_quality),
            "complex_rule": "advanced_faces >= min_faces OR edge_curves >= min_edges",
        },
        "summary": summary,
        "face_stats": int_stats(advanced_faces),
        "edge_stats": int_stats(edge_curves),
        "step_byte_stats": int_stats(lengths),
        "quality_gate_reasons": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "warnings": warnings,
        "entries": entries,
    }


def markdown_text(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    protocol = audit["protocol"]
    lines = [
        "# BrepARG Baseline Output Audit",
        "",
        f"Created: {audit['created']}",
        f"Run directory: `{audit['run_dir']}`",
        "",
        "## Protocol",
        "",
        f"- Complex threshold: `advanced_faces >= {protocol['min_faces']}` or `edge_curves >= {protocol['min_edges']}`",
        f"- Max threshold: `advanced_faces <= {protocol['max_faces']}` and `edge_curves <= {protocol['max_edges']}`",
        f"- Strict BRep validity requires quality manifest: `{protocol['require_quality_manifest_for_brep_valid']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "step_files",
        "png_files",
        "stl_files",
        "quality_manifest_rows",
        "step_read_ok",
        "brep_valid",
        "files_solid_closed_no_open_shell",
        "files_with_nonplanar_surfaces",
        "complex_by_step_entities_12faces_or_20edges",
        "complex_and_closed",
        "complex_and_brep_valid",
        "complex_and_brep_valid_closed",
        "strict_quality_accepted",
        "simple_or_rejected",
        "accepted_fraction",
        "complex_fraction",
    ):
        lines.append(f"| `{key}` | {summary.get(key)} |")

    lines.extend(
        [
            "",
            "## Face/Edge Stats",
            "",
            "| Group | min | median | mean | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            stats_row("advanced_faces", audit["face_stats"]),
            stats_row("edge_curves", audit["edge_stats"]),
        ]
    )
    if audit["quality_gate_reasons"]:
        lines.extend(["", "## Reject Reasons", "", "| Reason | Count |", "| --- | ---: |"])
        for reason, count in audit["quality_gate_reasons"].items():
            lines.append(f"| `{reason}` | {count} |")
    if audit["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This adapter audits upstream BrepARG output layouts and normalizes them to the same "
            "face/edge complexity and quality-gate vocabulary used for V13 generated samples. "
            "When a quality manifest is absent, entity complexity is still measured from STEP text, "
            "but strict BRep validity remains unknown.",
            "",
        ]
    )
    return "\n".join(lines)


def stats_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {stats.get('min')} | {stats.get('median')} | "
        f"{stats.get('mean')} | {stats.get('p95')} | {stats.get('max')} |"
    )


def write_outputs(
    audit: dict[str, Any],
    output: Path | None,
    markdown_output: Path | None,
    manifest_output: Path | None,
) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown_text(audit) + "\n", encoding="utf-8")
    if manifest_output:
        write_jsonl(manifest_output, audit["entries"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--min-faces", type=int, default=DEFAULT_MIN_FACES)
    parser.add_argument("--min-edges", type=int, default=DEFAULT_MIN_EDGES)
    parser.add_argument("--max-faces", type=int, default=DEFAULT_MAX_FACES)
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)
    parser.add_argument(
        "--no-require-quality",
        action="store_true",
        help="Do not require a quality manifest for strict acceptance; useful for STEP-text-only smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_breparg_baseline_outputs(
        args.run_dir,
        min_faces=args.min_faces,
        min_edges=args.min_edges,
        max_faces=args.max_faces,
        max_edges=args.max_edges,
        require_quality=not args.no_require_quality,
    )
    write_outputs(audit, args.output, args.markdown_output, args.manifest_output)
    print(json.dumps(audit["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
