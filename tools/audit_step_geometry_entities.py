from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


PLANAR_SURFACE_ENTITIES = ("PLANE",)

NONPLANAR_SURFACE_ENTITIES = (
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
    "BEZIER_SURFACE",
    "SURFACE_OF_REVOLUTION",
    "SURFACE_OF_LINEAR_EXTRUSION",
    "OFFSET_SURFACE",
)

CURVE_ENTITIES = (
    "LINE",
    "CIRCLE",
    "ELLIPSE",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_CURVE",
    "BEZIER_CURVE",
)

TOPOLOGY_ENTITIES = (
    "ADVANCED_FACE",
    "EDGE_CURVE",
    "VERTEX_POINT",
    "ORIENTED_EDGE",
    "FACE_OUTER_BOUND",
    "FACE_BOUND",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "MANIFOLD_SOLID_BREP",
    "SHELL_BASED_SURFACE_MODEL",
)

ENTITIES_TO_COUNT = (
    PLANAR_SURFACE_ENTITIES
    + NONPLANAR_SURFACE_ENTITIES
    + CURVE_ENTITIES
    + TOPOLOGY_ENTITIES
)

DEFAULT_COMPLEX_MIN_ADVANCED_FACES = 12
DEFAULT_COMPLEX_MIN_EDGE_CURVES = 20


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fallback_rows_from_steps(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, path in enumerate(sorted((run_dir / "steps").glob("*.step"))):
        rows.append(
            {
                "index": idx,
                "status": "saved",
                "step_saved": True,
                "step_path": str(path),
            }
        )
    return rows


def resolve_step_path(row: dict[str, Any], run_dir: Path) -> Path | None:
    raw = row.get("step_path")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
        fallback = run_dir / "steps" / path.name
        if fallback.exists():
            return fallback
    index = row.get("index")
    if index is not None:
        matches = sorted((run_dir / "steps").glob(f"*{int(index):06d}*.step"))
        if matches:
            return matches[0]
    return None


def count_step_entities(text: str) -> dict[str, int]:
    upper = text.upper()
    counts: dict[str, int] = {}
    for entity in ENTITIES_TO_COUNT:
        counts[entity] = len(re.findall(rf"\b{re.escape(entity)}\s*\(", upper))
    return counts


def audit_step_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    counts = count_step_entities(text)
    planar_surfaces = sum(counts[item] for item in PLANAR_SURFACE_ENTITIES)
    nonplanar_surfaces = sum(counts[item] for item in NONPLANAR_SURFACE_ENTITIES)
    curve_entities = sum(counts[item] for item in CURVE_ENTITIES)
    solid_breps = counts["MANIFOLD_SOLID_BREP"]
    closed_shells = counts["CLOSED_SHELL"]
    open_shells = counts["OPEN_SHELL"]
    advanced_faces = counts["ADVANCED_FACE"]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "entity_counts": counts,
        "planar_surfaces": planar_surfaces,
        "nonplanar_surfaces": nonplanar_surfaces,
        "curve_entities": curve_entities,
        "advanced_faces": advanced_faces,
        "has_nonplanar_surfaces": nonplanar_surfaces > 0,
        "all_planar_surfaces": planar_surfaces > 0 and nonplanar_surfaces == 0,
        "has_manifold_solid_brep": solid_breps > 0,
        "has_closed_shell": closed_shells > 0,
        "has_open_shell": open_shells > 0,
        "solid_closed_no_open_shell": solid_breps > 0 and closed_shells > 0 and open_shells == 0,
    }


def audit_step_geometry_entities(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "reconstruction_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    rows = manifest_rows or fallback_rows_from_steps(run_dir)
    saved_rows = [row for row in rows if row.get("step_saved", True)]
    entries = []
    entity_totals: Counter[str] = Counter()
    missing_step_files = 0

    for row in saved_rows:
        step_path = resolve_step_path(row, run_dir)
        if step_path is None:
            missing_step_files += 1
            continue
        file_audit = audit_step_file(step_path)
        entity_totals.update(file_audit["entity_counts"])
        entries.append(
            {
                "index": row.get("index"),
                "brep_valid": bool(row.get("brep_valid")),
                "grammar_faces": row.get("grammar_faces"),
                "grammar_edges": row.get("grammar_edges"),
                **file_audit,
            }
        )

    strict_valid_entries = [entry for entry in entries if entry["brep_valid"]]
    step_files_audited = len(entries)
    files_with_nonplanar = sum(1 for entry in entries if entry["has_nonplanar_surfaces"])
    files_solid_closed = sum(1 for entry in entries if entry["solid_closed_no_open_shell"])
    files_with_many_faces = sum(
        1 for entry in entries if entry["advanced_faces"] >= DEFAULT_COMPLEX_MIN_ADVANCED_FACES
    )
    files_with_many_edges = sum(
        1 for entry in entries if entry["entity_counts"]["EDGE_CURVE"] >= DEFAULT_COMPLEX_MIN_EDGE_CURVES
    )
    strict_valid_with_nonplanar = sum(
        1 for entry in strict_valid_entries if entry["has_nonplanar_surfaces"]
    )
    strict_valid_solid_closed = sum(
        1 for entry in strict_valid_entries if entry["solid_closed_no_open_shell"]
    )
    strict_valid_with_many_faces = sum(
        1 for entry in strict_valid_entries if entry["advanced_faces"] >= DEFAULT_COMPLEX_MIN_ADVANCED_FACES
    )
    strict_valid_with_many_edges = sum(
        1 for entry in strict_valid_entries if entry["entity_counts"]["EDGE_CURVE"] >= DEFAULT_COMPLEX_MIN_EDGE_CURVES
    )
    advanced_face_values = [entry["advanced_faces"] for entry in entries]
    edge_curve_values = [entry["entity_counts"]["EDGE_CURVE"] for entry in entries]

    summary = {
        "manifest_rows_total": len(rows),
        "saved_rows_total": len(saved_rows),
        "step_files_audited": step_files_audited,
        "missing_step_files": missing_step_files,
        "files_with_nonplanar_surfaces": files_with_nonplanar,
        "files_all_planar_surfaces": sum(1 for entry in entries if entry["all_planar_surfaces"]),
        "files_with_manifold_solid_brep": sum(1 for entry in entries if entry["has_manifold_solid_brep"]),
        "files_with_closed_shell": sum(1 for entry in entries if entry["has_closed_shell"]),
        "files_with_open_shell": sum(1 for entry in entries if entry["has_open_shell"]),
        "files_solid_closed_no_open_shell": files_solid_closed,
        "files_with_at_least_12_advanced_faces": files_with_many_faces,
        "files_with_at_least_20_edge_curves": files_with_many_edges,
        "strict_valid_entries": len(strict_valid_entries),
        "strict_valid_with_nonplanar_surfaces": strict_valid_with_nonplanar,
        "strict_valid_solid_closed_no_open_shell": strict_valid_solid_closed,
        "strict_valid_with_at_least_12_advanced_faces": strict_valid_with_many_faces,
        "strict_valid_with_at_least_20_edge_curves": strict_valid_with_many_edges,
        "mean_advanced_faces_per_file": (
            sum(advanced_face_values) / len(advanced_face_values) if advanced_face_values else None
        ),
        "max_advanced_faces_per_file": max(advanced_face_values) if advanced_face_values else None,
        "mean_edge_curves_per_file": (
            sum(edge_curve_values) / len(edge_curve_values) if edge_curve_values else None
        ),
        "max_edge_curves_per_file": max(edge_curve_values) if edge_curve_values else None,
        "nonplanar_surface_file_fraction": (
            files_with_nonplanar / step_files_audited if step_files_audited else None
        ),
        "solid_closed_no_open_shell_fraction": (
            files_solid_closed / step_files_audited if step_files_audited else None
        ),
    }

    warnings = []
    if step_files_audited == 0:
        warnings.append("no STEP files were audited")
    if files_with_nonplanar == 0:
        warnings.append("no retained STEP files contain non-planar surface entities")
    if strict_valid_with_nonplanar == 0:
        warnings.append("no strict-valid retained STEP files contain non-planar surface entities")
    if files_solid_closed < step_files_audited:
        warnings.append("some retained STEP files are missing solid closed-shell evidence")
    if strict_valid_with_many_faces == 0:
        warnings.append("no strict-valid retained STEP files have at least 12 advanced faces")

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "summary": summary,
        "entity_totals": {entity: int(entity_totals.get(entity, 0)) for entity in ENTITIES_TO_COUNT},
        "warnings": warnings,
        "entries": entries,
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    summary = audit["summary"]
    lines = [
        "# STEP Geometry Entity Audit",
        "",
        f"Created: {audit['created']}",
        f"Run directory: `{audit['run_dir']}`",
        "",
        "## Summary",
        "",
        "```text",
        f"step_files_audited: {summary['step_files_audited']}",
        f"strict_valid_entries: {summary['strict_valid_entries']}",
        f"files_with_nonplanar_surfaces: {summary['files_with_nonplanar_surfaces']}",
        f"strict_valid_with_nonplanar_surfaces: {summary['strict_valid_with_nonplanar_surfaces']}",
        f"files_solid_closed_no_open_shell: {summary['files_solid_closed_no_open_shell']}",
        f"strict_valid_solid_closed_no_open_shell: {summary['strict_valid_solid_closed_no_open_shell']}",
        f"files_with_at_least_12_advanced_faces: {summary['files_with_at_least_12_advanced_faces']}",
        f"strict_valid_with_at_least_12_advanced_faces: {summary['strict_valid_with_at_least_12_advanced_faces']}",
        f"files_with_at_least_20_edge_curves: {summary['files_with_at_least_20_edge_curves']}",
        f"strict_valid_with_at_least_20_edge_curves: {summary['strict_valid_with_at_least_20_edge_curves']}",
        f"mean_advanced_faces_per_file: {summary['mean_advanced_faces_per_file']}",
        f"mean_edge_curves_per_file: {summary['mean_edge_curves_per_file']}",
        f"nonplanar_surface_file_fraction: {summary['nonplanar_surface_file_fraction']}",
        f"solid_closed_no_open_shell_fraction: {summary['solid_closed_no_open_shell_fraction']}",
        "```",
        "",
        "## Entity Totals",
        "",
        "| Entity | Count |",
        "| --- | ---: |",
    ]
    for entity, count in audit["entity_totals"].items():
        if count:
            lines.append(f"| `{entity}` | {count} |")
    if audit["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a STEP-text entity audit, not a replacement for OpenCascade validation or visual review. "
            "It is useful for distinguishing exact file uniqueness from geometric variety: a run can have "
            "unique STEP hashes while still containing mostly planar or simple solid entities.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit STEP surface and shell entities in a reconstruction run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    audit = audit_step_geometry_entities(args.run_dir)
    if args.output:
        args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.markdown_output:
        write_markdown(audit, args.markdown_output)
    print(json.dumps(audit["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
