"""Audit parsed ABC geometry pool quality before VQ-VAE recovery training.

The recovery launcher intentionally oversamples complex and curved geometry.
This preflight audit checks that the parsed pickle pool can actually supply
those cases before rented GPU time is spent.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


try:
    from breparg_improvements.vqvae_sampling import patch_records_from_parsed
except ModuleNotFoundError:  # pragma: no cover - supports direct execution from tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "breparg_improvements"))
    from vqvae_sampling import patch_records_from_parsed


DEFAULT_MAX_FILES = 2048
DEFAULT_COMPLEX_MIN_FACES = 12
DEFAULT_COMPLEX_MIN_EDGES = 20
DEFAULT_CURVED_SCORE_THRESHOLD = 0.02
DEFAULT_MIN_PARSED_FILES = 64
DEFAULT_MIN_COMPLEX_SOURCES = 16
DEFAULT_MIN_COMPLEX_SOURCE_FRACTION = 0.05
DEFAULT_MIN_CURVED_PATCHES = 64
DEFAULT_MIN_CURVED_PATCH_FRACTION = 0.01
DEFAULT_MAX_LOAD_FAILURE_FRACTION = 0.05


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def fraction(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0:
        return 0.0
    return round(float(numerator) / denom, 6)


def iter_pickle_paths(root: Path, max_files: int | None) -> Iterable[Path]:
    limit = None if max_files is None or int(max_files) <= 0 else int(max_files)
    count = 0
    for path in root.rglob("*.pkl"):
        if not path.is_file():
            continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            break


def patch_counts(data: dict[str, Any]) -> tuple[int, int]:
    surfaces = np.asarray(data.get("surf_ncs") if data.get("surf_ncs") is not None else [])
    edges = np.asarray(data.get("edge_ncs") if data.get("edge_ncs") is not None else [])
    n_faces = int(len(surfaces)) if surfaces.ndim == 4 else 0
    n_edges = int(len(edges)) if edges.ndim == 3 else 0
    return n_faces, n_edges


def empty_report(
    parsed_pool: Path,
    *,
    status: str,
    blocking_reasons: list[str],
    max_files: int | None,
    complex_min_faces: int,
    complex_min_edges: int,
    curved_score_threshold: float,
    min_parsed_files: int,
    min_complex_sources: int,
    min_complex_source_fraction: float,
    min_curved_patches: int,
    min_curved_patch_fraction: float,
    max_load_failure_fraction: float,
) -> dict[str, Any]:
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "quality_ready": status == "PARSED_POOL_QUALITY_READY",
        "parsed_pool": str(parsed_pool),
        "blocking_reasons": blocking_reasons,
        "thresholds": {
            "max_files": None if max_files is None else int(max_files),
            "complex_min_faces": int(complex_min_faces),
            "complex_min_edges": int(complex_min_edges),
            "curved_score_threshold": float(curved_score_threshold),
            "min_parsed_files": int(min_parsed_files),
            "min_complex_sources": int(min_complex_sources),
            "min_complex_source_fraction": float(min_complex_source_fraction),
            "min_curved_patches": int(min_curved_patches),
            "min_curved_patch_fraction": float(min_curved_patch_fraction),
            "max_load_failure_fraction": float(max_load_failure_fraction),
        },
        "summary": {
            "files_scanned": 0,
            "loaded_files": 0,
            "load_failures": 0,
            "load_failure_fraction": 0.0,
            "surface_patch_records": 0,
            "edge_patch_records": 0,
            "patch_records": 0,
            "complex_source_files": 0,
            "complex_source_fraction": 0.0,
            "complex_patch_records": 0,
            "curved_source_files": 0,
            "curved_source_fraction": 0.0,
            "curved_patch_records": 0,
            "curved_patch_fraction": 0.0,
            "max_faces": 0,
            "max_edges": 0,
            "mean_faces": 0.0,
            "mean_edges": 0.0,
        },
        "load_errors": [],
    }


def audit_parsed_pool_quality(
    parsed_pool: str | Path,
    *,
    max_files: int | None = DEFAULT_MAX_FILES,
    complex_min_faces: int = DEFAULT_COMPLEX_MIN_FACES,
    complex_min_edges: int = DEFAULT_COMPLEX_MIN_EDGES,
    curved_score_threshold: float = DEFAULT_CURVED_SCORE_THRESHOLD,
    min_parsed_files: int = DEFAULT_MIN_PARSED_FILES,
    min_complex_sources: int = DEFAULT_MIN_COMPLEX_SOURCES,
    min_complex_source_fraction: float = DEFAULT_MIN_COMPLEX_SOURCE_FRACTION,
    min_curved_patches: int = DEFAULT_MIN_CURVED_PATCHES,
    min_curved_patch_fraction: float = DEFAULT_MIN_CURVED_PATCH_FRACTION,
    max_load_failure_fraction: float = DEFAULT_MAX_LOAD_FAILURE_FRACTION,
) -> dict[str, Any]:
    pool = Path(parsed_pool)
    thresholds = {
        "max_files": max_files,
        "complex_min_faces": complex_min_faces,
        "complex_min_edges": complex_min_edges,
        "curved_score_threshold": curved_score_threshold,
        "min_parsed_files": min_parsed_files,
        "min_complex_sources": min_complex_sources,
        "min_complex_source_fraction": min_complex_source_fraction,
        "min_curved_patches": min_curved_patches,
        "min_curved_patch_fraction": min_curved_patch_fraction,
        "max_load_failure_fraction": max_load_failure_fraction,
    }
    if not pool.exists():
        return empty_report(pool, status="PARSED_POOL_QUALITY_FAILED", blocking_reasons=["parsed_pool_missing"], **thresholds)
    if not pool.is_dir():
        return empty_report(pool, status="PARSED_POOL_QUALITY_FAILED", blocking_reasons=["parsed_pool_not_dir"], **thresholds)

    files_scanned = 0
    loaded_files = 0
    load_failures = 0
    surface_patch_records = 0
    edge_patch_records = 0
    complex_source_files = 0
    complex_patch_records = 0
    curved_source_files = 0
    curved_patch_records = 0
    face_counts: list[int] = []
    edge_counts: list[int] = []
    load_errors: list[dict[str, str]] = []
    threshold = float(curved_score_threshold)

    for path in iter_pickle_paths(pool, max_files):
        files_scanned += 1
        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, dict):
                raise TypeError(f"expected dict, got {type(data).__name__}")
            records = patch_records_from_parsed(data, path, complex_min_faces, complex_min_edges)
            n_faces, n_edges = patch_counts(data)
        except Exception as exc:
            load_failures += 1
            if len(load_errors) < 20:
                load_errors.append({"path": str(path), "reason": str(exc)})
            continue

        loaded_files += 1
        face_counts.append(n_faces)
        edge_counts.append(n_edges)
        source_complex = n_faces >= int(complex_min_faces) or n_edges >= int(complex_min_edges)
        source_curved = False
        if source_complex:
            complex_source_files += 1
        for record in records:
            kind = str(record.get("kind", ""))
            if kind == "surface":
                surface_patch_records += 1
            elif kind == "edge":
                edge_patch_records += 1
            if bool(record.get("is_complex_source")):
                complex_patch_records += 1
            if finite_float(record.get("curvature_score")) >= threshold:
                curved_patch_records += 1
                source_curved = True
        if source_curved:
            curved_source_files += 1

    patch_records = surface_patch_records + edge_patch_records
    summary = {
        "files_scanned": files_scanned,
        "loaded_files": loaded_files,
        "load_failures": load_failures,
        "load_failure_fraction": fraction(load_failures, files_scanned),
        "surface_patch_records": surface_patch_records,
        "edge_patch_records": edge_patch_records,
        "patch_records": patch_records,
        "complex_source_files": complex_source_files,
        "complex_source_fraction": fraction(complex_source_files, loaded_files),
        "complex_patch_records": complex_patch_records,
        "curved_source_files": curved_source_files,
        "curved_source_fraction": fraction(curved_source_files, loaded_files),
        "curved_patch_records": curved_patch_records,
        "curved_patch_fraction": fraction(curved_patch_records, patch_records),
        "max_faces": max(face_counts) if face_counts else 0,
        "max_edges": max(edge_counts) if edge_counts else 0,
        "mean_faces": round(float(sum(face_counts)) / len(face_counts), 3) if face_counts else 0.0,
        "mean_edges": round(float(sum(edge_counts)) / len(edge_counts), 3) if edge_counts else 0.0,
    }

    blocking_reasons: list[str] = []
    if files_scanned == 0:
        blocking_reasons.append("no_pickle_files_found")
    if loaded_files < int(min_parsed_files):
        blocking_reasons.append("parsed_files_below_minimum")
    if patch_records == 0:
        blocking_reasons.append("no_patch_records")
    if summary["load_failure_fraction"] > float(max_load_failure_fraction):
        blocking_reasons.append("load_failure_fraction_too_high")
    if complex_source_files < int(min_complex_sources):
        blocking_reasons.append("complex_sources_below_minimum")
    if summary["complex_source_fraction"] < float(min_complex_source_fraction):
        blocking_reasons.append("complex_source_fraction_below_minimum")
    if curved_patch_records < int(min_curved_patches):
        blocking_reasons.append("curved_patches_below_minimum")
    if summary["curved_patch_fraction"] < float(min_curved_patch_fraction):
        blocking_reasons.append("curved_patch_fraction_below_minimum")

    status = "PARSED_POOL_QUALITY_READY" if not blocking_reasons else "PARSED_POOL_QUALITY_FAILED"
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "quality_ready": status == "PARSED_POOL_QUALITY_READY",
        "parsed_pool": str(pool),
        "blocking_reasons": blocking_reasons,
        "thresholds": empty_report(pool, status=status, blocking_reasons=[], **thresholds)["thresholds"],
        "summary": summary,
        "load_errors": load_errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# V13 Parsed Pool Quality Audit",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Quality ready: {payload['quality_ready']}",
        f"- Parsed pool: `{payload['parsed_pool']}`",
        f"- Blocking reasons: {', '.join(payload['blocking_reasons']) if payload['blocking_reasons'] else 'none'}",
        "",
        "## Summary",
        "",
        f"- Files scanned: {summary['files_scanned']}",
        f"- Loaded files: {summary['loaded_files']}",
        f"- Load failures: {summary['load_failures']} ({summary['load_failure_fraction']})",
        f"- Patch records: {summary['patch_records']} surfaces={summary['surface_patch_records']} edges={summary['edge_patch_records']}",
        f"- Complex source files: {summary['complex_source_files']} ({summary['complex_source_fraction']})",
        f"- Curved patch records: {summary['curved_patch_records']} ({summary['curved_patch_fraction']})",
        f"- Max faces / edges: {summary['max_faces']} / {summary['max_edges']}",
        f"- Mean faces / edges: {summary['mean_faces']} / {summary['mean_edges']}",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in payload["thresholds"].items():
        lines.append(f"- {key}: {value}")
    if payload["load_errors"]:
        lines.extend(["", "## Load Errors", ""])
        for item in payload["load_errors"]:
            lines.append(f"- `{item['path']}`: {item['reason']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parsed ABC pool quality before VQ-VAE recovery.")
    parser.add_argument("parsed_pool", type=Path, nargs="?", default=Path("/workspace/ABC/processed/abc_parsed_full"))
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--complex-min-faces", type=int, default=DEFAULT_COMPLEX_MIN_FACES)
    parser.add_argument("--complex-min-edges", type=int, default=DEFAULT_COMPLEX_MIN_EDGES)
    parser.add_argument("--curved-score-threshold", type=float, default=DEFAULT_CURVED_SCORE_THRESHOLD)
    parser.add_argument("--min-parsed-files", type=int, default=DEFAULT_MIN_PARSED_FILES)
    parser.add_argument("--min-complex-sources", type=int, default=DEFAULT_MIN_COMPLEX_SOURCES)
    parser.add_argument("--min-complex-source-fraction", type=float, default=DEFAULT_MIN_COMPLEX_SOURCE_FRACTION)
    parser.add_argument("--min-curved-patches", type=int, default=DEFAULT_MIN_CURVED_PATCHES)
    parser.add_argument("--min-curved-patch-fraction", type=float, default=DEFAULT_MIN_CURVED_PATCH_FRACTION)
    parser.add_argument("--max-load-failure-fraction", type=float, default=DEFAULT_MAX_LOAD_FAILURE_FRACTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_parsed_pool_quality(
        args.parsed_pool,
        max_files=args.max_files,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
        curved_score_threshold=args.curved_score_threshold,
        min_parsed_files=args.min_parsed_files,
        min_complex_sources=args.min_complex_sources,
        min_complex_source_fraction=args.min_complex_source_fraction,
        min_curved_patches=args.min_curved_patches,
        min_curved_patch_fraction=args.min_curved_patch_fraction,
        max_load_failure_fraction=args.max_load_failure_fraction,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["quality_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
