"""Correlate true-token reconstruction outcomes with FSQ patch errors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * float(q)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(finite) - 1)
    frac = rank - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def numeric_stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.mean(finite),
        "median": statistics.median(finite),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
    }


def shape_patch_stats(patches: list[dict[str, Any]]) -> dict[str, Any]:
    chamfers = [float(row["chamfer"]) for row in patches if row.get("chamfer") is not None]
    mses = [float(row["mse"]) for row in patches if row.get("mse") is not None]
    return {
        "patch_count": len(patches),
        "chamfer_median": statistics.median(chamfers) if chamfers else None,
        "chamfer_p95": percentile(chamfers, 0.95),
        "chamfer_max": max(chamfers) if chamfers else None,
        "mse_median": statistics.median(mses) if mses else None,
        "mse_p95": percentile(mses, 0.95),
        "mse_max": max(mses) if mses else None,
    }


def status_groups(row: dict[str, Any]) -> list[str]:
    groups = ["all"]
    if row.get("status"):
        groups.append(str(row["status"]))
    if row.get("step_saved"):
        groups.append("step_saved")
    else:
        groups.append("step_not_saved")
    if row.get("brep_valid"):
        groups.append("brep_valid")
    else:
        groups.append("brep_invalid_or_failed")
    return groups


def analyze_correlation(manifest_path: Path, patch_metrics_path: Path, top_k: int = 10) -> dict[str, Any]:
    manifest_rows = load_jsonl(Path(manifest_path))
    patch_rows = load_jsonl(Path(patch_metrics_path))
    patches_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        source = row.get("source_relpath")
        if source:
            patches_by_source[str(source)].append(row)

    shape_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        source = str(row.get("source_relpath") or "")
        stats = shape_patch_stats(patches_by_source.get(source, []))
        shape_rows.append(
            {
                "source_relpath": source,
                "status": row.get("status"),
                "step_saved": bool(row.get("step_saved")),
                "brep_valid": bool(row.get("brep_valid")),
                "grammar_faces": int(row.get("grammar_faces", 0) or 0),
                "grammar_edges": int(row.get("grammar_edges", 0) or 0),
                "sequence_length": int(row.get("sequence_length", 0) or 0),
                **stats,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shape_rows:
        for group in status_groups(row):
            grouped[group].append(row)

    group_report = {}
    for name, rows in sorted(grouped.items()):
        group_report[name] = {
            "count": len(rows),
            "step_saved": sum(1 for row in rows if row["step_saved"]),
            "brep_valid": sum(1 for row in rows if row["brep_valid"]),
            "chamfer_p95": numeric_stats([row["chamfer_p95"] for row in rows if row["chamfer_p95"] is not None]),
            "chamfer_max": numeric_stats([row["chamfer_max"] for row in rows if row["chamfer_max"] is not None]),
            "mse_p95": numeric_stats([row["mse_p95"] for row in rows if row["mse_p95"] is not None]),
        }

    top = sorted(
        (row for row in shape_rows if row["chamfer_p95"] is not None),
        key=lambda row: (float(row["chamfer_p95"]), float(row["chamfer_max"] or 0.0)),
        reverse=True,
    )[: max(0, int(top_k))]

    return {
        "manifest": str(manifest_path),
        "patch_metrics": str(patch_metrics_path),
        "shape_count": len(shape_rows),
        "patch_metric_sources": len(patches_by_source),
        "groups": group_report,
        "top_chamfer_p95": top,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reconstruction vs FSQ Error Correlation",
        "",
        f"- Shapes: `{report['shape_count']}`",
        f"- Sources with patch metrics: `{report['patch_metric_sources']}`",
        "",
        "## Status Groups",
        "",
        "| Group | Count | STEP saved | BRep valid | Chamfer p95 median | Chamfer p95 mean | MSE p95 median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in sorted(report["groups"].items()):
        lines.append(
            "| {name} | {count} | {step_saved} | {brep_valid} | {cmed} | {cmean} | {mmed} |".format(
                name=name,
                count=group["count"],
                step_saved=group["step_saved"],
                brep_valid=group["brep_valid"],
                cmed=_fmt((group["chamfer_p95"] or {}).get("median")),
                cmean=_fmt((group["chamfer_p95"] or {}).get("mean")),
                mmed=_fmt((group["mse_p95"] or {}).get("median")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Shape-Level Chamfer p95",
            "",
            "| Rank | Status | BRep valid | Faces | Edges | Length | Patches | Chamfer p95 | Chamfer max | Source |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(report["top_chamfer_p95"], start=1):
        lines.append(
            "| {idx} | {status} | {valid} | {faces} | {edges} | {length} | {patches} | {p95} | {maxv} | `{source}` |".format(
                idx=idx,
                status=row.get("status"),
                valid=int(bool(row.get("brep_valid"))),
                faces=row.get("grammar_faces"),
                edges=row.get("grammar_edges"),
                length=row.get("sequence_length"),
                patches=row.get("patch_count"),
                p95=_fmt(row.get("chamfer_p95")),
                maxv=_fmt(row.get("chamfer_max")),
                source=row.get("source_relpath"),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_correlation(args.manifest, args.patch_metrics, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
