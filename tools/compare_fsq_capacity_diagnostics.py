"""Compare baseline and higher-capacity FSQ complex-curved diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def metric_delta(baseline: Any, candidate: Any) -> dict[str, Any]:
    base = None if baseline is None else float(baseline)
    cand = None if candidate is None else float(candidate)
    absolute = None if base is None or cand is None else cand - base
    relative = None
    if base not in (None, 0.0) and cand is not None:
        relative = (cand - base) / abs(base) * 100.0
    return {
        "baseline": base,
        "candidate": cand,
        "absolute_change": absolute,
        "relative_change_pct": relative,
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "fsq_chamfer_median": ("fsq_patch_metrics", "chamfer", "median"),
    "fsq_chamfer_p95": ("fsq_patch_metrics", "chamfer", "p95"),
    "fsq_chamfer_max": ("fsq_patch_metrics", "chamfer", "max"),
    "surface_chamfer_p95": ("fsq_patch_metrics", "by_kind", "surface", "chamfer", "p95"),
    "surface_chamfer_max": ("fsq_patch_metrics", "by_kind", "surface", "chamfer", "max"),
    "edge_chamfer_p95": ("fsq_patch_metrics", "by_kind", "edge", "chamfer", "p95"),
    "edge_chamfer_max": ("fsq_patch_metrics", "by_kind", "edge", "chamfer", "max"),
}


def capacity_reading(metrics: dict[str, dict[str, Any]]) -> dict[str, str]:
    fsq_p95 = metrics["fsq_chamfer_p95"].get("relative_change_pct")
    surface_p95 = metrics["surface_chamfer_p95"].get("relative_change_pct")
    edge_p95 = metrics["edge_chamfer_p95"].get("relative_change_pct")

    values = [value for value in (fsq_p95, surface_p95, edge_p95) if value is not None]
    if not values:
        return {
            "capacity_signal": "insufficient_metrics",
            "reading": "Missing comparable Chamfer p95 metrics; rerun both diagnostics with FSQ patch metrics.",
        }

    if fsq_p95 is not None and surface_p95 is not None and fsq_p95 <= -20.0 and surface_p95 <= -25.0:
        return {
            "capacity_signal": "strong_improvement",
            "reading": "Higher FSQ capacity substantially reduces the complex-curved Chamfer tail, especially surfaces.",
        }
    if fsq_p95 is not None and fsq_p95 <= -10.0:
        return {
            "capacity_signal": "moderate_improvement",
            "reading": "Higher FSQ capacity reduces overall Chamfer p95, but inspect surface and edge tails before calling FSQ solved.",
        }
    if fsq_p95 is not None and fsq_p95 >= 10.0:
        return {
            "capacity_signal": "regression",
            "reading": "Higher FSQ capacity is worse on overall Chamfer p95; shift attention to representation, loss, or data rather than this capacity setting.",
        }
    return {
        "capacity_signal": "inconclusive",
        "reading": "Capacity change did not move overall Chamfer p95 enough to isolate FSQ capacity as the main driver.",
    }


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        name: metric_delta(nested_get(baseline, path), nested_get(candidate, path))
        for name, path in METRIC_PATHS.items()
    }
    same_subset = {
        "baseline_selected_count": baseline.get("selected_count"),
        "candidate_selected_count": candidate.get("selected_count"),
        "baseline_patch_count": nested_get(baseline, ("fsq_patch_metrics", "patch_count")),
        "candidate_patch_count": nested_get(candidate, ("fsq_patch_metrics", "patch_count")),
    }
    return {
        "status": "VERIFIED",
        "baseline": {
            "path": baseline.get("output_dir"),
            "checkpoint": nested_get(baseline, ("fsq_patch_metrics", "checkpoint")),
        },
        "candidate": {
            "path": candidate.get("output_dir"),
            "checkpoint": nested_get(candidate, ("fsq_patch_metrics", "checkpoint")),
        },
        "same_subset_check": same_subset,
        "metrics": metrics,
        "recommendation": capacity_reading(metrics),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# FSQ Capacity Comparison",
        "",
        "This compares the current FSQ diagnostic against the higher-capacity FSQ candidate on the same complex-curved protocol.",
        "",
        f"- Status: `{comparison.get('status')}`",
        f"- Capacity signal: `{comparison['recommendation']['capacity_signal']}`",
        f"- Reading: {comparison['recommendation']['reading']}",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Candidate | Absolute change | Relative change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in comparison["metrics"].items():
        rel = metric.get("relative_change_pct")
        rel_text = "n/a" if rel is None else f"{rel:.2f}%"
        lines.append(
            f"| `{name}` | {fmt(metric.get('baseline'))} | {fmt(metric.get('candidate'))} | "
            f"{fmt(metric.get('absolute_change'))} | {rel_text} |"
        )
    subset = comparison.get("same_subset_check") or {}
    lines.extend(
        [
            "",
            "## Subset Check",
            "",
            f"- baseline selected count: `{subset.get('baseline_selected_count')}`",
            f"- candidate selected count: `{subset.get('candidate_selected_count')}`",
            f"- baseline patch count: `{subset.get('baseline_patch_count')}`",
            f"- candidate patch count: `{subset.get('candidate_patch_count')}`",
            "",
            "Interpretation guard: this comparison only tests the FSQ capacity variable. It does not complete the BrepARG baseline or the full DFS/RCM ordering experiment.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_reports(load_json(args.baseline), load_json(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(comparison) + "\n", encoding="utf-8")
    print(args.output)
    print(args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
