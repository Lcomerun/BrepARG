"""Summarize complex curved FSQ/AR diagnostic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def one_line(label: str, report: dict[str, Any]) -> str:
    fsq = report.get("fsq_patch_metrics", {})
    ar = report.get("ar_teacher_forcing", {})
    recon = report.get("teacher_reconstruction", {})
    return (
        f"| {label} | {report.get('selected_count')} | {fsq.get('patch_count')} | "
        f"{fmt((fsq.get('mse') or {}).get('median'))} | {fmt((fsq.get('mse') or {}).get('p95'))} | "
        f"{fmt((fsq.get('chamfer') or {}).get('median'))} | {fmt((fsq.get('chamfer') or {}).get('p95'))} | "
        f"{fmt(ar.get('token_weighted_ce'))} | "
        f"{recon.get('step_saved')}/{recon.get('attempted')} | {recon.get('brep_valid')}/{recon.get('attempted')} |"
    )


def ratio(numerator: Any, denominator: Any) -> str:
    if denominator in (None, 0, "0"):
        return "n/a"
    return f"{int(numerator or 0)}/{int(denominator)}"


def reconstruction_detail_lines(reports: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "",
        "## Reconstruction Detail",
        "",
        "| Run | Bucket type | Bucket | Attempted | STEP saved | BRep valid | Errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    found = False
    bucket_specs = (
        ("face", "by_face_bucket"),
        ("edge", "by_edge_bucket"),
        ("length", "by_length_bucket"),
    )
    for label, report in reports:
        recon = report.get("teacher_reconstruction") or {}
        if recon.get("skipped"):
            continue
        for bucket_type, key in bucket_specs:
            buckets = recon.get(key) or {}
            for bucket_name, bucket in sorted(buckets.items()):
                attempted = int(bucket.get("attempted") or bucket.get("count") or 0)
                if attempted <= 0:
                    continue
                found = True
                lines.append(
                    f"| {label} | {bucket_type} | {bucket_name} | {attempted} | "
                    f"{ratio(bucket.get('step_saved'), attempted)} | "
                    f"{ratio(bucket.get('brep_valid'), attempted)} | "
                    f"{int(bucket.get('errors') or 0)} |"
                )
    if not found:
        lines.append("| n/a | n/a | n/a | 0 | n/a | n/a | 0 |")
    return lines


def render_summary(reports: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "# Complex Curved FSQ/AR Diagnostic Summary",
        "",
        "These runs isolate reconstruction and teacher-forcing behavior on complex curved validation records. They do not sample from AR.",
        "",
        "| Run | Shapes | Patches | MSE median | MSE p95 | Chamfer median | Chamfer p95 | AR weighted CE | STEP saved | BRep valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, report in reports:
        lines.append(one_line(label, report))
    lines.extend(reconstruction_detail_lines(reports))
    lines.extend(
        [
            "",
            "## Current Reading",
            "",
            "- FSQ patch medians are low, but p95/max errors are much larger, especially on surfaces. That points to a heavy-tail capacity or loss-weight problem rather than a uniform failure.",
            "- AR teacher-forcing CE is high on the complex curved subset compared with the earlier global validation CE, so AR is also weaker on this subset before free-running exposure bias appears.",
            "- True-token reconstruction saves STEP for only part of the subset and strict BRep validity is lower still. This means the FSQ/OCC reconstruction path itself needs diagnosis before relying on generation-time filters.",
            "",
            "## BrepARG Baseline Note",
            "",
            "Official BrepARG ABC weights should be tried first before local retraining. Public source: `https://huggingface.co/qingtiannihao/BrepARG` lists `checkpoint/weights/abc_ar.pt` and `checkpoint/weights/abc_vqvae.pt`; the local `BrepARG/README.md` describes this repository as the official implementation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for item in args.report:
        if "=" not in item:
            raise SystemExit(f"--report must be LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        reports.append((label, load_report(Path(raw_path))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_summary(reports) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
