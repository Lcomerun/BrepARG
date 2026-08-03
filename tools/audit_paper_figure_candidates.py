from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


READY_DECISION = "ready_for_paper_figure_review"
HOLD_DECISION = "hold_for_failure_analysis"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def count_step_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.step") if item.is_file())


def audit_paper_figure_candidates(run_dir: Path, *, summary_path: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = Path(summary_path) if summary_path else run_dir / "generated_quality_summary.json"
    summary = read_json(summary_path)
    steps_dir = run_dir / "steps"
    manifest_path = run_dir / "reconstruction_manifest.jsonl"
    report_path = run_dir / "reconstruction_report.json"

    blocking_reasons: list[str] = []
    gate = (summary or {}).get("paper_gate") or {}
    summary_counts = (summary or {}).get("summary") or {}
    expected_steps = int(summary_counts.get("step_saved", 0) or 0)

    if summary is None:
        blocking_reasons.append("generated quality summary is missing")

    gate_decision = str(gate.get("decision", "missing"))
    gate_promoted = gate_decision == "promote_as_paper_candidates"
    if not gate_promoted:
        blocking_reasons.append("generated quality gate is not promoted")
        for reason in gate.get("reasons") or []:
            if isinstance(reason, str) and reason not in blocking_reasons:
                blocking_reasons.append(reason)

    contact_sheet = Path((summary or {}).get("contact_sheet") or (run_dir / "renders" / "contact_sheet.png"))
    if not contact_sheet.is_absolute():
        contact_sheet = (Path.cwd() / contact_sheet).resolve() if contact_sheet.parts[:1] else run_dir / "renders" / "contact_sheet.png"
        if not contact_sheet.exists():
            contact_sheet = run_dir / "renders" / "contact_sheet.png"
    contact_ok = contact_sheet.exists() and contact_sheet.is_file()
    if not contact_ok:
        blocking_reasons.append("rendered contact sheet is missing")

    step_count = count_step_files(steps_dir)
    step_ok = steps_dir.exists() and steps_dir.is_dir() and step_count >= max(1, expected_steps)
    if not step_ok:
        blocking_reasons.append("retained STEP files are missing or fewer than summary step_saved")

    summary_ok = summary is not None
    ready = summary_ok and gate_promoted and contact_ok and step_ok
    decision = READY_DECISION if ready else HOLD_DECISION
    paper_role = "positive_candidate_pending_human_review" if ready else "failure_analysis_only"

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "decision": decision,
        "paper_role": paper_role,
        "ready_for_human_review": ready,
        "human_review_required": True,
        "blocking_reasons": blocking_reasons,
        "artifacts": {
            "generated_quality_summary": {
                "path": str(summary_path),
                "exists": summary_path.exists(),
                "ok": summary_ok,
            },
            "contact_sheet": {
                "path": str(contact_sheet),
                "exists": contact_sheet.exists(),
                "ok": contact_ok,
            },
            "step_files": {
                "directory": str(steps_dir),
                "exists": steps_dir.exists(),
                "count": step_count,
                "expected_at_least": max(1, expected_steps),
                "ok": step_ok,
            },
            "reconstruction_manifest": {
                "path": str(manifest_path),
                "exists": manifest_path.exists(),
                "ok": manifest_path.exists() and manifest_path.is_file(),
            },
            "reconstruction_report": {
                "path": str(report_path),
                "exists": report_path.exists(),
                "ok": report_path.exists() and report_path.is_file(),
            },
        },
        "generated_quality_gate": {
            "decision": gate_decision,
            "promote": bool(gate.get("promote")),
            "requirements": gate.get("requirements") or {},
            "reasons": gate.get("reasons") or [],
        },
        "metrics": {
            "attempted": summary_counts.get("attempted") if summary else None,
            "step_saved": expected_steps if summary else None,
            "strict_valid": summary_counts.get("strict_valid") if summary else None,
            "strict_valid_complex": ((summary or {}).get("complexity") or {}).get("strict_valid_complex"),
            "nonprimitive_strict_valid": ((summary or {}).get("semantic_complexity") or {}).get(
                "nonprimitive_strict_valid"
            ),
            "primitive_like_strict_valid_fraction": ((summary or {}).get("semantic_complexity") or {}).get(
                "primitive_like_strict_valid_fraction"
            ),
            "top_two_topology_fraction": ((summary or {}).get("topology") or {}).get("top_two_fraction"),
            "unique_step_rate": ((summary or {}).get("step_hashes") or {}).get("unique_step_rate"),
        },
        "paper_update_rule": (
            "Machine gate is ready only for human visual review; replace positive paper figures "
            "only after inspection confirms nontrivial plausible CAD."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Paper Figure Candidate Audit",
        "",
        f"- Created: {payload['created']}",
        f"- Run: `{payload['run_dir']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Paper role: `{payload['paper_role']}`",
        f"- Human review required: {payload['human_review_required']}",
        "",
    ]
    if payload["blocking_reasons"]:
        lines.append("## Blocking Reasons")
        lines.append("")
        for reason in payload["blocking_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    lines.append("## Artifact Checks")
    lines.append("")
    for label, item in payload["artifacts"].items():
        path = item.get("path") or item.get("directory")
        lines.append(f"- `{label}`: ok={item.get('ok')} path=`{path}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for label, value in payload["metrics"].items():
        lines.append(f"- `{label}`: {value}")
    lines.append("")
    lines.append(payload["paper_update_rule"])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether a generated run is ready for paper figure review.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_paper_figure_candidates(args.run_dir, summary_path=args.summary)
    output = args.output or (args.run_dir / "paper_figure_candidate_audit.json")
    write_json(output, audit)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Paper figure candidate audit written to {output}")
    print(f"Decision: {audit['decision']}")
    for reason in audit["blocking_reasons"]:
        print(f"- {reason}")
    return 0 if audit["decision"] == READY_DECISION else 2


if __name__ == "__main__":
    raise SystemExit(main())
