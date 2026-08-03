#!/usr/bin/env python
"""Summarize the V13 quality-recovery pipeline across server gates.

This tool is intentionally read-only. It combines the guarded server preflight,
VQ-VAE recovery monitor, copy-back verification, sequence-rebuild verification,
generated-quality gate, and paper-figure audit into one small progress report.
The report is meant for the rented-server recovery loop: it tells the operator
where the pipeline is currently held and which stage should run next.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_PREFLIGHT = Path("local_reports/v13_server_quality_recovery_preflight_localdryrun_20260706.json")
DEFAULT_VQVAE_MONITOR = Path("local_reports/vqvae_recovery_monitor_latest.json")
DEFAULT_COPYBACK_VERIFY = Path("local_reports/v13_vqvae_copyback_verify_local_20260706.json")
DEFAULT_SEQUENCE_VERIFY = Path("local_reports/v13_sequence_rebuild_verify_server.json")
DEFAULT_HUMAN_REVIEW = Path("local_reports/v13_human_visual_review_latest.json")


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def nested(data: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def input_status(path: Path | None, data: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and Path(path).exists()),
        "valid_json": data is not None,
        "status": data.get("status") if isinstance(data, dict) else None,
        "state": data.get("state") if isinstance(data, dict) else None,
        "decision": data.get("decision") if isinstance(data, dict) else None,
    }


def report(
    *,
    status: str,
    current_stage: str,
    next_action: str,
    blocking_reasons: list[str] | None = None,
    can_train_ar_now: bool = False,
    positive_figures_allowed: bool = False,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "current_stage": current_stage,
        "next_action": next_action,
        "blocking_reasons": list(blocking_reasons or []),
        "can_train_ar_now": bool(can_train_ar_now),
        "positive_figures_allowed": bool(positive_figures_allowed),
        "inputs": inputs,
        "stage_order": [
            "server_preflight",
            "vqvae_complex_curved_recovery",
            "vqvae_copyback",
            "source_path_sequence_rebuild",
            "ar1536_or_ar2048",
            "generated_quality_gate",
            "paper_figure_audit",
            "human_visual_review",
        ],
    }


def preflight_blocking_reasons(preflight: dict[str, Any] | None) -> list[str]:
    reasons = list(nested(preflight, "plan", "blocking_reasons", default=[]) or [])
    if reasons:
        return [str(reason) for reason in reasons]
    stage_results = preflight.get("stage_results") if isinstance(preflight, dict) else {}
    if isinstance(stage_results, dict):
        return [
            str(label)
            for label, payload in stage_results.items()
            if isinstance(payload, dict) and int(payload.get("returncode", 0) or 0) != 0
        ]
    return []


def is_preflight_ready(preflight: dict[str, Any] | None) -> bool:
    if not isinstance(preflight, dict):
        return False
    if preflight.get("status") == "READY_TO_START_VQVAE_RECOVERY":
        return True
    gates = nested(preflight, "plan", "gates", default={}) or {}
    if isinstance(gates, dict) and gates:
        return all(bool((gate or {}).get("ready")) for gate in gates.values())
    return False


def is_vqvae_promoted(monitor: dict[str, Any] | None) -> bool:
    if not isinstance(monitor, dict):
        return False
    return bool(monitor.get("ready")) and monitor.get("state") == "ready_for_sequence_rebuild"


def is_vqvae_terminal_hold(monitor: dict[str, Any] | None) -> bool:
    if not isinstance(monitor, dict):
        return False
    return bool(monitor.get("terminal")) and not is_vqvae_promoted(monitor)


def vqvae_monitor_blocking_reasons(monitor: dict[str, Any] | None) -> list[str]:
    if not isinstance(monitor, dict):
        return ["missing_vqvae_monitor"]
    reasons: list[str] = []
    state = monitor.get("state")
    if state:
        reasons.append(str(state))
    reason = monitor.get("reason")
    if reason and str(reason) not in reasons:
        reasons.append(str(reason))
    benchmark_reasons = nested(monitor, "benchmark", "reasons", default=[]) or []
    for item in benchmark_reasons:
        text = str(item)
        if text not in reasons:
            reasons.append(text)
    missing_required = nested(monitor, "copy_back_manifest", "missing_required", default=[]) or []
    for item in missing_required:
        text = f"missing_copyback:{item}"
        if text not in reasons:
            reasons.append(text)
    return reasons or ["vqvae_recovery_not_promoted"]


def is_copyback_ready(copyback: dict[str, Any] | None) -> bool:
    if not isinstance(copyback, dict):
        return False
    return copyback.get("status") == "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD" or bool(copyback.get("copyback_ready"))


def is_sequence_ready(sequence: dict[str, Any] | None) -> bool:
    if not isinstance(sequence, dict):
        return False
    return sequence.get("status") == "READY_FOR_AR_LONG_CONTEXT" or bool(sequence.get("sequence_rebuild_ready"))


def is_generated_promoted(generated: dict[str, Any] | None) -> bool:
    if not isinstance(generated, dict):
        return False
    gate = generated.get("paper_gate") or {}
    return gate.get("decision") == "promote_as_paper_candidates" or bool(gate.get("promote"))


def is_paper_ready(paper: dict[str, Any] | None) -> bool:
    if not isinstance(paper, dict):
        return False
    return paper.get("decision") == "ready_for_paper_figure_review" and bool(paper.get("ready_for_human_review"))


def is_human_review_approved(review: dict[str, Any] | None) -> bool:
    if not isinstance(review, dict):
        return False
    decision = review.get("decision") or review.get("status")
    approved = bool(review.get("approved") or review.get("positive_figures_approved"))
    return decision in {"approved_for_paper_figures", "APPROVED_FOR_PAPER_FIGURES"} and approved


def summarize_quality_recovery_progress(
    *,
    repo_root: str | Path = ".",
    preflight: str | Path | None = None,
    vqvae_monitor: str | Path | None = None,
    copyback_verify: str | Path | None = None,
    sequence_verify: str | Path | None = None,
    generated_quality: str | Path | None = None,
    paper_audit: str | Path | None = None,
    human_review: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)

    paths = {
        "preflight": Path(preflight) if preflight is not None else root / DEFAULT_PREFLIGHT,
        "vqvae_monitor": Path(vqvae_monitor) if vqvae_monitor is not None else root / DEFAULT_VQVAE_MONITOR,
        "copyback_verify": Path(copyback_verify) if copyback_verify is not None else root / DEFAULT_COPYBACK_VERIFY,
        "sequence_verify": Path(sequence_verify) if sequence_verify is not None else root / DEFAULT_SEQUENCE_VERIFY,
        "generated_quality": Path(generated_quality) if generated_quality is not None else None,
        "paper_audit": Path(paper_audit) if paper_audit is not None else None,
        "human_review": Path(human_review) if human_review is not None else root / DEFAULT_HUMAN_REVIEW,
    }
    data = {label: read_json(path) for label, path in paths.items()}
    inputs = {label: input_status(paths[label], data[label]) for label in paths}

    preflight_data = data["preflight"]
    if preflight_data is None:
        return report(
            status="WAITING_FOR_SERVER_PREFLIGHT",
            current_stage="server_preflight",
            next_action="run_server_guarded_preflight",
            blocking_reasons=["missing_or_invalid_preflight_report"],
            inputs=inputs,
        )
    if not is_preflight_ready(preflight_data):
        return report(
            status=str(preflight_data.get("status", "HOLD_BEFORE_VQVAE_RECOVERY")),
            current_stage="server_preflight",
            next_action="fix_server_preflight_before_training",
            blocking_reasons=preflight_blocking_reasons(preflight_data),
            inputs=inputs,
        )

    monitor_data = data["vqvae_monitor"]
    if not is_vqvae_promoted(monitor_data):
        if is_vqvae_terminal_hold(monitor_data):
            return report(
                status="HOLD_AFTER_VQVAE_RECOVERY",
                current_stage="vqvae_complex_curved_recovery",
                next_action="inspect_vqvae_recovery_failure_before_sequence_rebuild",
                blocking_reasons=vqvae_monitor_blocking_reasons(monitor_data),
                can_train_ar_now=False,
                positive_figures_allowed=False,
                inputs=inputs,
            )
        monitor_state = str((monitor_data or {}).get("state", "missing_vqvae_monitor"))
        return report(
            status="WAITING_FOR_VQVAE_PROMOTION",
            current_stage="vqvae_complex_curved_recovery",
            next_action="continue_monitoring_vqvae_recovery",
            blocking_reasons=[monitor_state],
            inputs=inputs,
        )

    copyback_data = data["copyback_verify"]
    if not is_copyback_ready(copyback_data):
        return report(
            status="WAITING_FOR_VQVAE_COPYBACK",
            current_stage="vqvae_copyback",
            next_action="pull_and_verify_vqvae_copyback",
            blocking_reasons=["missing_or_unready_copyback_verification"],
            inputs=inputs,
        )

    sequence_data = data["sequence_verify"]
    if not is_sequence_ready(sequence_data):
        return report(
            status="WAITING_FOR_SOURCE_PATH_SEQUENCE_REBUILD",
            current_stage="source_path_sequence_rebuild",
            next_action="run_source_path_sequence_rebuild_and_verify",
            blocking_reasons=["missing_or_unready_sequence_verification"],
            can_train_ar_now=False,
            inputs=inputs,
        )

    generated_data = data["generated_quality"]
    if not is_generated_promoted(generated_data):
        return report(
            status="WAITING_FOR_PROMOTED_GENERATED_RUN",
            current_stage="generated_quality_gate",
            next_action="train_long_context_ar_generate_100_and_run_quality_gate",
            blocking_reasons=["missing_or_unpromoted_generated_quality_summary"],
            can_train_ar_now=True,
            inputs=inputs,
        )

    paper_data = data["paper_audit"]
    if not is_paper_ready(paper_data):
        return report(
            status="WAITING_FOR_PAPER_FIGURE_AUDIT",
            current_stage="paper_figure_audit",
            next_action="run_paper_figure_candidate_audit",
            blocking_reasons=["missing_or_unready_paper_figure_audit"],
            can_train_ar_now=True,
            inputs=inputs,
        )

    human_review_data = data["human_review"]
    if not is_human_review_approved(human_review_data):
        return report(
            status="WAITING_FOR_HUMAN_VISUAL_REVIEW",
            current_stage="human_visual_review",
            next_action="record_human_visual_review_decision",
            blocking_reasons=["missing_or_unapproved_human_visual_review"],
            can_train_ar_now=True,
            positive_figures_allowed=False,
            inputs=inputs,
        )

    return report(
        status="READY_FOR_PAPER_FIGURE_REPLACEMENT",
        current_stage="human_visual_review",
        next_action="replace_diagnostic_figures_with_approved_generated_outputs",
        can_train_ar_now=True,
        positive_figures_allowed=True,
        inputs=inputs,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Quality-Recovery Progress",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Current stage: `{payload['current_stage']}`",
        f"- Next action: `{payload['next_action']}`",
        f"- Can train AR now: `{str(payload['can_train_ar_now']).lower()}`",
        f"- Positive figures allowed: `{str(payload['positive_figures_allowed']).lower()}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if payload["blocking_reasons"]:
        lines.extend(f"- `{reason}`" for reason in payload["blocking_reasons"])
    else:
        lines.append("- None")
    lines.extend(["", "## Inputs", ""])
    for label, item in payload["inputs"].items():
        lines.append(
            f"- `{label}`: exists={item['exists']} valid_json={item['valid_json']} "
            f"status=`{item.get('status')}` state=`{item.get('state')}` decision=`{item.get('decision')}` "
            f"path=`{item.get('path')}`"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V13 quality-recovery progress across server gates.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--vqvae-monitor", type=Path)
    parser.add_argument("--copyback-verify", type=Path)
    parser.add_argument("--sequence-verify", type=Path)
    parser.add_argument("--generated-quality", type=Path)
    parser.add_argument("--paper-audit", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = summarize_quality_recovery_progress(
        repo_root=args.repo_root,
        preflight=args.preflight,
        vqvae_monitor=args.vqvae_monitor,
        copyback_verify=args.copyback_verify,
        sequence_verify=args.sequence_verify,
        generated_quality=args.generated_quality,
        paper_audit=args.paper_audit,
        human_review=args.human_review,
    )
    if args.output:
        write_json(args.output, payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["status"] == "READY_FOR_PAPER_FIGURE_REPLACEMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
