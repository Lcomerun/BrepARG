#!/usr/bin/env python
"""Decide the next V13 quality-recovery stage from current gate reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


G20_SUMMARY = Path(
    "local_runs/reconstruction_eval/"
    "eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag/"
    "generated_quality_summary.json"
)
G100_SUMMARY = Path(
    "local_runs/reconstruction_eval/"
    "eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342/"
    "generated_quality_summary.json"
)
VQVAE_SUMMARY = Path("local_runs/reconstruction_eval/vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json")
LENGTH_COVERAGE = Path("local_reports/v13_ar120_length_coverage_20260706.json")
SOURCE_PATH_AUDIT = Path("local_reports/v13_ar120_sequence_source_path_audit_20260706.json")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def generated_gate_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "decision": nested(data, "paper_gate", "decision", default="missing"),
        "promote": bool(nested(data, "paper_gate", "promote", default=False)),
        "attempted": nested(data, "summary", "attempted"),
        "step_saved": nested(data, "summary", "step_saved"),
        "strict_valid": nested(data, "summary", "strict_valid"),
        "top_two_fraction": nested(data, "topology", "top_two_fraction"),
        "strict_valid_complex": nested(data, "complexity", "strict_valid_complex"),
        "nonprimitive_strict_valid": nested(data, "semantic_complexity", "nonprimitive_strict_valid"),
        "primitive_like_strict_valid_fraction": nested(
            data,
            "semantic_complexity",
            "primitive_like_strict_valid_fraction",
        ),
        "unique_step_rate": nested(data, "step_hashes", "unique_step_rate"),
    }


def decide_quality_recovery_stage(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    g20 = load_json(repo_root / G20_SUMMARY)
    g100 = load_json(repo_root / G100_SUMMARY)
    vqvae = load_json(repo_root / VQVAE_SUMMARY)
    length = load_json(repo_root / LENGTH_COVERAGE)
    audit = load_json(repo_root / SOURCE_PATH_AUDIT)

    g100_gate = generated_gate_summary(g100)
    vqvae_decision = nested(vqvae, "promotion_gate", "decision", default="missing")
    vqvae_promoted = vqvae_decision == "promote_for_ar_rebuild"
    generated_promoted = g100_gate["decision"] == "promote_as_paper_candidates" or g100_gate["promote"]
    source_path_ready = bool(nested(audit, "validation_most_curved_ready", default=False))
    preferred_max_seq_len = nested(length, "recommendation", "preferred_max_seq_len")

    missing_inputs = [
        str(path)
        for path, data in [
            (G20_SUMMARY, g20),
            (G100_SUMMARY, g100),
            (VQVAE_SUMMARY, vqvae),
            (LENGTH_COVERAGE, length),
            (SOURCE_PATH_AUDIT, audit),
        ]
        if data is None
    ]

    blocking_reasons: list[str] = []
    if not generated_promoted:
        blocking_reasons.append("generated_quality_not_promoted")
    if not vqvae_promoted:
        blocking_reasons.append("vqvae_checkpoint_not_promoted")
    if not source_path_ready:
        blocking_reasons.append("source_path_sequence_not_ready")
    if preferred_max_seq_len and int(preferred_max_seq_len) > 1024:
        blocking_reasons.append("ar1024_length_limited")
    if missing_inputs:
        blocking_reasons.append("missing_required_evidence")

    if missing_inputs:
        status = "EVIDENCE_INCOMPLETE"
        next_stage = "collect_missing_reports"
        primary_bottleneck = "unknown_missing_evidence"
    elif not vqvae_promoted:
        status = "NEEDS_VQVAE_RECOVERY"
        next_stage = "vqvae_complex_curved_recovery"
        primary_bottleneck = "vqvae_reconstruction"
    elif not source_path_ready:
        status = "NEEDS_SOURCE_PATH_SEQUENCE_REBUILD"
        next_stage = "source_path_sequence_rebuild"
        primary_bottleneck = "sequence_provenance"
    elif preferred_max_seq_len and int(preferred_max_seq_len) > 1024:
        status = "NEEDS_LONG_CONTEXT_AR"
        next_stage = f"ar{preferred_max_seq_len}"
        primary_bottleneck = "ar_context_length"
    elif not generated_promoted:
        status = "NEEDS_AR_DIVERSITY_OR_SAMPLING_RECOVERY"
        next_stage = "topology_balanced_ar_or_sampling_ablation"
        primary_bottleneck = "ar_distribution"
    else:
        status = "READY_FOR_PAPER_FIGURE_REVIEW"
        next_stage = "human_visual_review_for_positive_figures"
        primary_bottleneck = "none_promoted_run_available"

    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "next_stage": next_stage,
        "can_train_ar_now": vqvae_promoted and source_path_ready and not missing_inputs,
        "blocking_reasons": blocking_reasons,
        "missing_inputs": missing_inputs,
        "evidence": {
            "primary_bottleneck": primary_bottleneck,
            "g20_generated": generated_gate_summary(g20),
            "g100_generated": g100_gate,
            "vqvae_baseline": {
                "decision": vqvae_decision,
                "longest_strict_valid": nested(vqvae, "slices", "longest", "brep_valid"),
                "most_faces_strict_valid": nested(vqvae, "slices", "most_faces", "brep_valid"),
            },
            "source_path_audit": {
                "validation_most_curved_ready": source_path_ready,
                "groups_with_source_path": nested(audit, "groups_with_source_path"),
            },
            "length_coverage": {
                "preferred_max_seq_len": preferred_max_seq_len,
                "complex_total": nested(length, "overall", "complex_total"),
                "coverage_1024": nested(length, "overall", "by_limit", "1024", "complex_allowed_fraction"),
                "coverage_1536": nested(length, "overall", "by_limit", "1536", "complex_allowed_fraction"),
                "coverage_2048": nested(length, "overall", "by_limit", "2048", "complex_allowed_fraction"),
            },
        },
        "stage_order": [
            "vqvae_complex_curved_recovery",
            "source_path_sequence_rebuild",
            "ar1536",
            "ar2048_if_needed",
            "generated_reconstruction_and_paper_gate",
        ],
        "server_recommendation": {
            "first_choice": "1x L40S 48GB or 1x RTX 6000 Ada/A6000 48GB",
            "upgrade_if_memory_limited": "1x A100 80GB",
            "deadline_speed_option": "1x H100 80GB",
        },
        "paper_figure_policy": {
            "positive_figures_allowed": generated_promoted and vqvae_promoted and source_path_ready,
            "current_g20_g100_role": "failure_analysis_only",
            "required_generated_decision": "promote_as_paper_candidates",
        },
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# V13 Quality-Recovery Stage Decision",
        "",
        f"Created: {report['created']}",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Next stage: `{report['next_stage']}`",
        "",
        f"Primary bottleneck: `{report['evidence']['primary_bottleneck']}`",
        "",
        f"Can train AR now: `{str(report['can_train_ar_now']).lower()}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    for reason in report["blocking_reasons"]:
        lines.append(f"- `{reason}`")
    if not report["blocking_reasons"]:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Stage Order",
            "",
        ]
    )
    for index, stage in enumerate(report["stage_order"], start=1):
        lines.append(f"{index}. `{stage}`")

    lines.extend(
        [
            "",
            "## Paper Figure Policy",
            "",
            f"Positive figures allowed: `{str(report['paper_figure_policy']['positive_figures_allowed']).lower()}`",
            "",
            f"Current G20/G100 role: `{report['paper_figure_policy']['current_g20_g100_role']}`",
            "",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = decide_quality_recovery_stage(args.repo_root)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.markdown_output)
    return 0 if not report["missing_inputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
