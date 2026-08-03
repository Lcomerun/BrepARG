#!/usr/bin/env python
"""Build a practical phase budget for the V13 rented-server recovery run.

The report is deliberately conservative. It does not start training. It turns
the current quality-recovery gates into an operator-facing plan that answers
which stage should run first, which GPU tier is appropriate, and which later
stages remain blocked until the VQ-VAE checkpoint is promoted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_PROGRESS = Path("local_reports/v13_quality_recovery_progress_localdryrun_20260706.json")
DEFAULT_STAGE_DECISION = Path("local_reports/v13_quality_recovery_stage_decision_20260706.json")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def progress_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "missing",
            "current_stage": "unknown",
            "can_train_ar_now": False,
            "positive_figures_allowed": False,
        }
    return {
        "status": str(payload.get("status", "unknown")),
        "current_stage": str(payload.get("current_stage", "unknown")),
        "next_action": str(payload.get("next_action", "unknown")),
        "can_train_ar_now": bool(payload.get("can_train_ar_now")),
        "positive_figures_allowed": bool(payload.get("positive_figures_allowed")),
    }


def stage_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "missing",
            "next_stage": "unknown",
            "can_train_ar_now": False,
            "positive_figures_allowed": False,
        }
    return {
        "status": str(payload.get("status", "unknown")),
        "next_stage": str(payload.get("next_stage", "unknown")),
        "can_train_ar_now": bool(payload.get("can_train_ar_now")),
        "positive_figures_allowed": bool((payload.get("paper_figure_policy") or {}).get("positive_figures_allowed")),
    }


def vqvae_command() -> str:
    return "\n".join(
        [
            "bash tools/run_vqvae_complex_recovery.sh \\",
            "  --repo-root /workspace/V13 \\",
            "  --python /opt/conda/bin/python \\",
            "  --pool /workspace/ABC/processed/abc_parsed_full \\",
            "  --outbase /workspace/ABC/processed/train_outputs \\",
            "  --run-name newscheme_full_vqvae_complex_recovery \\",
            "  --samples 450000 \\",
            "  --epochs 80 \\",
            "  --batch-size 128 \\",
            "  --lr 1e-4 \\",
            "  --complex-fraction 0.50 \\",
            "  --complex-min-faces 12 \\",
            "  --complex-min-edges 20 \\",
            "  --curved-fraction 0.35 \\",
            "  --run-benchmark \\",
            "  --sequence /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl \\",
            "  --benchmark-output-root /workspace/V13/local_runs/reconstruction_eval \\",
            "  --benchmark-prefix vq_complex_recovery_20260706",
        ]
    )


def build_hardware_tiers() -> list[dict[str, Any]]:
    return [
        {
            "id": "local_rtx3060_12gb",
            "name": "Local RTX 3060",
            "memory_gb": 12,
            "decision": "debug_only",
            "speed_tier": "local_debug",
            "best_use": "debugging, reconstruction, rendering, LaTeX, and smoke tests",
            "reason": "below the 40GB readiness gate",
        },
        {
            "id": "rtx4090_24gb",
            "name": "RTX 4090",
            "memory_gb": 24,
            "decision": "smoke_test_only",
            "speed_tier": "consumer_fast",
            "best_use": "environment smoke tests or short non-mainline experiments",
            "reason": "fast but below the 40GB formal recovery gate",
        },
        {
            "id": "l40s_48gb",
            "name": "L40S 48GB",
            "memory_gb": 48,
            "decision": "first_choice",
            "speed_tier": "balanced_fast",
            "best_use": "VQ-VAE complex/curved recovery and AR1536",
            "reason": "passes the memory gate with good cost/performance for the first paid run",
        },
        {
            "id": "rtx6000_ada_48gb",
            "name": "RTX 6000 Ada 48GB",
            "memory_gb": 48,
            "decision": "first_choice",
            "speed_tier": "balanced_fast",
            "best_use": "VQ-VAE complex/curved recovery and AR1536",
            "reason": "48GB workstation-class alternative when L40S is unavailable or expensive",
        },
        {
            "id": "rtx_a6000_48gb",
            "name": "RTX A6000 48GB",
            "memory_gb": 48,
            "decision": "budget_48gb_fallback",
            "speed_tier": "budget_steady",
            "best_use": "first VQ-VAE recovery run when budget matters more than speed",
            "reason": "passes the memory gate but is usually slower than newer 48GB options",
        },
        {
            "id": "a100_80gb",
            "name": "A100 80GB",
            "memory_gb": 80,
            "decision": "upgrade_for_ar2048_or_larger_batches",
            "speed_tier": "datacenter_stable",
            "best_use": "AR2048, larger VQ-VAE batches, or memory-limited recovery sweeps",
            "reason": "more memory headroom for long context and larger batch experiments",
        },
        {
            "id": "h100_80gb",
            "name": "H100 80GB",
            "memory_gb": 80,
            "decision": "deadline_speed_upgrade",
            "speed_tier": "fastest_deadline",
            "best_use": "fast sweeps when time matters more than rental cost",
            "reason": "highest speed tier, but not needed before the VQ-VAE-first plan is proven",
        },
    ]


def build_phases(next_stage: str) -> list[dict[str, Any]]:
    vq_gate = "run_next" if next_stage == "vqvae_complex_curved_recovery" else "inspect_current_stage"
    return [
        {
            "id": "server_preflight",
            "gate": "must_pass_before_training",
            "recommended_gpu": "48GB_single_gpu_or_better",
            "stop_if": [
                "transfer_verification_not_ready",
                "gpu_memory_below_minimum",
                "parsed_pool_quality_failed",
                "model_artifacts_failed",
            ],
            "acceptance": "READY_TO_START_VQVAE_RECOVERY before --start is used",
            "command": "python tools/run_server_quality_recovery.py ... --min-gpu-memory-gb 40",
        },
        {
            "id": "vqvae_complex_curved_recovery",
            "gate": vq_gate,
            "recommended_gpu": "48GB_single_gpu",
            "upgrade_gpu": "80GB_single_gpu if batch size or AR2048 becomes memory-limited",
            "stop_if": [
                "monitor exit code 2",
                "benchmark_summary decision hold_vqvae_checkpoint",
                "contact sheets still show broken complex/curved reconstructions",
            ],
            "acceptance": "benchmark summary decision promote_for_ar_rebuild plus complete copy_back_manifest.json",
            "command": vqvae_command(),
        },
        {
            "id": "source_path_sequence_rebuild",
            "gate": "blocked_until_vqvae_promoted",
            "recommended_gpu": "CPU_or_single_gpu_server",
            "acceptance": "verify_sequence_rebuild.py reports READY_FOR_AR_LONG_CONTEXT",
            "command": "bash tools/run_source_path_sequence_rebuild.sh ... --resume",
        },
        {
            "id": "ar1536_long_context",
            "gate": "blocked_until_sequence_ready",
            "recommended_gpu": "48GB_single_gpu",
            "acceptance": "AR1536 checkpoint plus generated100 run passes generated-quality gate",
            "command": "bash tools/run_ar_v13_long_context.sh ... --max-seq-len 1536 --batch-size 8 --no-auto-resume",
        },
        {
            "id": "ar2048_optional",
            "gate": "blocked_until_ar1536_or_memory_need",
            "recommended_gpu": "80GB_single_gpu_preferred",
            "acceptance": "AR2048 improves nonprimitive strict-valid outputs without severe validity loss",
            "command": "bash tools/run_ar_v13_long_context.sh ... --max-seq-len 2048 --batch-size 4 --no-auto-resume",
        },
        {
            "id": "generated100_paper_gate",
            "gate": "blocked_until_long_context_ar_checkpoint",
            "recommended_gpu": "server_or_local_for_rendering_after_copyback",
            "acceptance": (
                "generated_quality_summary promotes, step_geometry_entity_audit is nontrivial, "
                "paper_figure_candidate_audit is ready_for_paper_figure_review, and human review passes"
            ),
            "command": "evaluate_reconstruction_v13.py --source generated --max-samples 100 --write-step --validate-step",
        },
    ]


def build_training_phase_budget(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    progress = progress_summary(read_json(root / DEFAULT_PROGRESS))
    stage = stage_summary(read_json(root / DEFAULT_STAGE_DECISION))
    next_stage = stage["next_stage"]
    ready = next_stage == "vqvae_complex_curved_recovery" and not stage["can_train_ar_now"]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY_FOR_VQVAE_FIRST_SERVER_PLAN" if ready else "INSPECT_CURRENT_GATES_BEFORE_SERVER_PLAN",
        "repo_root": str(root),
        "progress": progress,
        "stage_decision": stage,
        "recommended_first_gpu": {
            "preferred": "1x L40S 48GB or 1x RTX 6000 Ada/A6000 48GB",
            "minimum_gpu_memory_gb": 40,
            "upgrade": "1x A100 80GB when AR2048 or larger VQ-VAE batches are memory-limited",
            "deadline_speed": "1x H100 80GB when speed matters more than cost",
        },
        "local_machine_policy": {
            "role": "debug_render_reconstruct_latex_smoke_tests_only",
            "reason": "RTX 3060 12GB is below the 40GB recovery-readiness gate",
        },
        "hardware_tiers": build_hardware_tiers(),
        "paper_policy": {
            "current_g20_g100_role": "failure_analysis_only",
            "positive_figures_allowed": bool(stage["positive_figures_allowed"] and progress["positive_figures_allowed"]),
        },
        "phases": build_phases(next_stage),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Training Phase Budget",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Current stage: `{payload['progress']['current_stage']}`",
        f"- Next stage: `{payload['stage_decision']['next_stage']}`",
        f"- Preferred first GPU: {payload['recommended_first_gpu']['preferred']}",
        f"- Minimum readiness GPU memory: {payload['recommended_first_gpu']['minimum_gpu_memory_gb']}GB",
        "",
        "Do not train AR before VQ-VAE promotion. The first expensive server job should be VQ-VAE complex/curved recovery.",
        "",
        "## Hardware",
        "",
        "- First choice: 1x L40S 48GB or 1x RTX 6000 Ada/A6000 48GB.",
        "- Upgrade: 1x A100 80GB if AR2048 or larger VQ-VAE batches are memory-limited.",
        "- Speed option: 1x H100 80GB when deadline speed matters more than cost.",
        "- Local RTX 3060 12GB: debugging, reconstruction, rendering, LaTeX, and smoke tests only.",
        "",
        "## Hardware Tiers",
        "",
        "| Hardware | VRAM GB | Decision | Speed tier | Best use |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for tier in payload["hardware_tiers"]:
        lines.append(
            f"| {tier['name']} | {tier['memory_gb']} | {tier['decision']} | "
            f"{tier['speed_tier']} | {tier['best_use']} |"
        )
    lines.extend(
        [
            "",
        "## Phases",
        "",
        ]
    )
    for index, phase in enumerate(payload["phases"], start=1):
        lines.extend(
            [
                f"### {index}. {phase['id']}",
                "",
                f"- Gate: `{phase['gate']}`",
                f"- Recommended GPU: `{phase['recommended_gpu']}`",
                f"- Acceptance: {phase['acceptance']}",
            ]
        )
        if phase.get("stop_if"):
            lines.append("- Stop if: " + ", ".join(f"`{item}`" for item in phase["stop_if"]))
        lines.extend(["", "```bash", phase["command"], "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V13 rented-server training phase budget.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_training_phase_budget(args.repo_root)
    if args.output:
        write_json(args.output, payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["status"] == "READY_FOR_VQVAE_FIRST_SERVER_PLAN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
