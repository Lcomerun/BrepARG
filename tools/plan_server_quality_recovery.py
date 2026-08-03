"""Build the gated server-side command plan for V13 quality recovery.

This is a small orchestration guardrail. It reads the three server-side gate
reports that must pass before spending GPU time and emits the VQ-VAE recovery
and monitor commands only when all gates are ready.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_TRANSFER_VERIFICATION = Path("/workspace/V13/local_reports/v13_server_transfer_verify_server.json")
DEFAULT_TRAINING_READINESS = Path("/workspace/V13/local_reports/v13_server_training_readiness_server.json")
DEFAULT_ARTIFACT_SANITY = Path("/workspace/V13/local_reports/v13_model_artifact_sanity_server.json")

DEFAULT_REPO_ROOT = "/workspace/V13"
DEFAULT_PYTHON = "/opt/conda/bin/python"
DEFAULT_PARSED_POOL = "/workspace/ABC/processed/abc_parsed_full"
DEFAULT_OUTBASE = "/workspace/ABC/processed/train_outputs"
DEFAULT_RUN_NAME = "newscheme_full_vqvae_complex_recovery"
DEFAULT_SEQUENCE = "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl"
DEFAULT_BENCHMARK_OUTPUT_ROOT = "/workspace/V13/local_runs/reconstruction_eval"
DEFAULT_BENCHMARK_PREFIX = "vq_complex_recovery_20260706"
DEFAULT_TARGET_EPOCH = 180


def read_gate_report(path: str | Path, expected_status: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {
            "path": str(resolved),
            "expected_status": expected_status,
            "actual_status": "missing",
            "ready": False,
            "reason": "missing_report",
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": str(resolved),
            "expected_status": expected_status,
            "actual_status": "invalid_json",
            "ready": False,
            "reason": str(exc),
        }
    actual = str(payload.get("status", "unknown"))
    return {
        "path": str(resolved),
        "expected_status": expected_status,
        "actual_status": actual,
        "ready": actual == expected_status,
        "reason": "ready" if actual == expected_status else "unexpected_status",
    }


def vqvae_recovery_command(
    *,
    repo_root: str,
    python: str,
    parsed_pool: str,
    outbase: str,
    run_name: str,
    sequence: str,
    benchmark_output_root: str,
    benchmark_prefix: str,
) -> str:
    return "\n".join(
        [
            "bash tools/run_vqvae_complex_recovery.sh \\",
            f"  --repo-root {repo_root} \\",
            f"  --python {python} \\",
            f"  --pool {parsed_pool} \\",
            f"  --outbase {outbase} \\",
            f"  --run-name {run_name} \\",
            "  --samples 450000 \\",
            "  --epochs 80 \\",
            "  --batch-size 128 \\",
            "  --lr 1e-4 \\",
            "  --complex-fraction 0.50 \\",
            "  --complex-min-faces 12 \\",
            "  --complex-min-edges 20 \\",
            "  --curved-fraction 0.35 \\",
            "  --run-benchmark \\",
            f"  --sequence {sequence} \\",
            f"  --benchmark-output-root {benchmark_output_root} \\",
            f"  --benchmark-prefix {benchmark_prefix}",
        ]
    )


def recovery_monitor_command(
    *,
    python: str,
    outbase: str,
    run_name: str,
    benchmark_output_root: str,
    benchmark_prefix: str,
    target_epoch: int,
) -> str:
    run_dir = f"{outbase.rstrip('/')}/{run_name}"
    return "\n".join(
        [
            f"{python} tools/monitor_vqvae_recovery_gate.py \\",
            f"  --run-dir {run_dir} \\",
            f"  --benchmark-summary {benchmark_output_root.rstrip('/')}/{benchmark_prefix}_benchmark_summary.json \\",
            f"  --copy-back-manifest {run_dir}/copy_back_manifest.json \\",
            f"  --target-epoch {int(target_epoch)} \\",
            "  --interval-seconds 300 \\",
            "  --status-log /workspace/V13/local_reports/vqvae_recovery_monitor_20260706.jsonl",
        ]
    )


def build_server_recovery_plan(
    *,
    transfer_verification: str | Path = DEFAULT_TRANSFER_VERIFICATION,
    training_readiness: str | Path = DEFAULT_TRAINING_READINESS,
    artifact_sanity: str | Path = DEFAULT_ARTIFACT_SANITY,
    repo_root: str = DEFAULT_REPO_ROOT,
    python: str = DEFAULT_PYTHON,
    parsed_pool: str = DEFAULT_PARSED_POOL,
    outbase: str = DEFAULT_OUTBASE,
    run_name: str = DEFAULT_RUN_NAME,
    sequence: str = DEFAULT_SEQUENCE,
    benchmark_output_root: str = DEFAULT_BENCHMARK_OUTPUT_ROOT,
    benchmark_prefix: str = DEFAULT_BENCHMARK_PREFIX,
    target_epoch: int = DEFAULT_TARGET_EPOCH,
) -> dict[str, Any]:
    gates = {
        "transfer_verification": read_gate_report(transfer_verification, "READY_FOR_SERVER_RUN"),
        "training_readiness": read_gate_report(training_readiness, "READY_FOR_VQVAE_TRAINING"),
        "artifact_sanity": read_gate_report(artifact_sanity, "MODEL_ARTIFACTS_READY"),
    }
    blocking_reasons = [
        f"{name}_not_ready"
        for name, gate in gates.items()
        if not gate["ready"]
    ]
    training_start_allowed = not blocking_reasons
    recovery_commands: list[str] = []
    if training_start_allowed:
        recovery_commands = [
            vqvae_recovery_command(
                repo_root=repo_root,
                python=python,
                parsed_pool=parsed_pool,
                outbase=outbase,
                run_name=run_name,
                sequence=sequence,
                benchmark_output_root=benchmark_output_root,
                benchmark_prefix=benchmark_prefix,
            ),
            recovery_monitor_command(
                python=python,
                outbase=outbase,
                run_name=run_name,
                benchmark_output_root=benchmark_output_root,
                benchmark_prefix=benchmark_prefix,
                target_epoch=target_epoch,
            ),
        ]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY_TO_START_VQVAE_RECOVERY" if training_start_allowed else "HOLD_BEFORE_VQVAE_RECOVERY",
        "training_start_allowed": bool(training_start_allowed),
        "blocking_reasons": blocking_reasons,
        "gates": gates,
        "next_action": "start_vqvae_recovery_and_monitor" if training_start_allowed else "fix_failed_gate_reports_before_training",
        "precheck_order": [
            "verify_server_transfer",
            "verify_server_training_readiness",
            "verify_model_artifacts",
            "plan_server_quality_recovery",
        ],
        "recovery_commands": recovery_commands,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Quality-Recovery Start Plan",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Training start allowed: {payload['training_start_allowed']}",
        f"- Next action: `{payload['next_action']}`",
        "",
        "## Gates",
        "",
        "| Gate | Expected | Actual | Ready |",
        "|---|---|---|---:|",
    ]
    for name, gate in payload["gates"].items():
        lines.append(
            f"| {name} | `{gate['expected_status']}` | `{gate['actual_status']}` | {gate['ready']} |"
        )
    lines.extend(["", "## Recovery Commands", ""])
    if not payload["recovery_commands"]:
        lines.append("No recovery command is emitted until every gate is ready.")
    else:
        for command in payload["recovery_commands"]:
            lines.append(f"```bash\n{command}\n```")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the gated V13 server quality-recovery command plan.")
    parser.add_argument("--transfer-verification", type=Path, default=DEFAULT_TRANSFER_VERIFICATION)
    parser.add_argument("--training-readiness", type=Path, default=DEFAULT_TRAINING_READINESS)
    parser.add_argument("--artifact-sanity", type=Path, default=DEFAULT_ARTIFACT_SANITY)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--parsed-pool", default=DEFAULT_PARSED_POOL)
    parser.add_argument("--outbase", default=DEFAULT_OUTBASE)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--benchmark-output-root", default=DEFAULT_BENCHMARK_OUTPUT_ROOT)
    parser.add_argument("--benchmark-prefix", default=DEFAULT_BENCHMARK_PREFIX)
    parser.add_argument("--target-epoch", type=int, default=DEFAULT_TARGET_EPOCH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_server_recovery_plan(
        transfer_verification=args.transfer_verification,
        training_readiness=args.training_readiness,
        artifact_sanity=args.artifact_sanity,
        repo_root=args.repo_root,
        python=args.python,
        parsed_pool=args.parsed_pool,
        outbase=args.outbase,
        run_name=args.run_name,
        sequence=args.sequence,
        benchmark_output_root=args.benchmark_output_root,
        benchmark_prefix=args.benchmark_prefix,
        target_epoch=args.target_epoch,
    )
    if args.output:
        write_json(args.output, plan)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=True))
    return 0 if plan["training_start_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
