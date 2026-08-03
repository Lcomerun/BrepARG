"""Run the server-side preflight gates for V13 quality recovery.

Default behavior is conservative: run transfer verification, training
readiness, artifact sanity, and write the start plan. The VQ-VAE recovery job is
launched only when ``start_training``/``--start`` is explicitly enabled and the
start plan is ready.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from plan_server_quality_recovery import build_server_recovery_plan, render_markdown as render_plan_markdown, write_json
from verify_server_training_readiness import DEFAULT_MIN_GPU_MEMORY_GB
from verify_server_transfer import parse_path_map, resolve_server_path


Runner = Callable[..., Any]


def completed_returncode(completed: Any) -> int:
    return int(getattr(completed, "returncode", 0))


def run_gate_command(command: Sequence[str], *, runner: Runner, repo_root: Path) -> dict[str, Any]:
    completed = runner(list(command), check=False, capture_output=True, text=True, cwd=str(repo_root))
    return {
        "command": list(command),
        "returncode": completed_returncode(completed),
        "stdout": str(getattr(completed, "stdout", "") or "")[-2000:],
        "stderr": str(getattr(completed, "stderr", "") or "")[-2000:],
    }


def server_report_paths(reports_dir: Path) -> dict[str, Path]:
    return {
        "transfer": reports_dir / "v13_server_transfer_verify_server.json",
        "transfer_md": reports_dir / "v13_server_transfer_verify_server.md",
        "readiness": reports_dir / "v13_server_training_readiness_server.json",
        "readiness_md": reports_dir / "v13_server_training_readiness_server.md",
        "artifacts": reports_dir / "v13_model_artifact_sanity_server.json",
        "artifacts_md": reports_dir / "v13_model_artifact_sanity_server.md",
        "plan": reports_dir / "v13_server_quality_recovery_start_plan.json",
        "plan_md": reports_dir / "v13_server_quality_recovery_start_plan.md",
    }


def local_dry_run_path(path: str | Path, path_map: dict[str, str | Path]) -> Path:
    return resolve_server_path(str(path), path_map) if path_map else Path(path)


def format_gpu_memory_gb(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def run_quality_recovery_preflight(
    *,
    repo_root: str | Path = "/workspace/V13",
    python: str = "/opt/conda/bin/python",
    reports_dir: str | Path | None = None,
    transfer_manifest: str | Path | None = None,
    parsed_pool: str | Path = "/workspace/ABC/processed/abc_parsed_full",
    vqvae_checkpoint: str | Path = "/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
    ar_checkpoint: str | Path = "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt",
    sequence: str | Path = "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
    split: str | Path = "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl",
    transfer_path_maps: Sequence[str] | None = None,
    skip_data_requirements: bool = False,
    min_gpu_memory_gb: float = DEFAULT_MIN_GPU_MEMORY_GB,
    start_training: bool = False,
    command_runner: Runner = subprocess.run,
) -> dict[str, Any]:
    root = Path(repo_root)
    report_root = Path(reports_dir) if reports_dir is not None else root / "local_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    paths = server_report_paths(report_root)
    manifest = Path(transfer_manifest) if transfer_manifest is not None else report_root / "v13_server_transfer_manifest_20260706.json"
    dry_run_path_map = parse_path_map(list(transfer_path_maps or []))
    gate_parsed_pool = local_dry_run_path(parsed_pool, dry_run_path_map)
    gate_vqvae_checkpoint = local_dry_run_path(vqvae_checkpoint, dry_run_path_map)
    gate_ar_checkpoint = local_dry_run_path(ar_checkpoint, dry_run_path_map)
    gate_sequence = local_dry_run_path(sequence, dry_run_path_map)
    gate_split = local_dry_run_path(split, dry_run_path_map)

    commands = {
        "transfer_verification": [
            python,
            "tools/verify_server_transfer.py",
            "--manifest",
            str(manifest),
            "--output",
            str(paths["transfer"]),
            "--markdown-output",
            str(paths["transfer_md"]),
        ],
        "training_readiness": [
            python,
            "tools/verify_server_training_readiness.py",
            "--repo-root",
            str(root),
            "--parsed-pool",
            str(gate_parsed_pool),
            "--vqvae-checkpoint",
            str(gate_vqvae_checkpoint),
            "--sequence",
            str(gate_sequence),
            "--split",
            str(gate_split),
            "--transfer-verification",
            str(paths["transfer"]),
            "--min-gpu-memory-gb",
            format_gpu_memory_gb(min_gpu_memory_gb),
            "--output",
            str(paths["readiness"]),
            "--markdown-output",
            str(paths["readiness_md"]),
        ],
        "artifact_sanity": [
            python,
            "tools/verify_model_artifacts.py",
            "--vqvae-checkpoint",
            str(gate_vqvae_checkpoint),
            "--ar-checkpoint",
            str(gate_ar_checkpoint),
            "--sequence",
            str(gate_sequence),
            "--split",
            str(gate_split),
            "--output",
            str(paths["artifacts"]),
            "--markdown-output",
            str(paths["artifacts_md"]),
        ],
    }
    if transfer_path_maps:
        for item in transfer_path_maps:
            commands["transfer_verification"].extend(["--path-map", str(item)])
    if skip_data_requirements:
        commands["transfer_verification"].append("--skip-data-requirements")

    stage_results: dict[str, Any] = {}
    for stage, command in commands.items():
        stage_results[stage] = run_gate_command(command, runner=command_runner, repo_root=root)

    plan = build_server_recovery_plan(
        transfer_verification=paths["transfer"],
        training_readiness=paths["readiness"],
        artifact_sanity=paths["artifacts"],
        repo_root=str(root),
        python=python,
        parsed_pool=str(parsed_pool),
        sequence=str(sequence),
    )
    write_json(paths["plan"], plan)
    paths["plan_md"].write_text(render_plan_markdown(plan), encoding="utf-8")

    started_recovery = False
    recovery_returncode: int | None = None
    if start_training and plan["training_start_allowed"]:
        recovery_command = plan["recovery_commands"][0]
        completed = command_runner(recovery_command, check=False, shell=True, text=True, cwd=str(root))
        started_recovery = True
        recovery_returncode = completed_returncode(completed)

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "RECOVERY_STARTED" if started_recovery else plan["status"],
        "started_recovery": started_recovery,
        "recovery_returncode": recovery_returncode,
        "executed_stages": list(commands.keys()),
        "stage_results": stage_results,
        "plan": plan,
        "reports": {key: str(path) for key, path in paths.items()},
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Quality-Recovery Preflight",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Started recovery: {payload['started_recovery']}",
        f"- Plan status: `{payload['plan']['status']}`",
        "",
        "## Gate Commands",
        "",
    ]
    for stage in payload["executed_stages"]:
        result = payload["stage_results"][stage]
        lines.append(f"- {stage}: return code {result['returncode']}")
    lines.extend(["", "## Reports", ""])
    for label, path in payload["reports"].items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V13 server quality-recovery preflight gates.")
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/V13"))
    parser.add_argument("--python", default="/opt/conda/bin/python")
    parser.add_argument("--reports-dir", type=Path, default=Path("/workspace/V13/local_reports"))
    parser.add_argument("--transfer-manifest", type=Path, default=Path("/workspace/V13/local_reports/v13_server_transfer_manifest_20260706.json"))
    parser.add_argument("--parsed-pool", type=Path, default=Path("/workspace/ABC/processed/abc_parsed_full"))
    parser.add_argument("--vqvae-checkpoint", type=Path, default=Path("/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt"))
    parser.add_argument("--ar-checkpoint", type=Path, default=Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt"))
    parser.add_argument("--sequence", type=Path, default=Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl"))
    parser.add_argument("--split", type=Path, default=Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl"))
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional local dry-run path map passed through to verify_server_transfer.py.",
    )
    parser.add_argument(
        "--skip-data-requirements",
        action="store_true",
        help="Optional local dry-run flag passed through to verify_server_transfer.py.",
    )
    parser.add_argument(
        "--min-gpu-memory-gb",
        type=float,
        default=DEFAULT_MIN_GPU_MEMORY_GB,
        help="Minimum largest-GPU memory required by the training-readiness gate.",
    )
    parser.add_argument("--start", action="store_true", help="Launch VQ-VAE recovery only if every gate is ready.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_quality_recovery_preflight(
        repo_root=args.repo_root,
        python=args.python,
        reports_dir=args.reports_dir,
        transfer_manifest=args.transfer_manifest,
        parsed_pool=args.parsed_pool,
        vqvae_checkpoint=args.vqvae_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        sequence=args.sequence,
        split=args.split,
        transfer_path_maps=args.path_map,
        skip_data_requirements=args.skip_data_requirements,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        start_training=args.start,
    )
    if args.output:
        write_json(args.output, payload)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if payload["started_recovery"]:
        return 0 if payload["recovery_returncode"] == 0 else 2
    return 0 if payload["plan"]["training_start_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
