"""Build a compact execution packet for the next V13 rented-server run.

The packet is a human-readable and machine-readable command card. It does not
copy files or start training; it gathers the current stage decision, transfer
manifest, preflight status, and the safest next commands for a VQ-VAE-first
quality-recovery run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_SERVER_REPO_ROOT = "/workspace/V13"
DEFAULT_SERVER_PYTHON = "/opt/conda/bin/python"
DEFAULT_REPORTS_DIR = "/workspace/V13/local_reports"
DEFAULT_TRANSFER_MANIFEST = "/workspace/V13/local_reports/v13_server_transfer_manifest_20260706.json"
DEFAULT_MIN_GPU_MEMORY_GB = 40.0
DEFAULT_VQVAE_RUN_DIR = "/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery"
DEFAULT_VQVAE_BENCHMARK_SUMMARY = "/workspace/V13/local_runs/reconstruction_eval/vq_complex_recovery_20260706_benchmark_summary.json"
DEFAULT_COPYBACK_MANIFEST = f"{DEFAULT_VQVAE_RUN_DIR}/copy_back_manifest.json"

LOCAL_STAGE_DECISION = "local_reports/v13_quality_recovery_stage_decision_20260706.json"
LOCAL_TRANSFER_MANIFEST = "local_reports/v13_server_transfer_manifest_20260706.json"
LOCAL_HANDOFF_PREFLIGHT = "local_reports/v13_server_handoff_preflight_20260706.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def transfer_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "missing",
            "ready": False,
            "entries_total": 0,
            "repo_source_entries": 0,
            "repo_tooling_entries": 0,
        }
    entries = list(payload.get("entries") or [])
    return {
        "status": str(payload.get("status", "unknown")),
        "ready": bool(payload.get("required_sources_complete")) and payload.get("status") == "READY_TO_TRANSFER",
        "entries_total": len(entries),
        "repo_source_entries": sum(1 for item in entries if item.get("transfer_group") == "repo_source"),
        "repo_tooling_entries": sum(1 for item in entries if item.get("transfer_group") == "repo_tooling"),
        "missing_entries": [item.get("label", "unknown") for item in payload.get("missing_entries") or []],
    }


def stage_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "missing",
            "next_stage": "unknown",
            "can_train_ar_now": False,
            "positive_figures_allowed": False,
            "blocking_reasons": ["missing_stage_decision"],
        }
    return {
        "status": str(payload.get("status", "unknown")),
        "next_stage": str(payload.get("next_stage", "unknown")),
        "can_train_ar_now": bool(payload.get("can_train_ar_now")),
        "positive_figures_allowed": bool((payload.get("paper_figure_policy") or {}).get("positive_figures_allowed")),
        "blocking_reasons": list(payload.get("blocking_reasons") or []),
    }


def preflight_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "missing", "ready": False}
    return {
        "status": str(payload.get("status", "unknown")),
        "ready": bool(payload.get("required_artifacts_complete"))
        and payload.get("status") == "READY_FOR_SERVER_VQVAE_RECOVERY",
    }


def guarded_preflight_command(
    *,
    server_repo_root: str,
    server_python: str,
    reports_dir: str,
    transfer_manifest: str,
    min_gpu_memory_gb: float = DEFAULT_MIN_GPU_MEMORY_GB,
    start: bool,
) -> str:
    min_memory_text = str(int(min_gpu_memory_gb)) if float(min_gpu_memory_gb).is_integer() else str(min_gpu_memory_gb)
    lines = [
        f"{server_python} tools/run_server_quality_recovery.py \\",
        f"  --repo-root {server_repo_root} \\",
        f"  --python {server_python} \\",
        f"  --reports-dir {reports_dir} \\",
        f"  --transfer-manifest {transfer_manifest} \\",
        f"  --min-gpu-memory-gb {min_memory_text} \\",
        f"  --output {reports_dir}/v13_server_quality_recovery_preflight.json \\",
        f"  --markdown-output {reports_dir}/v13_server_quality_recovery_preflight.md",
    ]
    if start:
        lines[-1] += " \\"
        lines.append("  --start")
    return "\n".join(lines)


def monitor_command(*, server_python: str, run_dir: str, benchmark_summary: str, copyback_manifest: str) -> str:
    return "\n".join(
        [
            f"{server_python} tools/monitor_vqvae_recovery_gate.py \\",
            f"  --run-dir {run_dir} \\",
            f"  --benchmark-summary {benchmark_summary} \\",
            f"  --copy-back-manifest {copyback_manifest} \\",
            "  --target-epoch 180 \\",
            "  --interval-seconds 300 \\",
            "  --status-log /workspace/V13/local_reports/vqvae_recovery_monitor_20260706.jsonl",
        ]
    )


def progress_summary_command(*, server_python: str, reports_dir: str) -> str:
    return "\n".join(
        [
            f"{server_python} tools/summarize_quality_recovery_progress.py \\",
            "  --repo-root /workspace/V13 \\",
            f"  --preflight {reports_dir}/v13_server_quality_recovery_preflight.json \\",
            f"  --output {reports_dir}/v13_quality_recovery_progress_server.json \\",
            f"  --markdown-output {reports_dir}/v13_quality_recovery_progress_server.md",
        ]
    )


def local_upload_command() -> str:
    return "\n".join(
        [
            "$Remote = 'USER@SERVER'",
            "& 'C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe' tools\\build_server_transfer_manifest.py `",
            "  --repo-root . `",
            "  --remote $Remote `",
            "  --output local_reports\\v13_server_transfer_manifest_20260706.json `",
            "  --markdown-output local_reports\\v13_server_transfer_manifest_20260706.md `",
            "  --powershell-upload-output local_reports\\v13_server_upload_from_manifest_20260706.ps1",
            "& .\\local_reports\\v13_server_upload_from_manifest_20260706.ps1",
        ]
    )


def local_phase_budget_command() -> str:
    return "\n".join(
        [
            "& 'C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe' tools\\build_server_training_phase_budget.py `",
            "  --repo-root . `",
            "  --output local_reports\\v13_server_training_phase_budget_20260706.json `",
            "  --markdown-output local_reports\\v13_server_training_phase_budget_20260706.md",
        ]
    )


def local_copyback_command() -> str:
    return "\n".join(
        [
            "& 'C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe' tools\\verify_vqvae_copyback.py `",
            "  --manifest ABC\\processed\\train_outputs\\newscheme_full_vqvae_complex_recovery\\copy_back_manifest.json `",
            "  --repo-root . `",
            "  --path-map /workspace/V13=. `",
            "  --path-map /workspace/ABC/processed/train_outputs=ABC/processed/train_outputs `",
            "  --output local_reports\\v13_vqvae_copyback_verify_local_20260706.json `",
            "  --markdown-output local_reports\\v13_vqvae_copyback_verify_local_20260706.md",
        ]
    )


def local_pull_copyback_command() -> str:
    return "\n".join(
        [
            "$Remote = 'USER@SERVER'",
            "New-Item -ItemType Directory -Force ABC\\processed\\train_outputs\\newscheme_full_vqvae_complex_recovery | Out-Null",
            "New-Item -ItemType Directory -Force local_runs\\reconstruction_eval | Out-Null",
            "New-Item -ItemType Directory -Force local_reports | Out-Null",
            "rsync -av --progress \"${Remote}:'/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery/'\" 'ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery/'",
            "rsync -av --progress \"${Remote}:'/workspace/V13/local_runs/reconstruction_eval/'\" 'local_runs/reconstruction_eval/'",
            "rsync -av --progress \"${Remote}:'/workspace/V13/local_reports/'\" 'local_reports/'",
            "if (-not (Test-Path 'ABC\\processed\\train_outputs\\newscheme_full_vqvae_complex_recovery\\copy_back_manifest.json')) {",
            "  throw 'copy_back_manifest.json was not copied back; do not delete the server yet.'",
            "}",
        ]
    )


def source_path_rebuild_command() -> str:
    return "\n".join(
        [
            "bash tools/run_source_path_sequence_rebuild.sh \\",
            "  --repo-root /workspace/V13 \\",
            "  --python /opt/conda/bin/python \\",
            "  --outbase /workspace/V13/local_runs/ar_training/train_outputs \\",
            "  --run-name newscheme_full_v13_sourcepath_sequence \\",
            "  --split /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl \\",
            "  --vqvae-checkpoint /workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery/fsq_vqvae_best.pt \\",
            "  --workers 4 \\",
            "  --resume",
        ]
    )


def source_path_length_coverage_command() -> str:
    return "\n".join(
        [
            "/opt/conda/bin/python tools/summarize_ar_length_coverage.py \\",
            "  /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/sequences_fsq_rcm.pkl \\",
            "  --limits 1024,1536,2048 \\",
            "  --output /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/length_coverage.json \\",
            "  --markdown-output /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/length_coverage.md",
        ]
    )


def sequence_rebuild_verify_command() -> str:
    return "\n".join(
        [
            "/opt/conda/bin/python tools/verify_sequence_rebuild.py \\",
            "  --sequence /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/sequences_fsq_rcm.pkl \\",
            "  --source-path-audit /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/source_path_audit.json \\",
            "  --length-coverage /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/length_coverage.json \\",
            "  --output /workspace/V13/local_reports/v13_sequence_rebuild_verify_server.json \\",
            "  --markdown-output /workspace/V13/local_reports/v13_sequence_rebuild_verify_server.md",
        ]
    )


def ar1536_command() -> str:
    return "\n".join(
        [
            "bash tools/run_ar_v13_long_context.sh \\",
            "  --repo-root /workspace/V13 \\",
            "  --python /opt/conda/bin/python \\",
            "  --outbase /workspace/V13/local_runs/ar_training/train_outputs \\",
            "  --run-name newscheme_full_v13_ar1536 \\",
            "  --sequence-source /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/sequences_fsq_rcm.pkl \\",
            "  --split-source /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl \\",
            "  --target-epochs 120 \\",
            "  --lr 5e-4 \\",
            "  --max-seq-len 1536 \\",
            "  --batch-size 8 \\",
            "  --no-auto-resume",
        ]
    )


def ar2048_command() -> str:
    return "\n".join(
        [
            "bash tools/run_ar_v13_long_context.sh \\",
            "  --repo-root /workspace/V13 \\",
            "  --python /opt/conda/bin/python \\",
            "  --outbase /workspace/V13/local_runs/ar_training/train_outputs \\",
            "  --run-name newscheme_full_v13_ar2048 \\",
            "  --sequence-source /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/sequences_fsq_rcm.pkl \\",
            "  --split-source /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl \\",
            "  --target-epochs 120 \\",
            "  --lr 5e-4 \\",
            "  --max-seq-len 2048 \\",
            "  --batch-size 4 \\",
            "  --no-auto-resume",
        ]
    )


def generated_reconstruction_command(*, branch: str, run_name: str, max_new_tokens: int) -> str:
    return "\n".join(
        [
            "/opt/conda/bin/python tools/evaluate_reconstruction_v13.py \\",
            "  --sequence /workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_sourcepath_sequence/sequences_fsq_rcm.pkl \\",
            f"  --ar-checkpoint /workspace/V13/local_runs/ar_training/train_outputs/{branch}/ar_best.pt \\",
            "  --vqvae-checkpoint /workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery/fsq_vqvae_best.pt \\",
            "  --source generated \\",
            "  --max-samples 100 \\",
            "  --device cuda \\",
            "  --constrained-decoding \\",
            f"  --max-new-tokens {int(max_new_tokens)} \\",
            "  --temperature 0.9 \\",
            "  --top-p 0.92 \\",
            "  --write-step \\",
            "  --validate-step \\",
            f"  --run-name {run_name}",
        ]
    )


def render_generated_command(run_name: str) -> str:
    return "\n".join(
        [
            "/opt/conda/bin/python papers/aaai_v13/render_step_directory.py \\",
            f"  /workspace/V13/local_runs/reconstruction_eval/{run_name} \\",
            "  --cols 5",
        ]
    )


def generated_quality_gate_command(run_name: str) -> str:
    return "\n".join(
        [
            "/opt/conda/bin/python tools/summarize_generated_quality.py \\",
            f"  /workspace/V13/local_runs/reconstruction_eval/{run_name}",
        ]
    )


def step_geometry_entity_audit_command(run_name: str) -> str:
    run_dir = f"/workspace/V13/local_runs/reconstruction_eval/{run_name}"
    return "\n".join(
        [
            "/opt/conda/bin/python tools/audit_step_geometry_entities.py \\",
            f"  {run_dir} \\",
            f"  --output {run_dir}/step_geometry_entity_audit.json \\",
            f"  --markdown-output {run_dir}/step_geometry_entity_audit.md",
        ]
    )


def server_paper_figure_audit_command(run_name: str) -> str:
    run_dir = f"/workspace/V13/local_runs/reconstruction_eval/{run_name}"
    return "\n".join(
        [
            "set +e",
            "/opt/conda/bin/python tools/audit_paper_figure_candidates.py \\",
            f"  {run_dir} \\",
            f"  --output {run_dir}/paper_figure_candidate_audit.json \\",
            f"  --markdown-output {run_dir}/paper_figure_candidate_audit.md",
            "audit_status=$?",
            f"if [ ! -s {run_dir}/paper_figure_candidate_audit.json ]; then",
            '  if [ "$audit_status" -eq 0 ]; then exit 1; else exit "$audit_status"; fi',
            "fi",
            f"if [ ! -s {run_dir}/paper_figure_candidate_audit.md ]; then",
            '  if [ "$audit_status" -eq 0 ]; then exit 1; else exit "$audit_status"; fi',
            "fi",
            'if [ "$audit_status" -ne 0 ]; then',
            "  echo 'Paper figure audit held this run; copy the audit files back for failure analysis.'",
            "fi",
            "exit 0",
        ]
    )


def local_pull_generated_command(run_name: str) -> str:
    local_run = f"local_runs/reconstruction_eval/{run_name}"
    return "\n".join(
        [
            "$Remote = 'USER@SERVER'",
            "New-Item -ItemType Directory -Force local_runs\\reconstruction_eval | Out-Null",
            (
                f"rsync -av --progress "
                f"\"${{Remote}}:'/workspace/V13/local_runs/reconstruction_eval/{run_name}/'\" "
                f"'{local_run}/'"
            ),
            f"if (-not (Test-Path '{local_run}\\generated_quality_summary.json')) {{",
            f"  throw 'generated_quality_summary.json was not copied back for {run_name}.'",
            "}",
            f"if (-not (Test-Path '{local_run}\\step_geometry_entity_audit.json')) {{",
            f"  throw 'step_geometry_entity_audit.json was not copied back for {run_name}.'",
            "}",
            f"if (-not (Test-Path '{local_run}\\step_geometry_entity_audit.md')) {{",
            f"  throw 'step_geometry_entity_audit.md was not copied back for {run_name}.'",
            "}",
            f"if (-not (Test-Path '{local_run}\\renders\\contact_sheet.png')) {{",
            f"  throw 'Rendered contact sheet was not copied back for {run_name}.'",
            "}",
            f"if (-not (Test-Path '{local_run}\\paper_figure_candidate_audit.json')) {{",
            f"  throw 'paper_figure_candidate_audit.json was not copied back for {run_name}.'",
            "}",
            f"if (-not (Test-Path '{local_run}\\paper_figure_candidate_audit.md')) {{",
            f"  throw 'paper_figure_candidate_audit.md was not copied back for {run_name}.'",
            "}",
        ]
    )


def paper_figure_audit_command(*, run_name: str, branch_label: str) -> str:
    local_run = f"local_runs\\reconstruction_eval\\{run_name}"
    return "\n".join(
        [
            "& 'C:\\Users\\YU\\.conda\\envs\\brepgen_env\\python.exe' tools\\audit_paper_figure_candidates.py `",
            f"  {local_run} `",
            f"  --output local_reports\\v13_paper_figure_candidate_audit_{branch_label}_20260706.json `",
            f"  --markdown-output local_reports\\v13_paper_figure_candidate_audit_{branch_label}_20260706.md",
        ]
    )


def build_server_recovery_packet(
    repo_root: str | Path,
    *,
    server_repo_root: str = DEFAULT_SERVER_REPO_ROOT,
    server_python: str = DEFAULT_SERVER_PYTHON,
    reports_dir: str = DEFAULT_REPORTS_DIR,
    transfer_manifest: str = DEFAULT_TRANSFER_MANIFEST,
    min_gpu_memory_gb: float = DEFAULT_MIN_GPU_MEMORY_GB,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    stage = stage_summary(read_json(root / LOCAL_STAGE_DECISION))
    transfer = transfer_summary(read_json(root / LOCAL_TRANSFER_MANIFEST))
    preflight = preflight_summary(read_json(root / LOCAL_HANDOFF_PREFLIGHT))
    ready = (
        stage["next_stage"] == "vqvae_complex_curved_recovery"
        and not stage["can_train_ar_now"]
        and transfer["ready"]
        and preflight["ready"]
    )
    status = "READY_FOR_RENTED_SERVER_VQVAE_RECOVERY_PACKET" if ready else "HOLD_BEFORE_RENTED_SERVER_PACKET"
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "repo_root": str(root),
        "stage_decision": stage,
        "transfer": transfer,
        "handoff_preflight": preflight,
        "policy": {
            "first_training_stage": "vqvae_complex_curved_recovery",
            "do_not_train_ar_first": True,
            "positive_paper_figures_allowed": stage["positive_figures_allowed"],
        },
        "immediate_commands": {
            "local_refresh_training_phase_budget": {
                "where": "local_before_server",
                "command": local_phase_budget_command(),
            },
            "local_upload_transfer_manifest": {
                "where": "local_before_server",
                "command": local_upload_command(),
            },
            "server_guarded_preflight": {
                "where": "server",
                "command": guarded_preflight_command(
                    server_repo_root=server_repo_root,
                    server_python=server_python,
                    reports_dir=reports_dir,
                    transfer_manifest=transfer_manifest,
                    min_gpu_memory_gb=min_gpu_memory_gb,
                    start=False,
                ),
            },
            "server_progress_summary": {
                "where": "server",
                "guard": "run after server_guarded_preflight to summarize the current hold/start state",
                "command": progress_summary_command(
                    server_python=server_python,
                    reports_dir=reports_dir,
                ),
            },
            "server_start_vqvae_recovery": {
                "where": "server",
                "guard": "run only after server_guarded_preflight reports ready gates",
                "command": guarded_preflight_command(
                    server_repo_root=server_repo_root,
                    server_python=server_python,
                    reports_dir=reports_dir,
                    transfer_manifest=transfer_manifest,
                    min_gpu_memory_gb=min_gpu_memory_gb,
                    start=True,
                ),
            },
            "server_monitor_vqvae_recovery": {
                "where": "server",
                "command": monitor_command(
                    server_python=server_python,
                    run_dir=DEFAULT_VQVAE_RUN_DIR,
                    benchmark_summary=DEFAULT_VQVAE_BENCHMARK_SUMMARY,
                    copyback_manifest=DEFAULT_COPYBACK_MANIFEST,
                ),
            },
            "local_pull_vqvae_copyback": {
                "where": "local_after_monitor_success",
                "guard": "run only after server_monitor_vqvae_recovery exits 0",
                "command": local_pull_copyback_command(),
            },
            "local_verify_vqvae_copyback": {
                "where": "local_after_copyback",
                "command": local_copyback_command(),
            },
        },
        "deferred_commands": {
            "source_path_sequence_rebuild": {
                "only_after": "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD",
                "command": source_path_rebuild_command(),
            },
            "source_path_length_coverage": {
                "only_after": "source_path_sequence_rebuild_finished",
                "command": source_path_length_coverage_command(),
            },
            "verify_sequence_rebuild": {
                "only_after": "source_path_length_coverage_finished",
                "command": sequence_rebuild_verify_command(),
            },
            "ar1536": {
                "only_after": "READY_FOR_AR_LONG_CONTEXT",
                "command": ar1536_command(),
            },
            "ar2048": {
                "only_after": "ar1536_promising_or_memory_allows",
                "command": ar2048_command(),
            },
            "generated_reconstruction_ar1536": {
                "only_after": "ar1536_best_checkpoint_ready",
                "command": generated_reconstruction_command(
                    branch="newscheme_full_v13_ar1536",
                    run_name="eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD",
                    max_new_tokens=768,
                ),
            },
            "render_generated_ar1536": {
                "only_after": "generated_reconstruction_ar1536_finished",
                "command": render_generated_command("eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD"),
            },
            "step_geometry_entity_audit_ar1536": {
                "only_after": "render_generated_ar1536_finished",
                "command": step_geometry_entity_audit_command(
                    "eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD"
                ),
            },
            "generated_quality_gate_ar1536": {
                "only_after": "step_geometry_entity_audit_ar1536_finished",
                "command": generated_quality_gate_command("eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD"),
            },
            "server_paper_figure_audit_ar1536": {
                "where": "server",
                "only_after": "generated_quality_gate_ar1536_finished",
                "command": server_paper_figure_audit_command(
                    "eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD"
                ),
            },
            "local_pull_generated_ar1536": {
                "where": "local_after_generated_gate",
                "only_after": "server_paper_figure_audit_ar1536_finished",
                "command": local_pull_generated_command("eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD"),
            },
            "paper_figure_audit_ar1536": {
                "where": "local_after_generated_pull",
                "only_after": "local_pull_generated_ar1536_finished",
                "command": paper_figure_audit_command(
                    run_name="eval_generated100_ar1536_vqcomplex_temp09_topp92_max768_YYYYMMDD",
                    branch_label="ar1536",
                ),
            },
            "generated_reconstruction_ar2048": {
                "only_after": "ar2048_best_checkpoint_ready",
                "command": generated_reconstruction_command(
                    branch="newscheme_full_v13_ar2048",
                    run_name="eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD",
                    max_new_tokens=1024,
                ),
            },
            "render_generated_ar2048": {
                "only_after": "generated_reconstruction_ar2048_finished",
                "command": render_generated_command("eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD"),
            },
            "step_geometry_entity_audit_ar2048": {
                "only_after": "render_generated_ar2048_finished",
                "command": step_geometry_entity_audit_command(
                    "eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD"
                ),
            },
            "generated_quality_gate_ar2048": {
                "only_after": "step_geometry_entity_audit_ar2048_finished",
                "command": generated_quality_gate_command("eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD"),
            },
            "server_paper_figure_audit_ar2048": {
                "where": "server",
                "only_after": "generated_quality_gate_ar2048_finished",
                "command": server_paper_figure_audit_command(
                    "eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD"
                ),
            },
            "local_pull_generated_ar2048": {
                "where": "local_after_generated_gate",
                "only_after": "server_paper_figure_audit_ar2048_finished",
                "command": local_pull_generated_command("eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD"),
            },
            "paper_figure_audit_ar2048": {
                "where": "local_after_generated_pull",
                "only_after": "local_pull_generated_ar2048_finished",
                "command": paper_figure_audit_command(
                    run_name="eval_generated100_ar2048_vqcomplex_temp09_topp92_max1024_YYYYMMDD",
                    branch_label="ar2048",
                ),
            },
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_lf_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(normalized.encode("utf-8"))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Recovery Packet",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Next stage: `{payload['stage_decision']['next_stage']}`",
        f"- Can train AR now: {payload['stage_decision']['can_train_ar_now']}",
        f"- Transfer entries: {payload['transfer']['entries_total']}",
        f"- Tooling entries: {payload['transfer']['repo_tooling_entries']}",
        "",
        "Do not train AR first. The current packet keeps the first expensive server job as VQ-VAE complex/curved recovery.",
        "",
        "## Immediate Commands",
        "",
    ]
    for label, command in payload["immediate_commands"].items():
        lines.append(f"### {label}")
        lines.append("")
        if command.get("guard"):
            lines.append(f"Guard: {command['guard']}")
            lines.append("")
        info = "powershell" if str(command.get("where", "")).startswith("local") else "bash"
        lines.append(f"```{info}\n{command['command']}\n```")
        lines.append("")
    lines.append("## Deferred Commands")
    lines.append("")
    for label, command in payload["deferred_commands"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"Only after: `{command['only_after']}`")
        lines.append("")
        info = "powershell" if str(command.get("where", "")).startswith("local") else "bash"
        lines.append(f"```{info}\n{command['command']}\n```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def indent_script_block(command: str) -> str:
    return "\n".join("  " + line if line else "" for line in command.splitlines())


def render_server_first_hour_script(payload: dict[str, Any]) -> str:
    """Render a guarded server-side Bash entrypoint from a recovery packet."""

    commands = payload["immediate_commands"]
    preflight = commands["server_guarded_preflight"]["command"]
    progress = commands["server_progress_summary"]["command"]
    start = commands["server_start_vqvae_recovery"]["command"]
    monitor = commands["server_monitor_vqvae_recovery"]["command"]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'mode="${1:-preflight}"',
            "",
            "cd /workspace/V13",
            "",
            'case "${1:-preflight}" in',
            "  preflight)",
            "    echo '[V13] Running guarded preflight only. No training will start.'",
            "    set +e",
            indent_script_block(preflight),
            "    preflight_status=$?",
            "    set -e",
            "    echo '[V13] Writing progress summary after preflight.'",
            "    set +e",
            indent_script_block(progress),
            "    progress_status=$?",
            "    set -e",
            '    if [ "${preflight_status}" -ne 0 ]; then',
            '      exit "${preflight_status}"',
            "    fi",
            '    exit "${progress_status}"',
            "    ;;",
            "  start)",
            "    echo '[V13] Re-running guarded preflight with explicit --start for VQ-VAE recovery.'",
            "    set +e",
            indent_script_block(start),
            "    start_status=$?",
            "    set -e",
            "    echo '[V13] Writing progress summary after start attempt.'",
            "    set +e",
            indent_script_block(progress),
            "    progress_status=$?",
            "    set -e",
            '    if [ "${start_status}" -ne 0 ]; then',
            '      exit "${start_status}"',
            "    fi",
            '    exit "${progress_status}"',
            "    ;;",
            "  monitor)",
            "    echo '[V13] Monitoring VQ-VAE recovery gate.'",
            indent_script_block(monitor),
            "    ;;",
            "  *)",
            "    echo 'Usage: bash local_reports/v13_server_first_hour_from_packet_20260706.sh [preflight|start|monitor]' >&2",
            "    exit 2",
            "    ;;",
            "esac",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V13 rented-server recovery command packet.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--server-repo-root", default=DEFAULT_SERVER_REPO_ROOT)
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--transfer-manifest", default=DEFAULT_TRANSFER_MANIFEST)
    parser.add_argument("--min-gpu-memory-gb", type=float, default=DEFAULT_MIN_GPU_MEMORY_GB)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--server-script-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_server_recovery_packet(
        args.repo_root,
        server_repo_root=args.server_repo_root,
        server_python=args.server_python,
        reports_dir=args.reports_dir,
        transfer_manifest=args.transfer_manifest,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
    )
    if args.output:
        write_json(args.output, packet)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(packet), encoding="utf-8")
    if args.server_script_output:
        write_lf_text(args.server_script_output, render_server_first_hour_script(packet))
    print(json.dumps(packet, indent=2, ensure_ascii=True))
    return 0 if packet["status"] == "READY_FOR_RENTED_SERVER_VQVAE_RECOVERY_PACKET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
