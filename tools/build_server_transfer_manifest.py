"""Create a transfer manifest for the next V13 rented-server run.

The manifest maps local V13 artifacts to their intended server locations and
records file sizes plus optional SHA256 checksums. It is a pre-upload guardrail:
it does not copy files, start training, or mutate checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


REPO_TRANSFER_ITEMS = [
    ("ar_best_checkpoint", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt", "large_model_artifact"),
    ("ar_sequence_package", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl", "large_model_artifact"),
    ("ar_split_file", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl", "large_model_artifact"),
    ("vqvae_training_entry", "breparg_improvements/train.py", "repo_source"),
    ("vqvae_sampling_source", "breparg_improvements/vqvae_sampling.py", "repo_source"),
    ("fsq_quantiser_source", "breparg_improvements/fsq_quantise.py", "repo_source"),
    ("gnn_ordering_source", "breparg_improvements/gnn_ordering.py", "repo_source"),
    ("ar_training_utils_source", "breparg_improvements/ar_training_utils.py", "repo_source"),
    ("training_stability_source", "breparg_improvements/training_stability.py", "repo_source"),
    ("sequence_sharding_source", "breparg_improvements/sequence_sharding.py", "repo_source"),
    ("breparg_source_path_sequence", "BrepARG/2sequence.py", "repo_source"),
    ("breparg_utils_source", "BrepARG/utils.py", "repo_source"),
    ("vqvae_linux_launcher", "tools/run_vqvae_complex_recovery.sh", "repo_tooling"),
    ("sequence_rebuild_linux_launcher", "tools/run_source_path_sequence_rebuild.sh", "repo_tooling"),
    ("ar_long_context_linux_launcher", "tools/run_ar_v13_long_context.sh", "repo_tooling"),
    ("sequence_sharded_runner", "tools/run_sharded_sequence.py", "repo_tooling"),
    ("reconstruction_evaluator", "tools/evaluate_reconstruction_v13.py", "repo_tooling"),
    ("step_renderer", "papers/aaai_v13/render_step_directory.py", "repo_tooling"),
    ("vqvae_benchmark_tool", "tools/run_vqvae_slice_benchmark.py", "repo_tooling"),
    ("vqvae_recovery_monitor", "tools/monitor_vqvae_recovery_gate.py", "repo_tooling"),
    ("generated_quality_gate", "tools/summarize_generated_quality.py", "repo_tooling"),
    ("step_geometry_entity_audit", "tools/audit_step_geometry_entities.py", "repo_tooling"),
    ("parsed_pool_quality_audit", "tools/audit_parsed_pool_quality.py", "repo_tooling"),
    ("paper_figure_candidate_audit", "tools/audit_paper_figure_candidates.py", "repo_tooling"),
    ("source_path_audit_tool", "tools/audit_sequence_source_paths.py", "repo_tooling"),
    ("server_handoff_preflight_tool", "tools/prepare_server_handoff.py", "repo_tooling"),
    ("server_handoff_writer", "tools/write_vqvae_server_handoff.py", "repo_tooling"),
    ("server_transfer_verifier", "tools/verify_server_transfer.py", "repo_tooling"),
    ("server_training_readiness_verifier", "tools/verify_server_training_readiness.py", "repo_tooling"),
    ("model_artifact_sanity_verifier", "tools/verify_model_artifacts.py", "repo_tooling"),
    ("server_recovery_plan_tool", "tools/plan_server_quality_recovery.py", "repo_tooling"),
    ("server_quality_recovery_orchestrator", "tools/run_server_quality_recovery.py", "repo_tooling"),
    ("vqvae_copyback_verifier", "tools/verify_vqvae_copyback.py", "repo_tooling"),
    ("sequence_rebuild_verifier", "tools/verify_sequence_rebuild.py", "repo_tooling"),
    ("quality_recovery_stage_decider", "tools/decide_quality_recovery_stage.py", "repo_tooling"),
    ("quality_recovery_progress_summary", "tools/summarize_quality_recovery_progress.py", "repo_tooling"),
    ("server_training_phase_budget_builder", "tools/build_server_training_phase_budget.py", "repo_tooling"),
    ("server_recovery_packet_builder", "tools/build_server_recovery_packet.py", "repo_tooling"),
    ("server_first_hour_script", "local_reports/v13_server_first_hour_from_packet_20260706.sh", "repo_report"),
    ("server_runbook", "local_reports/v13_next_server_quality_recovery_runbook_20260706.md", "repo_report"),
    ("server_start_here_guide", "local_reports/v13_rented_server_start_here_20260706.md", "repo_report"),
    ("rental_gpu_decision_card", "local_reports/v13_rental_gpu_decision_card_20260706.md", "repo_report"),
    ("server_training_phase_budget_json", "local_reports/v13_server_training_phase_budget_20260706.json", "repo_report"),
    ("server_training_phase_budget_report", "local_reports/v13_server_training_phase_budget_20260706.md", "repo_report"),
    ("server_handoff_preflight", "local_reports/v13_server_handoff_preflight_20260706.json", "repo_report"),
]

ABC_TRANSFER_ITEMS = [
    (
        "vqvae_baseline_checkpoint",
        "ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
        "newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
        "large_model_artifact",
    ),
]

SERVER_DATA_REQUIREMENTS = [
    {
        "label": "parsed_abc_pool",
        "server_path": "/workspace/ABC/processed/abc_parsed_full",
        "reason": "VQ-VAE recovery reads parsed ABC geometry pickles from this pool.",
    },
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_server_path(*parts: str) -> str:
    string_parts = [str(part).replace("\\", "/") for part in parts]
    cleaned = [part.strip("/") for part in string_parts if part.strip("/")]
    if not cleaned:
        return ""
    prefix = "/" if string_parts[0].startswith("/") else ""
    return prefix + "/".join(cleaned)


def should_hash(size: int, hash_limit_bytes: int | None) -> bool:
    if hash_limit_bytes is None:
        return True
    return int(size) <= int(hash_limit_bytes)


def repo_entry(
    repo_root: Path,
    server_repo_root: str,
    label: str,
    relative_path: str,
    transfer_group: str,
    hash_limit_bytes: int | None,
) -> dict[str, Any]:
    local_path = repo_root / relative_path
    exists = local_path.exists()
    size = int(local_path.stat().st_size) if exists and local_path.is_file() else 0
    digest = sha256_file(local_path) if exists and local_path.is_file() and should_hash(size, hash_limit_bytes) else None
    return {
        "label": label,
        "transfer_group": transfer_group,
        "local_path": relative_path,
        "server_path": normalize_server_path(server_repo_root, relative_path),
        "exists": bool(exists),
        "bytes": size,
        "sha256": digest,
        "hash_skipped": bool(exists and local_path.is_file() and digest is None),
    }


def abc_entry(
    repo_root: Path,
    server_abc_train_root: str,
    label: str,
    local_relative_path: str,
    server_relative_path: str,
    transfer_group: str,
    hash_limit_bytes: int | None,
) -> dict[str, Any]:
    local_path = repo_root / local_relative_path
    exists = local_path.exists()
    size = int(local_path.stat().st_size) if exists and local_path.is_file() else 0
    digest = sha256_file(local_path) if exists and local_path.is_file() and should_hash(size, hash_limit_bytes) else None
    return {
        "label": label,
        "transfer_group": transfer_group,
        "local_path": local_relative_path,
        "server_path": normalize_server_path(server_abc_train_root, server_relative_path),
        "exists": bool(exists),
        "bytes": size,
        "sha256": digest,
        "hash_skipped": bool(exists and local_path.is_file() and digest is None),
    }


def quote_single(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def quote_powershell(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def upload_command(entry: dict[str, Any], remote: str) -> str:
    target_dir = str(Path(entry["server_path"]).parent).replace("\\", "/")
    return (
        "rsync -av --progress "
        f"{quote_single(entry['local_path'])} "
        f"{remote}:{quote_single(target_dir + '/')}"
    )


def verify_command(entry: dict[str, Any]) -> str:
    path = quote_single(entry["server_path"])
    if entry.get("sha256"):
        return f"test -s {path} && echo {entry['sha256']}  {path} | sha256sum -c -"
    return f"test -s {path}"


def render_powershell_upload_script(
    payload: dict[str, Any],
    remote: str = "USER@SERVER",
    manifest_local_path: str = "local_reports/v13_server_transfer_manifest_20260706.json",
    manifest_server_path: str = "/workspace/V13/local_reports/v13_server_transfer_manifest_20260706.json",
) -> str:
    if not payload.get("required_sources_complete", False):
        missing = ", ".join(entry.get("label", "unknown") for entry in payload.get("missing_entries", []))
        raise ValueError(f"Cannot render upload script for incomplete manifest: {missing}")

    entries = [entry for entry in payload.get("entries", []) if entry.get("exists")]
    manifest_entry = {
        "label": "transfer_manifest_for_server_verifier",
        "local_path": manifest_local_path.replace("\\", "/"),
        "server_path": manifest_server_path.replace("\\", "/"),
        "exists": True,
    }
    upload_entries = [manifest_entry] + entries
    target_dirs = sorted({str(Path(entry["server_path"]).parent).replace("\\", "/") for entry in upload_entries})
    mkdir_targets = " ".join(quote_single(path) for path in target_dirs)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$Remote = {quote_powershell(remote)}",
        "",
        f"ssh $Remote \"mkdir -p {mkdir_targets}\"",
        "",
    ]
    for entry in upload_entries:
        target_dir = str(Path(entry["server_path"]).parent).replace("\\", "/") + "/"
        lines.append(
            "rsync -av --progress "
            f"{quote_powershell(entry['local_path'])} "
            f"\"${{Remote}}:{quote_single(target_dir)}\""
        )
    lines.extend(
        [
            "",
            "Write-Host 'Upload complete. On the server, run:'",
            "Write-Host '/opt/conda/bin/python tools/verify_server_transfer.py "
            "--manifest local_reports/v13_server_transfer_manifest_20260706.json'",
        ]
    )
    return "\n".join(lines) + "\n"


def build_transfer_manifest(
    repo_root: str | Path,
    server_repo_root: str = "/workspace/V13",
    server_abc_train_root: str = "/workspace/ABC/processed/train_outputs",
    hash_limit_bytes: int | None = 32 * 1024 * 1024,
    remote: str = "USER@SERVER",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    entries = [
        repo_entry(root, server_repo_root, label, relative_path, group, hash_limit_bytes)
        for label, relative_path, group in REPO_TRANSFER_ITEMS
    ]
    entries.extend(
        abc_entry(root, server_abc_train_root, label, local_path, server_path, group, hash_limit_bytes)
        for label, local_path, server_path, group in ABC_TRANSFER_ITEMS
    )
    complete = all(entry["exists"] for entry in entries)
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(root),
        "status": "READY_TO_TRANSFER" if complete else "MISSING_TRANSFER_SOURCES",
        "required_sources_complete": bool(complete),
        "server_repo_root": server_repo_root,
        "server_abc_train_root": server_abc_train_root,
        "hash_limit_bytes": None if hash_limit_bytes is None else int(hash_limit_bytes),
        "entries": entries,
        "missing_entries": [entry for entry in entries if not entry["exists"]],
        "server_data_requirements": SERVER_DATA_REQUIREMENTS,
        "suggested_upload_commands": [upload_command(entry, remote) for entry in entries if entry["exists"]],
        "server_verify_commands": [verify_command(entry) for entry in entries if entry["exists"]],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Transfer Manifest",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Required sources complete: {payload['required_sources_complete']}",
        f"- Server repo root: `{payload['server_repo_root']}`",
        f"- Server ABC train root: `{payload['server_abc_train_root']}`",
        "",
        "## Entries",
        "",
        "| Label | Group | Bytes | SHA256 | Server path |",
        "|---|---|---:|---|---|",
    ]
    for entry in payload["entries"]:
        digest = entry["sha256"] or ("skipped" if entry["hash_skipped"] else "missing")
        lines.append(
            f"| {entry['label']} | {entry['transfer_group']} | {entry['bytes']} | "
            f"{digest} | `{entry['server_path']}` |"
        )
    lines.extend(["", "## Server Data Requirements", ""])
    for item in payload["server_data_requirements"]:
        lines.append(f"- `{item['server_path']}`: {item['reason']}")
    lines.extend(["", "## Verify Commands", ""])
    for command in payload["server_verify_commands"]:
        lines.append(f"```bash\n{command}\n```")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V13 rented-server transfer manifest.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--server-repo-root", default="/workspace/V13")
    parser.add_argument("--server-abc-train-root", default="/workspace/ABC/processed/train_outputs")
    parser.add_argument("--remote", default="USER@SERVER", help="Remote host placeholder for suggested rsync commands.")
    parser.add_argument("--hash-limit-mb", type=float, default=32.0, help="Hash files up to this size; use --hash-large for all.")
    parser.add_argument("--hash-large", action="store_true", help="Compute SHA256 for all files, including large checkpoints.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--powershell-upload-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hash_limit = None if args.hash_large else int(float(args.hash_limit_mb) * 1024 * 1024)
    manifest = build_transfer_manifest(
        repo_root=args.repo_root,
        server_repo_root=args.server_repo_root,
        server_abc_train_root=args.server_abc_train_root,
        hash_limit_bytes=hash_limit,
        remote=args.remote,
    )
    if args.output:
        write_json(args.output, manifest)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")
    if args.powershell_upload_output:
        args.powershell_upload_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_upload_path = args.output or Path("local_reports/v13_server_transfer_manifest_20260706.json")
        args.powershell_upload_output.write_text(
            render_powershell_upload_script(
                manifest,
                remote=args.remote,
                manifest_local_path=str(manifest_upload_path).replace("\\", "/"),
                manifest_server_path=normalize_server_path(
                    args.server_repo_root,
                    "local_reports",
                    Path(manifest_upload_path).name,
                ),
            ),
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0 if manifest["required_sources_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
