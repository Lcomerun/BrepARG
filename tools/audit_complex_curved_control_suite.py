"""Audit progress for the complex-curved FSQ/AR control workspace."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"status": "CORRUPT", "path": str(path)}


def script_command(scripts_root: Path, script_name: str) -> str:
    return f"powershell -ExecutionPolicy Bypass -File {scripts_root / script_name}"


def status_from_report(report: dict[str, Any] | None, *, complete_statuses: set[str] | None = None) -> str:
    if not report:
        return "missing"
    complete_statuses = complete_statuses or {"VERIFIED", "PASS", "complete"}
    if str(report.get("status")) in complete_statuses:
        return "complete"
    return "partial"


def metric_path(report: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system().lower() == "windows":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        import os

        os.kill(int(pid), 0)
        return True
    except (OSError, SystemError):
        return False


def windows_pid_command_matches(pid: int, patterns: list[str], *, root: Path | None = None) -> bool:
    if pid <= 0 or platform.system().lower() != "windows":
        return pid_is_alive(pid)
    escaped = ", ".join("'" + pattern.replace("'", "''") + "'" for pattern in patterns)
    root_text = str(root).replace("'", "''") if root is not None else ""
    root_setup = f"$root = '{root_text}'; " if root is not None else ""
    root_check = "($true)"
    if root is not None:
        root_check = "($cmd -like ('*' + $root + '*'))"
    script = (
        root_setup
        + f"$patterns = @({escaped}); "
        + f"$proc = Get-CimInstance Win32_Process -Filter \"ProcessId = {int(pid)}\" -ErrorAction SilentlyContinue; "
        + "if (-not $proc) { exit 1 }; "
        + "$cmd = $proc.CommandLine; "
        + "if (-not $cmd) { exit 1 }; "
        + f"if (-not {root_check}) {{ exit 1 }}; "
        + "foreach ($p in $patterns) { if ($cmd -match $p) { exit 0 } }; "
        + "exit 1"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def windows_process_count(patterns: list[str], *, root: Path | None = None) -> int:
    if platform.system().lower() != "windows":
        return 0
    escaped = ", ".join("'" + pattern.replace("'", "''") + "'" for pattern in patterns)
    root_filter = ""
    if root is not None:
        root_text = str(root).replace("'", "''")
        root_filter = "$root = '" + root_text + "'; "
    script = (
        root_filter
        +
        "$patterns = @(" + escaped + "); "
        "$count = 0; "
        "Get-CimInstance Win32_Process | ForEach-Object { "
        "$cmd = $_.CommandLine; "
        "if ($cmd -and $cmd -notmatch '\\s-Command\\s') { "
        "$rootOk = $true; "
        "if (Get-Variable -Name root -ErrorAction SilentlyContinue) { $rootOk = ($cmd -like ('*' + $root + '*')) }; "
        "if ($rootOk) { foreach ($p in $patterns) { if ($cmd -match $p) { $count++; break } } } "
        "} "
        "}; "
        "Write-Output $count"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return 0
        return int((completed.stdout or "0").strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def latest_log(log_dir: Path, pattern: str) -> Path | None:
    if not log_dir.exists():
        return None
    matches = sorted(log_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def latest_file(search_root: Path, pattern: str) -> Path | None:
    if not search_root.exists():
        return None
    matches = sorted(search_root.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def fresh_report_after_checkpoint(report_path: Path, checkpoint_path: Path) -> bool:
    """Return True only when an eval report was produced after its checkpoint."""
    if not report_path.exists() or not checkpoint_path.exists():
        return False
    return report_path.stat().st_mtime > checkpoint_path.stat().st_mtime


def merged_current_method_report(experiments_root: Path) -> tuple[Path, dict[str, Any] | None]:
    combined_path = experiments_root / "00_current_fsq_ar_teacher_reconstruction" / "complex_curved_diagnostics_report.json"
    combined = read_json(combined_path)
    if combined:
        return combined_path, combined

    fsq_path = experiments_root / "00_fsq_only_patch_metrics" / "complex_curved_diagnostics_report.json"
    teacher_path = experiments_root / "01_teacher_forcing_true_token_reconstruction" / "complex_curved_diagnostics_report.json"
    fsq_report = read_json(fsq_path)
    teacher_report = read_json(teacher_path)
    if not fsq_report and not teacher_report:
        return combined_path, None

    fsq_status = status_from_report(fsq_report)
    teacher_status = status_from_report(teacher_report)
    status = "VERIFIED" if fsq_status == "complete" and teacher_status == "complete" else "PARTIAL"
    merged: dict[str, Any] = {
        "status": status,
        "selected_count": metric_path(fsq_report, "selected_count") or metric_path(teacher_report, "selected_count"),
        "fsq_patch_metrics": metric_path(fsq_report, "fsq_patch_metrics"),
        "ar_teacher_forcing": metric_path(teacher_report, "ar_teacher_forcing"),
        "teacher_reconstruction": metric_path(teacher_report, "teacher_reconstruction"),
        "source_reports": {
            "fsq_only": str(fsq_path),
            "teacher_forcing": str(teacher_path),
        },
    }
    return fsq_path, merged


def experiment_entry(status: str, path: Path, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "path": str(path),
        "details": details or {},
    }


def audit_suite(root: Path) -> dict[str, Any]:
    root = Path(root)
    experiments_root = root / "experiments"
    scripts_root = root / "scripts"

    current_path, current = merged_current_method_report(experiments_root)
    capacity_train_path = (
        experiments_root
        / "01a_train_fsq_capacity_candidate"
        / "fsq_levels_16_16_8_8_complex_curved_20260715"
        / "fsq_vqvae_best.pt"
    )
    capacity_train_run_dir = capacity_train_path.parent
    capacity_train_report_path = capacity_train_run_dir / "train_report.json"
    capacity_train_report = read_json(capacity_train_report_path)
    capacity_train_history_path = capacity_train_run_dir / "vqvae_history.json"
    capacity_train_history = read_json(capacity_train_history_path)
    capacity_logs_dir = experiments_root / "01a_train_fsq_capacity_candidate" / "logs"
    capacity_resume_pid_path = capacity_logs_dir / "fsq_capacity_resume.pid"
    capacity_resume_pid = read_pid(capacity_resume_pid_path)
    capacity_resume_log = latest_log(capacity_logs_dir, "fsq_capacity_resume_*.out.log")
    capacity_resume_alive = (
        pid_is_alive(capacity_resume_pid or -1)
        and windows_pid_command_matches(
            capacity_resume_pid or -1,
            [
                "01a_resume_fsq_capacity_candidate\\.ps1",
                "breparg_improvements\\\\train\\.py --stage vqvae",
            ],
            root=root,
        )
    )
    capacity_patch_summary_path = experiments_root / "01a_train_fsq_capacity_candidate" / "vq_patch_shards_full" / "_summary.json"
    capacity_patch_summary = read_json(capacity_patch_summary_path)
    capacity_preflight_path = experiments_root / "01a_train_fsq_capacity_candidate" / "fsq_capacity_preflight.json"
    capacity_preflight = read_json(capacity_preflight_path)
    capacity_eval_path = experiments_root / "01_fsq_capacity_candidate" / "complex_curved_diagnostics_report.json"
    capacity_eval = read_json(capacity_eval_path)
    capacity_comparison_path = root / "fsq_capacity_comparison.json"
    capacity_comparison = read_json(capacity_comparison_path)

    dfs_sequence = experiments_root / "02_dfs_rcm_ordering" / "sequences_fsq_dfs.pkl"
    rcm_sequence = experiments_root / "02_dfs_rcm_ordering" / "sequences_fsq_rcm.pkl"
    split_smoke_summary_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "same_data_split_smoke"
        / "v13_same_data_split_summary.json"
    )
    dfs_smoke_summary_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "sequence_rebuild_smoke"
        / "sequences_fsq_dfs_summary.json"
    )
    rcm_smoke_summary_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "sequence_rebuild_smoke"
        / "sequences_fsq_rcm_summary.json"
    )
    split_smoke_summary = read_json(split_smoke_summary_path)
    dfs_smoke_summary = read_json(dfs_smoke_summary_path)
    rcm_smoke_summary = read_json(rcm_smoke_summary_path)
    dfs_medium_sequence = experiments_root / "02_dfs_rcm_ordering" / "sequence_rebuild_medium" / "sequences_fsq_dfs.pkl"
    rcm_medium_sequence = experiments_root / "02_dfs_rcm_ordering" / "sequence_rebuild_medium" / "sequences_fsq_rcm.pkl"
    dfs_medium_summary_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "sequence_rebuild_medium"
        / "sequences_fsq_dfs_summary.json"
    )
    rcm_medium_summary_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "sequence_rebuild_medium"
        / "sequences_fsq_rcm_summary.json"
    )
    dfs_medium_summary = read_json(dfs_medium_summary_path)
    rcm_medium_summary = read_json(rcm_medium_summary_path)
    dfs_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_dfs_matched_20260715" / "ar_best.pt"
    rcm_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_rcm_matched_20260715" / "ar_best.pt"
    dfs_rcm_train_processes = windows_process_count(
        [
            "02b_train_dfs_rcm_ar\\.ps1",
            "breparg_improvements\\\\train\\.py --stage ar",
        ],
        root=root,
    )
    dfs_rcm_eval_watcher_processes = windows_process_count(["02d_watch_matched_ar_then_eval\\.ps1"], root=root)
    dfs_medium_smoke_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_dfs_medium_smoke_20260715" / "ar_best.pt"
    rcm_medium_smoke_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_rcm_medium_smoke_20260715" / "ar_best.pt"
    dfs_medium_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_dfs_medium_safe_20260715" / "ar_best.pt"
    rcm_medium_ar = experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs" / "ar_rcm_medium_safe_20260715" / "ar_best.pt"
    dfs_eval_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "ar_complex_curved_eval"
        / "dfs_teacher_forcing"
        / "complex_curved_diagnostics_report.json"
    )
    rcm_eval_path = (
        experiments_root
        / "02_dfs_rcm_ordering"
        / "ar_complex_curved_eval"
        / "rcm_teacher_forcing"
        / "complex_curved_diagnostics_report.json"
    )
    dfs_eval = read_json(dfs_eval_path)
    rcm_eval = read_json(rcm_eval_path)

    official_baseline_path = experiments_root / "03_breparg_official_baseline" / "breparg_baseline_quality_summary.json"
    official_baseline = read_json(official_baseline_path)
    official_incompat_path = experiments_root / "03_breparg_official_baseline" / "official_baseline_incompatibility_report.json"
    official_incompat = read_json(official_incompat_path)
    breparg_logic_root = experiments_root / "04_breparg_logic_generation_baseline"
    breparg_logic_report_path = latest_file(breparg_logic_root, "*/breparg_logic_report.json")
    breparg_logic_report = read_json(breparg_logic_report_path) if breparg_logic_report_path else None
    breparg_logic_distribution_path = (
        breparg_logic_report_path.parent / "breparg_logic_saved100_distribution.json"
        if breparg_logic_report_path
        else breparg_logic_root / "breparg_logic_saved100_distribution.json"
    )
    breparg_logic_distribution = read_json(breparg_logic_distribution_path)
    same_data_inputs_path = (
        experiments_root
        / "03b_breparg_same_data_training_fallback"
        / "data"
        / "same_data_input_summary.json"
    )
    same_data_inputs = read_json(same_data_inputs_path)
    fallback_root = experiments_root / "03b_breparg_same_data_training_fallback"
    fallback_baseline_path = fallback_root / "breparg_same_data_quality_summary.json"
    if not fallback_baseline_path.exists():
        latest_fallback = latest_file(fallback_root, "breparg_same_data_quality_summary_*.json")
        if latest_fallback:
            fallback_baseline_path = latest_fallback
    fallback_baseline = read_json(fallback_baseline_path)
    fallback_preflight_path = fallback_root / "breparg_same_data_preflight.json"
    fallback_preflight = read_json(fallback_preflight_path)
    breparg_watcher_processes = windows_process_count(["03c_watch_then_start_breparg_3060_safe\\.ps1"], root=root)
    breparg_training_processes = windows_process_count(["03b_breparg_same_data_training_fallback"], root=root)

    experiments: dict[str, dict[str, Any]] = {}
    experiments["current_method"] = experiment_entry(
        status_from_report(current),
        current_path,
        {
            "selected_count": metric_path(current, "selected_count"),
            "fsq_chamfer_p95": metric_path(current, "fsq_patch_metrics", "chamfer", "p95"),
            "ar_token_weighted_ce": metric_path(current, "ar_teacher_forcing", "token_weighted_ce"),
            "teacher_brep_valid": metric_path(current, "teacher_reconstruction", "brep_valid"),
            "teacher_attempted": metric_path(current, "teacher_reconstruction", "attempted"),
        },
    )
    capacity_stage = metric_path(capacity_train_report, "stages", "vqvae")
    capacity_stage_status = str(metric_path(capacity_stage, "status") or "")
    capacity_history_records = metric_path(capacity_train_history, "history") or []
    capacity_last_history = capacity_history_records[-1] if isinstance(capacity_history_records, list) and capacity_history_records else {}
    capacity_training_complete = capacity_train_path.exists() and capacity_stage_status in {"VERIFIED", "PASS", "complete"}
    capacity_training_partial = (
        capacity_train_path.exists()
        or bool(capacity_train_report)
        or bool(capacity_train_history)
    )
    experiments["fsq_capacity_training"] = experiment_entry(
        "complete" if capacity_training_complete else "partial" if capacity_training_partial else "missing",
        capacity_train_path,
        {
            "checkpoint_best": capacity_train_path.exists(),
            "train_report_exists": capacity_train_report_path.exists(),
            "train_report_status": capacity_stage_status or None,
            "history_epochs": len(capacity_history_records) if isinstance(capacity_history_records, list) else None,
            "target_epoch": metric_path(capacity_train_history, "config", "target_epoch"),
            "last_epoch": capacity_last_history.get("epoch") if isinstance(capacity_last_history, dict) else None,
            "last_val": capacity_last_history.get("val_loss") if isinstance(capacity_last_history, dict) else None,
            "history_best_val": metric_path(capacity_train_history, "best_val_recon"),
            "history_best_epoch": metric_path(capacity_train_history, "best_epoch"),
            "resume_pid": capacity_resume_pid,
            "resume_alive": capacity_resume_alive,
            "resume_log": str(capacity_resume_log) if capacity_resume_log else None,
            "epochs_ran": metric_path(capacity_stage, "epochs_ran"),
            "best_val_recon": metric_path(capacity_stage, "best_val_recon"),
        },
    )
    patch_summary_complete = (
        bool(capacity_patch_summary)
        and str(capacity_patch_summary.get("status")) in {"BUILT", "SKIPPED_EXISTING"}
        and int(capacity_patch_summary.get("patch_shards") or 0) > 0
        and int(capacity_patch_summary.get("patches") or 0) > 0
    )
    experiments["fsq_capacity_patch_shards"] = experiment_entry(
        "complete" if patch_summary_complete else status_from_report(capacity_patch_summary, complete_statuses={"BUILT", "SKIPPED_EXISTING"}),
        capacity_patch_summary_path,
        {
            "patch_shards": metric_path(capacity_patch_summary, "patch_shards"),
            "patches": metric_path(capacity_patch_summary, "patches"),
            "source_records_failed": metric_path(capacity_patch_summary, "source_records_failed"),
            "source_records_skipped_by_cap": metric_path(capacity_patch_summary, "source_records_skipped_by_cap"),
        },
    )
    experiments["fsq_capacity_preflight"] = experiment_entry(
        "complete"
        if str(metric_path(capacity_preflight, "status")) == "READY"
        else status_from_report(capacity_preflight, complete_statuses={"READY"}),
        capacity_preflight_path,
        {
            "status": metric_path(capacity_preflight, "status"),
            "blocking_reasons": metric_path(capacity_preflight, "blocking_reasons"),
            "codebook_size": metric_path(capacity_preflight, "config", "codebook_size"),
            "samples": metric_path(capacity_preflight, "config", "samples"),
            "sample_cache_enabled": metric_path(capacity_preflight, "sample_cache", "enabled"),
            "sample_cache_exists": metric_path(capacity_preflight, "sample_cache", "exists"),
            "sample_cache_path": metric_path(capacity_preflight, "sample_cache", "path"),
            "next_command": metric_path(capacity_preflight, "next_command"),
        },
    )
    experiments["fsq_capacity_eval"] = experiment_entry(
        status_from_report(capacity_eval),
        capacity_eval_path,
        {
            "fsq_chamfer_p95": metric_path(capacity_eval, "fsq_patch_metrics", "chamfer", "p95"),
            "status": metric_path(capacity_eval, "status"),
        },
    )
    experiments["fsq_capacity_comparison"] = experiment_entry(
        status_from_report(capacity_comparison),
        capacity_comparison_path,
        {
            "capacity_signal": metric_path(capacity_comparison, "recommendation", "capacity_signal"),
            "reading": metric_path(capacity_comparison, "recommendation", "reading"),
            "fsq_chamfer_p95_baseline": metric_path(capacity_comparison, "metrics", "fsq_chamfer_p95", "baseline"),
            "fsq_chamfer_p95_candidate": metric_path(capacity_comparison, "metrics", "fsq_chamfer_p95", "candidate"),
            "fsq_chamfer_p95_relative_change_pct": metric_path(
                capacity_comparison,
                "metrics",
                "fsq_chamfer_p95",
                "relative_change_pct",
            ),
        },
    )
    sequence_status = "complete" if dfs_sequence.exists() and rcm_sequence.exists() else "partial" if dfs_sequence.exists() or rcm_sequence.exists() else "missing"
    smoke_status = (
        "complete"
        if status_from_report(split_smoke_summary) == "complete"
        and status_from_report(dfs_smoke_summary) == "complete"
        and status_from_report(rcm_smoke_summary) == "complete"
        else "partial"
        if split_smoke_summary or dfs_smoke_summary or rcm_smoke_summary
        else "missing"
    )
    experiments["dfs_rcm_sequence_rebuild_smoke"] = experiment_entry(
        smoke_status,
        experiments_root / "02_dfs_rcm_ordering" / "sequence_rebuild_smoke",
        {
            "split_total_written": metric_path(split_smoke_summary, "total_written"),
            "dfs_sequences": metric_path(dfs_smoke_summary, "sequences"),
            "rcm_sequences": metric_path(rcm_smoke_summary, "sequences"),
            "dfs_out_of_vocab": metric_path(dfs_smoke_summary, "out_of_vocab"),
            "rcm_out_of_vocab": metric_path(rcm_smoke_summary, "out_of_vocab"),
        },
    )
    medium_status = (
        "complete"
        if dfs_medium_sequence.exists()
        and rcm_medium_sequence.exists()
        and status_from_report(dfs_medium_summary) == "complete"
        and status_from_report(rcm_medium_summary) == "complete"
        else "partial"
        if dfs_medium_sequence.exists() or rcm_medium_sequence.exists() or dfs_medium_summary or rcm_medium_summary
        else "missing"
    )
    experiments["dfs_rcm_sequence_rebuild_medium"] = experiment_entry(
        medium_status,
        experiments_root / "02_dfs_rcm_ordering" / "sequence_rebuild_medium",
        {
            "dfs_sequence": dfs_medium_sequence.exists(),
            "rcm_sequence": rcm_medium_sequence.exists(),
            "dfs_sequences": metric_path(dfs_medium_summary, "sequences"),
            "rcm_sequences": metric_path(rcm_medium_summary, "sequences"),
            "dfs_out_of_vocab": metric_path(dfs_medium_summary, "out_of_vocab"),
            "rcm_out_of_vocab": metric_path(rcm_medium_summary, "out_of_vocab"),
        },
    )
    experiments["dfs_rcm_sequence_rebuild"] = experiment_entry(
        sequence_status,
        experiments_root / "02_dfs_rcm_ordering",
        {"dfs_sequence": dfs_sequence.exists(), "rcm_sequence": rcm_sequence.exists()},
    )
    ar_status = "complete" if dfs_ar.exists() and rcm_ar.exists() else "partial" if dfs_ar.exists() or rcm_ar.exists() else "missing"
    medium_smoke_ar_status = (
        "complete"
        if dfs_medium_smoke_ar.exists() and rcm_medium_smoke_ar.exists()
        else "partial"
        if dfs_medium_smoke_ar.exists() or rcm_medium_smoke_ar.exists()
        else "missing"
    )
    experiments["dfs_rcm_ar_training_medium_smoke"] = experiment_entry(
        medium_smoke_ar_status,
        experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs",
        {"dfs_ar_best": dfs_medium_smoke_ar.exists(), "rcm_ar_best": rcm_medium_smoke_ar.exists()},
    )
    medium_ar_status = (
        "complete"
        if dfs_medium_ar.exists() and rcm_medium_ar.exists()
        else "partial"
        if dfs_medium_ar.exists() or rcm_medium_ar.exists()
        else "missing"
    )
    experiments["dfs_rcm_ar_training_medium"] = experiment_entry(
        medium_ar_status,
        experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs",
        {"dfs_ar_best": dfs_medium_ar.exists(), "rcm_ar_best": rcm_medium_ar.exists()},
    )
    experiments["dfs_rcm_ar_training"] = experiment_entry(
        ar_status,
        experiments_root / "02_dfs_rcm_ordering" / "ar_train_outputs",
        {
            "dfs_ar_best": dfs_ar.exists(),
            "rcm_ar_best": rcm_ar.exists(),
            "running_processes": dfs_rcm_train_processes,
            "eval_watcher_processes": dfs_rcm_eval_watcher_processes,
        },
    )
    dfs_eval_fresh = fresh_report_after_checkpoint(dfs_eval_path, dfs_ar)
    rcm_eval_fresh = fresh_report_after_checkpoint(rcm_eval_path, rcm_ar)
    eval_status = (
        "complete"
        if (
            ar_status == "complete"
            and dfs_eval_fresh
            and rcm_eval_fresh
            and status_from_report(dfs_eval) == "complete"
            and status_from_report(rcm_eval) == "complete"
        )
        else "partial"
        if dfs_eval or rcm_eval or dfs_ar.exists() or rcm_ar.exists()
        else "missing"
    )
    experiments["dfs_rcm_teacher_forcing"] = experiment_entry(
        eval_status,
        experiments_root / "02_dfs_rcm_ordering" / "ar_complex_curved_eval",
        {
            "dfs_ce": metric_path(dfs_eval, "ar_teacher_forcing", "token_weighted_ce"),
            "rcm_ce": metric_path(rcm_eval, "ar_teacher_forcing", "token_weighted_ce"),
            "dfs_eval_fresh": dfs_eval_fresh,
            "rcm_eval_fresh": rcm_eval_fresh,
            "requires_full_matched_checkpoints": True,
        },
    )
    official_step_files = int(metric_path(official_baseline, "summary", "step_files") or 0)
    official_status = "complete" if official_step_files > 0 else "partial" if official_incompat else "missing"
    experiments["breparg_official_baseline"] = experiment_entry(
        official_status,
        official_baseline_path if official_status == "complete" else official_incompat_path,
        {
            "brep_valid": metric_path(official_baseline, "summary", "brep_valid"),
            "complex_valid_closed": metric_path(official_baseline, "summary", "complex_and_brep_valid_closed"),
            "compatibility_status": metric_path(official_incompat, "status"),
            "abc_ar_vocab": metric_path(official_incompat, "checkpoint_shapes", "abc_ar_transformer_wte"),
        },
    )
    breparg_logic_summary = metric_path(breparg_logic_report, "summary") or {}
    breparg_logic_saved = (
        metric_path(breparg_logic_distribution, "saved_png_step_rows")
        or metric_path(breparg_logic_summary, "accepted_visual")
        or metric_path(breparg_logic_summary, "png_saved")
        or 0
    )
    breparg_logic_status = (
        "complete"
        if str(metric_path(breparg_logic_report, "status")) == "VERIFIED" and int(breparg_logic_saved) > 0
        else "partial"
        if breparg_logic_report
        else "missing"
    )
    top_pairs = metric_path(breparg_logic_distribution, "top_face_edge_pairs") or []
    top_pair = top_pairs[0].get("pair") if isinstance(top_pairs, list) and top_pairs and isinstance(top_pairs[0], dict) else None
    experiments["breparg_logic_generation"] = experiment_entry(
        breparg_logic_status,
        breparg_logic_report_path or breparg_logic_root,
        {
            "saved_png_step_rows": breparg_logic_saved,
            "attempted_rows": metric_path(breparg_logic_distribution, "attempted_rows")
            or metric_path(breparg_logic_summary, "attempted"),
            "reconstruct_failed": metric_path(breparg_logic_distribution, "reconstruct_failed")
            or metric_path(breparg_logic_summary, "status_counts", "reconstruct_failed"),
            "complex_fraction": metric_path(breparg_logic_distribution, "complex_fraction"),
            "very_simple_fraction": metric_path(breparg_logic_distribution, "very_simple_fraction"),
            "faces_median": metric_path(breparg_logic_summary, "faces", "median"),
            "edges_median": metric_path(breparg_logic_summary, "edges", "median"),
            "top_face_edge_pair": top_pair,
            "temperature": metric_path(breparg_logic_report, "config", "temperature"),
            "top_p": metric_path(breparg_logic_report, "config", "top_p"),
            "device": metric_path(breparg_logic_report, "config", "device"),
        },
    )
    train_written = metric_path(same_data_inputs, "splits", "train", "written") or 0
    val_written = metric_path(same_data_inputs, "splits", "val", "written") or 0
    test_written = metric_path(same_data_inputs, "splits", "test", "written") or 0
    surface_patches = metric_path(same_data_inputs, "surface_patches") or 0
    edge_patches = metric_path(same_data_inputs, "edge_patches") or 0
    same_data_inputs_complete = (
        same_data_inputs is not None
        and str(same_data_inputs.get("status")) in {"VERIFIED", "SKIPPED_EXISTING"}
        and int(train_written) > 0
        and int(val_written) > 0
        and int(test_written) > 0
        and int(surface_patches) > 0
        and int(edge_patches) > 0
    )
    experiments["breparg_same_data_inputs"] = experiment_entry(
        "complete" if same_data_inputs_complete else status_from_report(
            same_data_inputs,
            complete_statuses={"VERIFIED", "SKIPPED_EXISTING"},
        ),
        same_data_inputs_path,
        {
            "train_written": train_written,
            "val_written": val_written,
            "test_written": test_written,
            "surface_patches": surface_patches,
            "edge_patches": edge_patches,
        },
    )
    experiments["breparg_same_data_preflight"] = experiment_entry(
        "complete"
        if str(metric_path(fallback_preflight, "status")) == "READY"
        else status_from_report(fallback_preflight, complete_statuses={"READY"}),
        fallback_preflight_path,
        {
            "status": metric_path(fallback_preflight, "status"),
            "blocking_reasons": metric_path(fallback_preflight, "blocking_reasons"),
        },
    )
    fallback_status = (
        "complete"
        if fallback_baseline and int(metric_path(fallback_baseline, "summary", "step_files") or 0) > 0
        else "missing"
    )
    experiments["breparg_same_data_fallback"] = experiment_entry(
        fallback_status,
        fallback_baseline_path,
        {
            "brep_valid": metric_path(fallback_baseline, "summary", "brep_valid"),
            "complex_valid_closed": metric_path(fallback_baseline, "summary", "complex_and_brep_valid_closed"),
            "watcher_processes": breparg_watcher_processes,
            "training_processes": breparg_training_processes,
        },
    )

    next_actions = build_next_actions(experiments, scripts_root)
    status_counts: dict[str, int] = {}
    for entry in experiments.values():
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "summary": {
            "total": len(experiments),
            "completed": status_counts.get("complete", 0),
            "partial": status_counts.get("partial", 0),
            "missing": status_counts.get("missing", 0),
        },
        "experiments": experiments,
        "next_actions": next_actions,
    }


def build_next_actions(experiments: dict[str, dict[str, Any]], scripts_root: Path) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []

    def add(label: str, script_name: str, reason: str) -> None:
        actions.append(
            {
                "label": label,
                "script": str(scripts_root / script_name),
                "command": script_command(scripts_root, script_name),
                "reason": reason,
            }
        )

    def add_manual(label: str, script_name: str, command: str, reason: str) -> None:
        actions.append(
            {
                "label": label,
                "script": str(scripts_root / script_name),
                "command": command,
                "reason": reason,
            }
        )

    if experiments["current_method"]["status"] != "complete":
        add("Run current FSQ/AR diagnostic", "00_current_fsq_ar_teacher_reconstruction.ps1", "Current-method complex-curved baseline is missing.")
        return actions

    active_dfs_rcm_details = experiments.get("dfs_rcm_ar_training", {}).get("details", {})
    if int(active_dfs_rcm_details.get("running_processes") or 0) > 0:
        add_manual(
            "Monitor running matched DFS/RCM AR branches",
            "02b_train_dfs_rcm_ar.ps1",
            (
                "Get-Content "
                + str(scripts_root.parent / "experiments" / "02_dfs_rcm_ordering" / "ar_train_outputs" / "matched_ar_train_len1536_bs4.out.log")
                + " -Tail 100 -Wait"
            ),
            "Matched DFS/RCM AR training is already running; do not start another GPU training job.",
        )

    active_breparg_details = experiments.get("breparg_same_data_fallback", {}).get("details", {})
    if int(active_breparg_details.get("watcher_processes") or 0) > 0:
        add_manual(
            "Monitor BrepARG same-data fallback watcher",
            "03c_watch_then_start_breparg_3060_safe.ps1",
            (
                "Get-Content "
                + str(
                    scripts_root.parent
                    / "experiments"
                    / "03b_breparg_same_data_training_fallback"
                    / "watch_then_start_breparg_3060_safe.log"
                )
                + " -Tail 80 -Wait"
            ),
            "BrepARG fallback watcher is already waiting for fresh DFS/RCM evals and GPU idle; do not start another copy.",
        )

    gpu_busy = any("do not start another" in action["reason"].lower() for action in actions)

    if experiments["fsq_capacity_patch_shards"]["status"] != "complete":
        if (scripts_root / "01a_build_fsq_capacity_patch_shards_full.ps1").exists():
            add(
                "Build full FSQ capacity patch shards",
                "01a_build_fsq_capacity_patch_shards_full.ps1",
                "The real capacity training launcher expects full patch shards before training.",
            )
        else:
            add(
                "Train FSQ capacity candidate",
                "01a_train_fsq_capacity_candidate.ps1",
                "Capacity hypothesis is the highest-priority unresolved experiment; this older workspace script expects you to stage patch shards or a parsed pool first.",
            )
    elif experiments["fsq_capacity_training"]["status"] != "complete":
        if experiments.get("fsq_capacity_preflight", {}).get("status") != "complete" and (
            scripts_root / "01a_preflight_fsq_capacity_candidate.ps1"
        ).exists():
            add(
                "Preflight FSQ capacity candidate",
                "01a_preflight_fsq_capacity_candidate.ps1",
                "Full patch shards exist; verify VQ-VAE training inputs and CLI compatibility before starting the long FSQ capacity run.",
            )
        preflight_details = experiments.get("fsq_capacity_preflight", {}).get("details", {})
        cache_next_command = str(preflight_details.get("next_command") or "")
        if (
            preflight_details.get("sample_cache_enabled") is True
            and preflight_details.get("sample_cache_exists") is False
            and "01a_build_fsq_capacity_sample_cache.ps1" in cache_next_command
            and (scripts_root / "01a_build_fsq_capacity_sample_cache.ps1").exists()
        ):
            add(
                "Build FSQ capacity sample cache",
                "01a_build_fsq_capacity_sample_cache.ps1",
                "Preflight is ready but the reusable 450k patch sample cache is absent; build it before the long FSQ capacity training run.",
            )
        training_details = experiments.get("fsq_capacity_training", {}).get("details", {})
        if training_details.get("resume_alive") is True:
            if (scripts_root / "01a_watch_fsq_capacity_then_eval.ps1").exists():
                add_manual(
                    "Monitor running FSQ capacity candidate",
                    "01a_watch_fsq_capacity_then_eval.ps1",
                    (
                        "Get-Content "
                        + str(scripts_root.parent / "experiments" / "01a_train_fsq_capacity_candidate" / "logs" / "fsq_capacity_watch_then_eval.log")
                        + " -Tail 20 -Wait"
                    ),
                    "FSQ capacity resume is already running; do not start another resume. Watcher will run FSQ-only eval, summaries, and suite audit after training exits cleanly.",
                )
            else:
                add_manual(
                    "Wait for running FSQ capacity candidate",
                    "01a_resume_fsq_capacity_candidate.ps1",
                    (
                        "Get-Content "
                        + str(training_details.get("resume_log") or "")
                        + " -Tail 100 -Wait"
                    ),
                    "FSQ capacity resume is already running; do not start another resume until this PID exits.",
                )
        elif (
            training_details.get("checkpoint_best") is True
            and training_details.get("train_report_exists") is False
            and (scripts_root / "01a_resume_fsq_capacity_candidate.ps1").exists()
        ):
            label = "Resume FSQ capacity candidate"
            reason = "A partial capacity checkpoint and history exist but train_report.json is missing; resume from fsq_vqvae_best.pt instead of restarting from scratch."
            if gpu_busy:
                label = "Resume FSQ capacity candidate after GPU is idle"
                reason = reason + " A GPU job is already running now, so defer this until current watchers finish."
            add(label, "01a_resume_fsq_capacity_candidate.ps1", reason)
        else:
            add("Train FSQ capacity candidate", "01a_train_fsq_capacity_candidate.ps1", "Capacity hypothesis is the highest-priority unresolved experiment.")
    elif experiments["fsq_capacity_eval"]["status"] != "complete":
        add("Evaluate FSQ capacity candidate", "01_fsq_capacity_candidate.ps1", "Capacity checkpoint exists but complex-curved FSQ-only evaluation is missing.")
    elif experiments["fsq_capacity_comparison"]["status"] != "complete" and (scripts_root / "04_summarize_reports.ps1").exists():
        add(
            "Compare FSQ capacity candidate",
            "04_summarize_reports.ps1",
            "Capacity evaluation exists; refresh summaries to compare baseline and higher-capacity FSQ diagnostics.",
        )

    if experiments["dfs_rcm_sequence_rebuild_smoke"]["status"] != "complete" and (
        scripts_root / "02_smoke_dfs_rcm_ordering_rebuild.ps1"
    ).exists():
        add(
            "Smoke DFS/RCM sequence rebuild",
            "02_smoke_dfs_rcm_ordering_rebuild.ps1",
            "Run the small ordering smoke first to verify same-data split materialization and both ordering paths.",
        )
    elif experiments["dfs_rcm_sequence_rebuild_medium"]["status"] != "complete" and (
        scripts_root / "02_medium_dfs_rcm_ordering_rebuild.ps1"
    ).exists():
        add(
            "Rebuild medium DFS/RCM sequence packages",
            "02_medium_dfs_rcm_ordering_rebuild.ps1",
            "Local disk is tight; the medium ordering rebuild reuses the already materialized same-data BrepARG input pool before the full SSD/server run.",
        )
    elif experiments["dfs_rcm_sequence_rebuild"]["status"] != "complete":
        add(
            "Rebuild full DFS/RCM sequence packages",
            "02_dfs_rcm_ordering_rebuild.ps1",
            "Full ordering control still needs both DFS and RCM sequence packages; run this on SSD/server if local disk is tight.",
        )
    if (
        experiments["dfs_rcm_sequence_rebuild_medium"]["status"] == "complete"
        and experiments["dfs_rcm_ar_training_medium_smoke"]["status"] != "complete"
        and (scripts_root / "02b_smoke_dfs_rcm_ar_medium_safe.ps1").exists()
    ):
        add(
            "Smoke train medium matched DFS/RCM AR branches",
            "02b_smoke_dfs_rcm_ar_medium_safe.ps1",
            "Medium DFS/RCM sequence packages exist; run a tiny 1-epoch AR smoke before longer local training.",
        )
    elif experiments["dfs_rcm_sequence_rebuild_medium"]["status"] == "complete" and experiments["dfs_rcm_ar_training_medium"]["status"] != "complete":
        add(
            "Train medium matched DFS/RCM AR branches",
            "02b_train_dfs_rcm_ar_medium_safe.ps1",
            "Medium DFS/RCM sequence packages exist; train short local-safe matched AR branches for an ordering diagnostic.",
        )
    elif experiments["dfs_rcm_ar_training"]["status"] != "complete":
        dfs_rcm_details = experiments["dfs_rcm_ar_training"].get("details", {})
        if int(dfs_rcm_details.get("running_processes") or 0) <= 0:
            add("Train matched DFS/RCM AR branches", "02b_train_dfs_rcm_ar.ps1", "Ordering control needs matched AR checkpoints.")
    elif experiments["dfs_rcm_teacher_forcing"]["status"] != "complete":
        add("Evaluate DFS/RCM teacher forcing", "02c_eval_dfs_rcm_ar_complex_curved.ps1", "Ordering control needs comparable complex-curved teacher-forcing reports.")

    official_status = experiments["breparg_official_baseline"]["status"]
    official_compat = experiments["breparg_official_baseline"]["details"].get("compatibility_status")
    if official_status == "partial" and official_compat == "INCOMPATIBLE":
        if experiments["breparg_same_data_fallback"]["status"] == "complete":
            pass
        elif experiments["breparg_same_data_inputs"]["status"] != "complete":
            add(
                "Prepare BrepARG same-data inputs",
                "03a_prepare_breparg_same_data_inputs.ps1",
                "Official ABC weights were downloaded but are incompatible with the local protocol; materialize the current sequence provenance into original BrepARG input files first.",
            )
            add(
                "Run BrepARG same-data fallback after inputs are ready",
                "03b_breparg_same_data_training_fallback.ps1",
                "After the same-data split and deduplicated SE source files exist, train BrepARG under the same data/eval protocol.",
            )
        else:
            if experiments.get("breparg_same_data_preflight", {}).get("status") != "complete" and (
                scripts_root / "03b_preflight_breparg_same_data_fallback.ps1"
            ).exists():
                add(
                    "Preflight BrepARG same-data fallback",
                    "03b_preflight_breparg_same_data_fallback.ps1",
                    "Inputs exist and official weights are incompatible; verify dependencies and CLI compatibility before starting long fallback training.",
                )
            fallback_details = experiments.get("breparg_same_data_fallback", {}).get("details", {})
            if int(fallback_details.get("watcher_processes") or 0) <= 0:
                add(
                    "Run BrepARG same-data fallback",
                    "03b_breparg_same_data_training_fallback.ps1",
                    "Official ABC weights were downloaded but are incompatible with the local protocol; train BrepARG under the same data/eval protocol.",
                )
    elif official_status != "complete":
        add("Run official BrepARG baseline", "03_breparg_official_baseline.ps1", "Official weights are preferred before same-data fallback training.")
    elif experiments["breparg_same_data_fallback"]["status"] != "complete":
        actions.append(
            {
                "label": "Run BrepARG same-data fallback only if official baseline is incompatible",
                "script": str(scripts_root / "03b_breparg_same_data_training_fallback.ps1"),
                "command": script_command(scripts_root, "03b_breparg_same_data_training_fallback.ps1"),
                "reason": "Fallback is optional unless official baseline cannot be used fairly.",
            }
        )

    return actions


def markdown_text(audit: dict[str, Any]) -> str:
    lines = [
        "# Complex Curved Control Suite Status",
        "",
        f"Created: {audit['created']}",
        f"Root: `{audit['root']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for key, value in audit["summary"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Experiments", "", "| Experiment | Status | Key Details |", "| --- | --- | --- |"])
    for name, entry in audit["experiments"].items():
        details = ", ".join(f"{key}={value}" for key, value in entry.get("details", {}).items() if value is not None)
        lines.append(f"| `{name}` | `{entry['status']}` | {details} |")
    lines.extend(["", "## Next Actions", ""])
    if audit["next_actions"]:
        for idx, action in enumerate(audit["next_actions"], start=1):
            lines.append(f"{idx}. **{action['label']}**")
            lines.append(f"   - Reason: {action['reason']}")
            lines.append(f"   - Command: `{action['command']}`")
    else:
        lines.append("No next actions; all tracked experiment artifacts are present.")
    lines.append("")
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_outputs(audit: dict[str, Any], output: Path | None, markdown_output: Path | None) -> None:
    if output:
        write_text_atomic(output, json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    if markdown_output:
        write_text_atomic(markdown_output, markdown_text(audit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_suite(args.root)
    write_outputs(audit, args.output, args.markdown_output)
    print(json.dumps(audit["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
