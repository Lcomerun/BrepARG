"""Build a preflight manifest for the next V13 rented-server run.

This tool does not start training. It verifies that the local workspace has the
artifacts needed to begin the VQ-VAE-first quality recovery path, summarizes the
current hold/promote gates, and writes a compact handoff manifest for the
server session.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = [
    ("ar_best_checkpoint", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt"),
    ("ar_sequence_package", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl"),
    ("ar_split_file", "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl"),
    ("vqvae_baseline_checkpoint", "ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt"),
    ("vqvae_linux_launcher", "tools/run_vqvae_complex_recovery.sh"),
    ("sequence_rebuild_linux_launcher", "tools/run_source_path_sequence_rebuild.sh"),
    ("ar_long_context_linux_launcher", "tools/run_ar_v13_long_context.sh"),
    ("vqvae_benchmark_tool", "tools/run_vqvae_slice_benchmark.py"),
    ("vqvae_recovery_monitor", "tools/monitor_vqvae_recovery_gate.py"),
    ("generated_quality_gate", "tools/summarize_generated_quality.py"),
    ("step_geometry_entity_audit", "tools/audit_step_geometry_entities.py"),
    ("parsed_pool_quality_audit", "tools/audit_parsed_pool_quality.py"),
    ("source_path_audit_tool", "tools/audit_sequence_source_paths.py"),
    ("server_runbook", "local_reports/v13_next_server_quality_recovery_runbook_20260706.md"),
    ("server_training_readiness_verifier", "tools/verify_server_training_readiness.py"),
    ("model_artifact_sanity_verifier", "tools/verify_model_artifacts.py"),
    ("server_recovery_plan_tool", "tools/plan_server_quality_recovery.py"),
    ("server_quality_recovery_orchestrator", "tools/run_server_quality_recovery.py"),
    ("vqvae_copyback_verifier", "tools/verify_vqvae_copyback.py"),
    ("sequence_rebuild_verifier", "tools/verify_sequence_rebuild.py"),
    ("quality_recovery_stage_decider", "tools/decide_quality_recovery_stage.py"),
    ("quality_recovery_progress_summary", "tools/summarize_quality_recovery_progress.py"),
    ("server_training_phase_budget_builder", "tools/build_server_training_phase_budget.py"),
    ("server_recovery_packet_builder", "tools/build_server_recovery_packet.py"),
    ("server_first_hour_script", "local_reports/v13_server_first_hour_from_packet_20260706.sh"),
    ("server_start_here_guide", "local_reports/v13_rented_server_start_here_20260706.md"),
    ("rental_gpu_decision_card", "local_reports/v13_rental_gpu_decision_card_20260706.md"),
    ("server_training_phase_budget_json", "local_reports/v13_server_training_phase_budget_20260706.json"),
    ("server_training_phase_budget_report", "local_reports/v13_server_training_phase_budget_20260706.md"),
]

REPORT_PATHS = {
    "g20_generated": "local_runs/reconstruction_eval/eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag/generated_quality_summary.json",
    "g100_generated": "local_runs/reconstruction_eval/eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342/generated_quality_summary.json",
    "vqvae_baseline": "local_runs/reconstruction_eval/vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json",
    "length_coverage": "local_reports/v13_ar120_length_coverage_20260706.json",
    "source_path_audit": "local_reports/v13_ar120_sequence_source_path_audit_20260706.json",
}

SERVER_PHASE_ORDER = [
    "vqvae_complex_curved_recovery",
    "vqvae_four_slice_benchmark",
    "source_path_sequence_rebuild",
    "ar1536_long_context",
    "ar2048_if_memory_allows",
    "generated_reconstruction_and_paper_gate",
]

SERVER_SYNTAX_CHECKS = [
    "bash -n tools/run_vqvae_complex_recovery.sh",
    "bash -n tools/run_source_path_sequence_rebuild.sh",
    "bash -n tools/run_ar_v13_long_context.sh",
    "bash -n local_reports/v13_server_first_hour_from_packet_20260706.sh",
]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def artifact_status(repo_root: Path, label: str, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    exists = path.exists()
    return {
        "label": label,
        "path": relative_path,
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
    }


def generated_gate_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"found": False, "decision": "missing"}
    return {
        "found": True,
        "decision": (payload.get("paper_gate") or {}).get("decision", "unknown"),
        "attempted": (payload.get("summary") or {}).get("attempted"),
        "strict_valid": (payload.get("summary") or {}).get("strict_valid"),
        "top_two_fraction": (payload.get("topology") or {}).get("top_two_fraction"),
        "strict_valid_complex": (payload.get("complexity") or {}).get("strict_valid_complex"),
        "nonprimitive_strict_valid": (payload.get("semantic_complexity") or {}).get("nonprimitive_strict_valid"),
        "primitive_like_strict_valid_fraction": (payload.get("semantic_complexity") or {}).get(
            "primitive_like_strict_valid_fraction"
        ),
    }


def vqvae_gate_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"found": False, "decision": "missing"}
    slices = payload.get("slices") or {}
    return {
        "found": True,
        "decision": (payload.get("promotion_gate") or {}).get("decision", "unknown"),
        "longest_strict_valid": (slices.get("longest") or {}).get("brep_valid"),
        "most_faces_strict_valid": (slices.get("most_faces") or {}).get("brep_valid"),
    }


def length_coverage_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"found": False, "recommendation": "missing"}
    overall = payload.get("overall") or {}
    by_limit = overall.get("by_limit") or {}
    recommendation = payload.get("recommendation") or {}
    return {
        "found": True,
        "recommendation": recommendation.get("action", "unknown"),
        "preferred_max_seq_len": recommendation.get("preferred_max_seq_len"),
        "complex_total": overall.get("complex_total"),
        "limits": {
            key: {
                "complex_allowed": value.get("complex_allowed"),
                "complex_allowed_fraction": value.get("complex_allowed_fraction"),
            }
            for key, value in by_limit.items()
        },
    }


def source_path_audit_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"found": False, "validation_most_curved_ready": False}
    return {
        "found": True,
        "validation_most_curved_ready": bool(payload.get("validation_most_curved_ready")),
        "groups_with_source_path": int(payload.get("groups_with_source_path", 0) or 0),
    }


def determine_status(required_complete: bool, gates: dict[str, Any]) -> tuple[str, str]:
    if not required_complete:
        return "MISSING_LOCAL_ARTIFACTS", "restore_missing_artifacts_before_renting_server"
    if gates["vqvae_baseline"].get("decision") != "promote_for_ar_rebuild":
        return "READY_FOR_SERVER_VQVAE_RECOVERY", "rent_single_gpu_and_run_vqvae_recovery"
    if not gates["source_path_audit"].get("validation_most_curved_ready"):
        return "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD", "rebuild_source_path_sequence_package"
    return "READY_FOR_LONG_CONTEXT_AR", "train_ar1536_then_ar2048_if_memory_allows"


def build_preflight_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifacts = [artifact_status(root, label, relative_path) for label, relative_path in REQUIRED_ARTIFACTS]
    required_complete = all(item["exists"] for item in artifacts)
    reports = {name: read_json(root / path) for name, path in REPORT_PATHS.items()}
    gates = {
        "g20_generated": generated_gate_summary(reports["g20_generated"]),
        "g100_generated": generated_gate_summary(reports["g100_generated"]),
        "vqvae_baseline": vqvae_gate_summary(reports["vqvae_baseline"]),
        "length_coverage": length_coverage_summary(reports["length_coverage"]),
        "source_path_audit": source_path_audit_summary(reports["source_path_audit"]),
    }
    status, next_action = determine_status(required_complete, gates)
    missing = [item for item in artifacts if not item["exists"]]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(root),
        "status": status,
        "next_action": next_action,
        "required_artifacts_complete": bool(required_complete),
        "missing_required_artifacts": missing,
        "required_artifacts": artifacts,
        "current_gates": gates,
        "server_phase_order": SERVER_PHASE_ORDER,
        "server_syntax_checks": SERVER_SYNTAX_CHECKS,
        "recommended_first_gpu": "1x L40S 48GB or 1x RTX 6000 Ada 48GB",
        "fallback_gpu": "1x A100 80GB for AR2048 or larger VQ-VAE batches",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Handoff Preflight",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Next action: `{payload['next_action']}`",
        f"- Required artifacts complete: {payload['required_artifacts_complete']}",
        f"- Recommended first GPU: {payload['recommended_first_gpu']}",
        "",
        "## Current Gates",
        "",
    ]
    gates = payload["current_gates"]
    lines.extend([
        f"- G20 generated: `{gates['g20_generated']['decision']}`",
        "  "
        f"nonprimitive={gates['g20_generated'].get('nonprimitive_strict_valid')} "
        f"primitive_like_fraction={gates['g20_generated'].get('primitive_like_strict_valid_fraction')}",
        f"- G100 generated: `{gates['g100_generated']['decision']}`",
        "  "
        f"nonprimitive={gates['g100_generated'].get('nonprimitive_strict_valid')} "
        f"primitive_like_fraction={gates['g100_generated'].get('primitive_like_strict_valid_fraction')}",
        f"- VQ-VAE baseline: `{gates['vqvae_baseline']['decision']}`",
        "- Length coverage preferred max seq len: "
        f"{gates['length_coverage'].get('preferred_max_seq_len')}",
        "- Source-path validation ready: "
        f"{gates['source_path_audit'].get('validation_most_curved_ready')}",
        "",
        "## Server Phase Order",
        "",
    ])
    for index, phase in enumerate(payload["server_phase_order"], start=1):
        lines.append(f"{index}. {phase}")
    lines.extend(["", "## Missing Required Artifacts", ""])
    if payload["missing_required_artifacts"]:
        for item in payload["missing_required_artifacts"]:
            lines.append(f"- {item['label']}: `{item['path']}`")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V13 server handoff preflight manifest.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_preflight_manifest(args.repo_root)
    if args.output:
        write_json(args.output, manifest)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0 if manifest["required_artifacts_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
