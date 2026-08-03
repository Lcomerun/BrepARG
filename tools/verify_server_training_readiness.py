"""Verify rented-server readiness before starting VQ-VAE recovery training.

This is a server-side gate that should run after the transfer verifier. It
checks the training environment, required data/checkpoint paths, CUDA
availability, and launcher syntax without starting a long training job.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from audit_parsed_pool_quality import (
    DEFAULT_COMPLEX_MIN_EDGES,
    DEFAULT_COMPLEX_MIN_FACES,
    DEFAULT_CURVED_SCORE_THRESHOLD,
    DEFAULT_MAX_FILES as DEFAULT_PARSED_QUALITY_MAX_FILES,
    DEFAULT_MAX_LOAD_FAILURE_FRACTION,
    DEFAULT_MIN_COMPLEX_SOURCE_FRACTION,
    DEFAULT_MIN_COMPLEX_SOURCES,
    DEFAULT_MIN_CURVED_PATCH_FRACTION,
    DEFAULT_MIN_CURVED_PATCHES,
    DEFAULT_MIN_PARSED_FILES,
    audit_parsed_pool_quality,
)


REQUIRED_PYTHON_MODULES = [
    "torch",
    "numpy",
    "OCC.Core.TopoDS",
]

REQUIRED_LAUNCHERS = [
    "tools/run_vqvae_complex_recovery.sh",
    "tools/run_source_path_sequence_rebuild.sh",
    "tools/run_ar_v13_long_context.sh",
]

DEFAULT_MIN_GPU_MEMORY_GB = 40.0


def directory_nonempty(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except (StopIteration, OSError):
        return False


def path_status(label: str, path: str | Path, kind: str = "file") -> dict[str, Any]:
    resolved = Path(path)
    exists = resolved.exists()
    is_file = resolved.is_file()
    is_dir = resolved.is_dir()
    nonempty = bool(
        (kind == "dir" and is_dir and directory_nonempty(resolved))
        or (kind == "file" and is_file and resolved.stat().st_size > 0)
    )
    issues: list[str] = []
    if not exists:
        issues.append("missing")
    elif kind == "file" and not is_file:
        issues.append("not_file")
    elif kind == "dir" and not is_dir:
        issues.append("not_dir")
    elif not nonempty:
        issues.append("empty")
    return {
        "label": label,
        "path": str(resolved),
        "kind": kind,
        "exists": bool(exists),
        "nonempty": bool(nonempty),
        "ok": len(issues) == 0,
        "issues": issues,
        "bytes": int(resolved.stat().st_size) if exists and is_file else 0,
    }


def check_python_module(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError) as exc:
        return {"module": name, "ok": False, "reason": str(exc)}
    return {
        "module": name,
        "ok": spec is not None,
        "reason": "available" if spec is not None else "missing",
    }


def probe_cuda() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on server image
        return {
            "torch_import_ok": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "reason": str(exc),
        }
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                }
            )
    return {
        "torch_import_ok": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": devices,
    }


def summarize_gpu_memory(cuda: dict[str, Any], minimum_required_gb: float) -> dict[str, Any]:
    devices = list(cuda.get("devices") or [])
    memories = [
        float(device.get("total_memory_gb", 0.0) or 0.0)
        for device in devices
        if isinstance(device, dict)
    ]
    largest = max(memories) if memories else 0.0
    meets = bool(cuda.get("cuda_available")) and largest >= float(minimum_required_gb)
    return {
        "minimum_required_gb": float(minimum_required_gb),
        "largest_device_memory_gb": round(largest, 2),
        "meets_minimum": meets,
        "reason": "ok" if meets else "largest_cuda_device_below_minimum",
        "recommended_first_gpu": "1x L40S 48GB or 1x RTX 6000 Ada/A6000 48GB",
        "upgrade_gpu": "1x A100 80GB for AR2048 or larger VQ-VAE batches",
    }


def check_bash_syntax(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "ok": False, "reason": "missing_script"}
    try:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {"path": str(path), "ok": False, "reason": "bash_not_found"}
    except subprocess.TimeoutExpired:
        return {"path": str(path), "ok": False, "reason": "bash_syntax_timeout"}
    return {
        "path": str(path),
        "ok": completed.returncode == 0,
        "reason": "syntax_ok" if completed.returncode == 0 else "syntax_failed",
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_transfer_status(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "status": "not_provided", "ok": False}
    resolved = Path(path)
    if not resolved.exists():
        return {"path": str(resolved), "status": "missing", "ok": False}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(resolved), "status": "invalid_json", "ok": False, "reason": str(exc)}
    status = payload.get("status", "unknown")
    return {
        "path": str(resolved),
        "status": status,
        "ok": status == "READY_FOR_SERVER_RUN",
        "summary": payload.get("summary"),
    }


def evaluate_training_readiness(
    repo_root: str | Path,
    parsed_pool: str | Path,
    vqvae_checkpoint: str | Path,
    sequence: str | Path,
    split: str | Path,
    transfer_verification: str | Path | None,
    module_checker: Callable[[str], dict[str, Any]] = check_python_module,
    cuda_probe: Callable[[], dict[str, Any]] = probe_cuda,
    script_checker: Callable[[Path], dict[str, Any]] = check_bash_syntax,
    min_gpu_memory_gb: float = DEFAULT_MIN_GPU_MEMORY_GB,
    parsed_quality_max_files: int = DEFAULT_PARSED_QUALITY_MAX_FILES,
    min_parsed_files: int = DEFAULT_MIN_PARSED_FILES,
    min_complex_sources: int = DEFAULT_MIN_COMPLEX_SOURCES,
    min_complex_source_fraction: float = DEFAULT_MIN_COMPLEX_SOURCE_FRACTION,
    min_curved_patches: int = DEFAULT_MIN_CURVED_PATCHES,
    min_curved_patch_fraction: float = DEFAULT_MIN_CURVED_PATCH_FRACTION,
    curved_score_threshold: float = DEFAULT_CURVED_SCORE_THRESHOLD,
    max_load_failure_fraction: float = DEFAULT_MAX_LOAD_FAILURE_FRACTION,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    required_paths = [
        path_status("repo_root", root, "dir"),
        path_status("parsed_abc_pool", parsed_pool, "dir"),
        path_status("vqvae_baseline_checkpoint", vqvae_checkpoint, "file"),
        path_status("ar_sequence_package", sequence, "file"),
        path_status("ar_split_file", split, "file"),
    ]
    for launcher in REQUIRED_LAUNCHERS:
        required_paths.append(path_status(f"launcher:{launcher}", root / launcher, "file"))
    modules = [module_checker(name) for name in REQUIRED_PYTHON_MODULES]
    cuda = cuda_probe()
    gpu_memory = summarize_gpu_memory(cuda, min_gpu_memory_gb)
    scripts = [script_checker(root / launcher) for launcher in REQUIRED_LAUNCHERS]
    transfer = read_transfer_status(transfer_verification)
    parsed_quality = audit_parsed_pool_quality(
        parsed_pool,
        max_files=parsed_quality_max_files,
        complex_min_faces=DEFAULT_COMPLEX_MIN_FACES,
        complex_min_edges=DEFAULT_COMPLEX_MIN_EDGES,
        curved_score_threshold=curved_score_threshold,
        min_parsed_files=min_parsed_files,
        min_complex_sources=min_complex_sources,
        min_complex_source_fraction=min_complex_source_fraction,
        min_curved_patches=min_curved_patches,
        min_curved_patch_fraction=min_curved_patch_fraction,
        max_load_failure_fraction=max_load_failure_fraction,
    )

    blocking_reasons: list[str] = []
    if any(not item["ok"] for item in required_paths):
        blocking_reasons.append("required_paths_failed")
    if any(not item.get("ok") for item in modules):
        blocking_reasons.append("python_modules_failed")
    if not cuda.get("torch_import_ok"):
        blocking_reasons.append("torch_import_failed")
    if not cuda.get("cuda_available"):
        blocking_reasons.append("cuda_unavailable")
    if cuda.get("cuda_available") and not gpu_memory["meets_minimum"]:
        blocking_reasons.append("gpu_memory_below_minimum")
    if any(not item.get("ok") for item in scripts):
        blocking_reasons.append("launcher_syntax_failed")
    if not transfer.get("ok"):
        blocking_reasons.append("transfer_verification_not_ready")
    if not parsed_quality.get("quality_ready"):
        blocking_reasons.append("parsed_pool_quality_failed")

    training_allowed = len(blocking_reasons) == 0
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY_FOR_VQVAE_TRAINING" if training_allowed else "SERVER_TRAINING_READINESS_FAILED",
        "training_allowed": training_allowed,
        "blocking_reasons": blocking_reasons,
        "summary": {
            "required_paths_total": len(required_paths),
            "required_paths_failed": sum(1 for item in required_paths if not item["ok"]),
            "python_modules_total": len(modules),
            "python_modules_failed": sum(1 for item in modules if not item.get("ok")),
            "launcher_scripts_total": len(scripts),
            "launcher_scripts_failed": sum(1 for item in scripts if not item.get("ok")),
            "parsed_pool_quality_status": parsed_quality.get("status"),
            "parsed_pool_loaded_files": (parsed_quality.get("summary") or {}).get("loaded_files"),
            "parsed_pool_complex_sources": (parsed_quality.get("summary") or {}).get("complex_source_files"),
            "parsed_pool_curved_patches": (parsed_quality.get("summary") or {}).get("curved_patch_records"),
        },
        "required_paths": required_paths,
        "python_modules": modules,
        "cuda": cuda,
        "gpu_memory": gpu_memory,
        "launcher_syntax": scripts,
        "transfer_verification": transfer,
        "parsed_pool_quality": parsed_quality,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Training Readiness",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Training allowed: {payload['training_allowed']}",
        f"- Blocking reasons: {', '.join(payload['blocking_reasons']) if payload['blocking_reasons'] else 'none'}",
        "",
        "## Summary",
        "",
        f"- Required paths failed: {payload['summary']['required_paths_failed']}/{payload['summary']['required_paths_total']}",
        f"- Python modules failed: {payload['summary']['python_modules_failed']}/{payload['summary']['python_modules_total']}",
        f"- Launcher syntax failed: {payload['summary']['launcher_scripts_failed']}/{payload['summary']['launcher_scripts_total']}",
        f"- CUDA available: {payload['cuda'].get('cuda_available')}",
        f"- Largest GPU memory: {payload['gpu_memory'].get('largest_device_memory_gb')} GB",
        f"- Minimum GPU memory: {payload['gpu_memory'].get('minimum_required_gb')} GB",
        f"- GPU memory suitable: {payload['gpu_memory'].get('meets_minimum')}",
        f"- Transfer verification: `{payload['transfer_verification'].get('status')}`",
        f"- Parsed pool quality: `{payload['parsed_pool_quality'].get('status')}`",
        "- Parsed pool loaded/complex/curved: "
        f"{payload['summary'].get('parsed_pool_loaded_files')} / "
        f"{payload['summary'].get('parsed_pool_complex_sources')} / "
        f"{payload['summary'].get('parsed_pool_curved_patches')}",
        "",
    ]
    failed_paths = [item for item in payload["required_paths"] if not item["ok"]]
    if failed_paths:
        lines.extend(["## Failed Required Paths", ""])
        for item in failed_paths:
            lines.append(f"- `{item['label']}` at `{item['path']}`: {', '.join(item['issues'])}")
        lines.append("")
    failed_modules = [item for item in payload["python_modules"] if not item.get("ok")]
    if failed_modules:
        lines.extend(["## Failed Python Modules", ""])
        for item in failed_modules:
            lines.append(f"- `{item['module']}`: {item.get('reason')}")
        lines.append("")
    parsed_quality = payload["parsed_pool_quality"]
    if not parsed_quality.get("quality_ready"):
        lines.extend(["## Parsed Pool Quality Hold", ""])
        lines.append(
            "- Blocking reasons: "
            + (", ".join(parsed_quality.get("blocking_reasons") or []) or "none")
        )
        summary = parsed_quality.get("summary") or {}
        lines.append(
            "- Loaded files / complex sources / curved patches: "
            f"{summary.get('loaded_files')} / {summary.get('complex_source_files')} / {summary.get('curved_patch_records')}"
        )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify server readiness before V13 VQ-VAE recovery training.")
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/V13"))
    parser.add_argument("--parsed-pool", type=Path, default=Path("/workspace/ABC/processed/abc_parsed_full"))
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt"),
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl"),
    )
    parser.add_argument(
        "--transfer-verification",
        type=Path,
        default=Path("/workspace/V13/local_reports/v13_server_transfer_verify_server.json"),
    )
    parser.add_argument(
        "--min-gpu-memory-gb",
        type=float,
        default=DEFAULT_MIN_GPU_MEMORY_GB,
        help="Minimum largest-GPU memory required before starting VQ-VAE recovery.",
    )
    parser.add_argument("--parsed-quality-max-files", type=int, default=DEFAULT_PARSED_QUALITY_MAX_FILES)
    parser.add_argument("--min-parsed-files", type=int, default=DEFAULT_MIN_PARSED_FILES)
    parser.add_argument("--min-complex-sources", type=int, default=DEFAULT_MIN_COMPLEX_SOURCES)
    parser.add_argument("--min-complex-source-fraction", type=float, default=DEFAULT_MIN_COMPLEX_SOURCE_FRACTION)
    parser.add_argument("--min-curved-patches", type=int, default=DEFAULT_MIN_CURVED_PATCHES)
    parser.add_argument("--min-curved-patch-fraction", type=float, default=DEFAULT_MIN_CURVED_PATCH_FRACTION)
    parser.add_argument("--curved-score-threshold", type=float, default=DEFAULT_CURVED_SCORE_THRESHOLD)
    parser.add_argument("--max-load-failure-fraction", type=float, default=DEFAULT_MAX_LOAD_FAILURE_FRACTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_training_readiness(
        repo_root=args.repo_root,
        parsed_pool=args.parsed_pool,
        vqvae_checkpoint=args.vqvae_checkpoint,
        sequence=args.sequence,
        split=args.split,
        transfer_verification=args.transfer_verification,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        parsed_quality_max_files=args.parsed_quality_max_files,
        min_parsed_files=args.min_parsed_files,
        min_complex_sources=args.min_complex_sources,
        min_complex_source_fraction=args.min_complex_source_fraction,
        min_curved_patches=args.min_curved_patches,
        min_curved_patch_fraction=args.min_curved_patch_fraction,
        curved_score_threshold=args.curved_score_threshold,
        max_load_failure_fraction=args.max_load_failure_fraction,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
