"""Read-only preflight for the complex-curved FSQ capacity candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODULES = ("torch", "numpy", "diffusers")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_levels(text: str) -> list[int]:
    levels = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not levels:
        raise ValueError("levels must not be empty")
    if any(value <= 1 for value in levels):
        raise ValueError(f"all FSQ levels must be > 1: {levels}")
    return levels


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def module_status(names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        out[name] = {"available": importlib.util.find_spec(name) is not None}
    return out


def cli_help_status(python_exe: Path, train_script: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(python_exe), str(train_script), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_head": (completed.stdout or "")[:500],
            "stderr_tail": (completed.stderr or "")[-1000:],
        }
    except Exception as exc:  # pragma: no cover - defensive process boundary
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_preflight(
    *,
    patch_shard_root: Path,
    outbase: Path,
    run_name: str,
    python_exe: Path,
    train_script: Path,
    samples: int,
    levels: str,
    sample_cache: Path | None = None,
    check_modules: list[str] | None = None,
    run_cli_help: bool = True,
) -> dict[str, Any]:
    patch_shard_root = Path(patch_shard_root)
    outbase = Path(outbase)
    python_exe = Path(python_exe)
    train_script = Path(train_script)
    sample_cache = Path(sample_cache) if sample_cache else None
    blocking: list[str] = []

    try:
        parsed_levels = parse_levels(levels)
        codebook_size = product(parsed_levels)
    except Exception as exc:
        parsed_levels = []
        codebook_size = 0
        blocking.append(f"invalid_levels:{type(exc).__name__}")

    if not patch_shard_root.exists():
        blocking.append("missing_patch_shard_root")
    summary_path = patch_shard_root / "_summary.json"
    summary = read_json(summary_path)
    if patch_shard_root.exists() and not summary:
        blocking.append("missing_patch_shard_summary")
    if summary:
        if str(summary.get("status")) not in {"BUILT", "SKIPPED_EXISTING"}:
            blocking.append(f"patch_shard_status:{summary.get('status')}")
        if int(summary.get("patch_shards") or 0) <= 0:
            blocking.append("patch_shards_zero")
        if int(summary.get("patches") or 0) < int(samples):
            blocking.append("patch_count_below_requested_samples")

    sample_shards = sorted(patch_shard_root.glob("vq_patch_shard_*.pkl*")) if patch_shard_root.exists() else []
    if patch_shard_root.exists() and not sample_shards:
        blocking.append("missing_patch_shard_files")

    if not train_script.exists():
        blocking.append("missing_train_script")

    modules = module_status(list(check_modules if check_modules is not None else DEFAULT_MODULES))
    for name, status in modules.items():
        if not status["available"]:
            blocking.append(f"missing_module:{name}")

    cli = {"ok": None, "skipped": True}
    if run_cli_help and train_script.exists():
        cli = cli_help_status(python_exe, train_script)
        if not cli.get("ok"):
            blocking.append("train_cli_help_failed")

    output_run = outbase / run_name
    scripts_root = outbase.parents[1] / "scripts"
    next_script = (
        scripts_root / "01a_build_fsq_capacity_sample_cache.ps1"
        if sample_cache is not None and not sample_cache.exists()
        else scripts_root / "01a_train_fsq_capacity_candidate.ps1"
    )
    report = {
        "status": "READY" if not blocking else "BLOCKED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_started": False,
        "blocking_reasons": blocking,
        "patch_shards": {
            "root": str(patch_shard_root),
            "summary_path": str(summary_path),
            "summary": summary,
            "sample_shard_count_seen": len(sample_shards),
            "sample_shards": [str(path) for path in sample_shards[:5]],
            "estimated_gb": round(sum(path.stat().st_size for path in sample_shards) / (1024**3), 3)
            if sample_shards
            else 0.0,
        },
        "config": {
            "levels": parsed_levels,
            "codebook_size": int(codebook_size),
            "samples": int(samples),
            "epochs": 180,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "complex_fraction": 0.50,
            "curved_fraction": 0.35,
            "max_source_faces": 50,
            "max_source_edges": 150,
        },
        "sample_cache": {
            "path": str(sample_cache) if sample_cache else None,
            "exists": sample_cache.exists() if sample_cache else False,
            "bytes": sample_cache.stat().st_size if sample_cache and sample_cache.exists() else 0,
            "enabled": sample_cache is not None,
        },
        "modules": modules,
        "cli": {"train": cli},
        "outputs": {
            "outbase": str(outbase),
            "run_name": str(run_name),
            "run_dir": str(output_run),
            "expected_best": str(output_run / "fsq_vqvae_best.pt"),
            "expected_history": str(output_run / "vqvae_history.json"),
        },
        "next_command": None if blocking else f"powershell -ExecutionPolicy Bypass -File {next_script}",
    }
    if report["patch_shards"]["sample_shard_count_seen"]:
        expected = int(summary.get("patch_shards") or 0) if summary else 0
        seen = int(report["patch_shards"]["sample_shard_count_seen"])
        if expected and seen != expected:
            report["patch_shards"]["warning"] = f"summary_patch_shards={expected} but files_seen={seen}"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-shard-root", type=Path, required=True)
    parser.add_argument("--outbase", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--train-script", type=Path, default=Path("breparg_improvements/train.py"))
    parser.add_argument("--samples", type=int, default=450000)
    parser.add_argument("--levels", default="16,16,8,8")
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--no-cli-help", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = [item.strip() for item in str(args.modules).split(",") if item.strip()]
    report = run_preflight(
        patch_shard_root=args.patch_shard_root,
        outbase=args.outbase,
        run_name=args.run_name,
        python_exe=args.python_exe,
        train_script=args.train_script,
        samples=args.samples,
        levels=args.levels,
        sample_cache=args.sample_cache,
        check_modules=modules,
        run_cli_help=not args.no_cli_help,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blocking_reasons": report["blocking_reasons"]}, indent=2))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
