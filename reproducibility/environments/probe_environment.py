from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MODULES = (
    "torch",
    "numpy",
    "scipy",
    "transformers",
    "diffusers",
    "einops",
    "zstandard",
)


def module_probe(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def nvidia_probe() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "status": "command_missing"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "devices": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        "stderr": completed.stderr.strip(),
    }


def cuda_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "status": "torch_import_failed", "error": repr(exc)}
    if not torch.cuda.is_available():
        return {
            "available": False,
            "status": "cuda_unavailable",
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
    try:
        left = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
        result = left @ left.T
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(result).all().item())
    except Exception as exc:
        return {"available": False, "status": "kernel_failed", "error": repr(exc)}
    return {
        "available": finite,
        "status": "ok" if finite else "nonfinite_kernel_result",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "device_count": torch.cuda.device_count(),
    }


def source_probe(source_root: Path | None) -> dict[str, Any]:
    if source_root is None:
        return {"configured": False}
    required = (
        "breparg_improvements/train.py",
        "breparg_improvements/fsq_quantise.py",
        "BrepARG/model.py",
        "tools/complex_curved_diagnostics.py",
    )
    missing = [relative for relative in required if not (source_root / relative).is_file()]
    return {
        "configured": True,
        "root": str(source_root.resolve()),
        "required_files": list(required),
        "missing_files": missing,
        "ready": not missing,
    }


def build_report(source_root: Path | None) -> dict[str, Any]:
    modules = {name: module_probe(name) for name in REQUIRED_MODULES}
    cuda = cuda_probe()
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "modules": modules,
        "nvidia": nvidia_probe(),
        "cuda": cuda,
        "source": source_probe(source_root),
        "core_imports_ready": all(row["available"] for row in modules.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = build_report(args.source_root)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["core_imports_ready"]:
        return 2
    if args.require_cuda and not report["cuda"]["available"]:
        return 3
    if args.source_root is not None and not report["source"]["ready"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
