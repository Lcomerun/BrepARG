"""Preflight checks for the same-data BrepARG fallback baseline.

This tool is intentionally read-only. It verifies the inputs and command-line
surface needed by ``03b_breparg_same_data_training_fallback.ps1`` without
starting VQ-VAE or AR training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_INPUTS = {
    "summary": "same_data_input_summary.json",
    "split": "same_data_split.pkl",
    "surfaces": "deduplicated_surface_source.pkl",
    "edges": "deduplicated_edge_source.pkl",
}

DEFAULT_MODULES = ("torch", "tensorboard", "diffusers", "transformers", "OCC", "occwl", "shutup", "tqdm")
CLI_HELP_TARGETS = {
    "train_vqvae": ("BrepARG/train_vqvae.py", "--help"),
    "2sequence": ("BrepARG/2sequence.py", "--help"),
    "train_ar": ("BrepARG/train_ar.py", "--help"),
    "generate_brep": ("BrepARG/generate_brep.py", "--help"),
    "audit_breparg_baseline_outputs": ("tools/audit_breparg_baseline_outputs.py", "--help"),
}
PLANNED_COMMANDS = {
    "train_vqvae": {
        "script": "BrepARG/train_vqvae.py",
        "args": [
            "--data_list",
            "--surface_list",
            "--edge_list",
            "--dataset_type",
            "--batch_size",
            "--train_epoch",
            "--test_epoch",
            "--save_epoch",
            "--max_face",
            "--max_edge",
            "--dir_name",
            "--env",
            "--loss_dir",
            "--tb_log_dir",
            "--no_aug",
            "--gpu",
        ],
    },
    "2sequence": {
        "script": "BrepARG/2sequence.py",
        "args": [
            "--data_list",
            "--output_file",
            "--vqvae_se_weight",
            "--dataset_type",
            "--max_face",
            "--max_edge",
            "--scale",
            "--aug",
            "--gpu",
        ],
    },
    "train_ar": {
        "script": "BrepARG/train_ar.py",
        "args": [
            "--sequence_file",
            "--dataset_type",
            "--batch_size",
            "--train_epoch",
            "--test_epoch",
            "--save_epoch",
            "--max_face",
            "--max_edge",
            "--max_seq_len",
            "--learning_rate",
            "--d_model",
            "--nhead",
            "--num_layers",
            "--dim_feedforward",
            "--dir_name",
            "--env",
            "--loss_dir",
            "--tb_log_dir",
        ],
    },
    "generate_brep": {
        "script": "BrepARG/generate_brep.py",
        "args": [
            "--dataset_type",
            "--config",
            "--ar_model",
            "--se_vqvae",
            "--num_samples",
            "--max_attempts",
            "--mode",
            "--max_length",
            "--temperature",
            "--top_p",
            "--output_dir",
            "--filename_prefix",
            "--device",
            "--gpu",
        ],
    },
    "audit_breparg_baseline_outputs": {
        "script": "tools/audit_breparg_baseline_outputs.py",
        "args": [
            "--run-dir",
            "--output",
            "--markdown-output",
            "--manifest-output",
            "--min-faces",
            "--min-edges",
            "--max-faces",
            "--max-edges",
        ],
    },
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def split_counts(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = payload.get(split)
        if not hasattr(rows, "__len__"):
            return None
        counts[split] = int(len(rows))
    return counts


def module_status(names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        out[name] = {"available": spec is not None}
    return out


def run_help_checks(python_exe: Path, targets: dict[str, tuple[str, str]], cwd: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, args in targets.items():
        cmd = [str(python_exe), *args]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=60,
            )
            out[name] = {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "stderr_tail": (completed.stderr or "")[-1000:],
                "stdout_head": (completed.stdout or "")[:500],
            }
        except Exception as exc:  # pragma: no cover - defensive process boundary
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def planned_commands() -> dict[str, dict[str, Any]]:
    return {
        name: {"script": spec["script"], "args": list(spec["args"])}
        for name, spec in PLANNED_COMMANDS.items()
    }


def check_required_cli_args(
    help_checks: dict[str, dict[str, Any]],
    names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected = names if names is not None else set(PLANNED_COMMANDS)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(selected):
        expected = list((PLANNED_COMMANDS.get(name) or {}).get("args") or [])
        status = help_checks.get(name) or {}
        help_text = "\n".join(
            str(status.get(key) or "")
            for key in ("stdout", "stderr", "stdout_head", "stderr_tail")
        )
        missing = [arg for arg in expected if arg not in help_text]
        out[name] = {
            "ok": bool(status.get("ok")) and not missing,
            "skipped": bool(status.get("skipped")),
            "expected": expected,
            "missing": missing,
        }
    return out


def official_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_checked"}
    report = read_json(path)
    if report is None:
        return {"status": "missing", "path": str(path)}
    return {
        "status": report.get("status") or "unknown",
        "path": str(path),
        "abc_ar_vocab": (((report.get("checkpoint_shapes") or {}).get("abc_ar_transformer_wte"))),
    }


def run_preflight(
    *,
    root: Path,
    data_dir: Path,
    python_exe: Path,
    check_modules: list[str] | None = None,
    run_cli_help: bool = True,
    official_incompat_report: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    data_dir = Path(data_dir)
    python_exe = Path(python_exe)
    blocking: list[str] = []
    files: dict[str, dict[str, Any]] = {}

    for label, filename in REQUIRED_INPUTS.items():
        path = data_dir / filename
        files[label] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        if not path.exists():
            blocking.append(f"missing:{filename}")

    summary = read_json(data_dir / REQUIRED_INPUTS["summary"])
    if summary and str(summary.get("status")) not in {"VERIFIED", "SKIPPED_EXISTING"}:
        blocking.append(f"summary_status:{summary.get('status')}")

    counts = None
    if files["split"]["exists"]:
        try:
            counts = split_counts(read_pickle(data_dir / REQUIRED_INPUTS["split"]))
            if counts is None:
                blocking.append("split_pickle_has_unexpected_shape")
            elif any(value <= 0 for value in counts.values()):
                blocking.append(f"empty_split:{counts}")
        except Exception as exc:
            blocking.append(f"split_pickle_unreadable:{type(exc).__name__}")

    for label in ("surfaces", "edges"):
        if files[label]["exists"]:
            try:
                payload = read_pickle(data_dir / REQUIRED_INPUTS[label])
                files[label]["items"] = len(payload) if hasattr(payload, "__len__") else None
                if not files[label]["items"]:
                    blocking.append(f"empty:{REQUIRED_INPUTS[label]}")
            except Exception as exc:
                blocking.append(f"pickle_unreadable:{REQUIRED_INPUTS[label]}:{type(exc).__name__}")

    modules = module_status(list(check_modules if check_modules is not None else DEFAULT_MODULES))
    for name, status in modules.items():
        if not status["available"]:
            blocking.append(f"missing_module:{name}")

    planned = planned_commands()
    cli = {name: {"ok": None, "skipped": True} for name in CLI_HELP_TARGETS}
    cli_required_args = {name: {"ok": None, "skipped": True} for name in CLI_HELP_TARGETS}
    if run_cli_help:
        cli = run_help_checks(python_exe, CLI_HELP_TARGETS, cwd=Path.cwd())
        for name, status in cli.items():
            if not status.get("ok"):
                blocking.append(f"cli_help_failed:{name}")
        cli_required_args = check_required_cli_args(cli)
        for name, status in cli_required_args.items():
            if not status.get("ok"):
                missing = ",".join(status.get("missing") or [])
                blocking.append(f"cli_arg_missing:{name}:{missing}")

    outputs = {
        "vqvae_run": str(root / "vqvae"),
        "sequence_run": str(root / "sequence"),
        "ar_run": str(root / "ar"),
        "generated": str(root / "generated"),
    }

    report = {
        "status": "READY" if not blocking else "BLOCKED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_started": False,
        "root": str(root),
        "data_dir": str(data_dir),
        "python_exe": str(python_exe),
        "blocking_reasons": blocking,
        "official_baseline": official_status(official_incompat_report),
        "inputs": {
            "files": files,
            "summary": summary,
            "split_counts": counts,
        },
        "modules": modules,
        "cli": cli,
        "cli_required_args": cli_required_args,
        "planned_commands": planned,
        "outputs": outputs,
        "next_command": (
            f"powershell -ExecutionPolicy Bypass -File {root.parents[1] / 'scripts' / '03b_breparg_same_data_training_fallback.ps1'}"
            if not blocking
            else None
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--official-incompat-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cli-help", action="store_true")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = [item.strip() for item in str(args.modules).split(",") if item.strip()]
    report = run_preflight(
        root=args.root,
        data_dir=args.data_dir,
        python_exe=args.python_exe,
        check_modules=modules,
        run_cli_help=not args.no_cli_help,
        official_incompat_report=args.official_incompat_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blocking_reasons": report["blocking_reasons"]}, indent=2))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
