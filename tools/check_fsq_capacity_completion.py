"""Check whether the FSQ capacity training run is safe to evaluate."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable


PASS_STATUSES = {"PASS", "VERIFIED", "complete", "COMPLETED"}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _history_summary(run_dir: Path) -> dict[str, Any]:
    payload = _load_json(run_dir / "vqvae_history.json")
    if not payload:
        return {"exists": False}
    history = payload.get("history") or []
    last = history[-1] if history else {}
    return {
        "exists": True,
        "count": len(history),
        "target_epoch": (payload.get("config") or {}).get("target_epoch"),
        "last_epoch": last.get("epoch"),
        "last_train": last.get("train_loss"),
        "last_val": last.get("val_loss"),
        "best_val": payload.get("best_val_recon"),
        "best_epoch": payload.get("best_epoch"),
        "stop_reason": payload.get("stop_reason"),
    }


def _train_report_status(run_dir: Path) -> str | None:
    payload = _load_json(run_dir / "train_report.json")
    if not payload:
        return None
    stages = payload.get("stages") or {}
    vqvae = stages.get("vqvae")
    if isinstance(vqvae, dict):
        return vqvae.get("status")
    if isinstance(vqvae, str):
        return vqvae
    return payload.get("status")


def _pid_is_alive(pid: int) -> bool:
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
            pass

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        process_query_limited_information = 0x1000
        handle = windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if handle:
            windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def check_completion(
    run_dir: Path,
    *,
    pids: Iterable[int] = (),
    live_pids: set[int] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    pids = [int(pid) for pid in pids if int(pid) > 0]
    if live_pids is None:
        alive = [pid for pid in pids if _pid_is_alive(pid)]
    else:
        alive = [pid for pid in pids if pid in live_pids]

    status = _train_report_status(run_dir)
    history = _history_summary(run_dir)
    reasons: list[str] = []
    if alive:
        reasons.append("training_process_alive")
    if status is None:
        reasons.append("train_report_missing")
    elif status not in PASS_STATUSES:
        reasons.append(f"train_report_status_not_pass:{status}")

    return {
        "complete": not reasons,
        "run_dir": str(run_dir),
        "pids": pids,
        "alive_pids": alive,
        "train_report_status": status,
        "history": history,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_completion(args.run_dir, pids=args.pid)
    text = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
