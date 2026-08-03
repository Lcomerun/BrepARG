"""Monitor AR training until a target epoch checkpoint is ready.

This tool is intentionally read-only: it never touches the active training
process and only inspects history/checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUT_DIR = Path(r"D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar")


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def periodic_checkpoint_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / "ar_checkpoints" / f"ar_epoch_{epoch:04d}.pt"


def summarize_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "train_ce": checkpoint.get("train_ce"),
        "val_ce": checkpoint.get("val_ce"),
        "best_val_ce": checkpoint.get("best_val_ce"),
        "has_model": bool(checkpoint.get("model_state_dict")),
        "has_optimizer": bool(checkpoint.get("optimizer_state_dict")),
        "has_scaler": checkpoint.get("scaler_state_dict") is not None,
    }


def latest_resume_checkpoint_summary(out_dir: Path) -> dict[str, Any] | None:
    for name in ("ar_latest.pt", "ar_best.pt"):
        path = out_dir / name
        if path.exists():
            return summarize_checkpoint(path)
    return None


def evaluate_epoch_gate(out_dir: Path, history_rows: list[dict[str, Any]], target_epoch: int) -> dict[str, Any]:
    latest_epoch = max((int(row.get("epoch", 0)) for row in history_rows), default=0)
    latest_checkpoint = None
    if latest_epoch == 0:
        try:
            latest_checkpoint = latest_resume_checkpoint_summary(out_dir)
        except Exception as exc:  # pragma: no cover - defensive for damaged checkpoints.
            latest_checkpoint = {"load_error": f"{type(exc).__name__}:{exc}"}
        if latest_checkpoint and latest_checkpoint.get("epoch") is not None:
            latest_epoch = int(latest_checkpoint["epoch"])
    result: dict[str, Any] = {
        "ready": False,
        "target_epoch": target_epoch,
        "latest_epoch": latest_epoch,
        "history_rows": len(history_rows),
    }
    if latest_checkpoint:
        result["checkpoint"] = latest_checkpoint
    if latest_epoch < target_epoch:
        result["reason"] = f"waiting_for_epoch_{target_epoch}"
        return result

    checkpoint_path = periodic_checkpoint_path(out_dir, target_epoch)
    if not checkpoint_path.exists():
        result["reason"] = f"waiting_for_checkpoint_{checkpoint_path.name}"
        return result

    try:
        checkpoint = summarize_checkpoint(checkpoint_path)
    except Exception as exc:  # pragma: no cover - defensive for damaged checkpoints.
        result["reason"] = f"checkpoint_load_failed:{type(exc).__name__}:{exc}"
        return result

    required = checkpoint["epoch"] == target_epoch and checkpoint["has_model"] and checkpoint["has_optimizer"]
    result["checkpoint"] = checkpoint
    result["ready"] = bool(required)
    if not result["ready"]:
        result["reason"] = "checkpoint_missing_required_resume_state"
    return result


def status_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def append_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, ensure_ascii=True) + "\n")


def monitor(out_dir: Path, target_epoch: int, interval_seconds: int, once: bool, status_log: Path | None) -> int:
    while True:
        rows = read_history(out_dir / "ar_history.jsonl")
        status = evaluate_epoch_gate(out_dir, rows, target_epoch)
        status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(status_json(status), flush=True)
        if status_log is not None:
            append_status(status_log, status)
        if status["ready"]:
            return 0
        if once:
            return 1
        time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AR epoch checkpoint gate monitor.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-epoch", type=int, default=40)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return monitor(
        out_dir=args.out_dir,
        target_epoch=args.target_epoch,
        interval_seconds=args.interval_seconds,
        once=args.once,
        status_log=args.status_log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
