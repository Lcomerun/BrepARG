"""Read-only analysis for the V13 AR training history.

The active AR trainer already writes one JSONL row per completed epoch. This
tool turns that history into a small decision report without touching the
running training process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_HISTORY = Path(
    r"D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\ar_history.jsonl"
)


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _finite_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _epoch_minutes(rows: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    previous_elapsed = 0.0
    for row in rows:
        elapsed = _finite_float(row, "elapsed_min")
        if elapsed is None:
            continue
        duration = elapsed - previous_elapsed
        if duration > 0:
            durations.append(duration)
        previous_elapsed = elapsed
    return durations


def analyze_history(
    rows: list[dict[str, Any]],
    target_epoch: int = 100,
    recent_window: int = 5,
    plateau_patience: int = 5,
    min_delta: float = 0.0,
    baseline_best_epoch: int | None = None,
    baseline_best_val_ce: float | None = None,
) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("epoch", 0)))
    valid_rows = [row for row in sorted_rows if _finite_float(row, "val_ce") is not None]
    if not valid_rows:
        if baseline_best_epoch is None or baseline_best_val_ce is None:
            return {
                "status": "EMPTY",
                "history_rows": len(rows),
                "recommendation": "wait_for_more_history",
                "reason": "no completed epochs with finite validation CE",
            }
        return {
            "status": "BASELINE_ONLY",
            "history_rows": len(rows),
            "best_epoch": int(baseline_best_epoch),
            "best_val_ce": float(baseline_best_val_ce),
            "target_epoch": int(target_epoch),
            "remaining_epochs": None,
            "recommendation": "wait_for_more_history",
            "reason": "only baseline best is available; wait for completed continuation epochs",
        }

    latest = valid_rows[-1]
    latest_epoch = int(latest.get("epoch", 0))
    best = min(valid_rows, key=lambda row: _finite_float(row, "val_ce"))
    best_epoch = int(best.get("epoch", 0))
    best_val = float(_finite_float(best, "val_ce"))
    if baseline_best_epoch is not None and baseline_best_val_ce is not None:
        baseline_val = float(baseline_best_val_ce)
        if baseline_val <= best_val:
            best_epoch = int(baseline_best_epoch)
            best_val = baseline_val
    latest_val = float(_finite_float(latest, "val_ce"))
    latest_train = _finite_float(latest, "train_ce")
    epochs_since_best = max(0, latest_epoch - best_epoch)

    window = valid_rows[-max(1, recent_window):]
    recent_vals = [float(_finite_float(row, "val_ce")) for row in window]
    recent_trains = [_finite_float(row, "train_ce") for row in window]
    recent_trains = [value for value in recent_trains if value is not None]
    recent_val_delta = recent_vals[-1] - recent_vals[0] if len(recent_vals) >= 2 else 0.0
    recent_train_delta = recent_trains[-1] - recent_trains[0] if len(recent_trains) >= 2 else 0.0
    train_val_gap = latest_val - latest_train if latest_train is not None else None

    durations = _epoch_minutes(valid_rows)
    recent_durations = durations[-max(1, recent_window):]
    avg_epoch_min = sum(recent_durations) / len(recent_durations) if recent_durations else None
    remaining_epochs = max(0, int(target_epoch) - latest_epoch)
    eta_hours = (remaining_epochs * avg_epoch_min / 60.0) if avg_epoch_min is not None else None

    plateau_signal = epochs_since_best >= plateau_patience
    latest_worse_than_best = latest_val > best_val + min_delta
    train_still_improving = recent_train_delta < -min_delta if recent_trains else False
    validation_worsening = recent_val_delta > min_delta
    overfit_signal = bool(plateau_signal and latest_worse_than_best and (train_still_improving or validation_worsening))

    if overfit_signal:
        recommendation = "consider_stop_or_lower_lr"
        reason = (
            f"validation has not improved for {epochs_since_best} epochs and "
            "recent train CE is still improving or validation CE is worsening"
        )
    elif plateau_signal and latest_worse_than_best:
        recommendation = "review_at_checkpoint"
        reason = f"validation has not improved for {epochs_since_best} epochs"
    else:
        recommendation = "continue_unchanged"
        reason = "validation is still improving recently or has not exceeded plateau patience"

    return {
        "status": "VERIFIED",
        "history_rows": len(rows),
        "latest_epoch": latest_epoch,
        "latest_train_ce": latest_train,
        "latest_val_ce": latest_val,
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "epochs_since_best": epochs_since_best,
        "target_epoch": int(target_epoch),
        "remaining_epochs": remaining_epochs,
        "avg_epoch_min_recent": avg_epoch_min,
        "eta_to_target_hours": eta_hours,
        "recent_window": len(window),
        "recent_val_delta": recent_val_delta,
        "recent_train_delta": recent_train_delta if recent_trains else None,
        "train_val_gap": train_val_gap,
        "plateau_signal": plateau_signal,
        "overfit_signal": overfit_signal,
        "recommendation": recommendation,
        "reason": reason,
    }


def status_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AR training history analysis.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--target-epoch", type=int, default=100)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--baseline-best-epoch", type=int)
    parser.add_argument("--baseline-best-val-ce", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_history(
        read_history(args.history),
        target_epoch=args.target_epoch,
        recent_window=args.recent_window,
        plateau_patience=args.plateau_patience,
        min_delta=args.min_delta,
        baseline_best_epoch=args.baseline_best_epoch,
        baseline_best_val_ce=args.baseline_best_val_ce,
    )
    rendered = status_json(report)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("status") == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
