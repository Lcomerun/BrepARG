"""Read-only analysis for VQ-VAE retraining history.

The VQ-VAE trainer writes a JSON file with one row per completed epoch. This
tool turns that file into a small gate report for the current local recovery
goal: reach an e-6-scale validation reconstruction loss without obvious
overfitting before rebuilding sequences or training AR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TARGET_LOSS = 1e-6


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_from_rows(rows: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    valid = [
        (int(row.get("epoch", 0)), _finite_float(row.get("val_loss")))
        for row in rows
        if _finite_float(row.get("val_loss")) is not None
    ]
    if not valid:
        return None, None
    epoch, value = min(valid, key=lambda item: item[1])
    return epoch, value


def analyze_history(
    rows: list[dict[str, Any]],
    target_loss: float = DEFAULT_TARGET_LOSS,
    target_epoch: int | None = None,
    recent_window: int = 5,
    plateau_patience: int = 10,
    min_delta: float = 1e-8,
    recorded_best_epoch: int | None = None,
    recorded_best_val: float | None = None,
) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("epoch", 0)))
    valid_rows = [row for row in sorted_rows if _finite_float(row.get("val_loss")) is not None]
    target_loss = float(target_loss)

    if not valid_rows:
        return {
            "status": "EMPTY",
            "history_rows": len(rows),
            "target_loss": target_loss,
            "target_reached": False,
            "recommendation": "wait_for_more_history",
            "reason": "no completed epochs with finite validation loss",
        }

    latest = valid_rows[-1]
    latest_epoch = int(latest.get("epoch", 0))
    latest_train = _finite_float(latest.get("train_loss"))
    latest_val = float(_finite_float(latest.get("val_loss")))

    row_best_epoch, row_best_val = _best_from_rows(valid_rows)
    best_epoch = row_best_epoch
    best_val = row_best_val
    if recorded_best_epoch is not None and recorded_best_val is not None:
        rec_val = _finite_float(recorded_best_val)
        if rec_val is not None and (best_val is None or rec_val <= best_val):
            best_epoch = int(recorded_best_epoch)
            best_val = rec_val
    if best_epoch is None or best_val is None:
        best_epoch = latest_epoch
        best_val = latest_val

    epochs_since_best = max(0, latest_epoch - int(best_epoch))
    window = valid_rows[-max(1, int(recent_window)) :]
    recent_vals = [float(_finite_float(row.get("val_loss"))) for row in window]
    recent_trains = [_finite_float(row.get("train_loss")) for row in window]
    recent_trains = [value for value in recent_trains if value is not None]
    recent_val_delta = recent_vals[-1] - recent_vals[0] if len(recent_vals) >= 2 else 0.0
    recent_train_delta = recent_trains[-1] - recent_trains[0] if len(recent_trains) >= 2 else 0.0
    train_val_gap = latest_val - latest_train if latest_train is not None else None

    target_reached = best_val <= target_loss
    plateau_signal = epochs_since_best >= int(plateau_patience)
    latest_worse_than_best = latest_val > best_val + float(min_delta)
    train_still_improving = recent_train_delta < -float(min_delta) if recent_trains else False
    validation_worsening = recent_val_delta > float(min_delta)
    overfit_signal = bool(
        not target_reached
        and plateau_signal
        and latest_worse_than_best
        and (train_still_improving or validation_worsening)
    )

    if target_reached and not overfit_signal:
        status = "TARGET_REACHED"
        recommendation = "run_reconstruction_gate_before_sequence_or_ar"
        reason = "best validation loss reached the requested target; verify geometry reconstruction before downstream work"
    elif overfit_signal:
        status = "HOLD_ABOVE_TARGET"
        recommendation = "stop_and_diagnose_before_sequence_or_ar"
        reason = (
            f"validation has not improved for {epochs_since_best} epochs while "
            "training loss improves or validation worsens"
        )
    elif plateau_signal and latest_worse_than_best:
        status = "HOLD_ABOVE_TARGET"
        recommendation = "review_learning_rate_or_model_capacity"
        reason = f"validation has not improved for {epochs_since_best} epochs and target loss is not reached"
    else:
        status = "IN_PROGRESS"
        recommendation = "continue_training"
        reason = "validation is improving or has not exceeded plateau patience"

    remaining_epochs = None if target_epoch is None else max(0, int(target_epoch) - latest_epoch)
    return {
        "status": status,
        "history_rows": len(rows),
        "finite_history_rows": len(valid_rows),
        "latest_epoch": latest_epoch,
        "latest_train_loss": latest_train,
        "latest_val_loss": latest_val,
        "best_epoch": int(best_epoch),
        "best_val_recon": best_val,
        "epochs_since_best": epochs_since_best,
        "target_epoch": int(target_epoch) if target_epoch is not None else None,
        "remaining_epochs": remaining_epochs,
        "target_loss": target_loss,
        "target_reached": target_reached,
        "recent_window": len(window),
        "recent_val_delta": recent_val_delta,
        "recent_train_delta": recent_train_delta if recent_trains else None,
        "train_val_gap": train_val_gap,
        "plateau_signal": plateau_signal,
        "overfit_signal": overfit_signal,
        "recommendation": recommendation,
        "reason": reason,
    }


def analyze_history_file(
    path: Path,
    target_loss: float = DEFAULT_TARGET_LOSS,
    target_epoch: int | None = None,
    recent_window: int = 5,
    plateau_patience: int = 10,
    min_delta: float = 1e-8,
) -> dict[str, Any]:
    payload = read_history(path)
    config = payload.get("config") or {}
    resolved_target_epoch = target_epoch
    if resolved_target_epoch is None and config.get("target_epoch") is not None:
        resolved_target_epoch = int(config["target_epoch"])
    return analyze_history(
        list(payload.get("history") or []),
        target_loss=target_loss,
        target_epoch=resolved_target_epoch,
        recent_window=recent_window,
        plateau_patience=plateau_patience,
        min_delta=min_delta,
        recorded_best_epoch=payload.get("best_epoch"),
        recorded_best_val=payload.get("best_val_recon"),
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VQ-VAE History Analysis",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Latest epoch: `{report.get('latest_epoch')}`",
        f"- Best validation reconstruction loss: `{report.get('best_val_recon')}`",
        f"- Best epoch: `{report.get('best_epoch')}`",
        f"- Target loss: `{report.get('target_loss')}`",
        f"- Target reached: `{str(report.get('target_reached')).lower()}`",
        f"- Overfit signal: `{str(report.get('overfit_signal')).lower()}`",
        f"- Plateau signal: `{str(report.get('plateau_signal')).lower()}`",
        f"- Recommendation: `{report.get('recommendation')}`",
        "",
        "## Reason",
        "",
        str(report.get("reason")),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze VQ-VAE history against e-6 target and overfit gates.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--target-loss", type=float, default=DEFAULT_TARGET_LOSS)
    parser.add_argument("--target-epoch", type=int)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_history_file(
        args.history,
        target_loss=args.target_loss,
        target_epoch=args.target_epoch,
        recent_window=args.recent_window,
        plateau_patience=args.plateau_patience,
        min_delta=args.min_delta,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if report.get("status") == "HOLD_ABOVE_TARGET":
        return 2
    return 0 if report.get("status") in {"IN_PROGRESS", "TARGET_REACHED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
