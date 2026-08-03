import math
import json
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


@dataclass
class VQVAEStopConfig:
    min_epochs: int = 12
    patience: int = 8
    max_nonfinite_val_epochs: int = 2
    min_delta: float = 1e-5


@dataclass
class VQVAEStopState:
    best_val: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    consecutive_nonfinite_val_epochs: int = 0
    stop_reason: str = ""


def parse_env_bool(value, default=False):
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return default


def safe_json_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return value


def finite_average(total, count):
    return total / count if count else float("inf")


def summarize_vqvae_history(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    history = list(data.get("history") or [])
    last = history[-1] if history else {}
    last_epoch = int(last.get("epoch", -1))
    best_val = data.get("best_val_recon")
    if best_val is None:
        best_val = last.get("best_val")
    return {
        "history_count": len(history),
        "last_epoch": last_epoch,
        "next_epoch": last_epoch + 1,
        "best_epoch": int(data.get("best_epoch", last.get("best_epoch", -1))),
        "best_val_recon": safe_json_number(best_val),
        "final_val": safe_json_number(last.get("val_loss")),
    }


def continuation_epoch_count(start_epoch, target_epoch):
    count = int(target_epoch) - int(start_epoch)
    if count <= 0:
        raise ValueError("target_epoch must be greater than start_epoch")
    return count


def _is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def update_vqvae_stop_state(epoch, val_loss, state, config):
    finite_val = _is_finite(val_loss)
    improved = False
    should_stop = False

    if finite_val:
        state.consecutive_nonfinite_val_epochs = 0
        if val_loss < state.best_val - config.min_delta:
            state.best_val = val_loss
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            improved = True
        else:
            state.epochs_without_improvement += 1
    else:
        state.consecutive_nonfinite_val_epochs += 1
        state.epochs_without_improvement += 1

    reached_min_epochs = epoch + 1 >= config.min_epochs
    if reached_min_epochs and config.max_nonfinite_val_epochs > 0:
        if state.consecutive_nonfinite_val_epochs >= config.max_nonfinite_val_epochs:
            state.stop_reason = f"nonfinite_val_epochs={state.consecutive_nonfinite_val_epochs}"
            should_stop = True

    if not should_stop and reached_min_epochs and config.patience > 0:
        if state.epochs_without_improvement >= config.patience:
            state.stop_reason = f"patience={config.patience}"
            should_stop = True

    return state, improved, should_stop
