from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}
CHECKPOINT_SCHEMA = "vq_training_state_v1"


class NonFiniteTrainingError(RuntimeError):
    """Raised as soon as a formal run observes non-finite numeric state."""


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    device_type: str
    autocast_dtype: torch.dtype | None
    grad_scaler_enabled: bool

    def autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def as_dict(self):
        return {
            "name": self.name,
            "device_type": self.device_type,
            "autocast_dtype": (
                str(self.autocast_dtype).removeprefix("torch.")
                if self.autocast_dtype is not None
                else None
            ),
            "grad_scaler_enabled": self.grad_scaler_enabled,
        }


def resolve_precision(name, *, cuda_available=None, bf16_supported=None):
    normalized = str(name).strip().lower()
    if normalized not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: fp32, fp16, bf16")
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if normalized in {"fp16", "bf16"} and not cuda_available:
        raise ValueError(f"{normalized} precision requires CUDA")
    if normalized == "bf16":
        if bf16_supported is None:
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            raise ValueError("the selected CUDA device does not support bf16")
    dtype = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[normalized]
    return PrecisionPolicy(
        name=normalized,
        device_type="cuda" if cuda_available else "cpu",
        autocast_dtype=dtype,
        grad_scaler_enabled=normalized == "fp16",
    )


def _is_finite_auditable_tensor(value):
    return torch.is_tensor(value) and (
        value.is_floating_point() or value.is_complex()
    )


def _nonfinite_tensor_count(value):
    if not _is_finite_auditable_tensor(value):
        return 0
    return int(torch.count_nonzero(~torch.isfinite(value.detach())).item())


def _assert_finite_tensor(value, name):
    count = _nonfinite_tensor_count(value)
    if count:
        raise NonFiniteTrainingError(f"{name} contains {count} non-finite values")


def _iter_optimizer_tensors(value, prefix):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_optimizer_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_optimizer_tensors(child, f"{prefix}[{index}]")


def clip_gradients_strict(model, max_norm):
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not named_gradients:
        raise NonFiniteTrainingError("no gradients were produced for this batch")
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            float(max_norm),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        # The normal path lets clip_grad_norm_ reduce all gradients together.
        # Only pay for per-tensor synchronization when a failure needs a name.
        for name, gradient in named_gradients:
            _assert_finite_tensor(gradient, f"gradient:{name}")
        raise NonFiniteTrainingError(f"gradient norm is non-finite: {exc}") from exc
    norm_value = float(norm.detach().cpu()) if torch.is_tensor(norm) else float(norm)
    if not math.isfinite(norm_value):
        raise NonFiniteTrainingError(f"gradient norm is non-finite: {norm_value!r}")
    return norm_value


def _finite_state_entries(model, optimizer=None):
    for name, value in model.state_dict().items():
        yield "model", f"model:{name}", value
    seen_pools = set()
    for module_name, module in model.named_modules():
        pool = getattr(module, "pool", None)
        if pool is None or id(pool) in seen_pools:
            continue
        seen_pools.add(id(pool))
        features = getattr(pool, "features", None)
        if features is not None:
            prefix = f"{module_name}." if module_name else ""
            yield (
                "feature_pool",
                f"model:{prefix}pool.features",
                features,
            )
    if optimizer is not None:
        for name, value in _iter_optimizer_tensors(optimizer.state_dict(), "optimizer"):
            yield "optimizer", name, value


def audit_finite_training_state(model, optimizer=None):
    """Audit complete floating-point training state with one scalar check per device.

    The finite path launches a reduction for every tensor but combines the scalar
    results on-device before reading them on the host. A detailed per-tensor scan
    is reserved for the failure path, where naming the corrupt tensor matters more
    than throughput.
    """
    entries = [
        (source, name, value)
        for source, name, value in _finite_state_entries(model, optimizer)
        if _is_finite_auditable_tensor(value)
    ]
    source_counts = {
        source: {"tensors": 0, "elements": 0}
        for source in ("model", "feature_pool", "optimizer")
    }
    device_checks = {}
    for source, _name, value in entries:
        detached = value.detach()
        source_counts[source]["tensors"] += 1
        source_counts[source]["elements"] += int(detached.numel())
        device = str(detached.device)
        tensor_finite = torch.isfinite(detached).all()
        if device in device_checks:
            device_checks[device] = torch.logical_and(
                device_checks[device], tensor_finite
            )
        else:
            device_checks[device] = tensor_finite

    failing_devices = {
        device
        for device, finite in device_checks.items()
        if not bool(finite.item())
    }
    if failing_devices:
        for _source, name, value in entries:
            if str(value.device) in failing_devices:
                _assert_finite_tensor(value, name)
        raise NonFiniteTrainingError(
            "training state contains non-finite values on "
            + ", ".join(sorted(failing_devices))
        )

    return {
        "status": "finite",
        "tensors": sum(item["tensors"] for item in source_counts.values()),
        "elements": sum(item["elements"] for item in source_counts.values()),
        "devices": sorted(device_checks),
        "scalar_device_checks": len(device_checks),
        "sources": source_counts,
    }


def assert_finite_training_state(model, optimizer=None):
    audit_finite_training_state(model, optimizer)
    return True


def build_experiment_signature(configuration):
    canonical = json.dumps(
        configuration,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def capture_feature_pools(model):
    result = {}
    seen_pools = set()
    for module_name, module in model.named_modules():
        pool = getattr(module, "pool", None)
        features = getattr(pool, "features", None)
        if pool is None or features is None or id(pool) in seen_pools:
            continue
        seen_pools.add(id(pool))
        result[module_name] = {
            "features": features.detach().cpu().clone(),
            "nums_features": int(getattr(pool, "nums_features", 0)),
        }
    return result


def restore_feature_pools(model, states):
    modules = dict(model.named_modules())
    for module_name, state in (states or {}).items():
        if module_name not in modules:
            raise ValueError(f"checkpoint feature-pool module is missing: {module_name!r}")
        pool = getattr(modules[module_name], "pool", None)
        if pool is None:
            raise ValueError(f"checkpoint target has no feature pool: {module_name!r}")
        current = getattr(pool, "features", None)
        features = state["features"]
        if torch.is_tensor(current):
            features = features.to(device=current.device, dtype=current.dtype)
        pool.features = features
        pool.nums_features = int(state["nums_features"])


def capture_training_checkpoint(
        *, model, optimizer, scaler, scheduler, epoch, history,
        stop_state, plateau_state, experiment_signature, extra=None,
        prevalidated_state_audit=None):
    state_audit = (
        audit_finite_training_state(model, optimizer)
        if prevalidated_state_audit is None
        else dict(prevalidated_state_audit)
    )
    if state_audit.get("status") != "finite":
        raise NonFiniteTrainingError(
            "prevalidated checkpoint state audit is not finite"
        )
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_signature": str(experiment_signature),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "stop_state": asdict(stop_state),
        "plateau_state": asdict(plateau_state),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "feature_pool_state": capture_feature_pools(model),
        "finite_state_audit": state_audit,
        "extra": dict(extra or {}),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        replace_error = None
        for _attempt in range(50):
            try:
                os.replace(temporary, path)
                replace_error = None
                break
            except PermissionError as exc:
                # Windows: 其他进程(status/validate/snapshot)可能正持有目标
                # checkpoint 的读句柄;退避重试而不是让训练进程直接中止。
                replace_error = exc
                time.sleep(0.2)
        if replace_error is not None:
            raise replace_error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_training_checkpoint(path, *, expected_signature, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("training checkpoint must contain a mapping")
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported or incomplete training checkpoint schema")
    if payload.get("experiment_signature") != str(expected_signature):
        raise ValueError("training checkpoint experiment signature mismatch")
    return payload


def restore_training_checkpoint(
        payload, *, model, optimizer, scaler, scheduler, expected_signature):
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported or incomplete training checkpoint schema")
    if payload.get("experiment_signature") != str(expected_signature):
        raise ValueError("training checkpoint experiment signature mismatch")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "stop_state",
        "plateau_state",
        "history",
        "rng_state",
        "feature_pool_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("training checkpoint is incomplete: " + ", ".join(missing))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler_state = payload.get("scaler_state_dict")
    if scaler is not None:
        if scaler_state is None:
            raise ValueError("training checkpoint has no gradient-scaler state")
        scaler.load_state_dict(scaler_state)
    elif scaler_state is not None:
        raise ValueError("training checkpoint has gradient-scaler state but runtime does not")
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("training checkpoint has no LR-scheduler state")
        scheduler.load_state_dict(scheduler_state)
    elif scheduler_state is not None:
        raise ValueError("training checkpoint has LR-scheduler state but runtime does not")
    restore_feature_pools(model, payload["feature_pool_state"])
    state_audit = audit_finite_training_state(model, optimizer)
    restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "next_epoch": int(payload["epoch"]) + 1,
        "history": list(payload["history"]),
        "stop_state": VQVAEStopState(**payload["stop_state"]),
        "plateau_state": VQVAEStopState(**payload["plateau_state"]),
        "finite_state_audit": state_audit,
        "extra": dict(payload.get("extra") or {}),
    }


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
    if config.max_nonfinite_val_epochs > 0:
        if state.consecutive_nonfinite_val_epochs >= config.max_nonfinite_val_epochs:
            state.stop_reason = f"nonfinite_val_epochs={state.consecutive_nonfinite_val_epochs}"
            should_stop = True

    if not should_stop and reached_min_epochs and config.patience > 0:
        if state.epochs_without_improvement >= config.patience:
            state.stop_reason = f"patience={config.patience}"
            should_stop = True

    return state, improved, should_stop
