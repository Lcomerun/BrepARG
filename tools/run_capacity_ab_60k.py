"""Fail-closed launcher for the 60k learned-VQ capacity A/B matrix.

The formal matrix has two arms (VQ-8192 and two-stage RVQ-2x4096), seeds 3
and 4, 60,000/12,000 patches, bf16, batch 128, 100 epochs and lr 3e-4.  A
task is resumed only when every signed input and its rolling checkpoint still
match.  This module intentionally does not know how to delete or overwrite a
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.run_p0b_stability_retest import output_root_writer_lock
except ImportError:  # direct execution from tools/
    from run_p0b_stability_retest import output_root_writer_lock


SCHEMA = "capacity-ab-60k-v1"
# Reuse the P0-B OS writer lock implementation and its lock-file convention.
# Capacity runs must use a distinct output root, so this serializes writers
# without making the existing lock appear as an unexpected output artifact.
WRITER_LOCK_NAME = ".p0b_writer.lock"
FORMAL_ARMS = ("vq_8192_64d_random", "rvq_2x4096_64d_random")
FORMAL_SEEDS = (3, 4)
FORMAL_TRAIN_CAP = 60_000
FORMAL_VAL_CAP = 12_000
FORMAL_BATCH_SIZE = 128
FORMAL_EPOCHS = 100
FORMAL_LEARNING_RATE = "3e-4"
FORMAL_PRECISION = "bf16"
FORMAL_GRAD_CLIP = "1.0"
FORMAL_MIN_PARENT_COVERAGE = 0.9
FORMAL_PROTOCOL_SHA256 = (
    "6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3"
)
FORMAL_SPLIT_PICKLE_SHA256 = (
    "6ff0a0c3ee6a04ee056fa1ab982eb436a9f59d3d21f21f17babf34e6dc701d29"
)
MAX_SMOKE_TRAIN_CAP = 2048
MAX_SMOKE_VAL_CAP = 512
MAX_SMOKE_BATCH_SIZE = 128
MAX_SMOKE_EPOCHS = 2
ZERO_FIELDS = (
    "skipped_train_batches",
    "nonfinite_loss_batches",
    "nonfinite_gradient_batches",
    "nonfinite_state_batches",
    "nonfinite_val_batches",
    "nonfinite_val_samples",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    protocol_dir: Path
    breparg_root: Path
    output_root: Path
    python: Path
    arms: tuple[str, ...] = FORMAL_ARMS
    seeds: tuple[int, ...] = FORMAL_SEEDS
    train_cap: int = FORMAL_TRAIN_CAP
    val_cap: int = FORMAL_VAL_CAP
    batch_size: int = FORMAL_BATCH_SIZE
    epochs: int = FORMAL_EPOCHS
    learning_rate: str = FORMAL_LEARNING_RATE
    precision: str = FORMAL_PRECISION
    smoke: bool = False

    def __post_init__(self) -> None:
        for name in ("repo_root", "protocol_dir", "breparg_root", "output_root", "python"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "learning_rate", str(self.learning_rate))
        object.__setattr__(self, "precision", str(self.precision).lower())
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be non-empty and unique")
        if set(self.arms) - set(FORMAL_ARMS):
            raise ValueError(f"capacity A/B only supports arms {FORMAL_ARMS}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(int(value) <= 0 for value in (self.train_cap, self.val_cap, self.batch_size, self.epochs)):
            raise ValueError("caps, batch size, and epochs must be positive")
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0:
            raise ValueError("learning rate must be positive and finite")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if not self.smoke:
            actual = (self.arms, self.seeds, self.train_cap, self.val_cap, self.batch_size,
                      self.epochs, self.learning_rate, self.precision)
            expected = (FORMAL_ARMS, FORMAL_SEEDS, FORMAL_TRAIN_CAP, FORMAL_VAL_CAP,
                        FORMAL_BATCH_SIZE, FORMAL_EPOCHS, FORMAL_LEARNING_RATE,
                        FORMAL_PRECISION)
            if actual != expected:
                raise ValueError("formal capacity A/B protocol is immutable; use --smoke for bounded overrides")
        elif (self.train_cap > MAX_SMOKE_TRAIN_CAP or self.val_cap > MAX_SMOKE_VAL_CAP
              or self.batch_size > MAX_SMOKE_BATCH_SIZE or self.epochs > MAX_SMOKE_EPOCHS):
            raise ValueError("smoke overrides exceed bounded limits")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("repo_root", "protocol_dir", "breparg_root", "output_root", "python"):
            value[name] = str(value[name])
        value["arms"] = list(self.arms)
        value["seeds"] = list(self.seeds)
        return value


def protocol_binding(config: RunConfig) -> dict[str, Any]:
    summary_path = config.protocol_dir / "protocol_summary.json"
    split_path = config.protocol_dir / "split.pkl"
    summary = read_json(summary_path)
    return {
        "status": summary.get("status"),
        "protocol_sha256": summary.get("protocol_sha256"),
        "split_pickle_sha256": sha256_file(split_path),
        "summary_split_pickle_sha256": summary.get("split_pickle_sha256"),
        "protocol_summary_sha256": sha256_file(summary_path),
        "parent_overlap_counts": summary.get("parent_overlap_counts"),
    }


def verify_protocol(config: RunConfig) -> dict[str, Any]:
    binding = protocol_binding(config)
    reasons = []
    if binding["status"] != "VERIFIED":
        reasons.append("status must be VERIFIED")
    overlaps = binding.get("parent_overlap_counts")
    if not isinstance(overlaps, Mapping) or set(overlaps) != {"train__val", "train__test", "val__test"}:
        reasons.append("parent overlap counts are incomplete")
    elif any(type(overlaps[name]) is not int or overlaps[name] != 0 for name in overlaps):
        reasons.append("all parent overlap counts must be zero")
    if binding["summary_split_pickle_sha256"] != binding["split_pickle_sha256"]:
        reasons.append("split.pkl SHA-256 does not match summary")
    if config.smoke:
        if not isinstance(binding.get("protocol_sha256"), str) or not binding["protocol_sha256"]:
            reasons.append("protocol hash is missing")
    else:
        if binding.get("protocol_sha256") != FORMAL_PROTOCOL_SHA256:
            reasons.append("protocol hash is not frozen Protocol V5 hash")
        if binding.get("split_pickle_sha256") != FORMAL_SPLIT_PICKLE_SHA256:
            reasons.append("split hash is not frozen Protocol V5 hash")
    if reasons:
        raise ValueError("protocol verification failed: " + "; ".join(reasons))
    return binding


def required_inputs(config: RunConfig) -> tuple[Path, ...]:
    return (
        config.python,
        config.repo_root / "breparg_improvements" / "train.py",
        config.repo_root / "breparg_improvements" / "training_stability.py",
        config.repo_root / "breparg_improvements" / "vqvae_sampling.py",
        config.repo_root / "breparg_improvements" / "vqvae_metrics.py",
        config.protocol_dir / "protocol_summary.json",
        config.protocol_dir / "split.pkl",
        config.breparg_root / "quantise.py",
    )


def verify_inputs(config: RunConfig) -> None:
    missing = [str(path) for path in required_inputs(config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("capacity A/B inputs missing: " + ", ".join(missing))
    discovered = None
    cursor = config.repo_root / "breparg_improvements"
    for _ in range(6):
        candidate = cursor / "BrepARG"
        if (candidate / "model.py").is_file():
            discovered = candidate.resolve()
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if discovered != config.breparg_root:
        raise ValueError(
            f"train.py will discover BrepARG at {discovered}, not {config.breparg_root}"
        )
    verify_protocol(config)


def arm_metadata(arm: str) -> dict[str, Any]:
    if arm == "vq_8192_64d_random":
        return {
            "kind": "learned_vq", "implementation": "BrepARG.quantise.VectorQuantiser",
            "codebook_size": 8192, "embedding_dim": 64,
            "distance": "cos", "anchor": "random", "first_batch": False,
            "contrastive_loss": True, "decay": 0.99, "downstream_compatible": False,
        }
    if arm == "rvq_2x4096_64d_random":
        return {
            "kind": "residual_vq", "implementation": "BrepARG.quantise.VectorQuantiser",
            "stage_count": 2,
            "stage_codebook_sizes": [4096, 4096], "codebook_size": 4096,
            "embedding_dim": 64, "distance": "cos", "anchor": "random",
            "first_batch": False, "contrastive_loss": True, "decay": 0.99,
            "loss_aggregation": "sum", "residual_stage1_detached": True,
            "effective_code_combinations": 4096 * 4096,
            "stage2_collapse_gate": {"min_unique_bins": 2, "min_perplexity_exclusive": 1.0},
            "downstream_compatible": False,
        }
    raise ValueError(f"unsupported capacity arm: {arm}")


def arm_codebook_size(arm: str) -> int:
    if arm == "vq_8192_64d_random":
        return 8192
    if arm == "rvq_2x4096_64d_random":
        return 4096
    raise ValueError(f"unsupported capacity arm: {arm}")


def task_root(config: RunConfig, arm: str, seed: int) -> Path:
    return config.output_root / "tasks" / arm / f"seed{seed}"


def task_signature_payload(config: RunConfig, arm: str, seed: int) -> dict[str, Any]:
    protocol = protocol_binding(config)
    source_names = ("train.py", "training_stability.py", "vqvae_sampling.py", "vqvae_metrics.py")
    sources = {
        name: sha256_file(config.repo_root / "breparg_improvements" / name)
        for name in source_names
    }
    return {
        "schema": SCHEMA, "arm": arm, "seed": int(seed),
        "train_cap": int(config.train_cap), "val_cap": int(config.val_cap),
        "batch_size": int(config.batch_size), "epochs": int(config.epochs),
        "learning_rate": config.learning_rate, "precision": config.precision,
        "gradient_clip": FORMAL_GRAD_CLIP, "strict_nonfinite": True,
        "scheduler": {"kind": "ReduceLROnPlateau",
                       "metric": "curved_parent_mse",
                       "factor": 0.5, "patience": 8,
                       "threshold": 1e-5, "threshold_mode": "abs", "min_lr": 1e-6},
        "sampling": {"balance_by_parent": not config.smoke,
                     "deduplicate_before_cap": not config.smoke,
                     "require_exact_caps": not config.smoke,
                     "min_parent_coverage": 0.0 if config.smoke else FORMAL_MIN_PARENT_COVERAGE},
        "quantizer": arm_metadata(arm),
        "protocol_status": protocol["status"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "split_pickle_sha256": protocol["split_pickle_sha256"],
        "parent_overlap_counts": protocol["parent_overlap_counts"],
        "source_sha256": sources,
        "breparg_quantise_path": str(config.breparg_root / "quantise.py"),
        "breparg_quantise_sha256": sha256_file(config.breparg_root / "quantise.py"),
    }


def build_task(config: RunConfig, arm: str, seed: int) -> dict[str, Any]:
    payload = task_signature_payload(config, arm, seed)
    signature = canonical_signature(payload)
    root = task_root(config, arm, seed)
    command = [str(config.python), str(config.repo_root / "breparg_improvements" / "train.py"), "--stage", "vqsweep"]
    env = {
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(config.repo_root).replace("\\", "/"),
        "NS_OUTBASE": str(config.output_root), "NS_OUT": str(root),
        "NS_PROTOCOL_DIR": str(config.protocol_dir), "NS_PROTOCOL_V2": "1",
        "NS_VQ_AUTO_RESUME": "1", "NS_VQ_BALANCE_BY_PARENT": "1" if not config.smoke else "0",
        "NS_VQ_BS": str(config.batch_size), "NS_VQ_EPOCHS": str(config.epochs),
        "NS_VQ_EXPERIMENT_SEED": str(seed), "NS_VQ_EXPERIMENT_SIGNATURE": signature,
        "NS_VQ_GRAD_CLIP": FORMAL_GRAD_CLIP, "NS_VQ_LR": config.learning_rate,
        "NS_VQ_PRECISION": config.precision, "NS_VQ_SAMPLES": str(config.train_cap),
        "NS_VQ_VAL_SAMPLES": str(config.val_cap), "NS_VQ_SWEEP_ARMS": arm,
        "NS_VQ_SWEEP_EPOCHS": str(config.epochs), "NS_VQ_SWEEP_TRAIN_CAP": str(config.train_cap),
        "NS_VQ_ROLLING_CHECKPOINT": str(root / f"{arm}_rolling.pt"),
        "NS_VQ_SAVE_FINAL": "1", "NS_VQ_STRICT_NONFINITE": "1",
        "NS_VQ_MIN_EPOCHS": str(config.epochs), "NS_VQ_PATIENCE": str(config.epochs),
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "1", "NS_VQ_MIN_PARENT_COVERAGE": str(payload["sampling"]["min_parent_coverage"]),
        "NS_VQ_REQUIRE_EXACT_CAPS": "1" if not config.smoke else "0",
        "NS_VQ_DEDUP_BEFORE_CAP": "1" if not config.smoke else "0",
        "NS_VQ_SCHEDULER_FACTOR": "0.5", "NS_VQ_SCHEDULER_PATIENCE": "8",
        "NS_VQ_SCHEDULER_THRESHOLD": "1e-5", "NS_VQ_SCHEDULER_MIN_LR": "1e-6",
        "NS_VQ_TB_LOG_DIR": str(root / "tensorboard"),
        "NS_VQ_COMPLEX_FRACTION": "0", "NS_VQ_CURVED_FRACTION": "0",
    }
    return {
        "arm": arm, "seed": int(seed), "task_id": f"{arm}:seed{seed}",
        "task_root": str(root), "signature": signature, "signature_payload": payload,
        "command": command, "environment": env, "history": str(root / f"{arm}_history.json"),
        "sweep": str(root / "vqvae_hp_sweep.json"), "best_checkpoint": str(root / f"{arm}_best.pt"),
        "final_checkpoint": str(root / f"{arm}_final.pt"), "rolling_checkpoint": str(root / f"{arm}_rolling.pt"),
        "manifest": str(root / "task_manifest.json"), "status": "PENDING", "attempts": [],
    }


def build_state(config: RunConfig) -> dict[str, Any]:
    tasks = [build_task(config, arm, seed) for arm in config.arms for seed in config.seeds]
    public = config.public_dict()
    return {"schema": SCHEMA, "status": "PENDING", "mode": "SMOKE" if config.smoke else "FORMAL",
            "formal_result_eligible": not config.smoke, "created_at": now(), "updated_at": now(),
            "configuration": public, "configuration_signature": canonical_signature(public),
            "active_task": None, "tasks": tasks}


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_inventory(value: Any, task: Mapping[str, Any], reasons: list[str], prefix: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != {"train", "val"}:
        reasons.append(f"{prefix} inventory must contain train and val")
        return None
    normalized = {}
    for split in ("train", "val"):
        item = value.get(split)
        if not isinstance(item, Mapping) or item.get("schema") != "vq-exact-hash-inventory-v1":
            reasons.append(f"{prefix} {split} inventory schema mismatch")
            continue
        count = item.get("count")
        cap = task["signature_payload"]["train_cap" if split == "train" else "val_cap"]
        if type(count) is not int or count <= 0 or count > cap or (not task["signature_payload"]["sampling"]["require_exact_caps"] and count <= 0):
            reasons.append(f"{prefix} {split} inventory count invalid")
        if task["signature_payload"]["sampling"]["require_exact_caps"] and count != cap:
            reasons.append(f"{prefix} {split} inventory count mismatch")
        for field in ("ordered_sha256", "sorted_sha256"):
            if not _valid_sha256(item.get(field)):
                reasons.append(f"{prefix} {split} inventory {field} invalid")
        normalized[split] = dict(item)
    return normalized if len(normalized) == 2 else None


def _validate_quantizer(value: Any, task: Mapping[str, Any], reasons: list[str], prefix: str) -> None:
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} quantizer metadata missing")
        return
    try:
        expected = arm_metadata(str(task["arm"]))
    except ValueError:
        reasons.append(f"{prefix} unsupported capacity arm")
        return
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            reasons.append(f"{prefix} quantizer {key} mismatch")


def _validate_formal_task_contract(task: Mapping[str, Any], reasons: list[str]) -> None:
    payload = task.get("signature_payload")
    if not isinstance(payload, Mapping):
        reasons.append("task signature payload missing")
        return
    expected = {
        "schema": SCHEMA,
        "seed": task.get("seed"),
        "train_cap": FORMAL_TRAIN_CAP,
        "val_cap": FORMAL_VAL_CAP,
        "batch_size": FORMAL_BATCH_SIZE,
        "epochs": FORMAL_EPOCHS,
        "learning_rate": FORMAL_LEARNING_RATE,
        "precision": FORMAL_PRECISION,
        "gradient_clip": FORMAL_GRAD_CLIP,
        "strict_nonfinite": True,
        "protocol_sha256": FORMAL_PROTOCOL_SHA256,
        "split_pickle_sha256": FORMAL_SPLIT_PICKLE_SHA256,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            reasons.append(f"formal signature {field} mismatch")
    if task.get("arm") not in FORMAL_ARMS or payload.get("arm") != task.get("arm"):
        reasons.append("formal signature arm mismatch")
    if task.get("seed") not in FORMAL_SEEDS or payload.get("seed") != task.get("seed"):
        reasons.append("formal signature seed mismatch")
    if task.get("arm") in FORMAL_ARMS:
        if payload.get("quantizer") != arm_metadata(str(task.get("arm"))):
            reasons.append("formal signature quantizer mismatch")
    if payload.get("scheduler") != {
        "kind": "ReduceLROnPlateau",
        "metric": "curved_parent_mse",
        "factor": 0.5,
        "patience": 8,
        "threshold": 1e-5,
        "threshold_mode": "abs",
        "min_lr": 1e-6,
    }:
        reasons.append("formal signature scheduler mismatch")
    if payload.get("sampling") != {
        "balance_by_parent": True,
        "deduplicate_before_cap": True,
        "require_exact_caps": True,
        "min_parent_coverage": FORMAL_MIN_PARENT_COVERAGE,
    }:
        reasons.append("formal signature sampling mismatch")


def _validate_signed_inputs(task: Mapping[str, Any], reasons: list[str]) -> None:
    """Re-hash every source/protocol input named by the signed task."""
    payload = task.get("signature_payload") or {}
    command = task.get("command") or []
    if len(command) < 2:
        reasons.append("task command is incomplete")
        return
    train_path = Path(str(command[1])).resolve()
    improvements = train_path.parent
    observed_sources = {}
    for name in ("train.py", "training_stability.py", "vqvae_sampling.py", "vqvae_metrics.py"):
        path = improvements / name
        if not path.is_file():
            reasons.append(f"signed source is missing: {name}")
            continue
        observed_sources[name] = sha256_file(path)
    if observed_sources != payload.get("source_sha256"):
        reasons.append("signed source SHA-256 set mismatch")
    quantise_path = Path(str(payload.get("breparg_quantise_path", "")))
    if not quantise_path.is_file():
        reasons.append("signed BrepARG quantise.py is missing")
    elif sha256_file(quantise_path) != payload.get("breparg_quantise_sha256"):
        reasons.append("signed BrepARG quantise.py SHA-256 mismatch")
    protocol_dir = Path(str((task.get("environment") or {}).get("NS_PROTOCOL_DIR", "")))
    summary_path = protocol_dir / "protocol_summary.json"
    split_path = protocol_dir / "split.pkl"
    if not summary_path.is_file() or not split_path.is_file():
        reasons.append("signed protocol files are missing")
        return
    if sha256_file(summary_path) != payload.get("protocol_summary_sha256"):
        reasons.append("signed protocol summary SHA-256 mismatch")
    if sha256_file(split_path) != payload.get("split_pickle_sha256"):
        reasons.append("signed split.pkl SHA-256 mismatch")


def _validate_protocol_context(
    value: Any,
    task: Mapping[str, Any],
    reasons: list[str],
    prefix: str,
    *,
    require_parent_overlaps: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} protocol context missing")
        return
    expected = task["signature_payload"]
    for field in ("protocol_sha256", "split_pickle_sha256"):
        if value.get(field) != expected.get(field):
            reasons.append(f"{prefix} {field} mismatch")
    observed_overlaps = value.get("parent_overlap_counts")
    if require_parent_overlaps and observed_overlaps != expected.get("parent_overlap_counts"):
        reasons.append(f"{prefix} parent_overlap_counts mismatch")
    elif observed_overlaps is not None and observed_overlaps != expected.get("parent_overlap_counts"):
        reasons.append(f"{prefix} parent_overlap_counts mismatch")


def _validate_run_manifest(
    value: Any,
    task: Mapping[str, Any],
    reasons: list[str],
    prefix: str,
    inventory: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} run_manifest missing")
        return None
    experiment = value.get("experiment")
    if not isinstance(experiment, Mapping):
        reasons.append(f"{prefix} run_manifest experiment missing")
        return None
    expected = task["signature_payload"]
    for field in ("seed", "train_cap", "val_cap", "epochs", "batch_size"):
        if experiment.get(field) != expected[field]:
            reasons.append(f"{prefix} run_manifest {field} mismatch")
    expected_sampling = expected["sampling"]
    for field in (
        "min_parent_coverage",
        "balance_by_parent",
        "deduplicate_before_cap",
        "require_exact_caps",
    ):
        if experiment.get(field) != expected_sampling[field]:
            reasons.append(f"{prefix} run_manifest sampling {field} mismatch")
    arms = experiment.get("arms")
    if not isinstance(arms, list) or len(arms) != 1 or not isinstance(arms[0], Mapping):
        reasons.append(f"{prefix} run_manifest arm missing")
    else:
        arm = arms[0]
        if arm.get("name") != task["arm"] or arm.get("codebook") != arm_codebook_size(task["arm"]):
            reasons.append(f"{prefix} run_manifest arm mismatch")
        _validate_quantizer(arm.get("quantizer"), task, reasons, prefix)
    _validate_protocol_context(
        experiment.get("protocol"), task, reasons, prefix,
        require_parent_overlaps=True,
    )
    if inventory is not None and experiment.get("inventory") != inventory:
        reasons.append(f"{prefix} run_manifest inventory mismatch")
    runtime = value.get("runtime_resume_compatibility")
    if not isinstance(runtime, Mapping) or not runtime:
        reasons.append(f"{prefix} runtime_resume_compatibility missing")
        return None
    return runtime


def _validate_signature_configuration(
    value: Any,
    task: Mapping[str, Any],
    reasons: list[str],
    prefix: str,
    inventory: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
) -> None:
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} signature_configuration missing")
        return
    expected = task["signature_payload"]
    for field in ("seed", "train_cap", "val_cap", "epochs", "batch_size", "precision"):
        if value.get(field) != expected[field]:
            reasons.append(f"{prefix} signature_configuration {field} mismatch")
    if value.get("lr") != float(expected["learning_rate"]):
        reasons.append(f"{prefix} signature_configuration lr mismatch")
    if value.get("grad_clip_norm") != float(FORMAL_GRAD_CLIP):
        reasons.append(f"{prefix} signature_configuration grad clip mismatch")
    arm = value.get("arm")
    if not isinstance(arm, Mapping) or arm.get("name") != task["arm"]:
        reasons.append(f"{prefix} signature_configuration arm mismatch")
    else:
        _validate_quantizer(arm.get("quantizer"), task, reasons, prefix)
    _validate_protocol_context(
        value.get("protocol"), task, reasons, prefix,
        require_parent_overlaps=True,
    )
    if inventory is not None and value.get("inventory") != inventory:
        reasons.append(f"{prefix} signature_configuration inventory mismatch")
    if runtime is not None and value.get("runtime_resume_compatibility") != runtime:
        reasons.append(f"{prefix} signature_configuration runtime mismatch")
    if value.get("scheduler") != expected["scheduler"]:
        reasons.append(f"{prefix} signature_configuration scheduler mismatch")
    observed_sampling = value.get("sampling") or {}
    for field, expected_value in expected["sampling"].items():
        if observed_sampling.get(field) != expected_value:
            reasons.append(f"{prefix} signature_configuration sampling {field} mismatch")


def _validate_stage_usage(usage: Any, task: Mapping[str, Any], reasons: list[str], prefix: str) -> None:
    if task["arm"] != "rvq_2x4096_64d_random":
        return
    if not isinstance(usage, Mapping):
        reasons.append(f"{prefix} stage usage missing")
        return
    for stage, size in (("stage1", 4096), ("stage2", 4096)):
        item = usage.get(stage)
        if not isinstance(item, Mapping):
            reasons.append(f"{prefix} {stage} usage missing")
            continue
        if type(item.get("tokens")) is not int or item["tokens"] <= 0:
            reasons.append(f"{prefix} {stage} tokens invalid")
        if type(item.get("unique_bins")) is not int or not 2 <= item["unique_bins"] <= size:
            reasons.append(f"{prefix} {stage} unique_bins invalid")
        try:
            finite = math.isfinite(float(item.get("entropy_perplexity")))
        except (TypeError, ValueError):
            finite = False
        if not finite or float(item["entropy_perplexity"]) <= 1.0:
            reasons.append(f"{prefix} {stage} perplexity is collapsed or non-finite")


def _load_torch(path: Path, reasons: list[str], label: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        reasons.append(f"{label} missing")
        return None
    try:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        reasons.append(f"{label} unreadable: {type(exc).__name__}")
        return None
    if not isinstance(value, Mapping):
        reasons.append(f"{label} must be a mapping")
        return None
    return value


def _validate_capacity_model_state(
    state: Any, task: Mapping[str, Any], reasons: list[str], prefix: str
) -> None:
    if not isinstance(state, Mapping) or not state:
        reasons.append(f"{prefix} model_state_dict missing")
        return
    # State-dict names are deliberately checked by semantic suffix so a future
    # harmless wrapper-name change does not silently turn RVQ into one VQ.
    tensors = {
        str(name): value for name, value in state.items()
        if getattr(value, "ndim", None) == 2
    }
    if task["arm"] == "vq_8192_64d_random":
        candidates = [
            value for name, value in tensors.items()
            if name.endswith("quantize.quantizer.embedding.weight")
        ]
        if len(candidates) != 1 or tuple(candidates[0].shape) != (8192, 64):
            reasons.append(f"{prefix} VQ-8192 embedding state is missing or has wrong shape")
        return
    required = {
        "stage1": "quantize.stage1.quantizer.embedding.weight",
        "stage2": "quantize.stage2.quantizer.embedding.weight",
    }
    for stage, suffix in required.items():
        candidates = [value for name, value in tensors.items() if name.endswith(suffix)]
        if len(candidates) != 1 or tuple(candidates[0].shape) != (4096, 64):
            reasons.append(f"{prefix} RVQ {stage} embedding state is missing or has wrong shape")


def _validate_rvq_feature_pools(value: Any, reasons: list[str], prefix: str) -> None:
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix} feature_pool_state missing")
        return
    for stage in ("stage1", "stage2"):
        matches = [
            state for name, state in value.items()
            if stage in str(name) and isinstance(state, Mapping)
        ]
        if len(matches) != 1:
            reasons.append(f"{prefix} must contain exactly one {stage} FeaturePool")
            continue
        features = matches[0].get("features")
        if getattr(features, "dtype", None) is None or str(features.dtype) != "torch.float32":
            reasons.append(f"{prefix} {stage} FeaturePool features must be fp32")
        nums_features = matches[0].get("nums_features")
        if type(nums_features) is not int or nums_features < 0:
            reasons.append(f"{prefix} {stage} FeaturePool nums_features invalid")


def _validate_single_feature_pool(value: Any, reasons: list[str], prefix: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 1:
        reasons.append(f"{prefix} must contain exactly one learned-VQ FeaturePool")
        return
    state = next(iter(value.values()))
    if not isinstance(state, Mapping):
        reasons.append(f"{prefix} learned-VQ FeaturePool state missing")
        return
    features = state.get("features")
    if getattr(features, "dtype", None) is None or str(features.dtype) != "torch.float32":
        reasons.append(f"{prefix} learned-VQ FeaturePool features must be fp32")
    if type(state.get("nums_features")) is not int or state["nums_features"] < 0:
        reasons.append(f"{prefix} learned-VQ FeaturePool nums_features invalid")


def _validate_checkpoint(path: Path, task: Mapping[str, Any], reasons: list[str], label: str, expected_epoch: int | None, terminal: int, inventory: Mapping[str, Any] | None) -> None:
    payload = _load_torch(path, reasons, label)
    if payload is None:
        return
    _validate_capacity_model_state(payload.get("model_state_dict"), task, reasons, label)
    epoch = payload.get("checkpoint_epoch")
    if type(epoch) is not int or not 0 <= epoch <= terminal:
        reasons.append(f"{label} checkpoint_epoch invalid")
    elif expected_epoch is not None and epoch != expected_epoch:
        reasons.append(f"{label} checkpoint_epoch mismatch")
    if payload.get("fsq_levels") not in ([], None):
        reasons.append(f"{label} fsq_levels must be empty")
    _validate_quantizer(payload.get("quantizer"), task, reasons, label)
    context = payload.get("checkpoint_context")
    if not isinstance(context, Mapping):
        reasons.append(f"{label} checkpoint_context missing")
    else:
        _validate_protocol_context(context, task, reasons, label)
        if inventory is not None and context.get("inventory") != inventory:
            reasons.append(f"{label} inventory binding mismatch")
        _validate_run_manifest(
            context.get("run_manifest"), task, reasons, label, inventory
        )
    _validate_stage_usage(payload.get("val_stage_code_usage"), task, reasons, label)


def validate_task(task: Mapping[str, Any], *, formal: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if formal:
        _validate_formal_task_contract(task, reasons)
    _validate_signed_inputs(task, reasons)
    expected_epochs = int(task["signature_payload"]["epochs"])
    if canonical_signature(task["signature_payload"]) != task.get("signature"):
        reasons.append("task signature payload mismatch")
    history_payload = {}
    records = []
    history_path = Path(str(task["history"]))
    if not history_path.is_file():
        reasons.append("history missing")
    else:
        try:
            history_payload = read_json(history_path)
            records = history_payload.get("history") or []
            if not isinstance(records, list):
                raise ValueError("history is not a list")
        except Exception as exc:
            reasons.append(f"history unreadable: {type(exc).__name__}")
    epochs = [row.get("epoch") for row in records if isinstance(row, Mapping)]
    if epochs != list(range(expected_epochs)):
        reasons.append(f"epoch coverage must be 0..{expected_epochs - 1}")
    for row in records:
        if not isinstance(row, Mapping):
            reasons.append("history row is not an object")
            continue
        epoch = row.get("epoch")
        for field in ZERO_FIELDS:
            if row.get(field) != 0:
                reasons.append(f"epoch {epoch}: {field} must be zero")
        for field in ("gradients_finite", "training_state_finite", "grad_clip_active"):
            if row.get(field) is not True:
                reasons.append(f"epoch {epoch}: {field} must be true")
        if row.get("experiment_signature") != task["signature"]:
            reasons.append(f"epoch {epoch}: experiment signature mismatch")
        if task["arm"] == "rvq_2x4096_64d_random":
            _validate_stage_usage(row.get("train_stage_code_usage"), task, reasons, f"epoch {epoch} train")
            _validate_stage_usage(row.get("val_stage_code_usage"), task, reasons, f"epoch {epoch} val")
            health = row.get("stage_usage_health")
            if not isinstance(health, Mapping) or health.get("healthy") is not True:
                reasons.append(f"epoch {epoch}: stage2 health gate failed")
    config_record = history_payload.get("config") or {}
    if config_record.get("experiment_signature") != task["signature"]:
        reasons.append("history config signature mismatch")
    precision = config_record.get("precision")
    if isinstance(precision, Mapping):
        precision = precision.get("name")
    if precision != task["signature_payload"]["precision"]:
        reasons.append("history precision mismatch")
    if config_record.get("strict_nonfinite") is not True:
        reasons.append("history strict_nonfinite must be true")
    if config_record.get("grad_clip_norm") != 1.0:
        reasons.append("history grad_clip_norm mismatch")
    expected_scheduler = task["signature_payload"]["scheduler"]
    observed_scheduler = config_record.get("scheduler") or {}
    for field, value in expected_scheduler.items():
        if observed_scheduler.get(field) != value:
            reasons.append(f"history scheduler {field} mismatch")
    _validate_quantizer(config_record.get("quantizer"), task, reasons, "history")

    inventory = None
    run_runtime = None
    sweep_row = {}
    sweep_path = Path(str(task["sweep"]))
    if not sweep_path.is_file():
        reasons.append("sweep summary missing")
    else:
        try:
            sweep = read_json(sweep_path)
            rows = sweep.get("mse_ranking") or []
            by_arm = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
            if set(by_arm) != {task["arm"]}:
                reasons.append("sweep arm set mismatch")
            sweep_row = by_arm.get(task["arm"], {})
            if sweep_row.get("epochs_ran") != expected_epochs:
                reasons.append("sweep epochs_ran mismatch")
            if sweep_row.get("final_checkpoint_epoch") != expected_epochs - 1:
                reasons.append("sweep final checkpoint epoch mismatch")
            if sweep_row.get("experiment_signature") != task["signature"]:
                reasons.append("sweep signature mismatch")
            inventory = _validate_inventory(sweep_row.get("inventory"), task, reasons, "sweep")
            if formal and ((sweep_row.get("train_sampling") or {}).get("selected") != FORMAL_TRAIN_CAP
                           or (sweep_row.get("val_sampling") or {}).get("selected") != FORMAL_VAL_CAP):
                reasons.append("formal realized patch counts must be exactly 60000/12000")
            manifest = sweep.get("run_manifest") or {}
            run_runtime = _validate_run_manifest(
                manifest, task, reasons, "sweep", inventory
            )
        except Exception as exc:
            reasons.append(f"sweep unreadable: {type(exc).__name__}")
    best_epoch = sweep_row.get("checkpoint_epoch") if isinstance(sweep_row.get("checkpoint_epoch"), int) else None
    _validate_checkpoint(Path(str(task["best_checkpoint"])), task, reasons, "best_checkpoint", best_epoch, expected_epochs - 1, inventory)
    _validate_checkpoint(Path(str(task["final_checkpoint"])), task, reasons, "final_checkpoint", expected_epochs - 1, expected_epochs - 1, inventory)

    rolling = _load_torch(Path(str(task["rolling_checkpoint"])), reasons, "rolling_checkpoint")
    if rolling is not None:
        if rolling.get("checkpoint_schema") != "vq_training_state_v1":
            reasons.append("rolling checkpoint schema mismatch")
        if rolling.get("experiment_signature") != task["signature"]:
            reasons.append("rolling checkpoint signature mismatch")
        required_state = {
            "model_state_dict", "optimizer_state_dict", "scaler_state_dict",
            "scheduler_state_dict", "stop_state", "plateau_state", "history",
            "rng_state", "feature_pool_state", "finite_state_audit", "extra",
        }
        missing = sorted(required_state - set(rolling))
        if missing:
            reasons.append("rolling checkpoint incomplete: " + ", ".join(missing))
        _validate_capacity_model_state(rolling.get("model_state_dict"), task, reasons, "rolling checkpoint")
        if not isinstance(rolling.get("optimizer_state_dict"), Mapping) or not rolling.get("optimizer_state_dict"):
            reasons.append("rolling checkpoint optimizer state missing")
        if not isinstance(rolling.get("scheduler_state_dict"), Mapping) or not rolling.get("scheduler_state_dict"):
            reasons.append("rolling checkpoint scheduler state missing")
        if rolling.get("history") != records:
            reasons.append("rolling checkpoint embedded history mismatch")
        audit = rolling.get("finite_state_audit")
        if not isinstance(audit, Mapping) or audit.get("status") != "finite":
            reasons.append("rolling checkpoint finite-state audit missing or non-finite")
        rng = rolling.get("rng_state")
        if not isinstance(rng, Mapping) or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng):
            reasons.append("rolling checkpoint RNG state incomplete")
        if task["arm"] == "rvq_2x4096_64d_random":
            _validate_rvq_feature_pools(rolling.get("feature_pool_state"), reasons, "rolling checkpoint")
        else:
            _validate_single_feature_pool(
                rolling.get("feature_pool_state"), reasons, "rolling checkpoint"
            )
        extra = rolling.get("extra")
        if not isinstance(extra, Mapping):
            reasons.append("rolling checkpoint extra state missing")
        else:
            _validate_signature_configuration(
                extra.get("signature_configuration"), task, reasons,
                "rolling checkpoint", inventory, run_runtime,
            )
            precision_state = extra.get("precision")
            if not isinstance(precision_state, Mapping) or precision_state.get("name") != task["signature_payload"]["precision"]:
                reasons.append("rolling checkpoint precision context mismatch")
            selector = extra.get("selector_state")
            if not isinstance(selector, Mapping):
                reasons.append("rolling checkpoint selector state missing")
            elif task["arm"] == "rvq_2x4096_64d_random":
                health = selector.get("stage_usage_health")
                if not isinstance(health, Mapping) or health.get("healthy") is not True:
                    reasons.append("rolling checkpoint stage2 selector health missing or failed")
        if rolling.get("epoch") != expected_epochs - 1:
            reasons.append("rolling checkpoint terminal epoch mismatch")
    return {"task_id": task["task_id"], "valid": not reasons, "formal": formal,
            "epochs_observed": len(records), "last_epoch": epochs[-1] if epochs else None,
            "inventory": inventory, "runtime_resume_compatibility": run_runtime,
            "reasons": reasons}


def _inventory_consistent(results: Sequence[Mapping[str, Any]]) -> bool:
    inventories = [result.get("inventory") for result in results]
    return bool(inventories) and all(isinstance(value, Mapping) for value in inventories) and len({canonical_signature(value) for value in inventories}) == 1


def validation_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    formal = state.get("mode") == "FORMAL"
    tasks = list(state.get("tasks") or [])
    reasons: list[str] = []
    expected = {f"{arm}:seed{seed}" for arm in FORMAL_ARMS for seed in FORMAL_SEEDS}
    if formal and {task.get("task_id") for task in tasks} != expected:
        reasons.append("formal task matrix mismatch")
    validations = [validate_task(task, formal=formal) for task in tasks]
    for result in validations:
        reasons.extend(f"{result['task_id']}: {reason}" for reason in result["reasons"])
    consistent = _inventory_consistent(validations)
    if formal and not consistent:
        reasons.append("formal task inventories are missing or differ across tasks")
    runtimes = {
        canonical_signature(result["runtime_resume_compatibility"])
        for result in validations
        if isinstance(result.get("runtime_resume_compatibility"), Mapping)
    }
    if formal and (len(runtimes) != 1 or any(
            not isinstance(result.get("runtime_resume_compatibility"), Mapping)
            for result in validations
    )):
        reasons.append("formal runtime resume compatibility differs or is missing")
    runtime_consistent = len(runtimes) == 1 and all(
        isinstance(result.get("runtime_resume_compatibility"), Mapping)
        for result in validations
    )
    return {"schema": SCHEMA, "formal": formal, "formal_result_eligible": formal and not reasons,
            "valid": not reasons, "inventory_consistent": consistent,
            "runtime_resume_compatible": runtime_consistent, "validated_at": now(),
            "tasks": validations, "reasons": reasons}


def ensure_state(config: RunConfig) -> tuple[Path, dict[str, Any]]:
    path = config.output_root / "capacity_state.json"
    expected = build_state(config)
    if path.is_file():
        state = read_json(path)
        if state.get("schema") != SCHEMA or state.get("configuration_signature") != expected["configuration_signature"]:
            raise ValueError("existing capacity output root signature mismatch")
        actual_tasks = {task.get("task_id"): task for task in state.get("tasks", [])}
        expected_tasks = {task["task_id"]: task for task in expected["tasks"]}
        if set(actual_tasks) != set(expected_tasks):
            raise ValueError("existing capacity task matrix mismatch")
        for task_id, actual_task in actual_tasks.items():
            if actual_task.get("signature") != expected_tasks[task_id]["signature"]:
                raise ValueError(f"existing capacity task signature mismatch: {task_id}")
        return path, state
    if config.output_root.exists() and any(entry.name != WRITER_LOCK_NAME for entry in config.output_root.iterdir()):
        raise ValueError("new capacity output root is non-empty and has no matching state")
    config.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(path, expected)
    return path, expected


def _refresh(state: dict[str, Any]) -> dict[str, Any]:
    formal = state.get("mode") == "FORMAL"
    for task in state.get("tasks", []):
        result = validate_task(task, formal=formal)
        task["validation"] = result
        task["status"] = "COMPLETED" if result["valid"] else ("RUNNING" if task.get("status") == "RUNNING" else "INCOMPLETE")
    if state.get("tasks") and all(task.get("status") == "COMPLETED" for task in state["tasks"]):
        state["status"] = "COMPLETED" if _inventory_consistent([task["validation"] for task in state["tasks"]]) else "FAILED"
    elif any(task.get("status") == "RUNNING" for task in state.get("tasks", [])):
        state["status"] = "RUNNING"
    else:
        state["status"] = "INCOMPLETE"
    state["updated_at"] = now()
    return state


def _write_manifest(task: Mapping[str, Any]) -> None:
    path = Path(str(task["manifest"]))
    if path.is_file():
        if read_json(path).get("signature") != task["signature"]:
            raise ValueError("task manifest signature mismatch")
        return
    atomic_json(path, {"schema": SCHEMA, "task_id": task["task_id"], "signature": task["signature"],
                       "signature_payload": task["signature_payload"], "command": task["command"],
                       "environment": task["environment"], "created_at": now()})


def clean_environment(task: Mapping[str, Any]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("NS_")}
    env.update({str(key): str(value) for key, value in task["environment"].items()})
    return env


def run_cohort(config: RunConfig, *, dry_run: bool = False, lock_command: Sequence[str] | None = None) -> dict[str, Any]:
    verify_inputs(config)
    if dry_run:
        state = build_state(config)
        state["status"] = "DRY_RUN"
        state["note"] = "No output directory, lock, checkpoint, or process was created."
        for task in state["tasks"]:
            task["status"] = "PLANNED"
        return state
    with output_root_writer_lock(config.output_root, command=lock_command):
        state_path, state = ensure_state(config)
        _refresh(state)
        atomic_json(state_path, state)
        for task in state["tasks"]:
            if task.get("validation", {}).get("valid"):
                continue
            _write_manifest(task)
            root = Path(str(task["task_root"]))
            log_dir = root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            attempt_number = len(task.get("attempts", [])) + 1
            stdout_path, stderr_path = log_dir / f"attempt_{attempt_number:03d}.stdout.log", log_dir / f"attempt_{attempt_number:03d}.stderr.log"
            task.setdefault("attempts", []).append({"attempt": attempt_number, "started_at": now(), "stdout": str(stdout_path), "stderr": str(stderr_path)})
            task["status"] = "RUNNING"
            atomic_json(state_path, state)
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                completed = subprocess.run(task["command"], cwd=config.repo_root, env=clean_environment(task), stdout=stdout, stderr=stderr, check=False)
            task["attempts"][-1].update(returncode=completed.returncode, finished_at=now())
            result = validate_task(task, formal=not config.smoke)
            task["validation"] = result
            task["status"] = "COMPLETED" if completed.returncode == 0 and result["valid"] else "FAILED"
            atomic_json(state_path, state)
            if task["status"] != "COMPLETED":
                state["status"] = "FAILED"
                atomic_json(state_path, state)
                return state
        _refresh(state)
        atomic_json(state_path, state)
        return state


def load_and_refresh(output_root: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(output_root).resolve() / "capacity_state.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    state = read_json(path)
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported capacity state schema")
    return path, _refresh(copy.deepcopy(state))


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--precision", choices=("bf16", "fp32"), default=FORMAL_PRECISION)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", default=",".join(FORMAL_ARMS))
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--train-cap", type=int, default=FORMAL_TRAIN_CAP)
    parser.add_argument("--val-cap", type=int, default=FORMAL_VAL_CAP)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--learning-rate", default=FORMAL_LEARNING_RATE)
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(repo_root=args.repo_root, protocol_dir=args.protocol_dir, breparg_root=args.breparg_root,
                     output_root=args.output_root, python=args.python, arms=parse_csv(args.arms),
                     seeds=tuple(int(item) for item in parse_csv(args.seeds)), train_cap=args.train_cap,
                     val_cap=args.val_cap, batch_size=args.batch_size, epochs=args.epochs,
                     learning_rate=args.learning_rate, precision=args.precision, smoke=args.smoke)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    add_run_arguments(run_parser)
    probe_parser = subparsers.add_parser("probe")
    add_run_arguments(probe_parser)
    probe_parser.set_defaults(
        smoke=True,
        train_cap=64,
        val_cap=32,
        batch_size=8,
        epochs=1,
        precision="fp32",
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "status":
        _path, state = load_and_refresh(args.output_root)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if args.action == "validate":
        _path, state = load_and_refresh(args.output_root)
        result = validation_summary(state)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    config = config_from_args(args)
    result = run_cohort(config, dry_run=args.dry_run, lock_command=sys.argv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
