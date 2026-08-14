"""Coordinate the fail-closed P0-B learned-VQ 100-CAD assembly measurement.

This command deliberately runs *after* the four formal P0-B stability tasks.
It validates their histories and checkpoint bindings, freezes the historical
100-CAD cohort, delegates reconstruction/assembly to the existing calibration
oracle, and delegates native/strict STEP checks to the existing validity audit.
Heavy checkpoints and STEP files stay under ``--output-dir``.  Only the final
JSON, Markdown, and CSV summaries are written under ``--report-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = "p0b-vq-assembly-measurement-v1"
P0B_SCHEMA = "p0b-stability-retest-v1"
VQ_ARM = "vq_4096_64d_random"
BYPASS_ARM = "continuous_bypass_64d"
FORMAL_ARMS = (VQ_ARM, BYPASS_ARM)
FORMAL_SEEDS = (3, 4)
FORMAL_EPOCHS = 100
FORMAL_TRAIN_CAP = 60_000
FORMAL_VAL_CAP = 12_000
FORMAL_BATCH_SIZE = 128
SELECTION_SEED = 20260809
MAX_CADS = 100
JOINT_ITERATIONS = 200
HISTORICAL_STRICT_ONLY = {
    "original_gt": 84,
    "continuous_bypass_64d": 70,
    "fsq_8192_4d": 49,
}
ZERO_FIELDS = (
    "skipped_train_batches",
    "nonfinite_loss_batches",
    "nonfinite_gradient_batches",
    "nonfinite_state_batches",
    "nonfinite_val_batches",
    "nonfinite_val_samples",
)
TRUE_FIELDS = ("gradients_finite", "training_state_finite", "grad_clip_active")
REPORT_BASENAME = "p0b_vq_assembly_measurement"


class EvidenceError(RuntimeError):
    """Raised when an input or reusable artifact cannot prove its identity."""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"unreadable JSON evidence: {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON evidence must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceError(f"unreadable JSONL evidence: {path}: {type(exc).__name__}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"invalid JSONL evidence: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise EvidenceError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    result = float(value)
    if nonnegative:
        _require(result >= 0.0, f"{label} must be nonnegative")
    return result


def _validate_inventory(value: Any, task_id: str) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == {"train", "val"},
        f"{task_id}: inventory must contain exactly train and val",
    )
    normalized: dict[str, Any] = {}
    for split_name, expected_count in (("train", FORMAL_TRAIN_CAP), ("val", FORMAL_VAL_CAP)):
        item = value.get(split_name)
        _require(isinstance(item, Mapping), f"{task_id}: {split_name} inventory missing")
        _require(
            item.get("schema") == "vq-exact-hash-inventory-v1",
            f"{task_id}: {split_name} inventory schema mismatch",
        )
        _require(
            item.get("count") == expected_count,
            f"{task_id}: {split_name} inventory count mismatch",
        )
        for field in ("ordered_sha256", "sorted_sha256"):
            digest = item.get(field)
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and digest == digest.lower()
                and all(character in "0123456789abcdef" for character in digest),
                f"{task_id}: {split_name} inventory {field} invalid",
            )
        normalized[split_name] = dict(item)
    return normalized


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _bound_path(raw_path: Any, *, root: Path, label: str, must_exist: bool = True) -> Path:
    _require(isinstance(raw_path, str) and bool(raw_path.strip()), f"{label} path is missing")
    path = Path(raw_path).resolve()
    _require(_is_relative_to(path, root.resolve()), f"{label} escapes P0-B output root: {path}")
    if must_exist:
        _require(path.is_file(), f"{label} is missing: {path}")
    return path


def _assert_zero_nonfinite_tree(value: Any, label: str) -> None:
    """Reject any numeric counter whose key contains ``nonfinite`` and is nonzero."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if (
                "nonfinite" in str(key).lower()
                and type(child) in (int, float)
                and float(child) != 0.0
            ):
                raise EvidenceError(f"{child_label} must be zero, observed {child!r}")
            _assert_zero_nonfinite_tree(child, child_label)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_zero_nonfinite_tree(child, f"{label}[{index}]")


def default_checkpoint_loader(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"checkpoint payload is not a mapping: {path}")
    return payload


def _metric(metrics: Mapping[str, Any], keys: Sequence[str], label: str) -> float:
    value: Any = metrics
    for key in keys:
        _require(isinstance(value, Mapping) and key in value, f"{label} is missing")
        value = value[key]
    return _finite_number(value, label, nonnegative=True)


def _validate_run_manifest(
    run_manifest: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    protocol_sha256: str,
    split_pickle_sha256: str,
    expected_inventory: Mapping[str, Any],
) -> str:
    task_id = str(task["task_id"])
    git_state = run_manifest.get("git") or {}
    _require(git_state.get("dirty") is False, f"{task_id}: training Git evidence is dirty")
    commit = git_state.get("commit")
    _require(isinstance(commit, str) and bool(commit), f"{task_id}: training Git commit missing")
    experiment = run_manifest.get("experiment") or {}
    _require(experiment.get("seed") == task["seed"], f"{task_id}: run seed mismatch")
    _require(experiment.get("train_cap") == FORMAL_TRAIN_CAP, f"{task_id}: train cap mismatch")
    _require(experiment.get("val_cap") == FORMAL_VAL_CAP, f"{task_id}: val cap mismatch")
    _require(experiment.get("epochs") == FORMAL_EPOCHS, f"{task_id}: epoch budget mismatch")
    _require(experiment.get("batch_size") == FORMAL_BATCH_SIZE, f"{task_id}: batch size mismatch")
    _require(
        experiment.get("inventory") == expected_inventory,
        f"{task_id}: run inventory binding mismatch",
    )
    protocol = experiment.get("protocol") or {}
    _require(
        protocol.get("protocol_sha256") == protocol_sha256,
        f"{task_id}: run protocol SHA-256 mismatch",
    )
    _require(
        protocol.get("split_pickle_sha256") == split_pickle_sha256,
        f"{task_id}: run split SHA-256 mismatch",
    )
    arms = experiment.get("arms") or []
    _require(
        len(arms) == 1 and isinstance(arms[0], Mapping) and arms[0].get("name") == task["arm"],
        f"{task_id}: run arm binding mismatch",
    )
    relevant_env = (run_manifest.get("launch") or {}).get("relevant_env") or {}
    _require(
        relevant_env.get("NS_VQ_EXPERIMENT_SIGNATURE") == task["signature"],
        f"{task_id}: run signature environment mismatch",
    )
    _require(relevant_env.get("NS_VQ_STRICT_NONFINITE") == "1", f"{task_id}: strict fuse not bound")
    return commit


def _validate_task_evidence(
    task: Mapping[str, Any],
    *,
    output_root: Path,
    protocol_summary: Mapping[str, Any],
    protocol_summary_file_sha256: str,
    split_pickle_file_sha256: str,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    task_id = str(task.get("task_id"))
    arm = task.get("arm")
    seed = task.get("seed")
    _require(task_id == f"{arm}:seed{seed}", f"invalid task identity: {task_id}")
    _require(arm in FORMAL_ARMS and seed in FORMAL_SEEDS, f"unexpected formal task: {task_id}")
    _require(task.get("status") == "COMPLETED", f"{task_id}: task is not COMPLETED")
    validation = task.get("validation") or {}
    _require(validation.get("valid") is True, f"{task_id}: stored validation is not valid")
    _require(validation.get("last_epoch") == 99, f"{task_id}: stored last epoch is not 99")
    _require(validation.get("epochs_observed") == 100, f"{task_id}: stored epoch count is not 100")
    _require(not (validation.get("reasons") or []), f"{task_id}: stored validation has reasons")

    signature_payload = task.get("signature_payload") or {}
    expected_signature_values = {
        "arm": arm,
        "seed": seed,
        "train_cap": FORMAL_TRAIN_CAP,
        "val_cap": FORMAL_VAL_CAP,
        "batch_size": FORMAL_BATCH_SIZE,
        "epochs": FORMAL_EPOCHS,
        "strict_nonfinite_fuse": True,
    }
    for key, expected in expected_signature_values.items():
        _require(
            signature_payload.get(key) == expected,
            f"{task_id}: signature payload {key} mismatch",
        )
    _require(
        signature_payload.get("protocol_summary_sha256") == protocol_summary_file_sha256,
        f"{task_id}: protocol summary file binding mismatch",
    )
    _require(
        signature_payload.get("split_pickle_sha256") == split_pickle_file_sha256,
        f"{task_id}: split pickle file binding mismatch",
    )
    signature = task.get("signature")
    _require(
        signature == canonical_signature(signature_payload),
        f"{task_id}: task signature does not match its payload",
    )

    task_root = Path(str(task.get("task_root"))).resolve()
    _require(_is_relative_to(task_root, output_root), f"{task_id}: task root escapes output root")
    manifest_path = _bound_path(task.get("manifest"), root=output_root, label=f"{task_id} manifest")
    manifest = read_json(manifest_path)
    _require(manifest.get("schema") == P0B_SCHEMA, f"{task_id}: task manifest schema mismatch")
    _require(manifest.get("task_id") == task_id, f"{task_id}: task manifest identity mismatch")
    _require(manifest.get("signature") == signature, f"{task_id}: task manifest signature mismatch")
    _require(
        manifest.get("signature_payload") == signature_payload,
        f"{task_id}: task manifest payload mismatch",
    )

    history_path = _bound_path(task.get("history"), root=output_root, label=f"{task_id} history")
    history_payload = read_json(history_path)
    history_config = history_payload.get("config") or {}
    _require(
        history_config.get("experiment_signature") == signature,
        f"{task_id}: history config signature mismatch",
    )
    observed_precision = history_config.get("precision")
    if isinstance(observed_precision, Mapping):
        observed_precision = observed_precision.get("name")
    _require(
        observed_precision == signature_payload.get("precision"),
        f"{task_id}: history precision mismatch",
    )
    records = history_payload.get("history")
    _require(isinstance(records, list), f"{task_id}: history rows missing")
    epochs = [row.get("epoch") if isinstance(row, Mapping) else None for row in records]
    _require(epochs == list(range(100)), f"{task_id}: epoch coverage must be exactly 0..99")
    for epoch, row in enumerate(records):
        _require(isinstance(row, Mapping), f"{task_id}: epoch {epoch} is not an object")
        for field in ZERO_FIELDS:
            _require(row.get(field) == 0, f"{task_id}: epoch {epoch} {field} is not zero")
        for field in TRUE_FIELDS:
            _require(row.get(field) is True, f"{task_id}: epoch {epoch} {field} is not true")
        _require(
            row.get("nonfinite_state_audits") == 0,
            f"{task_id}: epoch {epoch} nonfinite_state_audits is not zero",
        )
        _require(
            row.get("finite_state_audit_cadence") == "lifecycle_v1",
            f"{task_id}: epoch {epoch} finite-state cadence mismatch",
        )
        _require(
            row.get("full_state_audits") == 1
            and row.get("per_batch_full_state_audits") == 0,
            f"{task_id}: epoch {epoch} full-state audit count mismatch",
        )
        _require(
            isinstance(row.get("finite_state_audit"), Mapping)
            and row["finite_state_audit"].get("status") == "finite",
            f"{task_id}: epoch {epoch} finite-state audit missing",
        )
        _require(
            row.get("experiment_signature") == signature,
            f"{task_id}: epoch {epoch} signature mismatch",
        )
        _finite_number(row.get("preclip_grad_norm"), f"{task_id}: epoch {epoch} gradient norm")
        train_batches = row.get("train_batches")
        val_batches = row.get("val_batches")
        _require(type(train_batches) is int and train_batches > 0, f"{task_id}: bad train batches")
        _require(row.get("finite_train_batches") == train_batches, f"{task_id}: finite train mismatch")
        _require(type(val_batches) is int and val_batches > 0, f"{task_id}: bad val batches")
        _require(row.get("finite_val_batches") == val_batches, f"{task_id}: finite val mismatch")
        _assert_zero_nonfinite_tree(row, f"{task_id}.epoch{epoch}")

    sweep_path = _bound_path(task.get("sweep"), root=output_root, label=f"{task_id} sweep")
    sweep = read_json(sweep_path)
    ranking = sweep.get("mse_ranking") or []
    _require(
        len(ranking) == 1 and isinstance(ranking[0], Mapping) and ranking[0].get("name") == arm,
        f"{task_id}: sweep must contain only its own arm",
    )
    sweep_row = ranking[0]
    task_inventory = _validate_inventory(sweep_row.get("inventory"), task_id)
    _require(
        validation.get("inventory") == task_inventory,
        f"{task_id}: stored validation inventory binding mismatch",
    )
    _require(
        (history_config.get("signature_configuration") or {}).get("inventory")
        == task_inventory,
        f"{task_id}: history inventory binding mismatch",
    )
    _require(sweep_row.get("epochs_ran") == 100, f"{task_id}: sweep epoch count mismatch")
    _require(sweep_row.get("final_checkpoint_epoch") == 99, f"{task_id}: final epoch is not 99")
    _require(
        sweep_row.get("experiment_signature") == signature,
        f"{task_id}: sweep experiment signature mismatch",
    )

    final_path = _bound_path(
        task.get("final_checkpoint"), root=output_root, label=f"{task_id} final checkpoint"
    )
    # The rolling checkpoint proves that the completed task retained resumable
    # optimizer and RNG state.  Evaluation loads a model-only best/final
    # checkpoint selected below, not the rolling payload.
    _bound_path(
        task.get("rolling_checkpoint"),
        root=output_root,
        label=f"{task_id} rolling checkpoint",
    )
    if arm == VQ_ARM:
        checkpoint_role = "best"
        checkpoint_path = _bound_path(
            task.get("best_checkpoint"),
            root=output_root,
            label=f"{task_id} best checkpoint",
        )
        _require(
            Path(str(sweep_row.get("checkpoint_best"))).resolve() == checkpoint_path,
            f"{task_id}: sweep best checkpoint path mismatch",
        )
    else:
        checkpoint_role = "final"
        checkpoint_path = final_path
        _require(
            Path(str(sweep_row.get("checkpoint_final"))).resolve() == checkpoint_path,
            f"{task_id}: sweep final checkpoint path mismatch",
        )

    payload = checkpoint_loader(checkpoint_path)
    _require(isinstance(payload, Mapping), f"{task_id}: checkpoint payload is not a mapping")
    model_state = payload.get("model_state_dict")
    _require(isinstance(model_state, Mapping) and bool(model_state), f"{task_id}: model state missing")
    checkpoint_epoch = payload.get("checkpoint_epoch")
    if checkpoint_role == "best":
        _require(
            type(checkpoint_epoch) is int and 0 <= checkpoint_epoch <= 99,
            f"{task_id}: checkpoint epoch is invalid",
        )
        _require(
            sweep_row.get("checkpoint_epoch") == checkpoint_epoch,
            f"{task_id}: sweep/checkpoint epoch mismatch",
        )
    else:
        _require(checkpoint_epoch == 99, f"{task_id}: final checkpoint epoch is not 99")
    quantizer = payload.get("quantizer") or {}
    expected_kind = "learned_vq" if arm == VQ_ARM else "continuous_bypass"
    _require(quantizer.get("kind") == expected_kind, f"{task_id}: quantizer kind mismatch")
    if arm == VQ_ARM:
        _require(quantizer.get("codebook_size") == 4096, f"{task_id}: VQ codebook mismatch")
        _require(quantizer.get("embedding_dim") == 64, f"{task_id}: VQ embedding mismatch")
        _require(quantizer.get("anchor") == "random", f"{task_id}: VQ anchor mismatch")

    metrics = payload.get("validation_metrics") or {}
    _require(isinstance(metrics, Mapping), f"{task_id}: checkpoint validation metrics missing")
    _assert_zero_nonfinite_tree(metrics, f"{task_id}.checkpoint_metrics")
    curved_parent_mse = _metric(
        metrics,
        ("parent_cluster_reconstruction_mse", "surface_curved_proxy", "mse"),
        f"{task_id}: best curved parent MSE",
    )
    parent_mse = _metric(
        metrics,
        ("parent_cluster_mse", "mse"),
        f"{task_id}: best parent MSE",
    )
    checkpoint_val_mse = _finite_number(
        payload.get("validation_loss"),
        f"{task_id}: {checkpoint_role} checkpoint validation MSE",
        nonnegative=True,
    )
    if checkpoint_role == "best":
        _require(
            sweep_row.get("best_val_metrics") == metrics,
            f"{task_id}: sweep best metrics do not match checkpoint",
        )
        _require(
            _finite_number(
                sweep_row.get("checkpoint_val_recon"),
                f"{task_id}: sweep best validation MSE",
                nonnegative=True,
            )
            == checkpoint_val_mse,
            f"{task_id}: sweep best validation MSE mismatch",
        )

    protocol_sha256 = str(protocol_summary.get("protocol_sha256"))
    split_pickle_sha256 = str(protocol_summary.get("split_pickle_sha256"))
    checkpoint_context = payload.get("checkpoint_context") or {}
    _require(
        checkpoint_context.get("inventory") == task_inventory,
        f"{task_id}: checkpoint inventory binding mismatch",
    )
    _require(
        checkpoint_context.get("protocol_sha256") == protocol_sha256,
        f"{task_id}: checkpoint protocol binding mismatch",
    )
    _require(
        checkpoint_context.get("split_pickle_sha256") == split_pickle_sha256,
        f"{task_id}: checkpoint split binding mismatch",
    )
    checkpoint_run_manifest = checkpoint_context.get("run_manifest") or {}
    sweep_run_manifest = sweep.get("run_manifest") or {}
    _require(
        checkpoint_run_manifest == sweep_run_manifest,
        f"{task_id}: checkpoint and sweep run manifests differ",
    )
    git_commit = _validate_run_manifest(
        checkpoint_run_manifest,
        task=task,
        protocol_sha256=protocol_sha256,
        split_pickle_sha256=split_pickle_sha256,
        expected_inventory=task_inventory,
    )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_role == "best":
        promotion = sweep_row.get("promotion") or {}
        binding = promotion.get("binding") or {}
        _require(
            binding.get("checkpoint_sha256") == checkpoint_sha256,
            f"{task_id}: checkpoint SHA binding mismatch",
        )
        _require(
            binding.get("checkpoint_epoch") == checkpoint_epoch,
            f"{task_id}: checkpoint epoch binding mismatch",
        )
        _require(
            binding.get("protocol_sha256") == protocol_sha256,
            f"{task_id}: protocol promotion binding mismatch",
        )
        _require(
            binding.get("split_pickle_sha256") == split_pickle_sha256,
            f"{task_id}: split promotion binding mismatch",
        )
        _require(
            binding.get("git_commit") == git_commit,
            f"{task_id}: Git promotion binding mismatch",
        )

    return {
        "task_id": task_id,
        "arm": arm,
        "seed": seed,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "curved_parent_mse": curved_parent_mse,
        "parent_mse": parent_mse,
        "checkpoint_val_mse": checkpoint_val_mse,
        "best_val_mse": checkpoint_val_mse if checkpoint_role == "best" else None,
        "experiment_signature": signature,
        "git_commit": git_commit,
        "protocol_sha256": protocol_sha256,
        "split_pickle_sha256": split_pickle_sha256,
        "inventory": task_inventory,
        "nonfinite_events": 0,
    }


def validate_p0b_evidence(
    output_root: Path,
    *,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = default_checkpoint_loader,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    state_path = output_root / "p0b_state.json"
    _require(state_path.is_file(), f"P0-B state missing: {state_path}")
    state = read_json(state_path)
    _require(state.get("schema") == P0B_SCHEMA, "unsupported P0-B state schema")
    _require(state.get("mode") == "FORMAL", "P0-B evidence is not a formal run")
    _require(state.get("formal_result_eligible") is True, "P0-B state is not formal-result eligible")
    _require(state.get("status") == "COMPLETED", "P0-B state is not COMPLETED")
    configuration = state.get("configuration") or {}
    _require(
        state.get("configuration_signature") == canonical_signature(configuration),
        "P0-B configuration signature mismatch",
    )
    _require(tuple(configuration.get("arms") or ()) == FORMAL_ARMS, "P0-B arm matrix mismatch")
    _require(tuple(configuration.get("seeds") or ()) == FORMAL_SEEDS, "P0-B seed matrix mismatch")
    _require(configuration.get("train_cap") == FORMAL_TRAIN_CAP, "P0-B train cap mismatch")
    _require(configuration.get("val_cap") == FORMAL_VAL_CAP, "P0-B validation cap mismatch")
    _require(configuration.get("batch_size") == FORMAL_BATCH_SIZE, "P0-B batch size mismatch")
    _require(configuration.get("epochs") == FORMAL_EPOCHS, "P0-B epoch budget mismatch")

    protocol_dir = Path(str(configuration.get("protocol_dir"))).resolve()
    protocol_summary_path = protocol_dir / "protocol_summary.json"
    split_path = protocol_dir / "split.pkl"
    _require(protocol_summary_path.is_file(), f"protocol summary missing: {protocol_summary_path}")
    _require(split_path.is_file(), f"protocol split missing: {split_path}")
    protocol_summary = read_json(protocol_summary_path)
    _require(protocol_summary.get("status") == "VERIFIED", "P0-B protocol is not VERIFIED")
    _require(
        all(value == 0 for value in (protocol_summary.get("parent_overlap_counts") or {}).values()),
        "P0-B protocol has parent overlap",
    )
    protocol_summary_file_sha256 = sha256_file(protocol_summary_path)
    split_pickle_file_sha256 = sha256_file(split_path)
    _require(
        protocol_summary.get("split_pickle_sha256") == split_pickle_file_sha256,
        "protocol split internal SHA-256 mismatch",
    )

    tasks = state.get("tasks") or []
    expected_ids = {f"{arm}:seed{seed}" for arm in FORMAL_ARMS for seed in FORMAL_SEEDS}
    observed_ids = [task.get("task_id") for task in tasks if isinstance(task, Mapping)]
    _require(len(observed_ids) == len(set(observed_ids)), "P0-B task IDs are duplicated")
    _require(set(observed_ids) == expected_ids, "P0-B task matrix is incomplete or contaminated")
    validated = [
        _validate_task_evidence(
            task,
            output_root=output_root,
            protocol_summary=protocol_summary,
            protocol_summary_file_sha256=protocol_summary_file_sha256,
            split_pickle_file_sha256=split_pickle_file_sha256,
            checkpoint_loader=checkpoint_loader,
        )
        for task in tasks
    ]
    inventory_signatures = {
        canonical_signature(task["inventory"]) for task in validated
    }
    _require(
        len(inventory_signatures) == 1,
        "P0-B task inventories differ across arm/seed tasks",
    )
    return {
        "state_path": str(state_path),
        "configuration_signature": state["configuration_signature"],
        "protocol_dir": str(protocol_dir),
        "protocol_sha256": protocol_summary["protocol_sha256"],
        "split_pickle_sha256": protocol_summary["split_pickle_sha256"],
        "tasks": sorted(validated, key=lambda item: (item["arm"], item["seed"])),
        "inventory": validated[0]["inventory"],
        "inventory_consistent": True,
        "zero_nonfinite": True,
    }


def select_healthy_vq_checkpoint(evidence: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [dict(task) for task in evidence.get("tasks", []) if task.get("arm") == VQ_ARM]
    _require(len(candidates) == 2, "exactly two healthy learned-VQ candidates are required")
    for candidate in candidates:
        for key in ("curved_parent_mse", "parent_mse", "best_val_mse"):
            _finite_number(candidate.get(key), f"{candidate.get('task_id')}: {key}", nonnegative=True)
        _require(candidate.get("nonfinite_events") == 0, f"{candidate.get('task_id')}: nonfinite candidate")
    # Curved parent-cluster MSE is the scientific selection metric.  Global
    # parent MSE and global best validation MSE resolve metric ties, followed by
    # seed and artifact identity for a deterministic final tie-break.
    candidates.sort(
        key=lambda item: (
            item["curved_parent_mse"],
            item["parent_mse"],
            item["best_val_mse"],
            item["seed"],
            item["checkpoint_sha256"],
            item["checkpoint_path"],
        )
    )
    selected = dict(candidates[0])
    selected["selection_metric"] = "lowest finite best-checkpoint curved parent-cluster MSE"
    selected["deterministic_tie_break"] = [
        "parent_mse",
        "best_val_mse",
        "seed",
        "checkpoint_sha256",
        "checkpoint_path",
    ]
    selected["candidate_ranking"] = [
        {
            key: candidate[key]
            for key in (
                "task_id",
                "seed",
                "checkpoint_sha256",
                "checkpoint_epoch",
                "curved_parent_mse",
                "parent_mse",
                "best_val_mse",
            )
        }
        for candidate in candidates
    ]
    return selected


def _cohort_identity(row: Mapping[str, Any], label: str) -> tuple[str, str]:
    cad_id = row.get("cad_id")
    parent_id = row.get("parent_id")
    _require(isinstance(cad_id, str) and bool(cad_id), f"{label}: CAD ID missing")
    _require(isinstance(parent_id, str) and bool(parent_id), f"{label}: parent ID missing")
    source_path = row.get("source_path", row.get("path"))
    _require(isinstance(source_path, str) and bool(source_path), f"{label}: source path missing")
    _require(Path(source_path).stem == cad_id, f"{label}: source filename/CAD ID mismatch")
    return cad_id, parent_id


def verify_fixed_cohort(
    protocol_dir: Path,
    historical_manifest: Path,
    *,
    protocol_sha256: str,
    selector: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    historical_manifest = Path(historical_manifest).resolve()
    _require(historical_manifest.is_file(), f"historical calibration manifest missing: {historical_manifest}")
    historical_rows = read_jsonl(historical_manifest)
    originals = [row for row in historical_rows if row.get("arm") == "original"]
    _require(len(originals) == MAX_CADS, "historical manifest must contain exactly 100 original rows")
    historical_identities: list[tuple[str, str]] = []
    for index, row in enumerate(originals):
        _require(row.get("selection_seed") == SELECTION_SEED, f"historical original row {index}: seed mismatch")
        _require(row.get("protocol_sha256") == protocol_sha256, f"historical original row {index}: protocol mismatch")
        _require(row.get("checkpoint_sha256") is None, f"historical original row {index}: checkpoint must be null")
        historical_identities.append(_cohort_identity(row, f"historical original row {index}"))
    _require(len(set(historical_identities)) == MAX_CADS, "historical original identities are duplicated")
    _require(len({parent for _, parent in historical_identities}) == MAX_CADS, "historical parents are not unique")

    if selector is None:
        from tools.run_assembly_calibration_oracle import select_validation_cads

        selector = select_validation_cads
    selected = list(selector(Path(protocol_dir), max_cads=MAX_CADS, seed=SELECTION_SEED))
    _require(len(selected) == MAX_CADS, "current protocol cannot select the frozen 100 CADs")
    selected_identities = [
        _cohort_identity(row, f"current selected row {index}") for index, row in enumerate(selected)
    ]
    _require(len(set(selected_identities)) == MAX_CADS, "current selected identities are duplicated")
    _require(
        set(selected_identities) == set(historical_identities),
        "current 100-CAD selection does not equal the historical original cohort",
    )
    identity_rows = [
        {"cad_id": cad_id, "parent_id": parent_id}
        for cad_id, parent_id in sorted(historical_identities)
    ]
    return {
        "selection_seed": SELECTION_SEED,
        "max_cads": MAX_CADS,
        "identities": identity_rows,
        "identity_sha256": canonical_signature(identity_rows),
        "historical_manifest": str(historical_manifest),
        "historical_manifest_sha256": sha256_file(historical_manifest),
    }


def _expected_identity_map(cohort: Mapping[str, Any]) -> dict[str, str]:
    return {str(row["cad_id"]): str(row["parent_id"]) for row in cohort["identities"]}


def _validate_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: Mapping[str, Any],
    selected: Mapping[str, Any],
    protocol_sha256: str,
    allow_partial: bool,
    verify_steps: bool,
) -> None:
    expected = _expected_identity_map(cohort)
    _require(len(rows) <= MAX_CADS, "calibration manifest has more than 100 rows")
    observed: set[str] = set()
    for index, row in enumerate(rows):
        cad_id = row.get("cad_id")
        _require(cad_id in expected, f"calibration row {index}: CAD is outside frozen cohort")
        _require(cad_id not in observed, f"calibration row {index}: duplicate CAD")
        observed.add(str(cad_id))
        _require(row.get("parent_id") == expected[cad_id], f"calibration row {index}: parent mismatch")
        _require(row.get("arm") == VQ_ARM, f"calibration row {index}: unexpected arm")
        _require(row.get("selection_seed") == SELECTION_SEED, f"calibration row {index}: seed mismatch")
        _require(row.get("protocol_sha256") == protocol_sha256, f"calibration row {index}: protocol mismatch")
        _require(
            row.get("checkpoint_sha256") == selected["checkpoint_sha256"],
            f"calibration row {index}: checkpoint binding mismatch",
        )
        _require(type(row.get("step_saved")) is bool, f"calibration row {index}: step_saved is not boolean")
        _require(type(row.get("brep_valid")) is bool, f"calibration row {index}: brep_valid is not boolean")
        if verify_steps and row.get("step_saved"):
            step_path = Path(str(row.get("step_path"))).resolve()
            _require(step_path.is_file(), f"calibration row {index}: STEP missing: {step_path}")
            _require(
                row.get("step_sha256") == sha256_file(step_path),
                f"calibration row {index}: STEP SHA-256 mismatch",
            )
    if not allow_partial:
        _require(observed == set(expected), "calibration manifest does not retain all 100 attempts")


def _calibration_is_complete(
    calibration_dir: Path,
    *,
    cohort: Mapping[str, Any],
    selected: Mapping[str, Any],
    protocol_sha256: str,
) -> bool:
    manifest_path = calibration_dir / "calibration_manifest.jsonl"
    state_path = calibration_dir / "calibration_state.json"
    if not manifest_path.exists():
        _require(not state_path.exists(), "calibration state exists without a manifest")
        return False
    rows = read_jsonl(manifest_path)
    _validate_calibration_rows(
        rows,
        cohort=cohort,
        selected=selected,
        protocol_sha256=protocol_sha256,
        allow_partial=True,
        verify_steps=True,
    )
    if len(rows) < MAX_CADS:
        return False
    # The oracle appends each attempt before writing its aggregate state.  A
    # process interruption after row 100 is therefore a recoverable partial
    # write: rerunning the oracle skips all bound rows and recreates the state.
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    _require(state.get("status") == "COMPLETED", "calibration state is not COMPLETED")
    _require(state.get("selected_cads") == MAX_CADS, "calibration selected_cads mismatch")
    _require(state.get("expected_rows") == MAX_CADS, "calibration expected_rows mismatch")
    _require(state.get("manifest_rows") == MAX_CADS, "calibration manifest_rows mismatch")
    _require(state.get("arms") == [VQ_ARM], "calibration state contains another arm")
    _require(state.get("protocol_sha256") == protocol_sha256, "calibration state protocol mismatch")
    checkpoint = (state.get("checkpoints") or {}).get(VQ_ARM) or {}
    _require(
        checkpoint.get("sha256") == selected["checkpoint_sha256"],
        "calibration state checkpoint mismatch",
    )
    _validate_calibration_rows(
        rows,
        cohort=cohort,
        selected=selected,
        protocol_sha256=protocol_sha256,
        allow_partial=False,
        verify_steps=True,
    )
    return True


def _validate_audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    allow_partial: bool,
) -> None:
    expected = _expected_identity_map(cohort)
    calibration_by_cad = {str(row["cad_id"]): row for row in calibration_rows}
    _require(len(rows) <= MAX_CADS, "validity audit has more than 100 rows")
    observed: set[str] = set()
    for index, row in enumerate(rows):
        cad_id = row.get("cad_id")
        _require(cad_id in expected, f"validity row {index}: CAD is outside frozen cohort")
        _require(cad_id not in observed, f"validity row {index}: duplicate CAD")
        observed.add(str(cad_id))
        _require(row.get("arm") == VQ_ARM, f"validity row {index}: unexpected arm")
        native = row.get("native_brep_valid")
        strict = row.get("strict_brep_valid")
        _require(native in (None, True, False), f"validity row {index}: invalid native value")
        _require(type(strict) is bool, f"validity row {index}: invalid strict value")
        source = calibration_by_cad.get(str(cad_id))
        _require(source is not None, f"validity row {index}: no calibration attempt")
        _require(row.get("source_status") == source.get("status"), f"validity row {index}: source status mismatch")
    if not allow_partial:
        _require(observed == set(expected), "validity audit does not retain all 100 attempts")


def _validity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = len(rows)
    step_saved = sum(row.get("native_brep_valid") is not None for row in rows)
    native = sum(row.get("native_brep_valid") is True for row in rows)
    strict = sum(row.get("strict_brep_valid") is True for row in rows)
    both = sum(
        row.get("native_brep_valid") is True and row.get("strict_brep_valid") is True
        for row in rows
    )
    return {
        "attempts": attempts,
        "step_saved": step_saved,
        "step_saved_rate": step_saved / attempts if attempts else None,
        "native_brep_valid": native,
        "native_brep_valid_rate": native / attempts if attempts else None,
        "strict_brep_valid": strict,
        "strict_brep_valid_rate": strict / attempts if attempts else None,
        "both_valid": both,
        "both_valid_rate": both / attempts if attempts else None,
        "native_only": sum(
            row.get("native_brep_valid") is True and row.get("strict_brep_valid") is False
            for row in rows
        ),
        "strict_only": sum(
            row.get("native_brep_valid") is False and row.get("strict_brep_valid") is True
            for row in rows
        ),
        "neither_valid": sum(
            row.get("native_brep_valid") is False and row.get("strict_brep_valid") is False
            for row in rows
        ),
        "no_step": attempts - step_saved,
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
    }


def _audit_is_complete(
    audit_dir: Path,
    *,
    cohort: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> bool:
    manifest_path = audit_dir / "step_validity_audit.jsonl"
    summary_path = audit_dir / "step_validity_summary.json"
    if not manifest_path.exists() and not summary_path.exists():
        return False
    if not manifest_path.exists() or not summary_path.exists():
        return False
    rows = read_jsonl(manifest_path)
    _validate_audit_rows(rows, cohort=cohort, calibration_rows=calibration_rows, allow_partial=True)
    if len(rows) < MAX_CADS:
        return False
    _validate_audit_rows(rows, cohort=cohort, calibration_rows=calibration_rows, allow_partial=False)
    observed = read_json(summary_path)
    expected = _validity_summary(rows)
    for key, value in expected.items():
        _require(observed.get(key) == value, f"validity summary mismatch: {key}")
    return True


def run_external(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("NS_")
    }
    with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=Path(cwd),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(completed.returncode)


def _ensure_measurement_state(output_dir: Path, intent: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    output_dir = Path(output_dir)
    state_path = output_dir / "measurement_state.json"
    signature = canonical_signature(intent)
    if state_path.is_file():
        state = read_json(state_path)
        _require(state.get("schema") == SCHEMA, "existing measurement state schema mismatch")
        _require(state.get("signature") == signature, "existing measurement intent signature mismatch")
        _require(state.get("intent") == intent, "existing measurement intent is dirty")
        return state_path, state
    if output_dir.exists():
        unexpected = [path for path in output_dir.iterdir() if path.name != "measurement_state.json"]
        _require(not unexpected, "measurement output is non-empty and has no matching state")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": SCHEMA,
        "signature": signature,
        "status": "PREPARED",
        "created_at": now(),
        "updated_at": now(),
        "intent": dict(intent),
        "commands": [],
    }
    atomic_json(state_path, state)
    return state_path, state


def _record_command(
    state_path: Path,
    state: dict[str, Any],
    *,
    stage: str,
    command: Sequence[str],
    command_runner: Callable[..., int],
    cwd: Path,
    logs_dir: Path,
) -> None:
    attempt = 1 + sum(item.get("stage") == stage for item in state.get("commands", []))
    stdout_path = logs_dir / f"{stage}_attempt_{attempt:03d}.stdout.log"
    stderr_path = logs_dir / f"{stage}_attempt_{attempt:03d}.stderr.log"
    record = {
        "stage": stage,
        "attempt": attempt,
        "command": list(command),
        "started_at": now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    state.setdefault("commands", []).append(record)
    state.update(status=stage.upper(), updated_at=now())
    atomic_json(state_path, state)
    returncode = command_runner(
        list(command), cwd=Path(cwd), stdout_path=stdout_path, stderr_path=stderr_path
    )
    record.update(returncode=int(returncode), finished_at=now())
    state["updated_at"] = now()
    if returncode != 0:
        state.update(status="FAILED", error=f"{stage} command returned {returncode}")
        atomic_json(state_path, state)
        raise RuntimeError(f"{stage} command failed with return code {returncode}")
    atomic_json(state_path, state)


def _render_reports(
    report_dir: Path,
    *,
    state: Mapping[str, Any],
    evidence: Mapping[str, Any],
    selected: Mapping[str, Any],
    cohort: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    calibration_by_cad = {str(row["cad_id"]): row for row in calibration_rows}
    audit_by_cad = {str(row["cad_id"]): row for row in audit_rows}
    attempt_rows: list[dict[str, Any]] = []
    for identity in cohort["identities"]:
        cad_id = str(identity["cad_id"])
        calibration = calibration_by_cad[cad_id]
        audit = audit_by_cad[cad_id]
        native = audit.get("native_brep_valid")
        strict = audit.get("strict_brep_valid") is True
        attempt_rows.append(
            {
                "cad_id": cad_id,
                "parent_id": identity["parent_id"],
                "calibration_status": calibration.get("status"),
                "audit_status": audit.get("status"),
                "step_saved": calibration.get("step_saved") is True,
                "native_brep_valid": native,
                "strict_brep_valid": strict,
                "both_valid": native is True and strict,
                "global_patch_mse": calibration.get("global_patch_mse"),
                "surface_mse": calibration.get("surface_mse"),
                "curved_mse": calibration.get("curved_mse"),
                "planar_mse": calibration.get("planar_mse"),
                "edge_mse": calibration.get("edge_mse"),
                "nonfinite_patches": calibration.get("nonfinite_patches"),
            }
        )
    summary = _validity_summary(audit_rows)
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "created_at": state["created_at"],
        "completed_at": state["completed_at"],
        "measurement_signature": state["signature"],
        "protocol": {
            "protocol_sha256": evidence["protocol_sha256"],
            "split_pickle_sha256": evidence["split_pickle_sha256"],
        },
        "p0b_gate": {
            "formal_tasks": 4,
            "completed_epoch": 99,
            "nonfinite_events": 0,
            "passed": True,
        },
        "selected_checkpoint": {
            key: selected[key]
            for key in (
                "task_id",
                "arm",
                "seed",
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_epoch",
                "curved_parent_mse",
                "parent_mse",
                "best_val_mse",
                "selection_metric",
                "deterministic_tie_break",
                "candidate_ranking",
            )
        },
        "cohort": {
            "selection_seed": cohort["selection_seed"],
            "attempts": cohort["max_cads"],
            "identity_sha256": cohort["identity_sha256"],
            "historical_manifest_sha256": cohort["historical_manifest_sha256"],
            "matches_historical_original_arm": True,
        },
        "learned_vq_dual_validity": summary,
        "historical_strict_only_comparison": {
            "attempts_each": 100,
            "strict_valid_counts": HISTORICAL_STRICT_ONLY,
            "note": (
                "GT 84, bypass 70, and FSQ 49 are historical strict-only runner values; "
                "native and both-valid were not reported by that historical runner."
            ),
        },
        "failure_denominator": {
            "attempts": 100,
            "all_reconstruction_assembly_and_no_step_failures_retained": True,
            "calibration_status_counts": dict(
                sorted(Counter(str(row.get("status")) for row in calibration_rows).items())
            ),
        },
        "artifact_policy": {
            "git_eligible": [
                f"{REPORT_BASENAME}.json",
                f"{REPORT_BASENAME}.md",
                f"{REPORT_BASENAME}.csv",
            ],
            "excluded_from_git": ["checkpoints (*.pt)", "STEP files (*.step)", "NPZ files (*.npz)"],
        },
    }

    json_path = report_dir / f"{REPORT_BASENAME}.json"
    markdown_path = report_dir / f"{REPORT_BASENAME}.md"
    csv_path = report_dir / f"{REPORT_BASENAME}.csv"
    atomic_json(json_path, payload)

    def rate(count: int) -> str:
        return f"{count / 100:.1%}"

    markdown = f"""# P0-B Learned-VQ Fixed-100-CAD Assembly Measurement

Status: **COMPLETED**. All four formal VQ/bypass seed 3/4 tasks reached epoch 99 with zero non-finite events and valid checkpoint bindings. The selected learned-VQ checkpoint is `{selected['task_id']}` (`{selected['checkpoint_sha256']}`).

The cohort is the exact historical original-arm cohort selected with seed `{SELECTION_SEED}`. All 100 attempts remain in the denominator, including reconstruction failures, assembly failures, invalid STEP files, and missing STEP files.

| Result | Native valid | Strict valid | Both valid | Attempts |
| --- | ---: | ---: | ---: | ---: |
| Learned VQ (current dual audit) | {summary['native_brep_valid']} ({rate(summary['native_brep_valid'])}) | {summary['strict_brep_valid']} ({rate(summary['strict_brep_valid'])}) | {summary['both_valid']} ({rate(summary['both_valid'])}) | 100 |
| GT/original (historical strict-only) | not measured | 84 (84.0%) | not measured | 100 |
| Continuous bypass (historical strict-only) | not measured | 70 (70.0%) | not measured | 100 |
| FSQ-8192/4D (historical strict-only) | not measured | 49 (49.0%) | not measured | 100 |

The 84/70/49 values are historical **strict-only runner values**. They must not be presented as native-valid or both-valid measurements. The current learned-VQ row reports native, project-strict, and their intersection separately.

## Checkpoint Selection

The learned-VQ checkpoint was selected by the lowest finite best-checkpoint curved parent-cluster MSE. Deterministic ties are resolved by parent MSE, global best validation MSE, seed, checkpoint SHA-256, and path. The selected curved parent MSE is `{selected['curved_parent_mse']:.9g}`.

## Artifact Boundary

Only this Markdown report and its JSON/CSV companions are suitable for Git. Checkpoints, STEP files, NPZ files, and other heavy evaluation artifacts remain in the local measurement output and are not copied into the report directory.
"""
    _atomic_text(markdown_path, markdown)

    columns = [
        "cad_id",
        "parent_id",
        "calibration_status",
        "audit_status",
        "step_saved",
        "native_brep_valid",
        "strict_brep_valid",
        "both_valid",
        "global_patch_mse",
        "surface_mse",
        "curved_mse",
        "planar_mse",
        "edge_mse",
        "nonfinite_patches",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(attempt_rows)
    _atomic_text(csv_path, buffer.getvalue())
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "csv": str(csv_path),
    }


def run_measurement(
    *,
    repo_root: Path,
    p0b_output_root: Path,
    historical_calibration_manifest: Path,
    breparg_root: Path,
    output_dir: Path,
    report_dir: Path,
    python: Path = Path(sys.executable),
    device: str = "auto",
    batch_size: int = 64,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = default_checkpoint_loader,
    selector: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    command_runner: Callable[..., int] = run_external,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    p0b_output_root = Path(p0b_output_root).resolve()
    breparg_root = Path(breparg_root).resolve()
    output_dir = Path(output_dir).resolve()
    report_dir = Path(report_dir).resolve()
    python = Path(python).resolve()
    _require((repo_root / "tools" / "run_assembly_calibration_oracle.py").is_file(), "calibration oracle missing")
    _require((repo_root / "tools" / "audit_assembly_step_validity.py").is_file(), "validity audit missing")
    _require((breparg_root / "utils.py").is_file(), "BrepARG utils.py missing")
    _require(python.is_file(), f"Python executable missing: {python}")
    _require(device in {"auto", "cpu", "cuda"}, f"unsupported device: {device}")
    _require(type(batch_size) is int and batch_size > 0, "batch size must be positive")

    evidence = validate_p0b_evidence(p0b_output_root, checkpoint_loader=checkpoint_loader)
    selected = select_healthy_vq_checkpoint(evidence)
    cohort = verify_fixed_cohort(
        Path(evidence["protocol_dir"]),
        historical_calibration_manifest,
        protocol_sha256=evidence["protocol_sha256"],
        selector=selector,
    )
    intent = {
        "p0b_configuration_signature": evidence["configuration_signature"],
        "p0b_task_signatures": {
            task["task_id"]: task["experiment_signature"] for task in evidence["tasks"]
        },
        "p0b_checkpoint_sha256": {
            task["task_id"]: task["checkpoint_sha256"] for task in evidence["tasks"]
        },
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_task_id": selected["task_id"],
        "protocol_sha256": evidence["protocol_sha256"],
        "split_pickle_sha256": evidence["split_pickle_sha256"],
        "cohort_identity_sha256": cohort["identity_sha256"],
        "historical_manifest_sha256": cohort["historical_manifest_sha256"],
        "arm": VQ_ARM,
        "selection_seed": SELECTION_SEED,
        "max_cads": MAX_CADS,
        "joint_iterations": JOINT_ITERATIONS,
        "preserve_all_failures_denominator": True,
    }
    if dry_run:
        return {
            "schema": SCHEMA,
            "status": "DRY_RUN",
            "intent": intent,
            "selected_checkpoint": selected,
            "cohort": {key: value for key, value in cohort.items() if key != "identities"},
            "note": "Evidence was validated; no output directories or evaluation processes were created.",
        }

    state_path, state = _ensure_measurement_state(output_dir, intent)
    calibration_dir = output_dir / "calibration"
    audit_dir = output_dir / "validity_audit"
    logs_dir = output_dir / "logs"
    try:
        if not _calibration_is_complete(
            calibration_dir,
            cohort=cohort,
            selected=selected,
            protocol_sha256=evidence["protocol_sha256"],
        ):
            calibration_command = [
                str(python),
                str(repo_root / "tools" / "run_assembly_calibration_oracle.py"),
                "--protocol-dir",
                evidence["protocol_dir"],
                "--max-cads",
                str(MAX_CADS),
                "--seed",
                str(SELECTION_SEED),
                "--breparg-root",
                str(breparg_root),
                "--checkpoint",
                f"{VQ_ARM}={selected['checkpoint_path']}",
                "--output-dir",
                str(calibration_dir),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--joint-iterations",
                str(JOINT_ITERATIONS),
            ]
            _record_command(
                state_path,
                state,
                stage="calibration",
                command=calibration_command,
                command_runner=command_runner,
                cwd=repo_root,
                logs_dir=logs_dir,
            )
        _require(
            _calibration_is_complete(
                calibration_dir,
                cohort=cohort,
                selected=selected,
                protocol_sha256=evidence["protocol_sha256"],
            ),
            "calibration did not produce a complete bound 100-attempt manifest",
        )
        calibration_rows = read_jsonl(calibration_dir / "calibration_manifest.jsonl")

        if not _audit_is_complete(
            audit_dir, cohort=cohort, calibration_rows=calibration_rows
        ):
            audit_command = [
                str(python),
                str(repo_root / "tools" / "audit_assembly_step_validity.py"),
                "--manifest",
                str(calibration_dir / "calibration_manifest.jsonl"),
                "--breparg-root",
                str(breparg_root),
                "--step-root",
                str(calibration_dir / "steps"),
                "--output-dir",
                str(audit_dir),
            ]
            _record_command(
                state_path,
                state,
                stage="validity_audit",
                command=audit_command,
                command_runner=command_runner,
                cwd=repo_root,
                logs_dir=logs_dir,
            )
        _require(
            _audit_is_complete(audit_dir, cohort=cohort, calibration_rows=calibration_rows),
            "validity audit did not produce a complete 100-attempt dual audit",
        )
        audit_rows = read_jsonl(audit_dir / "step_validity_audit.jsonl")
        state.setdefault("completed_at", now())
        state.update(status="COMPLETED", updated_at=now(), error=None)
        reports = _render_reports(
            report_dir,
            state=state,
            evidence=evidence,
            selected=selected,
            cohort=cohort,
            calibration_rows=calibration_rows,
            audit_rows=audit_rows,
        )
        state["reports"] = {
            name: {"path": path, "sha256": sha256_file(Path(path))}
            for name, path in reports.items()
        }
        state["summary"] = _validity_summary(audit_rows)
        atomic_json(state_path, state)
        return {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "measurement_signature": state["signature"],
            "selected_checkpoint": selected,
            "cohort_identity_sha256": cohort["identity_sha256"],
            "summary": state["summary"],
            "reports": state["reports"],
            "historical_strict_only": HISTORICAL_STRICT_ONLY,
        }
    except Exception as exc:
        state.update(status="FAILED", updated_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(state_path, state)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--p0b-output-root", type=Path, required=True)
    parser.add_argument("--historical-calibration-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Lightweight JSON/Markdown/CSV destination (default: OUTPUT/lightweight_reports).",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = args.report_dir or (args.output_dir / "lightweight_reports")
    try:
        result = run_measurement(
            repo_root=args.repo_root,
            p0b_output_root=args.p0b_output_root,
            historical_calibration_manifest=args.historical_calibration_manifest,
            breparg_root=args.breparg_root,
            output_dir=args.output_dir,
            report_dir=report_dir,
            python=args.python,
            device=args.device,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except (EvidenceError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
