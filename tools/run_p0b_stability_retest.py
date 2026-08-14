"""Launch and validate the fail-closed P0-B VQ/bypass stability retest.

Formal runs are intentionally immutable: learned VQ and continuous bypass,
seeds 3 and 4, 60k/12k patches, batch 128, 100 epochs, and LR 3e-4.  Each
arm/seed is a separate ``train.py`` process with its own output directory and
rolling checkpoint.  Repeating the same command resumes an interrupted task;
changing any signed input fails closed.
"""

from __future__ import annotations

import argparse
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


SCHEMA = "p0b-stability-retest-v1"
FORMAL_ARMS = ("vq_4096_64d_random", "continuous_bypass_64d")
FORMAL_SEEDS = (3, 4)
FORMAL_TRAIN_CAP = 60_000
FORMAL_VAL_CAP = 12_000
FORMAL_BATCH_SIZE = 128
FORMAL_EPOCHS = 100
FORMAL_LEARNING_RATE = "3e-4"
FORMAL_GRAD_CLIP = "1.0"
ALLOWED_FORMAL_PRECISIONS = ("fp32", "bf16")
MAX_SMOKE_TRAIN_CAP = 2_048
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
TRUE_FIELDS = ("gradients_finite", "training_state_finite", "grad_clip_active")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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
    precision: str = "fp32"
    smoke: bool = False

    def __post_init__(self) -> None:
        for field in ("repo_root", "protocol_dir", "breparg_root", "output_root", "python"):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "learning_rate", str(self.learning_rate))
        object.__setattr__(self, "precision", str(self.precision).lower())
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be non-empty and unique")
        if set(self.arms) - set(FORMAL_ARMS):
            raise ValueError(f"P0-B only supports arms {FORMAL_ARMS}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(value <= 0 for value in (self.train_cap, self.val_cap, self.batch_size, self.epochs)):
            raise ValueError("caps, batch size, and epochs must be positive")
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0:
            raise ValueError("learning rate must be positive and finite")
        if self.precision not in ALLOWED_FORMAL_PRECISIONS:
            raise ValueError(f"precision must be one of {ALLOWED_FORMAL_PRECISIONS}")
        if not self.smoke:
            actual = (
                self.arms,
                self.seeds,
                self.train_cap,
                self.val_cap,
                self.batch_size,
                self.epochs,
                self.learning_rate,
            )
            expected = (
                FORMAL_ARMS,
                FORMAL_SEEDS,
                FORMAL_TRAIN_CAP,
                FORMAL_VAL_CAP,
                FORMAL_BATCH_SIZE,
                FORMAL_EPOCHS,
                FORMAL_LEARNING_RATE,
            )
            if actual != expected:
                raise ValueError(
                    "formal P0-B protocol is immutable; use --smoke for bounded overrides"
                )
        elif (
            self.train_cap > MAX_SMOKE_TRAIN_CAP
            or self.val_cap > MAX_SMOKE_VAL_CAP
            or self.batch_size > MAX_SMOKE_BATCH_SIZE
            or self.epochs > MAX_SMOKE_EPOCHS
        ):
            raise ValueError(
                "smoke overrides exceed bounded limits: "
                f"train<={MAX_SMOKE_TRAIN_CAP}, val<={MAX_SMOKE_VAL_CAP}, "
                f"batch<={MAX_SMOKE_BATCH_SIZE}, epochs<={MAX_SMOKE_EPOCHS}"
            )

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in ("repo_root", "protocol_dir", "breparg_root", "output_root", "python"):
            payload[field] = str(payload[field])
        payload["arms"] = list(self.arms)
        payload["seeds"] = list(self.seeds)
        return payload


def required_inputs(config: RunConfig) -> tuple[Path, ...]:
    return (
        config.python,
        config.repo_root / "breparg_improvements" / "train.py",
        config.repo_root / "breparg_improvements" / "training_stability.py",
        config.protocol_dir / "protocol_summary.json",
        config.protocol_dir / "split.pkl",
        config.breparg_root / "quantise.py",
    )


def verify_inputs(config: RunConfig) -> None:
    missing = [str(path) for path in required_inputs(config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("P0-B inputs missing: " + ", ".join(missing))
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


def task_root(config: RunConfig, arm: str, seed: int) -> Path:
    return config.output_root / "tasks" / arm / f"seed{seed}"


def task_signature_payload(config: RunConfig, arm: str, seed: int) -> dict[str, Any]:
    train_source = config.repo_root / "breparg_improvements" / "train.py"
    stability_source = config.repo_root / "breparg_improvements" / "training_stability.py"
    return {
        "schema": SCHEMA,
        "arm": arm,
        "seed": seed,
        "train_cap": config.train_cap,
        "val_cap": config.val_cap,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "precision": config.precision,
        "gradient_clip": FORMAL_GRAD_CLIP,
        "strict_nonfinite_fuse": True,
        "scheduler": {
            "kind": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 8,
            "threshold": 1e-5,
            "threshold_mode": "abs",
            "min_lr": 1e-6,
        },
        "protocol_summary_sha256": sha256_file(config.protocol_dir / "protocol_summary.json"),
        "split_pickle_sha256": sha256_file(config.protocol_dir / "split.pkl"),
        "train_source_sha256": sha256_file(train_source),
        "training_stability_source_sha256": sha256_file(stability_source),
        "breparg_quantise_sha256": sha256_file(config.breparg_root / "quantise.py"),
    }


def build_task(config: RunConfig, arm: str, seed: int) -> dict[str, Any]:
    root = task_root(config, arm, seed)
    signature_payload = task_signature_payload(config, arm, seed)
    signature = canonical_signature(signature_payload)
    command = [
        str(config.python),
        str(config.repo_root / "breparg_improvements" / "train.py"),
        "--stage",
        "vqsweep",
    ]
    environment = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(config.repo_root).replace("\\", "/"),
        "NS_OUTBASE": str(config.output_root),
        # An absolute NS_OUT keeps train.py's heavy output and its lightweight
        # evidence mirror inside the same isolated task directory.
        "NS_OUT": str(root),
        "NS_PROTOCOL_DIR": str(config.protocol_dir),
        "NS_PROTOCOL_V2": "1",
        "NS_VQ_AUTO_RESUME": "1",
        "NS_VQ_BALANCE_BY_PARENT": "1",
        "NS_VQ_BS": str(config.batch_size),
        "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1",
        "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1",
        "NS_VQ_DEDUP_BEFORE_CAP": "1",
        "NS_VQ_EPOCHS": str(config.epochs),
        "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_EXPERIMENT_SIGNATURE": signature,
        "NS_VQ_GRAD_CLIP": FORMAL_GRAD_CLIP,
        "NS_VQ_LR": config.learning_rate,
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "1",
        "NS_VQ_MIN_DELTA": "1e-5",
        "NS_VQ_MIN_EPOCHS": str(config.epochs),
        "NS_VQ_MIN_PARENT_COVERAGE": "0.9",
        "NS_VQ_PATIENCE": str(config.epochs),
        "NS_VQ_PLATEAU_METRIC": "curved_parent_mse",
        "NS_VQ_PRECISION": config.precision,
        "NS_VQ_ROLLING_CHECKPOINT": str(root / f"{arm}_rolling.pt"),
        "NS_VQ_SAMPLES": str(config.train_cap),
        "NS_VQ_SAVE_FINAL": "1",
        "NS_VQ_STRICT_NONFINITE": "1",
        "NS_VQ_SCHEDULER_FACTOR": "0.5",
        "NS_VQ_SCHEDULER_MIN_LR": "1e-6",
        "NS_VQ_SCHEDULER_PATIENCE": "8",
        "NS_VQ_SCHEDULER_THRESHOLD": "1e-5",
        "NS_VQ_SWEEP_ARMS": arm,
        "NS_VQ_SWEEP_EPOCHS": str(config.epochs),
        "NS_VQ_SWEEP_TRAIN_CAP": str(config.train_cap),
        "NS_VQ_TB_LOG_DIR": str(root / "tensorboard"),
        "NS_VQ_VAL_SAMPLES": str(config.val_cap),
        "NS_DISABLE_AMP_VQVAE": "1" if config.precision == "fp32" else "0",
    }
    return {
        "arm": arm,
        "seed": seed,
        "task_id": f"{arm}:seed{seed}",
        "task_root": str(root),
        "signature": signature,
        "signature_payload": signature_payload,
        "command": command,
        "environment": environment,
        "history": str(root / f"{arm}_history.json"),
        "sweep": str(root / "vqvae_hp_sweep.json"),
        "rolling_checkpoint": str(root / f"{arm}_rolling.pt"),
        "best_checkpoint": str(root / f"{arm}_best.pt"),
        "final_checkpoint": str(root / f"{arm}_final.pt"),
        "manifest": str(root / "task_manifest.json"),
        "status": "PENDING",
        "attempts": [],
    }


def build_state(config: RunConfig) -> dict[str, Any]:
    tasks = [build_task(config, arm, seed) for arm in config.arms for seed in config.seeds]
    config_payload = config.public_dict()
    return {
        "schema": SCHEMA,
        "status": "PENDING",
        "mode": "SMOKE" if config.smoke else "FORMAL",
        "formal_result_eligible": not config.smoke,
        "created_at": now(),
        "updated_at": now(),
        "configuration": config_payload,
        "configuration_signature": canonical_signature(config_payload),
        "active_task": None,
        "tasks": tasks,
    }


def validate_task(task: Mapping[str, Any], *, formal: bool) -> dict[str, Any]:
    reasons: list[str] = []
    history_path = Path(str(task["history"]))
    sweep_path = Path(str(task["sweep"]))
    expected_epochs = int(task["signature_payload"]["epochs"])
    expected_signature = str(task["signature"])
    if canonical_signature(task["signature_payload"]) != expected_signature:
        reasons.append("task signature payload mismatch")
    records: list[Any] = []
    history_payload: dict[str, Any] = {}
    if not history_path.is_file():
        reasons.append("history missing")
    else:
        try:
            history_payload = read_json(history_path)
            records = history_payload.get("history") or []
            if not isinstance(records, list):
                raise ValueError("history is not a list")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"history unreadable: {type(exc).__name__}")
            records = []
    epochs = [row.get("epoch") for row in records if isinstance(row, dict)]
    if epochs != list(range(expected_epochs)):
        reasons.append(f"epoch coverage must be 0..{expected_epochs - 1}, observed {epochs[:3]}..{epochs[-3:] if epochs else []}")
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            reasons.append(f"epoch row {index} is not an object")
            continue
        epoch = row.get("epoch", index)
        for field in ZERO_FIELDS:
            if row.get(field) != 0:
                reasons.append(f"epoch {epoch}: {field} must be integer zero")
        for field in TRUE_FIELDS:
            if row.get(field) is not True:
                reasons.append(f"epoch {epoch}: {field} must be true")
        if row.get("experiment_signature") != expected_signature:
            reasons.append(f"epoch {epoch}: experiment signature mismatch")
        grad_norm = row.get("preclip_grad_norm")
        if not isinstance(grad_norm, (int, float)) or not math.isfinite(float(grad_norm)):
            reasons.append(f"epoch {epoch}: preclip_grad_norm must be finite")
        train_batches = row.get("train_batches")
        if type(train_batches) is not int or train_batches <= 0:
            reasons.append(f"epoch {epoch}: train_batches must be positive")
        elif row.get("finite_train_batches") != train_batches:
            reasons.append(f"epoch {epoch}: not all train batches are finite")
        val_batches = row.get("val_batches")
        if type(val_batches) is not int or val_batches <= 0:
            reasons.append(f"epoch {epoch}: val_batches must be positive")
        elif row.get("finite_val_batches") != val_batches:
            reasons.append(f"epoch {epoch}: not all validation batches are finite")
    config_record = history_payload.get("config") or {}
    if config_record.get("experiment_signature") != expected_signature:
        reasons.append("history config experiment signature mismatch")
    observed_precision = config_record.get("precision")
    if isinstance(observed_precision, dict):
        observed_precision = observed_precision.get("name")
    if observed_precision != task["signature_payload"]["precision"]:
        reasons.append("history precision mismatch")
    if config_record.get("strict_nonfinite") is not True:
        reasons.append("history strict_nonfinite must be true")
    if config_record.get("grad_clip_norm") != float(FORMAL_GRAD_CLIP):
        reasons.append("history grad_clip_norm mismatch")
    scheduler = config_record.get("scheduler") or {}
    expected_scheduler = task["signature_payload"]["scheduler"]
    for field in ("factor", "patience", "threshold", "threshold_mode", "min_lr"):
        if scheduler.get(field) != expected_scheduler[field]:
            reasons.append(f"history scheduler {field} mismatch")
    if not sweep_path.is_file():
        reasons.append("sweep summary missing")
    else:
        try:
            sweep = read_json(sweep_path)
            rows = sweep.get("mse_ranking") or []
            by_arm = {row.get("name"): row for row in rows if isinstance(row, dict)}
            if set(by_arm) != {task["arm"]}:
                reasons.append("sweep arm set mismatch")
            row = by_arm.get(task["arm"], {})
            if row.get("epochs_ran") != expected_epochs:
                reasons.append("sweep epochs_ran mismatch")
            if row.get("final_checkpoint_epoch") != expected_epochs - 1:
                reasons.append("sweep final checkpoint epoch mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"sweep unreadable: {type(exc).__name__}")
    # stage_vqsweep reads the selected checkpoint before it can emit a
    # completed sweep summary, so all three artifacts are part of operational
    # completion.  Bypass promotion eligibility remains irrelevant here.
    for field in ("best_checkpoint", "final_checkpoint", "rolling_checkpoint"):
        if not Path(str(task[field])).is_file():
            reasons.append(f"{field} missing")
    if formal and expected_epochs != FORMAL_EPOCHS:
        reasons.append("formal task does not request 100 epochs")
    return {
        "task_id": task["task_id"],
        "valid": not reasons,
        "formal": formal,
        "epochs_observed": len(records),
        "last_epoch": epochs[-1] if epochs else None,
        "reasons": reasons,
    }


def refresh_state(state: dict[str, Any]) -> dict[str, Any]:
    formal = state.get("mode") == "FORMAL"
    for task in state.get("tasks", []):
        validation = validate_task(task, formal=formal)
        task["validation"] = validation
        if validation["valid"]:
            task["status"] = "COMPLETED"
        elif task.get("status") == "RUNNING":
            # A separate ``status`` process may inspect a live blocking
            # launcher.  Keep RUNNING here; repeating ``run`` still retries an
            # invalid task and train.py decides whether its rolling state can
            # resume under the signed protocol.
            task["status"] = "RUNNING"
        elif task.get("status") not in {"FAILED", "PLANNED"}:
            task["status"] = "INCOMPLETE"
    statuses = [task.get("status") for task in state.get("tasks", [])]
    if statuses and all(status == "COMPLETED" for status in statuses):
        state["status"] = "COMPLETED"
    elif any(status == "FAILED" for status in statuses):
        state["status"] = "FAILED"
    elif any(status == "RUNNING" for status in statuses):
        state["status"] = "RUNNING"
    else:
        state["status"] = "INCOMPLETE"
    running_tasks = [
        task.get("task_id") for task in state.get("tasks", [])
        if task.get("status") == "RUNNING"
    ]
    state["active_task"] = running_tasks[0] if len(running_tasks) == 1 else None
    state["updated_at"] = now()
    return state


def ensure_state(config: RunConfig) -> tuple[Path, dict[str, Any]]:
    state_path = config.output_root / "p0b_state.json"
    expected = build_state(config)
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("schema") != SCHEMA:
            raise ValueError("existing output root has an unsupported P0-B state schema")
        if state.get("configuration_signature") != expected["configuration_signature"]:
            raise ValueError("existing output root configuration signature mismatch")
        expected_tasks = {task["task_id"]: task for task in expected["tasks"]}
        actual_tasks = {task.get("task_id"): task for task in state.get("tasks", [])}
        if set(actual_tasks) != set(expected_tasks):
            raise ValueError("existing output root task matrix mismatch")
        for task_id, task in actual_tasks.items():
            if task.get("signature") != expected_tasks[task_id]["signature"]:
                raise ValueError(f"existing task signature mismatch: {task_id}")
        return state_path, state
    if config.output_root.exists() and any(config.output_root.iterdir()):
        raise ValueError("new P0-B output root is non-empty and has no matching state")
    config.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(state_path, expected)
    return state_path, expected


def clean_environment(task: Mapping[str, Any]) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NS_")}
    environment.update({str(key): str(value) for key, value in task["environment"].items()})
    return environment


def write_task_manifest(task: Mapping[str, Any]) -> None:
    path = Path(str(task["manifest"]))
    if path.is_file():
        existing = read_json(path)
        if existing.get("signature") != task["signature"]:
            raise ValueError(f"task manifest signature mismatch: {task['task_id']}")
        return
    atomic_json(path, {
        "schema": SCHEMA,
        "task_id": task["task_id"],
        "signature": task["signature"],
        "signature_payload": task["signature_payload"],
        "command": task["command"],
        "environment": task["environment"],
        "created_at": now(),
    })


def run_cohort(config: RunConfig, *, dry_run: bool = False) -> dict[str, Any]:
    verify_inputs(config)
    if dry_run:
        plan = build_state(config)
        for task in plan["tasks"]:
            task["status"] = "PLANNED"
        plan["status"] = "DRY_RUN"
        plan["note"] = "No directories, checkpoints, or training processes were created."
        return plan
    state_path, state = ensure_state(config)
    refresh_state(state)
    for task in state["tasks"]:
        if task.get("status") == "FAILED":
            task["status"] = "INCOMPLETE"
    atomic_json(state_path, state)
    for task in state["tasks"]:
        if task["validation"]["valid"]:
            continue
        write_task_manifest(task)
        attempt_number = len(task.get("attempts", [])) + 1
        log_dir = Path(task["task_root"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"attempt_{attempt_number:03d}.stdout.log"
        stderr_path = log_dir / f"attempt_{attempt_number:03d}.stderr.log"
        attempt = {
            "attempt": attempt_number,
            "started_at": now(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "auto_resume": True,
        }
        task.setdefault("attempts", []).append(attempt)
        task["status"] = "RUNNING"
        state.update(status="RUNNING", active_task=task["task_id"], updated_at=now())
        atomic_json(state_path, state)
        with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stderr:
            completed = subprocess.run(
                task["command"], cwd=config.repo_root, env=clean_environment(task),
                stdout=stdout, stderr=stderr, check=False,
            )
        attempt.update(returncode=completed.returncode, finished_at=now())
        validation = validate_task(task, formal=not config.smoke)
        task["validation"] = validation
        task["status"] = "COMPLETED" if completed.returncode == 0 and validation["valid"] else "FAILED"
        state.update(active_task=None, updated_at=now())
        atomic_json(state_path, state)
        if task["status"] != "COMPLETED":
            state["status"] = "FAILED"
            atomic_json(state_path, state)
            return state
    refresh_state(state)
    atomic_json(state_path, state)
    return state


def load_and_refresh(output_root: Path) -> tuple[Path, dict[str, Any]]:
    state_path = Path(output_root).resolve() / "p0b_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"P0-B state missing: {state_path}")
    state = read_json(state_path)
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported P0-B state schema")
    refresh_state(state)
    atomic_json(state_path, state)
    return state_path, state


def validation_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    formal = state.get("mode") == "FORMAL"
    reasons: list[str] = []
    tasks = state.get("tasks") or []
    expected_ids = {f"{arm}:seed{seed}" for arm in FORMAL_ARMS for seed in FORMAL_SEEDS}
    observed_ids = {task.get("task_id") for task in tasks}
    if formal and observed_ids != expected_ids:
        reasons.append("formal task matrix mismatch")
    validations = [validate_task(task, formal=formal) for task in tasks]
    for result in validations:
        reasons.extend(f"{result['task_id']}: {reason}" for reason in result["reasons"])
    return {
        "schema": SCHEMA,
        "formal": formal,
        "formal_result_eligible": formal and not reasons,
        "valid": not reasons,
        "validated_at": now(),
        "tasks": validations,
        "reasons": reasons,
    }


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--precision", choices=ALLOWED_FORMAL_PRECISIONS, default="fp32")
    parser.add_argument("--smoke", action="store_true", help="Allow bounded non-formal overrides")
    parser.add_argument("--arms", default=",".join(FORMAL_ARMS))
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--train-cap", type=int, default=FORMAL_TRAIN_CAP)
    parser.add_argument("--val-cap", type=int, default=FORMAL_VAL_CAP)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--learning-rate", default=FORMAL_LEARNING_RATE)
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        repo_root=args.repo_root,
        protocol_dir=args.protocol_dir,
        breparg_root=args.breparg_root,
        output_root=args.output_root,
        python=args.python,
        arms=parse_csv(args.arms),
        seeds=tuple(int(item) for item in parse_csv(args.seeds)),
        train_cap=args.train_cap,
        val_cap=args.val_cap,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        precision=args.precision,
        smoke=args.smoke,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run or automatically resume the formal matrix")
    add_run_arguments(run_parser)
    probe_parser = subparsers.add_parser("probe", help="Run a bounded one-epoch forward/backward health probe")
    add_run_arguments(probe_parser)
    probe_parser.set_defaults(smoke=True, train_cap=128, val_cap=128, batch_size=8, epochs=1, seeds="3")
    for name in ("status", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"run", "probe"}:
        config = config_from_args(args)
        result = run_cohort(config, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0 if args.dry_run or result.get("status") == "COMPLETED" else 1
    _, state = load_and_refresh(args.output_root)
    if args.command == "status":
        print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    summary = validation_summary(state)
    atomic_json(Path(args.output_root).resolve() / "p0b_validation.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
