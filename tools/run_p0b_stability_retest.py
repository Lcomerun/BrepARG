"""Launch and validate the fail-closed P0-B VQ/bypass stability retest.

Formal runs are intentionally immutable: learned VQ and continuous bypass,
seeds 3 and 4, 60k/12k patches, batch 128, 100 epochs, and LR 3e-4.  Each
arm/seed is a separate ``train.py`` process with its own output directory and
rolling checkpoint.  Repeating the same command resumes an interrupted task;
changing any signed input fails closed.
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCHEMA = "p0b-stability-retest-v1"
WRITER_LOCK_SCHEMA = "p0b-output-writer-lock-v1"
WRITER_LOCK_NAME = ".p0b_writer.lock"
ROLLING_CHECKPOINT_SCHEMA = "vq_training_state_v1"
FORMAL_ARMS = ("vq_4096_64d_random", "continuous_bypass_64d")
FORMAL_SEEDS = (3, 4)
FORMAL_TRAIN_CAP = 60_000
FORMAL_VAL_CAP = 12_000
FORMAL_BATCH_SIZE = 128
FORMAL_EPOCHS = 100
FORMAL_LEARNING_RATE = "3e-4"
FORMAL_GRAD_CLIP = "1.0"
FORMAL_MIN_PARENT_COVERAGE = 0.9
FORMAL_PROTOCOL_SHA256 = (
    "6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3"
)
FORMAL_SPLIT_PICKLE_SHA256 = (
    "6ff0a0c3ee6a04ee056fa1ab982eb436a9f59d3d21f21f17babf34e6dc701d29"
)
REQUIRED_PARENT_OVERLAPS = ("train__val", "train__test", "val__test")
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


def protocol_binding(config: "RunConfig") -> dict[str, Any]:
    """Read the protocol identity embedded in task and checkpoint signatures."""
    summary_path = config.protocol_dir / "protocol_summary.json"
    split_path = config.protocol_dir / "split.pkl"
    summary = read_json(summary_path)
    overlaps = summary.get("parent_overlap_counts")
    return {
        "status": summary.get("status"),
        "protocol_sha256": summary.get("protocol_sha256"),
        "split_pickle_sha256": sha256_file(split_path),
        "summary_split_pickle_sha256": summary.get("split_pickle_sha256"),
        "parent_overlap_counts": dict(overlaps) if isinstance(overlaps, Mapping) else overlaps,
        "protocol_summary_sha256": sha256_file(summary_path),
    }


def verify_protocol(config: "RunConfig") -> dict[str, Any]:
    """Fail closed unless the protocol is internally valid and formally frozen."""
    binding = protocol_binding(config)
    reasons: list[str] = []
    if binding["status"] != "VERIFIED":
        reasons.append("status must be VERIFIED")
    overlaps = binding["parent_overlap_counts"]
    if not isinstance(overlaps, Mapping) or set(overlaps) != set(REQUIRED_PARENT_OVERLAPS):
        reasons.append("parent_overlap_counts must contain exactly the three split pairs")
    elif any(type(overlaps[name]) is not int or overlaps[name] != 0 for name in REQUIRED_PARENT_OVERLAPS):
        reasons.append("all parent overlap counts must be JSON integer zero")
    if binding["summary_split_pickle_sha256"] != binding["split_pickle_sha256"]:
        reasons.append("split.pkl SHA-256 does not match protocol_summary.json")
    if config.smoke:
        if not isinstance(binding["protocol_sha256"], str) or not binding["protocol_sha256"]:
            reasons.append("protocol_sha256 is missing")
    else:
        if binding["protocol_sha256"] != FORMAL_PROTOCOL_SHA256:
            reasons.append("protocol_sha256 is not the frozen Protocol V5 hash")
        if binding["split_pickle_sha256"] != FORMAL_SPLIT_PICKLE_SHA256:
            reasons.append("split_pickle_sha256 is not the frozen Protocol V5 split")
    if reasons:
        mode = "formal Protocol V5" if not config.smoke else "probe protocol"
        raise ValueError(f"{mode} verification failed: " + "; ".join(reasons))
    return binding


def _process_creation_identity(pid: int) -> dict[str, Any] | None:
    """Return a PID-reuse-resistant process identity when the OS exposes one."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            ]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                ticks = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
                return {"kind": "windows_filetime_100ns", "value": str(ticks)}
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close_paren = stat_text.rfind(")")
        fields = stat_text[close_paren + 2 :].split()
        start_ticks = fields[19]
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        boot_id = boot_id_path.read_text(encoding="ascii").strip() if boot_id_path.is_file() else None
        return {"kind": "proc_start_ticks", "value": start_ticks, "boot_id": boot_id}
    except (IndexError, OSError, ValueError):
        return None


def _read_lock_payload(handle: Any) -> dict[str, Any] | None:
    try:
        handle.seek(1)
        raw = handle.read()
        if not raw:
            return None
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_lock_payload(handle: Any, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    handle.seek(0)
    handle.write(b"\0" + encoded + b"\n")
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


def _acquire_os_lock(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def output_root_writer_lock(
    output_root: Path, *, command: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Serialize output writers; OS locks are released even after process death."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / WRITER_LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    payload: dict[str, Any] = {}
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            _acquire_os_lock(handle)
            acquired = True
        except OSError as exc:
            owner = _read_lock_payload(handle) or {"metadata": "unreadable"}
            raise RuntimeError(
                "P0-B output root already has an active writer; "
                f"lock={lock_path} owner={json.dumps(owner, sort_keys=True, ensure_ascii=True)}"
            ) from exc

        previous = _read_lock_payload(handle)
        argv = [str(item) for item in (command or (sys.executable, *sys.argv))]
        owner_identity = _process_creation_identity(os.getpid())
        previous_owner = previous.get("owner") if isinstance(previous, Mapping) else None
        previous_identity = (
            _process_creation_identity(previous_owner.get("pid"))
            if isinstance(previous_owner, Mapping) and type(previous_owner.get("pid")) is int
            else None
        )
        previous_unreleased = bool(previous) and not previous.get("released_at")
        # Acquiring the kernel lock proves that no writer still owns it. An
        # unreleased record is therefore stale even if Windows temporarily
        # keeps a terminated process object queryable after exit.
        stale_recovered = previous_unreleased
        payload = {
            "schema": WRITER_LOCK_SCHEMA,
            "lock_path": str(lock_path),
            "acquired_at": now(),
            "owner": {
                "pid": os.getpid(),
                "process_creation_identity": owner_identity,
            },
            "command": argv,
            "command_line": subprocess.list2cmdline(argv),
            "stale_lock_recovered": stale_recovered,
            "unreleased_lock_recovered": previous_unreleased,
            "previous_owner_identity_now": previous_identity,
            "previous_owner_identity_matches": bool(
                isinstance(previous_owner, Mapping)
                and previous_owner.get("process_creation_identity") == previous_identity
            ),
            "previous_lock": previous,
        }
        _write_lock_payload(handle, payload)
        yield payload
    finally:
        if acquired:
            payload["released_at"] = now()
            try:
                _write_lock_payload(handle, payload)
            finally:
                _release_os_lock(handle)
        handle.close()


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
        config.repo_root / "breparg_improvements" / "vqvae_sampling.py",
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
    verify_protocol(config)


def task_root(config: RunConfig, arm: str, seed: int) -> Path:
    return config.output_root / "tasks" / arm / f"seed{seed}"


def task_signature_payload(config: RunConfig, arm: str, seed: int) -> dict[str, Any]:
    train_source = config.repo_root / "breparg_improvements" / "train.py"
    stability_source = config.repo_root / "breparg_improvements" / "training_stability.py"
    sampling_source = config.repo_root / "breparg_improvements" / "vqvae_sampling.py"
    protocol = protocol_binding(config)
    formal_sampling = not config.smoke
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
        "sampling": {
            "balance_by_parent": formal_sampling,
            "deduplicate_before_cap": formal_sampling,
            "require_exact_caps": formal_sampling,
            "min_parent_coverage": (
                0.0 if config.smoke else FORMAL_MIN_PARENT_COVERAGE
            ),
            "curved_fraction": 0.0,
            "complex_fraction": 0.0,
        },
        "scheduler": {
            "kind": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 8,
            "threshold": 1e-5,
            "threshold_mode": "abs",
            "min_lr": 1e-6,
        },
        "protocol_status": protocol["status"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "split_pickle_sha256": protocol["split_pickle_sha256"],
        "parent_overlap_counts": protocol["parent_overlap_counts"],
        "train_source_sha256": sha256_file(train_source),
        "training_stability_source_sha256": sha256_file(stability_source),
        "vqvae_sampling_source_sha256": sha256_file(sampling_source),
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
        "NS_VQ_BALANCE_BY_PARENT": (
            "1" if signature_payload["sampling"]["balance_by_parent"] else "0"
        ),
        "NS_VQ_BS": str(config.batch_size),
        "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1",
        "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1",
        "NS_VQ_DEDUP_BEFORE_CAP": (
            "1" if signature_payload["sampling"]["deduplicate_before_cap"] else "0"
        ),
        "NS_VQ_EPOCHS": str(config.epochs),
        "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_EXPERIMENT_SIGNATURE": signature,
        "NS_VQ_GRAD_CLIP": FORMAL_GRAD_CLIP,
        "NS_VQ_LR": config.learning_rate,
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "1",
        "NS_VQ_MIN_DELTA": "1e-5",
        "NS_VQ_MIN_EPOCHS": str(config.epochs),
        "NS_VQ_MIN_PARENT_COVERAGE": str(
            signature_payload["sampling"]["min_parent_coverage"]
        ),
        "NS_VQ_PATIENCE": str(config.epochs),
        "NS_VQ_PLATEAU_METRIC": "curved_parent_mse",
        "NS_VQ_PRECISION": config.precision,
        "NS_VQ_REQUIRE_EXACT_CAPS": (
            "1" if signature_payload["sampling"]["require_exact_caps"] else "0"
        ),
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


def _checkpoint_protocol(task: Mapping[str, Any]) -> dict[str, Any]:
    signature = task["signature_payload"]
    return {
        "protocol_sha256": signature.get("protocol_sha256"),
        "split_pickle_sha256": signature.get("split_pickle_sha256"),
        "parent_overlap_counts": signature.get("parent_overlap_counts"),
    }


def _arm_checkpoint_contract(arm: str) -> dict[str, Any]:
    if arm == "vq_4096_64d_random":
        return {
            "kind": "learned_vq",
            "codebook_size": 4096,
            "embedding_dim": 64,
            "anchor": "random",
        }
    if arm == "continuous_bypass_64d":
        return {
            "kind": "continuous_bypass",
            "embedding_dim": 64,
        }
    raise ValueError(f"unsupported P0-B arm: {arm}")


def _arm_codebook_size(arm: str) -> int:
    return 4096 if arm == "vq_4096_64d_random" else 1


def _validate_quantizer(
    quantizer: Any, task: Mapping[str, Any], *, prefix: str, reasons: list[str]
) -> None:
    if not isinstance(quantizer, Mapping):
        reasons.append(f"{prefix} quantizer metadata missing")
        return
    expected = _arm_checkpoint_contract(str(task["arm"]))
    for field, value in expected.items():
        if quantizer.get(field) != value:
            reasons.append(f"{prefix} quantizer {field} mismatch")


def _validate_protocol_context(
    observed: Any, task: Mapping[str, Any], *, prefix: str, reasons: list[str]
) -> None:
    if not isinstance(observed, Mapping):
        reasons.append(f"{prefix} protocol context missing")
        return
    expected = _checkpoint_protocol(task)
    for field in ("protocol_sha256", "split_pickle_sha256"):
        if observed.get(field) != expected[field]:
            reasons.append(f"{prefix} {field} mismatch")
    observed_overlaps = observed.get("parent_overlap_counts")
    if observed_overlaps is not None and observed_overlaps != expected["parent_overlap_counts"]:
        reasons.append(f"{prefix} parent_overlap_counts mismatch")


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_inventory(
    observed: Any,
    task: Mapping[str, Any],
    *,
    prefix: str,
    reasons: list[str],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    start_reasons = len(reasons)
    if not isinstance(observed, Mapping) or set(observed) != {"train", "val"}:
        reasons.append(f"{prefix} inventory must contain exactly train and val")
        return None
    normalized: dict[str, Any] = {}
    expected_counts = {
        "train": task["signature_payload"]["train_cap"],
        "val": task["signature_payload"]["val_cap"],
    }
    exact_counts_required = bool(
        task["signature_payload"].get("sampling", {}).get("require_exact_caps")
    )
    for split_name in ("train", "val"):
        item = observed.get(split_name)
        if not isinstance(item, Mapping):
            reasons.append(f"{prefix} {split_name} inventory missing")
            continue
        normalized[split_name] = dict(item)
        if item.get("schema") != "vq-exact-hash-inventory-v1":
            reasons.append(f"{prefix} {split_name} inventory schema mismatch")
        count = item.get("count")
        cap = expected_counts[split_name]
        if type(count) is not int or count <= 0 or count > cap:
            reasons.append(f"{prefix} {split_name} inventory count invalid")
        elif exact_counts_required and count != cap:
            reasons.append(f"{prefix} {split_name} inventory count mismatch")
        for field in ("ordered_sha256", "sorted_sha256"):
            if not _valid_sha256(item.get(field)):
                reasons.append(f"{prefix} {split_name} inventory {field} invalid")
    if expected is not None and normalized != dict(expected):
        reasons.append(f"{prefix} inventory binding mismatch")
    return normalized if len(reasons) == start_reasons else None


def _validate_run_manifest(
    manifest: Any,
    task: Mapping[str, Any],
    *,
    prefix: str,
    reasons: list[str],
    expected_inventory: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(manifest, Mapping):
        reasons.append(f"{prefix} run_manifest missing")
        return
    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        reasons.append(f"{prefix} run_manifest experiment missing")
        return
    expected = task["signature_payload"]
    checks = {
        "seed": task["seed"],
        "train_cap": expected["train_cap"],
        "val_cap": expected["val_cap"],
        "epochs": expected["epochs"],
        "batch_size": expected["batch_size"],
    }
    for field, value in checks.items():
        if experiment.get(field) != value:
            reasons.append(f"{prefix} run_manifest {field} mismatch")
    arms = experiment.get("arms")
    if (
        not isinstance(arms, list)
        or len(arms) != 1
        or not isinstance(arms[0], Mapping)
        or arms[0].get("name") != task["arm"]
    ):
        reasons.append(f"{prefix} run_manifest arm mismatch")
    else:
        if arms[0].get("codebook") != _arm_codebook_size(str(task["arm"])):
            reasons.append(f"{prefix} run_manifest codebook mismatch")
        _validate_quantizer(
            arms[0].get("quantizer"), task, prefix=prefix, reasons=reasons
        )
    _validate_protocol_context(
        experiment.get("protocol"), task, prefix=f"{prefix} run_manifest", reasons=reasons
    )
    _validate_inventory(
        experiment.get("inventory"),
        task,
        prefix=f"{prefix} run_manifest",
        reasons=reasons,
        expected=expected_inventory,
    )


def _load_checkpoint(
    path: Path, *, label: str, reasons: list[str]
) -> Mapping[str, Any] | None:
    if not path.is_file():
        reasons.append(f"{label} missing")
        return None
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        reasons.append(f"{label} unreadable: {type(exc).__name__}")
        return None
    if not isinstance(payload, Mapping):
        reasons.append(f"{label} must contain a mapping")
        return None
    return payload


def _validate_model_state(
    payload: Mapping[str, Any], *, label: str, reasons: list[str]
) -> None:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        reasons.append(f"{label} model_state_dict must be a non-empty mapping")


def _validate_normal_checkpoint(
    path: Path,
    task: Mapping[str, Any],
    *,
    label: str,
    expected_epoch: int | None,
    terminal_epoch: int,
    expected_inventory: Mapping[str, Any] | None,
    reasons: list[str],
) -> None:
    payload = _load_checkpoint(path, label=label, reasons=reasons)
    if payload is None:
        return
    _validate_model_state(payload, label=label, reasons=reasons)
    checkpoint_epoch = payload.get("checkpoint_epoch")
    if type(checkpoint_epoch) is not int or not 0 <= checkpoint_epoch <= terminal_epoch:
        reasons.append(f"{label} checkpoint_epoch is invalid")
    elif expected_epoch is not None and checkpoint_epoch != expected_epoch:
        reasons.append(f"{label} checkpoint_epoch mismatch")
    if list(payload.get("fsq_levels") or []) != []:
        reasons.append(f"{label} fsq_levels must be empty for P0-B arms")
    _validate_quantizer(payload.get("quantizer"), task, prefix=label, reasons=reasons)
    context = payload.get("checkpoint_context")
    if not isinstance(context, Mapping):
        reasons.append(f"{label} checkpoint_context missing")
        return
    _validate_protocol_context(context, task, prefix=label, reasons=reasons)
    _validate_inventory(
        context.get("inventory"),
        task,
        prefix=label,
        reasons=reasons,
        expected=expected_inventory,
    )
    _validate_run_manifest(
        context.get("run_manifest"),
        task,
        prefix=label,
        reasons=reasons,
        expected_inventory=expected_inventory,
    )
    for coverage_name in ("train_parent_coverage", "val_parent_coverage"):
        coverage = context.get(coverage_name)
        if not isinstance(coverage, (int, float)) or not math.isfinite(float(coverage)):
            reasons.append(f"{label} {coverage_name} must be finite")


def _validate_signature_configuration(
    configuration: Any,
    task: Mapping[str, Any],
    *,
    label: str,
    reasons: list[str],
    expected_inventory: Mapping[str, Any] | None,
) -> None:
    if not isinstance(configuration, Mapping):
        reasons.append(f"{label} signature_configuration missing")
        return
    expected = task["signature_payload"]
    checks = {
        "seed": task["seed"],
        "train_cap": expected["train_cap"],
        "val_cap": expected["val_cap"],
        "epochs": expected["epochs"],
        "batch_size": expected["batch_size"],
        "precision": expected["precision"],
    }
    for field, value in checks.items():
        if configuration.get(field) != value:
            reasons.append(f"{label} signature_configuration {field} mismatch")
    try:
        observed_lr = float(configuration.get("lr"))
    except (TypeError, ValueError):
        observed_lr = math.nan
    if not math.isfinite(observed_lr) or observed_lr != float(expected["learning_rate"]):
        reasons.append(f"{label} signature_configuration lr mismatch")
    arm = configuration.get("arm")
    if not isinstance(arm, Mapping) or arm.get("name") != task["arm"]:
        reasons.append(f"{label} signature_configuration arm mismatch")
    else:
        if arm.get("codebook") != _arm_codebook_size(str(task["arm"])):
            reasons.append(f"{label} signature_configuration codebook mismatch")
        _validate_quantizer(arm.get("quantizer"), task, prefix=label, reasons=reasons)
    _validate_protocol_context(
        configuration.get("protocol"), task, prefix=label, reasons=reasons
    )
    _validate_inventory(
        configuration.get("inventory"),
        task,
        prefix=label,
        reasons=reasons,
        expected=expected_inventory,
    )


def _validate_rolling_checkpoint(
    path: Path,
    task: Mapping[str, Any],
    *,
    records: list[Any],
    terminal_epoch: int,
    expected_inventory: Mapping[str, Any] | None,
    reasons: list[str],
) -> None:
    label = "rolling_checkpoint"
    payload = _load_checkpoint(path, label=label, reasons=reasons)
    if payload is None:
        return
    required = {
        "checkpoint_schema",
        "experiment_signature",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "stop_state",
        "plateau_state",
        "history",
        "rng_state",
        "feature_pool_state",
        "finite_state_audit",
        "extra",
    }
    missing = sorted(required - set(payload))
    if missing:
        reasons.append(f"{label} incomplete: {', '.join(missing)}")
    if payload.get("checkpoint_schema") != ROLLING_CHECKPOINT_SCHEMA:
        reasons.append(f"{label} checkpoint_schema mismatch")
    if payload.get("experiment_signature") != task["signature"]:
        reasons.append(f"{label} experiment signature mismatch")
    if payload.get("epoch") != terminal_epoch:
        reasons.append(f"{label} terminal epoch mismatch")
    _validate_model_state(payload, label=label, reasons=reasons)
    if not isinstance(payload.get("optimizer_state_dict"), Mapping) or not payload.get(
        "optimizer_state_dict"
    ):
        reasons.append(f"{label} optimizer_state_dict must be complete")
    if not isinstance(payload.get("scheduler_state_dict"), Mapping) or not payload.get(
        "scheduler_state_dict"
    ):
        reasons.append(f"{label} scheduler_state_dict must be complete")
    if not isinstance(payload.get("stop_state"), Mapping):
        reasons.append(f"{label} stop_state missing")
    if not isinstance(payload.get("plateau_state"), Mapping):
        reasons.append(f"{label} plateau_state missing")
    embedded_history = payload.get("history")
    if not isinstance(embedded_history, list) or embedded_history != records:
        reasons.append(f"{label} embedded history mismatch")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Mapping) or not {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }.issubset(rng_state):
        reasons.append(f"{label} rng_state incomplete")
    if not isinstance(payload.get("feature_pool_state"), Mapping):
        reasons.append(f"{label} feature_pool_state missing")
    checkpoint_audit = payload.get("finite_state_audit")
    if not isinstance(checkpoint_audit, Mapping) or checkpoint_audit.get("status") != "finite":
        reasons.append(f"{label} finite_state_audit missing or non-finite")
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        reasons.append(f"{label} extra state missing")
        return
    _validate_signature_configuration(
        extra.get("signature_configuration"),
        task,
        label=label,
        reasons=reasons,
        expected_inventory=expected_inventory,
    )
    precision = extra.get("precision")
    if not isinstance(precision, Mapping) or precision.get("name") != task["signature_payload"][
        "precision"
    ]:
        reasons.append(f"{label} precision context mismatch")
    for field in ("selector_state", "meta"):
        if not isinstance(extra.get(field), Mapping):
            reasons.append(f"{label} extra {field} missing")


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
        if row.get("nonfinite_state_audits") != 0:
            reasons.append(f"epoch {epoch}: nonfinite_state_audits must be integer zero")
        if row.get("finite_state_audit_cadence") != "lifecycle_v1":
            reasons.append(f"epoch {epoch}: finite-state audit cadence mismatch")
        if row.get("full_state_audits") != 1:
            reasons.append(f"epoch {epoch}: exactly one full state audit is required")
        if row.get("per_batch_full_state_audits") != 0:
            reasons.append(f"epoch {epoch}: per-batch full state audits must be zero")
        state_audit = row.get("finite_state_audit")
        if not isinstance(state_audit, Mapping) or state_audit.get("status") != "finite":
            reasons.append(f"epoch {epoch}: finite-state audit evidence missing")
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
    expected_audit_cadence = {
        "policy": "lifecycle_v1",
        "startup_or_post_resume": True,
        "epoch_end_pre_save": True,
        "per_train_batch": False,
    }
    if config_record.get("finite_state_audit_cadence") != expected_audit_cadence:
        reasons.append("history finite-state audit cadence mismatch")
    scheduler = config_record.get("scheduler") or {}
    expected_scheduler = task["signature_payload"]["scheduler"]
    for field in ("factor", "patience", "threshold", "threshold_mode", "min_lr"):
        if scheduler.get(field) != expected_scheduler[field]:
            reasons.append(f"history scheduler {field} mismatch")
    sweep_row: Mapping[str, Any] = {}
    sweep_payload: Mapping[str, Any] = {}
    task_inventory: dict[str, Any] | None = None
    if not sweep_path.is_file():
        reasons.append("sweep summary missing")
    else:
        try:
            sweep_payload = read_json(sweep_path)
            rows = sweep_payload.get("mse_ranking") or []
            by_arm = {row.get("name"): row for row in rows if isinstance(row, dict)}
            if set(by_arm) != {task["arm"]}:
                reasons.append("sweep arm set mismatch")
            sweep_row = by_arm.get(task["arm"], {})
            if sweep_row.get("epochs_ran") != expected_epochs:
                reasons.append("sweep epochs_ran mismatch")
            if sweep_row.get("final_checkpoint_epoch") != expected_epochs - 1:
                reasons.append("sweep final checkpoint epoch mismatch")
            if sweep_row.get("experiment_signature") != expected_signature:
                reasons.append("sweep experiment signature mismatch")
            sweep_precision = sweep_row.get("precision")
            if isinstance(sweep_precision, Mapping):
                sweep_precision = sweep_precision.get("name")
            if sweep_precision != task["signature_payload"]["precision"]:
                reasons.append("sweep precision mismatch")
            task_inventory = _validate_inventory(
                sweep_row.get("inventory"),
                task,
                prefix="sweep",
                reasons=reasons,
            )
            _validate_run_manifest(
                sweep_payload.get("run_manifest"),
                task,
                prefix="sweep",
                reasons=reasons,
                expected_inventory=task_inventory,
            )
            train_selected = (sweep_row.get("train_sampling") or {}).get("selected")
            val_selected = (sweep_row.get("val_sampling") or {}).get("selected")
            if formal and train_selected != task["signature_payload"]["train_cap"]:
                reasons.append("sweep realized train patch count mismatch")
            if formal and val_selected != task["signature_payload"]["val_cap"]:
                reasons.append("sweep realized val patch count mismatch")
            if formal and (
                train_selected != FORMAL_TRAIN_CAP or val_selected != FORMAL_VAL_CAP
            ):
                reasons.append("formal realized patch counts must be exactly 60000/12000")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"sweep unreadable: {type(exc).__name__}")
    # stage_vqsweep reads the selected checkpoint before it can emit a completed
    # summary. Bypass promotion eligibility is irrelevant, but all artifacts
    # must be structurally complete and bound to this arm, seed, and protocol.
    best_epoch = sweep_row.get("checkpoint_epoch")
    if type(best_epoch) is not int:
        reasons.append("sweep checkpoint_epoch must be an integer")
        best_epoch = None
    _validate_normal_checkpoint(
        Path(str(task["best_checkpoint"])),
        task,
        label="best_checkpoint",
        expected_epoch=best_epoch,
        terminal_epoch=expected_epochs - 1,
        expected_inventory=task_inventory,
        reasons=reasons,
    )
    _validate_normal_checkpoint(
        Path(str(task["final_checkpoint"])),
        task,
        label="final_checkpoint",
        expected_epoch=expected_epochs - 1,
        terminal_epoch=expected_epochs - 1,
        expected_inventory=task_inventory,
        reasons=reasons,
    )
    _validate_rolling_checkpoint(
        Path(str(task["rolling_checkpoint"])),
        task,
        records=records,
        terminal_epoch=expected_epochs - 1,
        expected_inventory=task_inventory,
        reasons=reasons,
    )
    if formal and expected_epochs != FORMAL_EPOCHS:
        reasons.append("formal task does not request 100 epochs")
    return {
        "task_id": task["task_id"],
        "valid": not reasons,
        "formal": formal,
        "epochs_observed": len(records),
        "last_epoch": epochs[-1] if epochs else None,
        "inventory": task_inventory,
        "reasons": reasons,
    }


def _inventory_consistency(validations: Sequence[Mapping[str, Any]]) -> bool:
    inventory_signatures = {
        canonical_signature(result["inventory"])
        for result in validations
        if isinstance(result.get("inventory"), Mapping)
    }
    return bool(validations) and (
        len(inventory_signatures) == 1
        and all(isinstance(result.get("inventory"), Mapping) for result in validations)
    )


def refresh_state(state: dict[str, Any]) -> dict[str, Any]:
    formal = state.get("mode") == "FORMAL"
    validations = []
    for task in state.get("tasks", []):
        validation = validate_task(task, formal=formal)
        validations.append(validation)
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
        inventory_consistent = _inventory_consistency(validations)
        state["inventory_consistent"] = inventory_consistent
        state["status"] = (
            "COMPLETED" if not formal or inventory_consistent else "FAILED"
        )
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
    existing_entries = (
        [entry for entry in config.output_root.iterdir() if entry.name != WRITER_LOCK_NAME]
        if config.output_root.exists()
        else []
    )
    if existing_entries:
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


def _run_cohort_locked(config: RunConfig) -> dict[str, Any]:
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


def run_cohort(
    config: RunConfig,
    *,
    dry_run: bool = False,
    lock_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    verify_inputs(config)
    if dry_run:
        plan = build_state(config)
        for task in plan["tasks"]:
            task["status"] = "PLANNED"
        plan["status"] = "DRY_RUN"
        plan["note"] = "No directories, checkpoints, locks, or training processes were created."
        return plan
    with output_root_writer_lock(config.output_root, command=lock_command):
        return _run_cohort_locked(config)


def load_and_refresh(output_root: Path) -> tuple[Path, dict[str, Any]]:
    state_path = Path(output_root).resolve() / "p0b_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"P0-B state missing: {state_path}")
    state = read_json(state_path)
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported P0-B state schema")
    inspected = copy.deepcopy(state)
    refresh_state(inspected)
    return state_path, inspected


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
    inventory_consistent = _inventory_consistency(validations)
    if formal and not inventory_consistent:
        reasons.append("formal task inventories are missing or differ across arm/seed tasks")
    return {
        "schema": SCHEMA,
        "formal": formal,
        "formal_result_eligible": formal and not reasons,
        "valid": not reasons,
        "inventory_consistent": inventory_consistent,
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
        command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
        result = run_cohort(
            config, dry_run=args.dry_run, lock_command=command
        )
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0 if args.dry_run or result.get("status") == "COMPLETED" else 1
    _, state = load_and_refresh(args.output_root)
    if args.command == "status":
        print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    summary = validation_summary(state)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
