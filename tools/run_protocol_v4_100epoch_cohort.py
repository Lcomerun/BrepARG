"""Run the Protocol V4 three-arm FSQ cohort sequentially for three seeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ARM_NAMES = ("fsq_8192_4d", "fsq_4096_6d", "fsq_8192_6d")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class CohortConfig:
    repo_root: Path
    protocol_dir: Path
    output_root: Path
    python_executable: Path = Path(sys.executable)
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 100
    train_cap: int = 12000
    val_cap: int = 4637
    batch_size: int = 128
    learning_rate: str = "3e-4"
    min_parent_coverage: float = 0.9

    def __post_init__(self) -> None:
        for name in ("repo_root", "protocol_dir", "output_root", "python_executable"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        seeds = tuple(int(seed) for seed in self.seeds)
        object.__setattr__(self, "seeds", seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be unique non-negative integers")
        for name in ("epochs", "train_cap", "val_cap", "batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("learning_rate must be a positive finite number") from exc
        if not 0.0 < learning_rate < float("inf"):
            raise ValueError("learning_rate must be a positive finite number")
        if not 0.0 <= float(self.min_parent_coverage) <= 1.0:
            raise ValueError("min_parent_coverage must be between zero and one")


def build_seed_command(config: CohortConfig) -> list[str]:
    return [
        str(config.python_executable),
        str(config.repo_root / "breparg_improvements" / "train.py"),
        "--stage",
        "vqsweep",
    ]


def build_seed_environment(config: CohortConfig, seed: int) -> dict[str, str]:
    seed = int(seed)
    seed_root = (config.output_root / f"seed{seed}").resolve()
    return {
        "NS_OUTBASE": str(config.output_root),
        "NS_OUT": f"seed{seed}",
        "NS_PROTOCOL_DIR": str(config.protocol_dir),
        "NS_PROTOCOL_V2": "1",
        "NS_VQ_BS": str(config.batch_size),
        "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1",
        "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1",
        "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_LR": str(config.learning_rate),
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "2",
        "NS_VQ_MIN_EPOCHS": str(config.epochs),
        "NS_VQ_MIN_PARENT_COVERAGE": str(config.min_parent_coverage),
        "NS_VQ_PATIENCE": str(config.epochs),
        "NS_VQ_SWEEP_EPOCHS": str(config.epochs),
        "NS_VQ_SWEEP_TRAIN_CAP": str(config.train_cap),
        "NS_VQ_TB_LOG_DIR": str(seed_root / "tensorboard"),
        "NS_VQ_VAL_SAMPLES": str(config.val_cap),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_inputs(config: CohortConfig) -> None:
    required = (
        config.repo_root / "breparg_improvements" / "train.py",
        config.protocol_dir / "protocol_summary.json",
        config.protocol_dir / "split.pkl",
        config.python_executable,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required cohort inputs are missing: " + ", ".join(missing))
    reused = []
    for seed in config.seeds:
        seed_root = config.output_root / f"seed{seed}"
        if seed_root.exists() and any(seed_root.iterdir()):
            reused.append(str(seed_root))
    if reused:
        raise RuntimeError("existing seed output must not be reused: " + ", ".join(reused))


def _sweep_is_complete(path: Path, expected_epochs: int) -> tuple[bool, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"incomplete sweep: cannot read {path.name}: {type(exc).__name__}"
    manifest_epochs = (
        payload.get("run_manifest", {}).get("experiment", {}).get("epochs")
        if isinstance(payload, dict)
        else None
    )
    ranking = payload.get("mse_ranking") if isinstance(payload, dict) else None
    if manifest_epochs != expected_epochs or not isinstance(ranking, list):
        return False, "incomplete sweep: run manifest epoch count or ranking is missing"
    rows = {row.get("name"): row for row in ranking if isinstance(row, dict)}
    if set(rows) != set(ARM_NAMES):
        return False, "incomplete sweep: expected all three FSQ arms"
    if any(row.get("epochs_ran") != expected_epochs for row in rows.values()):
        return False, f"incomplete sweep: every arm must run {expected_epochs} epochs"
    return True, None


def _initial_state(config: CohortConfig) -> dict[str, Any]:
    return {
        "status": "PENDING",
        "created_at": _now(),
        "updated_at": _now(),
        "config": {
            "repo_root": str(config.repo_root),
            "protocol_dir": str(config.protocol_dir),
            "output_root": str(config.output_root),
            "python_executable": str(config.python_executable),
            "seeds": list(config.seeds),
            "epochs": config.epochs,
            "train_cap": config.train_cap,
            "val_cap": config.val_cap,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "min_parent_coverage": config.min_parent_coverage,
            "arms": list(ARM_NAMES),
        },
        "seeds": {
            str(seed): {
                "status": "PENDING",
                "output_dir": str(config.output_root / f"seed{seed}"),
            }
            for seed in config.seeds
        },
    }


def run_cohort(
    config: CohortConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    _validate_inputs(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    state_path = config.output_root / "cohort_state.json"
    if state_path.exists():
        raise RuntimeError(f"cohort state already exists; use a new output root: {state_path}")
    state = _initial_state(config)
    state["status"] = "RUNNING"
    state["updated_at"] = _now()
    _atomic_write_json(state_path, state)
    command = build_seed_command(config)

    for seed in config.seeds:
        seed_key = str(seed)
        seed_root = config.output_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        stdout_path = seed_root / "stdout.log"
        stderr_path = seed_root / "stderr.log"
        seed_state = state["seeds"][seed_key]
        seed_state.update(
            {
                "status": "RUNNING",
                "started_at": _now(),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
        state["updated_at"] = _now()
        _atomic_write_json(state_path, state)
        overlay = build_seed_environment(config, seed)
        environment = {
            name: value for name, value in os.environ.items() if not name.startswith("NS_")
        }
        environment.update(overlay)
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as stderr:
            completed = runner(
                command,
                cwd=str(config.repo_root),
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        seed_state["returncode"] = int(completed.returncode)
        seed_state["finished_at"] = _now()
        if completed.returncode != 0:
            seed_state["status"] = "FAILED"
            seed_state["error"] = f"training process exited with code {completed.returncode}"
            state["status"] = "FAILED"
            state["updated_at"] = _now()
            _atomic_write_json(state_path, state)
            return state
        complete, error = _sweep_is_complete(seed_root / "vqvae_hp_sweep.json", config.epochs)
        if not complete:
            seed_state["status"] = "FAILED"
            seed_state["error"] = error
            state["status"] = "FAILED"
            state["updated_at"] = _now()
            _atomic_write_json(state_path, state)
            return state
        seed_state["status"] = "COMPLETED"
        state["updated_at"] = _now()
        _atomic_write_json(state_path, state)

    state["status"] = "COMPLETED"
    state["finished_at"] = _now()
    state["updated_at"] = state["finished_at"]
    _atomic_write_json(state_path, state)
    return state


def parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", type=parse_seed_list, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-cap", type=int, default=12000)
    parser.add_argument("--val-cap", type=int, default=4637)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", default="3e-4")
    parser.add_argument("--min-parent-coverage", type=float, default=0.9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = CohortConfig(**vars(args))
    state = run_cohort(config)
    print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
