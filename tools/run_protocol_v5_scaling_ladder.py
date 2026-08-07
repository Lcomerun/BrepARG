"""Run the fail-closed Protocol V5 VQ-VAE scaling ladder."""

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


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class RungSpec:
    name: str
    train_cap: int
    arms: tuple[str, ...]


RUNG_SPECS = (
    RungSpec(
        "60k",
        60_000,
        ("fsq_8192_4d", "fsq_4096_6d", "vq_4096_64d_random"),
    ),
    RungSpec("300k", 300_000, ("fsq_8192_4d", "fsq_4096_6d")),
)


@dataclass(frozen=True)
class LadderConfig:
    repo_root: Path
    archive_root: Path
    workspace_root: Path
    v4_summary: Path
    python_executable: Path = Path(sys.executable)
    load_failure_allowlist: Path | None = None
    seeds: tuple[int, ...] = (0, 1)
    protocol_chunks: str = "0-9"
    protocol_record_cap: int = 15_000
    max_epochs: int = 100
    min_epochs: int = 40
    patience: int = 15
    batch_size: int = 128
    val_cap: int = 12_000
    learning_rate: str = "3e-4"
    min_parent_coverage: float = 0.9
    max_load_failures: int = 100
    max_load_failure_fraction: float = 0.001
    target_curved_mse: float = 5e-5
    projected_full_patches: int = 3_000_000

    def __post_init__(self) -> None:
        for name in (
            "repo_root",
            "archive_root",
            "workspace_root",
            "v4_summary",
            "python_executable",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.load_failure_allowlist is not None:
            allowlist = Path(self.load_failure_allowlist).resolve()
            if not allowlist.is_file():
                raise FileNotFoundError(f"load failure allowlist does not exist: {allowlist}")
            object.__setattr__(self, "load_failure_allowlist", allowlist)
        seeds = tuple(int(seed) for seed in self.seeds)
        object.__setattr__(self, "seeds", seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be unique non-negative integers")
        for name in (
            "protocol_record_cap",
            "max_epochs",
            "min_epochs",
            "patience",
            "batch_size",
            "val_cap",
            "projected_full_patches",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs must not exceed max_epochs")
        if self.max_load_failures < 0:
            raise ValueError("max_load_failures must be non-negative")
        if not 0 <= self.max_load_failure_fraction <= 1:
            raise ValueError("max_load_failure_fraction must be between zero and one")
        if not 0 <= self.min_parent_coverage <= 1:
            raise ValueError("min_parent_coverage must be between zero and one")
        if float(self.learning_rate) <= 0 or not float(self.learning_rate) < float("inf"):
            raise ValueError("learning_rate must be positive and finite")


def build_inventory_preflight_command(config: LadderConfig) -> list[str]:
    return [
        str(config.python_executable),
        str(config.repo_root / "tools" / "preflight_cad_archive_inventory.py"),
        "--archive-root",
        str(config.archive_root),
        "--chunks",
        "all",
        "--output",
        str(config.workspace_root / "evidence" / "full_archive_inventory.json"),
    ]


def build_master_protocol_command(config: LadderConfig) -> list[str]:
    command = [
        str(config.python_executable),
        str(config.repo_root / "tools" / "build_cad_protocol.py"),
        "--archive-root",
        str(config.archive_root),
        "--chunks",
        config.protocol_chunks,
        "--output-dir",
        str(config.workspace_root / "protocol"),
        "--materialize-root",
        str(config.workspace_root / "materialized"),
        "--max-eligible-records",
        str(config.protocol_record_cap),
        "--max-load-failures",
        str(config.max_load_failures),
        "--max-load-failure-fraction",
        str(config.max_load_failure_fraction),
        "--seed",
        "20260803",
    ]
    if config.load_failure_allowlist is not None:
        command.extend(["--load-failure-allowlist", str(config.load_failure_allowlist)])
    return command


def build_training_command(config: LadderConfig) -> list[str]:
    return [
        str(config.python_executable),
        str(config.repo_root / "breparg_improvements" / "train.py"),
        "--stage",
        "vqsweep",
    ]


def build_training_environment(
    config: LadderConfig, rung: RungSpec, seed: int
) -> dict[str, str]:
    rung_root = config.workspace_root / "rungs" / rung.name
    seed_root = rung_root / f"seed{seed}"
    return {
        "NS_OUTBASE": str(rung_root),
        "NS_OUT": f"seed{seed}",
        "NS_PROTOCOL_DIR": str(config.workspace_root / "protocol"),
        "NS_PROTOCOL_V2": "1",
        "NS_VQ_BS": str(config.batch_size),
        "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1",
        "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1",
        "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_LR": str(config.learning_rate),
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "2",
        "NS_VQ_MIN_EPOCHS": str(config.min_epochs),
        "NS_VQ_MIN_PARENT_COVERAGE": str(config.min_parent_coverage),
        "NS_VQ_PATIENCE": str(config.patience),
        "NS_VQ_PLATEAU_METRIC": "curved_parent_mse",
        "NS_VQ_SWEEP_ARMS": ",".join(rung.arms),
        "NS_VQ_SWEEP_EPOCHS": str(config.max_epochs),
        "NS_VQ_SWEEP_TRAIN_CAP": str(rung.train_cap),
        "NS_VQ_TB_LOG_DIR": str(seed_root / "tensorboard"),
        "NS_VQ_VAL_SAMPLES": str(config.val_cap),
    }


def build_oracle_environment(config: LadderConfig, seed: int) -> dict[str, str]:
    oracle = RungSpec("continuous_bypass_300k", 300_000, ("continuous_bypass_64d",))
    return build_training_environment(config, oracle, seed)


def build_analysis_command(config: LadderConfig) -> list[str]:
    return [
        str(config.python_executable),
        str(config.repo_root / "tools" / "summarize_protocol_v5_scaling.py"),
        "--v4-summary",
        str(config.v4_summary),
        "--rung-root",
        str(config.workspace_root / "rungs"),
        "--output-dir",
        str(config.workspace_root / "analysis"),
        "--target-curved-mse",
        str(config.target_curved_mse),
        "--projected-full-patches",
        str(config.projected_full_patches),
    ]


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


def _validate_runtime_inputs(config: LadderConfig) -> None:
    required = (
        config.repo_root / "breparg_improvements" / "train.py",
        config.repo_root / "tools" / "build_cad_protocol.py",
        config.repo_root / "tools" / "preflight_cad_archive_inventory.py",
        config.repo_root / "tools" / "summarize_protocol_v5_scaling.py",
        config.python_executable,
        config.v4_summary,
    )
    missing = [str(path) for path in required if not path.is_file()]
    for chunk in range(10):
        archive = config.archive_root / f"abc_{chunk:04d}_parsed.zip"
        if not archive.is_file():
            missing.append(str(archive))
    if missing:
        raise FileNotFoundError("required ladder inputs are missing: " + ", ".join(missing))
    launcher_control_files = {
        "launcher.stdout.log",
        "launcher.stderr.log",
        "launcher.pid",
    }
    if config.workspace_root.exists():
        unexpected = [
            path
            for path in config.workspace_root.iterdir()
            if path.name not in launcher_control_files or not path.is_file()
        ]
        if unexpected:
            raise RuntimeError(
                "workspace root must be new or empty except launcher control files: "
                f"{config.workspace_root}"
            )


def _initial_state(config: LadderConfig) -> dict[str, Any]:
    return {
        "status": "RUNNING",
        "phase": "PENDING",
        "created_at": _now(),
        "updated_at": _now(),
        "gpu_expected": False,
        "advance_to_ar": False,
        "config": {
            "repo_root": str(config.repo_root),
            "archive_root": str(config.archive_root),
            "workspace_root": str(config.workspace_root),
            "v4_summary": str(config.v4_summary),
            "load_failure_allowlist": (
                str(config.load_failure_allowlist)
                if config.load_failure_allowlist is not None
                else None
            ),
            "seeds": list(config.seeds),
            "protocol_chunks": config.protocol_chunks,
            "protocol_record_cap": config.protocol_record_cap,
            "max_epochs": config.max_epochs,
            "min_epochs": config.min_epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "val_cap": config.val_cap,
            "rungs": [
                {"name": rung.name, "train_cap": rung.train_cap, "arms": list(rung.arms)}
                for rung in RUNG_SPECS
            ],
        },
        "steps": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sweep_is_complete(path: Path, rung: RungSpec, config: LadderConfig) -> tuple[bool, str | None]:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read sweep: {type(exc).__name__}"
    experiment = (payload.get("run_manifest") or {}).get("experiment") or {}
    rows = payload.get("mse_ranking")
    if experiment.get("train_cap") != rung.train_cap:
        return False, "sweep train cap does not match rung"
    if experiment.get("arms") is None or not isinstance(rows, list):
        return False, "sweep manifest or ranking is missing"
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(rung.arms):
        return False, "sweep arm set does not match rung"
    for row in by_name.values():
        epochs = int(row.get("epochs_ran") or 0)
        if epochs < config.min_epochs or epochs > config.max_epochs:
            return False, "sweep arm did not train within plateau epoch bounds"
        sampling = row.get("train_sampling") or {}
        if sampling.get("requested_cap_met") is not True:
            return False, "sweep did not meet requested patch cap"
        if float(sampling.get("final_parent_coverage") or 0) < config.min_parent_coverage:
            return False, "sweep parent coverage is below gate"
    return True, None


def _run_step(
    config: LadderConfig,
    state: dict[str, Any],
    *,
    name: str,
    phase: str,
    command: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[Any]],
    environment: Mapping[str, str] | None = None,
) -> bool:
    state_path = config.workspace_root / "ladder_state.json"
    logs = config.workspace_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    step = {
        "name": name,
        "phase": phase,
        "status": "RUNNING",
        "started_at": _now(),
        "command": list(command),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    state["phase"] = phase
    state["gpu_expected"] = phase.startswith("GPU_TRAINING")
    state["updated_at"] = _now()
    state["steps"].append(step)
    _atomic_write_json(state_path, state)
    child_env = {name: value for name, value in os.environ.items() if not name.startswith("NS_")}
    if environment:
        child_env.update(environment)
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = runner(
            list(command),
            cwd=str(config.repo_root),
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    step["returncode"] = int(completed.returncode)
    step["finished_at"] = _now()
    step["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
    state["updated_at"] = _now()
    if completed.returncode != 0:
        state["status"] = "FAILED"
        state["error"] = f"{name} exited with code {completed.returncode}"
        state["gpu_expected"] = False
    _atomic_write_json(state_path, state)
    return completed.returncode == 0


def run_ladder(
    config: LadderConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    _validate_runtime_inputs(config)
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    state = _initial_state(config)
    state_path = config.workspace_root / "ladder_state.json"
    _atomic_write_json(state_path, state)

    steps = (
        (
            "full_inventory_preflight",
            "INVENTORY_PREFLIGHT",
            build_inventory_preflight_command(config),
            None,
        ),
        (
            "master_protocol",
            "CPU_IO_PROTOCOL_BUILD",
            build_master_protocol_command(config),
            None,
        ),
    )
    for name, phase, command, environment in steps:
        if not _run_step(
            config,
            state,
            name=name,
            phase=phase,
            command=command,
            environment=environment,
            runner=runner,
        ):
            return state
    protocol_summary = _read_json(config.workspace_root / "protocol" / "protocol_summary.json")
    if protocol_summary.get("status") != "VERIFIED":
        state.update(
            {
                "status": "FAILED",
                "phase": "CPU_IO_PROTOCOL_BUILD",
                "gpu_expected": False,
                "error": "master protocol did not pass fail-closed gates",
                "updated_at": _now(),
            }
        )
        _atomic_write_json(state_path, state)
        return state

    for rung in RUNG_SPECS:
        for seed in config.seeds:
            name = f"train_{rung.name}_seed{seed}"
            environment = build_training_environment(config, rung, seed)
            if not _run_step(
                config,
                state,
                name=name,
                phase=f"GPU_TRAINING_{rung.name.upper()}",
                command=build_training_command(config),
                environment=environment,
                runner=runner,
            ):
                return state
            sweep = config.workspace_root / "rungs" / rung.name / f"seed{seed}" / "vqvae_hp_sweep.json"
            complete, error = _sweep_is_complete(sweep, rung, config)
            if not complete:
                state.update(
                    {
                        "status": "FAILED",
                        "gpu_expected": False,
                        "error": f"{name} output validation failed: {error}",
                        "updated_at": _now(),
                    }
                )
                _atomic_write_json(state_path, state)
                return state

    return _run_analysis_and_oracle(config, state, runner)


def _run_analysis_and_oracle(
    config: LadderConfig,
    state: dict[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, Any]:
    state_path = config.workspace_root / "ladder_state.json"
    if not _run_step(
        config,
        state,
        name="analyze_scaling",
        phase="ANALYSIS",
        command=build_analysis_command(config),
        runner=runner,
    ):
        return state
    decision = _read_json(config.workspace_root / "analysis" / "scaling_summary.json")["decision"]
    oracle_status = "NOT_REQUIRED"
    if decision.get("continuous_bypass_oracle_recommended") is True:
        oracle = RungSpec("continuous_bypass_300k", 300_000, ("continuous_bypass_64d",))
        oracle_status = "RUNNING"
        for seed in config.seeds:
            name = f"continuous_bypass_300k_seed{seed}"
            if not _run_step(
                config,
                state,
                name=name,
                phase="GPU_ORACLE_CONTINUOUS_BYPASS",
                command=build_training_command(config),
                environment=build_oracle_environment(config, seed),
                runner=runner,
            ):
                return state
            sweep = (
                config.workspace_root
                / "rungs"
                / oracle.name
                / f"seed{seed}"
                / "vqvae_hp_sweep.json"
            )
            complete, error = _sweep_is_complete(sweep, oracle, config)
            if not complete:
                state.update(
                    {
                        "status": "FAILED",
                        "gpu_expected": False,
                        "error": f"{name} output validation failed: {error}",
                        "updated_at": _now(),
                    }
                )
                _atomic_write_json(state_path, state)
                return state
        oracle_status = "COMPLETED"
    state.update(
        {
            "status": "COMPLETED",
            "phase": "COMPLETED",
            "gpu_expected": False,
            "finished_at": _now(),
            "updated_at": _now(),
            "decision": decision,
            "continuous_bypass_oracle": oracle_status,
            "advance_to_ar": False,
        }
    )
    _atomic_write_json(state_path, state)
    return state


def resume_ladder_after_analysis(
    config: LadderConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Resume a failed analysis without repeating verified protocol or VQ sweeps."""
    state_path = config.workspace_root / "ladder_state.json"
    state = _read_json(state_path)
    if state.get("status") != "FAILED" or state.get("phase") != "ANALYSIS":
        raise RuntimeError("analysis recovery requires a FAILED state in ANALYSIS phase")
    protocol_summary = _read_json(config.workspace_root / "protocol" / "protocol_summary.json")
    if protocol_summary.get("status") != "VERIFIED":
        raise RuntimeError("analysis recovery requires a verified master protocol")
    for rung in RUNG_SPECS:
        for seed in config.seeds:
            sweep = (
                config.workspace_root
                / "rungs"
                / rung.name
                / f"seed{seed}"
                / "vqvae_hp_sweep.json"
            )
            complete, error = _sweep_is_complete(sweep, rung, config)
            if not complete:
                raise RuntimeError(
                    f"analysis recovery rejected {rung.name} seed{seed}: {error}"
                )
    state.update(
        {
            "status": "RUNNING",
            "phase": "ANALYSIS",
            "gpu_expected": False,
            "updated_at": _now(),
        }
    )
    state.pop("error", None)
    _atomic_write_json(state_path, state)
    return _run_analysis_and_oracle(config, state, runner)


def parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument(
        "--v4-summary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "protocol_v4"
        / "fsq_abc_100epoch_three_seed_20260805.json",
    )
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--load-failure-allowlist", type=Path)
    parser.add_argument("--seeds", type=parse_seed_list, default=(0, 1))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    config = LadderConfig(**vars(parse_args(argv)))
    state = run_ladder(config)
    print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
