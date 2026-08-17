"""Run Protocol V6: four representation arms, five seeds, exactly 100 epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = (
    "fsq_8192_4d",
    "fsq_4096_6d",
    "vq_4096_64d_random",
    "continuous_bypass_64d",
)
SEEDS = (0, 1, 2, 3, 4)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def training_environment(
    *, repo_root: Path, protocol_dir: Path, output_root: Path, seed: int,
    train_cap: int = 300000, val_cap: int = 12000, epochs: int = 100,
    batch_size: int = 128, learning_rate: str = "3e-4",
) -> dict[str, str]:
    seed_root = Path(output_root) / f"seed{seed}"
    return {
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(Path(repo_root).resolve()).replace("\\", "/"),
        "NS_OUTBASE": str(Path(output_root).resolve()), "NS_OUT": f"seed{seed}",
        "NS_PROTOCOL_DIR": str(Path(protocol_dir).resolve()), "NS_PROTOCOL_V2": "1",
        "NS_VQ_BS": str(batch_size), "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1", "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1", "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_LR": str(learning_rate), "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "2",
        "NS_VQ_MIN_EPOCHS": str(epochs), "NS_VQ_MIN_PARENT_COVERAGE": "0.9",
        "NS_VQ_PATIENCE": str(epochs), "NS_VQ_PLATEAU_METRIC": "curved_parent_mse",
        "NS_VQ_SAVE_FINAL": "1", "NS_VQ_SWEEP_ARMS": ",".join(ARMS),
        "NS_VQ_SWEEP_EPOCHS": str(epochs), "NS_VQ_SWEEP_TRAIN_CAP": str(train_cap),
        "NS_VQ_TB_LOG_DIR": str(seed_root / "tensorboard"),
        "NS_VQ_VAL_SAMPLES": str(val_cap),
    }


def validate_sweep(
    sweep_path: Path, *, train_cap: int, val_cap: int, epochs: int
) -> dict[str, Any]:
    report: dict[str, Any] = {"valid": False, "reasons": [], "checkpoints": []}
    if not Path(sweep_path).is_file():
        report["reasons"].append("sweep missing")
        return report
    try:
        payload = json.loads(Path(sweep_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["reasons"].append(f"sweep unreadable: {type(exc).__name__}")
        return report
    experiment = (payload.get("run_manifest") or {}).get("experiment") or {}
    seed_dir = Path(sweep_path).parent.name
    if seed_dir.startswith("seed") and seed_dir[4:].isdigit():
        try:
            run_seed = int(experiment.get("seed", -1))
        except (TypeError, ValueError):
            run_seed = -1
        if run_seed != int(seed_dir[4:]):
            report["reasons"].append(
                f"seed mismatch: run={experiment.get('seed')!r} dir={seed_dir}")
    rows = payload.get("mse_ranking") or []
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(ARMS):
        report["reasons"].append(f"arm set mismatch: {sorted(by_name)}")
    if int(experiment.get("train_cap") or 0) != train_cap:
        report["reasons"].append("train cap mismatch")
    if int(experiment.get("val_cap") or 0) != val_cap:
        report["reasons"].append("val cap mismatch")
    if int(experiment.get("epochs") or 0) != epochs:
        report["reasons"].append("requested epoch mismatch")
    for arm in ARMS:
        row = by_name.get(arm)
        if row is None:
            continue
        if int(row.get("epochs_ran") or 0) != epochs:
            report["reasons"].append(f"{arm}: epochs_ran != {epochs}")
        if int(row.get("final_checkpoint_epoch") or -1) != epochs - 1:
            report["reasons"].append(f"{arm}: final checkpoint epoch mismatch")
        sampling = row.get("train_sampling") or {}
        if sampling.get("requested_cap_met") is not True:
            report["reasons"].append(f"{arm}: train cap not met")
        if float(sampling.get("final_parent_coverage") or 0) < 0.9:
            report["reasons"].append(f"{arm}: parent coverage below 0.9")
        val_sampling = row.get("val_sampling") or {}
        if int(val_sampling.get("selected") or val_sampling.get("effective") or 0) != val_cap:
            report["reasons"].append(f"{arm}: val effective count mismatch")
        for checkpoint_kind in ("checkpoint_best", "checkpoint_final"):
            path = Path(str(row.get(checkpoint_kind) or ""))
            if not path.is_file():
                report["reasons"].append(f"{arm}: {checkpoint_kind} missing")
            else:
                report["checkpoints"].append({
                    "arm": arm, "kind": checkpoint_kind, "path": str(path.resolve()),
                    "bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
    report["valid"] = not report["reasons"]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--train-cap", type=int, default=300000)
    parser.add_argument("--val-cap", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", default="3e-4")
    parser.add_argument("--reconstruction-cads", type=int, default=100)
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve(); args.protocol_dir = args.protocol_dir.resolve()
    args.breparg_root = args.breparg_root.resolve(); args.output_root = args.output_root.resolve()
    args.python = args.python.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if seeds != SEEDS:
        raise ValueError(f"formal V6 seeds must be {SEEDS}, got {seeds}")
    required = [
        args.python, args.repo_root / "breparg_improvements" / "train.py",
        args.repo_root / "tools" / "evaluate_surface_reconstruction_cohort.py",
        args.protocol_dir / "protocol_summary.json", args.protocol_dir / "split.pkl",
        args.breparg_root / "quantise.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("V6 inputs missing: " + ", ".join(missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "cohort_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "status": "RUNNING", "created_at": now(), "updated_at": now(),
        "active_seed": None, "advance_to_ar": False, "steps": [],
        "configuration": {
            "arms": list(ARMS), "seeds": list(seeds), "train_cap": args.train_cap,
            "val_cap": args.val_cap, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "protocol_dir": str(args.protocol_dir),
            "reconstruction_cads": args.reconstruction_cads,
        },
    }
    state.update(status="RUNNING", active_seed=None, updated_at=now(), advance_to_ar=False)
    atomic_json(state_path, state)
    base_env = {key: value for key, value in os.environ.items() if not key.startswith("NS_")}
    command = [str(args.python), str(args.repo_root / "breparg_improvements" / "train.py"), "--stage", "vqsweep"]
    logs = args.output_root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        sweep_path = args.output_root / f"seed{seed}" / "vqvae_hp_sweep.json"
        validation = validate_sweep(
            sweep_path, train_cap=args.train_cap, val_cap=args.val_cap, epochs=args.epochs
        )
        if validation["valid"]:
            continue
        step = {
            "kind": "training", "seed": seed, "status": "RUNNING",
            "started_at": now(), "command": command,
            "stdout": str(logs / f"seed{seed}.stdout.log"),
            "stderr": str(logs / f"seed{seed}.stderr.log"),
        }
        state["steps"].append(step); state["active_seed"] = seed; state["updated_at"] = now()
        atomic_json(state_path, state)
        env = dict(base_env)
        env.update(training_environment(
            repo_root=args.repo_root, protocol_dir=args.protocol_dir,
            output_root=args.output_root, seed=seed, train_cap=args.train_cap,
            val_cap=args.val_cap, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ))
        with Path(step["stdout"]).open("w", encoding="utf-8", newline="\n") as stdout, Path(step["stderr"]).open("w", encoding="utf-8", newline="\n") as stderr:
            completed = subprocess.run(
                command, cwd=args.repo_root, env=env, stdout=stdout, stderr=stderr, check=False
            )
        validation = validate_sweep(
            sweep_path, train_cap=args.train_cap, val_cap=args.val_cap, epochs=args.epochs
        )
        step.update(
            status="COMPLETED" if completed.returncode == 0 and validation["valid"] else "FAILED",
            returncode=completed.returncode, finished_at=now(), validation=validation,
        )
        state["active_seed"] = None; state["updated_at"] = now()
        if step["status"] != "COMPLETED":
            state["status"] = "FAILED"; atomic_json(state_path, state); return 1
        atomic_json(state_path, state)

    reconstruction_dir = args.output_root / "surface_reconstruction"
    summary_path = reconstruction_dir / "surface_reconstruction_summary.json"
    reconstruction_complete = False
    if summary_path.is_file():
        try:
            reconstruction_complete = json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "COMPLETED"
        except (OSError, json.JSONDecodeError):
            reconstruction_complete = False
    if not reconstruction_complete:
        reconstruction_command = [
            str(args.python), str(args.repo_root / "tools" / "evaluate_surface_reconstruction_cohort.py"),
            "--training-root", str(args.output_root), "--protocol-dir", str(args.protocol_dir),
            "--breparg-root", str(args.breparg_root), "--output-dir", str(reconstruction_dir),
            "--seeds", ",".join(map(str, seeds)), "--arms", ",".join(ARMS),
            "--max-cads", str(args.reconstruction_cads), "--expected-epochs", str(args.epochs),
            "--device", "cuda", "--batch-size", "64",
        ]
        step = {
            "kind": "surface_reconstruction", "status": "RUNNING",
            "started_at": now(), "command": reconstruction_command,
            "stdout": str(logs / "surface_reconstruction.stdout.log"),
            "stderr": str(logs / "surface_reconstruction.stderr.log"),
        }
        state["steps"].append(step); state["phase"] = "SURFACE_RECONSTRUCTION"
        state["updated_at"] = now(); atomic_json(state_path, state)
        env = dict(base_env)
        env.update({
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(args.repo_root).replace("\\", "/"),
        })
        with Path(step["stdout"]).open("w", encoding="utf-8", newline="\n") as stdout, Path(step["stderr"]).open("w", encoding="utf-8", newline="\n") as stderr:
            completed = subprocess.run(
                reconstruction_command, cwd=args.repo_root, env=env,
                stdout=stdout, stderr=stderr, check=False,
            )
        reconstruction_complete = False
        if summary_path.is_file():
            try:
                reconstruction_complete = json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "COMPLETED"
            except (OSError, json.JSONDecodeError):
                pass
        step.update(
            status="COMPLETED" if completed.returncode == 0 and reconstruction_complete else "FAILED",
            returncode=completed.returncode, finished_at=now(), summary=str(summary_path),
        )
        if step["status"] != "COMPLETED":
            state["status"] = "FAILED"; state["updated_at"] = now()
            atomic_json(state_path, state); return 1
    state.update(
        status="COMPLETED", phase="COMPLETED", active_seed=None,
        finished_at=now(), updated_at=now(), advance_to_ar=False,
        surface_reconstruction_summary=str(summary_path),
    )
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
