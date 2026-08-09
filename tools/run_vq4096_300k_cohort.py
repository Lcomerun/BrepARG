"""Run the missing learned VQ-4096/64D Protocol V5 300k cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    train_cap: int, val_cap: int, epochs: int, min_epochs: int,
    patience: int, batch_size: int, learning_rate: str,
) -> dict[str, str]:
    seed_root = output_root / f"seed{seed}"
    return {
        # Git checks this process-local config before repository ownership, so
        # the training manifest can bind HEAD without changing global config.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(Path(repo_root).resolve()).replace("\\", "/"),
        "NS_OUTBASE": str(output_root), "NS_OUT": f"seed{seed}",
        "NS_PROTOCOL_DIR": str(protocol_dir), "NS_PROTOCOL_V2": "1",
        "NS_VQ_BS": str(batch_size), "NS_VQ_COMPLEX_FRACTION": "0",
        "NS_VQ_COMPLEX_LOSS_WEIGHT": "1", "NS_VQ_CURVED_FRACTION": "0",
        "NS_VQ_CURVED_LOSS_WEIGHT": "1", "NS_VQ_EXPERIMENT_SEED": str(seed),
        "NS_VQ_LR": str(learning_rate), "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": "2",
        "NS_VQ_MIN_EPOCHS": str(min_epochs), "NS_VQ_MIN_PARENT_COVERAGE": "0.9",
        "NS_VQ_PATIENCE": str(patience), "NS_VQ_PLATEAU_METRIC": "curved_parent_mse",
        "NS_VQ_SWEEP_ARMS": "vq_4096_64d_random", "NS_VQ_SWEEP_EPOCHS": str(epochs),
        "NS_VQ_SWEEP_TRAIN_CAP": str(train_cap),
        "NS_VQ_TB_LOG_DIR": str(seed_root / "tensorboard"),
        "NS_VQ_VAL_SAMPLES": str(val_cap),
    }


def sweep_complete(path: Path, *, train_cap: int, min_epochs: int, max_epochs: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    experiment = (payload.get("run_manifest") or {}).get("experiment") or {}
    ranking = payload.get("mse_ranking") or []
    if experiment.get("train_cap") != train_cap or len(ranking) != 1:
        return False
    row = ranking[0]
    sampling = row.get("train_sampling") or {}
    return (
        row.get("name") == "vq_4096_64d_random"
        and min_epochs <= int(row.get("epochs_ran") or 0) <= max_epochs
        and sampling.get("requested_cap_met") is True
        and float(sampling.get("final_parent_coverage") or 0) >= 0.9
    )


def mark_retry_running(state: dict[str, Any]) -> dict[str, Any]:
    state["status"] = "RUNNING"
    state["updated_at"] = now()
    state["active_seed"] = None
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--train-cap", type=int, default=300000)
    parser.add_argument("--val-cap", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", default="3e-4")
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve(); args.protocol_dir = args.protocol_dir.resolve()
    args.output_root = args.output_root.resolve(); args.python = args.python.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    required = [args.python, args.repo_root / "breparg_improvements" / "train.py",
                args.protocol_dir / "protocol_summary.json", args.protocol_dir / "split.pkl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("cohort inputs missing: " + ", ".join(missing))
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "cohort_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "status": "RUNNING", "created_at": now(), "updated_at": now(),
        "advance_to_ar": False, "seeds": list(seeds), "steps": [],
        "configuration": {
            "arm": "vq_4096_64d_random", "codebook_size": 4096,
            "embedding_dim": 64, "anchor": "random", "train_cap": args.train_cap,
            "val_cap": args.val_cap, "epochs": args.epochs, "min_epochs": args.min_epochs,
            "patience": args.patience, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "protocol_dir": str(args.protocol_dir),
        },
    }
    # A retry keeps prior failed steps as evidence, but the cohort itself is
    # running again.  Without this reset, status remains misleadingly FAILED
    # until both seeds complete.
    mark_retry_running(state)
    atomic_json(state_path, state)
    for seed in seeds:
        sweep = args.output_root / f"seed{seed}" / "vqvae_hp_sweep.json"
        if sweep_complete(sweep, train_cap=args.train_cap, min_epochs=args.min_epochs, max_epochs=args.epochs):
            continue
        log_root = args.output_root / "logs"; log_root.mkdir(parents=True, exist_ok=True)
        step = {"seed": seed, "status": "RUNNING", "started_at": now(),
                "stdout": str(log_root / f"seed{seed}.stdout.log"),
                "stderr": str(log_root / f"seed{seed}.stderr.log")}
        state["steps"].append(step); state["active_seed"] = seed; state["updated_at"] = now()
        atomic_json(state_path, state)
        env = {key: value for key, value in os.environ.items() if not key.startswith("NS_")}
        env.update(training_environment(
            repo_root=args.repo_root, protocol_dir=args.protocol_dir,
            output_root=args.output_root, seed=seed,
            train_cap=args.train_cap, val_cap=args.val_cap, epochs=args.epochs,
            min_epochs=args.min_epochs, patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ))
        command = [str(args.python), str(args.repo_root / "breparg_improvements" / "train.py"), "--stage", "vqsweep"]
        step["command"] = command
        with Path(step["stdout"]).open("w", encoding="utf-8", newline="\n") as stdout, Path(step["stderr"]).open("w", encoding="utf-8", newline="\n") as stderr:
            completed = subprocess.run(command, cwd=args.repo_root, env=env, stdout=stdout, stderr=stderr, check=False)
        step.update(status="COMPLETED" if completed.returncode == 0 else "FAILED",
                    returncode=completed.returncode, finished_at=now())
        state["updated_at"] = now(); state["active_seed"] = None
        if completed.returncode != 0 or not sweep_complete(sweep, train_cap=args.train_cap, min_epochs=args.min_epochs, max_epochs=args.epochs):
            state["status"] = "FAILED"; atomic_json(state_path, state); return 1
        atomic_json(state_path, state)
    state["status"] = "COMPLETED"; state["finished_at"] = now(); state["active_seed"] = None
    state["advance_to_ar"] = False; atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
