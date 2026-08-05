import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.run_protocol_v4_100epoch_cohort import (  # noqa: E402
    ARM_NAMES,
    CohortConfig,
    build_seed_command,
    build_seed_environment,
    run_cohort,
)


def make_config(tmp_path, **overrides):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    (protocol_dir / "protocol_summary.json").write_text("{}")
    (protocol_dir / "split.pkl").write_bytes(b"split")
    values = {
        "repo_root": REPO_ROOT,
        "protocol_dir": protocol_dir,
        "output_root": tmp_path / "cohort",
        "python_executable": Path(sys.executable),
    }
    values.update(overrides)
    return CohortConfig(**values)


def write_complete_sweep(env, *, epochs=100, arm_names=ARM_NAMES):
    output_dir = Path(env["NS_OUTBASE"]) / env["NS_OUT"]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_manifest": {"experiment": {"epochs": epochs}},
        "mse_ranking": [
            {"name": name, "epochs_ran": epochs}
            for name in arm_names
        ],
    }
    (output_dir / "vqvae_hp_sweep.json").write_text(json.dumps(payload))


def test_default_cohort_has_three_seeds_and_fixed_100_epoch_controls(tmp_path):
    config = make_config(tmp_path)

    assert config.seeds == (0, 1, 2)
    assert config.epochs == 100
    assert config.train_cap == 12000
    assert config.val_cap == 4637
    assert config.batch_size == 128
    assert config.learning_rate == "3e-4"
    assert ARM_NAMES == ("fsq_8192_4d", "fsq_4096_6d", "fsq_8192_6d")

    environments = [build_seed_environment(config, seed) for seed in config.seeds]
    for seed, env in zip(config.seeds, environments):
        assert env["NS_PROTOCOL_DIR"] == str(config.protocol_dir.resolve())
        assert env["NS_PROTOCOL_V2"] == "1"
        assert env["NS_VQ_EXPERIMENT_SEED"] == str(seed)
        assert env["NS_VQ_SWEEP_EPOCHS"] == "100"
        assert env["NS_VQ_MIN_EPOCHS"] == "100"
        assert env["NS_VQ_PATIENCE"] == "100"
        assert env["NS_VQ_SWEEP_TRAIN_CAP"] == "12000"
        assert env["NS_VQ_VAL_SAMPLES"] == "4637"
        assert env["NS_VQ_BS"] == "128"
        assert env["NS_VQ_LR"] == "3e-4"
        assert env["NS_VQ_MIN_PARENT_COVERAGE"] == "0.9"
        assert env["NS_VQ_COMPLEX_FRACTION"] == "0"
        assert env["NS_VQ_CURVED_FRACTION"] == "0"
        assert env["NS_VQ_COMPLEX_LOSS_WEIGHT"] == "1"
        assert env["NS_VQ_CURVED_LOSS_WEIGHT"] == "1"
        assert env["NS_VQ_MAX_NONFINITE_VAL_EPOCHS"] == "2"
        assert env["NS_OUT"] == f"seed{seed}"
        assert env["NS_VQ_TB_LOG_DIR"] == str(
            (config.output_root / f"seed{seed}" / "tensorboard").resolve()
        )

    ignored = {"NS_VQ_EXPERIMENT_SEED", "NS_OUT", "NS_VQ_TB_LOG_DIR"}
    reference = {key: value for key, value in environments[0].items() if key not in ignored}
    assert all(
        {key: value for key, value in env.items() if key not in ignored} == reference
        for env in environments[1:]
    )
    assert build_seed_command(config) == [
        str(config.python_executable.resolve()),
        str((REPO_ROOT / "breparg_improvements" / "train.py").resolve()),
        "--stage",
        "vqsweep",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": ()},
        {"seeds": (0, 0)},
        {"epochs": 0},
        {"train_cap": 0},
        {"val_cap": 0},
        {"batch_size": 0},
        {"learning_rate": "0"},
    ],
)
def test_cohort_rejects_invalid_controls(tmp_path, overrides):
    with pytest.raises(ValueError):
        make_config(tmp_path, **overrides)


def test_run_cohort_executes_seeds_sequentially_and_marks_only_verified_sweeps_complete(tmp_path):
    config = make_config(tmp_path)
    calls = []

    def runner(command, **kwargs):
        seed = int(kwargs["env"]["NS_VQ_EXPERIMENT_SEED"])
        calls.append((seed, list(command)))
        kwargs["stdout"].write(f"seed {seed} started\n")
        kwargs["stdout"].flush()
        write_complete_sweep(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)

    state = run_cohort(config, runner=runner)

    assert [seed for seed, _ in calls] == [0, 1, 2]
    assert state["status"] == "COMPLETED"
    assert all(state["seeds"][str(seed)]["status"] == "COMPLETED" for seed in config.seeds)
    persisted = json.loads((config.output_root / "cohort_state.json").read_text())
    assert persisted == state
    assert (config.output_root / "seed0" / "stdout.log").read_text() == "seed 0 started\n"


def test_run_cohort_stops_after_first_failed_seed(tmp_path):
    config = make_config(tmp_path)
    calls = []

    def runner(command, **kwargs):
        seed = int(kwargs["env"]["NS_VQ_EXPERIMENT_SEED"])
        calls.append(seed)
        if seed == 0:
            write_complete_sweep(kwargs["env"])
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 17)

    state = run_cohort(config, runner=runner)

    assert calls == [0, 1]
    assert state["status"] == "FAILED"
    assert state["seeds"]["0"]["status"] == "COMPLETED"
    assert state["seeds"]["1"]["status"] == "FAILED"
    assert state["seeds"]["1"]["returncode"] == 17
    assert state["seeds"]["2"]["status"] == "PENDING"


def test_run_cohort_rejects_success_exit_with_incomplete_sweep(tmp_path):
    config = make_config(tmp_path)

    def runner(command, **kwargs):
        write_complete_sweep(
            kwargs["env"],
            epochs=99,
            arm_names=("fsq_8192_4d", "fsq_4096_6d"),
        )
        return subprocess.CompletedProcess(command, 0)

    state = run_cohort(config, runner=runner)

    assert state["status"] == "FAILED"
    assert state["seeds"]["0"]["status"] == "FAILED"
    assert "incomplete sweep" in state["seeds"]["0"]["error"]
    assert state["seeds"]["1"]["status"] == "PENDING"
    assert state["seeds"]["2"]["status"] == "PENDING"


def test_run_cohort_does_not_inherit_unlisted_ns_controls(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setenv("NS_UNLISTED_EXPERIMENT_CONTROL", "contaminated")
    monkeypatch.setenv("NS_VQ_CURVED_FRACTION", "0.75")

    def runner(command, **kwargs):
        assert "NS_UNLISTED_EXPERIMENT_CONTROL" not in kwargs["env"]
        assert kwargs["env"]["NS_VQ_CURVED_FRACTION"] == "0"
        write_complete_sweep(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)

    state = run_cohort(config, runner=runner)

    assert state["status"] == "COMPLETED"


def test_run_cohort_rejects_existing_seed_artifacts_without_state(tmp_path):
    config = make_config(tmp_path)
    seed_root = config.output_root / "seed0"
    seed_root.mkdir(parents=True)
    (seed_root / "partial_history.json").write_text("{}")

    def runner(*args, **kwargs):
        raise AssertionError("training must not launch into a reused seed directory")

    with pytest.raises(RuntimeError, match="existing seed output"):
        run_cohort(config, runner=runner)
