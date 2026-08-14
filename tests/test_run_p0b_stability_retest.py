import json
import subprocess
from pathlib import Path

import pytest

from tools.run_p0b_stability_retest import (
    FORMAL_ARMS,
    FORMAL_SEEDS,
    RunConfig,
    build_state,
    build_task,
    run_cohort,
    validate_task,
)


def make_inputs(tmp_path: Path) -> RunConfig:
    repo = tmp_path / "repo"
    protocol = tmp_path / "protocol"
    breparg = repo / "BrepARG"
    (repo / "breparg_improvements").mkdir(parents=True)
    protocol.mkdir()
    breparg.mkdir()
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    (repo / "breparg_improvements" / "train.py").write_text("# train\n", encoding="utf-8")
    (repo / "breparg_improvements" / "training_stability.py").write_text(
        "# stability\n", encoding="utf-8"
    )
    (protocol / "protocol_summary.json").write_text('{"status":"VERIFIED"}\n', encoding="utf-8")
    (protocol / "split.pkl").write_bytes(b"split")
    (breparg / "model.py").write_text("# model\n", encoding="utf-8")
    (breparg / "quantise.py").write_text("# quantise\n", encoding="utf-8")
    return RunConfig(
        repo_root=repo,
        protocol_dir=protocol,
        breparg_root=breparg,
        output_root=tmp_path / "p0b-output",
        python=python,
    )


def healthy_epoch(epoch: int, signature: str) -> dict:
    return {
        "epoch": epoch,
        "train_batches": 469,
        "finite_train_batches": 469,
        "skipped_train_batches": 0,
        "val_batches": 94,
        "finite_val_batches": 94,
        "nonfinite_loss_batches": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_state_batches": 0,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "gradients_finite": True,
        "training_state_finite": True,
        "grad_clip_active": True,
        "preclip_grad_norm": 1.25,
        "experiment_signature": signature,
    }


def materialize_success(task: dict, epochs: int) -> None:
    root = Path(task["task_root"])
    root.mkdir(parents=True, exist_ok=True)
    Path(task["history"]).write_text(
        json.dumps(
            {
                "config": {
                    "experiment_signature": task["signature"],
                    "precision": {"name": task["signature_payload"]["precision"]},
                    "strict_nonfinite": True,
                    "grad_clip_norm": 1.0,
                    "scheduler": {
                        "factor": 0.5,
                        "patience": 8,
                        "threshold": 1e-5,
                        "threshold_mode": "abs",
                        "min_lr": 1e-6,
                    },
                },
                "history": [healthy_epoch(epoch, task["signature"]) for epoch in range(epochs)],
            }
        ),
        encoding="utf-8",
    )
    Path(task["sweep"]).write_text(
        json.dumps(
            {
                "mse_ranking": [
                    {
                        "name": task["arm"],
                        "epochs_ran": epochs,
                        "final_checkpoint_epoch": epochs - 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for field in ("best_checkpoint", "final_checkpoint", "rolling_checkpoint"):
        Path(task[field]).write_bytes(field.encode("ascii"))


def test_formal_protocol_is_fixed_and_smoke_overrides_are_explicit(tmp_path):
    config = make_inputs(tmp_path)
    assert config.arms == FORMAL_ARMS
    assert config.seeds == FORMAL_SEEDS
    assert config.train_cap == 60_000
    assert config.val_cap == 12_000
    assert config.batch_size == 128
    assert config.epochs == 100
    assert config.learning_rate == "3e-4"

    with pytest.raises(ValueError, match="immutable"):
        RunConfig(
            repo_root=config.repo_root,
            protocol_dir=config.protocol_dir,
            breparg_root=config.breparg_root,
            output_root=tmp_path / "changed",
            python=config.python,
            epochs=2,
        )

    smoke = RunConfig(
        repo_root=config.repo_root,
        protocol_dir=config.protocol_dir,
        breparg_root=config.breparg_root,
        output_root=tmp_path / "smoke",
        python=config.python,
        arms=(FORMAL_ARMS[0],),
        seeds=(3,),
        train_cap=16,
        val_cap=8,
        batch_size=4,
        epochs=2,
        smoke=True,
    )
    assert smoke.smoke is True

    with pytest.raises(ValueError, match="smoke overrides exceed"):
        RunConfig(
            repo_root=config.repo_root,
            protocol_dir=config.protocol_dir,
            breparg_root=config.breparg_root,
            output_root=tmp_path / "unbounded-smoke",
            python=config.python,
            arms=(FORMAL_ARMS[0],),
            seeds=(3,),
            epochs=3,
            smoke=True,
        )


def test_task_matrix_uses_independent_directories_and_complete_signed_environment(tmp_path):
    config = make_inputs(tmp_path)
    state = build_state(config)
    assert [task["task_id"] for task in state["tasks"]] == [
        "vq_4096_64d_random:seed3",
        "vq_4096_64d_random:seed4",
        "continuous_bypass_64d:seed3",
        "continuous_bypass_64d:seed4",
    ]
    assert len({task["task_root"] for task in state["tasks"]}) == 4
    assert len({task["signature"] for task in state["tasks"]}) == 4
    for task in state["tasks"]:
        env = task["environment"]
        assert env["NS_OUT"] == task["task_root"]
        assert env["NS_VQ_SWEEP_ARMS"] == task["arm"]
        assert env["NS_VQ_EXPERIMENT_SEED"] == str(task["seed"])
        assert env["NS_VQ_EXPERIMENT_SIGNATURE"] == task["signature"]
        assert env["NS_VQ_AUTO_RESUME"] == "1"
        assert env["NS_VQ_STRICT_NONFINITE"] == "1"
        assert env["NS_VQ_ROLLING_CHECKPOINT"] == task["rolling_checkpoint"]
        assert env["NS_VQ_PRECISION"] == "fp32"
        assert env["NS_VQ_GRAD_CLIP"] == "1.0"
        assert env["NS_VQ_SCHEDULER_FACTOR"] == "0.5"
        assert env["NS_VQ_SCHEDULER_PATIENCE"] == "8"
        assert env["NS_VQ_SCHEDULER_THRESHOLD"] == "1e-5"
        assert task["signature_payload"]["scheduler"]["threshold_mode"] == "abs"
        assert env["NS_VQ_SWEEP_TRAIN_CAP"] == "60000"
        assert env["NS_VQ_VAL_SAMPLES"] == "12000"
        assert env["NS_VQ_BS"] == "128"
        assert env["NS_VQ_SWEEP_EPOCHS"] == "100"


def test_validate_task_requires_exact_epoch_coverage_and_zero_nonfinite(tmp_path):
    config = make_inputs(tmp_path)
    task = build_task(config, FORMAL_ARMS[0], 3)
    materialize_success(task, 100)
    result = validate_task(task, formal=True)
    assert result["valid"] is True
    assert result["last_epoch"] == 99

    payload = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
    payload["history"][27]["nonfinite_gradient_batches"] = 1
    Path(task["history"]).write_text(json.dumps(payload), encoding="utf-8")
    result = validate_task(task, formal=True)
    assert result["valid"] is False
    assert any("epoch 27" in reason and "nonfinite_gradient_batches" in reason for reason in result["reasons"])


def test_bypass_stability_validation_ignores_usage_promotion_but_requires_artifacts(tmp_path):
    config = make_inputs(tmp_path)
    task = build_task(config, "continuous_bypass_64d", 4)
    materialize_success(task, 100)
    sweep = json.loads(Path(task["sweep"]).read_text(encoding="utf-8"))
    sweep["mse_ranking"][0]["promotion"] = {
        "eligible": False,
        "reasons": ["usage metric is not meaningful for continuous bypass"],
    }
    Path(task["sweep"]).write_text(json.dumps(sweep), encoding="utf-8")
    result = validate_task(task, formal=True)

    assert result["valid"] is True

    Path(task["best_checkpoint"]).unlink()
    result = validate_task(task, formal=True)
    assert result["valid"] is False
    assert "best_checkpoint missing" in result["reasons"]


def test_dry_run_plans_without_creating_output(tmp_path):
    config = make_inputs(tmp_path)
    result = run_cohort(config, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["formal_result_eligible"] is True
    for task in result["tasks"]:
        assert Path(task["environment"]["NS_OUT"]).is_absolute()
        assert task["environment"]["NS_OUT"] == task["task_root"]
    assert not config.output_root.exists()


def test_repeated_run_retries_from_same_rolling_task_and_skips_completed_task(
    tmp_path, monkeypatch
):
    formal = make_inputs(tmp_path)
    config = RunConfig(
        repo_root=formal.repo_root,
        protocol_dir=formal.protocol_dir,
        breparg_root=formal.breparg_root,
        output_root=formal.output_root,
        python=formal.python,
        arms=(FORMAL_ARMS[0],),
        seeds=(3,),
        train_cap=16,
        val_cap=8,
        batch_size=4,
        epochs=2,
        smoke=True,
    )
    calls = []

    def interrupted(command, **kwargs):
        calls.append((command, kwargs["env"]))
        task = build_task(config, FORMAL_ARMS[0], 3)
        Path(task["rolling_checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
        Path(task["rolling_checkpoint"]).write_bytes(b"rolling")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", interrupted)
    first = run_cohort(config)
    assert first["status"] == "FAILED"
    assert len(calls) == 1
    assert calls[0][1]["NS_VQ_AUTO_RESUME"] == "1"

    def resumed(command, **kwargs):
        calls.append((command, kwargs["env"]))
        task = build_task(config, FORMAL_ARMS[0], 3)
        assert Path(task["rolling_checkpoint"]).is_file()
        materialize_success(task, 2)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", resumed)
    second = run_cohort(config)
    assert second["status"] == "COMPLETED"
    assert len(calls) == 2
    assert len(second["tasks"][0]["attempts"]) == 2

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("completed task reran")),
    )
    third = run_cohort(config)
    assert third["status"] == "COMPLETED"
