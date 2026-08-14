import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tools.run_p0b_stability_retest as launcher

from tools.run_p0b_stability_retest import (
    FORMAL_ARMS,
    FORMAL_PROTOCOL_SHA256,
    FORMAL_SEEDS,
    FORMAL_SPLIT_PICKLE_SHA256,
    RunConfig,
    build_state,
    build_task,
    load_and_refresh,
    main,
    output_root_writer_lock,
    run_cohort,
    validate_task,
    validation_summary,
    verify_protocol,
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
    (repo / "breparg_improvements" / "vqvae_sampling.py").write_text(
        "# sampling\n", encoding="utf-8"
    )
    split_payload = b"split"
    split_sha256 = hashlib.sha256(split_payload).hexdigest()
    (protocol / "split.pkl").write_bytes(split_payload)
    (protocol / "protocol_summary.json").write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": FORMAL_PROTOCOL_SHA256,
                "split_pickle_sha256": split_sha256,
                "parent_overlap_counts": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
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
        "nonfinite_state_audits": 0,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "gradients_finite": True,
        "training_state_finite": True,
        "finite_state_audit_cadence": "lifecycle_v1",
        "finite_state_audit": {"status": "finite"},
        "full_state_audits": 1,
        "per_batch_full_state_audits": 0,
        "grad_clip_active": True,
        "preclip_grad_norm": 1.25,
        "experiment_signature": signature,
    }


def quantizer_metadata(task: dict) -> dict:
    if task["arm"] == "vq_4096_64d_random":
        return {
            "kind": "learned_vq",
            "implementation": "BrepARG.quantise.VectorQuantiser",
            "codebook_size": 4096,
            "embedding_dim": 64,
            "distance": "cos",
            "anchor": "random",
            "first_batch": False,
            "contrastive_loss": True,
            "decay": 0.99,
            "downstream_compatible": False,
        }
    return {
        "kind": "continuous_bypass",
        "embedding_dim": 64,
        "usage_metric_meaningful": False,
        "downstream_compatible": False,
    }


def arm_codebook_size(task: dict) -> int:
    return 4096 if task["arm"] == "vq_4096_64d_random" else 1


def checkpoint_protocol(task: dict) -> dict:
    return {
        "protocol_sha256": task["signature_payload"]["protocol_sha256"],
        "split_pickle_sha256": task["signature_payload"]["split_pickle_sha256"],
        "parent_overlap_counts": task["signature_payload"]["parent_overlap_counts"],
    }


def patch_inventory(task: dict, *, train_count=None, val_count=None) -> dict:
    return {
        "train": {
            "schema": "vq-exact-hash-inventory-v1",
            "count": (
                task["signature_payload"]["train_cap"]
                if train_count is None
                else train_count
            ),
            "ordered_sha256": "1" * 64,
            "sorted_sha256": "2" * 64,
        },
        "val": {
            "schema": "vq-exact-hash-inventory-v1",
            "count": (
                task["signature_payload"]["val_cap"]
                if val_count is None
                else val_count
            ),
            "ordered_sha256": "3" * 64,
            "sorted_sha256": "4" * 64,
        },
    }


def run_manifest(task: dict) -> dict:
    quantizer = quantizer_metadata(task)
    return {
        "git": {"commit": "a" * 40, "dirty": False},
        "experiment": {
            "seed": task["seed"],
            "protocol": checkpoint_protocol(task),
            "inventory": patch_inventory(task),
            "train_cap": task["signature_payload"]["train_cap"],
            "val_cap": task["signature_payload"]["val_cap"],
            "epochs": task["signature_payload"]["epochs"],
            "batch_size": task["signature_payload"]["batch_size"],
            "min_parent_coverage": task["signature_payload"]["sampling"][
                "min_parent_coverage"
            ],
            "balance_by_parent": task["signature_payload"]["sampling"][
                "balance_by_parent"
            ],
            "deduplicate_before_cap": task["signature_payload"]["sampling"][
                "deduplicate_before_cap"
            ],
            "require_exact_caps": task["signature_payload"]["sampling"][
                "require_exact_caps"
            ],
            "arms": [
                {
                    "name": task["arm"],
                    "levels": [],
                    "codebook": arm_codebook_size(task),
                    "lr": float(task["signature_payload"]["learning_rate"]),
                    "quantizer": quantizer,
                }
            ],
        },
    }


def checkpoint_context(task: dict) -> dict:
    return {
        **checkpoint_protocol(task),
        "inventory": patch_inventory(task),
        "git_commit": "a" * 40,
        "train_parent_coverage": 0.95,
        "val_parent_coverage": 1.0,
        "run_manifest": run_manifest(task),
    }


def signature_configuration(task: dict) -> dict:
    quantizer = quantizer_metadata(task)
    return {
        "git": {"commit": "a" * 40, "dirty": False},
        "protocol": checkpoint_protocol(task),
        "inventory": patch_inventory(task),
        "arm": {
            "name": task["arm"],
            "levels": [],
            "codebook": arm_codebook_size(task),
            "lr": float(task["signature_payload"]["learning_rate"]),
            "quantizer": quantizer,
        },
        "seed": task["seed"],
        "train_cap": task["signature_payload"]["train_cap"],
        "val_cap": task["signature_payload"]["val_cap"],
        "epochs": task["signature_payload"]["epochs"],
        "batch_size": task["signature_payload"]["batch_size"],
        "lr": float(task["signature_payload"]["learning_rate"]),
        "precision": task["signature_payload"]["precision"],
    }


def normal_checkpoint(task: dict, epoch: int) -> dict:
    return {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": [],
        "quantizer": quantizer_metadata(task),
        "checkpoint_epoch": epoch,
        "validation_metrics": {},
        "checkpoint_context": checkpoint_context(task),
        "validation_loss": 0.1,
        "checkpoint_selection": {"selected": True, "reasons": []},
    }


def materialize_success(task: dict, epochs: int) -> None:
    root = Path(task["task_root"])
    root.mkdir(parents=True, exist_ok=True)
    history = [healthy_epoch(epoch, task["signature"]) for epoch in range(epochs)]
    Path(task["history"]).write_text(
        json.dumps(
            {
                "config": {
                    "experiment_signature": task["signature"],
                    "precision": {"name": task["signature_payload"]["precision"]},
                    "strict_nonfinite": True,
                    "grad_clip_norm": 1.0,
                    "finite_state_audit_cadence": {
                        "policy": "lifecycle_v1",
                        "startup_or_post_resume": True,
                        "epoch_end_pre_save": True,
                        "per_train_batch": False,
                    },
                    "scheduler": {
                        "factor": 0.5,
                        "patience": 8,
                        "threshold": 1e-5,
                        "threshold_mode": "abs",
                        "min_lr": 1e-6,
                    },
                },
                "history": history,
            }
        ),
        encoding="utf-8",
    )
    Path(task["sweep"]).write_text(
        json.dumps(
            {
                "run_manifest": run_manifest(task),
                "mse_ranking": [
                    {
                        "name": task["arm"],
                        "epochs_ran": epochs,
                        "checkpoint_epoch": epochs - 1,
                        "final_checkpoint_epoch": epochs - 1,
                        "experiment_signature": task["signature"],
                        "precision": {
                            "name": task["signature_payload"]["precision"]
                        },
                        "inventory": patch_inventory(task),
                        "train_sampling": {
                            "selected": task["signature_payload"]["train_cap"]
                        },
                        "val_sampling": {
                            "selected": task["signature_payload"]["val_cap"]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    terminal_epoch = epochs - 1
    torch.save(normal_checkpoint(task, terminal_epoch), task["best_checkpoint"])
    torch.save(normal_checkpoint(task, terminal_epoch), task["final_checkpoint"])
    torch.save(
        {
            "checkpoint_schema": "vq_training_state_v1",
            "experiment_signature": task["signature"],
            "epoch": terminal_epoch,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 3e-4}]},
            "scaler_state_dict": None,
            "scheduler_state_dict": {"best": 0.1, "num_bad_epochs": 0},
            "stop_state": {"best_val": 0.1, "best_epoch": terminal_epoch},
            "plateau_state": {"best_val": 0.1, "best_epoch": terminal_epoch},
            "history": history,
            "rng_state": {
                "python": (3, (), None),
                "numpy": ("MT19937", torch.zeros(1).numpy(), 0, 0, 0.0),
                "torch_cpu": torch.zeros(1, dtype=torch.uint8),
                "torch_cuda": None,
            },
            "feature_pool_state": {},
            "finite_state_audit": {"status": "finite"},
            "extra": {
                "signature_configuration": signature_configuration(task),
                "precision": {"name": task["signature_payload"]["precision"]},
                "selector_state": {},
                "meta": {},
            },
        },
        task["rolling_checkpoint"],
    )


def test_formal_protocol_is_fixed_and_smoke_overrides_are_explicit(tmp_path):
    config = make_inputs(tmp_path)
    assert config.arms == FORMAL_ARMS
    assert config.seeds == FORMAL_SEEDS
    assert config.train_cap == 60_000
    assert config.val_cap == 12_000
    assert config.batch_size == 128
    assert config.epochs == 100
    assert config.learning_rate == "3e-4"
    assert FORMAL_PROTOCOL_SHA256 == (
        "6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3"
    )
    assert FORMAL_SPLIT_PICKLE_SHA256 == (
        "6ff0a0c3ee6a04ee056fa1ab982eb436a9f59d3d21f21f17babf34e6dc701d29"
    )

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
        assert task["signature_payload"]["sampling"] == {
            "balance_by_parent": True,
            "deduplicate_before_cap": True,
            "require_exact_caps": True,
            "min_parent_coverage": 0.9,
            "curved_fraction": 0.0,
            "complex_fraction": 0.0,
        }
        assert env["NS_VQ_MIN_PARENT_COVERAGE"] == "0.9"
        assert env["NS_VQ_REQUIRE_EXACT_CAPS"] == "1"
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


def test_validate_task_requires_lifecycle_finite_state_evidence(tmp_path):
    config = make_inputs(tmp_path)
    task = build_task(config, FORMAL_ARMS[0], 3)
    materialize_success(task, 100)
    payload = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
    payload["history"][12]["per_batch_full_state_audits"] = 1
    payload["history"][12]["finite_state_audit"] = {"status": "nonfinite"}
    Path(task["history"]).write_text(json.dumps(payload), encoding="utf-8")

    result = validate_task(task, formal=True)

    assert result["valid"] is False
    assert any("per-batch full state audits" in reason for reason in result["reasons"])
    assert any("finite-state audit evidence" in reason for reason in result["reasons"])


def test_validate_task_requires_realized_caps_and_checkpoint_contents(tmp_path):
    config = make_inputs(tmp_path)
    task = build_task(config, FORMAL_ARMS[0], 3)
    materialize_success(task, 100)

    sweep = json.loads(Path(task["sweep"]).read_text(encoding="utf-8"))
    sweep["mse_ranking"][0]["train_sampling"]["selected"] = 59_999
    Path(task["sweep"]).write_text(json.dumps(sweep), encoding="utf-8")
    result = validate_task(task, formal=True)
    assert result["valid"] is False
    assert "sweep realized train patch count mismatch" in result["reasons"]
    assert "formal realized patch counts must be exactly 60000/12000" in result["reasons"]


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


def test_dry_run_plans_without_creating_output(tmp_path, monkeypatch):
    config = make_inputs(tmp_path)
    monkeypatch.setattr(
        launcher,
        "FORMAL_SPLIT_PICKLE_SHA256",
        hashlib.sha256((config.protocol_dir / "split.pkl").read_bytes()).hexdigest(),
    )
    result = run_cohort(config, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["formal_result_eligible"] is True
    for task in result["tasks"]:
        assert Path(task["environment"]["NS_OUT"]).is_absolute()
        assert task["environment"]["NS_OUT"] == task["task_root"]
    assert not config.output_root.exists()


def test_probe_disables_impossible_parent_coverage_gate(tmp_path):
    formal = make_inputs(tmp_path)
    smoke = RunConfig(
        repo_root=formal.repo_root,
        protocol_dir=formal.protocol_dir,
        breparg_root=formal.breparg_root,
        output_root=tmp_path / "probe",
        python=formal.python,
        arms=FORMAL_ARMS,
        seeds=(3,),
        train_cap=128,
        val_cap=128,
        batch_size=8,
        epochs=1,
        smoke=True,
    )

    state = build_state(smoke)

    for task in state["tasks"]:
        assert task["signature_payload"]["sampling"]["balance_by_parent"] is False
        assert task["signature_payload"]["sampling"]["deduplicate_before_cap"] is False
        assert task["signature_payload"]["sampling"]["require_exact_caps"] is False
        assert task["signature_payload"]["sampling"]["min_parent_coverage"] == 0.0
        assert task["environment"]["NS_VQ_BALANCE_BY_PARENT"] == "0"
        assert task["environment"]["NS_VQ_DEDUP_BEFORE_CAP"] == "0"
        assert task["environment"]["NS_VQ_REQUIRE_EXACT_CAPS"] == "0"
        assert task["environment"]["NS_VQ_MIN_PARENT_COVERAGE"] == "0.0"


def test_probe_inventory_accepts_positive_realized_counts_below_requested_cap(tmp_path):
    formal = make_inputs(tmp_path)
    smoke = RunConfig(
        repo_root=formal.repo_root,
        protocol_dir=formal.protocol_dir,
        breparg_root=formal.breparg_root,
        output_root=tmp_path / "probe-realized-counts",
        python=formal.python,
        arms=(FORMAL_ARMS[0],),
        seeds=(3,),
        train_cap=128,
        val_cap=128,
        batch_size=8,
        epochs=1,
        smoke=True,
    )
    task = build_state(smoke)["tasks"][0]
    observed = patch_inventory(task, train_count=116, val_count=90)
    reasons = []

    normalized = launcher._validate_inventory(
        observed, task, prefix="probe", reasons=reasons
    )

    assert normalized == observed
    assert reasons == []


def test_formal_inventory_still_requires_exact_requested_counts(tmp_path):
    task = build_state(make_inputs(tmp_path))["tasks"][0]
    observed = patch_inventory(task, train_count=59_999, val_count=12_000)
    reasons = []

    assert launcher._validate_inventory(
        observed, task, prefix="formal", reasons=reasons
    ) is None
    assert reasons == ["formal train inventory count mismatch"]


def test_formal_protocol_verifier_checks_hash_status_and_integer_zero_overlaps(
    tmp_path, monkeypatch
):
    config = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen Protocol V5 split"):
        verify_protocol(config)

    fake_split_sha256 = hashlib.sha256(
        (config.protocol_dir / "split.pkl").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(launcher, "FORMAL_SPLIT_PICKLE_SHA256", fake_split_sha256)
    assert verify_protocol(config)["status"] == "VERIFIED"

    summary_path = config.protocol_dir / "protocol_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parent_overlap_counts"]["train__val"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON integer zero"):
        verify_protocol(config)


@pytest.mark.parametrize(
    ("corruption", "reason_fragment"),
    [
        ("unreadable_best", "best_checkpoint unreadable"),
        ("rolling_signature", "rolling_checkpoint experiment signature mismatch"),
        ("rolling_incomplete", "rolling_checkpoint incomplete: optimizer_state_dict"),
        ("final_wrong_arm", "final_checkpoint quantizer kind mismatch"),
        ("best_wrong_seed", "best_checkpoint run_manifest seed mismatch"),
        ("best_wrong_protocol", "best_checkpoint protocol_sha256 mismatch"),
    ],
)
def test_validate_task_rejects_corrupt_or_misbound_checkpoints(
    tmp_path, corruption, reason_fragment
):
    config = make_inputs(tmp_path)
    task = build_task(config, FORMAL_ARMS[0], 3)
    materialize_success(task, 100)

    if corruption == "unreadable_best":
        Path(task["best_checkpoint"]).write_bytes(b"not-a-torch-checkpoint")
    elif corruption in {"rolling_signature", "rolling_incomplete"}:
        path = Path(task["rolling_checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if corruption == "rolling_signature":
            payload["experiment_signature"] = "wrong-signature"
        else:
            payload.pop("optimizer_state_dict")
        torch.save(payload, path)
    else:
        path = Path(task["final_checkpoint" if corruption == "final_wrong_arm" else "best_checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if corruption == "final_wrong_arm":
            payload["quantizer"]["kind"] = "continuous_bypass"
        elif corruption == "best_wrong_seed":
            payload["checkpoint_context"]["run_manifest"]["experiment"]["seed"] = 4
        else:
            payload["checkpoint_context"]["protocol_sha256"] = "wrong-protocol"
        torch.save(payload, path)

    result = validate_task(task, formal=True)
    assert result["valid"] is False
    assert any(reason_fragment in reason for reason in result["reasons"])


def test_formal_validation_rejects_inventory_drift_across_tasks(tmp_path):
    config = make_inputs(tmp_path)
    state = build_state(config)
    for task in state["tasks"]:
        materialize_success(task, 100)

    drifted_task = state["tasks"][-1]
    drifted = patch_inventory(drifted_task)
    drifted["train"]["ordered_sha256"] = "a" * 64
    sweep_path = Path(drifted_task["sweep"])
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["mse_ranking"][0]["inventory"] = drifted
    sweep["run_manifest"]["experiment"]["inventory"] = drifted
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    for field in ("best_checkpoint", "final_checkpoint"):
        path = Path(drifted_task[field])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["checkpoint_context"]["inventory"] = drifted
        payload["checkpoint_context"]["run_manifest"]["experiment"]["inventory"] = drifted
        torch.save(payload, path)
    rolling_path = Path(drifted_task["rolling_checkpoint"])
    rolling = torch.load(rolling_path, map_location="cpu", weights_only=False)
    rolling["extra"]["signature_configuration"]["inventory"] = drifted
    torch.save(rolling, rolling_path)

    assert validate_task(drifted_task, formal=True)["valid"] is True
    refreshed = launcher.refresh_state(state)
    assert refreshed["status"] == "FAILED"
    assert refreshed["inventory_consistent"] is False
    summary = validation_summary(state)
    assert summary["valid"] is False
    assert summary["inventory_consistent"] is False
    assert "formal task inventories are missing or differ" in summary["reasons"][-1]


def test_output_root_writer_lock_is_cross_process_and_recovers_abnormal_exit(tmp_path):
    output_root = tmp_path / "locked-output"
    repo_root = Path(__file__).resolve().parents[1]
    holder_code = """
import sys
from pathlib import Path
from tools.run_p0b_stability_retest import output_root_writer_lock
with output_root_writer_lock(Path(sys.argv[1]), command=["holder", "--formal"]):
    print("LOCKED", flush=True)
    sys.stdin.readline()
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(output_root)],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        marker = holder.stdout.readline()
        assert marker == "LOCKED\n", holder.stderr.read()
        with pytest.raises(RuntimeError, match="already has an active writer"):
            with output_root_writer_lock(output_root, command=["second-writer"]):
                pass
    finally:
        if holder.stdin:
            holder.stdin.write("release\n")
            holder.stdin.flush()
        holder.wait(timeout=10)
    assert holder.returncode == 0

    crash_code = """
import os
import sys
from pathlib import Path
from tools.run_p0b_stability_retest import output_root_writer_lock
with output_root_writer_lock(Path(sys.argv[1]), command=["crashing-holder"]):
    print("LOCKED", flush=True)
    os._exit(0)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_code, str(output_root)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert crashed.returncode == 0
    assert crashed.stdout == "LOCKED\n"

    with output_root_writer_lock(output_root, command=["recovery", "--resume"]) as lock:
        assert lock["owner"]["pid"] == os.getpid()
        assert lock["command"] == ["recovery", "--resume"]
        assert lock["acquired_at"]
        assert lock["unreleased_lock_recovered"] is True
        assert lock["stale_lock_recovered"] is True


def test_status_and_validate_are_read_only(tmp_path, capsys):
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
    state = build_state(config)
    materialize_success(state["tasks"][0], 2)
    state_path = config.output_root / "p0b_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    original_state = state_path.read_bytes()
    original_files = sorted(
        str(path.relative_to(config.output_root))
        for path in config.output_root.rglob("*")
        if path.is_file()
    )

    _, refreshed = load_and_refresh(config.output_root)
    assert refreshed["status"] == "COMPLETED"
    assert main(["status", "--output-root", str(config.output_root)]) == 0
    capsys.readouterr()
    assert main(["validate", "--output-root", str(config.output_root)]) == 0
    capsys.readouterr()

    assert state_path.read_bytes() == original_state
    assert not (config.output_root / "p0b_validation.json").exists()
    assert not (config.output_root / launcher.WRITER_LOCK_NAME).exists()
    assert sorted(
        str(path.relative_to(config.output_root))
        for path in config.output_root.rglob("*")
        if path.is_file()
    ) == original_files


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
