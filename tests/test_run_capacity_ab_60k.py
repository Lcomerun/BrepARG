from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import tools.run_capacity_ab_60k as launcher


def make_inputs(tmp_path, monkeypatch, *, smoke=False):
    repo = tmp_path / "repo"
    protocol = tmp_path / "protocol"
    breparg = tmp_path / "BrepARG"
    improvements = repo / "breparg_improvements"
    improvements.mkdir(parents=True)
    protocol.mkdir()
    breparg.mkdir()
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    for name in (
        "train.py", "training_stability.py", "vqvae_sampling.py",
        "vqvae_metrics.py", "vqvae_sample_cache.py", "cad_protocol.py",
        "sharded_data.py", "fsq_quantise.py",
    ):
        (improvements / name).write_text(f"# {name}\n", encoding="utf-8")
    (breparg / "quantise.py").write_text("# quantise\n", encoding="utf-8")
    (breparg / "model.py").write_text("# model\n", encoding="utf-8")
    split = b"test split"
    split_hash = hashlib.sha256(split).hexdigest()
    (protocol / "split.pkl").write_bytes(split)
    (protocol / "protocol_summary.json").write_text(json.dumps({
        "status": "VERIFIED",
        "protocol_sha256": launcher.FORMAL_PROTOCOL_SHA256,
        "split_pickle_sha256": split_hash,
        "parent_overlap_counts": {"train__val": 0, "train__test": 0, "val__test": 0},
    }), encoding="utf-8")
    monkeypatch.setattr(launcher, "FORMAL_SPLIT_PICKLE_SHA256", split_hash)
    monkeypatch.setattr(
        launcher, "git_source_identity",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    return launcher.RunConfig(
        repo_root=repo,
        protocol_dir=protocol,
        breparg_root=breparg,
        output_root=tmp_path / "capacity-output",
        python=python,
        smoke=smoke,
    )


def test_signed_scheduler_contract_includes_selection_metric(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    task = launcher.build_task(config, "vq_8192_64d_random", 3)

    assert task["signature_payload"]["scheduler"] == {
        "kind": "ReduceLROnPlateau",
        "metric": "curved_parent_mse",
        "factor": 0.5,
        "patience": 8,
        "threshold": 1e-5,
        "threshold_mode": "abs",
        "min_lr": 1e-6,
    }
    assert task["signature_payload"]["git"] == {
        "commit": "a" * 40, "dirty": False
    }


def test_validator_rejects_declared_curved_scheduler_running_on_global_val(
    tmp_path, monkeypatch
):
    config = make_inputs(tmp_path, monkeypatch)
    task = launcher.build_task(config, "vq_8192_64d_random", 3)
    materialize_success(task)
    payload = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
    payload["history"][0]["plateau_metric"] = "global_val"
    payload["history"][0]["scheduler_metric"] = 0.2
    Path(task["history"]).write_text(json.dumps(payload), encoding="utf-8")

    result = launcher.validate_task(task, formal=True)

    assert result["valid"] is False
    assert any("plateau_metric mismatch" in reason for reason in result["reasons"])
    assert any("scheduler_metric is not curved" in reason for reason in result["reasons"])


def test_validator_compares_scheduler_to_serialized_curved_precision(
    tmp_path, monkeypatch
):
    config = make_inputs(tmp_path, monkeypatch)
    task = launcher.build_task(config, "vq_8192_64d_random", 3)
    materialize_success(task)
    payload = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
    payload["history"][0]["val_parent_cluster_reconstruction_mse"][
        "surface_curved_proxy"
    ]["mse"] = 0.10000000495337164
    Path(task["history"]).write_text(json.dumps(payload), encoding="utf-8")
    rolling = torch.load(task["rolling_checkpoint"], map_location="cpu", weights_only=False)
    rolling["history"] = payload["history"]
    torch.save(rolling, task["rolling_checkpoint"])

    result = launcher.validate_task(task, formal=True)

    assert result["valid"] is True, result["reasons"]


def test_existing_state_rejects_environment_drift(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    path, state = launcher.ensure_state(config)
    state["tasks"][0]["environment"]["NS_VQ_PLATEAU_METRIC"] = "global_val"
    launcher.atomic_json(path, state)

    with pytest.raises(ValueError, match="environment mismatch"):
        launcher.ensure_state(config)


def inventory(task):
    return {
        "train": {"schema": "vq-exact-hash-inventory-v1", "count": task["signature_payload"]["train_cap"], "ordered_sha256": "1" * 64, "sorted_sha256": "2" * 64},
        "val": {"schema": "vq-exact-hash-inventory-v1", "count": task["signature_payload"]["val_cap"], "ordered_sha256": "3" * 64, "sorted_sha256": "4" * 64},
    }


def healthy_stage_usage():
    return {
        "stage1": {"tokens": 16, "unique_bins": 4, "coverage": 4 / 4096, "entropy_perplexity": 3.5, "usage_fraction": 3.5 / 4096},
        "stage2": {"tokens": 16, "unique_bins": 3, "coverage": 3 / 4096, "entropy_perplexity": 2.5, "usage_fraction": 2.5 / 4096},
    }


def healthy_epoch(task, epoch):
    record = {
        "epoch": epoch,
        "train_batches": 469,
        "finite_train_batches": 469,
        "skipped_train_batches": 0,
        "nonfinite_loss_batches": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_state_batches": 0,
        "val_batches": 94,
        "finite_val_batches": 94,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "gradients_finite": True,
        "training_state_finite": True,
        "grad_clip_active": True,
        "experiment_signature": task["signature"],
        "finite_state_audit": {"status": "finite"},
        "full_state_audits": 1,
        "per_batch_full_state_audits": 0,
        "plateau_metric": "curved_parent_mse",
        "plateau_value": 0.1,
        "scheduler_metric": 0.1,
        "val_parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": 0.1}
        },
    }
    if task["arm"] == "rvq_2x4096_64d_random":
        record.update(train_stage_code_usage=healthy_stage_usage(), val_stage_code_usage=healthy_stage_usage(), stage_usage_health={"healthy": True, "reasons": []})
    return record


def checkpoint_context(task, inv):
    return {
        "protocol_sha256": task["signature_payload"]["protocol_sha256"],
        "split_pickle_sha256": task["signature_payload"]["split_pickle_sha256"],
        "parent_overlap_counts": task["signature_payload"]["parent_overlap_counts"],
        "inventory": inv,
        "run_manifest": run_manifest(task, inv),
    }


def runtime():
    return {"python_implementation": "CPython", "python_version": [3, 10, 0]}


def run_manifest(task, inv):
    return {
        "git": task["signature_payload"]["git"],
        "runtime_resume_compatibility": runtime(),
        "experiment": {
            "seed": task["seed"], "train_cap": task["signature_payload"]["train_cap"],
            "val_cap": task["signature_payload"]["val_cap"], "epochs": task["signature_payload"]["epochs"],
            "batch_size": task["signature_payload"]["batch_size"],
            "protocol": {key: task["signature_payload"][key] for key in ("protocol_sha256", "split_pickle_sha256", "parent_overlap_counts")},
            "inventory": inv, **task["signature_payload"]["sampling"],
            "curved_loss_weight": task["signature_payload"]["loss"]["curved_loss_weight"],
            "complex_loss_weight": task["signature_payload"]["loss"]["complex_loss_weight"],
            "curved_loss_threshold": task["signature_payload"]["loss"]["curved_loss_threshold"],
            **task["signature_payload"]["stop"],
            "arms": [{"name": task["arm"], "codebook": launcher.arm_codebook_size(task["arm"]), "quantizer": launcher.arm_metadata(task["arm"])}],
        },
    }


def signature_configuration(task, inv):
    return {
        "git": task["signature_payload"]["git"],
        "seed": task["seed"], "train_cap": task["signature_payload"]["train_cap"],
        "val_cap": task["signature_payload"]["val_cap"], "epochs": task["signature_payload"]["epochs"],
        "batch_size": task["signature_payload"]["batch_size"], "lr": 3e-4,
        "precision": task["signature_payload"]["precision"], "grad_clip_norm": 1.0,
        "protocol": {key: task["signature_payload"][key] for key in ("protocol_sha256", "split_pickle_sha256", "parent_overlap_counts")},
        "inventory": inv, "runtime_resume_compatibility": runtime(),
        "arm": {"name": task["arm"], "quantizer": launcher.arm_metadata(task["arm"])},
        "scheduler": task["signature_payload"]["scheduler"],
        "sampling": task["signature_payload"]["sampling"],
        "loss": task["signature_payload"]["loss"],
        "stop": task["signature_payload"]["stop"],
    }


def capacity_model_state(task):
    if task["arm"] == "vq_8192_64d_random":
        return {"quantize.quantizer.embedding.weight": torch.zeros(8192, 64)}
    return {
        "quantize.stage1.quantizer.embedding.weight": torch.zeros(4096, 64),
        "quantize.stage2.quantizer.embedding.weight": torch.zeros(4096, 64),
    }


def materialize_success(task):
    root = Path(task["task_root"])
    root.mkdir(parents=True, exist_ok=True)
    records = [healthy_epoch(task, epoch) for epoch in range(task["signature_payload"]["epochs"])]
    Path(task["history"]).write_text(json.dumps({
        "config": {"experiment_signature": task["signature"], "precision": {"name": task["signature_payload"]["precision"]}, "strict_nonfinite": True, "grad_clip_norm": 1.0, "plateau_metric": "curved_parent_mse", "scheduler": task["signature_payload"]["scheduler"], "signature_configuration": signature_configuration(task, inventory(task)), "quantizer": launcher.arm_metadata(task["arm"])},
        "history": records,
    }), encoding="utf-8")
    inv = inventory(task)
    manifest = run_manifest(task, inv)
    Path(task["sweep"]).write_text(json.dumps({
        "run_manifest": manifest,
        "mse_ranking": [{"name": task["arm"], "epochs_ran": 100, "final_checkpoint_epoch": 99,
                         "checkpoint_epoch": 99, "experiment_signature": task["signature"],
                         "inventory": inv, "train_sampling": {"selected": 60000}, "val_sampling": {"selected": 12000}}],
    }), encoding="utf-8")
    normal = {
        "model_state_dict": capacity_model_state(task), "fsq_levels": [],
        "quantizer": launcher.arm_metadata(task["arm"]), "checkpoint_epoch": 99,
        "checkpoint_context": checkpoint_context(task, inv),
        "val_stage_code_usage": healthy_stage_usage() if task["arm"].startswith("rvq_") else None,
    }
    torch.save(normal, task["best_checkpoint"])
    torch.save(normal, task["final_checkpoint"])
    pools = {}
    if task["arm"].startswith("rvq_"):
        pools = {
            "quantize.stage1.quantizer": {"features": torch.zeros(2, 2, dtype=torch.float32), "nums_features": 2},
            "quantize.stage2.quantizer": {"features": torch.zeros(2, 2, dtype=torch.float32), "nums_features": 2},
        }
    else:
        pools = {
            "quantize": {"features": torch.zeros(2, 2, dtype=torch.float32), "nums_features": 2},
        }
    torch.save({
        "checkpoint_schema": "vq_training_state_v1", "experiment_signature": task["signature"], "epoch": 99,
        "model_state_dict": capacity_model_state(task),
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 3e-4}]},
        "scaler_state_dict": None, "scheduler_state_dict": {"best": 0.1},
        "stop_state": {"best_val": 0.1, "best_epoch": 99}, "plateau_state": {"best_val": 0.1, "best_epoch": 99},
        "history": records,
        "rng_state": {"python": None, "numpy": None, "torch_cpu": torch.zeros(1, dtype=torch.uint8), "torch_cuda": None},
        "feature_pool_state": pools, "finite_state_audit": {"status": "finite"},
        "extra": {"signature_configuration": signature_configuration(task, inv), "precision": {"name": task["signature_payload"]["precision"]}, "selector_state": {"stage_usage_health": {"healthy": True, "reasons": []}}, "meta": {}},
    }, task["rolling_checkpoint"])


def test_formal_capacity_matrix_is_frozen_bf16_and_has_four_signed_tasks(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    assert config.arms == launcher.FORMAL_ARMS
    assert config.seeds == launcher.FORMAL_SEEDS
    assert config.precision == "bf16"
    with pytest.raises(ValueError, match="immutable"):
        launcher.RunConfig(repo_root=config.repo_root, protocol_dir=config.protocol_dir, breparg_root=config.breparg_root, output_root=tmp_path / "changed", python=config.python, precision="fp32")

    state = launcher.build_state(config)
    assert [task["task_id"] for task in state["tasks"]] == [
        "vq_8192_64d_random:seed3", "vq_8192_64d_random:seed4",
        "rvq_2x4096_64d_random:seed3", "rvq_2x4096_64d_random:seed4",
    ]
    for task in state["tasks"]:
        assert task["environment"]["NS_VQ_PRECISION"] == "bf16"
        assert task["environment"]["NS_VQ_BS"] == "128"
        assert task["environment"]["NS_VQ_SAMPLES"] == "60000"
        assert task["environment"]["NS_VQ_VAL_SAMPLES"] == "12000"
        assert task["environment"]["NS_VQ_SWEEP_EPOCHS"] == "100"
        assert task["environment"]["NS_VQ_AUTO_RESUME"] == "1"
        assert task["environment"]["NS_VQ_STRICT_NONFINITE"] == "1"
        assert task["environment"]["NS_VQ_PLATEAU_METRIC"] == "curved_parent_mse"
        assert task["signature_payload"]["loss"] == {
            "reconstruction": "weighted_mse",
            "curved_loss_weight": 1.0,
            "complex_loss_weight": 1.0,
            "curved_loss_threshold": 0.02,
        }


def test_dry_run_is_non_mutating_and_protocol_bound(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    state = launcher.run_cohort(config, dry_run=True)
    assert state["status"] == "DRY_RUN"
    assert all(task["status"] == "PLANNED" for task in state["tasks"])
    assert not config.output_root.exists()


def test_formal_validator_requires_complete_rvq_stage_evidence_and_pools(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    task = launcher.build_task(config, "rvq_2x4096_64d_random", 3)
    materialize_success(task)
    assert launcher.validate_task(task, formal=True)["valid"] is True

    history = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
    history["history"][10]["val_stage_code_usage"]["stage2"]["unique_bins"] = 1
    history["history"][10]["stage_usage_health"] = {"healthy": False, "reasons": ["stage2 collapsed"]}
    Path(task["history"]).write_text(json.dumps(history), encoding="utf-8")
    result = launcher.validate_task(task, formal=True)
    assert result["valid"] is False
    assert any("stage2" in reason for reason in result["reasons"])

    materialize_success(task)
    rolling = torch.load(task["rolling_checkpoint"], map_location="cpu", weights_only=False)
    rolling["feature_pool_state"].pop("quantize.stage2.quantizer")
    torch.save(rolling, task["rolling_checkpoint"])
    result = launcher.validate_task(task, formal=True)
    assert result["valid"] is False
    assert any("stage2 FeaturePool" in reason for reason in result["reasons"])


def test_summary_requires_identical_exact_inventory_across_all_four_tasks(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    state = launcher.build_state(config)
    for task in state["tasks"]:
        materialize_success(task)
    summary = launcher.validation_summary(state)
    assert summary["valid"] is True
    assert summary["inventory_consistent"] is True

    payload = json.loads(Path(state["tasks"][3]["sweep"]).read_text(encoding="utf-8"))
    payload["mse_ranking"][0]["inventory"]["train"]["ordered_sha256"] = "9" * 64
    Path(state["tasks"][3]["sweep"]).write_text(json.dumps(payload), encoding="utf-8")
    summary = launcher.validation_summary(state)
    assert summary["valid"] is False
    assert summary["inventory_consistent"] is False


def test_validator_rehashes_sources_and_existing_state_rejects_task_drift(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    task = launcher.build_task(config, "vq_8192_64d_random", 3)
    materialize_success(task)
    assert launcher.validate_task(task, formal=True)["valid"] is True

    train_source = config.repo_root / "breparg_improvements" / "train.py"
    train_source.write_text("# changed after signing\n", encoding="utf-8")
    result = launcher.validate_task(task, formal=True)
    assert result["valid"] is False
    assert "signed source SHA-256 set mismatch" in result["reasons"]

    # Restore the signed bytes, create state, then prove that a rewritten task
    # signature cannot be resumed from the same output root.
    train_source.write_text("# train.py\n", encoding="utf-8")
    state_config = launcher.RunConfig(
        repo_root=config.repo_root,
        protocol_dir=config.protocol_dir,
        breparg_root=config.breparg_root,
        output_root=tmp_path / "state-output",
        python=config.python,
    )
    state_path, state = launcher.ensure_state(state_config)
    state["tasks"][0]["signature"] = "0" * 64
    launcher.atomic_json(state_path, state)
    with pytest.raises(ValueError, match="task signature mismatch"):
        launcher.ensure_state(state_config)


def test_verify_inputs_rejects_breparg_path_that_train_will_not_discover(tmp_path, monkeypatch):
    config = make_inputs(tmp_path, monkeypatch)
    discoverable = config.repo_root / "breparg_improvements" / "BrepARG"
    discoverable.mkdir()
    (discoverable / "model.py").write_text("# model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train.py will discover BrepARG"):
        launcher.verify_inputs(config)
