import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "breparg_improvements"))

import training_stability as stability_module
from training_stability import (
    NonFiniteTrainingError,
    VQVAEStopConfig,
    VQVAEStopState,
    assert_finite_training_state,
    audit_finite_training_state,
    atomic_torch_save,
    build_experiment_signature,
    capture_training_checkpoint,
    clip_gradients_strict,
    load_training_checkpoint,
    resolve_precision,
    restore_training_checkpoint,
    update_vqvae_stop_state,
)


class PoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.pool = SimpleNamespace(
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            nums_features=1,
        )

    def forward(self, value):
        return self.linear(value)


def test_precision_policy_is_explicit_and_scaler_is_fp16_only():
    fp32 = resolve_precision("fp32", cuda_available=True, bf16_supported=True)
    fp16 = resolve_precision("fp16", cuda_available=True, bf16_supported=True)
    bf16 = resolve_precision("bf16", cuda_available=True, bf16_supported=True)

    assert fp32.autocast_dtype is None
    assert fp16.autocast_dtype is torch.float16 and fp16.grad_scaler_enabled
    assert bf16.autocast_dtype is torch.bfloat16 and not bf16.grad_scaler_enabled
    with pytest.raises(ValueError, match="CUDA"):
        resolve_precision("fp16", cuda_available=False)
    with pytest.raises(ValueError, match="does not support"):
        resolve_precision("bf16", cuda_available=True, bf16_supported=False)


def test_nonfinite_fuse_ignores_minimum_epoch_gate():
    state = VQVAEStopState()
    config = VQVAEStopConfig(
        min_epochs=100,
        patience=100,
        max_nonfinite_val_epochs=1,
    )

    state, improved, should_stop = update_vqvae_stop_state(
        3, float("inf"), state, config
    )

    assert not improved
    assert should_stop
    assert state.stop_reason == "nonfinite_val_epochs=1"


def test_strict_gradient_clip_reports_norm_and_rejects_nonfinite():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    norm = clip_gradients_strict(model, max_norm=0.1)
    assert np.isfinite(norm)
    assert norm > 0.1
    post_clip_norm = torch.linalg.vector_norm(
        torch.stack(
            [parameter.grad.detach().norm(2) for parameter in model.parameters()]
        ),
        2,
    )
    assert float(post_clip_norm) <= 0.100001

    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingError, match="weight"):
        clip_gradients_strict(model, max_norm=0.1)


def test_finite_state_check_includes_feature_pool_and_optimizer():
    model = PoolModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert_finite_training_state(model, optimizer)

    model.pool.features[0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingError, match="pool.features"):
        assert_finite_training_state(model, optimizer)

    model.pool.features.zero_()
    optimizer.state[next(iter(model.parameters()))]["exp_avg"][0, 0] = float("inf")
    with pytest.raises(NonFiniteTrainingError, match="optimizer"):
        assert_finite_training_state(model, optimizer)


def test_finite_state_audit_aggregates_normal_path_to_one_scalar_check_per_device(
    monkeypatch,
):
    model = PoolModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    def unexpected_detailed_scan(*_args, **_kwargs):
        raise AssertionError("finite audit used the per-tensor failure path")

    monkeypatch.setattr(
        stability_module, "_assert_finite_tensor", unexpected_detailed_scan
    )
    report = audit_finite_training_state(model, optimizer)

    assert report["status"] == "finite"
    assert report["devices"] == ["cpu"]
    assert report["scalar_device_checks"] == 1
    assert report["tensors"] > report["scalar_device_checks"]
    assert report["sources"]["model"]["tensors"] == 2
    assert report["sources"]["feature_pool"]["tensors"] == 1
    assert report["sources"]["optimizer"]["tensors"] == 6


def test_full_checkpoint_round_trip_restores_rng_pool_and_training_state(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = PoolModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step(0.5)
    signature = build_experiment_signature({"arm": "vq", "seed": 3})
    payload = capture_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        epoch=4,
        history=[{"epoch": 4, "val_loss": 0.5}],
        stop_state=VQVAEStopState(best_val=0.5, best_epoch=4),
        plateau_state=VQVAEStopState(best_val=0.5, best_epoch=4),
        experiment_signature=signature,
        extra={"precision": "fp32"},
    )
    assert payload["finite_state_audit"]["status"] == "finite"
    assert payload["finite_state_audit"]["scalar_device_checks"] == 1
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "rolling.pt"
    atomic_torch_save(payload, checkpoint)

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.pool.features.zero_()
    model.pool.nums_features = 0

    loaded = load_training_checkpoint(
        checkpoint, expected_signature=signature, map_location="cpu"
    )
    restored = restore_training_checkpoint(
        loaded,
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        expected_signature=signature,
    )

    assert restored["epoch"] == 4
    assert restored["history"] == [{"epoch": 4, "val_loss": 0.5}]
    assert restored["stop_state"] == VQVAEStopState(best_val=0.5, best_epoch=4)
    assert restored["finite_state_audit"]["status"] == "finite"
    assert model.pool.nums_features == 1
    assert torch.equal(
        model.pool.features, torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_parameters[name])
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_checkpoint_signature_mismatch_is_fail_closed(tmp_path):
    path = tmp_path / "rolling.pt"
    atomic_torch_save(
        {
            "checkpoint_schema": "vq_training_state_v1",
            "experiment_signature": build_experiment_signature({"seed": 3}),
        },
        path,
    )
    with pytest.raises(ValueError, match="signature"):
        load_training_checkpoint(
            path,
            expected_signature=build_experiment_signature({"seed": 4}),
        )


def test_restore_rejects_missing_runtime_scheduler(tmp_path):
    model = PoolModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    signature = build_experiment_signature({"seed": 3})
    payload = capture_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        epoch=0,
        history=[],
        stop_state=VQVAEStopState(),
        plateau_state=VQVAEStopState(),
        experiment_signature=signature,
    )
    atomic_torch_save(payload, tmp_path / "rolling.pt")

    with pytest.raises(ValueError, match="scheduler state but runtime does not"):
        restore_training_checkpoint(
            payload,
            model=model,
            optimizer=optimizer,
            scaler=None,
            scheduler=None,
            expected_signature=signature,
        )
