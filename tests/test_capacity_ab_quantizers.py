from __future__ import annotations

import importlib
import json
import pickle
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"


@pytest.fixture
def train_module(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "capacity_ab_test")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.setenv("NS_VQ_PRECISION", "fp32")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    module = importlib.import_module("train")
    yield module
    sys.modules.pop("train", None)


def test_capacity_configs_freeze_vq8192_and_residual_vq_metadata(train_module):
    vq, rvq = train_module.quantizer_comparison_configs(
        ("vq_8192_64d_random", "rvq_2x4096_64d_random")
    )

    assert vq["codebook_size"] == 8192
    assert vq["quantizer"]["anchor"] == "random"
    assert vq["quantizer"]["embedding_dim"] == 64
    assert rvq["stage_codebook_sizes"] == (4096, 4096)
    assert rvq["quantizer"]["effective_code_combinations"] == 16_777_216
    assert rvq["quantizer"]["residual_stage1_detached"] is True
    assert rvq["quantizer"]["stage2_collapse_gate"] == {
        "min_unique_bins": 2,
        "min_perplexity_exclusive": 1.0,
    }


def test_vq8192_has_expected_embedding_and_finite_forward_backward(train_module):
    config = train_module.quantizer_comparison_configs(("vq_8192_64d_random",))[0]
    train_module.seed_vq_experiment(3)
    model = train_module.build_quantized_vqvae(config)
    latent = torch.randn(2, 64, 2, 2, requires_grad=True)

    quantized, loss, info = model.quantize(latent)
    (quantized.square().mean() + loss).backward()

    assert model.quantize.quantizer.embedding.weight.shape == (8192, 64)
    assert info[2].shape == (8,)
    assert 0 <= int(info[2].min()) <= int(info[2].max()) < 8192
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_rvq_uses_detached_residual_two_independent_pools_and_one_st_path(
    train_module, monkeypatch
):
    config = train_module.quantizer_comparison_configs(("rvq_2x4096_64d_random",))[0]
    train_module.seed_vq_experiment(3)
    model = train_module.build_quantized_vqvae(config)
    rvq = model.quantize
    latent = torch.randn(2, 64, 2, 2, requires_grad=True)
    observed = {}
    real_stage2 = rvq.stage2.forward

    def capture_stage2(value, *args, **kwargs):
        observed["requires_grad"] = value.requires_grad
        observed["grad_fn"] = value.grad_fn
        return real_stage2(value, *args, **kwargs)

    monkeypatch.setattr(rvq.stage2, "forward", capture_stage2)
    quantized, loss, info = rvq(latent)
    decoder_only = torch.autograd.grad(
        quantized.sum(), latent, retain_graph=True
    )[0]
    (quantized.square().mean() + loss).backward()

    # The residual keeps an encoder gradient path through ``latent`` while the
    # stage-1 quantized value itself is detached from stage 2's loss path.
    assert observed["requires_grad"] is True
    assert observed["grad_fn"] is not None
    assert rvq.stage1.quantizer.embedding.weight.shape == (4096, 64)
    assert rvq.stage2.quantizer.embedding.weight.shape == (4096, 64)
    assert rvq.stage1.pool is not rvq.stage2.pool
    assert rvq.stage_codebook_sizes == (4096, 4096)
    assert quantized.shape == latent.shape
    assert len(info) == 3
    assert len(info.stage_indices) == 2
    assert all(indices.shape == (8,) for indices in info.stage_indices)
    assert info[2].shape == (16,)
    assert int(info[2][8:].min()) >= 4096
    assert torch.allclose(decoder_only, torch.ones_like(latent))
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_rvq_keeps_residual_subtraction_in_fp32_for_bf16_input(train_module, monkeypatch):
    config = train_module.quantizer_comparison_configs(("rvq_2x4096_64d_random",))[0]
    train_module.seed_vq_experiment(3)
    rvq = train_module.build_quantized_vqvae(config).quantize
    latent = torch.randn(1, 64, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    observed = {}
    real_stage2 = rvq.stage2.forward

    def capture_stage2(value, *args, **kwargs):
        observed["dtype"] = value.dtype
        return real_stage2(value, *args, **kwargs)

    monkeypatch.setattr(rvq.stage2, "forward", capture_stage2)
    quantized, loss, _info = rvq(latent)
    (quantized.float().square().mean() + loss).backward()

    assert observed["dtype"] == torch.float32
    assert quantized.dtype == torch.bfloat16
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()


def test_stage_tracker_reports_exact_usage_and_collapse_fail_closed(train_module):
    tracker = train_module.QuantizerStageUsageTracker((4, 4), device="cpu")
    info = train_module.QuantizerInfo(
        torch.tensor(2.0),
        None,
        torch.tensor([0, 1, 0, 2, 2, 2]),
        stage_indices=(torch.tensor([0, 1, 0]), torch.tensor([2, 2, 2])),
        stage_perplexities=(torch.tensor(2.0), torch.tensor(1.0)),
    )
    tracker.update(info)
    usage = tracker.summary()

    assert usage["stage1"]["tokens"] == 3
    assert usage["stage1"]["unique_bins"] == 2
    assert usage["stage1"]["coverage"] == 0.5
    assert usage["stage1"]["usage_fraction"] == pytest.approx(
        usage["stage1"]["entropy_perplexity"] / 4
    )
    assert usage["stage2"]["tokens"] == 3
    assert usage["stage2"]["unique_bins"] == 1
    health = train_module.stage_usage_health(usage, require_stage2=True)
    assert health["healthy"] is False
    assert any("stage2 unique_bins" in reason for reason in health["reasons"])
    assert train_module.stage_usage_health(
        {"stage1": usage["stage1"]}, require_stage2=True
    ) == {"healthy": False, "reasons": ["stage2 usage is missing"]}

    restored = pickle.loads(pickle.dumps(info))
    assert len(restored) == 3
    assert len(restored.stage_indices) == 2


def test_stage_tracker_rejects_missing_streams_and_out_of_range_indices(train_module):
    tracker = train_module.QuantizerStageUsageTracker((4, 4))
    with pytest.raises(ValueError, match="expected 2 stage index streams"):
        tracker.update((torch.tensor(1.0), None, torch.tensor([0, 1])))
    with pytest.raises(ValueError, match="stage 2 encoding index outside codebook"):
        tracker.update(train_module.QuantizerInfo(
            torch.tensor(1.0), None, torch.tensor([0, 9]),
            stage_indices=(torch.tensor([0]), torch.tensor([9])),
        ))


def test_checkpoint_selector_rejects_collapsed_stage2(train_module):
    metrics = {
        "code_usage": {"entropy_perplexity": 900.0, "coverage": 0.25},
        "parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": 0.01}
        },
        "nonfinite_val_samples": 0,
    }
    decision = train_module.evaluate_vq_checkpoint_candidate(
        metrics,
        best_curved_parent_mse=0.02,
        stage_usage_health_report={
            "healthy": False,
            "reasons": ["stage2 unique_bins must be >= 2"],
        },
    )

    assert decision["selected"] is False
    assert any("stage2 unique_bins" in reason for reason in decision["reasons"])


def test_rvq_selector_uses_independent_stage_usage_not_legacy_marginal(train_module):
    metrics = {
        # Deliberately nonsensical legacy marginal values.  RVQ selection must
        # be driven by its two independently named stage fractions.
        "code_usage": {"entropy_perplexity": 1.0, "coverage": 0.0001},
        "stage_code_usage": {
            "stage1": {"usage_fraction": 0.25},
            "stage2": {"usage_fraction": 0.20},
        },
        "parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": 0.01}
        },
        "nonfinite_val_samples": 0,
    }

    decision = train_module.evaluate_vq_checkpoint_candidate(
        metrics,
        prior_perplexities=[2000.0],
        prior_coverages=[0.8],
        best_curved_parent_mse=0.02,
        stage_usage_health_report={"healthy": True, "reasons": []},
        prior_stage_usage_fractions={
            "stage1": [0.26], "stage2": [0.21]
        },
    )

    assert decision["selected"] is True
    assert decision["observed"]["perplexity"] is None
    assert decision["observed"]["coverage"] is None


class TinyResidualQuantizer(nn.Module):
    stage_codebook_sizes = (4, 4)

    def forward(self, latent):
        token_count = latent.shape[0] * latent.shape[2] * latent.shape[3]
        first = torch.arange(token_count, device=latent.device) % 3
        second = (torch.arange(token_count, device=latent.device) + 1) % 3
        info = QuantizerInfoForTest(
            latent.new_tensor(3.0),
            None,
            torch.cat((first, second)),
            stage_indices=(first, second),
        )
        return latent, latent.sum() * 0.0, info


class QuantizerInfoForTest(tuple):
    def __new__(cls, perplexity, encodings, indices, stage_indices):
        value = tuple.__new__(cls, (perplexity, encodings, indices))
        value.stage_indices = tuple(stage_indices)
        return value


class TinyResidualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.quantize = TinyResidualQuantizer()

    def encoder(self, value):
        return value * self.scale

    def quant_conv(self, value):
        return value

    def post_quant_conv(self, value):
        return value

    def decoder(self, value):
        return value


def test_train_loop_records_separate_stage_usage_and_health(train_module, tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "VQ_PLATEAU_METRIC", "curved_parent_mse")
    samples = torch.ones(2, 3, 2, 2).numpy()
    history_path = tmp_path / "history.json"
    rolling = tmp_path / "rolling.pt"
    history, _best, meta = train_module._train_vqvae(
        TinyResidualAutoencoder(),
        samples,
        samples,
        epochs=1,
        bs=1,
        lr=1e-3,
        tag="tiny-rvq",
        history_path=str(history_path),
        save_path=str(tmp_path / "best.pt"),
        precision="fp32",
        strict_nonfinite=True,
        rolling_checkpoint_path=str(rolling),
        experiment_signature="tiny-rvq",
        signature_configuration={"name": "tiny-rvq"},
        val_buckets=["surface_curved_proxy", "surface_curved_proxy"],
        val_parent_ids=["p0", "p1"],
        codebook_size=4,
        quantizer_metadata={"kind": "residual_vq"},
        tb_log_dir=str(tmp_path / "tensorboard"),
    )

    assert len(history) == 1
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    record = history_payload["history"][0]
    assert history_payload["config"]["scheduler"]["kind"] == "ReduceLROnPlateau"
    assert record["train_stage_code_usage"]["stage1"]["tokens"] == 8
    assert record["val_stage_code_usage"]["stage2"]["tokens"] == 8
    assert record["stage_usage_health"] == {"healthy": True, "reasons": []}
    assert meta["last_val_metrics"]["stage_code_usage"] == record["val_stage_code_usage"]
    payload = torch.load(rolling, map_location="cpu", weights_only=False)
    assert payload["extra"]["selector_state"]["stage_usage_health"]["healthy"] is True
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags = set(EventAccumulator(str(tmp_path / "tensorboard")).Reload().Tags()["scalars"])
    assert "train/stage1/code_usage/entropy_perplexity" in tags
    assert "validation/stage2/code_usage/coverage" in tags
    assert "validation/stage_usage_health/healthy" in tags


class PoolOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = types.SimpleNamespace(
            features=torch.ones(2, 3, dtype=torch.float32),
            nums_features=2,
        )


def test_resume_requires_exact_independent_feature_pool_set(train_module):
    stability = sys.modules["training_stability"]
    model = nn.Module()
    model.stage1 = PoolOwner()
    model.stage2 = PoolOwner()
    states = stability.capture_feature_pools(model)

    assert set(states) == {"stage1", "stage2"}
    stability.restore_feature_pools(model, states)
    states.pop("stage2")
    with pytest.raises(ValueError, match="feature-pool module set mismatch"):
        stability.restore_feature_pools(model, states)


def test_checkpoint_context_preserves_parent_overlap_evidence(train_module):
    overlaps = {"train__val": 0, "train__test": 0, "val__test": 0}
    run_manifest = {
        "git": {"commit": "abc123"},
        "experiment": {
            "protocol": {
                "protocol_sha256": "protocol",
                "split_pickle_sha256": "split",
                "parent_overlap_counts": overlaps,
            }
        },
    }
    protocol_data = {
        "train_sampling": {"final_parent_coverage": 1.0},
        "val_sampling": {"parent_coverage": 1.0},
        "inventory": {},
    }

    context = train_module.checkpoint_context_from_run(run_manifest, protocol_data)

    assert context["parent_overlap_counts"] == overlaps
