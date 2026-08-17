import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"


@pytest.fixture
def train_module(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "p0b_train_integration")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.setenv("NS_VQ_PRECISION", "fp32")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    module = importlib.import_module("train")
    yield module
    sys.modules.pop("train", None)


class TinyAutoencoder(nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.nonfinite = nonfinite

    def encoder(self, value):
        return value * self.scale

    def quant_conv(self, value):
        return value

    def quantize(self, value):
        indices = torch.zeros(value.shape[0] * 4, dtype=torch.long, device=value.device)
        return value, value.sum() * 0.0, (value.new_tensor(1.0), None, indices)

    def post_quant_conv(self, value):
        return value

    def decoder(self, value):
        if self.nonfinite and self.training:
            return value * torch.tensor(float("nan"), device=value.device)
        return value


class PlateauValidationAutoencoder(TinyAutoencoder):
    def decoder(self, value):
        if self.training:
            return value
        return value * 0.0


class PoolCorruptingAutoencoder(TinyAutoencoder):
    def __init__(self):
        super().__init__()
        self.pool = type("Pool", (), {})()
        self.pool.features = torch.ones(2, 2)
        self.pool.nums_features = 1

    def quantize(self, value):
        result = super().quantize(value)
        if self.training:
            self.pool.features[0, 0] = float("nan")
        return result


def train_once(
    module,
    model,
    rolling,
    *,
    epochs,
    signature,
    auto_resume=False,
    signature_configuration=None,
):
    samples = np.ones((2, 3, 4, 4), dtype=np.float32)
    return module._train_vqvae(
        model,
        samples,
        samples,
        epochs=epochs,
        bs=1,
        lr=1e-3,
        tag="p0b-test",
        history_path=str(rolling.with_name("history.json")),
        precision="fp32",
        grad_clip_norm=0.1,
        strict_nonfinite=True,
        rolling_checkpoint_path=str(rolling),
        auto_resume=auto_resume,
        experiment_signature=signature,
        signature_configuration=(
            signature_configuration
            or {"formal": "p0b-test", "target_epochs": 3}
        ),
        scheduler_factor=0.5,
        scheduler_patience=0,
        scheduler_threshold=1e-5,
        scheduler_min_lr=1e-6,
        val_buckets=["surface_curved_proxy", "surface_curved_proxy"],
        val_parent_ids=["parent-a", "parent-b"],
        codebook_size=1,
    )


def test_strict_loop_records_finite_clip_scheduler_and_full_rolling_state(
    tmp_path, monkeypatch, train_module
):
    monkeypatch.setattr(train_module, "VQ_PLATEAU_METRIC", "curved_parent_mse")
    rolling = tmp_path / "rolling.pt"
    _, _, meta = train_once(
        train_module,
        PlateauValidationAutoencoder(),
        rolling,
        epochs=2,
        signature="signed-run",
    )

    payload = torch.load(rolling, map_location="cpu")
    history = payload["history"]
    assert payload["checkpoint_schema"] == "vq_training_state_v1"
    assert payload["epoch"] == 1
    assert payload["experiment_signature"] == "signed-run"
    assert payload["optimizer_state_dict"]
    assert payload["scheduler_state_dict"]
    assert payload["rng_state"]
    assert payload["scaler_state_dict"] is None
    assert payload["extra"]["selector_state"] == {
        "best_curved_parent_mse": pytest.approx(meta["best_val_metrics"][
            "parent_cluster_reconstruction_mse"
        ]["surface_curved_proxy"]["mse"]),
        "prior_perplexities": [1.0, 1.0],
        "prior_coverages": [1.0, 1.0],
    }
    for record in history:
        assert record["gradients_finite"] is True
        assert record["training_state_finite"] is True
        assert record["grad_clip_active"] is True
        assert record["preclip_grad_norm"] is not None
        assert record["finite_state_audit_cadence"] == "lifecycle_v1"
        assert record["full_state_audits"] == 1
        assert record["per_batch_full_state_audits"] == 0
        assert record["finite_state_audit"]["phase"] == "epoch_end_pre_save"
        assert record["finite_state_audit"]["status"] == "finite"
        audit = record["finite_state_audit"]
        assert audit["scalar_device_checks"] == len(audit["devices"])
        assert audit["scalar_device_checks"] <= 2
        for name in (
            "skipped_train_batches",
            "nonfinite_loss_batches",
            "nonfinite_gradient_batches",
            "nonfinite_state_batches",
            "nonfinite_val_batches",
            "nonfinite_val_samples",
        ):
            assert record[name] == 0
        curved_parent_mse = record["val_parent_cluster_reconstruction_mse"][
            "surface_curved_proxy"
        ]["mse"]
        assert record["scheduler_metric"] == pytest.approx(curved_parent_mse)
        assert record["plateau_metric"] == "curved_parent_mse"
        assert record["plateau_value"] == pytest.approx(curved_parent_mse)
    assert history[-1]["lr_after_scheduler"] < history[0]["lr"]
    assert meta["precision"]["name"] == "fp32"
    assert meta["finite_state_audit_cadence"] == {
        "policy": "lifecycle_v1",
        "startup_or_post_resume": True,
        "epoch_end_pre_save": True,
        "per_train_batch": False,
    }
    assert meta["startup_finite_state_audit"]["phase"] == "startup"
    assert meta["full_state_audits"] == 3
    assert payload["finite_state_audit"] == history[-1]["finite_state_audit"]
    history_document = json.loads(
        (tmp_path / "history.json").read_text(encoding="utf-8")
    )
    assert history_document["finite_state_audit_summary"]["full_state_audits"] == 3


def test_full_state_audit_runs_at_lifecycle_boundaries_not_per_batch(
    tmp_path, monkeypatch, train_module
):
    stability = sys.modules["training_stability"]
    real_audit = stability.audit_finite_training_state
    calls = []

    def tracked_audit(model, optimizer=None):
        calls.append((model.training, optimizer is not None))
        return real_audit(model, optimizer)

    monkeypatch.setattr(stability, "audit_finite_training_state", tracked_audit)
    monkeypatch.setattr(train_module, "audit_finite_training_state", tracked_audit)
    samples = np.ones((4, 3, 4, 4), dtype=np.float32)
    history_path = tmp_path / "history.json"
    rolling = tmp_path / "rolling.pt"

    train_module._train_vqvae(
        TinyAutoencoder(),
        samples,
        samples[:2],
        epochs=2,
        bs=1,
        lr=1e-3,
        tag="audit-cadence",
        history_path=str(history_path),
        precision="fp32",
        grad_clip_norm=1.0,
        strict_nonfinite=True,
        rolling_checkpoint_path=str(rolling),
        experiment_signature="audit-cadence",
        signature_configuration={"formal": "audit-cadence"},
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
    assert len(calls) == 3  # startup plus one pre-save audit for each epoch
    assert sum(record["train_batches"] for record in history) == 8
    assert all(record["per_batch_full_state_audits"] == 0 for record in history)
    assert all(record["full_state_audits"] == 1 for record in history)


def test_auto_resume_restores_full_state_and_starts_at_next_epoch(tmp_path, train_module):
    rolling = tmp_path / "rolling.pt"
    train_once(train_module, TinyAutoencoder(), rolling, epochs=1, signature="same-run")

    new_history, _, meta = train_once(
        train_module,
        TinyAutoencoder(),
        rolling,
        epochs=3,
        signature="same-run",
        auto_resume=True,
    )

    payload = torch.load(rolling, map_location="cpu")
    history_payload = json.loads(
        (tmp_path / "history.json").read_text(encoding="utf-8")
    )
    assert meta["resumed"] is True
    assert meta["resume_from_epoch"] == 0
    assert meta["start_epoch"] == 1
    assert meta["startup_finite_state_audit"]["phase"] == "post_resume"
    assert meta["full_state_audits"] == 5
    assert [record["epoch"] for record in payload["history"]] == [0, 1, 2]
    assert len(new_history) == 2
    assert payload["epoch"] == 2
    assert history_payload["config"]["start_epoch"] == 0
    assert history_payload["config"]["target_epoch"] == 3


def test_auto_resume_restarts_fresh_on_signature_mismatch(tmp_path, train_module, capsys):
    rolling = tmp_path / "rolling.pt"
    train_once(train_module, TinyAutoencoder(), rolling, epochs=1, signature="run-a")

    history, _, meta = train_once(
        train_module,
        TinyAutoencoder(),
        rolling,
        epochs=3,
        signature="run-b",
        auto_resume=True,
    )

    payload = torch.load(rolling, map_location="cpu")
    assert meta["resumed"] is False
    assert meta["start_epoch"] == 0
    assert len(history) == 3
    assert [record["epoch"] for record in payload["history"]] == [0, 1, 2]
    assert "signature stale; starting fresh" in capsys.readouterr().out


@pytest.mark.parametrize("drift_field", ["inventory", "runtime_resume_compatibility"])
def test_auto_resume_rejects_data_or_runtime_compatibility_drift(
    tmp_path, train_module, drift_field
):
    rolling = tmp_path / f"{drift_field}.pt"
    initial = {
        "formal": "p0b-test",
        "target_epochs": 3,
        "inventory": {"train": "inventory-a"},
        "runtime_resume_compatibility": {"pytorch": "runtime-a"},
    }
    changed = json.loads(json.dumps(initial))
    changed[drift_field] = {
        "train": "inventory-b"
    } if drift_field == "inventory" else {"pytorch": "runtime-b"}
    train_once(
        train_module,
        TinyAutoencoder(),
        rolling,
        epochs=1,
        signature="same-external-signature",
        signature_configuration=initial,
    )

    with pytest.raises(ValueError, match="signature configuration mismatch"):
        train_once(
            train_module,
            TinyAutoencoder(),
            rolling,
            epochs=3,
            signature="same-external-signature",
            auto_resume=True,
            signature_configuration=changed,
        )


def test_strict_nonfinite_loss_fuses_before_checkpoint(tmp_path, train_module):
    rolling = tmp_path / "rolling.pt"
    with pytest.raises(train_module.NonFiniteTrainingError, match="training loss"):
        train_once(
            train_module,
            TinyAutoencoder(nonfinite=True),
            rolling,
            epochs=1,
            signature="nonfinite-run",
        )
    assert not rolling.exists()


def test_strict_epoch_audit_catches_feature_pool_corruption_before_checkpoint(
    tmp_path, train_module
):
    rolling = tmp_path / "rolling.pt"
    with pytest.raises(train_module.NonFiniteTrainingError, match="pool.features"):
        train_once(
            train_module,
            PoolCorruptingAutoencoder(),
            rolling,
            epochs=1,
            signature="pool-corruption",
        )
    assert not rolling.exists()


def test_learned_vq_wrapper_disables_nested_autocast(train_module):
    class RecordingQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.autocast_seen = None

        def forward(self, value):
            self.autocast_seen = torch.is_autocast_cpu_enabled()
            indices = torch.zeros(value.shape[0], dtype=torch.long)
            return value, value.sum() * 0.0, (value.new_tensor(1.0), None, indices)

    quantizer = RecordingQuantizer()
    wrapper = train_module.AmpSafeLearnedVectorQuantiser(quantizer)
    value = torch.ones(2, 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        quantized, _, _ = wrapper(value)

    assert quantizer.autocast_seen is False
    assert quantized.dtype == value.dtype
