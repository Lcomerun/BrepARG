import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"


def source(parent, part=0, index=1):
    return f"abc_0000/{index:08d}_{parent}_step_{part:03d}.pkl"


def parsed(surface_value, edge_value):
    edge = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]
    edge = np.tile(edge, (1, 3)) + edge_value
    return {
        "surf_ncs": np.full((1, 32, 32, 3), surface_value, dtype=np.float32),
        "edge_ncs": np.asarray([edge], dtype=np.float32),
    }


@pytest.fixture
def train_module(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_training_test")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.delenv("NS_VQ_SAMPLE_CACHE", raising=False)
    monkeypatch.delenv("NS_VQ_PATCH_SHARD_ROOT", raising=False)
    monkeypatch.delenv("NS_VQ_PATCH_SHARDS", raising=False)
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    module = importlib.import_module("train")
    yield module
    sys.modules.pop("train", None)


def test_collect_protocol_vq_data_uses_disjoint_cad_splits_and_exact_dedup(
    tmp_path, train_module
):
    train_path = tmp_path / source("a" * 24)
    val_path = tmp_path / source("b" * 24)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as handle:
        pickle.dump(parsed(0.0, 0.0), handle)
    with val_path.open("wb") as handle:
        pickle.dump(parsed(1.0, 1.0), handle)

    data = train_module.collect_protocol_vq_data(
        {"train": [str(train_path)], "val": [str(val_path)], "test": []},
        train_cap=8,
        val_cap=8,
    )

    assert data["X_train"].shape == (2, 3, 32, 32)
    assert data["X_val"].shape == (2, 3, 32, 32)
    assert data["train_dedup"]["duplicates_removed"] == 0
    assert data["val_dedup"]["duplicates_removed"] == 0
    assert data["integrity"]["status"] == "VERIFIED"
    assert data["val_buckets"] == ["edge", "surface_planar_like"]


@pytest.mark.parametrize("asset_env", ["NS_VQ_SAMPLE_CACHE", "NS_VQ_PATCH_SHARD_ROOT"])
def test_protocol_v2_rejects_unverified_patch_assets(tmp_path, monkeypatch, asset_env):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_asset_test")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.setenv(asset_env, str(tmp_path / "legacy_asset"))
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="Protocol V2.*provenance"):
            module.collect_protocol_vq_data(
                {"train": [], "val": [], "test": []},
                train_cap=1,
                val_cap=1,
            )
    finally:
        sys.modules.pop("train", None)


def test_protocol_split_stage_accepts_only_verified_protocol_outputs(
    tmp_path, monkeypatch
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    split = {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []}
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump(split, handle)
    (protocol_dir / "protocol_summary.json").write_text(
        '{"status":"VERIFIED","protocol_sha256":"abc",'
        '"parent_overlap_counts":{"train__val":0,"train__test":0,"val__test":0}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_test")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        assert module.stage_split() is True
        assert module.SPLIT == str(protocol_dir / "split.pkl")
    finally:
        sys.modules.pop("train", None)


def test_protocol_split_stage_rejects_failed_summary(tmp_path, monkeypatch):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": [], "val": [], "test": []}, handle)
    (protocol_dir / "protocol_summary.json").write_text(
        '{"status":"FAILED","parent_overlap_counts":{"train__val":0}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_failed")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="VERIFIED"):
            module.stage_split()
    finally:
        sys.modules.pop("train", None)


class TinyQuantizedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encoder(self, value):
        return value * self.scale

    def quant_conv(self, value):
        return value

    def quantize(self, value):
        indices = torch.arange(value.shape[0] * 4, device=value.device) % 4
        zero = value.sum() * 0.0
        return value, zero, (torch.tensor(0.0, device=value.device), None, indices)

    def post_quant_conv(self, value):
        return value

    def decoder(self, value):
        return value


def test_train_vqvae_records_aggregate_usage_buckets_and_tensorboard(
    tmp_path, train_module
):
    history_path = tmp_path / "history.json"
    tensorboard_dir = tmp_path / "tensorboard"
    samples = np.zeros((2, 3, 32, 32), dtype=np.float32)

    _, _, meta = train_module._train_vqvae(
        TinyQuantizedAutoencoder(),
        samples,
        samples,
        epochs=1,
        bs=1,
        lr=1e-4,
        tag="tiny",
        history_path=str(history_path),
        amp_enabled=False,
        val_buckets=["edge", "surface_curved_proxy"],
        codebook_size=4,
        tb_log_dir=str(tensorboard_dir),
    )

    history = __import__("json").loads(history_path.read_text(encoding="utf-8"))["history"]
    record = history[-1]
    assert record["val_code_usage"] == {
        "tokens": 8,
        "unique_bins": 4,
        "coverage": 1.0,
        "entropy_perplexity": 4.0,
    }
    assert record["val_reconstruction_mse"]["edge"] == {"samples": 1, "mse": 0.0}
    assert meta["last_val_metrics"] == {
        "code_usage": record["val_code_usage"],
        "reconstruction_mse": record["val_reconstruction_mse"],
    }
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(tensorboard_dir)).Reload()
    assert "validation/code_usage/entropy_perplexity" in accumulator.Tags()["scalars"]
    assert "validation/reconstruction_mse/edge" in accumulator.Tags()["scalars"]
