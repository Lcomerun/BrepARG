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
    assert data["cross_split_exact"]["train_records_removed"] == 0
    assert data["integrity"]["status"] == "VERIFIED"
    assert data["val_buckets"] == ["edge", "surface_planar_like"]


@pytest.mark.parametrize("failure_kind", ["load", "parent"])
def test_collect_protocol_vq_data_rejects_any_unusable_requested_source(
    tmp_path, train_module, failure_kind
):
    train_path = tmp_path / source("a" * 24)
    val_path = tmp_path / source("b" * 24)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as handle:
        pickle.dump(parsed(0.0, 0.0), handle)
    with val_path.open("wb") as handle:
        pickle.dump(parsed(1.0, 1.0), handle)

    if failure_kind == "load":
        unusable_path = tmp_path / source("c" * 24, index=3)
    else:
        unusable_path = tmp_path / "abc_0000" / "unresolved_parent.pkl"
        with unusable_path.open("wb") as handle:
            pickle.dump(parsed(2.0, 2.0), handle)

    with pytest.raises(RuntimeError, match="required VQ source"):
        train_module.collect_protocol_vq_data(
            {
                "train": [str(train_path), str(unusable_path)],
                "val": [str(val_path)],
                "test": [],
            },
            train_cap=1,
            val_cap=1,
        )


def test_collect_protocol_vq_data_audits_identity_before_exact_overlap_filter(
    monkeypatch, train_module
):
    shared = np.zeros((32, 32, 3), dtype=np.float32)
    train_records = [
        {
            "record_id": "valid-train",
            "source_path": source("a" * 24),
            "source_key": source("a" * 24).casefold(),
            "parent_id": "a" * 24,
            "kind": "surface",
            "array": np.ones((32, 32, 3), dtype=np.float32),
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
        {
            "record_id": "invalid-overlap",
            "source_path": "",
            "source_key": "",
            "parent_id": "b" * 24,
            "kind": "surface",
            "array": shared,
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
    ]
    val_records = [
        {
            "record_id": "val",
            "source_path": source("c" * 24),
            "source_key": source("c" * 24).casefold(),
            "parent_id": "c" * 24,
            "kind": "surface",
            "array": shared.copy(),
            "curvature_score": 0.0,
            "is_complex_source": False,
        }
    ]

    def fake_collect(_paths, _cap, seed):
        records = train_records if seed == 0 else val_records
        return records, {"selected": len(records)}, {"unique_records": len(records)}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)

    with pytest.raises(ValueError, match="invalid source identity.*train"):
        train_module.collect_protocol_vq_data(
            {"train": ["train"], "val": ["val"], "test": []},
            train_cap=2,
            val_cap=1,
        )


@pytest.mark.parametrize("overlap_kind", ["source", "parent"])
def test_collect_protocol_vq_data_rejects_identity_overlap_before_exact_filter(
    monkeypatch, train_module, overlap_kind
):
    shared = np.zeros((32, 32, 3), dtype=np.float32)
    train_unique = np.ones((32, 32, 3), dtype=np.float32)
    shared_parent = "a" * 24
    train_shared_source = source(shared_parent)
    val_source = (
        train_shared_source
        if overlap_kind == "source"
        else source(shared_parent, index=2)
    )
    train_records = [
        {
            "record_id": "train-shared",
            "source_path": train_shared_source,
            "source_key": train_shared_source.casefold(),
            "parent_id": shared_parent,
            "kind": "surface",
            "array": shared,
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
        {
            "record_id": "train-unique",
            "source_path": source("b" * 24),
            "source_key": source("b" * 24).casefold(),
            "parent_id": "b" * 24,
            "kind": "surface",
            "array": train_unique,
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
    ]
    val_records = [
        {
            "record_id": "val-shared",
            "source_path": val_source,
            "source_key": val_source.casefold(),
            "parent_id": shared_parent,
            "kind": "surface",
            "array": shared.copy(),
            "curvature_score": 0.0,
            "is_complex_source": False,
        }
    ]

    def fake_collect(_paths, _cap, seed):
        records = train_records if seed == 0 else val_records
        return records, {"selected": len(records)}, {"unique_records": len(records)}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)

    with pytest.raises(ValueError, match=f"{overlap_kind}.*overlap"):
        train_module.collect_protocol_vq_data(
            {"train": ["train"], "val": ["val"], "test": []},
            train_cap=2,
            val_cap=1,
        )


def test_collect_protocol_vq_data_filters_content_overlap_then_audits_final_inventory(
    monkeypatch, train_module
):
    shared = np.zeros((32, 32, 3), dtype=np.float32)
    train_records = [
        {
            "record_id": "train-shared",
            "source_path": source("a" * 24),
            "source_key": source("a" * 24).casefold(),
            "parent_id": "a" * 24,
            "kind": "surface",
            "array": shared,
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
        {
            "record_id": "train-unique",
            "source_path": source("b" * 24),
            "source_key": source("b" * 24).casefold(),
            "parent_id": "b" * 24,
            "kind": "surface",
            "array": np.ones((32, 32, 3), dtype=np.float32),
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
    ]
    val_records = [
        {
            "record_id": "val-shared",
            "source_path": source("c" * 24),
            "source_key": source("c" * 24).casefold(),
            "parent_id": "c" * 24,
            "kind": "surface",
            "array": shared.copy(),
            "curvature_score": 0.0,
            "is_complex_source": False,
        }
    ]

    def fake_collect(_paths, _cap, seed):
        records = train_records if seed == 0 else val_records
        return records, {"selected": len(records)}, {"unique_records": len(records)}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)

    result = train_module.collect_protocol_vq_data(
        {"train": ["train"], "val": ["val"], "test": []},
        train_cap=2,
        val_cap=1,
    )

    assert result["cross_split_exact"]["train_records_removed"] == 1
    assert result["integrity"]["status"] == "VERIFIED"
    assert len(result["X_train"]) == 1
    assert len(result["X_val"]) == 1


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


def test_vq_stages_reject_disabling_protocol_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "legacy_vq_rejected")
    monkeypatch.setenv("NS_PROTOCOL_V2", "0")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        for stage in (module.stage_vqvae, module.stage_vqsweep):
            with pytest.raises(RuntimeError, match="requires Protocol V2"):
                stage()
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


@pytest.mark.parametrize(
    "overlap_counts",
    [
        {"train__val": 0, "train__test": 0},
        {
            "train__val": 0,
            "train__test": 0,
            "val__test": 0,
            "train__train": 0,
        },
    ],
)
def test_protocol_split_stage_requires_exact_parent_overlap_key_set(
    tmp_path, monkeypatch, overlap_counts
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump(
            {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []},
            handle,
        )
    (protocol_dir / "protocol_summary.json").write_text(
        __import__("json").dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": "abc",
                "parent_overlap_counts": overlap_counts,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_overlap_keys")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="parent_overlap_counts.*exactly"):
            module.stage_split()
    finally:
        sys.modules.pop("train", None)


@pytest.mark.parametrize("overlap_value", [0.5, -0.5, "0", False, None])
def test_protocol_split_stage_rejects_non_integer_zero_overlap_values(
    tmp_path, monkeypatch, overlap_value
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump(
            {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []},
            handle,
        )
    overlap_counts = {
        "train__val": overlap_value,
        "train__test": 0,
        "val__test": 0,
    }
    (protocol_dir / "protocol_summary.json").write_text(
        __import__("json").dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": "abc",
                "parent_overlap_counts": overlap_counts,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_overlap_value")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="JSON integer zero"):
            module.stage_split()
    finally:
        sys.modules.pop("train", None)


def test_protocol_split_stage_rejects_missing_overlap_evidence(tmp_path, monkeypatch):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump(
            {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []},
            handle,
        )
    (protocol_dir / "protocol_summary.json").write_text(
        '{"status":"VERIFIED","protocol_sha256":"abc","parent_overlap_counts":{}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_missing_evidence")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="parent_overlap_counts.*exactly"):
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


class ZeroDecoderAutoencoder(TinyQuantizedAutoencoder):
    def decoder(self, value):
        return torch.zeros_like(value)


class PartiallyNonfiniteDecoderAutoencoder(TinyQuantizedAutoencoder):
    def decoder(self, value):
        result = value.clone()
        if not self.training and len(result) > 1:
            result[1] = torch.nan
        return result


class AllNonfiniteDecoderAutoencoder(TinyQuantizedAutoencoder):
    def decoder(self, value):
        if self.training:
            return value
        return torch.full_like(value, torch.nan)


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


def test_train_vqvae_global_validation_mse_is_sample_weighted(
    tmp_path, train_module
):
    model = TinyQuantizedAutoencoder()
    model.scale.data.zero_()
    train_samples = np.zeros((1, 3, 32, 32), dtype=np.float32)
    val_samples = np.stack(
        [
            np.zeros((3, 32, 32), dtype=np.float32),
            np.zeros((3, 32, 32), dtype=np.float32),
            np.full((3, 32, 32), 2.0, dtype=np.float32),
        ]
    )

    history, best_val, _ = train_module._train_vqvae(
        model,
        train_samples,
        val_samples,
        epochs=1,
        bs=2,
        lr=1e-4,
        tag="sample-weighted-validation",
        history_path=str(tmp_path / "sample_weighted_history.json"),
        amp_enabled=False,
    )

    assert history[0][1] == pytest.approx(4.0 / 3.0)
    assert best_val == pytest.approx(4.0 / 3.0)


def test_train_vqvae_validation_loss_is_sample_weighted_across_uneven_batches(
    tmp_path, train_module
):
    samples = np.concatenate(
        [
            np.ones((2, 3, 32, 32), dtype=np.float32),
            np.full((1, 3, 32, 32), 3.0, dtype=np.float32),
        ]
    )

    history, _, _ = train_module._train_vqvae(
        ZeroDecoderAutoencoder(),
        samples,
        samples,
        epochs=1,
        bs=2,
        lr=0.0,
        tag="sample_weighted",
        history_path=str(tmp_path / "history.json"),
        amp_enabled=False,
        val_buckets=["edge", "edge", "edge"],
        codebook_size=4,
    )

    assert history[0][1] == pytest.approx(11.0 / 3.0)


def test_train_vqvae_partial_nonfinite_validation_invalidates_epoch(
    tmp_path, train_module
):
    samples = np.zeros((2, 3, 32, 32), dtype=np.float32)
    history_path = tmp_path / "partial_nonfinite_history.json"
    best_path = tmp_path / "partial_nonfinite_best.pt"

    history, best_val, meta = train_module._train_vqvae(
        PartiallyNonfiniteDecoderAutoencoder(),
        samples,
        samples,
        epochs=1,
        bs=2,
        lr=0.0,
        tag="partial-nonfinite-validation",
        save_path=str(best_path),
        history_path=str(history_path),
        amp_enabled=False,
        val_buckets=["edge", "edge"],
        codebook_size=4,
    )

    record = __import__("json").loads(history_path.read_text(encoding="utf-8"))["history"][0]
    assert np.isinf(history[0][1])
    assert np.isinf(best_val)
    assert record["val_loss"] is None
    assert record["finite_val_samples"] == 1
    assert record["nonfinite_val_samples"] == 1
    assert record["finite_val_batches"] == 0
    assert record["consecutive_nonfinite_val_epochs"] == 1
    assert record["train_loss"] == pytest.approx(0.0)
    assert meta["last_val_metrics"]["reconstruction_mse"]["edge"]["samples"] == 1
    assert not best_path.exists()


def test_train_vqvae_all_nonfinite_validation_cannot_become_best(
    tmp_path, train_module
):
    samples = np.zeros((2, 3, 32, 32), dtype=np.float32)
    history_path = tmp_path / "all_nonfinite_history.json"
    best_path = tmp_path / "all_nonfinite_best.pt"

    history, best_val, meta = train_module._train_vqvae(
        AllNonfiniteDecoderAutoencoder(),
        samples,
        samples,
        epochs=1,
        bs=2,
        lr=0.0,
        tag="all-nonfinite-validation",
        save_path=str(best_path),
        history_path=str(history_path),
        amp_enabled=False,
        val_buckets=["edge", "edge"],
        codebook_size=4,
    )

    record = __import__("json").loads(history_path.read_text(encoding="utf-8"))["history"][0]
    assert np.isinf(history[0][1])
    assert np.isinf(best_val)
    assert record["val_loss"] is None
    assert record["best_val"] is None
    assert record["best_epoch"] == -1
    assert record["finite_val_samples"] == 0
    assert record["nonfinite_val_samples"] == 2
    assert record["finite_val_batches"] == 0
    assert record["nonfinite_val_batches"] == 1
    assert record["consecutive_nonfinite_val_epochs"] == 1
    assert meta["last_val_metrics"]["reconstruction_mse"]["edge"] == {
        "samples": 0,
        "mse": None,
    }
    assert not best_path.exists()
