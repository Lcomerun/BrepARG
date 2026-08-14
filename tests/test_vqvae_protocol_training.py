import importlib
import hashlib
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


def test_exact_hash_inventory_binds_order_and_content(monkeypatch):
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sampling = importlib.import_module("vqvae_sampling")
    records = [
        {
            "kind": "surface",
            "array": np.full((32, 32, 3), value, dtype=np.float32),
        }
        for value in (0.0, 1.0, 2.0)
    ]

    original = sampling.summarize_exact_hash_inventory(records)
    reordered = sampling.summarize_exact_hash_inventory(list(reversed(records)))
    changed = sampling.summarize_exact_hash_inventory(
        [*records[:2], {"kind": "surface", "array": records[2]["array"] + 1.0}]
    )

    assert original["schema"] == "vq-exact-hash-inventory-v1"
    assert original["count"] == 3
    assert original["ordered_sha256"] != reordered["ordered_sha256"]
    assert original["sorted_sha256"] == reordered["sorted_sha256"]
    assert original["ordered_sha256"] != changed["ordered_sha256"]
    assert original["sorted_sha256"] != changed["sorted_sha256"]


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
    assert data["inventory"]["train"]["count"] == 2
    assert data["inventory"]["val"]["count"] == 2
    assert data["inventory"]["train"]["schema"] == "vq-exact-hash-inventory-v1"
    assert data["val_buckets"] == ["edge", "surface_planar_like"]
    assert data["val_parent_ids"] == ["b" * 24, "b" * 24]


@pytest.mark.parametrize(
    ("train_cap", "val_cap", "message"),
    [
        (2, 3, "validation inventory did not meet"),
        (3, 2, "train inventory did not meet"),
    ],
)
def test_collect_protocol_vq_data_exact_cap_gate_fails_closed(
    tmp_path, monkeypatch, train_module, train_cap, val_cap, message
):
    train_path = tmp_path / source("a" * 24)
    val_path = tmp_path / source("b" * 24)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as handle:
        pickle.dump(parsed(0.0, 0.0), handle)
    with val_path.open("wb") as handle:
        pickle.dump(parsed(1.0, 1.0), handle)
    monkeypatch.setattr(train_module, "VQ_REQUIRE_EXACT_CAPS", True)

    with pytest.raises(RuntimeError, match=message):
        train_module.collect_protocol_vq_data(
            {"train": [str(train_path)], "val": [str(val_path)], "test": []},
            train_cap=train_cap,
            val_cap=val_cap,
        )


def test_collect_protocol_vq_data_enforces_configured_parent_coverage(
    monkeypatch, train_module
):
    train_record = {
        "record_id": "train",
        "source_path": source("a" * 24),
        "source_key": source("a" * 24).casefold(),
        "parent_id": "a" * 24,
        "kind": "surface",
        "array": np.zeros((32, 32, 3), dtype=np.float32),
        "curvature_score": 0.0,
        "is_complex_source": False,
    }
    val_record = dict(
        train_record,
        record_id="val",
        source_path=source("b" * 24),
        source_key=source("b" * 24).casefold(),
        parent_id="b" * 24,
        array=np.ones((32, 32, 3), dtype=np.float32),
    )

    def fake_collect(_paths, _cap, seed):
        record_value = train_record if seed == 0 else val_record
        return [record_value], {
            "selected": 1,
            "parent_coverage": 0.5,
            "parent_cads_contributing": 1,
            "requested_parent_cads": 2,
        }, {"unique_records": 1}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)
    monkeypatch.setattr(train_module, "VQ_MIN_PARENT_COVERAGE", 0.9)

    with pytest.raises(RuntimeError, match="parent coverage.*train"):
        train_module.collect_protocol_vq_data(
            {"train": ["train-a", "train-b"], "val": ["val-a", "val-b"], "test": []},
            train_cap=1,
            val_cap=1,
        )


def test_final_parent_coverage_uses_direct_selected_parent_not_merged_provenance(
    monkeypatch, train_module
):
    train_record = {
        "record_id": "train",
        "source_path": source("a" * 24),
        "source_key": source("a" * 24).casefold(),
        "parent_id": "a" * 24,
        "provenance_parent_ids": ["a" * 24, "b" * 24],
        "kind": "surface",
        "array": np.zeros((32, 32, 3), dtype=np.float32),
        "curvature_score": 0.0,
        "is_complex_source": False,
    }
    val_record = {
        **train_record,
        "record_id": "val",
        "source_path": source("c" * 24),
        "source_key": source("c" * 24).casefold(),
        "parent_id": "c" * 24,
        "provenance_parent_ids": ["c" * 24],
        "array": np.ones((32, 32, 3), dtype=np.float32),
    }

    def fake_collect(_paths, _cap, seed):
        record_value = train_record if seed == 0 else val_record
        requested = 2 if seed == 0 else 1
        return [record_value], {
            "selected": 1,
            "requested_parent_cads": requested,
            "parent_cads_contributing": requested,
            "parent_coverage": 1.0,
        }, {"unique_records": 1}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)
    monkeypatch.setattr(train_module, "VQ_MIN_PARENT_COVERAGE", 0.75)

    with pytest.raises(RuntimeError, match="after exact filtering"):
        train_module.collect_protocol_vq_data(
            {"train": ["train-a", "train-b"], "val": ["val"], "test": []},
            train_cap=1,
            val_cap=1,
        )


def test_collect_protocol_vq_data_preserves_all_validation_parent_provenance(
    monkeypatch, train_module
):
    train_parent = "a" * 24
    val_parent_a = "b" * 24
    val_parent_b = "c" * 24
    train_record = {
        "record_id": "train",
        "source_path": source(train_parent),
        "source_key": source(train_parent).casefold(),
        "parent_id": train_parent,
        "provenance_parent_ids": [train_parent],
        "kind": "surface",
        "array": np.zeros((32, 32, 3), dtype=np.float32),
        "curvature_score": 0.0,
        "is_complex_source": False,
    }
    val_record = {
        **train_record,
        "record_id": "val",
        "source_path": source(val_parent_a),
        "source_key": source(val_parent_a).casefold(),
        "parent_id": val_parent_a,
        "provenance_parent_ids": [val_parent_a, val_parent_b],
        "array": np.ones((32, 32, 3), dtype=np.float32),
    }

    def fake_collect(_paths, _cap, seed):
        selected = train_record if seed == 0 else val_record
        return [selected], {
            "selected": 1,
            "requested_parent_cads": 1,
            "parent_cads_contributing": 1,
            "parent_coverage": 1.0,
        }, {"unique_records": 1}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)
    result = train_module.collect_protocol_vq_data(
        {"train": ["train"], "val": ["val"], "test": []},
        train_cap=1,
        val_cap=1,
    )

    assert result["val_parent_groups"] == [(val_parent_a, val_parent_b)]


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


def test_collect_protocol_vq_data_filters_before_cap_and_replenishes_train_inventory(
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
            "record_id": "train-unique-b",
            "source_path": source("b" * 24),
            "source_key": source("b" * 24).casefold(),
            "parent_id": "b" * 24,
            "kind": "surface",
            "array": np.ones((32, 32, 3), dtype=np.float32),
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
        {
            "record_id": "train-unique-c",
            "source_path": source("c" * 24),
            "source_key": source("c" * 24).casefold(),
            "parent_id": "c" * 24,
            "kind": "surface",
            "array": np.full((32, 32, 3), 2.0, dtype=np.float32),
            "curvature_score": 0.0,
            "is_complex_source": False,
        },
    ]
    val_records = [
        {
            "record_id": "val-shared",
            "source_path": source("d" * 24),
            "source_key": source("d" * 24).casefold(),
            "parent_id": "d" * 24,
            "kind": "surface",
            "array": shared.copy(),
            "curvature_score": 0.0,
            "is_complex_source": False,
        }
    ]

    def fake_collect(_paths, _cap, seed):
        records = train_records if seed == 0 else val_records
        return records, {
            "selected": len(records),
            "requested_parent_cads": len(records),
            "parent_cads_contributing": len(records),
            "parent_coverage": 1.0,
        }, {"unique_records": len(records)}

    monkeypatch.setattr(train_module, "_collect_protocol_inventory", fake_collect)

    result = train_module.collect_protocol_vq_data(
        {"train": ["train"], "val": ["val"], "test": []},
        train_cap=2,
        val_cap=1,
    )

    assert len(result["X_train"]) == 2
    assert result["cross_split_exact"]["train_records_removed"] == 1
    assert result["train_sampling"]["selected"] == 2
    assert result["train_sampling"]["effective_cap_met"] is True


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


def test_fsq_comparison_configs_split_dimension_and_codebook_confounds(train_module):
    configs = train_module.fsq_comparison_configs()

    assert [config["levels"] for config in configs] == [
        (8, 8, 8, 16),
        (4, 4, 4, 4, 4, 4),
        (4, 4, 4, 4, 4, 8),
    ]
    assert [int(np.prod(config["levels"])) for config in configs] == [8192, 4096, 8192]
    assert len({config["lr"] for config in configs}) == 1


def test_scaling_quantizer_configs_expose_matched_fsq_and_official_vq(train_module):
    configs = train_module.quantizer_comparison_configs(
        ("fsq_8192_4d", "fsq_4096_6d", "vq_4096_64d_random")
    )

    assert [config["name"] for config in configs] == [
        "fsq_8192_4d",
        "fsq_4096_6d",
        "vq_4096_64d_random",
    ]
    assert [config["codebook_size"] for config in configs] == [8192, 4096, 4096]
    assert configs[0]["quantizer"]["kind"] == "fsq"
    assert configs[1]["quantizer"]["latent_grid_dimensions"] == 6
    assert configs[2]["quantizer"] == {
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
    assert len({config["lr"] for config in configs}) == 1


def test_official_vq_4096_random_quantizer_forward_backward_contract(train_module):
    config = train_module.quantizer_comparison_configs(("vq_4096_64d_random",))[0]
    train_module.seed_vq_experiment(17)
    model = train_module.build_quantized_vqvae(config)
    latent = torch.randn(2, 64, 2, 2, requires_grad=True)

    quantized, loss, info = model.quantize(latent)
    (quantized.square().mean() + loss).backward()

    assert quantized.shape == latent.shape
    assert torch.isfinite(loss)
    assert latent.grad is not None
    assert info[2].shape == (8,)
    assert int(info[2].min()) >= 0
    assert int(info[2].max()) < 4096
    assert model.quantize.anchor == "random"
    assert model.quantize.embed_dim == 64
    assert model.quantize.num_embed == 4096


def test_official_vq_builder_adapts_half_latents_for_history_pool(train_module):
    config = train_module.quantizer_comparison_configs(("vq_4096_64d_random",))[0]
    train_module.seed_vq_experiment(17)
    model = train_module.build_quantized_vqvae(config)
    latent = torch.randn(2, 64, 2, 2, dtype=torch.float16, requires_grad=True)

    quantized, loss, info = model.quantize(latent)
    (quantized.float().square().mean() + loss.float()).backward()

    assert quantized.shape == latent.shape
    assert quantized.dtype == latent.dtype
    assert torch.isfinite(loss)
    assert latent.grad is not None
    assert info[2].shape == (8,)
    assert model.quantize.pool.features.dtype == torch.float32


def test_official_vq_4096_random_quantizer_aligns_history_pool_for_amp(train_module):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA AMP")
    config = train_module.quantizer_comparison_configs(("vq_4096_64d_random",))[0]
    train_module.seed_vq_experiment(17)
    model = train_module.build_quantized_vqvae(config).cuda()
    latent = torch.randn(2, 64, 2, 2, device="cuda", dtype=torch.float16)

    with torch.cuda.amp.autocast(enabled=True):
        quantized, loss, _ = model.quantize(latent)
        objective = quantized.square().mean() + loss
    objective.backward()

    assert torch.isfinite(objective).item()
    assert model.quantize.pool.features.dtype == torch.float32


def test_continuous_bypass_quantizer_preserves_latent_and_has_no_vq_loss(train_module):
    config = train_module.quantizer_comparison_configs(("continuous_bypass_64d",))[0]
    model = train_module.build_quantized_vqvae(config)
    latent = torch.randn(2, 64, 2, 2, requires_grad=True)

    quantized, loss, info = model.quantize(latent)
    (quantized.square().mean() + loss).backward()

    assert quantized is latent
    assert loss.item() == 0.0
    assert info[2].shape == (8,)
    assert latent.grad is not None
    assert config["quantizer"]["kind"] == "continuous_bypass"
    assert config["quantizer"]["downstream_compatible"] is False


def test_fsq_comparison_arms_share_all_non_quantizer_initialization(train_module):
    def shared_state_digests(levels):
        torch.manual_seed(17)
        np.random.seed(17)
        train_module.random.seed(17)
        model = train_module.build_fsq_vqvae(levels)
        return {
            name: hashlib.sha256(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for name, tensor in model.state_dict().items()
            if not name.startswith("quantize.")
        }

    arm_digests = [
        shared_state_digests(config["levels"])
        for config in train_module.fsq_comparison_configs()
    ]

    assert arm_digests[0]
    assert arm_digests[0] == arm_digests[1] == arm_digests[2]


def test_fsq_comparison_arms_reset_training_rng_after_model_construction(train_module):
    probes = []
    for config in train_module.fsq_comparison_configs():
        train_module.seed_vq_experiment(17)
        train_module.build_fsq_vqvae(config["levels"])
        train_module.seed_vq_experiment(17)
        probes.append(
            (
                torch.randperm(16).tolist(),
                np.random.permutation(16).tolist(),
                [train_module.random.random() for _ in range(4)],
            )
        )

    assert probes[0] == probes[1] == probes[2]


def test_vq_experiment_seed_controls_all_rngs(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "seeded_vq")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.setenv("NS_VQ_EXPERIMENT_SEED", "17")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        assert module.VQ_EXPERIMENT_SEED == 17
        assert module.load_report()["config"]["vq_experiment_seed"] == 17
    finally:
        sys.modules.pop("train", None)


def test_vq_run_manifest_captures_revision_runtime_protocol_and_controls(
    monkeypatch, train_module
):
    monkeypatch.setattr(train_module.sys, "argv", ["train.py", "--stage", "vqsweep"])
    monkeypatch.setenv("NS_MANIFEST_TEST_VALUE", "recorded")
    split_metadata = {
        "protocol_sha256": "protocol-sha",
        "split_pickle_sha256": "split-sha",
        "split_pickle_binding": "VERIFIED",
    }
    configs = train_module.fsq_comparison_configs()

    manifest = train_module.build_vq_run_manifest(split_metadata, configs)

    assert len(manifest["git"]["commit"]) == 40
    assert isinstance(manifest["git"]["dirty"], bool)
    assert manifest["launch"]["argv"] == ["train.py", "--stage", "vqsweep"]
    assert manifest["launch"]["relevant_env"]["NS_MANIFEST_TEST_VALUE"] == "recorded"
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["platform"]
    assert manifest["runtime"]["pytorch"] == torch.__version__
    assert "cuda" in manifest["runtime"]
    assert "cudnn" in manifest["runtime"]
    assert "gpu" in manifest["runtime"]
    assert manifest["runtime"]["tf32_matmul"] is train_module.torch.backends.cuda.matmul.allow_tf32
    assert manifest["runtime"]["tf32_cudnn"] is train_module.torch.backends.cudnn.allow_tf32
    assert manifest["runtime"]["amp"] is train_module.VQ_AMP
    assert manifest["experiment"]["seed"] == train_module.VQ_EXPERIMENT_SEED
    assert manifest["experiment"]["protocol"] == split_metadata
    assert manifest["experiment"]["train_cap"] == train_module.VQ_SWEEP_TRAIN_CAP
    assert manifest["experiment"]["val_cap"] == train_module.VQ_VAL_SAMPLES
    assert manifest["experiment"]["epochs"] == train_module.VQ_SWEEP_EPOCHS
    assert manifest["experiment"]["batch_size"] == train_module.VQ_BS
    assert manifest["experiment"]["arms"] == [
        {
            "name": config["name"],
            "levels": list(config["levels"]),
            "codebook": int(np.prod(config["levels"])),
            "lr": config["lr"],
        }
        for config in configs
    ]


def test_vq_run_manifest_records_stage_specific_training_controls(
    monkeypatch, train_module
):
    monkeypatch.setattr(
        train_module,
        "current_git_state",
        lambda: {"commit": "a" * 40, "dirty": False},
    )
    manifest = train_module.build_vq_run_manifest(
        {
            "protocol_sha256": "protocol-sha",
            "split_pickle_sha256": "split-sha",
        },
        [{"name": "vqvae", "levels": (2, 2), "lr": 1e-4}],
        train_cap=60000,
        val_cap=12000,
        epochs=120,
    )

    assert manifest["experiment"]["train_cap"] == 60000
    assert manifest["experiment"]["val_cap"] == 12000
    assert manifest["experiment"]["epochs"] == 120


def test_formal_vq_run_rejects_dirty_manifest(train_module):
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        train_module.require_clean_vq_run({"git": {"commit": "a" * 40, "dirty": True}})

    assert train_module.require_clean_vq_run(
        {"git": {"commit": "a" * 40, "dirty": False}}
    ) is True


@pytest.mark.parametrize("stage_name", ["stage_vqsweep", "stage_vqvae"])
def test_formal_vq_stage_rejects_dirty_manifest_before_sampling(
    monkeypatch, train_module, stage_name
):
    split = {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []}
    split_metadata = {
        "protocol_sha256": "protocol-sha",
        "split_pickle_sha256": "split-sha",
        "split_pickle_binding": "VERIFIED",
    }
    monkeypatch.setattr(
        train_module,
        "load_verified_protocol_split",
        lambda return_metadata=False: (split, split_metadata) if return_metadata else split,
    )
    monkeypatch.setattr(
        train_module,
        "build_vq_run_manifest",
        lambda *_args, **_kwargs: {"git": {"commit": "a" * 40, "dirty": True}},
    )
    monkeypatch.setattr(
        train_module,
        "collect_protocol_vq_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sampling started before clean-run gate")
        ),
    )

    with pytest.raises(RuntimeError, match="clean committed worktree"):
        getattr(train_module, stage_name)()


def promotion_metrics(
    perplexity=900.0, curved_parent_mse=4e-5, nonfinite=0, coverage=0.25
):
    return {
        "code_usage": {"entropy_perplexity": perplexity, "coverage": coverage},
        "parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": curved_parent_mse}
        },
        "nonfinite_val_samples": nonfinite,
        "train_parent_coverage": 0.95,
        "val_parent_coverage": 1.0,
    }


@pytest.mark.parametrize(
    "overrides, reason_fragment",
    [
        ({"perplexity": 799.0}, "perplexity"),
        ({"curved_parent_mse": 5.1e-5}, "curved parent-cluster MSE"),
        ({"nonfinite": 1}, "nonfinite"),
    ],
)
def test_vq_promotion_rejects_failed_representation_gates(
    train_module, overrides, reason_fragment
):
    promotion = train_module.evaluate_vq_promotion(
        promotion_metrics(**overrides),
        min_perplexity=800.0,
        max_curved_parent_mse=5e-5,
        min_parent_coverage=0.9,
    )

    assert promotion["eligible"] is False
    assert any(reason_fragment in reason for reason in promotion["reasons"])


def test_vq_promotion_requires_both_parent_coverage_gates(train_module):
    metrics = promotion_metrics()
    metrics["train_parent_coverage"] = 0.89

    promotion = train_module.evaluate_vq_promotion(
        metrics,
        min_perplexity=800.0,
        max_curved_parent_mse=5e-5,
        min_parent_coverage=0.9,
    )

    assert promotion["eligible"] is False
    assert any("train parent coverage" in reason for reason in promotion["reasons"])


def test_vq_promotion_accepts_all_finite_metrics_at_thresholds(train_module):
    promotion = train_module.evaluate_vq_promotion(
        promotion_metrics(perplexity=800.0, curved_parent_mse=5e-5),
        min_perplexity=800.0,
        max_curved_parent_mse=5e-5,
        min_parent_coverage=0.9,
    )

    assert promotion["eligible"] is True
    assert promotion["reasons"] == []


@pytest.mark.parametrize(
    "metrics, perplexity_history, coverage_history, best_curved, expected, reason",
    [
        (promotion_metrics(perplexity=900.0, curved_parent_mse=4e-5), [1000.0], [0.25], 5e-5, True, None),
        (promotion_metrics(perplexity=899.0, curved_parent_mse=4e-5), [1000.0], [0.25], 5e-5, False, "perplexity is below"),
        (promotion_metrics(perplexity=900.0, curved_parent_mse=4e-5, coverage=0.224), [1000.0], [0.25], 5e-5, False, "coverage is below"),
        (promotion_metrics(perplexity=900.0, curved_parent_mse=5e-5), [1000.0], [0.25], 5e-5, False, "curved parent-cluster MSE"),
        (promotion_metrics(perplexity=900.0, curved_parent_mse=4e-5, nonfinite=1), [1000.0], [0.25], 5e-5, False, "nonfinite"),
    ],
)
def test_vq_checkpoint_selector_requires_curved_improvement_and_stable_usage(
    train_module,
    metrics,
    perplexity_history,
    coverage_history,
    best_curved,
    expected,
    reason,
):
    decision = train_module.evaluate_vq_checkpoint_candidate(
        metrics,
        prior_perplexities=perplexity_history,
        prior_coverages=coverage_history,
        best_curved_parent_mse=best_curved,
    )

    assert decision["selected"] is expected
    if reason is not None:
        assert any(reason in item for item in decision["reasons"])


def test_vq_sweep_summary_does_not_promote_mse_leader_when_all_arms_fail(train_module):
    results = [
        {"name": "low-mse", "best_val_recon": 0.01, "promotion": {"eligible": False}},
        {"name": "higher-mse", "best_val_recon": 0.02, "promotion": {"eligible": False}},
    ]

    summary = train_module.summarize_vq_sweep(results)

    assert [item["name"] for item in summary["mse_ranking"]] == ["low-mse", "higher-mse"]
    assert summary["promotion_eligible_candidates"] == []
    assert summary["winner"] is None
    assert summary["status"] == "NO_PROMOTED_ARM"


def test_vq_sweep_summary_selects_lowest_mse_only_among_promoted_arms(train_module):
    results = [
        {"name": "failed-low", "best_val_recon": 0.01, "promotion": {"eligible": False}},
        {"name": "eligible-high", "best_val_recon": 0.03, "promotion": {"eligible": True}},
        {"name": "eligible-mid", "best_val_recon": 0.02, "promotion": {"eligible": True}},
    ]

    summary = train_module.summarize_vq_sweep(results)

    assert [item["name"] for item in summary["promotion_eligible_candidates"]] == [
        "eligible-mid",
        "eligible-high",
    ]
    assert summary["winner"]["name"] == "eligible-mid"
    assert summary["status"] == "PROMOTED_ARM_AVAILABLE"


def test_downstream_vq_gate_blocks_unpromoted_model_by_default(
    tmp_path, monkeypatch, train_module
):
    report = train_module.load_report()
    report["stages"]["vqvae"] = {
        "promotion": train_module.evaluate_vq_promotion(promotion_metrics(perplexity=10.0))
    }
    train_module.save_report(report)
    monkeypatch.setattr(train_module, "ALLOW_UNPROMOTED_VQ_DOWNSTREAM", False)

    with pytest.raises(RuntimeError, match="VQ promotion gate blocked sequence"):
        train_module.require_vq_promotion("sequence")

    gate = train_module.load_report()["downstream_vq_gate"]["sequence"]
    assert gate["allowed"] is False
    assert gate["override_enabled"] is False


def test_downstream_vq_gate_records_explicit_diagnostic_override(
    tmp_path, monkeypatch, train_module
):
    report = train_module.load_report()
    report["stages"]["vqvae"] = {
        "promotion": train_module.evaluate_vq_promotion(promotion_metrics(perplexity=10.0))
    }
    train_module.save_report(report)
    monkeypatch.setattr(train_module, "ALLOW_UNPROMOTED_VQ_DOWNSTREAM", True)

    gate = train_module.require_vq_promotion("ar")

    assert gate["allowed"] is True
    assert gate["eligible"] is False
    assert gate["override_enabled"] is True
    assert train_module.load_report()["downstream_vq_gate"]["ar"] == gate


def test_downstream_vq_gate_allows_promoted_model_without_missing_reason(
    train_module,
):
    report = train_module.load_report()
    report["stages"]["vqvae"] = {
        "promotion": train_module.evaluate_vq_promotion(promotion_metrics())
    }
    train_module.save_report(report)

    gate = train_module.require_vq_promotion("sequence")

    assert gate == {
        "eligible": True,
        "override_enabled": False,
        "allowed": True,
        "reasons": [],
    }


def test_vq_checkpoint_binding_rejects_replaced_checkpoint(
    tmp_path, monkeypatch, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_payload = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": list(train_module.FSQ_LEVELS),
        "checkpoint_context": {
            "git_commit": "a" * 40,
            "split_pickle_sha256": "split-sha",
            "protocol_sha256": "protocol-sha",
            "train_parent_coverage": 0.95,
            "val_parent_coverage": 1.0,
        },
        "checkpoint_epoch": 3,
        "validation_metrics": promotion_metrics(),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    promotion = train_module.build_checkpoint_promotion(
        checkpoint_path,
        checkpoint_payload,
        min_perplexity=800.0,
        max_curved_parent_mse=5e-5,
        min_parent_coverage=0.9,
    )

    torch.save({**checkpoint_payload, "checkpoint_epoch": 4}, checkpoint_path)

    with pytest.raises(RuntimeError, match="checkpoint SHA-256"):
        train_module.verify_vq_promotion_binding(
            promotion,
            checkpoint_path,
            {
                "protocol_sha256": "protocol-sha",
                "split_pickle_sha256": "split-sha",
            },
            current_git={"commit": "a" * 40, "dirty": False},
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("split", "split binding"),
        ("levels", "FSQ levels"),
        ("commit", "Git commit"),
    ],
)
def test_vq_checkpoint_binding_rejects_stale_context(
    tmp_path, monkeypatch, train_module, mutation, message
):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_payload = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": list(train_module.FSQ_LEVELS),
        "checkpoint_context": {
            "git_commit": "a" * 40,
            "split_pickle_sha256": "split-sha",
            "protocol_sha256": "protocol-sha",
            "train_parent_coverage": 0.95,
            "val_parent_coverage": 1.0,
        },
        "checkpoint_epoch": 3,
        "validation_metrics": promotion_metrics(),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    promotion = train_module.build_checkpoint_promotion(checkpoint_path, checkpoint_payload)
    split_metadata = {
        "protocol_sha256": "protocol-sha",
        "split_pickle_sha256": "split-sha",
    }
    current_git = {"commit": "a" * 40, "dirty": False}
    if mutation == "split":
        split_metadata["split_pickle_sha256"] = "different-split"
    elif mutation == "levels":
        monkeypatch.setattr(train_module, "FSQ_LEVELS", (4, 4, 4, 4, 4, 4))
    else:
        current_git["commit"] = "b" * 40

    with pytest.raises(RuntimeError, match=message):
        train_module.verify_vq_promotion_binding(
            promotion,
            checkpoint_path,
            split_metadata,
            current_git=current_git,
        )


def test_vq_checkpoint_binding_rejects_tampered_promotion_metrics(
    tmp_path, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_payload = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": list(train_module.FSQ_LEVELS),
        "checkpoint_context": {
            "git_commit": "a" * 40,
            "split_pickle_sha256": "split-sha",
            "protocol_sha256": "protocol-sha",
            "train_parent_coverage": 0.95,
            "val_parent_coverage": 1.0,
        },
        "checkpoint_epoch": 3,
        "validation_metrics": promotion_metrics(),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    promotion = train_module.build_checkpoint_promotion(checkpoint_path, checkpoint_payload)
    promotion["observed"]["perplexity"] = 9999.0

    with pytest.raises(RuntimeError, match="promotion metrics"):
        train_module.verify_vq_promotion_binding(
            promotion,
            checkpoint_path,
            {
                "protocol_sha256": "protocol-sha",
                "split_pickle_sha256": "split-sha",
            },
            current_git={"commit": "a" * 40, "dirty": False},
        )


def test_vq_checkpoint_binding_preserves_nondefault_promotion_thresholds(
    tmp_path, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_payload = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": list(train_module.FSQ_LEVELS),
        "checkpoint_context": {
            "git_commit": "a" * 40,
            "split_pickle_sha256": "split-sha",
            "protocol_sha256": "protocol-sha",
            "train_parent_coverage": 0.95,
            "val_parent_coverage": 1.0,
        },
        "checkpoint_epoch": 3,
        "validation_metrics": promotion_metrics(perplexity=450.0),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    promotion = train_module.build_checkpoint_promotion(
        checkpoint_path,
        checkpoint_payload,
        min_perplexity=400.0,
        max_curved_parent_mse=6e-5,
        min_parent_coverage=0.8,
    )

    assert train_module.verify_vq_promotion_binding(
        promotion,
        checkpoint_path,
        {
            "protocol_sha256": "protocol-sha",
            "split_pickle_sha256": "split-sha",
        },
        current_git={"commit": "a" * 40, "dirty": False},
    ) is True


def test_diagnostic_override_cannot_bypass_checkpoint_binding(
    tmp_path, monkeypatch, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_payload = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "fsq_levels": list(train_module.FSQ_LEVELS),
        "checkpoint_context": {
            "git_commit": "a" * 40,
            "split_pickle_sha256": "split-sha",
            "protocol_sha256": "protocol-sha",
            "train_parent_coverage": 0.95,
            "val_parent_coverage": 1.0,
        },
        "checkpoint_epoch": 3,
        "validation_metrics": promotion_metrics(perplexity=10.0),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    promotion = train_module.build_checkpoint_promotion(checkpoint_path, checkpoint_payload)
    report = train_module.load_report()
    report["stages"]["vqvae"] = {"promotion": promotion}
    train_module.save_report(report)
    monkeypatch.setattr(train_module, "VQVAE_PT", str(checkpoint_path))
    monkeypatch.setattr(train_module, "ALLOW_UNPROMOTED_VQ_DOWNSTREAM", True)
    torch.save({**checkpoint_payload, "checkpoint_epoch": 4}, checkpoint_path)

    with pytest.raises(RuntimeError, match="checkpoint SHA-256"):
        train_module.require_bound_vq_checkpoint(
            "sequence",
            {
                "protocol_sha256": "protocol-sha",
                "split_pickle_sha256": "split-sha",
            },
            current_git={"commit": "a" * 40, "dirty": False},
        )


def test_protocol_split_stage_accepts_only_verified_protocol_outputs(
    tmp_path, monkeypatch
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    split = {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []}
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump(split, handle)
    split_sha256 = hashlib.sha256((protocol_dir / "split.pkl").read_bytes()).hexdigest()
    (protocol_dir / "protocol_summary.json").write_text(
        '{"status":"VERIFIED","protocol_sha256":"abc",'
        f'"split_pickle_sha256":"{split_sha256}",'
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
        assert module.load_report()["stages"]["split"]["split_pickle_binding"] == "VERIFIED"
    finally:
        sys.modules.pop("train", None)


def test_protocol_split_stage_rejects_split_pickle_hash_mismatch(tmp_path, monkeypatch):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    with (protocol_dir / "split.pkl").open("wb") as handle:
        pickle.dump({"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []}, handle)
    (protocol_dir / "protocol_summary.json").write_text(
        '{"status":"VERIFIED","protocol_sha256":"abc",'
        '"split_pickle_sha256":"' + "0" * 64 + '",'
        '"parent_overlap_counts":{"train__val":0,"train__test":0,"val__test":0}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "protocol_split_hash_mismatch")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="split.pkl SHA-256"):
            module.stage_split()
    finally:
        sys.modules.pop("train", None)


@pytest.mark.parametrize("stage_name", ["stage_vqsweep", "stage_vqvae"])
def test_direct_vq_stage_rejects_split_pickle_hash_mismatch_before_sampling(
    tmp_path, monkeypatch, stage_name
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    split_path = protocol_dir / "split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(
            {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []},
            handle,
        )
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    (protocol_dir / "protocol_summary.json").write_text(
        __import__("json").dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": "protocol-hash",
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
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", f"direct_{stage_name}_hash_mismatch")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with split_path.open("wb") as handle:
            pickle.dump(
                {"train": [source("c" * 24)], "val": [source("d" * 24)], "test": []},
                handle,
            )

        def fail_if_sampling_starts(*_args, **_kwargs):
            raise AssertionError("sampling started before split verification")

        monkeypatch.setattr(module, "collect_protocol_vq_data", fail_if_sampling_starts)
        with pytest.raises(RuntimeError, match="split.pkl SHA-256"):
            getattr(module, stage_name)()
    finally:
        sys.modules.pop("train", None)


def test_verified_protocol_split_loader_rejects_unbound_legacy_summary(
    tmp_path, monkeypatch
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
                "protocol_sha256": "protocol-hash",
                "parent_overlap_counts": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "unbound_protocol_split")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="split_pickle_sha256"):
            module.load_verified_protocol_split()
    finally:
        sys.modules.pop("train", None)


def test_verified_protocol_split_loader_recomputes_parent_overlap_from_split(
    tmp_path, monkeypatch
):
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    shared_parent = "a" * 24
    split_path = protocol_dir / "split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(
            {
                "train": [source(shared_parent, index=1)],
                "val": [source(shared_parent, index=2)],
                "test": [],
            },
            handle,
        )
    (protocol_dir / "protocol_summary.json").write_text(
        __import__("json").dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": "protocol-hash",
                "split_pickle_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "parent_overlap_counts": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_PROTOCOL_DIR", str(protocol_dir))
    monkeypatch.setenv("NS_OUTBASE", str(tmp_path / "out"))
    monkeypatch.setenv("NS_OUT", "actual_parent_overlap")
    monkeypatch.setenv("NS_PROTOCOL_V2", "1")
    monkeypatch.syspath_prepend(str(IMPROVEMENTS_DIR))
    sys.modules.pop("train", None)
    try:
        module = importlib.import_module("train")
        with pytest.raises(RuntimeError, match="actual split parent overlap"):
            module.load_verified_protocol_split()
    finally:
        sys.modules.pop("train", None)


@pytest.mark.parametrize("stage_name", ["stage_sequence", "stage_ar", "stage_ar_sweep"])
def test_every_downstream_stage_verifies_protocol_split_before_inputs(
    monkeypatch, train_module, stage_name
):
    def split_sentinel(*_args, **_kwargs):
        raise RuntimeError("SPLIT_SENTINEL")

    monkeypatch.setattr(train_module, "load_verified_protocol_split", split_sentinel)
    monkeypatch.setattr(
        train_module,
        "require_vq_promotion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("promotion checked before split")
        ),
    )

    with pytest.raises(RuntimeError, match="SPLIT_SENTINEL"):
        getattr(train_module, stage_name)()


@pytest.mark.parametrize("stage_name", ["stage_sequence", "stage_ar", "stage_ar_sweep"])
def test_every_downstream_stage_requires_bound_checkpoint_before_inputs(
    monkeypatch, train_module, stage_name
):
    split = {"train": [source("a" * 24)], "val": [source("b" * 24)], "test": []}
    split_metadata = {
        "protocol_sha256": "protocol-sha",
        "split_pickle_sha256": "split-sha",
    }
    monkeypatch.setattr(
        train_module,
        "load_verified_protocol_split",
        lambda return_metadata=False: (split, split_metadata) if return_metadata else split,
    )
    monkeypatch.setattr(
        train_module,
        "require_bound_vq_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("BINDING_SENTINEL")),
        raising=False,
    )
    monkeypatch.setattr(
        train_module,
        "require_vq_promotion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy promotion-only gate was used")
        ),
    )

    with pytest.raises(RuntimeError, match="BINDING_SENTINEL"):
        getattr(train_module, stage_name)()


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


class ValidationDriftAutoencoder(TinyQuantizedAutoencoder):
    def __init__(self):
        super().__init__()
        self.validation_calls = 0

    def decoder(self, value):
        if self.training:
            return value
        self.validation_calls += 1
        if self.validation_calls == 1:
            return value
        return torch.zeros_like(value)


class GlobalImprovementAutoencoder(TinyQuantizedAutoencoder):
    def __init__(self):
        super().__init__()
        self.validation_calls = 0

    def decoder(self, value):
        if self.training:
            return value
        self.validation_calls += 1
        if self.validation_calls == 1:
            return torch.zeros_like(value)
        return value


def test_train_vqvae_checkpoint_selector_rejects_global_mse_only_improvement(
    tmp_path, monkeypatch, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    samples = np.ones((1, 3, 32, 32), dtype=np.float32)
    summaries = [
        promotion_metrics(perplexity=1000.0, curved_parent_mse=0.1),
        promotion_metrics(perplexity=1000.0, curved_parent_mse=0.2),
    ]
    for summary in summaries:
        summary["reconstruction_mse"] = {
            "surface_planar_like": {"samples": 0, "nonfinite_samples": 0, "mse": None},
            "surface_curved_proxy": {"samples": 1, "nonfinite_samples": 0, "mse": 0.1},
            "edge": {"samples": 0, "nonfinite_samples": 0, "mse": None},
        }
        summary["parent_cluster_mse"] = {
            "unique_patch_samples": 1,
            "parent_patch_contributions": 1,
            "parent_clusters": 1,
            "nonfinite_samples": 0,
            "nonfinite_parent_contributions": 0,
            "nonfinite_parents": 0,
            "mse": summary["parent_cluster_reconstruction_mse"]["surface_curved_proxy"]["mse"],
        }

    class FixedAccumulator:
        instances = 0

        def __init__(self, *_args, **_kwargs):
            self.index = type(self).instances
            type(self).instances += 1

        def update(self, *_args, **_kwargs):
            return None

        def summary(self):
            return summaries[self.index]

    monkeypatch.setattr(train_module, "VQValidationAccumulator", FixedAccumulator)

    history, best_val, meta = train_module._train_vqvae(
        GlobalImprovementAutoencoder(),
        samples,
        samples,
        epochs=2,
        bs=1,
        lr=0.0,
        tag="representation-selector",
        save_path=str(checkpoint_path),
        amp_enabled=False,
        val_buckets=["surface_curved_proxy"],
        val_parent_ids=["parent-a"],
        codebook_size=4,
        fsq_levels=(2, 2),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert history[0][1] == pytest.approx(1.0)
    assert history[1][1] == pytest.approx(0.0)
    assert best_val == pytest.approx(0.0)
    assert meta["global_best_epoch"] == 1
    assert meta["checkpoint_epoch"] == 0
    assert checkpoint["checkpoint_epoch"] == 0
    assert checkpoint["validation_metrics"] == summaries[0]


def test_train_vqvae_binds_best_epoch_metrics_and_context_to_checkpoint(
    tmp_path, train_module
):
    checkpoint_path = tmp_path / "best.pt"
    samples = np.ones((1, 3, 32, 32), dtype=np.float32)
    context = {
        "git_commit": "a" * 40,
        "split_pickle_sha256": "split-sha",
        "protocol_sha256": "protocol-sha",
        "train_parent_coverage": 0.95,
        "val_parent_coverage": 1.0,
    }

    _, _, meta = train_module._train_vqvae(
        ValidationDriftAutoencoder(),
        samples,
        samples,
        epochs=2,
        bs=1,
        lr=0.0,
        tag="best-checkpoint-binding",
        save_path=str(checkpoint_path),
        amp_enabled=False,
        val_buckets=["surface_curved_proxy"],
        val_parent_ids=["parent-a"],
        codebook_size=4,
        fsq_levels=(2, 2),
        checkpoint_context=context,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert meta["best_epoch"] == 0
    assert meta["best_val_metrics"]["reconstruction_mse"]["surface_curved_proxy"]["mse"] == 0.0
    assert meta["last_val_metrics"]["reconstruction_mse"]["surface_curved_proxy"]["mse"] == 1.0
    assert checkpoint["checkpoint_epoch"] == meta["checkpoint_epoch"]
    assert checkpoint["validation_metrics"] == meta["best_val_metrics"]
    assert checkpoint["checkpoint_context"] == context
    assert checkpoint["fsq_levels"] == [2, 2]


def test_ar_sequence_loader_rejects_stale_vq_binding(
    tmp_path, monkeypatch, train_module
):
    sequence_path = tmp_path / "sequences.pkl"
    with sequence_path.open("wb") as handle:
        pickle.dump(
            {
                "train": [],
                "val": [],
                "vq_binding": {"checkpoint_sha256": "old-checkpoint"},
            },
            handle,
        )
    monkeypatch.setattr(train_module, "SEQ_PKL", str(sequence_path))

    with pytest.raises(RuntimeError, match="sequence VQ binding"):
        train_module._load_ar_seqs(
            max_seq_len=2048,
            expected_vq_binding={"checkpoint_sha256": "current-checkpoint"},
        )


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
        val_parent_ids=["parent-a", "parent-b"],
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
    assert record["val_reconstruction_mse"]["edge"] == {
        "samples": 1,
        "nonfinite_samples": 0,
        "mse": 0.0,
    }
    assert record["val_parent_cluster_mse"] == {
        "unique_patch_samples": 2,
        "parent_patch_contributions": 2,
        "parent_clusters": 2,
        "nonfinite_samples": 0,
        "nonfinite_parent_contributions": 0,
        "nonfinite_parents": 0,
        "mse": 0.0,
    }
    assert meta["last_val_metrics"] == {
        "code_usage": record["val_code_usage"],
        "reconstruction_mse": record["val_reconstruction_mse"],
        "parent_cluster_mse": record["val_parent_cluster_mse"],
        "parent_cluster_reconstruction_mse": record["val_parent_cluster_reconstruction_mse"],
        "nonfinite_val_samples": 0,
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


def test_curved_plateau_mode_preserves_global_nonfinite_hard_stop(
    tmp_path, monkeypatch, train_module
):
    monkeypatch.setattr(train_module, "VQ_PLATEAU_METRIC", "curved_parent_mse")
    samples = np.zeros((2, 3, 32, 32), dtype=np.float32)
    stop_config = train_module.VQVAEStopConfig(
        min_epochs=1,
        patience=99,
        max_nonfinite_val_epochs=1,
        min_delta=1e-5,
    )

    _, _, meta = train_module._train_vqvae(
        PartiallyNonfiniteDecoderAutoencoder(),
        samples,
        samples,
        epochs=3,
        bs=2,
        lr=0.0,
        tag="curved-plateau-nonfinite-stop",
        amp_enabled=False,
        val_buckets=["surface_curved_proxy", "edge"],
        val_parent_ids=["parent-a", "parent-b"],
        codebook_size=4,
        stop_config=stop_config,
    )

    assert meta["epochs_ran"] == 1
    assert meta["stopped_early"] is True
    assert meta["stop_reason"] == "nonfinite_val_epochs=1"


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
        "nonfinite_samples": 2,
        "mse": None,
    }
    assert not best_path.exists()
