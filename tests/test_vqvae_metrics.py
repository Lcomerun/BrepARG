import math

import numpy as np
import pytest
import torch

from breparg_improvements.vqvae_metrics import (
    VQValidationAccumulator,
    aggregate_code_usage,
    patch_bucket,
    summarize_parent_cluster_mse,
    summarize_bucket_mse,
)


def test_patch_bucket_distinguishes_edge_planar_and_curved_surface():
    plane = np.zeros((32, 32, 3), dtype=np.float32)
    plane[..., 0] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]
    plane[..., 1] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :]
    curved = plane.copy()
    curved[..., 2] = 0.2 * np.sin(plane[..., 0] * math.pi)

    assert patch_bucket({"kind": "edge", "array": curved}) == "edge"
    assert patch_bucket({"kind": "surface", "array": plane}) == "surface_planar_like"
    assert patch_bucket({"kind": "surface", "array": curved}) == "surface_curved_proxy"


def test_patch_bucket_is_rotation_invariant_for_a_tilted_plane():
    plane = np.zeros((32, 32, 3), dtype=np.float32)
    plane[..., 0] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]
    plane[..., 1] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :]
    angle = math.radians(37.0)
    tilted = plane.copy()
    tilted[..., 0] = math.cos(angle) * plane[..., 0] + math.sin(angle) * plane[..., 2]
    tilted[..., 2] = -math.sin(angle) * plane[..., 0] + math.cos(angle) * plane[..., 2]

    assert patch_bucket({"kind": "surface", "array": tilted}) == "surface_planar_like"


def test_aggregate_code_usage_uses_complete_histogram():
    summary = aggregate_code_usage([2, 2, 0, 0], codebook_size=4)

    assert summary["tokens"] == 4
    assert summary["unique_bins"] == 2
    assert summary["coverage"] == 0.5
    assert summary["entropy_perplexity"] == 2.0


def test_aggregate_code_usage_rejects_wrong_histogram_size():
    with pytest.raises(ValueError, match="histogram length"):
        aggregate_code_usage([1, 2], codebook_size=4)


def test_bucket_mse_is_sample_weighted_and_keeps_empty_buckets_visible():
    summary = summarize_bucket_mse(
        ["edge", "edge", "surface_curved_proxy"],
        [1.0, 3.0, 2.0],
    )

    assert summary["edge"] == {"samples": 2, "nonfinite_samples": 0, "mse": 2.0}
    assert summary["surface_curved_proxy"] == {
        "samples": 1,
        "nonfinite_samples": 0,
        "mse": 2.0,
    }
    assert summary["surface_planar_like"] == {
        "samples": 0,
        "nonfinite_samples": 0,
        "mse": None,
    }


def test_bucket_mse_reports_nonfinite_samples_instead_of_silently_dropping_them():
    summary = summarize_bucket_mse(
        ["edge", "surface_curved_proxy"],
        [float("nan"), float("inf")],
    )

    assert summary["edge"] == {"samples": 0, "nonfinite_samples": 1, "mse": None}
    assert summary["surface_curved_proxy"] == {
        "samples": 0,
        "nonfinite_samples": 1,
        "mse": None,
    }


def test_bucket_mse_rejects_mismatched_inputs_instead_of_truncating():
    with pytest.raises(ValueError, match="same length"):
        summarize_bucket_mse(["edge", "edge"], [1.0])


def test_parent_cluster_mse_averages_within_parent_before_across_parents():
    summary = summarize_parent_cluster_mse(
        ["parent-a", "parent-a", "parent-b", "parent-c"],
        [1.0, 3.0, 5.0, float("nan")],
    )

    assert summary == {
        "unique_patch_samples": 3,
        "parent_patch_contributions": 3,
        "parent_clusters": 2,
        "nonfinite_samples": 1,
        "nonfinite_parent_contributions": 1,
        "nonfinite_parents": 1,
        "mse": 3.5,
    }


def test_parent_cluster_mse_expands_shared_patch_over_all_provenance_parents():
    summary = summarize_parent_cluster_mse(
        [("parent-a", "parent-b"), ("parent-a",)],
        [2.0, 4.0],
    )

    assert summary == {
        "unique_patch_samples": 2,
        "parent_patch_contributions": 3,
        "parent_clusters": 2,
        "nonfinite_samples": 0,
        "nonfinite_parent_contributions": 0,
        "nonfinite_parents": 0,
        "mse": 2.5,
    }


def test_validation_accumulator_uses_all_batches_for_usage_and_bucket_mse():
    accumulator = VQValidationAccumulator(
        codebook_size=4,
        buckets=["edge", "surface_curved_proxy", "edge"],
    )
    accumulator.update(torch.tensor([1.0, 3.0]), torch.tensor([0, 1, 1]))
    accumulator.update(torch.tensor([2.0]), torch.tensor([1, 2]))

    summary = accumulator.summary()

    assert summary["code_usage"]["tokens"] == 5
    assert summary["code_usage"]["unique_bins"] == 3
    assert summary["reconstruction_mse"]["edge"] == {
        "samples": 2,
        "nonfinite_samples": 0,
        "mse": 1.5,
    }
    assert summary["reconstruction_mse"]["surface_curved_proxy"] == {
        "samples": 1,
        "nonfinite_samples": 0,
        "mse": 3.0,
    }


def test_validation_accumulator_reports_parent_cluster_mse_for_each_bucket():
    accumulator = VQValidationAccumulator(
        codebook_size=4,
        buckets=["surface_curved_proxy", "surface_curved_proxy", "edge"],
        parent_ids=["parent-a", "parent-a", "parent-b"],
    )
    accumulator.update(torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0, 1, 2]))

    summary = accumulator.summary()

    assert summary["parent_cluster_reconstruction_mse"]["surface_curved_proxy"] == {
        "unique_patch_samples": 2,
        "parent_patch_contributions": 2,
        "parent_clusters": 1,
        "nonfinite_samples": 0,
        "nonfinite_parent_contributions": 0,
        "nonfinite_parents": 0,
        "mse": 2.0,
    }
