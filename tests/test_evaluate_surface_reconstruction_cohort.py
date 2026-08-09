import json

import numpy as np
import pytest

from tools.evaluate_surface_reconstruction_cohort import (
    aggregate_across_seeds,
    aggregate_checkpoint_rows,
    discover_final_checkpoints,
    latest_attempt_rows,
    surface_reconstruction_metrics,
)


def test_surface_metrics_bucket_curved_and_planar(monkeypatch):
    monkeypatch.setattr(
        "tools.evaluate_surface_reconstruction_cohort.surface_plane_residual",
        lambda surface: float(surface[0, 0, 0]),
    )
    target = np.zeros((2, 32, 32, 3), dtype=np.float32)
    target[1, 0, 0, 0] = 1.0
    reconstructed = target.copy(); reconstructed[0] += 1.0; reconstructed[1] += 2.0
    metrics = surface_reconstruction_metrics(target, reconstructed, curved_threshold=0.5)
    assert metrics["planar_surface_count"] == 1
    assert metrics["curved_surface_count"] == 1
    assert metrics["planar_mse"] == pytest.approx(1.0)
    assert metrics["curved_mse"] == pytest.approx(4.0)


def test_checkpoint_aggregation_is_cad_equal():
    summary = aggregate_checkpoint_rows([
        {"status": "saved", "surface_mse": 1.0, "curved_mse": 2.0, "planar_mse": 0.5,
         "surface_count": 100, "curved_surface_count": 10, "planar_surface_count": 90},
        {"status": "saved", "surface_mse": 3.0, "curved_mse": 4.0, "planar_mse": 1.5,
         "surface_count": 1, "curved_surface_count": 1, "planar_surface_count": 0},
    ])
    assert summary["cad_equal_surface_mse"] == 2.0
    assert summary["cad_equal_curved_mse"] == 3.0


def test_discover_final_checkpoints_requires_final_epoch(tmp_path):
    seed = tmp_path / "seed0"; seed.mkdir()
    checkpoint = seed / "arm_final.pt"; checkpoint.write_bytes(b"checkpoint")
    (seed / "vqvae_hp_sweep.json").write_text(json.dumps({
        "mse_ranking": [{"name": "arm", "epochs_ran": 100,
                         "final_checkpoint_epoch": 99,
                         "checkpoint_final": str(checkpoint)}]
    }), encoding="utf-8")
    rows = discover_final_checkpoints(tmp_path, seeds=(0,), arms=("arm",), expected_epoch=100)
    assert rows[0]["final_checkpoint_epoch"] == 99


def test_cross_seed_summary_reports_population_std():
    summary = aggregate_across_seeds([
        {"arm": "a", "seed": 0, "cad_equal_surface_mse": 1.0,
         "cad_equal_curved_mse": 2.0, "cad_equal_planar_mse": 3.0},
        {"arm": "a", "seed": 1, "cad_equal_surface_mse": 3.0,
         "cad_equal_curved_mse": 4.0, "cad_equal_planar_mse": 5.0},
    ])
    assert summary["a"]["cad_equal_surface_mse"]["mean"] == 2.0
    assert summary["a"]["cad_equal_surface_mse"]["std"] == 1.0


def test_latest_attempt_rows_replaces_failed_retry_without_duplicate_attempt():
    rows = latest_attempt_rows([
        {"checkpoint_sha256": "x", "cad_id": "a", "status": "failed"},
        {"checkpoint_sha256": "x", "cad_id": "a", "status": "saved"},
    ])
    assert len(rows) == 1
    assert rows[0]["status"] == "saved"
