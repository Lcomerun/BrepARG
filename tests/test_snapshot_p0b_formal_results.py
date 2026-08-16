from __future__ import annotations

from tools.snapshot_p0b_formal_results import (
    epoch_summary,
    summarize_task,
)


def _epoch(epoch: int, *, usage: dict | None = None) -> dict:
    return {
        "epoch": epoch,
        "train_loss": 0.2 - epoch * 0.01,
        "val_loss": 0.1 - epoch * 0.01,
        "train_batches": 4,
        "finite_train_batches": 4,
        "skipped_train_batches": 0,
        "nonfinite_loss_batches": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_state_batches": 0,
        "nonfinite_state_audits": 0,
        "val_batches": 2,
        "finite_val_batches": 2,
        "finite_val_samples": 8,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "training_state_finite": True,
        "finite_state_audit": {"status": "finite"},
        "val_code_usage": usage or {},
        "val_parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": 0.03 - epoch * 0.001},
            "surface_planar_like": {"mse": 0.01},
            "edge": {"mse": 0.02},
        },
        "lr": 3e-4,
        "lr_after_scheduler": 3e-4,
        "preclip_grad_norm": 0.5,
        "grad_clip_was_effective": False,
    }


def test_bypass_placeholder_usage_is_not_reported_as_codebook_health():
    task = {
        "task_id": "continuous_bypass_64d:seed3",
        "arm": "continuous_bypass_64d",
        "seed": 3,
        "status": "COMPLETED",
        "signature": "signature",
        "signature_payload": {"precision": "bf16"},
    }
    rows = [_epoch(0, usage={"entropy_perplexity": 1.0, "coverage": 1.0, "unique_bins": 1})]

    epoch = epoch_summary(task, rows[0])
    summary = summarize_task(task, {"config": {}}, rows)

    assert epoch["entropy_perplexity"] is None
    assert epoch["coverage"] is None
    assert epoch["unique_bins"] is None
    assert summary["final_perplexity"] is None
    assert summary["final_coverage"] is None
    assert summary["final_unique_bins"] is None


def test_vq_usage_is_preserved():
    task = {
        "task_id": "vq_4096_64d_random:seed3",
        "arm": "vq_4096_64d_random",
        "seed": 3,
        "status": "COMPLETED",
        "signature": "signature",
        "signature_payload": {"precision": "bf16"},
    }
    rows = [_epoch(0, usage={"entropy_perplexity": 1200.0, "coverage": 0.9, "unique_bins": 3686})]

    epoch = epoch_summary(task, rows[0])
    summary = summarize_task(task, {"config": {}}, rows)

    assert epoch["entropy_perplexity"] == 1200.0
    assert summary["final_perplexity"] == 1200.0
    assert summary["final_coverage"] == 0.9
    assert summary["final_unique_bins"] == 3686
