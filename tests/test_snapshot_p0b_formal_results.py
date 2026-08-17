from __future__ import annotations

from tools.snapshot_p0b_formal_results import (
    arm_aggregates,
    epoch_summary,
    summarize_task,
    validate_report,
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


def test_capacity_vq_and_rvq_usage_are_preserved():
    rows = []
    for arm, value in (("vq_8192_64d_random", 0.02), ("rvq_2x4096_64d_random", 0.01)):
        for seed in (3, 4):
            task = {
                "task_id": f"{arm}:seed{seed}",
                "arm": arm,
                "seed": seed,
                "status": "COMPLETED",
                "signature": f"{arm}-{seed}",
                "signature_payload": {"precision": "bf16"},
            }
            epoch = _epoch(0, usage={"entropy_perplexity": 1000.0, "coverage": 0.5, "unique_bins": 2048})
            epoch["val_loss"] = value
            epoch["val_parent_cluster_reconstruction_mse"]["surface_curved_proxy"]["mse"] = value
            assert epoch_summary(task, epoch)["entropy_perplexity"] == 1000.0
            rows.append(summarize_task(task, {"config": {}}, [epoch]))

    aggregates = arm_aggregates(
        rows, ("vq_8192_64d_random", "rvq_2x4096_64d_random")
    )
    assert aggregates["pairwise"]["best_curved_parent_mse_ratio_left_over_right"] == 2.0


def test_validate_report_accepts_multiple_tensorboard_segments_after_resume(tmp_path):
    for index in range(4):
        task_dir = tmp_path / "tasks" / f"arm{index}" / "seed3"
        task_dir.mkdir(parents=True)
        (task_dir / f"arm{index}_history.json").write_text("{}\n", encoding="utf-8")

        tensorboard_dir = tmp_path / "tensorboard" / f"arm{index}" / "seed3"
        tensorboard_dir.mkdir(parents=True)
        (tensorboard_dir / f"events.out.tfevents.{index}.0").write_bytes(b"event")

    resumed_dir = tmp_path / "tensorboard" / "arm3" / "seed3"
    (resumed_dir / "events.out.tfevents.3.1").write_bytes(b"resumed-event")

    validation = validate_report(
        tmp_path,
        expected_histories=4,
        expected_tensorboard_events=5,
    )

    assert validation["histories"] == 4
    assert validation["tensorboard_events"] == 5
