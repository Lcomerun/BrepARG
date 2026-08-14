import json
from pathlib import Path

import torch

from tools.snapshot_p0b_runtime_evidence import (
    compact_epoch,
    parse_lock_metadata,
    sha256_file,
)


def test_parse_lock_metadata_accepts_reserved_leading_lock_byte(tmp_path):
    path = tmp_path / ".p0b_writer.lock"
    path.write_bytes(b"\x00" + json.dumps({"released_at": "now"}).encode("utf-8"))

    assert parse_lock_metadata(path) == {"released_at": "now"}


def test_compact_epoch_keeps_finiteness_and_resume_evidence():
    row = {
        "epoch": 1,
        "train_loss": 0.2,
        "val_loss": 0.1,
        "preclip_grad_norm": 2.0,
        "train_batches": 4,
        "finite_train_batches": 4,
        "val_batches": 2,
        "finite_val_batches": 2,
        "nonfinite_loss_batches": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_state_audits": 0,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "training_state_finite": True,
        "finite_state_audit": {"status": "finite", "tensors": 3, "elements": 9},
        "resumed": True,
        "resume_from_epoch": 0,
        "val_code_usage": {"entropy_perplexity": 12.0, "coverage": 0.5},
        "val_parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": 0.03}
        },
    }

    compact = compact_epoch(row)

    assert compact["epoch"] == 1
    assert compact["curved_parent_mse"] == 0.03
    assert compact["entropy_perplexity"] == 12.0
    assert compact["finite_state_audit_status"] == "finite"
    assert compact["resumed"] is True
    assert compact["resume_from_epoch"] == 0


def test_sha256_file_streams_checkpoint_bytes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, path)

    first = sha256_file(path)
    second = sha256_file(path)

    assert len(first) == 64
    assert first == second
