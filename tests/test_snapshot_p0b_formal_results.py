from __future__ import annotations

import hashlib
import json

import pytest

from tools.snapshot_p0b_formal_results import (
    MACHINE_LOCAL_PATH,
    arm_aggregates,
    artifact_manifest,
    copy_lightweight,
    epoch_summary,
    redact_machine_local_paths,
    summarize_task,
    validate_report,
    write_csv,
    write_json,
    write_text_lf,
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
        write_json(task_dir / f"arm{index}_history.json", {})

        tensorboard_dir = tmp_path / "tensorboard" / f"arm{index}" / "seed3"
        tensorboard_dir.mkdir(parents=True)
        (tensorboard_dir / f"events.out.tfevents.{index}.0").write_bytes(
            b"\xff\x00event"
        )

    resumed_dir = tmp_path / "tensorboard" / "arm3" / "seed3"
    (resumed_dir / "events.out.tfevents.3.1").write_bytes(b"resumed-event")

    validation = validate_report(
        tmp_path,
        expected_histories=4,
        expected_tensorboard_events=5,
    )

    assert validation["histories"] == 4
    assert validation["tensorboard_events"] == 5
    assert validation["machine_absolute_paths"] is False


def test_redact_machine_local_paths_recurses_without_changing_relative_paths():
    payload = {
        "windows": r"D:\runs\seed3\best.pt",
        "posix": "/home/user/runs/history.json",
        "workspace": "/workspace/runs/report.json",
        "relative": "tasks/seed3/history.json",
        "nested": [r"C:/Users/YU/env/python.exe"],
        "command": r"python D:\runs\seed3\train.py --epochs 100",
    }

    redacted = redact_machine_local_paths(payload)

    assert redacted["windows"] == f"{MACHINE_LOCAL_PATH}/best.pt"
    assert redacted["posix"] == f"{MACHINE_LOCAL_PATH}/history.json"
    assert redacted["workspace"] == f"{MACHINE_LOCAL_PATH}/report.json"
    assert redacted["relative"] == "tasks/seed3/history.json"
    assert redacted["nested"] == [f"{MACHINE_LOCAL_PATH}/python.exe"]
    assert redacted["command"] == (
        f"python {MACHINE_LOCAL_PATH}/train.py --epochs 100"
    )


def test_copy_lightweight_redacts_json_and_binds_source_and_archive_hashes(tmp_path):
    run_root = tmp_path / "run"
    source = run_root / "tasks" / "seed3" / "history.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "checkpoint": r"D:\runs\seed3\best.pt",
                "relative": "tasks/seed3/best.pt",
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "report" / "tasks" / "arm" / "seed3" / source.name
    manifest = []

    copy_lightweight(source, target, run_root, tmp_path / "report", manifest)

    archived = json.loads(target.read_text(encoding="utf-8"))
    assert archived["checkpoint"] == f"{MACHINE_LOCAL_PATH}/best.pt"
    assert archived["relative"] == "tasks/seed3/best.pt"
    assert b"\r" not in target.read_bytes()
    assert manifest == [
        {
            "source_relative_path": "tasks/seed3/history.json",
            "archive_relative_path": "tasks/arm/seed3/history.json",
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_bytes": source.stat().st_size,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "archive_bytes": target.stat().st_size,
            "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "transformation": "json_machine_paths_redacted",
        }
    ]


def test_copy_lightweight_preserves_non_json_bytes(tmp_path):
    run_root = tmp_path / "run"
    source = run_root / "tensorboard" / "events.out.tfevents.1"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"event\r\nbytes\x00")
    target = tmp_path / "report" / "tensorboard" / source.name
    manifest = []

    copy_lightweight(source, target, run_root, tmp_path / "report", manifest)

    assert target.read_bytes() == source.read_bytes()
    assert manifest[0]["transformation"] == "identity"
    assert manifest[0]["source_sha256"] == manifest[0]["archive_sha256"]


def test_copy_lightweight_transcodes_windows_log_and_keeps_both_hashes(tmp_path):
    run_root = tmp_path / "run"
    source = run_root / "logs" / "stdout.log"
    source.parent.mkdir(parents=True)
    source.write_bytes("超参\r\nsecond\r\n".encode("gb18030"))
    target = tmp_path / "report" / "logs" / source.name
    manifest = []

    copy_lightweight(source, target, run_root, tmp_path / "report", manifest)

    assert target.read_bytes() == "超参\nsecond\n".encode("utf-8")
    assert manifest[0]["transformation"] == "text_utf8_lf"
    assert manifest[0]["source_sha256"] != manifest[0]["archive_sha256"]


def test_generated_text_is_canonical_lf_and_manifest_matches_bytes(tmp_path):
    report = tmp_path / "report"
    write_json(report / "summary.json", {"status": "ok"})
    write_csv(report / "metrics.csv", [{"epoch": 0}], ["epoch"])

    manifest = artifact_manifest(report)

    for artifact in manifest:
        data = (report / artifact["path"]).read_bytes()
        assert b"\r" not in data
        assert artifact["bytes"] == len(data)
        assert artifact["sha256"] == hashlib.sha256(data).hexdigest()


def test_copy_lightweight_redacts_absolute_path_in_log(tmp_path):
    run_root = tmp_path / "run"
    source = run_root / "logs" / "stdout.log"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint D:\\local\\best.pt\r\n")
    target = tmp_path / "report" / "logs" / source.name
    manifest = []

    copy_lightweight(source, target, run_root, tmp_path / "report", manifest)

    assert target.read_text(encoding="utf-8") == (
        f"checkpoint {MACHINE_LOCAL_PATH}/best.pt\n"
    )
    assert manifest[0]["transformation"] == "text_utf8_lf_machine_paths_redacted"


def test_validate_report_rejects_absolute_path_in_json(tmp_path):
    for index in range(4):
        task_dir = tmp_path / "tasks" / f"arm{index}" / "seed3"
        task_dir.mkdir(parents=True)
        write_json(task_dir / f"arm{index}_history.json", {"path": "relative"})
        tensorboard_dir = tmp_path / "tensorboard" / f"arm{index}" / "seed3"
        tensorboard_dir.mkdir(parents=True)
        (tensorboard_dir / f"events.out.tfevents.{index}.0").write_bytes(b"event")
    (tmp_path / "leak.json").write_text(
        json.dumps({"path": r"D:\local\secret.json"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="absolute path"):
        validate_report(tmp_path)


def test_validate_report_rejects_absolute_path_embedded_in_string(tmp_path):
    for index in range(4):
        task_dir = tmp_path / "tasks" / f"arm{index}" / "seed3"
        task_dir.mkdir(parents=True)
        write_json(task_dir / f"arm{index}_history.json", {"path": "relative"})
        tensorboard_dir = tmp_path / "tensorboard" / f"arm{index}" / "seed3"
        tensorboard_dir.mkdir(parents=True)
        (tensorboard_dir / f"events.out.tfevents.{index}.0").write_bytes(b"event")
    (tmp_path / "leak.json").write_text(
        json.dumps({"command": r"python D:\local\script.py"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="absolute path"):
        validate_report(tmp_path)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("leak.jsonl", '{"path": "/home/user/result.json"}\n'),
        ("leak.csv", "path\nD:\\local\\result.csv\n"),
        ("leak.md", "artifact: `/workspace/local/result.md`\n"),
        ("leak.txt", "artifact=/tmp/local/result.txt\n"),
        ("leak.log", "checkpoint C:\\local\\best.pt\n"),
    ],
)
def test_validate_report_rejects_absolute_path_in_every_text_type(
    tmp_path, name, content
):
    for index in range(4):
        task_dir = tmp_path / "tasks" / f"arm{index}" / "seed3"
        task_dir.mkdir(parents=True)
        write_json(task_dir / f"arm{index}_history.json", {"path": "relative"})
        tensorboard_dir = tmp_path / "tensorboard" / f"arm{index}" / "seed3"
        tensorboard_dir.mkdir(parents=True)
        (tensorboard_dir / f"events.out.tfevents.{index}.0").write_bytes(b"\xff\x00")
    write_text_lf(tmp_path / name, content)

    with pytest.raises(RuntimeError, match="absolute path"):
        validate_report(tmp_path)


def test_artifact_manifest_rejects_crlf_for_git_normalized_text(tmp_path):
    (tmp_path / "summary.json").write_bytes(b'{"status": "ok"}\r\n')

    with pytest.raises(RuntimeError, match="canonical LF"):
        artifact_manifest(tmp_path)
