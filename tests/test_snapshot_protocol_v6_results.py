import json
from pathlib import Path

from tools.snapshot_protocol_v6_results import snapshot, summarize_history


def history_row(epoch, finite=True, curved=0.01, perplexity=100.0):
    return {
        "epoch": epoch,
        "train_batches": 4,
        "finite_train_batches": 4 if finite else 3,
        "val_batches": 2,
        "finite_val_batches": 2 if finite else 1,
        "val_parent_cluster_reconstruction_mse": {
            "surface_curved_proxy": {"mse": curved}
        },
        "val_code_usage": {"entropy_perplexity": perplexity},
    }


def test_summary_does_not_treat_fixed_epoch_loop_as_numerically_healthy():
    payload = {
        "config": {"target_epoch": 3},
        "history": [history_row(0), history_row(1, finite=False), history_row(2)],
        "best_epoch": 0,
        "best_val_recon": 0.1,
    }
    row = summarize_history(0, "arm", payload, {"final_checkpoint_epoch": 2}, "abc")
    assert row["status"] == "completed"
    assert row["health"] == "NUMERICALLY_UNSTABLE"
    assert row["first_nonfinite_epoch"] == 1
    assert row["fully_finite_val_epochs"] == 2


def test_summary_reports_curved_metrics_only_from_finite_validation():
    payload = {
        "config": {"target_epoch": 3},
        "history": [
            history_row(0, curved=0.02),
            history_row(1, finite=False, curved=0.00001),
            history_row(2, curved=0.01, perplexity=900.0),
        ],
    }
    row = summarize_history(1, "arm", payload, None, None)
    assert row["best_curved_parent_epoch"] == 2
    assert row["best_curved_parent_mse"] == 0.01
    assert row["final_curved_parent_mse"] == 0.01
    assert row["final_perplexity"] == 900.0


def test_snapshot_copies_only_lightweight_allowlisted_artifacts(tmp_path: Path):
    run = tmp_path / "run"
    report = tmp_path / "report"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "seed0.stdout.log").write_text("ok\n", encoding="utf-8")
    (run / "cohort_state.json").write_text(json.dumps({
        "status": "RUNNING", "active_seed": 0, "configuration": {}
    }), encoding="utf-8")
    seed = run / "seed0"
    seed.mkdir()
    history = {"config": {"target_epoch": 1}, "history": [history_row(0)]}
    (seed / "fsq_8192_4d_history.json").write_text(json.dumps(history), encoding="utf-8")
    (seed / "fsq_8192_4d_final.pt").write_bytes(b"weight")
    arrays = run / "surface_reconstruction" / "arrays"
    arrays.mkdir(parents=True)
    (arrays / "cad.npz").write_bytes(b"array")
    (run / "surface_reconstruction" / "summary.json").write_text("{}", encoding="utf-8")
    report.mkdir()
    (report / "seed2_fsq_8192_4d_history.json").write_text("{}", encoding="utf-8")
    seed2 = run / "seed2"
    seed2.mkdir()
    (seed2 / "fsq_8192_4d_history.json").write_text(json.dumps(history), encoding="utf-8")

    snapshot(run, report)

    assert (report / "seed0" / "fsq_8192_4d_history.json").is_file()
    assert not list(report.rglob("*.pt"))
    assert not list(report.rglob("*.npz"))
    assert (report / "surface_reconstruction" / "summary.json").is_file()
    assert not (report / "seed2_fsq_8192_4d_history.json").exists()
    manifest = json.loads((report / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert all(not item["path"].endswith((".pt", ".npz")) for item in manifest["artifacts"])
