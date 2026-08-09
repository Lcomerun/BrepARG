import json
from pathlib import Path

from tools.run_protocol_v6_5seed_cohort import ARMS, SEEDS, training_environment, validate_sweep


def test_formal_environment_forces_four_arms_and_full_100_epochs(tmp_path):
    env = training_environment(
        repo_root=tmp_path / "repo", protocol_dir=tmp_path / "protocol",
        output_root=tmp_path / "out", seed=4,
    )
    assert tuple(env["NS_VQ_SWEEP_ARMS"].split(",")) == ARMS
    assert env["NS_VQ_SWEEP_EPOCHS"] == "100"
    assert env["NS_VQ_MIN_EPOCHS"] == "100"
    assert env["NS_VQ_PATIENCE"] == "100"
    assert env["NS_VQ_SAVE_FINAL"] == "1"
    assert SEEDS == (0, 1, 2, 3, 4)


def test_validate_sweep_requires_exact_epochs_caps_and_checkpoints(tmp_path):
    rows = []
    for arm in ARMS:
        best = tmp_path / f"{arm}_best.pt"; best.write_bytes(b"best")
        final = tmp_path / f"{arm}_final.pt"; final.write_bytes(b"final")
        rows.append({
            "name": arm, "epochs_ran": 100, "final_checkpoint_epoch": 99,
            "checkpoint_best": str(best), "checkpoint_final": str(final),
            "train_sampling": {"requested_cap_met": True, "final_parent_coverage": 0.95},
            "val_sampling": {"selected": 12000},
        })
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({
        "run_manifest": {"experiment": {"train_cap": 300000, "val_cap": 12000, "epochs": 100}},
        "mse_ranking": rows,
    }), encoding="utf-8")
    result = validate_sweep(sweep, train_cap=300000, val_cap=12000, epochs=100)
    assert result["valid"] is True
    assert len(result["checkpoints"]) == 8


def test_validate_sweep_rejects_early_stopped_arm(tmp_path):
    rows = [{"name": arm, "epochs_ran": 100, "final_checkpoint_epoch": 99,
             "train_sampling": {"requested_cap_met": True, "final_parent_coverage": 1.0},
             "val_sampling": {"selected": 12000}}
            for arm in ARMS]
    rows[0]["epochs_ran"] = 99
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({
        "run_manifest": {"experiment": {"train_cap": 300000, "val_cap": 12000, "epochs": 100}},
        "mse_ranking": rows,
    }), encoding="utf-8")
    result = validate_sweep(path, train_cap=300000, val_cap=12000, epochs=100)
    assert result["valid"] is False
    assert any("epochs_ran" in reason for reason in result["reasons"])


def test_vqsweep_wires_optional_final_checkpoint():
    source = Path("breparg_improvements/train.py").read_text(encoding="utf-8")
    assert "VQ_SAVE_FINAL = parse_env_bool" in source
    assert "save_final_path=arm_final_checkpoint" in source
    assert "'checkpoint_final': arm_final_checkpoint" in source
    assert "'final_checkpoint_epoch': arm_meta.get('end_epoch')" in source
