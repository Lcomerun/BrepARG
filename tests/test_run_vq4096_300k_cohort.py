import json

from tools.run_vq4096_300k_cohort import sweep_complete, training_environment


def test_training_environment_is_single_arm_300k_random_vq(tmp_path):
    env = training_environment(
        repo_root=tmp_path / "repo", protocol_dir=tmp_path / "protocol",
        output_root=tmp_path / "out", seed=1,
        train_cap=300000, val_cap=12000, epochs=100, min_epochs=40,
        patience=15, batch_size=128, learning_rate="3e-4",
    )
    assert env["NS_VQ_SWEEP_ARMS"] == "vq_4096_64d_random"
    assert env["NS_VQ_SWEEP_TRAIN_CAP"] == "300000"
    assert env["NS_VQ_EXPERIMENT_SEED"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert env["GIT_CONFIG_VALUE_0"].endswith("/repo")


def test_sweep_complete_checks_cap_arm_and_parent_coverage(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({
        "run_manifest": {"experiment": {"train_cap": 300000}},
        "mse_ranking": [{"name": "vq_4096_64d_random", "epochs_ran": 50,
                         "train_sampling": {"requested_cap_met": True,
                                            "final_parent_coverage": 0.95}}],
    }), encoding="utf-8")
    assert sweep_complete(path, train_cap=300000, min_epochs=40, max_epochs=100)
