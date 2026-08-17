import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.run_protocol_v5_scaling_ladder import (  # noqa: E402
    LadderConfig,
    RUNG_SPECS,
    _validate_runtime_inputs,
    build_master_protocol_command,
    build_oracle_environment,
    build_training_command,
    build_training_environment,
    resume_ladder_after_analysis,
    run_ladder,
)
from tools.summarize_protocol_v5_scaling import (  # noqa: E402
    render_scaling_pngs,
    summarize_scaling,
)


def make_config(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    for chunk in range(10):
        (archive_root / f"abc_{chunk:04d}_parsed.zip").write_bytes(b"zip")
    v4 = tmp_path / "v4.json"
    v4.write_text(json.dumps({"runs": []}))
    return LadderConfig(
        repo_root=REPO_ROOT,
        archive_root=archive_root,
        workspace_root=tmp_path / "workspace",
        v4_summary=v4,
        python_executable=Path(sys.executable),
    )


def test_ladder_uses_one_master_protocol_and_requested_matched_rungs(tmp_path):
    config = make_config(tmp_path)

    assert config.protocol_chunks == "0-9"
    assert config.protocol_record_cap == 15000
    assert config.seeds == (0, 1)
    assert config.max_epochs == 100
    assert config.min_epochs == 40
    assert config.patience == 15
    assert config.val_cap == 12000
    assert [(rung.name, rung.train_cap, rung.arms) for rung in RUNG_SPECS] == [
        (
            "60k",
            60000,
            ("fsq_8192_4d", "fsq_4096_6d", "vq_4096_64d_random"),
        ),
        ("300k", 300000, ("fsq_8192_4d", "fsq_4096_6d")),
    ]

    protocol_command = build_master_protocol_command(config)
    assert "--chunks" in protocol_command
    assert protocol_command[protocol_command.index("--chunks") + 1] == "0-9"
    assert protocol_command[protocol_command.index("--max-eligible-records") + 1] == "15000"
    assert "--load-failure-allowlist" not in protocol_command

    for rung in RUNG_SPECS:
        for seed in config.seeds:
            environment = build_training_environment(config, rung, seed)
            assert environment["NS_VQ_SWEEP_TRAIN_CAP"] == str(rung.train_cap)
            assert environment["NS_VQ_SWEEP_ARMS"] == ",".join(rung.arms)
            assert environment["NS_VQ_EXPERIMENT_SEED"] == str(seed)
            assert environment["NS_VQ_SWEEP_EPOCHS"] == "100"
            assert environment["NS_VQ_MIN_EPOCHS"] == "40"
            assert environment["NS_VQ_PATIENCE"] == "15"
            assert environment["NS_VQ_PLATEAU_METRIC"] == "curved_parent_mse"
            assert environment["NS_VQ_MIN_PARENT_COVERAGE"] == "0.9"
            command = build_training_command(config)
            assert command[-2:] == ["--stage", "vqsweep"]
            assert not any(token.lower() in {"sequence", "ar", "all"} for token in command)

    oracle = build_oracle_environment(config, seed=1)
    assert oracle["NS_VQ_SWEEP_ARMS"] == "continuous_bypass_64d"
    assert oracle["NS_VQ_SWEEP_TRAIN_CAP"] == "300000"
    assert oracle["NS_VQ_EXPERIMENT_SEED"] == "1"
    assert oracle["NS_VQ_PLATEAU_METRIC"] == "curved_parent_mse"


def test_ladder_allowlist_is_explicit_and_must_exist(tmp_path):
    config = make_config(tmp_path)
    allowlist = tmp_path / "approved.json"
    with pytest.raises(FileNotFoundError, match="allowlist"):
        LadderConfig(**{**config.__dict__, "load_failure_allowlist": allowlist})

    allowlist.write_text(json.dumps({"schema_version": 1, "entries": []}))
    approved = LadderConfig(**{**config.__dict__, "load_failure_allowlist": allowlist})
    command = build_master_protocol_command(approved)
    assert command[command.index("--load-failure-allowlist") + 1] == str(allowlist.resolve())


def test_runtime_validation_allows_only_launcher_control_files(tmp_path):
    config = make_config(tmp_path)
    config.workspace_root.mkdir()
    for name in ("launcher.stdout.log", "launcher.stderr.log", "launcher.pid"):
        (config.workspace_root / name).write_text("")

    _validate_runtime_inputs(config)


def test_runtime_validation_rejects_other_existing_workspace_files(tmp_path):
    config = make_config(tmp_path)
    config.workspace_root.mkdir()
    (config.workspace_root / "stale-result.json").write_text("{}")

    with pytest.raises(RuntimeError, match="new or empty"):
        _validate_runtime_inputs(config)


def run_row(seed, arm, codebook, curved, perplexity):
    return {
        "seed": seed,
        "arm": arm,
        "codebook_size": codebook,
        "curved_parent_mse": curved,
        "perplexity": perplexity,
    }


def test_scaling_summary_normalizes_usage_and_recommends_bypass_after_failed_projection():
    rows = [
        {"patches": 12000, "protocol_scope": "historical_reference", **run_row(0, "fsq_4096_6d", 4096, 1.0e-2, 640)},
        {"patches": 12000, "protocol_scope": "historical_reference", **run_row(1, "fsq_4096_6d", 4096, 1.1e-2, 680)},
        {"patches": 60000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 5.0e-3, 1024)},
        {"patches": 60000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 5.4e-3, 1100)},
        {"patches": 300000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 2.5e-3, 1300)},
        {"patches": 300000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 2.7e-3, 1400)},
        {"patches": 60000, "protocol_scope": "master", **run_row(0, "vq_4096_64d_random", 4096, 1.0e-3, 1800)},
        {"patches": 60000, "protocol_scope": "master", **run_row(1, "vq_4096_64d_random", 4096, 1.1e-3, 1900)},
    ]

    summary = summarize_scaling(rows, target_curved_mse=5e-5, projected_full_patches=3000000)

    vq = next(
        point for point in summary["points"]
        if point["patches"] == 60000 and point["arm"] == "vq_4096_64d_random"
    )
    assert vq["usage_fraction_mean"] == pytest.approx(((1800 / 4096) + (1900 / 4096)) / 2)
    assert summary["vq_control"]["vq_minus_fsq_curved_mse"] < 0
    assert summary["projection"]["source_points"] == [60000, 300000]
    assert summary["decision"]["status"] == "CONTINUE_CAPACITY_INVESTIGATION"
    assert summary["decision"]["continuous_bypass_oracle_recommended"] is True
    assert summary["decision"]["advance_to_ar"] is False


def test_scaling_summary_does_not_recommend_bypass_when_projected_target_is_plausible():
    rows = [
        {"patches": 60000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 1e-3, 1000)},
        {"patches": 60000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 1e-3, 1100)},
        {"patches": 300000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 1e-5, 1300)},
        {"patches": 300000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 1e-5, 1400)},
    ]

    summary = summarize_scaling(rows, target_curved_mse=5e-5, projected_full_patches=3000000)

    assert summary["decision"]["status"] == "TARGET_PLAUSIBLE"
    assert summary["decision"]["continuous_bypass_oracle_recommended"] is False
    assert summary["decision"]["advance_to_ar"] is False


def test_scaling_png_renderer_does_not_require_matplotlib(tmp_path):
    from PIL import Image

    summary = summarize_scaling(
        [
            {"patches": 60000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 5e-3, 1000)},
            {"patches": 60000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 4e-3, 1100)},
            {"patches": 300000, "protocol_scope": "master", **run_row(0, "fsq_4096_6d", 4096, 2e-3, 1300)},
            {"patches": 300000, "protocol_scope": "master", **run_row(1, "fsq_4096_6d", 4096, 2.2e-3, 1400)},
        ]
    )

    render_scaling_pngs(summary, tmp_path)

    for name in ("curved_mse_scaling.png", "usage_scaling.png"):
        path = tmp_path / name
        assert path.stat().st_size > 1000
        with Image.open(path) as image:
            image.verify()
            assert image.format == "PNG"


def test_full_ladder_runs_conditional_oracle_and_never_enters_ar(tmp_path):
    config = make_config(tmp_path)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((list(command), dict(kwargs.get("env") or {})))
        script = Path(command[1]).name
        if script == "preflight_cad_archive_inventory.py":
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"status": "VERIFIED"}))
        elif script == "build_cad_protocol.py":
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "protocol_summary.json").write_text(json.dumps({"status": "VERIFIED"}))
        elif script == "train.py":
            env = kwargs["env"]
            output = Path(env["NS_OUTBASE"]) / env["NS_OUT"]
            output.mkdir(parents=True, exist_ok=True)
            arms = env["NS_VQ_SWEEP_ARMS"].split(",")
            train_cap = int(env["NS_VQ_SWEEP_TRAIN_CAP"])
            payload = {
                "run_manifest": {
                    "experiment": {
                        "seed": int(env["NS_VQ_EXPERIMENT_SEED"]),
                        "train_cap": train_cap,
                        "arms": [{"name": arm} for arm in arms],
                    }
                },
                "mse_ranking": [
                    {
                        "name": arm,
                        "epochs_ran": 40,
                        "train_sampling": {
                            "requested_cap_met": True,
                            "final_parent_coverage": 0.95,
                        },
                    }
                    for arm in arms
                ],
            }
            (output / "vqvae_hp_sweep.json").write_text(json.dumps(payload))
        elif script == "summarize_protocol_v5_scaling.py":
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "scaling_summary.json").write_text(
                json.dumps(
                    {
                        "decision": {
                            "status": "CONTINUE_CAPACITY_INVESTIGATION",
                            "continuous_bypass_oracle_recommended": True,
                            "advance_to_ar": False,
                        }
                    }
                )
            )
        else:
            raise AssertionError(f"unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0)

    state = run_ladder(config, runner=fake_runner)

    assert state["status"] == "COMPLETED"
    assert state["continuous_bypass_oracle"] == "COMPLETED"
    assert state["advance_to_ar"] is False
    assert [step["phase"] for step in state["steps"]] == [
        "INVENTORY_PREFLIGHT",
        "CPU_IO_PROTOCOL_BUILD",
        "GPU_TRAINING_60K",
        "GPU_TRAINING_60K",
        "GPU_TRAINING_300K",
        "GPU_TRAINING_300K",
        "ANALYSIS",
        "GPU_ORACLE_CONTINUOUS_BYPASS",
        "GPU_ORACLE_CONTINUOUS_BYPASS",
    ]
    assert len(calls) == 9
    assert not any(
        token.lower() in {"sequence", "ar", "all"}
        for command, _ in calls
        if Path(command[1]).name == "train.py"
        for token in command
    )


def test_resume_after_analysis_failure_reuses_completed_training(tmp_path):
    config = make_config(tmp_path)
    workspace = config.workspace_root
    workspace.mkdir()
    (workspace / "protocol").mkdir()
    (workspace / "protocol" / "protocol_summary.json").write_text(json.dumps({"status": "VERIFIED"}))
    for rung in RUNG_SPECS:
        for seed in config.seeds:
            output = workspace / "rungs" / rung.name / f"seed{seed}"
            output.mkdir(parents=True)
            (output / "vqvae_hp_sweep.json").write_text(
                json.dumps(
                    {
                        "run_manifest": {
                            "experiment": {
                                "seed": seed,
                                "train_cap": rung.train_cap,
                                "arms": [{"name": arm} for arm in rung.arms],
                            }
                        },
                        "mse_ranking": [
                            {
                                "name": arm,
                                "epochs_ran": 40,
                                "train_sampling": {
                                    "requested_cap_met": True,
                                    "final_parent_coverage": 0.95,
                                },
                            }
                            for arm in rung.arms
                        ],
                    }
                )
            )
    state = {
        "status": "FAILED",
        "phase": "ANALYSIS",
        "steps": [],
        "gpu_expected": False,
    }
    (workspace / "ladder_state.json").write_text(json.dumps(state))
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((list(command), dict(kwargs.get("env") or {})))
        script = Path(command[1]).name
        if script == "summarize_protocol_v5_scaling.py":
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "scaling_summary.json").write_text(
                json.dumps(
                    {
                        "decision": {
                            "status": "CONTINUE_CAPACITY_INVESTIGATION",
                            "continuous_bypass_oracle_recommended": True,
                            "advance_to_ar": False,
                        }
                    }
                )
            )
        elif script == "train.py":
            env = kwargs["env"]
            output = Path(env["NS_OUTBASE"]) / env["NS_OUT"]
            output.mkdir(parents=True, exist_ok=True)
            (output / "vqvae_hp_sweep.json").write_text(
                json.dumps(
                    {
                        "run_manifest": {
                            "experiment": {
                                "seed": int(env["NS_VQ_EXPERIMENT_SEED"]),
                                "train_cap": 300000,
                                "arms": [{"name": "continuous_bypass_64d"}],
                            }
                        },
                        "mse_ranking": [
                            {
                                "name": "continuous_bypass_64d",
                                "epochs_ran": 40,
                                "train_sampling": {
                                    "requested_cap_met": True,
                                    "final_parent_coverage": 0.95,
                                },
                            }
                        ],
                    }
                )
            )
        else:
            raise AssertionError(f"unexpected recovery command: {command}")
        return subprocess.CompletedProcess(command, 0)

    result = resume_ladder_after_analysis(config, runner=fake_runner)

    assert result["status"] == "COMPLETED"
    assert result["continuous_bypass_oracle"] == "COMPLETED"
    assert result["advance_to_ar"] is False
    assert len(calls) == 3
    assert Path(calls[0][0][1]).name == "summarize_protocol_v5_scaling.py"
    assert all(Path(command[1]).name == "train.py" for command, _ in calls[1:])
