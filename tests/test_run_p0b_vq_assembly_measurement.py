import json
from pathlib import Path

import pytest

from tools.run_p0b_vq_assembly_measurement import (
    BYPASS_ARM,
    FORMAL_ARMS,
    FORMAL_SEEDS,
    HISTORICAL_STRICT_ONLY,
    MAX_CADS,
    SELECTION_SEED,
    VQ_ARM,
    EvidenceError,
    _calibration_is_complete,
    _validity_summary,
    canonical_signature,
    run_measurement,
    run_paired_measurement,
    select_fixed_seed3_checkpoints,
    select_healthy_vq_checkpoint,
    sha256_file,
    validate_p0b_evidence,
    verify_fixed_cohort,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _healthy_epoch(epoch: int, signature: str) -> dict:
    return {
        "epoch": epoch,
        "train_batches": 469,
        "finite_train_batches": 469,
        "skipped_train_batches": 0,
        "val_batches": 94,
        "finite_val_batches": 94,
        "nonfinite_loss_batches": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_state_batches": 0,
        "nonfinite_state_audits": 0,
        "nonfinite_val_batches": 0,
        "nonfinite_val_samples": 0,
        "gradients_finite": True,
        "training_state_finite": True,
        "finite_state_audit_cadence": "lifecycle_v1",
        "finite_state_audit": {"status": "finite"},
        "full_state_audits": 1,
        "per_batch_full_state_audits": 0,
        "grad_clip_active": True,
        "preclip_grad_norm": 1.25,
        "experiment_signature": signature,
    }


def _patch_inventory() -> dict:
    return {
        "train": {
            "schema": "vq-exact-hash-inventory-v1",
            "count": 60_000,
            "ordered_sha256": "1" * 64,
            "sorted_sha256": "2" * 64,
        },
        "val": {
            "schema": "vq-exact-hash-inventory-v1",
            "count": 12_000,
            "ordered_sha256": "3" * 64,
            "sorted_sha256": "4" * 64,
        },
    }


def _cohort_rows(protocol_sha256: str) -> tuple[list[dict], list[dict]]:
    originals = []
    selected = []
    for index in range(MAX_CADS):
        parent_id = f"{index:024x}"
        cad_id = f"{index:08d}_{parent_id}_step_000"
        source_path = str(Path("materialized") / "val" / f"{cad_id}.pkl")
        originals.append(
            {
                "arm": "original",
                "cad_id": cad_id,
                "parent_id": parent_id,
                "source_path": source_path,
                "selection_seed": SELECTION_SEED,
                "protocol_sha256": protocol_sha256,
                "checkpoint_sha256": None,
            }
        )
        selected.append({"cad_id": cad_id, "parent_id": parent_id, "path": source_path})
    return originals, selected


def make_evidence(tmp_path: Path) -> dict:
    output_root = tmp_path / "p0b"
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    split_path = protocol_dir / "split.pkl"
    split_path.write_bytes(b"frozen-split")
    split_sha = sha256_file(split_path)
    protocol_sha = "protocol-sha"
    protocol_summary_path = protocol_dir / "protocol_summary.json"
    _write_json(
        protocol_summary_path,
        {
            "status": "VERIFIED",
            "protocol_sha256": protocol_sha,
            "split_pickle_sha256": split_sha,
            "parent_overlap_counts": {
                "train__val": 0,
                "train__test": 0,
                "val__test": 0,
            },
        },
    )
    summary_file_sha = sha256_file(protocol_summary_path)
    configuration = {
        "repo_root": str(tmp_path / "repo"),
        "protocol_dir": str(protocol_dir),
        "breparg_root": str(tmp_path / "BrepARG"),
        "output_root": str(output_root),
        "python": str(tmp_path / "python.exe"),
        "arms": list(FORMAL_ARMS),
        "seeds": list(FORMAL_SEEDS),
        "train_cap": 60_000,
        "val_cap": 12_000,
        "batch_size": 128,
        "epochs": 100,
        "learning_rate": "3e-4",
        "precision": "fp32",
        "smoke": False,
    }
    checkpoint_payloads = {}
    tasks = []
    curved = {
        (VQ_ARM, 3): 0.003,
        (VQ_ARM, 4): 0.002,
        (BYPASS_ARM, 3): 0.001,
        (BYPASS_ARM, 4): 0.0009,
    }
    for arm in FORMAL_ARMS:
        for seed in FORMAL_SEEDS:
            task_root = output_root / "tasks" / arm / f"seed{seed}"
            task_root.mkdir(parents=True)
            signature_payload = {
                "schema": "p0b-stability-retest-v1",
                "arm": arm,
                "seed": seed,
                "train_cap": 60_000,
                "val_cap": 12_000,
                "batch_size": 128,
                "epochs": 100,
                "learning_rate": "3e-4",
                "precision": "fp32",
                "gradient_clip": "1.0",
                "strict_nonfinite_fuse": True,
                "protocol_summary_sha256": summary_file_sha,
                "split_pickle_sha256": split_sha,
            }
            signature = canonical_signature(signature_payload)
            task_id = f"{arm}:seed{seed}"
            history_path = task_root / f"{arm}_history.json"
            sweep_path = task_root / "vqvae_hp_sweep.json"
            best_path = task_root / f"{arm}_best.pt"
            final_path = task_root / f"{arm}_final.pt"
            rolling_path = task_root / f"{arm}_rolling.pt"
            manifest_path = task_root / "task_manifest.json"
            artifacts = [
                (best_path, f"best-{task_id}".encode()),
                (final_path, f"final-{task_id}".encode()),
                (rolling_path, f"rolling-{task_id}".encode()),
            ]
            for path, data in artifacts:
                path.write_bytes(data)
            metrics = {
                "code_usage": {"entropy_perplexity": 900.0, "coverage": 0.5},
                "nonfinite_val_samples": 0,
                "parent_cluster_mse": {
                    "mse": curved[(arm, seed)] / 2,
                    "nonfinite_samples": 0,
                    "nonfinite_parents": 0,
                },
                "parent_cluster_reconstruction_mse": {
                    "surface_curved_proxy": {
                        "mse": curved[(arm, seed)],
                        "nonfinite_samples": 0,
                        "nonfinite_parents": 0,
                    }
                },
            }
            run_manifest = {
                "git": {"commit": "clean-commit", "dirty": False},
                "launch": {
                    "relevant_env": {
                        "NS_VQ_EXPERIMENT_SIGNATURE": signature,
                        "NS_VQ_STRICT_NONFINITE": "1",
                    }
                },
                "experiment": {
                    "seed": seed,
                    "train_cap": 60_000,
                    "val_cap": 12_000,
                    "epochs": 100,
                    "batch_size": 128,
                    "inventory": _patch_inventory(),
                    "protocol": {
                        "protocol_sha256": protocol_sha,
                        "split_pickle_sha256": split_sha,
                    },
                    "arms": [{"name": arm}],
                },
            }
            checkpoint_epoch = 40 + seed
            validation_loss = curved[(arm, seed)] / 3
            quantizer = (
                {
                    "kind": "learned_vq",
                    "codebook_size": 4096,
                    "embedding_dim": 64,
                    "anchor": "random",
                }
                if arm == VQ_ARM
                else {"kind": "continuous_bypass", "embedding_dim": 64}
            )
            evaluated_checkpoint = best_path
            checkpoint_payloads[str(evaluated_checkpoint.resolve())] = {
                "model_state_dict": {"weight": object()},
                "checkpoint_epoch": checkpoint_epoch,
                "quantizer": quantizer,
                "validation_metrics": metrics,
                "validation_loss": validation_loss,
                "checkpoint_context": {
                    "protocol_sha256": protocol_sha,
                    "split_pickle_sha256": split_sha,
                    "inventory": _patch_inventory(),
                    "run_manifest": run_manifest,
                },
            }
            _write_json(
                history_path,
                {
                    "config": {
                        "experiment_signature": signature,
                        "precision": {"name": "fp32"},
                        "signature_configuration": {
                            "inventory": _patch_inventory()
                        },
                    },
                    "history": [_healthy_epoch(epoch, signature) for epoch in range(100)],
                },
            )
            checkpoint_sha = sha256_file(evaluated_checkpoint)
            sweep_row = {
                "name": arm,
                "epochs_ran": 100,
                "final_checkpoint_epoch": 99,
                "experiment_signature": signature,
                "inventory": _patch_inventory(),
                "checkpoint_best": str(best_path),
                "checkpoint_final": str(final_path),
                "checkpoint_epoch": checkpoint_epoch,
                "best_val_metrics": metrics,
                "checkpoint_val_recon": validation_loss,
                "promotion": {
                    "binding": {
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_epoch": checkpoint_epoch,
                        "protocol_sha256": protocol_sha,
                        "split_pickle_sha256": split_sha,
                        "git_commit": "clean-commit",
                    }
                },
            }
            _write_json(sweep_path, {"run_manifest": run_manifest, "mse_ranking": [sweep_row]})
            _write_json(
                manifest_path,
                {
                    "schema": "p0b-stability-retest-v1",
                    "task_id": task_id,
                    "signature": signature,
                    "signature_payload": signature_payload,
                },
            )
            tasks.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "task_id": task_id,
                    "task_root": str(task_root),
                    "signature": signature,
                    "signature_payload": signature_payload,
                    "history": str(history_path),
                    "sweep": str(sweep_path),
                    "best_checkpoint": str(best_path),
                    "final_checkpoint": str(final_path),
                    "rolling_checkpoint": str(rolling_path),
                    "manifest": str(manifest_path),
                    "status": "COMPLETED",
                    "validation": {
                        "valid": True,
                        "epochs_observed": 100,
                        "last_epoch": 99,
                        "inventory": _patch_inventory(),
                        "reasons": [],
                    },
                }
            )
    state = {
        "schema": "p0b-stability-retest-v1",
        "status": "COMPLETED",
        "mode": "FORMAL",
        "formal_result_eligible": True,
        "inventory_consistent": True,
        "configuration": configuration,
        "configuration_signature": canonical_signature(configuration),
        "tasks": tasks,
    }
    _write_json(output_root / "p0b_state.json", state)
    original_rows, selected_rows = _cohort_rows(protocol_sha)
    historical_manifest = tmp_path / "historical" / "calibration_manifest.jsonl"
    _write_jsonl(historical_manifest, original_rows)
    return {
        "output_root": output_root,
        "protocol_dir": protocol_dir,
        "protocol_sha": protocol_sha,
        "state": state,
        "checkpoint_payloads": checkpoint_payloads,
        "checkpoint_loader": lambda path: checkpoint_payloads[str(Path(path).resolve())],
        "historical_manifest": historical_manifest,
        "original_rows": original_rows,
        "selected_rows": selected_rows,
        "selector": lambda _protocol, max_cads, seed: (
            selected_rows if max_cads == MAX_CADS and seed == SELECTION_SEED else []
        ),
    }


def _make_runtime_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    (tools / "run_assembly_calibration_oracle.py").write_text("# oracle\n", encoding="utf-8")
    (tools / "audit_assembly_step_validity.py").write_text("# audit\n", encoding="utf-8")
    breparg = tmp_path / "BrepARG"
    breparg.mkdir()
    (breparg / "utils.py").write_text("# utils\n", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    return repo, breparg, python


def test_validates_all_four_tasks_and_selects_lowest_curved_vq_deterministically(tmp_path):
    fixture = make_evidence(tmp_path)
    evidence = validate_p0b_evidence(
        fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
    )
    assert len(evidence["tasks"]) == 4
    assert evidence["zero_nonfinite"] is True

    selected = select_healthy_vq_checkpoint(evidence)

    assert selected["task_id"] == f"{VQ_ARM}:seed4"
    assert selected["curved_parent_mse"] == pytest.approx(0.002)
    assert [row["seed"] for row in selected["candidate_ranking"]] == [4, 3]


def test_bypass_tasks_bind_best_with_promotion_binding(tmp_path):
    fixture = make_evidence(tmp_path)
    bypass_tasks = [
        task for task in fixture["state"]["tasks"] if task["arm"] == BYPASS_ARM
    ]
    assert len(bypass_tasks) == 2
    for task in bypass_tasks:
        assert Path(task["best_checkpoint"]).exists()
        sweep = json.loads(Path(task["sweep"]).read_text(encoding="utf-8"))
        assert "binding" in sweep["mse_ranking"][0]["promotion"]

    evidence = validate_p0b_evidence(
        fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
    )
    validated_bypass = [task for task in evidence["tasks"] if task["arm"] == BYPASS_ARM]
    validated_vq = [task for task in evidence["tasks"] if task["arm"] == VQ_ARM]

    assert len(validated_bypass) == 2
    assert all(task["checkpoint_role"] == "best" for task in validated_bypass)
    assert {
        Path(task["checkpoint_path"]).name for task in validated_bypass
    } == {f"{BYPASS_ARM}_best.pt"}
    assert all(task["checkpoint_role"] == "best" for task in validated_vq)
    assert select_healthy_vq_checkpoint(evidence)["task_id"] == f"{VQ_ARM}:seed4"
    selected = select_fixed_seed3_checkpoints(evidence)
    assert selected[VQ_ARM]["task_id"] == f"{VQ_ARM}:seed3"
    assert selected[BYPASS_ARM]["task_id"] == f"{BYPASS_ARM}:seed3"


@pytest.mark.parametrize("corruption", ["nonfinite", "checkpoint_sha", "dirty_git"])
def test_p0b_evidence_fails_closed_on_incomplete_or_dirty_binding(tmp_path, corruption):
    fixture = make_evidence(tmp_path)
    task = fixture["state"]["tasks"][0]
    if corruption == "nonfinite":
        history = json.loads(Path(task["history"]).read_text(encoding="utf-8"))
        history["history"][12]["nonfinite_gradient_batches"] = 1
        _write_json(Path(task["history"]), history)
    elif corruption == "checkpoint_sha":
        Path(task["best_checkpoint"]).write_bytes(b"mutated-after-binding")
    else:
        sweep = json.loads(Path(task["sweep"]).read_text(encoding="utf-8"))
        sweep["run_manifest"]["git"]["dirty"] = True
        _write_json(Path(task["sweep"]), sweep)
        payload = fixture["checkpoint_payloads"][str(Path(task["best_checkpoint"]).resolve())]
        payload["checkpoint_context"]["run_manifest"]["git"]["dirty"] = True

    with pytest.raises(EvidenceError):
        validate_p0b_evidence(
            fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
        )


def test_fixed_cohort_requires_historical_seed_and_exact_original_identities(tmp_path):
    fixture = make_evidence(tmp_path)
    cohort = verify_fixed_cohort(
        fixture["protocol_dir"],
        fixture["historical_manifest"],
        protocol_sha256=fixture["protocol_sha"],
        selector=fixture["selector"],
    )
    assert cohort["selection_seed"] == SELECTION_SEED
    assert len(cohort["identities"]) == 100

    changed = [dict(row) for row in fixture["selected_rows"]]
    changed[-1]["cad_id"] = "outside-cohort"
    with pytest.raises(EvidenceError, match="mismatch|does not equal"):
        verify_fixed_cohort(
            fixture["protocol_dir"],
            fixture["historical_manifest"],
            protocol_sha256=fixture["protocol_sha"],
            selector=lambda *_args, **_kwargs: changed,
        )


def _calibration_row(identity, fixture, selected_vq):
    return {
        "cad_id": identity["cad_id"],
        "parent_id": identity["parent_id"],
        "source_path": identity["path"],
        "arm": VQ_ARM,
        "selection_seed": SELECTION_SEED,
        "protocol_sha256": fixture["protocol_sha"],
        "checkpoint_sha256": selected_vq["checkpoint_sha256"],
        "status": "construct_brep_failed",
        "step_saved": False,
        "brep_valid": False,
    }


def test_calibration_partial_resume_and_full_manifest_without_state_are_recoverable(tmp_path):
    fixture = make_evidence(tmp_path)
    evidence = validate_p0b_evidence(
        fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
    )
    selected_vq = select_healthy_vq_checkpoint(evidence)
    cohort = verify_fixed_cohort(
        fixture["protocol_dir"],
        fixture["historical_manifest"],
        protocol_sha256=fixture["protocol_sha"],
        selector=fixture["selector"],
    )
    calibration_dir = tmp_path / "calibration"
    manifest = calibration_dir / "calibration_manifest.jsonl"
    rows = [
        _calibration_row(identity, fixture, selected_vq)
        for identity in fixture["selected_rows"]
    ]

    _write_jsonl(manifest, rows[:17])
    assert not _calibration_is_complete(
        calibration_dir,
        cohort=cohort,
        selected=selected_vq,
        protocol_sha256=fixture["protocol_sha"],
    )

    # The oracle writes each attempt before its aggregate state.  All 100
    # bound rows are therefore still a resumable state-recreation case.
    _write_jsonl(manifest, rows)
    assert not _calibration_is_complete(
        calibration_dir,
        cohort=cohort,
        selected=selected_vq,
        protocol_sha256=fixture["protocol_sha"],
    )


@pytest.mark.parametrize("corruption", ["duplicate", "outside_cohort", "checkpoint"])
def test_partial_calibration_manifest_rejects_dirty_identity_or_checkpoint_binding(
    tmp_path, corruption
):
    fixture = make_evidence(tmp_path)
    evidence = validate_p0b_evidence(
        fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
    )
    selected_vq = select_healthy_vq_checkpoint(evidence)
    cohort = verify_fixed_cohort(
        fixture["protocol_dir"],
        fixture["historical_manifest"],
        protocol_sha256=fixture["protocol_sha"],
        selector=fixture["selector"],
    )
    rows = [
        _calibration_row(identity, fixture, selected_vq)
        for identity in fixture["selected_rows"][:3]
    ]
    if corruption == "duplicate":
        rows[1] = dict(rows[0])
    elif corruption == "outside_cohort":
        rows[1].update(cad_id="outside-cohort", parent_id="outside-parent")
    else:
        rows[1]["checkpoint_sha256"] = "wrong-checkpoint"
    calibration_dir = tmp_path / "calibration"
    _write_jsonl(calibration_dir / "calibration_manifest.jsonl", rows)

    with pytest.raises(EvidenceError):
        _calibration_is_complete(
            calibration_dir,
            cohort=cohort,
            selected=selected_vq,
            protocol_sha256=fixture["protocol_sha"],
        )


def test_coordinator_launches_one_arm_preserves_denominator_and_reuses_complete_outputs(
    tmp_path,
):
    fixture = make_evidence(tmp_path)
    repo, breparg, python = _make_runtime_inputs(tmp_path)
    output_dir = tmp_path / "measurement"
    report_dir = tmp_path / "reports"
    commands = []

    selected_vq = next(
        task
        for task in validate_p0b_evidence(
            fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
        )["tasks"]
        if task["task_id"] == f"{VQ_ARM}:seed4"
    )

    def runner(command, *, cwd, stdout_path, stderr_path):
        commands.append(list(command))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if "run_assembly_calibration_oracle.py" in command[1]:
            assert command.count("--checkpoint") == 1
            checkpoint_arg = command[command.index("--checkpoint") + 1]
            assert checkpoint_arg == f"{VQ_ARM}={selected_vq['checkpoint_path']}"
            assert "--include-original-control" not in command
            assert command[command.index("--max-cads") + 1] == "100"
            assert command[command.index("--seed") + 1] == str(SELECTION_SEED)
            assert command[command.index("--joint-iterations") + 1] == "200"
            calibration_dir = Path(command[command.index("--output-dir") + 1])
            rows = []
            for index, identity in enumerate(fixture["selected_rows"]):
                step_saved = index < 90
                step_path = calibration_dir / "steps" / VQ_ARM / f"{identity['cad_id']}.step"
                row = {
                    "cad_id": identity["cad_id"],
                    "parent_id": identity["parent_id"],
                    "source_path": identity["path"],
                    "arm": VQ_ARM,
                    "selection_seed": SELECTION_SEED,
                    "protocol_sha256": fixture["protocol_sha"],
                    "checkpoint_sha256": selected_vq["checkpoint_sha256"],
                    "status": "saved" if step_saved else "construct_brep_failed",
                    "step_saved": step_saved,
                    "brep_valid": index < 75,
                    "global_patch_mse": index / 100_000,
                    "surface_mse": index / 100_000,
                    "curved_mse": index / 50_000,
                    "planar_mse": index / 200_000,
                    "edge_mse": index / 150_000,
                    "nonfinite_patches": 0,
                }
                if step_saved:
                    step_path.parent.mkdir(parents=True, exist_ok=True)
                    step_path.write_bytes(f"STEP-{index}".encode())
                    row.update(
                        step_path=str(step_path),
                        step_sha256=sha256_file(step_path),
                        step_bytes=step_path.stat().st_size,
                    )
                rows.append(row)
            _write_jsonl(calibration_dir / "calibration_manifest.jsonl", rows)
            _write_json(
                calibration_dir / "calibration_state.json",
                {
                    "status": "COMPLETED",
                    "selected_cads": 100,
                    "expected_rows": 100,
                    "manifest_rows": 100,
                    "arms": [VQ_ARM],
                    "protocol_sha256": fixture["protocol_sha"],
                    "checkpoints": {VQ_ARM: {"sha256": selected_vq["checkpoint_sha256"]}},
                },
            )
        else:
            assert "audit_assembly_step_validity.py" in command[1]
            calibration_manifest = Path(command[command.index("--manifest") + 1])
            calibration_rows = [
                json.loads(line)
                for line in calibration_manifest.read_text(encoding="utf-8").splitlines()
                if line
            ]
            audit_dir = Path(command[command.index("--output-dir") + 1])
            rows = []
            for index, source in enumerate(calibration_rows):
                saved = source["step_saved"]
                rows.append(
                    {
                        "cad_id": source["cad_id"],
                        "arm": VQ_ARM,
                        "source_status": source["status"],
                        "step_path": source.get("step_path"),
                        "native_brep_valid": index < 80 if saved else None,
                        "strict_brep_valid": index < 75,
                        "status": "audited" if saved else "no_step",
                    }
                )
            _write_jsonl(audit_dir / "step_validity_audit.jsonl", rows)
            _write_json(audit_dir / "step_validity_summary.json", _validity_summary(rows))
        return 0

    first = run_measurement(
        repo_root=repo,
        p0b_output_root=fixture["output_root"],
        historical_calibration_manifest=fixture["historical_manifest"],
        breparg_root=breparg,
        output_dir=output_dir,
        report_dir=report_dir,
        python=python,
        checkpoint_loader=fixture["checkpoint_loader"],
        selector=fixture["selector"],
        command_runner=runner,
    )

    assert first["status"] == "COMPLETED"
    assert first["summary"]["attempts"] == 100
    assert first["summary"]["step_saved"] == 90
    assert first["summary"]["native_brep_valid"] == 80
    assert first["summary"]["strict_brep_valid"] == 75
    assert first["summary"]["both_valid"] == 75
    assert first["historical_strict_only"] == HISTORICAL_STRICT_ONLY
    assert len(commands) == 2

    markdown = (report_dir / "p0b_vq_assembly_measurement.md").read_text(encoding="utf-8")
    assert "historical **strict-only runner values**" in markdown
    csv_rows = (report_dir / "p0b_vq_assembly_measurement.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 101
    assert not list(report_dir.rglob("*.step"))
    assert not list(report_dir.rglob("*.pt"))
    assert not list(report_dir.rglob("*.npz"))

    second_calls = []
    second = run_measurement(
        repo_root=repo,
        p0b_output_root=fixture["output_root"],
        historical_calibration_manifest=fixture["historical_manifest"],
        breparg_root=breparg,
        output_dir=output_dir,
        report_dir=report_dir,
        python=python,
        checkpoint_loader=fixture["checkpoint_loader"],
        selector=fixture["selector"],
        command_runner=lambda *args, **kwargs: second_calls.append((args, kwargs)),
    )
    assert second["status"] == "COMPLETED"
    assert second_calls == []


def test_dry_run_validates_without_creating_measurement_output(tmp_path):
    fixture = make_evidence(tmp_path)
    repo, breparg, python = _make_runtime_inputs(tmp_path)
    output_dir = tmp_path / "not-created"
    result = run_measurement(
        repo_root=repo,
        p0b_output_root=fixture["output_root"],
        historical_calibration_manifest=fixture["historical_manifest"],
        breparg_root=breparg,
        output_dir=output_dir,
        report_dir=tmp_path / "reports",
        python=python,
        checkpoint_loader=fixture["checkpoint_loader"],
        selector=fixture["selector"],
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert not output_dir.exists()


def test_paired_coordinator_uses_seed3_best_and_reports_delta_gates(tmp_path):
    fixture = make_evidence(tmp_path)
    repo, breparg, python = _make_runtime_inputs(tmp_path)
    evidence = validate_p0b_evidence(
        fixture["output_root"], checkpoint_loader=fixture["checkpoint_loader"]
    )
    selected = select_fixed_seed3_checkpoints(evidence)
    commands = []

    def runner(command, *, cwd, stdout_path, stderr_path):
        commands.append(list(command))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = Path(command[command.index("--output-dir") + 1])
        if "run_assembly_calibration_oracle.py" in command[1]:
            arm, checkpoint = command[command.index("--checkpoint") + 1].split("=", 1)
            assert checkpoint == selected[arm]["checkpoint_path"]
            assert selected[arm]["seed"] == 3
            rows = []
            for index, identity in enumerate(fixture["selected_rows"]):
                step_saved = index < (88 if arm == BYPASS_ARM else 83)
                row = {
                    "cad_id": identity["cad_id"], "parent_id": identity["parent_id"],
                    "source_path": identity["path"], "arm": arm,
                    "selection_seed": SELECTION_SEED, "protocol_sha256": fixture["protocol_sha"],
                    "checkpoint_sha256": selected[arm]["checkpoint_sha256"],
                    "status": "saved" if step_saved else "construct_brep_failed",
                    "step_saved": step_saved, "brep_valid": index < 70, "nonfinite_patches": 0,
                }
                if step_saved:
                    step = output / "steps" / arm / f"{identity['cad_id']}.step"
                    step.parent.mkdir(parents=True, exist_ok=True); step.write_bytes(b"STEP")
                    row.update(step_path=str(step), step_sha256=sha256_file(step), step_bytes=4)
                rows.append(row)
            _write_jsonl(output / "calibration_manifest.jsonl", rows)
            _write_json(output / "calibration_state.json", {
                "status": "COMPLETED", "selected_cads": 100, "expected_rows": 100,
                "manifest_rows": 100, "arms": [arm], "protocol_sha256": fixture["protocol_sha"],
                "checkpoints": {arm: {"sha256": selected[arm]["checkpoint_sha256"]}},
            })
        else:
            sources = [json.loads(line) for line in Path(command[command.index("--manifest") + 1]).read_text().splitlines()]
            arm = sources[0]["arm"]
            strict_limit = 72 if arm == BYPASS_ARM else 64
            audit = [{
                "cad_id": source["cad_id"], "arm": arm, "source_status": source["status"],
                "native_brep_valid": (index < strict_limit + 3) if source["step_saved"] else None,
                "strict_brep_valid": index < strict_limit,
                "status": "audited" if source["step_saved"] else "no_step",
            } for index, source in enumerate(sources)]
            _write_jsonl(output / "step_validity_audit.jsonl", audit)
            _write_json(output / "step_validity_summary.json", _validity_summary(audit))
        return 0

    result = run_paired_measurement(
        repo_root=repo, p0b_output_root=fixture["output_root"],
        historical_calibration_manifest=fixture["historical_manifest"], breparg_root=breparg,
        output_dir=tmp_path / "paired", report_dir=tmp_path / "paired_reports", python=python,
        checkpoint_loader=fixture["checkpoint_loader"], selector=fixture["selector"], command_runner=runner,
    )

    assert len(commands) == 4
    assert result["summary"][BYPASS_ARM]["strict_brep_valid"] == 72
    assert result["summary"][VQ_ARM]["strict_brep_valid"] == 64
    assert result["gates_percentage_points"]["delta_q_bypass60k_minus_vq60k"] == 8
    assert result["gates_percentage_points"]["delta_r_gt_minus_bypass60k"] == 12
    assert result["gates_percentage_points"]["decision"] == "CAPACITY_AB_FIRST"
    csv_rows = (tmp_path / "paired_reports" / "p0b_paired_assembly_measurement.csv").read_text().splitlines()
    assert len(csv_rows) == 201
