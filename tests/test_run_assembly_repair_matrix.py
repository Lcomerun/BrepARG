import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.assembly_repair import RepairProfile, parse_profiles
from tools.run_assembly_repair_matrix import (
    RUN_SCHEMA,
    WORKER_MARKER,
    bind_run_manifest,
    main,
    parse_worker_result,
    profile_kwargs,
    requires_isolated_worker,
    run_one_isolated,
    sha256_file,
    summarize_matrix,
    summarize_profile,
)


def _rows(profile, values):
    return [
        {
            "profile": profile, "cad_id": cad_id, "strict_brep_valid": strict,
            "native_brep_valid": strict, "both_valid": strict,
            "step_saved": True, "status": "both_valid" if strict else "step_invalid",
        }
        for cad_id, strict in values.items()
    ]


def test_profile_kwargs_keep_switches_independent():
    assert profile_kwargs(RepairProfile("baseline")) == {
        "directed_trim": False, "curve_fit_fallback": False,
        "curve_fit_rescue": False,
        "wire_continuity": False, "single_solid": False,
        "solid_topology_repair": False,
        "pcurve_self_intersection": False,
        "local_intersection_topology": False,
        "local_pcurve_continuity": False,
    }
    assert profile_kwargs(RepairProfile("directed_trim", ("directed_trim",)))[
        "directed_trim"
    ] is True
    assert profile_kwargs(
        RepairProfile(
            "local_intersection_topology", ("local_intersection_topology",)
        )
    )["local_intersection_topology"] is True
    assert profile_kwargs(
        RepairProfile("curve_fit_rescue", ("curve_fit_rescue",))
    )["curve_fit_rescue"] is True
    assert profile_kwargs(
        RepairProfile(
            "local_pcurve_continuity", ("local_pcurve_continuity",)
        )
    )["local_pcurve_continuity"] is True


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    [
        ("baseline", False),
        ("directed_trim", False),
        ("local_intersection_topology", True),
        ("local_pcurve_continuity", True),
        ("single_solid", True),
        ("directed_trim_local_intersection_topology", True),
        ("directed_trim_local_pcurve_continuity", True),
    ],
)
def test_local_face_repairs_require_isolated_worker(profile_name, expected):
    profile = parse_profiles([profile_name])[0]
    assert requires_isolated_worker(profile) is expected


def test_summary_records_restoration_and_regression():
    historical = {"kept": True, "lost": True, "restored": False, "still_bad": False}
    summary = summarize_profile(
        _rows("candidate", {"kept": True, "lost": False, "restored": True, "still_bad": False}),
        historical,
    )
    assert summary["restored_cad_ids"] == ["restored"]
    assert summary["regressed_cad_ids"] == ["lost"]
    assert summary["preserves_original_84"] is False


def test_gate_requires_95_and_zero_regression():
    historical = {f"cad{i:03d}": i < 84 for i in range(100)}
    candidate = {cad_id: old or int(cad_id[-3:]) < 95 for cad_id, old in historical.items()}
    summary = summarize_matrix(
        _rows("combined", candidate), [RepairProfile("combined")], historical
    )
    assert summary["gate_passed"] is True
    assert summary["profiles"][0]["strict_valid"] == 95
    assert len(summary["profiles"][0]["restored_cad_ids"]) == 11

    candidate["cad000"] = False
    failed = summarize_profile(_rows("combined", candidate), historical)
    assert failed["strict_valid"] == 94
    assert failed["meets_95_gate"] is False


def test_profile_summary_rejects_incomplete_cohort():
    with pytest.raises(ValueError, match="full frozen cohort"):
        summarize_profile(_rows("x", {"a": True}), {"a": True, "b": False})


def test_parse_worker_result_uses_only_final_valid_dict():
    valid = {"cad_id": "cad001", "status": "both_valid"}
    stdout = "OCC diagnostic\n" + WORKER_MARKER + json.dumps(valid) + "\n"
    assert parse_worker_result(stdout) == valid
    assert parse_worker_result(WORKER_MARKER + "[]") is None
    assert parse_worker_result(WORKER_MARKER + "{broken") is None
    assert parse_worker_result("OCC diagnostic only") is None


def test_isolated_worker_records_native_exit_in_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=3221225477,
            stdout="native diagnostic without sentinel\n",
            stderr="access violation\n",
        ),
    )
    profile = parse_profiles(["local_intersection_topology"])[0]
    row = run_one_isolated(
        {"cad_id": "cad-exit", "parent_id": "parent", "source_path": "input.pkl", "brep_valid": False},
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
    )

    assert row["status"] == "worker_process_exit"
    assert row["worker_returncode"] == 3221225477
    assert row["strict_brep_valid"] is False
    assert row["both_valid"] is False
    assert (tmp_path / "worker_logs" / profile.name / "cad-exit.stdout.log").is_file()


def test_isolated_worker_records_timeout_and_preserves_byte_logs(tmp_path, monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=3, output=b"partial stdout\n", stderr=b"partial stderr\n"
        )

    monkeypatch.setattr(subprocess, "run", time_out)
    profile = parse_profiles(["local_intersection_topology"])[0]
    row = run_one_isolated(
        {"cad_id": "cad-timeout", "parent_id": "parent", "source_path": "input.pkl", "brep_valid": False},
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
    )

    assert row["status"] == "worker_timeout"
    assert row["worker_returncode"] is None
    assert row["strict_brep_valid"] is False
    assert "partial stdout" in (tmp_path / "worker_logs" / profile.name / "cad-timeout.stdout.log").read_text()
    assert "partial stderr" in (tmp_path / "worker_logs" / profile.name / "cad-timeout.stderr.log").read_text()


def test_isolated_worker_accepts_only_zero_exit_with_sentinel(tmp_path, monkeypatch):
    source = {
        "cad_id": "cad-ok",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    profile = parse_profiles(["local_intersection_topology"])[0]
    payload = {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": "cad-ok",
        "parent_id": "parent",
        "profile": "local_intersection_topology",
        "switches": ["local_intersection_topology"],
        "historical_strict_valid": False,
        "source_path": "input.pkl",
        "status": "both_valid",
        "step_saved": True,
        "strict_brep_valid": True,
        "native_brep_valid": True,
        "both_valid": True,
    }
    def successful_worker(command, **kwargs):
        attempt_root = command[command.index("--output-dir") + 1]
        assert len(attempt_root) < 180
        step = (
            Path(attempt_root)
            / "steps"
            / profile.name
            / "cad-ok.step"
        )
        step.parent.mkdir(parents=True, exist_ok=True)
        step.write_bytes(b"valid step payload")
        payload["step_sha256"] = sha256_file(step)
        payload["step_bytes"] = step.stat().st_size
        return SimpleNamespace(
            returncode=0,
            stdout="OCC says hello\n" + WORKER_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", successful_worker)
    row = run_one_isolated(
        source,
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
    )

    assert row["status"] == "both_valid"
    assert row["worker_returncode"] == 0
    assert row["strict_brep_valid"] is True
    assert (tmp_path / "steps" / profile.name / "cad-ok.step").is_file()


@pytest.mark.parametrize(
    "changed",
    [
        {"schema": "old-schema"},
        {"cad_id": "other-cad"},
        {"profile": "other-profile"},
        {"switches": []},
        {"strict_brep_valid": True},
        {"both_valid": True},
    ],
)
def test_isolated_worker_rejects_identity_and_validity_drift(
    tmp_path, monkeypatch, changed
):
    source = {
        "cad_id": "cad-protocol",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    profile = parse_profiles(["local_intersection_topology"])[0]
    payload = {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": "cad-protocol",
        "parent_id": "parent",
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": False,
        "source_path": "input.pkl",
        "status": "assembly_error",
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
    } | changed
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=WORKER_MARKER + json.dumps(payload) + "\n",
            stderr="",
        ),
    )

    row = run_one_isolated(
        source,
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
    )

    assert row["status"] == "worker_protocol_error"
    assert row["strict_brep_valid"] is False
    assert row["both_valid"] is False


def test_run_manifest_allows_exact_resume_and_rejects_drift(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    payload = {"schema": RUN_SCHEMA, "joint_iterations": 200, "profiles": ["x"]}
    first = bind_run_manifest(root, payload)
    second = bind_run_manifest(root, payload)
    assert first["signature"] == second["signature"]

    with pytest.raises(RuntimeError, match="different run signature"):
        bind_run_manifest(root, payload | {"joint_iterations": 0})


def test_run_manifest_rejects_unsigned_existing_artifacts(tmp_path):
    root = tmp_path / "unsigned"
    root.mkdir()
    (root / "assembly_repair_matrix.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="no signed run manifest"):
        bind_run_manifest(root, {"schema": RUN_SCHEMA})


def test_cli_rejects_nonpositive_worker_timeout_before_reading_inputs(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--calibration-manifest", str(tmp_path / "missing.jsonl"),
                "--breparg-root", str(tmp_path / "BrepARG"),
                "--output-dir", str(tmp_path / "output"),
                "--worker-timeout-seconds", "0",
            ]
        )
    assert caught.value.code == 2
