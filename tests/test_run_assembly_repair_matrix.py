import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.run_assembly_repair_matrix as repair_matrix
from tools.assembly_repair import RepairProfile, parse_profiles
from tools.run_assembly_repair_matrix import (
    RUN_SCHEMA,
    WORKER_MARKER,
    build_run_payload,
    append_jsonl,
    bind_run_manifest,
    main,
    parse_worker_result,
    production_profile_topology_inputs,
    profile_kwargs,
    read_jsonl,
    requires_isolated_worker,
    run_one_isolated,
    sha256_file,
    source_pickle_binding,
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
        "curve_fit_rescue": False, "curve_interpolate": False,
        "wire_continuity": False, "single_solid": False,
        "surface_fit_precision": False,
        "solid_topology_repair": False,
        "pcurve_self_intersection": False,
        "local_intersection_topology": False,
        "local_pcurve_continuity": False,
    }
    assert profile_kwargs(RepairProfile("directed_trim", ("directed_trim",)))[
        "directed_trim"
    ] is True
    precision = profile_kwargs(
        parse_profiles(
            [
                "directed_trim_surface_precision_curve_rescue_"
                "local_intersection_topology"
            ]
        )[0]
    )
    assert precision["surface_fit_precision"] is True
    assert precision["curve_fit_rescue"] is True
    assert precision["local_intersection_topology"] is True
    precision_without_curve_rescue = profile_kwargs(
        parse_profiles(
            ["directed_trim_surface_precision_local_intersection_topology"]
        )[0]
    )
    assert precision_without_curve_rescue["surface_fit_precision"] is True
    assert precision_without_curve_rescue["curve_fit_rescue"] is False
    assert precision_without_curve_rescue["local_intersection_topology"] is True
    precision_interpolate = profile_kwargs(
        parse_profiles(["directed_trim_surface_precision_curve_interpolate"])[0]
    )
    assert precision_interpolate["directed_trim"] is True
    assert precision_interpolate["surface_fit_precision"] is True
    assert precision_interpolate["curve_interpolate"] is True
    assert precision_interpolate["curve_fit_rescue"] is False
    assert precision_interpolate["local_intersection_topology"] is False
    assert profile_kwargs(
        RepairProfile(
            "local_intersection_topology", ("local_intersection_topology",)
        )
    )["local_intersection_topology"] is True
    assert profile_kwargs(
        RepairProfile("curve_fit_rescue", ("curve_fit_rescue",))
    )["curve_fit_rescue"] is True
    assert profile_kwargs(
        RepairProfile("curve_interpolate", ("curve_interpolate",))
    )["curve_interpolate"] is True
    near = profile_kwargs(
        RepairProfile("near_vertex_reconciliation", ("near_vertex_reconciliation",))
    )
    assert near["single_solid"] is True
    assert near["solid_topology_repair"] is True
    assert profile_kwargs(
        RepairProfile(
            "local_pcurve_continuity", ("local_pcurve_continuity",)
        )
    )["local_pcurve_continuity"] is True


def test_production_single_solid_remaps_only_merged_endpoint_pair():
    points = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (1.0 + 5e-5, 1.0, 0.0),
    }
    edge_pairs = [(0, 1), (1, 2), (4, 3), (3, 0)]
    edge_wcs = np.asarray(
        [
            [points[left], points[right]]
            for left, right in edge_pairs
        ],
        dtype=np.float64,
    )
    adjacency = np.asarray(edge_pairs, dtype=np.int64)

    repaired_edges, remapped, diagnostics = production_profile_topology_inputs(
        RepairProfile("single_solid", ("single_solid",)),
        edge_wcs,
        adjacency,
        [[0, 1, 2, 3]],
    )

    assert remapped.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert diagnostics["solid_topology_repair"]["applied"] is True
    assert diagnostics["solid_topology_repair"]["production_endpoint_adjustment_count"] == 2
    np.testing.assert_allclose(repaired_edges[1, -1], repaired_edges[2, 0])
    np.testing.assert_allclose(repaired_edges[0], edge_wcs[0])


def test_production_topology_inputs_are_noop_without_single_solid():
    edge_wcs = np.zeros((1, 2, 3), dtype=np.float64)
    adjacency = np.asarray([[0, 0]], dtype=np.int64)

    repaired_edges, remapped, diagnostics = production_profile_topology_inputs(
        RepairProfile("baseline"),
        edge_wcs,
        adjacency,
        [[0]],
    )

    np.testing.assert_array_equal(repaired_edges, edge_wcs)
    np.testing.assert_array_equal(remapped, adjacency)
    assert diagnostics == {}


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    [
        ("baseline", False),
        ("directed_trim", False),
        ("near_vertex_reconciliation", True),
        ("curve_interpolate", False),
        ("directed_trim_curve_interpolate", False),
        ("directed_trim_curve_interpolate_local_intersection_topology", True),
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


def test_run_one_binds_exact_pickle_bytes_before_geometry_work(tmp_path, monkeypatch):
    source_path = tmp_path / "input.pkl"
    source_path.write_bytes(
        pickle.dumps(
            {
                "faceEdge_adj": [[0]],
                "edgeCorner_adj": [[0, 1]],
                "surf_ncs": [[[[]]]],
                "edge_ncs": [[[[]]]],
                "surf_bbox_wcs": [[0.0] * 6],
                "corner_unique": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            }
        )
    )
    expected_binding = source_pickle_binding(source_path)
    source = {
        "cad_id": "cad-direct-binding",
        "parent_id": "parent",
        "source_path": str(source_path),
        "brep_valid": False,
    }
    profile = parse_profiles(["baseline"])[0]

    def stop_before_occ(*args, **kwargs):
        raise RuntimeError("stop before OCC")

    monkeypatch.setattr(repair_matrix, "cpu_joint_optimize", stop_before_occ)
    row = repair_matrix.run_one(
        source,
        profile,
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=0,
        expected_source_binding=expected_binding,
    )

    assert row["status"] == "assembly_error"
    assert row["source_pickle_binding"] == expected_binding
    assert row["source_pickle_binding_after"] == expected_binding


def test_isolated_worker_rejects_mocked_source_pickle_hash_mismatch(tmp_path, monkeypatch):
    source_path = tmp_path / "input.pkl"
    source_path.write_bytes(b"signed pickle bytes")
    source = {
        "cad_id": "cad-binding",
        "parent_id": "parent",
        "source_path": str(source_path),
        "brep_valid": False,
    }
    profile = parse_profiles(["local_intersection_topology"])[0]
    expected_binding = source_pickle_binding(source_path)
    mismatched_binding = dict(expected_binding)
    mismatched_binding["sha256"] = "0" * 64
    payload = {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": source["cad_id"],
        "parent_id": source["parent_id"],
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": False,
        "source_path": source["source_path"],
        "status": "assembly_error",
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
        "source_pickle_binding": mismatched_binding,
        "source_pickle_binding_after": mismatched_binding,
    }

    def worker(command, **kwargs):
        binding_arg = command.index("--worker-source-binding-json") + 1
        assert json.loads(command[binding_arg]) == expected_binding
        assert str(source_path) not in command[binding_arg]
        return SimpleNamespace(
            returncode=0,
            stdout=WORKER_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", worker)
    row = run_one_isolated(
        source,
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
        expected_source_binding=expected_binding,
    )

    assert row["status"] == "worker_protocol_error"
    assert row["strict_brep_valid"] is False
    assert row["both_valid"] is False


def test_isolated_worker_forwards_selector_geometry_flag(tmp_path, monkeypatch):
    source = {
        "cad_id": "cad-gate",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    profile = parse_profiles(["near_vertex_reconciliation"])[0]
    payload = {
        "schema": "assembly-repair-matrix-v1",
        "cad_id": source["cad_id"],
        "parent_id": source["parent_id"],
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": False,
        "source_path": source["source_path"],
        "status": "assembly_error",
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
    }

    def worker(command, **kwargs):
        assert "--selector-geometry-gate" in command
        return SimpleNamespace(
            returncode=0,
            stdout=WORKER_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", worker)
    row = run_one_isolated(
        source,
        profile,
        calibration_manifest=tmp_path / "calibration.jsonl",
        output_dir=tmp_path,
        breparg_root=tmp_path / "BrepARG",
        joint_iterations=200,
        timeout_seconds=3,
        selector_geometry_gate=True,
    )
    assert row["status"] == "assembly_error"


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


def test_run_payload_records_selected_assembly_backend(tmp_path):
    manifest = tmp_path / "calibration_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    breparg_root = tmp_path / "BrepARG"
    breparg_root.mkdir()
    (breparg_root / "utils.py").write_text("# utility\n", encoding="utf-8")
    args = SimpleNamespace(
        calibration_manifest=manifest,
        breparg_root=breparg_root,
        joint_iterations=0,
        assembly_backend="production",
        historical_invalid_only=True,
        max_cads=None,
        isolate_cad_workers=True,
        worker_timeout_seconds=3.0,
    )

    payload = build_run_payload(
        args=args,
        full_rows=[],
        selected_rows=[],
        profiles=[RepairProfile("baseline")],
    )

    assert payload["assembly_backend"] == "production"
    assert payload["breparg_runtime"]["utils_sha256"] == sha256_file(
        breparg_root / "utils.py"
    )


def test_run_manifest_resume_clears_failure_only_exception_text(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    payload = {"schema": RUN_SCHEMA, "joint_iterations": 200, "profiles": ["x"]}
    first = bind_run_manifest(root, payload)
    first["status"] = "FAILED"
    first["error_type"] = "FileNotFoundError"
    first["error"] = r"D:\private\source.pkl"
    (root / "assembly_repair_run.json").write_text(json.dumps(first))

    resumed = bind_run_manifest(root, payload)

    assert resumed["status"] == "RUNNING"
    assert "error" not in resumed
    assert "error_type" not in resumed


def test_jsonl_resume_recovers_only_an_unterminated_final_torn_write(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_jsonl(path, {"cad": "first"})
    path.write_bytes(path.read_bytes() + b'{"cad":"torn"')

    rows = read_jsonl(path, recover_truncated_tail=True)

    assert rows == [{"cad": "first"}]
    assert path.read_text() == '{"cad": "first"}\n'

    path.write_bytes(path.read_bytes() + b'{"cad":"corrupt"\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, recover_truncated_tail=True)


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


def test_cli_requires_isolated_workers_for_production_backend(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--calibration-manifest", str(tmp_path / "missing.jsonl"),
                "--breparg-root", str(tmp_path / "BrepARG"),
                "--output-dir", str(tmp_path / "output"),
                "--assembly-backend", "production",
            ]
        )
    assert caught.value.code == 2
