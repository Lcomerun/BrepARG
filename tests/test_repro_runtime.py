import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from reproducibility.launchers import repro_runtime as runtime

from reproducibility.launchers.repro_runtime import (
    ReproError,
    create_run_context,
    load_artifact_specs,
    load_experiments,
    load_paths,
    run_experiment,
    verify_artifact,
    verify_package_checksums,
    verify_run,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    (root / "experiments" / "recommended").mkdir(parents=True)
    (root / "experiments" / "diagnostics").mkdir(parents=True)
    (root / "experiments" / "historical_failed").mkdir(parents=True)
    (root / "artifact_specs").mkdir()
    (root / "configs").mkdir()

    write_json(
        root / "experiments" / "recommended" / "ready.json",
        {
            "schema_version": 1,
            "id": "ready",
            "title": "Ready fixture",
            "category": "recommended",
            "state": "runnable",
            "command": [
                sys.executable,
                "-c",
                (
                    "import json, pathlib; "
                    "pathlib.Path('metrics.json').write_text("
                    "json.dumps({'loss': 0.25}), encoding='utf-8')"
                ),
            ],
            "smoke_command": [sys.executable, "-c", "print('smoke ok')"],
            "required_artifacts": [],
            "expected_outputs": [
                {
                    "path": "metrics.json",
                    "kind": "json",
                    "finite_fields": ["loss"],
                }
            ],
        },
    )
    write_json(
        root / "experiments" / "diagnostics" / "documentary.json",
        {
            "schema_version": 1,
            "id": "documentary",
            "title": "Documentary fixture",
            "category": "diagnostics",
            "state": "documentary",
            "blocked_reason": "Original command is not fully recoverable.",
            "required_artifacts": [],
            "expected_outputs": [],
        },
    )
    write_json(
        root / "experiments" / "historical_failed" / "failed.json",
        {
            "schema_version": 1,
            "id": "failed",
            "title": "Known failed fixture",
            "category": "historical_failed",
            "state": "runnable",
            "command": [sys.executable, "-c", "print('historical')"],
            "required_artifacts": [],
            "expected_outputs": [],
            "failure_reason": "Known nonfinite training branch.",
        },
    )
    return root


def test_load_paths_parses_values_without_shell_execution(tmp_path: Path) -> None:
    marker = tmp_path / "must_not_exist"
    paths_file = tmp_path / "paths.env"
    paths_file.write_text(
        "# local paths\n"
        "V13_RUN_ROOT=./runs\n"
        f'UNTRUSTED="$(touch {marker})"\n'
        "EMPTY=\n",
        encoding="utf-8",
    )

    paths = load_paths(paths_file)

    assert paths["V13_RUN_ROOT"] == "./runs"
    assert paths["UNTRUSTED"] == f"$(touch {marker})"
    assert paths["EMPTY"] == ""
    assert not marker.exists()


def test_load_experiments_rejects_duplicate_ids(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    duplicate = json.loads(
        (root / "experiments" / "recommended" / "ready.json").read_text(
            encoding="utf-8"
        )
    )
    write_json(root / "experiments" / "diagnostics" / "duplicate.json", duplicate)

    with pytest.raises(ReproError, match="duplicate experiment id: ready"):
        load_experiments(root)


def test_load_artifact_specs_rejects_duplicate_ids(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    spec = {
        "schema_version": 1,
        "id": "sequence",
        "type": "file",
        "path_var": "V13_SEQUENCE_PACKAGE",
    }
    write_json(root / "artifact_specs" / "one.json", spec)
    write_json(root / "artifact_specs" / "two.json", spec)

    with pytest.raises(ReproError, match="duplicate artifact id: sequence"):
        load_artifact_specs(root)


def test_verify_package_checksums_detects_mutation(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    payload = root / "START_HERE.md"
    payload.write_bytes(b"read me\n")
    (root / "SHA256SUMS").write_text(
        f"{sha256_bytes(payload.read_bytes())}  START_HERE.md\n", encoding="ascii"
    )

    assert verify_package_checksums(root) == [
        {
            "path": "START_HERE.md",
            "status": "ok",
            "expected_sha256": sha256_bytes(b"read me\n"),
            "actual_sha256": sha256_bytes(b"read me\n"),
        }
    ]

    payload.write_bytes(b"changed\n")
    result = verify_package_checksums(root)
    assert result[0]["status"] == "hash_mismatch"
    assert result[0]["actual_sha256"] == sha256_bytes(b"changed\n")


def test_verify_package_checksums_supports_utf8_paths(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    relative = Path("reports") / "\u590d\u76d8.md"
    payload = root / relative
    payload.parent.mkdir()
    payload.write_bytes(b"history\n")
    (root / "SHA256SUMS").write_text(
        f"{sha256_bytes(payload.read_bytes())}  {relative.as_posix()}\n",
        encoding="utf-8",
    )

    assert verify_package_checksums(root) == [
        {
            "path": relative.as_posix(),
            "status": "ok",
            "expected_sha256": sha256_bytes(b"history\n"),
            "actual_sha256": sha256_bytes(b"history\n"),
        }
    ]


def test_verify_artifact_checks_size_and_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "sequence.pkl"
    artifact.write_bytes(b"sequence fixture")
    spec = {
        "schema_version": 1,
        "id": "sequence",
        "type": "file",
        "path_var": "V13_SEQUENCE_PACKAGE",
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_bytes(artifact.read_bytes()),
    }

    ready = verify_artifact(spec, {"V13_SEQUENCE_PACKAGE": str(artifact)})
    assert ready["status"] == "ready"
    assert ready["verification_strength"] == "content_sha256"

    wrong = dict(spec, sha256="0" * 64)
    mismatch = verify_artifact(wrong, {"V13_SEQUENCE_PACKAGE": str(artifact)})
    assert mismatch["status"] == "hash_mismatch"
    assert mismatch["artifact_id"] == "sequence"


def test_verify_artifact_rejects_unresolved_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "unknown.pt"
    artifact.write_bytes(b"unknown")
    directory = tmp_path / "unknown_shards"
    directory.mkdir()
    (directory / "shard.bin").write_bytes(b"shard")

    file_result = verify_artifact(
        {
            "schema_version": 1,
            "id": "unknown_file",
            "type": "file",
            "path_var": "FILE",
        },
        {"FILE": str(artifact)},
    )
    directory_result = verify_artifact(
        {
            "schema_version": 1,
            "id": "unknown_directory",
            "type": "directory",
            "path_var": "DIRECTORY",
        },
        {"DIRECTORY": str(directory)},
    )

    assert file_result["status"] == "identity_unresolved"
    assert directory_result["status"] == "identity_unresolved"


def test_create_run_context_refuses_existing_directory(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    experiment = load_experiments(root)["ready"]
    paths = {"V13_RUN_ROOT": str(tmp_path / "runs")}
    now = datetime(2026, 8, 2, 10, 11, 12, tzinfo=timezone.utc)

    context = create_run_context(root, experiment, paths, now=now)

    assert Path(context["run_dir"]).is_dir()
    assert Path(context["run_dir"]).name.startswith("ready_20260802T101112Z_")
    assert (Path(context["run_dir"]) / "run_manifest.json").is_file()
    with pytest.raises(ReproError, match="run directory already exists"):
        create_run_context(root, experiment, paths, now=now)


def test_run_experiment_blocks_documentary_and_historical_by_default(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={tmp_path / 'runs'}\n", encoding="utf-8"
    )

    with pytest.raises(ReproError, match="not runnable"):
        run_experiment(root, "documentary")
    with pytest.raises(ReproError, match="--allow-historical-failed"):
        run_experiment(root, "failed")


def test_run_experiment_records_historical_opt_in(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    run_root = tmp_path / "runs"
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={run_root}\n", encoding="utf-8"
    )

    assert run_experiment(root, "failed", allow_historical_failed=True) == 0

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads(
        (run_dirs[0] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["historical_failed_opt_in"] is True
    assert manifest["status"] == "completed"


def test_run_experiment_applies_declared_working_directory_and_environment(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    source_root = root / "source" / "current"
    source_root.mkdir(parents=True)
    run_root = tmp_path / "runs"
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={run_root}\n", encoding="utf-8"
    )
    write_json(
        root / "experiments" / "diagnostics" / "configured.json",
        {
            "schema_version": 1,
            "id": "configured",
            "title": "Configured fixture",
            "category": "diagnostics",
            "state": "runnable",
            "working_directory": "${SOURCE_ROOT}",
            "environment": {"FIXTURE_OUTPUT": "${RUN_DIR}/from_env.txt"},
            "smoke_environment": {"FIXTURE_MODE": "smoke"},
            "command": [
                sys.executable,
                "-c",
                (
                    "import os, pathlib; "
                    "pathlib.Path(os.environ['FIXTURE_OUTPUT']).write_text("
                    "pathlib.Path.cwd().as_posix(), encoding='utf-8')"
                ),
            ],
            "smoke_command": [
                sys.executable,
                "-c",
                (
                    "import os, pathlib; "
                    "pathlib.Path(os.environ['FIXTURE_OUTPUT']).write_text("
                    "os.environ['FIXTURE_MODE'], encoding='utf-8')"
                ),
            ],
            "required_artifacts": [],
            "expected_outputs": [{"path": "from_env.txt", "kind": "file"}],
        },
    )

    assert run_experiment(root, "configured") == 0

    run_dir = next(run_root.iterdir())
    assert (run_dir / "from_env.txt").read_text(encoding="utf-8") == source_root.as_posix()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["working_directory"] == str(source_root.resolve())
    assert Path(manifest["environment_overrides"]["FIXTURE_OUTPUT"]).resolve() == (
        run_dir / "from_env.txt"
    ).resolve()

    smoke_root = tmp_path / "smoke_runs"
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={smoke_root}\n", encoding="utf-8"
    )
    assert run_experiment(root, "configured", smoke=True) == 0
    smoke_dir = next(smoke_root.iterdir())
    assert (smoke_dir / "from_env.txt").read_text(encoding="utf-8") == "smoke"
    smoke_manifest = json.loads(
        (smoke_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert smoke_manifest["environment_overrides"]["FIXTURE_MODE"] == "smoke"


def test_run_experiment_uses_control_environment_python_for_generic_command(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    source_root = root / "source" / "current"
    source_root.mkdir(parents=True)
    run_root = tmp_path / "runs"
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={run_root}\n", encoding="utf-8"
    )
    write_json(
        root / "experiments" / "diagnostics" / "generic_python.json",
        {
            "schema_version": 1,
            "id": "generic_python",
            "title": "Generic Python fixture",
            "category": "diagnostics",
            "state": "runnable",
            "command": [
                "python",
                "-c",
                "import pathlib,sys; pathlib.Path('python.txt').write_text(sys.executable)",
            ],
            "required_artifacts": [],
            "expected_outputs": [{"path": "python.txt", "kind": "file"}],
        },
    )

    assert run_experiment(root, "generic_python") == 0

    run_dir = next(run_root.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["command"][0]).resolve() == Path(sys.executable).resolve()
    assert Path((run_dir / "python.txt").read_text()).resolve() == Path(sys.executable).resolve()


def test_run_experiment_binds_verified_artifact_without_copying(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    run_root = tmp_path / "runs"
    artifact = tmp_path / "large_sequence.pkl"
    artifact.write_bytes(b"external sequence")
    (root / "configs" / "paths.env").write_text(
        f"V13_RUN_ROOT={run_root}\nV13_SEQUENCE_PACKAGE={artifact}\n",
        encoding="utf-8",
    )
    write_json(
        root / "artifact_specs" / "sequence.json",
        {
            "schema_version": 1,
            "id": "sequence",
            "type": "file",
            "path_var": "V13_SEQUENCE_PACKAGE",
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_bytes(artifact.read_bytes()),
        },
    )
    write_json(
        root / "experiments" / "diagnostics" / "binding.json",
        {
            "schema_version": 1,
            "id": "binding",
            "title": "Binding fixture",
            "category": "diagnostics",
            "state": "runnable",
            "artifact_bindings": {"sequence": "inputs/sequence.pkl"},
            "command": [
                sys.executable,
                "-c",
                (
                    "import pathlib; p=pathlib.Path('inputs/sequence.pkl'); "
                    "assert p.read_bytes() == b'external sequence'"
                ),
            ],
            "required_artifacts": ["sequence"],
            "expected_outputs": [],
        },
    )

    assert run_experiment(root, "binding") == 0

    run_dir = next(run_root.iterdir())
    bound = run_dir / "inputs" / "sequence.pkl"
    assert bound.read_bytes() == artifact.read_bytes()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_bindings"][0]["artifact_id"] == "sequence"
    assert manifest["artifact_bindings"][0]["method"] in {"hardlink", "symlink"}


def test_verify_run_rejects_nonfinite_json_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"loss": NaN}', encoding="utf-8")
    experiment = {
        "id": "metrics",
        "expected_outputs": [
            {"path": "metrics.json", "kind": "json", "finite_fields": ["loss"]}
        ],
    }

    result = verify_run(run_dir, experiment)

    assert result["status"] == "failed"
    assert result["outputs"][0]["status"] == "nonfinite"
    assert math.isnan(result["outputs"][0]["value"])


def test_preflight_reports_system_capabilities_without_optional_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = make_package(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(runtime, "_probe_import", lambda name: {"available": False})
    monkeypatch.setattr(
        runtime,
        "_disk_probe",
        lambda path: {"path": str(path), "free_bytes": 1234, "total_bytes": 5678},
    )

    report = runtime.package_preflight(package)

    capabilities = report["capabilities"]
    assert capabilities["disk"]["free_bytes"] == 1234
    assert capabilities["commands"]["conda"]["available"] is False
    assert capabilities["nvidia"]["available"] is False
    assert capabilities["python_modules"]["torch"]["available"] is False
    assert capabilities["python_modules"]["OCC"]["available"] is False
    assert capabilities["cuda"]["status"] == "torch_unavailable"
    assert report["target_ready"] is False
