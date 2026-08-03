import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import tools.repro_package_builder as builder
from tools.repro_package_builder import PackageBuildError, build_package, verify_archive


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def initialise_fixture_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=repo, check=True
    )
    write(repo / "README.md", "fixture repository\n")
    write(repo / "breparg_improvements" / "train.py", "VERSION = 'clean'\n")
    write(repo / "docs" / "full_experiment_postmortem_20260731.md", "# Postmortem\n")
    write(repo / "plans" / "first_execplan.md", "# First experiment\n")
    write(repo / "BrepARG" / "model.py", "UPSTREAM = True\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "clean fixture"], cwd=repo, check=True)
    clean_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    write(repo / "breparg_improvements" / "train.py", "VERSION = 'dirty'\n")
    write(repo / "tools" / "diagnostic.py", "print('diagnostic')\n")
    write(repo / "local_reports" / "result.md", "confirmed result\n")
    write(
        repo / "local_runs" / "failed" / "train.log",
        "start\n" + ("ordinary training batch record\n" * 12000)
        + "CUDA out of memory\nval_CE=nan\n",
    )
    artifact = repo / "ABC" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"external checkpoint")

    write(repo / "reproducibility" / "reproduce.sh", "#!/usr/bin/env bash\nexit 0\n")
    write(repo / "reproducibility" / "launchers" / "repro_cli.py", "print('cli')\n")
    write(repo / "reproducibility" / "launchers" / "repro_runtime.py", "VALUE = 1\n")
    write(repo / "reproducibility" / "configs" / "paths.env.example", "V13_RUN_ROOT=/workspace/v13/runs\n")
    write(repo / "reproducibility" / "environments" / "environment.linux-gpu.yml", "name: fixture\n")
    write(repo / "reproducibility" / "docs" / "START_HERE.md", "# Start\n")
    write(repo / "reproducibility" / "project_history" / "00_READ_ME_FIRST.md", "# History\n")
    experiments = {
        "schema_version": 1,
        "experiments": [
            {
                "schema_version": 1,
                "id": "diagnostic",
                "title": "Fixture diagnostic",
                "category": "diagnostics",
                "state": "runnable",
                "working_directory": "${SOURCE_ROOT}",
                "command": ["python", "${SOURCE_ROOT}/tools/diagnostic.py"],
                "smoke_command": ["python", "${SOURCE_ROOT}/tools/diagnostic.py"],
                "required_artifacts": ["model"],
                "expected_outputs": [],
            }
        ],
        "coverage_rules": [
            {"pattern": "plans/**", "classification": "diagnostic"},
            {"pattern": "local_reports/**", "classification": "diagnostic"},
            {"pattern": "local_runs/**", "classification": "diagnostic"},
        ],
    }
    (repo / "reproducibility" / "catalog").mkdir()
    (repo / "reproducibility" / "catalog" / "experiments.json").write_text(
        json.dumps(experiments), encoding="utf-8"
    )
    artifacts = {
        "schema_version": 1,
        "artifacts": [
            {
                "schema_version": 1,
                "id": "model",
                "title": "Fixture model",
                "type": "file",
                "path_var": "V13_AR_CHECKPOINT",
                "build_source_path": "ABC/model.pt",
                "required_identity": True,
            }
        ],
    }
    (repo / "reproducibility" / "catalog" / "artifacts.json").write_text(
        json.dumps(artifacts), encoding="utf-8"
    )
    return repo, clean_commit


def test_build_package_assembles_and_verifies_lightweight_archive(tmp_path: Path) -> None:
    repo, clean_commit = initialise_fixture_repository(tmp_path)
    output = tmp_path / "dist"
    epoch = int(datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc).timestamp())

    result = build_package(
        repo,
        output,
        "v13_repro_source_20260802.zip",
        epoch,
        clean_commit=clean_commit,
        history_roots=[repo / "local_runs"],
    )

    archive_path = Path(result["archive_path"])
    assert archive_path.is_file()
    assert Path(result["checksum_path"]).is_file()
    assert result["stage_validation"]["status"] == "ok"
    assert result["archive_verification"]["status"] == "ok"
    assert verify_archive(archive_path)["status"] == "ok"

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        prefix = "v13_repro_source_20260802/"
        assert prefix + "START_HERE.md" in names
        assert prefix + "SHA256SUMS" in names
        assert prefix + "source/current/breparg_improvements/train.py" in names
        assert prefix + "source/clean_head_16cf19b/breparg_improvements/train.py" in names
        assert prefix + "artifact_specs/model.json" in names
        assert prefix + "provenance/outer_untracked_manifest.json" in names
        assert prefix + "provenance/breparg_source_manifest.json" in names
        assert prefix + "project_history/05_original_records/local_reports/result.md" in names
        assert any(name.endswith("train.log.evidence.json") for name in names)
        assert prefix + "ABC/model.pt" not in names
        assert not any(name.endswith(".pt") for name in names)
        dirty = archive.read(
            prefix + "source/current/breparg_improvements/train.py"
        ).decode("utf-8")
        clean = archive.read(
            prefix + "source/clean_head_16cf19b/breparg_improvements/train.py"
        ).decode("utf-8")
        artifact_spec = json.loads(
            archive.read(prefix + "artifact_specs/model.json").decode("utf-8")
        )
        assert "dirty" in dirty
        assert "clean" in clean
        assert artifact_spec["size_bytes"] == len(b"external checkpoint")
        assert len(artifact_spec["sha256"]) == 64

        untracked = json.loads(
            archive.read(prefix + "provenance/outer_untracked_manifest.json").decode(
                "utf-8"
            )
        )
        assert any(
            row["path"] == "tools/diagnostic.py"
            and row["decision"] == "included_in_current_source"
            for row in untracked["files"]
        )
        assert any(
            row["path"] == "ABC/model.pt"
            and row["decision"] == "excluded_from_current_source"
            for row in untracked["files"]
        )
        breparg_manifest = json.loads(
            archive.read(prefix + "provenance/breparg_source_manifest.json").decode(
                "utf-8"
            )
        )
        assert breparg_manifest["file_count"] == 1
        assert breparg_manifest["files"][0]["path"] == "model.py"


def test_build_package_retains_stage_when_post_zip_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, clean_commit = initialise_fixture_repository(tmp_path)
    output = tmp_path / "dist"
    epoch = int(datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc).timestamp())

    def fail_verification(_archive_path: Path) -> dict[str, object]:
        raise PackageBuildError("deliberate post-zip verification failure")

    monkeypatch.setattr(builder, "verify_archive", fail_verification)

    with pytest.raises(PackageBuildError, match="deliberate post-zip"):
        build_package(
            repo,
            output,
            "v13_repro_source_20260802.zip",
            epoch,
            clean_commit=clean_commit,
            history_roots=[repo / "local_runs"],
        )

    stage = output / ".v13_repro_source_20260802.stage"
    assert stage.is_dir()
    execution = json.loads(
        (output / "v13_repro_source_20260802.build-execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["status"] == "failed"
    assert Path(execution["stage_retained"]).resolve() == stage.resolve()
