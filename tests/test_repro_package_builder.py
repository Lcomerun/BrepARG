import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.repro_package_builder import (
    PackageBuildError,
    _history_target_path,
    copy_or_summarize_history_file,
    export_clean_head,
    validate_catalogs,
    should_include_current_source,
    stable_json_bytes,
    validate_stage,
    write_checksum_manifest,
    write_deterministic_zip,
)


def test_stable_json_bytes_are_order_independent_and_finite() -> None:
    first = stable_json_bytes({"b": 2, "a": [3, 1]})
    second = stable_json_bytes({"a": [3, 1], "b": 2})

    assert first == second == b'{"a":[3,1],"b":2}\n'
    with pytest.raises(ValueError):
        stable_json_bytes({"loss": float("nan")})


@pytest.mark.parametrize(
    ("relative", "size", "included"),
    [
        ("breparg_improvements/train.py", 1024, True),
        ("BrepARG/model.py", 1024, True),
        ("tools/check.py", 1024, True),
        ("plans/experiment.md", 1024, True),
        ("papers/aaai_v13/main.tex", 1024, True),
        ("ABC/model.pt", 1024, False),
        ("local_runs/run/report.json", 1024, False),
        ("BrepARG/.git/config", 1024, False),
        ("BrepARG/__pycache__/model.pyc", 1024, False),
        ("papers/paper.pdf", 1024, False),
        ("tools/huge.py", 9 * 1024 * 1024, False),
        ("random.bin", 12, False),
    ],
)
def test_current_source_filter(relative: str, size: int, included: bool) -> None:
    assert should_include_current_source(Path(relative), size) is included


def test_checksum_manifest_excludes_itself_and_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z\n", encoding="utf-8")
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "b.txt").write_text("b\n", encoding="utf-8")

    checksum_path = write_checksum_manifest(tmp_path)
    lines = checksum_path.read_text(encoding="utf-8").splitlines()

    assert [line.split("  ", 1)[1] for line in lines] == ["a/b.txt", "z.txt"]
    assert all("SHA256SUMS" not in line for line in lines)


def test_checksum_manifest_supports_utf8_paths(tmp_path: Path) -> None:
    relative = Path("reports") / "\u590d\u76d8.md"
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_text("history\n", encoding="utf-8")

    checksum_path = write_checksum_manifest(tmp_path)
    line = checksum_path.read_text(encoding="utf-8").strip()

    assert line.endswith(f"  {relative.as_posix()}")


def test_deterministic_zip_has_stable_hash_order_and_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "v13_repro_source_20260802"
    source.mkdir()
    (source / "z.txt").write_text("z", encoding="utf-8")
    (source / "a.txt").write_text("a", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    epoch = int(datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc).timestamp())
    first_hash = write_deterministic_zip(source, first, epoch=epoch)
    second_hash = write_deterministic_zip(source, second, epoch=epoch)

    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash == hashlib.sha256(first.read_bytes()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "v13_repro_source_20260802/a.txt",
            "v13_repro_source_20260802/z.txt",
        ]
        assert {entry.date_time for entry in archive.infolist()} == {
            (2026, 8, 2, 8, 0, 0)
        }


def test_history_copy_and_summary_preserve_evidence(tmp_path: Path) -> None:
    small = tmp_path / "small.log"
    small.write_text("start\nloss=0.5\nDONE\n", encoding="utf-8")
    large = tmp_path / "large.log"
    large.write_text(
        "HEADER\n"
        + ("ordinary batch line\n" * 80)
        + "CUDA out of memory\n"
        + ("more batches\n" * 80)
        + "val_CE=nan\nTAIL\n",
        encoding="utf-8",
    )
    output = tmp_path / "history"

    small_record = copy_or_summarize_history_file(
        small,
        output / "small.log",
        original_label="local_runs/small.log",
        direct_copy_limit=1024,
    )
    large_record = copy_or_summarize_history_file(
        large,
        output / "large.log",
        original_label="local_runs/large.log",
        direct_copy_limit=128,
    )

    assert small_record["mode"] == "copied"
    assert (output / "small.log").read_bytes() == small.read_bytes()
    assert large_record["mode"] == "evidence_summary"
    evidence_path = Path(large_record["packaged_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["original_path"] == "local_runs/large.log"
    assert evidence["size_bytes"] == large.stat().st_size
    assert evidence["sha256"] == hashlib.sha256(large.read_bytes()).hexdigest()
    assert any("out of memory" in line.lower() for line in evidence["event_lines"])
    assert any("val_CE=nan" in line for line in evidence["event_lines"])
    assert evidence["head_excerpt"].startswith("HEADER")
    assert evidence["tail_excerpt"].rstrip().endswith("TAIL")


def test_history_target_path_compacts_deep_paths_deterministically(tmp_path: Path) -> None:
    package_history = tmp_path / "package" / "project_history"
    destination = Path("05_original_records/local_runs")
    short = Path("run/report.json")
    deep = Path("complex_curved_rootcause_suite_20260715") / Path(
        "experiments/01_teacher_forcing_true_token_reconstruction/"
        "complex_curved_diagnostics_report.json"
    )

    short_target, short_compacted = _history_target_path(
        package_history, destination, short
    )
    first, first_compacted = _history_target_path(package_history, destination, deep)
    second, second_compacted = _history_target_path(package_history, destination, deep)

    assert short_target == package_history / destination / short
    assert short_compacted is False
    assert first == second
    assert first_compacted is True
    assert second_compacted is True
    assert "_compacted" in first.parts
    assert first.suffix == ".json"
    assert len(first.relative_to(package_history.parent).as_posix()) <= 180


def test_history_target_path_compacts_before_windows_max_path_boundary(
    tmp_path: Path,
) -> None:
    package_history = tmp_path / "package" / "project_history"
    destination = Path("05_original_records/local_runs")
    relative = Path(
        "breparg_logic_compare_20260715/"
        "breparg_original_logic_100_20260715_rerun/"
        "breparg_baseline_quality_manifest.jsonl"
    )

    target, compacted = _history_target_path(
        package_history, destination, relative
    )

    assert compacted is True
    assert target.parent == package_history / "_compacted"
    assert len(target.relative_to(package_history.parent).as_posix()) <= 100


def test_validate_stage_rejects_forbidden_and_unsafe_content(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "START_HERE.md").write_text("portable\n", encoding="utf-8")
    (root / "reproduce.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    assert validate_stage(root)["status"] == "ok"

    checkpoint = root / "source" / "current" / "bad.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weight")
    with pytest.raises(PackageBuildError, match="forbidden packaged extension"):
        validate_stage(root)
    checkpoint.unlink()

    git_config = root / "source" / "current" / ".git" / "config"
    git_config.parent.mkdir(parents=True)
    git_config.write_text("gitdir", encoding="utf-8")
    with pytest.raises(PackageBuildError, match="nested .git"):
        validate_stage(root)
    git_config.unlink()
    git_config.parent.rmdir()

    experiment = root / "experiments" / "recommended" / "bad.json"
    experiment.parent.mkdir(parents=True)
    experiment.write_text('{"command":["/root/autodl-tmp/run.py"]}', encoding="utf-8")
    with pytest.raises(PackageBuildError, match="host-specific runtime path"):
        validate_stage(root)


def test_validate_stage_allows_legacy_paths_only_in_history_and_source(tmp_path: Path) -> None:
    root = tmp_path / "package"
    history = root / "project_history"
    source = root / "source" / "current"
    history.mkdir(parents=True)
    source.mkdir(parents=True)
    (history / "incident.md").write_text("D:\\old\\run\n", encoding="utf-8")
    (source / "legacy.py").write_text("ROOT='/root/autodl-tmp/workplace'\n", encoding="utf-8")

    assert validate_stage(root)["status"] == "ok"


def test_export_clean_head_exports_only_committed_files(tmp_path: Path) -> None:
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
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    target = tmp_path / "export"

    manifest = export_clean_head(repo, commit, target)

    assert (target / "tracked.txt").read_text(encoding="utf-8") == "clean\n"
    assert not (target / "untracked.txt").exists()
    assert manifest["commit"] == commit
    assert manifest["file_count"] == 1


def test_validate_catalogs_reports_complete_references(tmp_path: Path) -> None:
    catalog_root = tmp_path / "reproducibility" / "catalog"
    catalog_root.mkdir(parents=True)
    (catalog_root / "artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "schema_version": 1,
                        "id": "model",
                        "type": "file",
                        "path_var": "MODEL_PATH",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (catalog_root / "experiments.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [
                    {
                        "schema_version": 1,
                        "id": "probe",
                        "category": "diagnostics",
                        "state": "runnable",
                        "command": ["python", "probe.py"],
                        "required_artifacts": ["model"],
                    }
                ],
                "coverage_rules": [],
            }
        ),
        encoding="utf-8",
    )

    report = validate_catalogs(tmp_path)

    assert report["status"] == "ok"
    assert report["experiment_count"] == 1
    assert report["artifact_count"] == 1
    assert report["referenced_artifact_count"] == 1


def test_validate_catalogs_rejects_unknown_artifact_reference(tmp_path: Path) -> None:
    catalog_root = tmp_path / "reproducibility" / "catalog"
    catalog_root.mkdir(parents=True)
    (catalog_root / "artifacts.json").write_text(
        '{"schema_version":1,"artifacts":[]}', encoding="utf-8"
    )
    (catalog_root / "experiments.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [
                    {
                        "schema_version": 1,
                        "id": "bad",
                        "category": "diagnostics",
                        "state": "runnable",
                        "command": ["python", "bad.py"],
                        "required_artifacts": ["missing"],
                    }
                ],
                "coverage_rules": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PackageBuildError, match="unknown artifacts"):
        validate_catalogs(tmp_path)


def test_validate_catalogs_rejects_unsafe_binding_and_unknown_placeholder(
    tmp_path: Path,
) -> None:
    catalog_root = tmp_path / "reproducibility" / "catalog"
    catalog_root.mkdir(parents=True)
    (catalog_root / "artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "schema_version": 1,
                        "id": "model",
                        "type": "file",
                        "path_var": "MODEL_PATH",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (catalog_root / "experiments.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [
                    {
                        "schema_version": 1,
                        "id": "bad",
                        "category": "diagnostics",
                        "state": "runnable",
                        "command": ["python", "${UNKNOWN}/bad.py"],
                        "required_artifacts": ["model"],
                        "artifact_bindings": {"model": "../model.pt"},
                    }
                ],
                "coverage_rules": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PackageBuildError, match="unsafe artifact binding|unknown placeholder"):
        validate_catalogs(tmp_path)


def test_validate_catalogs_rejects_unsafe_expected_output(tmp_path: Path) -> None:
    catalog_root = tmp_path / "reproducibility" / "catalog"
    catalog_root.mkdir(parents=True)
    (catalog_root / "artifacts.json").write_text(
        '{"schema_version":1,"artifacts":[]}', encoding="utf-8"
    )
    (catalog_root / "experiments.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [
                    {
                        "schema_version": 1,
                        "id": "bad_output",
                        "category": "diagnostics",
                        "state": "runnable",
                        "command": ["python", "probe.py"],
                        "required_artifacts": [],
                        "expected_outputs": [{"path": "/tmp/result.json", "kind": "json"}],
                    }
                ],
                "coverage_rules": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PackageBuildError, match="unsafe expected output"):
        validate_catalogs(tmp_path)
