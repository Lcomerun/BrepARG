import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_host_cli_runs_as_a_direct_script() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/build_repro_source_package.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--validate-only" in completed.stdout


def test_validate_only_prints_catalog_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.build_repro_source_package as cli

    repo = tmp_path / "repo"
    repo.mkdir()
    expected = {"status": "ok", "experiment_count": 3, "artifact_count": 2}
    monkeypatch.setattr(cli, "validate_catalogs", lambda root: expected)

    assert cli.main(["--repo-root", str(repo), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_build_uses_source_date_epoch_and_default_history_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.build_repro_source_package as cli

    repo = tmp_path / "repo"
    output = tmp_path / "dist"
    repo.mkdir()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    observed: dict[str, object] = {}

    def fake_build_package(
        repo_root: Path,
        output_dir: Path,
        package_name: str,
        epoch: int,
        *,
        clean_commit: str,
        history_roots: list[Path],
    ) -> dict[str, object]:
        observed.update(
            repo_root=repo_root,
            output_dir=output_dir,
            package_name=package_name,
            epoch=epoch,
            clean_commit=clean_commit,
            history_roots=history_roots,
        )
        return {"status": "ok", "archive_sha256": "a" * 64}

    monkeypatch.setattr(cli, "build_package", fake_build_package)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1785657600")

    assert (
        cli.main(
            [
                "--repo-root",
                str(repo),
                "--output-dir",
                str(output),
                "--history-root",
                str(recovery),
            ]
        )
        == 0
    )
    assert observed["repo_root"] == repo.resolve()
    assert observed["output_dir"] == output.resolve()
    assert observed["package_name"] == "v13_repro_source_20260802.zip"
    assert observed["epoch"] == 1785657600
    assert observed["history_roots"] == [recovery.resolve()]
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_verify_archive_mode_does_not_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tools.build_repro_source_package as cli

    archive = tmp_path / "package.zip"
    archive.write_bytes(b"fixture")
    expected = {"status": "ok", "archive_sha256": "b" * 64}
    monkeypatch.setattr(cli, "verify_archive", lambda path: expected)
    monkeypatch.setattr(
        cli,
        "build_package",
        lambda *args, **kwargs: pytest.fail("build must not run in verify mode"),
    )

    assert cli.main(["--verify-archive", str(archive)]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize("value", ["not-an-integer", "-1", "0"])
def test_invalid_source_date_epoch_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.build_repro_source_package as cli

    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    with pytest.raises(SystemExit):
        cli.main([])
