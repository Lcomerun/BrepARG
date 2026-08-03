import argparse
import json
from pathlib import Path

import reproducibility.launchers.run_breparg_generation_pipeline as pipeline


def make_args(tmp_path: Path) -> argparse.Namespace:
    source_root = tmp_path / "source"
    (source_root / "tools").mkdir(parents=True)
    (source_root / "BrepARG").mkdir(parents=True)
    for relative in (
        "tools/run_breparg_generation_batches.py",
        "tools/validate_breparg_generated_directory.py",
        "BrepARG/generate_brep.py",
        "BrepARG/config.json",
    ):
        path = source_root / relative
        path.write_text("fixture\n", encoding="utf-8")
    ar = tmp_path / "ar.pt"
    vq = tmp_path / "vq.pt"
    ar.write_bytes(b"ar")
    vq.write_bytes(b"vq")
    return argparse.Namespace(
        python="python",
        source_root=source_root,
        ar_model=ar,
        se_vqvae=vq,
        output_dir=tmp_path / "output",
        target_count=3,
        batch_size=2,
        max_batches=4,
        batch_timeout_sec=30.0,
        max_attempts_per_batch=10,
        start_seed=43,
        max_length=1536,
        temperature=1.0,
        top_p=0.9,
        gpu=0,
        write_timeout=30,
        validation_timeout_sec=15,
    )


def test_build_commands_retains_generation_and_render_protocol(tmp_path: Path) -> None:
    args = make_args(tmp_path)

    commands = pipeline.build_commands(args)

    generation, validation = commands
    assert generation[:2] == [
        "python",
        str(args.source_root / "tools" / "run_breparg_generation_batches.py"),
    ]
    assert generation[generation.index("--ar-model") + 1] == str(args.ar_model)
    assert generation[generation.index("--target-count") + 1] == "3"
    assert validation[:2] == [
        "python",
        str(args.source_root / "tools" / "validate_breparg_generated_directory.py"),
    ]
    assert "--skip-preview" not in validation
    assert validation[validation.index("--run-dir") + 1] == str(args.output_dir)


def test_run_pipeline_validates_even_when_generation_is_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    args = make_args(tmp_path)
    returncodes = iter([1, 0])
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": next(returncodes)})()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    result = pipeline.run_pipeline(args)

    assert result == 1
    assert len(calls) == 2
    report = json.loads(
        (args.output_dir / "repro_pipeline_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "generation_incomplete"
    assert report["generation_returncode"] == 1
    assert report["validation_returncode"] == 0
