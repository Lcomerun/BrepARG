from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    generation = [
        str(args.python),
        str(source_root / "tools" / "run_breparg_generation_batches.py"),
        "--python",
        str(args.python),
        "--generator",
        str(source_root / "BrepARG" / "generate_brep.py"),
        "--config",
        str(source_root / "BrepARG" / "config.json"),
        "--ar-model",
        str(args.ar_model),
        "--se-vqvae",
        str(args.se_vqvae),
        "--output-dir",
        str(output_dir),
        "--target-count",
        str(args.target_count),
        "--batch-size",
        str(args.batch_size),
        "--max-batches",
        str(args.max_batches),
        "--batch-timeout-sec",
        str(args.batch_timeout_sec),
        "--max-attempts-per-batch",
        str(args.max_attempts_per_batch),
        "--start-seed",
        str(args.start_seed),
        "--max-length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--gpu",
        str(args.gpu),
        "--write-timeout",
        str(args.write_timeout),
    ]
    validation = [
        str(args.python),
        str(source_root / "tools" / "validate_breparg_generated_directory.py"),
        "--run-dir",
        str(output_dir),
        "--manifest-output",
        str(output_dir / "validated_manifest.jsonl"),
        "--summary-output",
        str(output_dir / "validated_summary.json"),
        "--timeout-sec",
        str(args.validation_timeout_sec),
    ]
    return [generation, validation]


def run_pipeline(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_command, validation_command = build_commands(args)
    generation = subprocess.run(generation_command, check=False)
    validation = subprocess.run(validation_command, check=False)
    if validation.returncode != 0:
        status = "validation_failed"
        returncode = validation.returncode
    elif generation.returncode != 0:
        status = "generation_incomplete"
        returncode = generation.returncode
    else:
        status = "completed"
        returncode = 0
    report = {
        "schema_version": 1,
        "status": status,
        "generation_returncode": generation.returncode,
        "validation_returncode": validation.returncode,
        "generation_command": generation_command,
        "validation_command": validation_command,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "repro_pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return int(returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BrepARG STEP files, then validate and render every survivor."
    )
    parser.add_argument("--python", default="python")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ar-model", type=Path, required=True)
    parser.add_argument("--se-vqvae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-attempts-per-batch", type=int, default=80)
    parser.add_argument("--start-seed", type=int, default=43)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write-timeout", type=int, default=120)
    parser.add_argument("--validation-timeout-sec", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
