"""Archive full Git-safe P0-A case, attempt, taxonomy, and ablation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from snapshot_p0a_assembly_chain import (
    build_ablation_summary,
    build_failure_taxonomy,
    detailed_attempt,
    read_json,
    read_jsonl,
    validate_evidence,
    write_json,
    write_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(run_root: Path, report_dir: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(run_root / "assembly_chain_summary.json")
    cases = read_json(run_root / "assembly_chain_cases.json")
    attempts = read_jsonl(run_root / "assembly_chain_attempts.jsonl")
    validate_evidence(summary, cases, attempts)

    cases = sorted(cases, key=lambda row: str(row.get("cad_id")))
    detailed = sorted(
        (detailed_attempt(row, run_root=run_root) for row in attempts),
        key=lambda row: (
            str(row.get("cad_id")),
            int(row.get("joint_iterations", -1)),
            float(row.get("sewing_tolerance", -1)),
        ),
    )
    write_jsonl(report_dir / "assembly_chain_cases.jsonl", cases)
    write_jsonl(report_dir / "assembly_chain_attempts_detailed.jsonl", detailed)
    write_json(report_dir / "failure_taxonomy.json", build_failure_taxonomy(cases))
    ablations = build_ablation_summary(attempts, cases)
    write_json(
        report_dir / "joint_optimize_ablation.json",
        ablations["joint_optimize_ablation"],
    )
    write_json(report_dir / "tolerance_scan.json", ablations["tolerance_scan"])

    artifacts = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(report_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        report_dir / "artifact_manifest.json",
        {
            "policy": "STEP, pickle, checkpoint, reconstructed-array, and machine-local path bytes remain local.",
            "artifacts": artifacts,
        },
    )
    return {
        "cases": len(cases),
        "attempts": len(detailed),
        "report_dir": str(report_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(snapshot(args.run_root, args.report_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
