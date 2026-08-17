"""Create a Git-safe snapshot of an assembly repair pilot or formal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_SUFFIXES = {".step", ".stp", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npy", ".npz"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    components = row.get("validity_components") or {}
    return {
        key: row.get(key)
        for key in (
            "schema", "cad_id", "parent_id", "profile", "switches",
            "historical_strict_valid", "status", "step_saved",
            "native_brep_valid", "strict_brep_valid", "both_valid",
            "step_bytes", "step_sha256", "error_type", "error",
            "elapsed_seconds",
        )
    } | {
        "validity_components": {
            key: components.get(key)
            for key in (
                "wire_order_failures", "wire_self_intersections", "free_edges",
                "shell_count", "shells_with_bad_edges", "solid_count",
            )
        },
        "source_bytes_archived": False,
        "step_bytes_archived": False,
    }


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(report_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def snapshot(run_root: Path, report_dir: Path, *, label: str) -> dict[str, Any]:
    run_root, report_dir = Path(run_root).resolve(), Path(report_dir).resolve()
    source = run_root / "assembly_repair_matrix.jsonl"
    summary_path = run_root / "assembly_repair_summary.json"
    rows = read_jsonl(source)
    if not rows or not summary_path.is_file():
        raise RuntimeError("assembly repair run is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_dir.mkdir(parents=True, exist_ok=True)
    if any(report_dir.iterdir()):
        raise RuntimeError(f"report directory must be empty: {report_dir}")
    compact = [compact_row(row) for row in rows]
    (report_dir / "assembly_repair_attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compact), encoding="utf-8"
    )
    source_binding = {
        "label": label, "source_run_name": run_root.name,
        "source_manifest_bytes": source.stat().st_size,
        "source_manifest_sha256": sha256_file(source),
        "step_files_local": sum(bool(row.get("step_saved")) for row in rows),
        "step_bytes_archived": False, "source_pickles_archived": False,
    }
    archived_summary = {**summary, "label": label, "generated_at": now(), "source_binding": source_binding}
    (report_dir / "assembly_repair_summary.json").write_text(
        json.dumps(archived_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile_lines = "\n".join(
        f"| {item['profile']} | {item['attempts']} | {item['step_readable']} | {item['native_valid']} | {item['strict_valid']} | {item['both_valid']} | {len(item['restored_cad_ids'])} | {len(item['regressed_cad_ids'])} |"
        for item in summary["profiles"]
    )
    readme = f"""# Assembly repair evidence: {label}

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{profile_lines}

Gate passed: `{summary.get('gate_passed')}`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    forbidden = [
        path.relative_to(report_dir).as_posix() for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden artifacts entered report: {forbidden}")
    validation = {
        "valid": True, "attempts": len(compact),
        "profiles": dict(Counter(str(row.get("profile")) for row in compact)),
        "forbidden_artifacts": forbidden,
    }
    (report_dir / "archive_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "artifact_manifest.json").write_text(
        json.dumps({"generated_at": now(), "artifacts": artifact_manifest(report_dir)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(snapshot(args.run_root, args.report_dir, label=args.label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
