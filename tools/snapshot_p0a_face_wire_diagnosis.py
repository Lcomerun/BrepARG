"""Archive P0-A face/wire diagnosis as a Git-safe evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .snapshot_assembly_repair import FORBIDDEN_SUFFIXES, sha256_file
except ImportError:  # direct script execution
    from snapshot_assembly_repair import FORBIDDEN_SUFFIXES, sha256_file


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(report_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def compact_case(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only measurements and hashes; remove every local source reference."""
    return {
        key: row.get(key)
        for key in (
            "schema", "cad_id", "parent_id", "historical_status",
            "historical_step_saved", "source_pickle_sha256", "source_topology",
            "step_diagnosis_available", "step_sha256", "step_diagnosis",
            "validity_components",
        )
    } | {"source_bytes_archived": False, "step_bytes_archived": False}


def _face_list(summary: Mapping[str, Any]) -> str:
    items = []
    for row in summary.get("self_intersection_faces_by_cad", []):
        faces = ", ".join(str(value) for value in row.get("face_indices", []))
        items.append(f"- `{row['cad_id']}`: faces {faces}")
    return "\n".join(items) if items else "- none"


def snapshot(run_root: Path, report_dir: Path) -> dict[str, Any]:
    run_root, report_dir = Path(run_root).resolve(), Path(report_dir).resolve()
    source_cases = run_root / "face_wire_cases.jsonl"
    source_summary = run_root / "face_wire_summary.json"
    if not source_cases.is_file() or not source_summary.is_file():
        raise RuntimeError("face/wire diagnosis output is incomplete")
    rows = read_jsonl(source_cases)
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if len(rows) != 16 or int(summary.get("cases", -1)) != 16:
        raise RuntimeError("P0-A face/wire snapshot requires all 16 frozen cases")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise RuntimeError(f"report directory must be empty: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    compact_rows = [compact_case(row) for row in rows]
    (report_dir / "face_wire_cases.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compact_rows),
        encoding="utf-8",
    )
    archived_summary = {
        key: value for key, value in summary.items() if key != "input_path"
    } | {
        "generated_at": now(),
        "source_run_name": run_root.name,
        "source_cases_sha256": sha256_file(source_cases),
        "source_bytes_archived": False,
        "step_bytes_archived": False,
    }
    (report_dir / "face_wire_summary.json").write_text(
        json.dumps(archived_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# P0-A face/wire-local diagnosis

This Git-safe report narrows the original 16 P0-A failures before any further
assembly repair. It contains no STEP, pickle, reconstruction, model, or
upstream-source bytes. Each input source is bound only by SHA-256.

- Frozen cases: `{summary['cases']}`
- Saved STEP cases with direct face/wire diagnosis: `{summary['step_diagnosis_available']}`
- Pre-STEP cases with no pcurve/wire observation available: `{summary['step_diagnosis_unavailable']}`
- Saved STEP cases with at least one self-intersecting wire: `{summary['self_intersection_cases']}`
- Saved STEP cases with wire-order failures: `{summary['order_failure_cases']}`

## Named self-intersection faces

{_face_list(summary)}

The five no-STEP cases are listed in `face_wire_summary.json`. Their source
topology inventory is evidence for a pre-STEP repair investigation, not an
assertion about absent pcurves. The next repair candidate must address only
the listed face/wire entities or a named pre-STEP failure, and must still pass
the fixed 100-CAD zero-regression gate.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    forbidden = [
        path.relative_to(report_dir).as_posix()
        for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden artifacts entered report: {forbidden}")
    validation = {
        "valid": True,
        "cases": len(compact_rows),
        "forbidden_artifacts": forbidden,
        "source_bytes_archived": False,
        "step_bytes_archived": False,
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
    args = parser.parse_args(argv)
    print(json.dumps(snapshot(args.run_root, args.report_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
