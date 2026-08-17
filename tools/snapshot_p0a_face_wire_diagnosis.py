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


V1_SCHEMA = "p0a-face-wire-diagnosis-v1"
V2_SCHEMA = "p0a-face-wire-crossing-diagnosis-v2"
SUPPORTED_SCHEMAS = {V1_SCHEMA, V2_SCHEMA}


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


def _occurrence_table(summary: Mapping[str, Any]) -> str:
    counts = summary.get("occurrence_counts") or {}
    case_counts = summary.get("occurrence_case_counts") or {}
    kinds = (
        "adjacent", "closure", "non_adjacent", "self_only",
        "pcurve_gap", "seam", "disconnected", "unavailable",
    )
    lines = ["| Kind | Occurrences | CADs |", "| --- | ---: | ---: |"]
    lines.extend(
        f"| `{kind}` | {int(counts.get(kind, 0))} | {int(case_counts.get(kind, 0))} |"
        for kind in kinds
    )
    return "\n".join(lines)


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
    schema = summary.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise RuntimeError(f"unsupported face/wire diagnosis schema: {schema!r}")
    row_schemas = {row.get("schema") for row in rows}
    if row_schemas != {schema}:
        raise RuntimeError(
            f"case/summary schema mismatch: cases={sorted(map(str, row_schemas))}, summary={schema}"
        )
    if schema == V2_SCHEMA and (
        int(summary.get("step_diagnosis_available", -1)) != 11
        or int(summary.get("step_diagnosis_unavailable", -1)) != 5
        or int(summary.get("historical_step_saved", -1)) != 11
    ):
        raise RuntimeError("v2 face/wire snapshot requires the stage-aware 11 STEP / 5 pre-STEP population")
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
    if schema == V2_SCHEMA:
        readme = f"""# P0-A face/wire crossing diagnosis v2

This Git-safe report extends, but does not replace, the v1 face/wire report.
It binds the same 16 stage-aware P0-A baseline cases and records independent
OCC crossing modes with one-based edge positions. No STEP, pickle,
reconstruction, model, local path, or upstream-source bytes are archived.

- Frozen cases: `{summary['cases']}`
- Saved STEP cases with direct OCC diagnosis: `{summary['step_diagnosis_available']}`
- Pre-STEP cases explicitly marked unavailable: `{summary['step_diagnosis_unavailable']}`
- Edge-position basis: `{summary['edge_position_basis']}`
- Aggregate self-intersecting wires: `{summary['self_intersection_wire_count']}`
- Wires with at least one classified occurrence: `{summary['self_intersection_wires_with_classified_occurrences']}`

## Occurrence taxonomy

{_occurrence_table(summary)}

`closure` is only the cyclic `(n, 1)` pair. `adjacent` is only `(i-1, i)`
for positions 2 through n. `non_adjacent` contains only pairs whose cyclic
distance exceeds one. `self_only`, `pcurve_gap`, `seam`, and `disconnected`
are independent evidence and may coexist on one wire. The `status` field
distinguishes detected geometry from unavailable pcurves and wrapped OCC
failures; missing evidence is never interpreted as a clean check.

The five no-STEP CADs remain pre-STEP investigations. This report localizes
evidence only and does not claim that an assembly repair has been implemented.
"""
    else:
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
        "schema": schema,
        "cases": len(compact_rows),
        "step_diagnosis_available": int(summary["step_diagnosis_available"]),
        "step_diagnosis_unavailable": int(summary["step_diagnosis_unavailable"]),
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
