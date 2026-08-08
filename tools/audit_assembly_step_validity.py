"""Audit assembly STEP files with OCC-native and project-strict validity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def summarize_validity_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    return _summarize_validity_rows(rows, include_arms=True)


def _summarize_validity_rows(
    rows: list[dict[str, Any]], *, include_arms: bool
) -> dict[str, Any]:
    step_saved = sum(row.get("native_brep_valid") is not None for row in rows)
    native_valid = sum(row.get("native_brep_valid") is True for row in rows)
    strict_valid = sum(row.get("strict_brep_valid") is True for row in rows)
    both_valid = sum(
        row.get("native_brep_valid") is True and row.get("strict_brep_valid") is True
        for row in rows
    )
    native_only = sum(
        row.get("native_brep_valid") is True and row.get("strict_brep_valid") is False
        for row in rows
    )
    strict_only = sum(
        row.get("native_brep_valid") is False and row.get("strict_brep_valid") is True
        for row in rows
    )
    neither = sum(
        row.get("native_brep_valid") is False and row.get("strict_brep_valid") is False
        for row in rows
    )
    result = {
        "attempts": len(rows),
        "step_saved": step_saved,
        "step_saved_rate": step_saved / len(rows) if rows else None,
        "native_brep_valid": native_valid,
        "native_brep_valid_rate": native_valid / len(rows) if rows else None,
        "strict_brep_valid": strict_valid,
        "strict_brep_valid_rate": strict_valid / len(rows) if rows else None,
        "both_valid": both_valid,
        "both_valid_rate": both_valid / len(rows) if rows else None,
        "native_only": native_only,
        "strict_only": strict_only,
        "neither_valid": neither,
        "native_true_strict_false": native_only,
        "native_false_strict_true": strict_only,
        "no_step": len(rows) - step_saved,
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
    }
    if include_arms:
        arms = sorted({str(row.get("arm")) for row in rows if row.get("arm") is not None})
        result["by_arm"] = {
            arm: _summarize_validity_rows(
                [row for row in rows if str(row.get("arm")) == arm], include_arms=False
            )
            for arm in arms
        }
    return result


def _resolve_step_path(
    source: Mapping[str, Any], *, manifest_path: Path, step_root: Path | None
) -> Path | None:
    raw_path = source.get("step_path")
    if raw_path:
        return Path(str(raw_path))
    if step_root is None:
        return None
    cad_id = source.get("cad_id")
    if not cad_id:
        return None
    arm = source.get("arm")
    candidates = []
    if arm:
        candidates.append(Path(step_root) / str(arm) / f"{cad_id}.step")
    candidates.append(Path(step_root) / f"{cad_id}.step")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return the deterministic primary candidate so the caller records no_step.
    return candidates[0] if candidates else None


def audit_manifest(
    manifest_path: Path, *, breparg_root: Path, step_root: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    breparg_root = Path(breparg_root).resolve()
    if str(breparg_root) not in sys.path:
        sys.path.insert(0, str(breparg_root))
    import utils as brep_utils
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    source_rows = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for source in source_rows:
        row = {
            "cad_id": source.get("cad_id"),
            "arm": source.get("arm"),
            "source_status": source.get("status"),
            "step_path": source.get("step_path"),
            "native_brep_valid": None,
            "strict_brep_valid": False,
            "status": "no_step",
        }
        step_path = _resolve_step_path(source, manifest_path=Path(manifest_path), step_root=step_root)
        row["step_path"] = str(step_path) if step_path else None
        if step_path is not None and step_path.is_file():
            try:
                reader = STEPControl_Reader()
                status = reader.ReadFile(str(step_path))
                if status != IFSelect_RetDone:
                    row["status"] = "step_read_failed"
                else:
                    reader.TransferRoots()
                    shape = reader.OneShape()
                    row["native_brep_valid"] = bool(BRepCheck_Analyzer(shape, True).IsValid())
                    row["strict_brep_valid"] = bool(brep_utils.check_brep_validity(str(step_path)))
                    row["status"] = "audited"
            except Exception as exc:
                row.update(status="audit_error", error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    return rows, summarize_validity_rows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument(
        "--step-root",
        type=Path,
        default=None,
        help="Fallback root for manifests without step_path; tries ROOT/arm/cad_id.step then ROOT/cad_id.step.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary = audit_manifest(
        args.manifest, breparg_root=args.breparg_root, step_root=args.step_root
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "step_validity_audit.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "step_validity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
