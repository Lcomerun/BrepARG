"""Apply a conservative OCC ShapeFix pass to matched saved-invalid STEP files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_matched_invalid_rows(rows: Sequence[dict[str, Any]], *, reference_arm: str = "original") -> list[dict[str, Any]]:
    """Select saved, strict-invalid reference CADs and all matching saved arms."""
    def has_step(row: dict[str, Any]) -> bool:
        return row.get("step_saved") is True or row.get("native_brep_valid") is not None

    reference_ids = {
        str(row["cad_id"])
        for row in rows
        if str(row.get("arm")) == reference_arm
        and has_step(row)
        and row.get("strict_brep_valid") is False
    }
    return [
        row for row in rows
        if str(row.get("cad_id")) in reference_ids and has_step(row)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_step(row: dict[str, Any], *, output_root: Path, breparg_root: Path) -> dict[str, Any]:
    breparg_root = Path(breparg_root).resolve()
    if str(breparg_root) not in sys.path:
        sys.path.insert(0, str(breparg_root))
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.DataExchange import write_step_file
    from OCC.Core.ShapeFix import ShapeFix_Shape
    import utils as brep_utils

    source = Path(str(row["step_path"]))
    result: dict[str, Any] = {
        "cad_id": row.get("cad_id"), "arm": row.get("arm"),
        "source_step": str(source), "source_sha256": sha256_file(source),
        "status": "pending", "step_saved": False,
    }
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone:
            result["status"] = "step_read_failed"
            return result
        reader.TransferRoots()
        shape = reader.OneShape()
        fixer = ShapeFix_Shape(shape)
        fixer.SetPrecision(1e-3)
        fixer.SetMaxTolerance(1.0)
        fixer.Perform()
        fixed = fixer.Shape()
        native = bool(BRepCheck_Analyzer(fixed, True).IsValid())
        target = Path(output_root) / str(row.get("arm")) / f"{row['cad_id']}.step"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_step_file(fixed, str(target))
        saved = target.is_file() and target.stat().st_size > 0
        strict = bool(brep_utils.check_brep_validity(str(target))) if saved else False
        result.update({
            "status": "repaired_valid" if strict else ("repaired_invalid" if saved else "step_save_failed"),
            "step_saved": saved, "step_path": str(target) if saved else None,
            "step_sha256": sha256_file(target) if saved else None,
            "step_bytes": target.stat().st_size if saved else 0,
            "native_brep_valid": native if saved else None,
            "strict_brep_valid": strict if saved else False,
        })
    except Exception as exc:
        result.update({"status": "repair_error", "error_type": type(exc).__name__, "error": str(exc)})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.audit_manifest)
    selected = select_matched_invalid_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "repair_pilot.jsonl"
    repaired = [repair_step(row, output_root=args.output_dir, breparg_root=args.breparg_root) for row in selected]
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in repaired), encoding="utf-8")
    summary = {"selected_attempts": len(selected), "results": repaired}
    (args.output_dir / "repair_pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_attempts": len(selected), "status_counts": {s: sum(r.get("status") == s for r in repaired) for s in sorted({r.get("status") for r in repaired})}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
