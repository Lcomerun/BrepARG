"""Report project validity components for saved STEP attempts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def diagnose_step(path: Path, *, breparg_root: Path) -> dict[str, Any]:
    if str(Path(breparg_root).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(breparg_root).resolve()))
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepTools import breptools_OuterWire
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Shell, ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Edge, topods_Face, topods_Shell, topods_Wire
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from occwl.io import load_step

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        return {"status": "step_read_failed"}
    reader.TransferRoots()
    shape = reader.OneShape()
    wire_order_failures = 0
    wire_self_intersections = 0
    wire_count = 0
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())
            fixer = ShapeFix_Wire(wire, face, 0.01)
            fixer.Load(wire); fixer.SetFace(face); fixer.SetPrecision(0.01)
            fixer.SetMaxTolerance(1); fixer.SetMinTolerance(0.0001); fixer.Perform()
            analysis = ShapeAnalysis_Wire(fixer.Wire(), face, 0.01)
            analysis.Load(fixer.Wire()); analysis.SetPrecision(0.01)
            analysis.SetSurface(BRep_Tool.Surface(face))
            wire_order_failures += int(analysis.CheckOrder() != 0)
            wire_self_intersections += int(bool(analysis.CheckSelfIntersection()))
            wire_count += 1
            wire_explorer.Next()
        face_explorer.Next()
    shell_bad_edges = 0
    shell_count = 0
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        analysis = ShapeAnalysis_Shell()
        analysis.LoadShells(topods_Shell(shell_explorer.Current()))
        shell_bad_edges += int(bool(analysis.HasBadEdges()))
        shell_count += 1
        shell_explorer.Next()
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_explorer = TopExp_Explorer(free_bounds.GetOpenWires(), TopAbs_EDGE)
    free_edges = 0
    while free_explorer.More():
        free_edges += 1
        free_explorer.Next()
    try:
        solid_count = len(load_step(str(path)))
    except Exception:
        solid_count = None
    return {
        "status": "diagnosed", "native_brep_valid": bool(BRepCheck_Analyzer(shape, True).IsValid()),
        "wire_count": wire_count, "wire_order_failures": wire_order_failures,
        "wire_self_intersections": wire_self_intersections, "shell_count": shell_count,
        "shells_with_bad_edges": shell_bad_edges, "free_edges": free_edges,
        "solid_count": solid_count,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnosed = [row for row in rows if row.get("status") == "diagnosed"]
    return {
        "attempts": len(rows), "diagnosed": len(diagnosed),
        "with_free_edges": sum((row.get("free_edges") or 0) > 0 for row in diagnosed),
        "with_wire_order_failures": sum((row.get("wire_order_failures") or 0) > 0 for row in diagnosed),
        "with_wire_self_intersections": sum((row.get("wire_self_intersections") or 0) > 0 for row in diagnosed),
        "with_bad_shell_edges": sum((row.get("shells_with_bad_edges") or 0) > 0 for row in diagnosed),
        "with_nonunit_solid_count": sum(row.get("solid_count") != 1 for row in diagnosed),
        "by_arm": dict(sorted(Counter(str(row.get("arm")) for row in rows).items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-strict-invalid", action="store_true")
    args = parser.parse_args(argv)
    source = [json.loads(line) for line in args.audit_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_strict_invalid:
        source = [row for row in source if row.get("native_brep_valid") is not None and row.get("strict_brep_valid") is False]
    rows = []
    for item in source:
        result = {"cad_id": item.get("cad_id"), "arm": item.get("arm"), "step_path": item.get("step_path")}
        try:
            result.update(diagnose_step(Path(str(item["step_path"])), breparg_root=args.breparg_root))
        except Exception as exc:
            result.update(status="diagnostic_error", error_type=type(exc).__name__, error=str(exc))
        rows.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validity_components.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarize(rows)
    (args.output_dir / "validity_components_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
