"""Locate the individual faces and wires behind P0-A assembly failures.

The earlier P0-A report establishes the *family* of every failure, but an OCC
``wire_self_intersection`` count alone does not say which trim loop needs
repair.  This CPU-only tool reads the frozen original-control manifest,
inspects every saved invalid STEP file face by face and wire by wire, and
records a Git-safe JSONL report.

The six original failures that never produced a STEP file are not hidden: the
tool records a source-topology inventory for them and labels their wire result
as unavailable.  No STEP, pickle, reconstructed array, or upstream source is
copied into the report; only identifiers, counts, Boolean diagnostics, and
SHA-256 values are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .diagnose_step_validity_components import diagnose_step
except ImportError:  # direct script execution
    from diagnose_step_validity_components import diagnose_step


SCHEMA = "p0a-face-wire-diagnosis-v1"
EXPECTED_INVALID_CADS = 16


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without retaining file contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def frozen_original_invalid_rows(
    manifest_path: Path, *, expected_count: int = EXPECTED_INVALID_CADS
) -> list[dict[str, Any]]:
    """Select the immutable 16 original invalid CADs in a deterministic order."""
    rows = [
        dict(row)
        for row in read_jsonl(manifest_path)
        if str(row.get("arm")) == "original" and row.get("brep_valid") is not True
    ]
    ids = [str(row.get("cad_id")) for row in rows]
    if len(rows) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique original invalid CADs, got {len(rows)} rows / {len(set(ids))} ids"
        )
    return sorted(rows, key=lambda row: str(row["cad_id"]))


def frozen_p0a_baseline_rows(
    attempts_path: Path, *, expected_count: int = EXPECTED_INVALID_CADS
) -> list[dict[str, Any]]:
    """Select the P0-A baseline variant after its stage-aware rerun.

    The historical 100-CAD calibration manifest predates P0-A and may omit a
    STEP that the stage-aware rerun successfully wrote.  This selector binds
    the local diagnosis to P0-A's explicit baseline instead: 200 joint
    optimisation iterations and sewing tolerance 1e-3.
    """
    rows = [
        dict(row)
        for row in read_jsonl(attempts_path)
        if int(row.get("joint_iterations", -1)) == 200
        and np.isclose(float(row.get("sewing_tolerance", np.nan)), 1e-3)
    ]
    ids = [str(row.get("cad_id")) for row in rows]
    if len(rows) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique P0-A baseline CADs, got {len(rows)} rows / {len(set(ids))} ids"
        )
    return sorted(rows, key=lambda row: str(row["cad_id"]))


def _edge_walk_issue(face_edges: Sequence[int], edge_vertex_adj: np.ndarray) -> str | None:
    """Return a source-topology reason when a face cannot form closed walks.

    This is only a topology inventory. It does not claim that an OCC pcurve is
    good or bad, because pcurves exist only after construction.
    """
    edge_ids = [int(edge_id) for edge_id in face_edges]
    if len(edge_ids) < 3:
        return "fewer_than_three_incident_edges"
    if len(set(edge_ids)) != len(edge_ids):
        return "duplicate_incident_edge_id"
    degree: Counter[int] = Counter()
    for edge_id in edge_ids:
        if edge_id < 0 or edge_id >= len(edge_vertex_adj):
            return "edge_id_out_of_range"
        endpoints = np.asarray(edge_vertex_adj[edge_id]).reshape(-1)
        if len(endpoints) != 2:
            return "edge_has_not_two_vertices"
        first, second = int(endpoints[0]), int(endpoints[1])
        if first != second:
            degree[first] += 1
            degree[second] += 1
    if degree and any(value != 2 for value in degree.values()):
        return "open_or_branching_vertex_incidence"
    return None


def source_topology_summary(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Extract safe source-level clues for all 16 CADs, including no-STEP cases."""
    face_edges = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    edge_ncs = np.asarray(parsed["edge_ncs"], dtype=np.float64)
    suspicious_faces = [
        {"face_index": index, "reason": reason}
        for index, values in enumerate(face_edges)
        if (reason := _edge_walk_issue(values, edge_vertex_adj)) is not None
    ]
    nonfinite_edges = [index for index, edge in enumerate(edge_ncs) if not np.isfinite(edge).all()]
    degenerate_edges: list[int] = []
    for index, edge in enumerate(edge_ncs):
        values = np.asarray(edge, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(values).all():
            continue
        span = float(np.linalg.norm(np.max(values, axis=0) - np.min(values, axis=0)))
        if span <= 1e-9:
            degenerate_edges.append(index)
    return {
        "face_count": len(face_edges),
        "edge_count": len(edge_vertex_adj),
        "max_incident_edges_per_face": max((len(values) for values in face_edges), default=0),
        "suspicious_face_count": len(suspicious_faces),
        "suspicious_faces": suspicious_faces,
        "nonfinite_edge_indices": nonfinite_edges,
        "degenerate_edge_indices": degenerate_edges,
    }


def _wire_row(*, face_index: int, wire_index: int, wire: Any, face: Any) -> dict[str, Any]:
    """Run the two established OCC wire checks and retain a stable local id."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    edge_count = 0
    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while edge_explorer.More():
        edge_count += 1
        edge_explorer.Next()
    fixer = ShapeFix_Wire(wire, face, 0.01)
    fixer.Load(wire)
    fixer.SetFace(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(1.0)
    fixer.SetMinTolerance(1e-4)
    fixer.Perform()
    analysis = ShapeAnalysis_Wire(fixer.Wire(), face, 0.01)
    analysis.Load(fixer.Wire())
    analysis.SetPrecision(0.01)
    analysis.SetSurface(BRep_Tool.Surface(face))
    order_code = int(analysis.CheckOrder())
    self_intersection = bool(analysis.CheckSelfIntersection())
    return {
        "face_index": int(face_index),
        "wire_index": int(wire_index),
        "edge_count": edge_count,
        "check_order_code": order_code,
        "order_failure": bool(order_code != 0),
        "self_intersection": self_intersection,
    }


def diagnose_step_face_wires(step_path: Path, *, breparg_root: Path) -> dict[str, Any]:
    """Diagnose every STEP face/wire with exactly the P0-A OCC wire semantics."""
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face, topods_Wire

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        return {"status": "step_read_failed", "faces": [], "wires": []}
    reader.TransferRoots()
    shape = reader.OneShape()
    wires: list[dict[str, Any]] = []
    faces: list[dict[str, Any]] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        face_wires: list[dict[str, Any]] = []
        wire_index = 0
        while wire_explorer.More():
            row = _wire_row(
                face_index=face_index,
                wire_index=wire_index,
                wire=topods_Wire(wire_explorer.Current()),
                face=face,
            )
            face_wires.append(row)
            wires.append(row)
            wire_explorer.Next()
            wire_index += 1
        faces.append(
            {
                "face_index": face_index,
                "wire_count": len(face_wires),
                "self_intersection_wire_indices": [
                    row["wire_index"] for row in face_wires if row["self_intersection"]
                ],
                "order_failure_wire_indices": [
                    row["wire_index"] for row in face_wires if row["order_failure"]
                ],
            }
        )
        face_explorer.Next()
        face_index += 1
    return {
        "status": "diagnosed",
        "faces": faces,
        "wires": wires,
        "self_intersection_faces": [
            row["face_index"] for row in faces if row["self_intersection_wire_indices"]
        ],
        "order_failure_faces": [
            row["face_index"] for row in faces if row["order_failure_wire_indices"]
        ],
    }


def build_case_row(source: Mapping[str, Any], *, breparg_root: Path) -> dict[str, Any]:
    """Produce one report row; never infer a STEP diagnosis when none exists."""
    source_path = Path(str(source["source_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with source_path.open("rb") as handle:
        parsed = pickle.load(handle)
    step_saved = bool(source.get("step_saved"))
    step_path = Path(str(source.get("step_path") or "")) if step_saved else None
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "historical_status": source.get("status"),
        "historical_step_saved": step_saved,
        "source_pickle_sha256": sha256_file(source_path),
        "source_topology": source_topology_summary(parsed),
    }
    if not step_saved or step_path is None or not step_path.is_file():
        row.update(
            step_diagnosis_available=False,
            step_diagnosis={"status": "unavailable_no_saved_step", "faces": [], "wires": []},
        )
        return row
    face_wire = diagnose_step_face_wires(step_path, breparg_root=breparg_root)
    row.update(
        step_diagnosis_available=True,
        step_sha256=sha256_file(step_path),
        step_diagnosis=face_wire,
        validity_components=diagnose_step(step_path, breparg_root=breparg_root),
    )
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize direct step observations without silently dropping no-STEP cases."""
    observed = [row for row in rows if row.get("step_diagnosis_available")]
    unavailable = [row for row in rows if not row.get("step_diagnosis_available")]
    self_intersections = [
        {"cad_id": row["cad_id"], "face_indices": row["step_diagnosis"].get("self_intersection_faces", [])}
        for row in observed if row["step_diagnosis"].get("self_intersection_faces")
    ]
    order_failures = [
        {"cad_id": row["cad_id"], "face_indices": row["step_diagnosis"].get("order_failure_faces", [])}
        for row in observed if row["step_diagnosis"].get("order_failure_faces")
    ]
    return {
        "schema": SCHEMA,
        "cases": len(rows),
        "step_diagnosis_available": len(observed),
        "step_diagnosis_unavailable": len(unavailable),
        "self_intersection_cases": len(self_intersections),
        "self_intersection_faces_by_cad": self_intersections,
        "order_failure_cases": len(order_failures),
        "order_failure_faces_by_cad": order_failures,
        "source_topology_suspicious_cases": sum(bool((row.get("source_topology") or {}).get("suspicious_faces")) for row in rows),
        "unavailable_step_cad_ids": sorted(str(row["cad_id"]) for row in unavailable),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration-manifest", type=Path)
    inputs.add_argument("--p0a-attempts", type=Path)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.p0a_attempts is not None:
        selected = frozen_p0a_baseline_rows(args.p0a_attempts)
        input_path = args.p0a_attempts
        input_kind = "p0a_stage_aware_baseline_attempts"
    else:
        selected = frozen_original_invalid_rows(args.calibration_manifest)
        input_path = args.calibration_manifest
        input_kind = "historical_calibration_manifest"
    rows = [build_case_row(row, breparg_root=args.breparg_root) for row in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "face_wire_cases.jsonl", rows)
    summary = summarize(rows)
    summary.update(
        input_kind=input_kind,
        input_path=str(Path(input_path).resolve()),
        input_sha256=sha256_file(input_path),
    )
    (args.output_dir / "face_wire_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
