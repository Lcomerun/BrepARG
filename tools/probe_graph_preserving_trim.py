"""Probe topology-preserving face fixing and sewing in isolated workers.

The original surface-precision/interpolated candidate can become OCC-valid by
deleting short edges or merging source vertices.  This tool keeps the existing
schema-v2 selector gate authoritative while testing whether a minimal,
topology-disabled face repair preserves the source graph.  STEP files remain in
the local output directory; the JSON output contains hashes and compact metrics
only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .assembly_selector_geometry import (
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from .directed_trim_assembly import construct_brep_directed
    from .diagnose_assembly_face_wires import collect_wire_occurrences
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .run_assembly_repair_matrix import strict_validate_step
except ImportError:  # direct script execution
    from assembly_selector_geometry import (
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from directed_trim_assembly import construct_brep_directed
    from diagnose_assembly_face_wires import collect_wire_occurrences
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from run_assembly_repair_matrix import strict_validate_step


SCHEMA = "graph-preserving-trim-probe-v1"
WORKER_MARKER = "__GRAPH_TRIM_WORKER_RESULT__="
DEFAULT_VARIANTS = (
    ("historical_1e3", "historical", 1e-5, 1e-4, 1e-3),
    ("minimal_no_topology_1e3", "minimal_no_topology", 1e-7, 1e-6, 1e-3),
    ("minimal_no_topology_1e4", "minimal_no_topology", 1e-7, 1e-6, 1e-4),
    ("minimal_no_topology_1e5", "minimal_no_topology", 1e-7, 1e-6, 1e-5),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def compact_stage(stage: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(stage or {})
    return {
        key: values.get(key)
        for key in (
            "face_count",
            "edge_count",
            "vertex_count",
            "face_edge_occurrences",
            "face_edge_incidence_counts",
            "edge_face_incidence_counts",
            "vertex_edge_incidence_counts",
        )
        if key in values
    }


def shape_topology_summary(shape: Any) -> dict[str, Any]:
    """Return counts and incidence only; never serialize OCC geometry."""
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    def unique_shapes(kind: int) -> list[Any]:
        explorer = TopExp_Explorer(shape, kind)
        result: list[Any] = []
        while explorer.More():
            current = explorer.Current()
            if not any(existing.IsSame(current) for existing in result):
                result.append(current)
            explorer.Next()
        return result

    faces = unique_shapes(TopAbs_FACE)
    edges = unique_shapes(TopAbs_EDGE)
    vertices = unique_shapes(TopAbs_VERTEX)
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    vertex_edges = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    topexp_MapShapesAndAncestors(shape, TopAbs_VERTEX, TopAbs_EDGE, vertex_edges)

    def ancestor_counts(items: Sequence[Any], mapping: Any) -> list[int]:
        values: list[int] = []
        for item in items:
            index = int(mapping.FindIndex(item))
            values.append(int(mapping.FindFromIndex(index).Size()) if index > 0 else 0)
        return sorted(values)

    face_edge_counts: list[int] = []
    for face in faces:
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        occurrences = 0
        while explorer.More():
            occurrences += 1
            explorer.Next()
        face_edge_counts.append(occurrences)
    return {
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "face_edge_occurrences": int(sum(face_edge_counts)),
        "face_edge_incidence_counts": sorted(face_edge_counts),
        "edge_face_incidence_counts": ancestor_counts(edges, edge_faces),
        "vertex_edge_incidence_counts": ancestor_counts(vertices, vertex_edges),
    }


def _minimal_no_topology_face_fix(face: Any, *, precision: float, max_tolerance: float) -> Any:
    """Run only the non-topology ShapeFix subset used by the isolated probe."""
    from OCC.Core.ShapeFix import ShapeFix_Face

    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(float(precision))
    fixer.SetMinTolerance(float(precision))
    fixer.SetMaxTolerance(float(max_tolerance))
    fixer.SetAutoCorrectPrecisionMode(False)
    fixer.SetFixAddNaturalBoundMode(False)
    fixer.SetFixIntersectingWiresMode(False)
    fixer.SetFixLoopWiresMode(False)
    fixer.SetFixMissingSeamMode(False)
    fixer.SetFixPeriodicDegeneratedMode(False)
    fixer.SetFixSmallAreaWireMode(False)
    fixer.SetFixSplitFaceMode(False)
    fixer.SetRemoveSmallAreaFaceMode(False)
    wire_tool = fixer.FixWireTool()
    wire_tool.SetModifyTopologyMode(False)
    wire_tool.SetModifyRemoveLoopMode(False)
    wire_tool.SetFixConnectedMode(False)
    wire_tool.SetFixDegeneratedMode(False)
    wire_tool.SetFixGaps2dMode(False)
    wire_tool.SetFixGaps3dMode(False)
    wire_tool.SetFixLackingMode(False)
    wire_tool.SetFixNotchedEdgesMode(False)
    wire_tool.SetFixSmallMode(False)
    wire_tool.SetFixVertexToleranceMode(False)
    fixer.Perform()
    fixer.FixOrientation()
    return fixer.Face()


@contextmanager
def isolated_constructor_policy(
    *,
    breparg_root: Path,
    face_fix_mode: str,
    face_fix_precision: float,
    face_fix_max_tolerance: float,
    sewing_tolerance: float,
):
    """Patch the imported OCC symbols only inside one already-isolated worker.

    The production constructor deliberately has no experimental face-fix or
    sewing parameters.  This context manager keeps those probes local to this
    diagnostic process and restores both module attributes in ``finally``.
    """
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils
    import OCC.Core.BRepBuilderAPI as builder_api

    if face_fix_mode not in {"historical", "minimal_no_topology"}:
        raise ValueError(f"unsafe or unknown diagnostic face_fix_mode: {face_fix_mode!r}")
    original_fix_face = brep_utils.fix_face
    original_sewing = builder_api.BRepBuilderAPI_Sewing

    class SewingWithFixedTolerance:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = original_sewing(*args, **kwargs)

        def SetTolerance(self, _ignored: float) -> Any:
            return self._delegate.SetTolerance(float(sewing_tolerance))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._delegate, name)

    try:
        if face_fix_mode == "minimal_no_topology":
            brep_utils.fix_face = lambda face: _minimal_no_topology_face_fix(
                face,
                precision=float(face_fix_precision),
                max_tolerance=float(face_fix_max_tolerance),
            )
        builder_api.BRepBuilderAPI_Sewing = SewingWithFixedTolerance
        yield
    finally:
        builder_api.BRepBuilderAPI_Sewing = original_sewing
        brep_utils.fix_face = original_fix_face


def _read_step_shape(step_path: Path) -> Any:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError("candidate STEP reader rejected the file")
    reader.TransferRoots()
    return reader.OneShape()


def nearest_vertex_correspondence(
    step_path: Path,
    edge_wcs: np.ndarray,
    edge_vertex_adj: np.ndarray,
    *,
    coordinate_scale: float,
) -> dict[str, Any]:
    """Map source vertex ids to nearest STEP vertices without storing points."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Vertex

    source_points: dict[int, list[np.ndarray]] = {}
    for edge_index, (start_vertex, end_vertex) in enumerate(edge_vertex_adj):
        source_points.setdefault(int(start_vertex), []).append(edge_wcs[edge_index, 0])
        source_points.setdefault(int(end_vertex), []).append(edge_wcs[edge_index, -1])
    source_means = {
        vertex_id: np.mean(np.asarray(points, dtype=np.float64), axis=0)
        for vertex_id, points in source_points.items()
    }
    shape = _read_step_shape(step_path)
    candidate_points: list[np.ndarray] = []
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    candidate_vertices: list[Any] = []
    while explorer.More():
        vertex = topods_Vertex(explorer.Current())
        if not any(existing.IsSame(vertex) for existing in candidate_vertices):
            candidate_vertices.append(vertex)
            point = BRep_Tool.Pnt(vertex)
            candidate_points.append(
                np.asarray((point.X(), point.Y(), point.Z()), dtype=np.float64)
            )
        explorer.Next()
    if not candidate_points:
        raise ValueError("candidate STEP has no vertices")
    candidates = np.stack(candidate_points)
    groups: dict[int, list[dict[str, Any]]] = {}
    scale = max(float(coordinate_scale), 1e-12)
    for source_vertex_id, point in sorted(source_means.items()):
        distances = np.linalg.norm(candidates - point, axis=1)
        candidate_index = int(np.argmin(distances))
        groups.setdefault(candidate_index, []).append(
            {
                "source_vertex_id": int(source_vertex_id),
                "distance_normalized": float(distances[candidate_index] / scale),
            }
        )
    merged = [
        {
            "candidate_vertex_ordinal": int(candidate_index),
            "source_vertices": rows,
        }
        for candidate_index, rows in sorted(groups.items())
        if len(rows) > 1
    ]
    all_distances = [row["distance_normalized"] for rows in groups.values() for row in rows]
    return {
        "basis": "nearest_spatial_vertex_normalized_by_source_bbox_diagonal",
        "source_vertex_count": len(source_means),
        "candidate_vertex_count": len(candidate_points),
        "candidate_vertices_with_source_matches": len(groups),
        "merged_group_count": len(merged),
        "merged_source_vertex_count": sum(len(row["source_vertices"]) for row in merged),
        "max_source_to_candidate_distance_normalized": max(all_distances, default=0.0),
        "merged_groups": merged,
    }


def _edge_curve_samples(edge: Any, *, sample_count: int = 16) -> np.ndarray:
    from OCC.Core.BRep import BRep_Tool

    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        raise ValueError("candidate edge has no 3-D curve")
    values = []
    for parameter in np.linspace(float(first), float(last), sample_count):
        point = curve.Value(float(parameter))
        values.append((point.X(), point.Y(), point.Z()))
    return np.asarray(values, dtype=np.float64)


def _samples_to_polyline_rms(samples: np.ndarray, polyline: np.ndarray) -> float:
    starts = np.asarray(polyline[:-1], dtype=np.float64)
    deltas = np.asarray(polyline[1:] - polyline[:-1], dtype=np.float64)
    squared_lengths = np.sum(np.square(deltas), axis=1)
    valid = squared_lengths > 1e-18
    if not np.any(valid):
        return float("inf")
    starts = starts[valid]
    deltas = deltas[valid]
    squared_lengths = squared_lengths[valid]
    residuals = []
    for point in np.asarray(samples, dtype=np.float64):
        offset = point - starts
        parameters = np.clip(
            np.sum(offset * deltas, axis=1) / squared_lengths, 0.0, 1.0
        )
        nearest = starts + parameters[:, None] * deltas
        residuals.append(float(np.sqrt(np.min(np.sum(np.square(nearest - point), axis=1)))))
    return float(np.sqrt(np.mean(np.square(residuals))))


def _match_source_edge(
    edge: Any, edge_wcs: np.ndarray, *, coordinate_scale: float
) -> dict[str, Any]:
    samples = _edge_curve_samples(edge)
    residuals = np.asarray(
        [_samples_to_polyline_rms(samples, source) for source in edge_wcs],
        dtype=np.float64,
    )
    order = np.argsort(residuals)
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else first
    scale = max(float(coordinate_scale), 1e-12)
    best = float(residuals[first] / scale)
    second_best = float(residuals[second] / scale)
    return {
        "best_source_edge_index": first,
        "best_rms_normalized": best,
        "second_source_edge_index": second,
        "second_rms_normalized": second_best,
        "second_to_best_ratio": (
            float(second_best / best) if best > 1e-15 else None
        ),
    }


def self_intersection_source_mapping(
    step_path: Path,
    edge_wcs: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
    *,
    coordinate_scale: float,
) -> dict[str, Any]:
    """Identify every OCC self-intersection and map its edges to source ids."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face, topods_Wire

    shape = _read_step_shape(step_path)
    source_face_sets = [set(map(int, values)) for values in face_edge_adj]
    rows: list[dict[str, Any]] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        wire_index = 0
        while wire_explorer.More():
            fixer = ShapeFix_Wire(topods_Wire(wire_explorer.Current()), face, 0.01)
            fixer.Load(topods_Wire(wire_explorer.Current()))
            fixer.SetFace(face)
            fixer.SetPrecision(0.01)
            fixer.SetMaxTolerance(1.0)
            fixer.SetMinTolerance(1e-4)
            fixer.Perform()
            fixed_wire = fixer.Wire()
            analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            analysis.Load(fixed_wire)
            analysis.SetPrecision(0.01)
            analysis.SetSurface(BRep_Tool.Surface(face))
            aggregate = bool(analysis.CheckSelfIntersection())
            if aggregate:
                wire_data = analysis.WireData()
                edge_count = int(analysis.NbEdges())
                pcurves: set[int] = set()
                seams: set[int] = set()
                position_matches: dict[int, dict[str, Any]] = {}
                for position in range(1, edge_count + 1):
                    edge = wire_data.Edge(position)
                    try:
                        pcurve, _first, _last = BRep_Tool.CurveOnSurface(edge, face)
                        if pcurve is not None:
                            pcurves.add(position)
                    except Exception:
                        pass
                    try:
                        if BRep_Tool.IsClosed(edge, face):
                            seams.add(position)
                    except Exception:
                        pass
                    position_matches[position] = _match_source_edge(
                        edge, edge_wcs, coordinate_scale=coordinate_scale
                    )
                occurrences = collect_wire_occurrences(
                    analysis,
                    edge_count=edge_count,
                    pcurve_positions=pcurves,
                    seam_positions=seams,
                )
                detected = [row for row in occurrences if row.get("status") == "detected"]
                matched_edges = {
                    int(match["best_source_edge_index"])
                    for match in position_matches.values()
                }
                scored_faces = [
                    (
                        len(matched_edges & source_edges),
                        -len(matched_edges ^ source_edges),
                        source_face_index,
                    )
                    for source_face_index, source_edges in enumerate(source_face_sets)
                ]
                _overlap, _negative_difference, source_face_index = max(scored_faces)
                rows.append(
                    {
                        "candidate_face_ordinal": face_index,
                        "candidate_wire_ordinal": wire_index,
                        "edge_count": edge_count,
                        "source_face_index_best_match": int(source_face_index),
                        "source_face_edge_indices": sorted(
                            int(value) for value in source_face_sets[source_face_index]
                        ),
                        "source_face_overlap_count": int(_overlap),
                        "source_face_symmetric_difference_count": int(-_negative_difference),
                        "detected_occurrences": [
                            {
                                **occurrence,
                                "source_edge_matches": [
                                    {
                                        "wire_edge_position": int(position),
                                        **position_matches[int(position)],
                                    }
                                    for position in occurrence.get("edge_positions", [])
                                    if int(position) in position_matches
                                ],
                            }
                            for occurrence in detected
                        ],
                    }
                )
            wire_explorer.Next()
            wire_index += 1
        face_explorer.Next()
        face_index += 1
    return {
        "basis": "OCC_wire_position_to_nearest_source_edge_polyline",
        "self_intersection_wire_count": len(rows),
        "wires": rows,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_pickle)
    output_dir = Path(args.output_dir)
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "cad_id": args.cad_id,
        "variant": args.variant,
        "policy": {
            "face_fix_mode": args.face_fix_mode,
            "face_fix_precision": args.face_fix_precision,
            "face_fix_max_tolerance": args.face_fix_max_tolerance,
            "sewing_tolerance": args.sewing_tolerance,
        },
        "source_pickle": {
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "archived": False,
        },
        "step_archived": False,
        "status": "running",
    }
    try:
        with source.open("rb") as handle:
            parsed = pickle.load(handle)
        face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
        edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
        surf_wcs, edge_wcs = cpu_joint_optimize(
            np.asarray(parsed["surf_ncs"], dtype=np.float32),
            np.asarray(parsed["edge_ncs"], dtype=np.float32),
            np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
            np.asarray(parsed["corner_unique"], dtype=np.float32),
            edge_vertex_adj,
            face_edge_adj,
            iterations=int(args.joint_iterations),
        )
        input_signature = input_geometry_signature(
            surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj
        )
        with isolated_constructor_policy(
            breparg_root=args.breparg_root,
            face_fix_mode=args.face_fix_mode,
            face_fix_precision=float(args.face_fix_precision),
            face_fix_max_tolerance=float(args.face_fix_max_tolerance),
            sewing_tolerance=float(args.sewing_tolerance),
        ):
            solid, diagnostics = construct_brep_directed(
                surf_wcs,
                edge_wcs,
                face_edge_adj,
                edge_vertex_adj,
                breparg_root=args.breparg_root,
                directed_trim=True,
                curve_fit_fallback=False,
                curve_interpolate=True,
                wire_continuity=False,
                single_solid=False,
                surface_fit_precision=True,
            )
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Extend.DataExchange import write_step_file

        in_memory_native = bool(BRepCheck_Analyzer(solid, True).IsValid())
        step_path = output_dir / "steps" / args.variant / f"{args.cad_id}.step"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        write_step_file(solid, str(step_path))
        if not step_path.is_file() or step_path.stat().st_size <= 0:
            raise RuntimeError("STEP writer produced no non-empty file")
        validity = strict_validate_step(step_path, breparg_root=args.breparg_root)
        candidate = candidate_step_signature(
            step_path,
            input_edge_samples=sample_input_edge_points(edge_wcs),
            input_edge_polylines=edge_wcs,
            input_signature=input_signature,
            validity_components=validity["validity_components"],
        )
        gate = geometry_topology_gate(input_signature, candidate)
        topology_stages = {"solid_before_step": shape_topology_summary(solid)}
        vertex_correspondence = nearest_vertex_correspondence(
            step_path,
            edge_wcs,
            edge_vertex_adj,
            coordinate_scale=float(input_signature["coordinate_scale"]),
        )
        self_intersection_mapping = self_intersection_source_mapping(
            step_path,
            edge_wcs,
            face_edge_adj,
            coordinate_scale=float(input_signature["coordinate_scale"]),
        )
        row.update(
            status="completed",
            in_memory_native_valid=in_memory_native,
            step_bytes=step_path.stat().st_size,
            step_sha256=sha256_file(step_path),
            step_readable=bool(validity.get("validity_components", {}).get("status") == "diagnosed"),
            native_brep_valid=bool(validity["native_brep_valid"]),
            strict_brep_valid=bool(validity["strict_brep_valid"]),
            both_valid=bool(validity["both_valid"]),
            geometry_topology_gate=gate,
            selector_eligible=bool(validity["both_valid"] and gate["accepted"]),
            topology_stages={
                name: compact_stage(value) for name, value in topology_stages.items()
            },
            nearest_vertex_correspondence=vertex_correspondence,
            self_intersection_source_mapping=self_intersection_mapping,
            shell_count=diagnostics.get("shell_count"),
            solid_count=diagnostics.get("solid_count"),
            validity_components=validity.get("validity_components"),
        )
    except Exception as exc:
        row.update(
            status="error", error_type=type(exc).__name__, error=str(exc)
        )
    return row


def parse_case(value: str) -> tuple[str, Path]:
    cad_id, separator, raw_path = value.partition("=")
    if not separator or not cad_id or not raw_path:
        raise argparse.ArgumentTypeError("case must be CAD_ID=SOURCE_PICKLE")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"source pickle does not exist: {path}")
    return cad_id, path


def run_parent(args: argparse.Namespace) -> int:
    cases = [parse_case(value) for value in args.case]
    output_dir = Path(args.output_dir)
    requested_variants = set(args.variant_filter or ())
    unknown_variants = requested_variants - {item[0] for item in DEFAULT_VARIANTS}
    if unknown_variants:
        raise ValueError(f"unknown variant filters: {sorted(unknown_variants)}")
    variants = [
        item
        for item in DEFAULT_VARIANTS
        if not requested_variants or item[0] in requested_variants
    ]
    rows = []
    for cad_id, source in cases:
        for label, mode, precision, max_tolerance, sewing in variants:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--cad-id",
                cad_id,
                "--source-pickle",
                str(source.resolve()),
                "--breparg-root",
                str(Path(args.breparg_root).resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--variant",
                label,
                "--face-fix-mode",
                mode,
                "--face-fix-precision",
                str(precision),
                "--face-fix-max-tolerance",
                str(max_tolerance),
                "--sewing-tolerance",
                str(sewing),
                "--joint-iterations",
                str(args.joint_iterations),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(args.worker_timeout_seconds),
                check=False,
            )
            sentinel = next(
                (
                    line[len(WORKER_MARKER) :]
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith(WORKER_MARKER)
                ),
                None,
            )
            if sentinel is None:
                row = {
                    "schema": SCHEMA,
                    "cad_id": cad_id,
                    "variant": label,
                    "status": "worker_protocol_failure",
                    "worker_returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
            else:
                row = json.loads(sentinel)
                row["worker_returncode"] = completed.returncode
            rows.append(row)
            print(
                json.dumps(
                    {
                        "cad_id": cad_id,
                        "variant": label,
                        "status": row.get("status"),
                        "both_valid": row.get("both_valid"),
                        "gate": (row.get("geometry_topology_gate") or {}).get("accepted"),
                        "eligible": row.get("selector_eligible"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "schema": SCHEMA,
        "attempts": len(rows),
        "worker_or_protocol_failures": sum(
            row.get("status") in {"error", "worker_protocol_failure"}
            for row in rows
        ),
        "selector_eligible": [
            {"cad_id": row["cad_id"], "variant": row["variant"]}
            for row in rows
            if row.get("selector_eligible")
        ],
        "rows": rows,
    }
    atomic_json(output_dir / "graph_trim_probe.json", summary)
    return 0 if summary["worker_or_protocol_failures"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--variant-filter", action="append", default=None)
    parser.add_argument("--cad-id")
    parser.add_argument("--source-pickle", type=Path)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="manual", help=argparse.SUPPRESS)
    parser.add_argument("--face-fix-mode", default="historical", help=argparse.SUPPRESS)
    parser.add_argument("--face-fix-precision", type=float, default=1e-5, help=argparse.SUPPRESS)
    parser.add_argument("--face-fix-max-tolerance", type=float, default=1e-4, help=argparse.SUPPRESS)
    parser.add_argument("--sewing-tolerance", type=float, default=1e-3, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.cad_id or args.source_pickle is None:
            parser.error("worker requires --cad-id and --source-pickle")
        row = run_worker(args)
        print(WORKER_MARKER + json.dumps(row, sort_keys=True), flush=True)
        return 0
    if not args.case:
        parser.error("at least one --case CAD_ID=SOURCE_PICKLE is required")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
