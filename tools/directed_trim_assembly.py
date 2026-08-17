"""BRep construction pilot with topology-directed trim loops.

This module intentionally lives outside the upstream ``BrepARG`` tree.  It
keeps the original fitting and tolerance choices while fixing two trim-loop
semantics: local face-edge positions are mapped to global edge ids for outer
loop selection, and edges are oriented along the vertex walk before insertion
into an OCC wire.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        orient_ordered_loop,
        sanitize_curve_points,
        validate_directed_loop,
    )
except ImportError:  # direct script execution
    from assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        orient_ordered_loop,
        sanitize_curve_points,
        validate_directed_loop,
    )


def construct_brep_directed(
    surf_wcs: np.ndarray,
    edge_wcs: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
    edge_vertex_adj: np.ndarray,
    *,
    breparg_root: Path,
    directed_trim: bool = True,
    curve_fit_fallback: bool = True,
    wire_continuity: bool = True,
    single_solid: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Construct one solid using directed trim loops and fail-closed OCC checks."""
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Shell

    surf_wcs = np.asarray(surf_wcs, dtype=np.float64)
    edge_wcs = np.asarray(edge_wcs, dtype=np.float64)
    edge_vertex_adj = np.asarray(edge_vertex_adj, dtype=np.int64)
    diagnostics: dict[str, Any] = {
        "faces": len(surf_wcs), "edges": len(edge_wcs), "loop_count": 0,
        "reversed_edge_uses": 0, "multi_loop_faces": 0,
        "curve_fit_attempts": [],
    }

    surfaces = []
    for face_index, points in enumerate(surf_wcs):
        grid = TColgp_Array2OfPnt(1, 32, 1, 32)
        for u_index in range(1, 33):
            for v_index in range(1, 33):
                point = points[u_index - 1, v_index - 1]
                grid.SetValue(u_index, v_index, gp_Pnt(*map(float, point)))
        fitter = GeomAPI_PointsToBSplineSurface(grid, 3, 8, GeomAbs_C2, 5e-2)
        if not fitter.IsDone():
            raise RuntimeError(f"surface_fit_not_done face={face_index}")
        surfaces.append(fitter.Surface())

    edges = []
    curve_tolerances = []
    for edge_index, points in enumerate(edge_wcs):
        if curve_fit_fallback:
            cleaned, point_stats = sanitize_curve_points(points)
            fit_attempts = curve_fit_attempts()
        else:
            cleaned = np.asarray(points, dtype=np.float64)
            point_stats = {"input_points": len(cleaned), "retained_points": len(cleaned)}
            fit_attempts = ((0, 8, 5e-3), (0, 8, 8e-3), (0, 8, 5e-2))
        values = TColgp_Array1OfPnt(1, len(cleaned))
        for point_index, point in enumerate(cleaned, 1):
            values.SetValue(point_index, gp_Pnt(*map(float, point)))
        curve = None
        for min_degree, max_degree, tolerance in fit_attempts:
            attempt = {
                "edge_index": edge_index, "min_degree": min_degree,
                "max_degree": max_degree, "tolerance": tolerance,
                **point_stats, "status": "pending",
            }
            try:
                fitter = GeomAPI_PointsToBSpline(
                    values, min_degree, max_degree, GeomAbs_C2, tolerance
                )
                if fitter.IsDone():
                    curve = fitter.Curve()
                    curve_tolerances.append(tolerance)
                    attempt["status"] = "succeeded"
                    diagnostics["curve_fit_attempts"].append(attempt)
                    break
                attempt["status"] = "not_done"
            except Exception as exc:
                attempt.update(
                    status="failed", error_type=type(exc).__name__, error=str(exc)
                )
            diagnostics["curve_fit_attempts"].append(attempt)
            if curve is not None:
                break
            else:
                continue
        if curve is None:
            raise RuntimeError(f"curve_fit_not_done edge={edge_index}")
        builder = BRepBuilderAPI_MakeEdge(curve)
        if not builder.IsDone():
            raise RuntimeError(f"edge_builder_not_done edge={edge_index}")
        edges.append(builder.Edge())

    faces = []
    for face_index, (surface, incident) in enumerate(zip(surfaces, face_edge_adj)):
        if directed_trim:
            historical = historical_face_loops(incident, edge_vertex_adj)
            try:
                loops = [orient_ordered_loop(loop, edge_vertex_adj) for loop in historical]
            except ValueError:
                loops = directed_face_loops(incident, edge_vertex_adj)
        else:
            loops = historical_face_loops(incident, edge_vertex_adj)
        diagnostics["loop_count"] += len(loops)
        diagnostics["multi_loop_faces"] += int(len(loops) > 1)
        spans = [loop_bbox_diagonal(loop, edge_wcs) for loop in loops]
        outer_index = int(np.argmax(np.asarray(spans)))
        wires = []
        for loop_index, loop in enumerate(loops):
            if wire_continuity:
                validate_directed_loop(loop, edge_vertex_adj)
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_id, reverse in loop:
                edge = edges[edge_id].Reversed() if reverse else edges[edge_id]
                diagnostics["reversed_edge_uses"] += int(reverse)
                wire_builder.Add(edge)
            if not wire_builder.IsDone():
                raise RuntimeError(
                    f"wire_builder_not_done face={face_index} loop={loop_index} "
                    f"error={wire_builder.Error()}"
                )
            wires.append(wire_builder.Wire())
        face_builder = BRepBuilderAPI_MakeFace(surface, wires[outer_index])
        for loop_index, wire in enumerate(wires):
            if loop_index != outer_index:
                face_builder.Add(wire)
        if not face_builder.IsDone():
            raise RuntimeError(f"face_builder_not_done face={face_index}")
        face = face_builder.Shape()
        brep_utils.fix_wires(face)
        brep_utils.add_pcurves_to_edges(face)
        brep_utils.fix_wires(face)
        faces.append(brep_utils.fix_face(face))

    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(1e-3)
    for face in faces:
        sewing.Add(face)
    sewing.Perform()
    sewn = sewing.SewedShape()
    shell_explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    shells = []
    while shell_explorer.More():
        shells.append(topods_Shell(shell_explorer.Current()))
        shell_explorer.Next()
    diagnostics["shell_count"] = len(shells)
    diagnostics["curve_tolerance_counts"] = {
        str(value): curve_tolerances.count(value) for value in sorted(set(curve_tolerances))
    }
    if single_solid and len(shells) != 1:
        raise RuntimeError(f"sewing_produced_shell_count={len(shells)}")
    if not shells:
        raise RuntimeError("sewing_produced_no_shell")
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(shells[0])
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("solid_builder_not_done")
    solid = maker.Solid()
    solid_explorer = TopExp_Explorer(solid, TopAbs_SOLID)
    solid_count = 0
    while solid_explorer.More():
        solid_count += 1
        solid_explorer.Next()
    diagnostics["solid_count"] = solid_count
    if single_solid and solid_count != 1:
        raise RuntimeError(f"solid_builder_produced_solid_count={solid_count}")
    return solid, diagnostics
