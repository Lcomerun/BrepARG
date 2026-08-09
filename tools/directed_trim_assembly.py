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


def directed_face_loops(
    face_edge_ids: Sequence[int], edge_vertex_adj: np.ndarray
) -> list[list[tuple[int, bool]]]:
    """Return closed loops as ``(global_edge_id, reversed)`` pairs.

    ``reversed`` is true when the edge must be traversed from its second
    recorded vertex to its first.  Malformed, branching, or open topology is
    rejected instead of being silently assembled.
    """
    edge_vertex_adj = np.asarray(edge_vertex_adj, dtype=np.int64)
    edge_ids = [int(value) for value in face_edge_ids]
    if not edge_ids:
        raise ValueError("face has no incident edge")
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("face contains duplicate edge ids")
    unused = set(edge_ids)
    loops: list[list[tuple[int, bool]]] = []
    while unused:
        first = min(unused)
        start_vertex, current_vertex = map(int, edge_vertex_adj[first])
        loop = [(first, False)]
        unused.remove(first)
        while current_vertex != start_vertex:
            candidates = [
                edge_id
                for edge_id in sorted(unused)
                if current_vertex in edge_vertex_adj[edge_id]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"trim loop is open or branching at vertex {current_vertex}: {candidates}"
                )
            edge_id = candidates[0]
            vertex_a, vertex_b = map(int, edge_vertex_adj[edge_id])
            if vertex_a == current_vertex:
                reverse = False
                current_vertex = vertex_b
            elif vertex_b == current_vertex:
                reverse = True
                current_vertex = vertex_a
            else:  # guarded by candidate selection
                raise AssertionError("selected edge is not incident to current vertex")
            loop.append((edge_id, reverse))
            unused.remove(edge_id)
            if len(loop) > len(edge_ids):
                raise ValueError("trim loop traversal exceeded incident edge count")
        loops.append(loop)
    return loops


def loop_bbox_diagonal(loop: Sequence[tuple[int, bool]], edge_wcs: np.ndarray) -> float:
    """Measure one loop using its global edge ids, not face-local positions."""
    points = np.concatenate([np.asarray(edge_wcs[edge_id]).reshape(-1, 3) for edge_id, _ in loop])
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def construct_brep_directed(
    surf_wcs: np.ndarray,
    edge_wcs: np.ndarray,
    face_edge_adj: Sequence[Sequence[int]],
    edge_vertex_adj: np.ndarray,
    *,
    breparg_root: Path,
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
    from OCC.Core.TopAbs import TopAbs_SHELL
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Shell

    surf_wcs = np.asarray(surf_wcs, dtype=np.float64)
    edge_wcs = np.asarray(edge_wcs, dtype=np.float64)
    edge_vertex_adj = np.asarray(edge_vertex_adj, dtype=np.int64)
    diagnostics: dict[str, Any] = {
        "faces": len(surf_wcs), "edges": len(edge_wcs), "loop_count": 0,
        "reversed_edge_uses": 0, "multi_loop_faces": 0,
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
        values = TColgp_Array1OfPnt(1, 32)
        for point_index in range(1, 33):
            values.SetValue(point_index, gp_Pnt(*map(float, points[point_index - 1])))
        curve = None
        for tolerance in (5e-3, 8e-3, 5e-2):
            try:
                fitter = GeomAPI_PointsToBSpline(values, 0, 8, GeomAbs_C2, tolerance)
                if fitter.IsDone():
                    curve = fitter.Curve()
                    curve_tolerances.append(tolerance)
                    break
            except Exception:
                continue
        if curve is None:
            raise RuntimeError(f"curve_fit_not_done edge={edge_index}")
        builder = BRepBuilderAPI_MakeEdge(curve)
        if not builder.IsDone():
            raise RuntimeError(f"edge_builder_not_done edge={edge_index}")
        edges.append(builder.Edge())

    faces = []
    for face_index, (surface, incident) in enumerate(zip(surfaces, face_edge_adj)):
        loops = directed_face_loops(incident, edge_vertex_adj)
        diagnostics["loop_count"] += len(loops)
        diagnostics["multi_loop_faces"] += int(len(loops) > 1)
        spans = [loop_bbox_diagonal(loop, edge_wcs) for loop in loops]
        outer_index = int(np.argmax(np.asarray(spans)))
        wires = []
        for loop_index, loop in enumerate(loops):
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
    if len(shells) != 1:
        raise RuntimeError(f"sewing_produced_shell_count={len(shells)}")
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(shells[0])
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("solid_builder_not_done")
    return maker.Solid(), diagnostics
