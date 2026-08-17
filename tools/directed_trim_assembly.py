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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        guarded_directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        sanitize_curve_points,
        validate_directed_loop,
    )
    from .solid_topology_repair import reconcile_near_vertices
except ImportError:  # direct script execution
    from assembly_repair import (
        curve_fit_attempts,
        directed_face_loops,
        guarded_directed_face_loops,
        historical_face_loops,
        loop_bbox_diagonal,
        sanitize_curve_points,
        validate_directed_loop,
    )
    from solid_topology_repair import reconcile_near_vertices


def effective_topology_summary(edge_vertex_adj: Any) -> dict[str, Any]:
    """Return path-free counts for the adjacency actually used to build edges.

    Near-vertex reconciliation can remap endpoint ids before OCC edge creation.
    The selector must compare the candidate STEP with that effective topology,
    not with the source pickle's pre-reconciliation vertex ids.
    """
    adjacency = np.asarray(edge_vertex_adj, dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape[1:] != (2,):
        raise ValueError("effective edge_vertex_adj must have shape (edge_count, 2)")
    if len(adjacency) == 0 or np.any(adjacency < 0):
        raise ValueError("effective edge_vertex_adj must contain nonnegative endpoints")
    _, counts = np.unique(adjacency.reshape(-1), return_counts=True)
    return {
        "edge_count": int(len(adjacency)),
        "vertex_count": int(len(counts)),
        "vertex_edge_incidence_counts": sorted(int(value) for value in counts),
    }


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
    solid_topology_repair: bool = False,
    pcurve_self_intersection: bool = False,
    local_intersection_topology: bool = False,
    curve_fit_rescue: bool = False,
    curve_interpolate: bool = False,
    local_pcurve_continuity: bool = False,
    periodic_pcurve_branches: bool = False,
    surface_fit_precision: bool = False,
    stage_observer: Callable[[str, Any | None, Mapping[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct one solid using directed trim loops and fail-closed OCC checks.

    ``solid_topology_repair`` is deliberately separate from ``single_solid``.
    The latter is a validation guard retained for historical profile parity;
    the former enables the narrowly scoped near-vertex reconciliation needed
    by the P0-A non-unit-solid case.
    """
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import utils as brep_utils
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.GeomAPI import (
        GeomAPI_Interpolate,
        GeomAPI_PointsToBSpline,
        GeomAPI_PointsToBSplineSurface,
    )
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Shell
    from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
    from OCC.Extend.TopologyUtils import TopologyExplorer

    surface_fit_tolerance = 1e-4 if surface_fit_precision else 5e-2
    surf_wcs = np.asarray(surf_wcs, dtype=np.float64)
    edge_wcs = np.asarray(edge_wcs, dtype=np.float64)
    edge_vertex_adj = np.asarray(edge_vertex_adj, dtype=np.int64)
    if sum(
        bool(value)
        for value in (curve_fit_fallback, curve_fit_rescue, curve_interpolate)
    ) > 1:
        raise ValueError(
            "curve_fit_fallback, curve_fit_rescue, and curve_interpolate are "
            "mutually exclusive"
        )
    diagnostics: dict[str, Any] = {
        "faces": len(surf_wcs), "edges": len(edge_wcs), "loop_count": 0,
        "reversed_edge_uses": 0, "multi_loop_faces": 0,
        "curve_fit_attempts": [], "directed_trim_loop_policies": [],
    }

    def observe(
        stage: str, shape: Any | None = None, **metadata: Any
    ) -> None:
        if stage_observer is not None:
            stage_observer(str(stage), shape, dict(metadata))
    topology_edge_vertex_adj = edge_vertex_adj
    shared_vertex_points: dict[int, np.ndarray] = {}
    if solid_topology_repair:
        (
            topology_edge_vertex_adj,
            shared_vertex_points,
            near_vertex_diagnostics,
        ) = reconcile_near_vertices(edge_wcs, edge_vertex_adj, face_edge_adj)
        diagnostics["solid_topology_repair"] = near_vertex_diagnostics
    diagnostics["effective_input_topology"] = effective_topology_summary(
        topology_edge_vertex_adj
    )

    surfaces = []
    for face_index, points in enumerate(surf_wcs):
        grid = TColgp_Array2OfPnt(1, 32, 1, 32)
        for u_index in range(1, 33):
            for v_index in range(1, 33):
                point = points[u_index - 1, v_index - 1]
                grid.SetValue(u_index, v_index, gp_Pnt(*map(float, point)))
        fitter = GeomAPI_PointsToBSplineSurface(
            grid, 3, 8, GeomAbs_C2, surface_fit_tolerance
        )
        if not fitter.IsDone():
            raise RuntimeError(f"surface_fit_not_done face={face_index}")
        surfaces.append(fitter.Surface())
    diagnostics["surface_fit"] = {
        "precision_enabled": bool(surface_fit_precision),
        "tolerance": surface_fit_tolerance,
    }
    observe(
        "surfaces_constructed",
        surface_count=len(surfaces),
        surface_fit_tolerance=surface_fit_tolerance,
    )

    shared_vertices: dict[int, Any] = {}
    if shared_vertex_points:
        vertex_builder = BRep_Builder()
        for vertex_id, point in shared_vertex_points.items():
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
            # The representative is the mean of endpoints that were already
            # no farther than 2e-4 apart.  Its tolerance must cover both
            # original curve endpoints without allowing a global snap.
            vertex_builder.UpdateVertex(vertex, 2e-4)
            shared_vertices[int(vertex_id)] = vertex

    edges = []
    curve_tolerances = []
    historical_fit_attempts = ((0, 8, 5e-3), (0, 8, 8e-3), (0, 8, 5e-2))
    for edge_index, points in enumerate(edge_wcs):
        raw_points = np.asarray(points, dtype=np.float64)
        fit_passes = []
        if curve_interpolate:
            cleaned, point_stats = sanitize_curve_points(raw_points)
            fit_passes.append(("interpolate", cleaned, point_stats, ()))
        elif curve_fit_fallback:
            cleaned, point_stats = sanitize_curve_points(raw_points)
            fit_passes.append(
                ("fallback_sanitized", cleaned, point_stats, curve_fit_attempts())
            )
        else:
            fit_passes.append(
                (
                    "historical",
                    raw_points,
                    {"input_points": len(raw_points), "retained_points": len(raw_points)},
                    historical_fit_attempts,
                )
            )
        curve = None
        for fit_mode, candidate_points, point_stats, fit_attempts in fit_passes:
            values = TColgp_Array1OfPnt(1, len(candidate_points))
            for point_index, point in enumerate(candidate_points, 1):
                values.SetValue(point_index, gp_Pnt(*map(float, point)))
            if fit_mode == "interpolate":
                from OCC.Core.TColgp import TColgp_HArray1OfPnt

                point_array = TColgp_HArray1OfPnt(1, len(candidate_points))
                for point_index, point in enumerate(candidate_points, 1):
                    point_array.SetValue(point_index, gp_Pnt(*map(float, point)))
                periodic = bool(
                    len(candidate_points) >= 3
                    and np.linalg.norm(candidate_points[0] - candidate_points[-1])
                    <= 1e-7
                )
                attempt = {
                    "edge_index": edge_index,
                    "fit_mode": fit_mode,
                    **point_stats,
                    "periodic": periodic,
                    "status": "pending",
                }
                try:
                    fitter = GeomAPI_Interpolate(point_array, periodic, 1e-7)
                    fitter.Perform()
                    if fitter.IsDone():
                        curve = fitter.Curve()
                        attempt["status"] = "succeeded"
                    else:
                        attempt["status"] = "not_done"
                except Exception as exc:
                    attempt.update(
                        status="failed", error_type=type(exc).__name__, error=str(exc)
                    )
                diagnostics["curve_fit_attempts"].append(attempt)
                if curve is not None:
                    break
                continue
            for min_degree, max_degree, tolerance in fit_attempts:
                attempt = {
                    "edge_index": edge_index, "fit_mode": fit_mode,
                    "min_degree": min_degree,
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
        if curve is None and curve_fit_rescue and not curve_fit_fallback:
            try:
                cleaned, point_stats = sanitize_curve_points(raw_points)
                values = TColgp_Array1OfPnt(1, len(cleaned))
                for point_index, point in enumerate(cleaned, 1):
                    values.SetValue(point_index, gp_Pnt(*map(float, point)))
                for min_degree, max_degree, tolerance in curve_fit_attempts():
                    attempt = {
                        "edge_index": edge_index, "fit_mode": "rescue_sanitized",
                        "min_degree": min_degree,
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
            except ValueError as exc:
                diagnostics["curve_fit_attempts"].append(
                    {
                        "edge_index": edge_index,
                        "fit_mode": "rescue_sanitized",
                        "status": "sanitize_failed",
                        "input_points": len(raw_points),
                        "retained_points": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if curve is None:
            observe(
                "curve_fit_failure",
                edge_index=edge_index,
                curve_fit_attempts=[
                    dict(attempt)
                    for attempt in diagnostics["curve_fit_attempts"]
                    if int(attempt.get("edge_index", -1)) == edge_index
                ],
            )
            raise RuntimeError(f"curve_fit_not_done edge={edge_index}")
        if shared_vertices:
            start_vertex, end_vertex = map(int, topology_edge_vertex_adj[edge_index])
            if start_vertex != end_vertex:
                builder = BRepBuilderAPI_MakeEdge(
                    curve, shared_vertices[start_vertex], shared_vertices[end_vertex]
                )
            else:
                # A closed edge cannot use one explicit vertex twice through
                # this OCC overload.  Preserve the original curve-only path.
                builder = BRepBuilderAPI_MakeEdge(curve)
        else:
            builder = BRepBuilderAPI_MakeEdge(curve)
        if not builder.IsDone():
            raise RuntimeError(f"edge_builder_not_done edge={edge_index}")
        edges.append(builder.Edge())
    observe(
        "edges_constructed",
        edge_count=len(edges),
        curve_fit_attempt_count=len(diagnostics["curve_fit_attempts"]),
    )

    faces = []
    for face_index, (surface, incident) in enumerate(zip(surfaces, face_edge_adj)):
        if directed_trim:
            loops, loop_policy = guarded_directed_face_loops(
                incident, topology_edge_vertex_adj
            )
            diagnostics["directed_trim_loop_policies"].append(
                {"face_index": face_index, **loop_policy}
            )
        else:
            loops = historical_face_loops(incident, topology_edge_vertex_adj)
        diagnostics["loop_count"] += len(loops)
        diagnostics["multi_loop_faces"] += int(len(loops) > 1)
        loop_endpoint_max_gaps: list[float] = []
        if stage_observer is not None:
            for loop in loops:
                endpoint_gaps: list[float] = []
                for loop_position, (edge_id, reverse) in enumerate(loop):
                    next_edge_id, next_reverse = loop[
                        (loop_position + 1) % len(loop)
                    ]
                    edge_points = np.asarray(edge_wcs[edge_id], dtype=np.float64)
                    next_edge_points = np.asarray(
                        edge_wcs[next_edge_id], dtype=np.float64
                    )
                    edge_end = edge_points[0] if reverse else edge_points[-1]
                    next_start = (
                        next_edge_points[-1]
                        if next_reverse
                        else next_edge_points[0]
                    )
                    endpoint_gaps.append(
                        float(np.linalg.norm(edge_end - next_start))
                    )
                loop_endpoint_max_gaps.append(max(endpoint_gaps, default=0.0))
        spans = [loop_bbox_diagonal(loop, edge_wcs) for loop in loops]
        outer_index = int(np.argmax(np.asarray(spans)))
        wires = []
        for loop_index, loop in enumerate(loops):
            if wire_continuity:
                validate_directed_loop(loop, topology_edge_vertex_adj)
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
        observe(
            "face_raw",
            face,
            face_index=face_index,
            loop_count=len(loops),
            outer_loop_index=outer_index,
            loop_3d_endpoint_max_gaps=loop_endpoint_max_gaps,
            face_3d_endpoint_max_gap=max(loop_endpoint_max_gaps, default=0.0),
        )
        brep_utils.fix_wires(face)
        brep_utils.add_pcurves_to_edges(face)
        if local_pcurve_continuity:
            try:
                from .local_wire_topology_repair import repair_face_local_pcurve
            except ImportError:  # direct script execution
                from local_wire_topology_repair import repair_face_local_pcurve

            repaired_face, repair_diagnostics = repair_face_local_pcurve(face)
            diagnostics.setdefault("local_pcurve_continuity", []).append(
                {"face_index": face_index, **repair_diagnostics}
            )
            if repair_diagnostics["accepted"]:
                face = repaired_face
            else:
                brep_utils.fix_wires(face)
                face = brep_utils.fix_face(face)
        elif periodic_pcurve_branches:
            try:
                from .local_wire_topology_repair import (
                    repair_face_periodic_pcurve_branches,
                )
            except ImportError:  # direct script execution
                from local_wire_topology_repair import (
                    repair_face_periodic_pcurve_branches,
                )

            repaired_face, repair_diagnostics = (
                repair_face_periodic_pcurve_branches(face)
            )
            diagnostics.setdefault("periodic_pcurve_branches", []).append(
                {"face_index": face_index, **repair_diagnostics}
            )
            if repair_diagnostics["accepted"]:
                face = repaired_face
            else:
                brep_utils.fix_wires(face)
                face = brep_utils.fix_face(face)
        elif local_intersection_topology:
            try:
                from .local_wire_topology_repair import repair_face_local_topology
            except ImportError:  # direct script execution
                from local_wire_topology_repair import repair_face_local_topology

            repaired_face, repair_diagnostics = repair_face_local_topology(face)
            diagnostics.setdefault("local_intersection_topology", []).append(
                {"face_index": face_index, **repair_diagnostics}
            )
            if repair_diagnostics["accepted"]:
                face = repaired_face
            else:
                brep_utils.fix_wires(face)
                face = brep_utils.fix_face(face)
        elif pcurve_self_intersection:
            face_fixer = ShapeFix_Face(face)
            face_fixer.SetPrecision(0.01)
            face_fixer.SetMaxTolerance(0.1)
            face_fixer.SetFixWireMode(True)
            face_fixer.SetFixIntersectingWiresMode(True)
            face_fixer.SetFixLoopWiresMode(True)
            wire_tool = face_fixer.FixWireTool()
            wire_tool.SetPrecision(0.01)
            wire_tool.SetMinTolerance(1e-4)
            wire_tool.SetMaxTolerance(0.1)
            wire_tool.SetClosedWireMode(True)
            wire_tool.SetFixReorderMode(True)
            wire_tool.SetFixConnectedMode(True)
            wire_tool.SetFixEdgeCurvesMode(True)
            wire_tool.SetFixAddPCurveMode(True)
            wire_tool.SetFixReversed2dMode(True)
            wire_tool.SetFixSameParameterMode(True)
            wire_tool.SetFixSelfIntersectionMode(True)
            wire_tool.SetFixSelfIntersectingEdgeMode(True)
            wire_tool.SetFixIntersectingEdgesMode(True)
            wire_tool.SetFixNonAdjacentIntersectingEdgesMode(True)
            wire_tool.SetModifyGeometryMode(True)
            wire_tool.SetModifyTopologyMode(False)
            wire_tool.SetModifyRemoveLoopMode(False)
            fixed_wires = sum(1 for _ in TopologyExplorer(face).wires())
            face_fixer.Perform()
            face_fixer.FixIntersectingWires()
            face_fixer.FixOrientation()
            face = face_fixer.Face()
            diagnostics.setdefault("pcurve_repair", {"faces": 0, "wires": 0})
            diagnostics["pcurve_repair"]["faces"] += 1
            diagnostics["pcurve_repair"]["wires"] += fixed_wires
        else:
            brep_utils.fix_wires(face)
            face = brep_utils.fix_face(face)
        observe(
            "face_final",
            face,
            face_index=face_index,
            loop_count=len(loops),
            outer_loop_index=outer_index,
            loop_3d_endpoint_max_gaps=loop_endpoint_max_gaps,
            face_3d_endpoint_max_gap=max(loop_endpoint_max_gaps, default=0.0),
        )
        faces.append(face)

    if stage_observer is not None:
        compound_builder = BRep_Builder()
        from OCC.Core.TopoDS import TopoDS_Compound

        faces_compound = TopoDS_Compound()
        compound_builder.MakeCompound(faces_compound)
        for face in faces:
            compound_builder.Add(faces_compound, face)
        observe("faces_compound", faces_compound, face_count=len(faces))

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
    observe("sewn_shape", sewn, shell_count=len(shells))
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
    observe("solid", solid, shell_count=len(shells), solid_count=solid_count)
    if single_solid and solid_count != 1:
        raise RuntimeError(f"solid_builder_produced_solid_count={solid_count}")
    return solid, diagnostics
