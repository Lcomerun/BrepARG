"""Stage-aware diagnosis of the frozen original-control assembly failures.

The runner intentionally leaves the upstream ``BrepARG`` checkout unchanged.
It reconstructs the historical assembly stages with explicit fail-closed stage
boundaries, scans only sewing tolerance, and independently decomposes the
saved STEP validity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from .diagnose_step_validity_components import diagnose_step
    from .run_assembly_calibration_oracle import cpu_joint_optimize
except ImportError:  # direct script execution
    from diagnose_step_validity_components import diagnose_step
    from run_assembly_calibration_oracle import cpu_joint_optimize


RUNNER_VERSION = "p0a-v1"
BASELINE_JOINT_ITERATIONS = 200
BASELINE_SEWING_TOLERANCE = 1e-3
DEFAULT_JOINT_ITERATIONS = (200, 0)
DEFAULT_SEWING_TOLERANCES = (1e-4, 1e-3, 1e-2)


class StageFailure(RuntimeError):
    """A serializable exception tied to one assembly stage and entity."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        entity_kind: str | None = None,
        entity_index: int | None = None,
        cause_type: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.entity_kind = entity_kind
        self.entity_index = entity_index
        self.cause_type = cause_type
        self.details = dict(details or {})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def select_frozen_failures(
    manifest_path: Path, expected_count: int = 16
) -> list[dict[str, Any]]:
    """Return the exact unique invalid ``original`` controls from a manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    selected = [
        dict(row)
        for row in _read_jsonl(manifest_path)
        if str(row.get("arm")) == "original" and row.get("brep_valid") is not True
    ]
    by_cad: dict[str, dict[str, Any]] = {}
    for row in selected:
        cad_id = str(row.get("cad_id") or "")
        source_path = Path(str(row.get("source_path") or ""))
        if not cad_id:
            raise ValueError("an invalid original-control row has no cad_id")
        if cad_id in by_cad:
            raise ValueError(f"duplicate invalid original-control CAD: {cad_id}")
        if not source_path.is_file():
            raise FileNotFoundError(f"source pickle is missing for {cad_id}: {source_path}")
        frozen = dict(row)
        frozen["source_path"] = str(source_path.resolve())
        frozen["source_manifest"] = str(manifest_path)
        frozen["source_manifest_sha256"] = manifest_sha256
        by_cad[cad_id] = frozen
    failures = [by_cad[cad_id] for cad_id in sorted(by_cad)]
    if len(failures) != int(expected_count):
        raise RuntimeError(
            f"expected {expected_count} frozen invalid original CADs, found {len(failures)}"
        )
    return failures


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("joint iterations must be unique non-negative integers")
    return result


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or any(not np.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("sewing tolerances must be finite and positive")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sewing tolerances must be unique")
    return result


def build_variants(
    joint_iterations: Sequence[int], sewing_tolerances: Sequence[float]
) -> list[dict[str, Any]]:
    return [
        {
            "joint_iterations": int(iterations),
            "sewing_tolerance": float(tolerance),
        }
        for iterations in joint_iterations
        for tolerance in sewing_tolerances
    ]


def attempt_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row.get("source_manifest_sha256")),
        str(row.get("cad_id")),
        int(row.get("joint_iterations")),
        format(float(row.get("sewing_tolerance")), ".12g"),
        str(row.get("runner_version")),
    )


def _raise_stage(
    stage: str,
    exc: BaseException,
    *,
    entity_kind: str | None = None,
    entity_index: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    if isinstance(exc, StageFailure):
        raise exc
    raise StageFailure(
        stage,
        str(exc) or repr(exc),
        entity_kind=entity_kind,
        entity_index=entity_index,
        cause_type=type(exc).__name__,
        details=details,
    ) from exc


def _ordered_face_loops(
    face_edge_ids: Sequence[int], edge_vertex_adj: np.ndarray, *, face_index: int
) -> tuple[list[int], list[list[int]]]:
    """Mirror the historical BrepARG face-local ordering with termination guards."""
    edge_ids = [int(value) for value in face_edge_ids]
    if not edge_ids:
        raise StageFailure(
            "topology_order", "face has no incident edge", entity_kind="face", entity_index=face_index
        )
    corner_indices = np.asarray(edge_vertex_adj, dtype=np.int64)[edge_ids]
    ordered = [0]
    seen_corners = [int(corner_indices[0, 0]), int(corner_indices[0, 1])]
    next_index = int(corner_indices[0, 1])
    loops: list[list[int]] = []
    guard = 0
    while len(ordered) < len(corner_indices):
        while True:
            next_rows = [
                index
                for index, edge in enumerate(corner_indices)
                if next_index in edge and index not in ordered
            ]
            if not next_rows:
                break
            ordered.extend(next_rows)
            difference = list(set(map(int, corner_indices[next_rows][0])) - set(seen_corners))
            if not difference:
                break
            next_index = int(difference[0])
            seen_corners.extend(map(int, corner_indices[next_rows][0]))
            guard += 1
            if guard > len(corner_indices) * 2:
                raise StageFailure(
                    "topology_order",
                    "face-edge traversal exceeded guarded iteration count",
                    entity_kind="face",
                    entity_index=face_index,
                )
        consumed = sum(len(loop) for loop in loops)
        loops.append(ordered[consumed:])
        remaining = list(set(range(len(corner_indices))) - set(ordered))
        if not remaining:
            break
        next_corner = int(remaining[0])
        ordered.append(next_corner)
        seen_corners.extend(map(int, corner_indices[next_corner]))
        next_index = int(corner_indices[next_corner, 1])
        guard += 1
        if guard > len(corner_indices) * 2:
            raise StageFailure(
                "topology_order",
                "face-loop discovery exceeded guarded iteration count",
                entity_kind="face",
                entity_index=face_index,
            )
    if not loops:
        loops = [ordered]
    if sorted(ordered) != list(range(len(edge_ids))):
        raise StageFailure(
            "topology_order",
            f"ordered face edges do not cover all incident edges: {ordered}",
            entity_kind="face",
            entity_index=face_index,
        )
    return ordered, loops


def _strict_components(path: Path, *, breparg_root: Path) -> dict[str, Any]:
    components = diagnose_step(path, breparg_root=breparg_root)
    if components.get("status") != "diagnosed":
        raise RuntimeError(f"component diagnosis returned {components.get('status')}")
    return {
        "native_brep_valid": bool(components.get("native_brep_valid")),
        "wire_count": int(components.get("wire_count") or 0),
        "wire_order_failures": int(components.get("wire_order_failures") or 0),
        "wire_self_intersections": int(components.get("wire_self_intersections") or 0),
        "shell_count": int(components.get("shell_count") or 0),
        "shells_with_bad_edges": int(components.get("shells_with_bad_edges") or 0),
        "free_edges": int(components.get("free_edges") or 0),
        "solid_count": components.get("solid_count"),
    }


def _saved_step_details(step_path: Path) -> dict[str, Any]:
    saved = Path(step_path).is_file() and Path(step_path).stat().st_size > 0
    return {
        "step_saved": saved,
        "step_path": str(Path(step_path)) if saved else None,
        "step_bytes": Path(step_path).stat().st_size if saved else 0,
        "step_sha256": sha256_file(step_path) if saved else None,
    }


def _execute_pipeline(
    parsed: Mapping[str, Any],
    *,
    joint_iterations: int,
    sewing_tolerance: float,
    step_path: Path,
    breparg_root: Path,
) -> dict[str, Any]:
    """Execute the historical assembly stages with explicit OCC boundaries."""
    root = Path(breparg_root).resolve()
    if not (root / "utils.py").is_file():
        raise StageFailure("preflight", f"BrepARG utils.py is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import utils as brep_utils
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_Sewing,
        )
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.GeomAbs import GeomAbs_C2
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
        from OCC.Extend.DataExchange import write_step_file
    except Exception as exc:
        _raise_stage("imports", exc)

    required = (
        "surf_ncs",
        "edge_ncs",
        "surf_bbox_wcs",
        "corner_unique",
        "edgeCorner_adj",
        "faceEdge_adj",
    )
    missing = [name for name in required if name not in parsed]
    if missing:
        raise StageFailure("input_validation", "missing fields: " + ", ".join(missing))
    surfaces_ncs = np.asarray(parsed["surf_ncs"], dtype=np.float32)
    edges_ncs = np.asarray(parsed["edge_ncs"], dtype=np.float32)
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    face_edge_adj = [list(map(int, row)) for row in parsed["faceEdge_adj"]]
    if not np.isfinite(surfaces_ncs).all() or not np.isfinite(edges_ncs).all():
        raise StageFailure("input_validation", "surface or edge patch contains non-finite values")

    try:
        surf_wcs, edge_wcs = cpu_joint_optimize(
            surfaces_ncs,
            edges_ncs,
            np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
            np.asarray(parsed["corner_unique"], dtype=np.float32),
            edge_vertex_adj,
            face_edge_adj,
            iterations=int(joint_iterations),
        )
    except Exception as exc:
        _raise_stage("joint_optimize", exc)
    if not np.isfinite(surf_wcs).all() or not np.isfinite(edge_wcs).all():
        raise StageFailure("joint_optimize", "joint optimization returned non-finite points")

    fitted_surfaces = []
    for face_index, points in enumerate(np.asarray(surf_wcs, dtype=np.float64)):
        try:
            grid = TColgp_Array2OfPnt(1, 32, 1, 32)
            for u_index in range(1, 33):
                for v_index in range(1, 33):
                    point = points[u_index - 1, v_index - 1]
                    grid.SetValue(u_index, v_index, gp_Pnt(*map(float, point)))
            fitter = GeomAPI_PointsToBSplineSurface(grid, 3, 8, GeomAbs_C2, 5e-2)
            if hasattr(fitter, "IsDone") and not fitter.IsDone():
                raise RuntimeError("surface fitter IsDone returned false")
            fitted_surfaces.append(fitter.Surface())
        except Exception as exc:
            _raise_stage("surface_fit", exc, entity_kind="face", entity_index=face_index)

    fitted_curves = []
    curve_attempts: list[dict[str, Any]] = []
    for edge_index, points in enumerate(np.asarray(edge_wcs, dtype=np.float64)):
        values = TColgp_Array1OfPnt(1, 32)
        try:
            for point_index in range(1, 33):
                values.SetValue(point_index, gp_Pnt(*map(float, points[point_index - 1])))
        except Exception as exc:
            _raise_stage("curve_points", exc, entity_kind="edge", entity_index=edge_index)
        curve = None
        last_exception: BaseException | None = None
        for curve_tolerance in (5e-3, 8e-3, 5e-2):
            attempt = {
                "edge_index": edge_index,
                "tolerance": curve_tolerance,
                "status": "pending",
            }
            try:
                fitter = GeomAPI_PointsToBSpline(values, 0, 8, GeomAbs_C2, curve_tolerance)
                if hasattr(fitter, "IsDone") and not fitter.IsDone():
                    raise RuntimeError("curve fitter IsDone returned false")
                curve = fitter.Curve()
                attempt["status"] = "succeeded"
                curve_attempts.append(attempt)
                break
            except Exception as exc:
                last_exception = exc
                attempt.update(
                    status="failed", error_type=type(exc).__name__, error=str(exc)
                )
                curve_attempts.append(attempt)
        if curve is None:
            _raise_stage(
                "curve_fit",
                last_exception or RuntimeError("all curve tolerances failed"),
                entity_kind="edge",
                entity_index=edge_index,
                details={"curve_fit_attempts": curve_attempts},
            )
        fitted_curves.append(curve)

    built_edges = []
    for edge_index, curve in enumerate(fitted_curves):
        try:
            builder = BRepBuilderAPI_MakeEdge(curve)
            if hasattr(builder, "IsDone") and not builder.IsDone():
                raise RuntimeError(f"edge builder error={builder.Error()}")
            built_edges.append(builder.Edge())
        except Exception as exc:
            _raise_stage("edge_build", exc, entity_kind="edge", entity_index=edge_index)

    post_faces = []
    topology_diagnostics: list[dict[str, Any]] = []
    for face_index, (surface, incident) in enumerate(zip(fitted_surfaces, face_edge_adj)):
        try:
            ordered, loops = _ordered_face_loops(
                incident, edge_vertex_adj, face_index=face_index
            )
        except Exception as exc:
            _raise_stage("topology_order", exc, entity_kind="face", entity_index=face_index)
        spans = []
        for loop in loops:
            loop_points = np.asarray(edge_wcs)[np.asarray(incident)[loop]].reshape(-1, 3)
            spans.append(float(np.linalg.norm(np.max(loop_points, axis=0) - np.min(loop_points, axis=0))))
        outer_index = int(np.argmax(np.asarray(spans)))
        wires = []
        for loop_index, loop in enumerate(loops):
            try:
                wire_builder = BRepBuilderAPI_MakeWire()
                for local_edge_index in loop:
                    wire_builder.Add(built_edges[incident[local_edge_index]])
                if hasattr(wire_builder, "IsDone") and not wire_builder.IsDone():
                    raise RuntimeError(f"wire builder error={wire_builder.Error()}")
                wires.append(wire_builder.Wire())
            except Exception as exc:
                _raise_stage(
                    "wire_build",
                    exc,
                    entity_kind=f"face[{face_index}].loop",
                    entity_index=loop_index,
                )
        try:
            face_builder = BRepBuilderAPI_MakeFace(surface, wires[outer_index])
            for loop_index, wire in enumerate(wires):
                if loop_index != outer_index:
                    face_builder.Add(wire)
            if hasattr(face_builder, "IsDone") and not face_builder.IsDone():
                raise RuntimeError("face builder IsDone returned false")
            face = face_builder.Shape()
        except Exception as exc:
            _raise_stage("face_build", exc, entity_kind="face", entity_index=face_index)
        for stage, operation in (
            ("wire_fix_pre_pcurve", lambda: brep_utils.fix_wires(face)),
            ("pcurve_add", lambda: brep_utils.add_pcurves_to_edges(face)),
            ("wire_fix_post_pcurve", lambda: brep_utils.fix_wires(face)),
        ):
            try:
                operation()
            except Exception as exc:
                _raise_stage(stage, exc, entity_kind="face", entity_index=face_index)
        try:
            post_faces.append(brep_utils.fix_face(face))
        except Exception as exc:
            _raise_stage("face_fix", exc, entity_kind="face", entity_index=face_index)
        topology_diagnostics.append(
            {
                "face_index": face_index,
                "incident_edges": len(incident),
                "ordered_local_edges": ordered,
                "loop_count": len(loops),
                "outer_loop_index": outer_index,
            }
        )

    try:
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(float(sewing_tolerance))
        for face in post_faces:
            sewing.Add(face)
        sewing.Perform()
        sewn_shape = sewing.SewedShape()
    except Exception as exc:
        _raise_stage("sewing", exc)

    try:
        maker = BRepBuilderAPI_MakeSolid()
        maker.Add(sewn_shape)
        maker.Build()
        if hasattr(maker, "IsDone") and not maker.IsDone():
            raise RuntimeError("solid builder IsDone returned false")
        solid = maker.Solid()
    except Exception as exc:
        _raise_stage("solid_build", exc)

    try:
        construction_native_valid = bool(BRepCheck_Analyzer(solid, True).IsValid())
    except Exception as exc:
        _raise_stage("construction_native_check", exc)

    try:
        step_path.parent.mkdir(parents=True, exist_ok=True)
        write_step_file(solid, str(step_path))
        if not step_path.is_file() or step_path.stat().st_size <= 0:
            raise RuntimeError("STEP writer did not create a non-empty file")
    except Exception as exc:
        _raise_stage("step_write", exc)

    try:
        components = _strict_components(step_path, breparg_root=root)
    except Exception as exc:
        _raise_stage(
            "step_reimport_or_component_check",
            exc,
            details=_saved_step_details(step_path),
        )
    try:
        strict_valid = bool(brep_utils.check_brep_validity(str(step_path)))
    except Exception as exc:
        _raise_stage(
            "strict_check",
            exc,
            details={
                **_saved_step_details(step_path),
                "native_brep_valid": bool(components["native_brep_valid"]),
                "validity_components": components,
            },
        )
    both_valid = bool(components["native_brep_valid"] and strict_valid)
    return {
        "status": "both_valid" if both_valid else "step_invalid",
        "step_saved": True,
        "step_path": str(step_path),
        "step_bytes": step_path.stat().st_size,
        "step_sha256": sha256_file(step_path),
        "construction_native_brep_valid": construction_native_valid,
        "strict_brep_valid": strict_valid,
        "both_valid": both_valid,
        "validity_components": components,
        "curve_fit_attempts": curve_attempts,
        "topology_diagnostics": topology_diagnostics,
        "stages_completed": [
            "joint_optimize",
            "surface_fit",
            "curve_fit",
            "edge_build",
            "topology_order",
            "wire_build",
            "face_build",
            "sewing",
            "solid_build",
            "construction_native_check",
            "step_write",
            "step_reimport_or_component_check",
            "strict_check",
        ],
    }


PipelineRunner = Callable[..., Mapping[str, Any]]


def run_attempt(
    parsed: Mapping[str, Any],
    case: Mapping[str, Any],
    joint_iterations: int,
    sewing_tolerance: float,
    output_dir: Path,
    breparg_root: Path,
    *,
    pipeline_runner: PipelineRunner = _execute_pipeline,
) -> dict[str, Any]:
    """Run one variant and return a fully serializable fail-closed row."""
    cad_id = str(case["cad_id"])
    variant_name = f"joint{int(joint_iterations)}_sew{float(sewing_tolerance):.0e}"
    step_path = Path(output_dir) / "steps" / variant_name / f"{cad_id}.step"
    row: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "cad_id": cad_id,
        "parent_id": case.get("parent_id"),
        "source_path": str(case.get("source_path")),
        "source_manifest": str(case.get("source_manifest")),
        "source_manifest_sha256": str(case.get("source_manifest_sha256")),
        "historical_status": case.get("status"),
        "historical_error_type": case.get("error_type"),
        "historical_error": case.get("error"),
        "joint_iterations": int(joint_iterations),
        "sewing_tolerance": float(sewing_tolerance),
        "step_saved": False,
        "strict_brep_valid": False,
        "native_brep_valid": False,
        "both_valid": False,
        "status": "running",
    }
    started = time.perf_counter()
    try:
        result = dict(
            pipeline_runner(
                parsed,
                joint_iterations=int(joint_iterations),
                sewing_tolerance=float(sewing_tolerance),
                step_path=step_path,
                breparg_root=Path(breparg_root),
            )
        )
        row.update(result)
        components = row.get("validity_components") or {}
        row["native_brep_valid"] = bool(components.get("native_brep_valid"))
        row["strict_brep_valid"] = bool(row.get("strict_brep_valid"))
        row["both_valid"] = bool(row["native_brep_valid"] and row["strict_brep_valid"])
        row["status"] = "both_valid" if row["both_valid"] else str(row.get("status") or "step_invalid")
    except StageFailure as exc:
        row.update(
            status="stage_error",
            failure_stage=exc.stage,
            failure_entity_kind=exc.entity_kind,
            failure_entity_index=exc.entity_index,
            error_type=exc.cause_type or type(exc).__name__,
            error=str(exc),
        )
        row.update(exc.details)
    except Exception as exc:
        row.update(
            status="runner_error",
            failure_stage="unscoped_runner_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def _is_baseline(row: Mapping[str, Any]) -> bool:
    return int(row.get("joint_iterations", -1)) == BASELINE_JOINT_ITERATIONS and np.isclose(
        float(row.get("sewing_tolerance", np.nan)), BASELINE_SEWING_TOLERANCE
    )


def _validity_failures(row: Mapping[str, Any]) -> list[str]:
    components = dict(row.get("validity_components") or {})
    failures = []
    if components and not components.get("native_brep_valid"):
        failures.append("native_brep_invalid")
    if (components.get("wire_order_failures") or 0) > 0:
        failures.append("wire_order_failure")
    if (components.get("wire_self_intersections") or 0) > 0:
        failures.append("wire_self_intersection")
    if (components.get("shells_with_bad_edges") or 0) > 0:
        failures.append("bad_shell_edges")
    if (components.get("free_edges") or 0) > 0:
        failures.append("free_edges")
    if components and components.get("solid_count") != 1:
        failures.append("nonunit_solid_count")
    return failures


def _outcome_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    components = dict(row.get("validity_components") or {})
    return (
        row.get("failure_stage"),
        bool(row.get("step_saved")),
        row.get("construction_native_brep_valid"),
        row.get("native_brep_valid"),
        bool(row.get("strict_brep_valid")),
        bool(row.get("both_valid")),
        components.get("wire_order_failures"),
        components.get("wire_self_intersections"),
        components.get("shells_with_bad_edges"),
        components.get("free_edges"),
        components.get("shell_count"),
        components.get("solid_count"),
    )


def classify_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assign a concrete primary cause and paired sensitivity evidence."""
    rows = [dict(row) for row in rows]
    cad_ids = {str(row.get("cad_id")) for row in rows}
    if len(cad_ids) != 1:
        raise ValueError("classify_case requires rows for exactly one CAD")
    baselines = [row for row in rows if _is_baseline(row)]
    if len(baselines) != 1:
        return {
            "cad_id": next(iter(cad_ids)),
            "primary_cause": "missing_or_duplicate_baseline",
            "attributed": False,
            "attempts": len(rows),
        }
    baseline = baselines[0]
    joint_control = next(
        (
            row
            for row in rows
            if int(row.get("joint_iterations", -1)) == 0
            and np.isclose(float(row.get("sewing_tolerance", np.nan)), BASELINE_SEWING_TOLERANCE)
        ),
        None,
    )
    tolerance_controls = [
        row
        for row in rows
        if int(row.get("joint_iterations", -1)) == BASELINE_JOINT_ITERATIONS
        and not np.isclose(float(row.get("sewing_tolerance", np.nan)), BASELINE_SEWING_TOLERANCE)
    ]
    baseline_signature = _outcome_signature(baseline)
    joint_sensitive = bool(
        joint_control
        and _outcome_signature(joint_control) != baseline_signature
    )
    tolerance_sensitive = any(
        _outcome_signature(row) != baseline_signature
        for row in tolerance_controls
    )
    strict_failures = _validity_failures(baseline)
    evidence: list[str] = []
    primary = "unknown"
    if baseline.get("failure_stage") and baseline.get("failure_stage") != "unscoped_runner_error":
        stage_side = "post_step" if baseline.get("step_saved") else "pre_step"
        primary = f"{stage_side}:{baseline['failure_stage']}"
        evidence.append(f"baseline failed at {baseline['failure_stage']}")
    elif baseline.get("step_saved") and not baseline.get("strict_brep_valid") and strict_failures:
        preferred = (
            "wire_self_intersection",
            "wire_order_failure",
            "free_edges",
            "bad_shell_edges",
            "nonunit_solid_count",
            "native_brep_invalid",
        )
        primary = next(name for name in preferred if name in strict_failures)
        evidence.append("baseline strict components: " + ", ".join(strict_failures))
    elif joint_sensitive:
        primary = "joint_optimize_sensitivity"
    elif tolerance_sensitive:
        primary = "sewing_tolerance_sensitivity"
    elif baseline.get("step_saved") and not baseline.get("strict_brep_valid"):
        primary = "strict_checker_disagreement"
        evidence.append("strict checker failed without a matching decomposed strict component")
    if joint_sensitive:
        evidence.append("joint=0 changes baseline stage or both-valid outcome")
    if tolerance_sensitive:
        evidence.append("sewing tolerance changes baseline stage or both-valid outcome")
    attributed = primary not in {
        "unknown",
        "missing_or_duplicate_baseline",
        "strict_checker_disagreement",
    }
    return {
        "cad_id": next(iter(cad_ids)),
        "attempts": len(rows),
        "primary_cause": primary,
        "attributed": attributed,
        "baseline_signature": list(baseline_signature),
        "baseline_failure_stage": baseline.get("failure_stage"),
        "baseline_strict_failures": strict_failures,
        "joint_sensitive": joint_sensitive,
        "tolerance_sensitive": tolerance_sensitive,
        "any_variant_both_valid": any(bool(row.get("both_valid")) for row in rows),
        "secondary_evidence": evidence,
    }


def summarize_cases(
    cases: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int = 16,
    expected_variants_per_case: int = 6,
    attribution_threshold: float = 0.8,
) -> dict[str, Any]:
    cases = [dict(case) for case in cases]
    attempts = [dict(row) for row in attempts]
    attributed = sum(bool(case.get("attributed")) for case in cases)
    complete_cases = sum(
        int(case.get("attempts", 0)) == int(expected_variants_per_case) for case in cases
    )
    attribution_rate = attributed / len(cases) if cases else 0.0
    matrix_complete = (
        len(cases) == int(expected_cases)
        and complete_cases == int(expected_cases)
        and len(attempts) == int(expected_cases) * int(expected_variants_per_case)
    )
    gate_passed = bool(matrix_complete and attribution_rate >= attribution_threshold)
    return {
        "runner_version": RUNNER_VERSION,
        "cases": len(cases),
        "expected_cases": int(expected_cases),
        "attempts": len(attempts),
        "expected_attempts": int(expected_cases) * int(expected_variants_per_case),
        "complete_cases": complete_cases,
        "attributed_cases": attributed,
        "attribution_rate": attribution_rate,
        "attribution_threshold": attribution_threshold,
        "matrix_complete": matrix_complete,
        "gate_passed": gate_passed,
        "primary_cause_counts": dict(
            sorted(Counter(str(case.get("primary_cause")) for case in cases).items())
        ),
        "failure_stage_counts": dict(
            sorted(
                Counter(
                    str(row.get("failure_stage"))
                    for row in attempts
                    if row.get("failure_stage")
                ).items()
            )
        ),
        "joint_sensitive_cases": sum(bool(case.get("joint_sensitive")) for case in cases),
        "tolerance_sensitive_cases": sum(
            bool(case.get("tolerance_sensitive")) for case in cases
        ),
        "cases_with_any_both_valid_variant": sum(
            bool(case.get("any_variant_both_valid")) for case in cases
        ),
        "advance_to_boundary_consistency": False,
    }


REPAIR_ACTIONS = {
    "curve_fit": "Inspect the recorded edge index and curve fallback attempts; add a bounded degenerate-curve policy or a validated lower-degree fallback.",
    "wire_build": "Validate edge endpoint continuity and orientation before wire construction; reject or reorder only the affected face loop.",
    "topology_order": "Repair face-loop extraction with directed vertex walks and explicit open/branching topology handling.",
    "solid_build": "Require exactly one sewn shell and convert the explicit shell rather than passing an unchecked compound to MakeSolid.",
    "wire_self_intersection": "Trace the offending face/wire and correct trim orientation or pcurve construction before applying broad shape repair.",
    "wire_order_failure": "Use topology-directed edge order and preserve edge orientation in the OCC wire.",
    "free_edges": "Measure unmatched sewn boundaries and tune or repair only the implicated face-edge pairs.",
    "bad_shell_edges": "Audit shell sewing and edge sharing for the implicated shell before solid conversion.",
    "nonunit_solid_count": "Require one closed shell and one solid; report compounds and empty solids as separate construction failures.",
    "native_brep_invalid": "Run BRepCheck status enumeration on the saved shape and retain the failing subshape identity.",
    "strict_checker_disagreement": "Make strict checking deterministic and report each fixed-wire component instead of returning one opaque boolean.",
    "joint_optimize_sensitivity": "Bound or disable surface translation for the affected topology and compare endpoint-to-surface residuals.",
    "sewing_tolerance_sensitivity": "Select a scale-aware sewing tolerance only after the paired scan identifies stable recovery.",
}


def write_repair_checklist(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(case.get("primary_cause")) for case in cases)
    lines = [
        "# P0-A assembly repair checklist",
        "",
        "This checklist is generated from frozen original-control failures. It does not authorize broad automatic repair.",
        "",
    ]
    for cause, count in sorted(counts.items()):
        family = cause.split(":", 1)[-1] if cause.startswith(("pre_step:", "post_step:")) else cause
        action = REPAIR_ACTIONS.get(family, "Collect a narrower stage-local reproduction before changing the assembly algorithm.")
        lines.extend([f"- [ ] `{cause}` ({count} case(s)): {action}"])
    lines.extend(
        [
            "",
            "Acceptance requires at least 80% of the frozen 16 cases to have a named cause. Sequence and AR work remain blocked.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joint-iterations", type=parse_int_csv, default=DEFAULT_JOINT_ITERATIONS
    )
    parser.add_argument(
        "--sewing-tolerances", type=parse_float_csv, default=DEFAULT_SEWING_TOLERANCES
    )
    parser.add_argument("--expected-invalid-cads", type=int, default=16)
    parser.add_argument("--max-cads", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants = build_variants(args.joint_iterations, args.sewing_tolerances)
    if len(variants) != 6:
        raise ValueError(f"P0-A requires exactly six variants, got {len(variants)}")
    frozen = select_frozen_failures(
        args.calibration_manifest, expected_count=args.expected_invalid_cads
    )
    selected = frozen[: args.max_cads] if args.max_cads is not None else frozen
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "assembly_chain_attempts.jsonl"
    existing = _read_jsonl(manifest_path)
    completed = {attempt_key(row) for row in existing}
    for case in selected:
        with Path(case["source_path"]).open("rb") as handle:
            parsed = pickle.load(handle)
        for variant in variants:
            prospective = {
                **case,
                **variant,
                "runner_version": RUNNER_VERSION,
            }
            if attempt_key(prospective) in completed:
                continue
            row = run_attempt(
                parsed,
                case,
                variant["joint_iterations"],
                variant["sewing_tolerance"],
                output_dir,
                args.breparg_root,
            )
            _append_jsonl(manifest_path, row)
            completed.add(attempt_key(row))
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "cad_id",
                            "joint_iterations",
                            "sewing_tolerance",
                            "status",
                            "failure_stage",
                            "both_valid",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    all_rows = [
        row
        for row in _read_jsonl(manifest_path)
        if str(row.get("source_manifest_sha256")) == str(frozen[0]["source_manifest_sha256"])
        and str(row.get("runner_version")) == RUNNER_VERSION
    ]
    selected_ids = {str(case["cad_id"]) for case in selected}
    requested_variants = {
        (int(variant["joint_iterations"]), format(float(variant["sewing_tolerance"]), ".12g"))
        for variant in variants
    }
    selected_rows = [
        row
        for row in all_rows
        if str(row.get("cad_id")) in selected_ids
        and (
            int(row.get("joint_iterations")),
            format(float(row.get("sewing_tolerance")), ".12g"),
        )
        in requested_variants
    ]
    cases = [
        classify_case([row for row in selected_rows if str(row.get("cad_id")) == cad_id])
        for cad_id in sorted(selected_ids)
    ]
    summary = summarize_cases(
        cases,
        selected_rows,
        expected_cases=args.expected_invalid_cads,
        expected_variants_per_case=len(variants),
    )
    summary.update(
        {
            "source_manifest": str(Path(args.calibration_manifest).resolve()),
            "source_manifest_sha256": frozen[0]["source_manifest_sha256"],
            "selected_case_ids": sorted(selected_ids),
            "partial_run": len(selected) != len(frozen),
            "joint_iterations": list(args.joint_iterations),
            "sewing_tolerances": list(args.sewing_tolerances),
        }
    )
    (output_dir / "assembly_chain_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "assembly_chain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_repair_checklist(output_dir / "repair_checklist.md", cases)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.allow_partial and summary["partial_run"]:
        return 0
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
