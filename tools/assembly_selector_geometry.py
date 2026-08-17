"""Geometry and topology gates for failure-triggered STEP selection.

The assembly selector compares independently built fallback STEP files against
the same optimized input geometry.  A fallback is eligible only when it is
native-valid, project-strict-valid, and passes every invariant below.  The
module keeps its numerical comparison functions free of OpenCascade imports so
their fail-closed behavior is covered by ordinary unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GEOMETRY_GATE_SCHEMA = "assembly-selector-geometry-gate-v2"
PROJECTION_SAMPLES_PER_EDGE = 8
MAX_BBOX_RELATIVE_DELTA = 0.02
MAX_EDGE_LENGTH_RELATIVE_DELTA = 0.05
MAX_EDGE_SAMPLE_RMS_NORMALIZED = 0.01
MAX_EDGE_SAMPLE_MAX_NORMALIZED = 0.05
GEOMETRY_GATE_CHECK_NAMES = (
    "face_count_equal",
    "edge_count_equal",
    "vertex_count_equal",
    "face_edge_occurrences_equal",
    "face_edge_incidence_equal",
    "edge_face_incidence_equal",
    "vertex_edge_incidence_equal",
    "single_solid",
    "no_free_edges",
    "no_wire_order_failures",
    "no_wire_self_intersections",
    "bbox_within_tolerance",
    "edge_length_within_tolerance",
    "all_candidate_edges_projectable",
    "all_candidate_curve_samples_evaluated",
    "input_projection_accounting_consistent",
    "all_input_edge_samples_projected",
    "input_to_candidate_rms_within_tolerance",
    "input_to_candidate_max_within_tolerance",
    "all_candidate_edge_samples_projected",
    "candidate_projection_accounting_consistent",
    "candidate_to_input_rms_within_tolerance",
    "candidate_to_input_max_within_tolerance",
)


def geometry_gate_thresholds() -> dict[str, float]:
    return {
        "max_bbox_relative_delta": MAX_BBOX_RELATIVE_DELTA,
        "max_edge_length_relative_delta": MAX_EDGE_LENGTH_RELATIVE_DELTA,
        "max_edge_sample_rms_normalized": MAX_EDGE_SAMPLE_RMS_NORMALIZED,
        "max_edge_sample_max_normalized": MAX_EDGE_SAMPLE_MAX_NORMALIZED,
    }


def _points(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.shape[-1:] != (3,):
        raise ValueError(f"{name} must contain non-empty XYZ points")
    flattened = array.reshape(-1, 3)
    if not np.isfinite(flattened).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return flattened


def _bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("cannot measure an empty point set")
    return np.min(points, axis=0), np.max(points, axis=0)


def sample_input_edge_points(
    edge_wcs: Any, *, samples_per_edge: int = PROJECTION_SAMPLES_PER_EDGE
) -> np.ndarray:
    """Take a deterministic, bounded set of samples from every input edge."""
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least two")
    edges = np.asarray(edge_wcs, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[-1] != 3 or len(edges) == 0:
        raise ValueError("edge_wcs must have shape (edge, point, 3)")
    if edges.shape[1] < 2 or not np.isfinite(edges).all():
        raise ValueError("edge_wcs must contain at least two finite points per edge")
    indices = np.unique(
        np.linspace(
            0,
            edges.shape[1] - 1,
            min(int(samples_per_edge), int(edges.shape[1])),
            dtype=np.int64,
        )
    )
    return np.ascontiguousarray(edges[:, indices, :].reshape(-1, 3))


def input_geometry_signature(
    surf_wcs: Any,
    edge_wcs: Any,
    face_edge_adj: Sequence[Sequence[int]],
    edge_vertex_adj: Any,
    *,
    effective_vertex_count: int | None = None,
    effective_vertex_edge_incidence_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize the optimized input geometry without retaining raw CAD bytes."""
    surfaces = np.asarray(surf_wcs, dtype=np.float64)
    edges = np.asarray(edge_wcs, dtype=np.float64)
    if surfaces.ndim != 4 or surfaces.shape[-1] != 3:
        raise ValueError("surf_wcs must have shape (face, u, v, 3)")
    if edges.ndim != 3 or edges.shape[-1] != 3:
        raise ValueError("edge_wcs must have shape (edge, point, 3)")
    if len(surfaces) != len(face_edge_adj):
        raise ValueError("face_edge_adj must have one row per surface")
    if len(edges) == 0 or edges.shape[1] < 2:
        raise ValueError("edge_wcs must contain at least one sampled edge")
    adjacency = np.asarray(edge_vertex_adj, dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape != (len(edges), 2):
        raise ValueError("edge_vertex_adj must have shape (edge_count, 2)")
    if np.any(adjacency < 0):
        raise ValueError("edge_vertex_adj contains negative vertex indices")
    face_edge_incidence_counts: list[int] = []
    edge_face_incidence_counts = [0 for _ in range(len(edges))]
    vertex_edge_incidence_counts: dict[int, int] = {}
    for start_vertex, end_vertex in adjacency:
        for vertex_id in (int(start_vertex), int(end_vertex)):
            vertex_edge_incidence_counts[vertex_id] = (
                vertex_edge_incidence_counts.get(vertex_id, 0) + 1
            )
    for face_index, edge_ids in enumerate(face_edge_adj):
        face_edge_incidence_counts.append(len(edge_ids))
        unique_face_edges: set[int] = set()
        for edge_id in edge_ids:
            if not isinstance(edge_id, (int, np.integer)) or not 0 <= int(edge_id) < len(edges):
                raise ValueError(
                    f"face_edge_adj contains invalid edge index at face {face_index}"
                )
            unique_face_edges.add(int(edge_id))
        for edge_id in unique_face_edges:
            edge_face_incidence_counts[edge_id] += 1
    surface_points = _points(surfaces, name="surf_wcs")
    edge_points = _points(edges, name="edge_wcs")
    minimum, maximum = _bbox(np.concatenate((surface_points, edge_points), axis=0))
    diagonal = float(np.linalg.norm(maximum - minimum))
    coordinate_scale = max(diagonal, float(np.max(np.abs(maximum - minimum))), 1e-12)
    segment_lengths = np.linalg.norm(np.diff(edges, axis=1), axis=2)
    return {
        "face_count": int(len(surfaces)),
        "edge_count": int(len(edges)),
        "vertex_count": int(
            effective_vertex_count
            if effective_vertex_count is not None
            else len(np.unique(adjacency.reshape(-1)))
        ),
        "face_edge_occurrences": int(sum(len(row) for row in face_edge_adj)),
        # The entity identifiers in a source pickle and a regenerated STEP file
        # need not have the same order.  Sorted incidence-degree multisets retain
        # topology information without assuming a fragile identity correspondence.
        "face_edge_incidence_counts": sorted(face_edge_incidence_counts),
        "edge_face_incidence_counts": sorted(edge_face_incidence_counts),
        "vertex_edge_incidence_counts": sorted(
            int(value)
            for value in (
                effective_vertex_edge_incidence_counts
                if effective_vertex_edge_incidence_counts is not None
                else vertex_edge_incidence_counts.values()
            )
        ),
        "bbox": [float(value) for value in np.concatenate((minimum, maximum))],
        "coordinate_scale": float(coordinate_scale),
        "edge_polyline_length": float(np.sum(segment_lengths)),
        "edge_sample_count": int(edges.shape[0] * edges.shape[1]),
        "projection_sample_count": int(
            len(sample_input_edge_points(edges))
        ),
    }


def _shape_counts(shape: Any) -> dict[str, int]:
    from OCC.Extend.TopologyUtils import TopologyExplorer

    topology = TopologyExplorer(shape, ignore_orientation=True)
    faces = list(topology.faces())
    edges = list(topology.edges())
    vertices = list(topology.vertices())
    face_edge_incidence_counts: list[int] = []
    edge_face_incidence_counts = [0 for _ in edges]
    vertex_edge_incidence_counts = [0 for _ in vertices]
    for face in faces:
        # Keep occurrences here: a seam repeated inside one wire must not be
        # hidden by orientation-insensitive global topology iteration.
        face_edges = list(TopologyExplorer(face, ignore_orientation=False).edges())
        face_edge_incidence_counts.append(len(face_edges))
        face_edge_ids: set[int] = set()
        for face_edge in face_edges:
            matched = next(
                (
                    index
                    for index, global_edge in enumerate(edges)
                    if face_edge.IsSame(global_edge)
                ),
                None,
            )
            if matched is None:
                raise ValueError("candidate STEP face edge is absent from global topology")
            face_edge_ids.add(matched)
        for edge_id in face_edge_ids:
            edge_face_incidence_counts[edge_id] += 1
    for edge in edges:
        # Count endpoint occurrences rather than unique endpoint identities: a
        # closed edge contributes degree two to its one shared vertex.
        endpoints = list(TopologyExplorer(edge, ignore_orientation=False).vertices())
        if len(endpoints) != 2:
            raise ValueError("candidate STEP edge does not expose two endpoints")
        for endpoint in endpoints:
            matched = next(
                (
                    index
                    for index, global_vertex in enumerate(vertices)
                    if endpoint.IsSame(global_vertex)
                ),
                None,
            )
            if matched is None:
                raise ValueError("candidate STEP endpoint is absent from global topology")
            vertex_edge_incidence_counts[matched] += 1
    return {
        "face_count": len(faces),
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "wire_count": sum(1 for _ in topology.wires()),
        "solid_count": sum(1 for _ in topology.solids()),
        "face_edge_occurrences": int(sum(face_edge_incidence_counts)),
        "face_edge_incidence_counts": sorted(face_edge_incidence_counts),
        "edge_face_incidence_counts": sorted(edge_face_incidence_counts),
        "vertex_edge_incidence_counts": sorted(vertex_edge_incidence_counts),
    }


def _shape_bbox(shape: Any) -> list[float]:
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box

    box = Bnd_Box()
    # The ordinary Add() path may use a loose BSpline control-polygon bound;
    # on valid interpolated trims that overestimated one axis by nearly six
    # percent.  AddOptimal measures the represented trimmed shape and keeps
    # the bbox gate about geometry rather than bounding-algorithm slack.
    brepbndlib.AddOptimal(shape, box, True, False)
    if box.IsVoid():
        raise ValueError("candidate STEP has a void bounding box")
    values = [float(value) for value in box.Get()]
    if not np.isfinite(values).all():
        raise ValueError("candidate STEP bounding box is non-finite")
    return values


def _candidate_curves_and_length(
    shape: Any,
) -> tuple[list[tuple[Any, float, float]], float, dict[str, int]]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Extend.TopologyUtils import TopologyExplorer

    curves: list[tuple[Any, float, float]] = []
    total_length = 0.0
    unprojectable_edge_count = 0
    for edge in TopologyExplorer(shape).edges():
        properties = GProp_GProps()
        brepgprop.LinearProperties(edge, properties)
        length = abs(float(properties.Mass()))
        if not np.isfinite(length):
            raise ValueError("candidate STEP has non-finite edge length")
        total_length += length
        curve, first, last = BRep_Tool.Curve(edge)
        if (
            curve is None
            or (hasattr(curve, "IsNull") and curve.IsNull())
            or not np.isfinite((float(first), float(last))).all()
        ):
            unprojectable_edge_count += 1
            continue
        curves.append((curve, float(first), float(last)))
    if not curves:
        raise ValueError("candidate STEP has no projectable edge curves")
    return curves, total_length, {
        "projectable_edge_count": int(len(curves)),
        "unprojectable_edge_count": int(unprojectable_edge_count),
    }


def _input_to_candidate_projection_metrics(
    samples: np.ndarray, curves: Sequence[tuple[Any, float, float]], *, coordinate_scale: float
) -> dict[str, Any]:
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.gp import gp_Pnt

    residuals: list[float] = []
    failures = 0
    for point in np.asarray(samples, dtype=np.float64):
        best = float("inf")
        query = gp_Pnt(*map(float, point))
        for curve, first, last in curves:
            try:
                projection = GeomAPI_ProjectPointOnCurve(query, curve, first, last)
                if projection.NbPoints() <= 0:
                    continue
                distance = float(projection.LowerDistance())
            except Exception:
                continue
            if np.isfinite(distance):
                best = min(best, distance)
        if np.isfinite(best):
            residuals.append(best)
        else:
            failures += 1
    if not residuals:
        raise ValueError("no input samples could be projected to candidate edges")
    values = np.asarray(residuals, dtype=np.float64)
    scale = max(float(coordinate_scale), 1e-12)
    return {
        "sample_count": int(len(samples)),
        "projected_sample_count": int(len(values)),
        "projection_failure_count": int(failures),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "rms_normalized": float(np.sqrt(np.mean(np.square(values))) / scale),
        "max_normalized": float(np.max(values) / scale),
    }


def _candidate_curve_samples(
    curves: Sequence[tuple[Any, float, float]], *, samples_per_curve: int
) -> tuple[np.ndarray, dict[str, int]]:
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least two")
    points: list[np.ndarray] = []
    sampling_failures = 0
    for curve, first, last in curves:
        parameters = np.linspace(
            min(float(first), float(last)),
            max(float(first), float(last)),
            int(samples_per_curve),
            dtype=np.float64,
        )
        for parameter in parameters:
            try:
                point = curve.Value(float(parameter))
                values = np.asarray(
                    (float(point.X()), float(point.Y()), float(point.Z())),
                    dtype=np.float64,
                )
            except Exception:
                sampling_failures += 1
                continue
            if np.isfinite(values).all():
                points.append(values)
            else:
                sampling_failures += 1
    if not points:
        raise ValueError("candidate STEP curves could not be sampled")
    samples = np.ascontiguousarray(np.stack(points, axis=0))
    return samples, {
        "requested_sample_count": int(len(curves) * samples_per_curve),
        "successful_sample_count": int(len(samples)),
        "sampling_failure_count": int(sampling_failures),
    }


def _candidate_to_input_projection_metrics(
    candidate_samples: np.ndarray,
    input_edge_polylines: np.ndarray,
    *,
    coordinate_scale: float,
) -> dict[str, Any]:
    """Measure every candidate curve sample against all input polyline segments."""
    edges = np.asarray(input_edge_polylines, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[-1] != 3 or edges.shape[1] < 2:
        raise ValueError("input edge polylines are malformed")
    starts = np.ascontiguousarray(edges[:, :-1, :].reshape(-1, 3))
    ends = np.ascontiguousarray(edges[:, 1:, :].reshape(-1, 3))
    deltas = ends - starts
    squared_lengths = np.sum(np.square(deltas), axis=1)
    valid_segments = squared_lengths > 1e-18
    if not np.any(valid_segments):
        raise ValueError("input edge polylines have no non-degenerate segments")
    starts, deltas, squared_lengths = (
        starts[valid_segments],
        deltas[valid_segments],
        squared_lengths[valid_segments],
    )
    residuals: list[float] = []
    for point in np.asarray(candidate_samples, dtype=np.float64):
        offset = point - starts
        parameters = np.clip(
            np.sum(offset * deltas, axis=1) / squared_lengths,
            0.0,
            1.0,
        )
        nearest = starts + parameters[:, None] * deltas
        residuals.append(float(np.sqrt(np.min(np.sum(np.square(nearest - point), axis=1)))))
    values = np.asarray(residuals, dtype=np.float64)
    scale = max(float(coordinate_scale), 1e-12)
    return {
        "sample_count": int(len(values)),
        "projected_sample_count": int(len(values)),
        "projection_failure_count": 0,
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "rms_normalized": float(np.sqrt(np.mean(np.square(values))) / scale),
        "max_normalized": float(np.max(values) / scale),
    }


def candidate_step_signature(
    step_path: Path,
    *,
    input_edge_samples: np.ndarray,
    input_edge_polylines: np.ndarray,
    input_signature: Mapping[str, Any],
    validity_components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a candidate STEP without returning its local path or CAD bytes."""
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(Path(step_path))) != IFSelect_RetDone:
        raise ValueError("candidate STEP reader rejected the file")
    reader.TransferRoots()
    shape = reader.OneShape()
    counts = _shape_counts(shape)
    curves, edge_length, curve_counts = _candidate_curves_and_length(shape)
    components = dict(validity_components or {})
    input_to_candidate = _input_to_candidate_projection_metrics(
        input_edge_samples,
        curves,
        coordinate_scale=float(input_signature["coordinate_scale"]),
    )
    candidate_samples, candidate_curve_sampling = _candidate_curve_samples(
        curves, samples_per_curve=PROJECTION_SAMPLES_PER_EDGE
    )
    candidate_to_input = _candidate_to_input_projection_metrics(
        candidate_samples,
        input_edge_polylines,
        coordinate_scale=float(input_signature["coordinate_scale"]),
    )
    return {
        **counts,
        **curve_counts,
        "bbox": _shape_bbox(shape),
        "edge_length": float(edge_length),
        "free_edges": components.get("free_edges"),
        "wire_order_failures": components.get("wire_order_failures"),
        "wire_self_intersections": components.get("wire_self_intersections"),
        "input_to_candidate_projection": input_to_candidate,
        "candidate_to_input_projection": candidate_to_input,
        "candidate_curve_sampling": candidate_curve_sampling,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(
        np.isfinite(float(value))
    )


def _nonnegative_integer_sequence(
    value: Any, *, name: str, expected_length: int
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer sequence")
    if len(value) != expected_length:
        raise ValueError(f"{name} length does not match topology count")
    result: list[int] = []
    for item in value:
        if not _finite_number(item) or int(item) != float(item) or int(item) < 0:
            raise ValueError(f"{name} contains an invalid count")
        result.append(int(item))
    return tuple(result)


def geometry_topology_gate(
    input_signature: Mapping[str, Any], candidate_signature: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a serializable, fail-closed decision for one fallback candidate."""
    required_input = (
        "face_count",
        "edge_count",
        "vertex_count",
        "face_edge_occurrences",
        "face_edge_incidence_counts",
        "edge_face_incidence_counts",
        "vertex_edge_incidence_counts",
        "coordinate_scale",
        "bbox",
        "edge_polyline_length",
        "projection_sample_count",
    )
    required_candidate = (
        "face_count",
        "edge_count",
        "vertex_count",
        "face_edge_occurrences",
        "face_edge_incidence_counts",
        "edge_face_incidence_counts",
        "vertex_edge_incidence_counts",
        "solid_count",
        "free_edges",
        "wire_order_failures",
        "wire_self_intersections",
        "bbox",
        "edge_length",
        "projectable_edge_count",
        "unprojectable_edge_count",
        "input_to_candidate_projection",
        "candidate_to_input_projection",
        "candidate_curve_sampling",
    )
    missing = [
        *(f"input:{key}" for key in required_input if key not in input_signature),
        *(
            f"candidate:{key}"
            for key in required_candidate
            if key not in candidate_signature
        ),
    ]
    thresholds = geometry_gate_thresholds()
    if missing:
        return {
            "schema": GEOMETRY_GATE_SCHEMA,
            "accepted": False,
            "checks": {"signature_complete": False},
            "rejection_reasons": [f"missing_signature_field:{key}" for key in missing],
            "thresholds": thresholds,
        }
    try:
        input_bbox = [float(value) for value in input_signature["bbox"]]
        candidate_bbox = [float(value) for value in candidate_signature["bbox"]]
        if len(input_bbox) != 6 or len(candidate_bbox) != 6:
            raise ValueError("bbox must contain six values")
        scale = float(input_signature["coordinate_scale"])
        edge_length = float(input_signature["edge_polyline_length"])
        candidate_length = float(candidate_signature["edge_length"])
        input_face_count = int(input_signature["face_count"])
        input_edge_count = int(input_signature["edge_count"])
        input_vertex_count = int(input_signature["vertex_count"])
        input_face_edge_occurrences = int(input_signature["face_edge_occurrences"])
        candidate_face_count = int(candidate_signature["face_count"])
        candidate_edge_count = int(candidate_signature["edge_count"])
        candidate_vertex_count = int(candidate_signature["vertex_count"])
        candidate_face_edge_occurrences = int(
            candidate_signature["face_edge_occurrences"]
        )
        solid_count = int(candidate_signature["solid_count"])
        free_edges = int(candidate_signature["free_edges"])
        wire_order_failures = int(candidate_signature["wire_order_failures"])
        wire_self_intersections = int(candidate_signature["wire_self_intersections"])
        projectable_edge_count = int(candidate_signature["projectable_edge_count"])
        unprojectable_edge_count = int(candidate_signature["unprojectable_edge_count"])
        input_face_edge_incidence = _nonnegative_integer_sequence(
            input_signature["face_edge_incidence_counts"],
            name="input face_edge_incidence_counts",
            expected_length=input_face_count,
        )
        candidate_face_edge_incidence = _nonnegative_integer_sequence(
            candidate_signature["face_edge_incidence_counts"],
            name="candidate face_edge_incidence_counts",
            expected_length=candidate_face_count,
        )
        input_edge_face_incidence = _nonnegative_integer_sequence(
            input_signature["edge_face_incidence_counts"],
            name="input edge_face_incidence_counts",
            expected_length=input_edge_count,
        )
        candidate_edge_face_incidence = _nonnegative_integer_sequence(
            candidate_signature["edge_face_incidence_counts"],
            name="candidate edge_face_incidence_counts",
            expected_length=candidate_edge_count,
        )
        input_vertex_edge_incidence = _nonnegative_integer_sequence(
            input_signature["vertex_edge_incidence_counts"],
            name="input vertex_edge_incidence_counts",
            expected_length=input_vertex_count,
        )
        candidate_vertex_edge_incidence = _nonnegative_integer_sequence(
            candidate_signature["vertex_edge_incidence_counts"],
            name="candidate vertex_edge_incidence_counts",
            expected_length=candidate_vertex_count,
        )
        input_projection_sample_count = int(input_signature["projection_sample_count"])
        input_to_candidate = dict(candidate_signature["input_to_candidate_projection"])
        candidate_to_input = dict(candidate_signature["candidate_to_input_projection"])
        candidate_curve_sampling = dict(candidate_signature["candidate_curve_sampling"])
        input_to_candidate_sample_count = int(input_to_candidate["sample_count"])
        input_to_candidate_projected_count = int(
            input_to_candidate["projected_sample_count"]
        )
        input_to_candidate_rms = float(input_to_candidate["rms_normalized"])
        input_to_candidate_max = float(input_to_candidate["max_normalized"])
        input_to_candidate_failures = int(
            input_to_candidate["projection_failure_count"]
        )
        candidate_to_input_rms = float(candidate_to_input["rms_normalized"])
        candidate_to_input_max = float(candidate_to_input["max_normalized"])
        candidate_to_input_failures = int(
            candidate_to_input["projection_failure_count"]
        )
        candidate_to_input_sample_count = int(candidate_to_input["sample_count"])
        candidate_to_input_projected_count = int(
            candidate_to_input["projected_sample_count"]
        )
        candidate_curve_requested_count = int(
            candidate_curve_sampling["requested_sample_count"]
        )
        candidate_curve_successful_count = int(
            candidate_curve_sampling["successful_sample_count"]
        )
        candidate_curve_sampling_failures = int(
            candidate_curve_sampling["sampling_failure_count"]
        )
        numeric_values = [
            *input_bbox,
            *candidate_bbox,
            scale,
            edge_length,
            candidate_length,
            input_to_candidate_rms,
            input_to_candidate_max,
            candidate_to_input_rms,
            candidate_to_input_max,
        ]
        if not all(_finite_number(value) for value in numeric_values) or scale <= 0.0:
            raise ValueError("signature contains non-finite geometry metrics")
        if any(
            value < 0
            for value in (
                input_face_count,
                input_edge_count,
                input_vertex_count,
                input_face_edge_occurrences,
                candidate_face_count,
                candidate_edge_count,
                candidate_vertex_count,
                candidate_face_edge_occurrences,
                solid_count,
                free_edges,
                wire_order_failures,
                wire_self_intersections,
                input_to_candidate_failures,
                candidate_to_input_failures,
                projectable_edge_count,
                unprojectable_edge_count,
                input_projection_sample_count,
                input_to_candidate_sample_count,
                input_to_candidate_projected_count,
                candidate_to_input_sample_count,
                candidate_to_input_projected_count,
                candidate_curve_requested_count,
                candidate_curve_successful_count,
                candidate_curve_sampling_failures,
            )
        ):
            raise ValueError("signature contains negative counts")
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": GEOMETRY_GATE_SCHEMA,
            "accepted": False,
            "checks": {"signature_well_formed": False},
            "rejection_reasons": [f"malformed_signature:{type(exc).__name__}"],
            "thresholds": thresholds,
        }
    bbox_relative_delta = max(
        abs(first - second) for first, second in zip(input_bbox, candidate_bbox)
    ) / max(scale, 1e-12)
    edge_length_relative_delta = abs(candidate_length - edge_length) / max(
        abs(edge_length), 1e-12
    )
    checks = {
        "face_count_equal": candidate_face_count == input_face_count,
        "edge_count_equal": candidate_edge_count == input_edge_count,
        "vertex_count_equal": candidate_vertex_count == input_vertex_count,
        "face_edge_occurrences_equal": (
            candidate_face_edge_occurrences == input_face_edge_occurrences
        ),
        "face_edge_incidence_equal": (
            candidate_face_edge_incidence == input_face_edge_incidence
        ),
        "edge_face_incidence_equal": (
            candidate_edge_face_incidence == input_edge_face_incidence
        ),
        "vertex_edge_incidence_equal": (
            candidate_vertex_edge_incidence == input_vertex_edge_incidence
        ),
        "single_solid": solid_count == 1,
        "no_free_edges": free_edges == 0,
        "no_wire_order_failures": wire_order_failures == 0,
        "no_wire_self_intersections": wire_self_intersections == 0,
        "bbox_within_tolerance": bbox_relative_delta <= MAX_BBOX_RELATIVE_DELTA,
        "edge_length_within_tolerance": (
            edge_length_relative_delta <= MAX_EDGE_LENGTH_RELATIVE_DELTA
        ),
        "all_candidate_edges_projectable": (
            projectable_edge_count == candidate_edge_count
            and unprojectable_edge_count == 0
        ),
        "all_candidate_curve_samples_evaluated": (
            candidate_curve_requested_count
            == candidate_curve_successful_count + candidate_curve_sampling_failures
            and candidate_curve_successful_count == candidate_curve_requested_count
            and candidate_curve_sampling_failures == 0
        ),
        "input_projection_accounting_consistent": (
            input_to_candidate_sample_count == input_projection_sample_count
            and input_to_candidate_projected_count + input_to_candidate_failures
            == input_projection_sample_count
        ),
        "all_input_edge_samples_projected": input_to_candidate_failures == 0,
        "input_to_candidate_rms_within_tolerance": (
            input_to_candidate_rms <= MAX_EDGE_SAMPLE_RMS_NORMALIZED
        ),
        "input_to_candidate_max_within_tolerance": (
            input_to_candidate_max <= MAX_EDGE_SAMPLE_MAX_NORMALIZED
        ),
        "all_candidate_edge_samples_projected": candidate_to_input_failures == 0,
        "candidate_projection_accounting_consistent": (
            candidate_to_input_sample_count == candidate_curve_successful_count
            and candidate_to_input_projected_count + candidate_to_input_failures
            == candidate_curve_successful_count
        ),
        "candidate_to_input_rms_within_tolerance": (
            candidate_to_input_rms <= MAX_EDGE_SAMPLE_RMS_NORMALIZED
        ),
        "candidate_to_input_max_within_tolerance": (
            candidate_to_input_max <= MAX_EDGE_SAMPLE_MAX_NORMALIZED
        ),
    }
    if tuple(checks) != GEOMETRY_GATE_CHECK_NAMES:
        raise RuntimeError("geometry gate checks drifted from registered schema")
    return {
        "schema": GEOMETRY_GATE_SCHEMA,
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [
            f"geometry_gate:{name}" for name, accepted in checks.items() if not accepted
        ],
        "bbox_relative_delta": float(bbox_relative_delta),
        "edge_length_relative_delta": float(edge_length_relative_delta),
        "input_face_count": int(input_face_count),
        "candidate_face_count": int(candidate_face_count),
        "input_edge_count": int(input_edge_count),
        "candidate_edge_count": int(candidate_edge_count),
        "input_vertex_count": int(input_vertex_count),
        "candidate_vertex_count": int(candidate_vertex_count),
        "input_face_edge_occurrences": int(input_face_edge_occurrences),
        "candidate_face_edge_occurrences": int(candidate_face_edge_occurrences),
        "input_face_edge_incidence_counts": list(input_face_edge_incidence),
        "candidate_face_edge_incidence_counts": list(candidate_face_edge_incidence),
        "input_edge_face_incidence_counts": list(input_edge_face_incidence),
        "candidate_edge_face_incidence_counts": list(candidate_edge_face_incidence),
        "input_vertex_edge_incidence_counts": list(input_vertex_edge_incidence),
        "candidate_vertex_edge_incidence_counts": list(
            candidate_vertex_edge_incidence
        ),
        "projectable_edge_count": int(projectable_edge_count),
        "unprojectable_edge_count": int(unprojectable_edge_count),
        "input_projection_sample_count": int(input_projection_sample_count),
        "input_to_candidate_sample_count": int(input_to_candidate_sample_count),
        "input_to_candidate_projected_sample_count": int(
            input_to_candidate_projected_count
        ),
        "candidate_to_input_sample_count": int(candidate_to_input_sample_count),
        "candidate_to_input_projected_sample_count": int(
            candidate_to_input_projected_count
        ),
        "candidate_curve_requested_sample_count": int(candidate_curve_requested_count),
        "candidate_curve_successful_sample_count": int(candidate_curve_successful_count),
        "candidate_curve_sampling_failure_count": int(candidate_curve_sampling_failures),
        "input_to_candidate_rms_normalized": float(input_to_candidate_rms),
        "input_to_candidate_max_normalized": float(input_to_candidate_max),
        "input_to_candidate_projection_failure_count": int(input_to_candidate_failures),
        "candidate_to_input_rms_normalized": float(candidate_to_input_rms),
        "candidate_to_input_max_normalized": float(candidate_to_input_max),
        "candidate_to_input_projection_failure_count": int(candidate_to_input_failures),
        "thresholds": thresholds,
    }


def validate_accepted_geometry_gate(
    gate: Mapping[str, Any] | Any,
) -> tuple[bool, list[str]]:
    """Validate a serialized positive gate instead of trusting its accepted bit."""
    reasons: list[str] = []
    if not isinstance(gate, Mapping):
        return False, ["accepted_gate_not_mapping"]
    if gate.get("schema") != GEOMETRY_GATE_SCHEMA:
        reasons.append("accepted_gate_schema_mismatch")
    if gate.get("accepted") is not True:
        reasons.append("accepted_gate_flag_not_true")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        reasons.append("accepted_gate_checks_missing")
    else:
        actual_names = set(str(name) for name in checks)
        expected_names = set(GEOMETRY_GATE_CHECK_NAMES)
        if actual_names != expected_names:
            reasons.append("accepted_gate_checks_schema_mismatch")
        if any(checks.get(name) is not True for name in GEOMETRY_GATE_CHECK_NAMES):
            reasons.append("accepted_gate_check_not_true")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, Mapping) or dict(thresholds) != (
        geometry_gate_thresholds()
    ):
        reasons.append("accepted_gate_thresholds_mismatch")
    if list(gate.get("rejection_reasons") or ()):
        reasons.append("accepted_gate_has_rejection_reasons")

    continuous_keys = (
        "bbox_relative_delta",
        "edge_length_relative_delta",
        "input_to_candidate_rms_normalized",
        "input_to_candidate_max_normalized",
        "candidate_to_input_rms_normalized",
        "candidate_to_input_max_normalized",
    )
    count_keys = (
        "input_face_count",
        "candidate_face_count",
        "input_edge_count",
        "candidate_edge_count",
        "input_vertex_count",
        "candidate_vertex_count",
        "input_face_edge_occurrences",
        "candidate_face_edge_occurrences",
        "projectable_edge_count",
        "unprojectable_edge_count",
        "input_projection_sample_count",
        "input_to_candidate_sample_count",
        "input_to_candidate_projected_sample_count",
        "input_to_candidate_projection_failure_count",
        "candidate_to_input_sample_count",
        "candidate_to_input_projected_sample_count",
        "candidate_to_input_projection_failure_count",
        "candidate_curve_requested_sample_count",
        "candidate_curve_successful_sample_count",
        "candidate_curve_sampling_failure_count",
    )
    continuous: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key in continuous_keys:
        value = gate.get(key)
        if not _finite_number(value) or float(value) < 0.0:
            reasons.append(f"accepted_gate_invalid_metric:{key}")
        else:
            continuous[key] = float(value)
    for key in count_keys:
        value = gate.get(key)
        if (
            not _finite_number(value)
            or int(value) != float(value)
            or int(value) < 0
        ):
            reasons.append(f"accepted_gate_invalid_count:{key}")
        else:
            counts[key] = int(value)

    incidence_specs = (
        ("input_face_edge_incidence_counts", "input_face_count"),
        ("candidate_face_edge_incidence_counts", "candidate_face_count"),
        ("input_edge_face_incidence_counts", "input_edge_count"),
        ("candidate_edge_face_incidence_counts", "candidate_edge_count"),
        ("input_vertex_edge_incidence_counts", "input_vertex_count"),
        ("candidate_vertex_edge_incidence_counts", "candidate_vertex_count"),
    )
    incidence: dict[str, tuple[int, ...]] = {}
    for key, count_key in incidence_specs:
        if count_key not in counts:
            reasons.append(f"accepted_gate_incidence_count_missing:{key}")
            continue
        try:
            incidence[key] = _nonnegative_integer_sequence(
                gate.get(key), name=key, expected_length=counts[count_key]
            )
        except (TypeError, ValueError):
            reasons.append(f"accepted_gate_invalid_incidence:{key}")

    required_positive = (
        "input_face_count",
        "candidate_face_count",
        "input_edge_count",
        "candidate_edge_count",
        "input_vertex_count",
        "candidate_vertex_count",
        "input_projection_sample_count",
        "candidate_curve_requested_sample_count",
    )
    if all(key in counts for key in required_positive) and any(
        counts[key] <= 0 for key in required_positive
    ):
        reasons.append("accepted_gate_required_count_not_positive")
    if all(key in counts for key in count_keys):
        if counts["projectable_edge_count"] != counts["candidate_edge_count"]:
            reasons.append("accepted_gate_projectable_edge_count_mismatch")
        if counts["unprojectable_edge_count"] != 0:
            reasons.append("accepted_gate_has_unprojectable_edges")
        if (
            counts["input_to_candidate_sample_count"]
            != counts["input_projection_sample_count"]
            or counts["input_to_candidate_projected_sample_count"]
            + counts["input_to_candidate_projection_failure_count"]
            != counts["input_projection_sample_count"]
        ):
            reasons.append("accepted_gate_input_projection_accounting_mismatch")
        if (
            counts["candidate_curve_successful_sample_count"]
            + counts["candidate_curve_sampling_failure_count"]
            != counts["candidate_curve_requested_sample_count"]
            or counts["candidate_curve_sampling_failure_count"] != 0
        ):
            reasons.append("accepted_gate_curve_sampling_accounting_mismatch")
        if (
            counts["candidate_to_input_sample_count"]
            != counts["candidate_curve_successful_sample_count"]
            or counts["candidate_to_input_projected_sample_count"]
            + counts["candidate_to_input_projection_failure_count"]
            != counts["candidate_curve_successful_sample_count"]
        ):
            reasons.append("accepted_gate_candidate_projection_accounting_mismatch")
        if (
            counts["input_to_candidate_projection_failure_count"] != 0
            or counts["candidate_to_input_projection_failure_count"] != 0
        ):
            reasons.append("accepted_gate_has_projection_failures")
    if all(key in continuous for key in continuous_keys):
        threshold_values = geometry_gate_thresholds()
        if continuous["bbox_relative_delta"] > threshold_values[
            "max_bbox_relative_delta"
        ]:
            reasons.append("accepted_gate_bbox_exceeds_threshold")
        if continuous["edge_length_relative_delta"] > threshold_values[
            "max_edge_length_relative_delta"
        ]:
            reasons.append("accepted_gate_edge_length_exceeds_threshold")
        for key in (
            "input_to_candidate_rms_normalized",
            "candidate_to_input_rms_normalized",
        ):
            if continuous[key] > threshold_values["max_edge_sample_rms_normalized"]:
                reasons.append(f"accepted_gate_rms_exceeds_threshold:{key}")
        for key in (
            "input_to_candidate_max_normalized",
            "candidate_to_input_max_normalized",
        ):
            if continuous[key] > threshold_values["max_edge_sample_max_normalized"]:
                reasons.append(f"accepted_gate_max_exceeds_threshold:{key}")
    if all(key in incidence for key, _ in incidence_specs):
        if (
            incidence["input_face_edge_incidence_counts"]
            != incidence["candidate_face_edge_incidence_counts"]
            or incidence["input_edge_face_incidence_counts"]
            != incidence["candidate_edge_face_incidence_counts"]
            or incidence["input_vertex_edge_incidence_counts"]
            != incidence["candidate_vertex_edge_incidence_counts"]
        ):
            reasons.append("accepted_gate_incidence_mismatch")
        if all(
            key in counts
            for key in (
                "input_face_edge_occurrences",
                "candidate_face_edge_occurrences",
                "input_edge_count",
                "candidate_edge_count",
            )
        ):
            if sum(incidence["input_face_edge_incidence_counts"]) != counts[
                "input_face_edge_occurrences"
            ]:
                reasons.append("accepted_gate_input_face_incidence_sum_mismatch")
            if sum(incidence["candidate_face_edge_incidence_counts"]) != counts[
                "candidate_face_edge_occurrences"
            ]:
                reasons.append("accepted_gate_candidate_face_incidence_sum_mismatch")
            if sum(incidence["input_vertex_edge_incidence_counts"]) != 2 * counts[
                "input_edge_count"
            ]:
                reasons.append("accepted_gate_input_vertex_degree_sum_mismatch")
            if sum(incidence["candidate_vertex_edge_incidence_counts"]) != 2 * counts[
                "candidate_edge_count"
            ]:
                reasons.append("accepted_gate_candidate_vertex_degree_sum_mismatch")
    unique_reasons = list(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons


__all__ = [
    "GEOMETRY_GATE_SCHEMA",
    "GEOMETRY_GATE_CHECK_NAMES",
    "MAX_BBOX_RELATIVE_DELTA",
    "MAX_EDGE_LENGTH_RELATIVE_DELTA",
    "MAX_EDGE_SAMPLE_MAX_NORMALIZED",
    "MAX_EDGE_SAMPLE_RMS_NORMALIZED",
    "PROJECTION_SAMPLES_PER_EDGE",
    "candidate_step_signature",
    "geometry_topology_gate",
    "geometry_gate_thresholds",
    "input_geometry_signature",
    "sample_input_edge_points",
    "validate_accepted_geometry_gate",
]
