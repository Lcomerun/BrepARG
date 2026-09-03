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
V2_SCHEMA = "p0a-face-wire-crossing-diagnosis-v2"
EXPECTED_INVALID_CADS = 16
EXPECTED_STEP_CASES = 11
EXPECTED_PRE_STEP_CASES = 5
OCCURRENCE_KINDS = (
    "adjacent",
    "closure",
    "non_adjacent",
    "self_only",
    "pcurve_gap",
    "seam",
    "disconnected",
    "unavailable",
)


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


def crossing_pair_candidates(edge_count: int) -> list[dict[str, Any]]:
    """Return every cyclic edge-pair check exactly once using OCC positions.

    OCC wire positions are one-based.  Position 1's predecessor is the final
    edge, so that pair is reported as ``closure``.  Consecutive positions 2
    through n are ``adjacent``.  Every remaining unordered pair has cyclic
    distance greater than one and is ``non_adjacent``.
    """
    count = int(edge_count)
    if count < 2:
        return []
    candidates = [
        {
            "kind": "closure",
            "edge_positions": [count, 1],
            "check_arguments": [1],
        }
    ]
    candidates.extend(
        {
            "kind": "adjacent",
            "edge_positions": [current - 1, current],
            "check_arguments": [current],
        }
        for current in range(2, count + 1)
    )
    for first in range(1, count + 1):
        for second in range(first + 1, count + 1):
            cyclic_distance = min(second - first, count - (second - first))
            if cyclic_distance <= 1:
                continue
            candidates.append(
                {
                    "kind": "non_adjacent",
                    "edge_positions": [first, second],
                    "check_arguments": [first, second],
                }
            )
    return candidates


def _occurrence(
    kind: str,
    edge_positions: Sequence[int],
    status: str,
    **details: Any,
) -> dict[str, Any]:
    row = {
        "kind": str(kind),
        "edge_positions": [int(value) for value in edge_positions],
        "status": str(status),
    }
    row.update(details)
    return row


def _call_occ_check(
    analysis: Any,
    method_name: str,
    arguments: Sequence[int],
    *,
    kind: str,
    edge_positions: Sequence[int],
    pcurve_available: bool = True,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Turn one OCC Boolean check into explicit detected/unavailable evidence."""
    extra = dict(details or {})
    if not pcurve_available:
        return _occurrence(
            kind,
            edge_positions,
            "occ_unavailable_no_pcurve",
            occ_method=method_name,
            **extra,
        )
    try:
        detected = bool(getattr(analysis, method_name)(*map(int, arguments)))
    except Exception as exc:  # OCC raises several wrapped Standard_* failures.
        return _occurrence(
            kind,
            edge_positions,
            "occ_fail",
            occ_method=method_name,
            occ_error_type=type(exc).__name__,
            **extra,
        )
    if not detected:
        return None
    return _occurrence(
        kind,
        edge_positions,
        "detected",
        occ_method=method_name,
        **extra,
    )


def collect_wire_occurrences(
    analysis: Any,
    *,
    edge_count: int,
    pcurve_positions: set[int] | None = None,
    seam_positions: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Collect independent crossing, gap, seam, and connectivity evidence."""
    count = int(edge_count)
    pcurves = set(range(1, count + 1)) if pcurve_positions is None else set(pcurve_positions)
    seams = set() if seam_positions is None else set(seam_positions)
    occurrences: list[dict[str, Any]] = []

    for position in range(1, count + 1):
        row = _call_occ_check(
            analysis,
            "CheckSelfIntersectingEdge",
            [position],
            kind="self_only",
            edge_positions=[position],
            pcurve_available=position in pcurves,
        )
        if row is not None:
            occurrences.append(row)

    for candidate in crossing_pair_candidates(count):
        positions = candidate["edge_positions"]
        row = _call_occ_check(
            analysis,
            "CheckIntersectingEdges",
            candidate["check_arguments"],
            kind=candidate["kind"],
            edge_positions=positions,
            pcurve_available=all(position in pcurves for position in positions),
        )
        if row is not None:
            occurrences.append(row)

    boundary_pairs = [
        ([count, 1], 1, "closure"),
        *[([position - 1, position], position, "adjacent") for position in range(2, count + 1)],
    ] if count >= 2 else []
    for positions, current, relation in boundary_pairs:
        gap = _call_occ_check(
            analysis,
            "CheckGap2d",
            [current],
            kind="pcurve_gap",
            edge_positions=positions,
            pcurve_available=all(position in pcurves for position in positions),
            details={"relation": relation},
        )
        if gap is not None:
            if gap["status"] == "detected":
                try:
                    gap["distance_2d"] = float(analysis.MinDistance2d())
                except Exception as exc:
                    gap["distance_status"] = "occ_fail"
                    gap["distance_error_type"] = type(exc).__name__
            occurrences.append(gap)
        disconnected = _call_occ_check(
            analysis,
            "CheckConnected",
            [current],
            kind="disconnected",
            edge_positions=positions,
            details={"relation": relation},
        )
        if disconnected is not None:
            occurrences.append(disconnected)

    for position in sorted(seams):
        row = _call_occ_check(
            analysis,
            "CheckSeam",
            [position],
            kind="seam",
            edge_positions=[position],
            pcurve_available=position in pcurves,
        )
        if row is not None:
            occurrences.append(row)
    return occurrences


def enrich_wire_occurrences_with_source_edges(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    observed_wire: Any,
    occurrence_edges: Mapping[int, Any],
    source_mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind fixed-wire occurrences to source edges using identity proof only.

    ``edge_positions`` use OCC's one-based wire positions, while
    ``occurrence_edges`` contains the corresponding edges from the exact
    post-ShapeFix ``ShapeAnalysis_Wire.WireData`` used by the diagnosis.  The
    observed pre-ShapeFix wire must be ``IsSame`` to exactly one proof row, and
    every fixed occurrence edge must be ``IsSame`` to candidates carrying
    exactly one source edge id.  Explorer ordinals are never correspondence.
    """

    enriched = [dict(occurrence) for occurrence in occurrences]
    for occurrence in enriched:
        occurrence.pop("source_edge_ids", None)
        occurrence.pop("source_mapping_status", None)
        occurrence.pop("source_mapping_reason", None)

    def mark_all(status: str, reason: str) -> list[dict[str, Any]]:
        for occurrence in enriched:
            occurrence["source_mapping_status"] = status
            occurrence["source_mapping_reason"] = reason
        return enriched

    if not isinstance(source_mapping, Mapping):
        return mark_all("unavailable", "source_mapping_not_mapping")

    mapping_status = source_mapping.get("status")
    if mapping_status not in {
        "exact_identity",
        "exact_face_local_geometry",
        "exact_sewing_history",
        "exact_sewing_face_local_geometry",
    }:
        status = "ambiguous" if mapping_status == "ambiguous" else "unavailable"
        return mark_all(status, "source_mapping_status_not_exact")

    wire_rows = source_mapping.get("wire_rows")
    if not isinstance(wire_rows, Sequence) or isinstance(wire_rows, (str, bytes)):
        return mark_all("unavailable", "source_mapping_wire_rows_missing")

    matching_rows: list[Mapping[str, Any]] = []
    for row in wire_rows:
        if not isinstance(row, Mapping):
            continue
        candidate_wire = row.get("observed_wire")
        if candidate_wire is None:
            continue
        try:
            same_wire = bool(observed_wire.IsSame(candidate_wire))
        except Exception:
            return mark_all("unavailable", "source_wire_identity_measurement_failed")
        if not same_wire:
            continue
        matching_rows.append(row)

    if not matching_rows:
        return mark_all("unavailable", "source_wire_mapping_not_found")
    if len(matching_rows) != 1:
        return mark_all("ambiguous", "source_wire_mapping_not_unique")

    candidates = matching_rows[0].get("source_edge_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return mark_all("unavailable", "source_edge_candidates_missing")

    for occurrence in enriched:
        positions = occurrence.get("edge_positions")
        if (
            not isinstance(positions, Sequence)
            or isinstance(positions, (str, bytes))
            or not positions
        ):
            occurrence["source_mapping_status"] = "unavailable"
            occurrence["source_mapping_reason"] = "occurrence_edge_positions_missing"
            continue

        source_edge_ids: list[int] = []
        failure_reason: str | None = None
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                failure_reason = "occurrence_edge_position_invalid"
                break
            if position not in occurrence_edges:
                failure_reason = "occurrence_edge_position_out_of_range"
                break
            occurrence_edge = occurrence_edges[position]
            matching_source_ids: set[int] = set()
            identity_failed = False
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    identity_failed = True
                    break
                source_edge_id = candidate.get("source_edge_id")
                candidate_edge = candidate.get("observed_edge")
                if (
                    isinstance(source_edge_id, bool)
                    or not isinstance(source_edge_id, int)
                    or source_edge_id < 0
                    or candidate_edge is None
                ):
                    identity_failed = True
                    break
                try:
                    same_edge = bool(occurrence_edge.IsSame(candidate_edge))
                except Exception:
                    identity_failed = True
                    break
                if same_edge:
                    matching_source_ids.add(int(source_edge_id))
            if identity_failed:
                failure_reason = "source_edge_identity_measurement_failed"
                break
            if len(matching_source_ids) != 1:
                failure_reason = (
                    "source_edge_identity_not_found"
                    if not matching_source_ids
                    else "source_edge_identity_ambiguous"
                )
                break
            source_edge_ids.append(next(iter(matching_source_ids)))

        if failure_reason is not None:
            occurrence["source_mapping_status"] = "unavailable"
            occurrence["source_mapping_reason"] = failure_reason
            continue
        occurrence["source_edge_ids"] = source_edge_ids
        occurrence["source_mapping_status"] = "mapped"
    return enriched


def _wire_row_v2(*, face_index: int, wire_index: int, wire: Any, face: Any) -> dict[str, Any]:
    """Locate the exact 1-based OCC edge positions behind every wire defect."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.ShapeFix import ShapeFix_Wire

    fixer = ShapeFix_Wire(wire, face, 0.01)
    fixer.Load(wire)
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
    wire_data = analysis.WireData()
    edge_count = int(analysis.NbEdges())
    occurrence_edges = {
        int(position): wire_data.Edge(position)
        for position in range(1, edge_count + 1)
    }
    aggregate_occurrences: list[dict[str, Any]] = []
    try:
        aggregate_self_intersection = bool(analysis.CheckSelfIntersection())
        crossing_detail_status = (
            "diagnosed" if aggregate_self_intersection
            else "not_applicable_aggregate_clean"
        )
    except Exception as exc:
        aggregate_self_intersection = None
        crossing_detail_status = "occ_fail"
        aggregate_occurrences.append(
            _occurrence(
                "unavailable",
                [],
                "occ_fail",
                occ_method="CheckSelfIntersection",
                occ_error_type=type(exc).__name__,
            )
        )
    pcurve_positions: set[int] = set()
    seam_positions: set[int] = set()
    pcurve_probe_failures: list[dict[str, Any]] = []
    for position in range(1, edge_count + 1):
        edge = occurrence_edges[position]
        try:
            pcurve, _first, _last = BRep_Tool.CurveOnSurface(edge, face)
            if pcurve is not None:
                pcurve_positions.add(position)
        except Exception as exc:
            pcurve_probe_failures.append(
                _occurrence(
                    "unavailable",
                    [position],
                    "occ_fail",
                    occ_method="BRep_Tool.CurveOnSurface",
                    occ_error_type=type(exc).__name__,
                )
            )
        try:
            if BRep_Tool.IsClosed(edge, face):
                seam_positions.add(position)
        except Exception as exc:
            pcurve_probe_failures.append(
                _occurrence(
                    "unavailable",
                    [position],
                    "occ_fail",
                    occ_method="BRep_Tool.IsClosed",
                    occ_error_type=type(exc).__name__,
                )
            )
    occurrences = list(aggregate_occurrences)
    if aggregate_self_intersection is True:
        occurrences.extend(
            collect_wire_occurrences(
                analysis,
                edge_count=edge_count,
                pcurve_positions=pcurve_positions,
                seam_positions=seam_positions,
            )
        )
        occurrences.extend(pcurve_probe_failures)
    return {
        "face_index": int(face_index),
        "wire_index": int(wire_index),
        "edge_count": edge_count,
        "edge_position_basis": "occ_1_based",
        "aggregate_self_intersection": aggregate_self_intersection,
        "crossing_detail_status": crossing_detail_status,
        "pcurve_edge_positions": sorted(pcurve_positions),
        "seam_edge_positions": sorted(seam_positions),
        "occurrences": occurrences,
        "occurrence_kinds": sorted({row["kind"] for row in occurrences}),
        # Private in-memory evidence consumed by ``diagnose_face_wires_v2``.
        # It is removed before a public/JSON-safe diagnosis is returned.
        "_observed_wire": wire,
        "_occurrence_edges": occurrence_edges,
    }


def diagnose_face_wires_v2(
    face: Any,
    *,
    face_index: int | None = None,
    source_face_index: int | None = None,
    source_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose one OCC face and optionally bind defects to source edges.

    The returned schema is the one-face form of
    :func:`diagnose_step_face_wires_v2`. Omitting ``source_mapping`` keeps the
    established STEP diagnostic rows compatible. Supplying a mapping
    annotates each occurrence with either exact source edge ids or an explicit
    unavailable/ambiguous reason.
    """
    from OCC.Core.TopAbs import TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Wire

    if face_index is None and source_face_index is None:
        raise ValueError("face_index or source_face_index is required")
    local_face_index = (
        int(source_face_index) if face_index is None else int(face_index)
    )

    face_wires: list[dict[str, Any]] = []
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    wire_index = 0
    while wire_explorer.More():
        try:
            row = _wire_row_v2(
                face_index=local_face_index,
                wire_index=wire_index,
                wire=topods_Wire(wire_explorer.Current()),
                face=face,
            )
        except Exception as exc:
            row = {
                "face_index": local_face_index,
                "wire_index": int(wire_index),
                "edge_count": 0,
                "edge_position_basis": "occ_1_based",
                "pcurve_edge_positions": [],
                "seam_edge_positions": [],
                "occurrences": [
                    _occurrence(
                        "unavailable",
                        [],
                        "occ_fail",
                        occ_method="wire_diagnosis",
                        occ_error_type=type(exc).__name__,
                    )
                ],
                "occurrence_kinds": ["unavailable"],
            }
        if source_mapping is not None:
            row["occurrences"] = enrich_wire_occurrences_with_source_edges(
                row["occurrences"],
                observed_wire=row.get("_observed_wire", wire_explorer.Current()),
                occurrence_edges=row.get("_occurrence_edges", {}),
                source_mapping=source_mapping,
            )
        row.pop("_observed_wire", None)
        row.pop("_occurrence_edges", None)
        if source_face_index is not None:
            row["source_face_index"] = int(source_face_index)
        face_wires.append(row)
        wire_explorer.Next()
        wire_index += 1

    face_occurrences = [
        occurrence
        for wire_row in face_wires
        for occurrence in wire_row["occurrences"]
    ]
    face_row: dict[str, Any] = {
        "face_index": local_face_index,
        "wire_count": len(face_wires),
        "wires_with_occurrences": [
            row["wire_index"] for row in face_wires if row["occurrences"]
        ],
        "occurrence_kinds": sorted({row["kind"] for row in face_occurrences}),
    }
    if source_face_index is not None:
        face_row["source_face_index"] = int(source_face_index)

    occurrences: list[dict[str, Any]] = []
    for row in face_wires:
        location = {
            "face_index": row["face_index"],
            "wire_index": row["wire_index"],
        }
        if source_face_index is not None:
            location["source_face_index"] = int(source_face_index)
        occurrences.extend(
            {**location, **occurrence} for occurrence in row["occurrences"]
        )
    return {
        "status": "diagnosed",
        "edge_position_basis": "occ_1_based",
        "faces": [face_row],
        "wires": face_wires,
        "occurrences": occurrences,
        "occurrence_kinds": sorted({row["kind"] for row in occurrences}),
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


def diagnose_step_face_wires_v2(step_path: Path, *, breparg_root: Path) -> dict[str, Any]:
    """Diagnose exact crossing modes while preserving unavailable OCC evidence."""
    root = Path(breparg_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face

    reader = STEPControl_Reader()
    try:
        read_status = reader.ReadFile(str(step_path))
    except Exception as exc:
        return {
            "status": "occ_fail",
            "occ_error_type": type(exc).__name__,
            "faces": [],
            "wires": [],
            "occurrences": [
                _occurrence(
                    "unavailable",
                    [],
                    "occ_fail",
                    occ_method="STEPControl_Reader.ReadFile",
                    occ_error_type=type(exc).__name__,
                )
            ],
        }
    if read_status != IFSelect_RetDone:
        return {
            "status": "step_read_failed",
            "faces": [],
            "wires": [],
            "occurrences": [
                _occurrence("unavailable", [], "step_read_failed")
            ],
        }
    reader.TransferRoots()
    shape = reader.OneShape()
    wires: list[dict[str, Any]] = []
    faces: list[dict[str, Any]] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        face_diagnosis = diagnose_face_wires_v2(face, face_index=face_index)
        faces.extend(face_diagnosis["faces"])
        wires.extend(face_diagnosis["wires"])
        face_explorer.Next()
        face_index += 1
    occurrences = [
        {
            "face_index": row["face_index"],
            "wire_index": row["wire_index"],
            **occurrence,
        }
        for row in wires
        for occurrence in row["occurrences"]
    ]
    return {
        "status": "diagnosed",
        "edge_position_basis": "occ_1_based",
        "faces": faces,
        "wires": wires,
        "occurrences": occurrences,
        "occurrence_kinds": sorted({row["kind"] for row in occurrences}),
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


def build_case_row_v2(source: Mapping[str, Any], *, breparg_root: Path) -> dict[str, Any]:
    """Produce one v2 row with explicit no-STEP and OCC-failure states."""
    source_path = Path(str(source["source_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with source_path.open("rb") as handle:
        parsed = pickle.load(handle)
    step_saved = bool(source.get("step_saved"))
    step_path = Path(str(source.get("step_path") or "")) if step_saved else None
    row: dict[str, Any] = {
        "schema": V2_SCHEMA,
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
            step_diagnosis={
                "status": "unavailable_no_saved_step",
                "edge_position_basis": "occ_1_based",
                "faces": [],
                "wires": [],
                "occurrences": [
                    _occurrence("unavailable", [], "unavailable_no_saved_step")
                ],
                "occurrence_kinds": ["unavailable"],
            },
        )
        return row
    face_wire = diagnose_step_face_wires_v2(step_path, breparg_root=breparg_root)
    diagnosed = face_wire.get("status") == "diagnosed"
    row.update(
        step_diagnosis_available=diagnosed,
        step_sha256=sha256_file(step_path),
        step_diagnosis=face_wire,
    )
    if diagnosed:
        row["validity_components"] = diagnose_step(step_path, breparg_root=breparg_root)
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


def summarize_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate v2 occurrence kinds without treating missing evidence as clean."""
    observed = [row for row in rows if row.get("step_diagnosis_available")]
    unavailable = [row for row in rows if not row.get("step_diagnosis_available")]
    occurrences = [
        {"cad_id": str(row["cad_id"]), **occurrence}
        for row in rows
        for occurrence in (row.get("step_diagnosis") or {}).get("occurrences", [])
    ]
    kind_counts = Counter(str(row["kind"]) for row in occurrences)
    kind_cases = {
        kind: len({row["cad_id"] for row in occurrences if row["kind"] == kind})
        for kind in OCCURRENCE_KINDS
    }
    status_counts = Counter(str(row["status"]) for row in occurrences)
    wire_keys = {
        (str(row["cad_id"]), int(row["face_index"]), int(row["wire_index"]))
        for row in occurrences
        if row["kind"] != "unavailable"
    }
    detailed_wire_keys = {
        (str(row["cad_id"]), int(wire["face_index"]), int(wire["wire_index"]))
        for row in observed
        for wire in (row.get("step_diagnosis") or {}).get("wires", [])
        if wire.get("aggregate_self_intersection") is True
    }
    return {
        "schema": V2_SCHEMA,
        "cases": len(rows),
        "edge_position_basis": "occ_1_based",
        "step_diagnosis_available": len(observed),
        "step_diagnosis_unavailable": len(unavailable),
        "historical_step_saved": sum(bool(row.get("historical_step_saved")) for row in rows),
        "occurrence_counts": {
            kind: int(kind_counts.get(kind, 0)) for kind in OCCURRENCE_KINDS
        },
        "occurrence_case_counts": kind_cases,
        "occurrence_status_counts": dict(sorted(status_counts.items())),
        "self_intersection_wire_count": len(detailed_wire_keys),
        "self_intersection_wires_with_classified_occurrences": len(
            detailed_wire_keys & wire_keys
        ),
        "self_intersection_wires_without_classified_occurrences": [
            {"cad_id": cad_id, "face_index": face_index, "wire_index": wire_index}
            for cad_id, face_index, wire_index in sorted(detailed_wire_keys - wire_keys)
        ],
        "occurrences_by_cad": occurrences,
        "source_topology_suspicious_cases": sum(
            bool((row.get("source_topology") or {}).get("suspicious_faces"))
            for row in rows
        ),
        "unavailable_step_cad_ids": sorted(
            str(row["cad_id"]) for row in unavailable
        ),
    }


def validate_v2_population(summary: Mapping[str, Any]) -> None:
    """Fail closed unless v2 uses the complete stage-aware 16-case baseline."""
    expected = {
        "cases": EXPECTED_INVALID_CADS,
        "step_diagnosis_available": EXPECTED_STEP_CASES,
        "step_diagnosis_unavailable": EXPECTED_PRE_STEP_CASES,
        "historical_step_saved": EXPECTED_STEP_CASES,
    }
    mismatches = [
        f"{key}={summary.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError("v2 P0-A population mismatch: " + "; ".join(mismatches))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration-manifest", type=Path)
    inputs.add_argument("--p0a-attempts", type=Path)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args(argv)

    if args.schema_version == "v2" and args.p0a_attempts is None:
        parser.error("v2 requires --p0a-attempts from the stage-aware P0-A run")

    if args.p0a_attempts is not None:
        selected = frozen_p0a_baseline_rows(args.p0a_attempts)
        input_path = args.p0a_attempts
        input_kind = "p0a_stage_aware_baseline_attempts"
    else:
        selected = frozen_original_invalid_rows(args.calibration_manifest)
        input_path = args.calibration_manifest
        input_kind = "historical_calibration_manifest"
    builder = build_case_row_v2 if args.schema_version == "v2" else build_case_row
    rows = [builder(row, breparg_root=args.breparg_root) for row in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "face_wire_cases.jsonl", rows)
    summary = summarize_v2(rows) if args.schema_version == "v2" else summarize(rows)
    if args.schema_version == "v2":
        validate_v2_population(summary)
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
