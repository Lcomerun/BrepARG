"""Isolate and classify closure/single-shell failures at assembly stage boundaries.

The parent process never imports Open CASCADE.  It launches one child for each
CAD/profile pair, validates a structured sentinel, and keeps crashes/timeouts
in the result.  The child may write local diagnostic STEP files, but the
parent report retains only counts, scalar diagnostics, sizes, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import (
        BASELINE_PROFILE,
        DIRECTED_CURVE_INTERPOLATE_PROFILE,
        DIRECTED_LOCAL_TOPOLOGY_PROFILE,
        DIRECTED_LOCAL_PCURVE_PROFILE,
        DIRECTED_SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE,
        RepairProfile,
    )
    from .assembly_selector_geometry import (
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from .diagnose_step_validity_components import diagnose_step
    from .diagnose_assembly_face_wires import _wire_row_v2
    from .directed_trim_assembly import construct_brep_directed
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .run_assembly_repair_matrix import (
        frozen_original_rows,
        profile_kwargs,
        sha256_file,
        source_pickle_binding,
        source_pickle_binding_from_bytes,
        strict_validate_step,
    )
except ImportError:  # direct script execution
    from assembly_repair import (
        BASELINE_PROFILE,
        DIRECTED_CURVE_INTERPOLATE_PROFILE,
        DIRECTED_LOCAL_TOPOLOGY_PROFILE,
        DIRECTED_LOCAL_PCURVE_PROFILE,
        DIRECTED_SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE,
        RepairProfile,
    )
    from assembly_selector_geometry import (
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )
    from diagnose_step_validity_components import diagnose_step
    from diagnose_assembly_face_wires import _wire_row_v2
    from directed_trim_assembly import construct_brep_directed
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from run_assembly_repair_matrix import (
        frozen_original_rows,
        profile_kwargs,
        sha256_file,
        source_pickle_binding,
        source_pickle_binding_from_bytes,
        strict_validate_step,
    )


SCHEMA = "closure-shell-stage-probe-v1"
RUN_SCHEMA = "closure-shell-stage-probe-run-v1"
WORKER_MARKER = "__CLOSURE_SHELL_STAGE_WORKER_RESULT__="
TARGET_CAD_IDS = (
    "00061931_dcdd8a95feac4121adfd341f_step_000",
    "00087341_6a73c5e821934d3fe4d0d555_step_000",
    "00095733_8b325d2fcb27ec9e79388602_step_000",
)
VARIANT_PROFILES: dict[str, RepairProfile] = {
    "historical": BASELINE_PROFILE,
    "directed_interpolate": DIRECTED_CURVE_INTERPOLATE_PROFILE,
    "selector_primary": DIRECTED_LOCAL_TOPOLOGY_PROFILE,
    "surface_interpolate": DIRECTED_SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE,
    "directed_interpolate_periodic": DIRECTED_CURVE_INTERPOLATE_PROFILE,
    "directed_local_pcurve": DIRECTED_LOCAL_PCURVE_PROFILE,
}
DEFAULT_VARIANTS = tuple(VARIANT_PROFILES)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(
        (
            json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
    )
    temporary.replace(target)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _count_subshapes(shape: Any, kind: Any) -> int:
    from OCC.Core.TopExp import TopExp_Explorer

    count = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def diagnose_occ_shape(shape: Any) -> dict[str, Any]:
    """Return path-free in-memory validity components for one OCC shape."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeAnalysis import (
        ShapeAnalysis_FreeBounds,
        ShapeAnalysis_Shell,
        ShapeAnalysis_Wire,
    )
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face, topods_Shell, topods_Wire

    wire_count = 0
    wire_order_failures = 0
    wire_self_intersections = 0
    bad_wires: list[dict[str, Any]] = []
    face_index = 0
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_index = 0
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())
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
            wire_order_failures += int(order_code != 0)
            wire_self_intersections += int(self_intersection)
            if order_code != 0 or self_intersection:
                bad_wires.append(
                    {
                        "face_index": face_index,
                        "wire_index": wire_index,
                        "order_code": order_code,
                        "self_intersection": self_intersection,
                    }
                )
            wire_count += 1
            wire_index += 1
            wire_explorer.Next()
        face_index += 1
        face_explorer.Next()

    shell_count = 0
    shells_with_bad_edges = 0
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(topods_Shell(shell_explorer.Current()))
        shells_with_bad_edges += int(bool(shell_analysis.HasBadEdges()))
        shell_count += 1
        shell_explorer.Next()

    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = _count_subshapes(free_bounds.GetOpenWires(), TopAbs_EDGE)
    return {
        "status": "diagnosed",
        "native_brep_valid": bool(BRepCheck_Analyzer(shape, True).IsValid()),
        "face_count": _count_subshapes(shape, TopAbs_FACE),
        "edge_count": _count_subshapes(shape, TopAbs_EDGE),
        "vertex_count": _count_subshapes(shape, TopAbs_VERTEX),
        "wire_count": wire_count,
        "wire_order_failures": wire_order_failures,
        "wire_self_intersections": wire_self_intersections,
        "bad_wires": bad_wires,
        "shell_count": shell_count,
        "shells_with_bad_edges": shells_with_bad_edges,
        "free_edges": free_edges,
        "solid_count": _count_subshapes(shape, TopAbs_SOLID),
    }


def diagnose_shape_crossings(shape: Any) -> dict[str, Any]:
    """Return exact OCC crossing occurrences for an in-memory shape."""
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face, topods_Wire

    rows: list[dict[str, Any]] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        wire_index = 0
        while wire_explorer.More():
            rows.append(
                _wire_row_v2(
                    face_index=face_index,
                    wire_index=wire_index,
                    wire=topods_Wire(wire_explorer.Current()),
                    face=face,
                )
            )
            wire_index += 1
            wire_explorer.Next()
        face_index += 1
        face_explorer.Next()
    return {
        "wire_count": len(rows),
        "wires": rows,
        "occurrences": [
            occurrence
            for row in rows
            for occurrence in row.get("occurrences", [])
        ],
    }


def _event_components(event: Mapping[str, Any], key: str = "in_memory") -> Mapping[str, Any]:
    value = event.get(key)
    return value if isinstance(value, Mapping) else {}


def classify_first_defective_stage(
    events: Sequence[Mapping[str, Any]],
    *,
    status: str,
    final_components: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> str:
    """Choose the earliest evidenced stage without inferring past missing data."""
    if any(event.get("stage") == "curve_fit_failure" for event in events):
        return "curve_fit"

    for stage in ("face_raw", "face_final"):
        for event in events:
            if event.get("stage") != stage:
                continue
            if int(_event_components(event).get("wire_self_intersections") or 0) > 0:
                return stage
            if int(_event_components(event, "roundtrip").get("wire_self_intersections") or 0) > 0:
                return f"{stage}_roundtrip"

    compound = next(
        (event for event in events if event.get("stage") == "faces_compound"), None
    )
    if compound is not None and int(
        _event_components(compound).get("wire_self_intersections") or 0
    ) > 0:
        return "faces_compound"

    sewn = next(
        (event for event in events if event.get("stage") == "sewn_shape"), None
    )
    if sewn is not None:
        components = _event_components(sewn)
        if int(components.get("shell_count") or 0) != 1:
            return "sewing_shell_count"
        if int(components.get("wire_self_intersections") or 0) > 0:
            return "sewing_wire_self_intersection"

    solid = next((event for event in events if event.get("stage") == "solid"), None)
    if solid is not None:
        components = _event_components(solid)
        if int(components.get("solid_count") or 0) != 1:
            return "solid_construction"
        if int(components.get("wire_self_intersections") or 0) > 0:
            return "solid_wire_self_intersection"

    final = dict(final_components or {})
    if status == "both_valid":
        return "none"
    if final:
        if int(final.get("solid_count") or 0) != 1:
            return "step_roundtrip_solid_count"
        if int(final.get("wire_self_intersections") or 0) > 0:
            return "step_roundtrip_wire_self_intersection"
        if not bool(final.get("native_brep_valid")):
            return "step_roundtrip_native_validity"
        return "step_roundtrip_strict_validity"

    message = str(error or "")
    if "trim loop is open or branching" in message:
        return "source_topology_walk"
    if "wire_builder_not_done" in message:
        return "wire_construction"
    if "face_builder_not_done" in message:
        return "face_construction"
    if "sewing_produced" in message:
        return "sewing"
    if "solid_builder" in message:
        return "solid_construction"
    return "worker_or_unclassified"


def compact_stage_event(event: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "event_index",
        "stage",
        "metadata",
        "in_memory",
        "roundtrip",
        "step_bytes",
        "step_sha256",
        "diagnostic_error_type",
        "roundtrip_error_type",
        "wire_crossing_diagnosis",
        "wire_crossing_error_type",
    }
    return {key: event[key] for key in keep if key in event}


def build_stage_observer(
    *, output_dir: Path, breparg_root: Path
) -> tuple[list[dict[str, Any]], Any]:
    events: list[dict[str, Any]] = []
    event_path = Path(output_dir) / "stage_events.jsonl"
    stage_dir = Path(output_dir) / "stage_steps"

    def observe(
        stage: str, shape: Any | None, metadata: Mapping[str, Any]
    ) -> None:
        event: dict[str, Any] = {
            "schema": SCHEMA,
            "event_index": len(events),
            "stage": str(stage),
            "metadata": dict(metadata),
        }
        if shape is not None:
            try:
                event["in_memory"] = diagnose_occ_shape(shape)
            except BaseException as exc:
                event["diagnostic_error_type"] = type(exc).__name__
            if (
                stage in {"face_raw", "face_final"}
                and int(_event_components(event).get("wire_self_intersections") or 0)
                > 0
            ):
                try:
                    event["wire_crossing_diagnosis"] = diagnose_shape_crossings(
                        shape
                    )
                except BaseException as exc:
                    event["wire_crossing_error_type"] = type(exc).__name__

            components = _event_components(event)
            should_roundtrip = bool(
                stage in {"faces_compound", "sewn_shape", "solid"}
                or int(components.get("wire_self_intersections") or 0) > 0
                or int(components.get("shell_count") or 0) > 1
            )
            if should_roundtrip:
                try:
                    from OCC.Extend.DataExchange import write_step_file

                    stage_dir.mkdir(parents=True, exist_ok=True)
                    face_suffix = metadata.get("face_index")
                    suffix = f"_{int(face_suffix):03d}" if face_suffix is not None else ""
                    step_path = stage_dir / f"{len(events):04d}_{stage}{suffix}.step"
                    write_step_file(shape, str(step_path))
                    if not step_path.is_file() or step_path.stat().st_size <= 0:
                        raise RuntimeError("stage STEP writer produced no bytes")
                    event["step_path"] = str(step_path)
                    event["step_bytes"] = step_path.stat().st_size
                    event["step_sha256"] = sha256_file(step_path)
                    event["roundtrip"] = diagnose_step(
                        step_path, breparg_root=breparg_root
                    )
                except BaseException as exc:
                    event["roundtrip_error_type"] = type(exc).__name__
        events.append(event)
        append_jsonl(event_path, event)

    return events, observe


def run_worker(
    *,
    source: Mapping[str, Any],
    variant: str,
    output_dir: Path,
    breparg_root: Path,
    joint_iterations: int,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    cad_id = str(source["cad_id"])
    profile = VARIANT_PROFILES[variant]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "cad_id": cad_id,
        "parent_id": source.get("parent_id"),
        "variant": variant,
        "profile": profile.name,
        "status": "running",
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events, observer = build_stage_observer(
        output_dir=output_dir, breparg_root=breparg_root
    )
    try:
        source_path = Path(str(source["source_path"]))
        source_bytes = source_path.read_bytes()
        before_binding = source_pickle_binding_from_bytes(source_bytes)
        result["source_pickle_binding"] = before_binding
        if dict(expected_binding) != before_binding:
            raise RuntimeError("source binding mismatched before deserialization")
        parsed = pickle.loads(source_bytes)
        after_binding = source_pickle_binding(source_path)
        result["source_pickle_binding_after"] = after_binding
        if after_binding != before_binding:
            raise RuntimeError("source binding changed during deserialization")

        face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
        edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
        surf_wcs, edge_wcs = cpu_joint_optimize(
            np.asarray(parsed["surf_ncs"], dtype=np.float32),
            np.asarray(parsed["edge_ncs"], dtype=np.float32),
            np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
            np.asarray(parsed["corner_unique"], dtype=np.float32),
            edge_vertex_adj,
            face_edge_adj,
            iterations=int(joint_iterations),
        )
        result["post_joint_counts"] = {
            "face_count": len(surf_wcs),
            "edge_count": len(edge_wcs),
            "source_vertex_count": len(np.unique(edge_vertex_adj.reshape(-1))),
        }
        solid, diagnostics = construct_brep_directed(
            surf_wcs,
            edge_wcs,
            face_edge_adj,
            edge_vertex_adj,
            breparg_root=breparg_root,
            stage_observer=observer,
            **{
                **profile_kwargs(profile),
                "periodic_pcurve_branches": (
                    variant == "directed_interpolate_periodic"
                ),
            },
        )
        result["assembly_diagnostics"] = diagnostics

        from OCC.Extend.DataExchange import write_step_file

        step_path = output_dir / "final.step"
        write_step_file(solid, str(step_path))
        if not step_path.is_file() or step_path.stat().st_size <= 0:
            raise RuntimeError("final STEP writer produced no bytes")
        validity = strict_validate_step(step_path, breparg_root=breparg_root)
        result.update(
            step_saved=True,
            step_bytes=step_path.stat().st_size,
            step_sha256=sha256_file(step_path),
            native_brep_valid=bool(validity["native_brep_valid"]),
            strict_brep_valid=bool(validity["strict_brep_valid"]),
            both_valid=bool(validity["both_valid"]),
            validity_components=validity["validity_components"],
            status="both_valid" if validity["both_valid"] else "step_invalid",
        )

        if validity["both_valid"]:
            effective = diagnostics.get("effective_input_topology") or {}
            input_signature = input_geometry_signature(
                surf_wcs,
                edge_wcs,
                face_edge_adj,
                edge_vertex_adj,
                effective_vertex_count=effective.get("vertex_count"),
                effective_vertex_edge_incidence_counts=effective.get(
                    "vertex_edge_incidence_counts"
                ),
            )
            candidate_signature = candidate_step_signature(
                step_path,
                input_edge_samples=sample_input_edge_points(edge_wcs),
                input_edge_polylines=edge_wcs,
                input_signature=input_signature,
                validity_components=validity["validity_components"],
            )
            result["selector_geometry_topology_gate"] = geometry_topology_gate(
                input_signature, candidate_signature
            )
        else:
            result["selector_geometry_topology_gate"] = {
                "accepted": False,
                "rejection_reasons": ["not_evaluated_because_both_valid_false"],
            }
    except BaseException as exc:
        result.update(
            status="assembly_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )

    result["stage_events"] = [compact_stage_event(event) for event in events]
    result["first_defective_stage"] = classify_first_defective_stage(
        result["stage_events"],
        status=str(result["status"]),
        final_components=result.get("validity_components"),
        error=result.get("error"),
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def parse_worker_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout).splitlines()):
        if not line.startswith(WORKER_MARKER):
            continue
        try:
            value = json.loads(line[len(WORKER_MARKER) :])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def worker_failure_result(
    *,
    source: Mapping[str, Any],
    variant: str,
    status: str,
    returncode: int | None,
    partial_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    events = [compact_stage_event(event) for event in partial_events]
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "variant": variant,
        "profile": VARIANT_PROFILES[variant].name,
        "status": status,
        "step_saved": False,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
        "worker_returncode": returncode,
        "stage_events": events,
        "first_defective_stage": classify_first_defective_stage(
            events, status=status
        ),
    }


def validate_result(
    row: Mapping[str, Any], *, source: Mapping[str, Any], variant: str
) -> None:
    expected = {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "variant": variant,
        "profile": VARIANT_PROFILES[variant].name,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"worker result {key} mismatch: expected {value!r}, got {row.get(key)!r}"
            )
    for key in ("step_saved", "native_brep_valid", "strict_brep_valid", "both_valid"):
        if type(row.get(key)) is not bool:
            raise ValueError(f"worker result {key} must be Boolean")
    if bool(row["native_brep_valid"] and row["strict_brep_valid"]) != row["both_valid"]:
        raise ValueError("both-valid result is inconsistent")
    if row["both_valid"] and not row["step_saved"]:
        raise ValueError("both-valid result has no STEP")
    if not isinstance(row.get("first_defective_stage"), str):
        raise ValueError("worker result lacks a stage classification")


def run_isolated(
    *,
    source: Mapping[str, Any],
    variant: str,
    calibration_manifest: Path,
    breparg_root: Path,
    output_dir: Path,
    joint_iterations: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    cad_id = str(source["cad_id"])
    worker_dir = Path(output_dir) / ".workers" / cad_id / variant
    worker_dir.mkdir(parents=True, exist_ok=True)
    binding = source_pickle_binding(Path(str(source["source_path"])))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--calibration-manifest",
        str(Path(calibration_manifest).resolve()),
        "--breparg-root",
        str(Path(breparg_root).resolve()),
        "--output-dir",
        str(worker_dir.resolve()),
        "--joint-iterations",
        str(int(joint_iterations)),
        "--cad-id",
        cad_id,
        "--worker-cad-id",
        cad_id,
        "--worker-variant",
        variant,
        "--worker-expected-binding-json",
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        (worker_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (worker_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        return worker_failure_result(
            source=source,
            variant=variant,
            status="worker_timeout",
            returncode=None,
            partial_events=read_jsonl(worker_dir / "stage_events.jsonl"),
        )
    (worker_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (worker_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    row = parse_worker_result(completed.stdout)
    if completed.returncode != 0 or row is None:
        return worker_failure_result(
            source=source,
            variant=variant,
            status="worker_process_exit",
            returncode=int(completed.returncode),
            partial_events=read_jsonl(worker_dir / "stage_events.jsonl"),
        )
    validate_result(row, source=source, variant=variant)
    if row.get("source_pickle_binding") != binding:
        raise ValueError("worker source binding differs from parent preflight")
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cad: list[dict[str, Any]] = []
    for cad_id in sorted({str(row["cad_id"]) for row in rows}):
        cad_rows = [row for row in rows if str(row["cad_id"]) == cad_id]
        by_cad.append(
            {
                "cad_id": cad_id,
                "variants": [
                    {
                        "variant": str(row["variant"]),
                        "status": str(row["status"]),
                        "step_saved": bool(row["step_saved"]),
                        "native_brep_valid": bool(row["native_brep_valid"]),
                        "strict_brep_valid": bool(row["strict_brep_valid"]),
                        "both_valid": bool(row["both_valid"]),
                        "first_defective_stage": str(row["first_defective_stage"]),
                        "geometry_topology_gate_accepted": bool(
                            (row.get("selector_geometry_topology_gate") or {}).get(
                                "accepted"
                            )
                        ),
                    }
                    for row in sorted(cad_rows, key=lambda item: str(item["variant"]))
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "attempt_count": len(rows),
        "cad_count": len(by_cad),
        "worker_failure_count": sum(
            str(row.get("status", "")).startswith("worker_") for row in rows
        ),
        "both_valid_count": sum(bool(row.get("both_valid")) for row in rows),
        "geometry_topology_gate_pass_count": sum(
            bool((row.get("selector_geometry_topology_gate") or {}).get("accepted"))
            for row in rows
        ),
        "first_defective_stage_counts": dict(
            sorted(Counter(str(row["first_defective_stage"]) for row in rows).items())
        ),
        "cases": by_cad,
        "eligible_for_invalid16": any(
            bool(row.get("both_valid"))
            and bool((row.get("selector_geometry_topology_gate") or {}).get("accepted"))
            for row in rows
        ),
    }


def _selected_sources(
    calibration_manifest: Path, cad_ids: Iterable[str]
) -> list[dict[str, Any]]:
    rows = frozen_original_rows(calibration_manifest)
    by_id = {str(row["cad_id"]): row for row in rows}
    requested = list(cad_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("CAD ids must be non-empty and unique")
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"requested CAD ids are absent from frozen cohort: {missing}")
    return [by_id[cad_id] for cad_id in requested]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cad-id", action="append", default=None)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_PROFILES), default=None)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-cad-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", choices=tuple(VARIANT_PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-expected-binding-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.joint_iterations < 0:
        parser.error("--joint-iterations must be non-negative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if bool(args.worker_cad_id) != bool(args.worker_variant):
        parser.error("worker CAD and variant must be supplied together")

    requested_ids = tuple(args.cad_id or TARGET_CAD_IDS)
    sources = _selected_sources(args.calibration_manifest, requested_ids)
    if args.worker_cad_id:
        selected = [
            row for row in sources if str(row["cad_id"]) == args.worker_cad_id
        ]
        if len(selected) != 1 or args.worker_expected_binding_json is None:
            parser.error("worker input is not uniquely and cryptographically bound")
        expected_binding = json.loads(args.worker_expected_binding_json)
        row = run_worker(
            source=selected[0],
            variant=str(args.worker_variant),
            output_dir=args.output_dir,
            breparg_root=args.breparg_root,
            joint_iterations=args.joint_iterations,
            expected_binding=expected_binding,
        )
        print(WORKER_MARKER + json.dumps(row, sort_keys=True), flush=True)
        return 0

    variants = tuple(args.variant or DEFAULT_VARIANTS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_bindings = {
        str(source["cad_id"]): source_pickle_binding(Path(str(source["source_path"])))
        for source in sources
    }
    run_payload = {
        "schema": RUN_SCHEMA,
        "cad_ids": requested_ids,
        "variants": variants,
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "source_bindings": source_bindings,
        "probe_source_sha256": sha256_file(Path(__file__)),
        "assembler_source_sha256": sha256_file(
            Path(__file__).with_name("directed_trim_assembly.py")
        ),
    }
    run_payload["signature"] = canonical_sha256(run_payload)
    run_path = output_dir / "run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("signature") != run_payload["signature"]:
            raise RuntimeError("output root belongs to a different probe signature")
    else:
        atomic_json(run_path, {**run_payload, "status": "RUNNING"})

    attempts_path = output_dir / "attempts.jsonl"
    rows = read_jsonl(attempts_path)
    expected_keys = {
        (str(source["cad_id"]), variant)
        for source in sources
        for variant in variants
    }
    keys = [(str(row.get("cad_id")), str(row.get("variant"))) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected_keys):
        raise RuntimeError("existing attempt rows do not match this signed run")
    done = set(keys)
    for source in sources:
        for variant in variants:
            key = (str(source["cad_id"]), variant)
            if key in done:
                continue
            row = run_isolated(
                source=source,
                variant=variant,
                calibration_manifest=args.calibration_manifest,
                breparg_root=args.breparg_root,
                output_dir=output_dir,
                joint_iterations=args.joint_iterations,
                timeout_seconds=args.worker_timeout_seconds,
            )
            validate_result(row, source=source, variant=variant)
            append_jsonl(attempts_path, row)
            rows.append(row)
            done.add(key)
            print(
                json.dumps(
                    {
                        "cad_id": row["cad_id"],
                        "variant": row["variant"],
                        "status": row["status"],
                        "first_defective_stage": row["first_defective_stage"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = summarize(rows)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        run_path,
        {
            **run_payload,
            "status": "COMPLETED",
            "attempt_count": len(rows),
            "attempts_sha256": sha256_file(attempts_path),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
