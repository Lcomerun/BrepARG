"""Run a signed, read-only periodic-pcurve applicability census.

The parent process does not import Open CASCADE. It launches one child process
per frozen CAD and keeps timeouts, native exits, malformed sentinels, and source
binding mismatches in the denominator. The worker observes constructed faces
after pcurves exist and before any face repair; it never writes STEP or mutates
a candidate through the periodic repair helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCHEMA = "periodic-pcurve-applicability-v1"
RUN_SCHEMA = "periodic-pcurve-applicability-run-v1"
SUMMARY_SCHEMA = "periodic-pcurve-applicability-summary-v1"
WORKER_MARKER = "__PERIODIC_PCURVE_APPLICABILITY_RESULT__="
PROFILE = "directed_trim_curve_fit"
PCURVE_GAP_TOLERANCE = 1e-7
TARGET_CAD_IDS = (
    "00047472_197769bbdd814278b715d88a_step_000",
    "00063055_e309c689b9b44f0686f47966_step_000",
    "00032101_674d8fea687f4d9bbca6599b_step_000",
    "00076198_7fde7438ca5d3ccb8a1dd1f4_step_000",
    "00051602_7f1947595ae247e0a4a32f43_step_000",
)
RUN_MANIFEST_NAME = "periodic_pcurve_run.json"
ROWS_NAME = "periodic_pcurve_cases.jsonl"
SUMMARY_NAME = "periodic_pcurve_summary.json"
LOCK_NAME = ".periodic_pcurve_writer.lock"


@contextmanager
def output_writer_lock(output_dir: Path) -> Iterator[None]:
    """Hold a nonblocking OS lock so two parents cannot interleave JSONL rows."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("periodic-pcurve output already has an active writer") from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(payload),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
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


def source_binding(path: Path) -> dict[str, Any]:
    source = Path(path)
    return {"bytes": source.stat().st_size, "sha256": sha256_file(source)}


def payload_binding(payload: bytes) -> dict[str, Any]:
    """Bind the exact bytes handed to ``pickle.loads``."""
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def normalize_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise ValueError("source binding must contain exactly bytes and sha256")
    byte_count = value.get("bytes")
    digest = value.get("sha256")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError("source binding bytes must be a positive integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("source binding sha256 must be lowercase hexadecimal")
    return {"bytes": byte_count, "sha256": digest}


def parse_worker_result(stdout: str) -> dict[str, Any] | None:
    """Accept exactly one sentinel, and only as the final non-empty line."""
    lines = [line for line in str(stdout).splitlines() if line.strip()]
    marker_indices = [
        index for index, line in enumerate(lines) if line.startswith(WORKER_MARKER)
    ]
    if marker_indices != [len(lines) - 1]:
        return None
    line = lines[-1]
    try:
        value = json.loads(line[len(WORKER_MARKER) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def classify_wire_applicability(
    *,
    periodic_state: Mapping[str, Any],
    bad_wire_indices: Sequence[int],
    tolerance: float = PCURVE_GAP_TOLERANCE,
) -> dict[str, Any]:
    """Classify applicability without invoking the mutating repair helper."""
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    targets = sorted({int(index) for index in bad_wire_indices})
    base = {
        "periodic_gap_candidate": False,
        "profile_applicable": False,
        "applicable": False,
        "partial_only": False,
        "target_wire_indices": targets,
        "repairable_wire_indices": [],
        "wire_classifications": [],
    }
    if not targets:
        return {**base, "reason": "no_diagnosed_self_intersection"}
    if periodic_state.get("available") is not True:
        return {
            **base,
            "reason": str(periodic_state.get("reason") or "pcurve_state_unavailable"),
        }
    periods = list(periodic_state.get("periods") or [None, None])
    if len(periods) != 2:
        return {**base, "reason": "invalid_surface_periods"}
    if periods == [None, None]:
        return {**base, "reason": "surface_not_periodic"}
    for period in periods:
        if period is None:
            continue
        if (
            not isinstance(period, (int, float))
            or isinstance(period, bool)
            or not math.isfinite(float(period))
            or float(period) <= 0.0
        ):
            return {**base, "reason": "invalid_surface_periods"}
    wires = list(periodic_state.get("wires") or [])
    if any(index < 0 or index >= len(wires) for index in targets):
        return {**base, "reason": "diagnosed_wire_index_out_of_range"}
    repairable: list[int] = []
    wire_classifications: list[dict[str, Any]] = []
    for index in targets:
        wire = wires[index]
        if not isinstance(wire, Mapping):
            wire_classifications.append(
                {"wire_index": index, "repairable": False, "reason": "malformed_wire_state"}
            )
            continue
        edge_count = wire.get("edge_count")
        movable = wire.get("movable")
        plan_value = wire.get("plan")
        if (
            type(edge_count) is not int
            or edge_count <= 0
            or not isinstance(movable, list)
            or len(movable) != edge_count
            or any(type(value) is not bool for value in movable)
            or not isinstance(plan_value, Mapping)
        ):
            wire_classifications.append(
                {"wire_index": index, "repairable": False, "reason": "malformed_wire_state"}
            )
            continue
        plan = dict(plan_value)
        raw_changed = plan.get("changed_edge_indices")
        raw_offsets = plan.get("offsets")
        changed = list(raw_changed) if isinstance(raw_changed, (list, tuple)) else []
        offsets = list(raw_offsets) if isinstance(raw_offsets, (list, tuple)) else []
        before = plan.get("before_max_gap")
        after = plan.get("after_max_gap")
        reason = "plan_not_solved"
        valid_offsets = bool(
            len(offsets) == edge_count
            and all(
                isinstance(offset, (list, tuple))
                and len(offset) == 2
                and all(type(value) is int for value in offset)
                for offset in offsets
            )
        )
        valid_changed = bool(
            changed
            and all(type(value) is int for value in changed)
            and len(set(changed)) == len(changed)
            and all(0 <= value < edge_count for value in changed)
        )
        derived_changed = (
            [
                position
                for position, offset in enumerate(offsets)
                if tuple(offset) != (0, 0)
            ]
            if valid_offsets
            else []
        )
        numeric_gaps = bool(
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
            and math.isfinite(float(before))
            and math.isfinite(float(after))
            and float(before) >= 0.0
            and float(after) >= 0.0
        )
        is_repairable = False
        if plan.get("solved") is not True:
            reason = str(plan.get("reason") or "plan_not_solved")
        elif (
            not isinstance(raw_changed, (list, tuple))
            or not isinstance(raw_offsets, (list, tuple))
            or not valid_offsets
            or not valid_changed
            or sorted(changed) != derived_changed
        ):
            reason = "malformed_branch_offsets"
        elif any(not movable[position] for position in changed):
            reason = "changed_edge_is_seam_or_immovable"
        elif not numeric_gaps:
            reason = "nonfinite_or_invalid_gap"
        elif float(before) <= tolerance:
            reason = "no_preexisting_branch_gap"
        elif float(after) > tolerance:
            reason = "branch_plan_does_not_close"
        else:
            reason = "repairable_periodic_branch_gap"
            is_repairable = True
            repairable.append(index)
        wire_classifications.append(
            {"wire_index": index, "repairable": is_repairable, "reason": reason}
        )
    candidate = bool(repairable)
    profile_applicable = bool(candidate and len(repairable) == len(targets))
    return {
        "periodic_gap_candidate": candidate,
        "profile_applicable": profile_applicable,
        "applicable": profile_applicable,
        "partial_only": bool(candidate and not profile_applicable),
        "reason": (
            "repairable_periodic_branch_gap_all_targets"
            if profile_applicable
            else "partial_periodic_branch_gap"
            if candidate
            else "no_repairable_periodic_branch_gap"
        ),
        "target_wire_indices": targets,
        "repairable_wire_indices": repairable,
        "wire_classifications": wire_classifications,
    }


def _public_wire_state(wire: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(wire.get("plan") or {})
    return {
        "wire_index": int(wire.get("wire_index", -1)),
        "edge_count": int(wire.get("edge_count", 0)),
        "immovable_seam_edge_count": sum(
            not bool(value) for value in wire.get("movable", [])
        ),
        "movable": [bool(value) for value in wire.get("movable", [])],
        "plan": {
            key: plan.get(key)
            for key in (
                "solved",
                "reason",
                "changed_edge_indices",
                "offsets",
                "before_gaps",
                "after_gaps",
                "before_max_gap",
                "after_max_gap",
                "periods",
                "max_period_shift",
            )
            if key in plan
        },
    }


def build_face_observation(
    *, face_index: int, face: Any, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Collect path-free OCC evidence from a disposable face copy."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.TopoDS import topods

    try:
        from .diagnose_assembly_face_wires import _wire_row_v2
        from .local_wire_topology_repair import (
            face_topology_incidence_signature,
            periodic_pcurve_continuity_state,
            wire_self_intersection_state,
        )
    except ImportError:  # direct script execution
        from diagnose_assembly_face_wires import _wire_row_v2
        from local_wire_topology_repair import (
            face_topology_incidence_signature,
            periodic_pcurve_continuity_state,
            wire_self_intersection_state,
        )

    copier = BRepBuilderAPI_Copy(face, True, False)
    copied = topods.Face(copier.Shape())
    diagnosis = wire_self_intersection_state(copied)
    surface = BRep_Tool.Surface(copied)
    is_u_periodic = bool(surface.IsUPeriodic())
    is_v_periodic = bool(surface.IsVPeriodic())
    state = periodic_pcurve_continuity_state(copied)
    applicability = classify_wire_applicability(
        periodic_state=state,
        bad_wire_indices=diagnosis.get("bad_wire_indices", []),
    )
    detailed_wires = []
    from OCC.Extend.TopologyUtils import TopologyExplorer

    for wire_index, wire in enumerate(TopologyExplorer(copied).wires()):
        if wire_index not in diagnosis.get("bad_wire_indices", []):
            continue
        detailed_wires.append(
            _wire_row_v2(
                face_index=int(face_index),
                wire_index=int(wire_index),
                wire=wire,
                face=copied,
            )
        )
    result = {
        "face_index": int(face_index),
        "phase": str(metadata.get("phase")),
        "loop_count": int(metadata.get("loop_count", 0)),
        "outer_loop_index": int(metadata.get("outer_loop_index", 0)),
        "loop_3d_endpoint_max_gaps": [
            float(value)
            for value in metadata.get("loop_3d_endpoint_max_gaps", [])
        ],
        "face_3d_endpoint_max_gap": float(
            metadata.get("face_3d_endpoint_max_gap", 0.0)
        ),
        "surface_type": str(surface.DynamicType().Name()),
        "is_u_periodic": is_u_periodic,
        "is_v_periodic": is_v_periodic,
        "u_period": float(surface.UPeriod()) if is_u_periodic else None,
        "v_period": float(surface.VPeriod()) if is_v_periodic else None,
        "diagnosis": diagnosis,
        "topology_incidence": face_topology_incidence_signature(copied),
        "pcurve_state": {
            "available": bool(state.get("available")),
            "reason": state.get("reason"),
            "periods": state.get("periods"),
            "wires": [_public_wire_state(wire) for wire in state.get("wires", [])],
        },
        "bad_wire_details": detailed_wires,
        "observation_copy": True,
        "diagnosis_semantics": "project_strict_style_shape_fix_wire_on_copy",
        **applicability,
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} rows must be mappings")
        cad_id = row.get("cad_id")
        if not isinstance(cad_id, str) or not cad_id:
            raise ValueError(f"{name} CAD ids must be non-empty strings")
        if cad_id in by_id:
            raise ValueError(f"duplicate {name} CAD id: {cad_id}")
        by_id[cad_id] = row
    return by_id


def select_census_sources(
    calibration_rows: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
    *,
    target_ids: Sequence[str] = TARGET_CAD_IDS,
) -> list[dict[str, Any]]:
    """Prove the frozen 100 -> current 9 -> registered 5 identity chain."""
    if len(calibration_rows) not in (100, 300):
        raise ValueError("calibration must contain either 100 originals or the formal 3x100 arms")
    originals = [row for row in calibration_rows if row.get("arm") == "original"]
    if len(originals) != 100:
        raise ValueError("calibration must contain exactly 100 original rows")
    calibration_by_id = _unique_rows(originals, name="calibration original")
    if sum(row.get("brep_valid") is True for row in originals) != 84:
        raise ValueError("calibration must contain exactly 84 historically valid CADs")
    if any(type(row.get("brep_valid")) is not bool for row in originals):
        raise TypeError("calibration brep_valid values must be booleans")

    # A formal calibration manifest has three arms. If auxiliary arms are
    # present, prove that each is a complete view of the same CAD/parent/source
    # cohort. Unit-level callers may provide the 100 original rows alone.
    arms = sorted({str(row.get("arm")) for row in calibration_rows})
    if len(calibration_rows) == 300 and set(arms) != {
        "original",
        "continuous_bypass_64d",
        "fsq_8192_4d",
    }:
        raise ValueError("formal calibration must contain the registered three arms")
    original_order = [str(row["cad_id"]) for row in originals]
    for arm in arms:
        arm_rows = [row for row in calibration_rows if str(row.get("arm")) == arm]
        if len(arm_rows) != 100:
            raise ValueError(f"calibration arm {arm!r} does not contain 100 rows")
        arm_by_id = _unique_rows(arm_rows, name=f"calibration {arm}")
        if set(arm_by_id) != set(calibration_by_id):
            raise ValueError(f"calibration arm {arm!r} changes the CAD cohort")
        if [str(row["cad_id"]) for row in arm_rows] != original_order:
            raise ValueError(f"calibration arm {arm!r} changes the CAD order")
        for cad_id, original in calibration_by_id.items():
            candidate = arm_by_id[cad_id]
            for field in ("parent_id", "source_path"):
                if candidate.get(field) != original.get(field):
                    raise ValueError(
                        f"calibration arm {arm!r} changes {field} for {cad_id}"
                    )

    if len(selector_rows) != 100:
        raise ValueError("selector matrix must contain exactly 100 rows")
    selector_by_id = _unique_rows(selector_rows, name="selector")
    if set(selector_by_id) != set(calibration_by_id):
        raise ValueError("selector and calibration CAD cohorts differ")
    if [str(row["cad_id"]) for row in selector_rows] != original_order:
        raise ValueError("selector and calibration CAD order differ")
    selector_schemas = {row.get("schema") for row in selector_rows if "schema" in row}
    if selector_schemas and selector_schemas != {"assembly-repair-selector-v1"}:
        raise ValueError("selector matrix schema mismatch")
    for cad_id, original in calibration_by_id.items():
        selected = selector_by_id[cad_id]
        if selected.get("parent_id") != original.get("parent_id"):
            raise ValueError(f"selector parent identity drift for {cad_id}")
        if "source_path" in selected and selected.get("source_path") != original.get("source_path"):
            raise ValueError(f"selector source identity drift for {cad_id}")
        if type(selected.get("historical_strict_valid")) is not bool:
            raise TypeError("selector historical validity must be boolean")
        if selected["historical_strict_valid"] is not original["brep_valid"]:
            raise ValueError(f"selector historical validity drift for {cad_id}")
        if type(selected.get("strict_brep_valid")) is not bool:
            raise TypeError("selector strict validity must be boolean")

    strict_count = sum(row["strict_brep_valid"] for row in selector_rows)
    if strict_count != 91:
        raise ValueError(f"selector must have 91 strict-valid rows, got {strict_count}")
    if any(
        row["historical_strict_valid"] and not row["strict_brep_valid"]
        for row in selector_rows
    ):
        raise ValueError("selector regresses a historically valid control")
    residual_ids = {
        str(row["cad_id"])
        for row in selector_rows
        if row["strict_brep_valid"] is False
    }
    if len(residual_ids) != 9:
        raise ValueError(f"selector must have exactly 9 strict residuals, got {len(residual_ids)}")

    ordered_targets = [str(value) for value in target_ids]
    if len(ordered_targets) != 5 or len(set(ordered_targets)) != 5:
        raise ValueError("census target list must contain five unique CAD ids")
    if not set(ordered_targets).issubset(residual_ids):
        raise ValueError("every census target must belong to the current strict residual set")
    return [dict(calibration_by_id[cad_id]) for cad_id in ordered_targets]


def selected_sources(manifest: Path, selector_matrix: Path) -> list[dict[str, Any]]:
    return select_census_sources(
        read_jsonl(manifest),
        read_jsonl(selector_matrix),
        target_ids=TARGET_CAD_IDS,
    )


def git_identity(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if revision.returncode or status.returncode:
        raise RuntimeError("could not bind repository identity")
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "status_sha256": hashlib.sha256(status.stdout.encode("utf-8")).hexdigest(),
    }


def source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "tools/probe_periodic_pcurve_applicability.py",
        "tools/directed_trim_assembly.py",
        "tools/local_wire_topology_repair.py",
        "tools/assembly_repair.py",
        "tools/diagnose_assembly_face_wires.py",
        "tools/run_assembly_calibration_oracle.py",
        "tools/run_assembly_repair_matrix.py",
        "tools/solid_topology_repair.py",
    )
    return {path: sha256_file(Path(repo_root) / path) for path in paths}


def _validate_selector_run(
    selector_run: Path,
    *,
    calibration_manifest: Path,
    selector_matrix: Path,
    source_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = json.loads(Path(selector_run).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("selector run must be a JSON object")
    payload = value.get("payload")
    if (
        value.get("schema") != "assembly-repair-run-v2"
        or value.get("status") != "COMPLETED"
        or type(value.get("attempts")) is not int
        or value.get("attempts") != 100
        or not isinstance(payload, Mapping)
    ):
        raise ValueError("selector run is not a completed formal 100-CAD run")
    signature = value.get("signature")
    if not isinstance(signature, str) or canonical_sha256(payload) != signature:
        raise ValueError("selector run signature mismatch")
    calibration_digest = sha256_file(calibration_manifest)
    selector_digest = sha256_file(selector_matrix)
    if payload.get("calibration_manifest_sha256") != calibration_digest:
        raise ValueError("selector run calibration binding mismatch")
    if value.get("final_matrix_sha256") != selector_digest:
        raise ValueError("selector run matrix binding mismatch")
    if payload.get("full_cohort_count") != 100 or payload.get("selected_cohort_count") != 100:
        raise ValueError("selector run cohort count mismatch")
    if (
        payload.get("run_kind") != "assembly-repair-selector-v1"
        or payload.get("matrix_schema") != "assembly-repair-selector-v1"
        or payload.get("candidate_schema") != "assembly-selector-candidate-v1"
        or payload.get("historical_invalid_only") is not False
    ):
        raise ValueError("selector run protocol identity mismatch")
    registered = payload.get("selected_source_pickles")
    if not isinstance(registered, Mapping):
        raise ValueError("selector run lacks source pickle bindings")
    for binding in source_bindings:
        cad_id = str(binding["cad_id"])
        if normalize_binding(registered.get(cad_id) or {}) != normalize_binding(
            {key: binding[key] for key in ("bytes", "sha256")}
        ):
            raise ValueError(f"selector run source binding mismatch for {cad_id}")
    return {
        "bytes": Path(selector_run).stat().st_size,
        "sha256": sha256_file(selector_run),
        "signature": signature,
        "status": "COMPLETED",
    }


def _cohort_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cad_id": str(row["cad_id"]),
            "parent_id": str(row["parent_id"]),
            "historical_strict_valid": bool(row["historical_strict_valid"]),
            "strict_brep_valid": bool(row["strict_brep_valid"]),
        }
        for row in rows
    ]


def build_run_payload(
    args: argparse.Namespace,
    sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_utils = Path(args.breparg_root).resolve() / "utils.py"
    if not runtime_utils.is_file():
        raise FileNotFoundError(runtime_utils)
    bindings = []
    for row in sources:
        path = Path(str(row["source_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        bindings.append({"cad_id": str(row["cad_id"]), **source_binding(path)})
    repository = {**git_identity(repo_root), "source_sha256": source_hashes(repo_root)}
    if repository["dirty"]:
        raise RuntimeError("formal periodic-pcurve census requires a clean Git worktree")
    return {
        "schema": RUN_SCHEMA,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "selector_matrix_sha256": sha256_file(args.selector_matrix),
        "selector_run": _validate_selector_run(
            args.selector_run,
            calibration_manifest=args.calibration_manifest,
            selector_matrix=args.selector_matrix,
            source_bindings=bindings,
        ),
        "selector_cohort_signature": canonical_sha256(_cohort_identity(selector_rows)),
        "selector_strict_valid": sum(
            row.get("strict_brep_valid") is True for row in selector_rows
        ),
        "selector_historical_strict_valid": sum(
            row.get("historical_strict_valid") is True for row in selector_rows
        ),
        "selector_residual_ids": [
            str(row["cad_id"])
            for row in selector_rows
            if row.get("strict_brep_valid") is False
        ],
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "source_bindings": bindings,
        "profile": PROFILE,
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "repository": repository,
        "breparg_runtime": {"utils_sha256": sha256_file(runtime_utils)},
    }


def validate_bound_inputs(
    args: argparse.Namespace,
    *,
    payload: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute every external/code binding before launch and completion."""
    current = build_run_payload(args, sources, selector_rows)
    if current != payload:
        raise RuntimeError("signed census inputs, runtime, or source code drifted")


def validate_case_row(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    run_signature: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> None:
    if row.get("schema") != SCHEMA:
        raise ValueError("worker schema mismatch")
    if str(row.get("cad_id")) != str(source["cad_id"]):
        raise ValueError("worker CAD identity mismatch")
    if row.get("parent_id") != source.get("parent_id"):
        raise ValueError("worker parent identity mismatch")
    if row.get("profile") != PROFILE:
        raise ValueError("worker profile mismatch")
    if row.get("run_signature") != run_signature:
        raise ValueError("worker run signature mismatch")
    bound = normalize_binding(row.get("source_binding") or {})
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("worker status must be a non-empty string")
    pre_measurement_failure = status in {
        "worker_timeout",
        "worker_process_exit",
        "worker_protocol_error",
        "worker_error",
        "source_binding_mismatch",
    }
    loaded_value = row.get("source_binding_loaded_bytes")
    after_value = row.get("source_binding_after_load")
    if pre_measurement_failure and loaded_value is None and after_value is None:
        loaded = bound
        after = bound
    else:
        loaded = normalize_binding(loaded_value or {})
        after = normalize_binding(after_value or {})
    if loaded != bound or after != bound:
        raise ValueError("worker source binding stages differ")
    if expected_binding is not None and bound != normalize_binding(expected_binding):
        raise ValueError("worker source binding mismatches signed input")
    source_face_count = row.get("source_face_count")
    face_count = row.get("face_count")
    faces = row.get("faces")
    if type(source_face_count) is not int or source_face_count < 0:
        raise ValueError("source face count must be a nonnegative integer")
    if type(face_count) is not int or not isinstance(faces, list) or face_count != len(faces):
        raise ValueError("reported face count differs from face rows")
    indices = [face.get("face_index") for face in faces if isinstance(face, Mapping)]
    if len(indices) != len(faces) or any(type(index) is not int for index in indices):
        raise ValueError("face indices must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("face coverage contains duplicate indices")
    if sorted(indices) != list(range(source_face_count)):
        raise ValueError("face coverage must be complete and contiguous")
    if any(face.get("phase") != "post_add_pcurves_pre_repair" for face in faces):
        raise ValueError("worker observed an invalid construction phase")
    all_faces = bool(
        source_face_count > 0
        and sorted(indices) == list(range(source_face_count))
    )
    if row.get("all_faces_observed") is not all_faces:
        raise ValueError("all_faces_observed disagrees with face coverage")
    derived_bad = sorted(
        int(face["face_index"])
        for face in faces
        if (face.get("diagnosis") or {}).get("bad_wire_indices")
    )
    derived_periodic = sorted(
        int(face["face_index"])
        for face in faces
        if (face.get("diagnosis") or {}).get("bad_wire_indices")
        and (face.get("is_u_periodic") or face.get("is_v_periodic"))
    )
    derived_repairable = sorted(
        int(face["face_index"]) for face in faces if face.get("applicable") is True
    )
    for key, derived in (
        ("bad_face_indices", derived_bad),
        ("periodic_bad_face_indices", derived_periodic),
        ("repairable_face_indices", derived_repairable),
    ):
        if row.get(key) != derived:
            raise ValueError(f"derived {key} differs from face evidence")
    embedded_error = any(face.get("reason") == "occ_probe_error" for face in faces)
    if status == "completed":
        if not all_faces or embedded_error:
            raise ValueError("completed worker lacks complete error-free face coverage")
    elif status not in {
        "probe_error",
        "measurement_incomplete",
        "worker_timeout",
        "worker_process_exit",
        "worker_protocol_error",
        "worker_error",
        "source_binding_mismatch",
    }:
        raise ValueError("unknown worker status")


def worker_failure_row(
    source: Mapping[str, Any], *, run_signature: str, status: str,
    returncode: int | None, error_type: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    binding = normalize_binding(expected_binding)
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": PROFILE,
        "run_signature": run_signature,
        "source_binding": binding,
        # A parent-side timeout, native exit, or protocol failure does not
        # prove which bytes (if any) the child loaded. Preserve that distinction
        # instead of fabricating successful worker-stage bindings.
        "source_binding_loaded_bytes": None,
        "source_binding_after_load": None,
        "status": status,
        "worker_returncode": returncode,
        "error_type": error_type,
        "assembly_status": "not_run",
        "source_face_count": 0,
        "face_count": 0,
        "all_faces_observed": False,
        "faces": [],
        "bad_face_indices": [],
        "periodic_bad_face_indices": [],
        "repairable_face_indices": [],
    }


def run_worker(
    source: Mapping[str, Any], *, breparg_root: Path, joint_iterations: int,
    expected_binding: Mapping[str, Any], run_signature: str,
) -> dict[str, Any]:
    import numpy as np

    try:
        from .assembly_repair import DIRECTED_CURVE_PROFILE
        from .directed_trim_assembly import construct_brep_directed
        from .run_assembly_calibration_oracle import cpu_joint_optimize
        from .run_assembly_repair_matrix import profile_kwargs
    except ImportError:  # direct script execution
        from assembly_repair import DIRECTED_CURVE_PROFILE
        from directed_trim_assembly import construct_brep_directed
        from run_assembly_calibration_oracle import cpu_joint_optimize
        from run_assembly_repair_matrix import profile_kwargs

    started = time.perf_counter()
    source_path = Path(str(source["source_path"]))
    expected = normalize_binding(expected_binding)
    before = source_binding(source_path)
    if before != expected:
        raise RuntimeError("source_binding_mismatch_before_load")
    payload = source_path.read_bytes()
    loaded_binding = normalize_binding(payload_binding(payload))
    if loaded_binding != expected:
        raise RuntimeError("source_binding_mismatch_loaded_bytes")
    parsed = pickle.loads(payload)
    after_binding = source_binding(source_path)
    if after_binding != expected:
        raise RuntimeError("source_binding_mismatch_after_load")
    face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
    source_face_count = len(face_edge_adj)
    if source_face_count <= 0 or len(parsed["surf_ncs"]) != source_face_count:
        raise ValueError("source face adjacency and surface counts differ")
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
    faces: list[dict[str, Any]] = []

    def observe(face_index: int, face: Any, metadata: Mapping[str, Any]) -> None:
        try:
            faces.append(
                build_face_observation(
                    face_index=face_index, face=face, metadata=metadata
                )
            )
        except Exception as exc:
            faces.append(
                {
                    "face_index": int(face_index),
                    "phase": str(metadata.get("phase")),
                    "applicable": False,
                    "reason": "occ_probe_error",
                    "error_type": type(exc).__name__,
                }
            )

    assembly_status = "completed"
    assembly_error_type = None
    try:
        construct_brep_directed(
            surf_wcs,
            edge_wcs,
            face_edge_adj,
            edge_vertex_adj,
            breparg_root=breparg_root,
            **profile_kwargs(DIRECTED_CURVE_PROFILE),
            post_pcurve_face_observer=observe,
        )
    except Exception as exc:
        # Face evidence collected before a later sewing/solid failure remains
        # valid. A failure before any face observation is inconclusive.
        assembly_status = "assembly_error"
        assembly_error_type = type(exc).__name__
    repairable = sorted(
        int(face["face_index"]) for face in faces if face.get("applicable") is True
    )
    probe_errors = sum(face.get("reason") == "occ_probe_error" for face in faces)
    observed_indices = [face.get("face_index") for face in faces]
    all_faces_observed = bool(
        observed_indices == list(range(source_face_count))
        and len(set(observed_indices)) == source_face_count
    )
    status = (
        "completed"
        if all_faces_observed and probe_errors == 0
        else "probe_error"
        if probe_errors
        else "measurement_incomplete"
    )
    return {
        "schema": SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": PROFILE,
        "run_signature": run_signature,
        "source_binding": expected,
        "source_binding_loaded_bytes": loaded_binding,
        "source_binding_after_load": after_binding,
        "status": status,
        "assembly_status": assembly_status,
        "assembly_error_type": assembly_error_type,
        "source_face_count": source_face_count,
        "face_count": len(faces),
        "all_faces_observed": all_faces_observed,
        "faces": faces,
        "bad_face_indices": sorted(
            int(face["face_index"])
            for face in faces
            if (face.get("diagnosis") or {}).get("bad_wire_indices")
        ),
        "periodic_bad_face_indices": sorted(
            int(face["face_index"])
            for face in faces
            if (face.get("diagnosis") or {}).get("bad_wire_indices")
            and (face.get("is_u_periodic") or face.get("is_v_periodic"))
        ),
        "repairable_face_indices": repairable,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_isolated(
    source: Mapping[str, Any], *, args: argparse.Namespace,
    run_signature: str, expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    cad_id = str(source["cad_id"])
    log_dir = Path(args.output_dir) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--calibration-manifest",
        str(Path(args.calibration_manifest).resolve()),
        "--selector-matrix",
        str(Path(args.selector_matrix).resolve()),
        "--selector-run",
        str(Path(args.selector_run).resolve()),
        "--breparg-root",
        str(Path(args.breparg_root).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--joint-iterations",
        str(int(args.joint_iterations)),
        "--worker-timeout-seconds",
        str(float(args.worker_timeout_seconds)),
        "--worker-cad-id",
        cad_id,
        "--worker-run-signature",
        run_signature,
        "--worker-source-binding-json",
        json.dumps(dict(expected_binding), sort_keys=True, separators=(",", ":")),
        "--worker-source-path",
        str(Path(str(source["source_path"])).resolve()),
        "--worker-parent-id",
        str(source["parent_id"]),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(args.worker_timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def output_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

        (log_dir / f"{cad_id}.stdout.log").write_text(
            output_text(exc.stdout), encoding="utf-8"
        )
        (log_dir / f"{cad_id}.stderr.log").write_text(
            output_text(exc.stderr), encoding="utf-8"
        )
        return worker_failure_row(
            source, run_signature=run_signature, status="worker_timeout",
            returncode=None, error_type="TimeoutExpired",
            expected_binding=expected_binding,
        )
    except OSError as exc:
        return worker_failure_row(
            source,
            run_signature=run_signature,
            status="worker_process_exit",
            returncode=None,
            error_type=type(exc).__name__,
            expected_binding=expected_binding,
        )
    (log_dir / f"{cad_id}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_dir / f"{cad_id}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    row = parse_worker_result(completed.stdout)
    if completed.returncode != 0:
        return worker_failure_row(
            source, run_signature=run_signature, status="worker_process_exit",
            returncode=int(completed.returncode), error_type="NonzeroWorkerExit",
            expected_binding=expected_binding,
        )
    if row is None:
        return worker_failure_row(
            source, run_signature=run_signature, status="worker_protocol_error",
            returncode=int(completed.returncode), error_type="InvalidWorkerSentinel",
            expected_binding=expected_binding,
        )
    try:
        validate_case_row(
            row,
            source=source,
            run_signature=run_signature,
            expected_binding=expected_binding,
        )
        if source_binding(Path(str(source["source_path"]))) != normalize_binding(expected_binding):
            raise ValueError("source binding changed after worker completion")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return worker_failure_row(
            source, run_signature=run_signature, status="worker_protocol_error",
            returncode=int(completed.returncode), error_type=type(exc).__name__,
            expected_binding=expected_binding,
        )
    row["worker_returncode"] = int(completed.returncode)
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("cad_id")) for row in rows]
    if len(rows) != len(TARGET_CAD_IDS) or ids != list(TARGET_CAD_IDS):
        raise ValueError("census summary requires the ordered five target rows")
    complete = [
        row
        for row in rows
        if row.get("status") == "completed"
        and row.get("all_faces_observed") is True
        and not any(
            face.get("reason") == "occ_probe_error"
            for face in (row.get("faces") or [])
            if isinstance(face, Mapping)
        )
    ]
    repairable = [
        {"cad_id": str(row["cad_id"]), "face_indices": row.get("repairable_face_indices", [])}
        for row in rows
        if row.get("repairable_face_indices")
    ]
    failure_statuses = {
        "worker_timeout",
        "worker_process_exit",
        "worker_protocol_error",
        "worker_error",
        "probe_error",
        "measurement_incomplete",
        "source_binding_mismatch",
    }
    worker_failures = sum(str(row.get("status", "")).startswith("worker_") for row in rows)
    probe_failures = sum(row.get("status") in {"probe_error", "measurement_incomplete"} for row in rows)
    explicit_failures = sum(row.get("status") in failure_statuses for row in rows)
    conclusive = len(complete) == len(rows) and explicit_failures == 0
    decision = (
        "PROMOTE_TARGETED_REPAIR_PROBE"
        if conclusive and repairable
        else "CLOSE_PERIODIC_PCURVE_ROUTE"
        if conclusive
        else "INCONCLUSIVE_REQUIRES_RERUN"
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "cases": len(rows),
        "completed_cases": len(complete),
        "worker_or_protocol_failures": worker_failures,
        "probe_failures": probe_failures,
        "explicit_failures": explicit_failures,
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "bad_face_count": sum(len(row.get("bad_face_indices") or []) for row in rows),
        "periodic_bad_face_count": sum(len(row.get("periodic_bad_face_indices") or []) for row in rows),
        "repairable_bad_face_count": sum(len(item["face_indices"]) for item in repairable),
        "repairable_cases": repairable,
        "conclusive": conclusive,
        "decision": decision,
        "assembly_release_gate_before": {"strict_valid": 91, "required": 95},
        "authorizes_full_100cad": False,
        "authorizes_boundary_or_ar": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--selector-matrix", type=Path, required=True)
    parser.add_argument("--selector-run", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-cad-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-signature", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-binding-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-parent-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.joint_iterations < 0:
        parser.error("--joint-iterations must be non-negative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibration_rows = read_jsonl(args.calibration_manifest)
    selector_rows = read_jsonl(args.selector_matrix)
    sources = select_census_sources(
        calibration_rows,
        selector_rows,
        target_ids=TARGET_CAD_IDS,
    )
    if args.worker_cad_id is not None:
        if (
            not args.worker_run_signature
            or not args.worker_source_binding_json
            or args.worker_source_path is None
            or not args.worker_parent_id
        ):
            raise SystemExit(
                "worker mode requires run signature, source binding, source path, and parent id"
            )
        matches = [row for row in sources if str(row["cad_id"]) == args.worker_cad_id]
        if len(matches) != 1:
            raise SystemExit("worker CAD id is not unique in target cohort")
        worker_source = dict(matches[0])
        if (
            Path(str(worker_source["source_path"])).resolve()
            != args.worker_source_path.resolve()
            or worker_source.get("parent_id") != args.worker_parent_id
        ):
            raise SystemExit("worker source identity mismatches signed parent arguments")
        try:
            expected = normalize_binding(json.loads(args.worker_source_binding_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"worker source binding argument is invalid: {type(exc).__name__}"
            ) from exc
        try:
            row = run_worker(
                worker_source, breparg_root=args.breparg_root,
                joint_iterations=args.joint_iterations, expected_binding=expected,
                run_signature=args.worker_run_signature,
            )
        except Exception as exc:
            row = worker_failure_row(
                worker_source, run_signature=args.worker_run_signature,
                status="worker_error", returncode=0, error_type=type(exc).__name__,
                expected_binding=expected,
            )
        print(
            WORKER_MARKER
            + json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False),
            flush=True,
        )
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_writer_lock(output_dir):
        payload = build_run_payload(args, sources, selector_rows)
        signature = canonical_sha256(payload)
        run_path = output_dir / RUN_MANIFEST_NAME
        if run_path.is_file():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if (
                existing.get("schema") != RUN_SCHEMA
                or existing.get("signature") != signature
                or existing.get("payload") != payload
            ):
                raise RuntimeError("output directory belongs to a different signed census")
        else:
            unexpected = [
                path
                for path in output_dir.iterdir()
                if path.name not in {RUN_MANIFEST_NAME, LOCK_NAME}
            ]
            if unexpected:
                raise RuntimeError("unsigned census output directory is not empty")
            atomic_json(
                run_path,
                {"schema": RUN_SCHEMA, "signature": signature, "payload": payload, "status": "RUNNING"},
            )
        rows_path = output_dir / ROWS_NAME
        rows = read_jsonl(rows_path)
        done = {str(row.get("cad_id")) for row in rows}
        if len(done) != len(rows) or not done.issubset(set(TARGET_CAD_IDS)):
            raise RuntimeError("existing census rows do not match the signed target cohort")
        bindings = {item["cad_id"]: item for item in payload["source_bindings"]}
        sources_by_id = {str(source["cad_id"]): source for source in sources}
        for row in rows:
            cad_id = str(row.get("cad_id"))
            expected_binding = {
                key: bindings[cad_id][key] for key in ("bytes", "sha256")
            }
            validate_case_row(
                row,
                source=sources_by_id[cad_id],
                run_signature=signature,
                expected_binding=expected_binding,
            )
        validate_bound_inputs(
            args,
            payload=payload,
            sources=sources,
            selector_rows=selector_rows,
        )
        for source in sources:
            cad_id = str(source["cad_id"])
            if cad_id in done:
                continue
            row = run_isolated(
                source, args=args, run_signature=signature,
                expected_binding={key: bindings[cad_id][key] for key in ("bytes", "sha256")},
            )
            append_jsonl(rows_path, row)
            rows.append(row)
            done.add(cad_id)
            print(json.dumps({key: row.get(key) for key in ("cad_id", "status", "bad_face_indices", "repairable_face_indices")}, sort_keys=True), flush=True)
        by_id = {str(row["cad_id"]): row for row in rows}
        ordered = [by_id[cad_id] for cad_id in TARGET_CAD_IDS]
        validate_bound_inputs(
            args,
            payload=payload,
            sources=sources,
            selector_rows=selector_rows,
        )
        for row in ordered:
            cad_id = str(row["cad_id"])
            validate_case_row(
                row,
                source=sources_by_id[cad_id],
                run_signature=signature,
                expected_binding={
                    key: bindings[cad_id][key] for key in ("bytes", "sha256")
                },
            )
        summary = summarize(ordered)
        atomic_json(output_dir / SUMMARY_NAME, summary)
        record = json.loads(run_path.read_text(encoding="utf-8"))
        record.update(
            status="COMPLETED" if summary["conclusive"] else "INCONCLUSIVE",
            attempts=len(ordered),
            rows_sha256=sha256_file(rows_path),
            summary_sha256=sha256_file(output_dir / SUMMARY_NAME),
        )
        atomic_json(run_path, record)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["conclusive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
