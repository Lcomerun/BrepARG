"""Archive a completed exact-CAD four-cell run as Git-safe evidence.

The formal runner keeps source pickles, promoted STEP files, and raw worker
stdout/stderr outside Git.  This module revalidates the signed four-cell
decision and writes only provenance hashes, compact validity measurements,
and the causal gates needed to review why each candidate was rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .probe_downstream_bad_wire_lineage import assert_path_free_evidence
    from .probe_periodic_pcurve_applicability import (
        canonical_sha256,
        normalize_binding,
        sha256_file,
    )
    from .run_exact_cad_repair_feasibility import (
        RUN_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        VARIANTS,
        summarize,
        validate_attempt_row,
        validate_saved_step,
        validate_terminal_artifact_hashes,
    )
except ImportError:  # pragma: no cover - direct script execution
    from probe_downstream_bad_wire_lineage import assert_path_free_evidence
    from probe_periodic_pcurve_applicability import (
        canonical_sha256,
        normalize_binding,
        sha256_file,
    )
    from run_exact_cad_repair_feasibility import (
        RUN_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        VARIANTS,
        summarize,
        validate_attempt_row,
        validate_saved_step,
        validate_terminal_artifact_hashes,
    )


ARCHIVE_SCHEMA = "exact-cad-repair-feasibility-archive-v1"
ARCHIVE_ATTEMPT_SCHEMA = "exact-cad-repair-feasibility-archive-attempt-v1"
ARCHIVE_RUN_SCHEMA = "exact-cad-repair-feasibility-archive-run-v1"

EXPECTED_REPORT_FILES = {
    "README.md",
    "archive_validation.json",
    "artifact_manifest.json",
    RUN_NAME,
    ROWS_NAME,
    SUMMARY_NAME,
}
FORBIDDEN_SUFFIXES = {
    ".step",
    ".stp",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".log",
}
REQUIRED_SOURCE_HASHES = {
    "tools/run_exact_cad_repair_feasibility.py",
    "tools/targeted_nonperiodic_pcurve_repair.py",
    "tools/post_sewing_graph_repair.py",
    "tools/run_assembly_repair_matrix.py",
    "tools/assembly_selector_geometry.py",
    "tools/probe_downstream_bad_wire_lineage.py",
    "tools/directed_trim_assembly.py",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
NATIVE_HANDLE_PATTERN = re.compile(
    r"(?:OCC\.Core|TopoDS_|SwigPyObject|<[^>]+ object at 0x[0-9a-f]+>)",
    re.IGNORECASE,
)
PRIVATE_HANDLE_KEYS = {
    "shape",
    "face",
    "edge",
    "wire",
    "observed_shape",
    "observed_face",
    "observed_edge",
    "observed_wire",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON artifact {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {Path(path).name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSONL artifact {Path(path).name}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"JSONL rows must be objects in {Path(path).name}")
    return values


def _write_text_lf(path: Path, value: str) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, value: Any) -> None:
    _write_text_lf(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_lf(
        path,
        "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value.lower()):
        raise RuntimeError(f"{label} is not a SHA-256 digest")
    return value.lower()


def _assert_no_native_handles(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.startswith("_") or key_text.lower() in PRIVATE_HANDLE_KEYS:
                raise RuntimeError(f"{label} contains private/native field {key_text!r}")
            _assert_no_native_handles(child, label=f"{label}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_no_native_handles(child, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and NATIVE_HANDLE_PATTERN.search(value):
        raise RuntimeError(f"{label} contains a serialized native handle")
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")


def _assert_git_safe_json(value: Any, *, label: str) -> None:
    try:
        assert_path_free_evidence(value, label=label)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not path-free") from exc
    _assert_no_native_handles(value, label=label)
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc


def _compact_mapping_gate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    curve_bindings = value.get("curve_bindings")
    binding_summary = None
    if isinstance(curve_bindings, list):
        deltas = [
            item.get("max_sample_delta")
            for item in curve_bindings
            if isinstance(item, Mapping)
            and isinstance(item.get("max_sample_delta"), (int, float))
        ]
        binding_summary = {
            "count": len(curve_bindings),
            "accepted_count": sum(
                isinstance(item, Mapping) and item.get("accepted") is True
                for item in curve_bindings
            ),
            "max_sample_delta": max(deltas) if deltas else None,
            "source_edge_ids": [
                item.get("source_edge_id")
                for item in curve_bindings
                if isinstance(item, Mapping)
            ],
        }
    result = {
        key: value.get(key)
        for key in (
            "accepted",
            "mapping_status",
            "upstream_mapping_status",
            "reason",
            "edge_occurrence_count",
            "wire_count",
            "source_edge_ids",
            "wire_source_edge_ids",
        )
        if key in value
    }
    if binding_summary is not None:
        result["curve_binding_summary"] = binding_summary
    return result


def _compact_target_selection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = {
        key: value.get(key)
        for key in (
            "accepted",
            "reason",
            "target_count",
            "source_edge_pairs",
            "target_source_edge_ids",
        )
        if key in value
    }
    targets = value.get("targets")
    if isinstance(targets, list):
        result["targets"] = [
            {
                key: target.get(key)
                for key in (
                    "wire_index",
                    "kind",
                    "edge_positions",
                    "source_edge_ids",
                    "source_edge_pair",
                )
                if key in target
            }
            for target in targets
            if isinstance(target, Mapping)
        ]
    return result


def _compact_pcurve_operations(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    keys = (
        "source_edge_id",
        "seam",
        "accepted",
        "reason",
        "remove_returned",
        "remove_reported",
        "pcurve_absent_after_remove",
        "add_returned",
        "add_reported",
        "pcurve_present_after_add",
        "curve_3d_preserved",
        "max_curve_3d_sample_delta",
    )
    return [
        {key: value.get(key) for key in keys if key in value}
        for value in values
        if isinstance(value, Mapping)
    ]


def _compact_simple_gate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: value.get(key)
        for key in (
            "accepted",
            "attempted",
            "status",
            "reason",
            "mapping_status",
            "upstream_mapping_status",
            "u_periodic",
            "v_periodic",
            "checked_edge_count",
            "target_source_edge_ids",
            "checks",
            "rejection_reasons",
        )
        if key in value
    }


def _compact_defect_occurrences(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [
        {
            key: value.get(key)
            for key in ("source_face_index", "kind", "status", "source_edge_ids")
            if key in value
        }
        for value in values
        if isinstance(value, Mapping)
    ]


def _compact_whole_cad_gate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = {
        key: value.get(key)
        for key in (
            "accepted",
            "lineage_status",
            "mapping_exact",
            "mapping_failures",
            "target_definition_complete",
            "occurrences_complete",
            "malformed_occurrence_count",
            "final_occurrence_count",
            "target_defects_removed",
            "no_new_non_target_defects",
        )
        if key in value
    }
    result["target_residuals"] = _compact_defect_occurrences(
        value.get("target_residuals")
    )
    result["non_target_defects"] = _compact_defect_occurrences(
        value.get("non_target_defects")
    )
    proof = value.get("geometry_incidence_proof")
    if isinstance(proof, Mapping):
        result["geometry_incidence_proof"] = {
            key: proof.get(key)
            for key in (
                "status",
                "failure_codes",
                "tolerance_normalized",
                "mapped_face_count",
                "mapped_edge_occurrence_count",
                "face_matching_count_capped",
            )
            if key in proof
        }
    return result


def _compact_face_mutation(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value.get(key)
        for key in (
            "source_face_index",
            "attempted",
            "accepted",
            "reason",
            "strategy",
        )
        if key in value
    }
    for source_key, output_key in (
        ("surface_gate", "surface_gate"),
        ("edge_preflight", "edge_preflight"),
        ("copy_gate", "copy_gate"),
    ):
        compact = _compact_simple_gate(value.get(source_key))
        if compact is not None:
            result[output_key] = compact
    for source_key in ("source_mapping_gate", "candidate_mapping_gate"):
        compact = _compact_mapping_gate(value.get(source_key))
        if compact is not None:
            result[source_key] = compact
    selection = _compact_target_selection(value.get("target_selection"))
    if selection is not None:
        result["target_selection"] = selection
    result["pcurve_operations"] = _compact_pcurve_operations(value.get("surgery"))
    return result


def _compact_post_sewing(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value.get(key)
        for key in (
            "attempted",
            "accepted",
            "reason",
            "strategy",
            "projection_precision",
            "target_source_face_index",
            "target_source_edge_ids",
            "target_face_before",
        )
        if key in value
    }
    selection = _compact_target_selection(value.get("target_selection"))
    if selection is not None:
        result["target_selection"] = selection
    for source_key in ("copy_topology_gate", "copy_source_edge_identity_gate"):
        gate = value.get(source_key)
        if isinstance(gate, Mapping):
            result[source_key] = {
                key: gate.get(key)
                for key in ("accepted", "checks", "rejection_reasons")
                if key in gate
            }
    reprojection = value.get("pcurve_reprojection")
    if isinstance(reprojection, Mapping):
        result["pcurve_reprojection"] = {
            "accepted": reprojection.get("accepted"),
            "reason": reprojection.get("reason"),
            "operations": _compact_pcurve_operations(reprojection.get("operations")),
        }
    return result


def _compact_candidate_application(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value.get(key)
        for key in ("attempted", "applied", "status")
        if key in value
    }
    diagnostics = value.get("diagnostics")
    causal: dict[str, Any] = {}
    if isinstance(diagnostics, Mapping):
        mutations = diagnostics.get("face_mutations")
        if isinstance(mutations, list):
            causal["targeted_face_mutations"] = [
                _compact_face_mutation(item)
                for item in mutations
                if isinstance(item, Mapping) and item.get("attempted") is True
            ]
        local_gates = diagnostics.get("local_face_gates")
        if isinstance(local_gates, Mapping):
            causal["local_face_gates"] = dict(local_gates)
        whole = _compact_whole_cad_gate(diagnostics.get("whole_cad_step_gate"))
        if whole is not None:
            causal["whole_cad_step_gate"] = whole
        post = diagnostics.get("post_sewing_mutation")
        if isinstance(post, Mapping):
            causal["post_sewing_mutation"] = _compact_post_sewing(post)
    if value.get("attempted") is True and not causal:
        raise RuntimeError("candidate application lacks compact causal evidence")
    if causal:
        result["causal_evidence"] = causal
    _assert_git_safe_json(result, label="compact_candidate_application")
    return result


def _compact_geometry_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "schema",
        "accepted",
        "rejection_reasons",
        "checks",
        "thresholds",
        "bbox_relative_delta",
        "edge_length_relative_delta",
        "input_to_candidate_rms_normalized",
        "input_to_candidate_max_normalized",
        "candidate_to_input_rms_normalized",
        "candidate_to_input_max_normalized",
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
        "input_to_candidate_projection_failure_count",
        "candidate_to_input_projection_failure_count",
        "candidate_curve_sampling_failure_count",
    )
    return {key: value.get(key) for key in scalar_keys if key in value}


def compact_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    application = row.get("candidate_application")
    defect = row.get("defect_gate")
    geometry = row.get("geometry_topology_gate")
    components = row.get("validity_components")
    if not all(isinstance(value, Mapping) for value in (application, defect, geometry)):
        raise RuntimeError("attempt lacks compactable decision gates")
    result: dict[str, Any] = {
        "schema": ARCHIVE_ATTEMPT_SCHEMA,
        "source_schema": row.get("schema"),
        "task_id": row.get("task_id"),
        "cad_id": row.get("cad_id"),
        "parent_id": row.get("parent_id"),
        "arm": row.get("arm"),
        "callback_ref": row.get("callback_ref"),
        "run_signature": row.get("run_signature"),
        "denominator": row.get("denominator"),
        "historical_strict_valid": row.get("historical_strict_valid"),
        "status": row.get("status"),
        "callback_completed": row.get("callback_completed"),
        "worker_returncode": row.get("worker_returncode"),
        "step_artifact": {
            "saved": row.get("step_saved"),
            "readable": row.get("step_readable"),
            "bytes": row.get("step_bytes"),
            "sha256": row.get("step_sha256"),
            "bytes_archived": False,
            "path_archived": False,
        },
        "validity": {
            "step_readable": row.get("step_readable"),
            "occ_native": row.get("native_brep_valid"),
            "project_strict": row.get("strict_brep_valid"),
            "both_valid": row.get("both_valid"),
        },
        "source_binding_chain": {
            "expected": row.get("source_binding_expected"),
            "before": row.get("source_binding_before"),
            "loaded_bytes": row.get("source_binding_loaded_bytes"),
            "after_load": row.get("source_binding_after_load"),
            "after_attempt": row.get("source_binding_after_attempt"),
            "all_equal": len(
                {
                    json.dumps(row.get(name), sort_keys=True)
                    for name in (
                        "source_binding_expected",
                        "source_binding_before",
                        "source_binding_loaded_bytes",
                        "source_binding_after_load",
                        "source_binding_after_attempt",
                    )
                }
            )
            == 1,
            "pickle_bytes_archived": False,
            "path_archived": False,
        },
        "candidate_application": _compact_candidate_application(application),
        "defect_gate": dict(defect),
        "geometry_topology_gate": _compact_geometry_gate(geometry),
        "raw_worker_logs_archived": False,
        "native_handles_archived": False,
    }
    if isinstance(components, Mapping):
        result["validity_components"] = {
            key: components.get(key)
            for key in (
                "status",
                "native_brep_valid",
                "wire_count",
                "wire_order_failures",
                "wire_self_intersections",
                "free_edges",
                "shell_count",
                "shells_with_bad_edges",
                "solid_count",
            )
            if key in components
        }
    control = row.get("control_expectation")
    if isinstance(control, Mapping):
        result["control_expectation"] = dict(control)
    _assert_git_safe_json(result, label="compact_attempt")
    return result


def _validate_repository(payload: Mapping[str, Any]) -> dict[str, Any]:
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise RuntimeError("signed repository binding is missing")
    commit = repository.get("commit")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit.lower()):
        raise RuntimeError("signed repository commit is invalid")
    if repository.get("dirty") is not False:
        raise RuntimeError("formal exact-CAD run did not use a clean worktree")
    status_sha = _require_sha256(
        repository.get("status_sha256"), "repository status binding"
    )
    source_hashes = repository.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != REQUIRED_SOURCE_HASHES:
        raise RuntimeError("signed source hash population is incomplete or unexpected")
    normalized = {}
    for name, digest in sorted(source_hashes.items()):
        path = Path(str(name))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("signed source hash has an unsafe repository path")
        normalized[str(name).replace("\\", "/")] = _require_sha256(
            digest, f"source hash {name}"
        )
    return {
        "commit": commit.lower(),
        "dirty": False,
        "status_sha256": status_sha,
        "source_sha256": normalized,
    }


def _validate_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    if payload.get("schema") != RUN_SCHEMA:
        raise RuntimeError("signed payload schema is not the exact-CAD v1 protocol")
    expected_tasks = [variant.task_id for variant in VARIANTS]
    if payload.get("ordered_cad_ids") != list(TARGET_CAD_IDS):
        raise RuntimeError("signed exact-CAD cohort order drifted")
    if payload.get("ordered_task_ids") != expected_tasks:
        raise RuntimeError("signed four-cell task order drifted")
    expected_variants = json.loads(json.dumps([asdict(value) for value in VARIANTS]))
    if payload.get("variants") != expected_variants:
        raise RuntimeError("signed exact-CAD variant contract drifted")
    if payload.get("joint_iterations") != 200:
        raise RuntimeError("signed exact-CAD joint-optimization protocol drifted")
    timeout = payload.get("worker_timeout_seconds")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise RuntimeError("signed worker timeout is invalid")

    for key in ("calibration_manifest_sha256", "selector_matrix_sha256"):
        _require_sha256(payload.get(key), f"signed input {key}")
    selector = payload.get("selector_run")
    if not isinstance(selector, Mapping) or selector.get("status") != "COMPLETED":
        raise RuntimeError("signed selector run binding is missing or incomplete")
    if type(selector.get("bytes")) is not int or selector["bytes"] <= 0:
        raise RuntimeError("signed selector run byte count is invalid")
    for key in ("sha256", "signature"):
        _require_sha256(selector.get(key), f"selector run {key}")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("signed lineage binding is missing")
    for key in ("cases_sha256", "run_sha256", "run_signature"):
        _require_sha256(lineage.get(key), f"lineage {key}")
    runtime = payload.get("breparg_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("signed upstream runtime binding is missing")
    _require_sha256(runtime.get("utils_sha256"), "upstream runtime utils binding")

    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != len(TARGET_CAD_IDS):
        raise RuntimeError("signed source inventory does not contain two CADs")
    source_by_id: dict[str, dict[str, Any]] = {}
    bindings = []
    for expected_cad, source in zip(TARGET_CAD_IDS, sources):
        if not isinstance(source, Mapping) or source.get("cad_id") != expected_cad:
            raise RuntimeError("signed source inventory order or identity drifted")
        parent = source.get("parent_id")
        if not isinstance(parent, str) or not parent:
            raise RuntimeError("signed source parent identity is missing")
        if source.get("historical_strict_valid") is not False:
            raise RuntimeError("exact-CAD target is not a historical invalid residual")
        if source.get("selector_strict_valid") is not False:
            raise RuntimeError("exact-CAD target is not a selector residual")
        try:
            binding = normalize_binding(source.get("binding"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("signed source binding is malformed") from exc
        source_by_id[expected_cad] = {
            "cad_id": expected_cad,
            "parent_id": parent,
            "brep_valid": False,
            "binding": binding,
        }
        bindings.append({"cad_id": expected_cad, **binding})
    if lineage.get("source_bindings") != bindings:
        raise RuntimeError("lineage and exact-CAD source bindings disagree")
    repository = _validate_repository(payload)
    _assert_git_safe_json(payload, label="signed_payload")
    return repository, source_by_id, dict(lineage)


def _readme(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    result_lines = []
    for row in rows:
        application = row.get("candidate_application") or {}
        result_lines.append(
            f"- `{row['task_id']}`: status `{row['status']}`; STEP readable "
            f"`{str(row['step_readable']).lower()}`; native "
            f"`{str(row['native_brep_valid']).lower()}`; strict "
            f"`{str(row['strict_brep_valid']).lower()}`; candidate attempted "
            f"`{str(application.get('attempted')).lower()}`; applied "
            f"`{str(application.get('applied')).lower()}`."
        )
    return f"""# Exact-CAD repair feasibility decision

This directory is the compact Git-safe snapshot of the signed four-cell
control/candidate experiment for CADs 47472 and 63055. The archive was emitted
only after rechecking the immutable run signature, terminal ledger and summary
hashes, all four source-binding chains, every promoted STEP hash, runner-level
row validation, and an exact summary recomputation.

## Result

- Decision: `{summary['decision']}`
- Controls reproduced: `{summary['controls_reproduced']}/2`
- Candidates rejected: `{len(summary['candidate_rejected_cad_ids'])}/2`
- Worker/protocol failures: `{summary['worker_or_protocol_failures']}`
- Non-finite observations: `{summary['nonfinite_count']}`
- Existing assembly gate: `91/100`; release requirement: `95/100`

{chr(10).join(result_lines)}

Both registered candidates were actually invoked but neither local OCC helper
could prove application. The compact attempt ledger retains the target
selection, pcurve remove/add outcome, whole-CAD defect gate, strict/native
validity, and schema-v2 geometry/topology rejection evidence needed to review
that negative result.

No STEP or pickle bytes, worker stdout/stderr, machine-local path, upstream
source tree, NumPy array, checkpoint, or OCC/native handle is archived. STEP,
source, input, runtime, and code identities are retained only as byte counts
and cryptographic hashes. This negative two-CAD feasibility result closes only
the two registered `FixRemovePCurve`-then-`FixAddPCurve` surgery
implementations at their signed precision and stage. It is not evidence
against other pcurve construction or replacement mechanisms, and it does not
authorize residual-family expansion, a 100-CAD promotion, full training,
sequence generation, or AR.
"""


def _artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(Path(report_dir).iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"archive text is not canonical LF: {path.name}")
        values.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return values


def _validate_report_inventory(report_dir: Path) -> None:
    inventory = list(Path(report_dir).iterdir())
    if {path.name for path in inventory} != EXPECTED_REPORT_FILES:
        raise RuntimeError("archive report inventory differs from its allowlist")
    if any(not path.is_file() for path in inventory):
        raise RuntimeError("archive report contains a directory")
    forbidden = [
        path.name
        for path in inventory
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
        or "worker" in path.name.lower()
        or "stdout" in path.name.lower()
        or "stderr" in path.name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"archive report contains forbidden artifacts: {forbidden}")


def snapshot(run_root: Path, report_dir: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    sources = {
        ROWS_NAME: run_root / ROWS_NAME,
        SUMMARY_NAME: run_root / SUMMARY_NAME,
        RUN_NAME: run_root / RUN_NAME,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise RuntimeError(f"completed exact-CAD artifact is missing: {name}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise RuntimeError("report directory must be empty")

    rows = _read_jsonl(sources[ROWS_NAME])
    summary = _read_json(sources[SUMMARY_NAME])
    run = _read_json(sources[RUN_NAME])
    payload = run.get("payload")
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "COMPLETED"
        or run.get("attempts") != 4
        or not isinstance(payload, Mapping)
    ):
        raise RuntimeError("exact-CAD archive requires a completed four-cell run")
    signature = _require_sha256(run.get("signature"), "run signature")
    if signature != canonical_sha256(payload):
        raise RuntimeError("exact-CAD run signature does not bind its payload")
    terminal_summary = validate_terminal_artifact_hashes(
        run, rows_path=sources[ROWS_NAME], summary_path=sources[SUMMARY_NAME]
    )
    if terminal_summary != summary:
        raise RuntimeError("terminal summary reload is inconsistent")

    repository, source_by_id, lineage = _validate_payload(payload)
    expected_tasks = [variant.task_id for variant in VARIANTS]
    if len(rows) != 4 or [row.get("task_id") for row in rows] != expected_tasks:
        raise RuntimeError("exact-CAD archive requires four ordered denominator rows")
    for row, variant in zip(rows, VARIANTS):
        source = source_by_id[variant.cad_id]
        binding = source["binding"]
        if row.get("schema") != SCHEMA:
            raise RuntimeError("exact-CAD attempt schema drifted")
        try:
            validate_attempt_row(
                row,
                source=source,
                variant=variant,
                run_signature=signature,
                expected_binding=binding,
            )
            validate_saved_step(row, output_dir=run_root)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"attempt validation failed for {variant.task_id}") from exc
        for name in (
            "source_binding_expected",
            "source_binding_before",
            "source_binding_loaded_bytes",
            "source_binding_after_load",
            "source_binding_after_attempt",
        ):
            if row.get(name) != binding:
                raise RuntimeError(f"source binding chain failed for {variant.task_id}")

    recomputed = summarize(rows)
    if recomputed != summary:
        raise RuntimeError("exact-CAD summary is not exactly derivable from its rows")
    controls = [row for row in rows if row.get("arm") == "control"]
    candidates = [row for row in rows if row.get("arm") == "candidate"]
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("conclusive") is not True
        or summary.get("decision") != "CLOSE_EXACT_CAD_CANDIDATES"
        or summary.get("denominator_rows") != 4
        or summary.get("controls_reproduced") != 2
        or summary.get("control_drift") != 0
        or summary.get("candidate_callbacks_complete") != 2
        or summary.get("candidate_hooks_unavailable") != 0
        or summary.get("candidate_accepted_cad_ids") != []
        or summary.get("candidate_rejected_cad_ids") != list(TARGET_CAD_IDS)
        or summary.get("worker_or_protocol_failures") != 0
        or summary.get("nonfinite_count") != 0
        or [row.get("status") for row in controls] != ["control_reproduced"] * 2
        or [row.get("status") for row in candidates] != ["candidate_rejected"] * 2
    ):
        raise RuntimeError("formal exact-CAD result is not a zero-failure negative decision")
    for row in candidates:
        application = row.get("candidate_application") or {}
        defect = row.get("defect_gate") or {}
        if (
            application.get("attempted") is not True
            or application.get("applied") is not False
            or defect.get("accepted") is not False
            or defect.get("nonfinite_count") != 0
            or not defect.get("rejection_reasons")
        ):
            raise RuntimeError("candidate rejection lacks attempted, finite, causal evidence")

    compact_rows = [compact_attempt(row) for row in rows]
    compact_summary = dict(summary)
    _assert_git_safe_json(compact_summary, label="compact_summary")
    compact_payload = dict(payload)
    compact_payload["repository"] = repository
    _assert_git_safe_json(compact_payload, label="compact_payload")
    source_artifact_bindings = {
        name: sha256_file(path) for name, path in sources.items()
    }
    compact_run = {
        "schema": ARCHIVE_RUN_SCHEMA,
        "source_schema": RUN_SCHEMA,
        "source_status": "COMPLETED",
        "attempts": 4,
        "signature": signature,
        "payload": compact_payload,
        "terminal_bindings": {
            "rows_sha256": run.get("rows_sha256"),
            "summary_sha256": run.get("summary_sha256"),
        },
        "source_artifact_bindings": source_artifact_bindings,
        "step_bytes_archived": False,
        "source_pickle_bytes_archived": False,
        "raw_worker_logs_archived": False,
        "native_handles_archived": False,
    }
    _assert_git_safe_json(compact_run, label="compact_run")

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(report_dir / ROWS_NAME, compact_rows)
    _write_json(report_dir / SUMMARY_NAME, compact_summary)
    _write_json(report_dir / RUN_NAME, compact_run)
    _write_text_lf(report_dir / "README.md", _readme(summary, rows))

    validation = {
        "schema": ARCHIVE_SCHEMA,
        "valid": True,
        "attempts": 4,
        "denominator_rows": 4,
        "conclusive": True,
        "decision": summary["decision"],
        "negative_scope": (
            "registered_FixRemovePCurve_then_FixAddPCurve_surgeries_only"
        ),
        "does_not_generalize_to_all_pcurve_mechanisms": True,
        "run_signature": signature,
        "signature_valid": True,
        "terminal_rows_hash_valid": True,
        "terminal_summary_hash_valid": True,
        "summary_recomputed_equal": True,
        "source_binding_chains_valid": True,
        "saved_step_hashes_revalidated": True,
        "repository_binding_valid": True,
        "repository_commit": repository["commit"],
        "repository_source_hash_count": len(repository["source_sha256"]),
        "input_hash_bindings_valid": True,
        "lineage_run_signature": lineage["run_signature"],
        "controls_reproduced": 2,
        "control_drift": 0,
        "candidate_callbacks_complete": 2,
        "candidate_attempted": 2,
        "candidate_applied": 0,
        "candidate_rejected": 2,
        "worker_or_protocol_failures": 0,
        "nonfinite_count": 0,
        "path_free": True,
        "step_bytes_archived": False,
        "source_pickle_bytes_archived": False,
        "raw_worker_logs_archived": False,
        "native_handles_archived": False,
        "source_artifact_bindings": source_artifact_bindings,
    }
    _write_json(report_dir / "archive_validation.json", validation)
    _write_json(
        report_dir / "artifact_manifest.json",
        {"schema": ARCHIVE_SCHEMA, "artifacts": _artifact_manifest(report_dir)},
    )
    _validate_report_inventory(report_dir)
    for path in report_dir.iterdir():
        if path.suffix.lower() == ".jsonl":
            archived: Any = _read_jsonl(path)
        elif path.suffix.lower() == ".json":
            archived = _read_json(path)
        else:
            continue
        _assert_git_safe_json(archived, label=f"archive.{path.name}")
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(snapshot(args.run_root, args.report_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
