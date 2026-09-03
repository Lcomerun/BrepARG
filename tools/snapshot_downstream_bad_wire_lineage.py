"""Create a compact Git-safe snapshot of a completed lineage decision run.

The local protocol intentionally keeps raw worker logs and detailed OCC
diagnostics beside the run.  This tool validates that evidence, then archives
only the signed provenance and the minimum source-face/source-edge lineage
facts needed to review the decision.  STEP and pickle bytes, native handles,
machine paths, and worker stdout/stderr never enter the report directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .probe_downstream_bad_wire_lineage import (
        ALL_PHASES,
        EXACT_LINEAGE_STATUSES,
        PROFILE,
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        assert_path_free_evidence,
        canonical_sha256,
        sha256_file,
        summarize,
        validate_case_row,
    )
except ImportError:  # direct script execution
    from probe_downstream_bad_wire_lineage import (
        ALL_PHASES,
        EXACT_LINEAGE_STATUSES,
        PROFILE,
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        assert_path_free_evidence,
        canonical_sha256,
        sha256_file,
        summarize,
        validate_case_row,
    )


ARCHIVE_SCHEMA = "downstream-bad-wire-lineage-archive-v1"
ARCHIVE_CASE_SCHEMA = "downstream-bad-wire-lineage-archive-case-v1"
ARCHIVE_RUN_SCHEMA = "downstream-bad-wire-lineage-archive-run-v1"
TERMINAL_DECISIONS = {
    "CLOSE_DOWNSTREAM_BAD_WIRE_ROUTE",
    "PROMOTE_TARGETED_NONPERIODIC_REPAIR_PROBE",
}
FAILURE_FIELDS = (
    "worker_or_protocol_failures",
    "source_binding_failures",
    "coverage_failures",
    "observation_failures",
    "mapping_failures",
    "explicit_failures",
)
EXPECTED_REPORT_FILES = {
    "README.md",
    "archive_validation.json",
    "artifact_manifest.json",
    RUN_MANIFEST_NAME,
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
PRIVATE_HANDLE_KEYS = {
    "shape",
    "observed_wire",
    "observed_edge",
    "occurrence_edges",
}
NATIVE_HANDLE_PATTERN = re.compile(
    r"(?:OCC\.Core|TopoDS_|SwigPyObject|<[^>]+ object at 0x[0-9a-f]+>)",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
REQUIRED_SOURCE_HASHES = {
    "tools/probe_downstream_bad_wire_lineage.py",
    "tools/probe_periodic_pcurve_applicability.py",
    "tools/directed_trim_assembly.py",
    "tools/diagnose_assembly_face_wires.py",
    "tools/local_wire_topology_repair.py",
    "tools/assembly_repair.py",
    "tools/run_assembly_calibration_oracle.py",
    "tools/run_assembly_repair_matrix.py",
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
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
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
    """Reject serialized traces of process-private OCC proof objects."""
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
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc


def _validate_repository_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise RuntimeError("signed repository binding is missing")
    commit = repository.get("commit")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit.lower()):
        raise RuntimeError("signed repository commit is invalid")
    if repository.get("dirty") is not False:
        raise RuntimeError("formal lineage run was not bound to a clean worktree")
    status_sha256 = _require_sha256(
        repository.get("status_sha256"), "repository status binding"
    )
    source_hashes = repository.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("signed source hash map is missing")
    if set(source_hashes) != REQUIRED_SOURCE_HASHES:
        raise RuntimeError("signed source hash population is incomplete or unexpected")
    normalized_hashes: dict[str, str] = {}
    for source_name, digest in sorted(source_hashes.items()):
        source_path = Path(str(source_name))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise RuntimeError("signed source hash has an unsafe repository path")
        normalized_hashes[str(source_name).replace("\\", "/")] = _require_sha256(
            digest, f"source hash {source_name}"
        )
    return {
        "commit": commit.lower(),
        "dirty": False,
        "status_sha256": status_sha256,
        "source_sha256": normalized_hashes,
    }


def _validate_source_bindings(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = payload.get("source_bindings")
    if not isinstance(values, list) or len(values) != len(TARGET_CAD_IDS):
        raise RuntimeError("signed source bindings do not contain exactly two cases")
    bindings: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise RuntimeError("signed source binding is not an object")
        cad_id = value.get("cad_id")
        byte_count = value.get("bytes")
        if (
            not isinstance(cad_id, str)
            or type(byte_count) is not int
            or byte_count <= 0
            or cad_id in bindings
        ):
            raise RuntimeError("signed source binding identity or byte count is invalid")
        bindings[cad_id] = {
            "bytes": byte_count,
            "sha256": _require_sha256(value.get("sha256"), f"source {cad_id}"),
        }
    if list(bindings) != list(TARGET_CAD_IDS):
        raise RuntimeError("signed source bindings do not preserve target order")
    return bindings


def _compact_occurrence(
    occurrence: Mapping[str, Any], *, source_face_index: int | None
) -> dict[str, Any]:
    source_ids = occurrence.get("source_edge_ids")
    if not isinstance(source_ids, list) or any(type(value) is not int for value in source_ids):
        raise RuntimeError("mapped occurrence lacks integer source edge ids")
    face_value = occurrence.get("source_face_index", source_face_index)
    if type(face_value) is not int:
        raise RuntimeError("mapped occurrence lacks source face identity")
    result = {
        "source_face_index": face_value,
        "wire_index_observation_label_only": occurrence.get("wire_index"),
        "kind": occurrence.get("kind"),
        "status": occurrence.get("status"),
        "source_mapping_status": occurrence.get("source_mapping_status"),
        "source_edge_ids": list(source_ids),
    }
    return result


def compact_case(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a validated worker row to identity and phase-level evidence."""
    phase_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {phase: [] for phase in ALL_PHASES}
    for observation in row.get("observations") or []:
        if not isinstance(observation, Mapping):
            raise RuntimeError("case observation is not an object")
        phase = observation.get("phase")
        if phase not in grouped:
            raise RuntimeError("case observation has an unregistered phase")
        grouped[str(phase)].append(observation)
    for phase in ALL_PHASES:
        observations = grouped[phase]
        lineage_counts = Counter(str(value.get("lineage_status")) for value in observations)
        diagnosis_counts = Counter(
            str((value.get("diagnosis") or {}).get("status"))
            for value in observations
            if isinstance(value.get("diagnosis"), Mapping)
        )
        defects = []
        geometry_proofs = []
        for observation in observations:
            diagnosis = observation.get("diagnosis")
            if not isinstance(diagnosis, Mapping):
                raise RuntimeError("case observation diagnosis is missing")
            source_face_index = observation.get("source_face_index")
            source_face_index = (
                source_face_index if type(source_face_index) is int else None
            )
            for occurrence in diagnosis.get("occurrences") or []:
                if not isinstance(occurrence, Mapping):
                    raise RuntimeError("diagnosis occurrence is not an object")
                if occurrence.get("source_mapping_status") != "mapped":
                    raise RuntimeError("conclusive case contains an unmapped occurrence")
                defects.append(
                    _compact_occurrence(
                        occurrence, source_face_index=source_face_index
                    )
                )
            proof = diagnosis.get("geometry_incidence_proof")
            if proof is not None:
                if not isinstance(proof, Mapping):
                    raise RuntimeError("geometry-incidence proof is not an object")
                geometry_proofs.append(
                    {
                        key: proof.get(key)
                        for key in (
                            "status",
                            "failure_codes",
                            "tolerance_normalized",
                            "face_candidate_degree_counts",
                            "face_matching_count_capped",
                            "mapped_face_count",
                            "mapped_edge_occurrence_count",
                        )
                        if key in proof
                    }
                )
        phase_rows.append(
            {
                "phase": phase,
                "observation_count": len(observations),
                "exact_lineage_count": sum(
                    value.get("lineage_status") in EXACT_LINEAGE_STATUSES
                    for value in observations
                ),
                "lineage_status_counts": dict(sorted(lineage_counts.items())),
                "diagnosis_status_counts": dict(sorted(diagnosis_counts.items())),
                "mapped_defect_occurrences": defects,
                "geometry_incidence_proofs": geometry_proofs,
            }
        )
    compact = {
        "schema": ARCHIVE_CASE_SCHEMA,
        "source_schema": row.get("schema"),
        "cad_id": row.get("cad_id"),
        "parent_id": row.get("parent_id"),
        "profile": row.get("profile"),
        "run_signature": row.get("run_signature"),
        "status": row.get("status"),
        "assembly_status": row.get("assembly_status"),
        "step_roundtrip_status": row.get("step_roundtrip_status"),
        "source_binding": row.get("source_binding"),
        "source_binding_loaded_bytes": row.get("source_binding_loaded_bytes"),
        "source_binding_after_load": row.get("source_binding_after_load"),
        "source_face_count": row.get("source_face_count"),
        "source_edge_count": row.get("source_edge_count"),
        "phase_counts": row.get("phase_counts"),
        "all_stages_observed": row.get("all_stages_observed"),
        "coverage_failure_count": row.get("coverage_failure_count"),
        "observation_failure_count": row.get("observation_failure_count"),
        "mapping_failure_count": row.get("mapping_failure_count"),
        "mapped_defect_count": row.get("mapped_defect_count"),
        "first_bad_phase": row.get("first_bad_phase"),
        "first_bad_occurrences": row.get("first_bad_occurrences") or [],
        "phase_evidence": phase_rows,
        "source_pickle_bytes_archived": False,
        "step_bytes_archived": False,
        "native_handles_archived": False,
        "worker_logs_archived": False,
    }
    _assert_git_safe_json(compact, label="compact_case")
    return compact


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    repository = _validate_repository_binding(payload)
    # Preserve the complete signed payload: it is already path-free, and its
    # canonical bytes are required to independently recheck the run signature.
    result = dict(payload)
    result["repository"] = repository
    _assert_git_safe_json(result, label="signed_payload")
    return result


def _readme(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    target_lines = "\n".join(
        f"- `{row['cad_id']}`: first bad phase = "
        f"`{row.get('first_bad_phase') or 'none'}`; mapped defects = "
        f"`{row.get('mapped_defect_count')}`."
        for row in rows
    )
    return f"""# Downstream bad-wire lineage decision

This directory is the compact Git-safe snapshot of one signed two-CAD
downstream-lineage run. The local run was accepted only after both isolated
workers completed, every construction and STEP-roundtrip phase had complete
source-face/source-edge lineage, and all protocol, binding, coverage,
observation, and mapping failure counts were zero.

## Result

- Decision: `{summary['decision']}`
- Conclusive cases: `{summary['completed_cases']}/{summary['cases']}`
- Assembly release gate before this probe: `91/100` strict-valid; required:
  `95/100`

{target_lines}

The case ledger contains phase-level counts and mapped defect occurrences only.
Explorer positions, when retained, are explicitly observation labels and are
not source identity. The signed run payload binds the clean repository commit,
every relevant source-file SHA-256, the two source-pickle SHA-256 values, the
selector evidence, and the upstream runtime hash.

No STEP or pickle payload, OCC/native handle, reconstructed array, checkpoint,
worker stdout/stderr, machine-local path, or upstream source tree is archived.
This snapshot records the preregistered two-CAD decision only; it does not by
itself authorize a full 100-CAD run, boundary-loss training, sequence work, or
AR training.
"""


def _artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(Path(report_dir).iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"archive text is not canonical LF: {path.name}")
        result.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return result


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
        RUN_MANIFEST_NAME: run_root / RUN_MANIFEST_NAME,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise RuntimeError(f"completed lineage artifact is missing: {name}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise RuntimeError("report directory must be empty")

    rows = _read_jsonl(sources[ROWS_NAME])
    summary = _read_json(sources[SUMMARY_NAME])
    run = _read_json(sources[RUN_MANIFEST_NAME])
    # Reject unsafe material before invoking the protocol validators.  This
    # ensures even a field that the compact schema would omit cannot enter or
    # influence a Git-safe snapshot.
    for label, value in (("run", run), ("summary", summary), ("cases", rows)):
        _assert_git_safe_json(value, label=label)
    payload = run.get("payload")
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "COMPLETED"
        or run.get("attempts") != len(TARGET_CAD_IDS)
        or not isinstance(payload, Mapping)
        or payload.get("schema") != RUN_SCHEMA
    ):
        raise RuntimeError("lineage run must be a completed two-case v1 run")
    if run.get("signature") != canonical_sha256(payload):
        raise RuntimeError("lineage run signature does not bind its payload")
    signature = str(run["signature"])
    _require_sha256(signature, "run signature")
    if sha256_file(sources[ROWS_NAME]) != run.get("rows_sha256"):
        raise RuntimeError("lineage case ledger hash differs from completed run")
    if sha256_file(sources[SUMMARY_NAME]) != run.get("summary_sha256"):
        raise RuntimeError("lineage summary hash differs from completed run")
    if len(rows) != len(TARGET_CAD_IDS) or [row.get("cad_id") for row in rows] != list(
        TARGET_CAD_IDS
    ):
        raise RuntimeError("lineage archive requires the ordered frozen two-CAD cohort")
    if payload.get("ordered_cad_ids") != list(TARGET_CAD_IDS):
        raise RuntimeError("signed payload target order differs from the case ledger")
    if payload.get("profile") != PROFILE:
        raise RuntimeError("signed lineage profile differs from the fixed profile")

    bindings = _validate_source_bindings(payload)
    repository = _validate_repository_binding(payload)
    runtime = payload.get("breparg_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("upstream runtime hash binding is missing")
    _require_sha256(runtime.get("utils_sha256"), "upstream runtime utils binding")
    for key in (
        "calibration_manifest_sha256",
        "selector_matrix_sha256",
        "selector_cohort_signature",
    ):
        _require_sha256(payload.get(key), f"signed payload {key}")
    selector_run = payload.get("selector_run")
    if not isinstance(selector_run, Mapping) or selector_run.get("status") != "COMPLETED":
        raise RuntimeError("signed selector run binding is missing or incomplete")
    for key in ("sha256", "signature"):
        _require_sha256(selector_run.get(key), f"selector run {key}")

    parents: dict[str, str] = {}
    for row in rows:
        cad_id = str(row.get("cad_id"))
        if row.get("schema") != SCHEMA or row.get("status") != "completed":
            raise RuntimeError("lineage archive requires two completed case rows")
        parent_id = row.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise RuntimeError("lineage case parent identity is missing")
        parents[cad_id] = parent_id
        validate_case_row(
            row,
            source={"cad_id": cad_id, "parent_id": parent_id},
            run_signature=signature,
            expected_binding=bindings[cad_id],
        )
        if row.get("all_stages_observed") is not True or any(
            row.get(field) != 0
            for field in (
                "coverage_failure_count",
                "observation_failure_count",
                "mapping_failure_count",
            )
        ):
            raise RuntimeError("lineage case contains incomplete or failed evidence")

    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("conclusive") is not True
        or summary.get("cases") != len(TARGET_CAD_IDS)
        or summary.get("completed_cases") != len(TARGET_CAD_IDS)
        or summary.get("decision") not in TERMINAL_DECISIONS
        or any(summary.get(field) != 0 for field in FAILURE_FIELDS)
    ):
        raise RuntimeError("lineage summary is not a zero-failure conclusive decision")
    if summarize(rows) != summary:
        raise RuntimeError("lineage summary is not exactly derivable from its cases")

    compact_payload = _compact_payload(payload)
    compact_rows = [compact_case(row) for row in rows]
    compact_summary = dict(summary)
    _assert_git_safe_json(compact_summary, label="compact_summary")
    compact_run = {
        "schema": ARCHIVE_RUN_SCHEMA,
        "source_schema": RUN_SCHEMA,
        "source_status": "COMPLETED",
        "attempts": len(rows),
        "signature": signature,
        "payload": compact_payload,
        "source_artifact_bindings": {
            ROWS_NAME: sha256_file(sources[ROWS_NAME]),
            SUMMARY_NAME: sha256_file(sources[SUMMARY_NAME]),
            RUN_MANIFEST_NAME: sha256_file(sources[RUN_MANIFEST_NAME]),
        },
        "source_pickle_bytes_archived": False,
        "step_bytes_archived": False,
        "native_handles_archived": False,
        "worker_logs_archived": False,
    }
    _assert_git_safe_json(compact_run, label="compact_run")

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(report_dir / ROWS_NAME, compact_rows)
    _write_json(report_dir / SUMMARY_NAME, compact_summary)
    _write_json(report_dir / RUN_MANIFEST_NAME, compact_run)
    _write_text_lf(report_dir / "README.md", _readme(summary, rows))

    validation = {
        "schema": ARCHIVE_SCHEMA,
        "valid": True,
        "cases": len(rows),
        "completed_cases": len(rows),
        "conclusive": True,
        "decision": summary["decision"],
        "failure_counts": {field: 0 for field in FAILURE_FIELDS},
        "run_signature": signature,
        "signature_valid": True,
        "rows_binding_valid": True,
        "summary_binding_valid": True,
        "summary_recomputed_equal": True,
        "source_bindings_valid": True,
        "repository_binding_valid": True,
        "repository_commit": repository["commit"],
        "repository_source_hash_count": len(repository["source_sha256"]),
        "path_free": True,
        "source_pickle_bytes_archived": False,
        "step_bytes_archived": False,
        "native_handles_archived": False,
        "worker_logs_archived": False,
        "source_artifact_bindings": compact_run["source_artifact_bindings"],
    }
    _write_json(report_dir / "archive_validation.json", validation)
    _write_json(
        report_dir / "artifact_manifest.json",
        {"schema": ARCHIVE_SCHEMA, "artifacts": _artifact_manifest(report_dir)},
    )
    _validate_report_inventory(report_dir)
    for path in report_dir.iterdir():
        if path.suffix.lower() in {".json", ".jsonl"}:
            if path.suffix.lower() == ".jsonl":
                archived_value: Any = _read_jsonl(path)
            else:
                archived_value = _read_json(path)
            _assert_git_safe_json(archived_value, label=f"archive.{path.name}")
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
