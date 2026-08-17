"""Create a Git-safe snapshot of an assembly repair pilot or formal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


FORBIDDEN_SUFFIXES = {".step", ".stp", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npy", ".npz"}
RUN_MANIFEST_NAME = "assembly_repair_run.json"
COMPLETED_RUN_STATUSES = {"COMPLETED", "COMPLETED_PARTIAL"}
SOLID_TOPOLOGY_DIAGNOSIS_SCHEMA = "solid-topology-diagnosis-v1"
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|(?:^|[\"'\s])/(?:home|users|private)(?:[\\/]))"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_lf(path: Path, content: str) -> None:
    """Write report text with canonical LF bytes on Windows and Linux."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_path_free(value: Any, location: str = "selection") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized == "path" or normalized.endswith("_path"):
                raise RuntimeError(f"selector evidence contains path field at {location}.{key}")
            _assert_path_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_path_free(child, f"{location}[{index}]")
    elif isinstance(value, str) and ABSOLUTE_PATH_PATTERN.search(value):
        raise RuntimeError(f"selector evidence contains absolute path text at {location}")


def _compact_selector_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    _assert_path_free(gate, "selection.geometry_topology_gate")
    return {
        key: gate.get(key)
        for key in (
            "schema",
            "accepted",
            "checks",
            "rejection_reasons",
            "bbox_relative_delta",
            "edge_length_relative_delta",
            "input_to_candidate_rms_normalized",
            "input_to_candidate_max_normalized",
            "input_to_candidate_sample_count",
            "input_to_candidate_projected_sample_count",
            "input_to_candidate_projection_failure_count",
            "candidate_to_input_rms_normalized",
            "candidate_to_input_max_normalized",
            "candidate_to_input_sample_count",
            "candidate_to_input_projected_sample_count",
            "candidate_to_input_projection_failure_count",
            "input_face_count",
            "candidate_face_count",
            "input_edge_count",
            "candidate_edge_count",
            "input_vertex_count",
            "candidate_vertex_count",
            "input_face_edge_occurrences",
            "candidate_face_edge_occurrences",
            "input_face_edge_incidence_counts",
            "candidate_face_edge_incidence_counts",
            "input_edge_face_incidence_counts",
            "candidate_edge_face_incidence_counts",
            "input_vertex_edge_incidence_counts",
            "candidate_vertex_edge_incidence_counts",
            "projectable_edge_count",
            "unprojectable_edge_count",
            "input_projection_sample_count",
            "candidate_curve_requested_sample_count",
            "candidate_curve_successful_sample_count",
            "candidate_curve_sampling_failure_count",
            "thresholds",
        )
    }


def compact_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist selector proof fields and reject all local execution paths."""
    _assert_path_free(selection)
    candidates: list[dict[str, Any]] = []
    for candidate in selection.get("candidates") or ():
        if not isinstance(candidate, Mapping):
            raise RuntimeError("selector evidence candidate is not an object")
        compact = {
            key: candidate.get(key)
            for key in (
                "profile",
                "switches",
                "status",
                "step_saved",
                "native_brep_valid",
                "strict_brep_valid",
                "both_valid",
                "step_bytes",
                "step_sha256",
                "candidate_result_sha256",
                "error_type",
                "elapsed_seconds",
                "worker_returncode",
                "rejection_reasons",
            )
        }
        components = candidate.get("validity_components") or {}
        compact["validity_components"] = {
            key: components.get(key)
            for key in (
                "status",
                "wire_count",
                "wire_order_failures",
                "wire_self_intersections",
                "free_edges",
                "shell_count",
                "shells_with_bad_edges",
                "solid_count",
            )
        }
        gate = candidate.get("geometry_topology_gate")
        if isinstance(gate, Mapping):
            compact["geometry_topology_gate"] = _compact_selector_gate(gate)
        candidates.append(compact)
    return {
        key: selection.get(key)
        for key in (
            "schema",
            "primary_profile",
            "fallback_order",
            "attempted_profiles",
            "selected_profile",
            "selected_reason",
            "fallback_accepted",
        )
    } | {"candidates": candidates}


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    components = row.get("validity_components") or {}
    compact = {
        key: row.get(key)
        for key in (
            "schema", "cad_id", "parent_id", "profile", "switches",
            "historical_strict_valid", "status", "step_saved",
            "native_brep_valid", "strict_brep_valid", "both_valid",
            "step_bytes", "step_sha256", "error_type",
            "elapsed_seconds",
        )
    } | {
        "validity_components": {
            key: components.get(key)
            for key in (
                "wire_order_failures", "wire_self_intersections", "free_edges",
                "shell_count", "shells_with_bad_edges", "solid_count",
            )
        },
        "source_bytes_archived": False,
        "step_bytes_archived": False,
    }
    if isinstance(row.get("selection"), Mapping):
        compact["selection"] = compact_selection(row["selection"])
    return compact


def compact_run_manifest(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transient exception text before archiving a completed run contract."""
    compact = dict(run_manifest)
    compact.pop("error", None)
    compact.pop("error_type", None)
    _assert_path_free(compact, "run_manifest")
    return compact


def selector_snapshot_binding(
    *,
    run_root: Path,
    matrix_path: Path,
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate selector ledger and matrix bindings without archiving the ledger."""
    payload = run_manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("run_kind") != (
        "assembly-repair-selector-v1"
    ):
        return {"selector_run": False}
    expected_matrix_sha = run_manifest.get("final_matrix_sha256")
    matrix_sha = sha256_file(matrix_path)
    if not isinstance(expected_matrix_sha, str) or expected_matrix_sha != matrix_sha:
        raise RuntimeError("selector final matrix hash does not match signed run manifest")
    candidate_path = Path(run_root) / "assembly_selector_candidates.jsonl"
    if not candidate_path.is_file():
        raise RuntimeError("selector candidate ledger is missing")
    expected_candidate_sha = run_manifest.get("candidate_manifest_sha256")
    candidate_sha = sha256_file(candidate_path)
    if (
        not isinstance(expected_candidate_sha, str)
        or expected_candidate_sha != candidate_sha
    ):
        raise RuntimeError(
            "selector candidate ledger hash does not match signed run manifest"
        )
    try:
        from .run_assembly_repair_matrix import read_jsonl
        from .run_assembly_repair_selector import (
            validate_candidate_ledger,
            validate_final_candidate_bindings,
        )
    except ImportError:  # direct script execution
        from run_assembly_repair_matrix import read_jsonl
        from run_assembly_repair_selector import (
            validate_candidate_ledger,
            validate_final_candidate_bindings,
        )
    candidate_entries = read_jsonl(candidate_path)
    if int(run_manifest.get("candidate_attempts", -1)) != len(candidate_entries):
        raise RuntimeError(
            "selector candidate ledger count does not match signed run manifest"
        )
    source_rows = [
        {
            "cad_id": row.get("cad_id"),
            "parent_id": row.get("parent_id"),
            "source_path": row.get("source_path"),
            "brep_valid": row.get("historical_strict_valid"),
        }
        for row in rows
    ]
    validate_candidate_ledger(candidate_entries, source_rows)
    validate_final_candidate_bindings(rows, candidate_entries)
    return {
        "selector_run": True,
        "candidate_attempts": len(candidate_entries),
        "candidate_ledger_sha256": candidate_sha,
        "final_matrix_sha256": matrix_sha,
        "candidate_final_binding_valid": True,
    }


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(report_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def repair_diagnostics_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate repair decisions without archiving OCC objects or source paths."""
    directed_modes: Counter[str] = Counter()
    directed_cads: dict[str, set[str]] = {}
    local_reasons: Counter[str] = Counter()
    local_attempted_cads: set[str] = set()
    local_accepted_cads: set[str] = set()
    local_attempts = 0
    local_accepted = 0
    solid_candidate_pairs = 0
    solid_mutual_pairs = 0
    solid_merged_vertices = 0
    solid_applied_cads: set[str] = set()
    for row in rows:
        cad_id = str(row.get("cad_id"))
        diagnostics = row.get("assembly_diagnostics") or {}
        for item in diagnostics.get("directed_trim_loop_policies") or []:
            mode = str(item.get("mode") or "missing")
            directed_modes[mode] += 1
            directed_cads.setdefault(mode, set()).add(cad_id)
        for item in diagnostics.get("local_intersection_topology") or []:
            if not item.get("attempted"):
                continue
            local_attempts += 1
            local_attempted_cads.add(cad_id)
            reason = str(item.get("reason") or "missing")
            local_reasons[reason] += 1
            if item.get("accepted"):
                local_accepted += 1
                local_accepted_cads.add(cad_id)
        solid = diagnostics.get("solid_topology_repair") or {}
        if isinstance(solid, Mapping):
            solid_candidate_pairs += int(solid.get("candidate_pair_count") or 0)
            solid_mutual_pairs += int(solid.get("mutual_pair_count") or 0)
            solid_merged_vertices += int(solid.get("merged_vertex_count") or 0)
            if solid.get("applied"):
                solid_applied_cads.add(cad_id)
    result: dict[str, Any] = {}
    if directed_modes:
        result["directed_trim_loop_policies"] = {
            "face_decision_count": sum(directed_modes.values()),
            "mode_counts": dict(sorted(directed_modes.items())),
            "cad_ids_by_mode": {
                mode: sorted(cad_ids)
                for mode, cad_ids in sorted(directed_cads.items())
            },
        }
    if local_attempts:
        result["local_intersection_topology"] = {
            "attempted_face_count": local_attempts,
            "accepted_face_count": local_accepted,
            "reason_counts": dict(sorted(local_reasons.items())),
            "attempted_cad_ids": sorted(local_attempted_cads),
            "accepted_cad_ids": sorted(local_accepted_cads),
        }
    if solid_candidate_pairs or solid_mutual_pairs or solid_applied_cads:
        result["solid_topology_repair"] = {
            "candidate_pair_count": solid_candidate_pairs,
            "mutual_pair_count": solid_mutual_pairs,
            "merged_vertex_count": solid_merged_vertices,
            "applied_cad_ids": sorted(solid_applied_cads),
        }
    return result


def cohort_equivalence(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_attempts: Path,
) -> dict[str, Any]:
    """Prove CAD identity/control-map equality without carrying local paths."""
    reference_path = Path(reference_attempts)
    reference_rows = read_jsonl(reference_path)

    def normalized(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        order: list[str] = []
        values: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            cad_id = str(row.get("cad_id") or "")
            if not cad_id:
                raise RuntimeError(f"{label} cohort row {index} has no cad_id")
            if cad_id in values:
                raise RuntimeError(f"{label} cohort repeats CAD {cad_id}")
            historical = row.get("historical_strict_valid")
            if type(historical) is not bool:
                raise RuntimeError(
                    f"{label} cohort CAD {cad_id} has non-boolean historical strict value"
                )
            values[cad_id] = {
                "parent_id": row.get("parent_id"),
                "historical_strict_valid": historical,
            }
            order.append(cad_id)
        return order, values

    candidate_order, candidate_map = normalized(candidate_rows, label="candidate")
    reference_order, reference_map = normalized(reference_rows, label="reference")
    candidate_set = sorted(candidate_map)
    reference_set = sorted(reference_map)
    id_set_digest = hashlib.sha256(
        ("\n".join(candidate_set) + "\n").encode("utf-8")
    ).hexdigest()
    equivalent = bool(candidate_set == reference_set and candidate_map == reference_map)
    if not equivalent:
        raise RuntimeError(
            "reference report does not match the candidate CAD/control cohort"
        )
    return {
        "schema": "assembly-cohort-equivalence-v1",
        "reference_attempts_sha256": sha256_file(reference_path),
        "reference_attempts": len(reference_rows),
        "candidate_attempts": len(candidate_rows),
        "cad_id_set_sha256": id_set_digest,
        "reference_cad_order_sha256": hashlib.sha256(
            ("\n".join(reference_order) + "\n").encode("utf-8")
        ).hexdigest(),
        "candidate_cad_order_sha256": hashlib.sha256(
            ("\n".join(candidate_order) + "\n").encode("utf-8")
        ).hexdigest(),
        "same_cad_set": True,
        "same_parent_and_historical_strict_map": True,
        "same_cad_order": reference_order == candidate_order,
        "valid": True,
    }


def archive_solid_topology_diagnosis(
    source: Path, report_dir: Path
) -> dict[str, Any]:
    """Copy one proven diagnosis only after rejecting path-bearing payloads."""
    source = Path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != SOLID_TOPOLOGY_DIAGNOSIS_SCHEMA:
        raise RuntimeError("solid topology diagnosis schema is unsupported")

    def reject_private_values(value: Any, location: str = "root") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized == "path" or normalized.endswith("_path"):
                    raise RuntimeError(
                        f"solid topology diagnosis contains path field at {location}.{key}"
                    )
                reject_private_values(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_private_values(child, f"{location}[{index}]")
        elif isinstance(value, str) and ABSOLUTE_PATH_PATTERN.search(value):
            raise RuntimeError(
                f"solid topology diagnosis contains absolute path text at {location}"
            )

    reject_private_values(payload)
    target = report_dir / "solid_topology_diagnosis.json"
    write_text_lf(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "archived": True,
        "schema": payload["schema"],
        "cad_id": payload.get("cad_id"),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "archived_bytes": target.stat().st_size,
        "archived_sha256": sha256_file(target),
    }


def snapshot(
    run_root: Path,
    report_dir: Path,
    *,
    label: str,
    solid_topology_diagnosis: Optional[Path] = None,
    reference_report_dir: Optional[Path] = None,
) -> dict[str, Any]:
    run_root, report_dir = Path(run_root).resolve(), Path(report_dir).resolve()
    source = run_root / "assembly_repair_matrix.jsonl"
    summary_path = run_root / "assembly_repair_summary.json"
    run_manifest_path = run_root / RUN_MANIFEST_NAME
    rows = read_jsonl(source)
    if not rows or not summary_path.is_file() or not run_manifest_path.is_file():
        raise RuntimeError("assembly repair run is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_status = str(run_manifest.get("status"))
    if run_manifest.get("schema") != "assembly-repair-run-v2":
        raise RuntimeError("assembly repair run manifest schema is not signed v2")
    if run_status not in COMPLETED_RUN_STATUSES:
        raise RuntimeError(f"assembly repair run status is not complete: {run_status}")
    if int(run_manifest.get("attempts", -1)) != len(rows):
        raise RuntimeError("assembly repair run attempt count does not match matrix")
    matrix_sha256 = sha256_file(source)
    if (
        run_manifest.get("final_matrix_sha256") is not None
        and run_manifest.get("final_matrix_sha256") != matrix_sha256
    ):
        raise RuntimeError("assembly repair matrix hash does not match signed run manifest")
    summary_sha256 = sha256_file(summary_path)
    if run_manifest.get("summary_sha256") != summary_sha256:
        raise RuntimeError("assembly repair summary hash does not match signed run manifest")
    cohort_binding = {"provided": False}
    if reference_report_dir is not None:
        reference_attempts = (
            Path(reference_report_dir).resolve() / "assembly_repair_attempts.jsonl"
        )
        if not reference_attempts.is_file():
            raise FileNotFoundError(reference_attempts)
        cohort_binding = {
            "provided": True,
            "reference_report_name": Path(reference_report_dir).resolve().name,
            **cohort_equivalence(rows, reference_attempts),
        }
    selector_binding = selector_snapshot_binding(
        run_root=run_root,
        matrix_path=source,
        rows=rows,
        run_manifest=run_manifest,
    )
    archived_run_manifest = compact_run_manifest(run_manifest)
    report_dir.mkdir(parents=True, exist_ok=True)
    if any(report_dir.iterdir()):
        raise RuntimeError(f"report directory must be empty: {report_dir}")
    compact = [compact_row(row) for row in rows]
    write_text_lf(
        report_dir / "assembly_repair_attempts.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compact),
    )
    write_text_lf(
        report_dir / RUN_MANIFEST_NAME,
        json.dumps(archived_run_manifest, indent=2, sort_keys=True) + "\n",
    )
    repair_summary = repair_diagnostics_summary(rows)
    write_text_lf(
        report_dir / "repair_diagnostics_summary.json",
        json.dumps(repair_summary, indent=2, sort_keys=True) + "\n",
    )
    topology_diagnosis_binding = (
        archive_solid_topology_diagnosis(solid_topology_diagnosis, report_dir)
        if solid_topology_diagnosis is not None
        else {"archived": False}
    )
    if cohort_binding["provided"]:
        write_text_lf(
            report_dir / "cohort_equivalence.json",
            json.dumps(cohort_binding, indent=2, sort_keys=True) + "\n",
        )
    source_binding = {
        "label": label, "source_run_name": run_root.name,
        "source_matrix_bytes": source.stat().st_size,
        "source_matrix_sha256": matrix_sha256,
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "run_signature": run_manifest.get("signature"),
        "run_status": run_status,
        "summary_sha256": summary_sha256,
        "selector_ledger_binding": selector_binding,
        "step_files_local": sum(bool(row.get("step_saved")) for row in rows),
        "step_bytes_archived": False, "source_pickles_archived": False,
        "solid_topology_diagnosis": topology_diagnosis_binding,
        "cohort_equivalence": cohort_binding,
    }
    archived_summary = {**summary, "label": label, "generated_at": now(), "source_binding": source_binding}
    write_text_lf(
        report_dir / "assembly_repair_summary.json",
        json.dumps(archived_summary, indent=2, sort_keys=True) + "\n",
    )
    profile_lines = "\n".join(
        f"| {item['profile']} | {item['attempts']} | {item['step_readable']} | {item['native_valid']} | {item['strict_valid']} | {item['both_valid']} | {len(item['restored_cad_ids'])} | {len(item['regressed_cad_ids'])} |"
        for item in summary["profiles"]
    )
    readme = f"""# Assembly repair evidence: {label}

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{profile_lines}

Gate passed: `{summary.get('gate_passed')}`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
"""
    write_text_lf(report_dir / "README.md", readme)
    forbidden = [
        path.relative_to(report_dir).as_posix() for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden artifacts entered report: {forbidden}")
    validation = {
        "valid": True, "attempts": len(compact),
        "profiles": dict(Counter(str(row.get("profile")) for row in compact)),
        "run_signature": run_manifest.get("signature"),
        "run_status": run_status,
        "summary_sha256": summary_sha256,
        "final_matrix_sha256": matrix_sha256,
        "selector_ledger_binding": selector_binding,
        "repair_diagnostics_present": bool(repair_summary),
        "solid_topology_diagnosis_archived": bool(
            topology_diagnosis_binding["archived"]
        ),
        "cohort_equivalence_valid": (
            bool(cohort_binding["valid"])
            if cohort_binding["provided"]
            else None
        ),
        "forbidden_artifacts": forbidden,
    }
    write_text_lf(
        report_dir / "archive_validation.json",
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
    )
    write_text_lf(
        report_dir / "artifact_manifest.json",
        json.dumps(
            {"generated_at": now(), "artifacts": artifact_manifest(report_dir)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--solid-topology-diagnosis", type=Path)
    parser.add_argument(
        "--reference-report-dir",
        type=Path,
        help="Require CAD, parent, and historical strict-control equality with a prior snapshot.",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            snapshot(
                args.run_root,
                args.report_dir,
                label=args.label,
                solid_topology_diagnosis=args.solid_topology_diagnosis,
                reference_report_dir=args.reference_report_dir,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
