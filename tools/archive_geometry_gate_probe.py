"""Archive path-free geometry/topology-gate evidence for one repair probe.

The assembly matrix keeps the full OCC gate in its local JSONL, while the
generic snapshot intentionally strips worker diagnostics.  This small
archiver preserves only the gate fields needed to audit a candidate and never
copies STEP, pickle, log, or array bytes into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_SUFFIXES = {
    ".step", ".stp", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npy", ".npz",
}
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|[\"'\s])/(?:home|users|private)(?:[\\/]))")
GATE_KEYS = (
    "schema", "accepted", "checks", "rejection_reasons", "thresholds",
    "bbox_relative_delta", "edge_length_relative_delta",
    "input_to_candidate_rms_normalized", "input_to_candidate_max_normalized",
    "candidate_to_input_rms_normalized", "candidate_to_input_max_normalized",
    "input_face_count", "candidate_face_count", "input_edge_count",
    "candidate_edge_count", "input_vertex_count", "candidate_vertex_count",
    "input_face_edge_occurrences", "candidate_face_edge_occurrences",
    "input_face_edge_incidence_counts", "candidate_face_edge_incidence_counts",
    "input_edge_face_incidence_counts", "candidate_edge_face_incidence_counts",
    "input_vertex_edge_incidence_counts", "candidate_vertex_edge_incidence_counts",
    "projectable_edge_count", "unprojectable_edge_count",
    "input_projection_sample_count", "input_to_candidate_sample_count",
    "input_to_candidate_projected_sample_count",
    "input_to_candidate_projection_failure_count",
    "candidate_to_input_sample_count", "candidate_to_input_projected_sample_count",
    "candidate_to_input_projection_failure_count",
    "candidate_curve_requested_sample_count", "candidate_curve_successful_sample_count",
    "candidate_curve_sampling_failure_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_path_free(value: Any, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name == "path" or name.endswith("_path"):
                raise ValueError(f"path-bearing field at {location}.{key}")
            _assert_path_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_path_free(child, f"{location}[{index}]")
    elif isinstance(value, str) and PATH_PATTERN.search(value):
        raise ValueError(f"absolute path text at {location}")


def compact_gate(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("selector geometry gate must be an object or null")
    _assert_path_free(value, "geometry_gate")
    gate = {key: value.get(key) for key in GATE_KEYS if key in value}
    _assert_path_free(gate, "geometry_gate")
    return gate


def failure_family(row: Mapping[str, Any]) -> str:
    status = str(row.get("status") or "")
    error = str(row.get("error") or "").lower()
    if "curve_fit" in error or "curve collapses" in error:
        return "curve_fit"
    components = row.get("validity_components")
    if isinstance(components, Mapping):
        if int(components.get("wire_self_intersections") or 0) > 0:
            return "closure_or_self_intersection"
        if int(components.get("shell_count") or 0) != 1 or int(
            components.get("solid_count") or 0
        ) != 1:
            return "shell_or_connectivity"
    if bool(row.get("native_brep_valid")) and not bool(row.get("strict_brep_valid")):
        return "strict_topology_or_pcurve"
    if status.startswith("worker") or status.startswith("source_"):
        return "worker_or_protocol"
    if status == "both_valid":
        return "valid"
    return "other"


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    components = row.get("validity_components")
    compact: dict[str, Any] = {
        key: row.get(key)
        for key in (
            "schema", "cad_id", "parent_id", "profile", "switches",
            "historical_strict_valid", "status", "step_saved",
            "native_brep_valid", "strict_brep_valid", "both_valid",
            "step_bytes", "step_sha256", "error_type", "elapsed_seconds",
        )
    }
    compact["validity_components"] = {
        key: components.get(key) if isinstance(components, Mapping) else None
        for key in (
            "wire_order_failures", "wire_self_intersections", "free_edges",
            "shell_count", "shells_with_bad_edges", "solid_count",
        )
    }
    compact["failure_family"] = failure_family(row)
    compact["selector_geometry_topology_gate"] = compact_gate(
        row.get("selector_geometry_topology_gate")
    )
    compact["source_bytes_archived"] = False
    compact["step_bytes_archived"] = False
    _assert_path_free(compact, "attempt")
    return compact


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def archive(run_root: Path, report_dir: Path, *, label: str) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    matrix = run_root / "assembly_repair_matrix.jsonl"
    manifest = run_root / "assembly_repair_run.json"
    if not matrix.is_file() or not manifest.is_file():
        raise FileNotFoundError("probe run must contain matrix and run manifest")
    rows = [json.loads(line) for line in matrix.read_text(encoding="utf-8").splitlines() if line]
    run = json.loads(manifest.read_text(encoding="utf-8"))
    if run.get("schema") != "assembly-repair-run-v2":
        raise ValueError("unsupported run manifest schema")
    if int(run.get("attempts", -1)) != len(rows):
        raise ValueError("run attempt count does not match matrix")
    compact = [compact_row(row) for row in rows]
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in report_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden artifact already present: {path.name}")
    with (report_dir / "geometry_gate_attempts.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
                for row in compact
            )
        )
    gates = [row["selector_geometry_topology_gate"] for row in compact]
    accepted = [
        row["cad_id"] for row in compact
        if isinstance(row["selector_geometry_topology_gate"], Mapping)
        and row["selector_geometry_topology_gate"].get("accepted") is True
    ]
    rejected = [
        row for row in compact
        if isinstance(row["selector_geometry_topology_gate"], Mapping)
        and row["selector_geometry_topology_gate"].get("accepted") is False
    ]
    rejection_reasons = Counter(
        reason
        for row in rejected
        for reason in row["selector_geometry_topology_gate"].get("rejection_reasons") or ()
    )
    summary = {
        "schema": "assembly-geometry-gate-probe-v1",
        "label": label,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "attempts": len(compact),
        "profiles": dict(Counter(str(row.get("profile")) for row in compact)),
        "status_counts": dict(Counter(str(row.get("status")) for row in compact)),
        "failure_family_counts": dict(Counter(str(row.get("failure_family")) for row in compact)),
        "gate_present": sum(isinstance(gate, Mapping) for gate in gates),
        "gate_accepted": len(accepted),
        "gate_rejected": len(rejected),
        "gate_missing": sum(gate is None for gate in gates),
        "gate_accepted_cad_ids": sorted(accepted),
        "gate_rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "worker_or_protocol_failures": sum(
            row["failure_family"] == "worker_or_protocol" for row in compact
        ),
        "source_binding": {
            "run_signature": run.get("signature"),
            "run_status": run.get("status"),
            "matrix_sha256": sha256_file(matrix),
            "run_manifest_sha256": sha256_file(manifest),
        },
        "production_eligible": False,
        "eligibility_reason": (
            "gate probe is historical-invalid-only and accepted IDs must still be "
            "checked against the full selector cohort"
        ),
    }
    _write_json(report_dir / "geometry_gate_summary.json", summary)
    binding = {
        "schema": "assembly-geometry-gate-archive-v1",
        "label": label,
        "run_signature": run.get("signature"),
        "matrix_sha256": sha256_file(matrix),
        "attempts": len(compact),
        "forbidden_artifacts": [],
    }
    _write_json(report_dir / "geometry_gate_archive_validation.json", binding)
    artifacts = []
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append({
            "path": path.relative_to(report_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    _write_json(report_dir / "artifact_manifest.json", {"artifacts": artifacts})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(archive(args.run_root, args.report_dir, label=args.label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
