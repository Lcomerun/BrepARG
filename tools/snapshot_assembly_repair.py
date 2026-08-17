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


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    components = row.get("validity_components") or {}
    return {
        key: row.get(key)
        for key in (
            "schema", "cad_id", "parent_id", "profile", "switches",
            "historical_strict_valid", "status", "step_saved",
            "native_brep_valid", "strict_brep_valid", "both_valid",
            "step_bytes", "step_sha256", "error_type", "error",
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
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
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
        "source_manifest_bytes": source.stat().st_size,
        "source_manifest_sha256": sha256_file(source),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "run_signature": run_manifest.get("signature"),
        "run_status": run_status,
        "summary_sha256": summary_sha256,
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
