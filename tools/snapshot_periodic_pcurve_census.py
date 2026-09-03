"""Archive a completed periodic-pcurve census as path-free Git evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .probe_periodic_pcurve_applicability import (
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        canonical_sha256,
        summarize,
        validate_case_row,
    )
    from .snapshot_assembly_repair import (
        FORBIDDEN_SUFFIXES,
        _assert_path_free,
        artifact_manifest,
        sha256_file,
        write_text_lf,
    )
except ImportError:  # direct script execution
    from probe_periodic_pcurve_applicability import (
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        ROWS_NAME,
        SCHEMA,
        SUMMARY_NAME,
        SUMMARY_SCHEMA,
        TARGET_CAD_IDS,
        canonical_sha256,
        summarize,
        validate_case_row,
    )
    from snapshot_assembly_repair import (
        FORBIDDEN_SUFFIXES,
        _assert_path_free,
        artifact_manifest,
        sha256_file,
        write_text_lf,
    )


ARCHIVE_SCHEMA = "periodic-pcurve-applicability-archive-v1"
EXPECTED_DECISIONS = {
    "CLOSE_PERIODIC_PCURVE_ROUTE",
    "PROMOTE_TARGETED_REPAIR_PROBE",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {Path(path).name}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bad_face_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        bad_indices = {int(value) for value in row.get("bad_face_indices") or ()}
        for face in row.get("faces") or ():
            if int(face.get("face_index", -1)) not in bad_indices:
                continue
            kinds = sorted(
                {
                    str(occurrence.get("kind"))
                    for detail in face.get("bad_wire_details") or ()
                    for occurrence in detail.get("occurrences") or ()
                }
            )
            result.append(
                {
                    "cad_id": str(row.get("cad_id")),
                    "face_index": int(face["face_index"]),
                    "surface_type": str(face.get("surface_type")),
                    "u_periodic": bool(face.get("is_u_periodic")),
                    "v_periodic": bool(face.get("is_v_periodic")),
                    "bad_wire_indices": [
                        int(value)
                        for value in face.get("diagnosis", {}).get("bad_wire_indices") or ()
                    ],
                    "occurrence_kinds": kinds,
                    "reason": str(face.get("reason")),
                }
            )
    return result


def _readme(summary: Mapping[str, Any], bad_faces: Sequence[Mapping[str, Any]]) -> str:
    face_lines = "\n".join(
        "- `{cad}` face `{face}`: `{surface}`, U/V periodic = `{u}/{v}`, "
        "occurrences = `{kinds}`, decision reason = `{reason}`.".format(
            cad=row["cad_id"],
            face=row["face_index"],
            surface=row["surface_type"],
            u=str(row["u_periodic"]).lower(),
            v=str(row["v_periodic"]).lower(),
            kinds=", ".join(row["occurrence_kinds"]),
            reason=row["reason"],
        )
        for row in bad_faces
    ) or "- No bad face was observed."
    return f"""# Periodic-pcurve construction-stage applicability census

This is the Git-safe snapshot of the signed, read-only five-CAD census. Every
CAD ran in an isolated Open CASCADE worker using the same
`directed_trim_curve_fit` construction profile. Faces were observed after the
baseline wire fix and pcurve attachment and before any optional strategy
repair. The census did not mutate a face and did not write STEP candidates.

## Result

- Decision: `{summary['decision']}`
- Cases completed: `{summary['completed_cases']}/{summary['cases']}`
- Worker, protocol, binding, or measurement failures: `{summary['worker_or_protocol_failures']}`
- Fully observed bad faces: `{summary['bad_face_count']}`
- Periodic bad faces: `{summary['periodic_bad_face_count']}`
- Repairable periodic bad faces: `{summary['repairable_bad_face_count']}`
- Assembly selector remains: `91/100` strict-valid; release gate: `>=95/100`

All six diagnosed bad faces are non-periodic fitted B-spline surfaces:

{face_lines}

The conclusive result closes the periodic-pcurve branch-translation route only
for this frozen five-CAD cohort. It does not claim that the overall assembly
problem is solved and does not authorize boundary-consistency training,
full-scale VQ training, sequence regeneration, AR training, or a full 100-CAD
rerun. The next assembly investigation should target the observed two-dimensional
trim/wire intersection and shell/connectivity failure families without relaxing
the schema-v2 topology and geometry gates.

`periodic_pcurve_cases.jsonl`, `periodic_pcurve_summary.json`, and
`periodic_pcurve_run.json` are byte-identical to the completed local run and are
bound by SHA-256. No STEP, source pickle, worker log, NumPy array, checkpoint,
upstream source, or machine-local absolute path is archived.
"""


def snapshot(
    run_root: Path, report_dir: Path, *, replace_existing: bool = False
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    source_rows = run_root / ROWS_NAME
    source_summary = run_root / SUMMARY_NAME
    source_run = run_root / RUN_MANIFEST_NAME
    for source in (source_rows, source_summary, source_run):
        if not source.is_file():
            raise RuntimeError(f"completed census artifact is missing: {source.name}")
    if report_dir.exists() and any(report_dir.iterdir()):
        if not replace_existing:
            raise RuntimeError(f"report directory must be empty: {report_dir}")
        expected_names = {
            "README.md",
            "archive_validation.json",
            "artifact_manifest.json",
            RUN_MANIFEST_NAME,
            ROWS_NAME,
            SUMMARY_NAME,
        }
        existing = list(report_dir.iterdir())
        if any(not path.is_file() or path.name not in expected_names for path in existing):
            raise RuntimeError("refusing to replace an unexpected report directory")
        for path in existing:
            path.unlink()

    rows = _read_rows(source_rows)
    summary = _read_json(source_summary)
    run = _read_json(source_run)
    payload = run.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("census run payload is missing")
    if run.get("schema") != RUN_SCHEMA or run.get("status") != "COMPLETED":
        raise RuntimeError("census run must have the completed v1 schema")
    if payload.get("schema") != RUN_SCHEMA:
        raise RuntimeError("census payload schema mismatch")
    if run.get("signature") != canonical_sha256(payload):
        raise RuntimeError("census run signature mismatch")
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("conclusive") is not True:
        raise RuntimeError("census summary must be conclusive")
    if summary.get("decision") not in EXPECTED_DECISIONS:
        raise RuntimeError("census summary has no preregistered terminal decision")
    if len(rows) != len(TARGET_CAD_IDS) or int(run.get("attempts", -1)) != len(rows):
        raise RuntimeError("census archive requires exactly five attempts")
    if [row.get("cad_id") for row in rows] != list(TARGET_CAD_IDS):
        raise RuntimeError("census rows do not preserve the frozen target order")
    payload_bindings = payload.get("source_bindings")
    if (
        not isinstance(payload_bindings, list)
        or len(payload_bindings) != len(TARGET_CAD_IDS)
        or any(not isinstance(binding, Mapping) for binding in payload_bindings)
    ):
        raise RuntimeError("signed census source bindings are missing")
    bindings_by_id = {
        str(binding.get("cad_id")): {
            "bytes": binding.get("bytes"),
            "sha256": binding.get("sha256"),
        }
        for binding in payload_bindings
        if isinstance(binding, Mapping)
    }
    if set(bindings_by_id) != set(TARGET_CAD_IDS):
        raise RuntimeError("signed census source bindings do not match the target cohort")
    sources = {
        row["cad_id"]: {"cad_id": row["cad_id"], "parent_id": row["parent_id"]}
        for row in rows
    }
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise RuntimeError("census case schema mismatch")
        validate_case_row(
            row,
            source=sources[str(row["cad_id"])],
            run_signature=str(run["signature"]),
            expected_binding=bindings_by_id[str(row["cad_id"])],
        )
        if row.get("status") != "completed" or row.get("all_faces_observed") is not True:
            raise RuntimeError("census archive requires five complete all-face observations")
    if sha256_file(source_rows) != run.get("rows_sha256"):
        raise RuntimeError("case ledger does not match the completed run binding")
    if sha256_file(source_summary) != run.get("summary_sha256"):
        raise RuntimeError("summary does not match the completed run binding")
    if summarize(rows) != summary:
        raise RuntimeError("census summary does not equal the rows-derived summary")
    for name, value in (("run", run), ("summary", summary), ("rows", rows)):
        _assert_path_free(value, name)

    bad_faces = _bad_face_rows(rows)
    if len(bad_faces) != int(summary.get("bad_face_count", -1)):
        raise RuntimeError("derived bad-face count does not match summary")

    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_rows, report_dir / ROWS_NAME)
    shutil.copyfile(source_summary, report_dir / SUMMARY_NAME)
    shutil.copyfile(source_run, report_dir / RUN_MANIFEST_NAME)
    write_text_lf(report_dir / "README.md", _readme(summary, bad_faces))

    validation = {
        "schema": ARCHIVE_SCHEMA,
        "valid": True,
        "decision": summary["decision"],
        "cases": len(rows),
        "completed_cases": sum(row.get("status") == "completed" for row in rows),
        "all_faces_observed_cases": sum(row.get("all_faces_observed") is True for row in rows),
        "observed_faces": sum(int(row.get("face_count", 0)) for row in rows),
        "bad_faces": len(bad_faces),
        "periodic_bad_faces": int(summary["periodic_bad_face_count"]),
        "repairable_bad_faces": int(summary["repairable_bad_face_count"]),
        "worker_or_protocol_failures": int(summary["worker_or_protocol_failures"]),
        "probe_failures": int(summary["probe_failures"]),
        "explicit_failures": int(summary["explicit_failures"]),
        "run_signature": run["signature"],
        "signature_valid": True,
        "rows_binding_valid": True,
        "summary_binding_valid": True,
        "summary_recomputed_equal": True,
        "source_bindings_valid": True,
        "path_free": True,
        "source_bindings": {
            ROWS_NAME: sha256_file(source_rows),
            SUMMARY_NAME: sha256_file(source_summary),
            RUN_MANIFEST_NAME: sha256_file(source_run),
        },
        "absolute_path_matches": 0,
        "path_key_matches": 0,
        "forbidden_artifacts": [],
        "source_bytes_archived": False,
        "step_bytes_archived": False,
        "worker_logs_archived": False,
    }
    write_text_lf(
        report_dir / "archive_validation.json",
        json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
    )

    forbidden = [
        path.relative_to(report_dir).as_posix()
        for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden artifacts entered census report: {forbidden}")
    write_text_lf(
        report_dir / "artifact_manifest.json",
        json.dumps(
            {"schema": ARCHIVE_SCHEMA, "artifacts": artifact_manifest(report_dir)},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            snapshot(
                args.run_root,
                args.report_dir,
                replace_existing=args.replace_existing,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
