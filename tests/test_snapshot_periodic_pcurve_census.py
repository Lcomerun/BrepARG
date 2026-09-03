import copy
import json

import pytest

from tools.probe_periodic_pcurve_applicability import (
    RUN_MANIFEST_NAME,
    RUN_SCHEMA,
    ROWS_NAME,
    SCHEMA,
    SUMMARY_NAME,
    SUMMARY_SCHEMA,
    TARGET_CAD_IDS,
    canonical_sha256,
    sha256_file,
    summarize,
)
from tools.snapshot_periodic_pcurve_census import snapshot


def _case(cad_id, *, bad=False):
    face = {
        "face_index": 0,
        "phase": "post_add_pcurves_pre_repair",
        "applicable": False,
        "periodic_gap_candidate": False,
        "partial_only": False,
        "reason": "surface_not_periodic" if bad else "no_diagnosed_self_intersection",
        "diagnosis": {"bad_wire_indices": [0] if bad else []},
        "bad_wire_details": [],
        "surface_type": "Geom_BSplineSurface",
        "is_u_periodic": False,
        "is_v_periodic": False,
    }
    return {
        "schema": SCHEMA,
        "cad_id": cad_id,
        "parent_id": f"parent-{cad_id}",
        "profile": "directed_trim_curve_fit",
        "run_signature": "",
        "source_binding": {"bytes": 1, "sha256": "a" * 64},
        "source_binding_loaded_bytes": {"bytes": 1, "sha256": "a" * 64},
        "source_binding_after_load": {"bytes": 1, "sha256": "a" * 64},
        "status": "completed",
        "assembly_status": "completed",
        "source_face_count": 1,
        "face_count": 1,
        "all_faces_observed": True,
        "faces": [face],
        "bad_face_indices": [0] if bad else [],
        "periodic_bad_face_indices": [],
        "repairable_face_indices": [],
    }


def _write_run(root, *, add_path=False):
    rows = [_case(cad_id, bad=index == 0) for index, cad_id in enumerate(TARGET_CAD_IDS)]
    payload = {
        "schema": RUN_SCHEMA,
        "ordered_cad_ids": list(TARGET_CAD_IDS),
        "profile": "directed_trim_curve_fit",
        "source_bindings": [
            {"cad_id": cad_id, "bytes": 1, "sha256": "a" * 64}
            for cad_id in TARGET_CAD_IDS
        ],
    }
    if add_path:
        payload["source_path"] = "C:/private/source.pkl"
    signature = canonical_sha256(payload)
    for row in rows:
        row["run_signature"] = signature
    rows_path = root / ROWS_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows)
    summary_path = root / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    run = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETED",
        "attempts": 5,
        "payload": payload,
        "signature": signature,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    (root / RUN_MANIFEST_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")
    return rows, summary, run


def test_snapshot_archives_only_signed_path_free_evidence(tmp_path):
    run_root = tmp_path / "run"
    report = tmp_path / "report"
    run_root.mkdir()
    _write_run(run_root)

    result = snapshot(run_root, report)

    assert result["valid"] is True
    assert result["cases"] == result["completed_cases"] == 5
    assert result["all_faces_observed_cases"] == 5
    assert result["bad_faces"] == 1
    assert sorted(path.name for path in report.iterdir()) == sorted([
        "README.md",
        "archive_validation.json",
        "artifact_manifest.json",
        RUN_MANIFEST_NAME,
        ROWS_NAME,
        SUMMARY_NAME,
    ])
    manifest = json.loads((report / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in manifest["artifacts"]}
    assert paths == {
        "README.md",
        "archive_validation.json",
        RUN_MANIFEST_NAME,
        ROWS_NAME,
        SUMMARY_NAME,
    }
    for item in manifest["artifacts"]:
        artifact = report / item["path"]
        assert artifact.stat().st_size == item["bytes"]
        assert sha256_file(artifact) == item["sha256"]
    assert (report / ROWS_NAME).read_bytes() == (run_root / ROWS_NAME).read_bytes()
    assert (report / SUMMARY_NAME).read_bytes() == (run_root / SUMMARY_NAME).read_bytes()
    assert (report / RUN_MANIFEST_NAME).read_bytes() == (run_root / RUN_MANIFEST_NAME).read_bytes()


def test_snapshot_rejects_path_bearing_signed_payload(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_run(run_root, add_path=True)

    with pytest.raises(RuntimeError, match="path field"):
        snapshot(run_root, tmp_path / "report")


def test_snapshot_rejects_case_ledger_hash_drift(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows, _summary, _run = _write_run(run_root)
    changed = copy.deepcopy(rows)
    changed[0]["cad_id"] = "drifted"
    (run_root / ROWS_NAME).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in changed),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="target order|ledger"):
        snapshot(run_root, tmp_path / "report")


def test_snapshot_rejects_case_binding_drift_even_if_ledger_hash_is_updated(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows, _summary, run = _write_run(run_root)
    rows[0]["source_binding"] = {"bytes": 2, "sha256": "b" * 64}
    rows[0]["source_binding_loaded_bytes"] = dict(rows[0]["source_binding"])
    rows[0]["source_binding_after_load"] = dict(rows[0]["source_binding"])
    rows_path = run_root / ROWS_NAME
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    run["rows_sha256"] = sha256_file(rows_path)
    (run_root / RUN_MANIFEST_NAME).write_text(json.dumps(run) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="signed input"):
        snapshot(run_root, tmp_path / "report")


def test_snapshot_replace_existing_accepts_only_its_known_report_files(tmp_path):
    run_root = tmp_path / "run"
    report = tmp_path / "report"
    run_root.mkdir()
    _write_run(run_root)
    snapshot(run_root, report)

    result = snapshot(run_root, report, replace_existing=True)

    assert result["valid"] is True
    (report / "unexpected.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected report directory"):
        snapshot(run_root, report, replace_existing=True)
    assert (report / "unexpected.txt").read_text(encoding="utf-8") == "do not delete"
