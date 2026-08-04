import json
import hashlib
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
if str(IMPROVEMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPROVEMENTS_DIR))


from cad_protocol import (  # noqa: E402
    ProtocolConfig,
    assign_parent_splits,
    build_manifest_row,
    build_protocol,
    inspect_cad_record,
    parent_cad_id,
)


def source(parent: str, part: int = 0, index: int = 1) -> str:
    return f"abc_0000/{index:08d}_{parent}_step_{part:03d}.pkl"


def make_cad(faces: int = 10, edges: int = 12, face_edges=None) -> dict:
    if face_edges is None:
        face_edges = [[index % edges] for index in range(faces)]
    return {
        "surf_ncs": [None] * faces,
        "edge_ncs": [None] * edges,
        "faceEdge_adj": face_edges,
    }


def eligible_row(parent: str, parts: int = 1) -> list[dict]:
    config = ProtocolConfig()
    return [build_manifest_row(source(parent, part, part + 1), make_cad(), config) for part in range(parts)]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (source("a" * 24), "a" * 24),
        (source("B" * 32), "b" * 32),
        (f"C:\\parsed\\{source('c' * 24)}", "c" * 24),
        (f"archive.zip!/{source('d' * 24)}", "d" * 24),
        (f"abc_0000/{'e' * 24}_step_001.pkl", "e" * 24),
        (f"abc_0000/{'f' * 24}_step_1.pkl", None),
        (f"abc_0000/{'0' * 24}_step_0001.pkl", None),
        (f"abc_0000/{'1' * 24}_step_001.pkl\n", None),
        (f"abc_0000/prefix_{'2' * 24}_step_001.pkl", None),
        (f"abc_0000/123_{'3' * 23}_step_001.pkl", None),
        (f"abc_0000/123_{'4' * 33}_step_001.pkl", None),
        ("abc_0000/train_a.pkl", None),
    ],
)
def test_parent_cad_id_is_strict_and_path_independent(path, expected):
    assert parent_cad_id(path) == expected


@pytest.mark.parametrize(
    ("data", "expected_reason"),
    [
        (make_cad(faces=9), "too_few_faces"),
        (make_cad(faces=51), "too_many_faces"),
        (make_cad(faces=10, edges=151), "too_many_global_edges"),
        (make_cad(faces=10, edges=31, face_edges=[list(range(31))] + [[0]] * 9), "too_many_edges_per_face"),
        ({"edge_ncs": [None] * 12, "faceEdge_adj": [[0]] * 10}, "missing_surf_ncs"),
        ({"surf_ncs": [None] * 10, "faceEdge_adj": [[0]] * 10}, "missing_edge_ncs"),
        ({"surf_ncs": [None] * 10, "edge_ncs": [None] * 12}, "missing_faceEdge_adj"),
        (make_cad(face_edges=[[0]] * 9), "face_edge_adjacency_length_mismatch"),
        (make_cad(face_edges=[["0"]] + [[0]] * 9), "non_integer_edge_index"),
        (make_cad(face_edges=[[-1]] + [[0]] * 9), "edge_index_out_of_range"),
        (make_cad(face_edges=[[12]] + [[0]] * 9), "edge_index_out_of_range"),
    ],
)
def test_protocol_rejects_invalid_records_with_stable_reasons(data, expected_reason):
    row = build_manifest_row(source("a" * 24), data, ProtocolConfig())

    assert row["protocol_eligible"] is False
    assert row["reject_reason"] == expected_reason
    assert row["split"] is None


@pytest.mark.parametrize(
    "data",
    [
        make_cad(faces=10, edges=12),
        make_cad(faces=50, edges=150, face_edges=[list(range(30))] + [[0]] * 49),
    ],
)
def test_protocol_accepts_inclusive_boundaries_and_populates_manifest(data):
    row = build_manifest_row(source("f" * 24), data, ProtocolConfig())

    assert row == {
        "protocol_version": "abc-parent-isolated-v2",
        "source_path": source("f" * 24),
        "source_key": source("f" * 24),
        "parent_id": "f" * 24,
        "num_faces": len(data["surf_ncs"]),
        "global_edges": len(data["edge_ncs"]),
        "max_edges_per_face": max(len(indices) for indices in data["faceEdge_adj"]),
        "protocol_eligible": True,
        "reject_reason": None,
        "split": None,
    }


def test_protocol_rejects_unknown_parent_and_load_errors():
    config = ProtocolConfig()

    unknown = build_manifest_row("abc_0000/train_a.pkl", make_cad(), config)
    failed = build_manifest_row(source("a" * 24), None, config, load_error="UnpicklingError")

    assert unknown["reject_reason"] == "unknown_parent_id"
    assert failed["reject_reason"] == "load_failed:UnpicklingError"


def test_rejection_reason_is_record_level_and_has_stable_priority():
    config = ProtocolConfig()
    missing_and_unknown = build_manifest_row("abc_0000/train_a.pkl", {}, config)
    mixed = make_cad(
        face_edges=[["0"]] + [list(range(31))] + [[0]] * 8,
        edges=31,
    )

    assert missing_and_unknown["reject_reason"] == "missing_surf_ncs"
    assert build_manifest_row(source("a" * 24), mixed, config)["reject_reason"] == "too_many_edges_per_face"

    mixed_order = make_cad(
        face_edges=[list(range(31))] + [["0"]] + [[0]] * 8,
        edges=31,
    )
    assert build_manifest_row(source("b" * 24), mixed_order, config)["reject_reason"] == "too_many_edges_per_face"


def test_record_inspection_scans_all_adjacency_before_selecting_reason():
    data = make_cad(
        faces=9,
        edges=151,
        face_edges=[["not-an-index"], list(range(31)), list(range(47))] + [[151]] * 6,
    )

    result = inspect_cad_record(data, ProtocolConfig())

    assert result["num_faces"] == 9
    assert result["global_edges"] == 151
    assert result["max_edges_per_face"] == 47
    assert result["reject_reason"] == "too_few_faces"


def test_invalid_field_has_priority_but_other_measurements_are_retained():
    data = {
        "surf_ncs": object(),
        "edge_ncs": [None] * 12,
        "faceEdge_adj": [[0], [0, 1, 2]],
    }

    result = inspect_cad_record(data, ProtocolConfig())

    assert result["num_faces"] is None
    assert result["global_edges"] == 12
    assert result["max_edges_per_face"] == 3
    assert result["reject_reason"] == "invalid_surf_ncs"


def test_unknown_parent_does_not_hide_record_validation_failure():
    row = build_manifest_row("abc_0000/not-a-parent.pkl", make_cad(faces=9), ProtocolConfig())

    assert row["num_faces"] == 9
    assert row["protocol_eligible"] is False
    assert row["reject_reason"] == "too_few_faces"


def test_parent_split_is_deterministic_balanced_and_never_fragments_parent():
    rows = (
        eligible_row("a" * 24, 4)
        + eligible_row("b" * 24, 3)
        + eligible_row("c" * 24)
        + eligible_row("d" * 24)
        + eligible_row("e" * 24)
        + eligible_row("f" * 24)
        + eligible_row("1" * 24)
    )
    config = ProtocolConfig(seed=20260803)

    first = assign_parent_splits(rows, config)
    second = assign_parent_splits(list(reversed(rows)), config)

    assert first == second
    assert set(first) == {row["parent_id"] for row in rows}
    assert set(first.values()) == {"train", "val", "test"}
    for parent in set(first):
        assert len({first[row["parent_id"]] for row in rows if row["parent_id"] == parent}) == 1

    counts = {name: 0 for name in ("train", "val", "test")}
    for row in rows:
        counts[first[row["parent_id"]]] += 1
    assert counts["train"] >= counts["val"]
    assert counts["train"] >= counts["test"]


def test_build_protocol_scans_zip_writes_rejects_and_has_zero_parent_overlap(tmp_path):
    archive = tmp_path / "abc_0000_parsed.zip"
    records = [
        (source("a" * 24, 0, 1), make_cad()),
        (source("a" * 24, 1, 2), make_cad()),
        (source("b" * 24, 0, 3), make_cad()),
        (source("c" * 24, 0, 4), make_cad()),
        (source("d" * 24, 0, 5), make_cad()),
        (source("e" * 24, 0, 6), make_cad(faces=9)),
    ]
    with zipfile.ZipFile(archive, "w") as handle:
        for member, data in records:
            handle.writestr(member, pickle.dumps(data))

    output = tmp_path / "out"
    rows, split, summary = build_protocol(
        archive_paths=[archive],
        config=ProtocolConfig(seed=7),
        output_dir=output,
        materialize_root=tmp_path / "materialized",
        max_scan_records=0,
        max_eligible_records=0,
    )

    assert len(rows) == 6
    assert summary["status"] == "VERIFIED"
    assert summary["records_scanned"] == 6
    assert summary["records_eligible"] == 5
    assert summary["records_rejected"] == 1
    assert summary["reject_reasons"] == {"too_few_faces": 1}
    assert summary["experiment_scale"] == "full"
    assert len(summary["protocol_sha256"]) == 64
    assert len(summary["split_pickle_sha256"]) == 64
    assert sum(map(len, split.values())) == 5
    assert {row["split"] for row in rows if row["parent_id"] == "a" * 24} <= {"train", "val", "test"}
    assert len({row["split"] for row in rows if row["parent_id"] == "a" * 24}) == 1

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_parents = {parent_cad_id(path) for path in split[left]}
        right_parents = {parent_cad_id(path) for path in split[right]}
        assert left_parents.isdisjoint(right_parents)

    assert (output / "protocol_manifest.jsonl").exists()
    assert (output / "protocol_summary.json").exists()
    assert (output / "split.pkl").exists()
    assert (output / "split_integrity.json").exists()
    written_rows = [json.loads(line) for line in (output / "protocol_manifest.jsonl").read_text().splitlines()]
    assert len(written_rows) == 6
    with (output / "split.pkl").open("rb") as handle:
        written_split = pickle.load(handle)
    assert written_split == split
    assert hashlib.sha256((output / "split.pkl").read_bytes()).hexdigest() == summary[
        "split_pickle_sha256"
    ]
    assert all(Path(path).exists() for paths in split.values() for path in paths)


def test_build_protocol_preserves_archive_identity_for_duplicate_member_names(tmp_path):
    member = source("a" * 24, part=0, index=1)
    archives = [
        tmp_path / "abc_0000_parsed.zip",
        tmp_path / "abc_0001_parsed.zip",
    ]
    for index, archive in enumerate(archives):
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, pickle.dumps(make_cad(edges=12 + index)))

    _, split, summary = build_protocol(
        archive_paths=archives,
        config=ProtocolConfig(seed=7),
        output_dir=tmp_path / "out",
        materialize_root=tmp_path / "materialized",
    )

    materialized = [Path(path) for paths in split.values() for path in paths]
    assert summary["records_selected"] == 2
    assert len(materialized) == 2
    assert len({path.resolve() for path in materialized}) == 2
    assert {path.relative_to(tmp_path / "materialized").parts[1] for path in materialized} == {
        "abc_0000_parsed",
        "abc_0001_parsed",
    }
    assert all(path.exists() for path in materialized)


def test_scan_limit_is_labeled_smoke_and_eligible_cap_preserves_whole_parents(tmp_path):
    archive = tmp_path / "abc_0000_parsed.zip"
    records = [
        (source("a" * 24, 0, 1), make_cad()),
        (source("a" * 24, 1, 2), make_cad()),
        (source("b" * 24, 0, 3), make_cad()),
        (source("c" * 24, 0, 4), make_cad()),
    ]
    with zipfile.ZipFile(archive, "w") as handle:
        for member, data in records:
            handle.writestr(member, pickle.dumps(data))

    rows, split, summary = build_protocol(
        archive_paths=[archive],
        config=ProtocolConfig(seed=3),
        output_dir=tmp_path / "out",
        materialize_root=tmp_path / "materialized",
        max_scan_records=3,
        max_eligible_records=1,
    )

    selected_rows = [row for row in rows if row["split"]]
    selected_parents = {row["parent_id"] for row in selected_rows}
    assert summary["experiment_scale"] == "smoke"
    assert summary["records_scanned"] == 3
    assert len(selected_parents) == 1
    selected_parent = next(iter(selected_parents))
    assert sum(row["parent_id"] == selected_parent for row in selected_rows) == 1
    assert sum(map(len, split.values())) == len(selected_rows)


def test_protocol_summary_fails_when_no_eligible_records_exist(tmp_path):
    archive = tmp_path / "abc_0000_parsed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(source("a" * 24), pickle.dumps(make_cad(faces=9)))

    _, split, summary = build_protocol(
        archive_paths=[archive],
        config=ProtocolConfig(seed=3),
        output_dir=tmp_path / "out",
        materialize_root=tmp_path / "materialized",
    )

    assert split == {"train": [], "val": [], "test": []}
    assert summary["status"] == "FAILED"
    assert summary["failure_reasons"] == ["no_eligible_records"]


def test_cap_overshoot_uses_only_one_indivisible_parent_group(tmp_path):
    archive = tmp_path / "abc_0000_parsed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for part in range(2):
            handle.writestr(source("a" * 24, part, part + 1), pickle.dumps(make_cad()))
        for part in range(3):
            handle.writestr(source("b" * 24, part, part + 3), pickle.dumps(make_cad()))

    rows, _, summary = build_protocol(
        archive_paths=[archive],
        config=ProtocolConfig(seed=3),
        output_dir=tmp_path / "out",
        materialize_root=tmp_path / "materialized",
        max_eligible_records=1,
    )

    selected = [row for row in rows if row["split"]]
    assert len(selected) == 2
    assert len({row["parent_id"] for row in selected}) == 1
    assert summary["max_eligible_records"] == 1
    assert summary["eligible_cap_overshoot_records"] == 1
    assert summary["eligible_cap_overshoot_parent_id"] == "a" * 24


def test_protocol_summary_fails_when_an_archive_member_cannot_be_loaded(tmp_path):
    archive = tmp_path / "abc_0000_parsed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(source("a" * 24, index=1), pickle.dumps(make_cad()))
        handle.writestr(source("b" * 24, index=2), b"not a pickle")

    _, _, summary = build_protocol(
        archive_paths=[archive],
        config=ProtocolConfig(seed=3),
        output_dir=tmp_path / "out",
        materialize_root=tmp_path / "materialized",
    )

    assert summary["status"] == "FAILED"
    assert summary["archive_member_load_failures"] == 1
    assert summary["failure_reasons"] == ["archive_member_load_failures"]


def test_cli_returns_nonzero_for_failed_protocol(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "abc_0000_parsed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(source("a" * 24), pickle.dumps(make_cad(faces=9)))

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_cad_protocol.py"),
            "--archive-root",
            str(archive_root),
            "--chunks",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
            "--materialize-root",
            str(tmp_path / "materialized"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["failure_reasons"] == ["no_eligible_records"]
