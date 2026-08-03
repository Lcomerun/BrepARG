import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
if str(IMPROVEMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPROVEMENTS_DIR))


from vqvae_sampling import (  # noqa: E402
    audit_train_val_inventories,
    canonical_patch_hash,
    deduplicate_patch_records,
    remove_train_exact_hash_overlap,
    patch_records_from_parsed,
    rounded_patch_hash,
)


def source(parent, part=0, index=1):
    return f"abc_0000/{index:08d}_{parent}_step_{part:03d}.pkl"


def parsed(surface=None, edge=None):
    surface = np.zeros((32, 32, 3), dtype=np.float32) if surface is None else surface
    edge = np.zeros((32, 3), dtype=np.float32) if edge is None else edge
    return {
        "surf_ncs": np.asarray([surface], dtype=np.float32),
        "edge_ncs": np.asarray([edge], dtype=np.float32),
    }


def record(record_id, source_path, parent_id, array=None, kind="surface", **extra):
    array = np.zeros((32, 32, 3), dtype=np.float32) if array is None else array
    return {
        "record_id": record_id,
        "source_path": source_path,
        "source_key": source_path.replace("\\", "/").casefold(),
        "parent_id": parent_id,
        "kind": kind,
        "array": np.asarray(array),
        **extra,
    }


def inventory_signature(records):
    return [
        {
            key: value
            for key, value in item.items()
            if key != "array"
        }
        | {"array": np.asarray(item["array"]).tolist()}
        for item in records
    ]


def test_patch_records_preserve_source_and_derive_parent_provenance():
    source_path = source("A" * 24)

    records = patch_records_from_parsed(parsed(), source_path, require_parent_id=True)

    assert {item["source_path"] for item in records} == {source_path}
    assert {item["source_key"] for item in records} == {source_path.casefold()}
    assert {item["parent_id"] for item in records} == {"a" * 24}
    assert {item["kind"] for item in records} == {"surface", "edge"}


def test_patch_records_fail_closed_when_parent_identity_is_unknown():
    with pytest.raises(ValueError, match="unknown parent CAD ID"):
        patch_records_from_parsed(
            parsed(),
            "parsed/train_shape.pkl",
            require_parent_id=True,
        )


def test_patch_records_keep_legacy_default_for_unknown_parent():
    records = patch_records_from_parsed(parsed(), "parsed/train_shape.pkl")

    assert len(records) == 2
    assert {item["source_path"] for item in records} == {"parsed/train_shape.pkl"}
    assert {item["parent_id"] for item in records} == {None}


def test_canonical_hash_uses_kind_shape_and_little_endian_c_float32_bytes():
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    fortran_big_endian = np.asfortranarray(values.astype(">f4"))

    expected = canonical_patch_hash("surface", values)

    assert canonical_patch_hash("surface", fortran_big_endian) == expected
    assert canonical_patch_hash("edge", values) != expected
    assert canonical_patch_hash("surface", values.reshape(1, 4, 3)) != expected


def test_rounded_hash_is_audit_only_and_does_not_drive_exact_deduplication():
    first = np.zeros((32, 32, 3), dtype=np.float32)
    second = first.copy()
    second[0, 0, 0] = 0.00004
    records = [
        record("a", source("a" * 24), "a" * 24, first),
        record("b", source("b" * 24), "b" * 24, second),
    ]

    deduplicated, summary = deduplicate_patch_records(records)

    assert canonical_patch_hash("surface", first) != canonical_patch_hash("surface", second)
    assert rounded_patch_hash("surface", first) == rounded_patch_hash("surface", second)
    assert len(deduplicated) == 2
    assert summary["duplicates_removed"] == 0
    assert summary["rounded_only_duplicate_groups"] == 1


def test_exact_duplicates_merge_without_losing_provenance_or_original_fields():
    patch = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    records = [
        record(
            "record-b",
            source("b" * 24, index=2),
            "b" * 24,
            patch,
            marker="non-representative",
        ),
        record(
            "record-a",
            source("a" * 24, index=1),
            "a" * 24,
            patch.copy(),
            marker="representative",
        ),
    ]

    deduplicated, summary = deduplicate_patch_records(records)

    assert len(deduplicated) == 1
    representative = deduplicated[0]
    assert representative["record_id"] == "record-a"
    assert representative["source_path"] == source("a" * 24, index=1)
    assert representative["parent_id"] == "a" * 24
    assert representative["kind"] == "surface"
    assert representative["marker"] == "representative"
    np.testing.assert_array_equal(representative["array"], patch)
    assert representative["provenance_record_ids"] == ["record-a", "record-b"]
    assert representative["provenance_source_paths"] == [
        source("a" * 24, index=1),
        source("b" * 24, index=2),
    ]
    assert representative["provenance_parent_ids"] == ["a" * 24, "b" * 24]
    assert representative["duplicate_count"] == 1
    assert representative["exact_hash"] == canonical_patch_hash("surface", patch)
    assert representative["rounded_hash"] == rounded_patch_hash("surface", patch)
    assert summary == {
        "input_records": 2,
        "unique_records": 1,
        "duplicates_removed": 1,
        "exact_duplicate_groups": 1,
        "rounded_only_duplicate_groups": 0,
    }


def test_deduplication_representatives_output_and_summary_ignore_input_order():
    shared = np.ones((32, 32, 3), dtype=np.float32)
    unique = np.full((32, 32, 3), 2.0, dtype=np.float32)
    records = [
        record("duplicate-z", source("b" * 24, index=2), "b" * 24, shared),
        record("unique", source("c" * 24, index=3), "c" * 24, unique),
        record("duplicate-a", source("a" * 24, index=1), "a" * 24, shared),
    ]

    first_records, first_summary = deduplicate_patch_records(records)
    second_records, second_summary = deduplicate_patch_records(list(reversed(records)))

    assert inventory_signature(first_records) == inventory_signature(second_records)
    assert first_summary == second_summary
    assert {item["record_id"] for item in first_records} == {"duplicate-a", "unique"}


def test_deduplication_accepts_an_iterable_without_corrupting_summary_counts():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    records = (
        item
        for item in [
            record("a", source("a" * 24), "a" * 24, patch),
            record("b", source("b" * 24), "b" * 24, patch),
        ]
    )

    deduplicated, summary = deduplicate_patch_records(records)

    assert len(deduplicated) == 1
    assert summary["input_records"] == 2
    assert summary["duplicates_removed"] == 1


def test_train_val_inventory_audit_accepts_disjoint_provenance():
    train = [record("train", source("a" * 24), "a" * 24)]
    val_array = np.ones((32, 32, 3), dtype=np.float32)
    val = [record("val", source("b" * 24), "b" * 24, val_array)]

    summary = audit_train_val_inventories(train, val)

    assert summary == {
        "status": "VERIFIED",
        "train_records": 1,
        "val_records": 1,
        "train_source_keys": 1,
        "val_source_keys": 1,
        "train_parent_ids": 1,
        "val_parent_ids": 1,
        "train_exact_hashes": 1,
        "val_exact_hashes": 1,
        "source_key_overlap": [],
        "parent_id_overlap": [],
        "exact_hash_overlap": [],
    }


def test_train_val_inventory_audit_canonicalizes_explicit_source_keys():
    train = [
        record(
            "train",
            source("a" * 24),
            "a" * 24,
            source_key="ABC_0000\\SHARED.PKL",
        )
    ]
    val = [
        record(
            "val",
            source("b" * 24),
            "b" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
            source_key="abc_0000/shared.pkl",
        )
    ]

    with pytest.raises(ValueError, match="source_key overlap"):
        audit_train_val_inventories(train, val)


@pytest.mark.parametrize(
    ("train", "val", "message"),
    [
        (
            [
                record(
                    "train",
                    source("a" * 24),
                    "a" * 24,
                    source_key="shared/source.pkl",
                )
            ],
            [
                record(
                    "val",
                    source("b" * 24),
                    "b" * 24,
                    np.ones((32, 32, 3), dtype=np.float32),
                    source_key="shared/source.pkl",
                )
            ],
            "source_key overlap",
        ),
        (
            [record("train", source("a" * 24, part=0), "a" * 24)],
            [
                record(
                    "val",
                    source("a" * 24, part=1),
                    "a" * 24,
                    np.ones((32, 32, 3), dtype=np.float32),
                )
            ],
            "parent_id overlap",
        ),
        (
            [record("train", source("a" * 24), "a" * 24)],
            [record("val", source("b" * 24), "b" * 24)],
            "exact_hash overlap",
        ),
    ],
)
def test_train_val_inventory_audit_blocks_each_cross_split_overlap(train, val, message):
    with pytest.raises(ValueError, match=message):
        audit_train_val_inventories(train, val)


def test_train_val_inventory_audit_uses_all_merged_provenance():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    other = np.ones((32, 32, 3), dtype=np.float32)
    train, _ = deduplicate_patch_records(
        [
            record("train-a", source("a" * 24), "a" * 24, patch),
            record("train-b", source("b" * 24), "b" * 24, patch),
        ]
    )
    val = [record("val-b", source("b" * 24, part=1), "b" * 24, other)]

    with pytest.raises(ValueError, match="parent_id overlap"):
        audit_train_val_inventories(train, val)


def test_train_val_inventory_audit_fails_closed_on_unknown_parent():
    train = [record("train", source("a" * 24), None)]
    val = [
        record(
            "val",
            source("b" * 24),
            "b" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="unknown parent_id.*train"):
        audit_train_val_inventories(train, val)


def test_remove_train_exact_hash_overlap_keeps_validation_authoritative():
    shared = np.zeros((32, 32, 3), dtype=np.float32)
    train_unique = np.ones((32, 32, 3), dtype=np.float32)
    val_unique = np.full((32, 32, 3), 2.0, dtype=np.float32)
    train, _ = deduplicate_patch_records(
        [
            record("train-shared", source("a" * 24), "a" * 24, shared),
            record("train-unique", source("b" * 24), "b" * 24, train_unique),
        ]
    )
    val, _ = deduplicate_patch_records(
        [
            record("val-shared", source("c" * 24), "c" * 24, shared),
            record("val-unique", source("d" * 24), "d" * 24, val_unique),
        ]
    )

    filtered, summary = remove_train_exact_hash_overlap(train, val)

    assert [item["record_id"] for item in filtered] == ["train-unique"]
    assert summary == {
        "train_records_before": 2,
        "train_records_after": 1,
        "train_records_removed": 1,
        "overlap_hashes_removed": 1,
        "removed_fraction": 0.5,
    }
    assert audit_train_val_inventories(filtered, val)["status"] == "VERIFIED"


def test_train_val_inventory_audit_fails_on_unknown_merged_parent_provenance():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    train, _ = deduplicate_patch_records(
        [
            record("known", source("a" * 24), "a" * 24, patch),
            record("unknown", "parsed/unknown.pkl", None, patch),
        ]
    )
    val = [
        record(
            "val",
            source("b" * 24),
            "b" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="unknown parent_id.*train"):
        audit_train_val_inventories(train, val)
