import hashlib
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
if str(IMPROVEMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(IMPROVEMENTS_DIR))


from vqvae_sampling import (  # noqa: E402
    _balanced_round_robin_order,
    audit_train_val_inventories,
    balanced_round_robin_records,
    canonical_patch_hash,
    collect_vqvae_sample_records,
    deduplicate_patch_records,
    remove_train_exact_hash_overlap,
    patch_records_from_parsed,
    rounded_patch_hash,
    surface_curvature_proxy,
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


def test_require_all_paths_validates_remaining_sources_without_retaining_patches(
    tmp_path,
):
    paths = []
    for index, parent in enumerate(("a" * 24, "b" * 24, "c" * 24), start=1):
        path = tmp_path / source(parent, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(parsed(), handle)
        paths.append(path)

    records, summary = collect_vqvae_sample_records(
        paths,
        cap=1,
        oversample_factor=1.0,
        require_parent_id=True,
        require_all_paths=True,
    )

    assert len(records) == 1
    assert summary["loaded_paths"] == 3
    assert summary["failed_paths"] == 0
    assert summary["source_records_available"] == 2


def test_protocol_sampling_deduplicates_before_cap_and_balances_parent_coverage(tmp_path):
    paths = []
    for index, parent in enumerate(("a" * 24, "b" * 24, "c" * 24, "d" * 24), start=1):
        path = tmp_path / source(parent, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = float(index)
        parsed_data = {
            "surf_ncs": np.full((1, 32, 32, 3), value, dtype=np.float32),
            "edge_ncs": np.full((1, 32, 3), value, dtype=np.float32),
        }
        with path.open("wb") as handle:
            pickle.dump(parsed_data, handle)
        paths.append(path)

    records, summary = collect_vqvae_sample_records(
        paths,
        cap=4,
        seed=0,
        require_parent_id=True,
        require_all_paths=True,
        deduplicate_before_cap=True,
        balance_by_parent=True,
        min_parent_coverage=0.75,
    )

    assert len(records) == 4
    assert len({record["parent_id"] for record in records}) == 4
    assert summary["scan_complete"] is True
    assert summary["unique_records_before_cap"] == 8
    assert summary["parent_cads_contributing"] == 4
    assert summary["parent_coverage"] == 1.0
    assert summary["dedup_before_cap"]["duplicates_removed"] == 0


def test_balanced_round_robin_records_is_deterministic_and_parent_fair():
    records = [
        record(f"record-{parent}-{index}", source(parent, index=index), parent)
        for parent in ("a" * 24, "b" * 24, "c" * 24)
        for index in (1, 2, 3)
    ]

    first = balanced_round_robin_records(records, target=6, seed=17)
    second = balanced_round_robin_records(list(reversed(records)), target=6, seed=17)

    assert [item["record_id"] for item in first] == [item["record_id"] for item in second]
    assert {item["parent_id"] for item in first} == {"a" * 24, "b" * 24, "c" * 24}
    assert {item["parent_id"] for item in first}.issuperset({"a" * 24, "b" * 24, "c" * 24})


def test_balanced_round_robin_records_exhausts_uneven_sources():
    parent = "a" * 24
    records = [
        record("short", source(parent, index=1), parent),
        *[
            record(f"long-{index}", source(parent, index=2), parent)
            for index in range(4)
        ],
    ]

    selected = balanced_round_robin_records(records, target=5, seed=0)

    assert {item["record_id"] for item in selected} == {
        "short",
        "long-0",
        "long-1",
        "long-2",
        "long-3",
    }


def test_balanced_round_robin_records_honors_curved_fraction_and_parent_coverage():
    parents = ("a" * 24, "b" * 24, "c" * 24, "d" * 24)
    records = []
    for index, parent in enumerate(parents):
        records.append(
            {
                **record(f"curved-{index}", source(parent, index=1), parent),
                "curvature_score": 1.0 - index * 0.1,
            }
        )
        records.append(
            {
                **record(f"flat-{index}", source(parent, index=2), parent),
                "curvature_score": 0.0,
            }
        )

    selected = balanced_round_robin_records(
        records,
        target=4,
        curved_fraction=0.5,
        seed=0,
    )

    assert {item["record_id"] for item in selected}.issuperset({"curved-0", "curved-1"})
    assert {item["parent_id"] for item in selected} == set(parents)


def test_zero_curved_fraction_uses_seeded_round_robin_representatives():
    parents = ("a" * 24, "b" * 24, "c" * 24, "d" * 24)
    records = []
    for index, parent in enumerate(parents):
        records.extend(
            [
                {
                    **record(f"curved-{index}", source(parent, index=1), parent),
                    "curvature_score": 1.0,
                },
                {
                    **record(f"flat-{index}", source(parent, index=2), parent),
                    "curvature_score": 0.0,
                },
            ]
        )
    expected = []
    seen_parents = set()
    for candidate in _balanced_round_robin_order(records, seed=8):
        if candidate["parent_id"] not in seen_parents:
            expected.append(candidate["record_id"])
            seen_parents.add(candidate["parent_id"])

    selected = balanced_round_robin_records(
        records,
        target=4,
        curved_fraction=0.0,
        seed=8,
    )

    assert [item["record_id"] for item in selected] == expected
    assert sum(item["curvature_score"] > 0 for item in selected) < len(selected)


def test_balanced_sampling_prioritizes_parent_coverage_over_concentrated_curved_quota():
    parents = ("a" * 24, "b" * 24, "c" * 24, "d" * 24)
    records = [
        {
            **record(f"curved-{index}", source(parents[0], index=index + 1), parents[0]),
            "curvature_score": 1.0 - index * 0.1,
        }
        for index in range(3)
    ]
    records.extend(
        {
            **record(f"flat-{index}", source(parent, index=9), parent),
            "curvature_score": 0.0,
        }
        for index, parent in enumerate(parents[1:], start=1)
    )

    selected = balanced_round_robin_records(
        records,
        target=4,
        curved_fraction=0.5,
        seed=0,
    )

    assert {item["parent_id"] for item in selected} == set(parents)
    assert sum(item["curvature_score"] > 0 for item in selected) == 1


def test_curved_replacement_never_duplicates_records_when_target_exceeds_parents():
    parents = ("a" * 24, "b" * 24)
    records = []
    for parent_index, parent in enumerate(parents):
        for patch_index in range(6):
            records.append(
                {
                    **record(
                        f"record-{parent_index}-{patch_index}",
                        source(parent, index=patch_index + 1),
                        parent,
                    ),
                    "curvature_score": 1.0 if patch_index >= 3 else 0.0,
                }
            )

    selected = balanced_round_robin_records(
        records,
        target=8,
        curved_fraction=0.75,
        seed=1,
    )

    assert len(selected) == 8
    assert len({item["record_id"] for item in selected}) == 8


def test_parent_coverage_counts_direct_selected_parents_not_merged_provenance(tmp_path):
    first_parent = "a" * 24
    second_parent = "b" * 24
    paths = []
    for index, parent in enumerate((first_parent, second_parent), start=1):
        path = tmp_path / source(parent, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(parsed(), handle)
        paths.append(path)

    records, summary = collect_vqvae_sample_records(
        paths,
        cap=1,
        seed=0,
        require_parent_id=True,
        require_all_paths=True,
        deduplicate_before_cap=True,
        balance_by_parent=True,
    )

    assert len(records) == 1
    assert summary["parent_cads_contributing"] == 1
    assert summary["parent_coverage"] == 0.5


def test_surface_curvature_proxy_is_rotation_invariant_for_tilted_plane():
    plane = np.zeros((32, 32, 3), dtype=np.float32)
    plane[..., 0] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]
    plane[..., 1] = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :]
    angle = np.deg2rad(37.0)
    tilted = plane.copy()
    tilted[..., 0] = np.cos(angle) * plane[..., 0] + np.sin(angle) * plane[..., 2]
    tilted[..., 2] = -np.sin(angle) * plane[..., 0] + np.cos(angle) * plane[..., 2]

    assert surface_curvature_proxy(plane) == pytest.approx(0.0, abs=1e-7)
    assert surface_curvature_proxy(tilted) == pytest.approx(0.0, abs=1e-7)


def test_require_all_paths_rejects_a_requested_source_with_zero_patches(tmp_path):
    valid_path = tmp_path / source("a" * 24)
    empty_path = tmp_path / source("b" * 24, index=2)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with valid_path.open("wb") as handle:
        pickle.dump(parsed(), handle)
    with empty_path.open("wb") as handle:
        pickle.dump({"surf_ncs": [], "edge_ncs": []}, handle)

    with pytest.raises(RuntimeError, match="zero geometry patches"):
        collect_vqvae_sample_records(
            [valid_path, empty_path],
            cap=1,
            oversample_factor=1.0,
            require_parent_id=True,
            require_all_paths=True,
        )


@pytest.mark.parametrize("filtered_position", [0, 1])
def test_require_all_paths_rejects_source_fully_removed_by_source_caps(
    tmp_path, filtered_position
):
    valid_path = tmp_path / source("a" * 24)
    filtered_path = tmp_path / source("b" * 24, index=2)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with valid_path.open("wb") as handle:
        pickle.dump(parsed(), handle)
    with filtered_path.open("wb") as handle:
        filtered = parsed()
        filtered["surf_ncs"] = np.zeros((60, 32, 32, 3), dtype=np.float32)
        pickle.dump(filtered, handle)
    paths = [valid_path, filtered_path]
    if filtered_position == 0:
        paths.reverse()

    with pytest.raises(RuntimeError, match="zero usable geometry patches"):
        collect_vqvae_sample_records(
            paths,
            cap=1,
            seed=0,
            oversample_factor=1.0,
            max_source_faces=50,
            require_parent_id=True,
            require_all_paths=True,
        )


def test_canonical_hash_uses_kind_shape_and_little_endian_c_float32_bytes():
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    fortran_big_endian = np.asfortranarray(values.astype(">f4"))

    expected = canonical_patch_hash("surface", values)

    assert canonical_patch_hash("surface", fortran_big_endian) == expected
    assert canonical_patch_hash("edge", values) != expected
    assert canonical_patch_hash("surface", values.reshape(1, 4, 3)) != expected


def test_canonical_hash_matches_independent_framed_byte_contract():
    kind = "曲面"
    values = np.asfortranarray(
        np.arange(24, dtype=np.float64).reshape(2, 4, 3).astype(">f8")
    )
    canonical = np.ascontiguousarray(values, dtype="<f4")
    framed = b"".join(
        [
            struct.pack("<Q", len(kind.encode("utf-8"))),
            kind.encode("utf-8"),
            struct.pack("<Q", canonical.ndim),
            *(struct.pack("<Q", dimension) for dimension in canonical.shape),
            canonical.tobytes(order="C"),
        ]
    )

    assert canonical_patch_hash(kind, values) == hashlib.sha256(framed).hexdigest()


def test_float32_exact_hash_equivalence_implies_rounded_hash_equivalence():
    below_rounding_boundary = np.asarray([0.00005 - 2e-13], dtype=np.float64)
    above_rounding_boundary = np.asarray([0.00005 + 2e-13], dtype=np.float64)

    np.testing.assert_array_equal(
        below_rounding_boundary.astype(np.float32),
        above_rounding_boundary.astype(np.float32),
    )
    assert canonical_patch_hash("edge", below_rounding_boundary) == canonical_patch_hash(
        "edge", above_rounding_boundary
    )
    assert rounded_patch_hash("edge", below_rounding_boundary) == rounded_patch_hash(
        "edge", above_rounding_boundary
    )


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


def test_deduplication_merges_singular_and_existing_plural_provenance():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    representative_path = source("a" * 24)
    historical_path = source("b" * 24, index=2)
    records = [
        record(
            "representative",
            representative_path,
            "a" * 24,
            patch,
            provenance_record_ids=["historical"],
            provenance_source_paths=[historical_path],
            provenance_source_keys=[historical_path.casefold()],
            provenance_parent_ids=["b" * 24],
        )
    ]

    deduplicated, _ = deduplicate_patch_records(records)

    assert deduplicated[0]["provenance_record_ids"] == ["historical", "representative"]
    assert deduplicated[0]["provenance_source_paths"] == [
        representative_path,
        historical_path,
    ]
    assert deduplicated[0]["provenance_source_keys"] == [
        representative_path.casefold(),
        historical_path.casefold(),
    ]
    assert deduplicated[0]["provenance_parent_ids"] == ["a" * 24, "b" * 24]


def test_deduplication_does_not_hide_blank_plural_source_provenance_from_audit():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    train, _ = deduplicate_patch_records(
        [
            record(
                "train",
                source("a" * 24),
                "a" * 24,
                patch,
                provenance_source_paths=[""],
            )
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

    with pytest.raises(ValueError, match="invalid source identity.*train"):
        audit_train_val_inventories(train, val)


def test_deduplication_does_not_hide_blank_source_from_exact_duplicate_member():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    train, _ = deduplicate_patch_records(
        [
            record("a-valid", source("a" * 24), "a" * 24, patch),
            record("z-blank", "", "b" * 24, patch.copy()),
        ]
    )
    val = [
        record(
            "val",
            source("c" * 24),
            "c" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="invalid source identity.*train"):
        audit_train_val_inventories(train, val)


def test_deduplication_does_not_hide_missing_source_from_exact_duplicate_member():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    missing_source = record(
        "z-missing",
        source("b" * 24),
        "b" * 24,
        patch.copy(),
    )
    missing_source.pop("source_path")
    missing_source.pop("source_key")
    train, _ = deduplicate_patch_records(
        [
            record("a-valid", source("a" * 24), "a" * 24, patch),
            missing_source,
        ]
    )
    val = [
        record(
            "val",
            source("c" * 24),
            "c" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="invalid source identity.*train"):
        audit_train_val_inventories(train, val)


def test_deduplication_does_not_hide_missing_parent_from_exact_duplicate_member():
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    missing_parent = record(
        "z-missing-parent",
        source("b" * 24),
        "b" * 24,
        patch.copy(),
    )
    missing_parent.pop("parent_id")
    train, _ = deduplicate_patch_records(
        [
            record("a-valid", source("a" * 24), "a" * 24, patch),
            missing_parent,
        ]
    )
    val = [
        record(
            "val",
            source("c" * 24),
            "c" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="unknown parent_id.*train"):
        audit_train_val_inventories(train, val)


def test_deduplication_preserves_non_string_parent_for_strict_validation():
    patch = np.zeros((32, 32, 3), dtype=np.float32)

    class HexLookingParent:
        def __str__(self):
            return "a" * 24

    invalid = record(
        "z-invalid-parent",
        source("b" * 24),
        HexLookingParent(),
        patch.copy(),
    )
    train, _ = deduplicate_patch_records(
        [
            record("a-valid", source("a" * 24), "a" * 24, patch),
            invalid,
        ]
    )
    val = [
        record(
            "val",
            source("c" * 24),
            "c" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="invalid parent_id.*train"):
        audit_train_val_inventories(train, val)


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


def test_cross_split_exact_removal_reports_kind_and_parent_breakdown():
    shared_surface = np.zeros((32, 32, 3), dtype=np.float32)
    shared_edge = np.ones((32, 32, 3), dtype=np.float32)
    train = [
        record("surface", source("a" * 24), "a" * 24, shared_surface),
        record("edge", source("b" * 24), "b" * 24, shared_edge, kind="edge"),
    ]
    val = [
        record("val-surface", source("c" * 24), "c" * 24, shared_surface.copy()),
        record("val-edge", source("d" * 24), "d" * 24, shared_edge.copy(), kind="edge"),
    ]

    kept, summary = remove_train_exact_hash_overlap(train, val)

    assert kept == []
    assert summary["removed_by_kind"] == {"edge": 1, "surface": 1}
    assert summary["train_parents_affected"] == ["a" * 24, "b" * 24]
    assert summary["train_parent_count_affected"] == 2


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


@pytest.mark.parametrize(
    "identity_fields",
    [
        {},
        {"source_path": "", "source_key": ""},
        {"source_path": " \t ", "source_key": " \n "},
    ],
)
def test_train_val_inventory_audit_rejects_missing_or_blank_source_identity(
    identity_fields,
):
    invalid = record("train", source("a" * 24), "a" * 24)
    invalid.pop("source_path")
    invalid.pop("source_key")
    invalid.update(identity_fields)
    train = [
        record(
            "valid-train",
            source("c" * 24, index=3),
            "c" * 24,
            np.full((32, 32, 3), 2.0, dtype=np.float32),
        ),
        invalid,
    ]
    val = [
        record(
            "val",
            source("b" * 24),
            "b" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="invalid source identity.*train"):
        audit_train_val_inventories(train, val)


@pytest.mark.parametrize(
    "parent_id",
    [
        "not-a-hex-parent",
        " " + "a" * 24,
        float("nan"),
        "a" * 23,
        "a" * 33,
    ],
)
def test_train_val_inventory_audit_rejects_invalid_parent_identity(parent_id):
    train = [
        record(
            "valid-train",
            source("c" * 24, index=3),
            "c" * 24,
            np.full((32, 32, 3), 2.0, dtype=np.float32),
        ),
        record("invalid-train", source("a" * 24), parent_id),
    ]
    val = [
        record(
            "val",
            source("b" * 24),
            "b" * 24,
            np.ones((32, 32, 3), dtype=np.float32),
        )
    ]

    with pytest.raises(ValueError, match="invalid parent_id.*train"):
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
        "removed_by_kind": {"surface": 1},
        "train_parents_affected": ["a" * 24],
        "train_parent_count_affected": 1,
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
