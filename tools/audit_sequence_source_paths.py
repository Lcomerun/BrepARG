from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


SOURCE_PATH_KEYS = ("source_path", "path", "file_path", "pkl_path")
SPLIT_KEYS = ("train", "val", "test")


def source_path_of(group: dict[str, Any]) -> str | None:
    for key in SOURCE_PATH_KEYS:
        value = group.get(key)
        if value:
            return str(value)
    original = group.get("original")
    if isinstance(original, dict):
        for key in SOURCE_PATH_KEYS:
            value = original.get(key)
            if value:
                return str(value)
    return None


def _record_has_source_path(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return any(record.get(key) for key in SOURCE_PATH_KEYS)


def _split_summary(groups: list[Any], sample_limit: int | None) -> dict[str, Any]:
    scanned_groups = groups if sample_limit is None else groups[:sample_limit]
    total = len(scanned_groups)
    groups_with_source = 0
    original_with_source = 0
    augmented_records = 0
    augmented_with_source = 0
    example_missing_indices: list[int] = []
    example_source_paths: list[str] = []

    for index, group in enumerate(scanned_groups):
        if not isinstance(group, dict):
            if len(example_missing_indices) < 5:
                example_missing_indices.append(index)
            continue

        source_path = source_path_of(group)
        if source_path:
            groups_with_source += 1
            if len(example_source_paths) < 5:
                example_source_paths.append(source_path)
        elif len(example_missing_indices) < 5:
            example_missing_indices.append(index)

        original = group.get("original")
        if _record_has_source_path(original):
            original_with_source += 1

        augmented = group.get("augmented") or []
        if isinstance(augmented, list):
            for record in augmented:
                augmented_records += 1
                if _record_has_source_path(record):
                    augmented_with_source += 1

    coverage = (groups_with_source / total) if total else 0.0
    return {
        "groups_total_in_package": len(groups),
        "groups_scanned": total,
        "groups_with_source_path": groups_with_source,
        "groups_missing_source_path": total - groups_with_source,
        "source_path_coverage": coverage,
        "original_with_source_path": original_with_source,
        "augmented_records": augmented_records,
        "augmented_with_source_path": augmented_with_source,
        "example_missing_indices": example_missing_indices,
        "example_source_paths": example_source_paths,
    }


def summarize_sequence_source_paths(package: dict[str, Any], sample_limit: int | None = None) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    total_groups = 0
    groups_with_source = 0

    for split in SPLIT_KEYS:
        groups = package.get(split, [])
        if not isinstance(groups, list):
            groups = []
        split_summary = _split_summary(groups, sample_limit=sample_limit)
        splits[split] = split_summary
        total_groups += split_summary["groups_scanned"]
        groups_with_source += split_summary["groups_with_source_path"]

    nonempty_splits = [summary for summary in splits.values() if summary["groups_scanned"] > 0]
    all_ready = bool(nonempty_splits) and all(summary["source_path_coverage"] == 1.0 for summary in nonempty_splits)
    val_summary = splits["val"]
    validation_ready = val_summary["groups_scanned"] > 0 and val_summary["source_path_coverage"] == 1.0
    overall_coverage = (groups_with_source / total_groups) if total_groups else 0.0
    recommendation = "source_path_ready" if all_ready else "rebuild_or_refresh_missing_source_paths"

    return {
        "total_groups_scanned": total_groups,
        "groups_with_source_path": groups_with_source,
        "groups_missing_source_path": total_groups - groups_with_source,
        "source_path_coverage": overall_coverage,
        "validation_most_curved_ready": validation_ready,
        "all_splits_source_path_ready": all_ready,
        "recommendation": recommendation,
        "splits": splits,
    }


def load_sequence_package(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        package = pickle.load(handle)
    if not isinstance(package, dict):
        raise ValueError(f"Expected a dict sequence package, got {type(package).__name__}")
    return package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a sequence package preserves parsed-geometry source_path metadata."
    )
    parser.add_argument("sequence", type=Path, help="Path to sequences_fsq_rcm.pkl or another sequence package.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Scan at most this many groups per split.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = load_sequence_package(args.sequence)
    summary = summarize_sequence_source_paths(package, sample_limit=args.sample_limit)
    summary["sequence"] = str(args.sequence)
    summary["sample_limit"] = args.sample_limit

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
