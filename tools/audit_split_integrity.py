"""Audit train/validation/test split overlap and sequence-length balance.

The ABC corpus can contain several ``*_step_NNN.pkl`` parts produced from one
parent CAD UUID.  Exact path de-duplication therefore is insufficient: two
different part files can still expose geometry from the same source CAD across
training, validation, and test splits.  This tool reports both levels.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
SOURCE_KEYS = ("source_relpath", "source_path", "path", "file_path", "pkl_path")
PARENT_CAD_RE = re.compile(
    r"^(?:\d+_)?(?P<cad>[0-9a-f]{24,32})_step_\d+\.pkl$",
    flags=re.IGNORECASE,
)


def source_path_of(record: Any) -> str | None:
    if isinstance(record, (str, Path)):
        return str(record)
    if not isinstance(record, dict):
        return None
    for key in SOURCE_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    original = record.get("original")
    if isinstance(original, dict):
        for key in SOURCE_KEYS:
            value = original.get(key)
            if value:
                return str(value)
    return None


def sequence_length_of(record: Any) -> int | None:
    if not isinstance(record, dict):
        return None
    original = record.get("original")
    if isinstance(original, dict):
        ids = original.get("input_ids")
    else:
        ids = record.get("input_ids")
    if ids is None:
        return None
    try:
        return int(len(ids))
    except TypeError:
        return None


def _path_name(value: str) -> str:
    normalized = str(value).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def normalize_source_path(value: str) -> str:
    """Normalize separators and case while retaining the complete source path."""
    return str(value).replace("\\", "/").strip().casefold()


def canonical_source_path(value: str) -> str:
    """Return a split-root-independent source key where provenance permits it."""
    normalized = normalize_source_path(value)
    if "!/" in normalized:
        return normalized.split("!/", 1)[1].lstrip("/")
    match = re.search(r"(?:^|/)(abc_\d{4}/[^/]+\.pkl)$", normalized)
    if match:
        return match.group(1)
    return normalized


def parent_cad_id(value: str) -> str:
    name = _path_name(value)
    match = PARENT_CAD_RE.match(name)
    if match:
        return match.group("cad").casefold()
    return re.sub(r"_step_\d+\.pkl$", "", name, flags=re.IGNORECASE).casefold()


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return int(ordered[index])


def summarize_lengths(values: Iterable[int]) -> dict[str, Any] | None:
    values = [int(value) for value in values]
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "allowed_at_1024": sum(value <= 1024 for value in values),
        "allowed_at_1536": sum(value <= 1536 for value in values),
        "allowed_at_2048": sum(value <= 2048 for value in values),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def audit_package(package: dict[str, Any], source: str = "") -> dict[str, Any]:
    rows: dict[str, list[Any]] = {}
    paths: dict[str, list[str]] = {}
    normalized_paths: dict[str, set[str]] = {}
    canonical_paths: dict[str, set[str]] = {}
    basenames: dict[str, set[str]] = {}
    parent_ids: dict[str, list[str]] = {}
    parent_sets: dict[str, set[str]] = {}
    lengths: dict[str, list[int]] = {}

    package_kind = "path_split"
    for split in SPLITS:
        split_rows = package.get(split, [])
        if not isinstance(split_rows, list):
            split_rows = []
        rows[split] = split_rows
        if any(isinstance(row, dict) for row in split_rows):
            package_kind = "sequence_package"
        split_paths = [source_path_of(row) for row in split_rows]
        paths[split] = [path for path in split_paths if path]
        normalized_paths[split] = {normalize_source_path(path) for path in paths[split]}
        canonical_paths[split] = {canonical_source_path(path) for path in paths[split]}
        basenames[split] = {_path_name(path).casefold() for path in paths[split]}
        parent_ids[split] = [parent_cad_id(path) for path in paths[split]]
        parent_sets[split] = set(parent_ids[split])
        lengths[split] = [
            length for length in (sequence_length_of(row) for row in split_rows) if length is not None
        ]

    pairwise: dict[str, dict[str, Any]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_parents = parent_sets[left] & parent_sets[right]
        key = f"{left}__{right}"
        pairwise[key] = {
            "exact_source_path_overlap": len(normalized_paths[left] & normalized_paths[right]),
            "canonical_source_path_overlap": len(canonical_paths[left] & canonical_paths[right]),
            "basename_overlap": len(basenames[left] & basenames[right]),
            "parent_cad_overlap": len(shared_parents),
            f"{left}_records_in_shared_parent_cads": sum(
                parent in shared_parents for parent in parent_ids[left]
            ),
            f"{right}_records_in_shared_parent_cads": sum(
                parent in shared_parents for parent in parent_ids[right]
            ),
            "example_shared_parent_cads": sorted(shared_parents)[:20],
        }

    split_reports: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        other_parents = set().union(*(parent_sets[name] for name in SPLITS if name != split))
        shared_parents = parent_sets[split] & other_parents
        affected_records = sum(parent in other_parents for parent in parent_ids[split])
        split_reports[split] = {
            "records": len(rows[split]),
            "records_with_source_path": len(paths[split]),
            "unique_exact_source_paths": len(normalized_paths[split]),
            "unique_canonical_source_paths": len(canonical_paths[split]),
            "unique_basenames": len(basenames[split]),
            "unique_parent_cads": len(parent_sets[split]),
            "records_sharing_parent_with_other_split": affected_records,
            "records_sharing_parent_with_other_split_fraction": _fraction(
                affected_records, len(parent_ids[split])
            ),
            "parent_cads_shared_with_other_split": len(shared_parents),
            "parent_cads_shared_with_other_split_fraction": _fraction(
                len(shared_parents), len(parent_sets[split])
            ),
            "sequence_length": summarize_lengths(lengths[split]),
        }

    return {
        "status": "LEAKAGE_DETECTED"
        if any(row["parent_cad_overlap"] for row in pairwise.values())
        else "NO_PARENT_CAD_OVERLAP_DETECTED",
        "source": str(source),
        "package_kind": package_kind,
        "parent_cad_rule": "hex UUID from <optional-index>_<uuid>_step_<part>.pkl",
        "splits": split_reports,
        "pairwise": pairwise,
        "all_three_parent_cad_overlap": len(
            parent_sets["train"] & parent_sets["val"] & parent_sets["test"]
        ),
        "interpretation": (
            "Exact source records may be disjoint while different STEP parts from the same parent CAD "
            "occur in multiple splits. Parent-CAD overlap invalidates an independent split protocol."
        ),
    }


def load_package(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        package = pickle.load(handle)
    if not isinstance(package, dict):
        raise ValueError(f"expected a dict package, got {type(package).__name__}")
    return package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact-record and parent-CAD overlap across train/val/test splits."
    )
    parser.add_argument("input", type=Path, help="A split.pkl or sequence package pickle.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_package(load_package(args.input), source=str(args.input.resolve()))
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
