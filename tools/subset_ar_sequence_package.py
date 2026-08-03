"""Create a small AR training sequence package while preserving metadata."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any


def input_ids_of(group: Any) -> list[int]:
    if not isinstance(group, dict):
        return []
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in (original.get("input_ids") or [])]
    return [int(item) for item in (group.get("input_ids") or [])]


def load_package(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_package(path: Path, package: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(package, handle)


def select_split(groups: list[Any], limit: int, max_seq_len: int) -> tuple[list[Any], dict[str, int]]:
    selected: list[Any] = []
    seen = 0
    skipped_empty = 0
    skipped_long = 0
    for group in groups:
        seen += 1
        ids = input_ids_of(group)
        if not ids:
            skipped_empty += 1
            continue
        if len(ids) > int(max_seq_len):
            skipped_long += 1
            continue
        selected.append(group)
        if int(limit) > 0 and len(selected) >= int(limit):
            break
    return selected, {
        "seen": seen,
        "selected": len(selected),
        "skipped_empty": skipped_empty,
        "skipped_long": skipped_long,
    }


def subset_package(
    package: dict[str, Any],
    *,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    max_seq_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subset = {key: value for key, value in package.items() if key not in {"train", "val", "test"}}
    limits = {"train": int(train_limit), "val": int(val_limit), "test": int(test_limit)}
    stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        selected, split_stats = select_split(list(package.get(split, [])), limits[split], int(max_seq_len))
        subset[split] = selected
        stats[split] = split_stats
    total_selected = sum(int(item["selected"]) for item in stats.values())
    summary = {
        "status": "VERIFIED" if stats["train"]["selected"] > 0 and stats["val"]["selected"] > 0 else "FAILED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limits": {
            "train": int(train_limit),
            "val": int(val_limit),
            "test": int(test_limit),
            "max_seq_len": int(max_seq_len),
        },
        "splits": stats,
        "total_selected": int(total_selected),
    }
    return subset, summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--test-limit", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = load_package(args.sequence)
    subset, summary = subset_package(
        package,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        max_seq_len=args.max_seq_len,
    )
    summary["sequence"] = str(args.sequence)
    summary["output"] = str(args.output)
    write_package(args.output, subset)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0 if summary["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
