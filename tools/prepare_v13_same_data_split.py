"""Materialize a V13 same-data split from sequence provenance.

``tools/run_sharded_sequence.py`` expects a split pickle mapping
``train``/``val``/``test`` to real parsed ``.pkl`` files. Current V13 sequence
packages may only store parsed-archive member provenance, so this tool
materializes the referenced archive members without building BrepARG SE
VQ-VAE surface/edge source arrays.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_breparg_same_data_inputs import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_SEQUENCE,
    load_parsed_from_archive,
    source_relpath_from_group,
    valid_record,
    write_pickle,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "local_runs"
    / "complex_curved_rootcause_suite_20260715"
    / "experiments"
    / "02_dfs_rcm_ordering"
    / "same_data_split"
)


def read_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def iter_split_groups(package: dict[str, Any], split: str, limit: int) -> list[dict[str, Any]]:
    groups = list(package.get(split, []) or [])
    if int(limit) > 0:
        return groups[: int(limit)]
    return groups


def target_path_for_relpath(output_dir: Path, split: str, relpath: str) -> Path:
    rel = str(relpath).replace("\\", "/").lstrip("/")
    return Path(output_dir) / "parsed_pool" / split / rel


def materialize_group(
    *,
    split: str,
    index: int,
    group: dict[str, Any],
    archive_root: Path,
    output_dir: Path,
    max_faces: int,
    max_edges: int,
    resume_existing: bool,
) -> dict[str, Any]:
    relpath = source_relpath_from_group(group)
    if not relpath:
        return {"status": "skipped", "reason": "missing_source_relpath"}

    target = target_path_for_relpath(output_dir, split, relpath)
    if resume_existing and target.exists() and target.stat().st_size > 0:
        try:
            existing = read_pickle(target)
            ok, reason = valid_record(existing, max_faces=max_faces, max_edges=max_edges)
        except Exception:
            ok = False
            reason = "existing_corrupt"
        if ok:
            return {
                "status": "written",
                "path": str(target),
                "manifest": {
                    "split": split,
                    "sequence_index": int(index),
                    "source_relpath": relpath,
                    "materialized_path": str(target),
                    "faces": int(len(existing["surf_ncs"])),
                    "edges": int(len(existing["edge_ncs"])),
                    "reused_existing": True,
                },
            }
        if reason == "existing_corrupt":
            target.unlink(missing_ok=True)

    try:
        record = load_parsed_from_archive(relpath, archive_root)
    except Exception as exc:
        return {"status": "skipped", "reason": f"load_failed:{type(exc).__name__}"}

    ok, reason = valid_record(record, max_faces=max_faces, max_edges=max_edges)
    if not ok:
        return {"status": "skipped", "reason": reason}

    write_pickle(target, record)
    return {
        "status": "written",
        "path": str(target),
        "manifest": {
            "split": split,
            "sequence_index": int(index),
            "source_relpath": relpath,
            "materialized_path": str(target),
            "faces": int(len(record["surf_ncs"])),
            "edges": int(len(record["edge_ncs"])),
            "reused_existing": False,
        },
    }


def prepare_v13_same_data_split(
    *,
    sequence_path: Path,
    archive_root: Path,
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    max_faces: int,
    max_edges: int,
    overwrite: bool = False,
    workers: int = 1,
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "v13_same_data_split_manifest.jsonl"
    summary_path = output_dir / "v13_same_data_split_summary.json"
    split_path = output_dir / "split.pkl"

    if overwrite:
        for path in (manifest_path, summary_path, split_path):
            if path.exists():
                path.unlink()
    elif split_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        summary["status"] = "SKIPPED_EXISTING"
        return summary

    if manifest_path.exists():
        manifest_path.unlink()

    package = read_pickle(Path(sequence_path))
    limits = {"train": int(train_limit), "val": int(val_limit), "test": int(test_limit)}
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_stats: dict[str, dict[str, int]] = {}
    skipped: dict[str, int] = {}

    for split in ("train", "val", "test"):
        groups = iter_split_groups(package, split, limits[split])
        seen = len(groups)
        written = 0
        jobs = (
            {
                "split": split,
                "index": index,
                "group": group,
                "archive_root": Path(archive_root),
                "output_dir": output_dir,
                "max_faces": int(max_faces),
                "max_edges": int(max_edges),
                "resume_existing": bool(resume_existing),
            }
            for index, group in enumerate(groups)
        )
        if int(workers) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                results = executor.map(lambda kwargs: materialize_group(**kwargs), jobs)
                for result in results:
                    if result.get("status") == "written":
                        split_paths[split].append(str(result["path"]))
                        written += 1
                        append_jsonl(manifest_path, result["manifest"])
                    else:
                        reason = str(result.get("reason") or "unknown")
                        skipped[reason] = skipped.get(reason, 0) + 1
        else:
            for kwargs in jobs:
                result = materialize_group(**kwargs)
                if result.get("status") == "written":
                    split_paths[split].append(str(result["path"]))
                    written += 1
                    append_jsonl(manifest_path, result["manifest"])
                else:
                    reason = str(result.get("reason") or "unknown")
                    skipped[reason] = skipped.get(reason, 0) + 1

        split_stats[split] = {"seen": int(seen), "written": int(written)}

    write_pickle(split_path, split_paths)
    summary = {
        "status": "VERIFIED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_path": str(sequence_path),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "split_path": str(split_path),
        "manifest": str(manifest_path),
        "limits": {
            "train": int(train_limit),
            "val": int(val_limit),
            "test": int(test_limit),
            "max_faces": int(max_faces),
            "max_edges": int(max_edges),
            "workers": int(workers),
            "resume_existing": bool(resume_existing),
        },
        "splits": split_stats,
        "total_written": int(sum(item["written"] for item in split_stats.values())),
        "skipped": dict(sorted(skipped.items())),
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--val-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--max-faces", type=int, default=50)
    parser.add_argument("--max-edges", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_v13_same_data_split(
        sequence_path=args.sequence,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        max_faces=args.max_faces,
        max_edges=args.max_edges,
        overwrite=args.overwrite,
        workers=args.workers,
        resume_existing=args.resume_existing,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
