"""Verify V13 parsed shard files without extracting them."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_parsed_shards import verify_parsed_shard
from sharded_data import iter_shard_records


def deep_unpickle_summary(path: Path, max_records: int = 0) -> dict:
    count = 0
    failures = 0
    total_faces = 0
    total_edges = 0
    iterator = iter_shard_records(path)
    next(iterator)
    for record in iterator:
        if max_records and count >= max_records:
            break
        count += 1
        try:
            data = pickle.loads(record["payload"])
            surfaces = data.get("surf_ncs")
            edges = data.get("edge_ncs")
            total_faces += len(surfaces) if surfaces is not None else 0
            total_edges += len(edges) if edges is not None else 0
        except Exception:
            failures += 1
    return {
        "deep_checked_records": count,
        "deep_unpickle_failures": failures,
        "deep_faces": total_faces,
        "deep_edges": total_edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify parsed shard payload hashes and optional unpickle health.")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--deep-unpickle", action="store_true")
    parser.add_argument("--max-deep-records", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    ok = True
    for path in args.paths:
        try:
            row = {
                "path": str(path),
                "status": "verified",
                **verify_parsed_shard(path),
            }
            if args.deep_unpickle:
                row.update(deep_unpickle_summary(path, max_records=args.max_deep_records))
                if row["deep_unpickle_failures"]:
                    row["status"] = "deep_unpickle_failed"
                    ok = False
        except Exception as exc:
            row = {
                "path": str(path),
                "status": "error",
                "error": repr(exc),
            }
            ok = False
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    payload = {
        "status": "VERIFIED" if ok else "FAILED",
        "shards": rows,
        "count": len(rows),
        "total_sources": sum(int(row.get("source_count") or 0) for row in rows),
        "total_payload_bytes": sum(int(row.get("payload_bytes") or 0) for row in rows),
        "total_shard_bytes": sum(int(row.get("shard_bytes") or 0) for row in rows),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
