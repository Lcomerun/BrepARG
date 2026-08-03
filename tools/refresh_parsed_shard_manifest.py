"""Refresh a parsed-shard manifest by verifying every shard in a root."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_parsed_shards import verify_parsed_shard


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite a manifest for all verified parsed shards in a directory.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    paths = sorted(args.shard_root.glob("parsed_abc_*.pkl*"))
    if not paths:
        raise SystemExit(f"no parsed shards found under {args.shard_root}")

    rows = []
    for path in paths:
        verified = verify_parsed_shard(path)
        row = {
            "chunk": verified["chunk"],
            "status": "verified_existing",
            "shard": str(path),
            "source_count": verified["source_count"],
            "payload_bytes": verified["payload_bytes"],
            "shard_bytes": verified["shard_bytes"],
            "refreshed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        rows.append(row)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "status": "VERIFIED",
        "shard_root": str(args.shard_root),
        "manifest": str(args.manifest),
        "shards": len(rows),
        "total_sources": sum(int(row["source_count"]) for row in rows),
        "total_payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
        "total_shard_bytes": sum(int(row["shard_bytes"]) for row in rows),
        "chunks": [row["chunk"] for row in rows],
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
