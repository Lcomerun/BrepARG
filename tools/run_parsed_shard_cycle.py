"""Extract one archived parsed chunk at a time, shard it, verify it, then delete it.

This is the disk-safe driver for machines that cannot hold the full extracted
parsed pool. It leaves the original zip archives untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_parsed_shards import build_chunk_shard, chunk_output_path, parse_chunks, verify_parsed_shard


def archive_path_for_chunk(archive_root: Path, chunk_id: str) -> Path:
    return archive_root / f"{chunk_id}_parsed.zip"


def validate_zip_members(zip_path: Path, expected_chunk: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            parts = [part for part in name.split("/") if part]
            if not parts:
                continue
            if parts[0] != expected_chunk:
                raise ValueError(f"{zip_path} contains unexpected top-level path {parts[0]!r}")
            if any(part == ".." for part in parts):
                raise ValueError(f"{zip_path} contains unsafe member {member.filename!r}")


def extract_chunk(zip_path: Path, parsed_root: Path, chunk_id: str) -> Path:
    validate_zip_members(zip_path, chunk_id)
    parsed_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(parsed_root)
    chunk_dir = parsed_root / chunk_id
    if not chunk_dir.exists():
        raise FileNotFoundError(f"{zip_path} did not extract {chunk_dir}")
    return chunk_dir


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Disk-safe parsed archive -> parsed shard cycle.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cycle-manifest", type=Path)
    parser.add_argument("--chunks", default="all")
    parser.add_argument("--compression", choices=("zstd", "gzip", "none"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete-after-verify", action="store_true")
    args = parser.parse_args()

    selected = parse_chunks(args.chunks)
    if selected is None:
        selected = {f"abc_{idx:04d}" for idx in range(100)}
    cycle_manifest = args.cycle_manifest or (args.shard_root / "_cycle_manifest.jsonl")

    for chunk_id in sorted(selected):
        started = time.time()
        shard_path = chunk_output_path(args.shard_root, chunk_id, args.compression)
        if args.resume and shard_path.exists():
            verified = verify_parsed_shard(shard_path, expected_chunk=chunk_id)
            row = {
                "chunk": chunk_id,
                "status": "skipped_existing_verified",
                "shard": str(shard_path),
                **verified,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            append_jsonl(cycle_manifest, row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            continue

        zip_path = archive_path_for_chunk(args.archive_root, chunk_id)
        if not zip_path.exists():
            raise FileNotFoundError(f"missing archive for {chunk_id}: {zip_path}")

        chunk_dir = args.parsed_root / chunk_id
        if not chunk_dir.exists():
            print(f"extract {chunk_id} from {zip_path}", flush=True)
            chunk_dir = extract_chunk(zip_path, args.parsed_root, chunk_id)
        else:
            print(f"use existing extracted {chunk_dir}", flush=True)

        row = build_chunk_shard(
            chunk_dir,
            parsed_root=args.parsed_root,
            shard_root=args.shard_root,
            compression=args.compression,
            compression_level=args.compression_level,
            resume=args.resume,
            delete_after_verify=args.delete_after_verify,
        )
        row["cycle_elapsed_seconds"] = round(time.time() - started, 3)
        append_jsonl(args.manifest, row)
        append_jsonl(cycle_manifest, row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
