"""Build streaming parsed-data shards from extracted ABC parsed chunk folders.

The tool does not unpickle source `.pkl` files. Each shard stores raw source
pickle bytes plus metadata, so it can run in a lightweight Python environment
without numpy. A later server-side step can unpickle records and build VQ patch
shards in the training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))

from sharded_data import PARSED_SHARD_FORMAT, dump_shard_record, iter_shard_records, open_shard_writer, shard_suffix


def parse_chunks(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    chunks: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start.replace("abc_", ""))
            end_idx = int(end.replace("abc_", ""))
            for idx in range(start_idx, end_idx + 1):
                chunks.add(f"abc_{idx:04d}")
        else:
            chunks.add(f"abc_{int(part.replace('abc_', '')):04d}")
    return chunks


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def chunk_output_path(shard_root: Path, chunk_id: str, compression: str) -> Path:
    return shard_root / f"parsed_{chunk_id}{shard_suffix(compression)}"


def verify_parsed_shard(path: Path, expected_chunk: str | None = None, expected_count: int | None = None) -> dict:
    iterator = iter_shard_records(path)
    header = next(iterator)
    if header.get("format") != PARSED_SHARD_FORMAT:
        raise ValueError(f"{path} is not a parsed shard: {header.get('format')}")
    if expected_chunk is not None and header.get("chunk_id") != expected_chunk:
        raise ValueError(f"{path} chunk mismatch: {header.get('chunk_id')} != {expected_chunk}")

    count = 0
    payload_bytes = 0
    bad_hashes = 0
    for record in iterator:
        if record.get("record_type") != "parsed_source":
            raise ValueError(f"{path} contains unexpected record type: {record.get('record_type')}")
        payload = record.get("payload")
        if not isinstance(payload, bytes):
            raise ValueError(f"{path} contains a parsed_source without bytes payload")
        count += 1
        payload_bytes += len(payload)
        if sha256_bytes(payload) != record.get("source_sha256"):
            bad_hashes += 1
    if expected_count is not None and count != expected_count:
        raise ValueError(f"{path} source count mismatch: {count} != {expected_count}")
    if bad_hashes:
        raise ValueError(f"{path} has {bad_hashes} payload hash mismatches")
    return {
        "chunk": header.get("chunk_id"),
        "source_count": count,
        "payload_bytes": payload_bytes,
        "shard_bytes": path.stat().st_size,
    }


def ensure_delete_target_is_safe(chunk_dir: Path, parsed_root: Path) -> None:
    resolved_root = parsed_root.resolve()
    resolved_chunk = chunk_dir.resolve()
    if resolved_chunk == resolved_root or resolved_root not in resolved_chunk.parents:
        raise ValueError(f"refusing to delete outside parsed root: {chunk_dir}")
    if not chunk_dir.name.startswith("abc_"):
        raise ValueError(f"refusing to delete non-chunk directory: {chunk_dir}")


def build_chunk_shard(
    chunk_dir: Path,
    *,
    parsed_root: Path,
    shard_root: Path,
    compression: str,
    compression_level: int,
    resume: bool,
    delete_after_verify: bool,
) -> dict:
    started = time.time()
    chunk_id = chunk_dir.name
    source_files = sorted(chunk_dir.glob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"no .pkl files found in {chunk_dir}")

    shard_root.mkdir(parents=True, exist_ok=True)
    final_path = chunk_output_path(shard_root, chunk_id, compression)
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    if resume and final_path.exists():
        verified = verify_parsed_shard(final_path, expected_chunk=chunk_id)
        row = {
            "chunk": chunk_id,
            "status": "skipped_existing_verified",
            "shard": str(final_path),
            "source_files_on_disk": len(source_files),
            **verified,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if delete_after_verify and chunk_dir.exists():
            ensure_delete_target_is_safe(chunk_dir, parsed_root)
            shutil.rmtree(chunk_dir)
            row["deleted_extracted_chunk"] = True
        return row

    if tmp_path.exists():
        tmp_path.unlink()

    source_bytes = 0
    with open_shard_writer(tmp_path, compression=compression, level=compression_level) as handle:
        header = {
            "record_type": "parsed_shard_header",
            "format": PARSED_SHARD_FORMAT,
            "chunk_id": chunk_id,
            "source_root": str(parsed_root),
            "source_count": len(source_files),
            "payload_encoding": "raw_pickle_bytes",
            "compression": compression,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        dump_shard_record(handle, header)
        for source_path in source_files:
            payload = source_path.read_bytes()
            source_bytes += len(payload)
            relpath = safe_relpath(source_path, parsed_root)
            dump_shard_record(
                handle,
                {
                    "record_type": "parsed_source",
                    "chunk_id": chunk_id,
                    "source_relpath": relpath,
                    "source_name": source_path.name,
                    "source_bytes": len(payload),
                    "source_sha256": sha256_bytes(payload),
                    "payload": payload,
                },
            )

    tmp_path.replace(final_path)
    verified = verify_parsed_shard(final_path, expected_chunk=chunk_id, expected_count=len(source_files))
    row = {
        "chunk": chunk_id,
        "status": "built_verified",
        "shard": str(final_path),
        "source_files_on_disk": len(source_files),
        "source_bytes_on_disk": source_bytes,
        **verified,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if delete_after_verify:
        ensure_delete_target_is_safe(chunk_dir, parsed_root)
        shutil.rmtree(chunk_dir)
        row["deleted_extracted_chunk"] = True
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Build parsed chunk shards from extracted ABC parsed chunk directories.")
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", default="all", help="all, a single chunk such as 7 or abc_0007, or ranges like 0-4")
    parser.add_argument("--compression", choices=("zstd", "gzip", "none"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete-after-verify", action="store_true")
    args = parser.parse_args()

    selected = parse_chunks(args.chunks)
    chunk_dirs = [path for path in sorted(args.parsed_root.glob("abc_*")) if path.is_dir()]
    if selected is not None:
        chunk_dirs = [path for path in chunk_dirs if path.name in selected]
    if not chunk_dirs:
        print(f"no matching chunk directories under {args.parsed_root}", flush=True)
        return 1

    print(
        f"parsed_root={args.parsed_root} shard_root={args.shard_root} chunks={len(chunk_dirs)} "
        f"compression={args.compression} delete_after_verify={args.delete_after_verify}",
        flush=True,
    )
    for chunk_dir in chunk_dirs:
        try:
            row = build_chunk_shard(
                chunk_dir,
                parsed_root=args.parsed_root,
                shard_root=args.shard_root,
                compression=args.compression,
                compression_level=args.compression_level,
                resume=args.resume,
                delete_after_verify=args.delete_after_verify,
            )
            append_jsonl(args.manifest, row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        except Exception as exc:
            row = {
                "chunk": chunk_dir.name,
                "status": "error",
                "error": repr(exc),
                "elapsed_seconds": 0,
            }
            append_jsonl(args.manifest, row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
