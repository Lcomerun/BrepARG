"""Build VQ-VAE patch shards from parsed shards.

This tool should run in the training environment because it unpickles parsed
records and therefore needs numpy. It writes streaming patch shards that the
VQ-VAE trainer can read directly with `NS_VQ_PATCH_SHARD_ROOT`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))

from sharded_data import PATCH_SHARD_FORMAT, dump_shard_record, iter_shard_records, open_shard_writer, shard_suffix
from vqvae_sampling import patch_records_from_parsed


def iter_parsed_payloads(paths: list[Path]):
    for shard_path in paths:
        iterator = iter_shard_records(shard_path)
        header = next(iterator)
        for record in iterator:
            if record.get("record_type") != "parsed_source":
                continue
            yield shard_path, header, record, pickle.loads(record["payload"])


def output_path_for_index(output_root: Path, index: int, compression: str) -> Path:
    return output_root / f"vq_patch_shard_{index:04d}{shard_suffix(compression)}"


def close_patch_shard(handle, path: Path, row: dict) -> dict:
    handle.close()
    row["shard_bytes"] = path.stat().st_size
    row["status"] = "built"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VQ-VAE patch shards from parsed source shards.")
    parser.add_argument("--parsed-shard-root", type=Path, required=True)
    parser.add_argument("--patch-shard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compression", choices=("zstd", "gzip", "none"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--patches-per-shard", type=int, default=100000)
    parser.add_argument("--complex-min-faces", type=int, default=12)
    parser.add_argument("--complex-min-edges", type=int, default=20)
    parser.add_argument("--max-source-faces", type=int, default=0)
    parser.add_argument("--max-source-edges", type=int, default=0)
    args = parser.parse_args()

    parsed_paths = sorted(args.parsed_shard_root.glob("parsed_abc_*.pkl*"))
    if not parsed_paths:
        raise SystemExit(f"no parsed shards found under {args.parsed_shard_root}")
    args.patch_shard_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    current_path = output_path_for_index(args.patch_shard_root, shard_index, args.compression)
    handle_cm = open_shard_writer(current_path, compression=args.compression, level=args.compression_level)
    handle = handle_cm.__enter__()
    current_row = {
        "path": str(current_path),
        "patch_count": 0,
        "source_count": 0,
        "surface_count": 0,
        "edge_count": 0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_shard_record(
        handle,
        {
            "record_type": "vq_patch_shard_header",
            "format": PATCH_SHARD_FORMAT,
            "patches_per_shard": args.patches_per_shard,
            "compression": args.compression,
            "complex_min_faces": args.complex_min_faces,
            "complex_min_edges": args.complex_min_edges,
            "max_source_faces": args.max_source_faces,
            "max_source_edges": args.max_source_edges,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    rows = []
    skipped_by_cap = 0
    total_sources = 0
    for shard_path, _header, source_record, data in iter_parsed_payloads(parsed_paths):
        total_sources += 1
        source_relpath = source_record["source_relpath"]
        patch_records = patch_records_from_parsed(
            data,
            source_relpath,
            complex_min_faces=args.complex_min_faces,
            complex_min_edges=args.complex_min_edges,
        )
        if not patch_records:
            continue
        n_faces = int(patch_records[0]["n_faces"])
        n_edges = int(patch_records[0]["n_edges"])
        too_many_faces = args.max_source_faces > 0 and n_faces > args.max_source_faces
        too_many_edges = args.max_source_edges > 0 and n_edges > args.max_source_edges
        if too_many_faces or too_many_edges:
            skipped_by_cap += 1
            continue

        current_row["source_count"] += 1
        for patch in patch_records:
            patch = dict(patch)
            patch["record_type"] = "vq_patch"
            patch["parsed_shard"] = str(shard_path)
            patch["chunk_id"] = source_record["chunk_id"]
            dump_shard_record(handle, patch)
            current_row["patch_count"] += 1
            if patch["kind"] == "surface":
                current_row["surface_count"] += 1
            elif patch["kind"] == "edge":
                current_row["edge_count"] += 1

            if current_row["patch_count"] >= args.patches_per_shard:
                handle_cm.__exit__(None, None, None)
                current_row["shard_bytes"] = current_path.stat().st_size
                current_row["status"] = "built"
                rows.append(current_row)
                with args.manifest.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(current_row, ensure_ascii=True) + "\n")
                shard_index += 1
                current_path = output_path_for_index(args.patch_shard_root, shard_index, args.compression)
                handle_cm = open_shard_writer(current_path, compression=args.compression, level=args.compression_level)
                handle = handle_cm.__enter__()
                current_row = {
                    "path": str(current_path),
                    "patch_count": 0,
                    "source_count": 0,
                    "surface_count": 0,
                    "edge_count": 0,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                dump_shard_record(
                    handle,
                    {
                        "record_type": "vq_patch_shard_header",
                        "format": PATCH_SHARD_FORMAT,
                        "patches_per_shard": args.patches_per_shard,
                        "compression": args.compression,
                        "complex_min_faces": args.complex_min_faces,
                        "complex_min_edges": args.complex_min_edges,
                        "max_source_faces": args.max_source_faces,
                        "max_source_edges": args.max_source_edges,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

    handle_cm.__exit__(None, None, None)
    if current_row["patch_count"] > 0:
        current_row["shard_bytes"] = current_path.stat().st_size
        current_row["status"] = "built"
        rows.append(current_row)
        with args.manifest.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(current_row, ensure_ascii=True) + "\n")
    elif current_path.exists():
        current_path.unlink()

    summary = {
        "status": "BUILT",
        "parsed_shards": len(parsed_paths),
        "source_records_seen": total_sources,
        "source_records_skipped_by_cap": skipped_by_cap,
        "patch_shards": len(rows),
        "patches": sum(int(row["patch_count"]) for row in rows),
        "surfaces": sum(int(row["surface_count"]) for row in rows),
        "edges": sum(int(row["edge_count"]) for row in rows),
    }
    summary_path = args.patch_shard_root / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
