"""Build VQ-VAE patch shards directly from parsed zip archives.

This is the disk-safe path for local machines: it reads each
`abc_XXXX_parsed.zip` archive and writes the same VQ patch shard format used by
`build_vqvae_patch_shards.py`, without materializing parsed shard files first.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_parsed_shards import parse_chunks  # noqa: E402
from run_parsed_shard_cycle import validate_zip_members  # noqa: E402
from sharded_data import PATCH_SHARD_FORMAT, dump_shard_record, open_shard_writer, shard_suffix  # noqa: E402
from vqvae_sampling import patch_records_from_parsed  # noqa: E402


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def archive_path_for_chunk(archive_root: Path, chunk_id: str) -> Path:
    return Path(archive_root) / f"{chunk_id}_parsed.zip"


def output_path_for_index(output_root: Path, index: int, compression: str) -> Path:
    return output_root / f"vq_patch_shard_{index:04d}{shard_suffix(compression)}"


def discover_archive_chunks(archive_root: Path, chunks: str | None) -> list[str]:
    selected = parse_chunks(chunks)
    if selected is None:
        discovered = []
        for path in sorted(Path(archive_root).glob("abc_*_parsed.zip")):
            name = path.name
            if len(name) >= len("abc_0000_parsed.zip"):
                discovered.append(name.split("_parsed.zip", 1)[0])
        return sorted(discovered)
    return sorted(selected)


class PatchShardWriter:
    def __init__(
        self,
        output_root: Path,
        manifest: Path,
        *,
        compression: str,
        compression_level: int,
        patches_per_shard: int,
        header: dict[str, Any],
    ) -> None:
        self.output_root = Path(output_root)
        self.manifest = Path(manifest)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.patches_per_shard = int(patches_per_shard)
        self.header = dict(header)
        self.rows: list[dict[str, Any]] = []
        self.shard_index = 0
        self.handle_cm = None
        self.handle = None
        self.current_path: Path | None = None
        self.current_row: dict[str, Any] | None = None

    def __enter__(self) -> "PatchShardWriter":
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self._open_next()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._finish_current(remove_empty=True)
        elif self.handle_cm is not None:
            self.handle_cm.__exit__(exc_type, exc, tb)

    def _open_next(self) -> None:
        self.current_path = output_path_for_index(self.output_root, self.shard_index, self.compression)
        self.handle_cm = open_shard_writer(self.current_path, compression=self.compression, level=self.compression_level)
        self.handle = self.handle_cm.__enter__()
        self.current_row = {
            "path": str(self.current_path),
            "patch_count": 0,
            "source_count": 0,
            "surface_count": 0,
            "edge_count": 0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        dump_shard_record(self.handle, self.header)

    def _finish_current(self, *, remove_empty: bool) -> None:
        if self.handle_cm is None or self.current_path is None or self.current_row is None:
            return
        self.handle_cm.__exit__(None, None, None)
        self.handle_cm = None
        self.handle = None
        if int(self.current_row["patch_count"]) > 0:
            self.current_row["shard_bytes"] = self.current_path.stat().st_size
            self.current_row["status"] = "built"
            self.rows.append(self.current_row)
            append_jsonl(self.manifest, self.current_row)
        elif remove_empty and self.current_path.exists():
            self.current_path.unlink()
        self.current_path = None
        self.current_row = None

    def write_source_patches(self, patches: list[dict[str, Any]], *, chunk_id: str) -> None:
        if not patches:
            return
        assert self.handle is not None
        assert self.current_row is not None
        self.current_row["source_count"] += 1
        for patch in patches:
            record = dict(patch)
            record["record_type"] = "vq_patch"
            record["chunk_id"] = chunk_id
            dump_shard_record(self.handle, record)
            self.current_row["patch_count"] += 1
            if record["kind"] == "surface":
                self.current_row["surface_count"] += 1
            elif record["kind"] == "edge":
                self.current_row["edge_count"] += 1

            if int(self.current_row["patch_count"]) >= self.patches_per_shard:
                self._finish_current(remove_empty=False)
                self.shard_index += 1
                self._open_next()


def archive_members(zip_path: Path, chunk_id: str) -> list[str]:
    validate_zip_members(zip_path, chunk_id)
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = []
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.endswith(".pkl") and not member.is_dir():
                names.append(name)
    return sorted(names)


def build_patch_shards_from_archives(
    *,
    archive_root: Path,
    patch_shard_root: Path,
    manifest: Path,
    chunks: str | None = "all",
    compression: str = "zstd",
    compression_level: int = 6,
    patches_per_shard: int = 100000,
    complex_min_faces: int = 12,
    complex_min_edges: int = 20,
    max_source_faces: int = 0,
    max_source_edges: int = 0,
    resume: bool = False,
    overwrite_incomplete: bool = False,
) -> dict[str, Any]:
    archive_root = Path(archive_root)
    patch_shard_root = Path(patch_shard_root)
    manifest = Path(manifest)
    summary_path = patch_shard_root / "_summary.json"
    if resume and summary_path.exists() and any(patch_shard_root.glob("vq_patch_shard_*.pkl*")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = dict(summary)
        summary["status"] = "SKIPPED_EXISTING"
        return summary
    if overwrite_incomplete and patch_shard_root.exists() and not summary_path.exists():
        shutil.rmtree(patch_shard_root)

    patch_shard_root.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        manifest.unlink()

    header = {
        "record_type": "vq_patch_shard_header",
        "format": PATCH_SHARD_FORMAT,
        "patches_per_shard": int(patches_per_shard),
        "compression": compression,
        "complex_min_faces": int(complex_min_faces),
        "complex_min_edges": int(complex_min_edges),
        "max_source_faces": int(max_source_faces),
        "max_source_edges": int(max_source_edges),
        "source": "parsed_zip_archives",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    chunks_to_process = discover_archive_chunks(archive_root, chunks)
    skipped_by_cap = 0
    total_sources = 0
    failed_sources = 0
    archives_seen = 0

    with PatchShardWriter(
        patch_shard_root,
        manifest,
        compression=compression,
        compression_level=compression_level,
        patches_per_shard=patches_per_shard,
        header=header,
    ) as writer:
        for chunk_id in chunks_to_process:
            zip_path = archive_path_for_chunk(archive_root, chunk_id)
            if not zip_path.exists():
                raise FileNotFoundError(f"missing archive for {chunk_id}: {zip_path}")
            archives_seen += 1
            members = archive_members(zip_path, chunk_id)
            with zipfile.ZipFile(zip_path, "r") as archive:
                for member in members:
                    total_sources += 1
                    try:
                        with archive.open(member, "r") as handle:
                            data = pickle.load(handle)
                        patches = patch_records_from_parsed(
                            data,
                            member,
                            complex_min_faces=complex_min_faces,
                            complex_min_edges=complex_min_edges,
                        )
                    except Exception:
                        failed_sources += 1
                        continue
                    if not patches:
                        continue
                    n_faces = int(patches[0]["n_faces"])
                    n_edges = int(patches[0]["n_edges"])
                    too_many_faces = max_source_faces > 0 and n_faces > max_source_faces
                    too_many_edges = max_source_edges > 0 and n_edges > max_source_edges
                    if too_many_faces or too_many_edges:
                        skipped_by_cap += 1
                        continue
                    writer.write_source_patches(patches, chunk_id=chunk_id)

    rows = []
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "status": "BUILT",
        "archives_seen": archives_seen,
        "source_records_seen": total_sources,
        "source_records_failed": failed_sources,
        "source_records_skipped_by_cap": skipped_by_cap,
        "patch_shards": len(rows),
        "patches": sum(int(row["patch_count"]) for row in rows),
        "surfaces": sum(int(row["surface_count"]) for row in rows),
        "edges": sum(int(row["edge_count"]) for row in rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--patch-shard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", default="all")
    parser.add_argument("--compression", choices=("zstd", "gzip", "none"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--patches-per-shard", type=int, default=100000)
    parser.add_argument("--complex-min-faces", type=int, default=12)
    parser.add_argument("--complex-min-edges", type=int, default=20)
    parser.add_argument("--max-source-faces", type=int, default=0)
    parser.add_argument("--max-source-edges", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_patch_shards_from_archives(
        archive_root=args.archive_root,
        patch_shard_root=args.patch_shard_root,
        manifest=args.manifest,
        chunks=args.chunks,
        compression=args.compression,
        compression_level=args.compression_level,
        patches_per_shard=args.patches_per_shard,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
        max_source_faces=args.max_source_faces,
        max_source_edges=args.max_source_edges,
        resume=args.resume,
        overwrite_incomplete=args.overwrite_incomplete,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
