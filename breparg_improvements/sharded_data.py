"""Streaming helpers for V13 parsed and VQ patch shard files.

Shard files are compressed streams of pickle objects. The first object is a
header dictionary; every following object is one source record or patch record.
This keeps file count low without requiring a whole chunk to fit in memory.
"""

from __future__ import annotations

import gzip
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PARSED_SHARD_FORMAT = "v13.parsed_shard.v1"
PATCH_SHARD_FORMAT = "v13.vq_patch_shard.v1"


def shard_suffix(compression: str) -> str:
    compression = compression.lower()
    if compression == "zstd":
        return ".pkl.zst"
    if compression == "gzip":
        return ".pkl.gz"
    if compression == "none":
        return ".pkl"
    raise ValueError(f"unsupported compression: {compression}")


def infer_compression(path: str | Path) -> str:
    name = str(path).lower()
    if name.endswith(".zst"):
        return "zstd"
    if name.endswith(".gz"):
        return "gzip"
    return "none"


@contextmanager
def open_shard_writer(path: str | Path, compression: str | None = None, level: int = 10):
    path = Path(path)
    compression = compression or infer_compression(path)
    if compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required for .zst shards; install with `pip install zstandard`") from exc
        with path.open("wb") as raw:
            compressor = zstd.ZstdCompressor(level=int(level))
            with compressor.stream_writer(raw) as handle:
                yield handle
    elif compression == "gzip":
        with gzip.open(path, "wb", compresslevel=max(1, min(9, int(level)))) as handle:
            yield handle
    elif compression == "none":
        with path.open("wb") as handle:
            yield handle
    else:
        raise ValueError(f"unsupported compression: {compression}")


@contextmanager
def open_shard_reader(path: str | Path):
    path = Path(path)
    compression = infer_compression(path)
    if compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required for .zst shards; install with `pip install zstandard`") from exc
        with path.open("rb") as raw:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(raw) as handle:
                yield handle
    elif compression == "gzip":
        with gzip.open(path, "rb") as handle:
            yield handle
    elif compression == "none":
        with path.open("rb") as handle:
            yield handle
    else:
        raise ValueError(f"unsupported compression: {compression}")


def dump_shard_record(handle, record: dict) -> None:
    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)


def iter_shard_records(path: str | Path) -> Iterator[dict]:
    with open_shard_reader(path) as handle:
        while True:
            try:
                yield pickle.load(handle)
            except EOFError:
                break


def read_shard_header(path: str | Path) -> dict:
    iterator = iter_shard_records(path)
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(f"empty shard: {path}") from exc
