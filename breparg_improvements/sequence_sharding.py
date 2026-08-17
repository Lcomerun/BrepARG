import json
import os
import pickle
import re
from pathlib import Path


CHUNK_RE = re.compile(r"abc_\d{4}")

METADATA_KEYS = [
    "vocab_size",
    "special_token_size",
    "face_index_size",
    "se_codebook_size",
    "bbox_index_size",
    "face_index_offset",
    "se_token_offset",
    "bbox_token_offset",
    "se_tokens_per_element",
    "bbox_tokens_per_element",
    "special_tokens",
    "ordering",
]


def chunk_id_from_path(path):
    match = CHUNK_RE.search(str(path))
    if not match:
        raise ValueError(f"cannot find abc chunk id in path: {path}")
    return match.group(0)


def group_split_paths_by_chunk(split):
    grouped = {}
    for split_name in ("train", "val", "test"):
        for path in split.get(split_name, []):
            chunk = chunk_id_from_path(path)
            row = grouped.setdefault(chunk, {"train": [], "val": [], "test": []})
            row[split_name].append(path)
    return dict(sorted(grouped.items()))


def sequence_metadata(package):
    missing = [key for key in METADATA_KEYS if key not in package]
    if missing:
        raise ValueError(f"sequence package missing metadata keys: {missing}")
    return {key: package[key] for key in METADATA_KEYS}


def _input_ids_of(group):
    if not isinstance(group, dict):
        return []
    original = group.get("original")
    if isinstance(original, dict):
        return original.get("input_ids") or []
    return group.get("input_ids") or []


def summarize_sequence_package(package):
    total = sum(len(package.get(split, [])) for split in ("train", "val", "test"))
    vocab_size = int(package.get("vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise ValueError(
            "sequence package is missing a positive vocab_size; "
            "token range audit would be silently skipped")
    max_token = -1
    out_of_vocab = 0
    for split_name in ("train", "val", "test"):
        for group in package.get(split_name, []):
            ids = [int(x) for x in _input_ids_of(group)]
            if not ids:
                continue
            max_token = max(max_token, max(ids))
            if vocab_size and (max(ids) >= vocab_size or min(ids) < 0):
                out_of_vocab += 1
    return {
        "sequences": total,
        "train": len(package.get("train", [])),
        "val": len(package.get("val", [])),
        "test": len(package.get("test", [])),
        "vocab_size": vocab_size,
        "max_token": max_token,
        "out_of_vocab": out_of_vocab,
        "se_tokens_per_element": package.get("se_tokens_per_element"),
        "ordering": package.get("ordering", "RCM"),
    }


def load_sequence_package(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_sequence_package(path, package):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(package, f)
    os.replace(tmp, path)


def merge_sequence_shards(shard_paths, output_path, summary_path=None):
    shard_paths = [Path(path) for path in shard_paths]
    if not shard_paths:
        raise ValueError("no sequence shard paths were provided")
    resolved = [path.resolve() for path in shard_paths]
    duplicates = sorted({str(p) for p in resolved if resolved.count(p) > 1})
    if duplicates:
        raise ValueError(f"duplicate sequence shards would double-count samples: {duplicates}")

    merged = {"train": [], "val": [], "test": []}
    expected_meta = None
    shard_summaries = []

    for shard_path in shard_paths:
        package = load_sequence_package(shard_path)
        meta = sequence_metadata(package)
        if expected_meta is None:
            expected_meta = meta
            merged.update(meta)
            merged["ordering"] = package.get("ordering", "RCM")
        elif meta != expected_meta:
            raise ValueError(f"inconsistent sequence metadata in shard: {shard_path}")

        for split_name in ("train", "val", "test"):
            merged[split_name].extend(package.get(split_name, []))
        shard_summary = summarize_sequence_package(package)
        shard_summary["shard"] = shard_path.name
        shard_summaries.append(shard_summary)

    write_sequence_package(output_path, merged)
    summary = summarize_sequence_package(merged)
    summary["output"] = str(output_path)
    summary["shards"] = len(shard_paths)
    summary["shard_summaries"] = shard_summaries

    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
