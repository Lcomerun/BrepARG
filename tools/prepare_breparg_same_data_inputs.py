"""Prepare same-data input files for original BrepARG fallback training.

Original BrepARG training expects a split pickle whose values are real parsed
``.pkl`` paths, plus separate surface and edge patch source pickles for SE
VQ-VAE training. The V13 local data is stored as parsed zip archives and the
current sequence package only stores archive member provenance, so this tool
materializes a medium same-data pool without extracting the full archive tree.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "sequences_fsq_rcm.pkl"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "ABC" / "processed" / "abc_parsed_full_archives"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "local_runs"
    / "complex_curved_rootcause_suite_20260715"
    / "experiments"
    / "03b_breparg_same_data_training_fallback"
    / "data"
)


def read_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def source_relpath_from_group(group: dict[str, Any]) -> str | None:
    for key in ("source_relpath", "relpath"):
        value = group.get(key)
        if value:
            return str(value).replace("\\", "/")
    original = group.get("original")
    if isinstance(original, dict):
        for key in ("source_relpath", "relpath"):
            value = original.get(key)
            if value:
                return str(value).replace("\\", "/")
    for key in ("source_path", "path", "file_path", "pkl_path"):
        value = group.get(key)
        if not value and isinstance(original, dict):
            value = original.get(key)
        if not value:
            continue
        text = str(value).replace("\\", "/")
        if "!/" in text:
            return text.split("!/", 1)[1]
        match = re.search(r"(abc_\d{4}/[^?#]+?\.pkl)$", text)
        if match:
            return match.group(1)
    return None


def archive_path_for_relpath(relpath: str, archive_root: Path) -> Path:
    rel = str(relpath).replace("\\", "/").lstrip("/")
    chunk = rel.split("/", 1)[0]
    if not re.fullmatch(r"abc_\d{4}", chunk):
        raise ValueError(f"cannot infer ABC chunk from relpath: {relpath!r}")
    return Path(archive_root) / f"{chunk}_parsed.zip"


def load_parsed_from_archive(relpath: str, archive_root: Path) -> dict[str, Any]:
    archive = archive_path_for_relpath(relpath, archive_root)
    if not archive.exists():
        raise FileNotFoundError(f"missing parsed archive: {archive}")
    member = str(relpath).replace("\\", "/").lstrip("/")
    with zipfile.ZipFile(archive, "r") as zf:
        with zf.open(member, "r") as handle:
            return pickle.load(handle)


def materialized_path_for_relpath(output_dir: Path, split: str, relpath: str) -> Path:
    rel = str(relpath).replace("\\", "/").lstrip("/")
    return Path(output_dir) / "parsed_pool" / split / rel


def valid_record(record: dict[str, Any], max_faces: int, max_edges: int) -> tuple[bool, str]:
    surf = np.asarray(record.get("surf_ncs", []))
    edge = np.asarray(record.get("edge_ncs", []))
    if surf.ndim != 4 or surf.shape[1:] != (32, 32, 3):
        return False, "bad_surf_ncs_shape"
    if edge.ndim != 3 or edge.shape[1:] != (32, 3):
        return False, "bad_edge_ncs_shape"
    if len(surf) == 0 or len(edge) == 0:
        return False, "empty_surface_or_edge"
    if int(max_faces) > 0 and len(surf) > int(max_faces):
        return False, "too_many_faces"
    if int(max_edges) > 0 and len(edge) > int(max_edges):
        return False, "too_many_edges"
    required = ("surf_bbox_wcs", "edge_bbox_wcs", "edgeFace_adj", "faceEdge_adj")
    for key in required:
        if key not in record:
            return False, f"missing_{key}"
    return True, "ok"


def iter_split_groups(package: dict[str, Any], split: str, limit: int) -> list[dict[str, Any]]:
    groups = list(package.get(split, []) or [])
    if int(limit) > 0:
        return groups[: int(limit)]
    return groups


def prepare_same_data_inputs(
    *,
    sequence_path: Path,
    archive_root: Path,
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    max_faces: int,
    max_edges: int,
    surface_patch_limit: int,
    edge_patch_limit: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "same_data_input_manifest.jsonl"
    summary_path = output_dir / "same_data_input_summary.json"
    split_path = output_dir / "same_data_split.pkl"
    surfaces_path = output_dir / "deduplicated_surface_source.pkl"
    edges_path = output_dir / "deduplicated_edge_source.pkl"

    if overwrite:
        for path in (manifest_path, summary_path, split_path, surfaces_path, edges_path):
            if path.exists():
                path.unlink()
    elif split_path.exists() and surfaces_path.exists() and edges_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        summary["status"] = "SKIPPED_EXISTING"
        return summary

    if manifest_path.exists():
        manifest_path.unlink()

    package = read_pickle(Path(sequence_path))
    limits = {"train": int(train_limit), "val": int(val_limit), "test": int(test_limit)}
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_stats: dict[str, dict[str, int]] = {}
    surface_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    skipped: dict[str, int] = {}

    for split in ("train", "val", "test"):
        seen = 0
        written = 0
        for index, group in enumerate(iter_split_groups(package, split, limits[split])):
            seen += 1
            relpath = source_relpath_from_group(group)
            if not relpath:
                skipped["missing_source_relpath"] = skipped.get("missing_source_relpath", 0) + 1
                continue
            try:
                record = load_parsed_from_archive(relpath, Path(archive_root))
            except Exception as exc:
                reason = f"load_failed:{type(exc).__name__}"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            ok, reason = valid_record(record, max_faces=max_faces, max_edges=max_edges)
            if not ok:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            target = materialized_path_for_relpath(output_dir, split, relpath)
            write_pickle(target, record)
            split_paths[split].append(str(target))
            written += 1

            surf = np.asarray(record["surf_ncs"], dtype=np.float32)
            edge = np.asarray(record["edge_ncs"], dtype=np.float32)
            if split == "train":
                current_surfaces = sum(len(chunk) for chunk in surface_chunks)
                current_edges = sum(len(chunk) for chunk in edge_chunks)
                if int(surface_patch_limit) <= 0 or current_surfaces < int(surface_patch_limit):
                    remaining = None if int(surface_patch_limit) <= 0 else int(surface_patch_limit) - current_surfaces
                    surface_chunks.append(surf[:remaining] if remaining is not None else surf)
                if int(edge_patch_limit) <= 0 or current_edges < int(edge_patch_limit):
                    remaining = None if int(edge_patch_limit) <= 0 else int(edge_patch_limit) - current_edges
                    edge_chunks.append(edge[:remaining] if remaining is not None else edge)

            append_jsonl(
                manifest_path,
                {
                    "split": split,
                    "sequence_index": int(index),
                    "source_relpath": relpath,
                    "materialized_path": str(target),
                    "faces": int(len(surf)),
                    "edges": int(len(edge)),
                },
            )

        split_stats[split] = {"seen": int(seen), "written": int(written)}

    surfaces = (
        np.concatenate(surface_chunks, axis=0).astype(np.float32)
        if surface_chunks
        else np.zeros((0, 32, 32, 3), dtype=np.float32)
    )
    edges = (
        np.concatenate(edge_chunks, axis=0).astype(np.float32)
        if edge_chunks
        else np.zeros((0, 32, 3), dtype=np.float32)
    )

    write_pickle(split_path, split_paths)
    write_pickle(surfaces_path, surfaces)
    write_pickle(edges_path, edges)

    summary = {
        "status": "VERIFIED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_path": str(sequence_path),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "split_path": str(split_path),
        "surface_list": str(surfaces_path),
        "edge_list": str(edges_path),
        "manifest": str(manifest_path),
        "limits": {
            "train": int(train_limit),
            "val": int(val_limit),
            "test": int(test_limit),
            "surface_patch_limit": int(surface_patch_limit),
            "edge_patch_limit": int(edge_patch_limit),
            "max_faces": int(max_faces),
            "max_edges": int(max_edges),
        },
        "splits": split_stats,
        "surface_patches": int(len(surfaces)),
        "edge_patches": int(len(edges)),
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
    parser.add_argument("--surface-patch-limit", type=int, default=1000000)
    parser.add_argument("--edge-patch-limit", type=int, default=1500000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_same_data_inputs(
        sequence_path=args.sequence,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        max_faces=args.max_faces,
        max_edges=args.max_edges,
        surface_patch_limit=args.surface_patch_limit,
        edge_patch_limit=args.edge_patch_limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
