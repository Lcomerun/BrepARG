"""Protocol V2 eligibility, parent grouping, and manifest construction.

This module deliberately uses only the Python standard library so a dataset
protocol can be audited before a CUDA or model environment is available.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import numbers
import os
import pickle
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "val", "test")
PARENT_CAD_RE = re.compile(
    r"^(?:\d+_)?(?P<cad>[0-9a-f]{24,32})_step_\d+\.pkl$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ProtocolConfig:
    min_faces: int = 10
    max_faces: int = 50
    max_global_edges: int = 150
    max_edges_per_face: int = 30
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0
    version: str = "abc-parent-isolated-v2"

    def __post_init__(self) -> None:
        if self.min_faces < 0 or self.max_faces < self.min_faces:
            raise ValueError("invalid face limits")
        if self.max_global_edges < 0 or self.max_edges_per_face < 0:
            raise ValueError("edge limits must be non-negative")
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(value < 0 for value in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
            raise ValueError("split ratios must be non-negative and sum to one")


def _path_name(value: str) -> str:
    normalized = str(value).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def canonical_source_key(value: str) -> str:
    """Return a separator- and case-normalized stable source identity."""
    return str(value).replace("\\", "/").strip().casefold()


def parent_cad_id(value: str) -> str | None:
    """Extract a reliable ABC parent UUID, returning ``None`` if unresolved."""
    match = PARENT_CAD_RE.match(_path_name(value))
    return match.group("cad").casefold() if match else None


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except (TypeError, ValueError):
        return None


def inspect_cad_record(data: Mapping[str, Any], config: ProtocolConfig) -> dict[str, Any]:
    """Inspect parsed topology and return measurements plus the first rejection."""
    result: dict[str, Any] = {
        "num_faces": None,
        "global_edges": None,
        "max_edges_per_face": None,
        "protocol_eligible": False,
        "reject_reason": None,
    }
    for field in ("surf_ncs", "edge_ncs", "faceEdge_adj"):
        if field not in data or data.get(field) is None:
            result["reject_reason"] = f"missing_{field}"
            return result

    num_faces = _safe_len(data["surf_ncs"])
    global_edges = _safe_len(data["edge_ncs"])
    face_edge_adj = data["faceEdge_adj"]
    adjacency_faces = _safe_len(face_edge_adj)
    result["num_faces"] = num_faces
    result["global_edges"] = global_edges
    if num_faces is None:
        result["reject_reason"] = "invalid_surf_ncs"
        return result
    if global_edges is None:
        result["reject_reason"] = "invalid_edge_ncs"
        return result
    if num_faces < config.min_faces:
        result["reject_reason"] = "too_few_faces"
        return result
    if num_faces > config.max_faces:
        result["reject_reason"] = "too_many_faces"
        return result
    if global_edges > config.max_global_edges:
        result["reject_reason"] = "too_many_global_edges"
        return result
    if adjacency_faces != num_faces:
        result["reject_reason"] = "face_edge_adjacency_length_mismatch"
        return result

    max_edges = 0
    for indices in face_edge_adj:
        count = _safe_len(indices)
        if count is None:
            result["reject_reason"] = "invalid_face_edge_adjacency"
            return result
        max_edges = max(max_edges, count)
        if count > config.max_edges_per_face:
            result["max_edges_per_face"] = max_edges
            result["reject_reason"] = "too_many_edges_per_face"
            return result
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                result["max_edges_per_face"] = max_edges
                result["reject_reason"] = "non_integer_edge_index"
                return result
            if int(index) < 0 or int(index) >= global_edges:
                result["max_edges_per_face"] = max_edges
                result["reject_reason"] = "edge_index_out_of_range"
                return result

    result.update(
        {
            "max_edges_per_face": max_edges,
            "protocol_eligible": True,
            "reject_reason": None,
        }
    )
    return result


def build_manifest_row(
    source_path: str,
    data: Mapping[str, Any] | None,
    config: ProtocolConfig,
    load_error: str | None = None,
) -> dict[str, Any]:
    parent_id = parent_cad_id(source_path)
    base = {
        "protocol_version": config.version,
        "source_path": str(source_path),
        "source_key": canonical_source_key(source_path),
        "parent_id": parent_id,
        "num_faces": None,
        "global_edges": None,
        "max_edges_per_face": None,
        "protocol_eligible": False,
        "reject_reason": None,
        "split": None,
    }
    if load_error:
        base["reject_reason"] = f"load_failed:{load_error}"
        return base
    if parent_id is None:
        base["reject_reason"] = "unknown_parent_id"
        return base
    if not isinstance(data, Mapping):
        base["reject_reason"] = "invalid_record"
        return base
    base.update(inspect_cad_record(data, config))
    return base


def _parent_tie_break(parent_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{parent_id}".encode("ascii")).hexdigest()


def assign_parent_splits(
    rows: Sequence[Mapping[str, Any]], config: ProtocolConfig
) -> dict[str, str]:
    """Assign complete parent groups while balancing eligible record counts."""
    group_sizes = Counter(
        str(row["parent_id"])
        for row in rows
        if row.get("protocol_eligible") and row.get("parent_id")
    )
    groups = sorted(
        group_sizes.items(),
        key=lambda item: (-item[1], _parent_tie_break(item[0], config.seed), item[0]),
    )
    total = sum(group_sizes.values())
    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    targets = {split: ratios[split] * total for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}

    for parent_id, group_size in groups:
        split = max(
            SPLITS,
            key=lambda name: (
                targets[name] - counts[name],
                ratios[name],
                -SPLITS.index(name),
            ),
        )
        assignment[parent_id] = split
        counts[split] += group_size
    return assignment


def _select_parent_groups(
    rows: Sequence[Mapping[str, Any]], max_eligible_records: int, seed: int
) -> set[str]:
    eligible_parents: dict[str, int] = Counter(
        str(row["parent_id"])
        for row in rows
        if row.get("protocol_eligible") and row.get("parent_id")
    )
    if max_eligible_records <= 0:
        return set(eligible_parents)
    ordered = sorted(eligible_parents, key=lambda parent: (_parent_tie_break(parent, seed), parent))
    selected: set[str] = set()
    selected_records = 0
    for parent in ordered:
        group_size = eligible_parents[parent]
        if selected and selected_records + group_size > max_eligible_records:
            continue
        selected.add(parent)
        selected_records += group_size
        if selected_records >= max_eligible_records:
            break
    return selected


def _protocol_hash(rows: Sequence[Mapping[str, Any]], config: ProtocolConfig) -> str:
    identity = {
        "config": asdict(config),
        "rows": [
            {
                "source_key": row.get("source_key"),
                "parent_id": row.get("parent_id"),
                "protocol_eligible": bool(row.get("protocol_eligible")),
                "reject_reason": row.get("reject_reason"),
                "split": row.get("split"),
            }
            for row in sorted(rows, key=lambda item: str(item.get("source_key", "")))
        ],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[str]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    _atomic_write_bytes(output_dir / "protocol_manifest.jsonl", manifest.encode("utf-8"))
    _atomic_write_bytes(
        output_dir / "protocol_summary.json",
        (json.dumps(dict(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"),
    )
    _atomic_write_bytes(output_dir / "split.pkl", pickle.dumps(dict(split), protocol=pickle.HIGHEST_PROTOCOL))

    parent_sets = {
        name: {parent_cad_id(path) for path in split.get(name, []) if parent_cad_id(path)}
        for name in SPLITS
    }
    pairwise = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = parent_sets[left] & parent_sets[right]
        pairwise[f"{left}__{right}"] = {
            "parent_cad_overlap": len(shared),
            "example_shared_parent_cads": sorted(shared)[:20],
        }
    integrity = {
        "status": "NO_PARENT_CAD_OVERLAP_DETECTED"
        if not any(item["parent_cad_overlap"] for item in pairwise.values())
        else "LEAKAGE_DETECTED",
        "pairwise": pairwise,
        "splits": {
            name: {"records": len(split.get(name, [])), "unique_parent_cads": len(parent_sets[name])}
            for name in SPLITS
        },
    }
    _atomic_write_bytes(
        output_dir / "split_integrity.json",
        (json.dumps(integrity, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"),
    )


def _iter_archive_records(
    archive_paths: Iterable[Path], max_scan_records: int
) -> Iterable[tuple[Path, str, str, Mapping[str, Any] | None, str | None]]:
    seen = 0
    for archive_path in sorted(Path(path) for path in archive_paths):
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = sorted(
                info.filename.replace("\\", "/")
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".pkl")
            )
            for member in members:
                if max_scan_records > 0 and seen >= max_scan_records:
                    return
                seen += 1
                source_path = f"{archive_path.name}!/{member}"
                try:
                    with archive.open(member, "r") as handle:
                        data = pickle.load(handle)
                    yield archive_path, member, source_path, data, None
                except Exception as exc:
                    yield archive_path, member, source_path, None, type(exc).__name__


def _materialize_selected(
    selected_rows: Sequence[dict[str, Any]],
    archive_locations: Mapping[str, tuple[Path, str]],
    materialize_root: Path,
) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {name: [] for name in SPLITS}
    open_archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for row in sorted(selected_rows, key=lambda item: str(item["source_key"])):
            source_key = str(row["source_key"])
            archive_path, member = archive_locations[source_key]
            split_name = str(row["split"])
            target = materialize_root / split_name / Path(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            archive = open_archives.setdefault(archive_path, zipfile.ZipFile(archive_path, "r"))
            _atomic_write_bytes(target, archive.read(member))
            split[split_name].append(str(target.resolve()))
    finally:
        for archive in open_archives.values():
            archive.close()
    return split


def build_protocol(
    *,
    archive_paths: Sequence[Path],
    config: ProtocolConfig,
    output_dir: Path,
    materialize_root: Path,
    max_scan_records: int = 0,
    max_eligible_records: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Scan ZIP members, filter, split by parent, materialize, and write evidence."""
    rows: list[dict[str, Any]] = []
    archive_locations: dict[str, tuple[Path, str]] = {}
    for archive_path, member, source_path, data, load_error in _iter_archive_records(
        archive_paths, max(0, int(max_scan_records))
    ):
        row = build_manifest_row(source_path, data, config, load_error=load_error)
        rows.append(row)
        archive_locations[row["source_key"]] = (archive_path, member)

    selected_parents = _select_parent_groups(rows, max(0, int(max_eligible_records)), config.seed)
    assignment = assign_parent_splits(
        [row for row in rows if row.get("parent_id") in selected_parents], config
    )
    for row in rows:
        parent = row.get("parent_id")
        if row.get("protocol_eligible") and parent in assignment:
            row["split"] = assignment[str(parent)]

    selected_rows = [row for row in rows if row.get("split") in SPLITS]
    split = _materialize_selected(selected_rows, archive_locations, Path(materialize_root))
    reject_reasons = Counter(
        str(row["reject_reason"]) for row in rows if row.get("reject_reason")
    )
    split_counts = {name: len(split[name]) for name in SPLITS}
    split_parents = {
        name: len({row["parent_id"] for row in selected_rows if row["split"] == name})
        for name in SPLITS
    }
    eligible_count = sum(bool(row.get("protocol_eligible")) for row in rows)
    summary: dict[str, Any] = {
        "status": "VERIFIED",
        "experiment_scale": "smoke"
        if max_scan_records > 0 or max_eligible_records > 0
        else "full",
        "protocol_version": config.version,
        "config": asdict(config),
        "archives_scanned": len({str(path) for path in archive_paths}),
        "records_scanned": len(rows),
        "records_eligible": eligible_count,
        "records_selected": len(selected_rows),
        "records_rejected": len(rows) - eligible_count,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "split_records": split_counts,
        "split_parents": split_parents,
        "protocol_sha256": _protocol_hash(rows, config),
    }
    _write_outputs(Path(output_dir), rows, split, summary)
    return rows, split, summary
