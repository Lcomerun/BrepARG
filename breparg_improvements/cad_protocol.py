"""Protocol V2 eligibility, parent grouping, and manifest construction.

The module itself imports only the Python standard library. Scanning real ABC
archives still requires an environment that can import every type stored in
the pickle payload, normally NumPy arrays. It does not require CUDA, PyTorch,
or OpenCascade.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import numbers
import os
import pickle
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "val", "test")
PARENT_CAD_RE = re.compile(
    r"(?:\d+_)?(?P<cad>[0-9a-f]{24,32})_step_\d{3}\.pkl",
    flags=re.IGNORECASE,
)

REJECTION_PRIORITY = (
    "missing_surf_ncs",
    "missing_edge_ncs",
    "missing_faceEdge_adj",
    "invalid_surf_ncs",
    "invalid_edge_ncs",
    "invalid_face_edge_adjacency",
    "too_few_faces",
    "too_many_faces",
    "too_many_global_edges",
    "face_edge_adjacency_length_mismatch",
    "too_many_edges_per_face",
    "non_integer_edge_index",
    "edge_index_out_of_range",
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


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: Path
    original_name: str
    normalized_name: str
    source_path: str
    source_key: str
    materialization_key: str
    crc32: int
    file_size: int


def _path_name(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def canonical_source_key(value: str) -> str:
    """Return a separator- and case-normalized stable source identity."""
    return str(value).replace("\\", "/").strip().casefold()


def _safe_archive_member_name(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    parts = normalized.split("/")
    path = PurePosixPath(normalized)
    drive_like = bool(re.match(r"^[A-Za-z]:", normalized))
    if (
        not normalized
        or path.is_absolute()
        or drive_like
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe archive member path: {value!r}")
    return path.as_posix()


def validate_archive_member_inventory(archive_paths: Iterable[Path]) -> tuple[ArchiveMember, ...]:
    """Return a globally unique, materialization-safe pickle inventory."""
    archive_names: dict[str, Path] = {}
    source_keys: set[str] = set()
    materialization_keys: set[str] = set()
    inventory: list[ArchiveMember] = []
    paths = sorted((Path(path) for path in archive_paths), key=lambda path: str(path.resolve()).casefold())
    for archive_path in paths:
        archive_name_key = archive_path.name.casefold()
        previous = archive_names.get(archive_name_key)
        if previous is not None:
            raise ValueError(
                f"duplicate archive basename: {archive_path.name!r} from {previous} and {archive_path}"
            )
        archive_names[archive_name_key] = archive_path
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = sorted(
                (info for info in archive.infolist() if not info.is_dir()),
                key=lambda info: (info.filename.replace("\\", "/").casefold(), info.filename),
            )
            for info in infos:
                if not info.filename.lower().endswith(".pkl"):
                    continue
                normalized = _safe_archive_member_name(info.filename)
                source_path = f"{archive_path.name}!/{normalized}"
                source_key = canonical_source_key(source_path)
                if source_key in source_keys:
                    raise ValueError(f"duplicate archive member identity: {source_path}")
                source_keys.add(source_key)
                materialization_key = canonical_source_key(
                    f"{archive_path.stem}/{normalized}"
                )
                if materialization_key in materialization_keys:
                    raise ValueError(
                        f"duplicate materialization target identity: {archive_path.stem}/{normalized}"
                    )
                materialization_keys.add(materialization_key)
                inventory.append(
                    ArchiveMember(
                        archive_path=archive_path,
                        original_name=info.filename,
                        normalized_name=normalized,
                        source_path=source_path,
                        source_key=source_key,
                        materialization_key=materialization_key,
                        crc32=int(info.CRC),
                        file_size=int(info.file_size),
                    )
                )
    return tuple(inventory)


def summarize_archive_member_inventory(
    inventory: Sequence[ArchiveMember],
) -> dict[str, Any]:
    identities = [
        {
            "source_key": member.source_key,
            "materialization_key": member.materialization_key,
            "crc32": member.crc32,
            "file_size": member.file_size,
        }
        for member in inventory
    ]
    payload = json.dumps(
        identities, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "archives": len({member.archive_path.name.casefold() for member in inventory}),
        "pickle_members": len(inventory),
        "unique_source_keys": len({member.source_key for member in inventory}),
        "unique_materialization_keys": len(
            {member.materialization_key for member in inventory}
        ),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_failure_allowlist(path: Path | None) -> tuple[set[tuple[str, int, int, str]], dict[str, Any]]:
    if path is None:
        return set(), {"path": None, "sha256": None, "entries": 0}
    path = Path(path)
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
        raise ValueError("load failure allowlist must use schema_version 1 and an entries list")
    entries: set[tuple[str, int, int, str]] = set()
    for item in payload["entries"]:
        if not isinstance(item, Mapping):
            raise ValueError("load failure allowlist entries must be objects")
        try:
            identity = (
                canonical_source_key(item["source_key"]),
                int(item["crc32"]),
                int(item["file_size"]),
                str(item["error_type"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid load failure allowlist entry") from exc
        if identity in entries:
            raise ValueError(f"duplicate load failure allowlist entry: {identity[0]}")
        entries.add(identity)
    return entries, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "entries": len(entries),
    }


def _load_failure_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(row.get("reject_reason") or "")
    return {
        "source_key": canonical_source_key(row.get("source_key", "")),
        "crc32": int(row["archive_crc32"]),
        "file_size": int(row["archive_file_size"]),
        "error_type": reason.split(":", 1)[1],
    }


def _allowlist_identity(candidate: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        canonical_source_key(candidate["source_key"]),
        int(candidate["crc32"]),
        int(candidate["file_size"]),
        str(candidate["error_type"]),
    )


def parent_cad_id(value: str) -> str | None:
    """Extract a reliable ABC parent UUID, returning ``None`` if unresolved."""
    match = PARENT_CAD_RE.fullmatch(_path_name(value))
    return match.group("cad").casefold() if match else None


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except (TypeError, ValueError):
        return None


def inspect_cad_record(data: Mapping[str, Any], config: ProtocolConfig) -> dict[str, Any]:
    """Inspect all available topology and apply the stable rejection priority."""
    result: dict[str, Any] = {
        "num_faces": None,
        "global_edges": None,
        "max_edges_per_face": None,
        "protocol_eligible": False,
        "reject_reason": None,
    }
    missing = [
        f"missing_{field}"
        for field in ("surf_ncs", "edge_ncs", "faceEdge_adj")
        if field not in data or data.get(field) is None
    ]
    num_faces = _safe_len(data.get("surf_ncs"))
    global_edges = _safe_len(data.get("edge_ncs"))
    face_edge_adj = data.get("faceEdge_adj")
    adjacency_faces = _safe_len(face_edge_adj)
    result["num_faces"] = num_faces
    result["global_edges"] = global_edges
    reasons = list(missing)
    if "missing_surf_ncs" not in reasons and num_faces is None:
        reasons.append("invalid_surf_ncs")
    if "missing_edge_ncs" not in reasons and global_edges is None:
        reasons.append("invalid_edge_ncs")
    if "missing_faceEdge_adj" not in reasons and adjacency_faces is None:
        reasons.append("invalid_face_edge_adjacency")
    if num_faces is not None and num_faces < config.min_faces:
        reasons.append("too_few_faces")
    if num_faces is not None and num_faces > config.max_faces:
        reasons.append("too_many_faces")
    if global_edges is not None and global_edges > config.max_global_edges:
        reasons.append("too_many_global_edges")
    if adjacency_faces is not None and num_faces is not None and adjacency_faces != num_faces:
        reasons.append("face_edge_adjacency_length_mismatch")

    saw_non_integer = False
    saw_out_of_range = False
    max_edges = 0
    if adjacency_faces is not None:
        for indices in face_edge_adj:
            count = _safe_len(indices)
            if count is None:
                reasons.append("invalid_face_edge_adjacency")
                continue
            max_edges = max(max_edges, count)
            try:
                iterator = iter(indices)
            except TypeError:
                reasons.append("invalid_face_edge_adjacency")
                continue
            for index in iterator:
                if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                    saw_non_integer = True
                elif global_edges is not None and (int(index) < 0 or int(index) >= global_edges):
                    saw_out_of_range = True
        result["max_edges_per_face"] = max_edges
        if max_edges > config.max_edges_per_face:
            reasons.append("too_many_edges_per_face")
    if saw_non_integer:
        reasons.append("non_integer_edge_index")
    if saw_out_of_range:
        reasons.append("edge_index_out_of_range")

    reason_set = set(reasons)
    result["reject_reason"] = next(
        (reason for reason in REJECTION_PRIORITY if reason in reason_set),
        None,
    )
    result["protocol_eligible"] = result["reject_reason"] is None
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
    if isinstance(data, Mapping):
        base.update(inspect_cad_record(data, config))
    else:
        base["reject_reason"] = "invalid_record"
    if parent_id is None and base["reject_reason"] is None:
        base["protocol_eligible"] = False
        base["reject_reason"] = "unknown_parent_id"
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
    ordered = sorted(
        eligible_parents,
        key=lambda parent: (eligible_parents[parent], _parent_tie_break(parent, seed), parent),
    )
    selected: set[str] = set()
    selected_records = 0
    for parent in ordered:
        group_size = eligible_parents[parent]
        if selected_records + group_size > max_eligible_records:
            continue
        selected.add(parent)
        selected_records += group_size
        if selected_records == max_eligible_records:
            break
    if not selected:
        smallest_size = min(eligible_parents.values(), default=0)
        candidates = [
            parent for parent in ordered if eligible_parents[parent] == smallest_size
        ]
        if candidates:
            selected.add(candidates[0])
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
    quarantine = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n"
        for row in rows
        if str(row.get("reject_reason") or "").startswith("load_failed:")
    )
    allowlist_candidates = "".join(
        json.dumps(_load_failure_candidate(row), sort_keys=True, ensure_ascii=True) + "\n"
        for row in rows
        if str(row.get("reject_reason") or "").startswith("load_failed:")
    )
    split_payload = pickle.dumps(dict(split), protocol=pickle.HIGHEST_PROTOCOL)
    summary = dict(summary)
    summary["split_pickle_sha256"] = hashlib.sha256(split_payload).hexdigest()
    _atomic_write_bytes(output_dir / "protocol_manifest.jsonl", manifest.encode("utf-8"))
    _atomic_write_bytes(
        output_dir / "quarantined_pickle_members.jsonl", quarantine.encode("utf-8")
    )
    _atomic_write_bytes(
        output_dir / "load_failure_allowlist_candidates.jsonl",
        allowlist_candidates.encode("utf-8"),
    )
    _atomic_write_bytes(
        output_dir / "protocol_summary.json",
        (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"),
    )
    _atomic_write_bytes(output_dir / "split.pkl", split_payload)

    integrity = _summarize_split_integrity(split)
    _atomic_write_bytes(
        output_dir / "split_integrity.json",
        (json.dumps(integrity, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"),
    )


def _summarize_split_integrity(
    split: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
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
    return {
        "status": "NO_PARENT_CAD_OVERLAP_DETECTED"
        if not any(item["parent_cad_overlap"] for item in pairwise.values())
        else "LEAKAGE_DETECTED",
        "pairwise": pairwise,
        "splits": {
            name: {"records": len(split.get(name, [])), "unique_parent_cads": len(parent_sets[name])}
            for name in SPLITS
        },
    }


def _iter_archive_records(
    inventory: Sequence[ArchiveMember], max_scan_records: int
) -> Iterable[tuple[ArchiveMember, Mapping[str, Any] | None, str | None]]:
    seen = 0
    for archive_path, members in itertools.groupby(inventory, key=lambda item: item.archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in members:
                if max_scan_records > 0 and seen >= max_scan_records:
                    return
                seen += 1
                try:
                    with archive.open(member.original_name, "r") as handle:
                        data = pickle.load(handle)
                    yield member, data, None
                except Exception as exc:
                    yield member, None, type(exc).__name__


def _materialize_selected(
    selected_rows: Sequence[dict[str, Any]],
    archive_locations: Mapping[str, ArchiveMember],
    materialize_root: Path,
) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {name: [] for name in SPLITS}
    open_archives: dict[Path, zipfile.ZipFile] = {}
    materialized_targets: set[Path] = set()
    try:
        for row in sorted(selected_rows, key=lambda item: str(item["source_key"])):
            source_key = str(row["source_key"])
            member = archive_locations[source_key]
            split_name = str(row["split"])
            target = (
                materialize_root
                / split_name
                / member.archive_path.stem
                / Path(member.normalized_name)
            )
            resolved_target = target.resolve()
            if resolved_target in materialized_targets:
                raise RuntimeError(f"duplicate materialization target: {resolved_target}")
            materialized_targets.add(resolved_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            archive = open_archives.setdefault(
                member.archive_path, zipfile.ZipFile(member.archive_path, "r")
            )
            _atomic_write_bytes(target, archive.read(member.original_name))
            split[split_name].append(str(resolved_target))
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
    max_load_failures: int = 100,
    max_load_failure_fraction: float = 0.001,
    load_failure_allowlist_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Scan ZIP members, filter, split by parent, materialize, and write evidence."""
    max_load_failures = int(max_load_failures)
    max_load_failure_fraction = float(max_load_failure_fraction)
    if max_load_failures < 0:
        raise ValueError("max_load_failures must be non-negative")
    if not 0.0 <= max_load_failure_fraction <= 1.0:
        raise ValueError("max_load_failure_fraction must be between zero and one")
    inventory = validate_archive_member_inventory(archive_paths)
    inventory_summary = summarize_archive_member_inventory(inventory)
    allowlist, allowlist_metadata = _load_failure_allowlist(load_failure_allowlist_path)
    rows: list[dict[str, Any]] = []
    archive_locations: dict[str, ArchiveMember] = {}
    for member, data, load_error in _iter_archive_records(
        inventory, max(0, int(max_scan_records))
    ):
        row = build_manifest_row(member.source_path, data, config, load_error=load_error)
        row["archive_crc32"] = member.crc32
        row["archive_file_size"] = member.file_size
        rows.append(row)
        archive_locations[row["source_key"]] = member

    selected_parents = _select_parent_groups(rows, max(0, int(max_eligible_records)), config.seed)
    assignment = assign_parent_splits(
        [row for row in rows if row.get("parent_id") in selected_parents], config
    )
    for row in rows:
        parent = row.get("parent_id")
        if row.get("protocol_eligible") and parent in assignment:
            row["split"] = assignment[str(parent)]

    planned_selected_rows = [row for row in rows if row.get("split") in SPLITS]
    reject_reasons = Counter(
        str(row["reject_reason"]) for row in rows if row.get("reject_reason")
    )
    eligible_count = sum(bool(row.get("protocol_eligible")) for row in rows)
    planned_selected_count = len(planned_selected_rows)
    selected_parent_ids = sorted({str(row["parent_id"]) for row in planned_selected_rows})
    cap_overshoot = (
        max(0, planned_selected_count - max_eligible_records)
        if max_eligible_records > 0
        else 0
    )
    load_failures = sum(
        str(row.get("reject_reason") or "").startswith("load_failed:") for row in rows
    )
    load_failure_fraction = load_failures / len(rows) if rows else 0.0
    load_failure_count_within_limit = load_failures <= max_load_failures
    load_failure_fraction_within_limit = (
        load_failure_fraction <= max_load_failure_fraction
    )
    load_failure_candidates = [
        _load_failure_candidate(row)
        for row in rows
        if str(row.get("reject_reason") or "").startswith("load_failed:")
    ]
    approved_load_failures = sum(
        _allowlist_identity(candidate) in allowlist for candidate in load_failure_candidates
    )
    unapproved_load_failures = load_failures - approved_load_failures
    failure_reasons = []
    if eligible_count == 0:
        failure_reasons.append("no_eligible_records")
    elif planned_selected_count == 0:
        failure_reasons.append("no_selected_records")
    if not load_failure_count_within_limit:
        failure_reasons.append("archive_member_load_failure_count_exceeded")
    if not load_failure_fraction_within_limit:
        failure_reasons.append("archive_member_load_failure_fraction_exceeded")
    if unapproved_load_failures:
        failure_reasons.append("unapproved_archive_member_load_failure")
    if failure_reasons:
        split = {name: [] for name in SPLITS}
    else:
        split = _materialize_selected(
            planned_selected_rows, archive_locations, Path(materialize_root)
        )
    split_integrity = _summarize_split_integrity(split)
    if split_integrity["status"] == "LEAKAGE_DETECTED":
        failure_reasons.append("parent_overlap")
    if failure_reasons:
        split = {name: [] for name in SPLITS}
        for row in rows:
            row["split"] = None
    selected_rows = planned_selected_rows if not failure_reasons else []
    selected_count = len(selected_rows)
    split_counts = {name: len(split[name]) for name in SPLITS}
    split_parents = {
        name: len({row["parent_id"] for row in selected_rows if row["split"] == name})
        for name in SPLITS
    }
    summary: dict[str, Any] = {
        "status": "FAILED" if failure_reasons else "VERIFIED",
        "failure_reasons": failure_reasons,
        "experiment_scale": "smoke"
        if max_scan_records > 0 or max_eligible_records > 0
        else "full",
        "protocol_version": config.version,
        "config": asdict(config),
        "archives_scanned": len({str(path) for path in archive_paths}),
        "archive_inventory": inventory_summary,
        "records_scanned": len(rows),
        "records_eligible": eligible_count,
        "records_selected": selected_count,
        "records_rejected": len(rows) - eligible_count,
        "archive_member_load_failures": load_failures,
        "quarantined_pickle_members": load_failures,
        "quarantined_pickle_members_file": "quarantined_pickle_members.jsonl",
        "load_failure_policy": {
            "status": "WITHIN_LIMITS"
            if load_failure_count_within_limit and load_failure_fraction_within_limit
            else "EXCEEDED",
            "max_count": max_load_failures,
            "max_fraction": max_load_failure_fraction,
            "observed_count": load_failures,
            "observed_fraction": load_failure_fraction,
            "count_within_limit": load_failure_count_within_limit,
            "fraction_within_limit": load_failure_fraction_within_limit,
        },
        "load_failure_allowlist": {
            **allowlist_metadata,
            "status": (
                "NO_FAILURES"
                if load_failures == 0
                else "ALL_FAILURES_APPROVED"
                if unapproved_load_failures == 0
                else "UNAPPROVED_FAILURES"
            ),
            "approved_failures": approved_load_failures,
            "unapproved_failures": unapproved_load_failures,
            "candidate_file": "load_failure_allowlist_candidates.jsonl",
        },
        "max_eligible_records": max(0, int(max_eligible_records)),
        "eligible_cap_overshoot_records": cap_overshoot,
        "eligible_cap_overshoot_parent_id": selected_parent_ids[0]
        if cap_overshoot and len(selected_parent_ids) == 1
        else None,
        "parent_overlap_counts": {
            pair: int(item["parent_cad_overlap"])
            for pair, item in split_integrity["pairwise"].items()
        },
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "split_records": split_counts,
        "split_parents": split_parents,
        "protocol_sha256": _protocol_hash(rows, config),
    }
    summary["split_pickle_sha256"] = hashlib.sha256(
        pickle.dumps(dict(split), protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    quarantine_payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n"
        for row in rows
        if str(row.get("reject_reason") or "").startswith("load_failed:")
    ).encode("utf-8")
    summary["quarantined_pickle_members_sha256"] = hashlib.sha256(
        quarantine_payload
    ).hexdigest()
    candidate_payload = "".join(
        json.dumps(candidate, sort_keys=True, ensure_ascii=True) + "\n"
        for candidate in load_failure_candidates
    ).encode("utf-8")
    summary["load_failure_allowlist_candidates_sha256"] = hashlib.sha256(
        candidate_payload
    ).hexdigest()
    _write_outputs(Path(output_dir), rows, split, summary)
    return rows, split, summary
