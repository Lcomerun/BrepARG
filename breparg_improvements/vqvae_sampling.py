import hashlib
import math
import pickle
import random
from pathlib import Path

import numpy as np

from cad_protocol import canonical_source_key, parent_cad_id
from sharded_data import PATCH_SHARD_FORMAT, iter_shard_records
from vqvae_metrics import surface_plane_residual


def _clamp_fraction(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _as_array(value, ndim):
    arr = np.asarray(value if value is not None else [], dtype=np.float32)
    if arr.ndim != ndim:
        return np.zeros((0,) + ((32, 32, 3) if ndim == 4 else (32, 3)), dtype=np.float32)
    return arr


def surface_curvature_proxy(surface):
    return surface_plane_residual(surface)


def edge_curvature_proxy(edge):
    points = np.asarray(edge, dtype=np.float32).reshape(-1, 3)
    if len(points) < 2:
        return 0.0
    start = points[0]
    chord = points[-1] - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 1e-8:
        return 0.0
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float32)[:, None]
    line = start[None, :] + t * chord[None, :]
    deviation = np.linalg.norm(points - line, axis=1)
    return float(np.max(deviation) / chord_len)


def edge_to_surface_patch(edge):
    edge = np.asarray(edge, dtype=np.float32)
    return np.tile(edge[:, None, :], (1, 32, 1))


def _canonical_patch_bytes(kind, array, decimals=None):
    kind_bytes = str(kind).encode("utf-8")
    values = np.ascontiguousarray(array, dtype="<f4")
    if decimals is not None:
        values = np.round(values, decimals=decimals)
    values = np.ascontiguousarray(values, dtype="<f4")
    shape = np.asarray(values.shape, dtype="<u8").tobytes()
    return (
        len(kind_bytes).to_bytes(8, "little")
        + kind_bytes
        + values.ndim.to_bytes(8, "little")
        + shape
        + values.tobytes(order="C")
    )


def canonical_patch_hash(kind, array):
    return hashlib.sha256(_canonical_patch_bytes(kind, array)).hexdigest()


def rounded_patch_hash(kind, array):
    return hashlib.sha256(_canonical_patch_bytes(kind, array, decimals=4)).hexdigest()


def _stable_value_key(value):
    if value is None:
        return ("none",)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("int", str(int(value)))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ("float", "nan")
        if math.isinf(number):
            return ("float", "inf" if number > 0 else "-inf")
        return ("float", number.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            "ndarray",
            str(array.dtype),
            tuple(array.shape),
            hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (_stable_value_key(key), _stable_value_key(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_stable_value_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted(_stable_value_key(item) for item in value)))
    return (type(value).__qualname__, repr(value))


def _record_sort_key(record):
    return (
        str(record.get("record_id", "")),
        str(record.get("source_path", "")),
        str(record.get("parent_id", "")),
        str(record.get("kind", "")),
        _stable_value_key(record),
    )


def _record_values(record, plural_key, singular_key, keep_missing=False):
    plural_present = plural_key in record
    values = record.get(plural_key)
    if values is None:
        values = [None] if plural_present else []
    elif isinstance(values, (str, bytes, Path)):
        values = [values]
    else:
        try:
            values = list(values)
        except TypeError:
            values = [values]
    if singular_key in record:
        values.append(record.get(singular_key))
    if keep_missing:
        return list(values)
    return [value for value in values if value is not None and str(value)]


def _source_path_values_with_missing_identity(record):
    paths = _record_values(
        record,
        "provenance_source_paths",
        "source_path",
        keep_missing=True,
    )
    keys = _record_values(
        record,
        "provenance_source_keys",
        "source_key",
        keep_missing=True,
    )
    return paths if paths or keys else [None]


def _parent_values_with_missing_identity(record):
    values = _record_values(
        record,
        "provenance_parent_ids",
        "parent_id",
        keep_missing=True,
    )
    return values or [None]


def _unique_sorted_values(values):
    unique = {}
    for value in values:
        unique.setdefault(_stable_value_key(value), value)
    return [unique[key] for key in sorted(unique)]


def deduplicate_patch_records(records):
    records = list(records)
    groups = {}
    rounded_groups = {}
    for original in records:
        record = dict(original)
        kind = record.get("kind")
        array = record.get("array")
        exact_hash = canonical_patch_hash(kind, array)
        audit_hash = rounded_patch_hash(kind, array)
        record["exact_hash"] = exact_hash
        record["rounded_hash"] = audit_hash
        groups.setdefault(exact_hash, []).append(record)
        rounded_groups.setdefault(audit_hash, set()).add(exact_hash)

    deduplicated = []
    for exact_hash in sorted(groups):
        group = sorted(groups[exact_hash], key=_record_sort_key)
        representative = dict(group[0])
        representative["provenance_record_ids"] = sorted(
            {
                str(value)
                for record in group
                for value in _record_values(record, "provenance_record_ids", "record_id")
            }
        )
        representative["provenance_source_paths"] = sorted(
            {
                None if value is None else str(value)
                for record in group
                for value in _source_path_values_with_missing_identity(record)
            },
            key=lambda value: "" if value is None else value,
        )
        representative["provenance_source_keys"] = sorted(
            {
                None if value is None else canonical_source_key(value)
                for record in group
                for value in _record_values(
                    record,
                    "provenance_source_keys",
                    "source_key",
                    keep_missing=True,
                )
            },
            key=lambda value: "" if value is None else value,
        )
        representative["provenance_parent_ids"] = _unique_sorted_values(
            value
            for record in group
            for value in _parent_values_with_missing_identity(record)
        )
        representative["duplicate_count"] = sum(
            int(record.get("duplicate_count", 0)) + 1 for record in group
        ) - 1
        deduplicated.append(representative)

    deduplicated.sort(key=_record_sort_key)
    input_records = sum(int(record.get("duplicate_count", 0)) + 1 for record in records)
    summary = {
        "input_records": input_records,
        "unique_records": len(deduplicated),
        "duplicates_removed": input_records - len(deduplicated),
        "exact_duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "rounded_only_duplicate_groups": sum(len(hashes) > 1 for hashes in rounded_groups.values()),
    }
    return deduplicated, summary


def _inventory_exact_hashes(records):
    return {
        canonical_patch_hash(record.get("kind"), record.get("array"))
        for record in records
    }


def _inventory_sources(records, split_name):
    sources = set()
    for record in records:
        values = _record_values(
            record,
            "provenance_source_paths",
            "source_path",
            keep_missing=True,
        )
        values.extend(
            _record_values(
                record,
                "provenance_source_keys",
                "source_key",
                keep_missing=True,
            )
        )
        if not values or any(value is None or not str(value).strip() for value in values):
            raise ValueError(f"invalid source identity in {split_name} inventory")
        sources.update(canonical_source_key(value) for value in values)
    return sources


def _canonical_parent_identity(value):
    if not isinstance(value, str) or value != value.casefold():
        return None
    if parent_cad_id(f"{value}_step_000.pkl") != value:
        return None
    return value


def _inventory_parents(records, split_name):
    parents = set()
    unknown = False
    for record in records:
        values = _record_values(
            record,
            "provenance_parent_ids",
            "parent_id",
            keep_missing=True,
        )
        if not values:
            unknown = True
        for value in values:
            if value is None or not str(value).strip():
                unknown = True
            else:
                parent = _canonical_parent_identity(value)
                if parent is None:
                    raise ValueError(f"invalid parent_id in {split_name} inventory: {value!r}")
                parents.add(parent)
    return parents, unknown


def validate_inventory_identities(records, split_name):
    records = list(records)
    sources = _inventory_sources(records, split_name)
    parents, unknown = _inventory_parents(records, split_name)
    if unknown:
        raise ValueError(f"unknown parent_id in {split_name} inventory")
    return sources, parents


def remove_train_exact_hash_overlap(train_records, val_records):
    """Keep validation fixed and remove exact-content overlaps from training."""
    train_records = list(train_records)
    val_hashes = _inventory_exact_hashes(val_records)
    kept = [
        record
        for record in train_records
        if canonical_patch_hash(record.get("kind"), record.get("array")) not in val_hashes
    ]
    removed_hashes = {
        canonical_patch_hash(record.get("kind"), record.get("array"))
        for record in train_records
        if canonical_patch_hash(record.get("kind"), record.get("array")) in val_hashes
    }
    removed = len(train_records) - len(kept)
    removed_by_kind = {}
    removed_parents = set()
    for record in train_records:
        if canonical_patch_hash(record.get("kind"), record.get("array")) in val_hashes:
            kind = str(record.get("kind", "unknown"))
            removed_by_kind[kind] = removed_by_kind.get(kind, 0) + 1
            removed_parents.update(_record_parent_ids(record))
    return kept, {
        "train_records_before": len(train_records),
        "train_records_after": len(kept),
        "train_records_removed": removed,
        "overlap_hashes_removed": len(removed_hashes),
        "removed_fraction": removed / len(train_records) if train_records else 0.0,
        "removed_by_kind": dict(sorted(removed_by_kind.items())),
        "train_parents_affected": sorted(removed_parents),
        "train_parent_count_affected": len(removed_parents),
    }


def audit_train_val_inventories(train_records, val_records):
    train_records = list(train_records)
    val_records = list(val_records)
    train_sources, train_parents = validate_inventory_identities(train_records, "train")
    val_sources, val_parents = validate_inventory_identities(val_records, "val")

    train_hashes = _inventory_exact_hashes(train_records)
    val_hashes = _inventory_exact_hashes(val_records)
    source_overlap = sorted(train_sources & val_sources)
    parent_overlap = sorted(train_parents & val_parents)
    hash_overlap = sorted(train_hashes & val_hashes)
    failures = []
    if source_overlap:
        failures.append(f"source_key overlap: {source_overlap}")
    if parent_overlap:
        failures.append(f"parent_id overlap: {parent_overlap}")
    if hash_overlap:
        failures.append(f"exact_hash overlap: {hash_overlap}")
    if failures:
        raise ValueError("train/val inventory audit failed: " + "; ".join(failures))

    return {
        "status": "VERIFIED",
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_source_keys": len(train_sources),
        "val_source_keys": len(val_sources),
        "train_parent_ids": len(train_parents),
        "val_parent_ids": len(val_parents),
        "train_exact_hashes": len(train_hashes),
        "val_exact_hashes": len(val_hashes),
        "source_key_overlap": source_overlap,
        "parent_id_overlap": parent_overlap,
        "exact_hash_overlap": hash_overlap,
    }


def patch_records_from_parsed(
    data,
    source_path,
    complex_min_faces=12,
    complex_min_edges=20,
    require_parent_id=False,
):
    surfaces = _as_array(data.get("surf_ncs"), 4)
    edges = _as_array(data.get("edge_ncs"), 3)
    n_faces = int(len(surfaces))
    n_edges = int(len(edges))
    is_complex = n_faces >= int(complex_min_faces) or n_edges >= int(complex_min_edges)
    source = str(source_path)
    parent = parent_cad_id(source)
    if require_parent_id and parent is None:
        raise ValueError(f"unknown parent CAD ID for source path: {source}")
    source_key = canonical_source_key(source)
    records = []

    for index, surface in enumerate(surfaces):
        records.append(
            {
                "record_id": f"{source}:surface:{index}",
                "source_path": source,
                "source_key": source_key,
                "parent_id": parent,
                "kind": "surface",
                "array": np.asarray(surface, dtype=np.float32),
                "curvature_score": surface_curvature_proxy(surface),
                "n_faces": n_faces,
                "n_edges": n_edges,
                "is_complex_source": is_complex,
            }
        )

    for index, edge in enumerate(edges):
        records.append(
            {
                "record_id": f"{source}:edge:{index}",
                "source_path": source,
                "source_key": source_key,
                "parent_id": parent,
                "kind": "edge",
                "array": edge_to_surface_patch(edge),
                "curvature_score": edge_curvature_proxy(edge),
                "n_faces": n_faces,
                "n_edges": n_edges,
                "is_complex_source": is_complex,
            }
        )

    return records


def load_patch_records(
    path,
    complex_min_faces=12,
    complex_min_edges=20,
    require_parent_id=False,
):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return patch_records_from_parsed(
        data,
        path,
        complex_min_faces,
        complex_min_edges,
        require_parent_id=require_parent_id,
    )


def select_patch_records(records, target, curved_fraction=0.0, seed=0, exclude_ids=None):
    target = max(0, int(target))
    if target == 0:
        return []
    excluded_ids = set(exclude_ids or [])
    candidates = [record for record in records if record.get("record_id") not in excluded_ids]
    if not candidates:
        return []

    curved_fraction = _clamp_fraction(curved_fraction)
    curved_target = int(math.ceil(target * curved_fraction)) if curved_fraction > 0 else 0
    curved_target = min(curved_target, target, len(candidates))

    selected = []
    if curved_target:
        by_curvature = sorted(
            candidates,
            key=lambda record: (
                -float(record.get("curvature_score", 0.0)),
                str(record.get("record_id", "")),
            ),
        )
        selected.extend(by_curvature[:curved_target])

    selected_ids = {record.get("record_id") for record in selected}
    remaining = [record for record in candidates if record.get("record_id") not in selected_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target - len(selected))])
    return selected


def _record_parent_ids(record):
    return {
        value
        for value in _record_values(
            record,
            "provenance_parent_ids",
            "parent_id",
            keep_missing=True,
        )
        if _canonical_parent_identity(value) is not None
    }


def _balanced_round_robin_order(candidates, seed):
    grouped = {}
    for record in candidates:
        parent = _canonical_parent_identity(record.get("parent_id"))
        if parent is None:
            raise ValueError(
                f"balanced sampling requires a canonical parent_id: {record.get('parent_id')!r}"
            )
        source = canonical_source_key(record.get("source_key") or record.get("source_path", ""))
        grouped.setdefault(parent, {}).setdefault(source, []).append(record)

    rng = random.Random(seed)
    parent_order = sorted(grouped)
    rng.shuffle(parent_order)
    source_orders = {}
    for parent in parent_order:
        source_orders[parent] = sorted(grouped[parent])
        rng.shuffle(source_orders[parent])
        for source in source_orders[parent]:
            rng.shuffle(grouped[parent][source])

    ordered = []
    parent_queues = {}
    for parent in parent_order:
        queue = []
        sources = source_orders[parent]
        depth = 0
        while len(queue) < sum(len(grouped[parent][source]) for source in sources):
            for source in sources:
                group = grouped[parent][source]
                if depth < len(group):
                    queue.append(group[depth])
            depth += 1
        parent_queues[parent] = queue
    depth = 0
    while len(ordered) < len(candidates):
        added = False
        for parent in parent_order:
            queue = parent_queues[parent]
            if depth < len(queue):
                ordered.append(queue[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def balanced_round_robin_records(
    records,
    target,
    curved_fraction=0.0,
    seed=0,
    exclude_ids=None,
):
    """Select deterministically while giving every parent/source one turn per pass."""
    target = max(0, int(target))
    excluded_ids = set(exclude_ids or [])
    candidates = sorted(
        (
            record
            for record in records
            if record.get("record_id") not in excluded_ids
        ),
        key=_record_sort_key,
    )
    if target == 0 or not candidates:
        return []

    curved_fraction = _clamp_fraction(curved_fraction)
    curved_target = min(
        target,
        len(candidates),
        int(math.ceil(target * curved_fraction)) if curved_fraction > 0 else 0,
    )
    full_order = _balanced_round_robin_order(candidates, seed)
    parent_order = list(
        dict.fromkeys(
            _canonical_parent_identity(record.get("parent_id")) for record in full_order
        )
    )
    by_parent = {parent: [] for parent in parent_order}
    for record in candidates:
        by_parent[_canonical_parent_identity(record.get("parent_id"))].append(record)

    natural_representative = {}
    for record in full_order:
        parent = _canonical_parent_identity(record.get("parent_id"))
        natural_representative.setdefault(parent, record)

    selected = []
    selected_ids = set()
    for parent in parent_order[:target]:
        record = natural_representative[parent]
        selected.append(record)
        selected_ids.add(record.get("record_id"))

    curved_selected = sum(
        float(record.get("curvature_score", 0.0)) > 0.0 for record in selected
    )
    curved_order = sorted(
        (
            record
            for record in candidates
            if record.get("record_id") not in selected_ids
            and float(record.get("curvature_score", 0.0)) > 0.0
        ),
        key=lambda record: (
            -float(record.get("curvature_score", 0.0)),
            _record_sort_key(record),
        ),
    )
    for index, record in enumerate(list(selected)):
        if curved_selected >= curved_target:
            break
        if float(record.get("curvature_score", 0.0)) > 0.0:
            continue
        parent = _canonical_parent_identity(record.get("parent_id"))
        replacements = [
            candidate
            for candidate in by_parent[parent]
            if candidate.get("record_id") not in selected_ids
            and float(candidate.get("curvature_score", 0.0)) > 0.0
        ]
        if not replacements:
            continue
        replacement = min(
            replacements,
            key=lambda item: (
                -float(item.get("curvature_score", 0.0)),
                _record_sort_key(item),
            ),
        )
        selected_ids.remove(record.get("record_id"))
        selected_ids.add(replacement.get("record_id"))
        selected[index] = replacement
        curved_selected += 1
    for record in curved_order:
        if len(selected) == target or curved_selected >= curved_target:
            break
        if record.get("record_id") in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record.get("record_id"))
        curved_selected += 1

    remaining = [
        record for record in candidates if record.get("record_id") not in selected_ids
    ]
    remaining_order = _balanced_round_robin_order(remaining, seed)
    for record in remaining_order:
        if len(selected) == target:
            break
        if record.get("record_id") not in selected_ids:
            selected.append(record)
            selected_ids.add(record.get("record_id"))
    return selected


def collect_vqvae_sample_records(
    paths,
    cap,
    seed=0,
    complex_fraction=0.0,
    complex_min_faces=12,
    complex_min_edges=20,
    curved_fraction=0.0,
    max_source_faces=0,
    max_source_edges=0,
    oversample_factor=1.2,
    require_parent_id=False,
    require_all_paths=False,
    deduplicate_before_cap=False,
    balance_by_parent=False,
    min_parent_coverage=0.0,
):
    cap = max(0, int(cap))
    paths = [Path(path) for path in paths]
    rng = random.Random(seed)
    rng.shuffle(paths)
    complex_fraction = _clamp_fraction(complex_fraction)
    curved_fraction = _clamp_fraction(curved_fraction)
    complex_target = int(round(cap * complex_fraction))
    complex_target = min(cap, max(0, complex_target))
    all_target = cap
    scan_target = max(cap, int(math.ceil(cap * float(oversample_factor))))
    complex_scan_target = max(complex_target, int(math.ceil(complex_target * float(oversample_factor))))
    max_source_faces = int(max_source_faces or 0)
    max_source_edges = int(max_source_edges or 0)

    all_records = []
    complex_records = []
    loaded_paths = 0
    failed_paths = 0
    dropped_records_source_cap = 0
    retain_full_inventory = bool(deduplicate_before_cap or balance_by_parent)
    requested_parents = {
        parent
        for path in paths
        for parent in [parent_cad_id(path)]
        if parent is not None
    }

    for path in paths:
        try:
            records = load_patch_records(
                path,
                complex_min_faces,
                complex_min_edges,
                require_parent_id=require_parent_id,
            )
        except Exception as exc:
            failed_paths += 1
            if require_all_paths:
                raise RuntimeError(f"failed to load required VQ source {path}: {exc}") from exc
            continue
        loaded_paths += 1
        if require_all_paths and not records:
            raise RuntimeError(f"required VQ source produced zero geometry patches: {path}")
        if max_source_faces > 0 or max_source_edges > 0:
            kept_records = []
            for record in records:
                too_many_faces = max_source_faces > 0 and int(record["n_faces"]) > max_source_faces
                too_many_edges = max_source_edges > 0 and int(record["n_edges"]) > max_source_edges
                if too_many_faces or too_many_edges:
                    dropped_records_source_cap += 1
                    continue
                kept_records.append(record)
            records = kept_records
        if require_all_paths and not records:
            raise RuntimeError(
                f"required VQ source produced zero usable geometry patches after source caps: {path}"
            )
        enough_all = len(all_records) >= scan_target
        enough_complex = complex_target == 0 or len(complex_records) >= complex_scan_target
        if require_all_paths and enough_all and enough_complex and not retain_full_inventory:
            continue
        all_records.extend(records)
        complex_records.extend([record for record in records if record["is_complex_source"]])
        enough_all = len(all_records) >= scan_target
        enough_complex = complex_target == 0 or len(complex_records) >= complex_scan_target
        if enough_all and enough_complex and not require_all_paths and not retain_full_inventory:
            break

    dedup_summary = None
    if deduplicate_before_cap:
        all_records, dedup_summary = deduplicate_patch_records(all_records)
        complex_records = [record for record in all_records if record["is_complex_source"]]

    selector = balanced_round_robin_records if balance_by_parent else select_patch_records
    complex_selected = selector(
        complex_records,
        complex_target,
        seed=seed + 1,
        curved_fraction=curved_fraction,
    )
    selected_ids = {record["record_id"] for record in complex_selected}
    uniform_target = max(0, all_target - len(complex_selected))
    uniform_selected = selector(
        all_records,
        uniform_target,
        seed=seed + 2,
        exclude_ids=selected_ids,
        curved_fraction=curved_fraction,
    )
    selected = complex_selected + uniform_selected
    contributing_parents = {
        parent
        for record in selected
        for parent in [_canonical_parent_identity(record.get("parent_id"))]
        if parent is not None
    }
    parent_coverage = (
        len(contributing_parents & requested_parents) / len(requested_parents)
        if requested_parents
        else 0.0
    )
    min_parent_coverage = _clamp_fraction(min_parent_coverage)
    if min_parent_coverage and parent_coverage < min_parent_coverage:
        raise RuntimeError(
            "VQ parent coverage below configured gate: "
            f"{parent_coverage:.6f} < {min_parent_coverage:.6f} "
            f"({len(contributing_parents & requested_parents)}/{len(requested_parents)})"
        )

    summary = {
        "requested": cap,
        "selected": len(selected),
        "seed": int(seed),
        "complex_fraction": complex_fraction,
        "complex_target": complex_target,
        "complex_min_faces": int(complex_min_faces),
        "complex_min_edges": int(complex_min_edges),
        "complex_records_available": len(complex_records),
        "complex_records_selected": sum(1 for record in selected if record["is_complex_source"]),
        "curved_fraction": curved_fraction,
        "max_source_faces": max_source_faces,
        "max_source_edges": max_source_edges,
        "dropped_records_source_cap": dropped_records_source_cap,
        "loaded_paths": loaded_paths,
        "failed_paths": failed_paths,
        "source_records_available": len(all_records),
        "scan_complete": loaded_paths + failed_paths == len(paths),
        "unique_records_before_cap": len(all_records),
        "dedup_before_cap": dedup_summary,
        "balanced_by_parent": bool(balance_by_parent),
        "requested_parent_cads": len(requested_parents),
        "parent_cads_contributing": len(contributing_parents & requested_parents),
        "parent_coverage": parent_coverage,
        "min_parent_coverage": min_parent_coverage,
    }
    return selected, summary


def collect_vqvae_patch_shard_records(
    paths,
    cap,
    seed=0,
    complex_fraction=0.0,
    complex_min_faces=12,
    complex_min_edges=20,
    curved_fraction=0.0,
    max_source_faces=0,
    max_source_edges=0,
    oversample_factor=1.2,
):
    cap = max(0, int(cap))
    paths = [Path(path) for path in paths]
    rng = random.Random(seed)
    rng.shuffle(paths)
    complex_fraction = _clamp_fraction(complex_fraction)
    curved_fraction = _clamp_fraction(curved_fraction)
    complex_target = min(cap, max(0, int(round(cap * complex_fraction))))
    scan_target = max(cap, int(math.ceil(cap * float(oversample_factor))))
    complex_scan_target = max(complex_target, int(math.ceil(complex_target * float(oversample_factor))))
    max_source_faces = int(max_source_faces or 0)
    max_source_edges = int(max_source_edges or 0)

    all_records = []
    complex_records = []
    loaded_shards = 0
    failed_shards = 0
    dropped_records_source_cap = 0
    source_paths = set()

    for path in paths:
        try:
            iterator = iter_shard_records(path)
            header = next(iterator)
            if header.get("format") != PATCH_SHARD_FORMAT:
                raise ValueError(f"not a VQ patch shard: {path}")
            loaded_shards += 1
            for record in iterator:
                if record.get("record_type") != "vq_patch":
                    continue
                too_many_faces = max_source_faces > 0 and int(record["n_faces"]) > max_source_faces
                too_many_edges = max_source_edges > 0 and int(record["n_edges"]) > max_source_edges
                if too_many_faces or too_many_edges:
                    dropped_records_source_cap += 1
                    continue
                record = dict(record)
                record.pop("record_type", None)
                all_records.append(record)
                source_paths.add(str(record.get("source_path", "")))
                if record.get("is_complex_source"):
                    complex_records.append(record)
                enough_all = len(all_records) >= scan_target
                enough_complex = complex_target == 0 or len(complex_records) >= complex_scan_target
                if enough_all and enough_complex:
                    break
        except Exception:
            failed_shards += 1
            continue
        enough_all = len(all_records) >= scan_target
        enough_complex = complex_target == 0 or len(complex_records) >= complex_scan_target
        if enough_all and enough_complex:
            break

    complex_selected = select_patch_records(
        complex_records,
        complex_target,
        curved_fraction=curved_fraction,
        seed=seed + 1,
    )
    selected_ids = {record["record_id"] for record in complex_selected}
    uniform_target = max(0, cap - len(complex_selected))
    uniform_selected = select_patch_records(
        all_records,
        uniform_target,
        curved_fraction=curved_fraction,
        seed=seed + 2,
        exclude_ids=selected_ids,
    )
    selected = complex_selected + uniform_selected

    summary = {
        "requested": cap,
        "selected": len(selected),
        "seed": int(seed),
        "complex_fraction": complex_fraction,
        "complex_target": complex_target,
        "complex_min_faces": int(complex_min_faces),
        "complex_min_edges": int(complex_min_edges),
        "complex_records_available": len(complex_records),
        "complex_records_selected": sum(1 for record in selected if record["is_complex_source"]),
        "curved_fraction": curved_fraction,
        "max_source_faces": max_source_faces,
        "max_source_edges": max_source_edges,
        "dropped_records_source_cap": dropped_records_source_cap,
        "loaded_shards": loaded_shards,
        "failed_shards": failed_shards,
        "source_records_available": len(all_records),
        "unique_sources_available": len(source_paths),
    }
    return selected, summary


def records_to_chw_array(records):
    records = list(records)
    if not records:
        return np.zeros((0, 3, 32, 32), dtype=np.float32)
    return np.stack([np.asarray(record["array"], dtype=np.float32) for record in records]).transpose(0, 3, 1, 2)


def _positive_weight(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number) or number <= 0:
        return 1.0
    return number


def records_to_patch_weights(
    records,
    complex_weight=1.0,
    curved_weight=1.0,
    curved_threshold=0.02,
):
    records = list(records)
    complex_weight = _positive_weight(complex_weight)
    curved_weight = _positive_weight(curved_weight)
    try:
        curved_threshold = float(curved_threshold)
    except (TypeError, ValueError):
        curved_threshold = 0.02
    if not math.isfinite(curved_threshold):
        curved_threshold = 0.02

    weights = []
    for record in records:
        weight = 1.0
        if bool(record.get("is_complex_source", False)):
            weight *= complex_weight
        try:
            curvature = float(record.get("curvature_score", 0.0))
        except (TypeError, ValueError):
            curvature = 0.0
        if math.isfinite(curvature) and curvature >= curved_threshold:
            weight *= curved_weight
        weights.append(weight)
    return np.asarray(weights, dtype=np.float32)


def collect_vqvae_samples(paths, cap, return_summary=False, **kwargs):
    records, summary = collect_vqvae_sample_records(paths, cap, **kwargs)
    samples = records_to_chw_array(records)
    if return_summary:
        return samples, summary
    return samples
