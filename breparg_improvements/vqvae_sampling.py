import math
import pickle
import random
from pathlib import Path

import numpy as np

from sharded_data import PATCH_SHARD_FORMAT, iter_shard_records


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
    points = np.asarray(surface, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return 0.0
    span = np.ptp(points, axis=0)
    largest = float(np.max(span))
    if largest <= 1e-8:
        return 0.0
    return float(np.min(span) / largest)


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


def patch_records_from_parsed(data, source_path, complex_min_faces=12, complex_min_edges=20):
    surfaces = _as_array(data.get("surf_ncs"), 4)
    edges = _as_array(data.get("edge_ncs"), 3)
    n_faces = int(len(surfaces))
    n_edges = int(len(edges))
    is_complex = n_faces >= int(complex_min_faces) or n_edges >= int(complex_min_edges)
    source = str(source_path)
    records = []

    for index, surface in enumerate(surfaces):
        records.append(
            {
                "record_id": f"{source}:surface:{index}",
                "source_path": source,
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
                "kind": "edge",
                "array": edge_to_surface_patch(edge),
                "curvature_score": edge_curvature_proxy(edge),
                "n_faces": n_faces,
                "n_edges": n_edges,
                "is_complex_source": is_complex,
            }
        )

    return records


def load_patch_records(path, complex_min_faces=12, complex_min_edges=20):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return patch_records_from_parsed(data, path, complex_min_faces, complex_min_edges)


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

    for path in paths:
        try:
            records = load_patch_records(path, complex_min_faces, complex_min_edges)
        except Exception:
            failed_paths += 1
            continue
        loaded_paths += 1
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
        all_records.extend(records)
        complex_records.extend([record for record in records if record["is_complex_source"]])
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
    uniform_target = max(0, all_target - len(complex_selected))
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
        "loaded_paths": loaded_paths,
        "failed_paths": failed_paths,
        "source_records_available": len(all_records),
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
