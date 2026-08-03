"""Run FSQ and AR diagnostics on a complex curved sequence subset.

The goal is to separate three questions that generation-time filters cannot
answer:

1. Can the FSQ VQ-VAE reconstruct complex curved surface/edge patches?
2. Can the current AR model predict true complex curved token sequences under
   teacher forcing?
3. Do true token sequences reconstruct through the current FSQ/OCC path?

The script reads local ABC parsed zip archives directly, so the large parsed
tree does not need to be extracted before running diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"

for item in (REPO_ROOT, TOOLS_DIR, IMPROVEMENTS_DIR, BREPARG_DIR):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

from evaluate_reconstruction_v13 import (  # noqa: E402
    grammar_validation,
    infer_device,
    load_ar_model,
    load_fsq_vqvae,
    normalize_vocab_info,
    reconstruct_one,
    summarize_manifest_rows,
)
from vqvae_sampling import patch_records_from_parsed, records_to_chw_array  # noqa: E402


DEFAULT_SEQUENCE = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "sequences_fsq_rcm.pkl"
DEFAULT_VQVAE = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "fsq_vqvae_best.pt"
DEFAULT_AR = REPO_ROOT / "ABC" / "processed" / "train_outputs" / "ubuntu" / "ar_best.pt"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "ABC" / "processed" / "abc_parsed_full_archives"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_runs" / "complex_curved_diagnostics_20260715" / "smoke"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def input_ids_of(group: dict[str, Any]) -> list[int]:
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in original.get("input_ids") or []]
    return [int(item) for item in group.get("input_ids") or []]


def source_relpath_from_group(group: dict[str, Any]) -> str | None:
    """Return an archive member path such as abc_0000/foo.pkl if available."""

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
    archive_path = archive_path_for_relpath(relpath, archive_root)
    if not archive_path.exists():
        raise FileNotFoundError(f"missing parsed archive: {archive_path}")
    member = str(relpath).replace("\\", "/").lstrip("/")
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open(member, "r") as handle:
            return pickle.load(handle)


def shape_patch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "patch_count": 0,
            "surface_count": 0,
            "edge_count": 0,
            "max_curvature": 0.0,
            "p95_curvature": 0.0,
            "p90_curvature": 0.0,
            "mean_curvature": 0.0,
        }
    curvatures = [float(record.get("curvature_score", 0.0) or 0.0) for record in records]
    return {
        "patch_count": int(len(records)),
        "surface_count": int(sum(1 for record in records if record.get("kind") == "surface")),
        "edge_count": int(sum(1 for record in records if record.get("kind") == "edge")),
        "max_curvature": float(max(curvatures)),
        "p95_curvature": float(percentile(curvatures, 0.95) or 0.0),
        "p90_curvature": float(percentile(curvatures, 0.90) or 0.0),
        "mean_curvature": float(statistics.mean(curvatures)),
    }


def sequence_length_bucket(length: int) -> str:
    length = int(length)
    if length <= 512:
        return "len_0000_0512"
    if length <= 1024:
        return "len_0513_1024"
    if length <= 1536:
        return "len_1025_1536"
    if length <= 2048:
        return "len_1537_2048"
    return "len_gt_2048"


def face_count_bucket(faces: int) -> str:
    faces = int(faces)
    if faces <= 11:
        return "faces_00_11"
    if faces <= 19:
        return "faces_12_19"
    if faces <= 29:
        return "faces_20_29"
    if faces <= 50:
        return "faces_30_50"
    return "faces_gt_50"


def edge_count_bucket(edges: int) -> str:
    edges = int(edges)
    if edges <= 19:
        return "edges_00_19"
    if edges <= 39:
        return "edges_20_39"
    if edges <= 79:
        return "edges_40_79"
    if edges <= 150:
        return "edges_80_150"
    return "edges_gt_150"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(q)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def numeric_stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(statistics.mean(finite)),
        "median": float(statistics.median(finite)),
        "p90": percentile(finite, 0.90),
        "p95": percentile(finite, 0.95),
        "max": float(max(finite)),
    }


def skipped_stage_report(stage: str, reason: str) -> dict[str, Any]:
    return {
        "skipped": True,
        "stage": str(stage),
        "reason": str(reason),
        "sample_count": 0,
        "token_count": 0,
        "token_weighted_ce": None,
        "teacher_ce": numeric_stats([]),
        "by_face_bucket": {},
        "by_edge_bucket": {},
        "by_length_bucket": {},
    }


def select_complex_curved_subset(
    package: dict[str, Any],
    archive_root: Path,
    split: str,
    max_samples: int,
    max_scan: int,
    max_seq_len: int,
    complex_min_faces: int,
    complex_min_edges: int,
    curved_threshold: float,
    max_source_faces: int,
    max_source_edges: int,
    curvature_rank_key: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vocab_info = normalize_vocab_info(package)
    groups = list(package.get(split, []))
    rng = random.Random(seed)
    scanned = 0
    grammar_ok = 0
    complex_candidates = 0
    parsed_loaded = 0
    parsed_failed = 0
    selected: list[dict[str, Any]] = []
    failures: dict[str, int] = {}

    for index, group in enumerate(groups):
        if scanned >= int(max_scan):
            break
        scanned += 1
        ids = input_ids_of(group)
        if not ids:
            failures["empty_sequence"] = failures.get("empty_sequence", 0) + 1
            continue
        if len(ids) > int(max_seq_len):
            failures["too_long"] = failures.get("too_long", 0) + 1
            continue
        grammar = grammar_validation(ids, vocab_info)
        if not grammar["ok"]:
            failures[f"grammar:{grammar['reason']}"] = failures.get(f"grammar:{grammar['reason']}", 0) + 1
            continue
        grammar_ok += 1
        faces = int(grammar["n_faces"])
        edges = int(grammar["n_edges"])
        if faces < int(complex_min_faces) and edges < int(complex_min_edges):
            failures["not_complex"] = failures.get("not_complex", 0) + 1
            continue
        if int(max_source_faces) > 0 and faces > int(max_source_faces):
            failures["too_many_faces"] = failures.get("too_many_faces", 0) + 1
            continue
        if int(max_source_edges) > 0 and edges > int(max_source_edges):
            failures["too_many_edges"] = failures.get("too_many_edges", 0) + 1
            continue
        complex_candidates += 1
        relpath = source_relpath_from_group(group)
        if not relpath:
            failures["missing_source_relpath"] = failures.get("missing_source_relpath", 0) + 1
            continue
        try:
            parsed = load_parsed_from_archive(relpath, archive_root)
            records = patch_records_from_parsed(parsed, relpath, complex_min_faces, complex_min_edges)
        except Exception as exc:
            parsed_failed += 1
            reason = f"parsed_load_failed:{type(exc).__name__}"
            failures[reason] = failures.get(reason, 0) + 1
            continue
        parsed_loaded += 1
        patch_summary = shape_patch_summary(records)
        rank_key = str(curvature_rank_key)
        curvature_rank_score = float(patch_summary.get(f"{rank_key}_curvature", patch_summary["max_curvature"]))
        if curvature_rank_score < float(curved_threshold):
            failures["not_curved"] = failures.get("not_curved", 0) + 1
            continue
        selected.append(
            {
                "source": "complex_curved_subset",
                "split": split,
                "index": int(index),
                "length": int(len(ids)),
                "sequence": ids,
                "source_relpath": relpath,
                "source_path": str(group.get("source_path", relpath)),
                "grammar_ok": True,
                "grammar_faces": faces,
                "grammar_edges": edges,
                "curvature_score": float(patch_summary["max_curvature"]),
                "curvature_rank_key": rank_key,
                "curvature_rank_score": curvature_rank_score,
                "p95_curvature": float(patch_summary["p95_curvature"]),
                "p90_curvature": float(patch_summary["p90_curvature"]),
                "mean_curvature": float(patch_summary["mean_curvature"]),
                "patch_count": int(patch_summary["patch_count"]),
                "surface_count": int(patch_summary["surface_count"]),
                "edge_count": int(patch_summary["edge_count"]),
                "length_bucket": sequence_length_bucket(len(ids)),
                "face_bucket": face_count_bucket(faces),
                "edge_bucket": edge_count_bucket(edges),
            }
        )

    selected.sort(
        key=lambda row: (
            -float(row["curvature_rank_score"]),
            -float(row["curvature_score"]),
            -int(row["grammar_faces"]),
            -int(row["grammar_edges"]),
            -int(row["length"]),
            rng.random(),
        )
    )
    selected = selected[: max(0, int(max_samples))]
    summary = {
        "split": split,
        "requested": int(max_samples),
        "selected": int(len(selected)),
        "max_scan": int(max_scan),
        "scanned": int(scanned),
        "grammar_ok": int(grammar_ok),
        "complex_candidates": int(complex_candidates),
        "parsed_loaded": int(parsed_loaded),
        "parsed_failed": int(parsed_failed),
        "curved_threshold": float(curved_threshold),
        "complex_min_faces": int(complex_min_faces),
        "complex_min_edges": int(complex_min_edges),
        "max_source_faces": int(max_source_faces),
        "max_source_edges": int(max_source_edges),
        "curvature_rank_key": str(curvature_rank_key),
        "failures": dict(sorted(failures.items())),
    }
    return selected, summary


def patch_chamfer(
    target: torch.Tensor,
    recon: torch.Tensor,
    point_stride: int,
    chunk_size: int,
) -> torch.Tensor:
    stride = max(1, int(point_stride))
    x = target.permute(0, 2, 3, 1).reshape(target.shape[0], -1, 3)[:, ::stride, :]
    y = recon.permute(0, 2, 3, 1).reshape(recon.shape[0], -1, 3)[:, ::stride, :]
    outputs: list[torch.Tensor] = []
    for start in range(0, x.shape[0], max(1, int(chunk_size))):
        xb = x[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        distances = torch.cdist(xb, yb)
        chamfer = distances.min(dim=2).values.mean(dim=1) + distances.min(dim=1).values.mean(dim=1)
        outputs.append(chamfer.detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.zeros((0,), dtype=torch.float32)


def evaluate_fsq_patches(
    selected: list[dict[str, Any]],
    archive_root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    chamfer_point_stride: int,
    chamfer_chunk_size: int,
    complex_min_faces: int,
    complex_min_edges: int,
) -> dict[str, Any]:
    model = load_fsq_vqvae(Path(checkpoint), device)
    patch_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    for row in selected:
        parsed = load_parsed_from_archive(str(row["source_relpath"]), archive_root)
        patch_records = patch_records_from_parsed(parsed, row["source_relpath"], complex_min_faces, complex_min_edges)
        for record in patch_records:
            records.append(record)
            owner_rows.append(row)

    patches = records_to_chw_array(records)
    mse_values: list[float] = []
    chamfer_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), max(1, int(batch_size))):
            batch_records = records[start : start + batch_size]
            batch_owners = owner_rows[start : start + batch_size]
            xb = torch.from_numpy(patches[start : start + batch_size]).to(device)
            h = model.encoder(xb)
            h = model.quant_conv(h)
            zq, _, _ = model.quantize(h)
            recon = model.decoder(model.post_quant_conv(zq))
            per_mse = (recon - xb).pow(2).flatten(1).mean(dim=1).detach().cpu()
            per_chamfer = patch_chamfer(
                xb.detach(),
                recon.detach(),
                point_stride=chamfer_point_stride,
                chunk_size=chamfer_chunk_size,
            )
            for offset, (record, owner) in enumerate(zip(batch_records, batch_owners)):
                mse = float(per_mse[offset].item())
                chamfer = float(per_chamfer[offset].item())
                mse_values.append(mse)
                chamfer_values.append(chamfer)
                patch_rows.append(
                    {
                        "source_relpath": owner["source_relpath"],
                        "sequence_index": int(owner["index"]),
                        "kind": str(record.get("kind", "")),
                        "record_id": str(record.get("record_id", "")),
                        "n_faces": int(record.get("n_faces", 0)),
                        "n_edges": int(record.get("n_edges", 0)),
                        "curvature_score": float(record.get("curvature_score", 0.0) or 0.0),
                        "mse": mse,
                        "chamfer": chamfer,
                        "face_bucket": owner["face_bucket"],
                        "edge_bucket": owner["edge_bucket"],
                        "length_bucket": owner["length_bucket"],
                    }
                )

    def grouped_stats(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in patch_rows:
            groups.setdefault(str(row.get(key, "unknown")), []).append(row)
        return {
            name: {
                "count": len(rows),
                "mse": numeric_stats([float(row["mse"]) for row in rows]),
                "chamfer": numeric_stats([float(row["chamfer"]) for row in rows]),
            }
            for name, rows in sorted(groups.items())
        }

    return {
        "patch_count": int(len(patch_rows)),
        "shape_count": int(len(selected)),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "chamfer_point_stride": int(max(1, chamfer_point_stride)),
        "mse": numeric_stats(mse_values),
        "chamfer": numeric_stats(chamfer_values),
        "by_kind": grouped_stats("kind"),
        "by_face_bucket": grouped_stats("face_bucket"),
        "by_edge_bucket": grouped_stats("edge_bucket"),
        "by_length_bucket": grouped_stats("length_bucket"),
        "patch_rows": patch_rows,
    }


def evaluate_ar_teacher_forcing(
    selected: list[dict[str, Any]],
    checkpoint: Path,
    vocab_info: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model = load_ar_model(Path(checkpoint), vocab_info, device)
    pad = int(vocab_info["PAD_TOKEN"])
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(selected), max(1, int(batch_size))):
            batch = selected[start : start + batch_size]
            max_len = max(int(row["length"]) for row in batch)
            ids = torch.full((len(batch), max_len), pad, dtype=torch.long, device=device)
            attention = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for row_index, row in enumerate(batch):
                seq = torch.tensor(row["sequence"], dtype=torch.long, device=device)
                ids[row_index, : seq.numel()] = seq
                attention[row_index, : seq.numel()] = 1
            outputs = model(input_ids=ids, attention_mask=attention, labels=None)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = ids[:, 1:].contiguous()
            label_mask = attention[:, 1:].contiguous().bool()
            labels_for_loss = labels.masked_fill(~label_mask, -100)
            token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels_for_loss.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
            token_loss = token_loss * label_mask.float()
            per_sample_tokens = label_mask.sum(dim=1)
            per_sample_loss = token_loss.sum(dim=1) / per_sample_tokens.clamp_min(1)
            for offset, row in enumerate(batch):
                token_count = int(per_sample_tokens[offset].item())
                ce = float(per_sample_loss[offset].item())
                total_loss += ce * token_count
                total_tokens += token_count
                rows.append(
                    {
                        "index": int(row["index"]),
                        "source_relpath": row["source_relpath"],
                        "length": int(row["length"]),
                        "token_count": token_count,
                        "teacher_ce": ce,
                        "grammar_faces": int(row["grammar_faces"]),
                        "grammar_edges": int(row["grammar_edges"]),
                        "curvature_score": float(row["curvature_score"]),
                        "curvature_rank_score": float(row.get("curvature_rank_score", row["curvature_score"])),
                        "face_bucket": row["face_bucket"],
                        "edge_bucket": row["edge_bucket"],
                        "length_bucket": row["length_bucket"],
                    }
                )

    def grouped_stats(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(key, "unknown")), []).append(row)
        return {
            name: {
                "count": len(items),
                "teacher_ce": numeric_stats([float(item["teacher_ce"]) for item in items]),
                "tokens": int(sum(int(item["token_count"]) for item in items)),
            }
            for name, items in sorted(groups.items())
        }

    return {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "sample_count": int(len(rows)),
        "token_count": int(total_tokens),
        "token_weighted_ce": (total_loss / total_tokens) if total_tokens else None,
        "teacher_ce": numeric_stats([float(row["teacher_ce"]) for row in rows]),
        "by_face_bucket": grouped_stats("face_bucket"),
        "by_edge_bucket": grouped_stats("edge_bucket"),
        "by_length_bucket": grouped_stats("length_bucket"),
        "rows": rows,
    }


def evaluate_true_token_reconstruction(
    selected: list[dict[str, Any]],
    vocab_info: dict[str, Any],
    vqvae_model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    write_step: bool,
    write_stl: bool,
    validate_step: bool,
    scale_factor: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    recon_dir = output_dir / "teacher_reconstruction"
    for row in selected:
        recon_row = reconstruct_one(
            row,
            vocab_info=vocab_info,
            vqvae_model=vqvae_model,
            device=device,
            output_dir=recon_dir,
            write_step=write_step,
            write_stl=write_stl,
            validate_step=validate_step,
            scale_factor=scale_factor,
        )
        recon_row["source_relpath"] = row["source_relpath"]
        recon_row["curvature_score"] = float(row["curvature_score"])
        recon_row["curvature_rank_score"] = float(row.get("curvature_rank_score", row["curvature_score"]))
        recon_row["face_bucket"] = row["face_bucket"]
        recon_row["edge_bucket"] = row["edge_bucket"]
        recon_row["length_bucket"] = row["length_bucket"]
        rows.append(recon_row)

    def grouped(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(key, "unknown")), []).append(row)
        out: dict[str, Any] = {}
        for name, items in sorted(groups.items()):
            summary = summarize_manifest_rows(items)
            summary["count"] = len(items)
            out[name] = summary
        return out

    summary = summarize_manifest_rows(rows)
    summary.update(
        {
            "write_step": bool(write_step),
            "write_stl": bool(write_stl),
            "validate_step": bool(validate_step),
            "by_face_bucket": grouped("face_bucket"),
            "by_edge_bucket": grouped("edge_bucket"),
            "by_length_bucket": grouped("length_bucket"),
            "rows": rows,
        }
    )
    return summary


def build_bucket_summary(
    selected: list[dict[str, Any]],
    fsq: dict[str, Any],
    ar: dict[str, Any],
    recon: dict[str, Any],
) -> dict[str, Any]:
    del selected
    return {
        "fsq_by_face_bucket": fsq.get("by_face_bucket", {}),
        "fsq_by_edge_bucket": fsq.get("by_edge_bucket", {}),
        "fsq_by_length_bucket": fsq.get("by_length_bucket", {}),
        "ar_by_face_bucket": ar.get("by_face_bucket", {}),
        "ar_by_edge_bucket": ar.get("by_edge_bucket", {}),
        "ar_by_length_bucket": ar.get("by_length_bucket", {}),
        "reconstruction_by_face_bucket": recon.get("by_face_bucket", {}),
        "reconstruction_by_edge_bucket": recon.get("by_edge_bucket", {}),
        "reconstruction_by_length_bucket": recon.get("by_length_bucket", {}),
    }


def render_markdown(report: dict[str, Any]) -> str:
    fsq = report["fsq_patch_metrics"]
    ar = report["ar_teacher_forcing"]
    recon = report["teacher_reconstruction"]
    ar_ce = "skipped" if ar.get("skipped") else ar["token_weighted_ce"]
    recon_success = "skipped" if recon.get("skipped") else f"{recon['reconstruct_success']}/{recon['attempted']}"
    step_saved = "skipped" if recon.get("skipped") else f"{recon['step_saved']}/{recon['attempted']}"
    brep_valid = "skipped" if recon.get("skipped") else f"{recon['brep_valid']}/{recon['attempted']}"
    lines = [
        "# Complex Curved Diagnostics",
        "",
        f"- Status: `{report['status']}`",
        f"- Selected records: `{report['selected_count']}`",
        f"- Output dir: `{report['output_dir']}`",
        f"- FSQ patch count: `{fsq['patch_count']}`",
        f"- FSQ MSE mean: `{fsq['mse']['mean']}`",
        f"- FSQ Chamfer mean: `{fsq['chamfer']['mean']}`",
        f"- AR token-weighted teacher CE: `{ar_ce}`",
        f"- True-token reconstruction success: `{recon_success}`",
        f"- True-token STEP saved: `{step_saved}`",
        f"- True-token BRep valid: `{brep_valid}`",
        "",
        "## Selection",
        "",
    ]
    for key, value in report["selection_summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Interpretation Hooks",
        "",
        "- If FSQ MSE and Chamfer are high on these real complex curved patches, prioritize FSQ capacity or loss changes before more AR sampling work.",
        "- If FSQ patch metrics are acceptable but AR teacher CE is high in long or high-face buckets, prioritize AR training distribution, context length, or ordering.",
        "- If true-token reconstruction fails while patch metrics are acceptable, inspect topology/OCC reconstruction and token-to-BRep assembly.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--complex-min-faces", type=int, default=12)
    parser.add_argument("--complex-min-edges", type=int, default=20)
    parser.add_argument("--curved-threshold", type=float, default=0.02)
    parser.add_argument("--curvature-rank-key", choices=["max", "p95", "p90", "mean"], default="p95")
    parser.add_argument("--max-source-faces", type=int, default=50)
    parser.add_argument("--max-source-edges", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ar-batch-size", type=int, default=4)
    parser.add_argument("--chamfer-point-stride", type=int, default=2)
    parser.add_argument("--chamfer-chunk-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--write-step", action="store_true")
    parser.add_argument("--write-stl", action="store_true")
    parser.add_argument("--validate-step", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    package = read_pickle(args.sequence)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, selection_summary = select_complex_curved_subset(
        package=package,
        archive_root=args.archive_root,
        split=args.split,
        max_samples=args.max_samples,
        max_scan=args.max_scan,
        max_seq_len=args.max_seq_len,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
        curved_threshold=args.curved_threshold,
        max_source_faces=args.max_source_faces,
        max_source_edges=args.max_source_edges,
        curvature_rank_key=args.curvature_rank_key,
        seed=args.seed,
    )
    if not selected:
        report = {
            "status": "FAILED",
            "reason": "no complex curved records selected",
            "selection_summary": selection_summary,
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "complex_curved_diagnostics_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 1

    write_jsonl(output_dir / "selected_subset.jsonl", selected)

    fsq_report = evaluate_fsq_patches(
        selected=selected,
        archive_root=args.archive_root,
        checkpoint=args.vqvae_checkpoint,
        device=device,
        batch_size=args.batch_size,
        chamfer_point_stride=args.chamfer_point_stride,
        chamfer_chunk_size=args.chamfer_chunk_size,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
    )
    fsq_patch_rows = fsq_report.pop("patch_rows")
    write_jsonl(output_dir / "fsq_patch_metrics.jsonl", fsq_patch_rows)
    write_json(output_dir / "fsq_patch_metrics.json", fsq_report)

    if args.skip_ar:
        ar_report = skipped_stage_report("ar_teacher_forcing", "skipped by --skip-ar")
        write_jsonl(output_dir / "ar_teacher_forcing.jsonl", [])
    else:
        ar_report = evaluate_ar_teacher_forcing(
            selected=selected,
            checkpoint=args.ar_checkpoint,
            vocab_info=vocab_info,
            device=device,
            batch_size=args.ar_batch_size,
        )
        ar_rows = ar_report.pop("rows")
        write_jsonl(output_dir / "ar_teacher_forcing.jsonl", ar_rows)
    write_json(output_dir / "ar_teacher_forcing.json", ar_report)

    if args.skip_reconstruction:
        reconstruction_report = skipped_stage_report(
            "teacher_reconstruction",
            "skipped by --skip-reconstruction",
        )
        reconstruction_report.update(
            {
                "attempted": 0,
                "grammar_valid": 0,
                "reconstruct_success": 0,
                "step_saved": 0,
                "stl_saved": 0,
                "brep_valid": 0,
                "errors": 0,
                "write_step": bool(args.write_step),
                "write_stl": bool(args.write_stl),
                "validate_step": bool(args.validate_step),
            }
        )
        write_jsonl(output_dir / "teacher_reconstruction_manifest.jsonl", [])
    else:
        recon_vqvae = load_fsq_vqvae(args.vqvae_checkpoint, device)
        reconstruction_report = evaluate_true_token_reconstruction(
            selected=selected,
            vocab_info=vocab_info,
            vqvae_model=recon_vqvae,
            device=device,
            output_dir=output_dir,
            write_step=args.write_step,
            write_stl=args.write_stl,
            validate_step=args.validate_step,
            scale_factor=args.scale_factor,
        )
        reconstruction_rows = reconstruction_report.pop("rows")
        write_jsonl(output_dir / "teacher_reconstruction_manifest.jsonl", reconstruction_rows)
    write_json(output_dir / "teacher_reconstruction_report.json", reconstruction_report)

    bucket_summary = build_bucket_summary(selected, fsq_report, ar_report, reconstruction_report)
    write_json(output_dir / "bucket_summary.json", bucket_summary)

    status = "VERIFIED" if len(selected) == int(args.max_samples) else "PARTIAL"
    report = {
        "status": status,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
        "sequence": str(args.sequence),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "ar_checkpoint": str(args.ar_checkpoint),
        "archive_root": str(args.archive_root),
        "output_dir": str(output_dir),
        "selected_count": int(len(selected)),
        "selection_summary": selection_summary,
        "fsq_patch_metrics": fsq_report,
        "ar_teacher_forcing": ar_report,
        "teacher_reconstruction": reconstruction_report,
        "bucket_summary": bucket_summary,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    write_json(output_dir / "complex_curved_diagnostics_report.json", report)
    (output_dir / "complex_curved_diagnostics_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": status, "selected": len(selected), "output": str(output_dir)}, indent=2))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
