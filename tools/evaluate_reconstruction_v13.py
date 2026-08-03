"""FSQ-aware reconstruction evaluation for the V13 AR run.

This tool evaluates the current V13 pipeline after VQ-VAE sequence generation:
it can reconstruct known validation/test token sequences with the FSQ VQ-VAE
checkpoint, and it can optionally sample token sequences from an AR checkpoint.
Outputs are written under local_runs so retained STEP files stay outside git.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"

for item in (str(REPO_ROOT), str(IMPROVEMENTS_DIR), str(BREPARG_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


DEFAULT_AR_OUT = REPO_ROOT / "local_runs" / "ar_training" / "train_outputs" / "newscheme_full_v13_ar"
DEFAULT_SEQUENCE = DEFAULT_AR_OUT / "sequences_fsq_rcm.pkl"
DEFAULT_AR_CHECKPOINT = DEFAULT_AR_OUT / "ar_best.pt"
DEFAULT_VQVAE = (
    REPO_ROOT
    / "ABC"
    / "processed"
    / "train_outputs"
    / "newscheme_full_vqvae_epoch100"
    / "fsq_vqvae_best.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local_runs" / "reconstruction_eval"


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_seed(seed: int, entropy=time.time_ns) -> int:
    """Return a numpy/torch-compatible seed, using entropy when seed is negative."""
    if int(seed) >= 0:
        return int(seed)
    return int(entropy()) % (2**32)


def build_sampling_config(args: argparse.Namespace, effective_seed: int) -> dict[str, Any]:
    return {
        "requested_seed": int(args.seed),
        "effective_seed": int(effective_seed),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "max_samples": int(args.max_samples),
    }


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _special_token(package: dict[str, Any], name: str, fallback: int) -> int:
    special = package.get("special_tokens") or {}
    if name in special:
        return int(special[name])
    if name in package:
        return int(package[name])
    return int(fallback)


def normalize_vocab_info(package: dict[str, Any]) -> dict[str, Any]:
    """Return the vocabulary dictionary expected by BrepARG.utils."""
    face_index_size = int(package["face_index_size"])
    se_codebook_size = int(package["se_codebook_size"])
    bbox_index_size = int(package["bbox_index_size"])

    face_index_offset = int(package.get("face_index_offset", 0))
    se_token_offset = int(package.get("se_token_offset", face_index_offset + face_index_size))
    bbox_token_offset = int(package.get("bbox_token_offset", se_token_offset + se_codebook_size))
    start_default = bbox_token_offset + bbox_index_size

    vocab = {
        "face_index_size": face_index_size,
        "se_codebook_size": se_codebook_size,
        "bbox_index_size": bbox_index_size,
        "face_index_offset": face_index_offset,
        "se_token_offset": se_token_offset,
        "bbox_token_offset": bbox_token_offset,
        "bbox_tokens_per_element": int(package.get("bbox_tokens_per_element", 6)),
        "se_tokens_per_element": int(package.get("se_tokens_per_element", 4)),
        "START_TOKEN": _special_token(package, "START_TOKEN", start_default),
        "SEP_TOKEN": _special_token(package, "SEP_TOKEN", start_default + 1),
        "END_TOKEN": _special_token(package, "END_TOKEN", start_default + 2),
        "PAD_TOKEN": _special_token(package, "PAD_TOKEN", start_default + 3),
    }
    vocab["vocab_size"] = int(package.get("vocab_size", vocab["PAD_TOKEN"] + 1))
    return vocab


def split_key(name: str) -> str:
    aliases = {
        "validation": "val",
        "valid": "val",
        "val": "val",
        "test": "test",
        "train": "train",
    }
    key = aliases.get(name.lower())
    if key is None:
        raise ValueError(f"Unknown split/source: {name}")
    return key


def _input_ids_of(group: dict[str, Any]) -> list[int]:
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in original.get("input_ids") or []]
    return [int(item) for item in group.get("input_ids") or []]


def _source_path_of(group: dict[str, Any]) -> str | None:
    """Return an optional parsed-geometry path carried by newer sequence packages."""
    keys = ("source_path", "path", "file_path", "pkl_path")
    for key in keys:
        value = group.get(key)
        if value:
            return str(value)
    original = group.get("original")
    if isinstance(original, dict):
        for key in keys:
            value = original.get(key)
            if value:
                return str(value)
    return None


def _shape_curvature_score(source_path: str | None, cache: dict[str, float]) -> float:
    if not source_path:
        return 0.0
    if source_path in cache:
        return cache[source_path]
    try:
        from vqvae_sampling import load_patch_records

        records = load_patch_records(Path(source_path))
        score = max((float(record.get("curvature_score", 0.0)) for record in records), default=0.0)
    except Exception:
        score = 0.0
    cache[source_path] = float(score)
    return cache[source_path]


def select_sequence_records(
    package: dict[str, Any],
    split: str,
    max_samples: int,
    order: str = "shortest",
    seed: int = 0,
    max_seq_len: int | None = 1024,
) -> list[dict[str, Any]]:
    """Select reconstruction candidates from a package split."""
    key = split_key(split)
    candidates: list[dict[str, Any]] = []
    needs_complexity = order in {"most_faces"}
    vocab_info = normalize_vocab_info(package) if needs_complexity else None
    curvature_cache: dict[str, float] = {}
    for index, group in enumerate(package.get(key, [])):
        ids = _input_ids_of(group)
        if not ids:
            continue
        if max_seq_len is not None and len(ids) > max_seq_len:
            continue
        source_path = _source_path_of(group)
        candidate = {
            "source": split,
            "split": key,
            "index": index,
            "length": len(ids),
            "sequence": ids,
        }
        if source_path:
            candidate["source_path"] = source_path
        if vocab_info is not None:
            grammar = grammar_validation(ids, vocab_info)
            candidate.update({
                "grammar_ok": bool(grammar["ok"]),
                "grammar_faces": int(grammar["n_faces"]),
                "grammar_edges": int(grammar["n_edges"]),
            })
        if order == "most_curved":
            candidate["curvature_score"] = _shape_curvature_score(source_path, curvature_cache)
        candidates.append(candidate)

    if order == "shortest":
        candidates.sort(key=lambda item: (item["length"], item["index"]))
    elif order == "longest":
        candidates.sort(key=lambda item: (-item["length"], item["index"]))
    elif order == "most_faces":
        candidates.sort(key=lambda item: (-item["grammar_faces"], -item["grammar_edges"], -item["length"], item["index"]))
    elif order == "most_curved":
        candidates.sort(key=lambda item: (-float(item.get("curvature_score", 0.0)), -item["length"], item["index"]))
    elif order == "original":
        pass
    elif order == "random":
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        raise ValueError(f"Unknown selection order: {order}")

    return candidates[: max(0, int(max_samples))]


def summarize_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reconstruct_success = [row for row in rows if row.get("status") == "saved" or row.get("solid_reconstructed")]
    return {
        "attempted": len(rows),
        "grammar_valid": sum(1 for row in rows if row.get("grammar_ok")),
        "reconstruct_success": len(reconstruct_success),
        "step_saved": sum(1 for row in rows if row.get("step_saved")),
        "stl_saved": sum(1 for row in rows if row.get("stl_saved")),
        "brep_valid": sum(1 for row in rows if row.get("brep_valid")),
        "errors": sum(1 for row in rows if row.get("status") not in {"saved", "dry_run"}),
    }


def load_fsq_vqvae(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the FSQ VQ-VAE used by the current V13 sequence files."""
    os.environ.setdefault("NS_OUTBASE", str(REPO_ROOT / "local_runs" / "reconstruction_eval" / "_train_import"))
    os.environ.setdefault("NS_OUT", "_fsq_import")
    from breparg_improvements import train as train_mod

    checkpoint = torch.load(checkpoint_path, map_location=device)
    levels = tuple(int(item) for item in checkpoint.get("fsq_levels", train_mod.FSQ_LEVELS))
    model = train_mod.build_fsq_vqvae(levels).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    materialize_fsq_decoder_embedding(model, device)
    return model


def materialize_fsq_decoder_embedding(model: torch.nn.Module, device: torch.device, chunk_size: int = 1024) -> None:
    """Expose a 64-channel decoder embedding for legacy BrepARG reconstruction.

    BrepARG.utils.decode_tokens_to_ncs expects quantize.embedding.weight to be
    the latent channel width consumed by post_quant_conv. The FSQ quantizer's
    placeholder embedding is only the scalar-code width, so this function
    materializes each FSQ index through indices_to_codes() and proj_out().
    """
    quantizer = model.quantize
    if not hasattr(quantizer, "fsq") or not hasattr(quantizer, "proj_out"):
        return

    num_embed = int(quantizer.fsq.codebook_size)
    out_channels = int(quantizer.proj_out.out_channels)
    weights: list[torch.Tensor] = []
    was_training = quantizer.training
    quantizer.eval()
    with torch.no_grad():
        for start in range(0, num_embed, chunk_size):
            ids = torch.arange(start, min(start + chunk_size, num_embed), dtype=torch.long, device=device)
            codes = quantizer.fsq.indices_to_codes(ids)
            codes_dchw = codes.reshape(-1, 1, 1, codes.shape[-1]).permute(0, 3, 1, 2).contiguous()
            projected = quantizer.proj_out(codes_dchw).reshape(-1, out_channels)
            weights.append(projected.detach().cpu())
    embedding = torch.nn.Embedding(num_embed, out_channels)
    embedding.weight.requires_grad_(False)
    embedding.weight.data.copy_(torch.cat(weights, dim=0))
    quantizer.embedding = embedding.to(device)
    if was_training:
        quantizer.train()


def infer_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_ar_model(checkpoint_path: Path, vocab_info: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from model import ARModel

    checkpoint = torch.load(checkpoint_path, map_location=device)
    d_model = int(checkpoint.get("d_model", 256))
    layers = int(checkpoint.get("layers", 8))
    model = ARModel(
        vocab_size=int(checkpoint.get("vocab_size", vocab_info["vocab_size"])),
        d_model=d_model,
        nhead=8,
        num_layers=layers,
        dim_feedforward=d_model * 4,
        dropout=0.1,
        max_seq_len=int((checkpoint.get("config") or {}).get("max_seq_len", 1024)),
        pad_token_id=int(vocab_info["PAD_TOKEN"]),
    ).to(device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.config.bos_token_id = int(vocab_info["START_TOKEN"])
    model.config.eos_token_id = int(vocab_info["END_TOKEN"])
    model.config.pad_token_id = int(vocab_info["PAD_TOKEN"])
    return model


def constraint_vocab(vocab_info: dict[str, Any]) -> Any:
    from constrained_decoding import BrepVocab

    return BrepVocab(
        face_index_size=int(vocab_info["face_index_size"]),
        se_codebook_size=int(vocab_info["se_codebook_size"]),
        bbox_index_size=int(vocab_info["bbox_index_size"]),
    )


def grammar_validation(sequence: list[int], vocab_info: dict[str, Any]) -> dict[str, Any]:
    """Validate the BrepARG token grammar before attempting OCC reconstruction."""
    from constrained_decoding import EDGE_BLOCK, EDGE_LEN, FACE_BLOCK, FACE_LEN

    V = constraint_vocab(vocab_info)
    body = [int(item) for item in sequence]
    if body and body[0] == V.START_TOKEN:
        body = body[1:]

    section = "face"
    fpos = 0
    declared: set[int] = set()
    used_faces_section: set[int] = set()
    edge_first: int | None = None
    seen_sep = False
    nfaces = 0
    nedges = 0

    for index, token in enumerate(body):
        if token == V.PAD_TOKEN:
            return {"ok": False, "reason": "PAD before END", "n_faces": nfaces, "n_edges": nedges}

        if section == "face":
            if fpos == 0 and token == V.SEP_TOKEN:
                if nfaces < 1:
                    return {"ok": False, "reason": "SEP before any face", "n_faces": nfaces, "n_edges": nedges}
                section = "edge"
                seen_sep = True
                fpos = 0
                used_faces_section = set()
                continue

            slot = FACE_BLOCK[fpos]
            if slot == "bbox" and not V.is_bbox(token):
                return {"ok": False, "reason": "face bbox slot", "n_faces": nfaces, "n_edges": nedges}
            if slot == "geo" and not V.is_geo(token):
                return {"ok": False, "reason": "face geo slot", "n_faces": nfaces, "n_edges": nedges}
            if slot == "idx":
                if not V.is_face_idx(token):
                    return {"ok": False, "reason": "face idx slot", "n_faces": nfaces, "n_edges": nedges}
                face_idx = V.face_idx_value(token)
                if face_idx in used_faces_section:
                    return {"ok": False, "reason": "dup face idx", "n_faces": nfaces, "n_edges": nedges}
                declared.add(face_idx)
                used_faces_section.add(face_idx)
                nfaces += 1
            fpos += 1
            if fpos == FACE_LEN:
                fpos = 0
            continue

        if fpos == 0 and token == V.END_TOKEN:
            rest = [item for item in body[index + 1:] if item != V.PAD_TOKEN]
            if rest:
                return {"ok": False, "reason": "tokens after END", "n_faces": nfaces, "n_edges": nedges}
            if not seen_sep:
                return {"ok": False, "reason": "END before SEP", "n_faces": nfaces, "n_edges": nedges}
            return {"ok": True, "reason": "ok", "n_faces": nfaces, "n_edges": nedges}

        slot = EDGE_BLOCK[fpos]
        if slot == "idx":
            if not V.is_face_idx(token):
                return {"ok": False, "reason": "edge idx slot", "n_faces": nfaces, "n_edges": nedges}
            face_idx = V.face_idx_value(token)
            if face_idx not in declared:
                return {"ok": False, "reason": "edge idx undeclared", "n_faces": nfaces, "n_edges": nedges}
            if fpos == 0:
                edge_first = face_idx
            elif face_idx == edge_first:
                return {"ok": False, "reason": "edge two faces equal", "n_faces": nfaces, "n_edges": nedges}
        elif slot == "bbox" and not V.is_bbox(token):
            return {"ok": False, "reason": "edge bbox slot", "n_faces": nfaces, "n_edges": nedges}
        elif slot == "geo" and not V.is_geo(token):
            return {"ok": False, "reason": "edge geo slot", "n_faces": nfaces, "n_edges": nedges}

        fpos += 1
        if fpos == EDGE_LEN:
            fpos = 0
            edge_first = None
            nedges += 1

    return {"ok": False, "reason": "truncated (no END)", "n_faces": nfaces, "n_edges": nedges}


def generate_ar_records(
    checkpoint_path: Path,
    vocab_info: dict[str, Any],
    count: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    constrained: bool,
) -> list[dict[str, Any]]:
    model = load_ar_model(checkpoint_path, vocab_info, device)
    prompt = torch.full((count, 1), int(vocab_info["START_TOKEN"]), dtype=torch.long, device=device)
    attention = torch.ones_like(prompt)
    logits_processor = None
    if constrained:
        from transformers import LogitsProcessorList
        from constrained_decoding import TopologyConstrainedLogitsProcessor

        processor = TopologyConstrainedLogitsProcessor(
            constraint_vocab(vocab_info),
            prompt_len=1,
            use_bbox_monotonic=True,
            enforce_face_unique=True,
            min_faces=1,
        )
        logits_processor = LogitsProcessorList([processor])
    with torch.no_grad():
        kwargs = {
            "input_ids": prompt,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 0,
            "pad_token_id": int(vocab_info["PAD_TOKEN"]),
            "eos_token_id": int(vocab_info["END_TOKEN"]),
        }
        if logits_processor is not None:
            kwargs["logits_processor"] = logits_processor
        generated = model.generate(**kwargs)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(generated.detach().cpu().tolist()):
        records.append({"source": "generated", "split": "generated", "index": index, "length": len(row), "sequence": row})
    return records


def _bbox_center_and_size(min_corner: np.ndarray, max_corner: np.ndarray) -> tuple[np.ndarray, float]:
    center = (np.asarray(min_corner, dtype=np.float32) + np.asarray(max_corner, dtype=np.float32)) / 2.0
    size = float(np.max(np.asarray(max_corner, dtype=np.float32) - np.asarray(min_corner, dtype=np.float32)))
    return center, size


def _bbox_minmax(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(point_cloud, dtype=np.float32)
    return np.min(points, axis=0), np.max(points, axis=0)


def cpu_safe_joint_optimize(
    surf_ncs: np.ndarray,
    edge_ncs: np.ndarray,
    surfPos: np.ndarray,
    unique_vertices: np.ndarray,
    EdgeVertexAdj: np.ndarray,
    FaceEdgeAdj: list[list[int]],
    num_edge: int,
    num_surf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU-safe replacement for BrepARG.utils.joint_optimize.

    The upstream helper hard-codes .cuda() and calls a chamferdist CUDA
    extension that is not available in this Windows environment. This local
    evaluator uses the same edge fitting and optimizes per-surface offsets
    with torch.cdist on CPU.
    """
    del num_edge, num_surf
    edge_ncs = np.asarray(edge_ncs, dtype=np.float32)
    surf_ncs = np.asarray(surf_ncs, dtype=np.float32)
    surfPos = np.asarray(surfPos, dtype=np.float32)
    unique_vertices = np.asarray(unique_vertices, dtype=np.float32)
    EdgeVertexAdj = np.asarray(EdgeVertexAdj, dtype=np.int64)

    edge_ncs_se = edge_ncs[:, [0, -1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]
    edge_wcs_chunks: list[np.ndarray] = []
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        scale_target = float(np.linalg.norm(vertex_se[0] - vertex_se[1]))
        scale_ncs = float(np.linalg.norm(ncs_se[0] - ncs_se[1]))
        edge_scale = scale_target / max(scale_ncs, 1e-8)

        edge_updated = wcs * edge_scale
        edge_se = ncs_se * edge_scale
        offset = vertex_se - edge_se
        offset_rev = vertex_se - edge_se[::-1]
        if np.abs(offset_rev[0] - offset_rev[1]).mean() < np.abs(offset[0] - offset[1]).mean():
            edge_updated = edge_updated[::-1]
            offset = offset_rev

        edge_updated = edge_updated + offset.mean(0)[np.newaxis, np.newaxis, :]
        edge_wcs_chunks.append(edge_updated.astype(np.float32))

    edge_wcs = np.vstack(edge_wcs_chunks) if edge_wcs_chunks else np.zeros((0, 32, 3), dtype=np.float32)

    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index, 0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index, 1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32) / 31)[:, np.newaxis], (1, 3))
        weighted_vec = np.tile(start_vec[np.newaxis, :], (32, 1)) * (1 - weight) + np.tile(end_vec, (32, 1)) * weight
        edge_wcs[index] += weighted_vec.astype(np.float32)

    surf_wcs_init: list[np.ndarray] = []
    face_edges: list[torch.Tensor] = []
    for adj, ncs, bbox in zip(FaceEdgeAdj, surf_ncs, surfPos):
        if len(adj) == 0:
            edges_perface = np.zeros((1, 32, 3), dtype=np.float32)
        else:
            edges_perface = edge_wcs[np.asarray(adj, dtype=np.int64)]
        face_edges.append(torch.as_tensor(edges_perface.reshape(-1, 3), dtype=torch.float32))

        surf_center, surf_scale = _bbox_center_and_size(bbox[0:3], bbox[3:])
        min_point, max_point = _bbox_minmax(edges_perface.reshape(-1, 3))
        _, edge_scale = _bbox_center_and_size(min_point, max_point)
        if surf_scale < edge_scale:
            surf_scale = 1.05 * edge_scale
        surf_wcs_init.append((ncs * (surf_scale / 2.0) + surf_center).astype(np.float32))

    if not surf_wcs_init:
        return np.zeros((0, 32, 32, 3), dtype=np.float32), edge_wcs

    surf = torch.as_tensor(np.stack(surf_wcs_init), dtype=torch.float32)
    offsets = torch.zeros((surf.shape[0], 1, 1, 3), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([offsets], lr=1e-3, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8)

    for _ in range(200):
        surf_updated = surf + offsets
        loss = torch.zeros((), dtype=torch.float32)
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            distances = torch.cdist(edge_pnts.reshape(1, -1, 3), surf_pnt.reshape(1, -1, 3))
            loss = loss + distances.min(dim=2).values.mean()
        loss = loss / max(1, len(face_edges))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (surf + offsets).detach().cpu().numpy(), edge_wcs


def patch_reconstruction_joint_optimize(brep_utils: Any) -> None:
    brep_utils.joint_optimize = cpu_safe_joint_optimize


def import_reconstruction_helpers():
    from OCC.Extend.DataExchange import write_step_file, write_stl_file
    from BrepARG import utils as brep_utils

    patch_reconstruction_joint_optimize(brep_utils)
    return brep_utils.reconstruct_cad_from_sequence, brep_utils.check_brep_validity, write_step_file, write_stl_file


def reconstruct_one(
    record: dict[str, Any],
    vocab_info: dict[str, Any],
    vqvae_model: torch.nn.Module | None,
    device: torch.device,
    output_dir: Path,
    write_step: bool,
    write_stl: bool,
    validate_step: bool,
    scale_factor: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": record["source"],
        "split": record["split"],
        "index": record["index"],
        "sequence_length": record["length"],
        "status": "dry_run",
        "solid_reconstructed": False,
        "step_saved": False,
        "stl_saved": False,
        "brep_valid": False,
    }
    if "source_path" in record:
        row["source_path"] = record["source_path"]
    if "curvature_score" in record:
        row["curvature_score"] = float(record["curvature_score"])
    grammar = grammar_validation(record["sequence"], vocab_info)
    row.update({
        "grammar_ok": bool(grammar["ok"]),
        "grammar_reason": grammar["reason"],
        "grammar_faces": grammar["n_faces"],
        "grammar_edges": grammar["n_edges"],
    })
    if not grammar["ok"]:
        row["status"] = "grammar_failed"
        return row
    if not write_step:
        return row

    reconstruct_cad_from_sequence, check_brep_validity, write_step_file, write_stl_file = import_reconstruction_helpers()
    try:
        solid = reconstruct_cad_from_sequence(
            sequence=record["sequence"],
            vocab_info=vocab_info,
            se_vqvae_model=vqvae_model,
            device=str(device),
            scale_factor=scale_factor,
            verbose=False,
        )
        if solid is None:
            row["status"] = "reconstruct_failed"
            row["error"] = "reconstruct_cad_from_sequence returned None"
            return row

        row["solid_reconstructed"] = True
        stem = f"{record['source']}_{int(record['index']):06d}_len{int(record['length']):04d}"
        step_path = output_dir / "steps" / f"{stem}.step"
        stl_path = output_dir / "stl" / f"{stem}.stl"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        stl_path.parent.mkdir(parents=True, exist_ok=True)

        write_step_file(solid, str(step_path))
        row["step_path"] = str(step_path)
        row["step_saved"] = step_path.exists() and step_path.stat().st_size > 0
        if write_stl:
            write_stl_file(solid, str(stl_path), linear_deflection=0.001, angular_deflection=0.5)
            row["stl_path"] = str(stl_path)
            row["stl_saved"] = stl_path.exists() and stl_path.stat().st_size > 0
        if validate_step and row["step_saved"]:
            row["brep_valid"] = bool(check_brep_validity(str(step_path)))
        row["status"] = "saved" if row["step_saved"] else "save_failed"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def build_output_dir(root: Path, name: str | None) -> Path:
    run_name = name or f"newscheme_full_v13_ar_best_{now_tag()}"
    return root / run_name


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    effective_seed = resolve_seed(args.seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    sequence_path = Path(args.sequence)
    package = read_pickle(sequence_path)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    output_dir = build_output_dir(Path(args.output_root), args.run_name)
    manifest_path = output_dir / "reconstruction_manifest.jsonl"
    report_path = output_dir / "reconstruction_report.json"

    records: list[dict[str, Any]]
    if args.source == "generated":
        records = generate_ar_records(
            Path(args.ar_checkpoint),
            vocab_info,
            count=args.max_samples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            constrained=args.constrained_decoding,
        )
    else:
        records = select_sequence_records(
            package,
            split=args.source,
            max_samples=args.max_samples,
            order=args.order,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
        )

    vqvae_model = None
    if args.write_step:
        vqvae_model = load_fsq_vqvae(Path(args.vqvae_checkpoint), device)

    rows = []
    for record in records:
        row = reconstruct_one(
            record,
            vocab_info=vocab_info,
            vqvae_model=vqvae_model,
            device=device,
            output_dir=output_dir,
            write_step=args.write_step,
            write_stl=args.write_stl,
            validate_step=args.validate_step,
            scale_factor=args.scale_factor,
        )
        rows.append(row)
        append_jsonl(manifest_path, row)

    summary = summarize_manifest_rows(rows)
    report = {
        "status": "VERIFIED" if rows and (not args.write_step or summary["step_saved"] > 0) else "FAILED",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": args.source,
        "sequence_path": str(sequence_path),
        "ar_checkpoint": str(args.ar_checkpoint),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "device": str(device),
        "write_step": bool(args.write_step),
        "write_stl": bool(args.write_stl),
        "validate_step": bool(args.validate_step),
        "constrained_decoding": bool(args.constrained_decoding),
        "sampling": build_sampling_config(args, effective_seed),
        "vocab": vocab_info,
        "summary": summary,
    }
    write_json(report_path, report)
    report["report"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V13 FSQ-aware CAD reconstruction and retain STEP files.")
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--source", choices=["validation", "val", "test", "train", "generated"], default="validation")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--order", choices=["shortest", "longest", "most_faces", "most_curved", "original", "random"], default="shortest")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducible sampling; use -1 for a fresh random seed.")
    parser.add_argument("--write-step", action="store_true")
    parser.add_argument("--write-stl", action="store_true")
    parser.add_argument("--validate-step", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--constrained-decoding", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report.get("status") == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
