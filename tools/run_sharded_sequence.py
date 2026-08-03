import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS = REPO_ROOT / "breparg_improvements"
BREPARG = REPO_ROOT / "BrepARG"

sys.path.insert(0, str(IMPROVEMENTS))
sys.path.insert(0, str(BREPARG))
sys.path.insert(0, str(BREPARG / "process_data"))

from gnn_ordering import rcm_face_ordering
from sequence_sharding import (
    chunk_id_from_path,
    group_split_paths_by_chunk,
    merge_sequence_shards,
    summarize_sequence_package,
)
from prepare_breparg_same_data_inputs import load_parsed_from_archive, source_relpath_from_group, valid_record


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_chunks(value):
    if not value:
        return None
    chunks = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for idx in range(int(start), int(end) + 1):
                chunks.add(f"abc_{idx:04d}")
        else:
            idx = int(part.replace("abc_", ""))
            chunks.add(f"abc_{idx:04d}")
    return chunks


def normalize_ordering(value):
    text = str(value or "rcm").strip().lower()
    if text not in {"rcm", "dfs"}:
        raise ValueError(f"unsupported ordering: {value!r}")
    return text


def load_sequence_module(ordering="rcm"):
    ordering = normalize_ordering(ordering)
    spec = importlib.util.spec_from_file_location("breparg_2sequence", BREPARG / "2sequence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if ordering == "rcm":
        module.dfs_face_ordering_from_core = lambda efp, nf: rcm_face_ordering(efp, nf)
    return module


def make_preprocessor(sequence_module, model, device):
    args = types.SimpleNamespace(max_face=50, max_edge=150, scale=1.0, aug=False)
    pre = object.__new__(sequence_module.ARDataPreprocessor)
    pre.data_list = None
    pre.se_vqvae_model = model
    pre.args = args
    pre.device = device
    pre.se_tokens_per_element, pre.bbox_tokens_per_element = sequence_module.calculate_tokens_per_element(
        model, device
    )
    pre.face_index_size = 50
    pre.se_codebook_size = 8192
    pre.bbox_index_size = 2048
    pre.special_token_size = 4
    pre.face_index_offset = 0
    pre.se_token_offset = pre.face_index_offset + pre.face_index_size
    pre.bbox_token_offset = pre.se_token_offset + pre.se_codebook_size
    special_token_offset = pre.bbox_token_offset + pre.bbox_index_size
    pre.START_TOKEN = special_token_offset
    pre.SEP_TOKEN = special_token_offset + 1
    pre.END_TOKEN = special_token_offset + 2
    pre.PAD_TOKEN = special_token_offset + 3
    pre.vocab_size = pre.face_index_size + pre.se_codebook_size + pre.bbox_index_size + pre.special_token_size
    pre.group_cache = []
    return pre


def metadata_from_preprocessor(pre, ordering="rcm"):
    return {
        "vocab_size": pre.vocab_size,
        "special_token_size": pre.special_token_size,
        "face_index_size": pre.face_index_size,
        "se_codebook_size": pre.se_codebook_size,
        "bbox_index_size": pre.bbox_index_size,
        "face_index_offset": pre.face_index_offset,
        "se_token_offset": pre.se_token_offset,
        "bbox_token_offset": pre.bbox_token_offset,
        "se_tokens_per_element": pre.se_tokens_per_element,
        "bbox_tokens_per_element": pre.bbox_tokens_per_element,
        "special_tokens": {
            "START_TOKEN": pre.START_TOKEN,
            "SEP_TOKEN": pre.SEP_TOKEN,
            "END_TOKEN": pre.END_TOKEN,
            "PAD_TOKEN": pre.PAD_TOKEN,
        },
        "ordering": normalize_ordering(ordering).upper(),
    }


def encode_record_group(sequence_module, pre, cad, source, split_name):
    surf_ncs = np.array(cad.get("surf_ncs", []), dtype=np.float32)
    edge_ncs = np.array(cad.get("edge_ncs", []), dtype=np.float32)
    edge_bbox_wcs = np.array(cad.get("edge_bbox_wcs", []), dtype=np.float32)
    surf_bbox_wcs = np.array(cad.get("surf_bbox_wcs", []), dtype=np.float32)
    edge_face_adj = cad.get("edgeFace_adj", [])
    face_edge_adj = cad.get("faceEdge_adj", None)

    if len(surf_ncs) == 0 or len(edge_ncs) == 0:
        return None
    if len(surf_ncs) > int(pre.args.max_face) or len(edge_ncs) > int(pre.args.max_edge):
        return None
    if face_edge_adj is None:
        return None

    threshold_value = 0.05
    scaled_value = 3
    surf_bbox = surf_bbox_wcs * scaled_value
    surf_bbox_reshaped = surf_bbox.reshape(len(surf_bbox), 2, 3)
    non_repeat = surf_bbox_reshaped[:1]
    for bbox in surf_bbox_reshaped:
        diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
        if (diff < threshold_value).sum() >= 1:
            continue
        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
    if len(non_repeat) != len(surf_bbox_reshaped):
        return None

    se_bbox = []
    for adj in face_edge_adj:
        if len(edge_bbox_wcs[adj]) == 0:
            return None
        se_bbox.append(edge_bbox_wcs[adj] * scaled_value)

    for bbox_group in se_bbox:
        edge_bbox_reshaped = bbox_group.reshape(len(bbox_group), 2, 3)
        non_repeat = edge_bbox_reshaped[:1]
        for bbox in edge_bbox_reshaped:
            diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
            if (diff < threshold_value).sum() >= 1:
                continue
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
        if len(non_repeat) != len(edge_bbox_reshaped):
            return None

    rotations = [0, 90, 180, 270] if split_name == "train" and bool(pre.args.aug) else [0]
    if split_name == "train":
        group = {"original": None, "augmented": []}
        for rot in rotations:
            tokens, attn = pre._encode_single_rotation(
                surf_ncs,
                edge_ncs,
                surf_bbox_wcs,
                edge_bbox_wcs,
                edge_face_adj,
                rotation_angle=rot,
            )
            item = {"input_ids": tokens, "attention_mask": attn}
            if rot == 0:
                group["original"] = item
            else:
                group["augmented"].append(item)
        if group["original"] is None:
            return None
    else:
        tokens, attn = pre._encode_single_rotation(
            surf_ncs,
            edge_ncs,
            surf_bbox_wcs,
            edge_bbox_wcs,
            edge_face_adj,
            rotation_angle=0,
        )
        group = {"original": {"input_ids": tokens, "attention_mask": attn}}
    return sequence_module.attach_sequence_source_path(group, source)


def process_chunk(chunk, paths_by_split, checkpoint, output_path, seed_base, ordering, archive_root=""):
    started = time.time()
    try:
        ordering = normalize_ordering(ordering)
        seed = int(seed_base) + int(chunk[-4:])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        import train as train_mod

        device = train_mod.DEVICE
        sequence_module = load_sequence_module(ordering)
        model = train_mod.build_fsq_vqvae(train_mod.FSQ_LEVELS).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        pre = make_preprocessor(sequence_module, model, device)

        package = {"train": [], "val": [], "test": []}
        package.update(metadata_from_preprocessor(pre, ordering=ordering))
        input_counts = {name: len(paths_by_split.get(name, [])) for name in ("train", "val", "test")}

        for split_name in ("train", "val", "test"):
            for path_or_relpath in paths_by_split.get(split_name, []):
                if archive_root:
                    cad = load_parsed_from_archive(str(path_or_relpath), Path(archive_root))
                    ok, _reason = valid_record(cad, max_faces=50, max_edges=150)
                    if not ok:
                        continue
                    group = encode_record_group(
                        sequence_module,
                        pre,
                        cad,
                        f"{Path(archive_root) / (str(path_or_relpath).split('/', 1)[0] + '_parsed.zip')}!/{path_or_relpath}",
                        split_name,
                    )
                else:
                    group = pre._process_single_cad(path_or_relpath, split_name)
                if group:
                    package[split_name].append(group)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
        with tmp_path.open("wb") as f:
            pickle.dump(package, f)
        os.replace(tmp_path, output_path)

        summary = summarize_sequence_package(package)
        summary.update({
            "chunk": chunk,
            "status": "done",
            "path": str(output_path),
            "input_train": input_counts["train"],
            "input_val": input_counts["val"],
            "input_test": input_counts["test"],
            "filtered_or_failed": sum(input_counts.values()) - summary["sequences"],
            "elapsed_seconds": round(time.time() - started, 3),
        })
        return summary
    except Exception as exc:
        return {
            "chunk": chunk,
            "status": "error",
            "path": str(output_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - started, 3),
        }


def provenance_split_from_sequence(sequence_path, limits):
    with Path(sequence_path).open("rb") as f:
        package = pickle.load(f)
    split = {"train": [], "val": [], "test": []}
    skipped = {}
    for split_name in ("train", "val", "test"):
        limit = int(limits.get(split_name, 0) or 0)
        groups = list(package.get(split_name, []) or [])
        if limit > 0:
            groups = groups[:limit]
        for group in groups:
            relpath = source_relpath_from_group(group)
            if not relpath:
                skipped["missing_source_relpath"] = skipped.get("missing_source_relpath", 0) + 1
                continue
            split[split_name].append(relpath)
    return split, skipped


def append_manifest(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_report(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def existing_shard_summary(chunk, shard_path):
    try:
        with Path(shard_path).open("rb") as f:
            package = pickle.load(f)
        summary = summarize_sequence_package(package)
        summary.update({"chunk": chunk, "status": "skipped_existing", "path": str(shard_path)})
        return summary
    except Exception as exc:
        return {"chunk": chunk, "status": "stale_existing", "path": str(shard_path), "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser(description="Generate sequence shards per ABC chunk and merge them.")
    parser.add_argument("--split", default="")
    parser.add_argument("--sequence-provenance", default="")
    parser.add_argument("--archive-root", default="")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--merge-output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunks", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--ordering", choices=["rcm", "dfs"], default="rcm")
    args = parser.parse_args()

    if not args.split and not args.sequence_provenance:
        raise SystemExit("either --split or --sequence-provenance is required")
    if args.sequence_provenance and not args.archive_root:
        raise SystemExit("--archive-root is required with --sequence-provenance")

    split_path = Path(args.split) if args.split else None
    shard_dir = Path(args.shard_dir)
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary)
    merge_output = Path(args.merge_output)
    selected_chunks = parse_chunks(args.chunks)
    ordering = normalize_ordering(args.ordering)

    provenance_skipped = {}
    if args.sequence_provenance:
        split, provenance_skipped = provenance_split_from_sequence(
            args.sequence_provenance,
            {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit},
        )
    else:
        with split_path.open("rb") as f:
            split = pickle.load(f)
    grouped = group_split_paths_by_chunk(split)
    if selected_chunks:
        grouped = {chunk: paths for chunk, paths in grouped.items() if chunk in selected_chunks}

    shard_dir.mkdir(parents=True, exist_ok=True)
    log(f"chunks={len(grouped)} workers={args.workers} ordering={ordering.upper()} shard_dir={shard_dir}")

    rows = []
    if not args.merge_only:
        pending = {}
        for chunk, paths_by_split in grouped.items():
            shard_path = shard_dir / f"{chunk}.pkl"
            if args.resume and shard_path.exists():
                row = existing_shard_summary(chunk, shard_path)
                rows.append(row)
                append_manifest(manifest_path, row)
                log(f"skip {chunk}: existing shard sequences={row.get('sequences')}")
                continue
            pending[chunk] = (paths_by_split, shard_path)

        if pending:
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
                future_to_chunk = {
                    executor.submit(
                        process_chunk,
                        chunk,
                        paths_by_split,
                        args.checkpoint,
                        str(shard_path),
                        args.seed_base,
                        ordering,
                        args.archive_root,
                    ): chunk
                    for chunk, (paths_by_split, shard_path) in pending.items()
                }
                for future in as_completed(future_to_chunk):
                    row = future.result()
                    rows.append(row)
                    append_manifest(manifest_path, row)
                    if row.get("status") == "done":
                        log(
                            f"done {row['chunk']}: seq={row['sequences']} "
                            f"train/val/test={row['train']}/{row['val']}/{row['test']} "
                            f"filtered={row['filtered_or_failed']} sec={row['elapsed_seconds']}"
                        )
                    else:
                        log(f"{row.get('status')} {row.get('chunk')}: {row.get('error')}")

    shard_paths = [shard_dir / f"{chunk}.pkl" for chunk in sorted(grouped) if (shard_dir / f"{chunk}.pkl").exists()]
    merge_summary = merge_sequence_shards(shard_paths, merge_output, summary_path=summary_path)
    merge_summary["status"] = "VERIFIED" if (
        merge_summary["out_of_vocab"] == 0 and merge_summary["se_tokens_per_element"] == 4
    ) else "FAILED"
    summary_path.write_text(json.dumps(merge_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"merged shards={merge_summary['shards']} sequences={merge_summary['sequences']} "
        f"out_of_vocab={merge_summary['out_of_vocab']} -> {merge_output}"
    )

    if args.report:
        report = load_report(args.report)
        stages = report.setdefault("stages", {})
        stages["sequence_sharded"] = {
            "status": merge_summary["status"],
            "sequences": merge_summary["sequences"],
            "train": merge_summary["train"],
            "val": merge_summary["val"],
            "test": merge_summary["test"],
            "vocab_size": merge_summary["vocab_size"],
            "max_token": merge_summary["max_token"],
            "out_of_vocab": merge_summary["out_of_vocab"],
            "se_tokens_per_element": merge_summary["se_tokens_per_element"],
            "ordering": merge_summary["ordering"],
            "requested_ordering": ordering.upper(),
            "shards": merge_summary["shards"],
            "provenance_skipped": provenance_skipped,
            "shard_dir": str(shard_dir),
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "merged_output": str(merge_output),
        }
        overall = report.setdefault("overall", {})
        overall["sequence_sharded"] = "PASS" if merge_summary["status"] == "VERIFIED" else "FAIL"
        save_report(args.report, report)

    if merge_summary["status"] != "VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
