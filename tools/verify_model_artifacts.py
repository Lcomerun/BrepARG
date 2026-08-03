"""Verify that uploaded model artifacts are loadable before training.

This server-side gate is intentionally lightweight: it does not instantiate the
models or start a training job. It catches the common failure mode where files
exist and pass byte-count checks but the checkpoint or pickle contents are not
the expected V13 artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any


DEFAULT_VQVAE = Path("/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt")
DEFAULT_AR = Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt")
DEFAULT_SEQUENCE = Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl")
DEFAULT_SPLIT = Path("/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl")

REQUIRED_SEQUENCE_KEYS = [
    "train",
    "val",
    "test",
    "vocab_size",
    "face_index_size",
    "se_codebook_size",
    "bbox_index_size",
    "special_tokens",
]
REQUIRED_SPECIAL_TOKENS = ["START_TOKEN", "SEP_TOKEN", "END_TOKEN", "PAD_TOKEN"]
REQUIRED_SPLITS = ["train", "val", "test"]


def file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(path.is_file()) if exists else False,
        "bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
    }


def load_torch_checkpoint(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"checkpoint_not_dict:{type(payload).__name__}"
    return payload, None


def load_pickle_payload(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return payload, None


def checkpoint_tensor_count(model_state: Any) -> int:
    if not isinstance(model_state, dict):
        return 0
    return len(model_state)


def summarize_vqvae_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    summary = file_status(path)
    summary.update({"ok": False, "issues": [], "fsq_levels": None, "model_state_keys": 0})
    if not summary["exists"]:
        summary["issues"].append("missing")
        return summary
    if not summary["is_file"] or summary["bytes"] <= 0:
        summary["issues"].append("not_nonempty_file")
        return summary
    payload, error = load_torch_checkpoint(path)
    if error:
        summary["issues"].append("torch_load_failed")
        summary["load_error"] = error
        return summary
    model_state = payload.get("model_state_dict") if payload is not None else None
    if not isinstance(model_state, dict) or not model_state:
        summary["issues"].append("missing_model_state_dict")
    levels = payload.get("fsq_levels") if payload is not None else None
    if levels is None:
        summary["issues"].append("missing_fsq_levels")
    else:
        try:
            summary["fsq_levels"] = [int(item) for item in levels]
        except (TypeError, ValueError):
            summary["issues"].append("invalid_fsq_levels")
    summary["model_state_keys"] = checkpoint_tensor_count(model_state)
    summary["ok"] = not summary["issues"]
    return summary


def summarize_ar_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    summary = file_status(path)
    summary.update(
        {
            "ok": False,
            "issues": [],
            "epoch": None,
            "best_val_ce": None,
            "vocab_size": None,
            "max_seq_len": None,
            "model_state_keys": 0,
        }
    )
    if not summary["exists"]:
        summary["issues"].append("missing")
        return summary
    if not summary["is_file"] or summary["bytes"] <= 0:
        summary["issues"].append("not_nonempty_file")
        return summary
    payload, error = load_torch_checkpoint(path)
    if error:
        summary["issues"].append("torch_load_failed")
        summary["load_error"] = error
        return summary
    model_state = payload.get("model_state_dict") if payload is not None else None
    if not isinstance(model_state, dict) or not model_state:
        summary["issues"].append("missing_model_state_dict")
    for key in ("epoch", "best_val_ce", "vocab_size"):
        summary[key] = payload.get(key)
    config = payload.get("config") or {}
    summary["max_seq_len"] = config.get("max_seq_len")
    summary["model_state_keys"] = checkpoint_tensor_count(model_state)
    if summary["epoch"] is None:
        summary["issues"].append("missing_epoch")
    summary["ok"] = not summary["issues"]
    return summary


def count_split_rows(payload: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for split in REQUIRED_SPLITS:
        value = payload.get(split)
        counts[split] = len(value) if isinstance(value, list) else 0
    return counts


def first_ids(row: Any) -> list[int]:
    if isinstance(row, dict):
        original = row.get("original")
        if isinstance(original, dict) and isinstance(original.get("input_ids"), list):
            return [int(item) for item in original["input_ids"]]
        if isinstance(row.get("input_ids"), list):
            return [int(item) for item in row["input_ids"]]
    return []


def summarize_sequence_package(path: str | Path, sample_rows_per_split: int = 1) -> dict[str, Any]:
    path = Path(path)
    summary = file_status(path)
    summary.update({"ok": False, "issues": [], "split_counts": {}, "vocab_size": None, "max_sample_token": None})
    if not summary["exists"]:
        summary["issues"].append("missing")
        return summary
    if not summary["is_file"] or summary["bytes"] <= 0:
        summary["issues"].append("not_nonempty_file")
        return summary
    payload, error = load_pickle_payload(path)
    if error:
        summary["issues"].append("pickle_load_failed")
        summary["load_error"] = error
        return summary
    if not isinstance(payload, dict):
        summary["issues"].append(f"package_not_dict:{type(payload).__name__}")
        return summary

    missing_keys = [key for key in REQUIRED_SEQUENCE_KEYS if key not in payload]
    for key in missing_keys:
        summary["issues"].append(f"missing_key:{key}")
    for split in REQUIRED_SPLITS:
        value = payload.get(split)
        if not isinstance(value, list):
            summary["issues"].append(f"missing_split:{split}")
        elif not value:
            summary["issues"].append(f"empty_split:{split}")
    special = payload.get("special_tokens")
    if not isinstance(special, dict):
        summary["issues"].append("special_tokens_not_dict")
    else:
        for token in REQUIRED_SPECIAL_TOKENS:
            if token not in special:
                summary["issues"].append(f"missing_special_token:{token}")
    vocab_size = payload.get("vocab_size")
    try:
        vocab_size = int(vocab_size)
    except (TypeError, ValueError):
        summary["issues"].append("invalid_vocab_size")
        vocab_size = 0
    if vocab_size <= 0:
        summary["issues"].append("nonpositive_vocab_size")
    summary["vocab_size"] = vocab_size
    summary["split_counts"] = count_split_rows(payload)

    max_token = None
    for split in REQUIRED_SPLITS:
        rows = payload.get(split)
        if not isinstance(rows, list):
            continue
        for row in rows[: max(0, int(sample_rows_per_split))]:
            ids = first_ids(row)
            if ids:
                row_max = max(ids)
                max_token = row_max if max_token is None else max(max_token, row_max)
                if vocab_size and row_max >= vocab_size:
                    summary["issues"].append(f"sample_token_out_of_vocab:{split}")
                if min(ids) < 0:
                    summary["issues"].append(f"sample_token_negative:{split}")
    summary["max_sample_token"] = max_token
    summary["ok"] = not summary["issues"]
    return summary


def summarize_split_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    summary = file_status(path)
    summary.update({"ok": False, "issues": [], "split_counts": {}})
    if not summary["exists"]:
        summary["issues"].append("missing")
        return summary
    if not summary["is_file"] or summary["bytes"] <= 0:
        summary["issues"].append("not_nonempty_file")
        return summary
    payload, error = load_pickle_payload(path)
    if error:
        summary["issues"].append("pickle_load_failed")
        summary["load_error"] = error
        return summary
    if not isinstance(payload, dict):
        summary["issues"].append(f"split_not_dict:{type(payload).__name__}")
        return summary
    for split in REQUIRED_SPLITS:
        value = payload.get(split)
        if not isinstance(value, list):
            summary["issues"].append(f"missing_split:{split}")
        elif not value:
            summary["issues"].append(f"empty_split:{split}")
    summary["split_counts"] = count_split_rows(payload)
    summary["ok"] = not summary["issues"]
    return summary


def evaluate_model_artifacts(
    *,
    vqvae_checkpoint: str | Path,
    ar_checkpoint: str | Path,
    sequence: str | Path,
    split: str | Path,
    sample_rows_per_split: int = 1,
) -> dict[str, Any]:
    vqvae = summarize_vqvae_checkpoint(vqvae_checkpoint)
    ar = summarize_ar_checkpoint(ar_checkpoint)
    sequence_summary = summarize_sequence_package(sequence, sample_rows_per_split=sample_rows_per_split)
    split_summary = summarize_split_file(split)

    blocking_reasons = []
    if not vqvae["ok"]:
        blocking_reasons.append("vqvae_checkpoint_invalid")
    if not ar["ok"]:
        blocking_reasons.append("ar_checkpoint_invalid")
    if not sequence_summary["ok"]:
        blocking_reasons.append("sequence_package_invalid")
    if not split_summary["ok"]:
        blocking_reasons.append("split_file_invalid")

    ready = not blocking_reasons
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "MODEL_ARTIFACTS_READY" if ready else "MODEL_ARTIFACTS_FAILED",
        "artifacts_ready": ready,
        "blocking_reasons": blocking_reasons,
        "vqvae_checkpoint": vqvae,
        "ar_checkpoint": ar,
        "sequence_package": sequence_summary,
        "split_file": split_summary,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Model Artifact Sanity",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Artifacts ready: {payload['artifacts_ready']}",
        f"- Blocking reasons: {', '.join(payload['blocking_reasons']) if payload['blocking_reasons'] else 'none'}",
        "",
        "## Summary",
        "",
    ]
    for key, title in [
        ("vqvae_checkpoint", "VQ-VAE checkpoint"),
        ("ar_checkpoint", "AR checkpoint"),
        ("sequence_package", "Sequence package"),
        ("split_file", "Split file"),
    ]:
        item = payload[key]
        issues = ", ".join(item.get("issues") or []) or "none"
        lines.append(f"- {title}: ok={item.get('ok')} issues={issues}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify V13 model artifacts can be loaded before server training.")
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--ar-checkpoint", type=Path, default=DEFAULT_AR)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--sample-rows-per-split", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_model_artifacts(
        vqvae_checkpoint=args.vqvae_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        sequence=args.sequence,
        split=args.split,
        sample_rows_per_split=args.sample_rows_per_split,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["artifacts_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
