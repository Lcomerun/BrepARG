import json
from pathlib import Path

import torch


def ar_checkpoint_paths(out_dir):
    out_dir = Path(out_dir)
    return {
        "best": out_dir / "ar_best.pt",
        "latest": out_dir / "ar_latest.pt",
        "checkpoint_dir": out_dir / "ar_checkpoints",
        "history": out_dir / "ar_history.jsonl",
    }


def periodic_checkpoint_path(checkpoint_dir, epoch):
    return Path(checkpoint_dir) / f"ar_epoch_{int(epoch):04d}.pt"


def _input_ids_of(group):
    if not isinstance(group, dict):
        return []
    original = group.get("original")
    if isinstance(original, dict):
        return original.get("input_ids") or []
    return group.get("input_ids") or []


def summarize_ar_sequences(package, max_seq_len=1024):
    vocab_size = int(package.get("vocab_size", 0) or 0)
    summary = {
        "raw_train": len(package.get("train", [])),
        "raw_val": len(package.get("val", [])),
        "raw_test": len(package.get("test", [])),
        "usable_train": 0,
        "usable_val": 0,
        "usable_test": 0,
        "vocab_size": vocab_size,
        "max_token": -1,
        "min_token": None,
        "out_of_vocab": 0,
        "empty_sequences": 0,
        "over_max_len": 0,
        "max_seq_len": int(max_seq_len),
    }
    for split_name in ("train", "val", "test"):
        for group in package.get(split_name, []):
            ids = [int(x) for x in _input_ids_of(group)]
            if not ids:
                summary["empty_sequences"] += 1
                continue
            summary["max_token"] = max(summary["max_token"], max(ids))
            summary["min_token"] = min(ids) if summary["min_token"] is None else min(summary["min_token"], min(ids))
            if vocab_size and (max(ids) >= vocab_size or min(ids) < 0):
                summary["out_of_vocab"] += 1
            if len(ids) > max_seq_len:
                summary["over_max_len"] += 1
                continue
            summary[f"usable_{split_name}"] += 1
    if summary["min_token"] is None:
        summary["min_token"] = -1
    summary["raw_total"] = summary["raw_train"] + summary["raw_val"] + summary["raw_test"]
    summary["usable_total"] = summary["usable_train"] + summary["usable_val"] + summary["usable_test"]
    return summary


def validate_ar_sequence_package(package, max_seq_len=1024):
    missing = [key for key in ("train", "val", "test", "vocab_size", "special_tokens") if key not in package]
    summary = summarize_ar_sequences(package, max_seq_len=max_seq_len)
    pad_token = None
    if isinstance(package.get("special_tokens"), dict):
        pad_token = package["special_tokens"].get("PAD_TOKEN")
    status = "VERIFIED"
    if missing or pad_token is None or summary["out_of_vocab"] or not summary["usable_train"] or not summary["usable_val"]:
        status = "FAILED"
    summary.update({"status": status, "missing_keys": missing, "pad_token": pad_token})
    return summary


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_ar_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ar_checkpoint(path, map_location):
    return torch.load(path, map_location=map_location)
