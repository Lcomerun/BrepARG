import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS = REPO_ROOT / "breparg_improvements"
BREPARG = REPO_ROOT / "BrepARG"
sys.path.insert(0, str(IMPROVEMENTS))
sys.path.insert(0, str(BREPARG))

from ar_training_utils import summarize_ar_sequences, validate_ar_sequence_package
from model import ARModel


def load_package(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def usable_sequences(package, split_name, max_seq_len, limit):
    seqs = []
    for group in package.get(split_name, []):
        ids = group.get("original", {}).get("input_ids", [])
        if ids and len(ids) <= max_seq_len:
            seqs.append([int(x) for x in ids])
        if limit and len(seqs) >= limit:
            break
    return seqs


def make_batch(seqs, pad_token, batch_size):
    batch = seqs[:batch_size]
    width = max(len(seq) for seq in batch)
    ids = torch.full((len(batch), width), int(pad_token), dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    for row, seq in enumerate(batch):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        att[row, : len(seq)] = 1
    return ids, att


def run_preflight(args):
    started = time.time()
    package = load_package(args.sequence)
    validation = validate_ar_sequence_package(package, max_seq_len=args.max_seq_len)
    summary = summarize_ar_sequences(package, max_seq_len=args.max_seq_len)
    report = {
        "sequence": str(args.sequence),
        "summary": summary,
        "validation": validation,
        "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        "d_model": args.d_model,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
    }
    if validation["status"] != "VERIFIED":
        report["status"] = "FAILED"
        return report

    train = usable_sequences(package, "train", args.max_seq_len, args.max_samples)
    if len(train) < args.batch_size:
        report["status"] = "FAILED"
        report["error"] = f"not enough usable train sequences for batch_size={args.batch_size}"
        return report

    device = torch.device(report["device"])
    pad = package["special_tokens"]["PAD_TOKEN"]
    model = ARModel(
        vocab_size=package["vocab_size"],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.d_model * 4,
        dropout=0.1,
        max_seq_len=args.max_seq_len,
        pad_token_id=pad,
    ).to(device)
    model.train()
    ids, att = make_batch(train, pad, args.batch_size)
    ids = ids.to(device)
    att = att.to(device)
    labels = ids.clone()
    labels[labels == pad] = -100
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
    if not torch.isfinite(loss):
        report["status"] = "FAILED"
        report["error"] = "non-finite smoke loss"
        report["smoke_loss"] = None
        return report
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    report["status"] = "VERIFIED"
    report["smoke_loss"] = float(loss.detach().cpu())
    report["elapsed_seconds"] = round(time.time() - started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description="Preflight AR training input and one tiny optimization step.")
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    report = run_preflight(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.get("status") == "VERIFIED" else 2)


if __name__ == "__main__":
    main()
