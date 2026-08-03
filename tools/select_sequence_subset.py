"""Create a small sequence package subset selected by grammar complexity."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
BREPARG_DIR = REPO_ROOT / "BrepARG"
for item in (TOOLS_DIR, IMPROVEMENTS_DIR, BREPARG_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_reconstruction_v13 import grammar_validation, normalize_vocab_info


def input_ids(group: dict) -> list[int]:
    original = group.get("original")
    if isinstance(original, dict):
        return [int(item) for item in original.get("input_ids") or []]
    return [int(item) for item in group.get("input_ids") or []]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-faces", type=int, default=12)
    parser.add_argument("--max-faces", type=int, default=24)
    parser.add_argument("--min-edges", type=int, default=20)
    parser.add_argument("--max-edges", type=int, default=80)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    with args.sequence.open("rb") as handle:
        package = pickle.load(handle)
    vocab_info = normalize_vocab_info(package)
    selected = []
    for index, group in enumerate(package.get(args.split, [])):
        ids = input_ids(group)
        if not ids or len(ids) > args.max_seq_len:
            continue
        grammar = grammar_validation(ids, vocab_info)
        if not grammar["ok"]:
            continue
        faces = int(grammar["n_faces"])
        edges = int(grammar["n_edges"])
        if not (args.min_faces <= faces <= args.max_faces):
            continue
        if not (args.min_edges <= edges <= args.max_edges):
            continue
        clone = dict(group)
        clone["source_original_index"] = index
        clone["selected_faces"] = faces
        clone["selected_edges"] = edges
        selected.append(clone)
        if len(selected) >= args.max_samples:
            break

    subset = {key: value for key, value in package.items() if key not in {"train", "val", "test"}}
    subset["train"] = []
    subset["val"] = selected
    subset["test"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(subset, handle)
    print(f"selected={len(selected)} output={args.output}")
    if selected:
        print(
            "faces_range",
            min(int(item["selected_faces"]) for item in selected),
            max(int(item["selected_faces"]) for item in selected),
        )
        print(
            "edges_range",
            min(int(item["selected_edges"]) for item in selected),
            max(int(item["selected_edges"]) for item in selected),
        )
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
