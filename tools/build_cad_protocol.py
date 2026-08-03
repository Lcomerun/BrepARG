"""Build Protocol V2 manifests and parent-isolated splits from parsed ZIPs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))

from cad_protocol import ProtocolConfig, build_protocol  # noqa: E402


def parse_chunk_selection(value: str) -> set[str] | None:
    value = str(value or "all").strip().lower()
    if value in {"", "all", "*"}:
        return None
    selected: set[str] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"invalid chunk range: {token}")
            selected.update(f"abc_{index:04d}" for index in range(start, end + 1))
        else:
            index = int(token.removeprefix("abc_"))
            selected.add(f"abc_{index:04d}")
    return selected


def discover_archives(root: Path, chunks: str) -> list[Path]:
    selected = parse_chunk_selection(chunks)
    paths = sorted(Path(root).glob("abc_*_parsed.zip"))
    if selected is not None:
        paths = [path for path in paths if re.sub(r"_parsed\.zip$", "", path.name) in selected]
    if not paths:
        raise FileNotFoundError(f"no parsed archives selected under {root}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--chunks", default="all", help="all, comma list, or inclusive ranges such as 0-4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize-root", type=Path, required=True)
    parser.add_argument("--max-scan-records", type=int, default=0)
    parser.add_argument("--max-eligible-records", type=int, default=0)
    parser.add_argument("--min-faces", type=int, default=10)
    parser.add_argument("--max-faces", type=int, default=50)
    parser.add_argument("--max-global-edges", type=int, default=150)
    parser.add_argument("--max-edges-per-face", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProtocolConfig(
        min_faces=args.min_faces,
        max_faces=args.max_faces,
        max_global_edges=args.max_global_edges,
        max_edges_per_face=args.max_edges_per_face,
        seed=args.seed,
    )
    archives = discover_archives(args.archive_root, args.chunks)
    _, _, summary = build_protocol(
        archive_paths=archives,
        config=config,
        output_dir=args.output_dir,
        materialize_root=args.materialize_root,
        max_scan_records=args.max_scan_records,
        max_eligible_records=args.max_eligible_records,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    return 0 if summary.get("status") == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
