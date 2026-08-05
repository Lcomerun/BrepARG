"""Validate global parsed-archive member identities without unpickling payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))

from cad_protocol import (  # noqa: E402
    summarize_archive_member_inventory,
    validate_archive_member_inventory,
)
from build_cad_protocol import discover_archives  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--chunks", default="all")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    archives = discover_archives(args.archive_root, args.chunks)
    inventory = validate_archive_member_inventory(archives)
    summary = {
        "status": "VERIFIED",
        "archive_root": str(args.archive_root.resolve()),
        "chunks": args.chunks,
        **summarize_archive_member_inventory(inventory),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
