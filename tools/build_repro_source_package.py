from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.repro_package_builder import (
    PackageBuildError,
    build_package,
    validate_catalogs,
    verify_archive,
)


DEFAULT_PACKAGE_NAME = "v13_repro_source_20260802.zip"
DEFAULT_RELEASE_EPOCH = 1785657600
DEFAULT_CLEAN_COMMIT = "16cf19bb79b6bfa8beb4660e88f8d9dc813216e2"


def _positive_epoch(value: str) -> int:
    try:
        epoch = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("release epoch must be an integer") from exc
    if epoch <= 0:
        raise argparse.ArgumentTypeError("release epoch must be positive")
    return epoch


def _default_epoch() -> int:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return DEFAULT_RELEASE_EPOCH
    return _positive_epoch(value)


def build_parser() -> argparse.ArgumentParser:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build or verify the lightweight V13 reproducibility source archive."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
    parser.add_argument("--clean-commit", default=DEFAULT_CLEAN_COMMIT)
    parser.add_argument("--release-epoch", type=_positive_epoch)
    parser.add_argument("--history-root", type=Path, action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--verify-archive", type=Path)
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    if args.validate_only:
        _print_json(validate_catalogs(repo_root))
        return 0

    if args.verify_archive is not None:
        _print_json(verify_archive(args.verify_archive.resolve()))
        return 0

    try:
        epoch = args.release_epoch if args.release_epoch is not None else _default_epoch()
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    output_dir = (args.output_dir or (repo_root / "dist")).resolve()
    history_roots = [path.resolve() for path in args.history_root]
    result = build_package(
        repo_root,
        output_dir,
        args.package_name,
        epoch,
        clean_commit=args.clean_commit,
        history_roots=history_roots,
    )
    _print_json(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackageBuildError as exc:
        print(f"package build failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
