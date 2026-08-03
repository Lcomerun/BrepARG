from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repro_runtime import (  # type: ignore
        ReproError,
        format_json,
        latest_run,
        load_experiments,
        load_paths,
        package_preflight,
        run_experiment,
        verify_run,
    )
else:
    from .repro_runtime import (
        ReproError,
        format_json,
        latest_run,
        load_experiments,
        load_paths,
        package_preflight,
        run_experiment,
        verify_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and run the V13 reproducibility source package."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("list")
    explain = subparsers.add_parser("explain")
    explain.add_argument("experiment_id")
    for name in ("smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("experiment_id")
        command.add_argument("--allow-historical-failed", action="store_true")
    for name in ("status", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("experiment_id")
        command.add_argument("--run-dir", type=Path)
    return parser


def _run_root(package_root: Path) -> Path:
    paths = load_paths(package_root / "configs" / "paths.env")
    value = paths.get("V13_RUN_ROOT", "")
    if not value:
        raise ReproError("V13_RUN_ROOT is not configured")
    return Path(value).expanduser()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.package_root.resolve()
    try:
        if args.action == "list":
            experiments = load_experiments(root)
            for experiment in sorted(
                experiments.values(), key=lambda item: (item["category"], item["id"])
            ):
                print(
                    f"{experiment['id']:<42} "
                    f"{experiment['category']:<18} "
                    f"{experiment['state']:<24} "
                    f"{experiment.get('title', '')}"
                )
            return 0
        if args.action == "explain":
            experiments = load_experiments(root)
            if args.experiment_id not in experiments:
                raise ReproError(f"unknown experiment id: {args.experiment_id}")
            print(format_json(experiments[args.experiment_id]))
            return 0
        if args.action == "preflight":
            report = package_preflight(root)
            print(format_json(report))
            return 0 if report["target_ready"] else 2
        if args.action == "bootstrap":
            script = root / "environments" / "bootstrap.sh"
            if not script.is_file():
                raise ReproError(f"bootstrap script is missing: {script}")
            return subprocess.run(["bash", str(script)], cwd=root, check=False).returncode
        if args.action in {"smoke", "run"}:
            return run_experiment(
                root,
                args.experiment_id,
                allow_historical_failed=args.allow_historical_failed,
                smoke=args.action == "smoke",
            )
        experiments = load_experiments(root)
        if args.experiment_id not in experiments:
            raise ReproError(f"unknown experiment id: {args.experiment_id}")
        run_dir = args.run_dir or latest_run(_run_root(root), args.experiment_id)
        if args.action == "status":
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise ReproError(f"run manifest is missing: {manifest_path}")
            print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), indent=2))
            return 0
        report = verify_run(run_dir, experiments[args.experiment_id])
        print(format_json(report))
        return 0 if report["status"] == "verified" else 3
    except ReproError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
