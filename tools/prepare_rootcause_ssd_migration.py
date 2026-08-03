"""Prepare or execute migration of the complex-curved rootcause suite to SSD.

The migration is intentionally non-destructive: it creates/regenerates a
portable suite on the destination drive, copies existing experiment artifacts,
and optionally copies referenced model inputs and parsed archives. It never
deletes source or destination files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SUITE = REPO_ROOT / "local_runs" / "complex_curved_rootcause_suite_20260715"
DEFAULT_DEST_ROOT = Path(r"E:\V13_rootcause_20260715")
SUITE_NAME = "complex_curved_rootcause_suite_20260715"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "files": 0, "dirs": 0}
    if path.is_file():
        return {"exists": True, "bytes": int(path.stat().st_size), "files": 1, "dirs": 0}
    total_bytes = 0
    files = 0
    dirs = 0
    for item in path.rglob("*"):
        if item.is_dir():
            dirs += 1
        elif item.is_file():
            files += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
    return {"exists": True, "bytes": int(total_bytes), "files": int(files), "dirs": int(dirs)}


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def default_manifest_path(suite_root: Path) -> Path:
    return suite_root / "ssd_migration_plan.json"


def default_commands_path(suite_root: Path) -> Path:
    return suite_root / "ssd_migration_commands.md"


def load_suite_config(suite_root: Path) -> dict[str, Any]:
    config_path = suite_root / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing suite config: {config_path}")
    return read_json(config_path)


def reference_dest(dest_root: Path, source: Path) -> Path:
    source = Path(source)
    anchor_parts = list(source.parts)
    if "ABC" in anchor_parts:
        idx = anchor_parts.index("ABC")
        rel = Path(*anchor_parts[idx:])
    else:
        rel = Path("reference_inputs") / source.name
    return dest_root / "reference_inputs" / rel


def build_items(
    suite_root: Path,
    dest_root: Path,
    *,
    copy_reference_models: bool,
    copy_archives: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite_root = Path(suite_root)
    dest_root = Path(dest_root)
    dest_suite = dest_root / SUITE_NAME
    config = load_suite_config(suite_root)

    items: list[dict[str, Any]] = [
        {
            "label": "experiments",
            "kind": "directory",
            "required": True,
            "source": str(suite_root / "experiments"),
            "destination": str(dest_suite / "experiments"),
            "reason": "existing diagnostic outputs, smoke models, patch shards, and same-data BrepARG inputs",
        },
        {
            "label": "rootcause_readout",
            "kind": "file_glob",
            "required": False,
            "source": str(suite_root / "rootcause_readout_20260715.md"),
            "destination": str(dest_suite / "rootcause_readout_20260715.md"),
            "reason": "human-readable root-cause readout",
        },
        {
            "label": "requirement_audit",
            "kind": "file",
            "required": False,
            "source": str(suite_root / "requirement_audit_20260715.md"),
            "destination": str(dest_suite / "requirement_audit_20260715.md"),
            "reason": "requirement-by-requirement evidence audit for the active root-cause goal",
        },
        {
            "label": "fsq_capacity_completion_handoff",
            "kind": "file",
            "required": False,
            "source": str(suite_root / "fsq_capacity_completion_handoff_20260715.md"),
            "destination": str(dest_suite / "fsq_capacity_completion_handoff_20260715.md"),
            "reason": "safe commands for monitoring and evaluating the active FSQ capacity run",
        },
        {
            "label": "partial_epoch5_fsq_capacity_readout",
            "kind": "file",
            "required": False,
            "source": str(suite_root / "partial_epoch5_fsq_capacity_readout_20260715.md"),
            "destination": str(dest_suite / "partial_epoch5_fsq_capacity_readout_20260715.md"),
            "reason": "early non-conclusive FSQ capacity checkpoint readout",
        },
        {
            "label": "suite_status_json",
            "kind": "file",
            "required": False,
            "source": str(suite_root / "suite_status.json"),
            "destination": str(dest_suite / "suite_status.json"),
            "reason": "latest local suite audit",
        },
        {
            "label": "suite_status_md",
            "kind": "file",
            "required": False,
            "source": str(suite_root / "suite_status.md"),
            "destination": str(dest_suite / "suite_status.md"),
            "reason": "latest local suite audit markdown",
        },
    ]

    for status_path in sorted(suite_root.glob("current_status_answer_*.md")):
        items.append(
            {
                "label": "current_status_answer",
                "kind": "file",
                "required": False,
                "source": str(status_path),
                "destination": str(dest_suite / status_path.name),
                "reason": "Chinese status answer explaining completed evidence, FSQ training rationale, and next steps",
            }
        )

    sequence = Path(config["sequence_path"])
    vqvae = Path(config["vqvae_checkpoint"])
    ar = Path(config["ar_checkpoint"])
    archive_root = Path(config["archive_root"])

    migrated_paths = {
        "sequence_path": sequence,
        "vqvae_checkpoint": vqvae,
        "ar_checkpoint": ar,
        "archive_root": archive_root,
    }

    if copy_reference_models:
        for label, src in (
            ("reference_sequence", sequence),
            ("reference_vqvae", vqvae),
            ("reference_ar", ar),
        ):
            dst = reference_dest(dest_root, src)
            migrated_paths[
                {
                    "reference_sequence": "sequence_path",
                    "reference_vqvae": "vqvae_checkpoint",
                    "reference_ar": "ar_checkpoint",
                }[label]
            ] = dst
            items.append(
                {
                    "label": label,
                    "kind": "file",
                    "required": True,
                    "source": str(src),
                    "destination": str(dst),
                    "reason": "reference input used by regenerated SSD suite scripts",
                }
            )

    if copy_archives:
        dst = reference_dest(dest_root, archive_root)
        migrated_paths["archive_root"] = dst
        items.append(
            {
                "label": "parsed_archives",
                "kind": "directory",
                "required": True,
                "source": str(archive_root),
                "destination": str(dst),
                "reason": "parsed zip archives needed to rerun complex-curved diagnostics and larger same-data prep",
            }
        )

    regenerated_config = {
        "python_exe": config["python_exe"],
        "sequence_path": str(migrated_paths["sequence_path"]),
        "vqvae_checkpoint": str(migrated_paths["vqvae_checkpoint"]),
        "ar_checkpoint": str(migrated_paths["ar_checkpoint"]),
        "archive_root": str(migrated_paths["archive_root"]),
    }
    return items, regenerated_config


def enrich_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        src = Path(item["source"])
        stats = file_stats(src)
        row = dict(item)
        row["source_exists"] = bool(stats["exists"])
        row["source_bytes"] = int(stats["bytes"])
        row["source_files"] = int(stats["files"])
        row["source_dirs"] = int(stats["dirs"])
        row["ready"] = bool(stats["exists"]) or not bool(item.get("required"))
        enriched.append(row)
    return enriched


def run_migration(
    *,
    suite_root: Path,
    dest_root: Path,
    copy_reference_models: bool,
    copy_archives: bool,
    execute: bool,
) -> dict[str, Any]:
    from tools.prepare_complex_curved_control_workspace import prepare_workspace

    suite_root = Path(suite_root)
    dest_root = Path(dest_root)
    dest_suite = dest_root / SUITE_NAME
    items, regenerated = build_items(
        suite_root,
        dest_root,
        copy_reference_models=copy_reference_models,
        copy_archives=copy_archives,
    )
    items = enrich_items(items)
    missing_required = [item for item in items if item["required"] and not item["source_exists"]]

    if execute and missing_required:
        labels = ", ".join(item["label"] for item in missing_required)
        raise FileNotFoundError(f"Missing required migration sources: {labels}")

    copied: list[dict[str, Any]] = []
    if execute:
        dest_root.mkdir(parents=True, exist_ok=True)
        prepare_workspace(
            output_dir=dest_suite,
            python_exe=regenerated["python_exe"],
            sequence_path=Path(regenerated["sequence_path"]),
            vqvae_checkpoint=Path(regenerated["vqvae_checkpoint"]),
            ar_checkpoint=Path(regenerated["ar_checkpoint"]),
            archive_root=Path(regenerated["archive_root"]),
        )
        for item in items:
            src = Path(item["source"])
            dst = Path(item["destination"])
            if not src.exists():
                continue
            copy_path(src, dst)
            copied.append({"label": item["label"], "destination": str(dst)})

    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "EXECUTED" if execute else "DRY_RUN",
        "suite_root": str(suite_root),
        "dest_root": str(dest_root),
        "dest_suite": str(dest_suite),
        "copy_reference_models": bool(copy_reference_models),
        "copy_archives": bool(copy_archives),
        "regenerated_config": regenerated,
        "items": items,
        "missing_required": missing_required,
        "copied": copied,
        "ready_to_execute": not missing_required,
    }
    return payload


def render_commands(payload: dict[str, Any], python_exe: str) -> str:
    suite = payload["suite_root"]
    dest = payload["dest_root"]
    archive_flag = " --copy-archives" if payload["copy_archives"] else ""
    model_flag = " --copy-reference-models" if payload["copy_reference_models"] else ""
    lines = [
        "# Rootcause Suite SSD Migration",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Destination suite: `{payload['dest_suite']}`",
        f"- Ready to execute: `{str(payload['ready_to_execute']).lower()}`",
        f"- Copy parsed archives: `{str(payload['copy_archives']).lower()}`",
        "",
        "Parsed archives are not required to continue from already-copied experiment artifacts and patch shards. Add `--copy-archives` if you want the SSD copy to rerun zip-backed complex-curved diagnostics or prepare larger same-data splits without reading the original workspace.",
        "",
        "## Dry Run",
        "",
        "```powershell",
        f"& '{python_exe}' tools\\prepare_rootcause_ssd_migration.py --suite-root '{suite}' --dest-root '{dest}'{model_flag}{archive_flag}",
        "```",
        "",
        "## Execute Copy",
        "",
        "```powershell",
        f"& '{python_exe}' tools\\prepare_rootcause_ssd_migration.py --suite-root '{suite}' --dest-root '{dest}'{model_flag}{archive_flag} --execute",
        "```",
        "",
        "## Continue Experiments From SSD",
        "",
        "```powershell",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\05_audit_suite_status.ps1'",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\01a_preflight_fsq_capacity_candidate.ps1'",
        "# If the copied suite represents an active/moving run, inspect status first and do not start a duplicate resume.",
        f"Get-Content '{payload['dest_suite']}\\experiments\\01a_train_fsq_capacity_candidate\\logs\\fsq_capacity_watch_then_eval.log' -Tail 20 -Wait",
        "# Resume only if no active FSQ capacity process is alive and train_report.json is still missing:",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\01a_resume_fsq_capacity_candidate.ps1'",
        "# If no partial checkpoint/history exists yet, start a fresh capacity run instead:",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\01a_train_fsq_capacity_candidate.ps1'",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\03b_preflight_breparg_same_data_fallback.ps1'",
        "# Run this after the FSQ capacity GPU job is done, not concurrently on the same GPU:",
        f"powershell -ExecutionPolicy Bypass -File '{payload['dest_suite']}\\scripts\\03b_breparg_same_data_training_fallback.ps1'",
        "```",
        "",
        "## Items",
        "",
        "| label | exists | files | GB | destination |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in payload["items"]:
        gb = item["source_bytes"] / 1e9
        lines.append(
            f"| `{item['label']}` | {str(item['source_exists']).lower()} | {item['source_files']} | {gb:.3f} | `{item['destination']}` |"
        )
    if payload["missing_required"]:
        lines.extend(["", "## Missing Required Sources", ""])
        lines.extend(f"- `{item['label']}`: `{item['source']}`" for item in payload["missing_required"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--copy-reference-models", action="store_true")
    parser.add_argument("--copy-archives", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commands-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or default_manifest_path(args.suite_root)
    commands_path = args.commands_output or default_commands_path(args.suite_root)
    payload = run_migration(
        suite_root=args.suite_root,
        dest_root=args.dest_root,
        copy_reference_models=args.copy_reference_models,
        copy_archives=args.copy_archives,
        execute=args.execute,
    )
    write_json(manifest_path, payload)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text(render_commands(payload, payload["regenerated_config"]["python_exe"]), encoding="utf-8")
    if args.execute:
        dest_manifest = Path(payload["dest_suite"]) / "ssd_migration_manifest.json"
        write_json(dest_manifest, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["ready_to_execute"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
