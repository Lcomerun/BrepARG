"""Verify copied-back VQ-VAE recovery artifacts before deleting a server.

The server writes ``copy_back_manifest.json`` after a VQ-VAE recovery run. This
local verifier re-checks the actual copied files in the paper/workspace machine
and decides whether the checkpoint can move to source-path sequence rebuild or
must stay held for missing artifacts or failed promotion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence


PROMOTE_DECISION = "promote_for_ar_rebuild"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_path_maps(items: Sequence[str]) -> dict[str, Path]:
    maps: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"path map must be SERVER_PREFIX=LOCAL_PREFIX: {item}")
        server, local = item.split("=", 1)
        maps[server.rstrip("/\\")] = Path(local)
    return maps


def mapped_path(server_path: str, path_maps: dict[str, Path]) -> Path | None:
    normalized = server_path.replace("\\", "/")
    for server_prefix, local_prefix in sorted(path_maps.items(), key=lambda pair: len(pair[0]), reverse=True):
        prefix = server_prefix.replace("\\", "/").rstrip("/")
        if normalized == prefix:
            return local_prefix
        if normalized.startswith(prefix + "/"):
            return local_prefix / normalized[len(prefix) + 1 :]
    return None


def resolve_artifact_path(artifact: dict[str, Any], repo_root: Path, path_maps: dict[str, Path]) -> Path:
    relative = artifact.get("relative_path")
    if relative and not str(relative).startswith(("/", "\\")):
        return repo_root / str(relative)
    raw_path = str(artifact.get("path") or "")
    mapped = mapped_path(raw_path, path_maps)
    if mapped is not None:
        return mapped
    return Path(raw_path)


def artifact_status(artifact: dict[str, Any], repo_root: Path, path_maps: dict[str, Path]) -> dict[str, Any]:
    local_path = resolve_artifact_path(artifact, repo_root, path_maps)
    exists = local_path.exists()
    is_file = local_path.is_file()
    return {
        "label": artifact.get("label"),
        "kind": artifact.get("kind"),
        "required": bool(artifact.get("required", True)),
        "manifest_path": artifact.get("path"),
        "relative_path": artifact.get("relative_path"),
        "local_path": str(local_path),
        "exists": bool(exists),
        "bytes": int(local_path.stat().st_size) if exists and is_file else 0,
        "ok": bool(exists and (not is_file or local_path.stat().st_size > 0)),
    }


def status_for(copyback_complete: bool, promotion_decision: str | None) -> tuple[str, str]:
    if not copyback_complete:
        return "COPYBACK_INCOMPLETE", "copy_missing_artifacts_before_deleting_server"
    if promotion_decision == PROMOTE_DECISION:
        return "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD", "run_source_path_sequence_rebuild"
    return "HOLD_VQVAE_RECOVERY_FOR_FAILURE_ANALYSIS", "inspect_benchmark_and_continue_vqvae_or_decoder_diagnosis"


def verify_vqvae_copyback(
    *,
    manifest_path: str | Path,
    repo_root: str | Path,
    path_maps: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root)
    maps = parse_path_maps(path_maps or [])
    manifest = load_json(manifest_path)
    artifacts = [
        artifact_status(item, repo_root, maps)
        for item in list(manifest.get("artifacts") or [])
    ]
    missing_required = [item for item in artifacts if item["required"] and not item["ok"]]
    copyback_complete = len(missing_required) == 0
    promotion_decision = manifest.get("promotion_decision")
    status, next_action = status_for(copyback_complete, promotion_decision)
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "next_action": next_action,
        "manifest_path": str(manifest_path),
        "repo_root": str(repo_root),
        "copyback_complete": bool(copyback_complete),
        "promotion_decision": promotion_decision,
        "promotion_reasons": list(manifest.get("promotion_reasons") or []),
        "artifacts_total": len(artifacts),
        "required_total": sum(1 for item in artifacts if item["required"]),
        "required_ok": sum(1 for item in artifacts if item["required"] and item["ok"]),
        "missing_required": missing_required,
        "artifacts": artifacts,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 VQ-VAE Copy-Back Verification",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Next action: `{payload['next_action']}`",
        f"- Promotion decision: `{payload.get('promotion_decision')}`",
        f"- Required artifacts: {payload['required_ok']}/{payload['required_total']} ok",
        "",
    ]
    if payload["missing_required"]:
        lines.extend(["## Missing Required Artifacts", ""])
        for item in payload["missing_required"]:
            lines.append(f"- {item['label']}: `{item['local_path']}`")
        lines.append("")
    lines.extend(["## Artifacts", "", "| Label | Required | OK | Local path |", "|---|---:|---:|---|"])
    for item in payload["artifacts"]:
        lines.append(f"| {item['label']} | {item['required']} | {item['ok']} | `{item['local_path']}` |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify copied-back VQ-VAE recovery artifacts.")
    parser.add_argument("--manifest", type=Path, required=True, help="Returned copy_back_manifest.json.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--path-map", action="append", default=[], help="Optional SERVER_PREFIX=LOCAL_PREFIX path map.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_vqvae_copyback(
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        path_maps=args.path_map,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD" else 2


if __name__ == "__main__":
    raise SystemExit(main())
