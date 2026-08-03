"""Verify a V13 rented-server transfer before starting training.

Run this on the rented server after uploading the files listed by
``tools/build_server_transfer_manifest.py``. It verifies that required files
exist, are nonempty, match recorded byte counts, and match SHA256 hashes when
the manifest includes them. It does not start training or modify artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_server_key(value: str) -> str:
    text = str(value).replace("\\", "/")
    if text != "/":
        text = text.rstrip("/")
    return text


def parse_path_map(values: list[str] | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"path map must use SERVER_PREFIX=LOCAL_PATH: {value}")
        server_prefix, local_path = value.split("=", 1)
        mapping[normalize_server_key(server_prefix)] = Path(local_path)
    return mapping


def resolve_server_path(server_path: str, path_map: dict[str, str | Path] | None = None) -> Path:
    normalized = normalize_server_key(server_path)
    mapping = {
        normalize_server_key(prefix): Path(local_root)
        for prefix, local_root in (path_map or {}).items()
    }
    for prefix in sorted(mapping, key=len, reverse=True):
        if normalized == prefix:
            return mapping[prefix]
        if normalized.startswith(prefix + "/"):
            suffix = normalized[len(prefix) + 1 :]
            return mapping[prefix].joinpath(*suffix.split("/"))
    return Path(normalized)


def verify_entry(entry: dict[str, Any], path_map: dict[str, str | Path] | None = None) -> dict[str, Any]:
    server_path = str(entry.get("server_path", ""))
    resolved = resolve_server_path(server_path, path_map)
    exists = resolved.exists()
    is_file = resolved.is_file()
    actual_bytes = int(resolved.stat().st_size) if exists and is_file else 0
    expected_bytes = entry.get("bytes")
    expected_int = int(expected_bytes) if expected_bytes is not None else None
    expected_sha = entry.get("sha256")
    actual_sha = sha256_file(resolved) if exists and is_file and expected_sha else None
    issues: list[str] = []

    if not exists:
        issues.append("missing")
    elif not is_file:
        issues.append("not_file")
    else:
        if actual_bytes <= 0:
            issues.append("empty")
        if expected_int is not None and actual_bytes != expected_int:
            issues.append("bytes_mismatch")
        if expected_sha and actual_sha != expected_sha:
            issues.append("sha256_mismatch")

    return {
        "label": entry.get("label"),
        "transfer_group": entry.get("transfer_group"),
        "server_path": server_path,
        "resolved_path": str(resolved),
        "exists": bool(exists),
        "is_file": bool(is_file),
        "expected_bytes": expected_int,
        "actual_bytes": actual_bytes,
        "bytes_match": bool(expected_int is None or actual_bytes == expected_int),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_match": None if not expected_sha else bool(actual_sha == expected_sha),
        "ok": len(issues) == 0,
        "issues": issues,
    }


def directory_nonempty(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def verify_data_requirement(
    requirement: dict[str, Any],
    path_map: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    server_path = str(requirement.get("server_path", ""))
    resolved = resolve_server_path(server_path, path_map)
    exists = resolved.exists()
    is_dir = resolved.is_dir()
    is_file = resolved.is_file()
    nonempty = bool(
        (is_dir and directory_nonempty(resolved))
        or (is_file and resolved.stat().st_size > 0)
    )
    issues: list[str] = []
    if not exists:
        issues.append("missing")
    elif not nonempty:
        issues.append("empty")
    return {
        "label": requirement.get("label"),
        "server_path": server_path,
        "resolved_path": str(resolved),
        "exists": bool(exists),
        "is_dir": bool(is_dir),
        "nonempty": bool(nonempty),
        "ok": len(issues) == 0,
        "issues": issues,
    }


def verify_transfer_manifest(
    manifest: dict[str, Any],
    path_map: dict[str, str | Path] | None = None,
    check_data_requirements: bool = True,
) -> dict[str, Any]:
    entries = [verify_entry(entry, path_map) for entry in manifest.get("entries", [])]
    data_requirements = (
        [
            verify_data_requirement(requirement, path_map)
            for requirement in manifest.get("server_data_requirements", [])
        ]
        if check_data_requirements
        else []
    )
    entries_ok = sum(1 for entry in entries if entry["ok"])
    data_ok = sum(1 for item in data_requirements if item["ok"])
    all_ok = entries_ok == len(entries) and data_ok == len(data_requirements)
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY_FOR_SERVER_RUN" if all_ok else "TRANSFER_VERIFICATION_FAILED",
        "summary": {
            "entries_total": len(entries),
            "entries_ok": entries_ok,
            "entries_failed": len(entries) - entries_ok,
            "data_requirements_total": len(data_requirements),
            "data_requirements_ok": data_ok,
            "data_requirements_failed": len(data_requirements) - data_ok,
        },
        "entries": entries,
        "server_data_requirements": data_requirements,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V13 Server Transfer Verification",
        "",
        f"- Created: {payload['created']}",
        f"- Status: `{payload['status']}`",
        f"- Entries: {payload['summary']['entries_ok']}/{payload['summary']['entries_total']} ok",
        "- Data requirements: "
        f"{payload['summary']['data_requirements_ok']}/{payload['summary']['data_requirements_total']} ok",
        "",
        "## Failed Entries",
        "",
    ]
    failed_entries = [entry for entry in payload["entries"] if not entry["ok"]]
    if failed_entries:
        for entry in failed_entries:
            lines.append(f"- `{entry['label']}` at `{entry['server_path']}`: {', '.join(entry['issues'])}")
    else:
        lines.append("- None")
    lines.extend(["", "## Failed Data Requirements", ""])
    failed_data = [item for item in payload["server_data_requirements"] if not item["ok"]]
    if failed_data:
        for item in failed_data:
            lines.append(f"- `{item['label']}` at `{item['server_path']}`: {', '.join(item['issues'])}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a V13 server transfer manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("local_reports/v13_server_transfer_manifest_20260706.json"),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON verification report.")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown verification report.")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional SERVER_PREFIX=LOCAL_PATH mapping for local dry-runs.",
    )
    parser.add_argument(
        "--skip-data-requirements",
        action="store_true",
        help="Only verify transferred files, not external data directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify_transfer_manifest(
        manifest,
        path_map=parse_path_map(args.path_map),
        check_data_requirements=not args.skip_data_requirements,
    )
    if args.output:
        write_json(args.output, report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "READY_FOR_SERVER_RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
