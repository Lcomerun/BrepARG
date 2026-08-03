from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PackageBuildError(RuntimeError):
    """Raised when a package cannot be built without violating its contract."""


TEXT_EXTENSIONS = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_PACKAGE_EXTENSIONS = {
    ".7z",
    ".ckpt",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".stl",
    ".step",
    ".stp",
    ".tar",
    ".tgz",
    ".zip",
    ".zst",
}

SOURCE_ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "PLANS.md",
    "PROJECT_INDEX.md",
    "README.md",
    "e_drive_drop_20260717_0613.md",
    "environment.server.yml",
    "local_training_config.json",
    "recovery_status_20260717_1024.md",
    "recovery_status_20260717_1040.md",
    "recovery_status_20260717_1108.md",
}

SOURCE_TOP_LEVEL_DIRECTORIES = {
    ".agents",
    "BrepARG",
    "breparg_improvements",
    "docs",
    "papers",
    "plans",
    "reproducibility",
    "tests",
    "tools",
}

SOURCE_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "latexmk-cache",
    "node_modules",
    "repro_outputs",
}

SOURCE_MAX_FILE_BYTES = 8 * 1024 * 1024
STAGE_MAX_FILE_BYTES = 16 * 1024 * 1024
STAGE_MAX_TOTAL_BYTES = 100 * 1024 * 1024
HISTORY_PACKAGE_RELATIVE_MAX = 120

EVENT_PATTERN = re.compile(
    r"(error|exception|traceback|failed|failure|out of memory|\boom\b|\bnan\b|"
    r"nonfinite|non-finite|early stop|saved best|\bdone\b|val[_ =]|train[_ =]|"
    r"epoch|resume|checkpoint)",
    re.IGNORECASE,
)

HOST_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9_])[CDE]:[\\/]"),
    re.compile(r"/root/autodl-tmp(?:/|\b)"),
)

CONTROL_PLANE_PREFIXES = {
    "artifact_specs",
    "configs",
    "environments",
    "experiments",
    "launchers",
    "tests",
}

SECRET_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}

SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
BUILTIN_PLACEHOLDERS = {"PACKAGE_ROOT", "RUN_DIR", "SOURCE_ROOT"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_stable_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def _relative_posix(path: Path) -> str:
    return path.as_posix().lstrip("./")


def _safe_catalog_relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageBuildError(f"{field} must be a non-empty relative path")
    normalized = Path(value.replace("\\", "/"))
    if (
        value.startswith(("/", "\\"))
        or normalized.is_absolute()
        or ".." in normalized.parts
    ):
        raise PackageBuildError(f"unsafe {field}: {value}")
    if re.match(r"^[A-Za-z]:", value):
        raise PackageBuildError(f"unsafe {field}: {value}")
    return normalized


def _placeholders(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(PLACEHOLDER_PATTERN.findall(value))
    elif isinstance(value, list):
        for item in value:
            found.update(_placeholders(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_placeholders(item))
    return found


def should_include_current_source(relative: Path, size_bytes: int) -> bool:
    """Return whether a working-tree file belongs in the lightweight source layer."""

    normalized = Path(_relative_posix(relative))
    parts = normalized.parts
    if not parts:
        return False
    if any(part in SOURCE_EXCLUDED_PARTS or part.startswith("._") for part in parts):
        return False
    if normalized.suffix.lower() in FORBIDDEN_PACKAGE_EXTENSIONS:
        return False
    if normalized.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    if size_bytes > SOURCE_MAX_FILE_BYTES:
        return False
    if len(parts) == 1:
        return normalized.name in SOURCE_ROOT_FILES or normalized.suffix.lower() in {
            ".md",
            ".txt",
        }
    if parts[0] not in SOURCE_TOP_LEVEL_DIRECTORIES:
        return False
    return True


def write_checksum_manifest(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    checksum_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
        newline="\n",
    )
    return checksum_path


def _zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    value = datetime.fromtimestamp(epoch, tz=timezone.utc)
    if value.year < 1980:
        value = datetime(1980, 1, 1, tzinfo=timezone.utc)
    return (value.year, value.month, value.day, value.hour, value.minute, value.second)


def write_deterministic_zip(source_root: Path, zip_path: Path, epoch: int) -> str:
    """Write a byte-stable ZIP containing source_root as its top-level directory."""

    if not source_root.is_dir():
        raise PackageBuildError(f"package source root is missing: {source_root}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _zip_datetime(epoch)
    root_name = source_root.name
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    temporary = zip_path.with_name(f".{zip_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for path in files:
                relative = path.relative_to(source_root).as_posix()
                entry = zipfile.ZipInfo(f"{root_name}/{relative}", date_time=timestamp)
                entry.create_system = 3
                mode = 0o755 if path.suffix.lower() == ".sh" else 0o644
                entry.external_attr = (stat.S_IFREG | mode) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.flag_bits |= 0x800
                archive.writestr(
                    entry,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(zip_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(zip_path)


def _decode_text_for_evidence(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _excerpt_lines(lines: list[str], *, first: bool, limit: int = 40) -> str:
    selected = lines[:limit] if first else lines[-limit:]
    return "\n".join(selected) + ("\n" if selected else "")


def copy_or_summarize_history_file(
    source: Path,
    target: Path,
    *,
    original_label: str,
    direct_copy_limit: int = 256 * 1024,
) -> dict[str, Any]:
    """Copy a small text record or write a hash/excerpt record for a large one."""

    if not source.is_file():
        raise PackageBuildError(f"history source is missing: {source}")
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) <= direct_copy_limit:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {
            "original_path": original_label,
            "packaged_path": str(target),
            "mode": "copied",
            "size_bytes": len(data),
            "sha256": digest,
        }
    text = _decode_text_for_evidence(data)
    lines = text.splitlines()
    event_lines: list[str] = []
    for index, line in enumerate(lines, start=1):
        if EVENT_PATTERN.search(line):
            event_lines.append(f"{index}: {line[:1000]}")
            if len(event_lines) >= 200:
                break
    evidence = {
        "schema_version": 1,
        "record_type": "large_text_evidence_summary",
        "original_path": original_label,
        "size_bytes": len(data),
        "sha256": digest,
        "line_count": len(lines),
        "head_excerpt": _excerpt_lines(lines, first=True),
        "tail_excerpt": _excerpt_lines(lines, first=False),
        "event_lines": event_lines,
        "note": (
            "The complete source record is intentionally excluded from the lightweight "
            "package; identity and diagnostic excerpts are preserved here."
        ),
    }
    evidence_target = target.with_suffix(target.suffix + ".evidence.json")
    write_stable_json(evidence_target, evidence)
    return {
        "original_path": original_label,
        "packaged_path": str(evidence_target),
        "mode": "evidence_summary",
        "size_bytes": len(data),
        "sha256": digest,
    }


def _history_target_path(
    package_history_root: Path,
    destination_root: Path,
    relative: Path,
) -> tuple[Path, bool]:
    direct = package_history_root / destination_root / relative
    package_relative = direct.relative_to(package_history_root.parent).as_posix()
    if len(package_relative) <= HISTORY_PACKAGE_RELATIVE_MAX:
        return direct, False

    source_label = (destination_root / relative).as_posix()
    digest = hashlib.sha256(source_label.encode("utf-8")).hexdigest()[:20]
    suffix = relative.suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.stem).strip("._-")
    if not stem:
        stem = "record"
    stem = stem[:48]
    target = (
        package_history_root
        / "_compacted"
        / f"{digest}_{stem}{suffix}"
    )
    return target, True


def _git(repo: Path, *args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if binary
            else completed.stderr
        )
        raise PackageBuildError(
            f"git {' '.join(args)} failed in {repo}: {stderr.strip()}"
        )
    return completed.stdout


def export_clean_head(repo: Path, commit: str, target: Path) -> dict[str, Any]:
    """Export committed file bytes without including Git metadata or dirty files."""

    listing = _git(repo, "ls-tree", "-r", "-z", "--name-only", commit, binary=True)
    assert isinstance(listing, bytes)
    names = [name.decode("utf-8") for name in listing.split(b"\0") if name]
    if target.exists():
        raise PackageBuildError(f"clean export target already exists: {target}")
    target.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        relative = Path(name)
        if any(part == ".git" or part.startswith("._") for part in relative.parts):
            continue
        content = _git(repo, "show", f"{commit}:{name}", binary=True)
        assert isinstance(content, bytes)
        if relative.suffix.lower() in FORBIDDEN_PACKAGE_EXTENSIONS:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "commit": commit,
        "file_count": len(records),
        "files": records,
    }


def _is_secret_name(relative: Path) -> bool:
    lowered = relative.name.lower()
    if lowered in SECRET_FILE_NAMES or relative.suffix.lower() in SECRET_SUFFIXES:
        return True
    return lowered.startswith(".env.") and lowered != ".env.example"


def _control_plane_file(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return relative.name in {
            "BUILD_REPORT.md",
            "PACKAGE_MANIFEST.json",
            "START_HERE.md",
            "reproduce.sh",
        }
    return relative.parts[0] in CONTROL_PLANE_PREFIXES


def _scan_host_paths(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    for pattern in HOST_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def validate_stage(stage_root: Path) -> dict[str, Any]:
    if not stage_root.is_dir():
        raise PackageBuildError(f"stage root is missing: {stage_root}")
    files = sorted(path for path in stage_root.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(stage_root)
        if ".git" in relative.parts:
            raise PackageBuildError(f"nested .git is forbidden: {relative.as_posix()}")
        if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
            raise PackageBuildError(f"cache directory is forbidden: {relative.as_posix()}")
        if relative.suffix.lower() in FORBIDDEN_PACKAGE_EXTENSIONS:
            raise PackageBuildError(
                f"forbidden packaged extension: {relative.as_posix()}"
            )
        if _is_secret_name(relative):
            raise PackageBuildError(f"secret-shaped file name: {relative.as_posix()}")
        size = path.stat().st_size
        if size > STAGE_MAX_FILE_BYTES:
            raise PackageBuildError(
                f"packaged file exceeds {STAGE_MAX_FILE_BYTES} bytes: "
                f"{relative.as_posix()} ({size})"
            )
        total_bytes += size
        if _control_plane_file(relative):
            match = _scan_host_paths(path)
            if match:
                raise PackageBuildError(
                    f"host-specific runtime path in {relative.as_posix()}: {match}"
                )
    if total_bytes > STAGE_MAX_TOTAL_BYTES:
        raise PackageBuildError(
            f"package payload exceeds {STAGE_MAX_TOTAL_BYTES} bytes: {total_bytes}"
        )
    return {
        "status": "ok",
        "file_count": len(files),
        "total_size_bytes": total_bytes,
    }


def _manifest_for_tree(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


def _copy_current_source(repo_root: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        raise PackageBuildError(f"current source target already exists: {target}")
    target.mkdir(parents=True)

    candidates: list[Path] = []
    for path in repo_root.iterdir():
        if path.is_file():
            candidates.append(path)
    for directory_name in sorted(SOURCE_TOP_LEVEL_DIRECTORIES):
        directory = repo_root / directory_name
        if not directory.is_dir():
            continue
        for current, directory_names, file_names in os.walk(directory):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in SOURCE_EXCLUDED_PARTS
                and name != ".git"
                and not name.startswith("._")
            )
            current_path = Path(current)
            for file_name in sorted(file_names):
                candidates.append(current_path / file_name)

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in sorted(set(candidates)):
        if source.is_symlink() or not source.is_file():
            continue
        relative = source.relative_to(repo_root)
        size = source.stat().st_size
        if not should_include_current_source(relative, size):
            excluded.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": size,
                    "reason": "source_filter",
                }
            )
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(destination),
            }
        )
    return {
        "schema_version": 1,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "excluded_candidates": excluded,
    }


def _copy_text_lf(source: Path, destination: Path) -> None:
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PackageBuildError(f"cannot read UTF-8 template {source}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        text.replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
        newline="\n",
    )


def _copy_template_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        return 0
    count = 0
    for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        relative = path.relative_to(source)
        if any(part in SOURCE_EXCLUDED_PARTS or part == ".git" for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_PACKAGE_EXTENSIONS:
            continue
        target = destination / relative
        if path.suffix.lower() in TEXT_EXTENSIONS:
            _copy_text_lf(path, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        count += 1
    return count


def _load_catalog(path: Path, list_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageBuildError(f"cannot load catalog {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get(list_key), list):
        raise PackageBuildError(f"catalog {path} must contain a {list_key} array")
    records = value[list_key]
    if not all(isinstance(record, dict) for record in records):
        raise PackageBuildError(f"catalog {path} contains a non-object record")
    ids: set[str] = set()
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_.-]*", record_id
        ):
            raise PackageBuildError(f"invalid id in {path}: {record_id!r}")
        if record_id in ids:
            raise PackageBuildError(f"duplicate id in {path}: {record_id}")
        ids.add(record_id)
    return value, records


def validate_catalogs(repo_root: Path) -> dict[str, Any]:
    """Validate source catalogs without staging artifacts or reading model data."""

    catalog_root = repo_root / "reproducibility" / "catalog"
    _, artifacts = _load_catalog(catalog_root / "artifacts.json", "artifacts")
    experiments_catalog, experiments = _load_catalog(
        catalog_root / "experiments.json", "experiments"
    )
    artifact_ids = {record["id"] for record in artifacts}
    path_variables = {record.get("path_var") for record in artifacts}
    path_variables.discard(None)
    categories = {"recommended", "baselines", "diagnostics", "historical_failed"}
    states = {"runnable", "documentary", "blocked_missing_evidence"}
    referenced: set[str] = set()
    for artifact in artifacts:
        artifact_type = artifact.get("type", "file")
        if artifact_type not in {"file", "directory"}:
            raise PackageBuildError(
                f"artifact {artifact['id']} has unsupported type: {artifact_type!r}"
            )
        path_var = artifact.get("path_var")
        if not isinstance(path_var, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", path_var
        ):
            raise PackageBuildError(
                f"artifact {artifact['id']} has invalid path_var: {path_var!r}"
            )
    for experiment in experiments:
        category = experiment.get("category")
        state = experiment.get("state")
        if category not in categories:
            raise PackageBuildError(
                f"experiment {experiment['id']} has invalid category: {category!r}"
            )
        if state not in states:
            raise PackageBuildError(
                f"experiment {experiment['id']} has invalid state: {state!r}"
            )
        required = experiment.get("required_artifacts", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise PackageBuildError(
                f"experiment {experiment['id']} has invalid required_artifacts"
            )
        unknown = sorted(set(required) - artifact_ids)
        if unknown:
            raise PackageBuildError(
                f"experiment {experiment['id']} references unknown artifacts: {unknown}"
            )
        referenced.update(required)
        bindings = experiment.get("artifact_bindings", {})
        if not isinstance(bindings, dict):
            raise PackageBuildError(
                f"experiment {experiment['id']} has invalid artifact_bindings"
            )
        unknown_bindings = sorted(set(bindings) - set(required))
        if unknown_bindings:
            raise PackageBuildError(
                f"experiment {experiment['id']} binds non-required artifacts: "
                f"{unknown_bindings}"
            )
        for artifact_id, target in bindings.items():
            _safe_catalog_relative_path(
                target,
                field=f"artifact binding for {experiment['id']}:{artifact_id}",
            )
        command = experiment.get("command")
        if state == "runnable" and (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise PackageBuildError(
                f"runnable experiment {experiment['id']} has no valid command array"
            )
        smoke_command = experiment.get("smoke_command")
        if smoke_command is not None and (
            not isinstance(smoke_command, list)
            or not smoke_command
            or not all(isinstance(item, str) and item for item in smoke_command)
        ):
            raise PackageBuildError(
                f"experiment {experiment['id']} has an invalid smoke_command"
            )
        expected_outputs = experiment.get("expected_outputs", [])
        if not isinstance(expected_outputs, list):
            raise PackageBuildError(
                f"experiment {experiment['id']} has invalid expected_outputs"
            )
        for output in expected_outputs:
            if not isinstance(output, dict):
                raise PackageBuildError(
                    f"experiment {experiment['id']} has a non-object expected output"
                )
            _safe_catalog_relative_path(
                output.get("path"),
                field=f"expected output for {experiment['id']}",
            )
        unknown_placeholders = sorted(
            _placeholders(experiment)
            - BUILTIN_PLACEHOLDERS
            - {str(value) for value in path_variables}
        )
        if unknown_placeholders:
            raise PackageBuildError(
                f"experiment {experiment['id']} has unknown placeholder(s): "
                f"{unknown_placeholders}"
            )
    coverage_rules = experiments_catalog.get("coverage_rules", [])
    if not isinstance(coverage_rules, list):
        raise PackageBuildError("experiment catalog coverage_rules must be an array")
    return {
        "status": "ok",
        "experiment_count": len(experiments),
        "artifact_count": len(artifacts),
        "referenced_artifact_count": len(referenced),
        "coverage_rule_count": len(coverage_rules),
    }


def _directory_inventory(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        count += 1
        total += size
    return count, total, digest.hexdigest()


def _expand_artifact_catalog(
    repo_root: Path,
    catalog_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    _, records = _load_catalog(catalog_path, "artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        source_value = record.pop("build_source_path", None)
        required_identity = bool(record.pop("required_identity", False))
        source: Path | None = None
        if isinstance(source_value, str) and source_value:
            source = Path(source_value)
            if not source.is_absolute():
                source = repo_root / source
        record["build_availability"] = "not_observed"
        if source is not None and source.exists():
            if record.get("type", "file") == "file":
                if not source.is_file():
                    raise PackageBuildError(
                        f"artifact {record['id']} expected a file: {source}"
                    )
                record["size_bytes"] = source.stat().st_size
                record["sha256"] = sha256_file(source)
                record["verification_strength"] = "content_sha256"
            elif record.get("type") == "directory":
                if not source.is_dir():
                    raise PackageBuildError(
                        f"artifact {record['id']} expected a directory: {source}"
                    )
                count, total, inventory_hash = _directory_inventory(source)
                record["file_count"] = count
                record["total_size_bytes"] = total
                record["inventory_sha256"] = inventory_hash
                record["verification_strength"] = "name_size_inventory"
            else:
                raise PackageBuildError(
                    f"artifact {record['id']} has unsupported type: {record.get('type')}"
                )
            record["build_availability"] = "observed_and_identified"
        elif required_identity:
            raise PackageBuildError(
                f"required artifact identity source is missing for {record['id']}: {source}"
            )
        else:
            record.setdefault("verification_strength", "unresolved")
        write_stable_json(output_root / f"{record['id']}.json", record)
        summary.append(
            {
                "id": record["id"],
                "build_availability": record["build_availability"],
                "verification_strength": record["verification_strength"],
            }
        )
    return {"count": len(summary), "artifacts": summary}


def _expand_experiment_catalog(
    catalog_path: Path,
    output_root: Path,
    artifact_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog, records = _load_catalog(catalog_path, "experiments")
    categories = {"recommended", "baselines", "diagnostics", "historical_failed"}
    states = {"runnable", "documentary", "blocked_missing_evidence"}
    counts: dict[str, int] = {category: 0 for category in sorted(categories)}
    states_count: dict[str, int] = {state: 0 for state in sorted(states)}
    output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        category = record.get("category")
        state = record.get("state")
        if category not in categories:
            raise PackageBuildError(
                f"experiment {record['id']} has invalid category: {category!r}"
            )
        if state not in states:
            raise PackageBuildError(
                f"experiment {record['id']} has invalid state: {state!r}"
            )
        unknown = sorted(set(record.get("required_artifacts", [])) - artifact_ids)
        if unknown:
            raise PackageBuildError(
                f"experiment {record['id']} references unknown artifacts: {unknown}"
            )
        if state == "runnable" and not isinstance(record.get("command"), list):
            raise PackageBuildError(
                f"runnable experiment {record['id']} has no command array"
            )
        write_stable_json(output_root / category / f"{record['id']}.json", record)
        counts[category] += 1
        states_count[state] += 1
    return (
        {
            "count": len(records),
            "categories": counts,
            "states": states_count,
        },
        list(catalog.get("coverage_rules", [])),
    )


def _history_sources(
    repo_root: Path,
    additional_roots: list[Path],
) -> list[tuple[Path, str, Path]]:
    sources: list[tuple[Path, str, Path]] = []
    defaults = [
        (repo_root / "plans", "plans", Path("04_plans_and_decisions/plans")),
        (
            repo_root / "local_reports",
            "local_reports",
            Path("05_original_records/local_reports"),
        ),
        (
            repo_root / "local_runs",
            "local_runs",
            Path("05_original_records/local_runs"),
        ),
    ]
    seen: set[Path] = set()
    for root, label, destination in defaults:
        if root.is_dir():
            resolved = root.resolve()
            seen.add(resolved)
            sources.append((root, label, destination))
    for index, root in enumerate(additional_roots):
        if not root.is_dir() or root.resolve() in seen:
            continue
        seen.add(root.resolve())
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name).strip("_")
        if not safe_name:
            safe_name = f"external_{index:02d}"
        sources.append(
            (
                root,
                f"external:{safe_name}",
                Path(f"05_original_records/external/{safe_name}"),
            )
        )
    return sources


def _classify_coverage(relative_label: str, rules: list[dict[str, Any]]) -> str | None:
    for rule in rules:
        pattern = rule.get("pattern")
        classification = rule.get("classification")
        if (
            isinstance(pattern, str)
            and isinstance(classification, str)
            and fnmatch.fnmatch(relative_label, pattern)
        ):
            return classification
    return None


def _curate_history(
    repo_root: Path,
    package_history_root: Path,
    additional_roots: list[Path],
    coverage_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_suffixes = {
        ".csv",
        ".env",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".ps1",
        ".sh",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
    records: list[dict[str, Any]] = []
    unclassified: list[str] = []
    for source_root, label_root, destination_root in _history_sources(
        repo_root, additional_roots
    ):
        for source in sorted(
            candidate for candidate in source_root.rglob("*") if candidate.is_file()
        ):
            relative = source.relative_to(source_root)
            if (
                source.suffix.lower() not in allowed_suffixes
                or any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts)
                or source.name.startswith("._")
            ):
                continue
            coverage_label = (
                f"{label_root}/{relative.as_posix()}"
                if not label_root.startswith("external:")
                else f"external/{label_root.split(':', 1)[1]}/{relative.as_posix()}"
            )
            classification = _classify_coverage(coverage_label, coverage_rules)
            if classification is None:
                unclassified.append(coverage_label)
                continue
            target, path_compacted = _history_target_path(
                package_history_root, destination_root, relative
            )
            record = copy_or_summarize_history_file(
                source,
                target,
                original_label=coverage_label,
            )
            packaged = Path(record["packaged_path"])
            record["packaged_path"] = packaged.relative_to(
                package_history_root.parent
            ).as_posix()
            record["classification"] = classification
            record["path_compacted"] = path_compacted
            records.append(record)
    if unclassified:
        examples = ", ".join(unclassified[:10])
        raise PackageBuildError(
            f"history coverage has {len(unclassified)} unclassified file(s): {examples}"
        )
    copied = sum(record["mode"] == "copied" for record in records)
    summarized = sum(record["mode"] == "evidence_summary" for record in records)
    inventory = {
        "schema_version": 1,
        "record_count": len(records),
        "copied_count": copied,
        "evidence_summary_count": summarized,
        "unclassified_count": 0,
        "records": records,
    }
    write_stable_json(
        package_history_root / "08_evidence_index" / "history_inventory.json",
        inventory,
    )
    return inventory


def _run_git_optional(repo: Path, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": ["git", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _capture_provenance(
    repo_root: Path,
    provenance_root: Path,
    clean_commit: str,
    current_manifest: dict[str, Any],
    clean_manifest: dict[str, Any],
) -> dict[str, Any]:
    provenance_root.mkdir(parents=True, exist_ok=True)
    head = _run_git_optional(repo_root, ["rev-parse", "HEAD"])
    status = _run_git_optional(repo_root, ["status", "--short", "--untracked-files=all"])
    diff = _run_git_optional(repo_root, ["diff", "--binary", clean_commit])
    log = _run_git_optional(
        repo_root,
        ["log", "-10", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%s"],
    )
    (provenance_root / "outer_status.txt").write_text(
        status["stdout"], encoding="utf-8", newline="\n"
    )
    (provenance_root / "outer_diff_from_clean.patch").write_text(
        diff["stdout"], encoding="utf-8", newline="\n"
    )
    (provenance_root / "outer_recent_commits.tsv").write_text(
        log["stdout"] + ("\n" if log["stdout"] else ""),
        encoding="utf-8",
        newline="\n",
    )
    write_stable_json(provenance_root / "current_source_manifest.json", current_manifest)
    write_stable_json(provenance_root / "clean_source_manifest.json", clean_manifest)

    included_paths = {record["path"] for record in current_manifest["files"]}
    untracked_probe = _run_git_optional(
        repo_root, ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    untracked_files: list[dict[str, Any]] = []
    if untracked_probe["returncode"] == 0:
        for name in sorted(item for item in untracked_probe["stdout"].split("\0") if item):
            relative = Path(name)
            source = repo_root / relative
            if not source.is_file():
                continue
            normalized = relative.as_posix()
            included = normalized in included_paths
            untracked_files.append(
                {
                    "path": normalized,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                    "decision": (
                        "included_in_current_source"
                        if included
                        else "excluded_from_current_source"
                    ),
                    "reason": "source_filter_passed" if included else "source_filter",
                }
            )
    write_stable_json(
        provenance_root / "outer_untracked_manifest.json",
        {
            "schema_version": 1,
            "probe_returncode": untracked_probe["returncode"],
            "probe_stderr": untracked_probe["stderr"],
            "file_count": len(untracked_files),
            "included_count": sum(
                row["decision"] == "included_in_current_source"
                for row in untracked_files
            ),
            "excluded_count": sum(
                row["decision"] == "excluded_from_current_source"
                for row in untracked_files
            ),
            "files": untracked_files,
        },
    )

    breparg_source = repo_root / "BrepARG"
    if breparg_source.is_dir():
        packaged_breparg = {
            "schema_version": 1,
            **_manifest_for_tree(
                provenance_root.parent / "source" / "current" / "BrepARG"
            ),
        }
    else:
        packaged_breparg = {
            "schema_version": 1,
            "file_count": 0,
            "total_size_bytes": 0,
            "files": [],
        }
    write_stable_json(
        provenance_root / "breparg_source_manifest.json", packaged_breparg
    )

    nested_root = repo_root / "BrepARG"
    nested: dict[str, Any] = {"available": nested_root.is_dir()}
    if nested_root.is_dir() and (nested_root / ".git").exists():
        nested_head = _run_git_optional(nested_root, ["rev-parse", "HEAD"])
        nested_status = _run_git_optional(
            nested_root, ["status", "--short", "--untracked-files=all"]
        )
        nested_diff = _run_git_optional(nested_root, ["diff", "--binary", "HEAD"])
        nested.update(
            {
                "commit": nested_head["stdout"].strip() or None,
                "head_probe_returncode": nested_head["returncode"],
                "status_returncode": nested_status["returncode"],
                "diff_returncode": nested_diff["returncode"],
                "stderr": "\n".join(
                    value
                    for value in (
                        nested_head["stderr"],
                        nested_status["stderr"],
                        nested_diff["stderr"],
                    )
                    if value
                ),
            }
        )
        (provenance_root / "breparg_status.txt").write_text(
            nested_status["stdout"], encoding="utf-8", newline="\n"
        )
        (provenance_root / "breparg_worktree.patch").write_text(
            nested_diff["stdout"], encoding="utf-8", newline="\n"
        )
    write_stable_json(provenance_root / "breparg_provenance.json", nested)
    record = {
        "schema_version": 1,
        "outer_head": head["stdout"].strip() or None,
        "clean_reference_commit": clean_commit,
        "outer_status_returncode": status["returncode"],
        "outer_diff_returncode": diff["returncode"],
        "nested_breparg": nested,
    }
    write_stable_json(provenance_root / "source_provenance.json", record)
    return record


def _verify_checksum_manifest(package_root: Path) -> dict[str, Any]:
    path = package_root / "SHA256SUMS"
    if not path.is_file():
        raise PackageBuildError("SHA256SUMS is missing from extracted package")
    checked = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise PackageBuildError(f"invalid SHA256SUMS line {line_number}")
        expected, value = match.groups()
        relative = Path(value.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise PackageBuildError(f"unsafe SHA256SUMS path: {value}")
        target = package_root / relative
        if not target.is_file():
            raise PackageBuildError(f"SHA256SUMS target is missing: {value}")
        actual = sha256_file(target)
        if actual != expected:
            raise PackageBuildError(f"SHA256SUMS mismatch for {value}")
        checked += 1
    return {"status": "ok", "checked_files": checked}


def verify_archive(archive_path: Path) -> dict[str, Any]:
    if not archive_path.is_file():
        raise PackageBuildError(f"archive is missing: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if not names:
            raise PackageBuildError("archive is empty")
        roots: set[str] = set()
        for name in names:
            normalized = name.replace("\\", "/")
            path = Path(normalized)
            if normalized.startswith("/") or ".." in path.parts:
                raise PackageBuildError(f"unsafe archive entry: {name}")
            if "\\" in name:
                raise PackageBuildError(f"Windows-style archive entry: {name}")
            roots.add(normalized.split("/", 1)[0])
        if len(roots) != 1:
            raise PackageBuildError(f"archive must have one package root: {sorted(roots)}")
        root_name = next(iter(roots))
        required = {
            f"{root_name}/START_HERE.md",
            f"{root_name}/reproduce.sh",
            f"{root_name}/PACKAGE_MANIFEST.json",
            f"{root_name}/SHA256SUMS",
            f"{root_name}/BUILD_REPORT.md",
        }
        missing = sorted(required - set(names))
        if missing:
            raise PackageBuildError(f"archive required entries are missing: {missing}")
        with tempfile.TemporaryDirectory(prefix="v13 repro verify ") as temporary:
            archive.extractall(temporary)
            package_root = Path(temporary) / root_name
            stage_result = validate_stage(package_root)
            checksum_result = _verify_checksum_manifest(package_root)
    return {
        "status": "ok",
        "archive_path": str(archive_path.resolve()),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "package_root": root_name,
        "stage_validation": stage_result,
        "checksum_validation": checksum_result,
    }


def _safe_remove_stage(stage_container: Path, output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_stage = stage_container.resolve()
    if resolved_stage.parent != resolved_output:
        raise PackageBuildError(
            f"refusing to remove stage outside output directory: {resolved_stage}"
        )
    if not resolved_stage.name.startswith(".v13_repro_source_") or not resolved_stage.name.endswith(
        ".stage"
    ):
        raise PackageBuildError(f"refusing to remove unexpected stage path: {resolved_stage}")
    if resolved_stage.exists():
        shutil.rmtree(resolved_stage)


def _build_report_markdown(manifest: dict[str, Any]) -> str:
    experiments = manifest["experiments"]
    history = manifest["history"]
    artifacts = manifest["artifacts"]
    lines = [
        "# V13 Reproducibility Package Build Report",
        "",
        "This report is part of the deterministic package payload. Host-specific execution time and paths are recorded beside the ZIP, not here.",
        "",
        "## Package Scope",
        "",
        f"- Release epoch: `{manifest['release_epoch']}`",
        f"- Clean reference commit: `{manifest['source']['clean_reference_commit']}`",
        f"- Current source files: `{manifest['source']['current_file_count']}`",
        f"- Clean reference files: `{manifest['source']['clean_file_count']}`",
        f"- Experiment descriptors: `{experiments['count']}`",
        f"- External artifact contracts: `{artifacts['count']}`",
        f"- Historical records: `{history['record_count']}`",
        f"- Historical records copied in full: `{history['copied_count']}`",
        f"- Large historical records summarized: `{history['evidence_summary_count']}`",
        "",
        "## Experiment Categories",
        "",
    ]
    for category, count in sorted(experiments["categories"].items()):
        lines.append(f"- `{category}`: `{count}`")
    lines.extend(
        [
            "",
            "## Safety Properties",
            "",
            "- Checkpoints, sequence packages, parsed archives, STEP, STL, PNG, PDF, caches, and nested Git metadata are excluded.",
            "- External artifacts are represented by identity contracts and are verified before runnable experiments start.",
            "- Historical failed experiments require explicit opt-in.",
            "- Current generated CAD quality is not claimed to be satisfactory.",
            "- Linux GPU, CUDA, OCC, and real-data smoke checks remain target-server acceptance steps.",
            "",
        ]
    )
    return "\n".join(lines)


def build_package(
    repo_root: Path,
    output_dir: Path,
    package_name: str,
    epoch: int,
    *,
    clean_commit: str = "16cf19bb79b6bfa8beb4660e88f8d9dc813216e2",
    history_roots: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not package_name.endswith(".zip") or Path(package_name).name != package_name:
        raise PackageBuildError(f"invalid package name: {package_name}")
    package_root_name = package_name[:-4]
    stage_container = output_dir / f".{package_root_name}.stage"
    _safe_remove_stage(stage_container, output_dir)
    stage_container.mkdir()
    stage_root = stage_container / package_root_name
    stage_root.mkdir()
    archive_path = output_dir / package_name
    checksum_path = output_dir / f"{package_name}.sha256"
    execution_path = output_dir / f"{package_root_name}.build-execution.json"

    build_succeeded = False
    try:
        template_root = repo_root / "reproducibility"
        required_templates = [
            template_root / "reproduce.sh",
            template_root / "launchers" / "repro_cli.py",
            template_root / "launchers" / "repro_runtime.py",
            template_root / "configs" / "paths.env.example",
            template_root / "environments" / "environment.linux-gpu.yml",
            template_root / "docs" / "START_HERE.md",
            template_root / "catalog" / "experiments.json",
            template_root / "catalog" / "artifacts.json",
        ]
        missing_templates = [str(path) for path in required_templates if not path.is_file()]
        if missing_templates:
            raise PackageBuildError(f"required package templates are missing: {missing_templates}")

        _copy_text_lf(template_root / "reproduce.sh", stage_root / "reproduce.sh")
        _copy_text_lf(
            template_root / "docs" / "START_HERE.md", stage_root / "START_HERE.md"
        )
        for directory in (
            "launchers",
            "configs",
            "environments",
            "reports",
            "tests",
        ):
            _copy_template_tree(template_root / directory, stage_root / directory)
        _copy_template_tree(
            template_root / "project_history", stage_root / "project_history"
        )

        current_manifest = _copy_current_source(
            repo_root, stage_root / "source" / "current"
        )
        clean_manifest = export_clean_head(
            repo_root,
            clean_commit,
            stage_root / "source" / "clean_head_16cf19b",
        )

        artifact_summary = _expand_artifact_catalog(
            repo_root,
            template_root / "catalog" / "artifacts.json",
            stage_root / "artifact_specs",
        )
        artifact_ids = {item["id"] for item in artifact_summary["artifacts"]}
        experiment_summary, coverage_rules = _expand_experiment_catalog(
            template_root / "catalog" / "experiments.json",
            stage_root / "experiments",
            artifact_ids,
        )

        postmortem = repo_root / "docs" / "full_experiment_postmortem_20260731.md"
        if postmortem.is_file():
            _copy_text_lf(
                postmortem,
                stage_root
                / "project_history"
                / "01_full_postmortem"
                / postmortem.name,
            )
            _copy_text_lf(postmortem, stage_root / "reports" / postmortem.name)
        history = _curate_history(
            repo_root,
            stage_root / "project_history",
            list(history_roots or []),
            coverage_rules,
        )

        provenance = _capture_provenance(
            repo_root,
            stage_root / "provenance",
            clean_commit,
            current_manifest,
            clean_manifest,
        )
        package_manifest = {
            "schema_version": 1,
            "package_name": package_root_name,
            "release_epoch": epoch,
            "target_platform": "linux-x86_64-nvidia-gpu",
            "default_source_layer": "source/current",
            "reference_source_layer": "source/clean_head_16cf19b",
            "source": {
                "clean_reference_commit": clean_commit,
                "current_head": provenance.get("outer_head"),
                "current_file_count": current_manifest["file_count"],
                "clean_file_count": clean_manifest["file_count"],
            },
            "experiments": experiment_summary,
            "artifacts": artifact_summary,
            "history": {
                key: history[key]
                for key in (
                    "record_count",
                    "copied_count",
                    "evidence_summary_count",
                    "unclassified_count",
                )
            },
            "heavy_artifacts_included": False,
        }
        write_stable_json(stage_root / "PACKAGE_MANIFEST.json", package_manifest)
        (stage_root / "BUILD_REPORT.md").write_text(
            _build_report_markdown(package_manifest), encoding="utf-8", newline="\n"
        )

        pre_checksum_validation = validate_stage(stage_root)
        write_checksum_manifest(stage_root)
        stage_validation = validate_stage(stage_root)
        archive_hash = write_deterministic_zip(stage_root, archive_path, epoch)
        checksum_path.write_text(
            f"{archive_hash}  {package_name}\n", encoding="ascii", newline="\n"
        )
        archive_verification = verify_archive(archive_path)
        execution = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "host_platform": platform.platform(),
            "python": platform.python_version(),
            "repo_root": str(repo_root),
            "output_dir": str(output_dir),
            "archive_path": str(archive_path),
            "archive_sha256": archive_hash,
            "status": "ok",
        }
        write_stable_json(execution_path, execution)
        result = {
            "status": "ok",
            "archive_path": str(archive_path),
            "checksum_path": str(checksum_path),
            "execution_report_path": str(execution_path),
            "archive_sha256": archive_hash,
            "archive_size_bytes": archive_path.stat().st_size,
            "pre_checksum_validation": pre_checksum_validation,
            "stage_validation": stage_validation,
            "archive_verification": archive_verification,
            "package_manifest": package_manifest,
        }
        build_succeeded = True
        return result
    except Exception as exc:
        write_stable_json(
            execution_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "host_platform": platform.platform(),
                "python": platform.python_version(),
                "repo_root": str(repo_root),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": repr(exc),
                "stage_retained": str(stage_container),
            },
        )
        raise
    finally:
        if build_succeeded and stage_container.exists():
            _safe_remove_stage(stage_container, output_dir)
