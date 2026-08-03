from __future__ import annotations

import hashlib
import json
import math
import os
import importlib.util
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ReproError(RuntimeError):
    """Raised when a reproducibility safety or readiness check fails."""


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReproError(f"cannot read JSON {path}: {exc}") from exc


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_relative_path(value: str, *, field: str) -> Path:
    candidate = Path(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReproError(f"unsafe {field}: {value}")
    return candidate


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_paths(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE file without evaluating shell syntax."""

    if not path.is_file():
        raise ReproError(
            f"path configuration is missing: {path}; create it from paths.env.example"
        )
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ReproError(f"invalid paths.env line {line_number}: missing '='")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_RE.fullmatch(key):
            raise ReproError(f"invalid paths.env key on line {line_number}: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif value.startswith(("'", '"')) or value.endswith(("'", '"')):
            raise ReproError(f"unbalanced quote in paths.env line {line_number}")
        if key in result:
            raise ReproError(f"duplicate paths.env key: {key}")
        result[key] = value
    return result


def _load_descriptors(
    root: Path,
    subdirectory: str,
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    descriptor_root = root / subdirectory
    if not descriptor_root.is_dir():
        raise ReproError(f"{kind} directory is missing: {descriptor_root}")
    descriptors: dict[str, dict[str, Any]] = {}
    for path in sorted(descriptor_root.rglob("*.json")):
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ReproError(f"{kind} descriptor must be an object: {path}")
        descriptor_id = value.get("id")
        if not isinstance(descriptor_id, str) or not _ID_RE.fullmatch(descriptor_id):
            raise ReproError(f"invalid {kind} id in {path}: {descriptor_id!r}")
        if descriptor_id in descriptors:
            raise ReproError(f"duplicate {kind} id: {descriptor_id}")
        value = dict(value)
        value["_descriptor_path"] = path.relative_to(root).as_posix()
        descriptors[descriptor_id] = value
    return descriptors


def load_experiments(package_root: Path) -> dict[str, dict[str, Any]]:
    experiments = _load_descriptors(
        package_root, "experiments", kind="experiment"
    )
    allowed_categories = {
        "recommended",
        "baselines",
        "diagnostics",
        "historical_failed",
    }
    allowed_states = {"runnable", "documentary", "blocked_missing_evidence"}
    for experiment_id, experiment in experiments.items():
        category = experiment.get("category")
        state = experiment.get("state")
        if category not in allowed_categories:
            raise ReproError(
                f"experiment {experiment_id} has invalid category: {category!r}"
            )
        if state not in allowed_states:
            raise ReproError(
                f"experiment {experiment_id} has invalid state: {state!r}"
            )
        command = experiment.get("command")
        if state == "runnable" and (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ReproError(
                f"runnable experiment {experiment_id} needs a command argument array"
            )
    return experiments


def load_artifact_specs(package_root: Path) -> dict[str, dict[str, Any]]:
    return _load_descriptors(package_root, "artifact_specs", kind="artifact")


def verify_package_checksums(package_root: Path) -> list[dict[str, Any]]:
    checksum_path = package_root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ReproError(f"package checksum manifest is missing: {checksum_path}")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if not match:
            raise ReproError(f"invalid SHA256SUMS line {line_number}")
        expected, relative_value = match.groups()
        expected = expected.lower()
        relative = _safe_relative_path(relative_value, field="checksum path")
        relative_posix = relative.as_posix()
        if relative_posix == "SHA256SUMS":
            raise ReproError("SHA256SUMS must not contain a checksum for itself")
        if relative_posix in seen:
            raise ReproError(f"duplicate SHA256SUMS path: {relative_posix}")
        seen.add(relative_posix)
        target = package_root / relative
        if not target.is_file():
            results.append(
                {
                    "path": relative_posix,
                    "status": "missing",
                    "expected_sha256": expected,
                    "actual_sha256": None,
                }
            )
            continue
        actual = _sha256_file(target)
        results.append(
            {
                "path": relative_posix,
                "status": "ok" if actual == expected else "hash_mismatch",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return results


def _directory_inventory(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return count, total_bytes, digest.hexdigest()


def verify_artifact(spec: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    artifact_id = str(spec.get("id", "<unknown>"))
    path_var = spec.get("path_var")
    if not isinstance(path_var, str) or not path_var:
        raise ReproError(f"artifact {artifact_id} has no path_var")
    configured = paths.get(path_var, "")
    if not configured:
        return {
            "artifact_id": artifact_id,
            "status": "path_not_configured",
            "path_var": path_var,
            "path": None,
            "verification_strength": "none",
        }
    path = Path(configured).expanduser()
    artifact_type = spec.get("type", "file")
    if artifact_type == "file":
        if not path.is_file():
            return {
                "artifact_id": artifact_id,
                "status": "missing",
                "path_var": path_var,
                "path": str(path),
                "verification_strength": "none",
            }
        actual_size = path.stat().st_size
        expected_size = spec.get("size_bytes")
        if expected_size is not None and actual_size != int(expected_size):
            return {
                "artifact_id": artifact_id,
                "status": "size_mismatch",
                "path_var": path_var,
                "path": str(path),
                "expected_size_bytes": int(expected_size),
                "actual_size_bytes": actual_size,
                "verification_strength": "size",
            }
        expected_hash = spec.get("sha256")
        if expected_hash:
            if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(
                expected_hash.lower()
            ):
                raise ReproError(f"artifact {artifact_id} has invalid sha256")
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash.lower():
                return {
                    "artifact_id": artifact_id,
                    "status": "hash_mismatch",
                    "path_var": path_var,
                    "path": str(path),
                    "expected_sha256": expected_hash.lower(),
                    "actual_sha256": actual_hash,
                    "verification_strength": "content_sha256",
                }
            return {
                "artifact_id": artifact_id,
                "status": "ready",
                "path_var": path_var,
                "path": str(path),
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "verification_strength": "content_sha256",
            }
        return {
            "artifact_id": artifact_id,
            "status": "identity_unresolved",
            "path_var": path_var,
            "path": str(path),
            "size_bytes": actual_size,
            "verification_strength": "none",
        }
    if artifact_type == "directory":
        if not path.is_dir():
            return {
                "artifact_id": artifact_id,
                "status": "missing",
                "path_var": path_var,
                "path": str(path),
                "verification_strength": "none",
            }
        count, total_bytes, inventory_hash = _directory_inventory(path)
        expected_count = spec.get("file_count")
        expected_bytes = spec.get("total_size_bytes")
        expected_inventory = spec.get("inventory_sha256")
        if not expected_inventory:
            status = "identity_unresolved"
        elif expected_count is not None and count != int(expected_count):
            status = "file_count_mismatch"
        elif expected_bytes is not None and total_bytes != int(expected_bytes):
            status = "size_mismatch"
        elif expected_inventory and inventory_hash != str(expected_inventory).lower():
            status = "inventory_mismatch"
        else:
            status = "ready"
        return {
            "artifact_id": artifact_id,
            "status": status,
            "path_var": path_var,
            "path": str(path),
            "file_count": count,
            "total_size_bytes": total_bytes,
            "inventory_sha256": inventory_hash,
            "verification_strength": "name_size_inventory",
        }
    raise ReproError(f"artifact {artifact_id} has unsupported type: {artifact_type}")


def _expand_argument(argument: str, variables: dict[str, str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            missing.add(key)
            return match.group(0)
        return variables[key]

    expanded = _PLACEHOLDER_RE.sub(replace, argument)
    if missing:
        raise ReproError(
            "command references undefined variable(s): " + ", ".join(sorted(missing))
        )
    return expanded


def _experiment_hash(experiment: dict[str, Any]) -> str:
    public = {key: value for key, value in experiment.items() if not key.startswith("_")}
    return hashlib.sha256(_stable_json(public).encode("utf-8")).hexdigest()


def create_run_context(
    package_root: Path,
    experiment: dict[str, Any],
    paths: dict[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    run_root_value = paths.get("V13_RUN_ROOT", "")
    if not run_root_value:
        raise ReproError("V13_RUN_ROOT is not configured")
    run_root = Path(run_root_value).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_utc = now.astimezone(timezone.utc)
    timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    descriptor_hash = _experiment_hash(experiment)
    run_dir = run_root / f"{experiment['id']}_{timestamp}_{descriptor_hash[:8]}"
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ReproError(f"run directory already exists: {run_dir}") from exc
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "experiment_category": experiment.get("category"),
        "experiment_descriptor_sha256": descriptor_hash,
        "experiment_descriptor_path": experiment.get("_descriptor_path"),
        "package_root": str(package_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "created_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "resolved_paths": dict(sorted(paths.items())),
        "status": "created",
        "historical_failed_opt_in": False,
    }
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _load_configured_paths(package_root: Path) -> dict[str, str]:
    configured = package_root / "configs" / "paths.env"
    if configured.is_file():
        return load_paths(configured)
    example = package_root / "configs" / "paths.env.example"
    if example.is_file():
        raise ReproError(
            f"path configuration is missing: {configured}; copy and edit {example.name}"
        )
    raise ReproError(f"path configuration is missing: {configured}")


def _verify_required_artifacts(
    package_root: Path,
    experiment: dict[str, Any],
    paths: dict[str, str],
) -> list[dict[str, Any]]:
    specs = load_artifact_specs(package_root)
    results: list[dict[str, Any]] = []
    for artifact_id in experiment.get("required_artifacts", []):
        if artifact_id not in specs:
            raise ReproError(
                f"experiment {experiment['id']} references unknown artifact: {artifact_id}"
            )
        result = verify_artifact(specs[artifact_id], paths)
        results.append(result)
        if result["status"] != "ready":
            raise ReproError(
                f"artifact {artifact_id} is not ready: {result['status']} "
                f"({specs[artifact_id].get('path_var')})"
            )
    return results


def _bind_artifacts(
    run_dir: Path,
    experiment: dict[str, Any],
    package_root: Path,
    paths: dict[str, str],
    verified_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bindings = experiment.get("artifact_bindings", {})
    if not bindings:
        return []
    if not isinstance(bindings, dict) or not all(
        isinstance(artifact_id, str) and isinstance(target, str) and target
        for artifact_id, target in bindings.items()
    ):
        raise ReproError(
            f"experiment {experiment['id']} has invalid artifact_bindings"
        )
    specs = load_artifact_specs(package_root)
    ready = {
        item["artifact_id"]: item
        for item in verified_results
        if item.get("status") == "ready"
    }
    records: list[dict[str, Any]] = []
    for artifact_id, target_value in sorted(bindings.items()):
        if artifact_id not in ready or artifact_id not in specs:
            raise ReproError(
                f"artifact binding {artifact_id} is not a verified required artifact"
            )
        relative = _safe_relative_path(target_value, field="artifact binding target")
        target = run_dir / relative
        if target.exists() or target.is_symlink():
            raise ReproError(f"artifact binding target already exists: {target}")
        source = Path(ready[artifact_id]["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact_type = specs[artifact_id].get("type", "file")
        method: str
        if artifact_type == "file":
            try:
                os.link(source, target)
                method = "hardlink"
            except OSError:
                try:
                    target.symlink_to(source.resolve(), target_is_directory=False)
                    method = "symlink"
                except OSError as exc:
                    raise ReproError(
                        f"cannot bind external artifact {artifact_id} without copying: {exc}"
                    ) from exc
        elif artifact_type == "directory":
            try:
                target.symlink_to(source.resolve(), target_is_directory=True)
                method = "symlink"
            except OSError as exc:
                raise ReproError(
                    f"cannot bind external artifact directory {artifact_id}: {exc}"
                ) from exc
        else:
            raise ReproError(
                f"artifact binding {artifact_id} has unsupported type: {artifact_type}"
            )
        records.append(
            {
                "artifact_id": artifact_id,
                "source": str(source.resolve()),
                "target": relative.as_posix(),
                "method": method,
            }
        )
    return records


def run_experiment(
    package_root: Path,
    experiment_id: str,
    allow_historical_failed: bool = False,
    smoke: bool = False,
) -> int:
    experiments = load_experiments(package_root)
    if experiment_id not in experiments:
        raise ReproError(f"unknown experiment id: {experiment_id}")
    experiment = experiments[experiment_id]
    if experiment.get("state") != "runnable":
        reason = experiment.get("blocked_reason", "descriptor is documentary")
        raise ReproError(f"experiment {experiment_id} is not runnable: {reason}")
    historical = experiment.get("category") == "historical_failed"
    if historical and not allow_historical_failed:
        raise ReproError(
            f"experiment {experiment_id} is a historical failure; "
            "pass --allow-historical-failed to run it explicitly"
        )
    paths = _load_configured_paths(package_root)
    artifact_results = _verify_required_artifacts(package_root, experiment, paths)
    context = create_run_context(package_root, experiment, paths)
    run_dir = Path(context["run_dir"])
    manifest_path = run_dir / "run_manifest.json"
    binding_records = _bind_artifacts(
        run_dir, experiment, package_root, paths, artifact_results
    )
    context["historical_failed_opt_in"] = bool(historical and allow_historical_failed)
    context["artifact_verification"] = artifact_results
    context["artifact_bindings"] = binding_records
    command_value = experiment.get("smoke_command") if smoke else None
    if not command_value:
        command_value = experiment["command"]
    variables = dict(paths)
    variables.update(
        {
            "PACKAGE_ROOT": str(package_root.resolve()),
            "SOURCE_ROOT": str((package_root / "source" / "current").resolve()),
            "RUN_DIR": str(run_dir.resolve()),
        }
    )
    command = [_expand_argument(item, variables) for item in command_value]
    if command and command[0] in {"python", "python3"}:
        command[0] = sys.executable
    working_directory_value = experiment.get("working_directory", "${RUN_DIR}")
    if not isinstance(working_directory_value, str) or not working_directory_value:
        raise ReproError(
            f"experiment {experiment_id} has an invalid working_directory"
        )
    working_directory = Path(
        _expand_argument(working_directory_value, variables)
    ).expanduser()
    if not working_directory.is_dir():
        raise ReproError(
            f"experiment working directory is missing: {working_directory}"
        )
    environment_value = experiment.get("environment", {})
    if not isinstance(environment_value, dict) or not all(
        isinstance(key, str)
        and _ENV_RE.fullmatch(key)
        and isinstance(value, str)
        for key, value in environment_value.items()
    ):
        raise ReproError(
            f"experiment {experiment_id} has an invalid environment mapping"
        )
    if smoke:
        smoke_environment = experiment.get("smoke_environment", {})
        if not isinstance(smoke_environment, dict) or not all(
            isinstance(key, str)
            and _ENV_RE.fullmatch(key)
            and isinstance(value, str)
            for key, value in smoke_environment.items()
        ):
            raise ReproError(
                f"experiment {experiment_id} has an invalid smoke_environment mapping"
            )
        environment_value = {**environment_value, **smoke_environment}
    environment_overrides = {
        key: _expand_argument(value, variables)
        for key, value in sorted(environment_value.items())
    }
    context["command"] = command
    context["working_directory"] = str(working_directory.resolve())
    context["environment_overrides"] = environment_overrides
    context["smoke"] = smoke
    context["status"] = "running"
    _atomic_write_json(manifest_path, context)
    environment = os.environ.copy()
    environment.update(paths)
    environment.update(
        {
            "V13_REPRO_PACKAGE_ROOT": variables["PACKAGE_ROOT"],
            "V13_REPRO_SOURCE_ROOT": variables["SOURCE_ROOT"],
            "V13_REPRO_RUN_DIR": variables["RUN_DIR"],
        }
    )
    environment.update(environment_overrides)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        context["returncode"] = completed.returncode
        context["status"] = "completed" if completed.returncode == 0 else "failed"
    except OSError as exc:
        context["returncode"] = None
        context["status"] = "failed"
        context["error"] = str(exc)
        _atomic_write_json(manifest_path, context)
        raise ReproError(f"failed to start experiment command: {exc}") from exc
    _atomic_write_json(manifest_path, context)
    return int(context["returncode"])


def _lookup_json_field(value: Any, field: str) -> Any:
    current = value
    for component in field.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(field)
        current = current[component]
    return current


def verify_run(run_dir: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    overall = "verified"
    for specification in experiment.get("expected_outputs", []):
        relative_value = specification.get("path")
        if not isinstance(relative_value, str):
            raise ReproError(
                f"experiment {experiment.get('id')} has an output without a path"
            )
        relative = _safe_relative_path(relative_value, field="expected output path")
        path = run_dir / relative
        result: dict[str, Any] = {"path": relative.as_posix()}
        if not path.exists():
            result["status"] = "missing"
            overall = "failed"
            outputs.append(result)
            continue
        if specification.get("kind") == "json":
            try:
                value = _read_json(path)
            except ReproError as exc:
                result.update(status="invalid_json", error=str(exc))
                overall = "failed"
                outputs.append(result)
                continue
            result["status"] = "ok"
            for field in specification.get("finite_fields", []):
                try:
                    field_value = _lookup_json_field(value, field)
                except KeyError:
                    result.update(status="missing_field", field=field)
                    overall = "failed"
                    break
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, (int, float))
                    or not math.isfinite(field_value)
                ):
                    result.update(status="nonfinite", field=field, value=field_value)
                    overall = "failed"
                    break
        else:
            result["status"] = "ok"
        outputs.append(result)
    report = {
        "schema_version": 1,
        "experiment_id": experiment.get("id"),
        "status": overall,
        "outputs": outputs,
    }
    _atomic_write_json(run_dir / "verification.json", report)
    return report


def latest_run(run_root: Path, experiment_id: str) -> Path:
    candidates = sorted(
        path
        for path in run_root.glob(f"{experiment_id}_*")
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    if not candidates:
        raise ReproError(f"no runs found for experiment: {experiment_id}")
    return candidates[-1]


def _probe_import(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"available": False, "error": str(exc)}
    return {"available": spec is not None}


def _disk_probe(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _command_probe(name: str) -> dict[str, Any]:
    resolved = shutil.which(name)
    return {"available": resolved is not None, "path": resolved}


def _nvidia_probe() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "status": "command_missing"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": completed.returncode == 0 and bool(devices),
        "status": "ok" if completed.returncode == 0 else "command_failed",
        "returncode": completed.returncode,
        "devices": devices,
        "stderr": completed.stderr.strip(),
    }


def _cuda_probe(torch_probe: dict[str, Any]) -> dict[str, Any]:
    if not torch_probe.get("available"):
        return {"available": False, "status": "torch_unavailable"}
    code = (
        "import json, torch; "
        "ok=bool(torch.cuda.is_available()); "
        "d={'available':ok,'status':'ok' if ok else 'cuda_unavailable',"
        "'torch_version':torch.__version__,'torch_cuda':torch.version.cuda}; "
        "d.update({'device_name':torch.cuda.get_device_name(0),"
        "'compute_capability':list(torch.cuda.get_device_capability(0))} if ok else {}); "
        "print(json.dumps(d, allow_nan=False))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "status": "probe_failed", "error": str(exc)}
    if completed.returncode != 0:
        return {
            "available": False,
            "status": "probe_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "status": "invalid_probe_output",
            "error": str(exc),
        }
    return value if isinstance(value, dict) else {"available": False, "status": "invalid_probe_output"}


def package_preflight(package_root: Path) -> dict[str, Any]:
    package_error: str | None = None
    try:
        checksum_results = verify_package_checksums(package_root)
    except ReproError as exc:
        checksum_results = []
        package_error = str(exc)
    package_ok = package_error is None and all(
        item["status"] == "ok" for item in checksum_results
    )
    experiments = load_experiments(package_root)
    artifacts = load_artifact_specs(package_root)
    configured_path = package_root / "configs" / "paths.env"
    paths: dict[str, str] = {}
    path_error: str | None = None
    if configured_path.is_file():
        try:
            paths = load_paths(configured_path)
        except ReproError as exc:
            path_error = str(exc)
    else:
        path_error = f"not configured: {configured_path}"
    artifact_results = [verify_artifact(spec, paths) for spec in artifacts.values()]
    system = {
        "platform": sys.platform,
        "linux": sys.platform.startswith("linux"),
        "python": sys.version.split()[0],
    }
    module_names = ("torch", "transformers", "numpy", "scipy", "zstandard", "OCC")
    module_probes = {name: _probe_import(name) for name in module_names}
    capabilities = {
        "disk": _disk_probe(package_root),
        "commands": {
            name: _command_probe(name) for name in ("bash", "conda", "git")
        },
        "nvidia": _nvidia_probe(),
        "python_modules": module_probes,
        "cuda": _cuda_probe(module_probes["torch"]),
    }
    return {
        "package_integrity": "ok" if package_ok else "failed",
        "package_error": package_error,
        "checksum_files": len(checksum_results),
        "experiments": len(experiments),
        "artifacts": artifact_results,
        "path_configuration": "ok" if path_error is None else "not_ready",
        "path_error": path_error,
        "system": system,
        "capabilities": capabilities,
        "target_ready": bool(
            package_ok
            and path_error is None
            and system["linux"]
            and capabilities["cuda"].get("available")
            and all(item["status"] == "ready" for item in artifact_results)
        ),
    }


def format_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
