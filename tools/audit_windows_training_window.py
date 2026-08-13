#!/usr/bin/env python3
"""Audit Windows restart risk and optionally set reversible active hours.

The default mode is read-only.  The only supported mutation writes the existing
``ActiveHoursStart`` and ``ActiveHoursEnd`` DWORD values under the Windows
Update UX settings key.  Policy keys, services, and scheduled tasks are never
modified.  Every mutation requires an explicit confirmation flag, a durable
backup, read-back verification, and rollback on partial failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import winreg
except ImportError:  # pragma: no cover - exercised only on non-Windows hosts
    winreg = None


SCHEMA_VERSION = 1
UX_SETTINGS_KEY = r"SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"
POLICY_SETTINGS_KEY = r"SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate"
ACTIVE_HOUR_NAMES = ("ActiveHoursStart", "ActiveHoursEnd")
UX_VALUE_NAMES = ACTIVE_HOUR_NAMES + ("SmartActiveHoursState",)
POLICY_VALUE_NAMES = ("SetActiveHours",) + ACTIVE_HOUR_NAMES
REG_DWORD_NAME = "REG_DWORD"
DEFAULT_RECOMMENDED_START = 12
DEFAULT_RECOMMENDED_END = 6
MAX_ACTIVE_HOURS = 18
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5:
        return "access_denied"
    if isinstance(exc, FileNotFoundError) or getattr(exc, "winerror", None) in {
        2,
        3,
    }:
        return "not_found"
    return "os_error"


def _registry_type_name(value_type: int) -> str:
    if winreg is not None:
        names = {
            winreg.REG_BINARY: "REG_BINARY",
            winreg.REG_DWORD: REG_DWORD_NAME,
            winreg.REG_EXPAND_SZ: "REG_EXPAND_SZ",
            winreg.REG_MULTI_SZ: "REG_MULTI_SZ",
            winreg.REG_NONE: "REG_NONE",
            winreg.REG_QWORD: "REG_QWORD",
            winreg.REG_SZ: "REG_SZ",
        }
        return names.get(value_type, f"REG_TYPE_{value_type}")
    return REG_DWORD_NAME if value_type == 4 else f"REG_TYPE_{value_type}"


class WindowsRegistry:
    """Minimal, fixed-scope registry adapter used by the audit and setter."""

    def __init__(self) -> None:
        if winreg is None:
            raise RuntimeError("Windows registry is unavailable")

    @staticmethod
    def _access(mask: int) -> int:
        return mask | getattr(winreg, "KEY_WOW64_64KEY", 0)

    def read_values(self, key_path: str, names: Sequence[str]) -> dict[str, Any]:
        values: dict[str, dict[str, Any]] = {
            name: {"present": False} for name in names
        }
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                key_path,
                0,
                self._access(winreg.KEY_READ),
            ) as key:
                for name in names:
                    try:
                        value, value_type = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    values[name] = {
                        "present": True,
                        "value": value,
                        "registry_type": _registry_type_name(value_type),
                    }
        except BaseException as exc:
            return {"status": _error_category(exc), "values": values}
        return {"status": "ok", "values": values}

    def key_exists(self, key_path: str) -> dict[str, Any]:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                key_path,
                0,
                self._access(winreg.KEY_READ),
            ):
                pass
        except FileNotFoundError:
            return {"status": "ok", "present": False}
        except BaseException as exc:
            return {"status": _error_category(exc), "present": None}
        return {"status": "ok", "present": True}

    def write_dword(self, key_path: str, name: str, value: int) -> None:
        if key_path != UX_SETTINGS_KEY or name not in ACTIVE_HOUR_NAMES:
            raise ValueError("registry write outside the active-hours allow-list")
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            key_path,
            0,
            self._access(winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE),
        ) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(value))


def active_hours_duration(start: int, end: int) -> int:
    return (end - start) % 24


def validate_active_hours(start: int, end: int) -> None:
    if not 0 <= start <= 23 or not 0 <= end <= 23:
        raise ValueError("active-hour endpoints must be integers from 0 through 23")
    duration = active_hours_duration(start, end)
    if duration == 0:
        raise ValueError("active-hour start and end must differ")
    if duration > MAX_ACTIVE_HOURS:
        raise ValueError(
            f"active-hour duration must not exceed {MAX_ACTIVE_HOURS} hours"
        )


def _valid_hour_entry(entry: Mapping[str, Any]) -> bool:
    value = entry.get("value")
    return (
        entry.get("present") is True
        and entry.get("registry_type") == REG_DWORD_NAME
        and isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 23
    )


def _extract_hour_pair(read_result: Mapping[str, Any]) -> tuple[int, int] | None:
    if read_result.get("status") != "ok":
        return None
    values = read_result.get("values", {})
    start_entry = values.get("ActiveHoursStart", {})
    end_entry = values.get("ActiveHoursEnd", {})
    if not _valid_hour_entry(start_entry) or not _valid_hour_entry(end_entry):
        return None
    return int(start_entry["value"]), int(end_entry["value"])


def _policy_override_present(policy: Mapping[str, Any]) -> bool | None:
    if policy.get("status") not in {"ok", "not_found"}:
        return None
    values = policy.get("values", {})
    return any(values.get(name, {}).get("present") is True for name in POLICY_VALUE_NAMES)


def collect_active_hours(registry: Any) -> dict[str, Any]:
    ux = registry.read_values(UX_SETTINGS_KEY, UX_VALUE_NAMES)
    policy = registry.read_values(POLICY_SETTINGS_KEY, POLICY_VALUE_NAMES)
    ux_pair = _extract_hour_pair(ux)
    policy_pair = _extract_hour_pair(policy)
    override = _policy_override_present(policy)

    effective_source = "unknown"
    effective_pair: tuple[int, int] | None = None
    set_policy = policy.get("values", {}).get("SetActiveHours", {})
    policy_enabled = (
        set_policy.get("present") is True
        and set_policy.get("registry_type") == REG_DWORD_NAME
        and isinstance(set_policy.get("value"), int)
        and not isinstance(set_policy.get("value"), bool)
        and set_policy.get("value") == 1
    )
    if policy_enabled and policy_pair is not None:
        effective_source = "policy"
        effective_pair = policy_pair
    elif override is False and ux_pair is not None:
        effective_source = "ux_settings"
        effective_pair = ux_pair

    effective: dict[str, Any] = {
        "source": effective_source,
        "start": None,
        "end": None,
        "duration_hours": None,
        "valid": False,
    }
    if effective_pair is not None:
        start, end = effective_pair
        duration = active_hours_duration(start, end)
        effective.update(
            {
                "start": start,
                "end": end,
                "duration_hours": duration,
                "valid": 0 < duration <= MAX_ACTIVE_HOURS,
            }
        )

    return {
        "ux_settings": ux,
        "policy_settings": policy,
        "policy_override_present": override,
        "effective": effective,
    }


PENDING_KEY_CHECKS = (
    (
        "component_based_servicing_reboot_pending",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
    ),
    (
        "windows_update_reboot_required",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired",
    ),
)
PENDING_VALUE_CHECKS = (
    (
        "pending_file_rename_operations",
        r"SYSTEM\CurrentControlSet\Control\Session Manager",
        "PendingFileRenameOperations",
    ),
    (
        "update_exe_volatile",
        r"SOFTWARE\Microsoft\Updates",
        "UpdateExeVolatile",
    ),
)


def _value_signals_restart(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, bytes, bytearray)):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() not in {"", "0"}
    if isinstance(value, int):
        return value != 0
    return bool(value)


def collect_pending_restart(registry: Any) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name, key_path in PENDING_KEY_CHECKS:
        probe = registry.key_exists(key_path)
        checks.append(
            {
                "name": name,
                "status": probe.get("status", "os_error"),
                "triggered": probe.get("present") is True,
            }
        )
    for name, key_path, value_name in PENDING_VALUE_CHECKS:
        read_result = registry.read_values(key_path, (value_name,))
        entry = read_result.get("values", {}).get(value_name, {})
        checks.append(
            {
                "name": name,
                "status": read_result.get("status", "os_error"),
                "triggered": entry.get("present") is True
                and _value_signals_restart(entry.get("value")),
            }
        )
    triggered = any(check["triggered"] for check in checks)
    complete = all(check["status"] in {"ok", "not_found"} for check in checks)
    detected: bool | None = True if triggered else (False if complete else None)
    return {"detected": detected, "checks": checks}


def _run_capture(command: Sequence[str]) -> dict[str, Any]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        creationflags=creation_flags,
    )
    return {"returncode": completed.returncode, "stdout": completed.stdout}


def _parse_number(value: str, *, integer: bool = False) -> float | int | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return int(value) if integer else round(float(value), 2)
    except ValueError:
        return None


def collect_gpu(
    command_runner: Callable[[Sequence[str]], Mapping[str, Any]] = _run_capture,
) -> dict[str, Any]:
    query = (
        "name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu"
    )
    try:
        result = command_runner(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
    except FileNotFoundError:
        return {"status": "nvidia_smi_unavailable", "devices": []}
    except (OSError, subprocess.SubprocessError):
        return {"status": "query_failed", "devices": []}
    if int(result.get("returncode", 1)) != 0:
        return {
            "status": "query_failed",
            "returncode": int(result.get("returncode", 1)),
            "devices": [],
        }

    devices: list[dict[str, Any]] = []
    for row in csv.reader(str(result.get("stdout", "")).splitlines()):
        if len(row) != 7:
            continue
        devices.append(
            {
                "name": row[0].strip(),
                "driver_version": row[1].strip(),
                "memory_total_mib": _parse_number(row[2], integer=True),
                "memory_used_mib": _parse_number(row[3], integer=True),
                "memory_free_mib": _parse_number(row[4], integer=True),
                "utilization_percent": _parse_number(row[5]),
                "temperature_c": _parse_number(row[6]),
            }
        )
    return {"status": "ok" if devices else "unparseable", "devices": devices}


def collect_storage(
    storage_targets: Mapping[str, Path],
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    volumes: list[dict[str, Any]] = []
    for label, path in storage_targets.items():
        if not LABEL_RE.fullmatch(label):
            raise ValueError(f"unsafe storage label: {label!r}")
        try:
            usage = disk_usage(path)
        except OSError as exc:
            volumes.append({"label": label, "status": _error_category(exc)})
            continue
        total = int(usage.total)
        free = int(usage.free)
        used = int(usage.used)
        volumes.append(
            {
                "label": label,
                "status": "ok",
                "total_gib": round(total / (1024**3), 2),
                "free_gib": round(free / (1024**3), 2),
                "used_percent": round((used / total) * 100, 1) if total else None,
            }
        )
    return {"volumes": volumes}


def iso_timestamp(now: datetime | None = None) -> str:
    moment = now or datetime.now().astimezone()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.isoformat(timespec="seconds")


def collect_audit(
    *,
    registry: Any | None = None,
    command_runner: Callable[[Sequence[str]], Mapping[str, Any]] = _run_capture,
    storage_targets: Mapping[str, Path] | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    platform_name: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_platform = platform_name or sys.platform
    is_windows = current_platform == "win32"
    if is_windows:
        registry = registry or WindowsRegistry()
        active_hours = collect_active_hours(registry)
        pending_restart = collect_pending_restart(registry)
    else:
        active_hours = {"status": "unsupported_non_windows"}
        pending_restart = {"detected": None, "status": "unsupported_non_windows"}

    targets = storage_targets or {"repository_volume": Path.cwd()}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso_timestamp(now),
        "scope": "p0b_windows_training_window_preflight",
        "privacy": {
            "hostname_recorded": False,
            "username_recorded": False,
            "absolute_paths_recorded": False,
            "network_identifiers_recorded": False,
            "gpu_serial_or_uuid_recorded": False,
        },
        "platform": {"windows": is_windows},
        "active_hours": active_hours,
        "pending_restart": pending_restart,
        "gpu": collect_gpu(command_runner),
        "storage": collect_storage(targets, disk_usage),
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_write_json(path: Path, payload: Any) -> None:
    """Create a small immutable backup without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _hour_pair_snapshot(read_result: Mapping[str, Any]) -> dict[str, Any] | None:
    pair = _extract_hour_pair(read_result)
    if pair is None:
        return None
    return {
        "ActiveHoursStart": {
            "value": pair[0],
            "registry_type": REG_DWORD_NAME,
        },
        "ActiveHoursEnd": {
            "value": pair[1],
            "registry_type": REG_DWORD_NAME,
        },
    }


def _read_pair_or_none(registry: Any) -> tuple[int, int] | None:
    return _extract_hour_pair(registry.read_values(UX_SETTINGS_KEY, ACTIVE_HOUR_NAMES))


def _rollback_written_values(
    registry: Any,
    rollback_values: Mapping[str, int],
) -> bool:
    rollback_ok = True
    for name in reversed(ACTIVE_HOUR_NAMES):
        try:
            registry.write_dword(UX_SETTINGS_KEY, name, int(rollback_values[name]))
        except BaseException:
            rollback_ok = False
    pair = _read_pair_or_none(registry)
    expected = (
        int(rollback_values["ActiveHoursStart"]),
        int(rollback_values["ActiveHoursEnd"]),
    )
    return rollback_ok and pair == expected


def _write_pair_transaction(
    registry: Any,
    *,
    target_start: int,
    target_end: int,
    rollback_start: int,
    rollback_end: int,
) -> dict[str, Any]:
    target = {
        "ActiveHoursStart": target_start,
        "ActiveHoursEnd": target_end,
    }
    rollback = {
        "ActiveHoursStart": rollback_start,
        "ActiveHoursEnd": rollback_end,
    }
    written: list[str] = []
    try:
        for name in ACTIVE_HOUR_NAMES:
            registry.write_dword(UX_SETTINGS_KEY, name, int(target[name]))
            written.append(name)
        observed = _read_pair_or_none(registry)
        if observed != (target_start, target_end):
            raise RuntimeError("readback_mismatch")
    except BaseException as exc:
        category = (
            "verification_failed"
            if isinstance(exc, RuntimeError) and str(exc) == "readback_mismatch"
            else _error_category(exc)
        )
        expected_rollback = (rollback_start, rollback_end)
        if _read_pair_or_none(registry) == expected_rollback:
            return {
                "status": "failed_no_change",
                "error_category": category,
                "written_values": len(written),
                "original_state_verified": True,
            }
        rollback_ok = _rollback_written_values(registry, rollback)
        return {
            "status": "failed_rolled_back" if rollback_ok else "rollback_incomplete",
            "error_category": category,
            "written_values": len(written),
            "rollback_verified": rollback_ok,
        }
    return {"status": "applied", "written_values": len(written), "verified": True}


def apply_active_hours(
    registry: Any,
    *,
    start: int,
    end: int,
    confirmed: bool,
    backup_path: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("active-hours change requires explicit confirmation")
    validate_active_hours(start, end)

    policy = registry.read_values(POLICY_SETTINGS_KEY, POLICY_VALUE_NAMES)
    override = _policy_override_present(policy)
    if override is None:
        return {
            "mode": "set_active_hours",
            "status": "refused_policy_state_unknown",
        }
    if override:
        return {"mode": "set_active_hours", "status": "refused_policy_override"}

    current_read = registry.read_values(UX_SETTINGS_KEY, UX_VALUE_NAMES)
    original = _hour_pair_snapshot(current_read)
    if original is None:
        return {
            "mode": "set_active_hours",
            "status": "refused_original_state_unreadable",
        }
    smart_state = current_read.get("values", {}).get("SmartActiveHoursState", {})
    if smart_state.get("present") is True and _value_signals_restart(
        smart_state.get("value")
    ):
        return {
            "mode": "set_active_hours",
            "status": "refused_smart_active_hours_enabled",
        }
    original_start = int(original["ActiveHoursStart"]["value"])
    original_end = int(original["ActiveHoursEnd"]["value"])
    backup = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "restore_windows_update_active_hours",
        "created_at": created_at or iso_timestamp(),
        "registry_scope": "HKLM",
        "registry_path": UX_SETTINGS_KEY,
        "values": original,
    }
    try:
        _exclusive_write_json(backup_path, backup)
    except FileExistsError:
        return {
            "mode": "set_active_hours",
            "status": "refused_backup_already_exists",
            "backup_file": backup_path.name,
        }
    except OSError as exc:
        return {
            "mode": "set_active_hours",
            "status": "refused_backup_write_failed",
            "error_category": _error_category(exc),
        }
    backup_sha256 = hashlib.sha256(backup_path.read_bytes()).hexdigest()

    transaction = _write_pair_transaction(
        registry,
        target_start=start,
        target_end=end,
        rollback_start=original_start,
        rollback_end=original_end,
    )
    return {
        "mode": "set_active_hours",
        **transaction,
        "requested": {
            "start": start,
            "end": end,
            "duration_hours": active_hours_duration(start, end),
        },
        "original": {"start": original_start, "end": original_end},
        "backup_file": backup_path.name,
        "backup_sha256": backup_sha256,
    }


def _load_restore_values(backup_path: Path) -> tuple[int, int]:
    payload = json.loads(backup_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("purpose") != "restore_windows_update_active_hours"
        or payload.get("registry_scope") != "HKLM"
        or payload.get("registry_path") != UX_SETTINGS_KEY
    ):
        raise ValueError("backup metadata is outside the active-hours allow-list")
    values = payload.get("values", {})
    for name in ACTIVE_HOUR_NAMES:
        entry = values.get(name, {})
        if entry.get("registry_type") != REG_DWORD_NAME:
            raise ValueError("backup registry type is not REG_DWORD")
        value = entry.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 23:
            raise ValueError("backup active-hour value is invalid")
    start = int(values["ActiveHoursStart"]["value"])
    end = int(values["ActiveHoursEnd"]["value"])
    validate_active_hours(start, end)
    return start, end


def restore_active_hours(
    registry: Any,
    *,
    backup_path: Path,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("active-hours restore requires explicit confirmation")
    policy = registry.read_values(POLICY_SETTINGS_KEY, POLICY_VALUE_NAMES)
    override = _policy_override_present(policy)
    if override is None:
        return {
            "mode": "restore_active_hours",
            "status": "refused_policy_state_unknown",
        }
    if override:
        return {"mode": "restore_active_hours", "status": "refused_policy_override"}
    try:
        target_start, target_end = _load_restore_values(backup_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"mode": "restore_active_hours", "status": "refused_invalid_backup"}

    current_read = registry.read_values(UX_SETTINGS_KEY, ACTIVE_HOUR_NAMES)
    current = _extract_hour_pair(current_read)
    if current is None:
        return {
            "mode": "restore_active_hours",
            "status": "refused_current_state_unreadable",
        }
    transaction = _write_pair_transaction(
        registry,
        target_start=target_start,
        target_end=target_end,
        rollback_start=current[0],
        rollback_end=current[1],
    )
    return {
        "mode": "restore_active_hours",
        **transaction,
        "restored": {"start": target_start, "end": target_end},
        "pre_restore": {"start": current[0], "end": current[1]},
        "backup_file": backup_path.name,
        "backup_sha256": hashlib.sha256(backup_path.read_bytes()).hexdigest(),
    }


def _format_window(effective: Mapping[str, Any]) -> str:
    if not effective.get("valid"):
        return "unknown / invalid"
    return (
        f"{int(effective['start']):02d}:00–{int(effective['end']):02d}:00 "
        f"({int(effective['duration_hours'])} h, source={effective['source']})"
    )


def _pending_text(pending_restart: Mapping[str, Any]) -> str:
    value = pending_restart.get("detected")
    if value is False:
        return "not detected by the four bounded registry probes"
    if value is None:
        return "unknown — at least one registry probe was unreadable"

    triggered = {
        str(check.get("name"))
        for check in pending_restart.get("checks", [])
        if check.get("triggered") is True
    }
    restart_specific = {
        "component_based_servicing_reboot_pending",
        "windows_update_reboot_required",
    }
    if triggered.isdisjoint(restart_specific) and triggered == {
        "pending_file_rename_operations"
    }:
        return (
            "generic PendingFileRenameOperations signal only; Windows Update "
            "RebootRequired and CBS markers are clear. Reboot at a convenient "
            "maintenance point; automatic resume remains required"
        )
    if not triggered.isdisjoint(restart_specific):
        return (
            "a Windows Update or CBS reboot-specific marker is present; schedule "
            "a controlled reboot when practical and retain automatic resume"
        )
    return (
        "one or more generic restart signals are present; inspect audit.json, "
        "reboot at a convenient maintenance point, and retain automatic resume"
    )


def render_readme(audit: Mapping[str, Any], *, report_dir_name: str) -> str:
    active = audit.get("active_hours", {})
    effective = active.get("effective", {})
    pre_action_effective = audit.get("pre_action_active_hours", {}).get(
        "effective", {}
    )
    pending_restart = audit.get("pending_restart", {})
    action = audit.get("action", {"mode": "audit", "status": "read_only"})
    gpu_devices = audit.get("gpu", {}).get("devices", [])
    gpu_lines = [
        (
            f"- GPU {index}: {device.get('name')}，driver {device.get('driver_version')}，"
            f"显存 {device.get('memory_used_mib')}/{device.get('memory_total_mib')} MiB，"
            f"利用率 {device.get('utilization_percent')}%，温度 {device.get('temperature_c')} °C。"
        )
        for index, device in enumerate(gpu_devices)
    ] or [f"- GPU 查询状态：`{audit.get('gpu', {}).get('status', 'unknown')}`。"]
    storage_lines = []
    for volume in audit.get("storage", {}).get("volumes", []):
        if volume.get("status") == "ok":
            storage_lines.append(
                f"- `{volume['label']}`：free {volume['free_gib']} GiB / "
                f"total {volume['total_gib']} GiB，used {volume['used_percent']}%。"
            )
        else:
            storage_lines.append(
                f"- `{volume['label']}`：查询状态 `{volume.get('status')}`。"
            )
    set_command = (
        "python tools/audit_windows_training_window.py "
        f"--report-dir reports/{report_dir_name} "
        f"--set-active-hours {DEFAULT_RECOMMENDED_START} {DEFAULT_RECOMMENDED_END} "
        "--confirm-active-hours-change"
    )
    restore_command = (
        "python tools/audit_windows_training_window.py "
        f"--report-dir reports/{report_dir_name} "
        f"--restore-active-hours reports/{report_dir_name}/active_hours_backup.json "
        "--confirm-active-hours-change"
    )
    return f"""# P0-B Windows 训练窗口预检

生成时间：`{audit.get('generated_at')}`。

## 当前结论

- Windows Update active hours：`{_format_window(effective)}`。
- 本次动作前原值：`{_format_window(pre_action_effective)}`。
- Pending restart：`{_pending_text(pending_restart)}`。
- 本次动作：`{action.get('mode')}` / `{action.get('status')}`。
- active hours 只能降低时段内自动重启概率，并不是训练不中断保证；正式训练仍必须使用原子 checkpoint 和自动续训。

## GPU 与存储

{chr(10).join(gpu_lines)}
{chr(10).join(storage_lines)}

## 建议与恢复

建议训练窗口使用 `{DEFAULT_RECOMMENDED_START:02d}:00–{DEFAULT_RECOMMENDED_END:02d}:00`（{MAX_ACTIVE_HOURS} 小时上限），把较可能的自动重启窗口留在白天。若当前窗口已经是该值，不必重复设置。设置命令必须在提升权限的终端中运行；权限不足、策略覆盖、自动 active hours、原值不可读或备份失败时，工具会拒绝写入并在 `audit.json` 留证：

    {set_command}

如果设置成功，工具会先生成不可覆盖的 `active_hours_backup.json`；同一报告目录再次设置会被拒绝，以免丢失原值。恢复原值：

    {restore_command}

工具只允许写入固定 Windows Update UX 键中的 `ActiveHoursStart` 与 `ActiveHoursEnd`，不禁用 Windows Update 服务，不改计划任务，也不写策略键。若企业/本机策略覆盖 active hours，工具会 fail closed，需由管理员按组织策略处理。

## 隐私与证据边界

报告不记录主机名、用户名、绝对路径、IP/内网地址、GPU UUID/序列号或进程列表。GPU 仅保留型号、驱动、显存、利用率和温度；磁盘仅用逻辑标签记录容量。`artifact_manifest.json` 为轻量报告文件提供 SHA-256，checkpoint、数据和原始系统日志不进入仓库。
"""


def _artifact_manifest(report_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(report_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "privacy_policy": "No host/user/path/network/GPU-serial identifiers are archived.",
        "artifacts": artifacts,
    }


def write_report(report_dir: Path, audit: Mapping[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(report_dir / "audit.json", audit)
    (report_dir / "README.md").write_text(
        render_readme(audit, report_dir_name=report_dir.name), encoding="utf-8"
    )
    _atomic_write_json(report_dir / "artifact_manifest.json", _artifact_manifest(report_dir))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/p0b_stability_preflight_20260813"),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--set-active-hours",
        nargs=2,
        metavar=("START", "END"),
        type=int,
    )
    action.add_argument("--restore-active-hours", type=Path)
    parser.add_argument("--confirm-active-hours-change", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report_dir = args.report_dir.resolve()
    registry = WindowsRegistry() if sys.platform == "win32" else None
    pre_audit = collect_audit(registry=registry)
    action: dict[str, Any] = {"mode": "audit", "status": "read_only"}

    if args.set_active_hours is not None:
        if registry is None:
            action = {"mode": "set_active_hours", "status": "unsupported_non_windows"}
        else:
            start, end = args.set_active_hours
            try:
                action = apply_active_hours(
                    registry,
                    start=start,
                    end=end,
                    confirmed=args.confirm_active_hours_change,
                    backup_path=report_dir / "active_hours_backup.json",
                    created_at=pre_audit["generated_at"],
                )
            except ValueError as exc:
                action = {
                    "mode": "set_active_hours",
                    "status": "refused_invalid_request",
                    "reason": str(exc),
                }
    elif args.restore_active_hours is not None:
        if registry is None:
            action = {
                "mode": "restore_active_hours",
                "status": "unsupported_non_windows",
            }
        else:
            try:
                action = restore_active_hours(
                    registry,
                    backup_path=args.restore_active_hours.resolve(),
                    confirmed=args.confirm_active_hours_change,
                )
            except ValueError as exc:
                action = {
                    "mode": "restore_active_hours",
                    "status": "refused_invalid_request",
                    "reason": str(exc),
                }

    post_audit = collect_audit(registry=registry)
    post_audit["pre_action_active_hours"] = pre_audit["active_hours"]
    post_audit["action"] = action
    post_audit["recommendation"] = {
        "start": DEFAULT_RECOMMENDED_START,
        "end": DEFAULT_RECOMMENDED_END,
        "duration_hours": active_hours_duration(
            DEFAULT_RECOMMENDED_START, DEFAULT_RECOMMENDED_END
        ),
        "warning": "Active hours reduce restart risk but do not guarantee continuity.",
    }
    write_report(report_dir, post_audit)
    print(json.dumps(action, ensure_ascii=False, sort_keys=True))
    return 0 if action.get("status") in {"read_only", "applied"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
