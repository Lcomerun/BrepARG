import json
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.audit_windows_training_window import (
    ACTIVE_HOUR_NAMES,
    POLICY_SETTINGS_KEY,
    POLICY_VALUE_NAMES,
    REG_DWORD_NAME,
    UX_SETTINGS_KEY,
    UX_VALUE_NAMES,
    _pending_text,
    apply_active_hours,
    collect_active_hours,
    collect_audit,
    collect_pending_restart,
    restore_active_hours,
    write_report,
)


DiskUsage = namedtuple("DiskUsage", "total used free")


class FakeRegistry:
    def __init__(
        self,
        *,
        start=8,
        end=17,
        policy=None,
        fail_write_call=None,
        persistent_write_failure=False,
    ):
        self.values = {
            UX_SETTINGS_KEY: {
                "ActiveHoursStart": (start, REG_DWORD_NAME),
                "ActiveHoursEnd": (end, REG_DWORD_NAME),
                "SmartActiveHoursState": (0, REG_DWORD_NAME),
            },
            POLICY_SETTINGS_KEY: policy or {},
        }
        self.existing_keys = set()
        self.writes = []
        self.fail_write_call = fail_write_call
        self.persistent_write_failure = persistent_write_failure
        self.write_attempts = 0

    def read_values(self, key_path, names):
        key_values = self.values.get(key_path, {})
        values = {}
        for name in names:
            if name in key_values:
                value, registry_type = key_values[name]
                values[name] = {
                    "present": True,
                    "value": value,
                    "registry_type": registry_type,
                }
            else:
                values[name] = {"present": False}
        return {"status": "ok", "values": values}

    def key_exists(self, key_path):
        return {"status": "ok", "present": key_path in self.existing_keys}

    def write_dword(self, key_path, name, value):
        self.write_attempts += 1
        if self.fail_write_call == self.write_attempts or (
            self.persistent_write_failure
            and self.fail_write_call is not None
            and self.write_attempts >= self.fail_write_call
        ):
            raise PermissionError("denied")
        assert key_path == UX_SETTINGS_KEY
        assert name in ACTIVE_HOUR_NAMES
        self.writes.append((name, value))
        self.values[key_path][name] = (value, REG_DWORD_NAME)


def _gpu_runner(command):
    assert command[0] == "nvidia-smi"
    return {
        "returncode": 0,
        "stdout": "NVIDIA GeForce RTX 3060, 999.1, 12288, 512, 11776, 7, 41\n",
    }


def _disk_usage(_path):
    return DiskUsage(total=100 * 1024**3, used=40 * 1024**3, free=60 * 1024**3)


def test_read_only_audit_is_privacy_bounded_and_writes_report(tmp_path):
    registry = FakeRegistry()
    audit = collect_audit(
        registry=registry,
        command_runner=_gpu_runner,
        storage_targets={"repository_volume": Path(r"C:\Users\Alice\private")},
        disk_usage=_disk_usage,
        platform_name="win32",
        now=datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc),
    )
    audit["action"] = {"mode": "audit", "status": "read_only"}
    write_report(tmp_path / "report", audit)

    serialized = json.dumps(audit)
    assert "Alice" not in serialized
    assert "private" not in serialized
    assert audit["active_hours"]["effective"] == {
        "source": "ux_settings",
        "start": 8,
        "end": 17,
        "duration_hours": 9,
        "valid": True,
    }
    assert audit["pending_restart"]["detected"] is False
    assert audit["gpu"]["devices"][0]["memory_total_mib"] == 12288
    assert audit["storage"]["volumes"][0]["free_gib"] == 60.0
    assert registry.writes == []
    assert (tmp_path / "report" / "artifact_manifest.json").is_file()


def test_apply_requires_confirmation_before_backup_or_registry_write(tmp_path):
    registry = FakeRegistry()

    with pytest.raises(ValueError, match="explicit confirmation"):
        apply_active_hours(
            registry,
            start=12,
            end=6,
            confirmed=False,
            backup_path=tmp_path / "backup.json",
        )

    assert registry.writes == []
    assert not (tmp_path / "backup.json").exists()


def test_apply_refuses_policy_override_without_writing(tmp_path):
    registry = FakeRegistry(
        policy={
            "SetActiveHours": (1, REG_DWORD_NAME),
            "ActiveHoursStart": (9, REG_DWORD_NAME),
            "ActiveHoursEnd": (3, REG_DWORD_NAME),
        }
    )

    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=tmp_path / "backup.json",
    )

    assert result["status"] == "refused_policy_override"
    assert registry.writes == []
    assert not (tmp_path / "backup.json").exists()


def test_malformed_policy_value_fails_closed_without_crashing_audit(tmp_path):
    registry = FakeRegistry(policy={"SetActiveHours": ("invalid", "REG_SZ")})

    active = collect_active_hours(registry)
    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=tmp_path / "backup.json",
    )

    assert active["effective"]["source"] == "unknown"
    assert active["policy_override_present"] is True
    assert result["status"] == "refused_policy_override"
    assert registry.writes == []


def test_apply_refuses_automatic_active_hours_without_writing(tmp_path):
    registry = FakeRegistry()
    registry.values[UX_SETTINGS_KEY]["SmartActiveHoursState"] = (1, REG_DWORD_NAME)

    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=tmp_path / "backup.json",
    )

    assert result["status"] == "refused_smart_active_hours_enabled"
    assert registry.writes == []
    assert not (tmp_path / "backup.json").exists()


def test_permission_failure_is_recorded_with_no_change(tmp_path):
    registry = FakeRegistry(fail_write_call=1)

    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=tmp_path / "backup.json",
    )

    assert result["status"] == "failed_no_change"
    assert result["error_category"] == "access_denied"
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursStart"][0] == 8
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursEnd"][0] == 17
    assert (tmp_path / "backup.json").is_file()


def test_persistent_permission_failure_after_partial_write_is_not_hidden(tmp_path):
    registry = FakeRegistry(fail_write_call=2, persistent_write_failure=True)

    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=tmp_path / "backup.json",
    )

    assert result["status"] == "rollback_incomplete"
    assert result["rollback_verified"] is False
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursStart"][0] == 12
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursEnd"][0] == 17


def test_partial_write_rolls_back_and_restore_replays_backup(tmp_path):
    backup_path = tmp_path / "backup.json"
    registry = FakeRegistry(fail_write_call=2)

    failed = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=backup_path,
    )

    assert failed["status"] == "failed_rolled_back"
    assert failed["rollback_verified"] is True
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursStart"][0] == 8
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursEnd"][0] == 17


def test_successful_apply_can_be_restored_from_immutable_backup(tmp_path):
    backup_path = tmp_path / "backup.json"
    registry = FakeRegistry()

    applied = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=backup_path,
    )
    assert applied["status"] == "applied"
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursStart"][0] == 12
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursEnd"][0] == 6

    restored = restore_active_hours(
        registry,
        backup_path=backup_path,
        confirmed=True,
    )
    assert restored["status"] == "applied"
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursStart"][0] == 8
    assert registry.values[UX_SETTINGS_KEY]["ActiveHoursEnd"][0] == 17


def test_existing_backup_is_never_overwritten(tmp_path):
    backup = tmp_path / "backup.json"
    backup.write_text('{"sentinel": true}\n', encoding="utf-8")
    registry = FakeRegistry()

    result = apply_active_hours(
        registry,
        start=12,
        end=6,
        confirmed=True,
        backup_path=backup,
    )

    assert result["status"] == "refused_backup_already_exists"
    assert backup.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert registry.writes == []


def test_restore_rejects_backup_with_arbitrary_registry_path(tmp_path):
    backup = tmp_path / "bad.json"
    backup.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "restore_windows_update_active_hours",
                "registry_scope": "HKLM",
                "registry_path": r"SOFTWARE\Other",
                "values": {},
            }
        ),
        encoding="utf-8",
    )
    registry = FakeRegistry()

    result = restore_active_hours(registry, backup_path=backup, confirmed=True)

    assert result["status"] == "refused_invalid_backup"
    assert registry.writes == []


def test_active_hour_queries_are_fixed_to_expected_names():
    assert UX_VALUE_NAMES == (
        "ActiveHoursStart",
        "ActiveHoursEnd",
        "SmartActiveHoursState",
    )
    assert POLICY_VALUE_NAMES == (
        "SetActiveHours",
        "ActiveHoursStart",
        "ActiveHoursEnd",
    )


def test_missing_optional_restart_value_means_absent_not_unknown():
    registry = FakeRegistry()

    result = collect_pending_restart(registry)

    assert result["detected"] is False


def test_pending_file_rename_signal_is_not_labeled_windows_update_reboot():
    message = _pending_text(
        {
            "detected": True,
            "checks": [
                {
                    "name": "component_based_servicing_reboot_pending",
                    "triggered": False,
                },
                {"name": "windows_update_reboot_required", "triggered": False},
                {"name": "pending_file_rename_operations", "triggered": True},
            ],
        }
    )

    assert "generic PendingFileRenameOperations signal only" in message
    assert "RebootRequired and CBS markers are clear" in message
    assert "automatic resume remains required" in message
