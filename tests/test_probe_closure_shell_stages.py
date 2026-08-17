from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.probe_closure_shell_stages import (
    SCHEMA,
    VARIANT_PROFILES,
    classify_first_defective_stage,
    parse_worker_result,
    run_isolated,
    summarize,
    validate_result,
)


def event(stage: str, **components: object) -> dict[str, object]:
    return {"stage": stage, "in_memory": dict(components)}


@pytest.mark.parametrize(
    ("events", "status", "components", "error", "expected"),
    [
        ([{"stage": "curve_fit_failure"}], "assembly_error", {}, None, "curve_fit"),
        ([event("face_raw", wire_self_intersections=1)], "assembly_error", {}, None, "face_raw"),
        ([event("faces_compound", wire_self_intersections=2)], "assembly_error", {}, None, "faces_compound"),
        ([event("sewn_shape", shell_count=5)], "assembly_error", {}, None, "sewing_shell_count"),
        ([event("sewn_shape", shell_count=1, wire_self_intersections=1)], "assembly_error", {}, None, "sewing_wire_self_intersection"),
        ([event("solid", solid_count=0)], "step_invalid", {}, None, "solid_construction"),
        ([], "step_invalid", {"solid_count": 0}, None, "step_roundtrip_solid_count"),
        ([], "step_invalid", {"solid_count": 1, "wire_self_intersections": 1}, None, "step_roundtrip_wire_self_intersection"),
        ([], "both_valid", {"solid_count": 1}, None, "none"),
        ([], "assembly_error", {}, "trim loop is open or branching at vertex 2", "source_topology_walk"),
    ],
)
def test_classify_first_defective_stage(
    events: list[dict[str, object]],
    status: str,
    components: dict[str, object],
    error: str | None,
    expected: str,
) -> None:
    assert (
        classify_first_defective_stage(
            events,
            status=status,
            final_components=components,
            error=error,
        )
        == expected
    )


def base_row(*, cad_id: str = "cad", variant: str = "historical") -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "cad_id": cad_id,
        "parent_id": "parent",
        "variant": variant,
        "profile": VARIANT_PROFILES[variant].name,
        "status": "step_invalid",
        "step_saved": True,
        "native_brep_valid": False,
        "strict_brep_valid": False,
        "both_valid": False,
        "first_defective_stage": "step_roundtrip_native_validity",
    }


def test_validate_result_rejects_inconsistent_both_valid() -> None:
    row = base_row()
    row.update(native_brep_valid=True, strict_brep_valid=True, both_valid=False)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_result(
            row,
            source={"cad_id": "cad", "parent_id": "parent"},
            variant="historical",
        )


def test_parse_worker_result_uses_last_sentinel() -> None:
    stdout = "noise\n__CLOSURE_SHELL_STAGE_WORKER_RESULT__={\"value\": 1}\n"
    assert parse_worker_result(stdout) == {"value": 1}
    assert parse_worker_result("noise only") is None


def test_summarize_requires_both_valid_and_gate_for_invalid16() -> None:
    first = base_row(cad_id="a")
    first.update(
        status="both_valid",
        native_brep_valid=True,
        strict_brep_valid=True,
        both_valid=True,
        first_defective_stage="none",
        selector_geometry_topology_gate={"accepted": False},
    )
    second = base_row(cad_id="b")
    summary = summarize([first, second])
    assert summary["both_valid_count"] == 1
    assert summary["geometry_topology_gate_pass_count"] == 0
    assert summary["eligible_for_invalid16"] is False
    first["selector_geometry_topology_gate"] = {"accepted": True}
    assert summarize([first, second])["eligible_for_invalid16"] is True


def test_run_isolated_turns_timeout_into_denominator_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.probe_closure_shell_stages as probe

    source_file = tmp_path / "source.pkl"
    source_file.write_bytes(b"payload")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    def timeout(*args: object, **kwargs: object) -> object:
        raise probe.subprocess.TimeoutExpired("worker", 1.0, output=b"partial", stderr=b"err")

    monkeypatch.setattr(probe.subprocess, "run", timeout)
    row = run_isolated(
        source={
            "cad_id": "cad",
            "parent_id": "parent",
            "source_path": str(source_file),
        },
        variant="historical",
        calibration_manifest=manifest,
        breparg_root=tmp_path,
        output_dir=tmp_path / "run",
        joint_iterations=200,
        timeout_seconds=1.0,
    )
    assert row["status"] == "worker_timeout"
    assert row["both_valid"] is False
    assert row["first_defective_stage"] == "worker_or_unclassified"


def test_run_isolated_rejects_wrong_worker_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.probe_closure_shell_stages as probe

    source_file = tmp_path / "source.pkl"
    source_file.write_bytes(b"payload")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    wrong = base_row(cad_id="different")
    wrong["source_pickle_binding"] = probe.source_pickle_binding(source_file)
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            probe.WORKER_MARKER + json.dumps(wrong, sort_keys=True) + "\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(ValueError, match="cad_id mismatch"):
        run_isolated(
            source={
                "cad_id": "cad",
                "parent_id": "parent",
                "source_path": str(source_file),
            },
            variant="historical",
            calibration_manifest=manifest,
            breparg_root=tmp_path,
            output_dir=tmp_path / "run",
            joint_iterations=200,
            timeout_seconds=1.0,
        )


def test_run_isolated_binds_requested_cad_id_in_worker_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.probe_closure_shell_stages as probe

    source_file = tmp_path / "source.pkl"
    source_file.write_bytes(b"payload")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        captured["command"] = command
        row = base_row(cad_id="cad")
        row["source_pickle_binding"] = probe.source_pickle_binding(source_file)
        return SimpleNamespace(
            returncode=0,
            stdout=probe.WORKER_MARKER + json.dumps(row, sort_keys=True) + "\n",
            stderr="",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    run_isolated(
        source={
            "cad_id": "cad",
            "parent_id": "parent",
            "source_path": str(source_file),
        },
        variant="historical",
        calibration_manifest=manifest,
        breparg_root=tmp_path,
        output_dir=tmp_path / "run",
        joint_iterations=200,
        timeout_seconds=1.0,
    )
    command = captured["command"]
    assert isinstance(command, list)
    cad_arg = command.index("--cad-id")
    assert command[cad_arg + 1] == "cad"
