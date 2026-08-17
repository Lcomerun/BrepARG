import pytest

from tools.assembly_repair import RepairProfile
from tools.run_assembly_repair_matrix import (
    profile_kwargs,
    summarize_matrix,
    summarize_profile,
)


def _rows(profile, values):
    return [
        {
            "profile": profile, "cad_id": cad_id, "strict_brep_valid": strict,
            "native_brep_valid": strict, "both_valid": strict,
            "step_saved": True, "status": "both_valid" if strict else "step_invalid",
        }
        for cad_id, strict in values.items()
    ]


def test_profile_kwargs_keep_switches_independent():
    assert profile_kwargs(RepairProfile("baseline")) == {
        "directed_trim": False, "curve_fit_fallback": False,
        "wire_continuity": False, "single_solid": False,
        "pcurve_self_intersection": False,
    }
    assert profile_kwargs(RepairProfile("directed_trim", ("directed_trim",)))[
        "directed_trim"
    ] is True


def test_summary_records_restoration_and_regression():
    historical = {"kept": True, "lost": True, "restored": False, "still_bad": False}
    summary = summarize_profile(
        _rows("candidate", {"kept": True, "lost": False, "restored": True, "still_bad": False}),
        historical,
    )
    assert summary["restored_cad_ids"] == ["restored"]
    assert summary["regressed_cad_ids"] == ["lost"]
    assert summary["preserves_original_84"] is False


def test_gate_requires_95_and_zero_regression():
    historical = {f"cad{i:03d}": i < 84 for i in range(100)}
    candidate = {cad_id: old or int(cad_id[-3:]) < 95 for cad_id, old in historical.items()}
    summary = summarize_matrix(
        _rows("combined", candidate), [RepairProfile("combined")], historical
    )
    assert summary["gate_passed"] is True
    assert summary["profiles"][0]["strict_valid"] == 95
    assert len(summary["profiles"][0]["restored_cad_ids"]) == 11

    candidate["cad000"] = False
    failed = summarize_profile(_rows("combined", candidate), historical)
    assert failed["strict_valid"] == 94
    assert failed["meets_95_gate"] is False


def test_profile_summary_rejects_incomplete_cohort():
    with pytest.raises(ValueError, match="full frozen cohort"):
        summarize_profile(_rows("x", {"a": True}), {"a": True, "b": False})
