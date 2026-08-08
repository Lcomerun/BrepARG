from tools.summarize_step_validity_audit import build_report


def test_build_report_preserves_attempts_and_arm_breakdown():
    report = build_report([
        {"arm": "a", "native_brep_valid": True, "strict_brep_valid": True},
        {"arm": "a", "native_brep_valid": None, "strict_brep_valid": False},
    ])
    assert report["attempts_denominator"] == 2
    assert report["arms"]["a"]["both_valid"] == 1
    assert report["arms"]["a"]["no_step"] == 1
