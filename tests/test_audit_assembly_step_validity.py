import json

from tools.audit_assembly_step_validity import _resolve_step_path, summarize_validity_rows


def test_summarize_validity_rows_keeps_native_and_strict_denominators():
    rows = [
        {"arm": "original", "native_brep_valid": True, "strict_brep_valid": True, "status": "saved"},
        {"arm": "original", "native_brep_valid": True, "strict_brep_valid": False, "status": "saved"},
        {"arm": "original", "native_brep_valid": False, "strict_brep_valid": True, "status": "saved"},
        {"arm": "original", "native_brep_valid": None, "strict_brep_valid": False, "status": "no_step"},
    ]
    summary = summarize_validity_rows(rows)
    assert summary["attempts"] == 4
    assert summary["step_saved"] == 3
    assert summary["native_brep_valid"] == 2
    assert summary["strict_brep_valid"] == 2
    assert summary["native_true_strict_false"] == 1
    assert summary["native_false_strict_true"] == 1
    assert summary["both_valid"] == 1
    assert summary["native_only"] == 1
    assert summary["strict_only"] == 1
    assert summary["neither_valid"] == 0
    assert summary["no_step"] == 1
    assert summary["by_arm"]["original"]["attempts"] == 4


def test_step_root_fallback_supports_arm_subdirectory(tmp_path):
    step_root = tmp_path / "steps"
    step_path = step_root / "original_direct_wcs" / "cad-1.step"
    step_path.parent.mkdir(parents=True)
    step_path.write_text("STEP", encoding="ascii")
    resolved = _resolve_step_path(
        {"cad_id": "cad-1", "arm": "original_direct_wcs"},
        manifest_path=tmp_path / "manifest.jsonl",
        step_root=step_root,
    )
    assert resolved == step_path
