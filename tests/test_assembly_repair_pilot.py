from tools.run_assembly_repair_pilot import select_matched_invalid_rows


def test_selects_saved_invalid_reference_and_matching_arms():
    rows = [
        {"cad_id": "a", "arm": "original", "step_saved": True, "strict_brep_valid": False},
        {"cad_id": "a", "arm": "bypass", "step_saved": True, "strict_brep_valid": True},
        {"cad_id": "b", "arm": "original", "step_saved": False, "strict_brep_valid": False},
    ]
    selected = select_matched_invalid_rows(rows)
    assert [(r["cad_id"], r["arm"]) for r in selected] == [("a", "original"), ("a", "bypass")]


def test_selects_audited_rows_without_source_step_saved_field():
    rows = [
        {"cad_id": "a", "arm": "original", "native_brep_valid": False, "strict_brep_valid": False},
        {"cad_id": "a", "arm": "bypass", "native_brep_valid": True, "strict_brep_valid": True},
    ]
    assert len(select_matched_invalid_rows(rows)) == 2
