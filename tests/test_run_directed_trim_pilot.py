from tools.run_directed_trim_pilot import selected_cad_ids


def test_selected_cad_ids_freezes_saved_strict_invalid_reference_subset():
    rows = [
        {"cad_id": "b", "arm": "original", "native_brep_valid": False, "strict_brep_valid": False},
        {"cad_id": "a", "arm": "original", "native_brep_valid": True, "strict_brep_valid": False},
        {"cad_id": "c", "arm": "original", "native_brep_valid": None, "strict_brep_valid": False},
        {"cad_id": "d", "arm": "other", "native_brep_valid": False, "strict_brep_valid": False},
    ]
    assert selected_cad_ids(rows, reference_arm="original") == ["a", "b"]
