from tools.analyze_paired_assembly_ceiling import paired_ceiling


def test_paired_ceiling_separates_retention_and_recovery():
    rows = [
        {"cad_id": "a", "arm": "original", "native_brep_valid": True, "strict_brep_valid": True},
        {"cad_id": "b", "arm": "original", "native_brep_valid": False, "strict_brep_valid": False},
        {"cad_id": "a", "arm": "model", "native_brep_valid": True, "strict_brep_valid": False},
        {"cad_id": "b", "arm": "model", "native_brep_valid": True, "strict_brep_valid": True},
    ]
    report = paired_ceiling(rows, reference_arm="original")["arms"]["model"]
    assert report["strict_valid_retained"] == 0
    assert report["strict_invalid_recovered"] == 1
    assert report["both_valid_retention_rate"] == 0.0
