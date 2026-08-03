import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_split_integrity.py"
SPEC = importlib.util.spec_from_file_location("audit_split_integrity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _group(relpath, length):
    return {
        "source_relpath": relpath,
        "original": {"input_ids": list(range(length))},
    }


def test_extracts_parent_cad_id_from_step_part_filename():
    path = "abc_0000/00000140_2fc54fcd110d4f49969163c4_step_003.pkl"

    assert MODULE.parent_cad_id(path) == "2fc54fcd110d4f49969163c4"


def test_reports_parent_overlap_and_affected_record_counts():
    package = {
        "train": [
            _group("abc_0000/00000001_aaaaaaaaaaaaaaaaaaaaaaaa_step_000.pkl", 100),
            _group("abc_0000/00000002_bbbbbbbbbbbbbbbbbbbbbbbb_step_000.pkl", 200),
        ],
        "val": [
            _group("abc_0000/00000003_aaaaaaaaaaaaaaaaaaaaaaaa_step_001.pkl", 150),
        ],
        "test": [
            _group("abc_0000/00000004_cccccccccccccccccccccccc_step_000.pkl", 250),
        ],
    }

    report = MODULE.audit_package(package, source="fixture.pkl")

    assert report["splits"]["train"]["records"] == 2
    assert report["splits"]["train"]["sequence_length"]["median"] == 150
    assert report["pairwise"]["train__val"]["exact_source_path_overlap"] == 0
    assert report["pairwise"]["train__val"]["parent_cad_overlap"] == 1
    assert report["pairwise"]["train__val"]["train_records_in_shared_parent_cads"] == 1
    assert report["pairwise"]["train__val"]["val_records_in_shared_parent_cads"] == 1
    assert report["splits"]["val"]["records_sharing_parent_with_other_split"] == 1
    assert report["splits"]["val"]["records_sharing_parent_with_other_split_fraction"] == 1.0
    assert report["all_three_parent_cad_overlap"] == 0


def test_accepts_plain_split_path_lists_without_sequence_lengths():
    split = {
        "train": [r"D:\\pool\\train\\00000001_aaaaaaaaaaaaaaaaaaaaaaaa_step_000.pkl"],
        "val": [r"D:\\pool\\val\\00000002_aaaaaaaaaaaaaaaaaaaaaaaa_step_001.pkl"],
        "test": [],
    }

    report = MODULE.audit_package(split, source="split.pkl")

    assert report["package_kind"] == "path_split"
    assert report["pairwise"]["train__val"]["parent_cad_overlap"] == 1
    assert report["splits"]["train"]["sequence_length"] is None
