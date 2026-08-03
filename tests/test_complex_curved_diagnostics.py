import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path


class ComplexCurvedDiagnosticsTests(unittest.TestCase):
    def test_source_relpath_from_server_uri(self):
        from tools.complex_curved_diagnostics import source_relpath_from_group

        group = {
            "source_path": "parsed-shard://parsed_abc_0007.pkl.zst!/abc_0007/example_step_000.pkl",
            "original": {"input_ids": [1, 2, 3]},
        }

        self.assertEqual(source_relpath_from_group(group), "abc_0007/example_step_000.pkl")

    def test_source_relpath_prefers_explicit_relpath(self):
        from tools.complex_curved_diagnostics import source_relpath_from_group

        group = {
            "source_relpath": r"abc_0003\shape.pkl",
            "source_path": "parsed-shard://parsed_abc_0007.pkl.zst!/abc_0007/wrong.pkl",
        }

        self.assertEqual(source_relpath_from_group(group), "abc_0003/shape.pkl")

    def test_archive_path_and_loading_from_zip(self):
        from tools.complex_curved_diagnostics import archive_path_for_relpath, load_parsed_from_archive

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "abc_0002_parsed.zip"
            payload = {"surf_ncs": "surface", "edge_ncs": "edge"}
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("abc_0002/example.pkl", pickle.dumps(payload))

            self.assertEqual(archive_path_for_relpath("abc_0002/example.pkl", root), archive_path)
            self.assertEqual(load_parsed_from_archive("abc_0002/example.pkl", root), payload)

    def test_bucket_labels_are_stable(self):
        from tools.complex_curved_diagnostics import edge_count_bucket, face_count_bucket, sequence_length_bucket

        self.assertEqual(sequence_length_bucket(512), "len_0000_0512")
        self.assertEqual(sequence_length_bucket(1537), "len_1537_2048")
        self.assertEqual(face_count_bucket(12), "faces_12_19")
        self.assertEqual(face_count_bucket(51), "faces_gt_50")
        self.assertEqual(edge_count_bucket(20), "edges_20_39")
        self.assertEqual(edge_count_bucket(151), "edges_gt_150")

    def test_skipped_stage_report_is_explicit(self):
        from tools.complex_curved_diagnostics import skipped_stage_report

        report = skipped_stage_report("ar_teacher_forcing", "new FSQ checkpoint has incompatible tokens")

        self.assertTrue(report["skipped"])
        self.assertEqual(report["stage"], "ar_teacher_forcing")
        self.assertEqual(report["reason"], "new FSQ checkpoint has incompatible tokens")
        self.assertEqual(report["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
