import json
import tempfile
import unittest
from pathlib import Path


class AnalyzeReconstructionFsqCorrelationTests(unittest.TestCase):
    def test_analyze_links_shape_chamfer_to_reconstruction_status(self):
        from tools.analyze_reconstruction_fsq_correlation import analyze_correlation

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "teacher_reconstruction_manifest.jsonl"
            patches = root / "fsq_patch_metrics.jsonl"
            manifest_rows = [
                {
                    "source_relpath": "abc_0000/a.pkl",
                    "status": "saved",
                    "step_saved": True,
                    "brep_valid": True,
                    "grammar_faces": 12,
                    "grammar_edges": 24,
                    "sequence_length": 400,
                },
                {
                    "source_relpath": "abc_0000/b.pkl",
                    "status": "reconstruct_failed",
                    "step_saved": False,
                    "brep_valid": False,
                    "grammar_faces": 30,
                    "grammar_edges": 80,
                    "sequence_length": 1500,
                },
            ]
            patch_rows = [
                {"source_relpath": "abc_0000/a.pkl", "chamfer": 0.01, "mse": 0.001, "kind": "surface"},
                {"source_relpath": "abc_0000/a.pkl", "chamfer": 0.02, "mse": 0.002, "kind": "edge"},
                {"source_relpath": "abc_0000/b.pkl", "chamfer": 0.50, "mse": 0.050, "kind": "surface"},
                {"source_relpath": "abc_0000/b.pkl", "chamfer": 1.00, "mse": 0.100, "kind": "edge"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
            patches.write_text("\n".join(json.dumps(row) for row in patch_rows) + "\n", encoding="utf-8")

            report = analyze_correlation(manifest, patches, top_k=1)

            self.assertEqual(report["shape_count"], 2)
            self.assertEqual(report["groups"]["brep_valid"]["count"], 1)
            self.assertEqual(report["groups"]["reconstruct_failed"]["count"], 1)
            self.assertGreater(
                report["groups"]["reconstruct_failed"]["chamfer_p95"]["median"],
                report["groups"]["brep_valid"]["chamfer_p95"]["median"],
            )
            self.assertEqual(report["top_chamfer_p95"][0]["source_relpath"], "abc_0000/b.pkl")


if __name__ == "__main__":
    unittest.main()
