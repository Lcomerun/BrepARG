import unittest


class SummarizeComplexCurvedDiagnosticsTests(unittest.TestCase):
    def test_render_summary_includes_reconstruction_failure_detail(self):
        from tools.summarize_complex_curved_diagnostics import render_summary

        report = {
            "selected_count": 4,
            "fsq_patch_metrics": {
                "patch_count": 10,
                "mse": {"median": 0.1, "p95": 0.9},
                "chamfer": {"median": 0.2, "p95": 1.1},
            },
            "ar_teacher_forcing": {"token_weighted_ce": 0.7},
            "teacher_reconstruction": {
                "attempted": 4,
                "step_saved": 2,
                "brep_valid": 1,
                "errors": 2,
                "by_face_bucket": {
                    "faces_12_19": {"attempted": 3, "step_saved": 2, "brep_valid": 1, "errors": 1},
                    "faces_30_50": {"attempted": 1, "step_saved": 0, "brep_valid": 0, "errors": 1},
                },
                "by_length_bucket": {
                    "len_0000_0512": {"attempted": 2, "step_saved": 2, "brep_valid": 1, "errors": 0},
                    "len_1537_2048": {"attempted": 2, "step_saved": 0, "brep_valid": 0, "errors": 2},
                },
            },
        }

        markdown = render_summary([("synthetic", report)])

        self.assertIn("## Reconstruction Detail", markdown)
        self.assertIn("faces_12_19", markdown)
        self.assertIn("1/3", markdown)
        self.assertIn("len_1537_2048", markdown)
        self.assertIn("0/2", markdown)


if __name__ == "__main__":
    unittest.main()
