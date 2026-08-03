import unittest


class CompareFsqCapacityDiagnosticsTests(unittest.TestCase):
    def test_compare_reports_computes_improvement_and_recommendation(self):
        from tools.compare_fsq_capacity_diagnostics import compare_reports, render_markdown

        baseline = {
            "selected_count": 50,
            "fsq_patch_metrics": {
                "patch_count": 100,
                "chamfer": {"median": 0.02, "p95": 0.20, "max": 1.5},
                "by_kind": {
                    "surface": {"chamfer": {"p95": 0.40, "max": 1.2}},
                    "edge": {"chamfer": {"p95": 0.10, "max": 0.8}},
                },
            },
        }
        candidate = {
            "selected_count": 50,
            "fsq_patch_metrics": {
                "patch_count": 100,
                "chamfer": {"median": 0.015, "p95": 0.14, "max": 0.9},
                "by_kind": {
                    "surface": {"chamfer": {"p95": 0.24, "max": 0.7}},
                    "edge": {"chamfer": {"p95": 0.09, "max": 0.6}},
                },
            },
        }

        comparison = compare_reports(baseline, candidate)

        self.assertEqual(comparison["status"], "VERIFIED")
        self.assertAlmostEqual(comparison["metrics"]["fsq_chamfer_p95"]["relative_change_pct"], -30.0)
        self.assertAlmostEqual(comparison["metrics"]["surface_chamfer_p95"]["relative_change_pct"], -40.0)
        self.assertEqual(comparison["recommendation"]["capacity_signal"], "strong_improvement")

        markdown = render_markdown(comparison)
        self.assertIn("FSQ Capacity Comparison", markdown)
        self.assertIn("surface_chamfer_p95", markdown)
        self.assertIn("strong_improvement", markdown)

    def test_compare_reports_marks_regression(self):
        from tools.compare_fsq_capacity_diagnostics import compare_reports

        baseline = {
            "selected_count": 50,
            "fsq_patch_metrics": {
                "patch_count": 100,
                "chamfer": {"p95": 0.20},
                "by_kind": {
                    "surface": {"chamfer": {"p95": 0.40}},
                    "edge": {"chamfer": {"p95": 0.10}},
                },
            },
        }
        candidate = {
            "selected_count": 50,
            "fsq_patch_metrics": {
                "patch_count": 100,
                "chamfer": {"p95": 0.25},
                "by_kind": {
                    "surface": {"chamfer": {"p95": 0.50}},
                    "edge": {"chamfer": {"p95": 0.12}},
                },
            },
        }

        comparison = compare_reports(baseline, candidate)

        self.assertEqual(comparison["recommendation"]["capacity_signal"], "regression")
        self.assertIn("worse", comparison["recommendation"]["reading"])


if __name__ == "__main__":
    unittest.main()
