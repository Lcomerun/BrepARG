import unittest


class GenerationQualityGateTests(unittest.TestCase):
    def test_accepts_complex_watertight_solid_with_preview(self):
        from tools.generation_quality_gate import quality_gate_decision

        row = {"grammar_ok": True, "grammar_faces": 18, "grammar_edges": 42, "step_saved": True}
        quality = {
            "brep_valid": True,
            "solid_closed_no_open_shell": True,
            "png_saved": True,
            "advanced_faces": 18,
            "edge_curves": 42,
        }

        decision = quality_gate_decision(row, quality, min_faces=12, min_edges=20)

        self.assertTrue(decision["accept"])
        self.assertEqual(decision["reasons"], [])

    def test_rejects_non_watertight_candidate(self):
        from tools.generation_quality_gate import quality_gate_decision

        row = {"grammar_ok": True, "grammar_faces": 18, "grammar_edges": 42, "step_saved": True}
        quality = {
            "brep_valid": False,
            "solid_closed_no_open_shell": False,
            "png_saved": True,
            "advanced_faces": 18,
            "edge_curves": 42,
        }

        decision = quality_gate_decision(row, quality, min_faces=12, min_edges=20)

        self.assertFalse(decision["accept"])
        self.assertIn("brep_not_valid", decision["reasons"])
        self.assertIn("not_solid_closed", decision["reasons"])

    def test_rejects_primitive_like_or_too_simple_candidate(self):
        from tools.generation_quality_gate import quality_gate_decision

        row = {"grammar_ok": True, "grammar_faces": 6, "grammar_edges": 12, "step_saved": True}
        quality = {
            "brep_valid": True,
            "solid_closed_no_open_shell": True,
            "png_saved": True,
            "advanced_faces": 6,
            "edge_curves": 12,
        }

        decision = quality_gate_decision(row, quality, min_faces=12, min_edges=20)

        self.assertFalse(decision["accept"])
        self.assertIn("too_simple", decision["reasons"])
        self.assertIn("primitive_like", decision["reasons"])

    def test_rejects_boundary_over_caps(self):
        from tools.generation_quality_gate import quality_gate_decision

        row = {"grammar_ok": True, "grammar_faces": 50, "grammar_edges": 144, "step_saved": True}
        quality = {
            "brep_valid": True,
            "solid_closed_no_open_shell": True,
            "png_saved": True,
            "advanced_faces": 50,
            "edge_curves": 144,
        }

        decision = quality_gate_decision(row, quality, min_faces=12, min_edges=20, max_faces=45, max_edges=120)

        self.assertFalse(decision["accept"])
        self.assertIn("too_many_faces", decision["reasons"])
        self.assertIn("too_many_edges", decision["reasons"])

    def test_v13_quality_gate_cli_can_allow_primitive_like_outputs(self):
        from pathlib import Path

        source = Path("tools/generate_quality_gated_step_png.py").read_text(encoding="utf-8")

        self.assertIn("--allow-primitive-like", source)
        self.assertIn("reject_primitive_like=not bool(args.allow_primitive_like)", source)


if __name__ == "__main__":
    unittest.main()
