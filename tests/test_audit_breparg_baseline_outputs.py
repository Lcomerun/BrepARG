import json
import tempfile
import unittest
from pathlib import Path


def write_fake_step(path: Path, faces: int, edges: int, *, curved: bool = False, open_shell: bool = False) -> None:
    lines = [
        "ISO-10303-21;",
        "DATA;",
        "#1=MANIFOLD_SOLID_BREP('',#2);",
        "#2=CLOSED_SHELL('',());" if not open_shell else "#2=OPEN_SHELL('',());",
    ]
    surface = "CYLINDRICAL_SURFACE" if curved else "PLANE"
    for idx in range(faces):
        lines.append(f"#{100 + idx}=ADVANCED_FACE('',(),#{200 + idx},.T.);")
        lines.append(f"#{200 + idx}={surface}('',#300,{1.0 + idx});")
    for idx in range(edges):
        lines.append(f"#{400 + idx}=EDGE_CURVE('',#500,#501,#600,.T.);")
        lines.append(f"#{600 + idx}=CIRCLE('',#700,{1.0 + idx});")
    lines.extend(["ENDSEC;", "END-ISO-10303-21;"])
    path.write_text("\n".join(lines), encoding="utf-8")


class AuditBrepArgbBaselineOutputsTests(unittest.TestCase):
    def test_audits_upstream_flat_output_directory_and_writes_reports(self):
        from tools.audit_breparg_baseline_outputs import audit_breparg_baseline_outputs, write_outputs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "breparg_flat"
            run_dir.mkdir()
            write_fake_step(run_dir / "sample_000.step", faces=6, edges=12)
            write_fake_step(run_dir / "sample_001.step", faces=14, edges=24, curved=True)
            (run_dir / "sample_000.png").write_bytes(b"png")
            (run_dir / "sample_001.png").write_bytes(b"png")
            (run_dir / "sample_000.stl").write_bytes(b"stl")

            audit = audit_breparg_baseline_outputs(
                run_dir,
                require_quality=False,
                min_faces=12,
                min_edges=20,
            )

            self.assertEqual(audit["summary"]["step_files"], 2)
            self.assertEqual(audit["summary"]["png_files"], 2)
            self.assertEqual(audit["summary"]["stl_files"], 1)
            self.assertEqual(audit["summary"]["strict_quality_accepted"], 1)
            self.assertEqual(audit["summary"]["simple_or_rejected"], 1)
            self.assertEqual(audit["summary"]["complex_by_step_entities_12faces_or_20edges"], 1)
            self.assertEqual(audit["summary"]["complex_and_closed"], 1)
            self.assertEqual(audit["summary"]["complex_and_brep_valid"], 0)
            self.assertEqual(audit["summary"]["files_solid_closed_no_open_shell"], 2)
            self.assertEqual(audit["face_stats"]["median"], 6)
            self.assertEqual(audit["face_stats"]["max"], 14)
            self.assertEqual(audit["edge_stats"]["median"], 12)
            self.assertEqual(audit["edge_stats"]["max"], 24)
            self.assertEqual(len(audit["entries"]), 2)

            output_json = run_dir / "audit.json"
            output_md = run_dir / "audit.md"
            output_manifest = run_dir / "audit.jsonl"
            write_outputs(audit, output_json, output_md, output_manifest)

            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            self.assertTrue(output_manifest.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["step_files"], 2)
            self.assertIn("BrepARG Baseline Output Audit", output_md.read_text(encoding="utf-8"))
            self.assertEqual(len(output_manifest.read_text(encoding="utf-8").splitlines()), 2)

    def test_can_use_quality_manifest_when_present(self):
        from tools.audit_breparg_baseline_outputs import audit_breparg_baseline_outputs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "breparg_with_quality"
            quality_dir = run_dir / "quality_check"
            run_dir.mkdir()
            quality_dir.mkdir()
            write_fake_step(run_dir / "sample_000.step", faces=14, edges=24, curved=True)
            (quality_dir / "step_quality_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "step": str(run_dir / "sample_000.step"),
                        "step_read_ok": True,
                        "brep_valid": True,
                        "solid_closed_no_open_shell": True,
                        "png_saved": True,
                        "advanced_faces": 14,
                        "edge_curves": 24,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = audit_breparg_baseline_outputs(run_dir, min_faces=12, min_edges=20)

            self.assertEqual(audit["summary"]["quality_manifest_rows"], 1)
            self.assertEqual(audit["summary"]["brep_valid"], 1)
            self.assertEqual(audit["summary"]["strict_quality_accepted"], 1)
            self.assertEqual(audit["summary"]["complex_and_brep_valid"], 1)

    def test_reads_prefixed_quality_fields_from_existing_v13_manifests(self):
        from tools.audit_breparg_baseline_outputs import audit_breparg_baseline_outputs

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "breparg_with_prefixed_quality"
            quality_dir = run_dir / "quality_check"
            run_dir.mkdir()
            quality_dir.mkdir()
            step_path = run_dir / "sample_000.step"
            write_fake_step(step_path, faces=14, edges=24, curved=True)
            (quality_dir / "step_quality_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "step_path": str(step_path),
                        "brep_valid": False,
                        "quality_step_read_ok": True,
                        "quality_brep_valid": True,
                        "quality_solid_closed_no_open_shell": True,
                        "quality_png_existing": True,
                        "quality_advanced_faces": 14,
                        "quality_edge_curves": 24,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = audit_breparg_baseline_outputs(run_dir, min_faces=12, min_edges=20)

            self.assertEqual(audit["summary"]["quality_manifest_rows"], 1)
            self.assertEqual(audit["summary"]["brep_valid"], 1)
            self.assertEqual(audit["summary"]["strict_quality_accepted"], 1)


if __name__ == "__main__":
    unittest.main()
