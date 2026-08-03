import json
import pickle
import tempfile
import unittest
from pathlib import Path


class BrepArgSameDataFallbackPreflightTests(unittest.TestCase):
    def test_preflight_reports_ready_without_starting_training(self):
        from tools.preflight_breparg_same_data_fallback import run_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            split = data / "same_data_split.pkl"
            surfaces = data / "deduplicated_surface_source.pkl"
            edges = data / "deduplicated_edge_source.pkl"
            summary = data / "same_data_input_summary.json"

            split_payload = {"train": [1, 2], "val": [3], "test": [4]}
            for path, payload in (
                (split, split_payload),
                (surfaces, [[0.0] * 3]),
                (edges, [[1.0] * 3]),
            ):
                with path.open("wb") as handle:
                    pickle.dump(payload, handle)
            summary.write_text(
                json.dumps(
                    {
                        "status": "VERIFIED",
                        "splits": {
                            "train": {"written": 2},
                            "val": {"written": 1},
                            "test": {"written": 1},
                        },
                        "surface_patches": 1,
                        "edge_patches": 1,
                    }
                ),
                encoding="utf-8",
            )

            report = run_preflight(
                root=root,
                data_dir=data,
                python_exe=Path("python"),
                check_modules=[],
                run_cli_help=False,
                official_incompat_report=None,
            )

        self.assertEqual(report["status"], "READY")
        self.assertFalse(report["training_started"])
        self.assertEqual(report["inputs"]["split_counts"], {"train": 2, "val": 1, "test": 1})
        self.assertIn("train_vqvae", report["cli"])
        self.assertEqual(report["official_baseline"]["status"], "not_checked")
        self.assertIn("planned_commands", report)
        self.assertEqual(
            report["planned_commands"]["train_vqvae"]["script"],
            "BrepARG/train_vqvae.py",
        )
        self.assertIn("--data_list", report["planned_commands"]["train_vqvae"]["args"])
        self.assertIn("--sequence_file", report["planned_commands"]["train_ar"]["args"])
        self.assertIn("--max_attempts", report["planned_commands"]["generate_brep"]["args"])

    def test_required_cli_arg_check_reports_missing_args(self):
        from tools.preflight_breparg_same_data_fallback import check_required_cli_args

        help_checks = {
            "train_vqvae": {
                "ok": True,
                "stdout": "--data_list PATH --surface_list PATH --edge_list PATH --batch_size 4",
                "stderr": "",
            },
            "generate_brep": {
                "ok": True,
                "stdout": "--ar_model X --se_vqvae X --num_samples 2",
                "stderr": "",
            },
        }

        checked = check_required_cli_args(help_checks, {"train_vqvae", "generate_brep"})

        self.assertFalse(checked["train_vqvae"]["ok"])
        self.assertIn("--dataset_type", checked["train_vqvae"]["missing"])
        self.assertFalse(checked["generate_brep"]["ok"])
        self.assertIn("--max_attempts", checked["generate_brep"]["missing"])

    def test_preflight_blocks_when_required_input_is_missing(self):
        from tools.preflight_breparg_same_data_fallback import run_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            report = run_preflight(
                root=root,
                data_dir=data,
                python_exe=Path("python"),
                check_modules=[],
                run_cli_help=False,
                official_incompat_report=None,
            )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["training_started"])
        self.assertIn("missing:same_data_input_summary.json", report["blocking_reasons"])
        self.assertIn("missing:same_data_split.pkl", report["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
