import json
import tempfile
import unittest
from pathlib import Path


class FsqCapacityCandidatePreflightTests(unittest.TestCase):
    def test_preflight_ready_when_patch_shards_and_training_entrypoint_exist(self):
        from tools.preflight_fsq_capacity_candidate import run_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "vq_patch_shards_full"
            shards.mkdir()
            (shards / "_summary.json").write_text(
                json.dumps(
                    {
                        "status": "BUILT",
                        "patch_shards": 2,
                        "patches": 450000,
                        "surfaces": 100,
                        "edges": 200,
                    }
                ),
                encoding="utf-8",
            )
            (shards / "vq_patch_shard_0000.pkl.gz").write_bytes(b"placeholder")
            train_py = root / "train.py"
            train_py.write_text("print('train')\n", encoding="utf-8")
            report = run_preflight(
                patch_shard_root=shards,
                outbase=root / "out",
                run_name="fsq_test",
                python_exe=Path("python"),
                train_script=train_py,
                samples=450000,
                levels="16,16,8,8",
                sample_cache=root / "cache" / "samples.npz",
                check_modules=[],
                run_cli_help=False,
            )

        self.assertEqual(report["status"], "READY")
        self.assertFalse(report["training_started"])
        self.assertEqual(report["patch_shards"]["summary"]["patches"], 450000)
        self.assertEqual(report["config"]["levels"], [16, 16, 8, 8])
        self.assertEqual(report["config"]["codebook_size"], 16384)
        self.assertEqual(report["sample_cache"]["path"], str(root / "cache" / "samples.npz"))
        self.assertFalse(report["sample_cache"]["exists"])
        self.assertIn("01a_build_fsq_capacity_sample_cache.ps1", report["next_command"])

    def test_preflight_blocks_when_patch_shards_are_missing(self):
        from tools.preflight_fsq_capacity_candidate import run_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_preflight(
                patch_shard_root=root / "missing",
                outbase=root / "out",
                run_name="fsq_test",
                python_exe=Path("python"),
                train_script=root / "train.py",
                samples=450000,
                levels="16,16,8,8",
                sample_cache=None,
                check_modules=[],
                run_cli_help=False,
            )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["training_started"])
        self.assertIn("missing_patch_shard_root", report["blocking_reasons"])
        self.assertIn("missing_train_script", report["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
