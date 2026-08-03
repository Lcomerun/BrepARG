import sys
import tempfile
import unittest
from pathlib import Path


from tools.run_breparg_generation_batches import (
    remaining_batch_size,
    run_command_with_timeout,
)


class BrepARGGenerationBatchTests(unittest.TestCase):
    def test_generation_only_launcher_uses_timeout_bounded_batches(self):
        source = Path("tools/run_breparg_long_generation_only_20260726.ps1").read_text(encoding="utf-8")

        self.assertIn("tools\\run_breparg_generation_batches.py", source)
        self.assertIn('"--batch-timeout-sec", "180"', source)
        self.assertIn('"--batch-size", "4"', source)
        self.assertIn('"--top-p", "0.9"', source)
        self.assertNotIn('"--top_p"', source)

    def test_remaining_batch_size_never_exceeds_remaining_target(self):
        self.assertEqual(remaining_batch_size(current=8, target=100, batch_size=4), 4)
        self.assertEqual(remaining_batch_size(current=98, target=100, batch_size=4), 2)
        self.assertEqual(remaining_batch_size(current=100, target=100, batch_size=4), 0)

    def test_timed_out_command_is_terminated_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_command_with_timeout(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout_sec=0.2,
                stdout_path=Path(tmp) / "stdout.log",
                stderr_path=Path(tmp) / "stderr.log",
            )

        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["returncode"])
        self.assertGreater(result["elapsed_sec"], 0)

    def test_successful_command_preserves_return_code_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout_path = Path(tmp) / "stdout.log"
            result = run_command_with_timeout(
                [sys.executable, "-c", "print('batch-ok')"],
                timeout_sec=10,
                stdout_path=stdout_path,
                stderr_path=Path(tmp) / "stderr.log",
            )

            output = stdout_path.read_text(encoding="utf-8")

        self.assertFalse(result["timed_out"])
        self.assertEqual(result["returncode"], 0)
        self.assertIn("batch-ok", output)


if __name__ == "__main__":
    unittest.main()
