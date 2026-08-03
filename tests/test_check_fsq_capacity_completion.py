import json
import tempfile
import unittest
from pathlib import Path


class CheckFsqCapacityCompletionTests(unittest.TestCase):
    def test_reports_incomplete_when_train_report_missing(self):
        from tools.check_fsq_capacity_completion import check_completion

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "vqvae_history.json").write_text(
                json.dumps(
                    {
                        "config": {"target_epoch": 180},
                        "history": [{"epoch": 5, "train_loss": 0.1, "val_loss": 0.2}],
                        "best_val_recon": 0.2,
                        "best_epoch": 5,
                        "stop_reason": "",
                    }
                ),
                encoding="utf-8",
            )

            report = check_completion(run, pids=[])

            self.assertFalse(report["complete"])
            self.assertIn("train_report_missing", report["reasons"])
            self.assertEqual(report["history"]["last_epoch"], 5)

    def test_reports_incomplete_when_live_pid_exists(self):
        from tools.check_fsq_capacity_completion import check_completion

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "train_report.json").write_text(
                json.dumps({"stages": {"vqvae": {"status": "PASS"}}}),
                encoding="utf-8",
            )

            report = check_completion(run, pids=[999], live_pids={999})

            self.assertFalse(report["complete"])
            self.assertIn("training_process_alive", report["reasons"])

    def test_reports_complete_when_report_passes_and_no_live_pid(self):
        from tools.check_fsq_capacity_completion import check_completion

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "train_report.json").write_text(
                json.dumps({"stages": {"vqvae": {"status": "PASS"}}}),
                encoding="utf-8",
            )

            report = check_completion(run, pids=[999], live_pids=set())

            self.assertTrue(report["complete"])
            self.assertEqual(report["train_report_status"], "PASS")

    def test_pid_probe_errors_are_treated_as_not_alive(self):
        from tools import check_fsq_capacity_completion as mod

        original_kill = mod.os.kill
        try:
            def boom(pid, sig):
                raise SystemError("windows pid probe failed")

            mod.os.kill = boom
            self.assertFalse(mod._pid_is_alive(12345))
        finally:
            mod.os.kill = original_kill

    def test_windows_pid_probe_uses_powershell_when_available(self):
        from tools import check_fsq_capacity_completion as mod

        calls = []
        original_system = mod.platform.system
        original_run = mod.subprocess.run
        try:
            mod.platform.system = lambda: "Windows"

            class Completed:
                returncode = 0

            def fake_run(*args, **kwargs):
                calls.append((args, kwargs))
                return Completed()

            mod.subprocess.run = fake_run

            self.assertTrue(mod._pid_is_alive(123))
            self.assertTrue(calls)
            self.assertIn("Get-Process -Id 123", calls[0][0][0][-1])
        finally:
            mod.platform.system = original_system
            mod.subprocess.run = original_run

    def test_windows_pid_probe_returns_false_when_powershell_reports_missing(self):
        from tools import check_fsq_capacity_completion as mod

        original_system = mod.platform.system
        original_run = mod.subprocess.run
        try:
            mod.platform.system = lambda: "Windows"

            class Completed:
                returncode = 1

            mod.subprocess.run = lambda *args, **kwargs: Completed()

            self.assertFalse(mod._pid_is_alive(123))
        finally:
            mod.platform.system = original_system
            mod.subprocess.run = original_run


if __name__ == "__main__":
    unittest.main()
