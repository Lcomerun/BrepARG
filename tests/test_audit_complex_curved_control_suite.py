import json
import os
import tempfile
import unittest
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class AuditComplexCurvedControlSuiteTests(unittest.TestCase):
    def test_audits_missing_and_completed_experiments_with_next_actions(self):
        from tools.audit_complex_curved_control_suite import audit_suite, write_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "00_current_fsq_ar_teacher_reconstruction.ps1",
                "01a_train_fsq_capacity_candidate.ps1",
                "01_fsq_capacity_candidate.ps1",
                "02_dfs_rcm_ordering_rebuild.ps1",
                "02b_train_dfs_rcm_ar.ps1",
                "02c_eval_dfs_rcm_ar_complex_curved.ps1",
                "03_breparg_official_baseline.ps1",
                "03a_prepare_breparg_same_data_inputs.ps1",
                "03b_breparg_same_data_training_fallback.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")

            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {
                    "status": "VERIFIED",
                    "selected_count": 50,
                    "fsq_patch_metrics": {"chamfer": {"p95": 0.15}},
                    "ar_teacher_forcing": {"token_weighted_ce": 0.75},
                    "teacher_reconstruction": {"attempted": 50, "brep_valid": 9},
                },
            )
            write_json(
                root / "experiments/03_breparg_official_baseline/breparg_baseline_quality_summary.json",
                {
                    "summary": {
                        "step_files": 10,
                        "brep_valid": 8,
                        "complex_and_brep_valid_closed": 2,
                    }
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["summary"]["completed"], 2)
            self.assertGreaterEqual(audit["summary"]["missing"], 3)
            self.assertEqual(audit["experiments"]["current_method"]["status"], "complete")
            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["status"], "missing")
            self.assertEqual(audit["experiments"]["breparg_official_baseline"]["status"], "complete")
            self.assertIn("scripts\\01a_train_fsq_capacity_candidate.ps1", audit["next_actions"][0]["command"])
            self.assertIn("FSQ", audit["next_actions"][0]["label"])

            output_json = root / "suite_status.json"
            output_md = root / "suite_status.md"
            write_outputs(audit, output_json, output_md)
            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            self.assertIn("Complex Curved Control Suite Status", output_md.read_text(encoding="utf-8"))

    def test_marks_fsq_capacity_patch_shards_complete_from_summary(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            patch_root = root / "experiments/01a_train_fsq_capacity_candidate/vq_patch_shards_full"
            write_json(
                patch_root / "_summary.json",
                {
                    "status": "BUILT",
                    "patch_shards": 344,
                    "patches": 34393215,
                    "source_records_failed": 458,
                },
            )
            (patch_root / "vq_patch_shard_0000.pkl.gz").write_bytes(b"fake")

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["fsq_capacity_patch_shards"]["status"], "complete")
            self.assertEqual(audit["experiments"]["fsq_capacity_patch_shards"]["details"]["patch_shards"], 344)
            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["status"], "missing")
            self.assertIn("01a_train_fsq_capacity_candidate.ps1", audit["next_actions"][0]["script"])

    def test_tracks_fsq_capacity_comparison_report(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"

            audit = audit_suite(root)
            self.assertEqual(audit["experiments"]["fsq_capacity_comparison"]["status"], "missing")

            write_json(
                root / "fsq_capacity_comparison.json",
                {
                    "status": "VERIFIED",
                    "recommendation": {
                        "capacity_signal": "moderate_improvement",
                        "reading": "Higher FSQ capacity reduces overall Chamfer p95.",
                    },
                    "metrics": {
                        "fsq_chamfer_p95": {
                            "baseline": 0.15,
                            "candidate": 0.11,
                            "relative_change_pct": -26.6,
                        }
                    },
                },
            )

            audit = audit_suite(root)
            entry = audit["experiments"]["fsq_capacity_comparison"]
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(entry["details"]["capacity_signal"], "moderate_improvement")
            self.assertEqual(entry["details"]["fsq_chamfer_p95_candidate"], 0.11)

    def test_points_to_fsq_sample_cache_before_training_when_preflight_requests_it(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "01a_preflight_fsq_capacity_candidate.ps1",
                "01a_build_fsq_capacity_sample_cache.ps1",
                "01a_train_fsq_capacity_candidate.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")

            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            patch_root = root / "experiments/01a_train_fsq_capacity_candidate/vq_patch_shards_full"
            write_json(
                patch_root / "_summary.json",
                {
                    "status": "BUILT",
                    "patch_shards": 344,
                    "patches": 34393215,
                },
            )
            write_json(
                root / "experiments/01a_train_fsq_capacity_candidate/fsq_capacity_preflight.json",
                {
                    "status": "READY",
                    "blocking_reasons": [],
                    "sample_cache": {
                        "enabled": True,
                        "exists": False,
                        "path": "suite/experiments/01a_train_fsq_capacity_candidate/vq_samples_450000_seed0.npz",
                    },
                    "next_command": (
                        "powershell -ExecutionPolicy Bypass -File "
                        "suite\\scripts\\01a_build_fsq_capacity_sample_cache.ps1"
                    ),
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["fsq_capacity_preflight"]["status"], "complete")
            self.assertEqual(Path(audit["next_actions"][0]["script"]).name, "01a_build_fsq_capacity_sample_cache.ps1")
            self.assertIn("sample cache", audit["next_actions"][0]["label"].lower())
            self.assertIn("01a_train_fsq_capacity_candidate.ps1", [Path(action["script"]).name for action in audit["next_actions"]])

    def test_does_not_mark_fsq_capacity_training_complete_from_early_checkpoint_only(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "01a_train_fsq_capacity_candidate.ps1").write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )

            run = (
                root
                / "experiments/01a_train_fsq_capacity_candidate/fsq_levels_16_16_8_8_complex_curved_20260715"
            )
            run.mkdir(parents=True)
            (run / "fsq_vqvae_best.pt").write_bytes(b"early checkpoint")
            write_json(
                run / "vqvae_history.json",
                {
                    "config": {"target_epoch": 180},
                    "history": [{"epoch": 0, "val_loss": 0.001}],
                    "stop_reason": "",
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["status"], "partial")
            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["details"]["checkpoint_best"], True)
            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["details"]["history_epochs"], 1)
            self.assertIn("01a_train_fsq_capacity_candidate.ps1", audit["next_actions"][0]["script"])

            write_json(
                run / "train_report.json",
                {"stages": {"vqvae": {"status": "VERIFIED", "epochs_ran": 44, "best_val_recon": 0.0001}}},
            )
            audit = audit_suite(root)
            self.assertEqual(audit["experiments"]["fsq_capacity_training"]["status"], "complete")

    def test_points_to_fsq_resume_when_partial_checkpoint_and_inputs_exist(self):
        from tools import audit_complex_curved_control_suite as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "01a_train_fsq_capacity_candidate.ps1",
                "01a_resume_fsq_capacity_candidate.ps1",
                "01a_watch_fsq_capacity_then_eval.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            write_json(
                root / "experiments/01a_train_fsq_capacity_candidate/vq_patch_shards_full/_summary.json",
                {"status": "BUILT", "patch_shards": 344, "patches": 34393215},
            )

            run = (
                root
                / "experiments/01a_train_fsq_capacity_candidate/fsq_levels_16_16_8_8_complex_curved_20260715"
            )
            run.mkdir(parents=True)
            (run / "fsq_vqvae_best.pt").write_bytes(b"partial checkpoint")
            write_json(
                run / "vqvae_history.json",
                {
                    "config": {"target_epoch": 180},
                    "history": [{"epoch": 5, "val_loss": 0.00036255}],
                    "best_val_recon": 0.00036255,
                    "best_epoch": 5,
                },
            )
            logs = root / "experiments/01a_train_fsq_capacity_candidate/logs"
            logs.mkdir(parents=True)
            (logs / "fsq_capacity_resume.pid").write_text("1234\n", encoding="utf-8")
            (logs / "fsq_capacity_resume_20260715_165543.out.log").write_text("resume\n", encoding="utf-8")

            original_pid_is_alive = mod.pid_is_alive
            original_windows_pid_command_matches = mod.windows_pid_command_matches
            try:
                mod.pid_is_alive = lambda pid: pid == 1234
                mod.windows_pid_command_matches = lambda pid, patterns, root=None: pid == 1234
                audit = mod.audit_suite(root)
            finally:
                mod.pid_is_alive = original_pid_is_alive
                mod.windows_pid_command_matches = original_windows_pid_command_matches

            training = audit["experiments"]["fsq_capacity_training"]
            self.assertEqual(training["status"], "partial")
            self.assertEqual(training["details"]["last_epoch"], 5)
            self.assertFalse(training["details"]["train_report_exists"])
            self.assertEqual(training["details"]["resume_pid"], 1234)
            self.assertTrue(training["details"]["resume_alive"])
            self.assertIn("fsq_capacity_resume_20260715_165543.out.log", training["details"]["resume_log"])
            self.assertEqual(Path(audit["next_actions"][0]["script"]).name, "01a_watch_fsq_capacity_then_eval.ps1")
            self.assertIn("monitor", audit["next_actions"][0]["label"].lower())
            self.assertIn("do not start another resume", audit["next_actions"][0]["reason"].lower())

    def test_ignores_stale_fsq_resume_pid_when_command_does_not_match_training(self):
        from tools import audit_complex_curved_control_suite as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "01a_train_fsq_capacity_candidate.ps1",
                "01a_resume_fsq_capacity_candidate.ps1",
                "01a_watch_fsq_capacity_then_eval.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            write_json(
                root / "experiments/01a_train_fsq_capacity_candidate/vq_patch_shards_full/_summary.json",
                {"status": "BUILT", "patch_shards": 344, "patches": 34393215},
            )

            run = (
                root
                / "experiments/01a_train_fsq_capacity_candidate/fsq_levels_16_16_8_8_complex_curved_20260715"
            )
            run.mkdir(parents=True)
            (run / "fsq_vqvae_best.pt").write_bytes(b"partial checkpoint")
            write_json(
                run / "vqvae_history.json",
                {
                    "config": {"target_epoch": 180},
                    "history": [{"epoch": 100, "val_loss": 0.00005424}],
                    "best_val_recon": 0.00005283,
                    "best_epoch": 82,
                },
            )
            logs = root / "experiments/01a_train_fsq_capacity_candidate/logs"
            logs.mkdir(parents=True)
            (logs / "fsq_capacity_resume.pid").write_text("999999\n", encoding="utf-8")
            (logs / "fsq_capacity_resume_20260715_165543.out.log").write_text("resume exited\n", encoding="utf-8")

            original_pid_is_alive = mod.pid_is_alive
            try:
                mod.pid_is_alive = lambda pid: True
                audit = mod.audit_suite(root)
            finally:
                mod.pid_is_alive = original_pid_is_alive

            training = audit["experiments"]["fsq_capacity_training"]
            self.assertEqual(training["status"], "partial")
            self.assertEqual(training["details"]["resume_pid"], 999999)
            self.assertFalse(training["details"]["resume_alive"])
            self.assertEqual(Path(audit["next_actions"][0]["script"]).name, "01a_resume_fsq_capacity_candidate.ps1")
            self.assertIn("resume", audit["next_actions"][0]["label"].lower())

    def test_marks_ordering_complete_only_when_both_reports_exist(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            dfs_ckpt = (
                root
                / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_matched_20260715/ar_best.pt"
            )
            rcm_ckpt = (
                root
                / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_matched_20260715/ar_best.pt"
            )
            dfs_report = (
                root
                / "experiments/02_dfs_rcm_ordering/ar_complex_curved_eval/dfs_teacher_forcing/complex_curved_diagnostics_report.json"
            )
            rcm_report = (
                root
                / "experiments/02_dfs_rcm_ordering/ar_complex_curved_eval/rcm_teacher_forcing/complex_curved_diagnostics_report.json"
            )
            dfs_ckpt.parent.mkdir(parents=True)
            rcm_ckpt.parent.mkdir(parents=True)
            dfs_ckpt.write_bytes(b"dfs")
            rcm_ckpt.write_bytes(b"rcm")
            os.utime(dfs_ckpt, (200.0, 200.0))
            os.utime(rcm_ckpt, (200.0, 200.0))
            write_json(
                dfs_report,
                {"status": "VERIFIED", "ar_teacher_forcing": {"token_weighted_ce": 0.7}},
            )
            os.utime(dfs_report, (300.0, 300.0))

            audit = audit_suite(root)
            self.assertEqual(audit["experiments"]["dfs_rcm_teacher_forcing"]["status"], "partial")

            write_json(
                rcm_report,
                {"status": "VERIFIED", "ar_teacher_forcing": {"token_weighted_ce": 0.8}},
            )
            os.utime(rcm_report, (100.0, 100.0))
            audit = audit_suite(root)
            self.assertEqual(audit["experiments"]["dfs_rcm_teacher_forcing"]["status"], "partial")
            self.assertFalse(audit["experiments"]["dfs_rcm_teacher_forcing"]["details"]["rcm_eval_fresh"])

            os.utime(rcm_report, (300.0, 300.0))
            audit = audit_suite(root)
            self.assertEqual(audit["experiments"]["dfs_rcm_teacher_forcing"]["status"], "complete")

    def test_marks_ordering_sequence_smoke_complete_from_three_summaries(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            write_json(
                root / "experiments/02_dfs_rcm_ordering/same_data_split_smoke/v13_same_data_split_summary.json",
                {"status": "VERIFIED", "total_written": 11},
            )
            write_json(
                root / "experiments/02_dfs_rcm_ordering/sequence_rebuild_smoke/sequences_fsq_dfs_summary.json",
                {"status": "VERIFIED", "sequences": 11, "out_of_vocab": 0},
            )
            write_json(
                root / "experiments/02_dfs_rcm_ordering/sequence_rebuild_smoke/sequences_fsq_rcm_summary.json",
                {"status": "VERIFIED", "sequences": 11, "out_of_vocab": 0},
            )

            audit = audit_suite(root)

            entry = audit["experiments"]["dfs_rcm_sequence_rebuild_smoke"]
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(entry["details"]["split_total_written"], 11)
            self.assertEqual(entry["details"]["dfs_sequences"], 11)
            self.assertEqual(entry["details"]["rcm_out_of_vocab"], 0)

    def test_marks_medium_ordering_sequence_rebuild_complete_from_outputs(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            medium = root / "experiments/02_dfs_rcm_ordering/sequence_rebuild_medium"
            (medium / "sequences_fsq_dfs.pkl").parent.mkdir(parents=True, exist_ok=True)
            (medium / "sequences_fsq_dfs.pkl").write_bytes(b"dfs")
            (medium / "sequences_fsq_rcm.pkl").write_bytes(b"rcm")
            write_json(
                medium / "sequences_fsq_dfs_summary.json",
                {"status": "VERIFIED", "sequences": 12000, "out_of_vocab": 0},
            )
            write_json(
                medium / "sequences_fsq_rcm_summary.json",
                {"status": "VERIFIED", "sequences": 12000, "out_of_vocab": 0},
            )

            audit = audit_suite(root)

            entry = audit["experiments"]["dfs_rcm_sequence_rebuild_medium"]
            self.assertEqual(entry["status"], "complete")
            self.assertTrue(entry["details"]["dfs_sequence"])
            self.assertEqual(entry["details"]["rcm_sequences"], 12000)

    def test_marks_medium_ordering_ar_training_complete_from_both_checkpoints(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            dfs = root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_medium_safe_20260715/ar_best.pt"
            rcm = root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_medium_safe_20260715/ar_best.pt"
            dfs.parent.mkdir(parents=True, exist_ok=True)
            rcm.parent.mkdir(parents=True, exist_ok=True)
            dfs.write_bytes(b"dfs")
            rcm.write_bytes(b"rcm")

            audit = audit_suite(root)

            entry = audit["experiments"]["dfs_rcm_ar_training_medium"]
            self.assertEqual(entry["status"], "complete")
            self.assertTrue(entry["details"]["dfs_ar_best"])
            self.assertTrue(entry["details"]["rcm_ar_best"])

    def test_marks_medium_ordering_ar_smoke_complete_from_both_checkpoints(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            dfs = root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_dfs_medium_smoke_20260715/ar_best.pt"
            rcm = root / "experiments/02_dfs_rcm_ordering/ar_train_outputs/ar_rcm_medium_smoke_20260715/ar_best.pt"
            dfs.parent.mkdir(parents=True, exist_ok=True)
            rcm.parent.mkdir(parents=True, exist_ok=True)
            dfs.write_bytes(b"dfs")
            rcm.write_bytes(b"rcm")

            audit = audit_suite(root)

            entry = audit["experiments"]["dfs_rcm_ar_training_medium_smoke"]
            self.assertEqual(entry["status"], "complete")
            self.assertTrue(entry["details"]["dfs_ar_best"])
            self.assertTrue(entry["details"]["rcm_ar_best"])

    def test_does_not_mark_empty_breparg_baseline_complete(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "00_current_fsq_ar_teacher_reconstruction.ps1").write_text("# script\n", encoding="utf-8")
            (scripts / "03_breparg_official_baseline.ps1").write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            write_json(
                root / "experiments/03_breparg_official_baseline/breparg_baseline_quality_summary.json",
                {
                    "summary": {
                        "step_files": 0,
                        "brep_valid": 0,
                        "complex_and_brep_valid_closed": 0,
                    }
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["breparg_official_baseline"]["status"], "missing")
            self.assertIn("03_breparg_official_baseline.ps1", [Path(action["script"]).name for action in audit["next_actions"]])

    def test_marks_incompatible_official_breparg_baseline_partial_and_points_to_fallback(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "00_current_fsq_ar_teacher_reconstruction.ps1",
                "01a_train_fsq_capacity_candidate.ps1",
                "03_breparg_official_baseline.ps1",
                "03b_breparg_same_data_training_fallback.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            incompat = root / "experiments/03_breparg_official_baseline/official_baseline_incompatibility_report.json"
            incompat.parent.mkdir(parents=True, exist_ok=True)
            incompat.write_text(
                "\ufeff"
                + json.dumps(
                    {
                        "status": "INCOMPATIBLE",
                        "decision": "Use same-data fallback",
                        "checkpoint_shapes": {"abc_ar_transformer_wte": [7222, 256]},
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_suite(root)

            official = audit["experiments"]["breparg_official_baseline"]
            self.assertEqual(official["status"], "partial")
            self.assertEqual(official["details"]["compatibility_status"], "INCOMPATIBLE")
            self.assertEqual(official["details"]["abc_ar_vocab"], [7222, 256])
            self.assertIn("03b_breparg_same_data_training_fallback.ps1", [Path(action["script"]).name for action in audit["next_actions"]])

    def test_does_not_point_to_fallback_when_same_data_fallback_is_complete(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            for name in (
                "00_current_fsq_ar_teacher_reconstruction.ps1",
                "03_breparg_official_baseline.ps1",
                "03b_breparg_same_data_training_fallback.ps1",
            ):
                (scripts / name).write_text("# script\n", encoding="utf-8")
            write_json(
                root / "experiments/00_current_fsq_ar_teacher_reconstruction/complex_curved_diagnostics_report.json",
                {"status": "VERIFIED"},
            )
            write_json(
                root / "experiments/03_breparg_official_baseline/official_baseline_incompatibility_report.json",
                {
                    "status": "INCOMPATIBLE",
                    "decision": "Use same-data fallback",
                    "checkpoint_shapes": {"abc_ar_transformer_wte": [7222, 256]},
                },
            )
            write_json(
                root / "experiments/03b_breparg_same_data_training_fallback/breparg_same_data_quality_summary.json",
                {
                    "summary": {
                        "step_files": 92,
                        "brep_valid": 75,
                        "complex_and_brep_valid_closed": 3,
                    }
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["breparg_same_data_fallback"]["status"], "complete")
            self.assertNotIn(
                "03b_breparg_same_data_training_fallback.ps1",
                [Path(action["script"]).name for action in audit["next_actions"]],
            )

    def test_marks_breparg_logic_generation_complete_from_latest_report(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            run = root / "experiments/04_breparg_logic_generation_baseline/breparg_logic_user_compare_100"
            write_json(
                run / "breparg_logic_report.json",
                {
                    "status": "VERIFIED",
                    "summary": {
                        "attempted": 111,
                        "accepted_visual": 100,
                        "status_counts": {"reconstruct_failed": 10},
                        "faces": {"median": 6},
                        "edges": {"median": 12},
                    },
                    "config": {"temperature": 1.0, "top_p": 0.9, "device": "cpu"},
                },
            )
            write_json(
                run / "breparg_logic_saved100_distribution.json",
                {
                    "saved_png_step_rows": 100,
                    "attempted_rows": 111,
                    "reconstruct_failed": 10,
                    "complex_fraction": 0.11,
                    "very_simple_fraction": 0.79,
                    "top_face_edge_pairs": [{"pair": "f6_e12", "count": 45}],
                },
            )

            audit = audit_suite(root)

            entry = audit["experiments"]["breparg_logic_generation"]
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(entry["details"]["saved_png_step_rows"], 100)
            self.assertEqual(entry["details"]["complex_fraction"], 0.11)
            self.assertEqual(entry["details"]["very_simple_fraction"], 0.79)
            self.assertEqual(entry["details"]["top_face_edge_pair"], "f6_e12")
            self.assertEqual(entry["details"]["temperature"], 1.0)

    def test_marks_breparg_same_data_inputs_complete_from_summary(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            write_json(
                root / "experiments/03b_breparg_same_data_training_fallback/data/same_data_input_summary.json",
                {
                    "status": "VERIFIED",
                    "splits": {
                        "train": {"written": 100},
                        "val": {"written": 10},
                        "test": {"written": 10},
                    },
                    "surface_patches": 1000,
                    "edge_patches": 1500,
                },
            )

            audit = audit_suite(root)

            entry = audit["experiments"]["breparg_same_data_inputs"]
            self.assertEqual(entry["status"], "complete")
            self.assertEqual(entry["details"]["train_written"], 100)
            self.assertEqual(entry["details"]["surface_patches"], 1000)

    def test_audits_split_rootcause_current_method_layout(self):
        from tools.audit_complex_curved_control_suite import audit_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rootcause_suite"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "01a_build_fsq_capacity_patch_shards_full.ps1").write_text("# script\n", encoding="utf-8")
            (scripts / "01a_train_fsq_capacity_candidate.ps1").write_text("# script\n", encoding="utf-8")

            write_json(
                root / "experiments/00_fsq_only_patch_metrics/complex_curved_diagnostics_report.json",
                {
                    "status": "VERIFIED",
                    "selected_count": 50,
                    "fsq_patch_metrics": {"chamfer": {"p95": 0.15}},
                },
            )
            write_json(
                root / "experiments/01_teacher_forcing_true_token_reconstruction/complex_curved_diagnostics_report.json",
                {
                    "status": "VERIFIED",
                    "selected_count": 50,
                    "ar_teacher_forcing": {"token_weighted_ce": 0.75},
                    "teacher_reconstruction": {"attempted": 50, "brep_valid": 9},
                },
            )

            audit = audit_suite(root)

            self.assertEqual(audit["experiments"]["current_method"]["status"], "complete")
            self.assertEqual(audit["experiments"]["current_method"]["details"]["selected_count"], 50)
            self.assertEqual(audit["experiments"]["current_method"]["details"]["fsq_chamfer_p95"], 0.15)
            self.assertEqual(audit["experiments"]["current_method"]["details"]["ar_token_weighted_ce"], 0.75)
            self.assertIn("01a_build_fsq_capacity_patch_shards_full.ps1", audit["next_actions"][0]["script"])
            self.assertIn(str(root), audit["next_actions"][0]["command"])


if __name__ == "__main__":
    unittest.main()
