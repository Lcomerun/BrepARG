import json
import tempfile
import unittest
from pathlib import Path


class PrepareComplexCurvedControlWorkspaceTests(unittest.TestCase):
    def test_prepare_workspace_writes_reproducible_experiment_entrypoints(self):
        from tools.prepare_complex_curved_control_workspace import prepare_workspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            result = prepare_workspace(
                output_dir=root,
                python_exe="PYTHON",
                sequence_path=Path("data/sequences_fsq_rcm.pkl"),
                vqvae_checkpoint=Path("data/fsq_vqvae_best.pt"),
                ar_checkpoint=Path("data/ar_best.pt"),
                archive_root=Path("data/archives"),
            )

            self.assertEqual(result["output_dir"], str(root))
            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "experiment_config.json").exists())

            config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["official_breparg_hf_repo"], "qingtiannihao/BrepARG")
            self.assertEqual(config["complex_subset"]["max_samples"], 50)
            self.assertEqual(config["complex_subset"]["complex_min_faces"], 12)
            self.assertEqual(config["complex_subset"]["complex_min_edges"], 20)

            expected_scripts = {
                "00_current_fsq_ar_teacher_reconstruction.ps1",
                "01a_preflight_fsq_capacity_candidate.ps1",
                "01a_build_fsq_capacity_sample_cache.ps1",
                "01a_train_fsq_capacity_candidate.ps1",
                "01a_resume_fsq_capacity_candidate.ps1",
                "01a_watch_fsq_capacity_then_eval.ps1",
                "01_fsq_capacity_candidate.ps1",
                "02a_prepare_v13_same_data_split.ps1",
                "02_smoke_dfs_rcm_ordering_rebuild.ps1",
                "02_medium_dfs_rcm_ordering_rebuild.ps1",
                "02_dfs_rcm_ordering_rebuild.ps1",
                "02b_smoke_dfs_rcm_ar_medium_safe.ps1",
                "02b_train_dfs_rcm_ar_medium_safe.ps1",
                "02b_train_dfs_rcm_ar.ps1",
                "02c_eval_dfs_rcm_ar_complex_curved.ps1",
                "03_breparg_official_baseline.ps1",
                "03a_prepare_breparg_same_data_inputs.ps1",
                "03a_prepare_breparg_same_data_inputs_full.ps1",
                "03b_preflight_breparg_same_data_fallback.ps1",
                "03b_smoke_breparg_same_data_training_fallback.ps1",
                "03b_breparg_same_data_training_fallback.ps1",
                "04_summarize_reports.ps1",
                "05_audit_suite_status.ps1",
                "06_prepare_external_ssd_migration.ps1",
            }
            actual_scripts = {path.name for path in (root / "scripts").glob("*.ps1")}
            self.assertTrue(expected_scripts.issubset(actual_scripts))

            current_script = (root / "scripts" / "00_current_fsq_ar_teacher_reconstruction.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\complex_curved_diagnostics.py", current_script)
            self.assertIn("--write-step", current_script)
            self.assertIn("--validate-step", current_script)

            capacity_script = (root / "scripts" / "01_fsq_capacity_candidate.ps1").read_text(encoding="utf-8")
            self.assertIn("--skip-ar", capacity_script)
            self.assertIn("--skip-reconstruction", capacity_script)
            self.assertIn("V13_CAPACITY_VQVAE", capacity_script)
            self.assertIn("fsq_levels_16_16_8_8_complex_curved_20260715", capacity_script)
            self.assertIn("fsq_vqvae_best.pt", capacity_script)
            self.assertIn("01a_train_fsq_capacity_candidate.ps1", capacity_script)
            self.assertNotIn("PATH\\TO\\capacity_candidate", capacity_script)

            train_capacity_script = (root / "scripts" / "01a_train_fsq_capacity_candidate.ps1").read_text(encoding="utf-8")
            self.assertIn("NS_LEVELS", train_capacity_script)
            self.assertIn("16,16,8,8", train_capacity_script)
            self.assertIn("--stage vqvae", train_capacity_script)
            self.assertIn("NS_VQ_PATCH_SHARD_ROOT", train_capacity_script)
            self.assertIn("NS_VQ_SAMPLE_CACHE", train_capacity_script)
            self.assertIn("vq_samples_450000_seed0.npz", train_capacity_script)
            self.assertIn("vq_patch_shards_full", train_capacity_script)
            self.assertNotIn("PATH\\TO\\vqvae_patch_shards", train_capacity_script)

            resume_capacity_script = (
                root / "scripts" / "01a_resume_fsq_capacity_candidate.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\check_fsq_capacity_completion.py", resume_capacity_script)
            self.assertIn("NS_VQ_RESUME_FROM", resume_capacity_script)
            self.assertIn("NS_VQ_HISTORY_IN", resume_capacity_script)
            self.assertIn("fsq_vqvae_best.pt", resume_capacity_script)
            self.assertIn("vqvae_history.json", resume_capacity_script)
            self.assertIn("NS_VQ_TARGET_EPOCH = \"180\"", resume_capacity_script)
            self.assertIn("--stage vqvae", resume_capacity_script)

            watch_capacity_script = (
                root / "scripts" / "01a_watch_fsq_capacity_then_eval.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("fsq_capacity_resume.pid", watch_capacity_script)
            self.assertIn("tools\\check_fsq_capacity_completion.py", watch_capacity_script)
            self.assertIn("01_fsq_capacity_candidate.ps1", watch_capacity_script)
            self.assertIn("04_summarize_reports.ps1", watch_capacity_script)
            self.assertIn("05_audit_suite_status.ps1", watch_capacity_script)
            self.assertIn("fsq_capacity_watch_then_eval.log", watch_capacity_script)

            fsq_preflight_script = (
                root / "scripts" / "01a_preflight_fsq_capacity_candidate.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\preflight_fsq_capacity_candidate.py", fsq_preflight_script)
            self.assertIn("vq_patch_shards_full", fsq_preflight_script)
            self.assertIn("--sample-cache $SAMPLE_CACHE", fsq_preflight_script)
            self.assertIn("fsq_capacity_preflight.json", fsq_preflight_script)

            fsq_cache_script = (
                root / "scripts" / "01a_build_fsq_capacity_sample_cache.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\build_vqvae_sample_cache.py", fsq_cache_script)
            self.assertIn("vq_samples_450000_seed0.npz", fsq_cache_script)
            self.assertIn("--samples 450000", fsq_cache_script)
            self.assertIn("--complex-fraction 0.50", fsq_cache_script)

            ordering_script = (root / "scripts" / "02_dfs_rcm_ordering_rebuild.ps1").read_text(encoding="utf-8")
            self.assertNotIn("PATH\\TO\\split.pkl", ordering_script)
            self.assertIn("same_data_split\\split.pkl", ordering_script)
            self.assertIn("02a_prepare_v13_same_data_split.ps1", ordering_script)
            self.assertIn("--ordering dfs", ordering_script)
            self.assertIn("--ordering rcm", ordering_script)

            prepare_v13_split_script = (root / "scripts" / "02a_prepare_v13_same_data_split.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\prepare_v13_same_data_split.py", prepare_v13_split_script)
            self.assertIn("--train-limit 50000", prepare_v13_split_script)
            self.assertIn("--val-limit 5000", prepare_v13_split_script)
            self.assertIn("v13_same_data_split_summary.json", prepare_v13_split_script)

            smoke_ordering_script = (root / "scripts" / "02_smoke_dfs_rcm_ordering_rebuild.ps1").read_text(
                encoding="utf-8"
            )
            self.assertIn("same_data_split_smoke", smoke_ordering_script)
            self.assertIn("sequence_rebuild_smoke", smoke_ordering_script)
            self.assertIn("--train-limit 5", smoke_ordering_script)
            self.assertIn("--val-limit 3", smoke_ordering_script)
            self.assertIn("--test-limit 3", smoke_ordering_script)
            self.assertIn("--ordering dfs", smoke_ordering_script)
            self.assertIn("--ordering rcm", smoke_ordering_script)

            medium_ordering_script = (root / "scripts" / "02_medium_dfs_rcm_ordering_rebuild.ps1").read_text(
                encoding="utf-8"
            )
            self.assertIn("03b_breparg_same_data_training_fallback", medium_ordering_script)
            self.assertIn("same_data_split.pkl", medium_ordering_script)
            self.assertIn("sequence_rebuild_medium", medium_ordering_script)
            self.assertIn("V13_ORDERING_WORKERS", medium_ordering_script)
            self.assertIn("--ordering dfs", medium_ordering_script)
            self.assertIn("--ordering rcm", medium_ordering_script)

            train_ordering_ar_script = (root / "scripts" / "02b_train_dfs_rcm_ar.ps1").read_text(encoding="utf-8")
            self.assertIn("sequences_fsq_dfs.pkl", train_ordering_ar_script)
            self.assertIn("sequence_rebuild_medium", train_ordering_ar_script)
            self.assertIn("sequences_fsq_rcm.pkl", train_ordering_ar_script)
            self.assertIn("breparg_improvements\\train.py --stage ar", train_ordering_ar_script)
            self.assertIn("tools\\preflight_ar_training.py", train_ordering_ar_script)
            self.assertIn("$env:NS_AR_MAX_SEQ_LEN", train_ordering_ar_script)

            medium_train_ordering_ar_script = (
                root / "scripts" / "02b_train_dfs_rcm_ar_medium_safe.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("ar_dfs_medium_safe_20260715", medium_train_ordering_ar_script)
            self.assertIn("ar_rcm_medium_safe_20260715", medium_train_ordering_ar_script)
            self.assertIn("V13_MEDIUM_AR_EPOCHS", medium_train_ordering_ar_script)
            self.assertIn('else { "5" }', medium_train_ordering_ar_script)
            self.assertIn("V13_MEDIUM_AR_BS", medium_train_ordering_ar_script)
            self.assertIn('else { "4" }', medium_train_ordering_ar_script)
            self.assertIn("tools\\summarize_ar_length_coverage.py", medium_train_ordering_ar_script)
            self.assertIn("tools\\preflight_ar_training.py", medium_train_ordering_ar_script)
            self.assertIn("train_report.json", medium_train_ordering_ar_script)

            smoke_train_ordering_ar_script = (
                root / "scripts" / "02b_smoke_dfs_rcm_ar_medium_safe.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("subset_ar_sequence_package.py", smoke_train_ordering_ar_script)
            self.assertIn("--train-limit 64", smoke_train_ordering_ar_script)
            self.assertIn("--val-limit 16", smoke_train_ordering_ar_script)
            self.assertIn('NS_AR_EPOCHS = "1"', smoke_train_ordering_ar_script)
            self.assertIn('NS_AR_BS = "2"', smoke_train_ordering_ar_script)
            self.assertIn("ar_dfs_medium_smoke_20260715", smoke_train_ordering_ar_script)
            self.assertIn("ar_rcm_medium_smoke_20260715", smoke_train_ordering_ar_script)
            self.assertIn("train_report.json", smoke_train_ordering_ar_script)

            eval_ordering_ar_script = (root / "scripts" / "02c_eval_dfs_rcm_ar_complex_curved.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\complex_curved_diagnostics.py", eval_ordering_ar_script)
            self.assertIn("sequences_fsq_dfs.pkl", eval_ordering_ar_script)
            self.assertIn("ar_dfs_medium_safe_20260715", eval_ordering_ar_script)
            self.assertIn("ar_best.pt", eval_ordering_ar_script)
            self.assertIn("--skip-reconstruction", eval_ordering_ar_script)

            baseline_script = (root / "scripts" / "03_breparg_official_baseline.ps1").read_text(encoding="utf-8")
            self.assertIn("function Invoke-Native", baseline_script)
            self.assertIn("if ($LASTEXITCODE -ne 0)", baseline_script)
            self.assertIn("huggingface_hub>=0.20.2,<0.26", baseline_script)
            self.assertIn("from huggingface_hub import hf_hub_download", baseline_script)
            self.assertNotIn("pip install -U huggingface_hub", baseline_script)
            self.assertNotIn("huggingface-cli download", baseline_script)
            self.assertIn("qingtiannihao/BrepARG", baseline_script)
            self.assertIn("checkpoint/weights/abc_ar.pt", baseline_script)
            self.assertIn("checkpoint/weights/abc_vqvae.pt", baseline_script)
            self.assertIn("tools\\audit_breparg_baseline_outputs.py", baseline_script)
            self.assertIn("breparg_baseline_quality_summary.json", baseline_script)
            self.assertIn("breparg_baseline_quality_summary.md", baseline_script)

            prepare_fallback_inputs_script = (
                root / "scripts" / "03a_prepare_breparg_same_data_inputs.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\prepare_breparg_same_data_inputs.py", prepare_fallback_inputs_script)
            self.assertIn("medium", prepare_fallback_inputs_script.lower())
            self.assertIn("--train-limit", prepare_fallback_inputs_script)
            self.assertIn("--train-limit 10000", prepare_fallback_inputs_script)
            self.assertIn("--val-limit 1000", prepare_fallback_inputs_script)
            self.assertIn("--test-limit 1000", prepare_fallback_inputs_script)
            self.assertIn("same_data_input_summary.json", prepare_fallback_inputs_script)

            prepare_fallback_inputs_full_script = (
                root / "scripts" / "03a_prepare_breparg_same_data_inputs_full.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\prepare_breparg_same_data_inputs.py", prepare_fallback_inputs_full_script)
            self.assertIn("--train-limit 50000", prepare_fallback_inputs_full_script)
            self.assertIn("--val-limit 5000", prepare_fallback_inputs_full_script)
            self.assertIn("--test-limit 5000", prepare_fallback_inputs_full_script)
            self.assertIn("data_full", prepare_fallback_inputs_full_script)

            fallback_script = (root / "scripts" / "03b_breparg_same_data_training_fallback.ps1").read_text(encoding="utf-8")
            self.assertIn("function Invoke-Native", fallback_script)
            self.assertIn("$previousPreference = $ErrorActionPreference", fallback_script)
            self.assertIn("$ErrorActionPreference = \"Continue\"", fallback_script)
            self.assertIn("$ErrorActionPreference = $previousPreference", fallback_script)
            self.assertIn("function Test-PythonModule", fallback_script)
            self.assertIn("tensorboard", fallback_script)
            self.assertIn("pip install tensorboard", fallback_script)
            self.assertIn("if ($exitCode -ne 0)", fallback_script)
            self.assertIn("BrepARG\\train_vqvae.py", fallback_script)
            self.assertIn("BrepARG\\2sequence.py", fallback_script)
            self.assertIn("BrepARG\\train_ar.py", fallback_script)
            self.assertIn("BrepARG\\generate_brep.py", fallback_script)
            self.assertIn("--dataset_type abc", fallback_script)
            self.assertIn("tools\\audit_breparg_baseline_outputs.py", fallback_script)
            self.assertIn("same_data_split.pkl", fallback_script)
            self.assertIn("deduplicated_surface_source.pkl", fallback_script)
            self.assertIn("deduplicated_edge_source.pkl", fallback_script)
            self.assertIn("same-data", fallback_script)
            self.assertIn("--tb_log_dir $VQVAE_TB", fallback_script)
            self.assertIn("--tb_log_dir $AR_TB", fallback_script)
            self.assertIn("--no_aug", fallback_script)

            fallback_preflight_script = (
                root / "scripts" / "03b_preflight_breparg_same_data_fallback.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("tools\\preflight_breparg_same_data_fallback.py", fallback_preflight_script)
            self.assertIn("--official-incompat-report", fallback_preflight_script)
            self.assertIn("breparg_same_data_preflight.json", fallback_preflight_script)

            fallback_smoke_script = (
                root / "scripts" / "03b_smoke_breparg_same_data_training_fallback.ps1"
            ).read_text(encoding="utf-8")
            self.assertIn("data_smoke", fallback_smoke_script)
            self.assertIn("same_data_breparg_fallback_smoke_manifest.json", fallback_smoke_script)
            self.assertIn("$VQVAE_EPOCHS = \"1\"", fallback_smoke_script)
            self.assertIn("$AR_EPOCHS = \"1\"", fallback_smoke_script)
            self.assertIn("--num_samples $GENERATE_SAMPLES", fallback_smoke_script)
            self.assertIn("breparg_same_data_smoke_quality_summary.json", fallback_smoke_script)
            self.assertIn("--no_aug", fallback_smoke_script)

            summarize_script = (root / "scripts" / "04_summarize_reports.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\analyze_reconstruction_fsq_correlation.py", summarize_script)
            self.assertIn("reconstruction_fsq_correlation.json", summarize_script)
            self.assertIn("00_fsq_only_patch_metrics", summarize_script)
            self.assertIn("01_teacher_forcing_true_token_reconstruction", summarize_script)

            audit_script = (root / "scripts" / "05_audit_suite_status.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\audit_complex_curved_control_suite.py", audit_script)
            self.assertIn("suite_status.json", audit_script)
            self.assertIn("suite_status.md", audit_script)

            migration_script = (root / "scripts" / "06_prepare_external_ssd_migration.ps1").read_text(encoding="utf-8")
            self.assertIn("param(", migration_script)
            self.assertIn("$DestRoot", migration_script)
            self.assertIn("$Execute", migration_script)
            self.assertIn("$CopyReferenceModels", migration_script)
            self.assertIn("$CopyArchives", migration_script)
            self.assertIn("tools\\prepare_rootcause_ssd_migration.py", migration_script)
            self.assertIn("ssd_migration_plan.json", migration_script)
            self.assertIn("ssd_migration_commands.md", migration_script)

            summarize_script = (root / "scripts" / "04_summarize_reports.ps1").read_text(encoding="utf-8")
            self.assertIn("tools\\compare_fsq_capacity_diagnostics.py", summarize_script)
            self.assertIn("fsq_capacity_comparison.json", summarize_script)
            self.assertIn("fsq_capacity_comparison.md", summarize_script)

    def test_rootcause_ssd_migration_dry_run_records_copy_plan(self):
        from tools.prepare_complex_curved_control_workspace import prepare_workspace
        from tools.prepare_rootcause_ssd_migration import render_commands, run_migration

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            refs = Path(tmp) / "refs"
            refs.mkdir()
            sequence = refs / "sequences_fsq_rcm.pkl"
            vqvae = refs / "fsq_vqvae_best.pt"
            ar = refs / "ar_best.pt"
            archives = refs / "archives"
            sequence.write_bytes(b"seq")
            vqvae.write_bytes(b"vq")
            ar.write_bytes(b"ar")
            archives.mkdir()
            (archives / "abc_0000_parsed.zip").write_bytes(b"zip")

            prepare_workspace(
                output_dir=root,
                python_exe="PYTHON",
                sequence_path=sequence,
                vqvae_checkpoint=vqvae,
                ar_checkpoint=ar,
                archive_root=archives,
            )
            (root / "experiments" / "marker.txt").write_text("ok", encoding="utf-8")
            (root / "requirement_audit_20260715.md").write_text("audit", encoding="utf-8")
            (root / "current_status_answer_20260715_1823.md").write_text("status", encoding="utf-8")
            (root / "current_status_answer_20260715_1851.md").write_text("latest", encoding="utf-8")
            (root / "fsq_capacity_completion_handoff_20260715.md").write_text("handoff", encoding="utf-8")
            (root / "partial_epoch5_fsq_capacity_readout_20260715.md").write_text(
                "partial", encoding="utf-8"
            )

            payload = run_migration(
                suite_root=root,
                dest_root=Path(tmp) / "ssd",
                copy_reference_models=True,
                copy_archives=False,
                execute=False,
            )

            self.assertEqual(payload["status"], "DRY_RUN")
            self.assertTrue(payload["ready_to_execute"])
            labels = {item["label"] for item in payload["items"]}
            self.assertIn("experiments", labels)
            self.assertIn("requirement_audit", labels)
            self.assertIn("current_status_answer", labels)
            status_items = [item for item in payload["items"] if item["label"] == "current_status_answer"]
            self.assertEqual(
                {Path(item["source"]).name for item in status_items},
                {"current_status_answer_20260715_1823.md", "current_status_answer_20260715_1851.md"},
            )
            self.assertIn("fsq_capacity_completion_handoff", labels)
            self.assertIn("partial_epoch5_fsq_capacity_readout", labels)
            self.assertIn("reference_sequence", labels)
            self.assertIn("reference_vqvae", labels)
            self.assertIn("reference_ar", labels)
            self.assertNotIn("parsed_archives", labels)
            self.assertEqual(payload["copied"], [])
            self.assertIn("sequence_path", payload["regenerated_config"])

            commands = render_commands(payload, "PYTHON")
            self.assertIn("01a_preflight_fsq_capacity_candidate.ps1", commands)
            self.assertIn("fsq_capacity_watch_then_eval.log", commands)
            self.assertIn("do not start a duplicate resume", commands)
            self.assertIn("01a_resume_fsq_capacity_candidate.ps1", commands)
            self.assertIn("01a_train_fsq_capacity_candidate.ps1", commands)
            self.assertIn("03b_preflight_breparg_same_data_fallback.ps1", commands)
            self.assertIn("Run this after the FSQ capacity GPU job is done", commands)
            self.assertIn("03b_breparg_same_data_training_fallback.ps1", commands)


if __name__ == "__main__":
    unittest.main()
