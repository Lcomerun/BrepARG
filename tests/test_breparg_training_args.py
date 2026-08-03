import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


class BrepARGTrainingArgsTests(unittest.TestCase):
    def test_se_args_expose_no_aug_flag_for_smoke_and_windows_runs(self):
        sys.path.insert(0, str(Path("BrepARG").resolve()))
        from utils import get_se_args

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_vqvae.py", "--no_aug"]
            args = get_se_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.no_aug)

    def test_training_args_expose_target_val_loss_stop_threshold(self):
        sys.path.insert(0, str(Path("BrepARG").resolve()))
        from utils import get_ar_args, get_se_args

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_vqvae.py", "--target_val_loss", "1e-6"]
            se_args = get_se_args()
            sys.argv = ["train_ar.py", "--target_val_loss", "0.1"]
            ar_args = get_ar_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(se_args.target_val_loss, 1e-6)
        self.assertEqual(ar_args.target_val_loss, 0.1)

    def test_breparg_best_checkpoints_do_not_force_epoch_checkpoints(self):
        source = Path("BrepARG/trainer.py").read_text(encoding="utf-8")

        self.assertIn("save_model(is_best=True, save_epoch=val_epoch, save_regular=False)", source)
        self.assertIn("save_checkpoint(is_best=True, save_regular=False)", source)
        self.assertIn("if save_regular:", source)

    def test_2sequence_passes_dataset_type_to_vqvae_loader(self):
        source = Path("BrepARG/2sequence.py").read_text(encoding="utf-8")
        self.assertIn(
            "load_se_vqvae_model(args.vqvae_se_weight, False, args.dataset_type, device)",
            source,
        )

    def test_vqvae_trainer_uses_abc_codebook_size(self):
        source = Path("BrepARG/trainer.py").read_text(encoding="utf-8")
        self.assertIn("num_vq_embeddings = 8192 if args.dataset_type == 'abc' else 4096", source)
        self.assertIn("num_vq_embeddings=num_vq_embeddings", source)

    def test_generate_brep_batch_exposes_max_attempts(self):
        source = Path("BrepARG/generate_brep.py").read_text(encoding="utf-8")
        self.assertIn("max_attempts: Optional[int] = None", source)
        self.assertIn("while saved_count < num_samples and total_attempts < effective_max_attempts", source)
        self.assertIn('parser.add_argument("--max_attempts"', source)
        self.assertIn("max_attempts=args.max_attempts", source)

    def test_generate_brep_batch_stops_at_max_attempts_when_all_attempts_fail(self):
        sys.path.insert(0, str(Path("BrepARG").resolve()))
        import generate_brep

        original = generate_brep.generate_and_reconstruct_single

        def always_fail(**_kwargs):
            return {
                "stl_saved": False,
                "step_saved": False,
                "error": "intentional test failure",
            }

        try:
            generate_brep.generate_and_reconstruct_single = always_fail
            with redirect_stdout(StringIO()):
                result = generate_brep.generate_and_reconstruct_batch(
                    ar_model=None,
                    se_vqvae_model=None,
                    vocab_info={},
                    device="cpu",
                    num_samples=2,
                    output_dir="local_runs/test_generate_brep_max_attempts",
                    max_attempts=3,
                )
        finally:
            generate_brep.generate_and_reconstruct_single = original

        self.assertEqual(result["total_attempts"], 3)
        self.assertEqual(result["saved_count"], 0)
        self.assertEqual(result["max_attempts"], 3)

    def test_generate_brep_uses_checkpoint_args_max_seq_len(self):
        sys.path.insert(0, str(Path("BrepARG").resolve()))
        import generate_brep

        checkpoint = {
            "args": SimpleNamespace(
                max_seq_len=1536,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
            )
        }

        with redirect_stdout(StringIO()):
            model = generate_brep.init_ar_model(
                vocab_size=128,
                pad_token_id=127,
                checkpoint=checkpoint,
                device="cpu",
            )

        self.assertEqual(model.max_seq_len, 1536)
        self.assertEqual(model.config.n_positions, 1536)

    def test_joint_optimize_does_not_hardcode_cuda(self):
        source = Path("BrepARG/utils.py").read_text(encoding="utf-8")

        self.assertIn("def joint_optimize_device", source)
        self.assertIn("model = model.to(opt_device).train()", source)
        self.assertIn("torch.FloatTensor(all_pnts).to(opt_device)", source)
        self.assertIn("torch.FloatTensor(surf_wcs_init).to(opt_device)", source)
        self.assertNotIn("model = model.cuda().train()", source)
        self.assertNotIn("torch.FloatTensor(all_pnts).cuda()", source)
        self.assertNotIn("torch.FloatTensor(surf_wcs_init).cuda()", source)

    def test_generate_brep_uses_serial_file_write_on_windows(self):
        source = Path("BrepARG/generate_brep.py").read_text(encoding="utf-8")

        self.assertIn("def write_worker_payload", source)
        self.assertIn('os.environ.get("BREPARG_SERIAL_WRITE", "0") == "1"', source)
        self.assertIn("if os.name == 'nt' or serial_write:", source)
        self.assertIn("result = write_worker_payload(temp_step_path, step_path, stl_path)", source)
        self.assertIn("status = result.get('status', 'error')", source)
        self.assertIn("error = result.get('error', 'Unknown error')", source)

    def test_smoke_fallback_passes_generation_max_attempts(self):
        source = Path("tools/prepare_complex_curved_control_workspace.py").read_text(encoding="utf-8")
        self.assertIn('generate_max_attempts = "20" if smoke else "5000"', source)
        self.assertIn('$GENERATE_MAX_ATTEMPTS = "{generate_max_attempts}"', source)
        self.assertIn("--max_attempts $GENERATE_MAX_ATTEMPTS", source)

    def test_validate_breparg_generated_directory_writes_quality_outputs(self):
        source = Path("tools/validate_breparg_generated_directory.py").read_text(encoding="utf-8")

        self.assertIn("validate_step_quality_once.py", source)
        self.assertIn("--manifest-output", source)
        self.assertIn("--summary-output", source)
        self.assertIn("quality_check", source)
        self.assertIn('"png_saved": sum(1 for row in rows if row.get("png_saved"))', source)

    def test_d_drive_breparg_fallback_stages_same_data_inputs_before_training(self):
        source = Path("tools/run_breparg_same_data_fallback_on_d.ps1").read_text(encoding="utf-8")

        self.assertIn('$StagedData = "$Root\\data_staged"', source)
        self.assertIn("function Stage-SameDataInputs", source)
        self.assertIn("function Invoke-RobocopyOk", source)
        self.assertIn("& robocopy @args", source)
        self.assertIn("-Source \"$SourceData\\parsed_pool\"", source)
        self.assertIn("-Destination \"$DestinationData\\parsed_pool\"", source)
        self.assertIn('V13_BREPARG_SOURCE_DATA', source)
        self.assertIn('V13_BREPARG_STAGED_DATA', source)
        self.assertIn('with (dest / "same_data_split.pkl").open("wb") as handle:', source)
        self.assertIn('"staged_data": str(dest)', source)

        stage_call = source.index("Stage-SameDataInputs -SourceData $SourceData -DestinationData $StagedData")
        staged_assignment = source.index("$Data = $StagedData")
        first_train_use = source.index("--data_list $Split")
        self.assertLess(stage_call, staged_assignment)
        self.assertLess(staged_assignment, first_train_use)

        self.assertIn("source_data = $SourceData", source)
        self.assertIn("staged_data = $StagedData", source)


if __name__ == "__main__":
    unittest.main()
