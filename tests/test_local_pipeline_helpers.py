import json
import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"


class LocalPipelineHelperTests(unittest.TestCase):
    def write_parsed_shape(self, path, n_faces, n_edges, curved=False):
        import pickle
        import numpy as np

        surf = np.zeros((n_faces, 32, 32, 3), dtype=np.float32)
        edge = np.zeros((n_edges, 32, 3), dtype=np.float32)
        if curved:
            grid = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            x, y = np.meshgrid(grid, grid, indexing="ij")
            z = 0.2 * np.sin(np.pi * x) * np.sin(np.pi * y)
            for face_index in range(n_faces):
                surf[face_index, :, :, 0] = x
                surf[face_index, :, :, 1] = y
                surf[face_index, :, :, 2] = z + face_index * 0.001
            t = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            for edge_index in range(n_edges):
                edge[edge_index, :, 0] = t
                edge[edge_index, :, 1] = 0.2 * np.sin(np.pi * t)
                edge[edge_index, :, 2] = edge_index * 0.001
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"surf_ncs": surf, "edge_ncs": edge}, handle)

    def test_select_chunks_uses_chunk_numbers_not_sorted_positions(self):
        from tools.prepare_ssd_pipeline import find_chunk_dirs, select_chunks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "abc_0004_step_v00").mkdir()
            (root / "abc_0002_step_v00").mkdir()
            (root / "not_a_chunk").mkdir()

            chunks = find_chunk_dirs(root)
            self.assertEqual([item["chunk"] for item in chunks], [2, 4])
            selected = select_chunks(chunks, "4")
            self.assertEqual([item["chunk"] for item in selected], [4])

    def test_chunk_range_selection_is_inclusive(self):
        from tools.prepare_ssd_pipeline import find_chunk_dirs, select_chunks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for chunk in [0, 1, 2, 4]:
                (root / f"abc_{chunk:04d}_step_v00").mkdir()

            chunks = find_chunk_dirs(root)
            selected = select_chunks(chunks, "1-3")
            self.assertEqual([item["chunk"] for item in selected], [1, 2])

    def test_training_env_renders_windows_paths(self):
        from tools.prepare_ssd_pipeline import build_training_env

        env = build_training_env(
            {
                "paths": {
                    "parsed_root": r"E:\breparg_data\abc_parsed_full",
                    "train_out_root": r"E:\breparg_data",
                },
                "training": {
                    "run_name": "newscheme_full_local",
                    "ns_n": 999999,
                    "ns_vq_samples": 300000,
                    "ns_vq_epochs": 200,
                    "ns_ar_epochs": 120,
                    "ns_vq_bs": 128,
                    "ns_ar_bs": 8,
                    "ns_ar_max_seq_len": 1536,
                    "ns_vq_min_epochs": 12,
                    "ns_vq_patience": 8,
                    "ns_vq_min_delta": 1e-5,
                    "ns_vq_max_nonfinite_val_epochs": 2,
                    "ns_disable_amp_vqvae": True,
                    "ns_vq_complex_fraction": 0.40,
                    "ns_vq_complex_min_faces": 12,
                    "ns_vq_complex_min_edges": 20,
                    "ns_vq_curved_fraction": 0.25,
                },
            }
        )
        self.assertEqual(env["NS_POOL"], r"E:\breparg_data\abc_parsed_full")
        self.assertEqual(env["NS_OUTBASE"], r"E:\breparg_data")
        self.assertEqual(env["NS_OUT"], "newscheme_full_local")
        self.assertEqual(env["NS_VQ_BS"], "128")
        self.assertEqual(env["NS_VQ_MIN_EPOCHS"], "12")
        self.assertEqual(env["NS_VQ_PATIENCE"], "8")
        self.assertEqual(env["NS_VQ_MIN_DELTA"], "1e-05")
        self.assertEqual(env["NS_VQ_MAX_NONFINITE_VAL_EPOCHS"], "2")
        self.assertEqual(env["NS_DISABLE_AMP_VQVAE"], "1")
        self.assertEqual(env["NS_VQ_COMPLEX_FRACTION"], "0.4")
        self.assertEqual(env["NS_VQ_COMPLEX_MIN_FACES"], "12")
        self.assertEqual(env["NS_VQ_COMPLEX_MIN_EDGES"], "20")
        self.assertEqual(env["NS_VQ_CURVED_FRACTION"], "0.25")
        self.assertEqual(env["NS_AR_MAX_SEQ_LEN"], "1536")

    def test_training_env_parses_disable_amp_string_false(self):
        from tools.prepare_ssd_pipeline import build_training_env

        env = build_training_env(
            {
                "paths": {
                    "parsed_root": r"E:\breparg_data\abc_parsed_full",
                    "train_out_root": r"E:\breparg_data",
                },
                "training": {
                    "ns_disable_amp_vqvae": "false",
                },
            }
        )
        self.assertEqual(env["NS_DISABLE_AMP_VQVAE"], "0")

    def test_process_abc_windows_out_dir_is_per_chunk(self):
        from tools.process_abc_windows import output_dir_for_chunk

        out = output_dir_for_chunk(Path(r"E:\breparg_data\abc_parsed_full"), 4)
        self.assertEqual(str(out), r"E:\breparg_data\abc_parsed_full\abc_0004")

    def test_gnn_node_features_does_not_crash_numpy_blas(self):
        code = r"""
import sys
import numpy as np
sys.path.insert(0, r'D:\luolin\V13\breparg_improvements')
from gnn_ordering import node_features
edge_face_pairs = [(i, i + 1) for i in range(11)]
face_geom = np.random.RandomState(0).randn(12, 6).astype('float32')
X, dim = node_features(edge_face_pairs, 12, face_geom)
print(tuple(X.shape), dim)
"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn("(12, 8) 8", proc.stdout)

    def test_archive_pipeline_discovers_and_selects_archives_by_chunk_id(self):
        from tools.run_ssd_archive_pipeline import discover_archives, discover_chunk_inputs, select_archives

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for chunk in [4, 0, 2]:
                (root / f"abc_{chunk:04d}_step_v00.7z").write_bytes(b"fake")
            (root / "abc_0001_step_v00").mkdir()
            (root / "success.zip").write_bytes(b"ignore")

            archives = discover_archives(root)
            self.assertEqual([item["chunk"] for item in archives], [0, 2, 4])
            selected = select_archives(archives, "2-4")
            self.assertEqual([item["chunk"] for item in selected], [2, 4])
            inputs = discover_chunk_inputs(root)
            self.assertEqual([item["chunk"] for item in inputs], [0, 1, 2, 4])
            self.assertIn("extract_dir", inputs[1])

    def test_archive_pipeline_paths_are_on_expected_roots(self):
        from tools.run_ssd_archive_pipeline import paths_for_archive

        row = {"chunk": 7, "archive": r"E:\ABC\step\abc_0007_step_v00.7z"}
        paths = paths_for_archive(row, Path(r"E:\ABC\processed\abc_parsed_full"))
        self.assertEqual(str(paths["extract_dir"]), r"E:\ABC\step\abc_0007_step_v00")
        self.assertEqual(str(paths["parsed_chunk_dir"]), r"E:\ABC\processed\abc_parsed_full\abc_0007")

    def test_process_abc_windows_manifest_resume_skips_terminal_rows(self):
        from tools.process_abc_windows import (
            load_manifest_by_step,
            manifest_path_for_chunk,
            previous_terminal_row,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            step = root / "abc_0000_step_v00" / "00000001" / "part.step"
            step.parent.mkdir(parents=True)
            step.write_text("fake", encoding="utf-8")
            manifest = manifest_path_for_chunk(out, 0)
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"chunk": 0, "status": "multi", "step_path": "' + str(step).replace("\\", "\\\\") + '"}\n',
                encoding="utf-8",
            )

            rows = load_manifest_by_step([manifest])
            task = {"chunk": 0, "step": step}
            previous = previous_terminal_row(task, out, rows)
            self.assertEqual(previous["status"], "multi")

    def test_process_abc_windows_failure_rate_threshold(self):
        from tools.process_abc_windows import chunk_has_too_many_failures

        self.assertFalse(chunk_has_too_many_failures({"error": 1, "timeout": 1}, 100, 0.05))
        self.assertTrue(chunk_has_too_many_failures({"error": 5, "timeout": 1}, 100, 0.05))

    def test_vqvae_stop_state_stops_after_consecutive_nonfinite_validation(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from training_stability import VQVAEStopConfig, VQVAEStopState, update_vqvae_stop_state

        cfg = VQVAEStopConfig(min_epochs=3, patience=0, max_nonfinite_val_epochs=2, min_delta=1e-5)
        state = VQVAEStopState()

        state, improved, should_stop = update_vqvae_stop_state(0, 0.01, state, cfg)
        self.assertTrue(improved)
        self.assertFalse(should_stop)
        self.assertEqual(state.best_epoch, 0)

        state, improved, should_stop = update_vqvae_stop_state(3, float("inf"), state, cfg)
        self.assertFalse(improved)
        self.assertFalse(should_stop)
        self.assertEqual(state.consecutive_nonfinite_val_epochs, 1)

        state, improved, should_stop = update_vqvae_stop_state(4, float("inf"), state, cfg)
        self.assertFalse(improved)
        self.assertTrue(should_stop)
        self.assertEqual(state.stop_reason, "nonfinite_val_epochs=2")

    def test_vqvae_stop_state_resets_nonfinite_counter_on_finite_improvement(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from training_stability import VQVAEStopConfig, VQVAEStopState, update_vqvae_stop_state

        cfg = VQVAEStopConfig(min_epochs=1, patience=4, max_nonfinite_val_epochs=2, min_delta=1e-5)
        state = VQVAEStopState()
        state, _, _ = update_vqvae_stop_state(0, 0.01, state, cfg)
        state, _, _ = update_vqvae_stop_state(1, float("inf"), state, cfg)
        self.assertEqual(state.consecutive_nonfinite_val_epochs, 1)

        state, improved, should_stop = update_vqvae_stop_state(2, 0.009, state, cfg)
        self.assertTrue(improved)
        self.assertFalse(should_stop)
        self.assertEqual(state.consecutive_nonfinite_val_epochs, 0)
        self.assertEqual(state.best_epoch, 2)

    def test_vqvae_metric_helpers_do_not_turn_missing_batches_into_zero(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from training_stability import finite_average, safe_json_number

        self.assertEqual(finite_average(0.0, 0), float("inf"))
        self.assertIsNone(safe_json_number(float("inf")))
        self.assertEqual(safe_json_number(0.00082), 0.00082)

    def test_vqvae_history_summary_supports_continuation_from_previous_best(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from training_stability import summarize_vqvae_history

        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "vqvae_history.json"
            history.write_text(
                """
{
  "tag": "vqvae",
  "history": [
    {"epoch": 38, "train_loss": 0.00040, "val_loss": 0.00058, "best_val": 0.00056, "best_epoch": 35},
    {"epoch": 39, "train_loss": 0.00038, "val_loss": 0.00055, "best_val": 0.00056, "best_epoch": 35}
  ],
  "best_val_recon": 0.00056,
  "best_epoch": 35
}
""".strip(),
                encoding="utf-8",
            )

            summary = summarize_vqvae_history(history)

        self.assertEqual(summary["history_count"], 2)
        self.assertEqual(summary["last_epoch"], 39)
        self.assertEqual(summary["next_epoch"], 40)
        self.assertEqual(summary["best_epoch"], 35)
        self.assertEqual(summary["best_val_recon"], 0.00056)
        self.assertEqual(summary["final_val"], 0.00055)

    def test_vqvae_continuation_epochs_rejects_non_increasing_target(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from training_stability import continuation_epoch_count

        self.assertEqual(continuation_epoch_count(40, 100), 60)
        with self.assertRaises(ValueError):
            continuation_epoch_count(40, 40)

    def test_vqvae_complex_sampler_reserves_complex_records(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        import pickle
        import numpy as np
        from vqvae_sampling import collect_vqvae_sample_records

        def write_shape(path, n_faces, n_edges):
            surf = np.zeros((n_faces, 32, 32, 3), dtype=np.float32)
            edge = np.zeros((n_edges, 32, 3), dtype=np.float32)
            with open(path, "wb") as handle:
                pickle.dump({"surf_ncs": surf, "edge_ncs": edge}, handle)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            simple = root / "simple.pkl"
            complex_shape = root / "complex.pkl"
            write_shape(simple, n_faces=2, n_edges=2)
            write_shape(complex_shape, n_faces=14, n_edges=24)

            records, summary = collect_vqvae_sample_records(
                [simple, complex_shape],
                cap=10,
                seed=0,
                complex_fraction=0.5,
                complex_min_faces=12,
                complex_min_edges=20,
            )

        complex_records = [record for record in records if record["is_complex_source"]]
        self.assertGreaterEqual(len(complex_records), 5)
        self.assertEqual(summary["complex_target"], 5)
        self.assertEqual(summary["complex_records_selected"], len(complex_records))

    def test_vqvae_curved_sampler_prefers_high_curvature_patches(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        import numpy as np
        from vqvae_sampling import select_patch_records

        flat_patch = np.zeros((32, 32, 3), dtype=np.float32)
        curved_patch = np.zeros((32, 32, 3), dtype=np.float32)
        curved_patch[:, :, 2] = 0.5

        records = [
            {"record_id": "flat", "array": flat_patch, "curvature_score": 0.0},
            {"record_id": "curved", "array": curved_patch, "curvature_score": 1.0},
        ]

        selected = select_patch_records(records, target=1, curved_fraction=1.0, seed=0)

        self.assertEqual([record["record_id"] for record in selected], ["curved"])

    def test_vqvae_sampler_excludes_previously_selected_ids(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        import numpy as np
        from vqvae_sampling import select_patch_records

        records = [
            {
                "record_id": f"patch_{index}",
                "array": np.zeros((32, 32, 3), dtype=np.float32),
                "curvature_score": 0.0,
            }
            for index in range(20)
        ]

        selected = select_patch_records(
            records,
            target=10,
            curved_fraction=0.0,
            seed=0,
            exclude_ids={f"patch_{index}" for index in range(8)},
        )

        selected_ids = {record["record_id"] for record in selected}
        self.assertEqual(len(selected), 10)
        self.assertTrue(selected_ids.isdisjoint({f"patch_{index}" for index in range(8)}))

    def test_vqvae_sampler_respects_downstream_source_caps(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from vqvae_sampling import collect_vqvae_sample_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "kept.pkl"
            too_large = root / "too_large.pkl"
            self.write_parsed_shape(kept, n_faces=12, n_edges=20, curved=True)
            self.write_parsed_shape(too_large, n_faces=60, n_edges=200, curved=True)

            records, summary = collect_vqvae_sample_records(
                [kept, too_large],
                cap=100,
                seed=0,
                max_source_faces=50,
                max_source_edges=150,
            )

        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["n_faces"] <= 50 and record["n_edges"] <= 150 for record in records))
        self.assertEqual(summary["dropped_records_source_cap"], 260)
        self.assertEqual(summary["max_source_faces"], 50)
        self.assertEqual(summary["max_source_edges"], 150)

    def test_vqvae_patch_weights_emphasize_complex_and_curved_records(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from vqvae_sampling import records_to_patch_weights

        records = [
            {"record_id": "simple_flat", "is_complex_source": False, "curvature_score": 0.0},
            {"record_id": "complex_flat", "is_complex_source": True, "curvature_score": 0.0},
            {"record_id": "simple_curved", "is_complex_source": False, "curvature_score": 0.05},
            {"record_id": "complex_curved", "is_complex_source": True, "curvature_score": 0.05},
        ]

        weights = records_to_patch_weights(
            records,
            complex_weight=1.5,
            curved_weight=2.0,
            curved_threshold=0.02,
        )

        self.assertEqual(weights.tolist(), [1.0, 1.5, 2.0, 3.0])

    def test_build_parsed_shard_preserves_source_payloads(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))
        import pickle
        from build_parsed_shards import build_chunk_shard, verify_parsed_shard
        from sharded_data import iter_shard_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed_root = root / "parsed"
            chunk_dir = parsed_root / "abc_0000"
            chunk_dir.mkdir(parents=True)
            for index in range(2):
                with (chunk_dir / f"shape_{index}.pkl").open("wb") as handle:
                    pickle.dump({"index": index}, handle)

            shard_root = root / "shards"
            row = build_chunk_shard(
                chunk_dir,
                parsed_root=parsed_root,
                shard_root=shard_root,
                compression="gzip",
                compression_level=1,
                resume=False,
                delete_after_verify=False,
            )

            self.assertEqual(row["status"], "built_verified")
            self.assertEqual(row["source_count"], 2)
            verified = verify_parsed_shard(Path(row["shard"]), expected_chunk="abc_0000", expected_count=2)
            self.assertEqual(verified["source_count"], 2)
            records = list(iter_shard_records(row["shard"]))
            self.assertEqual(records[0]["format"], "v13.parsed_shard.v1")
            self.assertEqual(pickle.loads(records[1]["payload"])["index"], 0)

    def test_vqvae_sampler_reads_patch_shards(self):
        sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))
        import numpy as np
        from sharded_data import PATCH_SHARD_FORMAT, dump_shard_record, open_shard_writer
        from vqvae_sampling import collect_vqvae_patch_shard_records

        with tempfile.TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / "vq_patch_shard_0000.pkl.gz"
            with open_shard_writer(shard_path, compression="gzip", level=1) as handle:
                dump_shard_record(
                    handle,
                    {
                        "record_type": "vq_patch_shard_header",
                        "format": PATCH_SHARD_FORMAT,
                    },
                )
                for index, complex_source in enumerate([False, True, True]):
                    dump_shard_record(
                        handle,
                        {
                            "record_type": "vq_patch",
                            "record_id": f"shape:{index}",
                            "source_path": f"abc_0000/shape_{index}.pkl",
                            "chunk_id": "abc_0000",
                            "kind": "surface",
                            "array": np.zeros((32, 32, 3), dtype=np.float32),
                            "curvature_score": 0.05 if complex_source else 0.0,
                            "n_faces": 20 if complex_source else 2,
                            "n_edges": 30 if complex_source else 2,
                            "is_complex_source": complex_source,
                        },
                    )

            records, summary = collect_vqvae_patch_shard_records(
                [shard_path],
                cap=2,
                seed=0,
                complex_fraction=0.5,
                curved_fraction=0.5,
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["loaded_shards"], 1)
        self.assertEqual(summary["failed_shards"], 0)
        self.assertGreaterEqual(summary["complex_records_selected"], 1)
        self.assertEqual(summary["unique_sources_available"], 3)

    def test_weighted_reconstruction_loss_uses_per_sample_weights(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        import torch
        import train

        recon = torch.tensor([[[[0.0]]], [[[2.0]]]], dtype=torch.float32)
        target = torch.zeros_like(recon)
        weights = torch.tensor([1.0, 3.0], dtype=torch.float32)

        loss = train.weighted_reconstruction_loss(recon, target, weights)

        self.assertAlmostEqual(float(loss.item()), 3.0)

    def test_vqvae_bucket_diagnostic_summarizes_complex_and_curved_losses(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from diagnose_vqvae_buckets import summarize_bucket_losses

        records = [
            {
                "record_id": "simple_flat",
                "kind": "surface",
                "is_complex_source": False,
                "curvature_score": 0.0,
            },
            {
                "record_id": "complex_flat",
                "kind": "surface",
                "is_complex_source": True,
                "curvature_score": 0.0,
            },
            {
                "record_id": "complex_curved",
                "kind": "edge",
                "is_complex_source": True,
                "curvature_score": 0.05,
            },
        ]
        losses = [1e-5, 2e-5, 8e-5]

        summary = summarize_bucket_losses(records, losses, target_loss=1e-6, curved_threshold=0.02)

        self.assertEqual(summary["all"]["count"], 3)
        self.assertAlmostEqual(summary["all"]["mean"], 3.6666666666666666e-5)
        self.assertEqual(summary["simple_source"]["count"], 1)
        self.assertEqual(summary["complex_source"]["count"], 2)
        self.assertEqual(summary["curved_patch"]["count"], 1)
        self.assertEqual(summary["edge"]["count"], 1)
        self.assertEqual(summary["complex_curved_patch"]["count"], 1)
        self.assertAlmostEqual(summary["complex_source"]["mean"], 5e-5)
        self.assertAlmostEqual(summary["curved_patch"]["target_ratio"], 80.0)

    def test_vqvae_bucket_diagnostic_compares_best_and_final_checkpoints(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from diagnose_vqvae_buckets import compare_bucket_summaries

        best = {
            "all": {"count": 4, "mean": 4.0e-5},
            "complex_source": {"count": 2, "mean": 5.0e-5},
            "curved_patch": {"count": 1, "mean": 7.0e-5},
        }
        final = {
            "all": {"count": 4, "mean": 5.5e-5},
            "complex_source": {"count": 2, "mean": 8.0e-5},
            "curved_patch": {"count": 1, "mean": 9.0e-5},
        }

        comparison = compare_bucket_summaries({"best": best, "final": final}, reference_label="best")

        self.assertEqual(comparison["reference_label"], "best")
        self.assertAlmostEqual(comparison["checkpoints"]["final"]["all"]["delta_mean"], 1.5e-5)
        self.assertAlmostEqual(comparison["checkpoints"]["final"]["complex_source"]["relative_mean"], 1.6)
        self.assertIn("complex_source", comparison["worst_regression_buckets"])

    def test_vqvae_bucket_diagnostic_reports_top_loss_records(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from diagnose_vqvae_buckets import top_loss_records

        records = [
            {
                "record_id": "low",
                "source_path": "shape_a.pkl",
                "kind": "surface",
                "is_complex_source": False,
                "curvature_score": 0.0,
                "n_faces": 2,
                "n_edges": 4,
            },
            {
                "record_id": "high",
                "source_path": "shape_b.pkl",
                "kind": "edge",
                "is_complex_source": True,
                "curvature_score": 0.1,
                "n_faces": 18,
                "n_edges": 30,
            },
        ]

        top = top_loss_records(records, {"ckpt": [1e-5, 9e-4]}, limit=1)

        self.assertEqual(top["ckpt"][0]["record_id"], "high")
        self.assertEqual(top["ckpt"][0]["source_path"], "shape_b.pkl")
        self.assertEqual(top["ckpt"][0]["buckets"], ["all", "edge", "complex_source", "curved_patch", "complex_curved_patch"])
        self.assertAlmostEqual(top["ckpt"][0]["loss"], 9e-4)

    def test_vqvae_bucket_diagnostic_filters_records_by_downstream_caps(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from diagnose_vqvae_buckets import filter_records_by_source_caps

        records = [
            {"record_id": "kept", "n_faces": 12, "n_edges": 20},
            {"record_id": "too_many_faces", "n_faces": 51, "n_edges": 20},
            {"record_id": "too_many_edges", "n_faces": 12, "n_edges": 151},
        ]

        kept, summary = filter_records_by_source_caps(records, max_faces=50, max_edges=150)

        self.assertEqual([record["record_id"] for record in kept], ["kept"])
        self.assertEqual(summary["input_records"], 3)
        self.assertEqual(summary["kept_records"], 1)
        self.assertEqual(summary["dropped_too_many_faces"], 1)
        self.assertEqual(summary["dropped_too_many_edges"], 1)

    def test_parsed_pool_quality_audit_requires_complex_and_curved_sources(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from audit_parsed_pool_quality import audit_parsed_pool_quality

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_pool = root / "ready_pool"
            self.write_parsed_shape(ready_pool / "simple.pkl", n_faces=2, n_edges=2)
            self.write_parsed_shape(ready_pool / "complex_curved.pkl", n_faces=14, n_edges=24, curved=True)

            ready = audit_parsed_pool_quality(
                ready_pool,
                max_files=8,
                complex_min_faces=12,
                complex_min_edges=20,
                curved_score_threshold=0.02,
                min_parsed_files=2,
                min_complex_sources=1,
                min_complex_source_fraction=0.25,
                min_curved_patches=1,
                min_curved_patch_fraction=0.02,
            )

            poor_pool = root / "poor_pool"
            self.write_parsed_shape(poor_pool / "simple_a.pkl", n_faces=2, n_edges=2)
            self.write_parsed_shape(poor_pool / "simple_b.pkl", n_faces=4, n_edges=4)
            poor = audit_parsed_pool_quality(
                poor_pool,
                max_files=8,
                complex_min_faces=12,
                complex_min_edges=20,
                curved_score_threshold=0.02,
                min_parsed_files=2,
                min_complex_sources=1,
                min_complex_source_fraction=0.25,
                min_curved_patches=1,
                min_curved_patch_fraction=0.02,
            )

        self.assertEqual(ready["status"], "PARSED_POOL_QUALITY_READY")
        self.assertTrue(ready["quality_ready"])
        self.assertEqual(ready["summary"]["loaded_files"], 2)
        self.assertEqual(ready["summary"]["complex_source_files"], 1)
        self.assertGreater(ready["summary"]["curved_patch_records"], 0)
        self.assertEqual(poor["status"], "PARSED_POOL_QUALITY_FAILED")
        self.assertFalse(poor["quality_ready"])
        self.assertIn("complex_sources_below_minimum", poor["blocking_reasons"])
        self.assertIn("curved_patches_below_minimum", poor["blocking_reasons"])

    def test_epoch100_continuation_script_sets_resume_controls(self):
        script = Path(r"D:\luolin\V13\tools\run_vqvae_continue_to_epoch100.ps1")
        text = script.read_text(encoding="utf-8")

        self.assertIn('$env:NS_OUT = "newscheme_full_vqvae_epoch100"', text)
        self.assertIn('$env:NS_VQ_TARGET_EPOCH = "100"', text)
        self.assertIn('$env:NS_VQ_LR = "1e-4"', text)
        self.assertIn('$env:NS_VQ_RESUME_FROM = "E:\\ABC\\processed\\train_outputs\\newscheme_full_vqvae_stable\\fsq_vqvae_best.pt"', text)
        self.assertIn('$env:NS_VQ_HISTORY_IN = "E:\\ABC\\processed\\train_outputs\\newscheme_full_vqvae_stable\\vqvae_history.json"', text)

    def test_complex_vqvae_recovery_script_sets_sampling_controls(self):
        script = Path(r"D:\luolin\V13\tools\run_vqvae_complex_recovery.ps1")
        text = script.read_text(encoding="utf-8")

        self.assertIn('$env:NS_OUT = $RunName', text)
        self.assertIn('$env:NS_VQ_COMPLEX_FRACTION = [string]$ComplexFraction', text)
        self.assertIn('$env:NS_VQ_COMPLEX_MIN_FACES = [string]$ComplexMinFaces', text)
        self.assertIn('$env:NS_VQ_COMPLEX_MIN_EDGES = [string]$ComplexMinEdges', text)
        self.assertIn('$env:NS_VQ_CURVED_FRACTION = [string]$CurvedFraction', text)
        self.assertIn('$env:NS_VQ_MAX_SOURCE_FACES = [string]$MaxSourceFaces', text)
        self.assertIn('$env:NS_VQ_MAX_SOURCE_EDGES = [string]$MaxSourceEdges', text)
        self.assertIn('$env:NS_VQ_COMPLEX_LOSS_WEIGHT = [string]$ComplexLossWeight', text)
        self.assertIn('$env:NS_VQ_CURVED_LOSS_WEIGHT = [string]$CurvedLossWeight', text)
        self.assertIn('$env:NS_VQ_CURVED_LOSS_THRESHOLD = [string]$CurvedLossThreshold', text)
        self.assertIn('$env:NS_VQ_RESUME_FROM = $ResumeFrom', text)
        self.assertIn('$env:NS_DISABLE_AMP_VQVAE = "1"', text)

    def test_linux_complex_vqvae_recovery_script_sets_sampling_controls(self):
        script = Path(r"D:\luolin\V13\tools\run_vqvae_complex_recovery.sh")
        text = script.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn('export NS_OUT="$RUN_NAME"', text)
        self.assertIn('export NS_VQ_COMPLEX_FRACTION="$COMPLEX_FRACTION"', text)
        self.assertIn('export NS_VQ_COMPLEX_MIN_FACES="$COMPLEX_MIN_FACES"', text)
        self.assertIn('export NS_VQ_COMPLEX_MIN_EDGES="$COMPLEX_MIN_EDGES"', text)
        self.assertIn('export NS_VQ_CURVED_FRACTION="$CURVED_FRACTION"', text)
        self.assertIn('export NS_VQ_MAX_SOURCE_FACES="$MAX_SOURCE_FACES"', text)
        self.assertIn('export NS_VQ_MAX_SOURCE_EDGES="$MAX_SOURCE_EDGES"', text)
        self.assertIn('export NS_VQ_COMPLEX_LOSS_WEIGHT="$COMPLEX_LOSS_WEIGHT"', text)
        self.assertIn('export NS_VQ_CURVED_LOSS_WEIGHT="$CURVED_LOSS_WEIGHT"', text)
        self.assertIn('export NS_VQ_CURVED_LOSS_THRESHOLD="$CURVED_LOSS_THRESHOLD"', text)
        self.assertIn('export NS_VQ_RESUME_FROM="$RESUME_FROM"', text)
        self.assertIn('export NS_DISABLE_AMP_VQVAE="1"', text)
        self.assertIn('server_run_ledger.txt', text)
        self.assertIn('"$PYTHON" breparg_improvements/train.py --stage vqvae', text)

    def test_ar_launcher_allows_lr_and_epoch_overrides(self):
        script = Path(r"D:\luolin\V13\tools\run_ar_v13_epoch100.ps1")
        text = script.read_text(encoding="utf-8")

        self.assertIn("[string]$LearningRate = \"5e-4\"", text)
        self.assertIn("[int]$TargetEpochs = 120", text)
        self.assertIn("[int]$MaxSeqLen = 1024", text)
        self.assertIn("$env:NS_AR_EPOCHS = [string]$TargetEpochs", text)
        self.assertIn("$env:NS_AR_LR = $LearningRate", text)
        self.assertIn("$env:NS_AR_MAX_SEQ_LEN = [string]$MaxSeqLen", text)

    def test_linux_ar_long_context_script_sets_max_seq_len(self):
        script = Path(r"D:\luolin\V13\tools\run_ar_v13_long_context.sh")
        text = script.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn('MAX_SEQ_LEN="1024"', text)
        self.assertIn('export NS_AR_MAX_SEQ_LEN="$MAX_SEQ_LEN"', text)
        self.assertIn('export NS_AR_EPOCHS="$TARGET_EPOCHS"', text)
        self.assertIn('export NS_AR_LR="$LEARNING_RATE"', text)
        self.assertIn('SEQUENCE_SOURCE=""', text)
        self.assertIn('SPLIT_SOURCE=""', text)
        self.assertIn('server_run_ledger.txt', text)
        self.assertIn('"$PYTHON" breparg_improvements/train.py --stage ar', text)

    def test_linux_source_path_sequence_rebuild_runs_shards_and_audit(self):
        script = Path(r"D:\luolin\V13\tools\run_source_path_sequence_rebuild.sh")
        text = script.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn('RUN_NAME="newscheme_full_v13_sourcepath_sequence"', text)
        self.assertIn('--vqvae-checkpoint', text)
        self.assertIn('--split', text)
        self.assertIn('--chunks', text)
        self.assertIn('--resume', text)
        self.assertIn('server_run_ledger.txt', text)
        self.assertIn('"$PYTHON" tools/run_sharded_sequence.py', text)
        self.assertIn('--merge-output "$SEQUENCE_PATH"', text)
        self.assertIn('"$PYTHON" tools/audit_sequence_source_paths.py "$SEQUENCE_PATH"', text)
        self.assertIn('source_path_audit.json', text)

    def test_sequence_shards_merge_preserves_order_and_metadata(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from sequence_sharding import merge_sequence_shards, summarize_sequence_package

        common = {
            "vocab_size": 10294,
            "special_token_size": 4,
            "face_index_size": 50,
            "se_codebook_size": 8192,
            "bbox_index_size": 2048,
            "face_index_offset": 0,
            "se_token_offset": 50,
            "bbox_token_offset": 8242,
            "se_tokens_per_element": 4,
            "bbox_tokens_per_element": 6,
            "special_tokens": {
                "START_TOKEN": 10290,
                "SEP_TOKEN": 10291,
                "END_TOKEN": 10292,
                "PAD_TOKEN": 10293,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0 = root / "abc_0000.pkl"
            shard1 = root / "abc_0001.pkl"
            out = root / "merged.pkl"
            with shard0.open("wb") as f:
                import pickle

                pickle.dump({**common, "train": [{"id": "tr0"}], "val": [{"id": "va0"}], "test": []}, f)
            with shard1.open("wb") as f:
                import pickle

                pickle.dump({**common, "train": [{"id": "tr1"}], "val": [], "test": [{"id": "te1"}]}, f)

            summary = merge_sequence_shards([shard0, shard1], out)
            with out.open("rb") as f:
                import pickle

                merged = pickle.load(f)

        self.assertEqual([item["id"] for item in merged["train"]], ["tr0", "tr1"])
        self.assertEqual([item["id"] for item in merged["val"]], ["va0"])
        self.assertEqual([item["id"] for item in merged["test"]], ["te1"])
        self.assertEqual(merged["vocab_size"], 10294)
        self.assertEqual(summary["sequences"], 4)
        self.assertEqual(summary["train"], 2)
        self.assertEqual(summary["val"], 1)
        self.assertEqual(summary["test"], 1)
        self.assertEqual(summarize_sequence_package(merged)["out_of_vocab"], 0)

    def test_sequence_shards_reject_inconsistent_metadata(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from sequence_sharding import merge_sequence_shards

        base = {
            "train": [],
            "val": [],
            "test": [],
            "vocab_size": 10294,
            "special_token_size": 4,
            "face_index_size": 50,
            "se_codebook_size": 8192,
            "bbox_index_size": 2048,
            "face_index_offset": 0,
            "se_token_offset": 50,
            "bbox_token_offset": 8242,
            "se_tokens_per_element": 4,
            "bbox_tokens_per_element": 6,
            "special_tokens": {
                "START_TOKEN": 10290,
                "SEP_TOKEN": 10291,
                "END_TOKEN": 10292,
                "PAD_TOKEN": 10293,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0 = root / "abc_0000.pkl"
            shard1 = root / "abc_0001.pkl"
            import pickle

            shard0.write_bytes(pickle.dumps(base))
            shard1.write_bytes(pickle.dumps({**base, "vocab_size": 999}))

            with self.assertRaises(ValueError):
                merge_sequence_shards([shard0, shard1], root / "merged.pkl")

    def test_breparg_sequence_groups_preserve_source_path_metadata(self):
        import importlib.util

        sys.path.insert(0, str(REPO_ROOT / "BrepARG"))
        spec = importlib.util.spec_from_file_location("breparg_2sequence_test", REPO_ROOT / "BrepARG" / "2sequence.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        group = {
            "original": {"input_ids": [1, 2], "attention_mask": [1, 1]},
            "augmented": [{"input_ids": [3, 4], "attention_mask": [1, 1]}],
        }

        annotated = module.attach_sequence_source_path(group, Path("abc_0001") / "shape.pkl")

        self.assertIs(annotated, group)
        self.assertEqual(annotated["source_path"], str(Path("abc_0001") / "shape.pkl"))
        self.assertEqual(annotated["original"]["source_path"], str(Path("abc_0001") / "shape.pkl"))
        self.assertEqual(annotated["augmented"][0]["source_path"], str(Path("abc_0001") / "shape.pkl"))

    def test_sequence_source_path_audit_reports_curved_readiness(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from audit_sequence_source_paths import summarize_sequence_source_paths

        package = {
            "train": [
                {"original": {"input_ids": [1, 2, 3]}},
            ],
            "val": [
                {
                    "source_path": "curved.pkl",
                    "original": {"input_ids": [4, 5], "source_path": "curved.pkl"},
                    "augmented": [
                        {"input_ids": [6], "source_path": "curved.pkl"},
                        {"input_ids": [7]},
                    ],
                }
            ],
            "test": [
                {"source_path": "heldout.pkl", "original": {"input_ids": [8], "source_path": "heldout.pkl"}},
            ],
        }

        summary = summarize_sequence_source_paths(package)

        self.assertEqual(summary["total_groups_scanned"], 3)
        self.assertEqual(summary["groups_with_source_path"], 2)
        self.assertFalse(summary["all_splits_source_path_ready"])
        self.assertTrue(summary["validation_most_curved_ready"])
        self.assertEqual(summary["splits"]["train"]["source_path_coverage"], 0.0)
        self.assertEqual(summary["splits"]["val"]["source_path_coverage"], 1.0)
        self.assertEqual(summary["splits"]["val"]["augmented_records"], 2)
        self.assertEqual(summary["splits"]["val"]["augmented_with_source_path"], 1)
        self.assertEqual(summary["recommendation"], "rebuild_or_refresh_missing_source_paths")

    def test_sequence_grouping_uses_chunk_id_from_paths(self):
        sys.path.insert(0, r"D:\luolin\V13\breparg_improvements")
        from sequence_sharding import group_split_paths_by_chunk

        split = {
            "train": [r"E:\ABC\processed\abc_parsed_full\abc_0002\a.pkl"],
            "val": [r"E:\ABC\processed\abc_parsed_full\abc_0002\b.pkl"],
            "test": [r"E:\ABC\processed\abc_parsed_full\abc_0003\c.pkl"],
        }

        grouped = group_split_paths_by_chunk(split)

        self.assertEqual(sorted(grouped), ["abc_0002", "abc_0003"])
        self.assertEqual(grouped["abc_0002"]["train"], [split["train"][0]])
        self.assertEqual(grouped["abc_0002"]["val"], [split["val"][0]])
        self.assertEqual(grouped["abc_0003"]["test"], [split["test"][0]])

    def test_ar_checkpoint_paths_and_periodic_names(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        from ar_training_utils import ar_checkpoint_paths, periodic_checkpoint_path

        paths = ar_checkpoint_paths(REPO_ROOT / "local_runs" / "ar_training" / "train_outputs" / "run_a")

        self.assertEqual(paths["best"].name, "ar_best.pt")
        self.assertEqual(paths["latest"].name, "ar_latest.pt")
        self.assertEqual(paths["history"].name, "ar_history.jsonl")
        self.assertEqual(periodic_checkpoint_path(paths["checkpoint_dir"], 20).name, "ar_epoch_0020.pt")
        self.assertEqual(periodic_checkpoint_path(paths["checkpoint_dir"], 120).name, "ar_epoch_0120.pt")

    def test_ar_checkpoint_helpers_roundtrip_payload_and_history(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import torch
        from ar_training_utils import append_jsonl, load_ar_checkpoint, save_ar_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "nested" / "ar_latest.pt"
            history = root / "history" / "ar_history.jsonl"
            payload = {
                "epoch": 3,
                "best_val_ce": 0.42,
                "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
                "optimizer_state_dict": {"state": {}},
                "scaler_state_dict": None,
            }

            save_ar_checkpoint(checkpoint, payload)
            loaded = load_ar_checkpoint(checkpoint, map_location="cpu")
            append_jsonl(history, {"epoch": 1, "val_ce": 0.8})
            append_jsonl(history, {"epoch": 2, "val_ce": 0.7})

            rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded["epoch"], 3)
            self.assertTrue(torch.equal(loaded["model_state_dict"]["weight"], payload["model_state_dict"]["weight"]))
            self.assertEqual([row["epoch"] for row in rows], [1, 2])

    def test_ar_tiny_training_saves_latest_best_periodic_and_history(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import torch
        import train
        from ar_training_utils import load_ar_checkpoint, periodic_checkpoint_path

        original_device = train.DEVICE
        original_amp = train.AMP
        original_log_every = train.AR_LOG_EVERY_BATCHES
        try:
            train.DEVICE = "cpu"
            train.AMP = False
            train.AR_LOG_EVERY_BATCHES = 0
            data = {"vocab_size": 12, "special_tokens": {"PAD_TOKEN": 11}}
            train_sequences = [[0, 1, 2, 3], [0, 2, 3], [1, 2, 3], [2, 3]]
            val_sequences = [[0, 1, 3], [1, 3]]

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                best = root / "ar_best.pt"
                latest = root / "ar_latest.pt"
                checkpoints = root / "ar_checkpoints"
                history = root / "ar_history.jsonl"

                meta = train._train_ar(
                    data,
                    train_sequences,
                    val_sequences,
                    dmodel=32,
                    layers=1,
                    lr=1e-3,
                    epochs=2,
                    bs=2,
                    tag="tiny_ar_test",
                    save_path=best,
                    latest_path=latest,
                    checkpoint_dir=checkpoints,
                    history_path=history,
                    save_every=1,
                    max_seq_len=64,
                )

                latest_payload = load_ar_checkpoint(latest, map_location="cpu")
                rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]

                self.assertTrue(best.exists())
                self.assertTrue(latest.exists())
                self.assertTrue(periodic_checkpoint_path(checkpoints, 1).exists())
                self.assertTrue(periodic_checkpoint_path(checkpoints, 2).exists())
                self.assertEqual(meta["end_epoch"], 2)
                self.assertEqual(latest_payload["epoch"], 2)
                self.assertEqual(latest_payload["batch_size"], 2)
                self.assertEqual(latest_payload["d_model"], 32)
                self.assertEqual(latest_payload["config"]["max_seq_len"], 64)
                self.assertIn("model_state_dict", latest_payload)
                self.assertIn("optimizer_state_dict", latest_payload)
                self.assertEqual([row["epoch"] for row in rows], [1, 2])
                self.assertTrue(torch.isfinite(torch.tensor(rows[-1]["val_ce"])))
        finally:
            train.DEVICE = original_device
            train.AMP = original_amp
            train.AR_LOG_EVERY_BATCHES = original_log_every

    def test_ar_resume_applies_requested_learning_rate_to_optimizer(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import train
        from ar_training_utils import load_ar_checkpoint

        original_device = train.DEVICE
        original_amp = train.AMP
        original_log_every = train.AR_LOG_EVERY_BATCHES
        try:
            train.DEVICE = "cpu"
            train.AMP = False
            train.AR_LOG_EVERY_BATCHES = 0
            data = {"vocab_size": 12, "special_tokens": {"PAD_TOKEN": 11}}
            train_sequences = [[0, 1, 2, 3], [0, 2, 3]]
            val_sequences = [[0, 1, 3]]

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first_latest = root / "first" / "ar_latest.pt"
                resumed_latest = root / "resumed" / "ar_latest.pt"

                train._train_ar(
                    data,
                    train_sequences,
                    val_sequences,
                    dmodel=32,
                    layers=1,
                    lr=1e-3,
                    epochs=1,
                    bs=2,
                    tag="tiny_ar_lr_source",
                    latest_path=first_latest,
                )

                train._train_ar(
                    data,
                    train_sequences,
                    val_sequences,
                    dmodel=32,
                    layers=1,
                    lr=2e-4,
                    epochs=2,
                    bs=2,
                    tag="tiny_ar_lr_resume",
                    latest_path=resumed_latest,
                    resume_from=first_latest,
                )

                payload = load_ar_checkpoint(resumed_latest, map_location="cpu")

            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["learning_rate"], 2e-4)
            self.assertEqual(payload["optimizer_state_dict"]["param_groups"][0]["lr"], 2e-4)
        finally:
            train.DEVICE = original_device
            train.AMP = original_amp
            train.AR_LOG_EVERY_BATCHES = original_log_every

    def test_ar_sequence_summary_counts_valid_training_tokens(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        from ar_training_utils import summarize_ar_sequences, validate_ar_sequence_package

        package = {
            "vocab_size": 10,
            "special_tokens": {"PAD_TOKEN": 9},
            "train": [
                {"original": {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}},
                {"original": {"input_ids": [1] * 1025, "attention_mask": [1] * 1025}},
            ],
            "val": [{"original": {"input_ids": [0, 8], "attention_mask": [1, 1]}}],
            "test": [{"original": {"input_ids": [3, 4], "attention_mask": [1, 1]}}],
        }

        summary = summarize_ar_sequences(package, max_seq_len=1024)

        self.assertEqual(summary["raw_train"], 2)
        self.assertEqual(summary["usable_train"], 1)
        self.assertEqual(summary["raw_val"], 1)
        self.assertEqual(summary["usable_val"], 1)
        self.assertEqual(summary["out_of_vocab"], 0)
        self.assertEqual(summary["max_token"], 8)
        self.assertEqual(validate_ar_sequence_package(package)["status"], "VERIFIED")

    def test_ar_sequence_validation_rejects_out_of_vocab(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        from ar_training_utils import validate_ar_sequence_package

        package = {
            "vocab_size": 10,
            "special_tokens": {"PAD_TOKEN": 9},
            "train": [{"original": {"input_ids": [1, 10], "attention_mask": [1, 1]}}],
            "val": [{"original": {"input_ids": [2, 3], "attention_mask": [1, 1]}}],
            "test": [],
        }

        result = validate_ar_sequence_package(package)

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["out_of_vocab"], 1)

    def test_ar_length_coverage_quantifies_complex_sequence_gains(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_ar_length_coverage import summarize_length_coverage

        vocab = {
            "face_index_size": 64,
            "se_codebook_size": 100,
            "bbox_index_size": 50,
            "face_index_offset": 0,
            "se_token_offset": 64,
            "bbox_token_offset": 164,
            "special_tokens": {
                "START_TOKEN": 214,
                "SEP_TOKEN": 215,
                "END_TOKEN": 216,
                "PAD_TOKEN": 217,
            },
            "vocab_size": 218,
        }

        def sequence(n_faces, n_edges):
            bbox = vocab["bbox_token_offset"]
            geo = vocab["se_token_offset"]
            seq = [vocab["special_tokens"]["START_TOKEN"]]
            for face in range(n_faces):
                seq.extend([bbox] * 6 + [geo] * 4 + [face])
            seq.append(vocab["special_tokens"]["SEP_TOKEN"])
            for edge in range(n_edges):
                left = edge % n_faces
                right = (edge + 1) % n_faces
                seq.extend([left, right] + [bbox] * 6 + [geo] * 4)
            seq.append(vocab["special_tokens"]["END_TOKEN"])
            return seq

        package = {
            **vocab,
            "train": [
                {"original": {"input_ids": sequence(2, 1)}},
                {"original": {"input_ids": sequence(12, 20)}},
                {"original": {"input_ids": sequence(50, 80)}},
                {"original": {"input_ids": sequence(55, 115)}},
                {"original": {"input_ids": sequence(60, 130)}},
                {"original": {"input_ids": []}},
            ],
            "val": [{"original": {"input_ids": sequence(2, 1)}}],
            "test": [{"original": {"input_ids": [214, 215, 216]}}],
        }

        summary = summarize_length_coverage(
            package,
            limits=[1024, 1536, 2048],
            complex_min_faces=12,
            complex_min_edges=20,
        )

        train = summary["splits"]["train"]
        self.assertEqual(train["total_groups"], 6)
        self.assertEqual(train["empty_sequences"], 1)
        self.assertEqual(train["complex_total"], 4)
        self.assertEqual(train["by_limit"]["1024"]["complex_allowed"], 1)
        self.assertEqual(train["by_limit"]["1536"]["complex_allowed"], 2)
        self.assertEqual(train["by_limit"]["2048"]["complex_allowed"], 3)
        self.assertEqual(summary["overall"]["by_limit"]["1024"]["complex_excluded"], 3)
        self.assertEqual(summary["overall"]["by_limit"]["2048"]["complex_excluded"], 1)
        self.assertEqual(summary["splits"]["test"]["grammar_failed"], 1)
        self.assertEqual(summary["recommendation"]["action"], "train_long_context_ar")

    def test_ar_v13_launcher_keeps_inputs_and_outputs_under_v13(self):
        script = REPO_ROOT / "tools" / "run_ar_v13_epoch100.ps1"
        text = script.read_text(encoding="utf-8")

        self.assertIn('[string]$OutBase = "D:\\luolin\\V13\\local_runs\\ar_training\\train_outputs"', text)
        self.assertIn('[string]$RunName = "newscheme_full_v13_ar"', text)
        self.assertIn('$env:NS_OUTBASE = $OutBase', text)
        self.assertIn('$env:NS_OUT = $RunName', text)
        self.assertIn('$env:NS_AR_SAVE_EVERY = "20"', text)
        self.assertIn('$env:NS_AR_BS = "8"', text)
        self.assertIn('$env:NS_AR_LOG_EVERY_BATCHES = "2000"', text)
        self.assertIn("param(", text)
        self.assertIn("$NoAutoResume", text)
        self.assertIn("ar_latest.pt", text)
        self.assertIn("$env:NS_AR_RESUME_FROM = $ResolvedResumeFrom", text)
        self.assertIn("$RunningAr = Get-CimInstance Win32_Process", text)
        self.assertIn("Another AR training process is already running", text)

    def test_ar_v13_launcher_supports_parametric_new_run_branches(self):
        script = REPO_ROOT / "tools" / "run_ar_v13_epoch100.ps1"
        text = script.read_text(encoding="utf-8")

        self.assertIn("[string]$RunName = \"newscheme_full_v13_ar\"", text)
        self.assertIn("[string]$OutBase = \"D:\\luolin\\V13\\local_runs\\ar_training\\train_outputs\"", text)
        self.assertIn("[string]$SequenceSource = \"\"", text)
        self.assertIn("[string]$SplitSource = \"\"", text)
        self.assertIn('$env:NS_OUTBASE = $OutBase', text)
        self.assertIn('$env:NS_OUT = $RunName', text)
        self.assertIn("Copied sequence input:", text)
        self.assertIn("Copied split input:", text)
        self.assertIn("Seeded new run from checkpoint:", text)
        self.assertIn("$LogPath = Join-Path $LogDir (\"ar_{0}_{1}.log\" -f $RunName", text)

    def test_ar_archive_status_tool_is_read_only_and_handles_utf16_logs(self):
        script = REPO_ROOT / "tools" / "check_ar_archive_status.py"
        text = script.read_text(encoding="utf-8")

        self.assertIn("Read-only status summary", text)
        self.assertIn(r"D:\luolin\V13\ABC\processed\abc_parsed_full_archives", text)
        self.assertIn(r"D:\luolin\V13\ABC\processed\logs", text)
        self.assertIn("raw.decode(\"utf-16-le\"", text)
        self.assertIn("nvidia-smi", text)
        self.assertIn("Get-PSDrive", text)
        self.assertIn("Get-CimInstance Win32_Process", text)
        self.assertNotIn("Remove-Item", text)
        self.assertNotIn("shutil.rmtree", text)

    def test_ar_archive_status_uses_configurable_ar_log_pattern(self):
        script = REPO_ROOT / "tools" / "check_ar_archive_status.py"
        text = script.read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--ar-log-pattern", default="ar_*.log")', text)
        self.assertIn('latest_file(args.ar_log_dir, getattr(args, "ar_log_pattern", "ar_*.log"))', text)

    def test_ar_status_tool_parses_progress_from_log_lines(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from check_ar_archive_status import parse_ar_progress

        progress = parse_ar_progress([
            "[14:54:39]   [ar] ep   1 train_CE=1.2521 val_CE=0.8038 best=0.8038",
            "[15:13:20]   [ar] ep   2 batch 12000/36289 train_CE_running=0.7133 elapsed_min=75.97",
            "[15:19:38]   [ar] ep   2 batch 16000/36289 train_CE_running=0.7026 elapsed_min=82.26",
        ])

        self.assertEqual(progress["epoch"], 2)
        self.assertEqual(progress["batch"], 16000)
        self.assertEqual(progress["total_batches"], 36289)
        self.assertEqual(progress["train_ce_running"], 0.7026)
        self.assertAlmostEqual(progress["epoch_progress_percent"], 44.09, places=2)
        self.assertAlmostEqual(progress["estimated_epoch_remaining_min"], 31.9, places=2)

    def test_ar_status_json_is_safe_for_gbk_console(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from check_ar_archive_status import status_json

        rendered = status_json({"log_tail": ["bad private char \ue160"]})

        rendered.encode("gbk")
        self.assertIn("\\ue160", rendered)

    def test_ar_status_drive_query_keeps_available_drives_when_one_is_missing(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import check_ar_archive_status

        class Result:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        calls = []

        def fake_run(cmd, capture_output, text):
            calls.append(cmd)
            script = cmd[-1]
            if "'D'" in script:
                return Result(0, '{"Name":"D","Used":10,"Free":20}')
            return Result(1, "", "Cannot find drive E")

        original_run = check_ar_archive_status.subprocess.run
        try:
            check_ar_archive_status.subprocess.run = fake_run
            drives = check_ar_archive_status.ps_drive_free(["D", "E"])
        finally:
            check_ar_archive_status.subprocess.run = original_run

        self.assertEqual(drives[0], {"Name": "D", "Used": 10, "Free": 20})
        self.assertEqual(drives[1]["Name"], "E")
        self.assertIn("Cannot find drive E", drives[1]["error"])
        self.assertEqual(len(calls), 2)

    def test_ar_status_default_drive_list_is_d_only_after_e_ejection(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import check_ar_archive_status

        class Args:
            ar_out = str(REPO_ROOT / "missing_ar_out")
            ar_log_dir = str(REPO_ROOT / "missing_logs")
            archive_root = str(REPO_ROOT / "missing_archives")
            archive_log_dir = str(REPO_ROOT / "missing_archive_logs")
            tail = 1
            drives = "D"

        seen = {}
        original_ps_drive_free = check_ar_archive_status.ps_drive_free
        original_nvidia_smi = check_ar_archive_status.nvidia_smi
        original_ps_processes = check_ar_archive_status.ps_processes
        try:
            def fake_ps_drive_free(names):
                seen["names"] = names
                return [{"Name": name, "Free": 1, "Used": 2} for name in names]

            check_ar_archive_status.ps_drive_free = fake_ps_drive_free
            check_ar_archive_status.nvidia_smi = lambda: {"available": False}
            check_ar_archive_status.ps_processes = lambda pattern: []
            status = check_ar_archive_status.build_status(Args())
        finally:
            check_ar_archive_status.ps_drive_free = original_ps_drive_free
            check_ar_archive_status.nvidia_smi = original_nvidia_smi
            check_ar_archive_status.ps_processes = original_ps_processes

        self.assertEqual(seen["names"], ["D"])
        self.assertEqual(status["drives"], [{"Name": "D", "Free": 1, "Used": 2}])

    def test_ar_batches_bucket_by_length_to_reduce_padding(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import train

        seqs = [[1] * 10, [2] * 2, [3] * 9, [4] * 3]
        batches = list(train._ar_batches(seqs, bs=2, pad=0, shuf=False, device="cpu"))

        self.assertEqual([tuple(ids.shape) for ids, _ in batches], [(2, 10), (2, 3)])
        self.assertEqual(batches[0][0][0, -1].item(), 1)
        self.assertEqual(batches[1][0][1, -1].item(), 0)

    def test_vqvae_report_metrics_preserve_e6_scale_losses(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import train

        self.assertEqual(train.metric_for_report(1e-6), 0.000001)
        self.assertEqual(train.metric_for_report(3.7e-4), 0.00037)

    def test_ar_epoch_monitor_waits_until_target_epoch_checkpoint_exists(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from monitor_ar_epoch_gate import evaluate_epoch_gate

        rows = [
            {"epoch": 38, "val_ce": 0.33, "best_val_ce": 0.33},
            {"epoch": 39, "val_ce": 0.331, "best_val_ce": 0.33},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            status = evaluate_epoch_gate(Path(tmp), rows, target_epoch=40)

        self.assertFalse(status["ready"])
        self.assertEqual(status["latest_epoch"], 39)
        self.assertEqual(status["target_epoch"], 40)
        self.assertIn("waiting_for_epoch", status["reason"])

    def test_ar_epoch_monitor_reports_resume_checkpoint_epoch_without_history(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import torch
        from monitor_ar_epoch_gate import evaluate_epoch_gate

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            torch.save(
                {
                    "epoch": 95,
                    "train_ce": 0.3012,
                    "val_ce": 0.2951,
                    "best_val_ce": 0.2951,
                    "model_state_dict": {"w": torch.tensor([1.0])},
                    "optimizer_state_dict": {"state": {}},
                    "scaler_state_dict": None,
                },
                out_dir / "ar_latest.pt",
            )

            status = evaluate_epoch_gate(out_dir, [], target_epoch=120)

        self.assertFalse(status["ready"])
        self.assertEqual(status["latest_epoch"], 95)
        self.assertEqual(status["checkpoint"]["epoch"], 95)
        self.assertEqual(status["checkpoint"]["path"].split("\\")[-1], "ar_latest.pt")
        self.assertIn("waiting_for_epoch_120", status["reason"])

    def test_ar_epoch_monitor_summarizes_target_checkpoint(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import torch
        from monitor_ar_epoch_gate import evaluate_epoch_gate

        rows = [{"epoch": 40, "val_ce": 0.32, "best_val_ce": 0.32}]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            checkpoint_dir = out_dir / "ar_checkpoints"
            checkpoint_dir.mkdir()
            torch.save(
                {
                    "epoch": 40,
                    "train_ce": 0.34,
                    "val_ce": 0.32,
                    "best_val_ce": 0.32,
                    "model_state_dict": {"w": torch.tensor([1.0])},
                    "optimizer_state_dict": {"state": {}},
                    "scaler_state_dict": {"scale": 1.0},
                },
                checkpoint_dir / "ar_epoch_0040.pt",
            )

            status = evaluate_epoch_gate(out_dir, rows, target_epoch=40)

        self.assertTrue(status["ready"])
        self.assertEqual(status["checkpoint"]["epoch"], 40)
        self.assertEqual(status["checkpoint"]["val_ce"], 0.32)
        self.assertTrue(status["checkpoint"]["has_model"])
        self.assertTrue(status["checkpoint"]["has_optimizer"])
        self.assertTrue(status["checkpoint"]["has_scaler"])

    def test_vqvae_recovery_monitor_waits_for_target_epoch(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from monitor_vqvae_recovery_gate import evaluate_vqvae_recovery_gate

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "vq_complex"
            run_dir.mkdir()
            (run_dir / "vqvae_history.json").write_text(
                json.dumps(
                    {
                        "config": {"target_epoch": 180},
                        "history": [
                            {
                                "epoch": 142,
                                "train_loss": 0.00041,
                                "val_loss": 0.00057,
                                "best_val": 0.00055,
                                "best_epoch": 140,
                                "finite_train_batches": 9,
                                "train_batches": 9,
                                "finite_val_batches": 3,
                                "val_batches": 3,
                            }
                        ],
                        "best_val_recon": 0.00055,
                        "best_epoch": 140,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "fsq_vqvae_best.pt").write_bytes(b"checkpoint")

            status = evaluate_vqvae_recovery_gate(run_dir)

        self.assertFalse(status["ready"])
        self.assertFalse(status["terminal"])
        self.assertEqual(status["exit_code"], 1)
        self.assertEqual(status["state"], "waiting_for_epoch_180")
        self.assertEqual(status["history"]["latest_epoch"], 142)
        self.assertEqual(status["history"]["target_epoch"], 180)
        self.assertEqual(status["checkpoints"]["best"]["exists"], True)

    def test_vqvae_recovery_monitor_promotes_after_benchmark_and_copyback(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from monitor_vqvae_recovery_gate import evaluate_vqvae_recovery_gate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "vq_complex"
            run_dir.mkdir()
            (run_dir / "vqvae_history.json").write_text(
                json.dumps(
                    {
                        "config": {"target_epoch": 180},
                        "history": [
                            {"epoch": 180, "train_loss": 0.00039, "val_loss": 0.00051, "best_val": 0.00051, "best_epoch": 180}
                        ],
                        "best_val_recon": 0.00051,
                        "best_epoch": 180,
                        "stop_reason": "",
                    }
                ),
                encoding="utf-8",
            )
            for name in ["fsq_vqvae_best.pt", "fsq_vqvae_final.pt"]:
                (run_dir / name).write_bytes(b"checkpoint")
            benchmark = root / "vq_complex_benchmark_summary.json"
            benchmark.write_text(
                json.dumps({"promotion_gate": {"decision": "promote_for_ar_rebuild", "reasons": []}}),
                encoding="utf-8",
            )
            copyback = run_dir / "copy_back_manifest.json"
            copyback.write_text(json.dumps({"complete": True, "missing_required": []}), encoding="utf-8")

            status = evaluate_vqvae_recovery_gate(
                run_dir,
                benchmark_summary=benchmark,
                copy_back_manifest=copyback,
            )

        self.assertTrue(status["ready"])
        self.assertTrue(status["terminal"])
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["state"], "ready_for_sequence_rebuild")
        self.assertEqual(status["benchmark"]["promotion_decision"], "promote_for_ar_rebuild")
        self.assertEqual(status["copy_back_manifest"]["complete"], True)

    def test_vqvae_recovery_monitor_blocks_hold_decision(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from monitor_vqvae_recovery_gate import evaluate_vqvae_recovery_gate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "vq_complex"
            run_dir.mkdir()
            (run_dir / "vqvae_history.json").write_text(
                json.dumps(
                    {
                        "config": {"target_epoch": 180},
                        "history": [{"epoch": 180, "val_loss": 0.00060, "best_val": 0.00056, "best_epoch": 150}],
                        "best_val_recon": 0.00056,
                        "best_epoch": 150,
                        "stop_reason": "patience=14",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "fsq_vqvae_best.pt").write_bytes(b"checkpoint")
            benchmark = root / "vq_complex_benchmark_summary.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "promotion_gate": {
                            "decision": "hold_vqvae_checkpoint",
                            "reasons": ["longest strict-valid count did not improve over baseline"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            status = evaluate_vqvae_recovery_gate(run_dir, benchmark_summary=benchmark)

        self.assertFalse(status["ready"])
        self.assertTrue(status["terminal"])
        self.assertEqual(status["exit_code"], 2)
        self.assertEqual(status["state"], "hold_vqvae_checkpoint")
        self.assertIn("longest strict-valid count did not improve", status["reason"])

    def test_ar_history_analysis_recommends_continue_after_recent_best(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from analyze_ar_training import analyze_history

        rows = [
            {"epoch": 1, "train_ce": 1.2, "val_ce": 0.8, "best_val_ce": 0.8, "improved": True, "elapsed_min": 60},
            {"epoch": 2, "train_ce": 0.7, "val_ce": 0.6, "best_val_ce": 0.6, "improved": True, "elapsed_min": 120},
            {"epoch": 3, "train_ce": 0.5, "val_ce": 0.55, "best_val_ce": 0.55, "improved": True, "elapsed_min": 180},
            {"epoch": 4, "train_ce": 0.45, "val_ce": 0.5, "best_val_ce": 0.5, "improved": True, "elapsed_min": 240},
        ]

        summary = analyze_history(rows, target_epoch=10, recent_window=3, plateau_patience=3)

        self.assertEqual(summary["latest_epoch"], 4)
        self.assertEqual(summary["best_epoch"], 4)
        self.assertEqual(summary["epochs_since_best"], 0)
        self.assertEqual(summary["recommendation"], "continue_unchanged")
        self.assertFalse(summary["overfit_signal"])
        self.assertAlmostEqual(summary["eta_to_target_hours"], 6 * 60 / 60)

    def test_ar_history_analysis_flags_overfit_or_plateau(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from analyze_ar_training import analyze_history

        rows = [
            {"epoch": 1, "train_ce": 0.8, "val_ce": 0.5, "best_val_ce": 0.5, "improved": True, "elapsed_min": 60},
            {"epoch": 2, "train_ce": 0.7, "val_ce": 0.45, "best_val_ce": 0.45, "improved": True, "elapsed_min": 120},
            {"epoch": 3, "train_ce": 0.6, "val_ce": 0.4, "best_val_ce": 0.4, "improved": True, "elapsed_min": 180},
            {"epoch": 4, "train_ce": 0.55, "val_ce": 0.43, "best_val_ce": 0.4, "improved": False, "elapsed_min": 240},
            {"epoch": 5, "train_ce": 0.52, "val_ce": 0.44, "best_val_ce": 0.4, "improved": False, "elapsed_min": 300},
            {"epoch": 6, "train_ce": 0.50, "val_ce": 0.46, "best_val_ce": 0.4, "improved": False, "elapsed_min": 360},
        ]

        summary = analyze_history(rows, target_epoch=10, recent_window=3, plateau_patience=3, min_delta=0.01)

        self.assertEqual(summary["best_epoch"], 3)
        self.assertEqual(summary["epochs_since_best"], 3)
        self.assertTrue(summary["overfit_signal"])
        self.assertEqual(summary["recommendation"], "consider_stop_or_lower_lr")
        self.assertIn("validation has not improved", summary["reason"])

    def test_ar_history_analysis_uses_baseline_best_for_continuation_branch(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from analyze_ar_training import analyze_history

        rows = [
            {"epoch": 58, "train_ce": 0.30, "val_ce": 0.2981, "best_val_ce": 0.2977, "improved": False, "elapsed_min": 60},
            {"epoch": 59, "train_ce": 0.299, "val_ce": 0.2982, "best_val_ce": 0.2977, "improved": False, "elapsed_min": 120},
        ]

        summary = analyze_history(
            rows,
            target_epoch=100,
            recent_window=2,
            plateau_patience=2,
            baseline_best_epoch=57,
            baseline_best_val_ce=0.2977,
        )

        self.assertEqual(summary["best_epoch"], 57)
        self.assertAlmostEqual(summary["best_val_ce"], 0.2977)
        self.assertEqual(summary["epochs_since_best"], 2)
        self.assertEqual(summary["recommendation"], "consider_stop_or_lower_lr")

    def test_vqvae_history_analysis_recommends_continue_toward_e6(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from analyze_vqvae_history import analyze_history

        rows = [
            {"epoch": 86, "train_loss": 0.00044, "val_loss": 0.000069, "best_val": 0.000069},
            {"epoch": 87, "train_loss": 0.00035, "val_loss": 0.000068, "best_val": 0.000068},
            {"epoch": 88, "train_loss": 0.00032, "val_loss": 0.000066, "best_val": 0.000066},
        ]

        summary = analyze_history(rows, target_loss=1e-6, target_epoch=260, plateau_patience=4, min_delta=1e-8)

        self.assertEqual(summary["status"], "IN_PROGRESS")
        self.assertEqual(summary["best_epoch"], 88)
        self.assertFalse(summary["target_reached"])
        self.assertFalse(summary["overfit_signal"])
        self.assertEqual(summary["recommendation"], "continue_training")

    def test_vqvae_history_analysis_flags_overfit_or_plateau_above_e6(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from analyze_vqvae_history import analyze_history

        rows = [
            {"epoch": 1, "train_loss": 0.00020, "val_loss": 0.000030, "best_val": 0.000030},
            {"epoch": 2, "train_loss": 0.00018, "val_loss": 0.000028, "best_val": 0.000028},
            {"epoch": 3, "train_loss": 0.00016, "val_loss": 0.000031, "best_val": 0.000028},
            {"epoch": 4, "train_loss": 0.00014, "val_loss": 0.000034, "best_val": 0.000028},
            {"epoch": 5, "train_loss": 0.00012, "val_loss": 0.000038, "best_val": 0.000028},
        ]

        summary = analyze_history(rows, target_loss=1e-6, target_epoch=20, plateau_patience=3, min_delta=1e-8)

        self.assertEqual(summary["status"], "HOLD_ABOVE_TARGET")
        self.assertEqual(summary["best_epoch"], 2)
        self.assertFalse(summary["target_reached"])
        self.assertTrue(summary["overfit_signal"])
        self.assertEqual(summary["recommendation"], "stop_and_diagnose_before_sequence_or_ar")
        self.assertIn("validation has not improved", summary["reason"])

    def test_reconstruction_tool_normalizes_sequence_metadata_for_breparg_utils(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import normalize_vocab_info

        package = {
            "face_index_size": 50,
            "se_codebook_size": 8192,
            "bbox_index_size": 2048,
            "vocab_size": 10294,
            "se_tokens_per_element": 4,
            "bbox_tokens_per_element": 6,
            "special_tokens": {
                "START_TOKEN": 10290,
                "SEP_TOKEN": 10291,
                "END_TOKEN": 10292,
                "PAD_TOKEN": 10293,
            },
        }

        vocab = normalize_vocab_info(package)

        self.assertEqual(vocab["face_index_offset"], 0)
        self.assertEqual(vocab["se_token_offset"], 50)
        self.assertEqual(vocab["bbox_token_offset"], 8242)
        self.assertEqual(vocab["START_TOKEN"], 10290)
        self.assertEqual(vocab["SEP_TOKEN"], 10291)
        self.assertEqual(vocab["END_TOKEN"], 10292)
        self.assertEqual(vocab["PAD_TOKEN"], 10293)
        self.assertEqual(vocab["se_tokens_per_element"], 4)

    def test_reconstruction_tool_selects_shortest_source_records(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import select_sequence_records

        package = {
            "val": [
                {"original": {"input_ids": [1] * 9, "attention_mask": [1] * 9}},
                {"original": {"input_ids": [2] * 3, "attention_mask": [1] * 3}},
                {"original": {"input_ids": [3] * 5, "attention_mask": [1] * 5}},
            ]
        }

        records = select_sequence_records(package, split="validation", max_samples=2, order="shortest")

        self.assertEqual([record["index"] for record in records], [1, 2])
        self.assertEqual([record["length"] for record in records], [3, 5])
        self.assertEqual(records[0]["source"], "validation")

    def test_reconstruction_tool_selects_longest_source_records(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import select_sequence_records

        package = {
            "val": [
                {"original": {"input_ids": [1] * 9, "attention_mask": [1] * 9}},
                {"original": {"input_ids": [2] * 3, "attention_mask": [1] * 3}},
                {"original": {"input_ids": [3] * 5, "attention_mask": [1] * 5}},
                {"original": {"input_ids": [4] * 11, "attention_mask": [1] * 11}},
            ]
        }

        records = select_sequence_records(package, split="validation", max_samples=2, order="longest")

        self.assertEqual([record["index"] for record in records], [3, 0])
        self.assertEqual([record["length"] for record in records], [11, 9])

    def test_reconstruction_tool_selects_most_faces_source_records(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import select_sequence_records

        bbox = 8242
        geo = 50
        start = 10290
        sep = 10291
        end = 10292

        def face(face_idx):
            return [bbox] * 6 + [geo] * 4 + [face_idx]

        def edge(a, b):
            return [a, b] + [bbox] * 6 + [geo] * 4

        one_face = [start] + face(0) + [sep, end]
        three_faces = [start] + face(0) + face(1) + face(2) + [sep] + edge(0, 1) + edge(1, 2) + [end]
        two_faces_longer = [start] + face(0) + face(1) + [sep] + edge(0, 1) + edge(0, 1) + edge(0, 1) + [end]

        package = {
            "face_index_size": 50,
            "se_codebook_size": 8192,
            "bbox_index_size": 2048,
            "special_tokens": {
                "START_TOKEN": start,
                "SEP_TOKEN": sep,
                "END_TOKEN": end,
                "PAD_TOKEN": 10293,
            },
            "val": [
                {"original": {"input_ids": one_face}},
                {"original": {"input_ids": two_faces_longer}},
                {"original": {"input_ids": three_faces}},
            ],
        }

        records = select_sequence_records(package, split="validation", max_samples=2, order="most_faces")

        self.assertEqual([record["index"] for record in records], [2, 1])
        self.assertEqual([record["grammar_faces"] for record in records], [3, 2])
        self.assertEqual([record["grammar_edges"] for record in records], [2, 3])

    def test_reconstruction_tool_selects_most_curved_source_records(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import pickle
        import numpy as np
        from evaluate_reconstruction_v13 import select_sequence_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat_path = root / "flat.pkl"
            curved_path = root / "curved.pkl"
            mildly_curved_path = root / "mild.pkl"

            flat_surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
            curved_surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
            mildly_curved_surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
            grid = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            xx, yy = np.meshgrid(grid, grid, indexing="ij")
            for surface in [flat_surface, curved_surface, mildly_curved_surface]:
                surface[0, :, :, 0] = xx
                surface[0, :, :, 1] = yy
            curved_surface[0, :, :, 2] = xx ** 2
            mildly_curved_surface[0, :, :, 2] = (xx ** 2) * 0.25

            for path, surface in [
                (flat_path, flat_surface),
                (curved_path, curved_surface),
                (mildly_curved_path, mildly_curved_surface),
            ]:
                with path.open("wb") as handle:
                    pickle.dump({"surf_ncs": surface, "edge_ncs": np.zeros((1, 32, 3), dtype=np.float32)}, handle)

            package = {
                "val": [
                    {"original": {"input_ids": [1] * 20}, "source_path": str(flat_path)},
                    {"original": {"input_ids": [2] * 20}, "source_path": str(curved_path)},
                    {"original": {"input_ids": [3] * 20}, "source_path": str(mildly_curved_path)},
                ]
            }

            records = select_sequence_records(package, split="validation", max_samples=2, order="most_curved")

        self.assertEqual([record["index"] for record in records], [1, 2])
        self.assertGreater(records[0]["curvature_score"], records[1]["curvature_score"])

    def test_reconstruction_tool_summarizes_manifest_rows(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import summarize_manifest_rows

        rows = [
            {"status": "saved", "step_saved": True, "stl_saved": True, "brep_valid": True},
            {"status": "reconstruct_failed", "step_saved": False, "stl_saved": False, "brep_valid": False},
            {"status": "saved", "step_saved": True, "stl_saved": False, "brep_valid": False},
        ]

        summary = summarize_manifest_rows(rows)

        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(summary["reconstruct_success"], 2)
        self.assertEqual(summary["step_saved"], 2)
        self.assertEqual(summary["stl_saved"], 1)
        self.assertEqual(summary["brep_valid"], 1)

    def test_reconstruction_manifest_preserves_source_path_and_curvature_metadata(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import reconstruct_one

        row = reconstruct_one(
            {
                "source": "validation",
                "split": "val",
                "index": 3,
                "length": 4,
                "sequence": [10290, 10291, 10292],
                "source_path": "shape.pkl",
                "curvature_score": 0.75,
            },
            vocab_info={
                "face_index_size": 50,
                "se_codebook_size": 8192,
                "bbox_index_size": 2048,
                "bbox_token_offset": 8242,
                "se_token_offset": 50,
                "START_TOKEN": 10290,
                "SEP_TOKEN": 10291,
                "END_TOKEN": 10292,
                "PAD_TOKEN": 10293,
            },
            vqvae_model=None,
            device="cpu",
            output_dir=Path("."),
            write_step=False,
            write_stl=False,
            validate_step=False,
            scale_factor=1.0,
        )

        self.assertEqual(row["source_path"], "shape.pkl")
        self.assertEqual(row["curvature_score"], 0.75)

    def test_vqvae_slice_benchmark_builds_optional_curved_reconstruction_command(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_vqvae_slice_benchmark import build_benchmark_plan

        plan = build_benchmark_plan(
            python="python",
            sequence=Path("seq.pkl"),
            vqvae_checkpoint=Path("vq.pt"),
            output_root=Path("eval_root"),
            run_prefix="candidate_vq",
            max_samples=10,
            max_seq_len=1024,
            device="cpu",
            seed=7,
            orders=["shortest", "random", "longest", "most_faces", "most_curved"],
            render=True,
            cols=5,
        )

        self.assertEqual([item["order"] for item in plan], ["shortest", "random", "longest", "most_faces", "most_curved"])
        self.assertEqual(
            [item["run_name"] for item in plan],
            [
                "candidate_vq_shortest10",
                "candidate_vq_random10_seed7",
                "candidate_vq_longest10",
                "candidate_vq_mostfaces10",
                "candidate_vq_mostcurved10",
            ],
        )

        random_command = plan[1]["evaluate_command"]
        self.assertIn("--seed", random_command)
        self.assertIn("7", random_command)
        for item in plan:
            command = item["evaluate_command"]
            self.assertIn(str(REPO_ROOT / "tools" / "evaluate_reconstruction_v13.py"), command)
            self.assertIn("--source", command)
            self.assertIn("validation", command)
            self.assertIn("--write-step", command)
            self.assertIn("--validate-step", command)
            self.assertIn("--max-seq-len", command)
            self.assertIn("1024", command)
            self.assertIn("--output-root", command)
            self.assertIn("eval_root", command)
            self.assertIsNotNone(item["render_command"])
            self.assertIn(str(REPO_ROOT / "papers" / "aaai_v13" / "render_step_directory.py"), item["render_command"])
            self.assertIn("--cols", item["render_command"])
            self.assertIn("5", item["render_command"])

    def test_vqvae_slice_benchmark_summary_flags_baseline_as_not_promotable(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_vqvae_slice_benchmark import build_benchmark_plan, summarize_benchmark_plan

        def write_report(run_dir, attempted, step_saved, brep_valid, errors):
            run_dir.mkdir(parents=True)
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "attempted": attempted,
                            "grammar_valid": attempted,
                            "reconstruct_success": step_saved,
                            "step_saved": step_saved,
                            "stl_saved": 0,
                            "brep_valid": brep_valid,
                            "errors": errors,
                        }
                    }
                ),
                encoding="utf-8",
            )
            renders = run_dir / "renders"
            renders.mkdir()
            (renders / "contact_sheet.png").write_bytes(b"fake-png")

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            plan = build_benchmark_plan(
                python="python",
                sequence=Path("seq.pkl"),
                vqvae_checkpoint=Path("vq.pt"),
                output_root=output_root,
                run_prefix="baseline_like",
                max_samples=10,
                max_seq_len=1024,
                device="cpu",
                seed=7,
                orders=["shortest", "random", "longest", "most_faces"],
                render=True,
                cols=5,
            )
            strict_counts = {"shortest": 8, "random": 6, "longest": 3, "most_faces": 5}
            saved_counts = {"shortest": 10, "random": 8, "longest": 7, "most_faces": 9}
            errors = {"shortest": 0, "random": 2, "longest": 3, "most_faces": 1}
            for item in plan:
                order = item["order"]
                write_report(Path(item["run_dir"]), 10, saved_counts[order], strict_counts[order], errors[order])

            summary = summarize_benchmark_plan(plan)

        gate = summary["promotion_gate"]
        self.assertFalse(gate["promote"])
        self.assertTrue(gate["requirements"]["reports_complete"])
        self.assertTrue(gate["requirements"]["renders_complete"])
        self.assertFalse(gate["requirements"]["longest_improved"])
        self.assertFalse(gate["requirements"]["most_faces_improved"])
        self.assertIn("longest strict-valid count did not improve over baseline", gate["reasons"])
        self.assertEqual(summary["slices"]["longest"]["brep_valid"], 3)
        self.assertEqual(summary["slices"]["longest"]["delta_brep_valid"], 0)
        self.assertAlmostEqual(summary["slices"]["longest"]["strict_valid_rate"], 0.3)

    def test_vqvae_slice_benchmark_summary_requires_complex_improvement_and_renders(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_vqvae_slice_benchmark import build_benchmark_plan, summarize_benchmark_plan

        def write_report(run_dir, brep_valid, with_render=True):
            run_dir.mkdir(parents=True)
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "attempted": 10,
                            "grammar_valid": 10,
                            "reconstruct_success": 10,
                            "step_saved": 10,
                            "stl_saved": 0,
                            "brep_valid": brep_valid,
                            "errors": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            if with_render:
                renders = run_dir / "renders"
                renders.mkdir()
                (renders / "contact_sheet.png").write_bytes(b"fake-png")

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            plan = build_benchmark_plan(
                python="python",
                sequence=Path("seq.pkl"),
                vqvae_checkpoint=Path("vq.pt"),
                output_root=output_root,
                run_prefix="candidate_good",
                max_samples=10,
                max_seq_len=1024,
                device="cpu",
                seed=7,
                orders=["shortest", "random", "longest", "most_faces"],
                render=True,
                cols=5,
            )
            counts = {"shortest": 8, "random": 6, "longest": 6, "most_faces": 7}
            for item in plan:
                write_report(Path(item["run_dir"]), counts[item["order"]])

            promoted = summarize_benchmark_plan(plan)
            missing_render_dir = Path(plan[2]["run_dir"]) / "renders" / "contact_sheet.png"
            missing_render_dir.unlink()
            blocked = summarize_benchmark_plan(plan)

        self.assertTrue(promoted["promotion_gate"]["promote"])
        self.assertEqual(promoted["promotion_gate"]["decision"], "promote_for_ar_rebuild")
        self.assertTrue(promoted["promotion_gate"]["requirements"]["longest_improved"])
        self.assertTrue(promoted["promotion_gate"]["requirements"]["most_faces_improved"])
        self.assertFalse(blocked["promotion_gate"]["promote"])
        self.assertFalse(blocked["promotion_gate"]["requirements"]["renders_complete"])
        self.assertIn("rendered contact sheets are incomplete", blocked["promotion_gate"]["reasons"])

    def test_vqvae_server_handoff_manifest_collects_training_and_benchmark_artifacts(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from write_vqvae_server_handoff import build_handoff_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "train_outputs" / "vq_candidate"
            run_dir.mkdir(parents=True)
            for name in ["fsq_vqvae_best.pt", "fsq_vqvae_final.pt", "vqvae_history.json", "server_run_ledger.txt"]:
                (run_dir / name).write_text(name, encoding="utf-8")

            eval_root = root / "reconstruction_eval"
            summary_path = eval_root / "candidate_benchmark_summary.json"
            slice_dir = eval_root / "candidate_longest10"
            (slice_dir / "renders").mkdir(parents=True)
            (slice_dir / "reconstruction_report.json").write_text("{}", encoding="utf-8")
            (slice_dir / "reconstruction_manifest.jsonl").write_text("{}", encoding="utf-8")
            (slice_dir / "renders" / "contact_sheet.png").write_bytes(b"fake")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "promotion_gate": {"decision": "promote_for_ar_rebuild"},
                        "slices": {
                            "longest": {
                                "run_dir": str(slice_dir),
                                "report_path": str(slice_dir / "reconstruction_report.json"),
                                "contact_sheet": str(slice_dir / "renders" / "contact_sheet.png"),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_handoff_manifest(
                run_dir=run_dir,
                benchmark_summary=summary_path,
                repo_root=root,
            )

        self.assertEqual(manifest["promotion_decision"], "promote_for_ar_rebuild")
        self.assertTrue(manifest["complete"])
        artifact_labels = {item["label"] for item in manifest["artifacts"]}
        self.assertIn("vqvae_best_checkpoint", artifact_labels)
        self.assertIn("server_run_ledger", artifact_labels)
        self.assertIn("benchmark_summary", artifact_labels)
        self.assertIn("longest_contact_sheet", artifact_labels)
        missing = [item for item in manifest["artifacts"] if not item["exists"]]
        self.assertEqual(missing, [])

    def test_vqvae_copyback_verifier_promotes_complete_returned_artifacts(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_vqvae_copyback import verify_vqvae_copyback

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "ABC" / "processed" / "train_outputs" / "vq_recovered"
            eval_dir = root / "local_runs" / "reconstruction_eval" / "vq_recovered_longest"
            (eval_dir / "renders").mkdir(parents=True)
            run_dir.mkdir(parents=True)
            for path in [
                run_dir / "fsq_vqvae_best.pt",
                run_dir / "fsq_vqvae_final.pt",
                run_dir / "vqvae_history.json",
                run_dir / "server_run_ledger.txt",
                root / "local_runs" / "reconstruction_eval" / "vq_recovered_benchmark_summary.json",
                eval_dir / "reconstruction_report.json",
                eval_dir / "reconstruction_manifest.jsonl",
                eval_dir / "renders" / "contact_sheet.png",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")
            manifest = {
                "promotion_decision": "promote_for_ar_rebuild",
                "artifacts": [
                    {"label": "vqvae_best_checkpoint", "relative_path": "ABC/processed/train_outputs/vq_recovered/fsq_vqvae_best.pt", "required": True},
                    {"label": "benchmark_summary", "relative_path": "local_runs/reconstruction_eval/vq_recovered_benchmark_summary.json", "required": True},
                    {"label": "longest_contact_sheet", "relative_path": "local_runs/reconstruction_eval/vq_recovered_longest/renders/contact_sheet.png", "required": True},
                ],
            }
            manifest_path = run_dir / "copy_back_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = verify_vqvae_copyback(manifest_path=manifest_path, repo_root=root)

        self.assertEqual(report["status"], "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD")
        self.assertTrue(report["copyback_complete"])
        self.assertEqual(report["next_action"], "run_source_path_sequence_rebuild")
        self.assertEqual(report["missing_required"], [])

    def test_vqvae_copyback_verifier_blocks_missing_required_artifacts(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_vqvae_copyback import verify_vqvae_copyback

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "ABC" / "processed" / "train_outputs" / "vq_recovered"
            run_dir.mkdir(parents=True)
            (run_dir / "fsq_vqvae_best.pt").write_text("checkpoint", encoding="utf-8")
            manifest_path = run_dir / "copy_back_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "promotion_decision": "promote_for_ar_rebuild",
                        "artifacts": [
                            {
                                "label": "vqvae_best_checkpoint",
                                "relative_path": "ABC/processed/train_outputs/vq_recovered/fsq_vqvae_best.pt",
                                "required": True,
                            },
                            {
                                "label": "longest_contact_sheet",
                                "relative_path": "local_runs/reconstruction_eval/vq_recovered_longest/renders/contact_sheet.png",
                                "required": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = verify_vqvae_copyback(manifest_path=manifest_path, repo_root=root)

        self.assertEqual(report["status"], "COPYBACK_INCOMPLETE")
        self.assertFalse(report["copyback_complete"])
        self.assertEqual(report["next_action"], "copy_missing_artifacts_before_deleting_server")
        self.assertEqual(report["missing_required"][0]["label"], "longest_contact_sheet")

    def test_sequence_rebuild_verifier_promotes_source_path_ready_package_for_ar(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_sequence_rebuild import verify_sequence_rebuild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = root / "local_runs" / "ar_training" / "train_outputs" / "seq" / "sequences_fsq_rcm.pkl"
            audit = sequence.parent / "source_path_audit.json"
            coverage = sequence.parent / "length_coverage.json"
            sequence.parent.mkdir(parents=True)
            sequence.write_bytes(b"sequence-package")
            audit.write_text(
                json.dumps(
                    {
                        "all_splits_source_path_ready": True,
                        "validation_most_curved_ready": True,
                        "source_path_coverage": 1.0,
                        "groups_missing_source_path": 0,
                    }
                ),
                encoding="utf-8",
            )
            coverage.write_text(
                json.dumps(
                    {
                        "recommendation": {
                            "action": "train_long_context_ar",
                            "preferred_max_seq_len": 2048,
                        },
                        "overall": {
                            "complex_total": 100,
                            "by_limit": {
                                "1536": {"complex_allowed": 90, "complex_allowed_fraction": 0.9},
                                "2048": {"complex_allowed": 98, "complex_allowed_fraction": 0.98},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = verify_sequence_rebuild(sequence=sequence, source_path_audit=audit, length_coverage=coverage)

        self.assertEqual(report["status"], "READY_FOR_AR_LONG_CONTEXT")
        self.assertTrue(report["sequence_rebuild_ready"])
        self.assertEqual(report["next_action"], "train_ar1536_then_ar2048_if_needed")
        self.assertEqual(report["recommended_max_seq_len"], 2048)
        self.assertEqual(report["blocking_reasons"], [])

    def test_sequence_rebuild_verifier_blocks_missing_source_paths(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_sequence_rebuild import verify_sequence_rebuild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = root / "sequences_fsq_rcm.pkl"
            audit = root / "source_path_audit.json"
            sequence.write_bytes(b"sequence-package")
            audit.write_text(
                json.dumps(
                    {
                        "all_splits_source_path_ready": False,
                        "validation_most_curved_ready": False,
                        "source_path_coverage": 0.5,
                        "groups_missing_source_path": 25,
                    }
                ),
                encoding="utf-8",
            )

            report = verify_sequence_rebuild(sequence=sequence, source_path_audit=audit)

        self.assertEqual(report["status"], "SEQUENCE_REBUILD_NOT_READY")
        self.assertFalse(report["sequence_rebuild_ready"])
        self.assertEqual(report["next_action"], "fix_sequence_rebuild_before_ar_training")
        self.assertIn("source_path_audit_not_ready", report["blocking_reasons"])

    def test_server_handoff_preflight_summarizes_current_gates_and_next_action(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from prepare_server_handoff import build_preflight_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            required_files = [
                root / "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt",
                root / "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
                root / "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl",
                root / "ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
                root / "tools/run_vqvae_complex_recovery.sh",
                root / "tools/run_source_path_sequence_rebuild.sh",
                root / "tools/run_ar_v13_long_context.sh",
                root / "tools/run_vqvae_slice_benchmark.py",
                root / "tools/monitor_vqvae_recovery_gate.py",
                root / "tools/summarize_generated_quality.py",
                root / "tools/audit_step_geometry_entities.py",
                root / "tools/audit_parsed_pool_quality.py",
                root / "tools/audit_sequence_source_paths.py",
                root / "tools/verify_server_training_readiness.py",
                root / "tools/verify_model_artifacts.py",
                root / "tools/plan_server_quality_recovery.py",
                root / "tools/run_server_quality_recovery.py",
                root / "tools/verify_vqvae_copyback.py",
                root / "tools/verify_sequence_rebuild.py",
                root / "tools/decide_quality_recovery_stage.py",
                root / "tools/summarize_quality_recovery_progress.py",
                root / "tools/build_server_training_phase_budget.py",
                root / "tools/build_server_recovery_packet.py",
                root / "local_reports/v13_server_first_hour_from_packet_20260706.sh",
                root / "local_reports/v13_next_server_quality_recovery_runbook_20260706.md",
                root / "local_reports/v13_rented_server_start_here_20260706.md",
                root / "local_reports/v13_rental_gpu_decision_card_20260706.md",
                root / "local_reports/v13_server_training_phase_budget_20260706.json",
                root / "local_reports/v13_server_training_phase_budget_20260706.md",
            ]
            for path in required_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            g20 = root / "local_runs/reconstruction_eval/eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag/generated_quality_summary.json"
            g100 = root / "local_runs/reconstruction_eval/eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342/generated_quality_summary.json"
            vq = root / "local_runs/reconstruction_eval/vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json"
            length = root / "local_reports/v13_ar120_length_coverage_20260706.json"
            audit = root / "local_reports/v13_ar120_sequence_source_path_audit_20260706.json"
            for path in [g20, g100, vq, length, audit]:
                path.parent.mkdir(parents=True, exist_ok=True)

            g20.write_text(
                json.dumps(
                    {
                        "paper_gate": {"decision": "hold_for_failure_analysis"},
                        "summary": {"attempted": 20, "strict_valid": 16},
                        "topology": {"top_two_fraction": 0.737},
                        "complexity": {"strict_valid_complex": 2},
                        "semantic_complexity": {
                            "nonprimitive_strict_valid": 1,
                            "primitive_like_strict_valid_fraction": 0.9375,
                        },
                    }
                ),
                encoding="utf-8",
            )
            g100.write_text(
                json.dumps(
                    {
                        "paper_gate": {"decision": "hold_for_failure_analysis"},
                        "summary": {"attempted": 100, "strict_valid": 78},
                        "topology": {"top_two_fraction": 0.713},
                        "complexity": {"strict_valid_complex": 0},
                        "semantic_complexity": {
                            "nonprimitive_strict_valid": 0,
                            "primitive_like_strict_valid_fraction": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            vq.write_text(
                json.dumps(
                    {
                        "promotion_gate": {"decision": "hold_vqvae_checkpoint"},
                        "slices": {"longest": {"brep_valid": 3}, "most_faces": {"brep_valid": 5}},
                    }
                ),
                encoding="utf-8",
            )
            length.write_text(
                json.dumps(
                    {
                        "recommendation": {"action": "train_long_context_ar", "preferred_max_seq_len": 2048},
                        "overall": {
                            "complex_total": 289076,
                            "by_limit": {
                                "1024": {"complex_allowed": 186502, "complex_allowed_fraction": 0.6452},
                                "1536": {"complex_allowed": 257096, "complex_allowed_fraction": 0.8894},
                                "2048": {"complex_allowed": 284148, "complex_allowed_fraction": 0.983},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit.write_text(
                json.dumps(
                    {
                        "validation_most_curved_ready": False,
                        "groups_with_source_path": 0,
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_preflight_manifest(root)

        self.assertEqual(manifest["status"], "READY_FOR_SERVER_VQVAE_RECOVERY")
        self.assertTrue(manifest["required_artifacts_complete"])
        self.assertEqual(manifest["current_gates"]["g100_generated"]["decision"], "hold_for_failure_analysis")
        self.assertEqual(manifest["current_gates"]["vqvae_baseline"]["decision"], "hold_vqvae_checkpoint")
        self.assertFalse(manifest["current_gates"]["source_path_audit"]["validation_most_curved_ready"])
        self.assertEqual(manifest["current_gates"]["length_coverage"]["preferred_max_seq_len"], 2048)
        self.assertEqual(manifest["current_gates"]["g20_generated"]["nonprimitive_strict_valid"], 1)
        self.assertEqual(manifest["current_gates"]["g100_generated"]["primitive_like_strict_valid_fraction"], 1.0)
        self.assertEqual(manifest["next_action"], "rent_single_gpu_and_run_vqvae_recovery")
        self.assertEqual(manifest["server_phase_order"][0], "vqvae_complex_curved_recovery")
        self.assertIn("bash -n tools/run_vqvae_complex_recovery.sh", manifest["server_syntax_checks"])
        self.assertIn(
            "bash -n local_reports/v13_server_first_hour_from_packet_20260706.sh",
            manifest["server_syntax_checks"],
        )
        required_labels = {item["label"] for item in manifest["required_artifacts"]}
        self.assertIn("vqvae_copyback_verifier", required_labels)
        self.assertIn("sequence_rebuild_verifier", required_labels)
        self.assertIn("quality_recovery_stage_decider", required_labels)
        self.assertIn("quality_recovery_progress_summary", required_labels)
        self.assertIn("server_training_phase_budget_builder", required_labels)
        self.assertIn("server_recovery_packet_builder", required_labels)
        self.assertIn("server_first_hour_script", required_labels)
        self.assertIn("server_start_here_guide", required_labels)
        self.assertIn("rental_gpu_decision_card", required_labels)
        self.assertIn("server_training_phase_budget_json", required_labels)
        self.assertIn("server_training_phase_budget_report", required_labels)
        self.assertIn("step_geometry_entity_audit", required_labels)
        self.assertIn("parsed_pool_quality_audit", required_labels)

    def test_quality_recovery_decision_prioritizes_vqvae_when_generated_and_vq_gates_hold(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from decide_quality_recovery_stage import decide_quality_recovery_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            g20 = root / "local_runs/reconstruction_eval/eval_generated20_lr5e6_epoch120_best_temp095_topp95_max512_random_cpu_20260705_diag/generated_quality_summary.json"
            g100 = root / "local_runs/reconstruction_eval/eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342/generated_quality_summary.json"
            vq = root / "local_runs/reconstruction_eval/vqvae_epoch100_complexity_benchmark_20260705_benchmark_summary.json"
            length = root / "local_reports/v13_ar120_length_coverage_20260706.json"
            audit = root / "local_reports/v13_ar120_sequence_source_path_audit_20260706.json"
            for path in [g20, g100, vq, length, audit]:
                path.parent.mkdir(parents=True, exist_ok=True)

            g20.write_text(
                json.dumps(
                    {
                        "paper_gate": {"decision": "hold_for_failure_analysis"},
                        "summary": {"attempted": 20, "strict_valid": 16, "step_saved": 19},
                        "topology": {"top_two_fraction": 0.737},
                        "complexity": {"strict_valid_complex": 2},
                        "semantic_complexity": {
                            "nonprimitive_strict_valid": 1,
                            "primitive_like_strict_valid_fraction": 0.9375,
                        },
                    }
                ),
                encoding="utf-8",
            )
            g100.write_text(
                json.dumps(
                    {
                        "paper_gate": {"decision": "hold_for_failure_analysis"},
                        "summary": {"attempted": 100, "strict_valid": 78, "step_saved": 87},
                        "topology": {"top_two_fraction": 0.713},
                        "complexity": {"strict_valid_complex": 0},
                        "semantic_complexity": {
                            "nonprimitive_strict_valid": 0,
                            "primitive_like_strict_valid_fraction": 1.0,
                        },
                        "step_hashes": {"unique_step_rate": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            vq.write_text(
                json.dumps(
                    {
                        "promotion_gate": {"decision": "hold_vqvae_checkpoint"},
                        "slices": {"longest": {"brep_valid": 3}, "most_faces": {"brep_valid": 5}},
                    }
                ),
                encoding="utf-8",
            )
            length.write_text(
                json.dumps(
                    {
                        "recommendation": {"preferred_max_seq_len": 2048},
                        "overall": {
                            "complex_total": 289076,
                            "by_limit": {
                                "1024": {"complex_allowed": 186502, "complex_allowed_fraction": 0.645},
                                "1536": {"complex_allowed": 257096, "complex_allowed_fraction": 0.889},
                                "2048": {"complex_allowed": 284148, "complex_allowed_fraction": 0.983},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit.write_text(
                json.dumps({"validation_most_curved_ready": False, "groups_with_source_path": 0}),
                encoding="utf-8",
            )

            decision = decide_quality_recovery_stage(root)

        self.assertEqual(decision["status"], "NEEDS_VQVAE_RECOVERY")
        self.assertEqual(decision["next_stage"], "vqvae_complex_curved_recovery")
        self.assertEqual(decision["evidence"]["primary_bottleneck"], "vqvae_reconstruction")
        self.assertFalse(decision["can_train_ar_now"])
        self.assertFalse(decision["paper_figure_policy"]["positive_figures_allowed"])
        self.assertIn("generated_quality_not_promoted", decision["blocking_reasons"])
        self.assertIn("vqvae_checkpoint_not_promoted", decision["blocking_reasons"])
        self.assertEqual(decision["evidence"]["g20_generated"]["nonprimitive_strict_valid"], 1)
        self.assertEqual(decision["evidence"]["g100_generated"]["primitive_like_strict_valid_fraction"], 1.0)
        self.assertEqual(decision["stage_order"][0], "vqvae_complex_curved_recovery")
        self.assertEqual(decision["server_recommendation"]["first_choice"], "1x L40S 48GB or 1x RTX 6000 Ada/A6000 48GB")

    def test_server_transfer_manifest_maps_local_artifacts_to_server_paths(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from build_server_transfer_manifest import build_transfer_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_paths = [
                "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt",
                "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
                "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/split.pkl",
                "ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
                "tools/run_vqvae_complex_recovery.sh",
                "tools/run_source_path_sequence_rebuild.sh",
                "tools/run_ar_v13_long_context.sh",
                "tools/run_vqvae_slice_benchmark.py",
                "tools/monitor_vqvae_recovery_gate.py",
                "tools/evaluate_reconstruction_v13.py",
                "tools/summarize_generated_quality.py",
                "tools/audit_step_geometry_entities.py",
                "tools/audit_parsed_pool_quality.py",
                "tools/audit_paper_figure_candidates.py",
                "tools/audit_sequence_source_paths.py",
                "tools/prepare_server_handoff.py",
                "tools/write_vqvae_server_handoff.py",
                "tools/verify_server_transfer.py",
                "tools/verify_server_training_readiness.py",
                "tools/verify_model_artifacts.py",
                "tools/plan_server_quality_recovery.py",
                "tools/run_server_quality_recovery.py",
                "tools/verify_vqvae_copyback.py",
                "tools/verify_sequence_rebuild.py",
                "tools/decide_quality_recovery_stage.py",
                "tools/summarize_quality_recovery_progress.py",
                "tools/build_server_training_phase_budget.py",
                "tools/build_server_recovery_packet.py",
                "local_reports/v13_server_first_hour_from_packet_20260706.sh",
                "papers/aaai_v13/render_step_directory.py",
                "BrepARG/2sequence.py",
                "BrepARG/utils.py",
                "breparg_improvements/train.py",
                "breparg_improvements/vqvae_sampling.py",
                "breparg_improvements/fsq_quantise.py",
                "breparg_improvements/gnn_ordering.py",
                "breparg_improvements/ar_training_utils.py",
                "breparg_improvements/training_stability.py",
                "breparg_improvements/sequence_sharding.py",
                "tools/run_sharded_sequence.py",
                "local_reports/v13_next_server_quality_recovery_runbook_20260706.md",
                "local_reports/v13_rented_server_start_here_20260706.md",
                "local_reports/v13_rental_gpu_decision_card_20260706.md",
                "local_reports/v13_server_training_phase_budget_20260706.json",
                "local_reports/v13_server_training_phase_budget_20260706.md",
                "local_reports/v13_server_handoff_preflight_20260706.json",
            ]
            for index, rel in enumerate(local_paths):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"artifact {index}", encoding="utf-8")

            manifest = build_transfer_manifest(
                root,
                server_repo_root="/workspace/V13",
                server_abc_train_root="/workspace/ABC/processed/train_outputs",
                hash_limit_bytes=1024,
            )

        self.assertEqual(manifest["status"], "READY_TO_TRANSFER")
        self.assertTrue(manifest["required_sources_complete"])
        entries = {item["label"]: item for item in manifest["entries"]}
        self.assertEqual(
            entries["ar_sequence_package"]["server_path"],
            "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
        )
        self.assertEqual(
            entries["vqvae_baseline_checkpoint"]["server_path"],
            "/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
        )
        self.assertEqual(entries["ar_best_checkpoint"]["transfer_group"], "large_model_artifact")
        self.assertEqual(entries["breparg_source_path_sequence"]["server_path"], "/workspace/V13/BrepARG/2sequence.py")
        self.assertEqual(entries["vqvae_training_entry"]["server_path"], "/workspace/V13/breparg_improvements/train.py")
        self.assertEqual(entries["vqvae_sampling_source"]["transfer_group"], "repo_source")
        self.assertEqual(entries["sequence_sharded_runner"]["server_path"], "/workspace/V13/tools/run_sharded_sequence.py")
        self.assertEqual(entries["reconstruction_evaluator"]["server_path"], "/workspace/V13/tools/evaluate_reconstruction_v13.py")
        self.assertEqual(entries["step_renderer"]["server_path"], "/workspace/V13/papers/aaai_v13/render_step_directory.py")
        self.assertEqual(entries["vqvae_recovery_monitor"]["server_path"], "/workspace/V13/tools/monitor_vqvae_recovery_gate.py")
        self.assertEqual(entries["step_geometry_entity_audit"]["server_path"], "/workspace/V13/tools/audit_step_geometry_entities.py")
        self.assertEqual(entries["step_geometry_entity_audit"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["parsed_pool_quality_audit"]["server_path"], "/workspace/V13/tools/audit_parsed_pool_quality.py")
        self.assertEqual(entries["parsed_pool_quality_audit"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["paper_figure_candidate_audit"]["server_path"], "/workspace/V13/tools/audit_paper_figure_candidates.py")
        self.assertEqual(entries["paper_figure_candidate_audit"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["server_transfer_verifier"]["server_path"], "/workspace/V13/tools/verify_server_transfer.py")
        self.assertEqual(entries["server_training_readiness_verifier"]["server_path"], "/workspace/V13/tools/verify_server_training_readiness.py")
        self.assertEqual(entries["model_artifact_sanity_verifier"]["server_path"], "/workspace/V13/tools/verify_model_artifacts.py")
        self.assertEqual(entries["server_recovery_plan_tool"]["server_path"], "/workspace/V13/tools/plan_server_quality_recovery.py")
        self.assertEqual(entries["server_quality_recovery_orchestrator"]["server_path"], "/workspace/V13/tools/run_server_quality_recovery.py")
        self.assertEqual(entries["vqvae_copyback_verifier"]["server_path"], "/workspace/V13/tools/verify_vqvae_copyback.py")
        self.assertEqual(entries["vqvae_copyback_verifier"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["sequence_rebuild_verifier"]["server_path"], "/workspace/V13/tools/verify_sequence_rebuild.py")
        self.assertEqual(entries["sequence_rebuild_verifier"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["quality_recovery_stage_decider"]["server_path"], "/workspace/V13/tools/decide_quality_recovery_stage.py")
        self.assertEqual(entries["quality_recovery_stage_decider"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["quality_recovery_progress_summary"]["server_path"], "/workspace/V13/tools/summarize_quality_recovery_progress.py")
        self.assertEqual(entries["quality_recovery_progress_summary"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["server_training_phase_budget_builder"]["server_path"], "/workspace/V13/tools/build_server_training_phase_budget.py")
        self.assertEqual(entries["server_training_phase_budget_builder"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["server_recovery_packet_builder"]["server_path"], "/workspace/V13/tools/build_server_recovery_packet.py")
        self.assertEqual(entries["server_recovery_packet_builder"]["transfer_group"], "repo_tooling")
        self.assertEqual(entries["server_first_hour_script"]["server_path"], "/workspace/V13/local_reports/v13_server_first_hour_from_packet_20260706.sh")
        self.assertEqual(entries["server_first_hour_script"]["transfer_group"], "repo_report")
        self.assertEqual(entries["server_start_here_guide"]["server_path"], "/workspace/V13/local_reports/v13_rented_server_start_here_20260706.md")
        self.assertEqual(entries["server_start_here_guide"]["transfer_group"], "repo_report")
        self.assertEqual(entries["rental_gpu_decision_card"]["server_path"], "/workspace/V13/local_reports/v13_rental_gpu_decision_card_20260706.md")
        self.assertEqual(entries["rental_gpu_decision_card"]["transfer_group"], "repo_report")
        self.assertEqual(entries["server_training_phase_budget_json"]["server_path"], "/workspace/V13/local_reports/v13_server_training_phase_budget_20260706.json")
        self.assertEqual(entries["server_training_phase_budget_json"]["transfer_group"], "repo_report")
        self.assertEqual(entries["server_training_phase_budget_report"]["server_path"], "/workspace/V13/local_reports/v13_server_training_phase_budget_20260706.md")
        self.assertEqual(entries["server_training_phase_budget_report"]["transfer_group"], "repo_report")
        self.assertRegex(entries["ar_best_checkpoint"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("rsync", manifest["suggested_upload_commands"][0])
        self.assertIn("test -s", manifest["server_verify_commands"][0])

    def test_server_transfer_manifest_renders_single_powershell_upload_script(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from build_server_transfer_manifest import render_powershell_upload_script

        manifest = {
            "status": "READY_TO_TRANSFER",
            "required_sources_complete": True,
            "entries": [
                {
                    "label": "ar_sequence_package",
                    "local_path": "local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
                    "server_path": "/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl",
                    "exists": True,
                },
                {
                    "label": "vqvae_baseline_checkpoint",
                    "local_path": "ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
                    "server_path": "/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
                    "exists": True,
                },
            ],
            "missing_entries": [],
        }

        script = render_powershell_upload_script(manifest, remote="gpu.example")

        self.assertIn("$ErrorActionPreference = 'Stop'", script)
        self.assertIn("$Remote = 'gpu.example'", script)
        self.assertIn(
            "ssh $Remote \"mkdir -p '/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100' "
            "'/workspace/V13/local_reports' "
            "'/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6'\"",
            script,
        )
        self.assertIn(
            "rsync -av --progress 'local_reports/v13_server_transfer_manifest_20260706.json' "
            "\"${Remote}:'/workspace/V13/local_reports/'\"",
            script,
        )
        self.assertIn(
            "rsync -av --progress 'local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/sequences_fsq_rcm.pkl' "
            "\"${Remote}:'/workspace/V13/local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/'\"",
            script,
        )
        self.assertIn(
            "rsync -av --progress 'ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt' "
            "\"${Remote}:'/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/'\"",
            script,
        )
        self.assertIn("tools/verify_server_transfer.py", script)

    def test_server_transfer_verifier_accepts_complete_uploaded_manifest(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import hashlib
        from verify_server_transfer import verify_transfer_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_repo = root / "workspace" / "V13"
            abc_train = root / "workspace" / "ABC" / "processed" / "train_outputs"
            abc_pool = root / "workspace" / "ABC" / "processed" / "abc_parsed_full"
            small = server_repo / "tools" / "run.py"
            large = server_repo / "local_runs" / "sequence.pkl"
            vqvae = abc_train / "newscheme_full_vqvae_epoch100" / "fsq_vqvae_best.pt"
            abc_pool.mkdir(parents=True)
            (abc_pool / "shape.pkl").write_text("parsed", encoding="utf-8")
            small.parent.mkdir(parents=True)
            large.parent.mkdir(parents=True)
            vqvae.parent.mkdir(parents=True)
            small.write_text("print('ok')", encoding="utf-8")
            large.write_bytes(b"sequence-bytes")
            vqvae.write_bytes(b"checkpoint")
            small_digest = hashlib.sha256(small.read_bytes()).hexdigest()

            manifest = {
                "entries": [
                    {
                        "label": "tool",
                        "server_path": "/workspace/V13/tools/run.py",
                        "bytes": small.stat().st_size,
                        "sha256": small_digest,
                    },
                    {
                        "label": "sequence",
                        "server_path": "/workspace/V13/local_runs/sequence.pkl",
                        "bytes": large.stat().st_size,
                        "sha256": None,
                        "hash_skipped": True,
                    },
                    {
                        "label": "vqvae",
                        "server_path": "/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt",
                        "bytes": vqvae.stat().st_size,
                        "sha256": None,
                        "hash_skipped": True,
                    },
                ],
                "server_data_requirements": [
                    {
                        "label": "parsed_abc_pool",
                        "server_path": "/workspace/ABC/processed/abc_parsed_full",
                    }
                ],
            }

            report = verify_transfer_manifest(
                manifest,
                path_map={
                    "/workspace/V13": server_repo,
                    "/workspace/ABC/processed/train_outputs": abc_train,
                    "/workspace/ABC/processed/abc_parsed_full": abc_pool,
                },
            )

        self.assertEqual(report["status"], "READY_FOR_SERVER_RUN")
        self.assertEqual(report["summary"]["entries_total"], 3)
        self.assertEqual(report["summary"]["entries_ok"], 3)
        self.assertEqual(report["summary"]["data_requirements_ok"], 1)
        self.assertEqual(report["entries"][0]["sha256_match"], True)
        self.assertEqual(report["entries"][1]["bytes_match"], True)
        self.assertTrue(report["entries"][0]["resolved_path"].endswith("tools\\run.py") or report["entries"][0]["resolved_path"].endswith("tools/run.py"))

    def test_server_transfer_verifier_reports_missing_and_checksum_failures(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_server_transfer import verify_transfer_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_repo = root / "workspace" / "V13"
            bad = server_repo / "tools" / "bad.py"
            bad.parent.mkdir(parents=True)
            bad.write_text("wrong", encoding="utf-8")
            manifest = {
                "entries": [
                    {
                        "label": "bad_hash",
                        "server_path": "/workspace/V13/tools/bad.py",
                        "bytes": bad.stat().st_size,
                        "sha256": "0" * 64,
                    },
                    {
                        "label": "missing",
                        "server_path": "/workspace/V13/tools/missing.py",
                        "bytes": 10,
                        "sha256": None,
                    },
                ],
                "server_data_requirements": [
                    {
                        "label": "parsed_abc_pool",
                        "server_path": "/workspace/ABC/processed/abc_parsed_full",
                    }
                ],
            }

            report = verify_transfer_manifest(manifest, path_map={"/workspace/V13": server_repo})

        self.assertEqual(report["status"], "TRANSFER_VERIFICATION_FAILED")
        entries = {entry["label"]: entry for entry in report["entries"]}
        data = {item["label"]: item for item in report["server_data_requirements"]}
        self.assertIn("sha256_mismatch", entries["bad_hash"]["issues"])
        self.assertIn("missing", entries["missing"]["issues"])
        self.assertIn("missing", data["parsed_abc_pool"]["issues"])
        self.assertEqual(report["summary"]["entries_failed"], 2)

    def test_server_training_readiness_accepts_complete_cuda_environment(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_server_training_readiness import evaluate_training_readiness

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed_pool = root / "workspace" / "ABC" / "processed" / "abc_parsed_full"
            repo = root / "workspace" / "V13"
            vqvae = root / "workspace" / "ABC" / "processed" / "train_outputs" / "newscheme_full_vqvae_epoch100" / "fsq_vqvae_best.pt"
            sequence = repo / "local_runs" / "ar_training" / "train_outputs" / "newscheme_full_v13_ar_lr5e6" / "sequences_fsq_rcm.pkl"
            split = sequence.parent / "split.pkl"
            launchers = [
                repo / "tools" / "run_vqvae_complex_recovery.sh",
                repo / "tools" / "run_source_path_sequence_rebuild.sh",
                repo / "tools" / "run_ar_v13_long_context.sh",
            ]
            transfer_report = repo / "local_reports" / "v13_server_transfer_verify_server.json"
            self.write_parsed_shape(parsed_pool / "complex_curved.pkl", n_faces=14, n_edges=24, curved=True)
            vqvae.parent.mkdir(parents=True)
            sequence.parent.mkdir(parents=True)
            launchers[0].parent.mkdir(parents=True)
            transfer_report.parent.mkdir(parents=True)
            vqvae.write_bytes(b"vqvae")
            sequence.write_bytes(b"sequence")
            split.write_bytes(b"split")
            for launcher in launchers:
                launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            transfer_report.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")

            report = evaluate_training_readiness(
                repo_root=repo,
                parsed_pool=parsed_pool,
                vqvae_checkpoint=vqvae,
                sequence=sequence,
                split=split,
                transfer_verification=transfer_report,
                module_checker=lambda name: {"module": name, "ok": True, "reason": "available"},
                cuda_probe=lambda: {"torch_import_ok": True, "cuda_available": True, "device_count": 1, "devices": [{"index": 0, "name": "L40S", "total_memory_gb": 48.0}]},
                script_checker=lambda path: {"path": str(path), "ok": True, "reason": "syntax_ok"},
                parsed_quality_max_files=8,
                min_parsed_files=1,
                min_complex_sources=1,
                min_complex_source_fraction=0.5,
                min_curved_patches=1,
                min_curved_patch_fraction=0.02,
                curved_score_threshold=0.02,
            )

        self.assertEqual(report["status"], "READY_FOR_VQVAE_TRAINING")
        self.assertTrue(report["training_allowed"])
        self.assertEqual(report["summary"]["required_paths_failed"], 0)
        self.assertEqual(report["summary"]["python_modules_failed"], 0)
        self.assertTrue(report["cuda"]["cuda_available"])
        self.assertTrue(report["gpu_memory"]["meets_minimum"])
        self.assertEqual(report["gpu_memory"]["largest_device_memory_gb"], 48.0)
        self.assertEqual(report["transfer_verification"]["status"], "READY_FOR_SERVER_RUN")
        self.assertEqual(report["parsed_pool_quality"]["status"], "PARSED_POOL_QUALITY_READY")

    def test_server_training_readiness_blocks_gpu_below_recovery_minimum(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_server_training_readiness import evaluate_training_readiness

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed_pool = root / "workspace" / "ABC" / "processed" / "abc_parsed_full"
            repo = root / "workspace" / "V13"
            vqvae = root / "workspace" / "ABC" / "processed" / "train_outputs" / "newscheme_full_vqvae_epoch100" / "fsq_vqvae_best.pt"
            sequence = repo / "local_runs" / "ar_training" / "train_outputs" / "newscheme_full_v13_ar_lr5e6" / "sequences_fsq_rcm.pkl"
            split = sequence.parent / "split.pkl"
            launchers = [
                repo / "tools" / "run_vqvae_complex_recovery.sh",
                repo / "tools" / "run_source_path_sequence_rebuild.sh",
                repo / "tools" / "run_ar_v13_long_context.sh",
            ]
            transfer_report = repo / "local_reports" / "v13_server_transfer_verify_server.json"
            self.write_parsed_shape(parsed_pool / "complex_curved.pkl", n_faces=14, n_edges=24, curved=True)
            vqvae.parent.mkdir(parents=True)
            sequence.parent.mkdir(parents=True)
            launchers[0].parent.mkdir(parents=True)
            transfer_report.parent.mkdir(parents=True)
            vqvae.write_bytes(b"vqvae")
            sequence.write_bytes(b"sequence")
            split.write_bytes(b"split")
            for launcher in launchers:
                launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            transfer_report.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")

            report = evaluate_training_readiness(
                repo_root=repo,
                parsed_pool=parsed_pool,
                vqvae_checkpoint=vqvae,
                sequence=sequence,
                split=split,
                transfer_verification=transfer_report,
                module_checker=lambda name: {"module": name, "ok": True, "reason": "available"},
                cuda_probe=lambda: {
                    "torch_import_ok": True,
                    "cuda_available": True,
                    "device_count": 1,
                    "devices": [{"index": 0, "name": "RTX 4090", "total_memory_gb": 24.0}],
                },
                script_checker=lambda path: {"path": str(path), "ok": True, "reason": "syntax_ok"},
                parsed_quality_max_files=8,
                min_parsed_files=1,
                min_complex_sources=1,
                min_complex_source_fraction=0.5,
                min_curved_patches=1,
                min_curved_patch_fraction=0.02,
                curved_score_threshold=0.02,
            )

        self.assertEqual(report["status"], "SERVER_TRAINING_READINESS_FAILED")
        self.assertFalse(report["training_allowed"])
        self.assertIn("gpu_memory_below_minimum", report["blocking_reasons"])
        self.assertFalse(report["gpu_memory"]["meets_minimum"])
        self.assertEqual(report["gpu_memory"]["minimum_required_gb"], 40.0)
        self.assertEqual(report["parsed_pool_quality"]["status"], "PARSED_POOL_QUALITY_READY")

    def test_server_training_readiness_blocks_low_quality_parsed_pool(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_server_training_readiness import evaluate_training_readiness

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed_pool = root / "workspace" / "ABC" / "processed" / "abc_parsed_full"
            repo = root / "workspace" / "V13"
            vqvae = root / "workspace" / "ABC" / "processed" / "train_outputs" / "newscheme_full_vqvae_epoch100" / "fsq_vqvae_best.pt"
            sequence = repo / "local_runs" / "ar_training" / "train_outputs" / "newscheme_full_v13_ar_lr5e6" / "sequences_fsq_rcm.pkl"
            split = sequence.parent / "split.pkl"
            launchers = [
                repo / "tools" / "run_vqvae_complex_recovery.sh",
                repo / "tools" / "run_source_path_sequence_rebuild.sh",
                repo / "tools" / "run_ar_v13_long_context.sh",
            ]
            transfer_report = repo / "local_reports" / "v13_server_transfer_verify_server.json"
            self.write_parsed_shape(parsed_pool / "simple_a.pkl", n_faces=2, n_edges=2)
            self.write_parsed_shape(parsed_pool / "simple_b.pkl", n_faces=4, n_edges=4)
            vqvae.parent.mkdir(parents=True)
            sequence.parent.mkdir(parents=True)
            launchers[0].parent.mkdir(parents=True)
            transfer_report.parent.mkdir(parents=True)
            vqvae.write_bytes(b"vqvae")
            sequence.write_bytes(b"sequence")
            split.write_bytes(b"split")
            for launcher in launchers:
                launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            transfer_report.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")

            report = evaluate_training_readiness(
                repo_root=repo,
                parsed_pool=parsed_pool,
                vqvae_checkpoint=vqvae,
                sequence=sequence,
                split=split,
                transfer_verification=transfer_report,
                module_checker=lambda name: {"module": name, "ok": True, "reason": "available"},
                cuda_probe=lambda: {
                    "torch_import_ok": True,
                    "cuda_available": True,
                    "device_count": 1,
                    "devices": [{"index": 0, "name": "L40S", "total_memory_gb": 48.0}],
                },
                script_checker=lambda path: {"path": str(path), "ok": True, "reason": "syntax_ok"},
                parsed_quality_max_files=8,
                min_parsed_files=2,
                min_complex_sources=1,
                min_complex_source_fraction=0.25,
                min_curved_patches=1,
                min_curved_patch_fraction=0.02,
                curved_score_threshold=0.02,
            )

        self.assertEqual(report["status"], "SERVER_TRAINING_READINESS_FAILED")
        self.assertFalse(report["training_allowed"])
        self.assertIn("parsed_pool_quality_failed", report["blocking_reasons"])
        self.assertEqual(report["parsed_pool_quality"]["status"], "PARSED_POOL_QUALITY_FAILED")
        self.assertIn("complex_sources_below_minimum", report["parsed_pool_quality"]["blocking_reasons"])
        self.assertIn("curved_patches_below_minimum", report["parsed_pool_quality"]["blocking_reasons"])

    def test_server_training_readiness_blocks_missing_cuda_and_data(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from verify_server_training_readiness import evaluate_training_readiness

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "workspace" / "V13"
            repo.mkdir(parents=True)
            transfer_report = repo / "local_reports" / "v13_server_transfer_verify_server.json"
            transfer_report.parent.mkdir(parents=True)
            transfer_report.write_text(json.dumps({"status": "TRANSFER_VERIFICATION_FAILED"}), encoding="utf-8")

            report = evaluate_training_readiness(
                repo_root=repo,
                parsed_pool=root / "workspace" / "ABC" / "processed" / "abc_parsed_full",
                vqvae_checkpoint=root / "missing" / "fsq_vqvae_best.pt",
                sequence=repo / "missing_sequences.pkl",
                split=repo / "missing_split.pkl",
                transfer_verification=transfer_report,
                module_checker=lambda name: {"module": name, "ok": name != "OCC.Core.TopoDS", "reason": "available" if name != "OCC.Core.TopoDS" else "missing"},
                cuda_probe=lambda: {"torch_import_ok": True, "cuda_available": False, "device_count": 0, "devices": []},
                script_checker=lambda path: {"path": str(path), "ok": False, "reason": "missing_bash_or_script"},
            )

        self.assertEqual(report["status"], "SERVER_TRAINING_READINESS_FAILED")
        self.assertFalse(report["training_allowed"])
        self.assertGreater(report["summary"]["required_paths_failed"], 0)
        self.assertGreater(report["summary"]["python_modules_failed"], 0)
        self.assertIn("cuda_unavailable", report["blocking_reasons"])
        self.assertIn("transfer_verification_not_ready", report["blocking_reasons"])

    def test_model_artifact_sanity_accepts_loadable_artifacts(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import pickle
        import torch
        from verify_model_artifacts import evaluate_model_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vqvae = root / "fsq_vqvae_best.pt"
            ar = root / "ar_best.pt"
            sequence = root / "sequences_fsq_rcm.pkl"
            split = root / "split.pkl"
            vocab = {
                "vocab_size": 10294,
                "face_index_size": 50,
                "se_codebook_size": 8192,
                "bbox_index_size": 2048,
                "special_tokens": {
                    "START_TOKEN": 10290,
                    "SEP_TOKEN": 10291,
                    "END_TOKEN": 10292,
                    "PAD_TOKEN": 10293,
                },
            }
            group = {"original": {"input_ids": [10290, 8242, 8243, 8244, 8245, 8246, 8247, 50, 51, 52, 53, 0, 10291, 10292]}}
            torch.save({"model_state_dict": {"w": torch.tensor([1.0])}, "fsq_levels": [8, 8, 8, 16]}, vqvae)
            torch.save(
                {
                    "model_state_dict": {"w": torch.tensor([2.0])},
                    "epoch": 120,
                    "best_val_ce": 0.2949,
                    "vocab_size": 10294,
                    "config": {"max_seq_len": 1024},
                },
                ar,
            )
            sequence.write_bytes(pickle.dumps({**vocab, "train": [group], "val": [group], "test": [group]}))
            split.write_bytes(pickle.dumps({"train": ["train.pkl"], "val": ["val.pkl"], "test": ["test.pkl"]}))

            report = evaluate_model_artifacts(vqvae_checkpoint=vqvae, ar_checkpoint=ar, sequence=sequence, split=split)

        self.assertEqual(report["status"], "MODEL_ARTIFACTS_READY")
        self.assertTrue(report["artifacts_ready"])
        self.assertEqual(report["vqvae_checkpoint"]["fsq_levels"], [8, 8, 8, 16])
        self.assertEqual(report["ar_checkpoint"]["epoch"], 120)
        self.assertEqual(report["sequence_package"]["split_counts"]["train"], 1)
        self.assertEqual(report["split_file"]["split_counts"]["test"], 1)
        self.assertEqual(report["blocking_reasons"], [])

    def test_model_artifact_sanity_blocks_bad_checkpoint_and_sequence_metadata(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import pickle
        import torch
        from verify_model_artifacts import evaluate_model_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vqvae = root / "bad_vqvae.pt"
            ar = root / "bad_ar.pt"
            sequence = root / "bad_sequence.pkl"
            split = root / "bad_split.pkl"
            torch.save({"fsq_levels": [8, 8, 8, 16]}, vqvae)
            torch.save({"epoch": 120}, ar)
            sequence.write_bytes(pickle.dumps({"train": [], "vocab_size": 0}))
            split.write_bytes(pickle.dumps({"train": []}))

            report = evaluate_model_artifacts(vqvae_checkpoint=vqvae, ar_checkpoint=ar, sequence=sequence, split=split)

        self.assertEqual(report["status"], "MODEL_ARTIFACTS_FAILED")
        self.assertFalse(report["artifacts_ready"])
        self.assertIn("vqvae_checkpoint_invalid", report["blocking_reasons"])
        self.assertIn("ar_checkpoint_invalid", report["blocking_reasons"])
        self.assertIn("sequence_package_invalid", report["blocking_reasons"])
        self.assertIn("split_file_invalid", report["blocking_reasons"])
        self.assertIn("missing_model_state_dict", report["vqvae_checkpoint"]["issues"])
        self.assertIn("missing_model_state_dict", report["ar_checkpoint"]["issues"])
        self.assertIn("missing_split:val", report["sequence_package"]["issues"])
        self.assertIn("missing_split:test", report["split_file"]["issues"])

    def test_server_recovery_plan_blocks_training_until_all_gates_pass(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from plan_server_quality_recovery import build_server_recovery_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer = root / "transfer.json"
            readiness = root / "readiness.json"
            artifacts = root / "artifacts.json"
            transfer.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")
            readiness.write_text(json.dumps({"status": "SERVER_TRAINING_READINESS_FAILED"}), encoding="utf-8")
            artifacts.write_text(json.dumps({"status": "MODEL_ARTIFACTS_READY"}), encoding="utf-8")

            plan = build_server_recovery_plan(
                transfer_verification=transfer,
                training_readiness=readiness,
                artifact_sanity=artifacts,
            )

        self.assertEqual(plan["status"], "HOLD_BEFORE_VQVAE_RECOVERY")
        self.assertFalse(plan["training_start_allowed"])
        self.assertIn("training_readiness_not_ready", plan["blocking_reasons"])
        self.assertEqual(plan["recovery_commands"], [])

    def test_server_recovery_plan_emits_vqvae_and_monitor_commands_when_ready(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from plan_server_quality_recovery import build_server_recovery_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer = root / "transfer.json"
            readiness = root / "readiness.json"
            artifacts = root / "artifacts.json"
            transfer.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")
            readiness.write_text(json.dumps({"status": "READY_FOR_VQVAE_TRAINING"}), encoding="utf-8")
            artifacts.write_text(json.dumps({"status": "MODEL_ARTIFACTS_READY"}), encoding="utf-8")

            plan = build_server_recovery_plan(
                transfer_verification=transfer,
                training_readiness=readiness,
                artifact_sanity=artifacts,
                benchmark_prefix="vq_complex_recovery_test",
            )

        self.assertEqual(plan["status"], "READY_TO_START_VQVAE_RECOVERY")
        self.assertTrue(plan["training_start_allowed"])
        joined = "\n".join(plan["recovery_commands"])
        self.assertIn("bash tools/run_vqvae_complex_recovery.sh", joined)
        self.assertIn("--benchmark-prefix vq_complex_recovery_test", joined)
        self.assertIn("tools/monitor_vqvae_recovery_gate.py", joined)

    def test_server_recovery_packet_prioritizes_guarded_vqvae_start_and_copyback(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from build_server_recovery_packet import build_server_recovery_packet, render_markdown

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir(parents=True)
            (reports / "v13_quality_recovery_stage_decision_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "NEEDS_VQVAE_RECOVERY",
                        "next_stage": "vqvae_complex_curved_recovery",
                        "can_train_ar_now": False,
                        "blocking_reasons": [
                            "generated_quality_not_promoted",
                            "vqvae_checkpoint_not_promoted",
                        ],
                        "paper_figure_policy": {"positive_figures_allowed": False},
                    }
                ),
                encoding="utf-8",
            )
            (reports / "v13_server_transfer_manifest_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "READY_TO_TRANSFER",
                        "required_sources_complete": True,
                        "entries": [
                            {"label": "a", "transfer_group": "repo_tooling"},
                            {"label": "b", "transfer_group": "repo_source"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reports / "v13_server_handoff_preflight_20260706.json").write_text(
                json.dumps({"status": "READY_FOR_SERVER_VQVAE_RECOVERY", "required_artifacts_complete": True}),
                encoding="utf-8",
            )

            packet = build_server_recovery_packet(root)
            markdown = render_markdown(packet)

        self.assertEqual(packet["status"], "READY_FOR_RENTED_SERVER_VQVAE_RECOVERY_PACKET")
        self.assertEqual(packet["stage_decision"]["next_stage"], "vqvae_complex_curved_recovery")
        self.assertFalse(packet["stage_decision"]["can_train_ar_now"])
        self.assertEqual(packet["transfer"]["entries_total"], 2)
        self.assertEqual(packet["transfer"]["repo_tooling_entries"], 1)
        immediate = packet["immediate_commands"]
        self.assertIn("local_refresh_training_phase_budget", immediate)
        self.assertIn("local_upload_transfer_manifest", immediate)
        self.assertIn("server_guarded_preflight", immediate)
        self.assertIn("server_progress_summary", immediate)
        self.assertIn("server_start_vqvae_recovery", immediate)
        self.assertIn("server_monitor_vqvae_recovery", immediate)
        self.assertIn("local_pull_vqvae_copyback", immediate)
        self.assertIn("local_verify_vqvae_copyback", immediate)
        self.assertEqual(immediate["local_upload_transfer_manifest"]["where"], "local_before_server")
        self.assertIn("build_server_transfer_manifest.py", immediate["local_upload_transfer_manifest"]["command"])
        self.assertIn("--powershell-upload-output", immediate["local_upload_transfer_manifest"]["command"])
        self.assertIn("v13_server_upload_from_manifest_20260706.ps1", immediate["local_upload_transfer_manifest"]["command"])
        self.assertEqual(immediate["local_refresh_training_phase_budget"]["where"], "local_before_server")
        self.assertIn("build_server_training_phase_budget.py", immediate["local_refresh_training_phase_budget"]["command"])
        self.assertIn("v13_server_training_phase_budget_20260706.json", immediate["local_refresh_training_phase_budget"]["command"])
        self.assertIn("v13_server_training_phase_budget_20260706.md", immediate["local_refresh_training_phase_budget"]["command"])
        self.assertIn("--min-gpu-memory-gb 40", immediate["server_guarded_preflight"]["command"])
        self.assertIn("summarize_quality_recovery_progress.py", immediate["server_progress_summary"]["command"])
        self.assertIn("v13_quality_recovery_progress_server.json", immediate["server_progress_summary"]["command"])
        self.assertIn("run_server_quality_recovery.py", immediate["server_start_vqvae_recovery"]["command"])
        self.assertIn("--min-gpu-memory-gb 40", immediate["server_start_vqvae_recovery"]["command"])
        self.assertIn("--start", immediate["server_start_vqvae_recovery"]["command"])
        self.assertNotIn("run_ar_v13_long_context.sh", immediate["server_start_vqvae_recovery"]["command"])
        self.assertEqual(immediate["local_pull_vqvae_copyback"]["where"], "local_after_monitor_success")
        self.assertIn("rsync -av --progress", immediate["local_pull_vqvae_copyback"]["command"])
        self.assertIn("newscheme_full_vqvae_complex_recovery", immediate["local_pull_vqvae_copyback"]["command"])
        self.assertIn("local_runs/reconstruction_eval", immediate["local_pull_vqvae_copyback"]["command"])
        self.assertIn("copy_back_manifest.json", immediate["local_pull_vqvae_copyback"]["command"])
        self.assertEqual(packet["deferred_commands"]["source_path_length_coverage"]["only_after"], "source_path_sequence_rebuild_finished")
        self.assertIn("summarize_ar_length_coverage.py", packet["deferred_commands"]["source_path_length_coverage"]["command"])
        self.assertEqual(packet["deferred_commands"]["verify_sequence_rebuild"]["only_after"], "source_path_length_coverage_finished")
        self.assertIn("verify_sequence_rebuild.py", packet["deferred_commands"]["verify_sequence_rebuild"]["command"])
        self.assertEqual(packet["deferred_commands"]["ar1536"]["only_after"], "READY_FOR_AR_LONG_CONTEXT")
        self.assertEqual(packet["deferred_commands"]["ar2048"]["only_after"], "ar1536_promising_or_memory_allows")
        self.assertIn("--max-seq-len 2048", packet["deferred_commands"]["ar2048"]["command"])
        self.assertEqual(packet["deferred_commands"]["generated_reconstruction_ar1536"]["only_after"], "ar1536_best_checkpoint_ready")
        self.assertIn("evaluate_reconstruction_v13.py", packet["deferred_commands"]["generated_reconstruction_ar1536"]["command"])
        self.assertEqual(packet["deferred_commands"]["step_geometry_entity_audit_ar1536"]["only_after"], "render_generated_ar1536_finished")
        self.assertIn("audit_step_geometry_entities.py", packet["deferred_commands"]["step_geometry_entity_audit_ar1536"]["command"])
        self.assertIn("summarize_generated_quality.py", packet["deferred_commands"]["generated_quality_gate_ar1536"]["command"])
        self.assertEqual(packet["deferred_commands"]["generated_quality_gate_ar1536"]["only_after"], "step_geometry_entity_audit_ar1536_finished")
        self.assertEqual(packet["deferred_commands"]["server_paper_figure_audit_ar1536"]["only_after"], "generated_quality_gate_ar1536_finished")
        self.assertEqual(packet["deferred_commands"]["server_paper_figure_audit_ar1536"]["where"], "server")
        self.assertIn("audit_paper_figure_candidates.py", packet["deferred_commands"]["server_paper_figure_audit_ar1536"]["command"])
        self.assertIn("paper_figure_candidate_audit.json", packet["deferred_commands"]["server_paper_figure_audit_ar1536"]["command"])
        self.assertEqual(packet["deferred_commands"]["local_pull_generated_ar1536"]["only_after"], "server_paper_figure_audit_ar1536_finished")
        self.assertEqual(packet["deferred_commands"]["local_pull_generated_ar1536"]["where"], "local_after_generated_gate")
        self.assertIn("rsync -av --progress", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("eval_generated100_ar1536_vqcomplex", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("generated_quality_summary.json", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("step_geometry_entity_audit.json", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("step_geometry_entity_audit.md", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("contact_sheet.png", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("paper_figure_candidate_audit.json", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertIn("paper_figure_candidate_audit.md", packet["deferred_commands"]["local_pull_generated_ar1536"]["command"])
        self.assertEqual(packet["deferred_commands"]["paper_figure_audit_ar1536"]["only_after"], "local_pull_generated_ar1536_finished")
        self.assertEqual(packet["deferred_commands"]["paper_figure_audit_ar1536"]["where"], "local_after_generated_pull")

    def test_server_recovery_packet_renders_safe_first_hour_bash_script(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from build_server_recovery_packet import build_server_recovery_packet, render_server_first_hour_script, write_lf_text

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir(parents=True)
            (reports / "v13_quality_recovery_stage_decision_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "NEEDS_VQVAE_RECOVERY",
                        "next_stage": "vqvae_complex_curved_recovery",
                        "can_train_ar_now": False,
                        "paper_figure_policy": {"positive_figures_allowed": False},
                    }
                ),
                encoding="utf-8",
            )
            (reports / "v13_server_transfer_manifest_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "READY_TO_TRANSFER",
                        "required_sources_complete": True,
                        "entries": [{"label": "tool", "transfer_group": "repo_tooling"}],
                    }
                ),
                encoding="utf-8",
            )
            (reports / "v13_server_handoff_preflight_20260706.json").write_text(
                json.dumps({"status": "READY_FOR_SERVER_VQVAE_RECOVERY", "required_artifacts_complete": True}),
                encoding="utf-8",
            )

            packet = build_server_recovery_packet(root)
            script = render_server_first_hour_script(packet)

        self.assertIn("#!/usr/bin/env bash", script)
        self.assertIn("set -euo pipefail", script)
        self.assertIn('case "${1:-preflight}" in', script)
        self.assertIn("preflight)", script)
        self.assertIn("tools/run_server_quality_recovery.py", script)
        self.assertIn("tools/summarize_quality_recovery_progress.py", script)
        self.assertIn("--min-gpu-memory-gb 40", script)
        self.assertIn("preflight_status=$?", script)
        self.assertIn("start_status=$?", script)
        self.assertIn("progress_status=$?", script)
        self.assertLess(script.index("preflight_status=$?"), script.index("Writing progress summary after preflight"))
        self.assertLess(script.index("start_status=$?"), script.index("Writing progress summary after start attempt"))
        self.assertIn("--start", script)
        self.assertIn("start)", script)
        self.assertIn("monitor)", script)
        self.assertIn("tools/monitor_vqvae_recovery_gate.py", script)
        self.assertLess(script.index("preflight)"), script.index("start)"))
        self.assertNotIn("run_ar_v13_long_context.sh", script)

        target = Path(tmp) / "local_reports" / "first_hour.sh"
        write_lf_text(target, "#!/usr/bin/env bash\r\necho ready\r\n")
        self.assertEqual(target.read_bytes(), b"#!/usr/bin/env bash\necho ready\n")

    def test_server_recovery_orchestrator_holds_when_a_gate_fails(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_server_quality_recovery import run_quality_recovery_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fake_runner(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index("--output") + 1])
                command_text = " ".join(command)
                if "verify_server_transfer.py" in command_text:
                    output.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")
                elif "verify_server_training_readiness.py" in command_text:
                    output.write_text(json.dumps({"status": "SERVER_TRAINING_READINESS_FAILED"}), encoding="utf-8")
                elif "verify_model_artifacts.py" in command_text:
                    output.write_text(json.dumps({"status": "MODEL_ARTIFACTS_READY"}), encoding="utf-8")
                else:
                    raise AssertionError(f"unexpected command: {command}")
                markdown = Path(command[command.index("--markdown-output") + 1])
                markdown.write_text("report", encoding="utf-8")
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            result = run_quality_recovery_preflight(
                repo_root=root,
                reports_dir=root / "local_reports",
                transfer_path_maps=["/workspace/V13=."],
                skip_data_requirements=True,
                start_training=True,
                command_runner=fake_runner,
            )

        self.assertEqual(result["plan"]["status"], "HOLD_BEFORE_VQVAE_RECOVERY")
        self.assertFalse(result["started_recovery"])
        self.assertEqual(result["executed_stages"], ["transfer_verification", "training_readiness", "artifact_sanity"])
        self.assertIn("--path-map", calls[0])
        self.assertIn("--skip-data-requirements", calls[0])
        self.assertFalse(any(isinstance(call, str) and "run_vqvae_complex_recovery" in call for call in calls))

    def test_server_recovery_orchestrator_maps_artifact_paths_for_local_dry_run(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_server_quality_recovery import run_quality_recovery_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fake_runner(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index("--output") + 1])
                command_text = " ".join(command)
                if "verify_server_transfer.py" in command_text:
                    output.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")
                elif "verify_server_training_readiness.py" in command_text:
                    output.write_text(json.dumps({"status": "SERVER_TRAINING_READINESS_FAILED"}), encoding="utf-8")
                elif "verify_model_artifacts.py" in command_text:
                    output.write_text(json.dumps({"status": "MODEL_ARTIFACTS_READY"}), encoding="utf-8")
                else:
                    raise AssertionError(f"unexpected command: {command}")
                markdown = Path(command[command.index("--markdown-output") + 1])
                markdown.write_text("report", encoding="utf-8")
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            run_quality_recovery_preflight(
                repo_root=root,
                reports_dir=root / "local_reports",
                transfer_path_maps=[
                    "/workspace/V13=.",
                    "/workspace/ABC/processed/train_outputs=ABC/processed/train_outputs",
                ],
                skip_data_requirements=True,
                min_gpu_memory_gb=48,
                command_runner=fake_runner,
            )

        readiness_command = calls[1]
        artifact_command = calls[2]
        self.assertIn("--min-gpu-memory-gb", readiness_command)
        self.assertEqual(readiness_command[readiness_command.index("--min-gpu-memory-gb") + 1], "48")
        self.assertIn("local_runs", readiness_command[readiness_command.index("--sequence") + 1])
        self.assertNotIn("/workspace/V13", artifact_command[artifact_command.index("--ar-checkpoint") + 1].replace("\\", "/"))
        self.assertIn("ABC/processed/train_outputs", artifact_command[artifact_command.index("--vqvae-checkpoint") + 1].replace("\\", "/"))

    def test_server_recovery_orchestrator_starts_recovery_only_when_ready(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_server_quality_recovery import run_quality_recovery_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fake_runner(command, **kwargs):
                calls.append(command)
                if isinstance(command, str):
                    return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
                output = Path(command[command.index("--output") + 1])
                command_text = " ".join(command)
                if "verify_server_transfer.py" in command_text:
                    output.write_text(json.dumps({"status": "READY_FOR_SERVER_RUN"}), encoding="utf-8")
                elif "verify_server_training_readiness.py" in command_text:
                    output.write_text(json.dumps({"status": "READY_FOR_VQVAE_TRAINING"}), encoding="utf-8")
                elif "verify_model_artifacts.py" in command_text:
                    output.write_text(json.dumps({"status": "MODEL_ARTIFACTS_READY"}), encoding="utf-8")
                else:
                    raise AssertionError(f"unexpected command: {command}")
                markdown = Path(command[command.index("--markdown-output") + 1])
                markdown.write_text("report", encoding="utf-8")
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            result = run_quality_recovery_preflight(
                repo_root=root,
                reports_dir=root / "local_reports",
                start_training=True,
                command_runner=fake_runner,
            )

        self.assertEqual(result["plan"]["status"], "READY_TO_START_VQVAE_RECOVERY")
        self.assertTrue(result["started_recovery"])
        self.assertTrue(any(isinstance(call, str) and "run_vqvae_complex_recovery.sh" in call for call in calls))

    def test_linux_complex_vqvae_recovery_can_run_post_training_benchmark(self):
        script = Path(r"D:\luolin\V13\tools\run_vqvae_complex_recovery.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn('--run-benchmark', text)
        self.assertIn('RUN_BENCHMARK="0"', text)
        self.assertIn('"$PYTHON" tools/run_vqvae_slice_benchmark.py', text)
        self.assertIn('--vqvae-checkpoint "$OUT_DIR/fsq_vqvae_best.pt"', text)
        self.assertIn('"$PYTHON" tools/write_vqvae_server_handoff.py', text)
        self.assertIn('copy_back_manifest.json', text)

    def test_generated_quality_summary_blocks_topology_collapse(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_generated_quality import summarize_generated_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            renders = run_dir / "renders"
            steps.mkdir()
            renders.mkdir()
            rows = []
            for idx in range(100):
                if idx < 70:
                    faces, edges = 6, 12
                elif idx < 90:
                    faces, edges = 4, 6
                else:
                    faces, edges = 14, 24
                step_path = steps / f"generated_{idx:06d}.step"
                step_path.write_text(f"ISO-10303-21; {idx}", encoding="utf-8")
                rows.append(
                    {
                        "source": "generated",
                        "index": idx,
                        "status": "saved",
                        "step_saved": True,
                        "brep_valid": True,
                        "grammar_ok": True,
                        "grammar_reason": "ok",
                        "grammar_faces": faces,
                        "grammar_edges": edges,
                        "step_path": str(step_path),
                    }
                )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps({"summary": {"attempted": 100, "step_saved": 100, "brep_valid": 100, "errors": 0}}),
                encoding="utf-8",
            )
            (renders / "contact_sheet.png").write_bytes(b"fake")

            summary = summarize_generated_run(run_dir)

        gate = summary["paper_gate"]
        self.assertEqual(gate["decision"], "hold_for_failure_analysis")
        self.assertFalse(gate["requirements"]["topology_not_collapsed"])
        self.assertAlmostEqual(summary["topology"]["top_two_fraction"], 0.9)
        self.assertIn("top two topology buckets dominate saved outputs", gate["reasons"])

    def test_generated_quality_summary_promotes_diverse_complex_run(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_generated_quality import summarize_generated_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            renders = run_dir / "renders"
            steps.mkdir()
            renders.mkdir()
            rows = []
            topologies = [
                (6, 12),
                (4, 6),
                (14, 24),
                (16, 28),
                (18, 32),
                (20, 36),
                (12, 22),
                (15, 30),
                (10, 24),
                (22, 40),
            ]
            for idx in range(100):
                faces, edges = topologies[idx % len(topologies)]
                step_path = steps / f"generated_{idx:06d}.step"
                step_path.write_text(f"ISO-10303-21; diverse {idx}", encoding="utf-8")
                rows.append(
                    {
                        "source": "generated",
                        "index": idx,
                        "status": "saved",
                        "step_saved": True,
                        "brep_valid": idx < 80,
                        "grammar_ok": True,
                        "grammar_reason": "ok",
                        "grammar_faces": faces,
                        "grammar_edges": edges,
                        "step_path": str(step_path),
                    }
                )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps({"summary": {"attempted": 100, "step_saved": 100, "brep_valid": 80, "errors": 0}}),
                encoding="utf-8",
            )
            (renders / "contact_sheet.png").write_bytes(b"fake")

            summary = summarize_generated_run(run_dir)

        gate = summary["paper_gate"]
        self.assertEqual(gate["decision"], "promote_as_paper_candidates")
        self.assertTrue(gate["requirements"]["complex_strict_valid_enough"])
        self.assertGreaterEqual(summary["complexity"]["strict_valid_complex"], 10)
        self.assertLessEqual(summary["topology"]["top_two_fraction"], 0.6)
        self.assertTrue(gate["requirements"]["nonprimitive_strict_valid_enough"])
        self.assertTrue(gate["requirements"]["primitive_fraction_not_dominant"])

    def test_generated_quality_summary_blocks_diverse_primitive_like_run(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_generated_quality import summarize_generated_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            renders = run_dir / "renders"
            steps.mkdir()
            renders.mkdir()
            rows = []
            topologies = [
                (2, 2),
                (4, 6),
                (6, 12),
                (8, 18),
                (9, 20),
                (10, 22),
                (11, 24),
                (12, 20),
                (12, 22),
                (12, 24),
            ]
            for idx in range(100):
                faces, edges = topologies[idx % len(topologies)]
                step_path = steps / f"generated_{idx:06d}.step"
                step_path.write_text(f"ISO-10303-21; primitive-like {idx}", encoding="utf-8")
                rows.append(
                    {
                        "source": "generated",
                        "index": idx,
                        "status": "saved",
                        "step_saved": True,
                        "brep_valid": True,
                        "grammar_ok": True,
                        "grammar_reason": "ok",
                        "grammar_faces": faces,
                        "grammar_edges": edges,
                        "step_path": str(step_path),
                    }
                )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps({"summary": {"attempted": 100, "step_saved": 100, "brep_valid": 100, "errors": 0}}),
                encoding="utf-8",
            )
            (renders / "contact_sheet.png").write_bytes(b"fake")

            summary = summarize_generated_run(run_dir)

        gate = summary["paper_gate"]
        self.assertEqual(gate["decision"], "hold_for_failure_analysis")
        self.assertTrue(gate["requirements"]["topology_not_collapsed"])
        self.assertTrue(gate["requirements"]["complex_strict_valid_enough"])
        self.assertFalse(gate["requirements"]["nonprimitive_strict_valid_enough"])
        self.assertFalse(gate["requirements"]["primitive_fraction_not_dominant"])
        self.assertIn("too few non-primitive strict-valid generated outputs", gate["reasons"])
        self.assertIn("primitive-like strict-valid outputs dominate the retained set", gate["reasons"])

    def test_step_geometry_entity_audit_counts_curved_and_closed_solids(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from audit_step_geometry_entities import audit_step_geometry_entities

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            steps.mkdir()
            planar = steps / "planar_box.step"
            curved = steps / "curved_cylinder.step"
            planar.write_text(
                "\n".join(
                    [
                        "ISO-10303-21;",
                        "#1 = MANIFOLD_SOLID_BREP('',#2);",
                        "#2 = CLOSED_SHELL('',(#3));",
                        "#3 = ADVANCED_FACE('',(),#4,.T.);",
                        "#4 = PLANE('',#5);",
                        "ENDSEC;",
                    ]
                ),
                encoding="utf-8",
            )
            curved.write_text(
                "\n".join(
                    [
                        "ISO-10303-21;",
                        "#1 = MANIFOLD_SOLID_BREP('',#2);",
                        "#2 = CLOSED_SHELL('',(#3));",
                        "#3 = ADVANCED_FACE('',(),#4,.T.);",
                        "#4 = CYLINDRICAL_SURFACE('',#5,1.0);",
                        "#5 = CIRCLE('',#6,1.0);",
                        "ENDSEC;",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "index": 0,
                                "step_saved": True,
                                "brep_valid": True,
                                "grammar_faces": 6,
                                "grammar_edges": 12,
                                "step_path": str(planar),
                            }
                        ),
                        json.dumps(
                            {
                                "index": 1,
                                "step_saved": True,
                                "brep_valid": True,
                                "grammar_faces": 13,
                                "grammar_edges": 28,
                                "step_path": str(curved),
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            audit = audit_step_geometry_entities(run_dir)

            self.assertEqual(audit["summary"]["step_files_audited"], 2)
            self.assertEqual(audit["summary"]["files_with_nonplanar_surfaces"], 1)
            self.assertEqual(audit["summary"]["files_solid_closed_no_open_shell"], 2)
            self.assertEqual(audit["summary"]["strict_valid_with_nonplanar_surfaces"], 1)
            self.assertEqual(audit["summary"]["files_with_at_least_12_advanced_faces"], 0)
            self.assertEqual(audit["summary"]["strict_valid_with_at_least_12_advanced_faces"], 0)
            self.assertEqual(audit["summary"]["mean_advanced_faces_per_file"], 1.0)
            self.assertEqual(audit["entity_totals"]["CYLINDRICAL_SURFACE"], 1)
            self.assertEqual(audit["entity_totals"]["PLANE"], 1)

    def test_paper_figure_candidate_audit_blocks_held_generated_run(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from audit_paper_figure_candidates import audit_paper_figure_candidates
        from summarize_generated_quality import summarize_generated_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            renders = run_dir / "renders"
            steps.mkdir()
            renders.mkdir()
            rows = []
            for idx in range(100):
                faces, edges = (6, 12) if idx < 80 else (14, 24)
                step_path = steps / f"generated_{idx:06d}.step"
                step_path.write_text(f"ISO-10303-21; collapse {idx}", encoding="utf-8")
                rows.append(
                    {
                        "source": "generated",
                        "index": idx,
                        "status": "saved",
                        "step_saved": True,
                        "brep_valid": True,
                        "grammar_ok": True,
                        "grammar_reason": "ok",
                        "grammar_faces": faces,
                        "grammar_edges": edges,
                        "step_path": str(step_path),
                    }
                )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps({"summary": {"attempted": 100, "step_saved": 100, "brep_valid": 100, "errors": 0}}),
                encoding="utf-8",
            )
            (renders / "contact_sheet.png").write_bytes(b"fake")
            summary = summarize_generated_run(run_dir)
            (run_dir / "generated_quality_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            audit = audit_paper_figure_candidates(run_dir)

        self.assertEqual(audit["decision"], "hold_for_failure_analysis")
        self.assertEqual(audit["paper_role"], "failure_analysis_only")
        self.assertFalse(audit["ready_for_human_review"])
        self.assertIn("generated quality gate is not promoted", audit["blocking_reasons"])

    def test_paper_figure_candidate_audit_requires_artifacts_before_human_review(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from audit_paper_figure_candidates import audit_paper_figure_candidates
        from summarize_generated_quality import summarize_generated_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            steps = run_dir / "steps"
            renders = run_dir / "renders"
            steps.mkdir()
            renders.mkdir()
            rows = []
            topologies = [
                (6, 12),
                (4, 6),
                (14, 24),
                (16, 28),
                (18, 32),
                (20, 36),
                (12, 22),
                (15, 30),
                (10, 24),
                (22, 40),
            ]
            for idx in range(100):
                faces, edges = topologies[idx % len(topologies)]
                step_path = steps / f"generated_{idx:06d}.step"
                step_path.write_text(f"ISO-10303-21; paper candidate {idx}", encoding="utf-8")
                rows.append(
                    {
                        "source": "generated",
                        "index": idx,
                        "status": "saved",
                        "step_saved": True,
                        "brep_valid": idx < 80,
                        "grammar_ok": True,
                        "grammar_reason": "ok",
                        "grammar_faces": faces,
                        "grammar_edges": edges,
                        "step_path": str(step_path),
                    }
                )
            (run_dir / "reconstruction_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run_dir / "reconstruction_report.json").write_text(
                json.dumps({"summary": {"attempted": 100, "step_saved": 100, "brep_valid": 80, "errors": 0}}),
                encoding="utf-8",
            )
            (renders / "contact_sheet.png").write_bytes(b"fake")
            summary = summarize_generated_run(run_dir)
            (run_dir / "generated_quality_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            audit = audit_paper_figure_candidates(run_dir)
            (renders / "contact_sheet.png").unlink()
            missing_render_audit = audit_paper_figure_candidates(run_dir)

        self.assertEqual(audit["decision"], "ready_for_paper_figure_review")
        self.assertEqual(audit["paper_role"], "positive_candidate_pending_human_review")
        self.assertTrue(audit["ready_for_human_review"])
        self.assertTrue(audit["human_review_required"])
        self.assertTrue(audit["artifacts"]["contact_sheet"]["ok"])
        self.assertTrue(audit["artifacts"]["step_files"]["ok"])
        self.assertEqual(missing_render_audit["decision"], "hold_for_failure_analysis")
        self.assertIn("rendered contact sheet is missing", missing_render_audit["blocking_reasons"])

    def test_reconstruction_tool_records_sampling_controls_and_random_seed(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import build_sampling_config, resolve_seed

        self.assertEqual(resolve_seed(123), 123)
        self.assertEqual(resolve_seed(-1, entropy=lambda: 987654321), 987654321)

        class Args:
            seed = -1
            temperature = 0.9
            top_p = 0.95
            max_new_tokens = 400
            max_samples = 5

        config = build_sampling_config(Args(), effective_seed=987654321)

        self.assertEqual(
            config,
            {
                "requested_seed": -1,
                "effective_seed": 987654321,
                "temperature": 0.9,
                "top_p": 0.95,
                "max_new_tokens": 400,
                "max_samples": 5,
            },
        )

    def test_reconstruction_tool_defaults_to_fresh_random_seed(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import evaluate_reconstruction_v13

        original_argv = sys.argv
        try:
            sys.argv = ["evaluate_reconstruction_v13.py"]
            args = evaluate_reconstruction_v13.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.seed, -1)

    def test_reconstruction_tool_validates_sequence_grammar_before_reconstructing(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from evaluate_reconstruction_v13 import grammar_validation

        vocab = {
            "face_index_size": 50,
            "se_codebook_size": 8192,
            "bbox_index_size": 2048,
            "bbox_token_offset": 8242,
            "se_token_offset": 50,
            "START_TOKEN": 10290,
            "SEP_TOKEN": 10291,
            "END_TOKEN": 10292,
            "PAD_TOKEN": 10293,
        }
        valid = [10290] + [8242] * 6 + [50] * 4 + [0] + [10291] + [10292]
        invalid = [10290, 10291, 10292]

        self.assertEqual(grammar_validation(valid, vocab)["reason"], "ok")
        self.assertFalse(grammar_validation(invalid, vocab)["ok"])

    def test_reconstruction_tool_uses_fsq_loader_instead_of_original_vq_loader(self):
        script = REPO_ROOT / "tools" / "evaluate_reconstruction_v13.py"
        text = script.read_text(encoding="utf-8")

        self.assertIn("build_fsq_vqvae", text)
        self.assertIn("FSQ-aware", text)
        self.assertIn("indices_to_codes", text)
        self.assertIn("proj_out", text)
        self.assertIn("cpu_safe_joint_optimize", text)
        self.assertIn("patch_reconstruction_joint_optimize", text)
        self.assertIn("torch.cdist", text)
        self.assertIn("TopologyConstrainedLogitsProcessor", text)
        self.assertIn("grammar_validation", text)
        self.assertNotIn("load_se_vqvae_model(", text)

    def test_ar_stage_status_accepts_finite_resumed_best_even_if_final_train_ce_plateaus(self):
        sys.path.insert(0, str(IMPROVEMENTS_DIR))
        import train

        meta = {
            "ce_init": 0.3012,
            "ce_final": 0.3018,
            "best_val_ce": 0.2951,
            "epochs_ran": 24,
            "end_epoch": 100,
        }

        self.assertTrue(train.ar_stage_verified(meta))

    def test_quality_recovery_progress_holds_on_local_preflight(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_quality_recovery_progress import summarize_quality_recovery_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            preflight = reports / "preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "status": "HOLD_BEFORE_VQVAE_RECOVERY",
                        "started_recovery": False,
                        "plan": {
                            "gates": {
                                "transfer_verification": {"ready": True},
                                "training_readiness": {"ready": False},
                                "artifact_sanity": {"ready": True},
                            },
                            "blocking_reasons": ["training_readiness_not_ready"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_quality_recovery_progress(repo_root=root, preflight=preflight)

        self.assertEqual(report["status"], "HOLD_BEFORE_VQVAE_RECOVERY")
        self.assertEqual(report["current_stage"], "server_preflight")
        self.assertEqual(report["next_action"], "fix_server_preflight_before_training")
        self.assertFalse(report["can_train_ar_now"])
        self.assertFalse(report["positive_figures_allowed"])
        self.assertIn("training_readiness_not_ready", report["blocking_reasons"])

    def test_quality_recovery_progress_requires_copyback_after_vqvae_promotion(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_quality_recovery_progress import summarize_quality_recovery_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            preflight = reports / "preflight.json"
            monitor = reports / "monitor.json"
            preflight.write_text(
                json.dumps({"status": "READY_TO_START_VQVAE_RECOVERY", "started_recovery": True}),
                encoding="utf-8",
            )
            monitor.write_text(
                json.dumps(
                    {
                        "state": "ready_for_sequence_rebuild",
                        "ready": True,
                        "terminal": True,
                        "exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_quality_recovery_progress(
                repo_root=root,
                preflight=preflight,
                vqvae_monitor=monitor,
            )

        self.assertEqual(report["status"], "WAITING_FOR_VQVAE_COPYBACK")
        self.assertEqual(report["current_stage"], "vqvae_copyback")
        self.assertEqual(report["next_action"], "pull_and_verify_vqvae_copyback")
        self.assertFalse(report["can_train_ar_now"])

    def test_quality_recovery_progress_holds_after_terminal_vqvae_failure(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_quality_recovery_progress import summarize_quality_recovery_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            preflight = reports / "preflight.json"
            monitor = reports / "monitor.json"
            preflight.write_text(
                json.dumps({"status": "READY_TO_START_VQVAE_RECOVERY", "started_recovery": True}),
                encoding="utf-8",
            )
            monitor.write_text(
                json.dumps(
                    {
                        "state": "hold_vqvae_checkpoint",
                        "ready": False,
                        "terminal": True,
                        "exit_code": 2,
                        "reason": "longest slice did not improve",
                        "benchmark": {
                            "reasons": [
                                "longest_strict_valid_not_improved",
                                "most_faces_contact_sheet_needs_review",
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_quality_recovery_progress(
                repo_root=root,
                preflight=preflight,
                vqvae_monitor=monitor,
            )

        self.assertEqual(report["status"], "HOLD_AFTER_VQVAE_RECOVERY")
        self.assertEqual(report["current_stage"], "vqvae_complex_curved_recovery")
        self.assertEqual(report["next_action"], "inspect_vqvae_recovery_failure_before_sequence_rebuild")
        self.assertFalse(report["can_train_ar_now"])
        self.assertFalse(report["positive_figures_allowed"])
        self.assertIn("hold_vqvae_checkpoint", report["blocking_reasons"])
        self.assertIn("longest slice did not improve", report["blocking_reasons"])
        self.assertIn("longest_strict_valid_not_improved", report["blocking_reasons"])

    def test_quality_recovery_progress_reaches_human_review_after_all_gates(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_quality_recovery_progress import summarize_quality_recovery_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            preflight = reports / "preflight.json"
            monitor = reports / "monitor.json"
            copyback = reports / "copyback.json"
            sequence = reports / "sequence.json"
            generated = reports / "generated_quality_summary.json"
            paper = reports / "paper_figure_candidate_audit.json"
            human = reports / "human_visual_review.json"
            preflight.write_text(
                json.dumps({"status": "READY_TO_START_VQVAE_RECOVERY", "started_recovery": True}),
                encoding="utf-8",
            )
            monitor.write_text(
                json.dumps({"state": "ready_for_sequence_rebuild", "ready": True, "exit_code": 0}),
                encoding="utf-8",
            )
            copyback.write_text(
                json.dumps({"status": "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD", "copyback_ready": True}),
                encoding="utf-8",
            )
            sequence.write_text(
                json.dumps({"status": "READY_FOR_AR_LONG_CONTEXT", "sequence_rebuild_ready": True}),
                encoding="utf-8",
            )
            generated.write_text(
                json.dumps({"paper_gate": {"decision": "promote_as_paper_candidates", "promote": True}}),
                encoding="utf-8",
            )
            paper.write_text(
                json.dumps(
                    {
                        "decision": "ready_for_paper_figure_review",
                        "ready_for_human_review": True,
                    }
                ),
                encoding="utf-8",
            )
            human.write_text(
                json.dumps(
                    {
                        "decision": "approved_for_paper_figures",
                        "approved": True,
                        "reviewer": "human",
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_quality_recovery_progress(
                repo_root=root,
                preflight=preflight,
                vqvae_monitor=monitor,
                copyback_verify=copyback,
                sequence_verify=sequence,
                generated_quality=generated,
                paper_audit=paper,
                human_review=human,
            )

        self.assertEqual(report["status"], "READY_FOR_PAPER_FIGURE_REPLACEMENT")
        self.assertEqual(report["current_stage"], "human_visual_review")
        self.assertEqual(report["next_action"], "replace_diagnostic_figures_with_approved_generated_outputs")
        self.assertTrue(report["can_train_ar_now"])
        self.assertTrue(report["positive_figures_allowed"])

    def test_quality_recovery_progress_requires_human_review_approval_before_positive_figures(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from summarize_quality_recovery_progress import summarize_quality_recovery_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            preflight = reports / "preflight.json"
            monitor = reports / "monitor.json"
            copyback = reports / "copyback.json"
            sequence = reports / "sequence.json"
            generated = reports / "generated_quality_summary.json"
            paper = reports / "paper_figure_candidate_audit.json"
            preflight.write_text(
                json.dumps({"status": "READY_TO_START_VQVAE_RECOVERY", "started_recovery": True}),
                encoding="utf-8",
            )
            monitor.write_text(
                json.dumps({"state": "ready_for_sequence_rebuild", "ready": True, "exit_code": 0}),
                encoding="utf-8",
            )
            copyback.write_text(
                json.dumps({"status": "READY_FOR_SOURCE_PATH_SEQUENCE_REBUILD", "copyback_ready": True}),
                encoding="utf-8",
            )
            sequence.write_text(
                json.dumps({"status": "READY_FOR_AR_LONG_CONTEXT", "sequence_rebuild_ready": True}),
                encoding="utf-8",
            )
            generated.write_text(
                json.dumps({"paper_gate": {"decision": "promote_as_paper_candidates", "promote": True}}),
                encoding="utf-8",
            )
            paper.write_text(
                json.dumps(
                    {
                        "decision": "ready_for_paper_figure_review",
                        "ready_for_human_review": True,
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_quality_recovery_progress(
                repo_root=root,
                preflight=preflight,
                vqvae_monitor=monitor,
                copyback_verify=copyback,
                sequence_verify=sequence,
                generated_quality=generated,
                paper_audit=paper,
            )

        self.assertEqual(report["status"], "WAITING_FOR_HUMAN_VISUAL_REVIEW")
        self.assertEqual(report["current_stage"], "human_visual_review")
        self.assertEqual(report["next_action"], "record_human_visual_review_decision")
        self.assertTrue(report["can_train_ar_now"])
        self.assertFalse(report["positive_figures_allowed"])
        self.assertIn("missing_or_unapproved_human_visual_review", report["blocking_reasons"])

    def test_run_sharded_sequence_metadata_records_ordering(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from run_sharded_sequence import metadata_from_preprocessor

        class Pre:
            vocab_size = 10294
            special_token_size = 4
            face_index_size = 50
            se_codebook_size = 8192
            bbox_index_size = 2048
            face_index_offset = 0
            se_token_offset = 50
            bbox_token_offset = 8242
            se_tokens_per_element = 4
            bbox_tokens_per_element = 6
            START_TOKEN = 10290
            SEP_TOKEN = 10291
            END_TOKEN = 10292
            PAD_TOKEN = 10293

        self.assertEqual(metadata_from_preprocessor(Pre(), ordering="dfs")["ordering"], "DFS")
        self.assertEqual(metadata_from_preprocessor(Pre(), ordering="rcm")["ordering"], "RCM")

    def test_sequence_sharding_rejects_mixed_ordering_shards(self):
        sys.path.insert(0, str(REPO_ROOT / "breparg_improvements"))
        import pickle
        from sequence_sharding import merge_sequence_shards

        def package(ordering):
            return {
                "train": [],
                "val": [],
                "test": [],
                "vocab_size": 10294,
                "special_token_size": 4,
                "face_index_size": 50,
                "se_codebook_size": 8192,
                "bbox_index_size": 2048,
                "face_index_offset": 0,
                "se_token_offset": 50,
                "bbox_token_offset": 8242,
                "se_tokens_per_element": 4,
                "bbox_tokens_per_element": 6,
                "special_tokens": {
                    "START_TOKEN": 10290,
                    "SEP_TOKEN": 10291,
                    "END_TOKEN": 10292,
                    "PAD_TOKEN": 10293,
                },
                "ordering": ordering,
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rcm = root / "abc_0000.pkl"
            dfs = root / "abc_0001.pkl"
            with rcm.open("wb") as handle:
                pickle.dump(package("RCM"), handle)
            with dfs.open("wb") as handle:
                pickle.dump(package("DFS"), handle)

            with self.assertRaises(ValueError):
                merge_sequence_shards([rcm, dfs], root / "merged.pkl")

    def test_server_training_phase_budget_prioritizes_vqvae_before_ar(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from build_server_training_phase_budget import build_training_phase_budget, render_markdown

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "local_reports"
            reports.mkdir()
            (reports / "v13_quality_recovery_progress_localdryrun_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "HOLD_BEFORE_VQVAE_RECOVERY",
                        "current_stage": "server_preflight",
                        "next_action": "fix_server_preflight_before_training",
                        "can_train_ar_now": False,
                        "positive_figures_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            (reports / "v13_quality_recovery_stage_decision_20260706.json").write_text(
                json.dumps(
                    {
                        "status": "NEEDS_VQVAE_RECOVERY",
                        "next_stage": "vqvae_complex_curved_recovery",
                        "can_train_ar_now": False,
                        "paper_figure_policy": {"positive_figures_allowed": False},
                    }
                ),
                encoding="utf-8",
            )

            budget = build_training_phase_budget(root)
            markdown = render_markdown(budget)

        self.assertEqual(budget["status"], "READY_FOR_VQVAE_FIRST_SERVER_PLAN")
        self.assertEqual(budget["recommended_first_gpu"]["minimum_gpu_memory_gb"], 40)
        self.assertIn("L40S 48GB", budget["recommended_first_gpu"]["preferred"])
        phase_by_id = {phase["id"]: phase for phase in budget["phases"]}
        self.assertEqual(phase_by_id["vqvae_complex_curved_recovery"]["gate"], "run_next")
        self.assertEqual(phase_by_id["vqvae_complex_curved_recovery"]["recommended_gpu"], "48GB_single_gpu")
        self.assertIn("run_vqvae_complex_recovery.sh", phase_by_id["vqvae_complex_curved_recovery"]["command"])
        self.assertEqual(phase_by_id["source_path_sequence_rebuild"]["gate"], "blocked_until_vqvae_promoted")
        self.assertEqual(phase_by_id["ar1536_long_context"]["gate"], "blocked_until_sequence_ready")
        self.assertEqual(phase_by_id["ar2048_optional"]["gate"], "blocked_until_ar1536_or_memory_need")
        self.assertFalse(budget["paper_policy"]["positive_figures_allowed"])
        hardware_by_id = {tier["id"]: tier for tier in budget["hardware_tiers"]}
        self.assertEqual(hardware_by_id["local_rtx3060_12gb"]["decision"], "debug_only")
        self.assertEqual(hardware_by_id["rtx4090_24gb"]["decision"], "smoke_test_only")
        self.assertEqual(hardware_by_id["l40s_48gb"]["decision"], "first_choice")
        self.assertEqual(hardware_by_id["rtx6000_ada_48gb"]["decision"], "first_choice")
        self.assertEqual(hardware_by_id["a100_80gb"]["decision"], "upgrade_for_ar2048_or_larger_batches")
        self.assertEqual(hardware_by_id["h100_80gb"]["speed_tier"], "fastest_deadline")
        self.assertIn("Do not train AR before VQ-VAE promotion", markdown)
        self.assertIn("1x L40S 48GB", markdown)
        self.assertIn("## Hardware Tiers", markdown)
        self.assertIn("| L40S 48GB | 48 | first_choice | balanced_fast |", markdown)
        self.assertIn("| RTX 4090 | 24 | smoke_test_only | consumer_fast |", markdown)


if __name__ == "__main__":
    unittest.main()
