import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class BuildVqvaeSampleCacheTests(unittest.TestCase):
    def test_builds_cache_from_patch_shards_and_writes_summary(self):
        from breparg_improvements.sharded_data import dump_shard_record
        from tools.build_vqvae_sample_cache import build_sample_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "vq_patch_shard_0000.pkl"
            with shard.open("wb") as handle:
                dump_shard_record(
                    handle,
                    {
                        "record_type": "vq_patch_shard_header",
                        "format": "v13.vq_patch_shard.v1",
                    },
                )
                for index in range(4):
                    dump_shard_record(
                        handle,
                        {
                            "record_type": "vq_patch",
                            "record_id": f"shape:{index}",
                            "source_path": f"abc_0000/shape_{index}.pkl",
                            "kind": "surface",
                            "array": np.full((32, 32, 3), index, dtype=np.float32),
                            "curvature_score": 0.05 if index % 2 else 0.0,
                            "n_faces": 20 if index >= 2 else 2,
                            "n_edges": 30 if index >= 2 else 2,
                            "is_complex_source": index >= 2,
                        },
                    )

            cache = root / "cache" / "samples.npz"
            summary_path = root / "summary.json"

            summary = build_sample_cache(
                patch_shard_root=root,
                output=cache,
                summary_output=summary_path,
                samples=3,
                seed=0,
                complex_fraction=0.5,
                complex_min_faces=12,
                complex_min_edges=20,
                curved_fraction=0.5,
                max_source_faces=50,
                max_source_edges=150,
                complex_loss_weight=1.25,
                curved_loss_weight=2.0,
                curved_loss_threshold=0.02,
            )

            self.assertTrue(cache.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["status"], "BUILT")
            self.assertEqual(summary["cache"]["samples"], 3)
            self.assertEqual(summary["sampling"]["selected"], 3)
            self.assertGreaterEqual(summary["sampling"]["complex_records_selected"], 1)
            written = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written["cache"]["path"], str(cache))


if __name__ == "__main__":
    unittest.main()
