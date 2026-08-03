import tempfile
import unittest
from pathlib import Path
import importlib
import os
import sys

import numpy as np


class VqvaeSampleCacheTests(unittest.TestCase):
    def test_round_trips_samples_weights_and_summary(self):
        from breparg_improvements.vqvae_sample_cache import load_vqvae_sample_cache, save_vqvae_sample_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "samples.npz"
            samples = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
            weights = np.array([1.0, 2.5], dtype=np.float32)
            summary = {"selected": 2, "source": "unit_test"}

            save_vqvae_sample_cache(cache, samples, weights, summary)
            loaded_samples, loaded_weights, loaded_summary = load_vqvae_sample_cache(cache, min_samples=2)

        np.testing.assert_array_equal(loaded_samples, samples)
        np.testing.assert_array_equal(loaded_weights, weights)
        self.assertEqual(loaded_summary["selected"], 2)
        self.assertEqual(loaded_summary["source"], "unit_test")
        self.assertEqual(loaded_summary["cache_path"], str(cache))
        self.assertEqual(loaded_summary["cache_samples"], 2)

    def test_rejects_cache_with_too_few_samples(self):
        from breparg_improvements.vqvae_sample_cache import load_vqvae_sample_cache, save_vqvae_sample_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "samples.npz"
            samples = np.zeros((1, 3, 4, 4), dtype=np.float32)
            weights = np.ones((1,), dtype=np.float32)
            save_vqvae_sample_cache(cache, samples, weights, {"selected": 1})

            with self.assertRaisesRegex(ValueError, "fewer samples"):
                load_vqvae_sample_cache(cache, min_samples=2)

    def test_train_collect_se_can_load_sample_cache(self):
        from breparg_improvements.vqvae_sample_cache import save_vqvae_sample_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "samples.npz"
            samples = np.arange(3 * 3 * 4 * 4, dtype=np.float32).reshape(3, 3, 4, 4)
            weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            save_vqvae_sample_cache(cache, samples, weights, {"selected": 3})

            old_env = os.environ.copy()
            sys.path.insert(0, str(Path("breparg_improvements").resolve()))
            try:
                os.environ["NS_VQ_SAMPLE_CACHE"] = str(cache)
                os.environ["NS_OUTBASE"] = str(Path(tmp) / "out")
                os.environ["NS_OUT"] = "cache_test"
                sys.modules.pop("train", None)
                train = importlib.import_module("train")

                loaded_samples, loaded_weights = train.collect_se([], 2, return_weights=True)
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                sys.modules.pop("train", None)

        np.testing.assert_array_equal(loaded_samples, samples[:2])
        np.testing.assert_array_equal(loaded_weights, weights[:2])


if __name__ == "__main__":
    unittest.main()
