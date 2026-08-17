"""Pure-python regression tests for the 2026-08-17 hardening batch.

These tests intentionally avoid numpy/torch so they can run on any box
(including the review environment) and act as fast CI smoke for the
fail-closed behaviours added by fix/full-hardening-20260817.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "breparg_improvements", REPO_ROOT / "tools"):
    p = str(extra)
    if p not in sys.path:
        sys.path.insert(0, p)

import sharded_data  # noqa: E402
import sequence_sharding  # noqa: E402
from summarize_assembly_calibration import summarize_calibration  # noqa: E402


def _cal_row(arm, cad, sha, valid=True, mse=1e-4, status="ok"):
    return {
        "arm": arm,
        "cad_id": cad,
        "checkpoint_sha256": sha,
        "brep_valid": valid,
        "step_saved": True,
        "curved_mse": mse,
        "status": status,
    }


class ShardedDataHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_gzip_roundtrip(self):
        path = self.tmp / "s.pkl.gz"
        with sharded_data.open_shard_writer(path) as handle:
            for i in range(5):
                pickle.dump({"i": i}, handle)
        rows = list(sharded_data.iter_shard_records(path))
        self.assertEqual([r["i"] for r in rows], list(range(5)))

    def test_truncated_gzip_shard_raises(self):
        path = self.tmp / "s.pkl.gz"
        with sharded_data.open_shard_writer(path) as handle:
            for i in range(50):
                pickle.dump({"i": i, "pad": "x" * 64}, handle)
        blob = path.read_bytes()
        truncated = self.tmp / "t.pkl.gz"
        truncated.write_bytes(blob[: int(len(blob) * 0.6)])
        with self.assertRaisesRegex(ValueError, "truncated"):
            list(sharded_data.iter_shard_records(truncated))

    def test_writer_is_atomic_on_failure(self):
        path = self.tmp / "s.pkl.gz"
        with self.assertRaises(RuntimeError):
            with sharded_data.open_shard_writer(path) as handle:
                pickle.dump({"x": 1}, handle)
                raise RuntimeError("boom")
        self.assertFalse(path.exists())
        self.assertEqual(list(self.tmp.glob("s.pkl.gz.tmp*")), [])


class SequenceShardingHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.package = {
            "train": [{"original": {"input_ids": [1, 2]}}],
            "val": [],
            "test": [],
            "vocab_size": 10,
            "special_tokens": {"PAD_TOKEN": 9},
        }

    def test_write_package_lands_on_final_name(self):
        path = self.tmp / "shard_a.pkl"
        sequence_sharding.write_sequence_package(path, self.package)
        self.assertTrue(path.exists())
        self.assertEqual(list(self.tmp.glob("shard_a.pkl.tmp*")), [])

    def test_duplicate_shards_rejected(self):
        path = self.tmp / "shard_a.pkl"
        sequence_sharding.write_sequence_package(path, self.package)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sequence_sharding.merge_sequence_shards([path, path], self.tmp / "m.pkl")

    def test_legacy_shards_default_missing_ordering_to_rcm(self):
        metadata = {
            "vocab_size": 10,
            "special_token_size": 4,
            "face_index_size": 2,
            "se_codebook_size": 2,
            "bbox_index_size": 2,
            "face_index_offset": 0,
            "se_token_offset": 2,
            "bbox_token_offset": 4,
            "se_tokens_per_element": 4,
            "bbox_tokens_per_element": 6,
            "special_tokens": {"PAD_TOKEN": 9},
        }
        shard_a = self.tmp / "abc_0000.pkl"
        shard_b = self.tmp / "abc_0001.pkl"
        sequence_sharding.write_sequence_package(
            shard_a,
            {**metadata, "train": [{"original": {"input_ids": [1]}}], "val": [], "test": []},
        )
        sequence_sharding.write_sequence_package(
            shard_b,
            {**metadata, "train": [], "val": [], "test": [{"original": {"input_ids": [2]}}]},
        )

        output = self.tmp / "merged.pkl"
        summary = sequence_sharding.merge_sequence_shards([shard_a, shard_b], output)

        self.assertEqual(summary["ordering"], "RCM")
        self.assertEqual(sequence_sharding.load_sequence_package(output)["ordering"], "RCM")


class CalibrationSummaryHardeningTests(unittest.TestCase):
    def test_latest_attempt_wins_per_arm_cad(self):
        rows = [
            _cal_row("armA", "c1", "s1"),
            _cal_row("armA", "c2", "s1"),
            _cal_row("armA", "c1", "s1", valid=False),
        ]
        summary = summarize_calibration(rows, min_cads=1)
        self.assertEqual(summary["arms"]["armA"]["attempts"], 2)
        self.assertEqual(summary["arms"]["armA"]["brep_valid"], 1)

    def test_mixed_checkpoints_within_arm_rejected(self):
        rows = [_cal_row("armA", "c1", "sha_new"), _cal_row("armA", "c3", "sha_old")]
        with self.assertRaisesRegex(RuntimeError, "mixes rows"):
            summarize_calibration(rows, min_cads=1)

    def test_none_checkpoint_sha_tolerated(self):
        rows = [_cal_row("original", "c1", None), _cal_row("armA", "c1", "sha")]
        summary = summarize_calibration(rows, min_cads=1)
        self.assertIn("original", summary["arms"])

    def test_all_none_curved_mse_does_not_crash(self):
        rows = [_cal_row("armB", f"c{i}", "s", mse=None) for i in range(5)]
        summary = summarize_calibration(rows, min_cads=1)
        self.assertIsNone(
            summary["arms"]["armB"]["empirical_curved_mse_gate_at_80pct_valid"]
        )

    def test_association_none_is_insufficient_evidence(self):
        rows = [
            _cal_row("continuous_bypass_64d", f"c{i}", "s", valid=(i % 3 != 0), mse=None)
            for i in range(10)
        ]
        summary = summarize_calibration(rows, min_cads=10)
        self.assertEqual(summary["decision"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_strong_association_path_intact(self):
        rows = (
            [_cal_row("continuous_bypass_64d", f"v{i}", "s", valid=True, mse=1e-5) for i in range(60)]
            + [_cal_row("continuous_bypass_64d", f"i{i}", "s", valid=False, mse=5e-3) for i in range(60)]
        )
        summary = summarize_calibration(rows, min_cads=100)
        self.assertEqual(
            summary["decision"]["status"], "REPRESENTATION_ERROR_CORRELATED"
        )


if __name__ == "__main__":
    unittest.main()
