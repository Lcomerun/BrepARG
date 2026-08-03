import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


class BuildVqvaePatchShardsFromArchivesTests(unittest.TestCase):
    def write_archive_member(self, archive: zipfile.ZipFile, relpath: str, n_faces: int, n_edges: int) -> None:
        payload = {
            "surf_ncs": np.zeros((n_faces, 32, 32, 3), dtype=np.float32),
            "edge_ncs": np.zeros((n_edges, 32, 3), dtype=np.float32),
        }
        archive.writestr(relpath, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def test_builds_patch_shards_directly_from_archives_without_parsed_shards(self):
        from breparg_improvements.sharded_data import iter_shard_records
        from tools.build_vqvae_patch_shards_from_archives import build_patch_shards_from_archives

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_root = root / "archives"
            archive_root.mkdir()
            with zipfile.ZipFile(archive_root / "abc_0000_parsed.zip", "w") as archive:
                self.write_archive_member(archive, "abc_0000/simple.pkl", n_faces=2, n_edges=3)
                self.write_archive_member(archive, "abc_0000/complex.pkl", n_faces=12, n_edges=20)

            patch_root = root / "patch_shards"
            manifest = patch_root / "manifest.jsonl"

            summary = build_patch_shards_from_archives(
                archive_root=archive_root,
                patch_shard_root=patch_root,
                manifest=manifest,
                chunks="0-0",
                compression="gzip",
                compression_level=1,
                patches_per_shard=10,
                complex_min_faces=12,
                complex_min_edges=20,
                max_source_faces=50,
                max_source_edges=150,
                resume=False,
            )

            self.assertEqual(summary["status"], "BUILT")
            self.assertEqual(summary["archives_seen"], 1)
            self.assertEqual(summary["source_records_seen"], 2)
            self.assertEqual(summary["source_records_skipped_by_cap"], 0)
            self.assertEqual(summary["patches"], 37)
            self.assertEqual(summary["surfaces"], 14)
            self.assertEqual(summary["edges"], 23)
            self.assertTrue((patch_root / "_summary.json").exists())
            self.assertFalse(list(root.glob("**/parsed_abc_*.pkl*")))

            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), 4)
            shard_paths = sorted(patch_root.glob("vq_patch_shard_*.pkl.gz"))
            self.assertGreaterEqual(len(shard_paths), 4)

            first_records = list(iter_shard_records(shard_paths[0]))
            self.assertEqual(first_records[0]["format"], "v13.vq_patch_shard.v1")
            self.assertEqual(first_records[1]["record_type"], "vq_patch")
            source_paths = set()
            for shard_path in shard_paths:
                for record in list(iter_shard_records(shard_path))[1:]:
                    source_paths.add(record["source_path"])
            self.assertEqual(source_paths, {"abc_0000/simple.pkl", "abc_0000/complex.pkl"})

    def test_resume_reuses_existing_summary_without_rewriting_shards(self):
        from tools.build_vqvae_patch_shards_from_archives import build_patch_shards_from_archives

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_root = root / "archives"
            archive_root.mkdir()
            with zipfile.ZipFile(archive_root / "abc_0000_parsed.zip", "w") as archive:
                self.write_archive_member(archive, "abc_0000/shape.pkl", n_faces=2, n_edges=2)

            patch_root = root / "patch_shards"
            manifest = patch_root / "manifest.jsonl"
            first = build_patch_shards_from_archives(
                archive_root=archive_root,
                patch_shard_root=patch_root,
                manifest=manifest,
                chunks="0-0",
                compression="gzip",
                compression_level=1,
                patches_per_shard=100,
                resume=False,
            )
            shard_path = next(patch_root.glob("vq_patch_shard_*.pkl.gz"))
            mtime = shard_path.stat().st_mtime

            second = build_patch_shards_from_archives(
                archive_root=archive_root,
                patch_shard_root=patch_root,
                manifest=manifest,
                chunks="0-0",
                compression="gzip",
                compression_level=1,
                patches_per_shard=100,
                resume=True,
            )

            self.assertEqual(first["patches"], second["patches"])
            self.assertEqual(shard_path.stat().st_mtime, mtime)
            self.assertEqual(second["status"], "SKIPPED_EXISTING")

    def test_overwrite_incomplete_removes_stale_shards_without_summary(self):
        from tools.build_vqvae_patch_shards_from_archives import build_patch_shards_from_archives

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_root = root / "archives"
            archive_root.mkdir()
            with zipfile.ZipFile(archive_root / "abc_0000_parsed.zip", "w") as archive:
                self.write_archive_member(archive, "abc_0000/shape.pkl", n_faces=1, n_edges=1)

            patch_root = root / "patch_shards"
            patch_root.mkdir()
            stale = patch_root / "vq_patch_shard_9999.pkl.gz"
            stale.write_bytes(b"stale")
            manifest = patch_root / "manifest.jsonl"

            summary = build_patch_shards_from_archives(
                archive_root=archive_root,
                patch_shard_root=patch_root,
                manifest=manifest,
                chunks="0-0",
                compression="gzip",
                compression_level=1,
                patches_per_shard=100,
                overwrite_incomplete=True,
            )

            self.assertEqual(summary["status"], "BUILT")
            self.assertFalse(stale.exists())
            self.assertTrue((patch_root / "vq_patch_shard_0000.pkl.gz").exists())


if __name__ == "__main__":
    unittest.main()
