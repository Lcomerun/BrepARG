import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


def make_record(faces: int, edges: int) -> dict:
    return {
        "surf_ncs": np.zeros((faces, 32, 32, 3), dtype=np.float32),
        "edge_ncs": np.ones((edges, 32, 3), dtype=np.float32),
        "surf_bbox_wcs": np.zeros((faces, 6), dtype=np.float32),
        "edge_bbox_wcs": np.zeros((edges, 6), dtype=np.float32),
        "edgeFace_adj": np.array([[0, 1] for _ in range(edges)], dtype=np.int32),
        "faceEdge_adj": [[i for i in range(edges)] for _ in range(faces)],
    }


def write_zip_record(archive_root: Path, relpath: str, record: dict) -> None:
    chunk = relpath.split("/", 1)[0]
    archive = archive_root / f"{chunk}_parsed.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        import io

        buffer = io.BytesIO()
        pickle.dump(record, buffer)
        zf.writestr(relpath, buffer.getvalue())


class PrepareV13SameDataSplitTests(unittest.TestCase):
    def test_materializes_split_without_surface_edge_sources(self):
        from tools.prepare_v13_same_data_split import prepare_v13_same_data_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_root = root / "archives"
            rel_train = "abc_0000/train_a.pkl"
            rel_val = "abc_0001/val_a.pkl"
            rel_test = "abc_0001/test_a.pkl"
            write_zip_record(archive_root, rel_train, make_record(2, 3))
            write_zip_record(archive_root, rel_val, make_record(4, 5))
            write_zip_record(archive_root, rel_test, make_record(6, 7))

            sequence = root / "sequences.pkl"
            with sequence.open("wb") as handle:
                pickle.dump(
                    {
                        "train": [{"source_relpath": rel_train}],
                        "val": [{"source_path": f"parsed-shard://x!/{rel_val}"}],
                        "test": [{"source_relpath": rel_test}],
                    },
                    handle,
                )

            out = root / "out"
            summary = prepare_v13_same_data_split(
                sequence_path=sequence,
                archive_root=archive_root,
                output_dir=out,
                train_limit=1,
                val_limit=1,
                test_limit=1,
                max_faces=50,
                max_edges=150,
            )

            split_path = out / "split.pkl"
            manifest_path = out / "v13_same_data_split_manifest.jsonl"
            summary_path = out / "v13_same_data_split_summary.json"

            self.assertEqual(summary["status"], "VERIFIED")
            self.assertEqual(summary["total_written"], 3)
            self.assertTrue(split_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertFalse((out / "deduplicated_surface_source.pkl").exists())
            self.assertFalse((out / "deduplicated_edge_source.pkl").exists())

            with split_path.open("rb") as handle:
                split = pickle.load(handle)
            self.assertEqual({key: len(value) for key, value in split.items()}, {"train": 1, "val": 1, "test": 1})
            for paths in split.values():
                for item in paths:
                    self.assertTrue(Path(item).exists())
                    with Path(item).open("rb") as handle:
                        loaded = pickle.load(handle)
                    self.assertIn("surf_ncs", loaded)

            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["split"] for row in rows], ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main()
