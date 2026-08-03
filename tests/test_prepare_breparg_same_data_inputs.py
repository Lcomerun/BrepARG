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


class PrepareBrepArgSameDataInputsTests(unittest.TestCase):
    def test_materializes_split_and_patch_sources_from_archives(self):
        from tools.prepare_breparg_same_data_inputs import prepare_same_data_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_root = root / "archives"
            rel_train_a = "abc_0000/train_a.pkl"
            rel_train_b = "abc_0000/train_b.pkl"
            rel_val = "abc_0001/val_a.pkl"
            rel_test = "abc_0001/test_a.pkl"
            write_zip_record(archive_root, rel_train_a, make_record(2, 3))
            write_zip_record(archive_root, rel_train_b, make_record(3, 2))
            write_zip_record(archive_root, rel_val, make_record(2, 1))
            write_zip_record(archive_root, rel_test, make_record(1, 1))

            package = {
                "train": [
                    {"source_relpath": rel_train_a, "original": {"input_ids": [1, 2, 3]}},
                    {"source_path": f"parsed-shard://x!/{rel_train_b}", "original": {"input_ids": [4, 5]}},
                ],
                "val": [{"source_relpath": rel_val, "original": {"input_ids": [6]}}],
                "test": [{"source_relpath": rel_test, "original": {"input_ids": [7]}}],
            }
            sequence = root / "sequences.pkl"
            with sequence.open("wb") as handle:
                pickle.dump(package, handle)

            out = root / "out"
            summary = prepare_same_data_inputs(
                sequence_path=sequence,
                archive_root=archive_root,
                output_dir=out,
                train_limit=2,
                val_limit=1,
                test_limit=1,
                max_faces=50,
                max_edges=150,
                surface_patch_limit=10,
                edge_patch_limit=10,
            )

            split_path = out / "same_data_split.pkl"
            surfaces_path = out / "deduplicated_surface_source.pkl"
            edges_path = out / "deduplicated_edge_source.pkl"
            manifest_path = out / "same_data_input_manifest.jsonl"
            summary_path = out / "same_data_input_summary.json"

            self.assertEqual(summary["status"], "VERIFIED")
            self.assertEqual(summary["splits"]["train"]["written"], 2)
            self.assertEqual(summary["splits"]["val"]["written"], 1)
            self.assertEqual(summary["splits"]["test"]["written"], 1)
            self.assertEqual(summary["surface_patches"], 5)
            self.assertEqual(summary["edge_patches"], 5)
            self.assertTrue(split_path.exists())
            self.assertTrue(surfaces_path.exists())
            self.assertTrue(edges_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())

            with split_path.open("rb") as handle:
                split = pickle.load(handle)
            self.assertEqual(set(split), {"train", "val", "test"})
            for paths in split.values():
                for item in paths:
                    self.assertTrue(Path(item).exists())
                    with Path(item).open("rb") as handle:
                        loaded = pickle.load(handle)
                    self.assertIn("surf_ncs", loaded)

            with surfaces_path.open("rb") as handle:
                surfaces = pickle.load(handle)
            with edges_path.open("rb") as handle:
                edges = pickle.load(handle)
            self.assertEqual(surfaces.shape, (5, 32, 32, 3))
            self.assertEqual(edges.shape, (5, 32, 3))

            manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(manifest_rows), 4)
            self.assertEqual(manifest_rows[0]["source_relpath"], rel_train_a)
            self.assertEqual(Path(manifest_rows[0]["materialized_path"]).suffix, ".pkl")


if __name__ == "__main__":
    unittest.main()
