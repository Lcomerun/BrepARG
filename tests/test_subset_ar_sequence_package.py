import unittest


class SubsetArSequencePackageTests(unittest.TestCase):
    def test_selects_usable_sequences_and_preserves_metadata(self):
        from tools.subset_ar_sequence_package import subset_package

        package = {
            "vocab_size": 10,
            "special_tokens": {"PAD_TOKEN": 9},
            "ordering": "DFS",
            "train": [
                {"original": {"input_ids": [1, 2, 3]}},
                {"original": {"input_ids": [1, 2, 3, 4, 5]}},
                {"original": {"input_ids": []}},
                {"original": {"input_ids": [4, 5]}},
            ],
            "val": [
                {"original": {"input_ids": [1]}},
                {"original": {"input_ids": [1, 2, 3, 4, 5, 6]}},
            ],
            "test": [{"original": {"input_ids": [2, 3]}}],
        }

        subset, summary = subset_package(
            package,
            train_limit=2,
            val_limit=1,
            test_limit=1,
            max_seq_len=4,
        )

        self.assertEqual(subset["vocab_size"], 10)
        self.assertEqual(subset["ordering"], "DFS")
        self.assertEqual(len(subset["train"]), 2)
        self.assertEqual(len(subset["val"]), 1)
        self.assertEqual(len(subset["test"]), 1)
        self.assertEqual(summary["status"], "VERIFIED")
        self.assertEqual(summary["splits"]["train"]["skipped_long"], 1)
        self.assertEqual(summary["total_selected"], 4)


if __name__ == "__main__":
    unittest.main()
