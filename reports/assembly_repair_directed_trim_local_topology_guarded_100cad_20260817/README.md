# Assembly repair evidence: guarded directed trim plus local topology full 100-CAD matrix

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim_local_intersection_topology | 100 | 97 | 88 | 87 | 84 | 3 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
