# Assembly repair evidence: baseline full 100-CAD parity matrix

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100 | 95 | 85 | 84 | 81 | 0 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
