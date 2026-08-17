# Assembly repair evidence: directed_trim full 100-CAD diagnostic matrix; rejected due to one regression

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim | 100 | 93 | 88 | 85 | 83 | 2 | 1 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
