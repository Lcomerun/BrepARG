# Assembly repair evidence: failure-triggered-selector-100cad-20260818

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| failure_triggered_selector | 100 | 97 | 90 | 91 | 88 | 7 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
