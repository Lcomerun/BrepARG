# Assembly repair evidence: 16-case independent switch pilot

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim | 16 | 10 | 7 | 2 | 2 | 2 | 0 |
| curve_fit_fallback | 16 | 14 | 5 | 0 | 0 | 0 | 0 |
| wire_continuity | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| single_solid | 16 | 11 | 4 | 0 | 0 | 0 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
