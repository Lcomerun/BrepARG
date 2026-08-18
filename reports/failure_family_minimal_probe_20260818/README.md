# Assembly repair evidence: failure-family-minimal-probe-20260818

> **SUPERSEDED / PRELIMINARY.** This snapshot was produced before the
> `--isolate-cad-workers` dispatch fix, so it does **not** contain the
> schema-v2 selector geometry gate. Do not use its 5/16 count as a promotion
> decision. The authoritative rerun is
> `reports/failure_family_minimal_probe_gate_v4_20260818/`.

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim_surface_precision_curve_interpolate | 16 | 14 | 6 | 5 | 5 | 5 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
