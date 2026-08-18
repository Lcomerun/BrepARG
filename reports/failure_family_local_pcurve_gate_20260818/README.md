# Assembly repair evidence: failure-family-local-pcurve-gate-20260818

This is the compact matrix snapshot for the local-pcurve arm. The canonical
gate-preserving archive for this arm (and the paired curve-fit control) is
`reports/failure_family_followup_v1_20260818/`; use that report for the
per-CAD gate decision because this compact snapshot predates the dedicated
geometry-gate archiver.

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim_local_pcurve_continuity | 16 | 13 | 7 | 2 | 2 | 2 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.
