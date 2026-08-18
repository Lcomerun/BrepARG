# Assembly repair evidence: failure-family-minimal-probe-gate-v4-20260818

This is a Git-safe snapshot. It excludes STEP bytes, source pickle bytes, model
weights, and reconstructed arrays. Every saved STEP remains bound by size and
SHA-256 in the compact per-attempt JSONL.

| Profile | Attempts | STEP-readable | Native | Strict | Both | Restored | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| directed_trim_surface_precision_curve_interpolate | 16 | 14 | 6 | 5 | 5 | 5 | 0 |

Gate passed: `False`. This snapshot does not authorize
boundary consistency, sequence regeneration, or AR training.

The full schema-v2 geometry/topology gate was evaluated inside isolated
workers. Three of five both-valid candidates passed it, and all three were
already covered by the production selector (`00002441...`, `00008763...`, and
`00029780...`). No new selector restoration was found. The two both-valid
candidates rejected by the gate were:

- `00076198...`: candidate topology changed from 106 to 104 edges and from 68
  to 65 vertices; the bounding-box relative delta was `0.0495977`, above the
  `0.02` limit.
- `00032101...`: candidate topology changed from 18 to 16 vertices and the
  vertex-edge incidence multiset changed.

Per-CAD gate fields and source/run hashes are in
`geometry_gate_attempts.jsonl` and `geometry_gate_summary.json`. The probe had
`0` worker/protocol failures. The accepted IDs are not a promotion: this was a
historical-invalid-only cohort and the 100-CAD selector gate remains `91/100`.
