# Downstream bad-wire lineage decision

This directory is the compact Git-safe snapshot of one signed two-CAD
downstream-lineage run. The local run was accepted only after both isolated
workers completed, every construction and STEP-roundtrip phase had complete
source-face/source-edge lineage, and all protocol, binding, coverage,
observation, and mapping failure counts were zero.

## Result

- Decision: `PROMOTE_TARGETED_NONPERIODIC_REPAIR_PROBE`
- Conclusive cases: `2/2`
- Assembly release gate before this probe: `91/100` strict-valid; required:
  `95/100`

- `00047472_197769bbdd814278b715d88a_step_000`: first bad phase = `post_add_pcurves_pre_repair`; mapped defects = `9`.
- `00063055_e309c689b9b44f0686f47966_step_000`: first bad phase = `post_sewing_pre_step`; mapped defects = `4`.

The case ledger contains phase-level counts and mapped defect occurrences only.
Explorer positions, when retained, are explicitly observation labels and are
not source identity. The signed run payload binds the clean repository commit,
every relevant source-file SHA-256, the two source-pickle SHA-256 values, the
selector evidence, and the upstream runtime hash.

No STEP or pickle payload, OCC/native handle, reconstructed array, checkpoint,
worker stdout/stderr, machine-local path, or upstream source tree is archived.
This snapshot records the preregistered two-CAD decision only; it does not by
itself authorize a full 100-CAD run, boundary-loss training, sequence work, or
AR training.
