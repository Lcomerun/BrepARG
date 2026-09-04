# Exact-CAD repair feasibility decision

This directory is the compact Git-safe snapshot of the signed four-cell
control/candidate experiment for CADs 47472 and 63055. The archive was emitted
only after rechecking the immutable run signature, terminal ledger and summary
hashes, all four source-binding chains, every promoted STEP hash, runner-level
row validation, and an exact summary recomputation.

## Result

- Decision: `CLOSE_EXACT_CAD_CANDIDATES`
- Controls reproduced: `2/2`
- Candidates rejected: `2/2`
- Worker/protocol failures: `0`
- Non-finite observations: `0`
- Existing assembly gate: `91/100`; release requirement: `95/100`

- `00047472_197769bbdd814278b715d88a_step_000::control`: status `control_reproduced`; STEP readable `true`; native `true`; strict `false`; candidate attempted `false`; applied `false`.
- `00047472_197769bbdd814278b715d88a_step_000::candidate`: status `candidate_rejected`; STEP readable `true`; native `true`; strict `false`; candidate attempted `true`; applied `false`.
- `00063055_e309c689b9b44f0686f47966_step_000::control`: status `control_reproduced`; STEP readable `true`; native `false`; strict `false`; candidate attempted `false`; applied `false`.
- `00063055_e309c689b9b44f0686f47966_step_000::candidate`: status `candidate_rejected`; STEP readable `true`; native `false`; strict `false`; candidate attempted `true`; applied `false`.

Both registered candidates were actually invoked but neither local OCC helper
could prove application. The compact attempt ledger retains the target
selection, pcurve remove/add outcome, whole-CAD defect gate, strict/native
validity, and schema-v2 geometry/topology rejection evidence needed to review
that negative result.

No STEP or pickle bytes, worker stdout/stderr, machine-local path, upstream
source tree, NumPy array, checkpoint, or OCC/native handle is archived. STEP,
source, input, runtime, and code identities are retained only as byte counts
and cryptographic hashes. This negative two-CAD feasibility result closes only
the two registered `FixRemovePCurve`-then-`FixAddPCurve` surgery
implementations at their signed precision and stage. It is not evidence
against other pcurve construction or replacement mechanisms, and it does not
authorize residual-family expansion, a 100-CAD promotion, full training,
sequence generation, or AR.
