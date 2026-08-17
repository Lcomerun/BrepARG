# Decision 0004: Keep the 100-CAD selector below the assembly release gate

Date: 2026-08-18

Status: Accepted — release blocked

## Context

The project uses a frozen 100-CAD validation cohort. The historical assembly chain is strict-valid on 84 CADs. The release requirement is at least 95/100 strict-valid while retaining every one of those original 84 successes. A failure-triggered selector was introduced to retain the guarded primary output and try isolated, geometry/topology-preserving fallbacks only when the primary is strict-invalid. Selector protocol success and assembly utility are separate claims.

## Evidence

The signed run is `a4f1208d4a74026be313a6dfff6b6a1dc92ce0c79c154f5ea9dc9bf113b55cf1`. Its Git-safe snapshot is `reports/assembly_selector_main_100cad_20260818/`. The run contains exactly 100 final rows and 146 isolated candidate attempts:

- 97 STEP-readable;
- 90 native-valid;
- 91 strict-valid;
- 88 both-valid;
- 7 previously invalid CADs restored;
- all 84 historical strict-valid CADs preserved;
- 0 worker or protocol failures.

The selector protocol passed: expected restoration and fallback identities, candidate ledger bindings, source hashes, cohort identity, and snapshot hashes all match. The assembly release gate did not pass because 91 is four below 95.

## Decision

Retain the selector implementation and its complete evidence as the current best no-regression candidate. Do not call it a released assembly chain, and do not start boundary-consistency loss, sequence regeneration, or AR training. Continue only with independently diagnosed, source/topology-preserving repair probes; every candidate must be measured on the same 100-CAD cohort and preserve the 84/84 control vector.

## Consequences

The capacity conclusion remains valid: VQ-8192 is the selected representation arm (`69/100` strict versus bypass `70/100` on the unchanged chain). A repaired-chain Delta-q measurement is deferred until the assembly release gate passes. The selector's 91/100 result is reusable evidence, but it cannot justify a boundary-loss experiment or downstream model training by itself.
