# P0-B runtime evidence

This directory is a lightweight, Git-safe snapshot of the P0-B CUDA preflight.
Checkpoint bytes and protocol data remain local; their local files are bound by
size and SHA-256 in `checkpoint_manifest.json`.

## Decision

- fp32 probe: `COMPLETED` with zero non-finite events.
- bf16 probe: `COMPLETED` with zero non-finite events.
- Selected formal precision: `bf16`.
- Both probes used the same ordered and sorted train/validation inventory digests.

## Resume and writer exclusion

The bf16 resume smoke was interrupted immediately after epoch 0 was atomically
saved. The next invocation restored the full rolling state and completed epoch 1.
Final history epochs are `[0, 1]` and rolling checkpoint epoch
is `1`. A concurrent second writer exited nonzero, and the
subsequent launcher recorded stale/unreleased-lock recovery before completing.

## Contents

- `runtime_evidence.json`: compact per-epoch metrics, inventory digests, precision
  decision, resume evidence, and lock evidence.
- `checkpoint_manifest.json`: checkpoint sizes and SHA-256 only; no model bytes.
- `logs/`: task stdout/stderr, including the rejected concurrent writer evidence.
- `tensorboard/`: small probe TensorBoard event files.
- `artifact_manifest.json`: size and SHA-256 of every tracked report artifact.

These probes establish numerical execution and restart behavior only. They are
not representation-quality comparisons. Boundary-consistency work remains gated
until all four formal 100-epoch tasks and the learned-VQ 100-CAD assembly measure
are complete.
