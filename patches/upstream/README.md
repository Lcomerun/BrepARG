# Upstream Production Assembly Overlay

`0001-fix-harden-production-assembly-topology.patch` is exported from the
clean upstream `123qiang06/BrepARG` source at `07970a4` to isolated commit
`835ed0d`. It changes only upstream `utils.py` and adds
`test_utils_topology.py`.

The patch preserves declared face IDs, rejects ambiguous face-edge mappings,
handles closed-edge scaling without endpoint division by zero, supports
one-edge trim loops, checks OCC builders and single-shell/single-solid output,
uses bounded curve-fit fallback, and applies copied-face wire repair only when
the candidate passes the geometry-preservation gate.

It is intentionally **held**, not applied to the shared
`D:\luolin\V13\BrepARG` checkout. The signed frozen 100-CAD production matrix
in `reports/assembly_production_hardening_local_topology_100cad_20260817/`
reached 87 strict-valid CADs with zero regressions, below the required 95.

Review the patch in a clean upstream worktree before any future application:

```powershell
git apply --check patches/upstream/0001-fix-harden-production-assembly-topology.patch
```
