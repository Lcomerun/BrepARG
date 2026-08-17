# P0-A face/wire-local diagnosis

This Git-safe report narrows the original 16 P0-A failures before any further
assembly repair. It contains no STEP, pickle, reconstruction, model, or
upstream-source bytes. Each input source is bound only by SHA-256.

- Frozen cases: `16`
- Saved STEP cases with direct face/wire diagnosis: `11`
- Pre-STEP cases with no pcurve/wire observation available: `5`
- Saved STEP cases with at least one self-intersecting wire: `10`
- Saved STEP cases with wire-order failures: `0`

## Named self-intersection faces

- `00002441_179a8df075c94d0596b1f20d_step_010`: faces 6, 8
- `00008763_affd199038524a1c8e3b7ad4_step_001`: faces 17
- `00029780_a4788d5955c04fe1886666b7_step_000`: faces 24
- `00032101_674d8fea687f4d9bbca6599b_step_000`: faces 3, 4
- `00047472_197769bbdd814278b715d88a_step_000`: faces 3, 12, 43
- `00051587_446e8810d6884cae80689579_step_000`: faces 15, 20
- `00063055_e309c689b9b44f0686f47966_step_000`: faces 3
- `00067160_2a27016aa44f42c69c1079f7_step_000`: faces 17
- `00076198_7fde7438ca5d3ccb8a1dd1f4_step_000`: faces 35, 36
- `00095733_8b325d2fcb27ec9e79388602_step_000`: faces 48

The five no-STEP cases are listed in `face_wire_summary.json`. Their source
topology inventory is evidence for a pre-STEP repair investigation, not an
assertion about absent pcurves. The next repair candidate must address only
the listed face/wire entities or a named pre-STEP failure, and must still pass
the fixed 100-CAD zero-regression gate.
