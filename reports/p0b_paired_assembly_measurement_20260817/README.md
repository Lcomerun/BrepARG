# P0-B paired 60k assembly measurement

This is the Git-safe archive of the fixed-cohort comparison between the learned VQ and continuous-bypass arms. Both arms use the seed-3 best checkpoint, the same 100 original CAD identities selected with seed `20260809`, `joint_iterations=200`, the same assembly chain, and an attempts denominator of 100. All 200 STEP files and both model checkpoints remain local.

## Result

| Arm | STEP readable | OCC native valid | Project strict valid | Both valid | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| bypass@60k | 95 (95%) | 73 (73%) | 70 (70%) | 64 (64%) | 100 |
| VQ-4096/64D@60k | 95 (95%) | 55 (55%) | 57 (57%) | 49 (49%) | 100 |

| Strict comparison | GT historical | bypass@300k historical | FSQ@300k historical | bypass@60k | VQ@60k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid / 100 | 84 | 70 | 49 | 70 | 57 |

- `Delta_q = bypass@60k - VQ@60k = 13 percentage points`.
- `Delta_r = GT - bypass@60k = 14 percentage points`.
- Gate decision: `CAPACITY_AB_FIRST` because `Delta_q > 5 pp`. The capacity comparison must precede boundary-consistency loss.

The GT 84%, bypass@300k 70%, and FSQ@300k 49% values are historical strict-only references. The two current 60k rows expose STEP readability, native validity, strict validity, and their conjunction independently. The per-CAD 200-row table is in `p0b_paired_assembly_measurement.csv`; the machine-readable gate and checkpoint bindings are in the JSON companion.

## Runtime evidence

`measurement_runtime_manifest.json` records the completed calibration/audit stages, return codes, cohort identity hash, checkpoint SHA-256 values, and the four local log hashes. It intentionally omits machine-local paths, STEP bytes, checkpoint bytes, and raw CAD data.
