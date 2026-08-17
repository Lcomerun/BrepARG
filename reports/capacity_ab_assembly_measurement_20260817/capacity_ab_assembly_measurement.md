# VQ capacity A/B fixed-100-CAD assembly measurement

The three arms use the same ordered parent-isolated 100-CAD cohort and the unchanged reconstruction, assembly, STEP, and OCC audit chain. All attempts, including construction and STEP failures, remain in the denominator.

| Arm | STEP readable | Native valid | Strict valid | Both valid | Attempts | Curved MSE median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous_bypass_64d | 95 | 73 | 70 | 64 | 100 | 3.33018e-05 |
| vq_8192_64d_random | 96 | 67 | 69 | 61 | 100 | 0.000221517 |
| rvq_2x4096_64d_random | 96 | 72 | 65 | 62 | 100 | 0.00022165 |

Strict validity is the preregistered decision outcome: bypass@60k=70/100, VQ-8192=69/100, RVQ=65/100. `Delta_q` for VQ-8192 is `1.0 pp`.

| Strict comparison | GT | bypass@300k | FSQ@300k | bypass@60k | VQ-8192@60k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid / 100 | 84 | 70 | 49 | 70 | 69 |

- `Delta_q = bypass@60k - VQ-8192@60k = 1.0 pp`.
- `Delta_r = GT - bypass@60k = 14 pp`.
- The registered numeric boundary-loss trigger is `true` because `Delta_r > 8 pp`. Execution remains held for the independent assembly-chain gate and result review; this report does not start boundary-loss training.

The RVQ-versus-VQ-8192 strict comparison has `5` RVQ-only successes and `9` VQ-only successes; exact two-sided McNemar `p=0.42395`. RVQ is accepted only when this improvement is positive and significant at alpha `0.05` because its preregistered estimated downstream sequence length is **+36%** (`1.36x`).

Decision: **VQ_8192_DIRECT_WIN**. VQ-8192 is within the preregistered 5 pp bypass gap; the extra RVQ sequence cost is unnecessary.

The GT, bypass@300k, and FSQ@300k columns are historical strict-only references. The bypass@60k row is loaded from the completed P0-B paired report (`f3bb8157120856fe042aa37b0c83bf3ae25fbed260b748e440834b49321ae691`). Checkpoints, STEP files, reconstruction arrays, and raw CAD remain outside this report directory.
