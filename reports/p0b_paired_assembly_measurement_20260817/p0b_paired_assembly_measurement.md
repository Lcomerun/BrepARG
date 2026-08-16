# P0-B fixed-100-CAD paired assembly measurement

Both 60k arms use their predeclared seed-3 best checkpoint, the identical ordered 100-CAD cohort, and the unchanged assembly and OCC audit chain. Every failure remains in the 100-attempt denominator.

| Arm | STEP readable | Native valid | Strict valid | Both valid | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| bypass@60k | 95 | 73 | 70 | 64 | 100 |
| VQ@60k | 95 | 55 | 57 | 49 | 100 |

| Strict comparison | GT | bypass@300k | FSQ@300k | bypass@60k | VQ@60k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valid / 100 | 84 | 70 | 49 | 70 | 57 |

- `Delta_q = bypass@60k - VQ@60k = 13 pp`.
- `Delta_r = GT - bypass@60k = 14 pp`.
- Gate decision: `CAPACITY_AB_FIRST`. Capacity A/B has precedence when `Delta_q > 5 pp`; otherwise boundary consistency starts when `Delta_r > 8 pp`.

The GT, bypass@300k, and FSQ@300k values are historical strict-only references. The two current 60k rows separately report STEP readability, OCC native validity, project strict validity, and their conjunction. Checkpoints, STEP files, and reconstruction arrays remain local.
