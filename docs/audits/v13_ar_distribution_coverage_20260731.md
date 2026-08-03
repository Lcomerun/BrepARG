# V13 AR Length Coverage

- Created: 2026-07-31 17:18:13
- Sequence package: `ABC\processed\train_outputs\ubuntu\sequences_fsq_rcm.pkl`
- Limits: 1024, 1536, 2048
- Complex threshold: faces >= 12 or edges >= 20
- Recommendation: `train_long_context_ar` (preferred max_seq_len=2048)

## Overall

| Limit | Allowed | Excluded | Complex allowed | Complex excluded | Complex allowed frac |
|---:|---:|---:|---:|---:|---:|
| 1024 | 322546 | 102574 | 186502 | 102574 | 0.6452 |
| 1536 | 393140 | 31980 | 257096 | 31980 | 0.8894 |
| 2048 | 420192 | 4928 | 284148 | 4928 | 0.9830 |

## Splits

| Split | Groups | Empty | Grammar ok | Grammar failed | Complex total | Max len |
|---|---:|---:|---:|---:|---:|---:|
| train | 382903 | 0 | 382903 | 0 | 260248 | 2353 |
| val | 21214 | 0 | 21214 | 0 | 14513 | 2353 |
| test | 21003 | 0 | 21003 | 0 | 14315 | 2353 |

## Split Distributions

| Split | Metric | Count | Min | P25 | Median | P75 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | lengths | 382903 | 49 | 259.0 | 542.0 | 997.0 | 1714.0 | 2071.0 | 2353 |
| train | faces | 382903 | 2 | 7.0 | 14.0 | 24.0 | 40.0 | 48.0 | 50 |
| train | edges | 382903 | 2 | 14.0 | 32.0 | 60.0 | 105.0 | 129.0 | 150 |
| val | lengths | 21214 | 49 | 260.0 | 563.0 | 1012.0 | 1717.0 | 2091.87 | 2353 |
| val | faces | 21214 | 2 | 8.0 | 14.0 | 24.0 | 41.0 | 48.0 | 50 |
| val | edges | 21214 | 2 | 15.0 | 33.0 | 61.0 | 107.0 | 131.87 | 150 |
| test | lengths | 21003 | 49 | 260.0 | 552.0 | 1011.0 | 1717.0 | 2067.0 | 2353 |
| test | faces | 21003 | 2 | 8.0 | 14.0 | 24.0 | 40.9 | 48.0 | 50 |
| test | edges | 21003 | 2 | 15.0 | 32.0 | 60.0 | 106.0 | 129.0 | 150 |

| Split | Grammar valid | Complex | Complex fraction |
|---|---:|---:|---:|
| train | 382903 | 260248 | 0.679671 |
| val | 21214 | 14513 | 0.684124 |
| test | 21003 | 14315 | 0.681569 |

## Interpretation

longer context admits more grammar-valid complex sequences into AR training/evaluation
