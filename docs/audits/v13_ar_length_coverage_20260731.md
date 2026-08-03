# V13 AR Length Coverage

- Created: 2026-07-31 16:46:45
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

## Interpretation

longer context admits more grammar-valid complex sequences into AR training/evaluation
