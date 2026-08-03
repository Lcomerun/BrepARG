# Complex Curved FSQ/AR Diagnostic Summary

These runs isolate reconstruction and teacher-forcing behavior on complex curved validation records. They do not sample from AR.

| Run | Shapes | Patches | MSE median | MSE p95 | Chamfer median | Chamfer p95 | AR weighted CE | STEP saved | BRep valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dfs | 50 | 3265 | 2.30321e-05 | 0.00676019 | 0.0130452 | 0.156588 | 1.25869 | 0/0 | 0/0 |
| rcm | 50 | 3265 | 2.30321e-05 | 0.00676019 | 0.0130452 | 0.156588 | 1.31302 | 0/0 | 0/0 |

## Reconstruction Detail

| Run | Bucket type | Bucket | Attempted | STEP saved | BRep valid | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| n/a | n/a | n/a | 0 | n/a | n/a | 0 |

## Current Reading

- FSQ patch medians are low, but p95/max errors are much larger, especially on surfaces. That points to a heavy-tail capacity or loss-weight problem rather than a uniform failure.
- AR teacher-forcing CE is high on the complex curved subset compared with the earlier global validation CE, so AR is also weaker on this subset before free-running exposure bias appears.
- True-token reconstruction saves STEP for only part of the subset and strict BRep validity is lower still. This means the FSQ/OCC reconstruction path itself needs diagnosis before relying on generation-time filters.

## BrepARG Baseline Note

Official BrepARG ABC weights should be tried first before local retraining. Public source: `https://huggingface.co/qingtiannihao/BrepARG` lists `checkpoint/weights/abc_ar.pt` and `checkpoint/weights/abc_vqvae.pt`; the local `BrepARG/README.md` describes this repository as the official implementation.

