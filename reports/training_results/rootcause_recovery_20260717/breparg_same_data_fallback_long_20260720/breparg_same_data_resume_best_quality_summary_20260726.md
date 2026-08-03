# BrepARG Baseline Output Audit

Created: 2026-07-26 17:27:10
Run directory: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_breparg_resume_best_20260726`

## Protocol

- Complex threshold: `advanced_faces >= 12` or `edge_curves >= 20`
- Max threshold: `advanced_faces <= 45` and `edge_curves <= 120`
- Strict BRep validity requires quality manifest: `True`

## Summary

| Metric | Value |
| --- | ---: |
| `step_files` | 100 |
| `png_files` | 0 |
| `stl_files` | 100 |
| `quality_manifest_rows` | 100 |
| `step_read_ok` | 100 |
| `brep_valid` | 86 |
| `files_solid_closed_no_open_shell` | 100 |
| `files_with_nonplanar_surfaces` | 100 |
| `complex_by_step_entities_12faces_or_20edges` | 13 |
| `complex_and_closed` | 13 |
| `complex_and_brep_valid` | 13 |
| `complex_and_brep_valid_closed` | 13 |
| `strict_quality_accepted` | 6 |
| `simple_or_rejected` | 94 |
| `accepted_fraction` | 0.06 |
| `complex_fraction` | 0.13 |

## Face/Edge Stats

| Group | min | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `advanced_faces` | 2 | 6 | 7.27 | 14 | 28 |
| `edge_curves` | 2 | 12 | 14.33 | 26 | 60 |

## Reject Reasons

| Reason | Count |
| --- | ---: |
| `primitive_like` | 94 |
| `step_entities_too_simple` | 87 |
| `too_simple` | 87 |
| `brep_not_valid` | 14 |

## Reading

This adapter audits upstream BrepARG output layouts and normalizes them to the same face/edge complexity and quality-gate vocabulary used for V13 generated samples. When a quality manifest is absent, entity complexity is still measured from STEP text, but strict BRep validity remains unknown.

