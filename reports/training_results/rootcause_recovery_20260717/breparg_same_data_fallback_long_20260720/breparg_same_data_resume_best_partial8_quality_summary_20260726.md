# BrepARG Baseline Output Audit

Created: 2026-07-26 16:56:50
Run directory: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_breparg_resume_best_20260726`

## Protocol

- Complex threshold: `advanced_faces >= 12` or `edge_curves >= 20`
- Max threshold: `advanced_faces <= 45` and `edge_curves <= 120`
- Strict BRep validity requires quality manifest: `True`

## Summary

| Metric | Value |
| --- | ---: |
| `step_files` | 8 |
| `png_files` | 0 |
| `stl_files` | 8 |
| `quality_manifest_rows` | 8 |
| `step_read_ok` | 8 |
| `brep_valid` | 8 |
| `files_solid_closed_no_open_shell` | 8 |
| `files_with_nonplanar_surfaces` | 8 |
| `complex_by_step_entities_12faces_or_20edges` | 1 |
| `complex_and_closed` | 1 |
| `complex_and_brep_valid` | 1 |
| `complex_and_brep_valid_closed` | 1 |
| `strict_quality_accepted` | 0 |
| `simple_or_rejected` | 8 |
| `accepted_fraction` | 0.0 |
| `complex_fraction` | 0.125 |

## Face/Edge Stats

| Group | min | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `advanced_faces` | 2 | 6 | 6.125 | 12 | 12 |
| `edge_curves` | 2 | 12 | 11.5 | 24 | 24 |

## Reject Reasons

| Reason | Count |
| --- | ---: |
| `primitive_like` | 8 |
| `step_entities_too_simple` | 7 |
| `too_simple` | 7 |

## Warnings

- complex STEP entities exist but none pass the strict quality gate

## Reading

This adapter audits upstream BrepARG output layouts and normalizes them to the same face/edge complexity and quality-gate vocabulary used for V13 generated samples. When a quality manifest is absent, entity complexity is still measured from STEP text, but strict BRep validity remains unknown.

