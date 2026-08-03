# BrepARG Baseline Output Audit

Created: 2026-07-20 14:27:24
Run directory: `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d`

## Protocol

- Complex threshold: `advanced_faces >= 12` or `edge_curves >= 20`
- Max threshold: `advanced_faces <= 45` and `edge_curves <= 120`
- Strict BRep validity requires quality manifest: `True`

## Summary

| Metric | Value |
| --- | ---: |
| `step_files` | 92 |
| `png_files` | 91 |
| `stl_files` | 92 |
| `quality_manifest_rows` | 92 |
| `step_read_ok` | 91 |
| `brep_valid` | 75 |
| `files_solid_closed_no_open_shell` | 91 |
| `files_with_nonplanar_surfaces` | 92 |
| `complex_by_step_entities_12faces_or_20edges` | 5 |
| `complex_and_closed` | 5 |
| `complex_and_brep_valid` | 3 |
| `complex_and_brep_valid_closed` | 3 |
| `strict_quality_accepted` | 0 |
| `simple_or_rejected` | 92 |
| `accepted_fraction` | 0.0 |
| `complex_fraction` | 0.05434782608695652 |

## Face/Edge Stats

| Group | min | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `advanced_faces` | 2 | 6 | 6.217391304347826 | 10 | 12 |
| `edge_curves` | 2 | 12 | 11.902173913043478 | 18 | 24 |

## Reject Reasons

| Reason | Count |
| --- | ---: |
| `primitive_like` | 92 |
| `step_entities_too_simple` | 87 |
| `too_simple` | 87 |
| `brep_not_valid` | 17 |
| `not_solid_closed` | 1 |

## Warnings

- complex STEP entities exist but none pass the strict quality gate

## Reading

This adapter audits upstream BrepARG output layouts and normalizes them to the same face/edge complexity and quality-gate vocabulary used for V13 generated samples. When a quality manifest is absent, entity complexity is still measured from STEP text, but strict BRep validity remains unknown.

