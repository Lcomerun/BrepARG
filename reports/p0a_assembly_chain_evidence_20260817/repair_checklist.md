# P0-A assembly repair checklist

This checklist is generated from frozen original-control failures. It does not authorize broad automatic repair.

- [ ] `nonunit_solid_count` (1 case(s)): Require one closed shell and one solid; report compounds and empty solids as separate construction failures.
- [ ] `pre_step:curve_fit` (3 case(s)): Inspect the recorded edge index and curve fallback attempts; add a bounded degenerate-curve policy or a validated lower-degree fallback.
- [ ] `pre_step:wire_build` (2 case(s)): Validate edge endpoint continuity and orientation before wire construction; reject or reorder only the affected face loop.
- [ ] `wire_self_intersection` (10 case(s)): Trace the offending face/wire and correct trim orientation or pcurve construction before applying broad shape repair.

Acceptance requires at least 80% of the frozen 16 cases to have a named cause. Sequence and AR work remain blocked.
