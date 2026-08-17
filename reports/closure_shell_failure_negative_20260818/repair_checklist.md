# Follow-up repair checklist

The closure/pcurve families tested in this report are closed as negative
results. Do not broaden sewing tolerance, disable the schema-v2 gate, or
apply global ShapeFix operations to recover these rows.

The next candidate must be a narrow failure-triggered operation for a different
failure family. It must run in an isolated worker, preserve source topology
and sampled 3-D curves, and pass one-CAD STEP/native/strict/both-valid checks
before any invalid16 or 100-CAD run. The existing 84/84 original strict-valid
controls and the 95/100 release gate remain mandatory.

Recommended order for the next investigation:

1. single-edge curve fallback, starting with a one-CAD endpoint/curve case;
2. multi-edge closure only when the source graph is unambiguous;
3. shell/connectivity cases after construction-stage evidence is available.

Boundary-consistency training, sequence regeneration, and AR remain blocked by
the assembly release gate.
