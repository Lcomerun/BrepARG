# ADR-0009: Use a source-bound stage census before new assembly repairs

## Status

Accepted for diagnostic implementation and one signed seven-CAD census. It is
not a production repair, selector profile, or training authorization.

## Date

2026-09-04

## Context

The frozen 100-CAD assembly selector is strict-valid for 91 CADs. It preserves
all 84 historically strict-valid controls and has zero worker or protocol
failures, but it remains below the release requirement of at least 95/100 with
zero regressions and an unchanged schema-v2 geometry/topology gate. The
representation capacity comparison is already resolved: VQ-8192/64D reaches
69/100 and the continuous bypass reaches 70/100 on the same unrepaired chain.
The active blocker is therefore assembly, not another GPU capacity sweep.

The selector has nine strict-invalid rows. A later exact-CAD four-cell
experiment tested 47472 and 63055. Both controls reproduced and both registered
`FixRemovePCurve` followed by `FixAddPCurve` candidates were invoked but could
not prove application. The run completed with four denominator rows, no
worker/protocol failure, no non-finite evidence, and the conclusive decision
`CLOSE_EXACT_CAD_CANDIDATES`. Its Git-safe archive is
`reports/exact_cad_repair_feasibility_20260904/` at commit
`afafeb81e1674078aa4e08c2987f4343d4734808`; run signature, terminal row hash,
and summary hash are respectively
`1d4f68839aadc8b3f8fb38eea642a1f7ea4f6d8d51b61152f943c725832ffcad`,
`f158f2ca7f9bf2adceb7a56434ca4925bed99e34d5791e8867ca476f32d70a34`,
and `545802b4e783a3f3f76039d70e983fba1ab5eb29af0748e5db032e731c925f60`.
That evidence closes the two registered implementations without claiming that
every possible pcurve or wire reconstruction is impossible.

The remaining seven CADs span different observable outcomes. The primary path
stops during assembly for 51602, 61931, and 87341. Other cases reach STEP but
remain invalid. In particular, current evidence records 32101 as
construction-native false and STEP-reimport native false, while 76198 is
construction-native false and STEP-reimport native true but project-strict
false. These distinctions show that a final STEP diagnosis alone cannot reveal
where a failure began. Another broad repair sweep would mix curve fitting,
topological construction, pcurve creation, optional face repair, sewing, solid
construction, and STEP transfer without a causal boundary.

Source topology also cannot be tracked by explorer index. ShapeFix and sewing
can copy or reorder entities, and STEP transfer destroys in-memory identity.
Any stage comparison based on matching face number to face number or taking the
nearest edge can attribute a defect to the wrong source entity. Before adding a
new mutation, the project needs a read-only census whose correspondence is
unique or explicitly inconclusive.

Implementation exposed an additional confound in the first observer draft. It
fitted every curve before constructing any edge, while the historical
constructor runs `fit(i) -> MakeEdge(i) -> fit(i+1)`. The draft would therefore
execute later curve fits after an early MakeEdge failure that stops the real
control. That globalized design was rejected. The accepted observer preserves
the historical loops and treats S1-S4 as source-bound distributed events, not
whole-CAD simultaneous snapshots.

One real-OCC smoke on historical calibration CAD
`00066307_28655d45c4dc4e378db24d63_step_000`, which has 30 faces and 86 edges,
supports that correction. With 200 joint-optimization iterations, observer-off
and observer-on construction had the same completed native result, shell count,
and solid count. The observer emitted S1=86, S2=86, S3=30, S4=30, S5=1, and
S6=1 in exact source order. A separate dynamic A/B against commit `afafeb81...`
matched the compared native status, shape counts, topology, fit settings, and
diagnostics. Both direct analyzers were native-invalid, so these smokes prove
path equivalence and event order only. No formal seven-CAD census row exists
until the code is frozen in a clean commit and the signed ten-cell protocol is
run.

A later full smoke on the same CAD exposed a distinction that the first
lineage fixtures did not model. Before sewing, its 86 independently constructed
edges expose 172 native endpoint handles for 58 source vertex labels. Face and
wire construction preserve only 38 of 172 endpoint occurrences as `IsSame` to
the standalone edge handles, although sewing subsequently produces the exact
source population of 30 faces, 86 edges, and 58 vertices. Native-handle sharing
is therefore not a valid cross-edge or cross-builder source-topology oracle at
S2-S4. STEP roundtrip shows a different, real topology change: the imported
shape has 30 faces, 90 edges, 60 vertices, and 180 face-edge occurrences rather
than 30, 86, 58, and 172. Source faces 1, 6, and 28 retain essentially identical
area and bounding boxes while their edge counts rise from 4/9/4 to 6/13/6. The
pre-sewing handle-copy behavior must not be mislabeled as failure, while the S7
edge split must remain fail-closed under schema-v2.

The final development smoke on that CAD completed in about 6.84 seconds with
zero non-finite and zero protocol failures. The exact prefix and S5/S6 global-
proof path localize the first bad boundary to S6 because construction-native
validity is false. S7 later becomes reimport-native-valid and project-strict-
valid, but retains the 86/58/172 to 90/60/180 topology change. The assessment is
conclusive with `first_bad_stage=S6`, reason `construction_native_invalid`, and
`valid_chain=false`: downstream validity recovery cannot erase an earlier bad
boundary. This smoke is not a formal denominator row. Its S7 outcome is a
registered scientific no-perfect-match/topology-split observation, not an
internal matching exception.

## Decision

Implement the living plan
`plans/source_bound_stage_census_7cad_execplan_20260904.md` and run one signed,
read-only source-bound stage census. Derive its cohort explicitly: validate the
frozen 100-CAD manifest and completed selector; require exactly nine current
strict-invalid CADs; exclude only the exact-negative 47472 and 63055 constants;
and require the remaining ordered seven to be 51602, 61931, 67160, 87341,
76198, 95733, and 32101. The exact-negative archive identities are frozen in
the code and run payload, but its local files are not added as fragile runtime
inputs.

Register exactly ten denominator cells. Seven are unchanged
`directed_trim_local_intersection_topology` primary controls. Three additional
cells run 51602, 61931, and 87341 with the existing directed-trim
`curve_interpolate` switch solely as reachability bridges. A bridge is not
`curve_fit_rescue`. It may expose later stages after a primary construction
error, but it cannot count as a repair, restored CAD, selector numerator,
schema-v2-accepted profile, residual expansion, or training authorization.

Observe these ordered boundary types. S1 through S4 are distributed events,
not whole-CAD snapshots:

- S1 once per source edge, after `fit(i)` and immediately before
  `MakeEdge(i)`;
- S2 once per source edge, after `MakeEdge(i)` and before `fit(i+1)`, producing
  the exact event stream S1(0), S2(0), S1(1), S2(1), and so on;
- S3 once per source face, after that face's pcurves are present and before
  optional repair of the same face;
- S4 once per source face, after that face's optional repair and before the
  next face, producing S3(0), S4(0), S3(1), S4(1), and so on;
- S5 after sewing and before solid construction;
- S6 after solid construction and before STEP, including in-memory
  construction-native validity;
- S7 only after STEP write and independent reimport, including reimport-native
  and project-strict validity.

Add one keyword-only, default-`None`, read-only
`assembly_stage_observer` to `construct_brep_directed`. Its return value is
ignored and it cannot replace a face or shape. The established
`assembly_stage_face_observer`, `post_pcurve_face_mutator`, and
`post_sewing_shape_mutator` interfaces and all default behavior remain
unchanged. An observer exception fails closed with its exact stage. The census
passes no mutator and must not invoke `FixRemovePCurve` or either exact-CAD
mutation route.

Classify distributed lineage with two different meanings. An
`exact_prefix_pass` proves that all canonical entities reached so far crossed
the current boundary before the paired next boundary stopped traversal; it is
positive preceding evidence and cannot itself be first-bad. A
`local_exact_failure` proves a unique terminal entity at the current boundary
and may be first-bad when every prerequisite is exact. The implementation may
accept `exact_prefix` as a serialized compatibility alias, but reports use the
clearer `exact_prefix_pass` meaning.

S1 and S2 terminals must be emitted explicitly at the instrumented curve-fit
and MakeEdge failure sites. S3 or S4 may infer a terminal only when an ordinary
construction exception follows an otherwise canonical alternating source-face
event prefix, making the next event unique. S5 may be inferred only after full
exact S4 coverage and an exception before any S5 observation; S6 may be inferred
only after exactly one exact S5 and an exception before S6. Never synthesize a
terminal from exception text, an observer exception, duplicate or reordered
events, missing prerequisite evidence, or ambiguous earlier lineage. Later
stages remain not reached after a proved terminal; missing or malformed evidence
remains inconclusive rather than being repaired by inference.

An inferred S5 or S6 terminal uses a dedicated whole-shape boundary scope. It
does not invent a source face or edge identifier: S5 and S6 each occur once for
the complete CAD, so exact canonical prerequisite evidence plus the ordinary
construction exception proves only that the next whole-shape boundary failed.

Keep the parent coordinator free of pythonocc, NumPy, and any transitive OCC
import. The parent validates JSON inputs, hashes code and data, signs a clean
commit, launches subprocesses, persists rows, and derives the summary with pure
functions. Every `(CAD, arm)` cell runs in its own fresh child process. A native
exit, timeout, spawn failure, malformed result, or internal measurement failure
remains one explicit worker/protocol-failure denominator row. A well-formed
missing, zero-match, nonunique, or ambiguous lineage proof remains a separate
scientific-inconclusive denominator row rather than being mislabeled as a
runtime failure.

Decode all signed or resumable JSON with duplicate-member rejection,
non-finite and overflow rejection, exact key schemas, and exact recursive JSON-
type comparison. Resume recomputes the stored payload signature and requires it
to equal both the stored and current signature; numeric equality such as
`96256.0 == 96256` is not identity.

Bind runtime drift with schema `source-bound-runtime-abi-sentinel-v1` and scope
`representative_abi_sentinel_not_complete_module_inventory`. This representative
ABI sentinel binds the Python executable name, bytes and SHA-256; Python and
NumPy versions; pythonocc version and `_Standard.pyd` name, bytes and SHA-256;
loaded `TKernel.dll` PE versions, name, bytes and SHA-256; and Python isolation
flags. It deliberately does not claim a complete inventory of every lazy-loaded
pythonocc/OCCT module. Launch workers with `python -I -c` plus a controlled
`runpy` bootstrap. The actual worker must measure its sentinel before input
selection, source-byte access, deserialization, or scientific imports and must
exactly match the signed and frozen value. Scientific completed or inconclusive
rows retain the path-free sentinel; worker/protocol failure rows retain `null`.
A separate probe child may bind the run payload but cannot substitute for this
same-process evidence.

Bind each source pickle through six comparisons: the parent-signed expected
byte identity, the child path before load, the exact bytes passed to
`pickle.loads`, the path after load, the path after all measurement, and the
parent path after the child returns. All six SHA-256 and byte counts must match. Terminal reopen separately
rehashes the current seven sources as a run-level audit. The formal run is
accepted only from a clean Git commit, and any change to code, input, task
order, stage schema, arguments, or runtime state detectable by the signed
representative ABI sentinel requires a new output root.

A full exact stage must have complete and unique correspondence for every
source population that exists at that boundary. A proved distributed prefix has
only the prefix scope it actually observed and must not claim a complete CAD
topology. From S2 onward, vertex endpoint labels are mandatory, but the proof
matches the topology that actually exists at each OCC lifecycle stage. S2 binds
each source edge authoritatively to the just-built standalone OCC edge and
checks its two local endpoint occurrences, preserving a repeated source label
for a self-loop. It does not require independently built edges that share a
source label to share an OCC vertex handle. S3 and S4 require a unique
source-face and source-edge occurrence mapping and a stage-local endpoint
relation; legal face or wire copies need not be `IsSame` to S2 handles. A
malformed local endpoint relation or ambiguous source mapping remains
non-exact. S5 and S6, after sewing has materialized global topology, may claim
exactness only with one unique source-to-observed vertex proof over the whole
shape; unchanged face and edge totals alone are insufficient. A registered
missing, ambiguous, or nonunique proof remains bounded scientific non-exact
evidence rather than becoming a worker failure. It cannot claim exactness or
first-bad on its own, and it cannot erase an earlier directly localized bad
boundary.

For every S7 reached by a formal task, retain an attempt-unique STEP file in the
machine-local run root and record its byte count and SHA-256. Resume and terminal
validation rehash that file. The Git-safe snapshot retains only the logical
identity, size, and digest, never STEP bytes or an absolute path. The coordinator
also rejects selector evidence containing a nonzero nested candidate worker
return code and rejects calibration or selector inputs whose SHA-256 differs
from the pre-registered values.

Use identity or authoritative OCC modification history where available before
STEP. After STEP, an exact claim requires one unique global face, edge, and
vertex geometry-and-incidence assignment. Face and edge comparison handles
curve direction and closed-curve phase. Vertex candidates are constrained by the exact multiset of
incident mapped source-edge IDs and a frozen normalized 3D tolerance, after
which every mapped edge's unordered endpoint labels must match the source,
including self-loop multiplicity. Explorer ordinal, nearest-neighbor coercion,
a split or merged source entity, zero or multiple perfect assignments, missing
entities, and non-finite measurements all fail closed. A zero or non-unique
perfect assignment is a registered scientific non-exact result and retains its
bounded proof fields and failure code. An exception raised while computing the
matching is an implementation/runtime failure, is promoted through
`StepGeometryIncidenceMatchingError`, and becomes a worker/protocol-failure row;
it must never be serialized as ordinary unavailable geometry. Construction-
native and STEP-reimport native values remain separate and are not forced into
a monotonic sequence.

The STEP source vertex point is derived from the optimized `edge_wcs` endpoint
samples carrying each ordered `edge_vertex_adj` label, using the same reimported
STEP bounding-box scale as the matching tolerance. It is not copied from stale
pre-optimization `corner_unique`. Both frozen census profiles require
`solid_topology_repair=False`; otherwise the endpoint labels would no longer
name the topology actually consumed by the constructor and the task fails
closed before measurement.

The census does not lower or reinterpret
`assembly-selector-geometry-gate-v2`. Its exact topology and incidence checks
and current continuous thresholds remain unchanged. The report may authorize
only the design of a new exact-CAD candidate when it exposes a unique local
first-bad transition. It cannot authorize a repair or expansion by itself.
Every new mechanism requires a separate pre-registered exact-CAD control and
candidate experiment before related residual-family, frozen invalid-16, and
fixed-100 expansion, in that order.

Archive only Git-safe evidence: compact path-free JSON/JSONL, Markdown, artifact
hashes, and validation results. STEP and pickle bytes, checkpoints, NumPy raw
arrays, absolute paths, raw worker logs, upstream source payload, and OCC/native
handles stay in machine-local storage.

## Alternatives Considered

### Repeat the 47472 and 63055 exact candidates with more parameters

Rejected because the signed four-cell result is a conclusive negative for the
registered `FixRemovePCurve`-then-`FixAddPCurve` implementations. Repeating the
same mechanism would not address the other seven residuals and would erase the
distinction between a new hypothesis and a parameter sweep of a closed route.

### Run another broad invalid-16 or 100-CAD repair profile

Rejected because a broad profile combines multiple causal stages and prior
apparent recoveries have changed edge, vertex, or incidence topology. A small
stage-exact census is cheaper and makes the next candidate falsifiable before
any larger denominator is exposed.

### Count curve interpolation as a recovery for the three assembly errors

Rejected because changing the curve construction path is a confound. The three
bridge rows exist only to observe downstream reachability. Their validity must
remain separate from the primary controls and cannot change 91/100.

### Run one process for all ten tasks

Rejected because OCC can terminate or corrupt its native process. One CAD/arm
per child confines a failure to one retained row and lets the parent complete
the denominator.

### Import OCC in the parent for convenience

Rejected because isolation would then begin after native initialization. The
parent must remain capable of recording a child crash without sharing the same
OCC process state.

### Match entities by explorer order or nearest curve

Rejected because construction repair, sewing, and STEP transfer can reorder,
copy, split, or merge topology, and repeated geometry can create tied nearest
matches. Only unique correspondence can support a causal first-bad claim.

### Make S1 and S2 global snapshots after all curves and all edges

Rejected after implementation showed that this requires moving curve fitting
out of the historical per-edge loop. It changes failure reachability and would
measure a reordered constructor. Distributed edge events retain exact causal
ordering without changing the control.

### Relax schema-v2 because this is diagnostic work

Rejected because a diagnostic that assumes topology drift is harmless cannot
select a release-safe repair. The census may observe drift, but no later
candidate can be promoted without the existing gate.

### Resume representation or AR training while assembly diagnosis runs

Rejected because capacity is already within one percentage point of bypass and
the known 91/100 assembly chain would be carried downstream. Full VQ, sequence,
and AR work remain gated on assembly release and repaired-chain validation.

### Claim a complete native module inventory

Rejected because the registered sentinel hashes representative interpreter and
ABI anchors but does not enumerate every lazily loaded pythonocc or OCCT module.
The explicit scope field preserves that evidence boundary without delaying this
census for an unrelated complete loader inventory.

### Use a normal script worker or trust only the probe child

Rejected because `PYTHONPATH`, user site packages, `sitecustomize`, or actual
worker imports could diverge from a separate isolated probe. The accepted path
uses an isolated `-I` bootstrap and same-process sentinel measurement before
the worker reads CAD data or begins scientific work.

## Consequences

The directed assembler gains a wider diagnostic view, but no production repair
surface: the new observer defaults to `None`, cannot replace its input, and is
absent from the historical path. Existing observer and mutator behavior remains
compatible. Because S1-S4 are distributed events, downstream reports must not
describe them as uniform whole-CAD states; aggregation retains source entity
IDs, canonical ordering, prefix scope, and any explicit or strictly inferred
terminal.

The formal result contains ten rows even when one or more children fail. It may
produce fewer than seven conclusive first-bad assignments because ambiguity is
reported rather than guessed. That is an acceptable negative or inconclusive
outcome; it calls for stronger observation evidence, not looser mapping.

The three bridge rows cost additional CPU/OCC inference but prevent the early
curve-construction failures from hiding every downstream boundary. Their
strict separation prevents them from inflating a selector score.

The project gains an attributable map from a construction transition to each
future repair hypothesis. This map and later repair-to-recovered-case mapping
are suitable evidence for the experiment record and paper motivation, while
the source data and native artifacts remain local.

The stronger vertex requirement can make a row scientifically inconclusive
even when its face and edge mapping is unique. This is intentional: endpoint
rewiring is a release-relevant topology change, not a detail that may be hidden
behind matching entity counts.

The stage-local S2-S4 proof is deliberately narrower than the whole-shape S5-S7
proof. This is not a relaxation of source correspondence: source edge and face
assignments must still be complete and unique. It prevents a known OCC object-
lifecycle detail from masquerading as a source vertex split while preserving
the stricter global proof at the stages where shared topology actually exists.

Strict JSON validation intentionally rejects evidence that may look
numerically equivalent but differs in syntax, type, key set, finiteness, or
self-signature. Each scientific row is larger because it carries its path-free
same-process sentinel; worker/protocol failures cannot impersonate that proof
and keep the field null. The representative sentinel reduces environment-drift
risk but does not eliminate the explicitly unclaimed risk from unenumerated
lazy native modules.

Registered S5/S6/S7 non-exact proofs remain useful archive evidence but cannot
authorize a repair. An unexpected matching implementation exception instead
makes the census protocol inconclusive, preventing broken code from producing a
plausible scientific diagnosis.

No selector score changes under this ADR. The release gate remains at least
95/100 strict-valid, historical controls 84/84, zero regressions, zero
worker/protocol failures, zero non-finite evidence, and unchanged schema-v2
acceptance for every selected repair. Until a separate exact candidate passes
and expands through the required cohorts, full training, sequence regeneration,
boundary-consistency work, and AR remain blocked.
