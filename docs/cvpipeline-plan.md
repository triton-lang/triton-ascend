# Cube/Vector pipeline analysis

The 0.6 implementation and validation results are recorded below. Section 6.1.2 of the [design](en/cvpipeline_flash_attention_design.md) gives the timing results and the repeat measurement.

The compiler separates a structural stage plan from the operation-pattern proofs
needed by particular transformations. The plan describes an ordered sequence of
Cube/Vector pairs and uses the existing CVP scheduler.

The reference implementation is recorded on `ebraheema/flashAtt/0.5.work`;
the review changes are on `ebraheema/flashAtt/0.6`. The current Source13 snapshot
has passed compiler qualification, including the actual two-/four-stage inputs
and normalized two-/four-/six-stage native replays. Its generic hardware matrix passed. The retained 0.5 Source12 results are identified separately below.
Functional changes and formatting are prepared as separate commits.

```text
Cube 0 -> Vector 1 -> Cube 2 -> Vector 3 -> ...
```

## Representation

`CubeVectorPipelineInfo` contains `CubeStage` and `VectorStage` descriptors, SSA
`DependencyEdge` records and `StateEdge` records for loop-carried values. Each
Cube owns a `MatmulGeometry { rows, reductionDim, cols }`; a dimension is an
`OpFoldResult`, carrying either a folded attribute or its actual SSA value.

For example, the normalized three-segment regression uses:

| Cube stage ID | rows | reductionDim | cols |
| --- | ---: | ---: | ---: |
| 0 | 16 | 16 | 32 |
| 2 | 16 | 32 | 64 |
| 4 | 16 | 64 | 16 |

The structural contract imposes no fixed stage count, tile size, scalar value or
element type. Target legality and a transformation's numerical or storage
requirements remain separate checks.

## Recognition and memory contract

Recognition starts with a flat `scf.for` containing one or more alternating
Cube/Vector segments. Each Vector segment must consume its preceding Cube
result. The first Cube can use two GM multiplicands. An accumulator that is read and an
optional bias require resident provenance and contribute dependency edges. A
cleared accumulator contributes no input data. A later Cube needs an operand
from the immediately preceding Vector stage and can use at most one GM operand.

Provenance follows supported tensor views, memory views/casts and uniquely
written private buffers. Source11 extends complete writer proofs through static
expand/collapse views as well as casts. A whole-buffer writer must be unambiguous; partial or
multiple writers and escaping aliases cannot establish that proof. Constants
and values already resident outside the loop are supported. A loop-carried
buffer retains its initial memory origin and must preserve its identity; a
carried GM view does not become a resident value.

The recognized region excludes GM stores, GM-to-UB transfers, unknown calls or
memory effects, nested regions and unresolved aliasing. Known transfers write
private loop allocations. Memory origins come from explicit memory spaces or
established importer/operation semantics, including imported GM arguments and
ND2NZ destinations. Early analysis can describe an unresolved origin; scheduled
verification requires resolved origins by default.

These facts do not certify numerical reassociation, buffer reuse, asynchronous
ownership or complete target resource feasibility. Existing transformation and
Cube address/span/resource checks remain responsible for those requirements.

## Operation-pattern proofs

`RunningSoftmaxPattern.h/.cpp` provides separate matching for exponential/reduction
and multiply/accumulate sequences and the combined reduction/rescale recurrence.
`RunningSoftmaxPattern` records the matched operations, exact literal scale,
state/reduction initializers, stage bindings and matrix geometry. Each optimized
rewrite owns its numerical restrictions. A valid structural chain does not imply
a match for a particular numerical rewrite.

The generic persistent transformation discovers the stage count,
preload lags, private induction counters and operand lifetimes from this
structural analysis. It also supports numerical tensor recurrences owned and
read by one discovered Vector stage. The first inner iteration of each outer
task uses the original initializer; later iterations use that stage's previous
update. After partitioning, explicit private UB banks and guarded snapshots
replace tensor carries. The first-inner path never reads an uninitialized bank.
Independent epilogues execute with the Vector stages that own their results.
The history path reserves stage-derived private UB banks and takes
guarded snapshots for later readers and combined epilogues. It checks actual
indices, owners, initialization branches and alias closure. Partitioned native compilation is qualified; all 66 planned hardware compilation and numerical checks passed. Preparation accepts an old state
read by a later Vector stage only after proving the supported use graph and
actual work-item order. A pending-storage marker blocks preload until the
persistent rewrite verifies its temporal storage. Independent stores to the
same GM argument fall back to a common ordered epilogue. The existing
running-softmax path retains its separate compact storage protocol.

History storage uses `D = stage_count + 1 - owner` banks for a state owned by
the given Vector stage. Current and previous reads use the actual logical
ordinal and guarded private snapshots. Verification reconstructs the selectors,
bank writes, reset predicates and aliases; the exact unit-step identity remains
valid after canonicalization removes division by one. This proof is specific to
the emitted storage and does not certify arbitrary memory reuse.

Core splitting preserves positional anchors and their result-free conditional
nesting on both cores for delayed synchronization. An opposite-core scope can
retain only this marker/TCB-metadata skeleton; executable work is rejected.
Partitioning must also preserve state-selection annotations when rebuilding
tensor or backing-buffer selects. The late state verifier reconstructs the
actual owner, reset predicate and use/yield relationships after slicing.
Resized allocations retain discardable ownership annotations; their structural
properties are derived from the new shape and dynamic operands. Unit-dimension
folding preserves the original memory space, dynamic sizes, alignment and
ownership attributes. Losing the memory space would prevent classification of
an otherwise valid Vector-to-Cube transfer.

History writes explicitly belong to their proved Vector owner. The materialize
operation does not have its own core-inference interface, so this assignment
prevents a write from surviving on Cube after its Vector producer is removed.
The verifier checks this assignment, and split-kernel regressions require that
history reads and writes occur only in the Vector half.

Invariant caching has a separate function-lifetime proof for ordinary structured
AIC consumers sharing one immutable B operand, including a single consumer.
It preserves the existing two-bank streamed-operand protocol and checks every
physical bank alternative, DMA completion and return credit. The retained paired
cache remains responsible for its shifted outer-task lifetime. See the
[planner/helper contract](cvp-invariant-cache-contract.md).

The existing rewrites retain their supported types, dimensions, initialization,
orientation, reduction policy and ownership requirements. Their geometry comes
from the identified matrix stages. An explicit request that lacks its required
pattern proof is rejected. Backup graphs retain the existing pass freeze policy.

## Serialized contract

The producer emits version 2 under `hivm.cvp.plan`. It contains:

- `stages`: ordered IDs/kinds and each Cube's `rows`, `reduction_dim`, `cols`;
- `dependencies`: the canonical directed graph between stage IDs;
- `state`: carried argument indices and producer stage IDs, with `-1` for an
  unscheduled producer;
- `memory_spaces_explicit`: whether memory origins have been resolved;
- an optional `vector_pattern` containing additional operation-specific facts.

`hivm.cvp.stage` identifies executable anchors. A dynamic dimension serializes as
`-1` and remains bound to the tagged operation's dimension operand. It is never
substituted with a guessed constant.

`hivm-cvp-plan-prepare` clears stale current annotations before recognition.
`cv-pipelining` records successful Skew application with `annotate-stages=true`.
`hivm-verify-cvp-plan` checks IDs, kinds, dimensions, scope/core order and actual
application, then analyzes a disposable flattened copy of the scheduled loop.
It checks the current graph, carried state and memory origins against the record,
and rematches any recorded numerical pattern. Returned analysis references are
mapped back to executable IR before the clone is erased. The executable scopes
are preserved.

Verified summaries are retained under `hivm.cvp.verified` only when requested by
later consumers. Annotation removal includes the stage and pattern markers.

## Public controls

Python options and native CLI controls describe mechanisms. Examples:

| Python option | Native CLI control |
| --- | --- |
| `enable_cvp_plan_analysis` | `--enable-cvp-plan-analysis` |
| `enable_cvp_rematerialization` | `--enable-cvp-rematerialization` |
| `enable_cvp_reduction_scaling` | `--enable-cvp-reduction-scaling` |
| `enable_cvp_reduction_major_layout` | `--enable-cvp-reduction-major-layout` |
| `enable_cvp_dense_operand` | `--enable-cvp-dense-operand` |
| `cvp_stage_fusion` | `--cvp-stage-fusion` |
| `enable_cvp_persistent_pipeline` | `--enable-cvp-persistent-pipeline` |

The fusion values are `none` and `reduce-exp-state`. Python `None` leaves a
control unspecified. The native defaults for the migrated controls are:

| Control | Native default |
| --- | --- |
| `enable_cvp_plan_analysis` | `false` |
| `enable_cvp_rematerialization` | `false` |
| `enable_cvp_reduction_scaling` | `false` |
| `enable_cvp_reduction_major_layout` | `false` |
| `enable_cvp_dense_operand` | `false` |
| `allow_cvp_sum_reassociation` | `false` |
| `cvp_stage_fusion` | `none` |
| `cvp_reduction_chains` | `1` |
| `enable_cvp_persistent_pipeline` | `false` |
| `enable_cvp_state_copy_forward` | `false` |
| `enable_cvp_final_output_forward` | `false` |
| `enable_cvp_output_copy_forward` | `false` |
| `enable_cvp_scalar_input_forward` | `false` |
| `cvp_persistent_pipeline_mode` | `skew` |
| `cvp_inner_unroll` | `1` |
| `cvp_invariant_cache_mode` | `none` |

The measured compiler recipe explicitly selects `auto` for invariant caching;
that selection is distinct from its native default. Existing active validation recipes are migrated by a one-to-one
name/value mapping; their shapes, tiles, host settings and selections are
preserved. Retired Python names are rejected before option filtering, and native
entrypoints do not register the retired controls or pass names.

Persistent issue coordinates take an explicit stage count and ring size. Buffer
handover restrictions remain with the transformation that owns the state, rather
than being implied by the coordinate arithmetic.

Downstream coordinates use `outer_start`, `outer_count`, `outer_ordinal`,
`outer_task`, `outer_slot`, `inner_offset`, `first_inner` and `last_inner`.
Storage records use `invariant_operand`, `maximum`, `sum`, `accumulator` and
`output` roles. Clocks describe outer iterations or accumulator updates. Rewrite
counters identify matrix-result rematerialization, exponential materialization,
reduction scaling and dense-operand layout.

## Regression coverage and archived fixtures

The new tests cover one/two/three segments, distinct matrix geometries, dynamic
SSA dimensions, carried state, memory/dataflow rejection, known views, ambiguous
writers, algebraic pattern proofs and malformed metadata. Normalized IR cases
exercise application of the scheduler and subsequent verification for 2/4/6
stages. Contract mutations exercise dimensions/types, dependencies, state,
anchors, application, schema version and an inserted GM-to-UB transfer.

Archived IR files and their manifests retain their original bytes. A test-only
snapshot reader translates their metadata for current checker programs; it is
not part of compiler entrypoints or deployment. Current controls and outputs are
checked directly. The numerical checks retain their independent operation and
index-mapping assertions.

## Validation status

The current 0.6 Source13 snapshot passed 112 focused tests and the full native
suite: 950 passed, 140 unsupported and one expected failure. The actual
two-/four-stage inputs and all three normalized two-/four-/six-stage cases
compile to executable ELFs. All fourteen reader/input/launch/UB/callback
comparisons passed, and their native bytes match Source12. Relative to the 0.5
reference, the two largest HD64 programs differ. Both passed numerical checks in the paired hardware run. The unchanged C++ unit binaries retain their
120-test pass; stock LLVM/MLIR retains the recorded baseline failure.

On NPU 1, Source13's four-stage zero/random checks and all three cache cases
(single consumer, two K pieces, three consumers) passed. Every cache case ran
uncached and cached, with changed B on a second invocation and actual cached
IR bound to the executed native bytes.
All 66 generic hardware checks passed. The paired eleven-shape run on NPU 7 measured a geometric-mean change of +0.036%. Section 6.1.2 of the [design](en/cvpipeline_flash_attention_design.md) gives every median and the two changed-binary results. The
0.6 review copy
and CPU acceptance record
identify the current evidence. These results do not replace the retained
reference qualification below.

The retained 0.5 Source12 implementation is built and locally validated. Source and
artifact identities, actual command receipts, XML and raw logs are retained in
`/home/ebraheema/bz-workflow/cvpipeline-generalize-20261114T042554.574004Z`.

| Check | Actual result |
| --- | --- |
| Native compiler/optimizer and normal template libraries | Build/install passed; nine libraries present |
| Fresh parent compiler/core/package | Build and assembly passed; dependency check reports no work after the final unused-helper removal |
| Native unit tests | 116 passed: 94 dialect utility and 22 pass tests |
| Python compiler contracts | 556 passed, 28 existing skips |
| Standalone frontend tests | 17 passed through their intended unittest entrypoint |
| Benchmark helper tests | All 54 passed across five existing modules |
| Focused compiler tests | 100 passed, zero unsupported |
| Full native suite | 942 passed, 140 unsupported, one expected failure; actual exit 0 |
| Full MLIR suite | 2,287 passed, 606 unsupported, one expected failure, one known baseline failure; actual exit 1 |
| Generic validator evidence extraction | Five rejection/completeness tests passed; pinned native reader certifies one segment/two stages from captured IR |
| Normal-route new programs and hardware numerics | All three passed: 1/2/3 recognized segments, 2/4/6 verified stages, real ELFs and exact CPU-reference matches |
| Eleven-case hardware correctness and fresh paired timing | Passed; 176 primary blocks, all eleven native kernels identical between compiler versions |
| Timing acceptance | Passed: the user selected comparison with the previous compiler on the same NPU/runtime |
| Historical absolute timings | Retained as context; different devices/runtimes produced different absolute durations |

Fresh stock and native readers reproduce the exact previously recorded SCF
bufferization function body, SHA256
`72f7b2e26688e6fde32c3fba1bdfd2aed3f2dab1d0e5346c2d81d139ea822c41`,
with three allocations and one copy. That classifies the sole MLIR failure; the
suite remains a recorded failure. No LLVM test was disabled or weakened.

The final cleanup removed one compiler-confirmed unreferenced static helper.
Its removal produced native executables byte-identical to the preceding
successful Source10 build. Four parent artifacts and all nine template libraries
retain their hashes. All 17 archived input fixtures/manifests are preserved.

Source12 preserves eight later formatting/comment/include-order edits. Their
non-include executable tokens and include sets match Source11; three include
orders changed. The affected tools were rebuilt and the local qualification
above was repeated. Only the compiler and optimizer binaries differ in the
209-file package; the remaining 207 files, including the host launcher and nine
template libraries, are byte-identical.

An additional CPU audit replays the actual retained normal-route bytecode
through both readers, requires exact equality with the hardware-qualified text,
and compiles each reader's actual output. All eleven production kernels and
three generic kernels are byte-identical to the executed native programs.
Parser fields, UB metadata, effective optional callbacks and launch policy are
bound to the qualified metadata. The host launch binary is unchanged. This
carries the recorded hardware correctness and accepted paired-timing evidence
forward; it repeats compiler qualification using the existing hardware evidence.

The current-source build/test report is
`validation-source12/SOURCE12-STAGE-SUMMARY.json`; the fourteen-case equivalence
proof is `codegen-equivalence-source12/EQUIVALENCE-RESULT.json`. The final
requirement audit is `COMPLETION-AUDIT-20261115.json` in the run directory.

At the precommit qualification checkpoint, changes were uncommitted. A
read-only audit found that the then-current native index matched HEAD and
differed from the recorded staged tree `5846b413` in five files, while all
5,254 live source files still matched Source11. That index was preserved;
the older staging checkpoint was not restored.
`CONTINUATION-PRESERVATION-AUDIT-20261115.json` records the exact difference.

Earlier failed attempts are retained separately. They include test migration
errors, missing suite tools, reused generated destinations, an incorrect pytest
invocation of the standalone unittest module, and a six-test benchmark discovery
subset. The final results above come from corrected commands and complete
coverage, without changing the test assertions.

## Normal-route and hardware validation

The real one/two/three-segment programs are in
`third_party/ascend/benchmarks/cvp_pipeline`. Their validator requires actual
recognition of 1/2/3 segments, verification of 2/4/6 ordered Skew stages, normal
native ELF generation and a numerical comparison against an independent CPU
reference with explicit FP16 boundaries. All three cases now pass on physical
NPU 5 using the unchanged Source11 frontend and native reader.

| Segments | Verified stages | Maximum absolute error |
| ---: | ---: | ---: |
| 1 | 2 | 0 |
| 2 | 4 | 0 |
| 3 | 6 | 0 |

The parent independently verified all 104 copied files, actual input/snapshot
hashes, the nine native-reader command receipts, per-stage geometries and
native ELF hashes. The first kernel remains byte-identical to the earlier
uninstrumented run. Worker and controller exited zero; the group was empty
before the original reservation was released at revision 3. The full report is
`generic-certified-hw-20261115T141629Z/GENERIC-HARDWARE-REPORT.md`; the independent
audit is `GENERIC-HARDWARE-PARENT-AUDIT-20261115.json` in the run directory.

The first admitted server run compiled and executed the one-segment program.
Its numerical comparison and effective-option checks passed, but the required
recognition remarks were absent, so the validator exited one before testing
the two- and three-segment programs. The frontend collects MLIR diagnostics
during tuning and suppresses them on success. A diagnostic IR capture shows
that recognition and scheduling did run. This first validator failure remains
retained; it is not relabelled as a passing three-case hardware run.

The corrected validator captures the actual normal compiler input and module
IR after plan preparation, before verification and after verification. The
pinned native reader reanalyzes the prepared operations and verifies the
scheduled operations, requiring exactly the expected recognized segment and
verified stage counts. It also parses the completed verifier snapshot. Missing,
truncated, ambiguous or wrong-kernel captures fail. Captured IR is evidence
only; it is never substituted into the runtime compilation. Numerical checks,
kernel source, options, repetitions and tolerances remain unchanged.

Five evidence-extraction tests pass. Before hardware execution, the native
reader also certified one segment and two stages from a CPU replay of the
recorded backend input. That separately qualified the evidence mechanism; the
three hardware cases above provide the actual normal-route qualification.
CPU evidence is retained under `generic-cpu-certification-02`; the original failure and copied raw payload
are independently audited in `GENERIC-FIRST-EVIDENCE-PARENT-AUDIT.json`.
No compiler implementation change or rebuild was needed for this correction.

A BZ AST compilation attempt reached TTIR, TTAdapter, MLIR bytecode and BCMLIR,
then stopped before native compilation. The existing backend constructs
`NPUUtils` for device-limit handling and queries the actual NPU core count during
binary generation. The selected CPU environment cannot finish that path. Later
attempts also encountered runtime initialization without an NPU. No successful
normal-route result, numerical result or kernel execution is inferred from
these attempts. The successful hardware cases above use the actual admitted
server runtime, with no mock device information, injected IR or custom bitcode.

The completed timing comparison retains the latest measured preservation
protocol: eleven production cases, eight balanced serial rounds between the
reference and candidate versions, and ten active operator tasks per block
(176 primary blocks). It uses the previously measured cleanup package as its
reference, with a verified 596-file source bundle. The current common policy
has 27 declared settings, equal to that reference after the one-to-one
name/value migration. Four fixed host settings are identical; one of those is
already in the common dictionary, giving 30 unique effective launch keywords.
`block_m` stays 128; `block_n` is 256 for HD64 and 128 for HD128.

The user-selected recovery restored a private B060 SDK and Python 3.11.15 under
`/home/e00977161`, inside the existing server113 container. Routine Python uses
the shared `fa-validation` venv with system-site-packages enabled. The recovered
runtime uses Torch 2.7.1+cpu and torch_npu 2.7.1.post4; the earlier measurement
used torch_npu 2.7.1.post3.dev20260129. Recovery did not restart or recreate the
container or change system packages, drivers or firmware.

Both compiler versions passed numerical qualification for all eleven production
cases on physical NPU 5, with seeds 12/0/42, complete-output AscendC comparisons,
independent CPU rows and the retained 32/64-iteration checkpoints at
`atol=rtol=0.005`. Eight balanced serial rounds per case produced 176 valid
primary profiler blocks, each containing ten operator tasks. The current
compiler's median differences against the freshly run reference range from
-0.182% to +0.276%, with a geometric-mean difference of -0.0044%. All eleven
native kernels are byte-identical to the reference and historical expected
kernel hashes. Raw CSVs, binaries and medians were independently audited.

The user confirmed the timing acceptance criterion: match the previous compiler
on the same NPU/runtime. The complete eleven-case comparison above passes that
criterion. An additional focused comparison on physical NPU 1 also passes:
1854.982 ms for the reference and 1854.022 ms for the current compiler, a
-0.052% difference, with identical native binaries and inputs. The parent
independently checked its sixteen primary timing blocks.

Historical absolute durations remain context. The largest case changed from
1644.223 ms to 1731.137 ms (+5.29%); the fresh reference is also slower. Saved
records identify physical NPU 0 for the earlier run and NPU 5 for the recovered
run, as well as the torch_npu difference above. The kernel source/selected AST,
28-core target, downstream compiler and common loaded CANN library hashes
match. NPU 1 is also 7.10% slower than the NPU 5 run for that case. These
observations do not isolate a single cause and do not reproduce the earlier
absolute duration; that is separate from the accepted compiler comparison.

The complete hardware report and independent measurement/lifecycle audits are
under `fa-hw-20261115T092957Z` in the run directory above. Interrupted partial
blocks were retained separately and excluded; all selected cases have eight
complete rounds. Final workers exited zero, their process group was empty and
both original SQLite reservations were released after verification.

Every hardware launch requires the original BZ reservation and a physical-host
NPU health/workload observation no older than five minutes. Peer processes are
preserved. All server writes remain within the user-authorized roots, with
mutating commands inside the existing container except the approved transfer
operations. Generic normal-route qualification and the accepted timing
comparison are complete. The focused NPU 1 run retained the same options and
eight balanced rounds, passed qualification, exited zero and released its
original reservations after empty-group verification. No peer was stopped.
The original development-runtime package was later verified through read-only
direct-path checks against its installation record. No runtime change or
legacy-package installation was made for this work.

## Requirement audit

Current source/build/local qualification refers to Source12. The recorded
normal-route hardware runs executed Source11-generated programs; exact
reader/native-program/launch-metadata equivalence establishes applicability to
Source12. Normalized IR tests remain separately labelled.

| Requirement | Current evidence and status |
| --- | --- |
| Neutral stage representation and per-matrix geometry | Implemented; unit checks include distinct geometries, dynamic SSA dimensions and carried state |
| One or more alternating segments | Local tests and all three normal-route hardware programs prove 1/2/3 segments and 2/4/6 scheduled stages |
| Memory and dependency contract | Positive and negative unit/IR cases cover views, copies, casts, GM origins, accumulators, bias, effects and ambiguous writers |
| Separate operation-pattern proofs | Independent exponential/reduction and multiply/accumulate tests pass; existing rewrite numerical restrictions remain |
| Consumer migration | Scheduler, layout, storage, cache and vector consumers build and pass focused/full native qualification; eleven production kernels retain their bytes |
| Versioned producer/verifier/consumer schema | Version 2 and malformed/stale/mutated metadata cases pass local qualification |
| Neutral public API | Current routing/recipes use mechanism names; retired names are rejected; archived inputs remain unchanged |
| Validation | Builds, local tests, classified baseline, generic hardware and the accepted eleven-case same-NPU/runtime timing comparison are qualified |
| Preservation | Live native source matches all 5,254 qualified files; archived fixtures remain retained; the current staging difference is recorded without restoring old index state |
| Documentation | Design, copied review and final requirement audit record completion, evidence and remaining context |

The machine-readable checklist is `REQUIREMENTS.json` in the run directory.
The known baseline MLIR failure remains explicitly classified with its actual
nonzero suite exit; it is not counted as a passing test.

## Settings disabled in the retained timing policy

This table records the current common recipe used by the completed eleven-case
hardware comparison. It does not establish the individual
performance contribution of each setting.

| Mechanism | Recorded setting | Meaning for later cleanup |
| --- | --- | --- |
| Dynamic Cube/Vector pipeline selection | `enable_dynamic_cv_pipeline=False` | The retained cases explicitly use the static pipeline path |
| Unit-flag synchronization | `unit_flag=False` | This synchronization alternative is not selected |
| Tightly coupled buffer reuse | `disable_tightly_coupled_buffer_reuse=True` | Reuse remains disabled to preserve the persistent storage policy |
| Inner-loop unrolling | `cvp_inner_unroll=1` | No additional inner-loop unrolling is selected; two reduction chains are a separate mechanism |

Fused multiply-add is enabled (`enable_fused_multiply_add=True`). Automatic block
synchronization injection is enabled (`disable_auto_inject_block_sync=False`),
and the fixed host settings retain synchronization solving. Invariant caching
is explicitly `auto`; its native default remains `none`. Unspecified options
are not classified as disabled solely because they are absent from this recipe.
