# Dynamic CV Pipeline Cost Model

> ### Experimental feature, enabled by the user
>
> Everything described here — the cost estimate, the operation reordering it
> ranks, and the search over those orderings — is **experimental**. It is
> switched on by the user, for one run, and the result is not something to
> build on without measuring:
>
> * the estimate ranks IR variants, it does not predict time (§1), and some of
>   its hardware numbers are **placeholders** (§11);
> * there is **no measured acceptance criterion** — nothing has been checked
>   against hardware for rank correlation (§14), so every claim here rests on
>   argument, not evidence;
> * the two known-signed biases (§13) are large: cube/vector overlap is assumed
>   free, and vector work is charged without fusion;
> * a seed the search likes is a **candidate to measure**, not a result. The
>   toolchain below this pass can still refuse it, and the guards that try to
>   predict that (§10) are heuristics.
>
> **Nothing searches unless asked.** `TRITON_ASCEND_CV_VARIANTS` or a seed
> table turns the search on; without either, one ordinary compilation runs. Use
> it at your own risk, per kernel, and keep the measurement.
>
> What does change by default, with nothing set, is listed in §13 so it is not
> a surprise: one extra reorder pass runs, and every CV-pipelined module pays
> the estimate's compile time.

This document describes everything added on top of `main-dev` for cost
estimation inside the dynamic Cube/Vector pipeline, together with the operation
reordering and the variant search the estimate exists to serve, and explains
why each decision was made. It is written so that someone who has never seen
this work can pick it up, and so that a decision is not silently reversed later
because its reason was not written down.

The merge base for all of this is `97f81c947`.

---

## 1. Purpose

The dynamic CV pipeline (`AddDynamicCVPipelinePass`, run inside the
`convert_ttir_to_linalg` stage) can **generate several candidate IR variants
and keep the cheapest one**. That needs a number to rank them by.

The cost model exists to produce that number. It is a *ranking* device: only
relative accuracy between variants matters, not absolute microseconds. A bias
that hits every variant equally is harmless; a bias that depends on how the
pipeline partitioned the work is not.

The variants themselves come from one decision: the order the operations are
linearised in, which `ReorderOpsByBlockId` makes and a seed selects. §10
describes that side; §3 to §9 describe the number.

### Why not reuse the existing cost model as-is

`third_party/ascend/costmodel` already contains a cost model
(`-ascend-perf-model`). It runs on **Triton IR, before `ttir_to_linalg`**, so it
has to *guess* what the compiler will do: `ConvertTritonToAscend` heuristically
marks Cube/Vector work around `tt.dot`.

By the end of the CV pipeline those decisions have already been made, and made
better — `OpClassifierPass` does a full BFS with alias analysis, a dozen
patterns, and `CUBE_AND_VECTOR` handling via cloning. So the new pass **reads
the decisions that were actually taken** instead of re-deriving them. Only the
hardware formulas (`HardwareConfig`) are reused from the old model; the
scheduling is new and works on compute blocks.

---

## 2. Where the pass runs, and why exactly there

`third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp`:

```cpp
pm.addPass(createSeparateMemoryFromComputePass());
// Must precede createRemoveSsbufAttrPass(): the estimate is driven by the
// ssbuffer.* attributes that pass strips.
pm.addPass(createEstimateCVPipelineCostPass());
pm.addPass(createRemoveSsbufAttrPass());
```

The position is not arbitrary. The pass reads `ssbuffer.core_type`,
`ssbuffer.block_id`, `ssbuffer.crossCoreDeps`, `ssbuffer.intraDeps`,
`ssbuffer.blockDeps` and the buffer-count attributes. `RemoveSsbufAttrPass`
deletes all of those. Running the estimate after it would silently produce a
meaningless number rather than an error, so the ordering is load-bearing and is
commented in place.

### Contract

The pass is advisory and must never be able to break a compilation:

* it does not mutate compute IR — it only adds module attributes;
* it never returns `failure()`;
* if `CVPipeline::hasFallbackAttr(module)` is set, the CV pipeline has already
  given up and the module will be replaced by its backup, so the pass returns
  immediately — costing a module that is about to be discarded is pointless;
* if the in-process cost model is not built (`TRITON_ASCEND_HAS_INPROC_COSTMODEL`
  is 0) the whole body compiles out and the pass is a no-op.

---

## 3. How a single operation is costed

`estimateOpCost` in
`third_party/ascend/lib/DynamicCVPipeline/EstimateCVPipelineCost.cpp`.

### Core assignment

Taken from `ssbuffer.core_type` via `CVPipeline::getOpCoreType`. If an
operation was never stamped, the fallback is the enclosing `scope::ScopeOp`'s
`tcore_type` (created by `SeparateCVScopePass`); failing that, Vector, matching
`OpClassifierPass`'s own default.

### Shapes: `ShapedType`, not `RankedTensorType`

At this point the IR is only partially bufferized: the same logical buffer is a
`tensor` before bufferization and a `memref` after. Every size helper is written
against `ShapedType`, which covers both. The old cost model's helpers only
handle tensors, which is why they could not be reused here.

### Data transfers

The engine that executes a transfer is derived from **the pair of address
spaces**, read from `hivm::AddressSpaceAttr` on the memref — not from the role
of the operation. That is what makes the newer topology expressible: the same
`memref.copy` may target HBM, UB or L1, and only the type says which.

`getTransferUnit(src, dst)`:

| src → dst | unit | note |
|---|---|---|
| `l0c` → `ub` | `fixpipe_ub` | accumulator drained on chip |
| `l0c` → anything else | `fixpipe` | accumulator drained to HBM |
| `ub` → `l1`/`l0a`/`l0b` | `mte3_ub_l1` | on-chip staging up to the Cube |
| other → `l1`/`l0a`/`l0b` | `cube_mte2` | Cube's own load engine |
| `l1` → `ub` | `mte1_l1_ub` | on-chip staging back down |
| other → `ub` | `vec_mte2` | ordinary vector load |
| `ub` → other | `mte3` | ordinary vector store |

Address-space names map to the bandwidth table's vocabulary; note `GM → "hbm"`,
because the migrated tables call global memory `hbm`.

Bandwidth is looked up per `src:dst` pair directly. **No routing through L2 is
assumed** — `ub:l1` and `l1:ub` are their own entries in the hardware profile.

#### Operand convention

`hivm.hir.fixpipe` and `hivm.hir.copy` are built as `(source, destination)` —
`InterCoreTransferAndSync` creates them with the operands labelled `// src` and
`// dst`. The roles come from that convention.

This matters and was a real bug once: a positional heuristic of "the first
operand carrying an address space is the source" finds only the destination,
because a fixpipe's source is often still a tensor with no address space. The
destination in UB was then taken for the source, and a Cube drain was
misclassified as a vector store, landing on `mte3` and serialising against
`vec_mte2` through the mutex clique.

The size of a transfer is taken from **the destination type**, since that is
what actually gets written.

### Vector compute

Mapped to a tilesim mnemonic (`VADD`, `VEXP`, `VDIV`, …) and costed from the
migrated per-`(intrinsic, dtype)` cycle table via
`HardwareConfig::estimateVectorCyclesFromTable`. This is what makes an `exp`
cost more than an `add`; the older `estimateVectorCycles` charges every vector
operation the same.

Reductions are costed separately: a reduction is not one pass over the data, it
is an elementwise pass plus a logarithmic tree inside each vector register and
another across registers.

### Zero-cost operations

Structure, metadata, addressing and scalar bookkeeping: module/function/return,
terminators, `scf` containers, `scope::ScopeOp`, `annotation::MarkOp`,
allocations, tensor/memref bridging, view-like operations, shape metadata
(`reshape`/`expand`/`collapse`/`cast`), constants, and scalar `arith`/`math`.

Operations inside a `linalg` body are also free — the parent is costed as a
whole, so charging its body again would double count.

### Confidence categories

A single "is this known" flag was tried first and caught the wrong thing: an
operation with no model was charged by element count and still reported as
*known*. There was no way to tell "computed precisely" from "guessed". Four
categories replaced it:

| Category | Meaning |
|---|---|
| `Modelled` | dedicated formula, all inputs statically known |
| `Generic` | charged from element count only; no model for this kind |
| `NotModelled` | recognised as real hardware occupancy but charged zero |
| `UnknownSize` | a dynamic shape, so nothing could be computed; contributes 0 |

`NotModelled` exists so cross-core synchronisation is not silently mistaken for
either "free" or "unknown size". It is real time that is accounted for
elsewhere (see §7).

---

## 4. How many times an operation runs

### Loop trip counts

Resolved in three steps, most trustworthy first:

1. constant bounds already in the IR (`analyzeScfForTripCount`);
2. the caller's argument bindings (`TRITON_ASCEND_CV_COST_ARG_BINDINGS`);
3. an assumed default (`TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT`, default 1).

Only a loop that reaches step 3 is reported, because only then is the number a
guess. Reporting is per *loop*, not per operation inside it: one loop with an
unknown trip count is one problem however many operations it contains.

#### Rounding divisions are mandatory

`AddControlFlowCondition/UpdateLoopIterTimes.cpp` rewrites every pipelined
loop's upper bound as

```
newUpperBound = lb + step * (ceildiv(ceildiv(ub - lb, step) * buffers, x) + ifCount)
```

Without `CeilDivUI`/`CeilDivSI`/`FloorDivSI` in the evaluator, a loop with
entirely constant bounds still looks dynamic. This is why the resolver
implements them.

A consequence worth remembering: the resolved trip count is **larger** than the
mathematical one, because the pipeline prologue is baked into it.

#### The launch grid is function arguments, not operations

`TritonToLinalg` does not lower `tl.program_id` to an operation. It appends the
launch grid to the entry function's arguments and replaces the op with one of
them (`TritonToLinalg/FunctionConverter.cpp`, `LAUNCH_GRID_RANK = 3`): the last
three arguments are `program_id x/y/z`, the three before them `num_programs
x/y/z`. Naming those slots is what makes `pid_x=0` work without the caller
counting arguments by hand.

#### Why a separate value resolver

The old cost model's `evaluateValue` binds **any** `BlockArgument` by index
without checking who owns it. A loop induction variable and a loop-carried value
are `BlockArgument`s too, so binding those by index would yield a plausible but
wrong number. `evaluateWithBindings` only binds arguments of the **entry block
of a `func.func`**. This is a latent bug in the old model; the restriction here
is deliberate and must not be "simplified" away.

### Mutually exclusive branches

The pipeline builds multi-buffering out of branches: with two buffers,
`AllocMultiCache` emits the producing operations twice, once per buffer, and
`AddControlFlowCondition` wraps them so exactly one copy runs per iteration.

Summing both copies makes multi-buffering look like twice the work. This was
observed directly: raising `inter_cache_num` from 1 to 2 doubled
`hivm.hir.fixpipe` from 2 operations to 4, and the entire increase in the
estimate was exactly those two extra operations — so a configuration that should
have been *faster* scored *worse*.

`computeBranchDivisors` pre-scans every `scf.if` and counts how many arms
contain at least one operation with non-zero cost:

* **two arms with work** — a genuine either/or. Over a long loop the predicate
  alternates, so each arm runs about half the iterations and the honest total is
  the average of the arms, i.e. every operation in either arm charged at half.
  For arms of equal cost, which is what buffer rotation produces, that is
  exactly one arm.
* **one arm** (no `else`, or an `else` that does no work) — a predicated stage.
  In steady state such a guard holds on all but the prologue and epilogue
  iterations, so it is charged in full.

Deciding by *"does this arm contain work"* rather than by *"does an else region
exist"* is what stops an empty `else` from halving real work. That was the main
robustness requirement: the rule has to be safe on kernels whose branches are
not buffer rotation.

Nested ifs multiply, so a three-deep rotation built from two-sided ifs comes out
at 1/8 rather than 1/3. Buffer counts that high are already warned against by
`BufferCountManager`, so this is left as is and documented.

---

## 5. Blocks and segments

### Why blocks

A compute block (`ssbuffer.block_id`, planned by `PlanComputeBlock`) is the unit
the pipeline schedules and synchronises. It is also the unit variants will
differ by. `ReorderOpsByBlockId` topologically sorts the block graph and
physically moves operations, so **a block is a contiguous run of operations** in
the IR — the rest of the pipeline depends on this too (`getBlockStartEnd` walks
linearly and stops at the first different block id).

### Segments

Within a block, every `SyncBlockSetOp`/`SyncBlockWaitOp` starts a new *segment*.
A block's cost is the sum over its segments of that segment's roofline; segments
are serial with respect to each other because a barrier stops the core.

This exists because "a block is one fused chain" is **not** always true. Most
barriers do sit on block boundaries — `InterCoreTransferAndSync` places its set
after the producing block's last operation and its wait before the consuming
block's first one — but the cube-to-vector direct-store path puts a wait
immediately before the store it guards, which can be anywhere inside the block.
On the validation kernel 5 of 11 blocks have such an interior barrier, so this
is not a hypothetical.

The count of such blocks is always reported, including when it is zero, because
zero is the useful statement that the simple model holds for that kernel.

### Intra-block fusion

Operations inside one block issue back to back on one core, so intermediates can
stay in registers instead of round-tripping through memory. Each operation is
nonetheless charged as a separate full pass over UB with its own startup
latency, which over-counts — softmax's `sub → exp → mul` is charged as three
passes rather than one.

Two factors compensate, per core because the two cores fuse for different
reasons (the Vector core chains elementwise instructions, the Cube core mostly
does not):

```
TRITON_ASCEND_CV_COST_FUSION_CUBE=0.8
TRITON_ASCEND_CV_COST_FUSION_VECTOR=0.8
```

Both default to **1.0, i.e. off** — an unmeasured factor should not silently
change everyone's numbers. Values outside `(0, 1]` are rejected: above 1 would
mean a block costs more than its parts, at or below 0 would make blocks free.

These belong in the hardware profile eventually; the environment is used for now
so they can be swept without a rebuild.

---

## 6. The block dependency graph

Two kinds of edge, both read from the IR rather than re-derived.

### Synchronisation edges

From `ssbuffer.crossCoreDeps` and `ssbuffer.intraDeps`, each an `I32ArrayAttr`
of `[groupId, role]` where role 1 produces and 0 consumes — the same format
`InitDependentMap::collectDepsByGroup` parses. These are the flag waits the
hardware will actually perform.

**SSA cannot show them.** After bufferization the two ends of a transfer are
joined through a buffer, not through a value, so a value-based walk sees nothing
exactly where the real barrier is. This is why they must be read, not inferred.

### Dataflow edges

From `ssbuffer.blockDeps`, a module attribute newly published by
`DataDependencyAnalysisPass` (see §12.1).

The first implementation walked SSA operands here. That was a worse
re-derivation of what `DataDependencyAnalysis` already computes properly: it
handles values carried through `scf.for` iter_args, producer/consumer pairs
joined only by a memory effect, and transposed operands. The SSA walk was
deleted.

Two caveats, both conservative: `RefineArgsBlockId` can move an operation to a
different block after the graph was recorded, so an edge may outlive the reason
it was added; and blocks created after the analysis (by `AllocMultiCache`,
`AddControlFlowCondition`) carry no edges at all. Either way the schedule only
loses ordering it could have enforced, which moves the estimate towards the
roofline rather than away from it.

---

## 7. Scheduling

### Ordering

Blocks are scheduled in a topological order computed by Kahn's algorithm with
ties broken by program order.

**Program order alone will not do.** `SeparateCVScope` groups the module by
core, so every Vector block can precede every Cube block in the IR even when a
Cube block produces what a Vector block consumes. Scheduling in that order
treats each such edge as loop-carried and drops it — silently discarding exactly
the cross-core barriers the model exists to honour. This was observed: block 7
started at cycle 656 while its producer block 2 finished at 3755, and the
reported barrier cost was zero.

**Dependency order alone will not do either.** It leaves an unconstrained block
free to float to the front and fill a gap the hardware does not have. So the
graph additionally chains consecutive blocks **on the same core in program
order** — a core issues its blocks in order and never reorders them. Without
that chain a 1684-cycle block hid inside a stall that does not exist, and the
critical path came out 13 % short.

A genuine cycle (a loop-carried dependency) cannot be ordered: the earliest
remaining block in program order is emitted anyway and its unsatisfied incoming
edges are dropped, which is the intended reading — this schedules one pass
through the graph.

### The schedule itself

ASAP at the granularity of a single hardware pipe. Each pipe has its own
availability; pipes that share physical hardware share one slot, decided by the
profile's mutex cliques (`getPipeResource`). A block's segments remain serial
with respect to each other, because a barrier inside a block really does stop
the core, but a block does **not** occupy its whole core: block B+1's loads run
underneath block B's compute whenever they need different pipes, which is how
the issue queue actually behaves.

An earlier version gave each core a single availability slot, so a block held
its entire core for its entire duration. That charges ordinary pipe overlap as
serial time. The damaging part for a ranking model is not the size of the error
but its shape: it grows as buffering shrinks, because the pessimism lands
hardest on the configuration with the least overlap to give away, so the model
mis-orders the very configurations it exists to compare.

One consequence is worth stating plainly: the resource bound is now
**partition-invariant**. Per-pipe busy time does not depend on how operations
were grouped into blocks, which is physically right. Sensitivity to the
partition lives in the schedule, and through it in the recurrence bound — also
right, since grouping changes where the barriers fall, not how much work exists.

### Synchronisation operations are charged zero cycles — on purpose

The stall a barrier causes is modelled by the schedule. Charging the
`sync_block_set`/`sync_block_wait` operations as well would count it twice.

---

## 8. The module estimate

The total is the larger of two bounds:

```
resource   = busiest core's total busy time
recurrence = max over dependency edges of (buffer occupancy / buffer depth)
total      = max(resource, recurrence)
```

### Why not `II * (N - 1) + latency`

That is the textbook modulo-scheduling formula and it was tried. It assumes the
pipeline reaches steady state, i.e. that buffering is deep enough to hide the
dependency chain. `BufferCountManager`'s inter-core default is **one**, so on a
kernel that alternates Cube and Vector nothing is hidden and the chain is paid
every iteration.

It also could not be composed correctly here: blocks run different numbers of
times, so `resource` and `II` end up describing different cores and
`steadyState = resource - II` is not `II * (N - 1)` of anything. On the
validation kernel `resource` was the Cube number while `II` was the Vector one.

### Why the recurrence bound is per edge

Each transfer has **its own** buffers. Iteration i+1 of a producer waits for
*its* consumer to release *its* buffer, not for the far end of the chain.
Charging a whole chain against a single buffer counts the same alternation once
per link, so a chain of k transfers is over-counted roughly k-fold — it
degenerates to the fully serialised bound as chains get longer.

### Why occupancy is measured on one iteration

Occupancy is `finish(consumer) − start(producer)` in the schedule of a **single
iteration**, so the result is a lower bound on the initiation interval. It is
scaled into whole-module units by `throughputBound / II`, which *is* the
iteration count expressed through two numbers already computed — no global N is
needed.

Measuring it on the loop-weighted schedule instead is wrong in a way that is
easy to miss, and an earlier version did exactly that. There each block occupies
its entire N-iteration duration before the next starts, so the span from a
producer's start to its consumer's finish covers most of the kernel. That is not
a buffer's lifetime: a buffer is held for one iteration's worth of
producer-to-consumer span and then reused. Measured that way the bound grows
with the trip count, so on a long enough loop it always overtakes the resource
bound, and the model starts crediting extra buffers with gains the hardware
cannot deliver.

Every kind of dependency counts: Cube→Vector, Cube→Cube, Vector→Vector. Only the
depth differs.

### Buffer depths

Read from the module, not assumed:

* `ssbuffer.intra_buf_count` — taken at face value.
  `AddMultiBufferInnerScope` creates exactly that many UB allocations from it.
* `ssbuffer.inter_core_buf_count` — **clamped to 2**.

The clamp is not a safety margin. `AddMultiBufferOuterScope` reduces the count
to `isDoubleBuf = (interCoreBufNum > 1)` and never looks at the number again, so
three buffers compile to exactly the IR that two do. Dividing by a larger number
would credit the schedule with an overlap the IR does not contain, and would
make the model prefer a configuration that compiles to something identical. When
a larger value was requested, the report says so.

`ssbuffer.load_store_buf_count` is read only to report it. GM-load prefetching is
applied by `DecoupleComputeAndMemory`, which the CV pipeline does not run
(`createDecoupleComputeAndMemoryPass` is defined but never called), so raising it
changes neither the IR nor the estimate. The report says that too, so it is not
discovered by experiment.

---

## 9. Output

### Module attributes

Declared in
`third_party/ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h`.

| Attribute | Type | Meaning |
|---|---|---|
| `ascend.cv_pipeline_estimated_cycles` | i64 | the estimate; lower is better |
| `ascend.cv_pipeline_cost_resource` | i64 | the occupancy bound |
| `ascend.cv_pipeline_cost_recurrence` | i64 | the dependency bound; the estimate is the larger of the two |
| `ascend.cv_pipeline_cost_roofline` | i64 | same module with unconditional Cube/Vector overlap; the gap is what barriers cost |
| `ascend.cv_pipeline_cost_hardware` | str | profile name; estimates are only comparable within one profile |
| `ascend.cv_pipeline_cost_ub_bytes` | i64 | Unified Buffer allocated, summed over allocations |
| `ascend.cv_pipeline_cost_ub_peak` | i64 | the most Unified Buffer live at one point of the schedule |
| `ascend.cv_pipeline_cost_ub_capacity` | i64 | what the profile says the part has |
| `ascend.cv_pipeline_cost_pipeline_depth` | i64 | the buffer rescaling factor (`requiredBuffers`), **not** the stage count |
| `ascend.cv_pipeline_cost_pipeline_stages` | i64 | predicated stages in the deepest loop's software pipeline |
| `ascend.cv_pipeline_cost_loop_extension` | i64 | iterations the rewritten bound added past the real trip count |
| `ascend.cv_pipeline_cost_unknown_ops` | i64 | operations whose own cost could not be computed |
| `ascend.cv_pipeline_cost_generic_ops` | i64 | operations charged by element count alone |
| `ascend.cv_pipeline_cost_dynamic_loops` | i64 | loops whose trip count no binding resolved |
| `ascend.cv_pipeline_blocks` | array of dicts | per-block results |
| `ascend.cv_pipeline_cost_quiet` | unit | **input**, not output: set by a variant trial to suppress the report |

Each block dictionary carries `id`, `core`, `cycles`, `iter_cycles`,
`work_cycles`, `iterations`, `iter_start`, `iter_finish`, `segments`, `ops`,
`bottleneck`, `depends_on`, `sync_depends_on`.

The two bounds are published separately because the total throws away which
constraint was slack, and the search needs that to order candidates that tie —
see the next section. The `ub_*` and `pipeline_*` figures exist for the same
reason: they are what the search's baseline guards compare.

Block results live on the **module**, not on the operations. A block is a set of
operations sharing an id, not an IR entity; stamping every operation would bury
the IR in annotations.

### Report

`TRITON_ASCEND_CV_COST_VERBOSE=1` prints a one-line summary,
`=2` adds:

* the derivation of the total, term by term, with `>` marking which bound won;
* the per-block schedule ordered by start time, so a critical path can be
  followed downwards through the `waits` column (`core` there means the block's
  own core was still busy, which is throughput pressure rather than a
  serialisation problem);
* the count of blocks with an interior barrier;
* the count of operations in mutually exclusive branches;
* both edge kinds;
* a breakdown by operation kind with the confidence of each, and the actionable
  lists of kinds worth teaching the model about.

`LLVM_DEBUG` prints the same under `-debug-only=estimate-cv-pipeline-cost`; the
environment variable exists because a debug build plus `-debug-only` is
impractical when driving compilation from Python.

---

---

## 10. Operation reordering and the variant search

The estimate exists to rank something. This section is that something.

### What varies

The pipeline is deterministic apart from one decision: the order the operations
of a block are linearised in. `ReorderOpsByBlockId`
(`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.cpp`)
makes it, and `ssbuffer.reorder_seed` on the module selects which legal order it
emits. Upstream the pass orders whole compute blocks topologically; the seed
makes it order *operations* under the same constraints, so a seed produces
genuinely different IR rather than a cosmetic permutation. Everything downstream
— where the block boundaries fall, where the barriers land, how deep the
software pipeline gets, how much Unified Buffer is live — follows from it.

The seed is consumed once. The run of the pass that sees it applies the
operation-level ordering, records what it used in
`ssbuffer.reorder_seed_applied`, and overwrites `ssbuffer.reorder_seed` with
`-1`; later runs of the same pass take the ordinary block-level path, so they do
not regroup underneath transfers and sync flags that already refer to block ids.
The two attributes are separate on purpose: the `-1` is a protocol between the
runs, while `reorder_seed_applied` is the record the cost report and the seed
table read. Both are stripped with the rest of `ssbuffer.*` before the module
leaves the pipeline.

`TRITON_ASCEND_REORDER_SEED` selects an ordering by hand, for reproducing one
candidate without running a search; `0` means the untouched pipeline, which is
the same thing the search calls seed 0. `TRITON_ASCEND_REORDER_POLICY=fifo`
switches which of the ready blocks is emitted first (last-ready by default).
Both rules are legal topological orders and only the result differs, so a
non-default choice announces itself: a run that silently ignored the variable
would be indistinguishable from one that honoured it.

### The search

`AddDynamicCVPipelinePass` runs the whole sub-pipeline once per candidate on a
private clone of the untouched input, reads the estimate back off each clone,
and then compiles the winner once more — so the module that survives and the
report that describes it are the same compilation, not two that happen to agree.

* **Cost.** One full pipeline run per candidate. That is why it is off unless
  `TRITON_ASCEND_CV_VARIANTS` asks for it.
* **Failures are not errors.** A candidate that trips a downstream assumption
  sets the fallback attribute; the search treats it as infinitely expensive and
  moves on. Rejections are counted by error code, because a search that kept
  nothing has to be distinguishable from one that was never asked to do
  anything.
* **Ties.** Two totals within 1/10000 of each other are called equal and decided
  on the bound the total did *not* come from. The resource bound does not depend
  on how operations were grouped, so once candidates reach it they score
  identically and the total has nothing left to say. Measured on flash
  attention: intra-core buffer depths 2 and 3 gave an identical total,
  recurrence bounds of 184 465 709 against 154 792 120, and 168 825 us against
  147 302 us on hardware — the second bound ranked them correctly where the
  total could not.
* **Parallel trials.** `TRITON_ASCEND_CV_VARIANT_THREADS` scores several
  candidates at once; the default of 1 is the serial search and stays the path
  to debug on. A trial's IR never leaves its thread, so the only shared state is
  the `MLIRContext`, and the one place a trial would still write to it — a
  `PassManager` loading dependent dialects — is pre-loaded before the fan-out.
  Threads additionally need the context's own multithreading enabled, which a
  pass cannot turn on from the inside; the triton patch therefore enables it in
  `python/src/ir.cc` exactly when `TRITON_ASCEND_CV_VARIANTS` asks for a search
  (see §12.6).
* **Trial output is swallowed** at the file descriptor and restored on the way
  out, unless the pass option `quiet-variant-trials=false` says otherwise. A
  trial drives the whole talkative pipeline; times two hundred candidates that
  is thousands of lines about IR which is being thrown away. Silencing pass by
  pass would not cover prints this pass does not own.

### Baseline guards

A winner the toolchain then refuses is worse than no winner. Two properties
predict that refusal, and both are compared **against seed 0** rather than
against an absolute, so nothing here is fitted to a kernel and the baseline
always passes its own test:

| Guard | Rejects a candidate that | Slack variable |
|---|---|---|
| Unified Buffer | holds more UB live than the baseline plus the allowance | `TRITON_ASCEND_CV_UB_SLACK_BYTES`, default `capacity - peak(baseline)` |
| Pipeline depth | needs a larger `requiredBuffers` factor than the baseline | `TRITON_ASCEND_CV_STAGE_SLACK`, default 0 |

The UB overrun is diagnosed by the binary compiler, in another process, after
this compilation has finished — so an unbuildable candidate looks perfectly
healthy here, and it is the *good* candidates that overrun, because the overlap
they win is paid for by materialising every value that crosses a new block
boundary. The absolute figures do not match the binary compiler's, but every
correction it applies applies to the baseline too and cancels in a difference;
what is left is the buffers the new cuts created, which this model does know.

The stage **count** is deliberately not guarded: it costs time rather than
correctness, and the estimate already charges the prologue and epilogue
directly, so a candidate that buys shorter dependency chains by adding stages
pays for them in its own score.

### What the search publishes

Four module attributes in the `triton_ascend.dynamic_cv_pipeline.*` namespace —
`chosen_seed`, `search_status`, `variants_usable`, `variants_total` — written on
**every** exit from the pass, including the ones that decide nothing. They are
not read by any pass: the Python driver copies them into the compilation
metadata and removes them, the way it already handles the return code, which is
why they are deliberately absent from `RemoveSsbufAttrPass` — that pass runs
inside the pipeline and would strip them before anyone could read them. They are
written at the very end because the fallback path replaces the module's whole
attribute dictionary from a backup.

`search_status` separates outcomes the seed alone cannot: no search was asked
for, a seed other than the baseline won, the baseline was still cheapest,
nothing could be scored, the seed was supplied by the caller, or the target was
not supported. Whether the pipeline then *survived* is a different question that
the return code answers, and the driver combines the two — see
`cv_search_status_name` in the next section.

`variants_usable` / `variants_total` are published because a winner means
something quite different when it beat 199 other orderings than when it was the
only one that compiled.

## 11. Configuration reference

### Cost model

| Variable | Meaning |
|---|---|
| `TRITON_ASCEND_CV_COST_VERBOSE` | `1` summary, `2` full report |
| `TRITON_ASCEND_CV_COST_HARDWARE_CONFIG` | path to a hardware profile JSON |
| `TRITON_ASCEND_CV_COST_ARG_BINDINGS` | e.g. `pid_x=0,num_programs_x=28,arg3=98432` |
| `TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT` | iterations assumed for unresolved loops |
| `TRITON_ASCEND_CV_COST_FUSION_CUBE` | intra-block fusion factor, Cube |
| `TRITON_ASCEND_CV_COST_FUSION_VECTOR` | intra-block fusion factor, Vector |

`DEFAULT_TRIP_COUNT` applies **per loop**, so two nested unresolved loops get
its square.

**Known limitation:** these are process-wide. They are fine for inspecting one
kernel but are the wrong channel for autotuning, where every configuration has
different sizes. Passing the values explicitly (pass option or module attribute)
is the intended replacement.

### Variant search

| Variable | Meaning |
|---|---|
| `TRITON_ASCEND_CV_VARIANTS` | how many operation orderings to score; 1 or unset is the ordinary single compilation |
| `TRITON_ASCEND_CV_VARIANT_THREADS` | how many candidates to score at once; a ceiling, capped at the candidate count |
| `TRITON_ASCEND_CV_UB_SLACK_BYTES` | extra Unified Buffer a candidate may hold over the baseline; unset derives it from what the baseline leaves unused |
| `TRITON_ASCEND_CV_STAGE_SLACK` | extra software-pipeline buffer factor a candidate may carry over the baseline; defaults to 0 |
| `TRITON_ASCEND_REORDER_SEED` | compile one ordering by hand, for reproducing a candidate; 0 is the untouched pipeline |
| `TRITON_ASCEND_REORDER_POLICY` | `lifo` (default) or `fifo`: which of the ready blocks `ReorderOpsByBlockId` emits first |
| `TRITON_ASCEND_REORDER_VERBOSE` | verbosity of that pass's own report; 0 is silent |

### Seed table

A search run over a suite of kernels has to answer one question per kernel:
did the cost model prefer some ordering to the untouched pipeline, and which.
That answer is recorded automatically and replayed on request.

**Recording needs nothing.** Every compilation that ran the CV pipeline puts
its decision into the compilation metadata, which triton serialises to
`<cache>/<hash>/<kernel>.json` beside each kernel and hands to
`triton.knobs.compilation.listener`. One file per kernel, written through a
temp file and a rename, so no two processes ever contend for one path and
there is nothing to merge.

`third_party/ascend/costmodel/tools/collect_cv_seeds.py` turns those records
into one table — standard library only, and it does not import triton, so it
also runs wherever the cache was copied to:

```bash
python third_party/ascend/costmodel/tools/collect_cv_seeds.py
```

```
status         seed  kernel_name     hash          cost       second_bound  ub_peak  pipeline_stages  usable  total
seed_found     7     attn_fwd        9f2c4e1a77b3  288578196  154792120     180224   3                82      200
seed_found     13    flash_attn_bwd  41bb90ce2f7a  91003120   70110200      131072   4                12      200
baseline_best  0     gemm_fp16       cc10ab7733d9  50221000   40118000      2048     1                200     200
ignored        -1    softmax_rows    77de01fa9b2c  -          -             -        -                0       0
```

Rows come out with the chosen orderings first and the expensive kernels before
the cheap ones, which is the order in which measuring them pays off.
`--only-found` keeps just the kernels where something other than the untouched
pipeline won, `--status` filters by outcome, and `--format jsonl|csv` writes
the table instead of printing it. The JSONL it writes is what the replay side
reads, so collect and replay are the same file with nothing in between:

```bash
python .../collect_cv_seeds.py --only-found --format jsonl --out seeds.jsonl
TRITON_ALWAYS_COMPILE=1 TRITON_ASCEND_CV_SEED_TABLE=seeds.jsonl pytest kernels/
```

| Metadata key | Meaning |
|---|---|
| `cv_kernel_name` | the kernel, as the table keys it together with `hash` |
| `cv_seed` | the ordering compiled; `0` is the untouched pipeline, `-1` none chosen |
| `cv_search_status` / `cv_search_status_name` | `no_search`, `seed_found`, `baseline_best`, `none_usable`, `from_table`, `wrong_chip`, `ignored`, `fallback` |
| `cv_variants_usable` / `cv_variants_total` | how many orderings were scored and how many could be |
| `cv_cost` | the estimate for the ordering that was compiled |
| `cv_cost_resource` / `cv_cost_recurrence` / `cv_second_bound` | the two bounds, and the one that decides ties |
| `cv_ub_peak` / `cv_ub_capacity` | Unified Buffer held live, against what the part has |
| `cv_pipeline_stages` / `cv_pipeline_depth` | software-pipeline stage count, and the buffer factor that predicts a refusal |
| `cv_rc` | the pipeline's return code; `0` is success |
| `cv_seed_requested` | the seed this compilation was told to use, or `-1` |

`cv_search_status_name` is the one field combining both halves of the outcome:
the search reports what it decided about the ordering, the return code reports
whether the pipeline then survived, and neither alone is the answer.

**Replaying needs the table.** Naming a file is what turns replay on; there is
no second variable for the mode.

| Variable | Meaning |
|---|---|
| `TRITON_ASCEND_CV_SEED_TABLE` | a table of seeds to compile instead of searching for them |

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_ASCEND_CV_SEED_TABLE=seeds.jsonl python run_kernels.py
```

JSONL or delimited text, by line, with `#` comments:

```
{"kernel_name": "attn_fwd", "hash": "9f2c...", "seed": 7}
kernel_name,hash,seed
attn_fwd,9f2c...,7
```

A kernel listed in the table is compiled with its seed and **no search runs**,
even if `TRITON_ASCEND_CV_VARIANTS` asked for one — a replay that re-derives
the seed is not a replay. A kernel that is not listed, or listed with no usable
seed, compiles exactly as it would with no table at all; both are logged,
because a caller who named a table is expecting a hit and a silent miss would
be indistinguishable from a replay. The table is re-read on every compilation,
so one edited during a run takes effect during it, and every lookup logs the
file's mtime — once the table can change mid-run, that is the only way to say
afterwards which version produced a result.

> **`TRITON_ALWAYS_COMPILE=1` is required**, for recording and for replay
> alike. Neither of these variables is cache-invalidating, so without it a
> kernel compiled once is served from the cache: no record is written, and a
> replayed seed is never applied. This is also what keeps the table's key
> stable, since the compilation hash is then the same in both runs.

### Hardware profile

`third_party/ascend/costmodel/configs/ascend_custom.json` describes the newer
target: it adds the `l0c:ub`, `ub:l1` and `l1:ub` data paths, the `fixpipe_ub`
mover, and the mutex-clique declaration.

> **The bandwidths for those three new paths are placeholders**, copied from
> comparable paths on an older part because no measurements were available. The
> status is written into the profile's `name` field so it prints on every report
> line. On the validation kernel `hivm.hir.fixpipe` accounts for ~31 % of all
> cycles, so this is the single most valuable number to replace. It is a JSON
> edit, no rebuild.

Mutex cliques express "these units share one physical pipeline and serialise".
Three questions are answered by adding a clique rather than by changing code:
whether `vec_mte2`/`mte3` really are a mutex on this target (inherited), whether
`fixpipe` and `fixpipe_ub` share a pipe, and whether the on-chip movers
`mte3_ub_l1`/`mte1_l1_ub` are truly independent of each other and of the
off-chip ones (currently modelled as independent, so UB can drain to L1 while
streaming to L2).

---

## 12. Changes outside the cost model

These are separable from the estimate itself and would stand on their own.

### 12.1 `DataDependencyAnalysis` publishes its block graph to the IR

`DataDependencyAnalysisPass` computes a block-level dependency graph
(`BlockInfo` plus v2c / c2v / memory dependency lists) and then throws it away:
the analysis is pass-local and holds raw `Operation *`.

Those pointers cannot be consumed later. Between `SplitDataflow` and the end of
the pipeline, `AllocMultiCache` and `AddControlFlowCondition` clone operations
and `UpdateLoopIterTimes` rewrites loop bounds. Worse,
`DataDependencyInfo::isInvalidated` returns `false` — the analysis declares
itself *never* invalidated, so MLIR would hand a later pass the stale object
rather than recompute it. It would not fail; it would quietly describe IR that
no longer exists.

A new step at the end of `runOnOperation` records the block-level edges on the
module under `ssbuffer.blockDeps`, as an array of `{producer, consumer, kind}`
dictionaries with `kind` in `v2c` / `c2v` / `mem`. Self-edges are dropped and
pairs deduplicated. Block **ids** survive cloning where pointers do not.

The pass is otherwise untouched: it adds one module attribute and changes
nothing else. If the attribute is absent, consumers simply have no dataflow
edges.

`c2cDependencies` is declared in the analysis but never populated, so it is not
persisted.

### 12.2 `RemoveSsbufAttrPass` strips the new attributes

Four attributes are added to the IR by this work and all four have to go before
the module leaves the pipeline:

| Attribute | On | Written by |
|---|---|---|
| `ssbuffer.blockDeps` | module | `DataDependencyAnalysis` (§12.1) |
| `ssbuffer.reorder_seed` | module | the variant search, and consumed by `ReorderOpsByBlockId` |
| `ssbuffer.reorder_seed_applied` | module | `ReorderOpsByBlockId`, late, so the report can still read it |
| `ssbuffer.iter_extension` | `scf.for` | `UpdateLoopIterTimes` (§12.3) |

`ssbuffer.blockDeps` is removed with an explicit `module->removeAttr` rather than
through the `kAttrsToRemove` array, because that array is applied inside
`module->walk` and whether a walk visits its own root is an MLIR detail this
should not depend on. The others go through the array; the lit test below is
what keeps that honest.

This matters downstream: there is precedent that hivmc rejects module attributes
it does not recognise — `compiler.py` deliberately strips `hacc.*` so they
"never reach hivmc". Coverage was added to
`unittest/Conversion/General/DynamicCVPipeline/test-remove-attrs.mlir`, with
`reorder_seed_applied` the one most worth checking: it is written last, by a
different pass, specifically so something downstream can still read the seed,
which makes forgetting to strip it the easy mistake.

The four `triton_ascend.dynamic_cv_pipeline.*` attributes the search publishes
are deliberately **not** in this list — see the end of §10.

### 12.3 `UpdateLoopIterTimes` records how it extended the loop

Software pipelining rewrites a pipelined loop's bound to

```
ceildiv(originalIterations * requiredBuffers, x) + ifCount
```

and the extra iterations are prologue and epilogue in which most stages are
predicated off. Nothing downstream can tell that from the loop alone: the
rewritten bound looks like an ordinary trip count, so anything asking how many
times a stage really runs would over-count it, and by a factor that depends on
the pipeline rather than on the kernel.

One `setAttr` at the end of `extendForOpIterationCount` records the three
numbers as `ssbuffer.iter_extension = [requiredBuffers, x, ifCount]`. The pass
is otherwise untouched; the cost model undoes the extension from these, and
publishes `pipeline_depth` / `pipeline_stages` / `loop_extension` out of them
(§9), which is also what the search's second baseline guard reads (§10).

### 12.4 `ReorderOpsByBlockId` learns a seed, and `SplitDataflow` runs it once more

The pass itself is upstream; the seed is not. With no seed it behaves as before,
ordering whole compute blocks topologically. With `ssbuffer.reorder_seed` set it
orders operations within the same dependency and memory constraints, which is
the one axis the variant search varies — §10 describes the protocol, including
why the seed is spent on first use and recorded separately.

`SplitDataflow` gains one more run of the pass, at the end, after
`RefineArgsBlockId`. By then the inter-core transfers and their sync flags refer
to block ids, so this run takes the block-level path by construction. **This is
the one default that moves** when nothing is configured: a module that sets no
seed still gets one block-level reorder that upstream does not do. It is listed
again in §13.

### 12.5 `Common/Utils.h`

Attribute-name constants for the four attributes above, the `SearchStatus` enum
and the four `triton_ascend.dynamic_cv_pipeline.*` names the driver reads, plus
`getKernelName(ModuleOp)` — used only to label log lines, because a cost report
in a log holding several hundred kernels is unusable without a subject. It
matches `func.func` / `tt.func` by name rather than taking the first symbol in
the body: a module can open with a `memref.global`, and labelling every report
with a buffer's name would be worse than labelling none.

`ssbuffer.inter_core_buf_downgraded` is declared here and read by the cost model,
which drops the inter-core depth to 1 when it is present, because
`AddMultiBufferOuterScope` compiles single-buffered when the double-buffer
groups would not fit in the flag budget while leaving the count attribute alone.
**Nothing in this branch writes it**, so that path is inert here; it is listed in
§14.

### 12.6 Triton patch: the context's multithreading flag

`third_party/ascend/patch/triton-ascend-3.6.0.patch` gains one hunk in
`python/src/ir.cc`. Upstream triton builds the `MLIRContext` with threading
disabled, so that async compilation does not spawn a thread pool per context.
A parallel variant search needs the context's uniquers locked, and MLIR refuses
to flip that flag from inside a pass — which is where this pass runs. The flag
therefore follows the search: the context is built with threading enabled
exactly when `TRITON_ASCEND_CV_VARIANTS` asks for more than one candidate, and
with it disabled, as upstream, otherwise.

`setup_patch.py` is untouched, so the patch has to apply cleanly as written.

### 12.7 Shared cost model additions

Small, additive changes to `third_party/ascend/costmodel`:

* **`HWUnit` enum** gained `FixPipeUB` (7), `MTE3ToL1` (8), `MTE1ToUB` (9).
  Values are **appended**, never inserted, so existing values stay stable and
  other branches working on the cost model are unaffected.
* **`PipelineScheduler::initPipelines`** registers all three. A unit without a
  pipeline is *silently dropped* by `schedule()`, so every `HWUnit` must appear —
  a trap worth knowing about.
* **`HardwareConfig::estimateVectorCyclesFromTable`** — cycles for one vector
  instruction from the migrated per-`(intrinsic, dtype)` table. Preferred over
  `estimateVectorCycles`, which charges every vector operation the same.
* **`PipelineAnalysisPass`** and the utilization report account for the new
  units. They are zero unless something upstream assigns them, so the standalone
  `-ascend-perf-model` pipeline behaves exactly as before.

### 12.8 Build integration

`third_party/ascend/CMakeLists.txt`: the cost model block was **moved above
`add_subdirectory(lib)`**. `include_directories()` and target definitions only
reach subdirectories added *after* them, so with the original ordering
`lib/DynamicCVPipeline` could see neither the cost model headers nor the
`AscendModelAnalysis` target. This ordering is a build requirement, not a
preference, and is commented in place.

`lib/DynamicCVPipeline/CMakeLists.txt` links `AscendModelAnalysis` and defines
`TRITON_ASCEND_HAS_INPROC_COSTMODEL` when the cost model is enabled; the pass
degrades to a no-op when it is not.

---

## 13. Known biases

Stated with direction, because a bias of known sign is usable and one of unknown
sign is not.

### What changes with nothing set

The search and the reordering are opt-in, but two things are not, and they are
the whole of what a build inherits from this work without asking for anything:

* **One extra reorder pass.** `SplitDataflow` runs `ReorderOpsByBlockId` once
  more at the end (§12.4). With no seed that is a block-level reorder, legal by
  construction, but it is IR that upstream would not have produced.
* **The estimate's compile time.** `EstimateCVPipelineCostPass` runs on every
  CV-pipelined module. It never mutates compute IR, never fails, and returns
  immediately if the pipeline has already fallen back — but it does build a
  schedule, so it is not free. It is not currently possible to switch it off
  separately from the CV pipeline itself.

Everything else — candidate scoring, the seed table, the threading flag in the
context — happens only when a variable or a pass option asks for it. The
attributes this work adds to the IR are all stripped before the module leaves
the pipeline (§12.2).

### Over-estimates

* **No operator fusion by default.** Each elementwise operation is a full pass
  with its own startup latency. The factors exist but are off and unmeasured.
* **Compounding.** Buffer occupancy is measured on a schedule that already
  contains the above, then divided by the buffer depth, so any pessimism in the
  schedule is carried into the recurrence bound as well.

### Under-estimates

* **Loop-carried dependencies are not in the graph.** The accumulator and
  running max/sum of an attention kernel cross iterations and form genuine
  cycles, which is where the longest recurrences live. Back edges are dropped,
  so the recurrence bound is a *lower* bound on the true one. This is the
  clearest remaining gap.
* **Scalar work is charged zero.** The old model's calibration constants
  (`aiv_scalar_overhead_factor`, `pipe_barrier_cycles_per_iter`) are deliberately
  not reused: they were measured on IR *without* the CV pipeline, so importing
  them would double count the sync overhead the schedule now models. Zero is
  not right either.
* **Blocks created after `DataDependencyAnalysis` carry no edges.**
* **Transfer size is taken from the destination**, so a masked or strided
  transfer may move more than is charged.

### Unknown direction

* the placeholder bandwidths (§11);
* whether the inherited `vec_mte2`/`mte3` mutex holds on this target;
* whether `fixpipe` and `fixpipe_ub` share a pipe;
* whether the on-chip movers are really independent.

---

## 14. Open questions

* **Is operation order the right axis?** It is the axis that exists (§10), and it
  is a real one: block boundaries, barrier placement and pipeline depth all
  follow from it. But the model only distinguishes orderings through the
  recurrence bound, so on a resource-bound kernel every ordering scores
  identically — which the report makes visible by marking the winning bound and
  by counting distinct scores. Buffer depths and block partitioning are the
  obvious further axes and nothing generates them yet.
* **`ssbuffer.inter_core_buf_downgraded` has no producer here.** The cost model
  reads it and would drop the inter-core depth to 1, which is the right answer
  when `AddMultiBufferOuterScope` ran out of flag budget and compiled
  single-buffered anyway. Nothing in this branch writes it, so the estimate
  still credits an overlap that may not be in the IR. Either the producer comes
  with the flag-budget work or the read should go.
* **No acceptance criterion exists.** For a ranking model the right metric is
  rank correlation against measurements, not absolute error. Nothing has been
  measured this way, so every improvement so far rests on argument rather than
  evidence. This is the highest-leverage next step.
* **Passing bindings during autotuning** — process-wide environment variables
  cannot work when every configuration has different sizes.
* **hivmc and the module attributes** — the four
  `triton_ascend.dynamic_cv_pipeline.*` attributes are read into `metadata` and
  removed by `record_decision`, which is the pattern that works. The
  `ascend.cv_pipeline_*` estimate attributes are **not**: they still travel down
  the compilation chain and have never been tested end to end. The fix if it
  bites is the same one, beside `_export_coalesce_metadata`.
* **Do the lit tests run?** `bin/RegisterTritonDialects.h` registers no ascend
  passes and the hand-written registration functions are never called, yet the
  existing tests apparently pass. The mechanism was never traced.
* **Report defect.** In the by-operation-kind table, `record()` assigns
  `stats.unit` unconditionally, so a kind split across units shows only the last
  one and the column does not add up.

---

## 15. Files

### New

| File | |
|---|---|
| `third_party/ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h` | pass and attribute declarations |
| `third_party/ascend/lib/DynamicCVPipeline/EstimateCVPipelineCost.cpp` | the whole model |
| `third_party/ascend/costmodel/configs/ascend_custom.json` | hardware profile for the newer target |
| `third_party/ascend/backend/cv_seed_table.py` | records the seed decision into `metadata`; reads a seed table back |
| `third_party/ascend/costmodel/tools/collect_cv_seeds.py` | turns the per-kernel records into one table |
| `third_party/ascend/costmodel/tools/cv_seeds_quickstart.md` | the shortest path from a suite run to a replayable table |
| `docs/en/dynamic_cv_pipeline_cost_model.md` | this document |

### Modified

| File | |
|---|---|
| `third_party/ascend/CMakeLists.txt` | cost model configured before `lib` |
| `third_party/ascend/lib/DynamicCVPipeline/CMakeLists.txt` | link and feature macro |
| `third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp` | pass insertion, the variant search, the seed report |
| `third_party/ascend/include/DynamicCVPipeline/Passes.h`, `Passes.td` | registration, and the search's pass options |
| `third_party/ascend/triton_ascend.cc` | the new pass options on the python binding |
| `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h` | attribute names, `SearchStatus`, `getKernelName` (§12.5) |
| `third_party/ascend/{include,lib}/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.{h,cpp}` | persist the block graph (§12.1) |
| `third_party/ascend/lib/DynamicCVPipeline/SplitDataflow.cpp` | one more `ReorderOpsByBlockId` run (§12.4) |
| `third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.cpp` | seed-driven operation ordering, ready policy, report (§10) |
| `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.cpp` | record `ssbuffer.iter_extension` (§12.3) |
| `third_party/ascend/lib/DynamicCVPipeline/RemoveAttributes.cpp` | strip the new attributes (§12.2) |
| `third_party/ascend/patch/triton-ascend-3.6.0.patch` | context multithreading for a parallel search (§12.6) |
| `third_party/ascend/backend/compiler.py` | pass options, seed lookup, record the decision |
| `third_party/ascend/costmodel/include/AscendModel/IR/AscendModelBase.td` | three new `HWUnit`s |
| `third_party/ascend/costmodel/{include,lib}/AscendModel/Analysis/HardwareConfig.{h,cpp}` | vector cycle table |
| `third_party/ascend/costmodel/lib/AscendModel/Analysis/PipelineAnalysis.cpp` | register new units |
| `third_party/ascend/costmodel/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp` | account for new units |
| `third_party/ascend/unittest/.../DynamicCVPipeline/test-remove-attrs.mlir` | cover the new attributes |
| `third_party/ascend/unittest/.../AddDynamicCVPipeline/fallback_existing_scope.mlir` | the seed decision a declined module reports |
| `third_party/ascend/unittest/pytest_ut/test_compiler_layout_memory_contract.py` | stub takes the pass's new options |
