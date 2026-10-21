# SplitIfByBlockId (ComputeBlockOpt)

Pass entry: `--split-if-by-block-id`
Source: `third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/SplitIfByBlockId.cpp`
UTs: `third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/`

## 1. Role & pipeline

Splits a single `scf.if` whose `then`/`else` regions mix CUBE and VECTOR
`ssbuffer.block_id` groups **with bidirectional cross-core data flow** into a
chain of single-`block_id` `scf.if`s (a split-if chain). Cross-group SSA values
are forwarded through the split-if results. Stage order in `runOnOperation`:

1. `getCandidate` — `groupOpsInBlock` per region, then
   `hasBidirectionalCrossCoreDeps` (SSA + memory-alias deps).
2. `preprocessScalarDependencies` — `ScalarClosure` clones scalar/index
   computation into each group.
3. `analyzeDependencies` — `scanRegion` builds the value-level cross-group map,
   `planYield` plans per-group yield signatures (Case A = no original yield,
   Case B = original yield present).
4. `materializeCandidate` — builds the split-if chain.
5. `rearrangeIfOp` — moves each split-if after its last dependency.
6. `hoistAndUnifyRedirectedOutput` — removes Case-B pass-through results.

## 2. `hoistAndUnifyRedirectedOutput`: old vs new

### Why it exists
In Case B the last split-if (`lastIf`) carries **all** original result slots.
A slot whose `then` operand is produced by an *earlier* split-if is a redundant
pass-through ("jump reference"). The hoist lets consumers read the earlier
split-if result directly and **back-propagates the original else value into that
earlier if** ("unify"). This removes the 二传手 `scf.if` result that downstream
provenance analyses (`AnalyzeSplitIf`, `InterCoreTransferAndSync`) dislike.

### Old implementation defects (commit `efbcb27c9` prep)
1. **Case B erases live bodies.** `hoistedCount == numResults` called
   `rewriter.replaceOp(ifOp, ...)` unconditionally, deleting `lastIf` and any
   real work still in its then/else body (e.g. an `arith.mulf`, `memref.store`,
   DMA).
2. **Scenario 1 drops real else values.** When `elseVal` was defined inside the
   if it was treated as a disposable placeholder. But `ensureLocalValue`
   clones real `ViewLikeOpInterface` ops into the else block, and Scene 3/4
   absorbed the other side's real ops into `lastIf.elseBlock()`; both are
   internal yet must not be dropped.
3. **Scenario 2 patches the upstream if blindly.** `upstreamIfOp.elseYield()`
   was rewritten without checking same-batch, same-condition, dominance or type,
   which can produce invalid SSA.
4. **Case C loses attributes.** `kCoreType` was set on the *old* `ifOp` (erased
   immediately after) and the rebuilt `newIfOp` inherited nothing, so
   `ssbuffer.splitted_if` / `ssbuffer.block_id` were lost.
5. No arity / type guards on the yields.

### New (hardened) implementation
A safety gate runs **before any mutation**. If it returns false the function
returns immediately and leaves the IR untouched; the pass-through form is still
valid IR (just less optimal). This is intentional: safety over optimisation.

- `canSafelyHoistRedirectedOutput(ifOp, thenYield, elseYield)` (~L1832) —
  arity/type checks, then for each hoistable slot:
  - `thenVal` internal → slot kept, nothing to check.
  - `elseVal` internal → only accepted when `isDisposablePlaceholder`.
  - `elseVal` external → `canSafelyRedirectToUpstreamIf` must accept.
  - Case B (all hoisted) additionally requires `!ifBodiesHaveRealWork`.
- `isDisposablePlaceholder(val, ifOp)` (~L1740) — only `arith.constant`,
  `tensor.empty`, `linalg.fill`, `memref.alloc(a)`, or a
  `memref.reinterpret_cast` whose source is itself disposable. ViewLike clones
  and absorbed real ops are rejected.
- `canSafelyRedirectToUpstreamIf(...)` (~L1773) — same enclosing block,
  same condition, same `ssbuffer.splitted_if` tag; valid result index; type
  match; `elseVal` `properlyDominates` the upstream else-yield; the existing
  upstream else operand must be `elseVal` or a disposable placeholder.
- `ifBodiesHaveRealWork(ifOp)` (~L1806) — Case B refuses to erase a body that
  holds anything other than terminators/placeholders.
- Case C now does `newIfOp->setAttrs(ifOp->getAttrs())` before overriding
  `kCoreType`, so `block_id`/`splitted_if` survive.

## 3. Verified UTs

`hoist_redirected_output.mlir` (the UT for `hoistAndUnifyRedirectedOutput`;
passes with the rebuilt `triton-opt`):

- `@hoist_caseb_live_body_bails` — Case B where `lastIf` still holds a live
  `arith.mulf` (block_id 96 but yielded value comes from group 95). The gate
  must bail; the test asserts all three split-ifs survive and the mulf is kept.
- `@hoist_casec_redirects_result` — Case C partial hoist. Verifies the positive
  path actually redirects: the loop yields the group-95 split-if result (`%M`)
  directly, the rebuilt group-96 if has a single result whose body consumes
  `%M`, and the rebuilt if keeps `ssbuffer.splitted_if = 0` with a recomputed
  `core_type = "VECTOR"` (attribute inheritance fix).

Discrimination check: temporarily commenting out the
`hoistAndUnifyRedirectedOutput(ifOp);` call makes `@hoist_casec_redirects_result`
fail (pre-hoist the group-96 if has 2 results and `core_type = "CUBE, VECTOR"`),
confirming the test really exercises the hoist rather than just rearrange.

## 4. Latent issues (NOT fixed — only recorded)

These were identified while auditing but deliberately left untouched; verify
before relying on them.

- **`buildElseYieldForGroup` (~L1257) arity bug** — `continue;` sits outside the
  `if (valueDominates(...))` guard. When the original else value does not
  dominate, no operand is pushed *and* the placeholder fallback is skipped,
  producing an `scf.yield` with the wrong arity (verifier crash). Fix: move
  `continue;` inside the guard. Symptom → Cause → Fix → Scope: only triggers
  when a Case-B else value fails the (already weak) dominance check.
- **`valueDominates` (~L1214) is wrong** — returns true for any `BlockArgument`
  and only checks block ancestry, not op order inside the block. Prefer
  `mlir::DominanceInfo::properlyDominates`.
- **`planYield` (~L783) non-determinism** — sorts cross-group values by
  `Value::getAsOpaquePointer()`; under ASLR the result order (and thus the
  split-if result order) can change between compiles. Use a `discoveryOrder`
  recorded in `scanRegion` and sort by `(fromGroupIdx, discoveryOrder)`.
- **`getCandidate` mutates IR before deciding** — `groupOpsInBlock` calls
  `op.setAttr(ssbuffer.block_id, ...)` on ambient ops even when `needsSplit()`
  later returns false. The analysis is not read-only.
- **`MemoryDependenceGraph` dangling pointers** — in `runOnOperation` the graph
  holds raw `Operation*`; `hoistAndUnifyRedirectedOutput` erases `ifOp` and may
  create `newIfOp`. A later `rearrangeIfOp` can then read a dangling op and miss
  the new one. Rebuild the graph after hoisting, or hoist all ifs first and
  rebuild before rearranging.

## 5. Debug / dev notes

- **FileCheck `CHECK-COUNT-*` is sequential.** A preceding
  `CHECK-COUNT-3: scf.if` advances the scan position past the closing `} {...}`
  lines, so a later attribute-count check misses the earlier matches. Interleave
  per-op / per-tag checks in source order instead.
- **Build:** always use `bash ~/personal/build_ta.sh &> build/build.log`. A raw
  `ninja triton-opt` inside `build/cmake.linux-x86_64-cpython-3.11/` can fail
  with `no member named 'CopyOpInterface' in namespace 'mlir'` when the build
  dir configuration is stale; the script's CMake reconfigure supplies
  `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__` (needed by the vendored
  `bishengir/Interfaces/CopyOpInterface.h`). Always run
  `python3 restore_sources.py` afterwards.
