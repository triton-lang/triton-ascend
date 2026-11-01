Experimental interleaved load packing
====================================

Problem and intended behavior
-----------------------------

An update kernel may read several fields from each row separately. For a
row width W=4, the addresses are ``base + 4*i + field``. Separate field
loads expose strided accesses and repeated conversion work. This pass
forms one contiguous row tile and extracts its columns. It is a TTIR
transformation, independent of the kernel name and batch/channel sizes.

Enable with the Ascend compilation option
``enable_interleaved_access_packing=True``. The default is False. Each
compilation transforms the current module in place and emits one kernel;
there is no runtime pointer test, dispatcher, or alternate executable.

Implementation guide
--------------------

* ``third_party/ascend/backend/compiler.py::make_ttir`` runs the optional
  pass after the normal TTIR pipeline, followed by canonicalization and CSE.
  ``NPUOptions`` makes the choice part of compilation options.
* ``third_party/ascend/triton_ascend.cc`` exposes the pass to Python.
* ``third_party/ascend/lib/TTIRLayoutMerge/PackInterleavedAccess.cpp`` contains
  all matching, legality checks and rewriting. Its standalone CMake target
  does not require the experimental costmodel/layout-merge implementation.
* ``group`` recognizes loads that can move to the first load. ``sinkingGroup``
  recognizes loads that can move to the last load. ``candidateGroup`` tries
  these in that order. Failed matching leaves the original operations intact.
* ``rewriteGroup`` broadcasts row bases to [N,W], adds field indices, loads
  a tile, and uses ``splitColumns`` to reconstruct the original vectors.
  FP16-to-FP32 extensions are exact and can be performed on the tile before
  splitting. Floating-point arithmetic is not reassociated.

For example, three FP16 cache loads of fields 1,2,3 can become::

    before: load(rows + 1, mask), load(rows + 2, mask), load(rows + 3, mask)
    after:  tile = load(rows[:, None] + [0,1,2,3],
                       mask[:, None] & ([0,1,2,3] != 0))
            split(tile)[1], split(tile)[2], split(tile)[3]

Field 0 remains masked: the transformation does not add an active read.
Every original store remains at its original memory-order position. Store
packing is deliberately absent from this implementation.

Safety argument and rejection conditions
-----------------------------------------

The matcher accepts static, unencoded rank-one FP16/FP32 pointer tensors
with 1..8192 rows and power-of-two row widths 2,4,8. Consecutive fields must
share the same SSA row base, mask, fallback value, pointer type and load
attributes. Volatile operations, regions and unknown memory effects stop
matching. The load's cache/eviction policy and mask/fallback semantics are
preserved. Unsupported layouts are unchanged.

For hoisting, a crossed write must have the same SSA row base and a field
outside the remaining fields being moved. Because row offsets are multiples
of W, different fields have different residues modulo W. Merely sharing a
kernel argument name is never sufficient evidence.

For sinking across different pointer arguments, ``alignedPointer`` requires
an existing entry-argument ``tt.divisibility`` contract sufficient for W
elements. ``multipleOf`` proves intermediate offsets preserve this alignment.
Only bounded-depth constant/multiply/add/extension/splat reasoning is used.
For element size S, accepted row bases are 0 modulo W*S bytes. Field f is
therefore f*S modulo W*S. A crossed store must be outside the interval of
already collected fields ``[firstField, nextField)``. These residue classes
are disjoint even if the base pointers are equal or overlap. Power-of-two W
preserves the residue argument under integer wraparound. No ``tt.noalias``
attribute is assumed, added or consulted. As with other compiler passes,
frontend-provided alignment contracts must be truthful.

``sinkUsers`` additionally checks the transitive consumers that would need
moving: they must stay in the same block, have no regions and be memory-effect
free. They move in original block order after the last load. A consumer that
stores a value before that point rejects the transformation. This protects
both SSA dominance and the load/store dependence order.

FP16 sinking packs a complete row; FP32 sinking is limited to the final
aligned field pair. Hoisting can also pack larger aligned FP32 groups when
the structural matcher permits them. This is a conservative layout policy,
not a list of benchmark shapes. Other valid layouts may receive no benefit.
The rewrite reduces the number of rank-one loads, so greedy matching
terminates; a second invocation must not change the canonicalized output.

Impact and limitations
----------------------

The default compilation path is unchanged. Enabling the option may alter
memory instruction shape, register pressure and downstream scheduling for
any matching Ascend kernel, including kernels inside control-flow regions
whose individual blocks match. The pass does not move operations between
blocks. It is not an inter-program weight cache and does not eliminate
communication or add synchronization. Programs with existing data races are
not made safe by this transformation. No universal speedup is claimed.

Validation evidence
-------------------

The standalone C++ pass was built and exercised with verifier checks. The
portable regression is::

    python test/Triton/pack_interleaved_access.py /path/to/triton-opt \
      test/Triton/pack-interleaved-access.mlir

It covers missing/insufficient alignment, misaligned rows, conflicting
stores, memory consumers, matching and mismatched masks, volatility,
non-power-of-two strides and both forms of noalias metadata. Every case
checks load/store counts, single-function output and idempotence.

Hardware exploration used CANN 9.2 beta.2 and fixed launch configurations.
Ten original update shapes passed three seeds with bitwise baseline output,
exact cache comparison and a CPU-reference RMS threshold of 1e-3. Twelve
additional shapes were exercised with and without genuine alignment
contracts, including tails, padded strides and widths 2,3,4,8. Forty pointer
pair/offset combinations plus two padded-stride cases passed for each dtype.
The overlap tests used one program/warp to avoid pre-existing inter-program
races; they do not exhaust all possible alias relationships.

Frozen, independent five-round measurements gave mean historical L20
latency / NPU MSPTI kernel latency 0.30562394 before and 0.31938034 after
(+4.50% in this ratio). A three-round selection run reached 0.32019200;
the independent result does not establish a stable 0.32 target. L20 values
were historical, not remeasured. This metric excludes host dispatch.

Each timing segment used 80 launches, discarding the first 50. Baseline and
candidate were interleaved with bracketing baseline measurements; accepted
rounds required CV <=15%, bracket drift <=10% and spread <=10%. Launch
parameters were held fixed. The largest tested shape changed from
16.12447 us to 15.18445 us. Ten inspected ELF files each exposed one global
kernel function, with no runtime pointer dispatch in their TTIR.

Merge readiness
---------------

This is intentionally a draft. The hardware results came from the standalone
pass and the experimental compiler toolchain, not a wheel rebuilt from the
latest upstream branch. The complete upstream wheel, Python binding/JIT
integration, full repository suite and additional hardware configurations
still require validation before merge. The cleaned and formatted PR pass was rebuilt, reran all 11 regressions,
and produced byte-identical canonicalized TTIR to the frozen measured pass
for all ten original shapes. Python AST/mock checks confirmed both option
branches return the same module and only the enabled branch adds this pass.
These checks do not load the rebuilt Python extension.

Structural regression success and
the static legality argument are complementary evidence, not a substitute
for those integration checks. Keep the option disabled by default until
review and integration validation are complete.
