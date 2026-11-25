# Invariant-cache planner and device-helper contract

The resource header `CVPCubeResourcePlan.h` and `LocalMmad.cpp` describe one
physical layout. The planner reserves storage supported by the helper; it does
not choose another chunking policy, bank count or fallback arrangement.

This document records the retained paired layout, the 0.6 function-lifetime
invariant-operand layout and their agreement checks. Source13 has passed build
and compiler qualification. All three ordinary Cube cache cases passed on NPU 1.

## Physical layout and chunk calculation

The supported target has 64 KiB each of L0A and L0B. The template splits each
into two 32 KiB ping-pong banks, uses 32-byte C0 packing and rounds reduction
chunks down to a multiple of 16 elements. The cached contract uses FP16
multiplicands, FP32 output and unit-flag mode zero.

`deriveTemplateReductionChunk` implements the existing device calculation.
Its row/column arguments have already received the template's L1-layout M/N
alignment. For each operand it divides bank capacity by element size and that
aligned dimension, rounds down to its C0 element count, then rounds the smaller
operand limit down to the matrix-fractal size. Both the planner and the actual
`L1Mmad` device implementation call this function.

The paired resource dictionary retains its existing neutral keys. If its
geometry is `(rows, cols, reduction_dim) = (R, C, D)`, the cached matrix's
actual `(M, K, N)` is `(C, D, R)`, and the co-resident matrix's is `(R, C, D)`.
Their chunks therefore use aligned `(M, N)` of `(C, R)` and `(R, D)`.

Each L0A bank remains the ordinary full streamed-operand bank. Each L0B bank
starts with the co-resident fragment and ends with the aligned cached operand
reservation. A cached operand must fit completely in one supported matrix
chunk. Every range must fit, and the two clients' ranges must be disjoint.
Checked multiplication/alignment rejects arithmetic overflow.

For `(R, C, D) = (128, 256, 64)` and FP16:

| Property | Retained value |
| --- | ---: |
| Cached matrix chunk | 64 |
| Co-resident matrix chunk/count | 128 / 2 |
| Co-resident B ranges | `[0, 16384)`, `[32768, 49152)` |
| Cached ranges | `[16384, 32768)`, `[49152, 65536)` |

`LocalMmad.cpp` checks the target capacities, C0 size, fractal size and two-bank
counter/event requirement with compile-time assertions. Ordinary non-cache
matrix calls retain their pre-existing full-capacity fallback when the normal
double-buffered chunk is zero. Cache eligibility requires a valid nonzero
double-buffered chunk and does not gain that fallback.

## Function-lifetime invariant operand

The independent function-lifetime contract uses `cached_clock = "function"`
and `tenant = "none"`. One or more actual Cube consumers must use the same
complete immutable B view, dimensions, transpose and effective K partition.
It supports a single Cube consumer without inventing a co-resident matrix.

The full invariant B must fit in one existing 32 KiB L0B half-bank, starting at
offset zero. `cached_slots = 1` describes that operand's storage, while
`counter_banks = 2` preserves the helper's ordinary two-bank streamed-A and
event protocol. This is a distinct lifetime proof, rather than a fallback for a
paired layout that does not fit. It uses the same chunk calculation; individual
K pieces use offsets `piece * chunk * cols * element_bytes` within the complete
reservation. Neither total L0 capacity nor the normal chunk policy changes.

Eligibility requires one complete ND2NZ initialization outside every loop,
dominating all consumers, with no overlapping or unknown writer. Physical
streamed-operand bank selections are checked over every bounded address
alternative; an unresolved or overlapping alternative rejects caching. An initialized
Boolean is threaded through the actual SCF loops and conditionals. The first
executed consumer copies B; later executed consumers reuse it. Skipped branches
and zero-iteration loops must preserve the actual initialization state. The
original matrix accumulator initialization remains unchanged.

Final readiness tracks issued and consumed MTE2-to-MTE1 events through the
actual SCF loops and conditionals. Completion may use an explicit wait or the
matrix helper's A-ready wait: the latter also completes the preceding invariant
B DMA on the same MTE2 pipe. Every consumer needs completed B initialization;
events cannot be consumed twice, overwritten while pending, leaked across an
iteration or disagree across branches. This closed proof requires resolved event
identities and does not infer a conditional helper B-ready protocol. Both
M-to-MTE1 bank credits and drains are verified separately. The typed resource
reader rebuilds this descriptor from target capacity and geometry; matching
attributes alone cannot substitute for the consumer and synchronization proofs.

## Ownership and synchronization are additional obligations

The cached/co-resident adapters share the actual `getL1MmadPingPongId` counter.
The streamed operand follows that counter; the cached operand uses its verified
generation-specific offset. Independent counters or merely matching offsets do
not establish a common bank sequence.

The compiler must prove invariance, initialization before use, lifetime through
the last read, bank/range ownership, selected-bank readiness and return credits.
It must also preserve required L0C/FIX completion, startup/tail predicates and
the exact drain signals. Capacity and compile-time layout assertions prove none
of those asynchronous facts by themselves.

Changes to the helper layout or reuse protocol require coordinated changes to
the resource derivation, `CVPInvariantCache.cpp`, typed matrix verification in
`HIVMMacroOps.cpp`, device adapters and final contract verification.

## Verification

`CVPCubeResourcePlanTest.cpp` owns the reference-only constant and its original
compile-time assertions, plus runtime capacity/range and planner/helper agreement
checks. The production header contains reusable types/calculations only.

A bounded local host check additionally compares 8,192 FP16 geometries with the
previous derivation and 486 chunk cases with the original device arithmetic.
These checks passed. The normal template checkpoint rebuilt all nine device
libraries. Resource/template sources have not changed since; the current
209-file package preserves their verified bytes. Source13 passed 112 focused
cases and the full native suite (950 passed, 140 unsupported and one expected
failure). The unchanged C++ unit binaries retain their 120-test pass.

Function-lifetime resource tests cover full-operand capacity and multiple normal
K pieces. Actual-IR tests exercise one/three consumers, conditional and empty
execution, transpose adapters, and mutations of writers, state carries, credit
events and resource fields. A proved-empty loop and its matrix call are removed
by conversion, preserving the no-execution case.

Source13's single-consumer, two-K-piece and three-consumer cases passed on NPU 1
in both uncached and cached modes, each with B changed for a second invocation.
For every cached case, both executed native artifacts match the normal-AST
compiler proof. The verified IR contains one/three actual cached consumers as
requested; the M256/K128/N128 case uses K chunk 64 and two pieces, filling the
existing 32 KiB B half-bank. Maximum absolute error was approximately 3.34e-6
for that case and 1.8e-7 for the other two. The broader history matrix also passed. The production timing results appear in section 6.1.2 of the [design](en/cvpipeline_flash_attention_design.md). The review work copy
tracks the current results without extending an earlier pass to unfinished cases.
