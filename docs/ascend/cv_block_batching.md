# Manual CV block batching: independent lanes and spatial concatenation

This opt-in extension integrates with the unified block-batching effect plan.
CV does not choose a factor automatically.

```python
compiled = kernel[original_grid](
    *args,
    block_batching_cv=True,
    block_batching_axis=1,  # a proven M-tile axis; use the kernel's actual axis
    block_batching_factor=4,
)
print(compiled.block_batching)
print(compiled.last_launch)
```

Pass the **original grid**, never a pre-divided grid. The existing guarded bundle
owns the grid conversion, three original-grid hidden arguments, binary, workspace,
and launch metadata as one complete artifact. Factor 1 compiles the original
kernel. Valid CV factors are powers of two from 2 to 64; backend resource admission
may reject any of them. The batch-matmul option selects the existing backend path;
it is not a new automatic selection mechanism.

See [block batching configuration and limits](block_batching.md) for
launch policies, diagnostics and test commands.

## Semantics and supported domain

The compiler chooses a lowering from IR dependence on the requested axis;
there are no Liger names, shape tables or host-side tile overrides in the pass.
Floating inputs may be FP16, BF16 or FP32 with FP32 accumulation. Signed INT8
inputs with INT32 accumulation are also admitted. Integer M/N concatenation
retains the original K extent/order, bit widths and accumulation operations;
it neither converts the dot to floating point nor combines reductions across
programs. Device validation
must be reported separately for the actual precision/shape tested. The original
memory-domain, integer-overflow, alias and exclusive-store guards remain required.
CV uses the same v3 source/plan fingerprint as VV. The independent masked-batch
path additionally checks its original-coordinate padding and reduction
obligations. VV's terminal-read relaxation for opaque device addresses does not
authorize a CV matrix transformation: those addresses are rejected for CV.

### Integer scope

In CV mode, byte-valued loop state and lane-wise integer add/subtract/multiply,
bitwise and/or/xor, signed/unsigned extension and truncation retain their
original semantics. Existing uniform-control, pointer-provenance, bounds,
alias and attribute checks remain required. Overflow promises (`nsw`/`nuw`),
general byte division/reductions, byte scalar host ABI and FP/integer byte
conversions are not enabled by this change. VV byte admission is unchanged.

The current official consumer cannot link
`_mlir_ciface_batch_mma_tile_int8_t_to_int32_t`. Independent integer matrix
pairs therefore use `static_integer_lanes_v1`: F independent rank-two dots
inside one grouped program, with the original masks and accumulators.
This is program grouping, not a single batch MMA or an M/N concatenation.
Uniform loops retain `static_uniform_control_lanes_v1` with
`CV_INTEGER_BATCH_MATMUL_UNAVAILABLE` as the fusion reason. The new static
integer report carries an owned rank/type contract and shares the verified
original-grid AutoBlockify policy. Shared-operand integer M/N paths still
produce one wider rank-two dot.

Supporting INT8 types does not establish new proofs for dynamic grouped-pid
division/remainder or nested pointer recurrences. Such kernels retain their
complete baseline when the existing address analysis cannot prove them.

### Shared B: concatenate M; shared A: concatenate N

For a single rank-two dot, the `spatial_concat_v1` path proves
one canonical coordinate `pid(axis) * T + arange(0, T)`. All uses of that program
ID must enter through this coordinate. It rewrites the coordinate to
`group * (F*T) + arange(0, F*T)` and propagates the expanded tensor dimension
through addresses, loads, structured masks, transpose, dot and elementwise
operations. `spatial_m_v1` requires only A's M axis to expand; `spatial_n_v1`
requires only B's N axis. The shared operand and K extent/order are unchanged.
The initial accumulator must be constant zero.

The same spatial proof now applies inside supported uniform `scf.for` and
`scf.if` regions. The loop's bounds and a branch's condition must be independent
of the selected program axis. The expansion analysis reaches a fixed point for
each loop's initial values, iter_args, yielded values and results, rather than
resizing just the dot. Such rewrites are reported as
`spatial_m_uniform_control_v1` or `spatial_n_uniform_control_v1`.

The proof is ordinary block matrix multiplication:

```
concat_rows(A0, A1) @ B = concat_rows(A0 @ B, A1 @ B)
A @ concat_cols(B0, B1) = concat_cols(A @ B0, A @ B1)
```

Output stores cover the same disjoint coordinates as the original F programs.
A shared `arange` definition is never resized globally: it may also feed K or
the unchanged matrix. Uniform tensors are resized per use only when proved
invariant along the expanded dimension (splat, singleton broadcast, or proven
pointwise/transpose compositions). Independence from `pid` alone is insufficient.
For example, an epilogue bias indexed by the original *local* M coordinate is
currently rejected, rather than silently interpreted as a larger global bias.

Boundary memory masks retain their exact expressions and zero load alternatives.
A loaded data mask in an elementwise epilogue, such as the QK neighborhood select,
is retained. Data-dependent memory predicates remain outside the admitted domain.
The first spatial implementation rejects scalar uses escaping the coordinate,
K-axis expansion, multiple dots, unproved reshape/reduction/scan, nonzero Acc,
and stores that do not concatenate. Rejection leaves the complete input IR intact.

### Independent matrix pairs

With `enable_hivm_batch_matmul=True`, `independent_batch_v1` stacks independent
matrix pairs into `[F,M,K] @ [F,K,N] -> [F,M,N]`. The added batch dimension never
becomes part of K, so each output lane depends only on its own input pair.
`rank3_masked_batch_v1` retains rectangular, zero-padded load masks and the
original output masks. It does not require every memory mask to be true.
Inside supported uniform regions this is reported as
`rank3_uniform_control_batch_v1`; each loop-carried accumulator has an
independent batch dimension. Neither the original reduction order nor the
number of iterations changes.

For supported prefix-masked, partially structured loads, the memory converter
zero-initializes the complete destination, then writes each lane's valid GM
data directly into its destination subview. This avoids an
intermediate L1-to-L1 copy, which the packaged backend cannot lower. Transposed
operands are loaded as a contiguous batch and then transposed as a whole.

The generated dot carries an internal `tt.cv_batch_full_mkn` marker. Lowering
passes its complete logical M/K/N tensor extents directly to `batchMmadL1`, with
`already_set_real_mkn`. The packaged consumer otherwise tries to infer those
extents from nested subview writes and can leave its zero placeholders intact.
These dimensions describe the mathematical, padded tensors; they do not assert
that a GM access mask is true. Input precision and accumulation semantics are
preserved. Unmarked rank-three dots keep their existing lowering.

Admission remains conservative: K must be fully present, and both physical
operand axes must retain the same final 32-byte padding extent as the complete
tile (quantum 8 for FP32, 16 for FP16/BF16). Empty or more distant tails select
the full baseline before launch. This is a backend-layout restriction in addition
to ordinary bounds and alias checks; arbitrary zero-padded K tails are not yet
supported. Numerical validation is required for each admitted layout and dtype.

For a directly decoded program ID `p = g*F + u`, `0 <= u < F`, a constant
positive head period `H` divisible by `F` permits the compile-time identities
`p // H = (g*F) // H` and `p % H = (g*F) % H + u`. The rewrite keeps division
scalar and preserves unit stride across lanes. The existing full-group and
signed-range proofs exclude overflow; no new runtime guard is required.
This avoids discontinuous tensor div/rem address lowering and its per-lane
assembly before batch MMA. If `F` does not divide `H`, the group may cross a
period boundary: batch fusion reports `CV_BATCH_QUOTIENT_GROUP_CROSSES_PERIOD`
and supported uniform regions retain the independent rank-two lane fallback.
The original masks, coordinates and byte-domain proof are unchanged.

The proof recognizes a block pointer's signed `x >= 0` predicate as `-1 < x`
without changing the device mask. This normalization is restricted to a zero
right operand. The lower bound must hold over the complete original domain,
with checked integer arithmetic; it cannot trim a negative prefix to justify
the padding proof. If one load supplies both dot operands, all reduction-axis
obligations are combined in that load's original coordinates. Auxiliary
rank-one accesses do not acquire an artificial rank-two reduction obligation,
but still require their ordinary memory and mask proofs.
If JIT specialization makes a prefix bound constant, a proven violation of these
same limits is rejected before calling the backend. This prevents a statically
ineligible `1x1x1` candidate from reaching unsupported ND2NZ lowering before its
runtime guard can run. Dynamic arguments and grid values are never guessed by
this early check; their existing launch checks remain mandatory.

Without the batch-matmul option, structured masked memory still uses
`static_masked_lanes_v1`, cloning the rank-two body F times. That path must not be
reported as a hardware batch-matmul optimization.

### Uniform control flow and fallback

The spatial and independent-batch proofs are attempted on a private copy of the
normalized module. If fusion cannot be proved, the original program's complete
regions can still be cloned per lane, reported as
`static_uniform_control_lanes_v1`. `cv_loop_fusion_rejection` records why genuine
fusion was rejected. This fallback retains each original two-dimensional dot;
it must not be counted as M/N or hardware batch fusion.

For canonical dot-accumulator and pointer recurrences, including supported
uniform branches, the original `scf.for` already preserves the zero-trip initial
result. An additional result-selecting `scf.if` is omitted for these recurrences
to avoid an unnecessary cross-region accumulator transfer in the consumer.
Invariant or unclassified yielded results keep the explicit zero-trip
workaround. Zero-trip semantics are never removed, and VV's existing rule is
unchanged.

For an exact pointer backedge `ptr_next = addptr(ptr, constant_splat)`, the
independent-batch loop rebuilds each iteration's address from its original
initial pointer and the scalar iteration count. The cumulative displacement
uses i64, preserving the sign extension of each original i32 pointer step.
This exposes the existing strides to memory lowering instead of carrying a
rank-three address tensor. Original masks, loop bounds, zero-trip behavior and
byte-domain proofs remain; unknown or conditional pointer steps keep their
original representation.

Induction gives the loop proof: if the accumulated state at iteration `t` is
the concatenation (or independent stack) of the original states, the unchanged
K-step dot produces that same relation at iteration `t+1`. The common loop
bounds and branch condition preserve the sequence for every lane. The zero
initial state establishes the base case; disjoint final stores establish the
same observable outputs.

### Common limits

CV requires `original_grid[axis] % F == 0`. A partial **group of programs** selects
the entire baseline. An original program's partial M/N/K tile can still qualify
for the spatial path when its original structured masks and memory guard prove
it safe. These are distinct conditions; no workload is rounded or omitted.

The masked independent-batch and shared-operand M/N spatial paths can use backend AutoBlockify after the
guarded CV grid transform. The launcher first maps the original grid to grouped
programs, then caps physical blocks; the backend loop visits every grouped
program at a stride of the physical core count. The original-grid hidden
arguments remain unchanged. This combination is admitted only for MIX mode,
an empty compiler-derived AutoBlockify blacklist and no competing persistent
cap. The shared allowlist includes `rank3_masked_batch_v1`, `spatial_m_v1`,
`spatial_n_v1`, their uniform control-flow fusion counterparts, and
`static_uniform_control_lanes_v1`.
Other CV lowerings retain their prior AutoBlockify exclusion. Correctness and
performance still require a device test with more grouped programs than physical
blocks; enabling the flag on a smaller grid does not exercise the loop.

Dynamic loops and conditions can qualify when uniform relative to the selected
axis and accepted by the recurrence and memory analyses above. Varying control
flow, dot-scaled, pre-batched rank-three dot, atomics and cross-program
synchronization remain outside this CV path. Independent batch fusion still
requires rectangular masks uniform along the selected axis and traceable
dot-input coordinates. Multi-dot graphs or transformed inputs that cannot be
traced may use the complete-lane fallback or reject. VV memory-view folding is
disabled for CV. CV owns logical grouping; AutoBlockify schedules the grouped
programs only in the admitted paths above.

The byte-domain proof permits positive strided layouts with holes when their
integer bounds, address ranges and exclusive writes are proved. A uniform CV
loop bound may depend on other grid axes; a conservative whole-grid envelope
covers those iterations. This can still reject a safe case when the envelope
overlaps another range. It does not authorize dependence on the selected axis,
negative strides, unchecked address overflow, or overlapping writes.

## Admission and failure

The baseline compiles independently first. Before accepting a CV binary, validate
its hash, matching native implementation, MIX/AIC mode, exact v3 grid contract,
workspace, task type, absence of conflicting mappings and global lock use.
Normal backend compilation admits resource usage. Recognized UB, L1/cbuf and
L0A/B/C (`ca`, `cb`, `cc`) overflow selects baseline before submission, preserving
the diagnostic. There is no runtime replay after device execution starts.
Unknown internal compiler errors remain errors and must be investigated.

Compilation success or baseline fallback is not optimization evidence. An
accepted measurement requires `last_launch.runtime_status == 'APPLIED'`, the
requested factor, numerical comparison and uncontended device timing. Compare
baseline and candidate in adjacent interleaved rounds on identical inputs.

The packaged BishengIR backend is unchanged; this revision changes TA's rewrite
and lowering into its existing operations. Tests and performance results are
scoped to the registered environment and must not imply all CV kernels work.
