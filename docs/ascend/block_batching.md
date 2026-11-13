# Block batching

Block batching groups independent Triton programs, or supported iterations of
a uniform loop, without changing the kernel's input interface. It is opt-in and
implemented in the Ascend frontend, compiler and launcher. It does not require
changes to the packaged BishengIR compiler.

## Configuration

```python
compiled = kernel[original_grid](
    *args,
    block_batching_factor=4,
    block_batching_axis=0,
)
print(compiled.block_batching)  # compile-time decision and rejection reason
print(compiled.last_launch)    # variant actually submitted for this call
```

Always pass the original grid. The launcher owns grouping and preserves the
original `tl.program_id` and `tl.num_programs` semantics. Factor 1 uses the normal
compiler path. Positive factors up to 64 are candidates, subject to proof and
backend resource limits. Dot-containing kernels automatically use CV batching,
which requires powers of two. An explicit axis can override automatic selection.

| Option | Default | Meaning |
| --- | --- | --- |
| `block_batching_factor` | `1` | Requested grouping factor; no automatic factor search. |
| `block_batching_axis` | `-1` | Original program axis, or compiler-selected axis for VV. |
| `block_batching_layout` | `"auto"` | Try proved memory-view folding; `"semantic_only"` retains lane grouping. |
| `enable_hivm_batch_matmul` | Backend default | Allow independent CV pairs to use the backend batch-MMA path. |

`TA_BLOCK_BATCHING_CORE_FLOOR` overrides the minimum grouped task count; zero
disables that scheduling policy. By default the policy uses the available vector
core count after the normal device-limit validation. A request that cannot
reduce the program count, or would fall below this floor, selects the baseline.
Supported iteration grouping retains its original program grid. These are
scheduling heuristics, not guarantees of a speedup.

## Supported transformations

- VV: lane-wise elementwise operations, supported reductions/scans, proved
  contiguous memory views, and supported uniform control flow. Selected loop
  forms group sequential iterations while preserving their carried state.
- CV: shared-right-operand M concatenation, shared-left-operand N concatenation,
  independent matrix batches, and supported uniform-loop variants. When fusion
  cannot be proved, a supported CV body can retain independent rank-two lanes.
  This fallback must not be reported as hardware batch-MMA or M/N fusion.
- Ordinary VV tails keep inactive lanes masked and use safe operands for
  operations that would otherwise evaluate outside their domain. CV requires
  full groups of programs. A matrix tile's M/N tail is a separate condition.
- The current target admission is Ascend950/Ascend910_95 and the Torch-NPU
  runtime. Pure SIMT requests select the baseline.

The native operation registry reports each operation's model, attribute policy,
inactive-input policy and implementation status. Admission of one operation does
not imply that every dtype, rank, attribute or combination is supported. See
[CV block batching](cv_block_batching.md) for matrix identities, mask rules,
integer scope and AutoBlockify integration.

## Memory and control-flow contract

Compile-time analysis and a per-call guard prove the modeled integer/address
domain, exclusive writes and noninterference. Read-only sharing is permitted.
Overlapping conservative byte ranges involving a write select the baseline,
even when a more precise analysis could prove the original kernel safe. The
guard does not query allocation capacity or validate user-provided buffer sizes.

Uniform loops and branches must preserve the selected-axis semantics and carried
values. Loop trip counts and transformed arithmetic are checked before choosing
iteration grouping; known overflow cannot be treated as an unknown dynamic
bound. Dynamic obligations remain guarded at launch.

Unknown address roots, unsupported scatter, atomics, cross-program
synchronization, unmodeled calls/assembly and unsupported varying control flow
are not admitted. A limited terminal-store proof permits some opaque device
reads in VV, without inspecting device indices on the host; it does not authorize
a CV matrix transformation. Volatile loads are not shared or broadcast and must
also have a proved nonoverlapping access domain.

Memory cache/eviction attributes and supported block-pointer boundary checks
retain their semantics. Rank-dependent axis hints are conservatively removed
when the rewrite cannot preserve them. Known pure external elementwise functions
are admitted individually; arbitrary external code is not assumed safe.

## Compilation, launch and fallback

Baseline and candidate compile from the same frozen frontend source as separate
artifacts. The candidate binds its plan, native implementation, binary hash,
original-grid ABI, workspace and launch metadata. Recognized proof or backend
resource rejections select the complete baseline before device submission.
Unknown internal compiler failures remain errors. A failed device execution is
never replayed automatically against the baseline.

Graph capture uses the baseline. Custom caches and IR overrides that cannot
preserve the artifact contract are rejected for candidates. The compiler's
ordinary diagnostics and resource checks remain authoritative; there is no
claim that all factors fit UB/L1/L0 or improve performance.

`compile_status` alone is not evidence that a call merged. Check
`last_launch.runtime_status == "APPLIED"`, `applied_factor`, the selected lowering
and correctness on the actual input. For performance, use identical inputs and
toolchains with adjacent, interleaved baseline/candidate measurements.

## Regression tests

Two device test files under `third_party/ascend/unittest/pytest_ut` cover the
feature's observable behavior:

- `test_block_batching_e2e.py`: explicit factors and the unbatched default,
  partial groups, original callable grids, empty launches, cached alias fallback,
  graph capture, reductions/scans and sequential iteration grouping.
- `test_block_batching_cv_e2e.py`: shared M/N and independent matrix grouping,
  structured matrix tails, loop accumulators and the supported INT8 paths.

Positive cases require an actually applied merge and check numerical results;
fallback cases check the selected baseline and its output. These correctness
tests explicitly disable the occupancy heuristic and do not measure speedup.
Use the normal test runner with an available supported NPU.

MLIR tests under `Conversion/General/BlockBatching` check tensor dimensions,
preserved reductions, loop fusion, overflow admission and transactional
rejection through the production binding. The adjacent TritonToLinalg and
TritonToUnstructure tests cover masked-load and safe-select lowering. They need
a matching built Ascend extension and FileCheck, but no device.
