# Scalar reads, byte aliases and strided DMA

Sparse attention gathers selected KV rows, reads UB index metadata in scalar
control flow, and reuses temporary UB storage across non-overlapping stages.
These interfaces express those operations directly through native builders;
they do not parse MLIR strings or patch the Python code generator.

## Read UB control data

```python
index = tla.scalar_load(indices, position)
index = tla.scalar_load(indices_2d[row, column])
```

`scalar_load` reads one int32 UB element through `memref.load`. It accepts one
Python integer or signed int32/int64 scalar tensor per dimension, preserving the source offset and strides.
A single index is sufficient for a rank-one buffer. Use it in scalar control
flow inside `tla.scope.vector()`, outside `tla.scope.simd()`.

Static out-of-bounds indices are rejected. Runtime indices must be in bounds.
The producer must complete before the read: a scalar load is not a barrier.
Automatic synchronization can see this ordinary memory effect; manually
managed pipelines remain responsible for their producer/consumer dependency.

## Reuse byte-backed UB storage

```python
storage = tla.alloc((1024,), tl.int8, space="ub")
fp32_tile = tla.byte_view(storage, (128,), tl.float32, byte_offset=128)
bf16_tile = tla.byte_view(storage, (128,), tl.bfloat16, byte_offset=128)
```

`byte_view(src, shape, dtype, byte_offset=0)` emits `memref.view`, with no copy
or numerical conversion. The source must be a zero-offset, contiguous,
rank-one int8 UB buffer. Shape and byte offset are compile-time values. The
range must fit the source and the byte offset must align to the result element
size. Scalar integer/floating-point types with 8, 16, 32 or 64 bits are accepted;
each consumer still imposes its backend dtype/alignment restrictions.

Overlapping views alias the same storage. Complete prior reads before writing
through a reused view. Memory planning and synchronization see the native
alias relationship; this API makes no no-alias assertion.

## Copy rows with runtime pitch and count

```python
# base is a rank-one, unit-stride GM view.
# tile is a rank-two UB view with contiguous columns.
tla.load_gm_to_ub_strided(base, tile, offset, row_stride, rows=count)
```

The transfer reads `count` rows of `tile.shape[1]` elements. `rows=None` selects
all destination rows. Offset, row stride and row count can be Python integers
fitting signed int64 or signed int32/int64 scalar tensors; offsets/strides are measured in elements. Unselected
destination rows and padding remain untouched. The source and destination
element types must match.

Source offset is relative to the **current source view**, including existing
entry/subview offsets. The implementation extracts and adds the source
descriptor offset before constructing a `memref.reinterpret_cast`. The
destination `memref.subview` preserves its existing pitch and offset.
The source descriptor has static row capacity; a source subview expresses a
runtime row count. This keeps pointer-entry lowering statically shaped. Only
the requested rows are accessed, even if the unused capacity extends beyond
the source allocation.

Requirements:

- Source: rank-one GM with unit stride.
- Destination: rank-two UB, inner stride one, non-overlapping rows.
- `offset >= 0`, `1 <= rows <= tile.shape[0]`, and
  `row_stride >= tile.shape[1]`.
- The full source range is in bounds; the native DMA backend additionally
  checks target-specific alignment, dtype and transfer limits.

Statically known violations are rejected. Runtime values remain the caller's
responsibility; no runtime bounds checks are inserted. The interface emits
ordinary memref views and one `hivm.hir.load`, with no allocation, index sorting,
implicit gather, padding or additional synchronization.

This scoped interface leaves the general TLA buffer type statically shaped.
Runtime memref dimensions are confined to the transfer itself.

## Validation

Three-stage tests under `third_party/ascend/unittest/tla_ut/` cover:

- `vector/scalar_load.py`: DMA/vector writes followed by scalar reads, including
  strided/nonzero-offset aliases and multidimensional indices.
- `vector/byte_view.py`: overlapping FP32/BF16 view lifetimes, exact typed
  outputs, full backing bytes and untouched guards.
- `dma/strided_load.py`: FP16/BF16 static and runtime row counts/pitches,
  composed pointer/entry/subview offsets and pitched destination guards.
  Zero-offset pointer-entry cases also check a one-row transfer at the end of
  the source allocation with no valid second row.
- `memory_frontend_contract.py`: host bounds/type/layout rejection and
  forwarding contracts; these do not replace the real IR/device tests.

Each device configuration runs two seeds with exact equality checks. See the
[TLA test README](../../../../third_party/ascend/unittest/tla_ut/README.md) for
explicit `TLA_TEST_STAGE=run` invocation and native build requirements.
