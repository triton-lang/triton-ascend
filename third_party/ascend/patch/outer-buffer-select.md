# Inter-core double-buffer selects

Outer-scope buffering uses the existing loop parity condition unchanged:

```mlir
%flag = arith.select %parity, %original_id, %spare_id : i64
hivm.hir.sync_block_wait[...] flag = %flag
%buffer = arith.select %parity, %original_buffer, %spare_buffer : memref<...>
// One original transfer, using %buffer.
hivm.hir.sync_block_set[...] flag = %flag
```

Selecting a memref chooses its descriptor; it does not execute a transfer or
read either buffer. The transfer and receiver view chain retain their original
position between wait and set. Initial and final notifications remain static.
The two selections use the same original condition, including for while loops.

## NPU-IR dependency

`npuir-cross-buffer-select.patch` supplies the corresponding native support:

- Trace both selected allocation roots for fixpipe tiling when their physical
  types agree.
- Propagate subblock slices down through both selected buffers; reconcile
  equivalent identity and explicit strided layouts at the select boundary.
- Propagate alignment requirements to both allocations and preserve the aligned
  layout on the selected descriptor.
- Select offset, sizes and strides as well as pointers during descriptor
  lowering.

The alignment change is necessary even when automatic subblock tiling is off.
For example a 3x3 FP32 fixpipe destination requires padded rows. Losing its
aligned layout changes the descriptor's row stride from 8 elements to 3 and
reads the wrong data despite an unchanged parity condition.

`TRITON_BUILD_NPUIR=ON` applies this patch before building the pinned NPU-IR
source and records `lib/cross-buffer-select-v1` in its payload. When bundling
an external `TRITON_ASCEND_BISHENGIR_PATH`, rebuild that payload with the same
patch and include this capability file. Do not add the capability file to an
unpatched compiler. The backend rejects double-buffer selects with an old
payload before invoking compilation, because older tools can silently lose
layout constraints. No automatic tiling or buffer-reuse optimization is
disabled by this change.

The patch must be reevaluated when the NPU-IR revision changes; the build fails
if it cannot apply cleanly. The marker is a payload packaging contract, not a
compiler version comparison. A pass-only IR test does not establish end-to-end
correctness: native tiling, alignment, alias analysis, memory planning and
synchronization must all preserve both possible buffer roots.
