# TLA Programming Interface

## 1. Overview

TLA (Triton Language for Ascend) is an independent programming namespace provided by Triton-Ascend for explicit on-chip memory, SIMD vector-register, and pipeline/inter-core synchronization programming on Ascend NPUs. TLA does not reuse the `al` / `bl` extension programming model; all capabilities are accessed through the `tla.*` namespace:

```python
import triton
import triton.language as tl
import triton.language.extra.tla as tla
```

TLA consists of the following modules:

| Module | Purpose |
| --- | --- |
| `tla` | Buffers, views, copies, scopes, and compiler hints |
| `tla.scope` | `vector` / `simd` / `cube` core scopes |
| `tla.simd` | SIMD vector-register load, store, compute, compare, and reduction |
| `tla.cube` | Cube matrix multiply-accumulate `mmad` |
| `tla.sync` | Pipeline synchronization, inter-core flags, and global synchronization |
| `tla.math` | TLA math helpers |
| `tla.device_print` / `tla.device_assert` | Device-side diagnostics |

> This document describes the TLA interfaces already merged into the current
> `main-dev` branch. The implementation lives in
> `third_party/ascend/language/tla/`, and examples live in
> `third_party/ascend/unittest/tla_ut/`. This document is refreshed when new
> interfaces are merged into `main-dev`.

## 2. Programming Model

TLA uses explicit on-chip buffers and vector registers. A typical data flow is:

```text
GM --tla.view--> buffer --tla.copy--> UB buffer
                                  --tla.simd.vlds--> vector
                                  --tla.simd.*-----> vector
                                  --tla.simd.vsts--> UB buffer
UB buffer --tla.copy--> GM
```

- `tla.view`: turns a kernel pointer or an existing buffer slice into a `tla.buffer`.
- `tla.alloc_buffer`: allocates a buffer in UB / L1 / L0A / L0B / L0C.
- `tla.copy`: performs DMA between buffers (GM↔UB, GM→L1, L0C→…).
- `tla.scope.simd()`: marks a SIMD vector-function region; `tla.simd.*` operates only on `tla.vector`.
- `tla.sync.*`: expresses pipeline and inter-core dependencies explicitly.

A minimal example:

```python
@triton.jit
def simd_only(X_ptr, O_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x_blk = tla.view(X_ptr, (N, ), strides=(1, ))
    o_blk = tla.view(O_ptr, (N, ), strides=(1, ))
    tla.copy(x_blk, x_ub)
    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], tla.simd.vadd(v, v))
    tla.copy(o_ub, o_blk)
```

## 3. Data Types

### 3.1 `tla.buffer` / `tla.buffer_type`

`tla.buffer` represents an on-chip memory view with shape, element type, address space, strides, and offset. It is not a Triton tensor.

| Attribute | Meaning |
| --- | --- |
| `shape` | Size of each dimension (positive compile-time integers) |
| `dtype` | Element type |
| `space` | Address space: `gm` / `ub` / `l1` / `l0a` / `l0b` / `l0c` |
| `strides` | Per-dimension strides; row-major by default |
| `offset` | Buffer offset |

`buf[i0, i1, ...]` returns a `buffer_access`, which can be used as the address argument of `vlds` / `vsts`, for example `tla.simd.vlds(x_ub[0, c * 64])`.

### 3.2 `tla.vector` / `tla.vector_type`

`tla.vector` represents a one-dimensional SIMD register value with `lanes` and an element type. It is used only by `tla.simd` operations. It does not support implicit conversion to a tensor and cannot be used as a scalar condition (`__bool__` raises).

## 4. Scopes

TLA marks core scopes with context managers:

| Interface | Purpose |
| --- | --- |
| `with tla.scope.vector():` | Regular vector-core region hosting `tla.copy`, control flow, and other non-SIMD operations; may nest `simd` |
| `with tla.scope.simd():` | SIMD vector-function region; the compiler outlines it as a `hivm.vector_function` |
| `with tla.scope.cube():` | Cube-core region (reserved; cube operators are not yet exposed through this scope) |

The core type is fixed at compile time; no runtime argument is required. A SIMD scope can be used standalone or nested inside a `vector` scope.

## 5. Memory Allocation and Data Movement

### 5.1 `tla.alloc` / `tla.alloc_buffer`

```python
tla.alloc_buffer(shape, dtype, space="ub", versions=1)
tla.alloc(shape, dtype, space="ub", versions=1)
```

Allocates an on-chip buffer and returns a `tla.buffer`.

| Parameter | Type | Description |
| --- | --- | --- |
| `shape` | `tuple[int, ...]` | Size of each dimension; positive compile-time integers |
| `dtype` | `tl.dtype` | Element type |
| `space` | `str` | Address space, default `"ub"`; `"gm"` is not allowed |
| `versions` | `int` | Number of multi-buffer versions; positive; `1` disables multi-buffering |

`alloc_buffer` is equivalent to `alloc`. GM buffers must be obtained from a kernel pointer through `tla.view`; they cannot be allocated with `alloc`.

### 5.2 `tla.copy`

```python
tla.copy(src, dst)
```

Performs DMA between two `tla.buffer` objects. Both source and destination must be `tla.buffer`.

Supported address-space pairs:

| `src` → `dst` | Semantics |
| --- | --- |
| `gm` → `ub` | Load GM into UB |
| `ub` → `gm` | Store UB into GM |
| `gm` → `l1` | ND → NZ copy into L1 |
| `l0c` → other | Fixpipe copy-out; a 2-D destination uses `NZ2ND`, otherwise normal mode |

Constraints:

- Source and destination element types must match.
- Source and destination shapes must match, except for the `gm→l1` ND→NZ relation and the `l0c` Fixpipe fractal relation.
- Other address-space pairs raise an error.

### 5.3 `tla.view`

```python
tla.view(src, shape, strides=None, offsets=None, space="gm")
```

Turns a kernel pointer or an existing buffer into a `tla.buffer` with the given shape.

| `src` type | Behavior |
| --- | --- |
| Pointer tensor | Pointer-entry view; `space` selects the GM address space, and `strides` are the final memref strides (row-major by default) |
| `tla.buffer` | Buffer slice (subview); `strides` are relative to the source view, and `space` must keep its default value |

`offsets` are per-dimension offsets and may be compile-time integers or scalar integer tensors. A pointer entry emits a 1-D `hivm.hir.pointer_cast` plus one `memref.reinterpret_cast`; a buffer slice emits `memref.subview`.

### 5.4 `tla.reshape`

```python
tla.reshape(src, shape)
```

Performs a pure shape transformation on a contiguous, row-major buffer with zero offset. The element count must be preserved. It uses `memref.collapse_shape` / `memref.expand_shape` internally. Use `tla.view` first for strided or offset slices.

### 5.5 `tla.annotate_buffer_versions`

```python
tla.annotate_buffer_versions(buf, versions)
```

Declares the number of multi-buffer versions for one buffer, consumed by the NPU-IR multi-buffer pass. `versions` must be a positive integer; `1` disables multi-buffering.

## 6. SIMD Vector-Register Interfaces

### 6.1 Element Types and Register Width

Supported SIMD element types:

| Element type | Tag | Lanes in a full register |
| --- | --- | --- |
| `int8` / `float8e4nv` | `i8` / `fp8e4nv` | 256 |
| `int16` / `float16` / `bfloat16` | `i16` / `f16` / `bf16` | 128 |
| `int32` / `float32` | `i32` / `f32` | 64 |
| `int64` / `uint64` | `i64` | 64 (4096-bit register) |

Other element types raise an error. Unsigned integers are handled as signed integers of the same width by the backend.

Predicate masks are `vector<lanes x int1>`. When no mask is provided, the interface uses an all-true `pge <ALL>` mask.

### 6.2 Load and Store

| Interface | Description |
| --- | --- |
| `tla.simd.vlds(ptr, offsets=None, *, dist="NORM", elem=None)` | Loads a full vector register from a UB buffer; `ptr` may be `buf[i, ...]`, and `elem` can override the result element type |
| `tla.simd.vsts(ptr, offsets_or_val, val=None, *, dist=None, mask=None)` | Masked store of a vector register to a UB buffer; `dist` defaults to `NORM_B32`, and `mask` defaults to all-true |

`dist` must be a load/store distribution string supported by AVE, such as `NORM`, `UNPK_B16`, `UNPK4_B8`, `BRC_B32`, `NORM_B32`, or `ONEPT_B32`.

### 6.3 Conversion and Layout Transformation

| Interface | Description |
| --- | --- |
| `tla.simd.vextf(src, target_dtype, mask=None, part="part_even")` | Widening float conversion; currently only converts to `f32`; the result is a full 64-lane register |
| `tla.simd.vtruncf(src, target_dtype, mask=None, part="part_even")` | Narrowing float conversion to `bf16` / `f16` / `fp8e4nv`; the total bit width is preserved |
| `tla.simd.vbitcast(src, dtype_str)` | Reinterprets the bits of a vector register; the result lane count is derived from the total bit width |
| `tla.simd.vintlv(src1, src2)` | Pairwise interleave of two same-typed registers; returns two result registers |
| `tla.simd.vdintlv(src1, src2)` | Pairwise deinterleave; the inverse of `vintlv` |

`part` accepts only `part_even` / `part_odd` and selects a sub-region of the source register.

### 6.4 Arithmetic and Unary Operations

| Interface | Description |
| --- | --- |
| `tla.simd.vmul(lhs, rhs)` | Elementwise multiply |
| `tla.simd.vadd(lhs, rhs)` | Elementwise add |
| `tla.simd.vdiv(lhs, rhs)` | Elementwise divide |
| `tla.simd.vxor(lhs, rhs)` | Elementwise bitwise XOR |
| `tla.simd.vmod(lhs, rhs)` | Elementwise integer modulo |
| `tla.simd.vfma(acc, lhs, rhs)` | Fused multiply-add: `acc + lhs * rhs` |
| `tla.simd.vneg(src)` | Elementwise negate |
| `tla.simd.vsqrt(src)` | Elementwise square root |
| `tla.simd.vrsqrt(src)` | Elementwise reciprocal square root |
| `tla.simd.vexp(src)` | Elementwise exponential |

Binary operations require both operands to have exactly the same type; unary operations preserve the element type. All operations use an all-true mask.

### 6.5 Compare and Select

| Interface | Description |
| --- | --- |
| `tla.simd.vcmp(lhs, rhs, cmp_type)` | Elementwise compare; `cmp_type` accepts `eq` / `ne` / `gt` / `ge` / `lt` / `le` and returns an `int1` predicate vector |
| `tla.simd.vsel(mask, true_val, false_val)` | Bitwise select; the mask lane count must match the value vectors, and the two value vectors must have the same type |

### 6.6 Reduction and Broadcast

| Interface | Description |
| --- | --- |
| `tla.simd.vreduce(src, kind)` | Horizontal reduction; `kind` is `add` / `min` / `max`; the result is placed at lane 0 |
| `tla.simd.vreduce_sum(src)` | Alias of `vreduce(src, "add")` |
| `tla.simd.vreduce_max(regs)` | Global maximum over one register or a list of registers, broadcast to all lanes |
| `tla.simd.vreduce_min(regs)` | Global minimum over one register or a list of registers, broadcast to all lanes |
| `tla.simd.vbroadcast(src, low=True)` | Broadcasts the low half (`low=True`) or high half of the source register to the full register |

`vreduce_max` / `vreduce_min` first fold multiple registers with elementwise `max` / `min`, then perform a horizontal reduction and broadcast, so the result can be used directly by `vcmp`.

### 6.7 Constants, Indices, and Predicates

| Interface | Description |
| --- | --- |
| `tla.simd.vfull(value, dtype, lanes)` | Broadcasts a scalar into a constant register with the given lane count |
| `tla.simd.vsplat(value, dtype)` | Derives the full-register lane count from the element type and broadcasts a constant |
| `tla.simd.vci(base)` | Generates a lane-index register where lane `i` holds `base + i`; fixed to `int32` / 64 lanes |
| `tla.simd.pge(n, lanes)` | Prefix predicate: the first `n` lanes are true and the rest are false; `n` supports 1/2/3/4/8/16/32/64/128 |

### 6.8 Memory Barrier

```python
tla.simd.membar(scope)
```

Inserts a memory barrier inside a VF. `scope` supports AVE membar kinds such as `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST`, `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST`, `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST`, `SS_ALL`, `ST_LD`, `LD_ST`, and `ST_ST`.

## 7. Cube Interface

### 7.1 `tla.cube.mmad`

```python
tla.cube.mmad(a, b, acc, transpose_b=False, clear_accum=False)
```

Computes `C = A × B (+ C)`:

| Parameter | Description |
| --- | --- |
| `a` | L1 buffer with shape `(M, K)` (2-D ND) or an equivalent 4-D zN shape |
| `b` | L1 buffer with shape `(K, N)` or an equivalent 4-D zN shape |
| `acc` | L0C buffer with shape `(M, N)` |
| `transpose_b` | Whether to transpose `b` |
| `clear_accum` | Whether to clear the accumulator before computing |

`a`, `b`, and `acc` must be `tla.buffer`; `a` / `b` must be in L1 and `acc` in L0C; `a` and `b` must have the same element type. `transpose_a` is not exposed; implement it by adjusting the layout during the GM→L1 copy.

## 8. Synchronization Interfaces

### 8.1 Pipeline Synchronization

| Interface | Description |
| --- | --- |
| `tla.sync.barrier(pipe)` | Waits for all preceding instructions on the given pipeline |
| `tla.sync.set_flag(pipe1, pipe2, event_id)` | The producer on `pipe1` signals an event to the consumer `pipe2`; `event_id` is in `[0, 7]` |
| `tla.sync.wait_flag(pipe1, pipe2, event_id)` | The consumer on `pipe2` waits for the event from `pipe1`; `event_id` is in `[0, 7]` |

### 8.2 Global Synchronization

```python
tla.sync.sync_all(mode)
```

Hardware global barrier: `mode=False` synchronizes AIV cores; `mode=True` synchronizes all AIC and AIV cores. `mode` must be a compile-time `bool`.

### 8.3 Inter-Core Flags

| Interface | Description |
| --- | --- |
| `tla.sync.block_set_flag(mode, pipe, flag)` | Signals an inter-core flag from the current core |
| `tla.sync.block_wait_flag(pipe, flag)` | Waits for an inter-core flag on the given pipeline |

`block_set_flag` `mode` values:

| `mode` | Meaning |
| --- | --- |
| `0` | inter-block |
| `1` | inter-subblock |
| `2` | intra-block |

`flag` is in `[0, 15]`; `block_wait_flag` must use the same flag ID as `block_set_flag`.

### 8.4 Pipeline Names

`pipe` is a compile-time string and may be written as `"V"`, `"MTE2"`, or the full names `"PIPE_V"`, `"PIPE_MTE2"`. Supported names:

`PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_MTE4`, `PIPE_MTE5`, `PIPE_V2`, `PIPE_FIX`.

`barrier("ALL")` is equivalent to `PIPE_ALL`.

## 9. Math Helpers

### 9.1 `tla.compensated_rstd`

```python
tla.compensated_rstd(square_sum, hidden, eps)
```

Computes the compensated reciprocal standard deviation (rstd) of `square_sum / hidden + eps` using TwoSum error recovery and a Newton step. It returns an `f32` vector register replicated per lane.

| Parameter | Description |
| --- | --- |
| `square_sum` | `f32` vector register holding `sum_h x[h]^2`, usually broadcast to all lanes |
| `hidden` | Hidden size; a positive Python / constexpr scalar |
| `eps` | Epsilon; must be at least the smallest normal FP32 value `2^-126` |

This interface is semantically consistent with `make_rstd_impl` in TileKernels `rstd.py`; it is not the plain RMSNorm formula.

## 10. Debug Interfaces

| Interface | Description |
| --- | --- |
| `tla.device_print(prefix, *args, hex=False)` | Prints values on device; only allowed outside `tla.scope` and only for UB buffers / UB element ranges / scalars |
| `tla.device_assert(cond, msg="")` | Runtime assertion; effective only when the `debug` option is enabled; `cond` must be a scalar `int1` |

## 11. Relationship to `al` / `bl` Extensions

TLA and `al` / `bl` are independent programming interfaces:

- `al` / `bl` center on Triton tensors and an implicit memory model; TLA uses explicit `tla.buffer` and `tla.vector`.
- `tla.sync.sync_all` uses a compile-time `bool` to select the synchronization scope, whereas `al.sync_block_all` uses a string `mode` and an `event_id`.
- `tla.sync.block_set_flag(mode, pipe, flag)` explicitly specifies the inter-core synchronization mode, whereas `al.sync_block_set(sender, receiver, event_id, ...)` derives it from the sender and receiver.
- NVIDIA-style tensor operations and implicit type conversion are not supported in TLA; data crossing interfaces must be moved explicitly with `tla.view` / `tla.copy`.

## 12. Complete Example

The following kernel computes `O = A + B` and explicitly inserts the GM→UB, vector-compute, and UB→GM dependencies:

```python
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def sync_intra(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.sync.set_flag("MTE2", "V", 0)
    tla.sync.wait_flag("MTE2", "V", 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    tla.sync.barrier("ALL")
    tla.sync.set_flag("V", "MTE3", 1)
    tla.sync.wait_flag("V", "MTE3", 1)
    tla.copy(o_ub, o)

    tla.sync.sync_all(False)
```

Launch with TLA mode enabled:

```python
sync_intra[(1, )](A, B, O, N=64, enable_tla_mode=True)
```

## 13. Constraints and Notes

1. Inputs and outputs of `tla.simd` must be `tla.vector`; Triton tensors cannot be passed directly.
2. Binary operations require both operands to have exactly the same type; a mask must be an `int1` predicate vector with the same lane count as the value vector.
3. SIMD registers are 2048 bits wide; `i64` / `u64` use 4096 bits. Unsupported element types raise an error instead of being implicitly mapped to a different width.
4. `tla.copy` accepts only `tla.buffer`; pointers and tensors cannot be copied directly.
5. `tla.device_print` cannot be used inside `tla.scope` and supports only UB-backed values.
6. The TLA interfaces described here follow the current `main-dev` implementation; this document should be updated when new interfaces are merged.
