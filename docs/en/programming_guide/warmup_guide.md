# Warmup User Guide

## Overview

`kernel.warmup(*args, grid=...)` compiles a kernel without executing it. The compiled artifact is written to the Triton cache, so a later real launch can load it directly. This makes warmup a good fit for paying the compilation cost at service startup or in CI.

Recommended for the following scenarios:

- **Lower first-inference latency**: real requests no longer wait on compilation, with no first-request stall;
- **Earlier compile-error exposure**: compile failures surface at startup or CI time instead of being deferred to runtime;
- **Compile without executing**: verify a kernel compiles without allocating outputs or running it.

## Using plain JIT kernels

Take a vector-add kernel as an example:

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (triton.cdiv(n, 256),)
```

### Scenario 1: Using dtype placeholders

`warmup()` depends only on argument types and pointer attributes, not tensor contents, so it needs no device memory allocation or initialization—just pass a dtype. This is especially convenient when device memory is tight, or when a startup script or CI job does not have input data ready:

```python
compiled = add_kernel.warmup(
    torch.float32, torch.float32, torch.float32, n,
    BLOCK_SIZE=256, grid=grid,
)
print(compiled.metadata)
```

There are two points to note when calling it:

- `grid` must be passed explicitly;
- `warmup()` returns a `CompiledKernel`; `metadata`, `asm`, and other compilation results are available.

### Scenario 2: Passing real tensors

A placeholder is treated as an ordinary tensor with a 16-byte-aligned pointer. If real inputs are slices or offset views with potentially misaligned pointers, pass real tensors so specialization matches the real launch:

```python
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.randn(n, device="npu", dtype=torch.float32)
out = torch.empty_like(x)
compiled = add_kernel.warmup(x, y, out, n, BLOCK_SIZE=256, grid=grid)
```

## Using autotune kernels

If a kernel uses `@triton.autotune`, warmup compiles every candidate configuration; the call is the same as for a plain kernel:

```python
import torch
import torch_npu
import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
]


@triton.autotune(configs=configs, key=["n_elements"])
@triton.jit
def add_kernel_autotune(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (1,)

handles = add_kernel_autotune.warmup(
    torch.float32, torch.float32, torch.float32, n, grid=grid,
)
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

Compared with a plain kernel, there are three differences:

- The return value is a list with one entry per candidate configuration. The Ascend backend compiles in parallel by default, so elements may be `FutureKernel`; call `.result()` to get a `CompiledKernel`, after which they can be handled uniformly;
- Automatic tiling with `configs=[]` is also supported; compilation errors from parallel compilation surface before `warmup()` returns;
- Warmup **does not benchmark or select a best configuration**; selection still happens on the first real launch, so do not treat warmup as "selection preheating".

## Notes

- Warmup does not read or write input/output tensors, and it cannot detect runtime problems such as out-of-bounds access, numerical errors, or UB overflow;
- Warmup's benefit depends on cache hits: the real launch should use the same Triton cache directory (`TRITON_CACHE_DIR`) and compiler options as warmup;
- Set `TRITON_DEBUG=1` to confirm a cache hit: once the real launch hits the cache, that kernel's compilation logs no longer appear.
