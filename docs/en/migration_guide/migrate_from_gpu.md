# Migrating Triton Operators from GPUs

This document describes the general procedure and common issues for migrating GPU Triton operators to Ascend NPUs. Start by replacing Python-side device and runtime interfaces, then check grid core allocation, memory access alignment, single-program computation, UB usage, and coreDim limits. The examples later in this document show how to apply these steps in code.

## General Migration Procedure

### Migrate Python-Side Device and Runtime Interfaces

Before modifying a specific Triton kernel, migrate the Python-side device code first:

1. Add `import torch_npu` to the Python file.
2. Find `device="cuda"`, `device='cuda'`, `.cuda()`, `.to("cuda")`, and similar device specifications, and change them to `device="npu"`, `device='npu'`, `.npu()`, or `.to("npu")`.
3. Find GPU-specific APIs such as `torch.cuda.*`, CUDA streams, CUDA events, and CUDA synchronization, then replace them with NPU counterparts or remove unnecessary synchronization logic.
4. Remove logic that exists only for GPU device discovery, such as assertions around `triton.runtime.driver.active.get_active_torch_device()`.
5. Keep the Triton kernel body unchanged at first, and use NPU tensors to verify compilation and correctness.

### Adjust Grid Core Allocation

GPU kernels often use a large logical grid and rely on the runtime and hardware to schedule programs onto SMs. On NPUs, first consider the physical AI Core count and operator type:

- Prefer 1D grids. NPU 2D adaptations are merged into 1D; for example, `(20,)` and `(4, 5)` produce equivalent execution results.
- For Vector-only operators, organize concurrent tasks around the Vector Core count. For operators containing `tl.dot`, organize concurrent tasks around the AI Core count.
- If the logical grid is much larger than the physical core count, consider letting each program process multiple tiles in an inner loop, or use `TRITON_ALL_BLOCKS_PARALLEL` when logical programs have no ordering dependency.
- `coreDim` cannot exceed `UINT16_MAX` (65535). For large shapes, control grid size through BLOCK_SIZE or tiling.

| Dimension | Core Structure | Operator Type |
|-----------|----------------|---------------|
| Ascend NPU | Multiple AI cores, categorized into Cube Cores for matrix multiplication and Vector Cores for vector computation | Vector-only operators -> concurrent task count = Vector Core count; operators containing `tl.dot` -> concurrent task count = AI Core count |
| NVIDIA/AMD GPU | Multiple CUDA cores for scalar/vector computation and Tensor Cores for matrix multiplication | GPU concurrency is generally determined by the compiler and hardware |

### Check Single-Program Data Transfer

After device replacement, check data movement inside each program:

- Vector operators require 32-byte memory access alignment, and cube-vector fused operators require 512-byte alignment.
- Keep tail masks and verify that boundary elements are not accessed out of bounds.
- Check on-chip memory usage for each tile to avoid UB overflow.
- Remove or replace GPU-specific synchronization APIs, such as CUDA thread, stream, event, or kernel synchronization interfaces.

### Check Single-Program Computation

NPU and GPU compute units differ in supported data types and execution behavior. After migration, verify correctness first, then adjust based on performance symptoms:

- For integer indices, offsets, and lengths, confirm whether the current dtype is efficiently supported by the NPU path.
- For operators containing `tl.dot`, check M/N/K tiling, accumulator dtype, and output dtype.
- For long sequence, long hidden size, or large K loops, use tiling to control the amount of data moved and computed at one time.

## Migration Examples

### Example 1: Complete Vector Addition Migration

```diff
import torch
+ import torch_npu  # [Added] Import Ascend NPUs' PyTorch adaptation library to support NPU devices.
import triton
import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()  #  [Deleted] GPU devices are automatically obtained. NPUs do not need this logic.

@triton.jit
def add_kernel(x_ptr, # Pointer to first input vector.
y_ptr, # Pointer to second input vector.
output_ptr, # Pointer to output vector.
n_elements, # Size of the vector.
BLOCK_SIZE: tl.constexpr, # Number of elements each program should process.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
-   assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE  # [Deleted] GPU devices have consistency checks. NPUs do not need explicit assertion.
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
- x = torch.rand(size, device='cuda')  # [Deleted] Specify the GPU device.
+ x = torch.rand(size, device='npu')  # [Modified] Specify the Ascend NPU device.
- y = torch.rand(size, device='cuda')  # [Deleted] Specify the GPU device.
+ y = torch.rand(size, device='npu')  # [Modified] Specify the Ascend NPU device.
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
f'{torch.max(torch.abs(output_torch - output_triton))}')
```

### Example 2: Device Replacement and Single-Program Data Transfer

The following example replaces CUDA tensors with NPU tensors and verifies correctness for a single-program data transfer case.

```diff
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize('shape', [(1,), (2,), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

-    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='cuda')
+    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='npu')
    std = torch.broadcast_to(x, (YS, XS))
-    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='cuda')
+    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='npu')
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
```

## FAQ

After completing the basic migration procedure, you may encounter the following two types of new issues: 

1. **coreDim** limit 
This issue is triggered when grid dimensions exceed the hardware limit of NPUs. 
Typical error message: `coreDim=xxxx can't be greater than UINT16_MAX`. 
2. UB space overflow 
Memory usage exceeds the NPU cache capacity. 
Typical error message: `ub overflow, requires xxxx bits while 1572684 bits available!`. 

### Solving the coreDim Limit Issue

Issue analysis: 
The **coreDim** parameter of NPUs cannot exceed **UINT16_MAX** (**65535**). When processing large-scale data, simplistic grid division may exceed this limit. 

Case: Optimizing the `zeros_like` function   
(data scale `N = 1073741824`; original `BLOCK_SIZE = 2048`; calculated `coreDim = 524288`, exceeding the limit of **65535**) 

Solution 1: 
To address the **coreDim** limit in the Ascend compiler, one solution is to set the environment variable *'TRITON_ALL_BLOCKS_PARALLEL'* to **1** by running this command: 
export TRITON_ALL_BLOCKS_PARALLEL=1  
Solution 2: 
Another solution is to increase **BLOCK_SIZE** to reduce the number of required cores and ensure that **coreDim** remains within the limit. 
The calculation follows: `coreDim = ceil(N / BLOCK_SIZE)`. → It needs to satisfy `ceil(N / BLOCK_SIZE) <= 65535 => BLOCK_SIZE >= ceil(N / 65535)`. Given `N = 1073741824`, we have `BLOCK_SIZE >= triton.next_power_of_2(triton.cdiv(1073741824, 65535)) = 32768`. Therefore, **32768** is the minimum safe value.

Code before optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # The original value is too small.
    return out
```

Code after optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535)) 
    BLOCK_SIZE = max(32768, min_block_size) # The minimum value is 32768.
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),) 

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE) 
    return out
```

### Dynamically Calculating **BLOCK_SIZE** to Ensure **coreDim** Remains Within the Limit

```diff
optimal_block_size = 32768 # Optimized value obtained after calculation

grid_fn = lambda meta: (triton.cdiv(N, optimal_block_size),)

zeros_kernel[grid_fn](out, N, BLOCK_SIZE=optimal_block_size)
return out
```

### Handling the Compound Issue: coreDim + UB Overflow

Issue analysis: 
In some scenarios, solving the **coreDim** limit issue may inadvertently trigger a new issue—UB overflow. This typically occurs when increasing **BLOCK_SIZE** causes the data volume processed by a single thread block to exceed the UB cache capacity of NPUs. 

Case:   
Data scale `N = 1073741824`; original `BLOCK_SIZE = 4096`; calculated `coreDim = 262144`, exceeding the limit of **65535**. After **BLOCK_SIZE** is adjusted to **32768**, **coreDim** is **32768** (within the limit), but UB overflow occurs. 

Solution: 
Introduce the **BLOCK_SIZE_SUB** parameter to further subdivide large blocks, thereby controlling memory usage while maintaining a reasonable **coreDim**. 
Code before optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)
    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)

def masked_fill(inp, mask, value):
    # ... Parameter verification code ...
    # inp.device = "npu"
    out = torch.zeros_like(inp)
    N = inp.numel()
    if N == 0:
        return out

    grid = lambda meta: (triton.cdiv(N, 4096),) # coreDim exceeds the limit.
    masked_fill_kernel[grid](inp, mask.to(torch.int), value, out, N, 4096)
    return out
```

Code after optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    # Calculate the number of sub-blocks to be processed.
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    # Process blocks to avoid UB overflow.
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # Load and process data in batches.
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        # First, write the original data.
        tl.store(out + offsets, input_vals, mask=mask)
        # Then overwrite the target value at the position where padding is required.
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

def masked_fill(inp, expand_mask, value):
    logger.debug("GEMS MASKED FILL")

    # ... Parameter verification code ...
    # inp.device = "npu"
    out = torch.zeros_like(inp)
    N = inp.numel()
    if N == 0:
        return out

    # Use optimized parameter settings.
    MAIN_BLOCK_SIZE = 32768  # Ensure that coreDim is within the limit.
    SUB_BLOCK_SIZE = 1024    # Control the UB usage.

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N, 
                        MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

### Why Does the UBSIZE Out of Memory Error Occur?

Improper data tiling can lead to excessive unaligned memory access or computation. Consider a 2D data transfer of shape `(64, 32)` as an example. The corresponding stride is `(12832, 128)`. If aligned memory access is required, the stride becomes `(32, 1)`. In unaligned access scenarios, an additional axis of size `1` is added to the innermost dimension, yielding a shape of `(64, 32, 4)`. Because the hardware mandates 32-byte UB memory alignment in vector operator scenarios, the corresponding stride is recalculated as `(12832, 128, 1)`, assuming `type=float16`.

### Discrete Memory Access and Inefficient Scalar Mapping Observed by Line-by-Line Code Comparison

Set the environment variable *TRITON_DEBUG* to **1**, save **~/.triton/cache/xxx.ttadapter**, and execute: 

```diff
bishengir-compile xxx.ttadapter --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync  
```

Compare the Triton kernel's logic with the internal operations of the output intermediate representations (IRs) to identify any operations that are not mapped to instructions. 
Check whether pure scalar transfer or computation exists in the HIVM IR phase without being mapped to SIMD instructions. If such cases exist, they will create a significant performance bottleneck.   

Problem: Discrete memory access and inefficient scalar mapping 
Given `b[1024, 32] = a[1024, 32]`, the original Triton code binds thread blocks to the lowest dimension `32` in `[1024, 32]`, and then splits `1024` into `16` parts, yielding `[64, 16, 32]`. Finally, it binds thread blocks to dimension `64`. 

```diff
chunk_fwd_kernel_o[(NT, B * H)](
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(1024,), # 1D tensor
        strides=(32,), # Contiguous memory
        offsets=(i_t * 16,), # Start position
        block_shape=(BT,), # Block size
        order=(0,) # Sequential access
    )
​)
```

Optimization Approach

Adjust **shape** and **stride** for **block_ptr** as follows: 
The shape (1024, 32) is treated as a 2D matrix, where the lowest dimension `32` is contiguous. Accordingly, the stride should be `(32, 1)` instead of `(32,)`. This enables each thread block to access 32 contiguous elements. Bind thread blocks to the row dimension `(1024)` and configure each thread to process all 32 elements in a row. This approach guarantees contiguous memory access and high memory affinity 

Example:

```diff 
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0) # Row-major: dim 1 is innermost (stride 1), dim 0 is outermost
)
```
