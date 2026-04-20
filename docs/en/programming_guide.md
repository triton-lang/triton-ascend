# Triton Operator Development Guide

This document focuses on the issues that need to be paid attention to during Triton operator development on NPUs, which are divided into three aspects: multi-core task parallelism, single-core data transfer, and single-core data computation. First, section "Multi-Core Task Parallelism" describes the basis for setting the maximum number of hardware cores and the specific implementation. Then, section "Single-Core Data Transfer" describes how to set the proper data block size in a loop, introduces the common optimization methods, and describes how to handle the UB overflow problem that may occur. Finally, section "Single-Core Data Computation" focuses on how to develop Triton operators and highlights the key points.

## Multi-Core Task Parallelism

### Setting the Maximum Number of Hardware Cores

In a Triton operator, the grid is usually used for core allocation. For GPUs, it contains dozens or hundreds of core SMs. However, for the Ascend NPU platform, it contains dozens of AI Cores for computation.\
Although the runtime interface allows a maximum of 65,535 concurrent tasks to be delivered, the tasks that exceed the number of physical cores are delivered in a new round. If the Triton operator on the GPU is directly executed on the Ascend platform, a large number of tasks will introduce considerable overhead during core startup and initialization, affecting the operator performance.\
Therefore, the core allocation logic needs to be modified based on the Ascend platform features. The most recommended method is to **fix the number of cores to the number of physical cores of the hardware** and perform more detailed data block division within the cores.

* For pure vector operators, the number of cores is equal to the **number of vector cores**.
* For CV fusion operators, the number of cores is equal to the **number of cube cores** (usually half of the number of vector cores). During operator execution, vector cores are called at a ratio of 1:2.

Generally, on an NPU card, a computing core (AI Core) consists of one cube core, with each cube core paired with two vector cores. So you can obtain the **number of vector cores(vectorcore_num)** and **number of cube cores(aicore_num)** through the following interfaces:

```python
import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

```

According to the sample code, fix the number of cores, and then process task blocks in batches through an internal loop.

```python
NUM_CORE = vectorcore_num
grid = (NUM_CORE ,) 
_attn_fwd[grid](Q, K, V, M, Out, acc, scale......)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,  
              ......
              stride_qz, stride_qh, 
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # Calculate the total number of tasks and flatten the three-dimensional tasks (Z, H, M) into a one-dimensional total number of tasks.
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Each core selects the task to be processed based on its own identifier.
    pid = tl.program_id(0)  # Unique ID of the current core.
    NUM_CORE = tl.num_programs(0)  # Obtain the total number of cores that are started.
    # Loop rule: range(pid, NUM_BLOCKS, NUM_CORE) implements step-based task allocation.
    # - Start value (pid): Each core obtains tasks from its own ID to avoid task overlapping.
    # - Step length (NUM_CORE): The step is based on the total number of cores to ensure that tasks are evenly allocated to each core.
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # Calculate the data offset of each task.
        # [Core: Reverse restoration of one-dimensional task index to original multi-dimensional index.]
        # block_idx is the one-dimensional task index after flattening. The original dimensions are restored through integer division and remainder.
        # 1. Split the Z+H combined axis and M block axis.
        #   - Exact division by NUM_BLOCKS_M: Extract the index (task_hz_idx) of the Z+H combined axis.
        #   - Remainder of NUM_BLOCKS_M: Extract the block index (task_m_idx) of the M dimension.
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2. Split the Z+H combined axis into the original Z axis and H axis.
        #   - Exact division by H: Restore the Z axis index (off_z).
        #   - Remainder of H: Restore the H axis index (off_h).
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3. Calculate the data offset: Locate the start position of the corresponding data in the Q/K/V tensor based on the restored Z/H index.
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```

## Single-Core Data Transfer

### Setting the Proper Data Block Size (BLOCK SIZE)

Take **add_kernel** as an example. The variables and operations determine the on-chip memory usage. You can change the value of **BLOCK_SIZE** to adjust the size of the data block in the loop and the size of the intermediate result. If the upper limit is exceeded, the expected usage size is displayed and an error is reported during operator compilation. To achieve the maximum compute-to-memory ratio, **BLOCK_SIZE** needs to be as large as possible without exceeding the on-chip space. You can set different **BLOCK_SIZE** values in advance by using [autotune](#triton-autotune) of Triton-Ascend. The optimal setting is automatically selected during running.

```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, 
               y_ptr, 
               out_ptr, 
               n,  # Total number of elements.
               BLOCK_SIZE: tl.constexpr,  # Number of block elements.
               ): 
    pid = tl.program_id(0) 
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # The block size is BLOCK_SIZE.
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # Load data of x and y to the on-chip memory.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```

### Aligning the Size of the Tail Axis of the Tensor

[Description] For VV operators, if the Vector core needs to be called for computation, the UB of the Ascend hardware requires that the size of the tail axis of the tensor be divisible by 32 bytes. For CV operators, if the Vector core and Cube core need to be called for computation, the size of the tail axis of the tensor must be divisible by 512 bytes. If the tail axis length is insufficient, the tail axis length will be automatically padded. Under this premise, the performance of operations with the shape of (2048,3) and (2048,1) tensors in the model deteriorates significantly due to automatic padding. In this case, you can perform the transpose operation to convert the alignment axis to a lower dimension until the store operation is performed, avoiding automatic padding and optimizing the computing speed. In addition, the transpose operation is also affected by the automatic padding rule. Therefore, special skills are required to avoid padding. The following is a tip for "borrowing axis for transpose", which is applicable to the scenario where **tensor.numel() % 256Byte == 0**:

- Note: VV operators indicate that only Vector Core is used during operator computation. CV operators indicate that both AI Core and Vector Core are used during operator computation.
- Example

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # It is considered as the 1D tensor load. In this case, numel is aligned and no padding is performed.
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # The long axis (2048) is split into an aligned axis (16) and lent to the short axis (3) to align the two axes.
```

### Transferring Data to the UB and Then Selecting the Target Value from the UB

[Description] In the discrete scenario of the NPU, data can be transferred to the UB and then the target value can be selected from **share**.

- Example

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # the original code
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # the modified code
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

- Performance analysis and comparison before and after optimization

You can use the msProf tool to execute the test case to obtain the **PROF_***\** folder, which contains the **op_summary_***\****.csv** file. This file can be used to analyze the pipeline. Note: *\** indicates the timestamp. For details, see the [performance data collection methods](./debug_guide/profiling.md).

||Op Name|aiv_mte2_time(us)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Unoptimized|pick_kernel|0.686|0.008|
|Optimized|pick_kernel|1.041|0.066|

According to the data in the table, the values of **aiv_mte2_time(us)** and **aiv_mte2_ratio** before and after the optimization are greatly different. The optimization solution first transfers most of the data to the UB, reducing the number of times that small batches of data are transferred to the UB through the L2 and the total time for transferring data to the UB through the L2.

### Parallel Storage and Computation

Triton-Ascend supports two data processing modes: serial storage and computation and parallel storage and computation.

Serial storage and computation: Data is first transferred from the global memory to the on-chip memory, and then the next batch of data is transferred after the computation is complete. This mode has a significant idle waiting time, and the efficiency is low.

Parallel storage and computation: Computing is performed when the first batch of data is transferred to the on-chip memory. Then, the second batch of data is transferred, and the "transfer + compute" pipeline operation is formed, significantly improving the overall throughput.

The key to implementing parallel storage and computation is to properly design the data tiling policy so that the data required for the next phase can be prepared in advance during the compute of the current batch of data, thereby implementing parallelization of data transfer and computing.
 Currently, the compiler is configured with **multiBuffer** set to **True** by default, and the parallel storage and computation are supported by default.

### Tiling Optimization

Before the AI Core performs computation, data needs to be transferred to the on-chip memory. The on-chip memory space is usually much smaller than the total data volume to be processed by the AI Core. For example, the on-chip memory capacity of Atlas 800T/I A2 is 192 KB. After doublebuffer is enabled by default, the capacity is reduced to half of the original capacity. Therefore, data needs to be tiled during operator computation, and only a small part of the data is loaded and processed each time.

- Example

```diff
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
- def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
+ def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
+   base_offset = pid * BLOCK_SIZE
    
+   # Calculate the total number of blocks that need to be processed
+   num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB
    
+   # Loop processing each sub block
+   for sub_block_idx in range(num_sub_blocks):
+       # Calculate the offset of the current sub block
+       sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
+       offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
-       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    
        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)
 
        # Overlay and write value at the position that needs to be filled
-       value_to_write = tl.full([BLOCK_SIZE], value, dtype=input_vals.dtype)
+       value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

### Triton Autotune

In tiling block optimization, the values of block parameters such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** directly affect the operator performance. However, manually debugging parameter combinations is inefficient and it is difficult to find the optimal values. triton.autotune is an autotune tool provided by the Triton framework. It can traverse preset parameter configurations, compare the performance of different parameter configurations, and automatically select the optimal parameter combination. It is the core auxiliary means of tiling optimization.

- Core functions 
Automatically traversal of the parameter space: Test the performance of different values of block parameters of the constexpr type such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** in batches. 
Performance benchmark comparison: Select the optimal parameters that adapt to the current hardware based on the operator execution duration. 
Tuning result caching: The optimal configuration after tuning is cached. The optimal configuration is reused when the operator is called, avoiding repeated tuning. 

- Simple example

    ```diff
    import triton.language as tl

    @triton.autotune(
    configs=[ # List of parameter configurations to be tested. The candidate parameter values must be powers of 2.
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # Tune dimension: input dimension on which the parameter value depends.
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

- Note: You can set the following environment variables to print the optimal parameter information.

    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```

### How Do I Avoid UB Overflow on the NPU?

[Description] On the NPU, the UB or L1 size has an upper limit. When this error occurs, reduce the amount of data transferred at a time and use the for loop to process long sequences.

```diff
E triton.compiler. errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```

[Note] The UB size of the A2 series products is 192 KB (1,572,864 bits).

## Single-Core Data Computation

### R&D Goals

Implement basic data operation operators (such as addition, subtraction, multiplication, division, activation functions, and simple matrix element operations) on the Ascend NPU single core. Ensure that operators are efficiently executed on a single core, laying a foundation for subsequent multi-core parallel processing and distributed expansion. 

### Development Procedure

1.Determine the operator function. 
-Determine the shapes and data types (such as float16, float32, and int32) of the input and output tensors. 
-Check whether broadcast and boundary processing are required. 
 
2.Write kernel functions. 
Single-kernel computation corresponds to block-level data processing.   
Single-kernel data computation example: vector addition 

```diff

@triton.jit
def add_kernel(x_ptr, # Pointer to first input vector.
    y_ptr, # Pointer to second input vector.
    output_ptr, # Pointer to output vector.
    n_elements, # Size of the vector.
    BLOCK_SIZE: tl.constexpr, # Number of elements each program should process.
    # NOTE: constexpr so it can be used as a shape value.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Calling:

 ```diff
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

Use the above function to compute **element-wise sum** of two torch.tensor objects and test its correctness.

 ```diff
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
f'{torch.max(torch.abs(output_torch - output_triton))}')
# Out:
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')  
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')  
# The maximum difference between torch and triton is 0.0
```

3.Key points of single-kernel computation
-Block-level data processing: Each computing block is responsible for a small segment of data, ensuring parallelism.

-Boundary check: Use **mask** or **if (tid < N)** to avoid out-of-bounds access.

-Block size selection: Properly set the block and grid.
 
4.Performance points 
(1) Memory access optimization 
-Ensure sequential access. 
-Use the aligned stride to avoid cross-row/cross-column jump access. 
-Align the data block size to the 32-byte boundary. 
Ensure that the input and output buffers are aligned during allocation to avoid memory access performance deterioration. 
Example: 

 ```diff
BLOCK_SIZE = 256 # 256 x 4 bytes = 1024 bytes, which are well-aligned.

@triton.jit
def vec_add_kernel(X, Y, Z, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # Compute the index range of the current block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # The mask is used to prevent out-of-bounds.
    mask = offsets < N

    # Contiguous memory access: The offsets are contiguous.
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    z = x + y

    # Contiguous writeback
    tl.store(Z + offsets, z, mask=mask)


def vec_add(x, y):
    assert x.numel() == y.numel()
    N = x.numel()

    # Allocate aligned memory. (PyTorch is aligned to 64 bytes by default.)
    z = torch.empty_like(x)

    # grid: Each block processes BLOCK_SIZE elements.
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z
```

(2) Sub-block division 
-Divide a large matrix into small blocks. Each block is computed in the UB. 
-Sub-block division should ensure both memory access continuity and computing unit utilization. 
Example: 

 ```diff
BLOCK_M = 64   # Each block processes 64 rows.
BLOCK_N = 64   # Each block processes 64 columns.
BLOCK_K = 32   # Internal dimension is accumulated.

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # ID of the block in the M direction.
    pid_n = tl.program_id(1)  # ID of the block in the N direction.

    # Start coordinates of the current block.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute blocks in the loop.
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # Write back the result.
    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```
