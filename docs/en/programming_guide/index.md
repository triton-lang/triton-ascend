# Triton Operator Development Guide

This document focuses on the issues that need to be paid attention to during Triton operator development on NPUs, which are divided into three aspects: multi-core task parallelism, single-core data transfer, and single-core data computation. First, section "Multi-Core Task Parallelism" describes the basis for setting the maximum number of hardware cores and the specific implementation. Then, section "Single-Core Data Transfer" describes how to set the proper data block size in a loop, introduces the common optimization methods, and describes how to handle the UB overflow problem that may occur. Finally, section "Single-Core Data Computation" focuses on how to develop Triton operators and highlights the key points.

## Documentation Structure

This guide separates common development rules from operator-specific development paths:

- This page covers common Triton-Ascend concerns, including core allocation, on-chip memory, memory access, tiling, and autotune.
- [Vector Operator Development](./vector_operator.md) describes element-wise, reduction, gather/scatter, and other operators mainly executed on Vector Cores.
- [Cube Operator Development](./cube_operator.md) describes operators whose main computation is `tl.dot`, matrix multiplication, or batched matrix multiplication.
- [CV Fusion Operator Development](./cv_fusion_operator.md) describes operators that combine Cube computation with Vector post-processing, reductions, softmax, or cross-core coordination.

For simple operators, refer to this repository's `docs/en/examples/` and `third_party/ascend/tutorials/`. For complex operators, refer to complete optimization cases in `tutorial/best_practice/` of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops).

## Common Multi-Core Task Parallelism

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
_attn_fwd[grid](Q, K, V, M, Out, acc, scale, ...)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ...,
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

## Common Single-Core Data Transfer

### Setting the Proper Data Block Size (BLOCK SIZE)

Take **add_kernel** as an example. The variables and operations determine the on-chip memory usage. You can change the value of **BLOCK_SIZE** to adjust the size of the data block in the loop and the size of the intermediate result. If the upper limit is exceeded, the expected usage size is displayed and an error is reported during operator compilation. To achieve the maximum compute-to-memory ratio, **BLOCK_SIZE** needs to be as large as possible without exceeding the on-chip space. You can set different **BLOCK_SIZE** values in advance by using [autotune](../examples/06_autotune_example.md) of Triton-Ascend. The optimal setting is automatically selected during running.

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

[Description] The UB (Unified Buffer) of the Ascend hardware imposes alignment requirements on the size of the tail axis (the last dimension) of a tensor: the data size of the tail axis must be an integer multiple of the alignment unit. If the tail axis length is insufficient, the hardware automatically pads it. The alignment requirements for different operator types are shown in the following table.

| Operator Type | Compute Unit Used | Tail Axis Alignment Requirement |
| --- | --- | --- |
| VV operators (Vector-Vector, pure vector computation operators) | Vector core only | Tail axis size must be divisible by 32B |
| CV operators (Cube-Vector, Cube core and Vector core fusion computation operators) | Cube core + Vector core | Tail axis size must be divisible by 512B |

[Description] Various operations on tensors with shapes (2048,3) and (2048,1) in the model suffer noticeable performance degradation due to automatic padding. In this case, consider transposing the alignment axis to a lower dimension, and transpose it back to the original state only when storing, thereby avoiding automatic padding and improving computation speed.
Since the transpose operation itself is also affected by the automatic padding rule, special techniques are needed to avoid padding. The following lists a "borrowing axis for transpose" tip, applicable to the scenario where **tensor.numel() % 256Byte == 0**:

- Note: VV operators indicate that only Vector Core is used during operator computation. CV operators indicate that both AI Core and Vector Core are used during operator computation.
- Example

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # It is considered as the 1D tensor load. In this case, numel is aligned and no padding is performed.
conv_state_T = conv_state.reshape(128, 16 * 3).transpose().reshape(16, 3 * 128).transpose().reshape(3 * 2048,) # The long axis (2048) is split into an aligned axis (16) and lent to the short axis (3) to align the two axes.
```

### Transferring Data to the UB and Then Selecting the Target Value from the UB

[Description] In the discrete scenario of the NPU, data can be transferred to the UB and then the target value can be selected from **share**.

- Example

```python
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

You can use the msProf tool to execute the test case to obtain the **PROF_***\** folder, which contains the **op_summary_***\****.csv** file. This file can be used to analyze the pipeline. Note: *\** indicates the timestamp. For details, see the [performance data collection methods](../debug_guide/profiling.md).

|Optimization State|Op Name|aiv_mte2_time(μs)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Unoptimized|pick_kernel|0.686|0.008|
|Optimized|pick_kernel|1.041|0.066|

Description of each metric:

- **aiv_mte2_time(μs)**: Time consumed during the MTE2 (Move Engine 2) transfer stage on the AI Vector (AIV) core, in microseconds (μs), reflecting the overhead of moving data from global memory to on-chip memory (UB).
- **aiv_mte2_ratio**: The ratio of MTE2 transfer time to the total operator execution time. A larger value indicates a higher proportion of transfer time, which can be used to evaluate the degree of overlap between transfer and computation.

According to the data in the table, the values of aiv_mte2_time(μs) and aiv_mte2_ratio before and after optimization differ significantly. The optimization solution first transfers most of the data to the UB, reducing the number of times small batches of data are transferred from the L2 to the UB, thereby reducing the total time of transferring data from the L2 to the UB.

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
+   num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

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

In tiling block optimization, the values of block parameters such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** directly affect operator performance. However, manually trying parameter combinations is inefficient and makes it difficult to find the best values. `triton.autotune` is the autotuning utility provided by the Triton framework. It can sweep over preset parameter configurations, compare their performance, and automatically select the best combination. It is a core tool for tiling optimization.

For the recommended Triton-Ascend usage of `configs=[]`, the scope of automatic tiling, see the [Triton-Ascend Autotune Guide](./../autotune_guide.md).

- Core functions
Automatic exploration of the parameter space: Test different values of constexpr block parameters such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** in batches.
Benchmark-based comparison: Select the optimal parameters for the current hardware based on execution time.
Caching of tuning results: Cache the best configuration after tuning so that later calls can reuse it without tuning again.

- Simple example

    ```python
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

    ```bash
    export TRITON_PRINT_AUTOTUNING=1
    ```

### Advanced: Using max_autotune for Autotuning

For Ascend NPU operators, achieving optimal performance requires tuning not only BLOCK_SIZE but also multiple hardware-related parameters such as num_stages, enable_hivm_auto_cv_balance, and tile_mix_vector_loop. Using @triton.autotune to manually enumerate all combinations would lead to an explosive growth in the configuration list, making the code difficult to maintain.

max_autotune is an extension decorator designed specifically for Ascend NPU (located in triton.backends.ascend.runtime), allowing users to provide only base configurations while passing other tuning parameters as lists. The decorator automatically generates a complete Config list of all combinations.

- Core Function
Developers only need to provide a few base configurations (such as BLOCK_SIZE), and all compiler options related to that operator type (for example, num_stages, enable_hivm_auto_cv_balance, tile_mix_vector_loop, enable_ubuf_saving, etc.) will be automatically included in the optimal combination search range through built-in reasonable default values. Developers don't need to explicitly enumerate them, achieving one-time automatic optimization of both optimal tiling and compiler option combinations. If developers want to constrain certain parameters, they can also override the default search range by explicitly passing lists.

- Simple Example

    ```python
    from triton.backends.ascend.runtime import max_autotune

    @max_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
        ],
        key=['n_elements'],
        kernel_type="vector",           # Operator type, supports cube/mix/vector
        enable_ubuf_saving=[True, False] # Optional, already included by default
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, **META):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

### How Do I Avoid UB Overflow on the NPU?

[Description] On the NPU, on-chip memories such as UB (Unified Buffer) and L1 (level-1 on-chip cache) have hardware capacity limits. When the on-chip buffer required by a single tile (block) for data transfer and computation in a kernel exceeds this limit, the compiler reports a `ub overflow` error during compilation, and the kernel cannot be compiled. This error is common in the following scenarios: the block size parameter (such as BLOCK_SIZE) is set too large, long sequences are transferred in a single pass, there are too many intermediate tensors in the kernel, and multi-buffer (parallel storage and computation with multiple buffers) is enabled by default, causing some tensors to require multiple copies. When this error occurs, you need to reduce the amount of data transferred at a time: reduce the block size parameter, or use a for loop within the kernel to process long sequences in blocks along the sequence dimension.

#### Error Message Example

The following is the Triton compilation failure log echoed by pytest: the `E` at the beginning of each line is a pytest error output prefix (indicating that the line comes from pytest-captured error output) and is not part of the log content itself.

```text
E triton.compiler.errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```

#### Error Cause Analysis

1. **Trigger phase:** This error occurs during kernel compilation — the Ascend backend fails when converting MLIR to binary, not during program runtime; compilation failure means no executable binary was generated for the kernel.

2. **Direct cause:** The compiler estimated that a single tile requires 3,072,256 bits of UB, while the hardware UB upper limit is 1,572,864 bits (A2 series, i.e., 192 KB), exceeding the limit by 1,499,392 bits (3,072,256 − 1,572,864 = 1,499,392), approximately 1.95 times the available capacity, resulting in a ub overflow.

3. **Root cause:** ① The block size parameter (such as BLOCK_SIZE) is too large, resulting in too many elements in a single tile; ② Multi-buffer parallel storage and computation is enabled by default (the compiler-side configuration is `multiBuffer=True`, controlled by the user via the `multibuffer` parameter in `triton.Config` or kernel launch, enabled by default), creating multiple tensor copies for pipeline overlap between data transfer and computation, with some operators requiring additional local buffers, multiplying UB usage; ③ Too many intermediate tensors in the kernel, with cumulative usage exceeding the limit.

The key log fields are interpreted as follows:

| Log Key Field | Meaning |
| --- | --- |
| `triton.compiler.errors.MLIRCompilationError` | Compilation error thrown by the Triton compiler during the MLIR compilation phase; the kernel did not generate an executable binary |
| `[ConvertLinalgRToBinary] encounters error`, `Failed to run BishengHIR pipeline` | The Ascend backend failed during the MLIR-to-binary conversion phase |
| `ub overflow` | UB (Unified Buffer) overflow: the estimated on-chip buffer requirement during compilation exceeds the hardware capacity limit |
| `requires 3072256 bits` | UB bits required for a single tile in this compilation: 3,072,256 bits (approximately 375 KB) |
| `1572864 bits available` | Current hardware available UB upper limit: 1,572,864 bits, i.e., 192 KB (A2 series) |
| `multi-buffer feature is enabled and some ops need extra local buffer` | Compiler hint: multi-buffer (parallel storage and computation with multiple buffers) is enabled, and some operators require additional local buffer copies, which amplifies UB usage |

#### Resolution Steps

Process in the following order; recompile and verify after each step; stop once the overflow disappears after any step:

1. **Reduce the block size parameter.** Reduce BLOCK_SIZE and other block size parameters by powers of 2 (e.g., from 4096 to 2048, 1024). The grid will automatically expand with the block size (grid = ceil(n_elements / BLOCK_SIZE)), and the total number of elements processed remains unchanged.

    Before modification (BLOCK_SIZE too large, triggering ub overflow):

    ```python
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=4096)
    ```

    After modification (reduced BLOCK_SIZE):

    ```python
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    ```

2. **Use a for loop to process long sequences in blocks along the sequence dimension.** When the sequence length is far greater than the number of elements that a single tile can safely hold, use a for loop within the kernel to split along the sequence axis: load, computation, and store are all completed within the loop body, processing only BLOCK_SIZE_SUB elements at a time, and only one sub-block of data resides in the UB at any given time. Parameter constraints: BLOCK_SIZE_SUB should not be greater than BLOCK_SIZE, and should preferably be a power of 2 (e.g., 1024, 2048) and evenly divide BLOCK_SIZE; when it cannot evenly divide, the tail block is automatically handled by the mask, which does not affect correctness. The complete kernel example is as follows:

    ```python
    @triton.jit
    def add_kernel_tiled(x_ptr, y_ptr, out_ptr, n_elements,
                        BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
        pid = tl.program_id(axis=0)
        base_offset = pid * BLOCK_SIZE

        # Calculate the total number of sub-blocks to be processed by the current program
        num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

        # Process in blocks along the sequence dimension: transfer/compute only BLOCK_SIZE_SUB elements at a time
        for sub_block_idx in range(num_sub_blocks):
            sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
            offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0)
            tl.store(out_ptr + offsets, x + y, mask=mask)
    ```

    Compared with the large-block approach that triggers overflow (loading the entire BLOCK_SIZE at once in the kernel, e.g., `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`), the key difference is: the blocked approach splits the same amount of data into the for loop for sub-block transfer and computation, reducing the single transfer data volume from BLOCK_SIZE to BLOCK_SIZE_SUB.

3. **Disable multi-buffer parallel storage and computation.** If reducing the block size still causes overflow, set `multibuffer=False` in the autotune configuration to disable multi-buffering (corresponding to the compiler-side `multiBuffer=True` configuration enabled by default), reducing the additional local buffer usage from tensor copies:

    ```python
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, multibuffer=False),
            triton.Config({'BLOCK_SIZE': 2048}, multibuffer=False),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    ```

    It can also be passed as a meta-parameter during kernel launch:

    ```python
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024, multibuffer=False)
    ```

    Disabling multi-buffer reduces the pipeline overlap between data transfer and computation and may affect performance. It is recommended only when UB is tight, and you should prioritize using autotune to compare and trade off between multiple configurations (see the "Parallel Storage and Computation" section of this document).

4. **Recompile and verify.** Rerun the kernel compilation or test cases and confirm that `ub overflow` and `MLIRCompilationError` errors no longer appear in the log. The kernel compiles successfully and the output results are consistent with expectations, which means the issue is resolved. If overflow still occurs after all the above steps, continue reducing the block parameters by comparing the `requires` and `available` values in the log, and verify the UB specifications of the target product (see the specification table at the end of this section).

#### Preventive Measures

| Preventive Measure | Description |
| --- | --- |
| Estimate UB usage before coding | The UB usage (bit) of a single tile ≈ number of tile elements × bytes per element × number of buffer copies × 8; in practice, additional intermediate tensors and alignment overhead must be added, so leave sufficient margin. It is recommended to start with BLOCK_SIZE_SUB from 1024 to 4096 and verify with autotune |
| Prioritize autotune | Use `triton.autotune` to enumerate multiple BLOCK_SIZE configurations (including the `multibuffer` switch), and let the runtime automatically select the combination that does not overflow and has the best performance, avoiding manual trial and error (see the "triton.autotune Automatic Tuning" section of this document) |
| Pay attention to multi-buffer impact | The compiler enables parallel storage and computation by default (`multiBuffer=True`), and multi-buffering increases UB usage proportionally with the number of buffer copies; when UB is tight, set `multibuffer=False` in Config (see the "Parallel Storage and Computation" section of this document) |
| Batch processing for long sequences | When the sequence length causes the estimated usage of a single tile to approach or exceed the UB limit, use a for loop within the kernel to load/compute/store in blocks along the sequence dimension, avoiding transferring the entire segment at once |

UB usage estimation example: With 65,536 float32 elements (4 bytes) and double buffering (2 copies), a single tensor requires 65,536 × 4 × 2 × 8 = 4,194,304 bits, which already exceeds the A2 series available 1,572,864 bits; based on the A2 series 1,572,864 bits limit, the theoretical upper limit for a single float16 tensor with double buffering is approximately 49,152 elements (1,572,864 ÷ 8 ÷ 2 ÷ 2 = 49,152).

[Note] The UB size of the A2 series products is 192KB (i.e., 1,572,864 bits). The on-chip memory specifications of each product series are as follows:

| Product Series | UB Capacity | L1 Capacity |
| --- | --- | --- |
| A2 series | 192K B | 512 KB |
| A3 series | 192K B | 512 KB |
| A5 series | 248K B | 512 KB |

## Common Single-Core Data Computation

### R&D Goals

Implement basic data operation operators (such as addition, subtraction, multiplication, division, activation functions, and simple matrix element operations) on the Ascend NPU single core. Ensure that operators are efficiently executed on a single core, laying a foundation for subsequent multi-core parallel processing and distributed expansion.

### Development Procedure

1.Determine the operator function.
-Determine the shapes and data types (such as float16, float32, and int32) of the input and output tensors.
-Check whether broadcast and boundary processing are required.

2.Write kernel functions.
Single-kernel computation corresponds to block-level data processing.
Single-kernel data computation example: vector addition

```python

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

 ```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

Use the above function to compute **element-wise sum** of two torch.tensor objects and test its correctness.

 ```python
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

 ```python
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

 ```python
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

## General Multi-Dimensional Tensor Tiling

When processing multi-dimensional tensors in Triton operators, the core idea is to map high-dimensional data to the hardware's Blocks, Cores, and hardware units. This section provides typical examples for processing two-dimensional and three-dimensional tensors.

### Two-Dimensional Tensor Tiling: A Matrix Multiplication (GEMM) Example

For two-dimensional matrix multiplication, two-dimensional tiling is typically performed along the height (M) and width (N) dimensions, with iterative looping along the depth (K) dimension.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 1. Task division: compute the coordinates of the current Block in the M and N dimensions.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Define Block Pointers to handle multi-dimensional strides.
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 3. Loop over the K dimension to perform accumulation.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, accumulator)
```

**Key Points**:

- `pid_m` / `pid_n` correspond to the block indices in the M and N dimensions respectively.

- `stride_*` explicitly handles multi-dimensional strides, avoiding assumptions about contiguous memory.

- The K dimension is accumulated through loop-based block tiling.

### Three-Dimensional and Higher Tensor Tiling: A Batched GEMM Example

When processing a three-dimensional tensor (e.g. `[Batch, M, N]`), the `Batch` dimension (B) can be mapped directly to a Triton `Grid` dimension, or it can be flattened together with the M/N dimensions and remapped.

#### Adding a `Batch` dimension to the Grid launch

```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), B)
```

#### Kernel implementation

```python
@triton.jit
def batched_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, B, ...):
    # Obtain the index of the current Batch.
    pid_b = tl.program_id(2)

    # Compute the base address offset in global memory based on the Batch index.
    a_batch_ptr = a_ptr + pid_b * M * K
    b_batch_ptr = b_ptr + pid_b * K * N
    c_batch_ptr = c_ptr + pid_b * M * N

    # Subsequent tiling of the M, N, and K dimensions is identical to the 2D GEMM;
    # only the base pointers need to be replaced.
    # ...
```

**Key Points**:

- `tl.program_id(2)` obtains the index of the Batch dimension.

- Each Batch independently computes its own `a_batch_ptr`, `b_batch_ptr`, and `c_batch_ptr`.

- Subsequent tiling logic for the M / N / K dimensions is consistent with the 2D GEMM.
