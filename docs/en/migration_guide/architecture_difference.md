# Development Differences Between Ascend and GPUs

## Multi-Core Task Parallelism Strategy

In Triton multi-core parallelism, NPUs use a strong physical-core binding model, which is a core difference from the GPU model of logical-dimension parallelism with automatic hardware mapping. The core comparison is shown in the following table:

|Dimension      |GPU (NVIDIA)|Ascend|
|-----------|--------------|-----------|
|Essence of grids| Logical task dimension (decoupled from physical cores)| Physical core group mapping (bound to the AI core topology)|
|Limit on the number of cores/dimensions| No hard limit on the grid dimensions/sizes| Grid size ≤ Total number of AI cores; topology matching required by 2D|

GPUs can be bound to multiple dimensions (a 3D grid of `[n, m, l]` is equivalent to`n × m × l` parallel threads). Each thread corresponds to only one kernel execution and executes only once.\
In NPUs, vector cores and cube cores belong to multiple physical cores. The number of cores varies with the generation of hardware. Each core executes only one block and can schedule the block execution repeatedly.

### Full Utilization of Cores

Ascend NPUs have multiple computing cores. Properly allocating and fully utilizing all available cores is one of the key factors to improve operator performance.
When calling Triton kernel functions, you can set the **launch** parameter to control the number of cores in use. Take the GELU operator as an example:

```Python
triton_gelu[n, 1, 1](...)  # The first parameter indicates the number of cores in use. n indicates that n cores are in use.
```

By optimizing the number of cores, you can fully schedule and utilize all computing resources, thereby maximizing the degree of parallelism (DOP) and throughput. Without `auto-blockify` (see below), the number of cores in the launched grid must be less than or equal to 65,535.

### Auto-Blockify: lifting the 65,535 logical-block limit

Upstream Triton on NVIDIA GPUs treats the grid as a pure logical dimension — `n` logical blocks map 1:1 to `n` hardware blocks, and the runtime expands the work across SMs without any per-block iteration. On Ascend, the strict physical-core binding above caps the launchable grid at 65,535, which is restrictive for kernels with millions of logical work items (autotuned reduce/scan, megablocks-style sparse kernels, etc.).

`auto-blockify` (the `SIMTAutoBlockify` compiler pass plus a matching runtime cap) removes that limit by treating the grid as logical at compile time and folding it onto the physical cores at launch:

- **Compile time**: a Triton pass wraps the kernel body in an `scf.for` over per-block work, indexed by `gpu.linear_block_id`. The chunk size is `ceildiv(logical_block_count, physical_core_count)`, so each physical block iterates through `chunk` logical block IDs.
- **Runtime**: the block-count argument passed to the launcher is clamped from the logical grid down to `physical_core_count`, mirroring the compile-time fold.

The two sides share the same gating metadata (`enable_auto_blockify` on `NPUOptions`, falling back to `TRITON_ALL_BLOCKS_PARALLEL`), so the compile-time loop wrap and the runtime cap are always in sync — a kernel never compiles for one mode and launches for another.

Practical implications when porting a GPU Triton kernel:

- A grid larger than 65,535 just works; no need to manually fold the outer dimension into the kernel body.
- Logical blocks must remain order-independent (the loop visits them in chunk order). Kernels that assume strict logical block-id ordering within a single launch (e.g., explicit cross-block synchronization on a particular order) need to be rewritten.
- Per-block workspace allocations become `O(physical_core_count)` rather than `O(logical_block_count)`, because workspace is reused across iterations of the inner `scf.for`.

## Single-Core Data Transfer Strategy

### Data Tiling

When you write Triton kernel functions, a proper data tiling strategy is essential for performance optimization. By adjusting tiling granularity parameters, you can balance computational workload and memory access efficiency across different dimensions.

Common tiling parameters include:

```text
ncore: the number of cores in use (cross-core tiling)
xblock: the size of inter-core data blocks (inter-core tiling)
xblock_sub: the granularity of intra-core tiling (fine-grained intra-core tiling)
```

By manually selecting the optimal tiling configurations based on your actual scenario, you can maximize the utilization of on-chip memory during each computation cycle, preventing performance bottlenecks caused by frequent access to the global memory.

Taking the GELU operator as an example, adjusting the tiling parameters helps effectively adapt to the on-chip cache capacity limit, thereby improving execution efficiency.

Note: Atlas 800T/I A2 has an on-chip memory capacity of 192 KB. When designing the tiling strategy, ensure that the data volume of each computation cycle does not exceed this capacity.

#### Example GELU Operator

The following demonstrates the development of an example GELU operator with three result computation methods.

`standard_unary` is standard Torch computation.

`triton_easy_kernel` is a simple implementation of Triton.

`triton_better_kernel` is a more efficient implementation of Triton.

#### Standard Torch Writing

After computing the input `tensor x0`, Torch implements the GELU operator and returns the result value.

```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```

#### Simple Triton Writing

The following is an example of a simple kernel written in Triton, demonstrating how to define and call a basic Triton kernel function. This example implements a simple mathematical operation (GELU activation function).

```Python
# Define the triton_kernel function.
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```

Precautions

1. Memory limit: In the preceding writing, all input data is loaded to memory at a time for computation. If the input tensor is too large, it may exceed the on-chip memory capacity of a single kernel, resulting in a memory overflow error.
Therefore, this simple writing is suitable for computing small-scale tensors or for understanding the basic writing and call method of Triton kernels.

2. Application scenarios: This method helps developers quickly understand and get started with Triton programming. However, for large-scale data sets or scenarios demanding high performance, developers are advised to use more complex data tiling strategies to fully utilize hardware resources and prevent memory overflow. In this way, developers can quickly get started with Triton programming and understand how to define, call, and optimize Triton kernel functions.

#### More Efficient Triton Writing

When using Triton to write high-performance operators on Ascend NPUs, developers need to use a data tiling strategy to fully utilize hardware resources, prevent memory overflow, and improve execution efficiency.
The following is an example of an optimized Triton kernel implementation suitable for large-scale tensor computation.

```Python
# Define the triton_kernel function.
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# Call the triton_kernel function.
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

Explanation of key code:

```Python
# Calculate the start offset address of the data block processed by the current core to implement inter-core tiling. Each core is responsible only for a data segment of size XBLOCK.
xoffset = tl.program_id(0) * XBLOCK

# Further split the data block within a single core to process data of size XBLOCK_SUB each time, which is known as intra-core tiling.
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

# Construct the data index array of the current iteration. This array is used to access the input and output tensors.
x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

# Set a mask to prevent out-of-bounds access and ensure that only data within the defined range is processed.
xmask = x_index < xnumel

# Load data from the global memory to the on-chip memory and write the computation results back to the global memory.
tl.load() and tl.store()
```

## Compilation Optimization

### Ascend NPU IR Optimization

The following table lists the compilation options for Ascend NPU IR optimization, which are adapted to the hardware and software features of Ascend.
**Usage**: During the autotune configuration phase, pass the values of the compilation options.
For example, to enable the `multibuffer` option, pass `'multibuffer': True` to `triton.Config` during the autotune configuration phase. For details, see [Autotune Example](../examples/06_autotune_example.md).

```python
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),]
```

| Option     | Capability      | Enabled or Not|
| ----------------- | ------------ | ----------------- |
| multibuffer                                   | Data transfer through parallel pipelines. | Default: **true**. Options: **true** and **false**. It is configurable during autotune.                    |
| unit_flag                                     | Optimization item for cube-out.                                        | Default: None. Options: **true** and **false**.  It is configurable during autotune.                    |
| limit_auto_multi_buffer_only_for_local_buffer | Optimization item for CV operators and cube-out.                        | Default: None. Options: **true** and **false**. It is configurable during autotune.|
| limit_auto_multi_buffer_of_local_buffer       | Scope of enabling double buffer for cube operators.                        | Default: None. Value: "no-limit" or "no-l0c". It is configurable during autotune.          |
| set_workspace_multibuffer                     | Configures workspace multi-buffer levels to enable multi-buffering for workspace-related data movement.| Default: None. Use a single value, for example 2 or 4. Candidate values can be configured during autotune.                           |
| enable_hivm_auto_cv_balance                   | Enables or disables automatic CV balance to balance Cube and Vector execution in CV fusion scenarios.| Default: None. Options: **true** and **false**. It is configurable during autotune.|
| tile_mix_vector_loop                          | Optimization item for CV operators. It specifies the number of segments into which the current vector can be split.                       | Default: None. Use a single value, for example 2, 4, or 8. Candidate values can be configured during autotune.                      |
| tile_mix_cube_loop                            | Optimization item for CV operators. It specifies the number of segments into which the current cube can be split.     | Default: None. Use a single value, for example 2, 4, or 8. Candidate values can be configured during autotune.                     |
| auto_blockify_size                            | Optimization item for TRITON_ALL_BLOCKS_PARALLEL. It specifies the size of leftmost dimension to be expanded.     | Default: 1. Use a single integer value, for example 2, 4, or 8. Candidate values can be configured during autotune.                     |
| enable_auto_blockify                          | Per-kernel override for the TRITON_ALL_BLOCKS_PARALLEL env var. When set to **true** or **false**, the kernel uses that value regardless of the env var; when left unset (None), the env var decides. Resolution order: this option > env var > off. Both the compile-time blockify pass and the runtime cap on the launched block count follow this resolved value, so they always agree. | Default: None. Options: **true**, **false**, None. |

- Note: The compilation optimization options are located in **ascend/backend/compiler.py**.
- Note: CV operators indicate that both AI cores and vector cores are used during operator computation.
