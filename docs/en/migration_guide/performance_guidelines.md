# NPU High-Performance Programming Guide

## Combining Grid Cores
### I. Principles for Automatically Combining Grid Cores

Some scenarios requiring migration of Triton operators from GPUs to NPUs. Due to architectural differences, the Triton operators developed on GPUs often utilize large grid core counts. When executed on NPUs, these operators cannot be scheduled all at once. Delivering them in batches introduces significant latency and degrades performance. To optimize NPU-based Triton operators, you need to check the grid core counts first. In cases with large grid core counts, set the environment variable *TRITON_ALL_BLOCKS_PARALLEL* to improve operator execution performance.

## Optimizing Instruction Parallelism

### I. Core Principles of Instruction Parallelism Optimization

When executing Triton operators, NPUs leverage parallel mechanisms such as multi-buffer and instruction parallelism to parallelize data-in, computation, and data-out, thereby enhancing performance. However, in certain scenarios, the multi-buffer mechanism cannot be enabled, which reduces the degree of parallelism (DOP) and degrades operator execution performance. If this issue occurs during performance optimization, consider the following aspects and implement optimizations based on the provided code examples:\
1. Data transfer and computation involve dependencies, which introduce synchronization. The memory transfer engine (MTE) can only be triggered after vector computation completes, resulting in low DOP.\
2. In cases where the operator lacks multiple data loads or a single execution completes without tiling, the multi-buffer mechanism cannot be enabled.\
3. The multi-buffer mechanism requires additional UB space. If the UB space is insufficient during computation, the multi-buffer mechanism cannot be enabled.

### II. Code Examples

- Example 1: Reducing synchronization for higher DOP

    In operator optimization, increasing instruction-level parallelism (DOP) is a critical strategy. In the `tl.load` statement below, when `N` > `M`, the loaded data fills only a portion of the tensor memory space pointed to by `data`. For the remaining unfilled portion, if users do not specify the `other` value, GPUs default to zero-padding. To reduce the adaptation workload of migration, NPUs maintain the same behavior as GPUs. NPUs first use the vector core to set all the memory space pointed to by `data` to a specified value (defaulting to `0` if no `other` value is provided). Subsequently, the MTE2 instruction transfers data to part of the memory space pointed to by `data`. This implementation results in a dependency between MTE2 and vector operations, which limits parallelism and degrades overall performance.

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        N :tl.constexpr = BLOCK_SIZE
        idx = tl.arange(0, N)
        mask = idx < M
        data = tl.load(input + idx, mask = mask) # Alternatively, specify a value such as other=-1.
    ```

    To increase DOP and enhance performance, when the loaded data fills only a portion of the memory space pointed to by `data`, add `care_padding=False` to the load statement to remove default-value padding, provided that the unfilled portion does not affect subsequent computation results. That is, the preceding operator can be optimized follows:

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.arange(0, N)
        mask = idx < M
    -   data = tl.load(input + idx, mask = mask) # Alternatively, specify a value such as other=-1.
    +   data = tl.load(input + idx, mask = mask, care_padding=False) # Alternatively, specify a value such as other=-1.
    ```

- Example 2: Using `for` loops in Triton operators to increase tiling and enhance DOP

    In Triton operator programming, `mask` operations are frequently employed in syntax such as `load`, `store`, and `where`. During performance optimization, you should prioritize identifying performance degradation caused by these operations. When the logic within Triton operators executes sequentially in a single pass (Start -> Data-in -> Computation -> Data-out -> End), instructions cannot be parallelized, resulting in low execution efficiency. By introducing `for` loops to increase tiling, you can process data in multiple passes (with each pass handling a reduced data volume), enabling parallel execution of data-in, computation, and data-out. This approach reduces serial waiting time and improves overall performance. Additionally, compared to monolithic (non-tiled) data processing, the use of `for` loops for tiling reduces UB space consumption.
    Note: Mathematical equivalence is an important aspect to consider when you increase data tiling.

    ```diff
    @triton.jit
    def alloc_extend_kernel(
            pre_lens_ptr,
            seq_lens_ptr,
            free_page_ptr,
            out_indices,
            bs_upper: tl.constexpr,
            page_size: tl.constexpr,
            max_num_extend_tokens: tl.constexpr,
    +       BLOCK_SIZE: tl.constexpr = 1024,
    ):
        pid = tl.program_id(0)

        load_offset = tl.arange(0, bs_upper)
        seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
        pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
        extend_lens = seq_lens - pre_lens

        seq_len = tl.load(seq_lens_ptr + pid)
        pre_len = tl.load(pre_lens_ptr + pid)
        extend_len = seq_len - pre_len

        sum_extend_lens = tl.sum(extend_lens)
        output_start_loc = sum_extend_lens - extend_len

        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
                pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages = tl.sum(num_new_pages)
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        # Part 2: fill the new full pages
        num_part2 = (
                seq_len // page_size * page_size
                - (pre_len + page_size - 1) // page_size * page_size
        )

    -   # load data at once
    -   offset_many_page = tl.arange(0, max_num_extend_tokens)
    -   page_start = tl.load(
    -       free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    -       mask=offset_many_page < num_part2,
    -   )
    -   tl.store(
    -       out_indices + output_start_loc + offset_many_page,
    -       page_start * page_size + offset_many_page % page_size,
    -       mask=offset_many_page < num_part2,
    -   )

    +   # load data using loop
    +   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    +   blk_offset = tl.arange(0, BLOCK_SIZE)
    +   for i in range(num_loop):
    +       offset_many_page = blk_offset + i * BLOCK_SIZE
    +       page_start = tl.load(
    +           free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    +       tl.store(
    +           out_indices + output_start_loc + offset_many_page,
    +           page_start * page_size + offset_many_page % page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    ```


## Optimizing Data Types

### I. Core Principles of Data Type Optimization

Some operations of the A2/A3 vector units do not support certain data types. In this case, the corresponding vector operations will degrade to scalar operations, affecting performance. If the overall operator accuracy is not affected, it is advisable to use supported data types to improve performance.
The following operations are involved.
|  **Operator Name** |  **Unsupported Data Type** |
|---|---|
| Vector Add| int64 |
| Vector Cmp| int64/int32 |

### II. Code Examples

- Example code of the Triton operator Vector Add

    For the following Triton operator, when the input tensors `x` and `y` utilize the int64 data type, `x1 + y1` is expanded into a scalar operation, which degrades performance. Provided that computational accuracy remains unaffected, it is advisable to use the int32 data type.
    ``` diff
    @triton.jit
    def npu_vector_add_kernel(
        x,                          # [Tensor] input tensor (1 x col)
        y,                          # [Tensor] input tensor (1 x col)
        z,                          # [Tensor] output tensor (1 x col)
        vector_len: tl.constexpr,   # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE)
        len_mask = offset < vector_len
        x1 = tl.load(x + offset, mask=len_mask)
        y1 = tl.load(y + offset, mask=len_mask)
        z1 = x1 + y1
        tl.store(z + offset, z1, mask=len_mask)
    ```

- Example code of the Triton operator Vector Cmp

    In the following Triton operator, the `mask` operation utilizes Cmp. However, Cmp does not support the int64 or int32 data type, causing the condition `cols < N` to be expanded into a scalar operation, which reduces performance. Provided that computational accuracy remains unaffected, it is advisable to use the FP32 data type.
    In Triton operator programming, `mask` operations are frequently employed in syntax such as `load`, `store`, and `where`. During performance optimization, you should prioritize identifying performance degradation caused by these operations.

    ``` diff
    @triton.jit
    def npu_vector_cmp_kernel(
        X,                 # [Tensor] input tensor (row x col)
        Out,               # [Tensor] output tensor (row x col)
        Mean,              # [Vector] mean tensor (row, ) of X
        Rstd,              # [Vector] std tensor (row, ) of X
        stride_x_row,      # [Scalar] stride of row of x
        stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
        M,                 # [Scalar] row number
        N,                 # [Scalar] col number
        eps,               # [Scalar] epsilon to avoid division by zeros
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        group_m = tl.program_id(0)
        group_n = tl.program_id(1)
        row = group_m

        # calculate index & offset
        Mean = Mean + group_n * M
        Rstd = Rstd + group_n * M
        X = X + row * stride_x_row + group_n * N
        Out = Out + row * stride_out_row + group_n * N

        cols = tl.arange(0, BLOCK_N)  # cols is int64
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

        # calculate mean & rstd
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        # [Changed begin]
    -   xbar = tl.where(cols < N, X - mean, 0.0)
    +   cols_cmp = cols.to(tl.float32)
    +   xbar = tl.where(cols_cmp < N, x - mean, 0.0)
        # [Changed end]

        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)

        # calculate Out
        mask = cols < N
        out = (x - mean) * rstd
        tl.store(Out + cols, out, mask=mask)
    ```
