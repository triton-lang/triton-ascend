# NPU高性能编程指南

## 合并Grid分核

### 一、自动合并Grid分核优化原则

部分场景下，Triton算子从GPU迁移到NPU。由于体系结构的差异，基于GPU开发的Triton算子Grid分核数较多。在NPU上执行时，无法一次全部调度，多轮下发导致下发时延过大，影响算子性能。基于NPU优Triton算子过程中，需要首先检查Grid分核数。当分核数较大时，使用TRITON_ALL_BLOCKS_PARALLEL环境变量提升算子执行性能。

## 指令并行优化

### 一、指令并行优化核心原则

Triton算子在NPU上执行时，为了提升性能，NPU底层提供multi buffer、指令并行等并行机制，将“数据搬入/数据计算/数据搬出”并行起来，以此来提升性能；但是某些场景下存在multi buffer无法使能问题，影响并行度，导致算子执行性能降低；在性能优化过程中，存在此类问题时，可以参考以下几点做排查，并按照代码示例优化：\
1、数据搬运和计算存在数据依赖，产生同步，必须依赖Vector运算后，才能触发MTE搬运，导致并行度低；\
2、算子内，无多个数据加载或者单次执行完成无Tiling切分，该场景下无法使能multi buffer；\
3、multi buffer需要额外增加UB空间的使用，计算过程中UB空间不足，无法使能multi buffer；

### 二、代码示例

- 示例1：减少同步，提升并行度

    在算子调优过程中，增加指令并行度是算子调优的重要手段。在如下的tl.load语句中，当N > M时, load加载的数据只能填充部分data指向的tensor内存空间中，剩下未填充的部分，如果用户未指定other值，则GPU默认填充为0，为了减少用户迁移的适配工作，NPU保持行为和GPU一致。NPU会先用Vector核对data指向的全部内存空间设置为指定值(如果用户未指定other值，同样设置为0)，然后在使用MTE2指令搬运数据到data指向的部分内存空间，这样就会导致MTE2和Vector产生依赖，无法高效并行，影响性能：

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
        data = tl.load(input + idx, mask = mask) # 或者指定other=-1等值
    ```

    为了提升性能，在load加载数据只能部分填充到指向的内存空间时，如果未填充的部分不影响后续的计算结果，可以在load语句中，添加care_padding=False来去掉默认值的填充，增加并行度，提升性能，上面算子的优化写法如下：

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
    -   data = tl.load(input + idx, mask = mask) # 或者指定other=-1等值
    +   data = tl.load(input + idx, mask = mask, care_padding=False) # 或者指定other=-1等值
    ```

- 示例2：在Triton算子内，使用for循环，增加Tiling，提升并行度

    在Triton算子编程中，mask运算经常运用在load/store/where等语法中，在性能优化过程中，需要特别注意这类运算导致的性能下降。当Triton算子内逻辑是单次顺序执行，开始->数据搬入->计算->数据搬出->结束，指令无法并行，执行效率低;可以通过在算子使用for循环增加tiling，将单次处理量减少，多次处理，能够让"数据搬入/计算/数据搬出"并行起来，减少串行的等待时间，提升整体性能；同时使用for循环增加Tiling，也能降低单次处理消耗的UB空间。
    需要注意：增加数据Tiling也同时需要考虑改变数据切块后的数学是否等价。

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

## 数据类型优化

### 一、数据类型优化核心原则

A2/A3向量运算单元的部分运算操作不支持某些数据类型，这种场景下，对应的矢量运算会退化为标量运算，影响性能，在确定不影响整体算子精度的情况下，建议使用支持的数据类型，提升性能
主要涉及以下操作

|  **OP名称**  |  **不支持的数据类型**  |
|---|---|
| Vector ADD | int64 |
| Vector CMP | int64/int32 |

### 二、代码示例

- Vector Add Triton算子示例代码

    如下Triton算子，当x, y input tensor使用的数据类型是int64时，会导致x1+y1运算展开为Scalar运算，降低性能，在不影响精度的情况下，建议使用int32数据类型。

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

- Vector Cmp Triton算子示例代码

    如下Triton算子，做mask运算时，使用到了Cmp操作，Cmp不支持int64/int32数据类型，会导致cols < N运算展开为Scalar运算，降低性能，在不影响精度的情况下，建议使用fp32数据类型。
    在Triton算子编程中，mask运算经常运用在load/store/where等语法中，在性能优化过程中，需要特别注意这类运算导致的性能下降。

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
