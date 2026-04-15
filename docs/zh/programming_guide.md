# Triton算子开发指南
概述：本文着重介绍了在NPU上进行Triton算子开发中值得注意的问题，分为三个方面：多核任务并行、单核数据搬运、单核数据运算。首先在多核任务并行中介绍了设置最大硬件核数的依据以及具体的实现。然后在单核数据搬运中具体描述了如何设置合适的循环内数据分块大小，并且介绍了过程中常用的优化手段，还补充了可能面临的UB OVERFLOW问题的处理方式。最后回归单个算子，在单核数据运算层面着重介绍了如何开发Triton算子，并强调了相关的关键点。

## 多核任务并行

### 设置最大硬件核数

在一个Triton算子中，通常使用grid进行分核操作。对于GPU而言，其计算核心SM通常是几十到几百量级。但是对于昇腾 NPU 平台而言，其计算核心AI Core的数量在几十个的量级。\
虽然运行时接口允许下发并发任务数最大为65535，但超过物理核数的部分是通过新一轮的下发来完成的。如果直接将GPU上的Triton算子拿到昇腾平台上运行，这些大量的任务会引入可观的核启动和核初始化时的额外开销，影响到算子性能表现。\
因此，需要针对昇腾平台特性修改分核逻辑。最推荐的做法是**将分核的数量直接固定为硬件的物理核数**，在核内做更为细致的数据分块：

* 对于纯Vector算子，分核数等于**Vector核数量**
* 对于CV融合算子，分核数等于**Cube核数量**（通常为Vector核数量的一半），算子执行时会按1：2的比例调用Vector核

可以通过以下接口获取**Vector核数**与**Cube核数量**：

```python
import torch_npu
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

```

参考示例代码，先固定核数，再通过内部循环分批处理任务分块：

```python
grid = (NUM_CORE ,)
_attn_fwd[grid](Q, K, V, M, Out, acc, scale......)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ......
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # 计算任务总量,将三维任务(Z,H,M)展平为一维总任务数
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # 每个核根据自己标识选取要处理的任务
    pid = tl.program_id(0)  # 当前核的唯一ID
    NUM_CORE = tl.num_programs(0)  # 获取固定启动的总核数
    # 循环规则：range(pid, NUM_BLOCKS, NUM_CORE) 实现"跨步分配任务"
    # - 起始值pid：每个核从自己的ID开始取任务，避免任务重叠
    # - 步长NUM_CORE：按总核数跨步，确保任务均匀分配到各个核
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 计算每次任务的数据偏移
        # 【核心：一维任务索引反向还原为原始多维索引】
        # block_idx是展平后的一维任务索引，通过整除/取余拆分回原始维度
        # 1.拆分Z+H合并轴 & M分块轴：
        #   - 整除NUM_BLOCKS_M：提取Z+H合并轴的索引（task_hz_idx）
        #   - 取余NUM_BLOCKS_M：提取M维度的分块索引（task_m_idx）
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2.拆分Z+H合并轴为原始Z轴和H轴：
        #   - 整除H：还原Z轴索引（off_z）
        #   - 取余H：还原H轴索引（off_h）
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3.计算数据偏移量：根据还原的Z/H索引，定位Q/K/V张量中对应的数据起始位置
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```


## 单核数据搬运

### 设置合适的循环内数据分块大小（BLOCK SIZE）
以add_kernel为例，变量和操作共同决定了片上内存空间的占用大，通过修改BLOCK_SIZE大小可以调整循环内数据分块和计算中间结果占用的大小。如果超过上限则算子编译时会提示预期占用大小并报错。要达到最大计算访存比，BLOCK_SIZE需要在不超出片上空间时尽可能大，这可以通过Triton-Ascend的[Autotune](#triton-autotune-自动调优)预先设置不同的BLOCK_SIZE，运行时会自动选取最优设置。

```python
@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # 元素总数量。
               BLOCK_SIZE: tl.constexpr,  # 分块元素数量。
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # 分块大小为 BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # 加载 x,y 数据到片上内存
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)
```

### 尽量保证Tensor的尾轴大小数据对齐

【描述】对于VV类算子需要调用Vector核计算时，昇腾硬件的UB要求Tensor的尾轴大小能被32Bytes整除，而对于CV类算子需要调用vector核和Cube核计算时，要求Tensor的尾轴大小能被512Bytes整除，若尾轴长度不足则会自动补齐。在此前提下，对模型中shape为(2048,3)和(2048,1)Tensor的种种操作，都会因为自动补齐导致性能明显恶化，此时可考虑通过转置操作将对齐轴转到低维，直到store时再转置为原始状态，从而规避自动补齐，优化计算速度。同时由于转置操作本身也受自动补齐规则的影响，因此同样需要特殊技巧来规避补齐。这里列出一个"借轴转置"的tip，适用于**tensor.numel() % 256Byte == 0**的场景，具体操作如下：

- 注：VV类算子表示该类算子在运算过程中只使用了Vector Core；CV类算子表示该类算子运算过程中既使用了AI Core又使用了Vector Core。
- 示例

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # 当成1D tensor load，此时由于numel对齐，不会自动补齐。
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，从而让两个轴都对齐
```

### 先将数据搬运到UB上，再从UB中select目标值

【描述】在NPU的离散场景下，可以先将数据搬运到UB，再从share中select目标值。

- 示例

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

-   val = tl.load(x_ptr + idx * stride_x, mask=mask)

+   rm = tl.arange(0, M)
+   x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
+   val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

- 优化前后性能分析和对比

通过msprof工具执行用例可得到PROF_*文件夹，里面包含了op_summary_\*.csv文件，该文件可以帮助分析流水情况。注：“\*”表示时间戳，[性能数据采集参考方法](./debug_guide/profiling.md)。

||Op Name|aiv_time(us)|
|:---- |:--------|:--------|
|未优化|pick_kernel|574.124|
|优化|pick_kernel|171.175|

通过分析表格中的数据可以发现，优化前后的aiv_mte2_time(us)和aiv_mte2_ratio差距较大，优化方案通过先将大部分数据搬运到UB上，减少小批量数据通过L2搬运到UB的次数，减少了L2搬运到UB上的总时间。

### 存算并行

Triton-Ascend支持两种数据处理模式：存算串行和存算并行。

存算串行：先从全局内存搬运数据到片上内存，完成计算后，再搬运下一批数据。这种方式存在明显的空闲等待时间，效率较低。

存算并行：在搬运第一批数据至片上内存的同时，已开始对其执行计算；随后继续搬运第二批数据，形成“搬运+计算”重叠的流水线式操作，显著提升整体吞吐率。

实现存算并行的关键在于合理设计数据切分（Tiling）策略，使得在当前批次数据计算过程中，能够提前准备下一阶段所需的数据，从而实现数据搬运与计算过程的并
行化。目前，编译器默认配置multiBuffer=True，默认支持存算并行。

### Tiling优化

AI Core进行计算的时候要先将数据搬运至片上内存，而片上内存的空间通常远小于AI Core要处理的总数据量，以Atlas 800T/I A2产品为例，片上内存容量为192KB，默认开启doublebuffer特性后容量还会减至原来的一半。因此算子计算时需要对数据进行分块操作，每次只加载处理其中的一小部分数据。

- 示例

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

### Triton Autotune 自动调优

在 Tiling 分块优化中，BLOCK_SIZE、BLOCK_SIZE_SUB等分块参数的取值直接影响算子性能，但手动调试参数组合效率低且难以找到最优值。triton.autotune是Triton框架提供的自动调优工具，能遍历预设的参数配置，通过实际运行对比性能，自动选择最优参数组合，是Tiling优化的核心配套手段。

- 核心作用
自动遍历参数空间：针对BLOCK_SIZE、BLOCK_SIZE_SUB等 constexpr 类型的分块参数，批量测试不同取值的性能。
性能基准对比：以算子的执行耗时为指标，筛选出适配当前硬件的最优参数。
缓存调优结果：调优后的最优配置会被缓存，后续调用算子时直接复用，避免重复调优。

- 简单示例
    ```diff
    @triton.autotune(
    configs=[ # 待测试的参数配置列表,参数候选值需要是2的幂次
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # 调优维度：参数取值依赖的输入维度
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
- 注：设置以下环境变量，便可打印出最优参数信息。
    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```

### 如何在NPU上避免UB OVERFLOW

【描述】在NPU上，UB或者L1 Size存在上限，当出现该错误时，需要减少单次搬运的数据量，以for循环的方式处理长序列场景。
```diff
E triton.compiler. errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```
【注意】A2系列产品UB大小为192KB(1572864 bits)。


## 单核数据运算
### 开发目标
在昇腾NPU单核上实现基础数据运算算子（如加减乘除、激活函数、简单矩阵元素运算）。保证算子在单核内高效执行，为后续多核并行和分布式扩展打下基础。


### 开发步骤
1.确定算子功能
-明确输入/输出张量的形状、数据类型（float16/float32/int32 等）。
-确认是否需要广播、边界处理。


2.编写核函数（kernel）
单核运算通常对应块级的数据处理。
单核数据运算示例：向量加法
```diff
import torch
import torch_npu

import triton
import triton.language as tl

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
调用：
 ```diff
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```
使用上述函数计算两个 torch.tensor 对象的 element-wise sum，并测试其正确性
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


3.单核运算的关键点

-块级数据处理：每个计算块负责一小段数据，保证并行性。

-边界检查：使用 mask 或 if (tid < N) 避免越界。

-块大小选择：合理设置 block 和 grid


4.性能要点：
(1)访存优化
-保证连续访问。
-使用对齐的 stride，避免跨行/跨列跳跃式访问。
-尽量让数据块大小对齐到 32 字节边界。
输入输出 buffer 在分配时保证对齐，避免访存性能下降。
例:
 ```diff
BLOCK_SIZE = 256  # 256 * 4 bytes = 1024 bytes，对齐良好

@triton.jit
def vec_add_kernel(X, Y, Z, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # 计算当前 block 负责的 index 范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask 防止越界
    mask = offsets < N

    # 连续访存：offsets 是连续的
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    z = x + y

    # 连续写回
    tl.store(Z + offsets, z, mask=mask)


def vec_add(x, y):
    assert x.numel() == y.numel()
    N = x.numel()

    # 分配对齐内存（PyTorch 默认已经对齐到 64 字节）
    z = torch.empty_like(x)

    # grid：每个 block 处理 BLOCK_SIZE 个元素
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z
```

(2)子块划分
-将大矩阵分解为小block，每个block在 UB 内完成计算。
-子块划分要兼顾访存连续性和计算单元利用率。
例：
 ```diff
BLOCK_M = 64   # 每个 block 处理 64 行
BLOCK_N = 64   # 每个 block 处理 64 列
BLOCK_K = 32   # 内部累加维度

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # block 在 M 方向的 id
    pid_n = tl.program_id(1)  # block 在 N 方向的 id

    # 当前 block 对应的起始坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 循环分块计算
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

    # 写回结果
    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```
