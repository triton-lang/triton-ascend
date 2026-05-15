# GPU Triton算子迁移

概述：本文介绍 GPU Triton 算子迁移到昇腾 NPU 时的通用处理思路和常见问题。迁移时建议先完成 Python 侧设备与运行时接口替换，再检查 grid 分核、访存对齐、单核计算、UB 空间和 coreDim 限制，最后结合具体示例完成代码修改和正确性验证。

## 通用迁移流程

### 迁移 Python 侧设备和运行时接口

在修改具体 Triton kernel 前，先完成 Python 侧设备迁移：

1. 在 Python 文件中增加 `import torch_npu`。
2. 查找 `device="cuda"`、`device='cuda'`、`.cuda()` 和 `.to("cuda")` 等设备指定方式，改为 `device="npu"`、`device='npu'`、`.npu()` 或 `.to("npu")`。
3. 查找 `torch.cuda.*`、CUDA stream、CUDA event、CUDA synchronize 等 GPU 专属接口，改为 NPU 对应接口或删除不必要的同步逻辑。
4. 删除只为 GPU 设备发现服务的逻辑，例如 `triton.runtime.driver.active.get_active_torch_device()` 相关设备断言。
5. 保持 Triton kernel 主体逻辑不变，先用 NPU Tensor 完成编译和正确性验证。

### 调整 grid 分核

GPU 上常见的写法会把 grid 设计为大量逻辑 program，由硬件和运行时调度到 SM 上执行。迁移到 NPU 时，应优先考虑物理 AI Core 数量和算子类型：

- grid 优先使用 1D；2D NPU 适配写法也会合并为 1D，例如 `(20,)` 与 `(4, 5)` 的效果相同。
- Vector-only 算子的并发任务数通常按 Vector Core 数量组织；包含 `tl.dot` 的算子通常按 AI Core 数量组织。
- 当逻辑 grid 远大于物理核数时，需要评估是否改成每个 program 内部循环处理多个 tile，或在逻辑核之间无顺序依赖时使用 `TRITON_ALL_BLOCKS_PARALLEL`。
- coreDim 不能超过 `UINT16_MAX`（65535），大 shape 算子需要结合 BLOCK_SIZE 或分块方式控制 grid 大小。

| 维度 | 核心结构 | 算子类型 |
|------|----------|----------|
| 昇腾 NPU (Ascend) | 多个 AI Core，分为 Cube Core（矩阵乘）和 Vector Core（向量计算） | Vector-only 算子 → 并发任务数 = Vector Core 数；含 `tl.dot` 算子 → 并发任务数 = AI Core 数 |
| GPU NVIDIA/AMD | 多个 CUDA Core（标量/向量计算） + Tensor Core（矩阵乘） | GPU 算子一般由编译器和硬件自动决定并发度 |

### 检查单核数据搬运

完成设备替换后，需要继续检查单个 program 内的数据搬运方式：

- Vector 算子场景下要求 32 字节访存对齐，cube-vector 融合算子场景下要求 512 字节对齐。
- 保留 tail mask，确认边界元素不会越界访问。
- 检查一次 tile 的片上内存占用，避免触发 UB 空间溢出。
- 移除或替换 GPU 专属同步 API，例如 CUDA thread、stream、event 或 kernel synchronize 相关接口。

### 检查单核数据运算

NPU 与 GPU 的计算单元和支持的数据类型存在差异。迁移后应先保证正确性，再根据性能问题调整：

- 对整数索引、offset、长度等中间值，优先确认当前数据类型是否被 NPU 路径高效支持。
- 对含 `tl.dot` 的算子，确认 M/N/K tile、累加 dtype 和输出 dtype 是否符合 NPU 后端要求。
- 对长序列、长 hidden size 或大 K 维循环，优先通过 tiling 控制单次搬入和计算规模。

## 迁移示例

### 示例 1：向量加法完整迁移

```diff
import torch
+ import torch_npu  # 【新增】导入昇腾NPU PyTorch适配库，提供NPU设备支持
import triton
import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()  # 【删除】GPU设备自动获取，NPU无需此逻辑

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
-   assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE  # 【删除】GPU设备一致性校验，NPU无需显式断言
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
- x = torch.rand(size, device='cuda')  # 【删除】GPU设备指定
+ x = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
- y = torch.rand(size, device='cuda')  # 【删除】GPU设备指定
+ y = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
f'{torch.max(torch.abs(output_torch - output_triton))}')
```

### 示例 2：设备替换与单核数据搬运

以下示例展示将设备从 CUDA 替换为 NPU 后，对单核数据搬运场景进行正确性验证：

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

## 常见问题概览

完成迁移基础步骤后，可能会遇到新的问题，新问题可归纳为以下两类：  
1.coreDim限制问题  
当网格维度超过NPU硬件限制时触发。  
典型错误信息：coreDim=xxxx can't be greater than UINT16_MAX  
2.UB空间溢出  
内存使用超出NPU缓存容量。  
典型错误信息：ub overflow, requires xxxx bits while 1572684 bits available!  

### 解决 coreDim 超限问题

问题分析:  
NPU的 coreDim 参数不能超过 UINT16_MAX（65535）。当处理大规模数据时，简单的grid划分可能导致该限制被突破。  

案例：zeros_like 函数优化    
数据规模：N = 1073741824，原始 BLOCK_SIZE = 2048，计算得到的 coreDim = 524288 > 65535（超限）  

解决思路1：  
昇腾编译器针对coreDim超限问题，有对应的解决方案，只需将环境变量'TRITON_ALL_BLOCKS_PARALLEL'设为1。设置命令如下：  
export TRITON_ALL_BLOCKS_PARALLEL=1  
解决思路2：  
通过增大 BLOCK_SIZE 来减少所需的核心数量，确保 coreDim 不超过限制。  
计算公式： coreDim = ceil(N / BLOCK_SIZE) → 需满足：ceil(N / BLOCK_SIZE) <= 65535 => BLOCK_SIZE >= ceil(N / 65535) 代入 N = 1073741824 得： BLOCK_SIZE >= triton.next_power_of_2(triton.cdiv(1073741824, 65535)) = 32768 -> 至少为 32768更稳妥

优化前的代码：

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

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # 原始值过小
    return out
```

优化后的代码：

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
    BLOCK_SIZE = max(32768, min_block_size) # 至少为 32768 
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),) 

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE) 
    return out
```

### 动态计算适合的 BLOCK_SIZE 以避免 coreDim 超限

```diff
optimal_block_size = 32768  # 根据计算得出的优化值

grid_fn = lambda meta: (triton.cdiv(N, optimal_block_size),)

zeros_kernel[grid_fn](out, N, BLOCK_SIZE=optimal_block_size)
return out
```

### 处理复合问题：coreDim + UB 溢出

问题分析:  
在某些情况下，解决了 coreDim 问题后可能引发新的UB溢出问题。这通常发生在增大 BLOCK_SIZE 后，单个线程块需要处理的数据量超出了NPU的UB缓存容量。  

案例：    
数据规模：N = 1073741824，原始 BLOCK_SIZE = 4096，计算得到的 coreDim = 262144 > 65535（超限），调整为 BLOCK_SIZE = 32768 后，coreDim = 32768（合规），但出现 UB 溢出  

解决思路：  
引入 BLOCK_SIZE_SUB 参数，将大块进一步细分，在保持合理 coreDim 的同时控制内存使用。  
优化前代码： 

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
    # ... 参数验证代码 ...
    # inp.device = "npu"
    out = torch.zeros_like(inp)
    N = inp.numel()
    if N == 0:
        return out

    grid = lambda meta: (triton.cdiv(N, 4096),)  # 导致 coreDim 超限
    masked_fill_kernel[grid](inp, mask.to(torch.int), value, out, N, 4096)
    return out
```

优化后代码：

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
    # 计算需要处理的子块数量
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    # 分块处理，避免 UB 溢出
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # 分批加载和处理数据
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        # 先写入原始数据
        tl.store(out + offsets, input_vals, mask=mask)
        # 然后在需要填充的位置覆写目标值
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

def masked_fill(inp, expand_mask, value):
    logger.debug("GEMS MASKED FILL")

    out = torch.zeros_like(inp)
    # ... 参数验证代码 ...
    # inp.device = "npu"
    N = inp.numel()
    if N == 0:
        return out

    # 使用优化的参数配置
    MAIN_BLOCK_SIZE = 32768  # 确保 coreDim 合规
    SUB_BLOCK_SIZE = 1024    # 控制 UB 使用量

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N, 
                        MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

### 为什么会出现UBSIZE超出内存的错误

切分不合理,存在过多的非对齐访存或者运算，例如对（64，32）二维数据搬运，对应stride(12832，128),如果是对齐数据的访存，对应的stride(32,1)。 对于非对齐访问内容，在最内轴新增一个大小为1的轴，变为（64，32，4） 由于硬件要求vector算子场景ub内存32bytes对齐 ，假设type=float16，对应stride应该为(12832, 128,1)

### 离散访存代码逐行对比观察scalar低效映射

设置环境变量TRITON_DEBUG=1, 保存~/.triton/cache/xxx.ttadapter，然后执行  

```diff
bishengir-compile xxx.ttadapter --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync  
```

会有输出IR ， 对比Triton 算子逻辑与IR内部的操作，观察是否有未映射成指令的操作。  
观察HIVM IR阶段是否存在纯scalar搬运或者计算， 没有映射为simd指令，这会成为性能瓶颈。    

问题：离散访存 && scalar低效映射  
b[1024, 32] = a[1024, 32]  Triton原先写法利用thread的方式 对[1024,32] 中的最低维度32绑定线程块, 再对1024切16，分为[64， 16， 32]，再对64绑定线程块  

```diff
chunk_fwd_kernel_o[(NT, B * H)](
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(1024,), # 一维张量
        strides=(32,), # 连续内存
        offsets=(i_t * 16,), # 从起始位置开始
        block_shape=(BT,), # 块大小
        order=(0,) # 连续访问
    )
​)
```

优化思路

调整 block_ptr 的 shape/stride:  
把 (1024, 32) 看成二维矩阵，最低维度 32 是连续的，所以 stride 应该是 (32, 1)，而不是 (32,)，这样每个线程块能访问连续的 32 元素。让线程块绑定到行维度（1024），每个线程处理一整行的 32 元素。这样访存就是连续的，亲和性好。  

比如： 

```diff 
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0) # 行优先布局：维度 1 最连续（stride 1），维度 0 最不连续
)
```
