# 昇腾与GPU的开发差异

## 多核任务并行策略

NPU在Triton多核并行中是物理核强绑定模式，与GPU逻辑维度并行+硬件自动物理映射的模式形成核心差异

- 核心对比

    |维度       |GPU（NVIDIA） |昇腾（Ascend）|
    |-----------|--------------|-----------|
    |grid 本质| 逻辑任务维度（和物理核解耦）| 物理核组映射（绑定 AI Core 拓扑）|
    |核数 / 维度限制| grid 维度 / 大小无硬限制| grid 大小≤AI Core 总数，2D 需匹配拓扑|

GPU:可绑定多个维度轴（三维grid=[n,m,l] 等同于乘积n×m×l个并行线程），每个线程仅对应一次kernel执行，且仅执行一次。\
NPU:Vector核，Cube核属于多个物理核，不同代际硬件核数不同，每个核仅执行一次Block,且支持对该Block重复调度执行。

### 充分利用核数

昇腾NPU具备多个计算核心，合理分配并充分利用所有可用核心，是提升算子性能的关键因素之一。
在调用Triton内核函数时，通过设置launch参数控制使用的核数量。以GELU算子为例：

```Python
triton_gelu[n, 1, 1](...)  # 第一个参数表示使用的核数，n表示使用n个核
```

通过对核数的调优，可实现对所有计算资源的充分调度和利用，从而最大化并行度与吞吐量。注意，当前版本核数需小于等于65535。

## 单核数据搬运策略

### 数据切分Tiling

写Triton内核函数时，合理的数据切分策略对性能优化至关重要。通过调整不同的切分粒度参数，可以在不同维度上平衡计算负载与内存访问效率。

常见的切分参数包括：

```text
ncore：使用的核数（跨核切分）
xblock：核间数据块大小（核间切分）
xblock_sub：核内切分粒度（核内细粒度划分）
```

开发者可根据实际场景手动选择最优的切分配置，使得每次计算尽可能充分利用片上内存（On-chip Memory），避免频繁访问全局内存（Global Memory）造成的
性能瓶颈。

以GELU算子为例，通过调整切分参数，可以有效适配片上缓存容量限制，从而提升执行效率。

注：Atlas 800T/I A2产品的片上内存容量为192KB，因此在设计切分策略时需考虑该限制，确保每轮计算的数据量不超过片上内存容量。

#### GELU算子示例

GELU算子开发示例，使用3种方式计算结果。

standard_unary      为标准Torch计算。

triton_easy_kernel  为简单Triton实现。

triton_better_kernel为更高效的Triton实现。

#### 标准Torch写法

输入tensor x0，经过torch计算实现 GELU 算子，返回结果值。

```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```

#### 简单Triton写法

以下是一个使用 Triton 编写的简单内核示例，用于展示如何定义和调用一个基本的Triton内核函数。此示例实现了一个简单的数学运算（GELU 激活函数）。

```Python
# 定义triton_kernel核函数
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```

注意事项

1. 内存限制：上述写法中，所有输入数据一次性被加载到内存中进行计算。如果输入张量过大，可能会超出单个内核的片上内存容量，导致内存溢出错误。
因此，这种简单的写法更适合于小规模张量的计算或用于理解 Triton 内核的基本写法和调用方式。

2. 适用场景：尽管这种方法有助于快速理解和入门 Triton 编程，但对于大规模数据集或高性能要求的应用场景，建议采用更复杂的数据切分策略（如 Tiling），
以充分利用硬件资源并避免内存溢出问题。通过这种方式，开发者可以快速上手 Triton 编程，同时了解如何定义、调用以及优化 Triton 内核函数。

#### 更高效triton写法

在昇腾 NPU 上使用 Triton 编写高性能算子时，为了充分利用硬件资源、避免内存溢出并提升执行效率，通常需要采用数据切分（Tiling）策略。
下面是一个经过优化的 Triton 内核实现示例，适用于大规模张量计算。

```Python
# 定义triton_kernel核函数
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# 调用triton_kernel核函数
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

关键代码解释

```Python
# 计算当前核处理数据块的起始偏移地址，实现核间切分。每个核仅负责 XBLOCK 大小的数据范围。
xoffset = tl.program_id(0) * XBLOCK

# 在单个核内部进一步细分数据块，每次处理 XBLOCK_SUB 大小的数据，实现核内切分。
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

# 构造当前迭代的数据索引数组，用于访问输入和输出张量。
x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

# 设置掩码以防止越界访问，确保只处理合法范围内的数据。
xmask = x_index < xnumel

# 分别用于从全局内存加载数据到片上内存，以及将计算结果写回全局内存。
tl.load() 和 tl.store()
```

## 编译优化能力

### AscendNPU IR优化

针对昇腾软硬件特性，适配了AscendNPU IR优化的编译选项，如下表所示。
**使用方法**：在autotune的配置阶段，传入编译选项的值
以开启`multibuffer`选项举例，在autotune的配置阶段，即`triton.Config`中，传入`'multibuffer': True`，详见[autotune示例](../examples/06_autotune_example.md)：

```python
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),]
```

| 选项      | 能力       | 是否开启 |
| ----------------- | ------------ | ----------------- |
| multibuffer                                   | 开启流水并行数据搬运  | 默认true； true , false。 autotune中可配置                     |
| unit_flag                                     | cube搬出的一个优化项                                         | 默认None；true , false。  autotune中可配置                     |
| limit_auto_multi_buffer_only_for_local_buffer | CV算子一个优化项，cube搬出的一个优化项                         | 默认None；true , false。 autotune中可配置 |
| limit_auto_multi_buffer_of_local_buffer       | cube算子开启double buffer具体的scope                         | 默认None；["no-limit", "no-l0c"]， autotune中可配置           |
| set_workspace_multibuffer                     | 只有在limit_auto_multi_buffer_only_for_local_buffer=false场景下生效 | 默认None；如[2,4]，autotune中可配置                            |
| enable_hivm_auto_cv_balance                   | set_workspace_multibuffer只有在limit_auto_multi_buffer_only_for_local_buffer=false场景下生效 | 默认None；true , false。 autotune中可配置 |
| tile_mix_vector_loop                          | CV算子的一个优化项,当前vector可以切几份                        | 默认None；如 [2,4,8]，autotune中可配置                       |
| tile_mix_cube_loop                            | CV算子一个优化项,当前cube可以切几份      | 默认None；如 [2,4,8]，autotune中可配置                      |
| auto_blockify_size                            | TRITON_ALL_BLOCKS_PARALLEL优化项，用于指定扩展的左起第一个维度的大小。 | 默认1；如 [2,4,8]，autotune中可配置                       |

- 注：优化编译选项在ascend/backend/compiler.py代码中。
- 注：CV算子表示该算子运算过程中既使用了AI Core又使用了Vector Core。
