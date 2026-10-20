# Triton-Ascend autotune使用指南

## 文档定位

本文面向已经会写Triton kernel、也了解社区版 `triton.autotune` 基本概念的用户，重点说明Triton-Ascend的推荐用法：

- Triton-Ascend上的推荐autotune写法；
- `configs=[]` 在Ascend后端中的含义；
- 自动Tiling模式的适用边界，以及何时回到手写 `triton.Config`。

## 快速上手

在Triton-Ascend上，推荐保留社区版 `@triton.autotune` 的基本写法；当希望系统自动生成并评估候选配置时，将 `configs` 设为 `[]`：

```python
"""
Vector Add
=============
"""


import torch
import torch_npu
import triton
import triton.language as tl
import triton.backends.ascend.runtime
from triton.backends.ascend.testing import do_bench_npu


@triton.autotune(configs=[], key=["n_elements"])
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )](x, y, output, n_elements)
    return output


def test_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)
    print(f"Vector Add {size} PASSED!")
    triton_us = do_bench_npu(lambda: add_autotune(x, y))
    torch_us = do_bench_npu(lambda: add_torch(x, y))
    print(f"Triton kernel execution time: {triton_us:.4f} ms")
    print(f"PyTorch kernel execution time: {torch_us:.4f} ms")


if __name__ == "__main__":
    test_add(98432)
```

这表示：

- `key` 的语义与社区版保持一致，用于决定哪些输入变化会触发重新选择配置；
- `configs=[]` 在Triton-Ascend中表示“由Ascend backend自动生成候选配置并完成寻优”，而不是“没有可选配置”。

### 1. 先启用Ascend对autotune的扩展

只有导入下面这行后，才会进入Triton-Ascend的autotune扩展路径：

```python
import triton.backends.ascend.runtime
```

如果没有这一步，使用的仍然是社区版 `triton.autotune`，`configs=[]` 也不会触发Ascend的自动Tiling生成。

### 2. `@triton.autotune` 必须直接包在 `@triton.jit` 外层

必须写成下面这种顺序：

```python
@triton.autotune(configs=[], key=["M", "N"])
@triton.jit
def kernel(...):
    ...
```

`@triton.autotune` 必须直接包在 `@triton.jit` 外层，不能在两者之间插入其他decorator。否则会导致无法对kernel DSL进行解析，从而无法进入Triton-Ascend的自动Tiling生成与寻优链路。

### 3. `key` 的含义与社区一致

`key` 的本质是autotune的cache key。凡是填入 `key` 的参数，只要取值发生变化，就会触发重新autotune。

大多数情况下，`key` 里放的是 `M/N/K`、`seq_len`、`hidden_size` 这类shape参数，因为它们往往会显著影响最优Tiling；但 `key` 并不只限于shape参数，只要某个参数变化会影响配置选择，也可以放入 `key`。

### 4. 希望参与自动调优的参数不要被提前固定

如果希望某个 `tl.constexpr` 参与自动Tiling生成，需要同时满足下面三点：

- 它本身必须是Tiling参数，也就是会影响每个block（逻辑核）处理的数据规模或tile大小的参数；
- 不要在launch时把它显式传值硬编码；
- 不要在kernel定义里给它设置默认值。

例如下面这种写法，`BLOCK_M` 会参与自动调优：

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
)
```

如果你在launch时显式传入：

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
    BLOCK_M=128,
)
```

那这个参数就已经被固定，不再属于自动生成范围。

同样，如果在kernel定义中给某个调优参数提供了默认值，例如：

```python
@triton.jit
def kernel(
    ...,
    BLOCK_M: tl.constexpr = 128,
):
    ...
```

那么这个参数也不会参与自动调优。对于希望交给框架自动生成和寻优的参数，应该把它保留为“未在launch时显式传值、且在kernel定义中也没有默认值”的 `tl.constexpr`。

### 5. 如果Tiling参数会影响grid，grid必须写成lambda形式

如果某个Tiling参数会影响grid大小，那么grid不能提前写成固定值或只依赖运行时参数的静态表达式，而必须写成依赖meta参数的 `lambda` 形式。这一点与社区autotune的要求一致。

```python
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
```

原因是autotune在评估不同候选配置时，`BLOCK_M` 这类参数的取值会变化；如果grid不随候选配置一起变化，就无法保证每个候选配置都以正确的发射方式执行。

## 使用注意事项

autotune的benchmark语义与社区一致，会多次执行kernel。如果kernel存在副作用，例如包含原子操作、inplace写入，或会修改输入/输出buffer的累积状态，仍然需要通过社区已有的hook机制处理。

## Triton-Ascend相比社区autotune的扩展

社区版 `triton.autotune` 的典型模式是：用户手工提供一组 `triton.Config`，框架做benchmark，然后缓存最优结果。

Triton-Ascend在保持这套接口习惯不变的前提下，主要扩展了下面几件事。

### 1. 支持 `configs=[]` 自动生成候选配置

这是最核心的扩展。用户不必先手写一组 `triton.Config`，而是可以把 `configs` 留空，让Ascend backend根据kernel DSL语义和运行时shape自动生成候选配置。

### 2. 支持多个config的并行编译

当autotune需要评估多个候选配置时，Triton-Ascend默认会并行编译这些候选配置，以缩短首次调优时延。

这一能力默认开启，可通过环境变量 `TRITON_AUTOTUNE_PARALLEL_COMPILE=0` 关闭。

### 3. 支持使用profiler采集kernel性能

Triton-Ascend支持在autotune做benchmark时切换性能采集方式：除了默认benchmark模式外，还可以使用profiler来采集每个候选config的kernel性能数据，它只关注kernel的片上计算时间，对于执行时间较短的kernel比默认性能采集方式更加精确，但会增加一些耗时。这项能力可通过环境变量 `TRITON_BENCH_METHOD='npu'` 开启。

## 自动Tiling生成能力的范围与行为

前面的第1点说明了Triton-Ascend支持 `configs=[]` 自动生成候选配置。对这项能力，进行以下几点说明。

### 1. 自动生成范围聚焦在Tiling参数

Ascend自动生成的重点是kernel中与Tiling相关的 `tl.constexpr` 参数，也就是影响每个block（逻辑核）处理的数据规模或tile大小的参数。

这套能力不等价于“自动帮你调所有参数”。像 `num_warps`、`num_stages` 这类编译参数，以及kernel的非Tiling参数，不属于当前自动生成范围。

### 2. 候选配置会带上Ascend硬件约束

Ascend backend在生成候选时，会结合NPU的片上存储容量、对齐约束、核数利用等边界做筛选，而不是单纯枚举一批配置再盲目benchmark。

### 3. 自动Tiling模式的目标是方便给出“性能不错”的配置

为了权衡寻优时间和寻优效果，当前自动Tiling会对生成配置数量做大量剪枝，因此并不保证自动生成结果一定能达到手工极致调优的性能上限。

这项能力的目标，是在尽量降低用户使用门槛和寻优成本的前提下，方便地给用户提供一个性能还不错的Tiling配置。

### 4. 自动Tiling生成失败时需要用户手写 `triton.Config`

如果自动Tiling模式无法生成任何可用候选配置，这时需要用户改为手写 `triton.Config`。同时也建议对这类场景提issue，帮助Triton-Ascend后续补齐解析和自动生成能力。

## 手写 `triton.Config` 模式

如果自动Tiling模式生成失败，或生成的Tiling性能未达到预期，直接回到社区标准写法即可。Triton-Ascend对这部分语义保持兼容：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def kernel(...):
    ...
```

这一模式下：

- 配置由用户手工提供；
- 框架负责benchmark、选择最优配置和缓存复用；
- 使用习惯与社区autotune保持一致。

### 查看调优日志

设置 `TRITON_PRINT_AUTOTUNING=1`，可查看调优日志。实际完成测速后，会打印调优耗时、最佳配置和各个成功配置的测速结果。以下为示意输出（`...` 表示省略的配置参数）：

```text
Triton autotuning for function kernel finished after 0.12s; best config selected: BLOCK_M: 128, BLOCK_N: 128, ...;
Triton autotuning benchmark results for function kernel:
  config=BLOCK_M: 128, BLOCK_N: 128, ...; p50=0.0100 ms, p20=0.0090 ms, p80=0.0110 ms [selected]
  config=BLOCK_M: 64, BLOCK_N: 256, ...; p50=0.0140 ms, p20=0.0130 ms, p80=0.0150 ms
```

`[selected]` 标记被选中的最佳配置。`p50`、`p20`、`p80` 分别为第50、20、80百分位耗时，单位为毫秒；若测速返回平均耗时，则显示 `mean=... ms`。

用户配置（`configs`，包括通过 `hints` 展开的配置）发生失败，或调优因异常中断时，会打印相应配置的状态和原因。以下分别为编译失败和调优中断的示意输出：

```text
Triton autotuning: config=BLOCK_M: 128, BLOCK_N: 128, ...; compile_failed; reason=CompileTimeAssertionFailure: <错误信息>
Triton autotuning: config=BLOCK_M: 64, BLOCK_N: 256, ...; not_evaluated; reason=No benchmark result was produced. Autotuning interrupted: RuntimeError: <错误信息>
```

其中，`compile_failed` 表示编译失败，`not_evaluated` 表示配置尚未完成评估，`reason` 给出失败或中断原因。

## 进阶用法：自动Tiling与其他参数联合调优

以下内容属于进阶用法，只有当用户希望在自动Tiling模式下，继续联合调优kernel的非Tiling参数或编译参数时，再考虑使用。

### 1. 社区autotune：手工枚举全部调优参数

社区标准写法是由用户手工枚举所有候选配置：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
            },
            num_warps=num_warps,
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 2. Triton-Ascend：手工枚举Tiling参数和Ascend编译参数

在Triton-Ascend中，如果希望继续走手工枚举模式，也可以把Ascend侧参数一起放进手工配置空间：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
                "multibuffer": MS,
            }
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for MS in [False, True]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 3. Triton-Ascend：自动生成Tiling，同时联合调优其他参数

如果希望Tiling参数继续由 `configs=[]` 自动生成，但又希望同时调优其他非Tiling参数或编译参数，可以把这些额外搜索维度通过 `hints` 传入：

```python
@triton.autotune(
    configs=[],
    key=["M", "N", "K"],
    hints={
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "multibuffer": [False, True],
    },
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...


grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * meta["GROUP_SIZE_M"], triton.cdiv(N, meta["BLOCK_N"]))
matmul_kernel[grid](a, b, M, N, K)
```

这种方式的含义是：

- Tiling相关参数仍然由Triton-Ascend自动生成；
- 非Tiling参数或编译参数（如 `GROUP_SIZE_M` 和 `multibuffer`）由用户通过 `hints` 显式给出候选集合；
- autotune会对两部分组合后的配置空间做评估。

## autotune示例

| autotune算子          | 示例 |
|---------------------|----|
| 向量加法autotune       |[01-vector-add.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/autotune_ut/01-vector-add.py)|
| 融合Softmax autotune |[02-fused-softmax.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/autotune_ut/02-fused-softmax.py)|
| Layer Norm autotune |[03-layer-norm.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/autotune_ut/03-layer-norm.py)|

## 小结

Triton-Ascend相比社区autotune的关键扩展，不是改变用户接口，而是在社区接口之上增加了“自动生成Tiling候选并完成寻优”的能力。对大多数用户来说，最推荐的使用方式就是：

- 保持社区版 `@triton.autotune` 的写法；
- 将 `configs` 设为 `[]`；
- 让Ascend backend基于kernel DSL和运行时shape自动完成候选生成、筛选、benchmark与缓存复用。

如果场景不适合自动Tiling模式，再回到手写 `triton.Config` 即可。
