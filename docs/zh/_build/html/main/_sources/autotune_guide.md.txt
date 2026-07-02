# Triton-Ascend autotune 使用指南

## 文档定位

本文面向已经会写 Triton kernel、也了解社区版 `triton.autotune` 基本概念的用户，重点说明 Triton-Ascend 的推荐用法：

- Triton-Ascend 上的推荐 autotune 写法；
- `configs=[]` 在 Ascend 后端中的含义；
- 自动 Tiling 模式的适用边界，以及何时回到手写 `triton.Config`。

## 快速上手

在 Triton-Ascend 上，推荐保留社区版 `@triton.autotune` 的基本写法；当希望系统自动生成并评估候选配置时，将 `configs` 设为 `[]`：

```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(
    configs=[],
    key=["M", "N"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```

这表示：

- `key` 的语义与社区版保持一致，用于决定哪些输入变化会触发重新选择配置；
- `configs=[]` 在 Triton-Ascend 中表示“由 Ascend backend 自动生成候选配置并完成寻优”，而不是“没有可选配置”。

### 1. 先启用 Ascend 对 autotune 的扩展

只有导入下面这行后，才会进入 Triton-Ascend 的 autotune 扩展路径：

```python
import triton.backends.ascend.runtime
```

如果没有这一步，使用的仍然是社区版 `triton.autotune`，`configs=[]` 也不会触发 Ascend 的自动 Tiling 生成。

### 2. `@triton.autotune` 必须直接包在 `@triton.jit` 外层

必须写成下面这种顺序：

```python
@triton.autotune(configs=[], key=["M", "N"])
@triton.jit
def kernel(...):
    ...
```

`@triton.autotune` 必须直接包在 `@triton.jit` 外层，不能在两者之间插入其他 decorator。否则会导致无法对 kernel DSL 进行解析，从而无法进入 Triton-Ascend 的自动 Tiling 生成与寻优链路。

### 3. `key` 的含义与社区一致

`key` 的本质是 autotune 的 cache key。凡是填入 `key` 的参数，只要取值发生变化，就会触发重新 autotune。

大多数情况下，`key` 里放的是 `M/N/K`、`seq_len`、`hidden_size` 这类 shape 参数，因为它们往往会显著影响最优 Tiling；但 `key` 并不只限于 shape 参数，只要某个参数变化会影响配置选择，也可以放入 `key`。

### 4. 希望参与自动调优的参数不要被提前固定

如果希望某个 `tl.constexpr` 参与自动 Tiling 生成，需要同时满足下面三点：

- 它本身必须是 Tiling 参数，也就是会影响每个 block（逻辑核）处理的数据规模或 tile 大小的参数；
- 不要在 launch 时把它显式传值写死；
- 不要在 kernel 定义里给它设置默认值。

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

如果你在 launch 时显式传入：

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

同样，如果在 kernel 定义中给某个调优参数提供了默认值，例如：

```python
@triton.jit
def kernel(
    ...,
    BLOCK_M: tl.constexpr = 128,
):
    ...
```

那么这个参数也不会参与自动调优。对于希望交给框架自动生成和寻优的参数，应该把它保留为“未在 launch 时显式传值、且在 kernel 定义中也没有默认值”的 `tl.constexpr`。

### 5. 如果 Tiling 参数会影响 grid，grid 必须写成 lambda 形式

如果某个 Tiling 参数会影响 grid 大小，那么 grid 不能提前写成固定值或只依赖运行时参数的静态表达式，而必须写成依赖 meta 参数的 `lambda` 形式。这一点与社区 autotune 的要求一致。

```python
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
```

原因是 autotune 在评估不同候选配置时，`BLOCK_M` 这类参数的取值会变化；如果 grid 不随候选配置一起变化，就无法保证每个候选配置都以正确的发射方式执行。

## 使用注意事项

autotune 的 benchmark 语义与社区一致，会多次执行 kernel。如果 kernel 存在副作用，例如包含原子操作、inplace 写入，或会修改输入/输出 buffer 的累积状态，仍然需要通过社区已有的 hook 机制处理。

## Triton-Ascend 相比社区 autotune 的扩展

社区版 `triton.autotune` 的典型模式是：用户手工提供一组 `triton.Config`，框架做 benchmark，然后缓存最优结果。

Triton-Ascend 在保持这套接口习惯不变的前提下，主要扩展了下面几件事。

### 1. 支持 `configs=[]` 自动生成候选配置

这是最核心的扩展。用户不必先手写一组 `triton.Config`，而是可以把 `configs` 留空，让 Ascend backend 根据 kernel DSL 语义和运行时 shape 自动生成候选配置。

### 2. 支持多个 config 的并行编译

当 autotune 需要评估多个候选配置时，Triton-Ascend 默认会并行编译这些候选配置，以缩短首次调优时延。

这一能力默认开启，可通过环境变量 `TRITON_AUTOTUNE_PARALLEL_COMPILE=0` 关闭。

### 3. 支持使用 profiler 采集 kernel 性能

Triton-Ascend 支持在 autotune 做 benchmark 时切换性能采集方式：除了默认 benchmark 模式外，还可以使用 profiler 来采集每个候选 config 的 kernel 性能数据，它只关注 kernel 的片上计算时间，对于执行时间较短的 kernel 比默认性能采集方式更加精确，但会增加一些耗时。这项能力可通过环境变量 `TRITON_BENCH_METHOD='npu'` 开启。

## 自动 Tiling 生成能力的范围与行为

前面的第 1 点说明了 Triton-Ascend 支持 `configs=[]` 自动生成候选配置。对这项能力，进行以下几点说明。

### 1. 自动生成范围聚焦在 Tiling 参数

Ascend 自动生成的重点是 kernel 中与 Tiling 相关的 `tl.constexpr` 参数，也就是影响每个 block（逻辑核）处理的数据规模或 tile 大小的参数。

这套能力不等价于“自动帮你调所有参数”。像 `num_warps`、`num_stages` 这类编译参数，以及 kernel 的非 Tiling 参数，不属于当前自动生成范围。

### 2. 候选配置会带上 Ascend 硬件约束

Ascend backend 在生成候选时，会结合 NPU 的片上存储容量、对齐约束、核数利用等边界做筛选，而不是单纯枚举一批配置再盲目 benchmark。

### 3. 自动 Tiling 模式的目标是方便给出“性能不错”的配置

为了权衡寻优时间和寻优效果，当前自动 Tiling 会对生成配置数量做大量剪枝，因此并不保证自动生成结果一定能达到手工极致调优的性能上限。

这项能力的目标，是在尽量降低用户使用门槛和寻优成本的前提下，方便地给用户提供一个性能还不错的 Tiling 配置。

### 4. 自动 Tiling 生成失败时需要用户手写 `triton.Config`

如果自动 Tiling 模式无法生成任何可用候选配置，这时需要用户改为手写 `triton.Config`。同时也建议对这类场景提 issue，帮助 Triton-Ascend 后续补齐解析和自动生成能力。

## 手写 `triton.Config` 模式

如果自动 Tiling 模式生成失败，或生成的 Tiling 性能未达到预期，直接回到社区标准写法即可。Triton-Ascend 对这部分语义保持兼容：

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
- 框架负责 benchmark、选择最优配置和缓存复用；
- 使用习惯与社区 autotune 保持一致。

## 进阶用法：自动 Tiling 与其他参数联合调优

以下内容属于进阶用法，只有当用户希望在自动 Tiling 模式下，继续联合调优 kernel 的非 Tiling 参数或编译参数时，再考虑使用。

### 1. 社区 autotune：手工枚举全部调优参数

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

### 2. Triton-Ascend：手工枚举 Tiling 参数和 Ascend 编译参数

在 Triton-Ascend 中，如果希望继续走手工枚举模式，也可以把 Ascend 侧参数一起放进手工配置空间：

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

### 3. Triton-Ascend：自动生成 Tiling，同时联合调优其他参数

如果希望 Tiling 参数继续由 `configs=[]` 自动生成，但又希望同时调优其他非 Tiling 参数或编译参数，可以把这些额外搜索维度通过 `hints` 传入：

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


matmul_kernel[grid](a, b, M, N, K)
```

这种方式的含义是：

- Tiling 相关参数仍然由 Triton-Ascend 自动生成；
- 非 Tiling 参数或编译参数由用户通过 `hints` 显式给出候选集合；
- autotune 会对两部分组合后的配置空间做评估。

## 小结

Triton-Ascend 相比社区 autotune 的关键扩展，不是改变用户接口，而是在社区接口之上增加了“自动生成 Tiling 候选并完成寻优”的能力。对大多数用户来说，最推荐的使用方式就是：

- 保持社区版 `@triton.autotune` 的写法；
- 将 `configs` 设为 `[]`；
- 让 Ascend backend 基于 kernel DSL 和运行时 shape 自动完成候选生成、筛选、benchmark 与缓存复用。

如果场景不适合自动 Tiling 模式，再回到手写 `triton.Config` 即可。
