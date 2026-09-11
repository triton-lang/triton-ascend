# Triton-Ascend max_autotune 使用指南

## 文档定位

本文面向已经熟悉 Triton-Ascend 自动调优机制的用户，介绍 `max_autotune` 装饰器的高级用法：

- `max_autotune` 与标准 `@triton.autotune` 的区别与联系；
- 如何利用笛卡尔积展开机制批量生成候选配置；
- 不同 `kernel_type` 支持的调优参数及其适用场景；
- 何时选择 `max_autotune` 而非手写配置列表。

## 快速上手

`max_autotune` 是 `@triton.autotune` 的扩展版装饰器，它允许在自动调优前将每个基础配置与额外的调优参数进行笛卡尔积展开，从而大幅减少用户需要手工枚举的配置数量。

### 1. 基本用法

```python
import triton
import triton.language as tl
from triton.backends.ascend.runtime import max_autotune


@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 256}),
    ],
    key=["M", "N"],
    kernel_type="mix",
    # 额外的调优参数，每个值必须是列表
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4],
)
@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    **META
):
    ...
```

上述配置展开后的等效配置数量为：

- 基础配置：2 个
- 用户提供的调优参数：`enable_hivm_auto_cv_balance`（2 种）、`tile_mix_vector_loop`（2 种）
- 使用默认值的参数：`set_workspace_multibuffer`（2 种）、`tile_mix_cube_loop`（2 种）、其他参数各 1 种

总配置数量：`2 × 2 × 2 × 2 × 2 = 32` 个配置。

> 注：`kernel_type="mix"` 支持的参数较多，未显式提供的参数会使用默认值参与展开。如果希望某个参数不参与展开，可在基础 `Config` 的 `kwargs` 中固定其值。

### 2. `max_autotune` 与 `@triton.autotune` 的关系

`max_autotune` 本质上是先调用 `get_max_configs` 对基础配置进行展开，然后将展开后的配置列表传递给标准的 `@triton.autotune`。因此：

- 所有 `@triton.autotune` 支持的参数（`key`、`prune_configs_by`、`reset_to_zero` 等）在 `max_autotune` 中同样有效；
- `max_autotune` 额外增加了 `kernel_type` 参数和 `**tuning_params` 调优参数空间；
- 最终仍然是 `@triton.autotune` 完成 benchmark、选优和缓存。

### 3. 必须导入 Ascend backend 扩展

与使用 `configs=[]` 自动 Tiling 一样，`max_autotune` 需要 Ascend backend 扩展支持。`max_autotune` 需要从 Ascend backend 模块单独导入：

```python
from triton.backends.ascend.runtime import max_autotune
```

## Kernel 类型与支持的参数

`max_autotune` 通过 `kernel_type` 参数区分不同类型的算子，每种类型支持不同的调优参数集合。

### 参数支持矩阵

| 参数 | cube | mix | vector | 默认值 | 有效值 | 说明 |
|------|:----:|:---:|:------:|--------|--------|------|
| `num_stages` | ✅ | ✅ | ✅ | `[2]` | `[1, 2]` | pipeline stages 数量 |
| `unit_flag` | ✅ | ✅ | ❌ | `[False]` | 布尔列表 | Cube 搬出相关同步优化项 |
| `limit_auto_multi_buffer_of_local_buffer` | ✅ | ✅ | ❌ | `["no-l0c"]` | `["no-limit", "no-l0c"]` | 配置 local buffer 自动 multi-buffer 的 scope |
| `limit_auto_multi_buffer_only_for_local_buffer` | ❌ | ✅ | ❌ | `[False]` | 布尔列表 | 限制自动 multi-buffer 只作用于 local buffer |
| `set_workspace_multibuffer` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4]` | 配置 workspace multi-buffer 档位 |
| `enable_hivm_auto_cv_balance` | ❌ | ✅ | ❌ | `[True]` | 布尔列表 | 启用或禁用自动 CV balance |
| `tile_mix_vector_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | 配置 Vector loop 的切分份数 |
| `tile_mix_cube_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | 配置 Cube loop 的切分份数 |
| `enable_ubuf_saving` | ❌ | ✅ | ✅ | `[True]` | 布尔列表 | 是否启用 ubuf 节省 |

### Kernel 类型说明

- **cube**：纯 cube（矩阵乘法类）算子，支持最少的调优参数；
- **vector**：纯向量算子，仅支持 `num_stages` 和 `enable_ubuf_saving`；
- **mix**：混合 cube+vector 算子（默认类型），支持最完整的调优参数集合。

## 参数值优先级与展开逻辑

### 参数值优先级

调优参数的值按以下优先级确定：

1. **tuning_params 参数**（最高优先级）：通过 `**tuning_params` 传递的候选值列表；
2. **基础配置中的值**：如果基础 `Config` 的 `kwargs` 中已存在该参数，则固定为该值（转换为单元素列表）；
3. **默认值**（最低优先级）：从内部默认值表获取。

### 展开示例

假设有以下配置：

```python
@max_autotune(
    configs=[triton.Config(kwargs={'BLOCK_M': 128}, num_stages=2)],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
)
@triton.jit
def kernel(...):
    ...
```

展开过程：

1. `kernel_type="vector"` 支持的参数为 `num_stages` 和 `enable_ubuf_saving`；
2. `num_stages` 在 `tuning_params` 中提供 `[1, 2]`，优先级最高；
3. `enable_ubuf_saving` 未提供，使用默认值 `[True]`；
4. 笛卡尔积展开后得到 2 个配置。

展开结果等效于：

```python
configs=[
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=1),
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=2),
]
```

### 在基础配置中固定参数

如果希望某个调优参数不参与展开，可以在基础配置中直接固定：

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': False}),
    ],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
    # enable_ubuf_saving 已在基础配置中固定为 False，不会使用默认值 [True]
)
@triton.jit
def kernel(...):
    ...
```

## 使用注意事项

### 1. 不支持的参数会被忽略

如果通过 `tuning_params` 传递了当前 `kernel_type` 不支持的参数，会产生警告并忽略该参数：

```python
# 警告：tile_mix_vector_loop 不支持 kernel_type="vector"
@max_autotune(
    configs=[...],
    key=["M"],
    kernel_type="vector",
    tile_mix_vector_loop=[2, 4],  # 将被忽略并产生警告
)
@triton.jit
def kernel(...):
    ...
```

### 2. 参数值必须是列表

`tuning_params` 中的每个值必须是列表或元组，且不能为空：

```python
# 正确写法
enable_hivm_auto_cv_balance=[True, False]

# 错误写法：不是列表
enable_hivm_auto_cv_balance=True  # 将导致验证错误
```

### 3. 配置数量增长

展开后的配置数量等于：
基础配置数量 × Π(每个 tuning_param 的列表长度)

例如，2 个基础配置 × 3 个参数（列表长度分别为 2、3、2）= 12 个展开配置。

过多的配置会增加首次调优时间，建议根据实际需要合理控制参数空间。

### 4. 与 `configs=[]` 的区别

`max_autotune` 与 `@triton.autotune(configs=[], ...)` 是两种不同的自动调优策略：

| 特性 | `max_autotune` | `@triton.autotune(configs=[])` |
|------|----------------|--------------------------------|
| Tiling 参数生成 | 需要用户在基础配置中指定 | Ascend backend 自动生成 |
| 编译参数调优 | 支持通过 `tuning_params` 展开 | 通过 `hints` 参数传入 |
| 适用场景 | 明确知道 Tiling 参数空间，需要调优编译参数 | 希望 Tiling 参数也自动生成 |

## 进阶用法

### 1. 结合多个调优参数

对于混合类型算子（`kernel_type="mix"`），可以同时调优多个参数：

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}),
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
    ],
    key=["M", "N", "K"],
    kernel_type="mix",
    num_stages=[1, 2],
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4, 8],
    tile_mix_cube_loop=[2, 4],
)
@triton.jit
def mixed_kernel(...):
    ...
```

展开后配置数量计算：

- 基础配置：2 个
- 用户提供的参数：`num_stages`（2）、`enable_hivm_auto_cv_balance`（2）、`tile_mix_vector_loop`（3）、`tile_mix_cube_loop`（2）
- 使用默认值的参数：`set_workspace_multibuffer`（默认 `[2, 4]` → 2）、其他参数各 1

总配置数量：`2 × 2 × 2 × 2 × 3 × 2 = 96` 个配置。

### 2. 针对 cube 类型算子的调优

cube 类型算子（如纯矩阵乘法）支持的参数较少：

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=["M", "N", "K"],
    kernel_type="cube",
    num_stages=[1, 2],
    unit_flag=[True, False],
    limit_auto_multi_buffer_of_local_buffer=["no-limit", "no-l0c"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

### 3. 针对向量算子的调优

向量算子支持最少的调优参数：

```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}),
    ],
    key=["N"],
    kernel_type="vector",
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def vector_kernel(...):
    ...
```

## 小结

`max_autotune` 是 Triton-Ascend 提供的高级自动调优工具，适合以下场景：

1. 已知 Tiling 参数空间，希望减少手工枚举配置的工作量；
2. 需要联合调优多个 Ascend 编译参数（如 `num_stages`、`enable_hivm_auto_cv_balance` 等）；
3. 希望通过笛卡尔积方式批量生成候选配置。

`max_autotune` 的核心价值在于：用少量基础配置 + 调优参数空间描述，自动展开出完整的候选配置集合，兼顾了灵活性与便利性。
