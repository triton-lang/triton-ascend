# al.sub_vec_num

## 1. 概述

`al.sub_vec_num` 在 CPU/JIT 编译阶段查询当前活动 NPU，返回每个 AI Core 对应的 Vector Core
数量，用于确定 Sub Vector 切分份数。它通常与 [`al.sub_vec_id`](./sub_vec_id.md) 配合使用。

当前实现查询活动 NPU 的 AIV Core 数量和 AIC Core 数量，并计算：

```text
AIV Core 数量 // AIC Core 数量
```

## 2. 接口说明

```python
al.sub_vec_num() -> tl.constexpr
```

### 返回值

返回 `tl.constexpr` 编译期常量。该值不是 NPU 运行时 ID，也不会生成查询 Sub Vector 数量的设备指令。

### 参数

无。

## 3. 使用方法

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def matmul_then_exp(
    a_ptr,
    b_ptr,
    out_ptr,
    temp_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    ks = tl.arange(0, K)

    a = tl.load(a_ptr + rows * K + ks[None, :])
    b = tl.load(b_ptr + ks[:, None] * N + cols)

    # Cube 计算，并通过 GM 临时缓冲区把结果交给 Vector。
    acc = tl.dot(a, b)
    tl.store(temp_ptr + rows * N + cols, acc)

    # 多个 Sub Vector 分别处理一部分行。
    sub_num: tl.constexpr = al.sub_vec_num()
    sub_id = al.sub_vec_id()
    rows_per_sub: tl.constexpr = M // sub_num
    sub_rows = sub_id * rows_per_sub + tl.arange(0, rows_per_sub)[:, None]

    temp = tl.load(temp_ptr + sub_rows * N + cols)
    tl.store(out_ptr + sub_rows * N + cols, tl.exp(temp))
```

该简化示例要求 `M` 能被 `sub_num` 整除。一般场景还需要为不能整除的尾块增加 mask。

## 4. 编译结果

`sub_vec_num()` 的结果在编译期间参与常量计算，不生成独立的 `get_sub_block_num` IR。
在上例中，`sub_num` 会被折叠为编译时当前活动 NPU 对应的常量；`sub_vec_id()` 则生成运行时 ID：

```mlir
%sub_id = hivm.hir.get_sub_block_idx -> i64
```

## 5. 约束说明

- 返回值来自 CPU/JIT 编译阶段对当前活动 NPU 的属性查询，而不是仅根据 IR 中的目标架构静态推导。
- 返回值依赖该设备的 AIV/AIC Core 数量比例，不应假定所有设备都固定返回 2。合法的
  `NPU_DEVICE_LIMIT` 配置会保持这一硬件比例，因此不会单独改变该比值。
- 数据规模不能整除切分数时，需要使用 mask 或其他尾块处理方式。
- `sub_vec_num()` 只提供数量；真实的运行时编号由 `sub_vec_id()` 返回。
- 调用 `sub_vec_id()` 后，开发者需要自行保证所有 Sub Vector 的分片覆盖正确。

## 6. 验证范围

仓库的 `test_sub_vec_num.py` 将该编译期常量用于 Cube 与 Vector 混合计算的数据切分，
并对 NPU 结果进行精度校验。具体返回值取决于目标设备的 AIV/AIC Core 数量比例。
