# al.debug_barrier

## 1. 概述

`al.debug_barrier` 声明 Vector/Scalar 访存流水之间的 Ascend 专用细粒度同步。
同步类型由 [`al.SYNC_IN_VF`](./sync_in_vf.md) 指定。

该接口与 `tl.debug_barrier()` 不同：前者先生成 VF 同步标记，并在支持该能力的后端路径中降低为
VF memory barrier；后者是 Triton 通用的线程块屏障。

## 2. 接口说明

```python
al.debug_barrier(sync_mode: al.SYNC_IN_VF) -> None
```

内部注入参数不属于用户接口，因此未在签名中列出。

### 参数

| 参数名 | 类型 | 含义 |
| --- | --- | --- |
| `sync_mode` | [`al.SYNC_IN_VF`](./sync_in_vf.md) | Vector/Scalar 访存流水的同步模式 |

### 返回值

无。

## 3. 使用方法

下面的示例等待屏障前的 Vector Load/Store 完成，再允许屏障后的 Vector Load/Store 继续：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def vector_barrier_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)

    with al.scope(core_mode="vector"):
        values = tl.load(x_ptr + offsets)
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        tl.store(y_ptr + offsets, values)
```

前端生成的关键 IR 为：

```mlir
%c0_i64 = arith.constant 0 : i64
annotation.mark %c0_i64 {SYNC_IN_VF = "VV_ALL"} : i64
```

在当前 RegBased 后端路径中，`ProcessMembar` 会继续把该标记降低为对应的 VF membar 指令。
当前源码将 Ascend310B 和 Ascend950 归为 RegBased 架构；Ascend910B、Ascend910_93 等
MemBased 架构不会经过这条降低路径。

## 4. 约束说明

- `sync_mode` 必须是 `al.SYNC_IN_VF` 枚举值。
- 建议在 `al.scope(core_mode="vector")` 中按真实访存依赖使用。
- 屏障模式必须与屏障前后的指令类型一致；多余的屏障可能引入性能开销，错误的模式可能无法覆盖
  真实依赖。
- 当前只有 RegBased 后端路径会把该标记降低为 VF membar；MemBased 后端路径不提供这一降低。

## 5. 验证范围

仓库的 `test_ascend_barrier.py` 验证了
`al.debug_barrier(al.SYNC_IN_VF.VV_ALL)` 能生成带 `SYNC_IN_VF = "VV_ALL"` 的
`annotation.mark`。该用例是编译检查，不等同于在所有目标芯片上执行全部同步模式。
