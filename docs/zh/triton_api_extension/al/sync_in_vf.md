# al.SYNC_IN_VF

## 1. 概述

`al.SYNC_IN_VF` 是 Vector/Scalar 访存流水之间的细粒度同步模式枚举。
它不是可直接调用的函数，只能作为 `al.debug_barrier` 的参数：

```python
al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
```

该接口与 `tl.debug_barrier()` 不同。`al.debug_barrier` 首先生成 Ascend 专用的 VF 同步标记；
在支持该能力的后端路径中，该标记才会继续降低为 VF memory barrier，而不是 Triton 通用的线程块屏障。

## 2. 命名规则

除 `VV_ALL`、`VS_ALL` 和 `SV_ALL` 外，枚举名 `A_B` 表示：等待先前的 A 类访存完成，
然后才允许后续的 B 类访存继续。

缩写含义如下：

- `VLD`：Vector Load；
- `VST`：Vector Store；
- `LD`：Scalar Load；
- `ST`：Scalar Store；
- `V`：Vector Load/Store；
- `S`：Scalar Load/Store。

## 3. 枚举值

| 枚举值 | 同步语义 |
| --- | --- |
| `VV_ALL` | 等待先前所有 Vector Load/Store 完成，再执行后续 Vector Load/Store |
| `VST_VLD` | 等待先前 Vector Store 完成，再执行后续 Vector Load |
| `VLD_VST` | 等待先前 Vector Load 完成，再执行后续 Vector Store |
| `VST_VST` | 等待先前 Vector Store 完成，再执行后续 Vector Store |
| `VS_ALL` | 等待先前所有 Vector Load/Store 完成，再执行后续 Scalar Load/Store |
| `VST_LD` | 等待先前 Vector Store 完成，再执行后续 Scalar Load |
| `VLD_ST` | 等待先前 Vector Load 完成，再执行后续 Scalar Store |
| `VST_ST` | 等待先前 Vector Store 完成，再执行后续 Scalar Store |
| `SV_ALL` | 等待先前所有 Scalar Load/Store 完成，再执行后续 Vector Load/Store |
| `ST_VLD` | 等待先前 Scalar Store 完成，再执行后续 Vector Load |
| `LD_VST` | 等待先前 Scalar Load 完成，再执行后续 Vector Store |
| `ST_VST` | 等待先前 Scalar Store 完成，再执行后续 Vector Store |

Python 枚举使用 `enum.auto()`，但前端实际向 IR 传递枚举名称，后端再将名称映射为目标指令。
不要把 Python 层的 `.value` 当作稳定的硬件编码。

## 4. 使用方法

下面的示例在先前的 Vector Load 与后续的 Vector Store 之间插入 VF 屏障：

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

在当前 RegBased 后端路径中，`ProcessMembar` 会继续将其降低为：

```mlir
%c0_i32 = arith.constant 0 : i32
ave.hir.membar %c0_i32
```

`VV_ALL` 最终对应 `hivm.mem.bar.vv.all` intrinsic；其他枚举值映射到各自的 membar intrinsic。
当前源码将 Ascend310B 和 Ascend950 归为 RegBased 架构；Ascend910B、Ascend910_93 等
MemBased 架构不会经过这条 `ProcessMembar` 路径，因此不能从前端生成 `annotation.mark` 推导出
目标设备程序一定包含 VF membar。

## 5. 约束说明

- `al.debug_barrier` 的参数必须是 `al.SYNC_IN_VF` 枚举值。
- 该接口面向 Vector/Scalar 访存流水同步，建议在 `al.scope` 中按真实访存依赖使用。
- 屏障模式必须与屏障前后的指令类型一致；多余的屏障可能引入性能开销，错误的模式可能无法覆盖
  真实依赖。
- 当前只有 RegBased 后端路径会把该标记降低为 VF membar；MemBased 后端路径不提供这一降低。

## 6. 验证范围

仓库的 `test_ascend_barrier.py` 验证了 `al.debug_barrier(al.SYNC_IN_VF.VV_ALL)`
能够生成带 `SYNC_IN_VF = "VV_ALL"` 的 `annotation.mark`。后端源码覆盖了 12 种模式到
membar intrinsic 的映射，但当前测试没有在所有目标芯片上逐项执行 12 种模式。
