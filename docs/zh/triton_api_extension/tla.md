# TLA 编程接口说明

## 1. 概述

TLA（Triton Language for Ascend）是 Triton-Ascend 提供的独立编程命名空间，面向 Ascend NPU 上的显式片上内存、SIMD 向量寄存器和流水/核间同步编程。TLA 不复用 `al` / `bl` 扩展接口的编程范式，所有能力通过 `tla.*` 命名空间访问：

```python
import triton.language as tl
import triton.language.extra.tla as tla
```

TLA 的核心组成如下：

| 模块 | 作用 |
| --- | --- |
| `tla` | 缓冲区、视图、拷贝、作用域、标量类型与编译提示 |
| `tla.dma` | GM / UB / L1 / L0 之间的显式 DMA 路由与 Fixpipe 配置 |
| `tla.scope` | 指定 `vector` / `simd` / `cube` 核心作用域 |
| `tla.simd` | SIMD 向量寄存器加载、存储、计算、比较与规约 |
| `tla.cube` | Cube 矩阵乘加 `mmad` / `mmad_mx` |
| `tla.sync` | 流水同步、核间 flag 与全局同步 |
| `tla.where` | 标量与 SIMD 向量选择 |
| `tla.math` | 数学辅助函数 `sqrt` / `truncmod` / `compensated_rstd` |
| `tla.jit` | TLA 前端内核装饰器 |
| `tla.device_print` / `tla.device_assert` | 设备侧调试 |

> 本文是 TLA 总览页，只保留编程模型、数据流和完整示例。各接口的参数、
> 约束和用例由代码 docstring 自动生成，见
> {doc}`../python-api/triton.language.extra.tla`。

接口实现位于 `third_party/ascend/language/tla/`，文档示例位于
`docs/zh/examples/tla/`。新增接口合入后，只要补充源码 docstring 和
`_ascend_constraints.py` 条目，API 参考页会自动刷新。

## 2. 编程模型

TLA 使用显式的片上缓冲区（buffer）和向量寄存器（vector），数据流一般为：

```text
GM --tla.view--> buffer --tla.copy--> UB buffer
                                  --tla.simd.vlds--> vector
                                  --tla.simd.*-----> vector
                                  --tla.simd.vsts--> UB buffer
UB buffer --tla.copy--> GM
```

- `tla.view` / `tla.subview` / `tla.reshape`：把内核指针或已有缓冲区转换为 `tla.buffer` 视图。
- `tla.alloc_buffer` / `tla.alloc`：在 UB / L1 / L0A / L0B / L0C 上分配缓冲区。
- `tla.copy`：按源/目标地址空间自动分派 DMA（GM↔UB、GM→L1、UB↔L1、L1→L0、L0C→…）。
- `tla.dma.*`：显式选择 DMA 路由，例如 `load_gm_to_ub`、`load_gm_to_l1`、`store_l0c_to_gm`。
- `tla.scope.simd()`：标记 SIMD 向量函数区域；`tla.simd.*` 只能操作 `tla.vector`。
- `tla.sync.*`：显式表达流水与核间依赖，`pipe` / `mode` 使用 `tla.sync.PIPE` / `tla.sync.MODE` 枚举。

一个最小示例（TLA 内核必须使用 `@tla.jit`）：

```python
@tla.jit
def simd_only(X_ptr, O_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x_blk = tla.view(X_ptr, (N, ), strides=(1, ))
    o_blk = tla.view(O_ptr, (N, ), strides=(1, ))
    tla.copy(x_blk, x_ub)
    with tla.scope.simd():
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], tla.simd.vadd(v, v))
    tla.copy(o_ub, o_blk)
```

## 3. 数据类型

### 3.1 `tla.buffer` / `tla.buffer_type`

`tla.buffer` 表示一块带形状、元素类型、地址空间、步长和偏移的片上内存视图，不对应 Triton tensor。

| 属性 | 含义 |
| --- | --- |
| `shape` | 各维大小（编译期正整数） |
| `dtype` | 元素类型 |
| `space` | 地址空间：`gm` / `ub` / `l1` / `l0a` / `l0b` / `l0c` |
| `strides` | 各维步长，默认行主序 |
| `offset` | 缓冲区偏移 |

`buf[i0, i1, ...]` 返回 `buffer_access`，可作为 `vlds` / `vsts` 的地址参数，例如 `tla.simd.vlds(x_ub[0, c * 64])`。

### 3.2 `tla.vector` / `tla.vector_type`

`tla.vector` 表示一维 SIMD 寄存器值，包含 `lanes` 和元素类型。它只用于 `tla.simd` 运算，不支持隐式转换为 tensor，也不能作为标量条件使用（`__bool__` 会报错）。

### 3.3 标量 dtype

`tla.bfloat16`、`tla.float16`、`tla.float32`、`tla.int8` / `int16` / `int32` / `int64` 和 `tla.uint8` / `uint16` / `uint32` / `uint64` 与 `triton.language` 中的 dtype 是同一对象，可直接用于 `buffer`、`vector` 和类型转换；`tla.dtype("fp32")` 使用 Triton 的规范名称。

## 4. 作用域

TLA 通过上下文管理器标记核心作用域：

| 接口 | 作用 |
| --- | --- |
| `with tla.scope.vector():` | Vector 核的普通区域，承载 `tla.copy`、控制流等非 SIMD 操作，可嵌套 `simd` |
| `with tla.scope.simd():` | SIMD 向量函数区域；编译器会 outline 为 `hivm.vector_function` |
| `with tla.scope.cube():` | Cube 核区域；在其中完成 GM→L1、`tla.cube.mmad` / `mmad_mx` 和 L0C Fixpipe |

作用域在编译期确定核心类型，不需要运行时参数。SIMD 作用域可以独立使用，也可以嵌套在 `vector` 作用域中；Cube 操作必须位于 `tla.scope.cube()` 内。

## 5. 与 `al` / `bl` 扩展接口的关系

TLA 与 `al` / `bl` 是两套独立的编程接口：

- `al` / `bl` 以 Triton tensor 和隐式内存模型为中心；TLA 使用显式 `tla.buffer` 和 `tla.vector`。
- `tla.sync.sync_all` 使用编译期 `bool` 选择同步范围，而 `al.sync_block_all` 使用字符串 `mode` 和 `event_id`。
- `tla.sync.block_set_flag(MODE, PIPE, flag)` 使用编译期枚举显式指定同步模式与流水；`al.sync_block_set(sender, receiver, event_id, ...)` 由发送端/接收端推导。
- TLA 的 NVIDIA 风格 tensor 运算和隐式类型转换不受支持；跨接口的数据需要使用 `tla.view` / `tla.copy` 显式搬运。

## 6. 完整示例

下面的内核完成 `O = A + B`，并显式插入 GM→UB、向量计算、UB→GM 的同步依赖：

```python
import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def sync_intra(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
    tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    tla.sync.barrier(tla.sync.PIPE.ALL)
    tla.sync.set_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
    tla.sync.wait_flag(tla.sync.PIPE.V, tla.sync.PIPE.MTE3, 1)
    tla.copy(o_ub, o)

    tla.sync.sync_all(False)
```

内核通过 `@tla.jit` 选择 TLA 前端后直接启动，不需要额外的 launch 参数：

```python
sync_intra[(1, )](A, B, O, N=64)
```

## 7. 约束与注意事项

1. TLA 内核必须使用 `@tla.jit`；`@triton.jit` 会走通用前端，不会识别 `tla.*` 接口。
2. `tla.simd` 的输入输出必须是 `tla.vector`，不能直接传入 Triton tensor。
3. 二元运算要求两个操作数类型完全一致；掩码必须是与值向量同 lane 数的 `int1` 谓词向量。
4. SIMD 寄存器宽度为 2048 位；`i64` / `u64` 使用 4096 位。不支持的元素类型会直接报错，不会做隐式位宽映射。
5. `tla.copy` 只接受 `tla.buffer`；非默认路由配置通过 `params=` 传入对应 `*Params` 对象，或改用 `tla.dma.*` 显式接口。
6. `tla.cube.mmad` / `mmad_mx` 和 L1/L0C DMA 必须位于 `tla.scope.cube()` 内。
7. `tla.device_print` 不能在 `tla.scope` 内使用，且只支持 UB 上的值。
8. 各接口的具体约束和用例见 {doc}`../python-api/triton.language.extra.tla`。
