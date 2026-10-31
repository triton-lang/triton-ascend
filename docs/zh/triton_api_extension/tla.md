# TLA 编程接口说明

## 1. 概述

TLA（Triton Language for Ascend）是 Triton-Ascend 提供的独立编程命名空间，面向 Ascend NPU 上的显式片上内存、SIMD 向量寄存器和流水/核间同步编程。TLA 不复用 `al` / `bl` 扩展接口的编程范式，所有能力通过 `tla.*` 命名空间访问：

```python
import triton
import triton.language as tl
import triton.language.extra.tla as tla
```

TLA 的核心组成如下：

| 模块 | 作用 |
| --- | --- |
| `tla` | 缓冲区、视图、拷贝、作用域与编译提示 |
| `tla.scope` | 指定 `vector` / `simd` / `cube` 核心作用域 |
| `tla.simd` | SIMD 向量寄存器加载、存储、计算、比较与规约 |
| `tla.cube` | Cube 矩阵乘加 `mmad` |
| `tla.sync` | 流水同步、核间 flag 与全局同步 |
| `tla.math` | TLA 数学辅助函数 |
| `tla.device_print` / `tla.device_assert` | 设备侧调试 |

> 本文档描述当前 `main-dev` 分支已经合入的 TLA 接口。接口实现位于
> `third_party/ascend/language/tla/`，示例位于
> `third_party/ascend/unittest/tla_ut/`。接口合入 `main-dev` 后同步刷新本文档。

## 2. 编程模型

TLA 使用显式的片上缓冲区（buffer）和向量寄存器（vector），数据流一般为：

```text
GM --tla.view--> buffer --tla.copy--> UB buffer
                                  --tla.simd.vlds--> vector
                                  --tla.simd.*-----> vector
                                  --tla.simd.vsts--> UB buffer
UB buffer --tla.copy--> GM
```

- `tla.view`：把内核指针或已有缓冲区切片转换为 `tla.buffer`。
- `tla.alloc_buffer`：在 UB / L1 / L0A / L0B / L0C 上分配缓冲区。
- `tla.copy`：执行 buffer 之间的 DMA（GM↔UB、GM→L1、L0C→…）。
- `tla.scope.simd()`：标记 SIMD 向量函数区域；`tla.simd.*` 只能操作 `tla.vector`。
- `tla.sync.*`：显式表达流水与核间依赖。

一个最小示例：

```python
@triton.jit
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

## 4. 作用域

TLA 通过上下文管理器标记核心作用域：

| 接口 | 作用 |
| --- | --- |
| `with tla.scope.vector():` | Vector 核的普通区域，承载 `tla.copy`、控制流等非 SIMD 操作，可嵌套 `simd` |
| `with tla.scope.simd():` | SIMD 向量函数区域；编译器会 outline 为 `hivm.vector_function` |
| `with tla.scope.cube():` | Cube 核区域（当前保留，Cube 算子尚未通过该作用域暴露） |

作用域在编译期确定核心类型，不需要运行时参数。SIMD 作用域可以独立使用，也可以嵌套在 `vector` 作用域中。

## 5. 内存分配与数据搬运

### 5.1 `tla.alloc` / `tla.alloc_buffer`

```python
tla.alloc_buffer(shape, dtype, space="ub", versions=1)
tla.alloc(shape, dtype, space="ub", versions=1)
```

在片上分配一个缓冲区，返回 `tla.buffer`。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `shape` | `tuple[int, ...]` | 各维大小，必须为正的编译期整数 |
| `dtype` | `tl.dtype` | 元素类型 |
| `space` | `str` | 地址空间，默认 `"ub"`；不能为 `"gm"` |
| `versions` | `int` | 多缓冲版本数，正整数；`1` 表示关闭多缓冲 |

`alloc_buffer` 与 `alloc` 等价。GM 缓冲区必须由内核指针经 `tla.view` 获得，不能用 `alloc` 分配。

### 5.2 `tla.copy`

```python
tla.copy(src, dst)
```

在两个 `tla.buffer` 之间执行 DMA，源和目标都必须是 `tla.buffer`。

当前支持的地址空间组合：

| `src` → `dst` | 语义 |
| --- | --- |
| `gm` → `ub` | GM 加载到 UB |
| `ub` → `gm` | UB 存储到 GM |
| `gm` → `l1` | ND → NZ，搬运到 L1 |
| `l0c` → 其他 | Fixpipe 搬出；2 维目标使用 `NZ2ND`，否则使用普通模式 |

约束：

- 源和目标元素类型必须一致。
- 除 `gm→l1` 的 ND→NZ 以及 `l0c` Fixpipe 的 fractal 关系外，源和目标形状必须一致。
- 其他地址空间组合会报错。

### 5.3 `tla.view`

```python
tla.view(src, shape, strides=None, offsets=None, space="gm")
```

把内核指针或已有缓冲区转换为指定形状的 `tla.buffer`。

| `src` 类型 | 行为 |
| --- | --- |
| 指针 tensor | 指针入口视图；`space` 指定 GM 地址空间，`strides` 为最终 memref 步长，默认行主序 |
| `tla.buffer` | 缓冲区切片（subview）；`strides` 相对于源视图，`space` 必须保持默认值 |

`offsets` 为各维偏移，支持编译期整数或标量整数 tensor。指针入口会生成 1 维 `hivm.hir.pointer_cast` 和一次 `memref.reinterpret_cast`；缓冲区切片生成 `memref.subview`。

### 5.4 `tla.reshape`

```python
tla.reshape(src, shape)
```

对连续的、行主序且无偏移的缓冲区做纯形状变换，元素总数必须保持不变。内部使用 `memref.collapse_shape` / `memref.expand_shape`。带步长或偏移的切片请先使用 `tla.view`。

### 5.5 `tla.annotate_buffer_versions`

```python
tla.annotate_buffer_versions(buf, versions)
```

为单个缓冲区声明多缓冲版本数，供 NPU-IR 多缓冲 pass 使用。`versions` 必须为正整数，`1` 表示关闭多缓冲。

## 6. SIMD 向量寄存器接口

### 6.1 数据类型与寄存器宽度

SIMD 支持的元素类型：

| 元素类型 | 标签 | 满寄存器 lane 数 |
| --- | --- | --- |
| `int8` / `float8e4nv` | `i8` / `fp8e4nv` | 256 |
| `int16` / `float16` / `bfloat16` | `i16` / `f16` / `bf16` | 128 |
| `int32` / `float32` | `i32` / `f32` | 64 |
| `int64` / `uint64` | `i64` | 64（4096 位寄存器） |

其他元素类型会报错。无符号整型在底层按同宽有符号整型处理。

谓词掩码为 `vector<lanes x int1>`；未显式提供掩码时，接口使用全真 `pge <ALL>` 掩码。

### 6.2 加载与存储

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vlds(ptr, offsets=None, *, dist="NORM", elem=None)` | 从 UB buffer 加载一个满向量寄存器；`ptr` 可为 `buf[i, ...]`，`elem` 可覆盖结果元素类型 |
| `tla.simd.vsts(ptr, offsets_or_val, val=None, *, dist=None, mask=None)` | 将向量寄存器掩码存储到 UB buffer；`dist` 默认 `NORM_B32`，`mask` 默认全真 |

`dist` 必须是 AVE 支持的 load/store 分布字符串，例如 `NORM`、`UNPK_B16`、`UNPK4_B8`、`BRC_B32`、`NORM_B32`、`ONEPT_B32` 等。

### 6.3 类型转换与布局变换

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vextf(src, target_dtype, mask=None, part="part_even")` | 拓宽浮点转换，当前仅支持转换到 `f32`；结果为 64 lane 满寄存器 |
| `tla.simd.vtruncf(src, target_dtype, mask=None, part="part_even")` | 收窄浮点转换，目标为 `bf16` / `f16` / `fp8e4nv`；总位宽保持不变 |
| `tla.simd.vbitcast(src, dtype_str)` | 按位重解释向量寄存器，结果 lane 数按总位宽换算 |
| `tla.simd.vintlv(src1, src2)` | 两个同类型寄存器的成对交织，返回两个结果寄存器 |
| `tla.simd.vdintlv(src1, src2)` | 成对解交织，是 `vintlv` 的逆操作 |

`part` 仅支持 `part_even` / `part_odd`，用于选择源寄存器的子区域。

### 6.4 算术与一元运算

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vmul(lhs, rhs)` | 逐元素乘 |
| `tla.simd.vadd(lhs, rhs)` | 逐元素加 |
| `tla.simd.vdiv(lhs, rhs)` | 逐元素除 |
| `tla.simd.vxor(lhs, rhs)` | 逐元素按位异或 |
| `tla.simd.vmod(lhs, rhs)` | 逐元素整数取模 |
| `tla.simd.vfma(acc, lhs, rhs)` | 融合乘加：`acc + lhs * rhs` |
| `tla.simd.vneg(src)` | 逐元素取负 |
| `tla.simd.vsqrt(src)` | 逐元素平方根 |
| `tla.simd.vrsqrt(src)` | 逐元素倒数平方根 |
| `tla.simd.vexp(src)` | 逐元素指数 |

二元运算要求两个操作数类型完全一致；一元运算保持元素类型不变。所有运算使用全真掩码。

### 6.5 比较与选择

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vcmp(lhs, rhs, cmp_type)` | 逐元素比较，`cmp_type` 支持 `eq` / `ne` / `gt` / `ge` / `lt` / `le`，返回 `int1` 谓词向量 |
| `tla.simd.vsel(mask, true_val, false_val)` | 按位选择；`mask` 的 lane 数必须与值向量一致，两个值向量类型必须一致 |

### 6.6 规约与广播

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vreduce(src, kind)` | 水平规约，`kind` 为 `add` / `min` / `max`，结果放在 lane 0 |
| `tla.simd.vreduce_sum(src)` | `vreduce(src, "add")` 的别名 |
| `tla.simd.vreduce_max(regs)` | 对一个寄存器或寄存器列表做全局最大值规约，并广播到所有 lane |
| `tla.simd.vreduce_min(regs)` | 对一个寄存器或寄存器列表做全局最小值规约，并广播到所有 lane |
| `tla.simd.vbroadcast(src, low=True)` | 把源寄存器的低半部分（`low=True`）或高半部分广播到整个寄存器 |

`vreduce_max` / `vreduce_min` 会先对多个寄存器做逐元素 `max` / `min`，再水平规约并广播，结果可直接用于 `vcmp`。

### 6.7 常量、索引与谓词

| 接口 | 说明 |
| --- | --- |
| `tla.simd.vfull(value, dtype, lanes)` | 将标量广播为指定 lane 数的常量寄存器 |
| `tla.simd.vsplat(value, dtype)` | 按元素类型推导满寄存器 lane 数并广播常量 |
| `tla.simd.vci(base)` | 生成 lane 索引寄存器，lane `i` 的值为 `base + i`；固定为 `int32` / 64 lane |
| `tla.simd.pge(n, lanes)` | 前缀谓词：前 `n` 个 lane 为真，其余为假；`n` 支持 1/2/3/4/8/16/32/64/128 |

### 6.8 内存屏障

```python
tla.simd.membar(scope)
```

插入 VF 内的内存屏障。`scope` 支持 `VV_ALL`、`VST_VLD`、`VLD_VST`、`VST_VST`、`VS_ALL`、`VST_LD`、`VLD_ST`、`VST_ST`、`SV_ALL`、`ST_VLD`、`LD_VST`、`ST_VST`、`SS_ALL`、`ST_LD`、`LD_ST`、`ST_ST` 等 AVE membar 类型。

## 7. Cube 接口

### 7.1 `tla.cube.mmad`

```python
tla.cube.mmad(a, b, acc, transpose_b=False, clear_accum=False)
```

计算 `C = A × B (+ C)`：

| 参数 | 说明 |
| --- | --- |
| `a` | L1 缓冲区，形状为 `(M, K)`（2 维 ND）或等价的 4 维 zN 形状 |
| `b` | L1 缓冲区，形状为 `(K, N)` 或等价的 4 维 zN 形状 |
| `acc` | L0C 缓冲区，形状为 `(M, N)` |
| `transpose_b` | 是否转置 `b` |
| `clear_accum` | 是否清空累加器后计算 |

`a`、`b`、`acc` 必须是 `tla.buffer`，且 `a` / `b` 位于 L1、`acc` 位于 L0C；`a` 和 `b` 元素类型必须一致。`transpose_a` 不在该接口暴露，可通过 GM→L1 拷贝时调整布局实现。

## 8. 同步接口

### 8.1 流水同步

| 接口 | 说明 |
| --- | --- |
| `tla.sync.barrier(pipe)` | 等待指定流水上的前序指令完成 |
| `tla.sync.set_flag(pipe1, pipe2, event_id)` | 生产者在 `pipe1` 上向消费者 `pipe2` 发送事件，`event_id` 范围 `[0, 7]` |
| `tla.sync.wait_flag(pipe1, pipe2, event_id)` | 消费者在 `pipe2` 上等待来自 `pipe1` 的事件，`event_id` 范围 `[0, 7]` |

### 8.2 全局同步

```python
tla.sync.sync_all(mode)
```

硬件全局屏障：`mode=False` 同步 AIV 核，`mode=True` 同步 AIC 和 AIV 全部核。`mode` 必须是编译期 `bool`。

### 8.3 核间 flag

| 接口 | 说明 |
| --- | --- |
| `tla.sync.block_set_flag(mode, pipe, flag)` | 从当前核心发送核间 flag |
| `tla.sync.block_wait_flag(pipe, flag)` | 在指定流水上等待核间 flag |

`block_set_flag` 的 `mode` 取值：

| `mode` | 含义 |
| --- | --- |
| `0` | inter-block |
| `1` | inter-subblock |
| `2` | intra-block |

`flag` 范围为 `[0, 15]`，`block_wait_flag` 必须使用与 `block_set_flag` 相同的 flag ID。

### 8.4 流水名称

`pipe` 为编译期字符串，可写作 `"V"`、`"MTE2"` 或完整名称 `"PIPE_V"`、`"PIPE_MTE2"`。支持：

`PIPE_S`、`PIPE_V`、`PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_ALL`、`PIPE_MTE4`、`PIPE_MTE5`、`PIPE_V2`、`PIPE_FIX`。

`barrier("ALL")` 等价于 `PIPE_ALL`。

## 9. 数学辅助

### 9.1 `tla.compensated_rstd`

```python
tla.compensated_rstd(square_sum, hidden, eps)
```

计算 `square_sum / hidden + eps` 的补偿倒数标准差（rstd），使用 TwoSum 误差恢复和 Newton 迭代，返回逐 lane 复制的 `f32` 向量寄存器。

| 参数 | 说明 |
| --- | --- |
| `square_sum` | `f32` 向量寄存器，保存 `sum_h x[h]^2`，通常为广播到所有 lane 的值 |
| `hidden` | 隐藏维大小，正的 Python / constexpr 标量 |
| `eps` | epsilon，必须不小于最小 FP32 正规数 `2^-126` |

该接口与 TileKernels `rstd.py` 的 `make_rstd_impl` 语义保持一致，不是普通 RMSNorm 公式。

## 10. 调试接口

| 接口 | 说明 |
| --- | --- |
| `tla.device_print(prefix, *args, hex=False)` | 在设备侧打印值；仅允许在 `tla.scope` 外使用，且只支持 UB 缓冲区 / UB 元素范围 / 标量 |
| `tla.device_assert(cond, msg="")` | 运行时断言；仅在开启 `debug` 选项时生效，`cond` 必须是标量 `int1` |

## 11. 与 `al` / `bl` 扩展接口的关系

TLA 与 `al` / `bl` 是两套独立的编程接口：

- `al` / `bl` 以 Triton tensor 和隐式内存模型为中心；TLA 使用显式 `tla.buffer` 和 `tla.vector`。
- `tla.sync.sync_all` 使用编译期 `bool` 选择同步范围，而 `al.sync_block_all` 使用字符串 `mode` 和 `event_id`。
- `tla.sync.block_set_flag(mode, pipe, flag)` 显式指定核间同步模式；`al.sync_block_set(sender, receiver, event_id, ...)` 由发送端/接收端推导。
- TLA 的 NVIDIA 风格 tensor 运算和隐式类型转换不受支持；跨接口的数据需要使用 `tla.view` / `tla.copy` 显式搬运。

## 12. 完整示例

下面的内核完成 `O = A + B`，并显式插入 GM→UB、向量计算、UB→GM 的同步依赖：

```python
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def sync_intra(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")

    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.sync.set_flag("MTE2", "V", 0)
    tla.sync.wait_flag("MTE2", "V", 0)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        tla.simd.vsts(o_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")

    tla.sync.barrier("ALL")
    tla.sync.set_flag("V", "MTE3", 1)
    tla.sync.wait_flag("V", "MTE3", 1)
    tla.copy(o_ub, o)

    tla.sync.sync_all(False)
```

启动时需要通过 TLA 模式编译：

```python
sync_intra[(1, )](A, B, O, N=64, enable_tla_mode=True)
```

## 13. 约束与注意事项

1. `tla.simd` 的输入输出必须是 `tla.vector`，不能直接传入 Triton tensor。
2. 二元运算要求两个操作数类型完全一致；掩码必须是与值向量同 lane 数的 `int1` 谓词向量。
3. SIMD 寄存器宽度为 2048 位；`i64` / `u64` 使用 4096 位。不支持的元素类型会直接报错，不会做隐式位宽映射。
4. `tla.copy` 只接受 `tla.buffer`，不支持指针或 tensor 直接拷贝。
5. `tla.device_print` 不能在 `tla.scope` 内使用，且只支持 UB 上的值。
6. 本文档所述的 TLA 接口以 `main-dev` 当前实现为准；新增接口合入后应同步更新本文档。
