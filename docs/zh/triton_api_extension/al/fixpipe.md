# al.fixpipe 接口文档

## 1. 硬件背景

昇腾 A5 及后续架构新增了 **L0C → UB** 的专用数据通路（fixpipe），用于将矩阵乘（Cube 单元）输出从 L0C 缓冲区高效搬运到 UB（Vector 单元输入缓冲区）。`al.fixpipe` 是该通路的前端显式调用接口，可在搬运过程中顺带完成格式转换（NZ↔ND）、预量化（pre-quant）、预激活（pre-ReLU）等操作。

## 2. 接口说明

```python
def fixpipe(
    src: tl.tensor,
    dst: bl.buffer,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    pre_quant_mode: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT,
    pre_relu_mode: FixpipePreReluMode = FixpipePreReluMode.NO_RELU,
    _builder=None,
) -> None:
```

### 相关枚举

```python
class FixpipeDMAMode(enum.Enum):
    NZ2DN = ascend_ir.FixpipeDMAMode.NZ2DN   # NZ 格式 → DN 格式
    NZ2ND = ascend_ir.FixpipeDMAMode.NZ2ND   # NZ 格式 → ND 格式
    NZ2NZ = ascend_ir.FixpipeDMAMode.NZ2NZ   # NZ 格式 → NZ 格式

class FixpipeDualDstMode(enum.Enum):
    NO_DUAL = ascend_ir.FixpipeDualDstMode.NO_DUAL           # 单目标
    COLUMN_SPLIT = ascend_ir.FixpipeDualDstMode.COLUMN_SPLIT # 列切分双目标
    ROW_SPLIT = ascend_ir.FixpipeDualDstMode.ROW_SPLIT       # 行切分双目标

class FixpipePreQuantMode(enum.Enum):
    NO_QUANT = ascend_ir.FixpipePreQuantMode.NO_QUANT  # 不做量化
    F322BF16 = ascend_ir.FixpipePreQuantMode.F322BF16  # f32 → bf16
    F322F16  = ascend_ir.FixpipePreQuantMode.F322F16   # f32 → f16
    S322I8   = ascend_ir.FixpipePreQuantMode.S322I8    # s32 → i8 量化

class FixpipePreReluMode(enum.Enum):
    NO_RELU     = ascend_ir.FixpipePreReluMode.NO_RELU      # 不做激活
    NORMAL_RELU = ascend_ir.FixpipePreReluMode.NORMAL_RELU  # ReLU
    LEAKY_RELU  = ascend_ir.FixpipePreReluMode.LEAKY_RELU   # Leaky ReLU
    P_RELU      = ascend_ir.FixpipePreReluMode.P_RELU       # PReLU
```

## 3. 参数说明

| 参数名 | 类型 | 是否必需 | 说明 |
|--------|------|----------|------|
| `src` | `tl.tensor` | 是 | 源张量，必须位于 L0C 内存区域（通常是 `tl.dot` 的输出） |
| `dst` | `bl.buffer` | 是 | 目标缓冲区，必须位于 UB 内存区域（`ascend_address_space.UB`） |
| `dma_mode` | `al.FixpipeDMAMode` | 否 | 数据搬运格式模式，默认 `NZ2ND`；可选 `NZ2DN` / `NZ2ND` / `NZ2NZ` |
| `dual_dst_mode` | `al.FixpipeDualDstMode` | 否 | 双目标模式控制，仅在 `NZ2ND`/普通模式下可启用；默认 `NO_DUAL` |
| `pre_quant_mode` | `al.FixpipePreQuantMode` | 否 | 搬运过程中的预量化/类型转换，默认 `NO_QUANT` |
| `pre_relu_mode` | `al.FixpipePreReluMode` | 否 | 搬运过程中的预激活函数，默认 `NO_RELU` |
| `_builder` | - | 内部参数 | 编译器自动传参，用户无需使用 |

## 4. 返回值

无返回值，数据直接写入传入的 `dst` buffer。

## 5. 约束说明

- `fixpipe` 仅支持 **L0C → UB** 方向的数据搬运。
- `src` 必须是 `tl.dot`（矩阵乘）的结果（Cube 单元输出，驻留 L0C）。
- `dst` 必须是 memscope 为 UB 的 buffer。
- `dual_dst_mode` 非 `NO_DUAL` 时仅支持 `NZ2ND`/`NZ2DN` 模式。

## 6. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def fixpipe_kernel(A_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)

    # 从 GM 加载 A 块（示例：简化场景）
    offs_i = tl.arange(0, M)[:, None]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (pid + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)  # [M, K]

    # 在 UB 分配目标 buffer
    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)

    # 将 L0C 上的 Cube 输出通过 fixpipe 搬到 UB
    al.fixpipe(a_vals, ub, dma_mode=al.FixpipeDMAMode.NZ2ND,
               dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)


@triton.jit
def fixpipe_with_preprocess(A_ptr, M: tl.constexpr, N: tl.constexpr,
                            K: tl.constexpr):
    """示例：fixpipe 同时做类型转换和激活。"""
    pid = tl.program_id(0)
    offs_i = tl.arange(0, M)[:, None]
    offs_k = tl.arange(0, K)
    a_vals = tl.load(A_ptr + (pid + offs_i) * K + offs_k[None, :])

    ub = bl.alloc(tl.float16, [M, N], al.ascend_address_space.UB)
    al.fixpipe(a_vals, ub,
               dma_mode=al.FixpipeDMAMode.NZ2ND,
               pre_quant_mode=al.FixpipePreQuantMode.F322F16,
               pre_relu_mode=al.FixpipePreReluMode.NORMAL_RELU)
```

## 7. 编译输出结果

以 `M=N=K=16` 为例，核心 IR 片段：

```mlir
// ... 前置 tl.load 等指令省略 ...

// UB 侧目标 buffer 分配
%alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
annotation.mark %alloc {effects = ["write", "read"]}
    : memref<16x16xf32, #hivm.address_space<ub>>

// fixpipe 对应 hivm.hir.fixpipe op
//   ins  ：源 tensor（Cube 输出，L0C）
//   outs ：目标 memref（UB）
//   属性 ：dma_mode、dual_dst_mode、pre_quant、pre_relu
hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
    ins(%13 : tensor<16x16xf32>)
    outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    dual_dst_mode = <NO_DUAL>
```

### 输出要点说明

- `bl.alloc` + `annotation.mark` 与 `bl.alloc` 文档一致。
- `al.fixpipe` 降低为 `hivm.hir.fixpipe` op，通过 `ins(...) outs(...)` 形式关联源 tensor 和目标 memref。
- `dma_mode` 以 `#hivm.dma_mode<...>` 枚举属性形式存在；`dual_dst_mode` 等为字符串/枚举属性。
- pre_quant / pre_relu 在开启时会作为附加属性出现，并直接由硬件在搬运过程中完成计算，无需额外 Vector 指令。

## 8. 相关接口

- [`bl.alloc`](../bl/alloc.md)：分配 UB 目标 buffer
- [`tl.dot`](../../triton_api/Linear_Algebra_Ops/dot.md)：矩阵乘（Cube 计算，输出驻留 L0C）
- [`al.ascend_address_space`](../al/ascend_address_space.md)：昇腾地址空间枚举
