# al.SYNC_IN_VF 接口文档

## 1. 硬件背景

Vector 核中的 Vector 和 Scalar 访存操作可能需要明确的先后顺序。例如，后续操作依赖前一次
写入的结果时，需要等待对应的写入完成。`al.SYNC_IN_VF` 用于选择这种访存同步的范围。

## 2. 接口说明

### 2.1 枚举定义

```python
class SYNC_IN_VF(enum.Enum):
    VV_ALL = enum.auto()
    VST_VLD = enum.auto()
    VLD_VST = enum.auto()
    VST_VST = enum.auto()
    VS_ALL = enum.auto()
    VST_LD = enum.auto()
    VLD_ST = enum.auto()
    VST_ST = enum.auto()
    SV_ALL = enum.auto()
    ST_VLD = enum.auto()
    LD_VST = enum.auto()
    ST_VST = enum.auto()
```

这些枚举值作为 `al.debug_barrier` 的参数使用：

```python
al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
```

前端向 IR 传递枚举名称，因此不应把 Python 枚举的 `.value` 当作硬件指令编码。

### 2.2 命名说明

缩写含义如下：

- `VLD`：Vector Load；
- `VST`：Vector Store；
- `LD`：Scalar Load；
- `ST`：Scalar Store；
- `V`：Vector Load/Store；
- `S`：Scalar Load/Store。

除三个 `ALL` 模式外，枚举名 `A_B` 表示：先等待屏障之前的 A 类访存完成，
再允许屏障之后的 B 类访存继续。

| 枚举值 | 同步范围 |
| --- | --- |
| `VV_ALL` | 之前的 Vector Load/Store → 之后的 Vector Load/Store |
| `VST_VLD` | 之前的 Vector Store → 之后的 Vector Load |
| `VLD_VST` | 之前的 Vector Load → 之后的 Vector Store |
| `VST_VST` | 之前的 Vector Store → 之后的 Vector Store |
| `VS_ALL` | 之前的 Vector Load/Store → 之后的 Scalar Load/Store |
| `VST_LD` | 之前的 Vector Store → 之后的 Scalar Load |
| `VLD_ST` | 之前的 Vector Load → 之后的 Scalar Store |
| `VST_ST` | 之前的 Vector Store → 之后的 Scalar Store |
| `SV_ALL` | 之前的 Scalar Load/Store → 之后的 Vector Load/Store |
| `ST_VLD` | 之前的 Scalar Store → 之后的 Vector Load |
| `LD_VST` | 之前的 Scalar Load → 之后的 Vector Store |
| `ST_VST` | 之前的 Scalar Store → 之后的 Vector Store |

## 3. 约束说明

- `al.debug_barrier` 的参数必须是一个 `al.SYNC_IN_VF` 枚举值。
- 同步模式应与屏障前后的访存类型相匹配；不必要的屏障可能带来额外开销。
- 该接口通常在 `al.scope(core_mode="vector")` 中使用。
- 当前走 AVE lowering 的 RegBased 后端路径会把前端标记继续降低为 VF 访存屏障；
  MemBased 后端路径不提供这一步降低。

## 4. 用例示例

下面是一个 Triton Kernel 函数示例。示例假设 `BLOCK` 是 2 的幂，以满足 `tl.arange`
对区间长度的要求。它在 Vector Load 与后续 Vector Store 之间插入 `VV_ALL` 屏障：

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

## 5. 编译输出结果

前端首先生成带有枚举名称的关键同步标记：

```mlir
%c0_i64 = arith.constant 0 : i64
annotation.mark %c0_i64 {SYNC_IN_VF = "VV_ALL"} : i64
```

在当前走 AVE lowering 的 RegBased 后端路径中，该标记会继续降低为：

```mlir
%c0_i32 = arith.constant 0 : i32
ave.hir.membar %c0_i32
```

因此，看到前端的 `annotation.mark` 只能说明同步模式已经记录到 IR；目标程序中是否生成
VF 访存屏障，还取决于目标后端是否支持对应的降低路径。
