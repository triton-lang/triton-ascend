# al.custom（CustomOp 和 CustomMacro）接口文档

## 1. 硬件背景

当需要把已有的设备侧实现接入 Triton 时，可以使用 CustomOp 将实现信息加入编译流程。注册类负责描述操作使用的核、执行模式、流水线和设备侧符号，Triton Kernel 通过 `al.custom` 调用已经注册的操作。

Python 层没有相互独立的 `CustomOp` 和 `CustomMacro` 类，两者使用相同的注册与调用接口，通过 `pipe` 的配置形式区分：单个 `al.PIPE` 表示普通 CustomOp，两个 `al.PIPE` 组成的 `tuple` 或 `list` 表示 CustomMacro。CustomMacro 还可以使用 `SyncEventSlot` 描述设备侧实现中的流水线同步关系。

## 2. 接口说明

接口签名片段：

```python
def register_custom_op(op):
    ...


def custom(name: str, *args, **kwargs):
    ...


class SyncEventSlot:
    def __init__(
        self,
        set_pipe=None,
        wait_pipe=None,
        sync=None,
        event=None
    ):
        ...
```

### 2.1 使用流程

1. 使用 `@al.register_custom_op` 装饰一个配置类，注册名称和设备侧实现信息。
2. 在 `@triton.jit` 函数中使用 `al.custom(name, ..., out=...)` 按注册名称调用。
3. 编译器根据注册类中的 `pipe` 配置生成普通 CustomOp 或 CustomMacro。

### 2.2 al.custom 入参

| 参数名 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `name` | `str` | 是 | 已经通过 `@al.register_custom_op` 注册的 CustomOp 或 CustomMacro 名称 |
| `*args` / `**kwargs` | - | 按操作定义 | 传给设备侧实现的输入；注册类定义 `__init__` 时，调用参数必须与其签名一致，否则按实际调用顺序传入 |
| `out` | `tl.tensor` / `tl.tuple` / `tuple` / `list` | 否 | 输出占位值 |

### 2.3 返回值

提供一个输出占位张量时，返回与其类型相同的 `tl.tensor`；提供多个时，返回 `tl.tuple`，其中各张量按顺序与对应的 `out` 保持相同类型；省略 `out` 时，返回 `None`。

### 2.4 注册类配置

| 字段 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `name` | `str` | 否 | 注册名称；省略时使用类名 |
| `core` | `al.CORE` | 是 | 操作使用的核类型 |
| `pipe` | `al.PIPE` / `tuple` / `list` | 是 | 单个 `PIPE` 表示 CustomOp；两个 `PIPE` 依次表示 CustomMacro 的输入、输出流水线 |
| `mode` | `al.MODE` | 条件必需 | `core` 不是 `al.CORE.CUBE` 时必须配置 |
| `symbol` | `str` | 是（用户注册） | 设备侧实现的函数符号；用户注册的 CustomOp 或 CustomMacro 必须配置 |
| `bitcode` | `str` / `path` | 是（用户注册） | 包含设备侧实现的 bitcode 文件路径；用户注册的 CustomOp 或 CustomMacro 必须配置，且文件必须存在 |
| `iterator_types` | `list[al.IteratorType]` | 否 | 按逻辑循环顺序描述各迭代维度的作用 |
| `sync_event_slots` | `list[al.SyncEventSlot]` / `tuple[al.SyncEventSlot, ...]` | 否 | CustomMacro 的同步槽列表；普通 CustomOp 不支持 |

注册类可以定义 `__init__`，用于校验 `al.custom` 的调用参数。定义后，构造函数签名必须接收调用时传入的全部参数；如果调用使用 `out`，构造函数应包含带默认值的 `out` 参数，例如 `out=None`。省略 `__init__` 时，注册类只保存静态配置。

### 2.5 CustomOp 与 CustomMacro

| 形式 | pipe 配置 | 生成的关键 IR | sync_event_slots |
| --- | --- | --- | --- |
| CustomOp | 一个 `al.PIPE` | `hivm.hir.custom` | 不支持 |
| CustomMacro | 恰好两个 `al.PIPE` 组成的 `tuple` 或 `list` | `hivm.hir.custom_macro` | 支持 |

CustomMacro 的第一个 `PIPE` 生成 `hivm.pipe_in`，第二个 `PIPE` 生成 `hivm.pipe_out`。

### 2.6 CORE

`al.CORE` 用于配置操作涉及的核类型，生成 `hivm.tcore_type` 属性。

| 枚举值 | 说明 |
| --- | --- |
| `al.CORE.VECTOR` | Vector 核 |
| `al.CORE.CUBE` | Cube 核 |
| `al.CORE.CUBE_OR_VECTOR` | 可涉及 Cube 或 Vector 核的类型标识 |
| `al.CORE.CUBE_AND_VECTOR` | 同时涉及 Cube 和 Vector 核的类型标识 |

### 2.7 MODE

`al.MODE` 用于配置非纯 Cube 操作的执行模式，生成 `hivm.vf_mode` 属性。

| 枚举值 | 说明 |
| --- | --- |
| `al.MODE.SIMD` | SIMD（单指令多数据）模式 |
| `al.MODE.SIMT` | SIMT（单指令多线程）模式 |
| `al.MODE.MIX` | 混合模式 |

### 2.8 PIPE

`al.PIPE` 用于标识操作或同步涉及的流水线。它只提供流水线信息。

| 枚举值 | 典型用途 |
| --- | --- |
| `al.PIPE.PIPE_S` | 标量流水线 |
| `al.PIPE.PIPE_V` | 向量计算流水线 |
| `al.PIPE.PIPE_M` | 矩阵计算流水线 |
| `al.PIPE.PIPE_MTE1` | 片上数据搬运流水线 |
| `al.PIPE.PIPE_MTE2` | 数据搬入流水线 |
| `al.PIPE.PIPE_MTE3` | 数据搬出流水线 |
| `al.PIPE.PIPE_ALL` | 所有流水线 |
| `al.PIPE.PIPE_FIX` | Cube 计算结果搬出流水线 |

表中列出的是典型用途，实际配置必须与设备侧实现保持一致。

### 2.9 IteratorType

`iterator_types` 是可选配置，用于说明设备侧实现中各逻辑循环维度的作用。列表顺序应与逻辑循环顺序一致，其长度不一定等于输出张量的维数。省略该字段时，当前前端不会生成 `iterator_types` 属性，也不会自动补成 `Parallel`。

| 枚举值 | 含义 |
| --- | --- |
| `al.IteratorType.Parallel` | 不同迭代之间可以独立执行 |
| `al.IteratorType.Broadcast` | 广播维度 |
| `al.IteratorType.Transpose` | 转置维度 |
| `al.IteratorType.Reduction` | 归约维度 |
| `al.IteratorType.Interleave` | 交织维度 |
| `al.IteratorType.Deinterleave` | 解交织维度 |
| `al.IteratorType.Inverse` | 逆序维度 |
| `al.IteratorType.Pad` | 填充维度 |
| `al.IteratorType.Concat` | 拼接维度 |
| `al.IteratorType.Gather` | Gather 索引维度 |
| `al.IteratorType.Cumulative` | 累积维度 |
| `al.IteratorType.Opaque` | 不按上述通用类型解释的维度 |

### 2.10 SyncEventSlot

当两条流水线之间存在先后依赖时，设置信号的一侧通知任务已经完成，等待信号的一侧等待同一个信号，避免后续任务过早继续执行。`al.SyncEventSlot` 用于描述 CustomMacro 设备侧实现中的一组此类同步关系，只能用于两条流水线的 CustomMacro。

| 参数名 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `set_pipe` | `al.PIPE` | 条件必需 | 设置信号的一侧流水线；使用 `WAIT` 或 `SET` 时必须与 `wait_pipe` 一起配置 |
| `wait_pipe` | `al.PIPE` | 条件必需 | 等待信号的一侧流水线；使用 `WAIT` 或 `SET` 时必须与 `set_pipe` 一起配置 |
| `sync` | `al.SYNC_HINT` | 否 | 说明设备侧实现承担的同步动作；省略时默认使用 `WAIT`，建议显式配置 |
| `event` | `al.EVENT_ID` | 否 | 固定该同步槽使用的 event ID；省略时不指定固定 event ID |

### 2.11 SYNC_HINT

`al.SYNC_HINT` 告诉编译器设备侧实现已经承担哪一侧的同步动作。

| 枚举值 | 设备侧实现中的动作 | 编译器在 CustomMacro 边界的处理 |
| --- | --- | --- |
| `al.SYNC_HINT.WAIT` | 内部执行等待 | 在 CustomMacro 前复用或补充匹配的设置信号 |
| `al.SYNC_HINT.SET` | 内部设置信号 | 在 CustomMacro 后补充匹配的等待 |
| `al.SYNC_HINT.INTERNAL` | 内部只使用事件号 | 不在边界补充设置或等待 |

### 2.12 EVENT_ID

`al.EVENT_ID` 提供 `EVENT_ID0` 至 `EVENT_ID7` 八个枚举值，用于固定 `SyncEventSlot.event` 的事件编号。

省略 `event` 不等同于显式指定 `al.EVENT_ID.EVENT_ID0`。具体 event ID 与流水线组合应和设备侧同步实现保持一致。

## 3. 约束说明

- `@al.register_custom_op` 必须装饰类，注册名称不能重复。
- `core` 和 `pipe` 必须配置，且分别使用 `al.CORE` 和 `al.PIPE` 枚举值。
- 普通 CustomOp 的 `pipe` 必须是一个 `al.PIPE`；CustomMacro 的 `pipe` 必须是长度为 2 的 `tuple` 或 `list`，且两个元素都必须是 `al.PIPE`。
- `core` 不是 `al.CORE.CUBE` 时必须配置 `mode`，并使用 `al.MODE` 枚举值。
- 用户注册的 CustomOp 或 CustomMacro 必须提供 `symbol` 和 `bitcode`；`bitcode` 路径必须存在，且应包含与 `symbol` 对应的设备侧实现。
- `al.custom` 应在 `@triton.jit` 函数中调用，`name` 必须与已经注册的名称一致。
- `iterator_types` 中的元素应为 `al.IteratorType`，顺序和含义必须与设备侧实现的逻辑循环一致。
- `sync_event_slots` 只支持 CustomMacro。使用 `al.SYNC_HINT.WAIT` 或 `al.SYNC_HINT.SET` 时，必须同时提供 `set_pipe` 和 `wait_pipe`；同步声明必须与设备侧实现一致。
- `al.EVENT_ID` 只用于 `al.SyncEventSlot.event`。

## 4. 用例示例

### 4.1 普通 CustomOp

以下是注册配置和 Kernel 调用片段。

```python
import triton.language.extra.cann.extension as al

BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class my_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_custom_func"
    bitcode = BITCODE_PATH
    iterator_types = [al.IteratorType.Parallel]


# 以下调用位于 @triton.jit 函数中，x 和 y 是已经生成的张量。
result = al.custom("my_custom_op", x, out=y)
```

注册类没有显式配置 `name`，因此使用类名 `my_custom_op` 作为注册名称。单个 `PIPE_V` 表示这是普通 CustomOp。

### 4.2 带同步槽的 CustomMacro

CustomMacro 与普通 CustomOp 使用相同的装饰器和调用接口，区别在于 `pipe` 包含两个流水线。

以下是注册配置和 Kernel 调用片段。

```python
import triton.language.extra.cann.extension as al

BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class my_custom_macro_sync_op:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "my_custom_macro_sync_func"
    bitcode = BITCODE_PATH
    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_MTE1,
            sync=al.SYNC_HINT.WAIT,
            event=al.EVENT_ID.EVENT_ID1,
        )
    ]


# 以下调用位于 @triton.jit 函数中，x 和 y 是已经生成的张量。
result = al.custom("my_custom_macro_sync_op", x, out=y)
```

`pipe` 中的 `PIPE_MTE2` 和 `PIPE_V` 分别表示输入、输出流水线。同步槽描述设备侧实现内部实际使用的同步流水线，不要求和 CustomMacro 的输入、输出流水线相同；其中 `WAIT` 表示设备侧实现内部执行等待，并固定使用 `EVENT_ID1`。

## 5. 编译输出结果

下面是与配置对应的关键 IR 字段摘录。

普通 CustomOp 示例生成的关键操作和属性为：

```mlir
hivm.hir.custom
hivm.tcore_type = #hivm.tcore_type<VECTOR>
hivm.vf_mode = #hivm.vf_mode<SIMT>
hivm.pipe = #hivm.pipe<PIPE_V>
iterator_types = [#hivm.iterator_type<parallel>]
```

CustomMacro 示例生成的关键操作和属性为：

```mlir
hivm.hir.custom_macro
hivm.tcore_type = #hivm.tcore_type<VECTOR>
hivm.vf_mode = #hivm.vf_mode<SIMD>
hivm.pipe_in = #hivm.pipe<PIPE_MTE2>
hivm.pipe_out = #hivm.pipe<PIPE_V>
sync_event_slots = [
  #hivm.sync_event_slot<
    #hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>
  >
]
```

CustomMacro 使用 `hivm.pipe_in` 和 `hivm.pipe_out`。
