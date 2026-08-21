# al.PIPE

## 1. 概述

`al.PIPE` 是 Ascend 扩展中的流水线枚举，用于向编译器描述操作所属的硬件流水线。
它本身不会执行计算、数据搬运或流水线切换。

`al.PIPE` 主要用于以下场景：

- 定制算子的 `pipe` 属性；
- 定制宏算子的输入、输出流水线；
- [`al.sync_block_set`](./sync_block_set.md) 和
  [`al.sync_block_wait`](./sync_block_wait.md) 的发送端、接收端流水线。

## 2. 枚举值

```python
class PIPE(enum.Enum):
    PIPE_S
    PIPE_V
    PIPE_M
    PIPE_MTE1
    PIPE_MTE2
    PIPE_MTE3
    PIPE_ALL
    PIPE_FIX
```

这些枚举值通过 `al.PIPE` 访问，例如 `al.PIPE.PIPE_V`。

| 枚举值 | 典型用途 |
| --- | --- |
| `PIPE_S` | 标量与控制类操作流水线 |
| `PIPE_V` | 向量计算流水线 |
| `PIPE_M` | 矩阵计算流水线 |
| `PIPE_MTE1` | 片上存储之间的数据搬运，例如 L1 到 L0A/L0B |
| `PIPE_MTE2` | 从 GM 向片上存储搬入数据 |
| `PIPE_MTE3` | 从 UB 等片上存储向 GM 或 L1 搬出数据 |
| `PIPE_ALL` | 表示全部流水线，主要用于屏障或同步范围 |
| `PIPE_FIX` | 搬出 Cube 计算结果，例如从 L0C 到 GM 或 L1 |

表中的数据路径是帮助理解流水线的典型用途，不是所有芯片和所有指令的完整路由表。
具体操作使用哪条流水线，应以该操作的后端定义和目标芯片能力为准。

## 3. 定制算子中的用法

普通定制算子使用一个 `al.PIPE`：

```python
@al.register_custom_op
class vector_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    symbol = "vector_custom_impl"
    bitcode = "/path/to/vector_custom_impl.bc"
```

该属性会表示为单流水线属性：

```mlir
hivm.pipe = #hivm.pipe<PIPE_V>
```

定制宏算子使用恰好两个 `al.PIPE`，分别描述输入和输出边界：

```python
@al.register_custom_op
class vector_custom_macro:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "vector_custom_macro_impl"
    bitcode = "/path/to/vector_custom_macro_impl.bc"
```

对应的关键 IR 属性为：

```mlir
hivm.pipe_in = #hivm.pipe<PIPE_MTE2>
hivm.pipe_out = #hivm.pipe<PIPE_V>
```

双流水线属性只描述宏算子的读写边界，不会自动在两条流水线之间插入同步。
真实计算和宏内部同步仍由 `symbol` 对应的设备实现提供；需要暴露宏内部同步需求时，使用
`sync_event_slots`、[`al.SYNC_HINT`](./sync_hint.md) 和
[`al.EVENT_ID`](./event_id.md)。

## 4. 约束说明

- 普通定制算子的 `pipe` 必须是一个 `al.PIPE` 枚举值。
- 定制宏算子的 `pipe` 必须是长度为 2 的 `tuple` 或 `list`，且两个元素都必须是
  `al.PIPE` 枚举值。
- `sync_event_slots` 只适用于具有双流水线属性的定制宏算子。
- 流水线声明必须与设备侧实现的真实读写行为一致。编译器不能完整验证 bitcode 内部行为，
  错误声明可能导致错误的依赖分析或同步插入。

## 5. 验证范围

仓库中的 `test_custom.py` 和 `test_custom_macro.py` 验证了单流水线、双流水线属性可以正确生成
HIVM IR。这些用例使用占位 bitcode，只验证编译期属性，不验证真实定制算子在 NPU 上的执行结果。
