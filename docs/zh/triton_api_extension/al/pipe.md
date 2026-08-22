# al.PIPE 接口文档

## 1. 硬件背景

Ascend NPU 使用不同的硬件流水线完成标量计算、向量计算、矩阵计算和数据搬运等操作。
`al.PIPE` 是流水线枚举，用于标识操作或同步所涉及的流水线。
它只提供流水线信息，本身不执行计算、数据搬运或同步。

## 2. 接口说明

### 2.1 枚举定义

```python
class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX
```

这些枚举值通过 `al.PIPE` 访问，例如 `al.PIPE.PIPE_V`。

### 2.2 枚举值说明

| 枚举值 | 典型用途 |
| --- | --- |
| `PIPE_S` | 标量流水线 |
| `PIPE_V` | 向量计算流水线 |
| `PIPE_M` | 矩阵计算流水线 |
| `PIPE_MTE1` | 片上数据搬运流水线 |
| `PIPE_MTE2` | 数据搬入流水线 |
| `PIPE_MTE3` | 数据搬出流水线 |
| `PIPE_ALL` | 所有流水线 |
| `PIPE_FIX` | Cube 计算结果搬出流水线 |

表中列出的是各流水线的典型用途。具体操作使用哪条流水线，应与设备侧实现保持一致。

### 2.3 使用方式

`al.PIPE` 可以用于以下位置：

- `CustomOp` 的 `pipe` 配置使用一个 `al.PIPE`。
- `CustomMacro` 的 `pipe` 配置使用两个 `al.PIPE`，第一个表示输入流水线，第二个表示输出流水线。
- [`al.sync_block_set`](./sync_block_set.md) 和
  [`al.sync_block_wait`](./sync_block_wait.md) 使用 `al.PIPE` 指定发送端和接收端流水线。

这里的流水线配置只描述操作使用的流水线，不会自动执行流水线切换或数据搬运。

## 3. 约束说明

`CustomOp` 是通过扩展接口接入 Triton 的设备侧操作，使用一个 `PIPE` 标识其流水线。
`CustomMacro` 是同时描述输入、输出流水线的设备侧操作，使用两个 `PIPE` 分别标识输入和输出流水线。

- `CustomOp` 的 `pipe` 必须是一个 `al.PIPE` 枚举值。
- `CustomMacro` 的 `pipe` 必须是长度为 2 的 `tuple` 或 `list`，且两个元素都必须是
  `al.PIPE` 枚举值。
- 流水线声明必须与设备侧实现保持一致。

## 4. 用例示例

### 4.1 注册配置中的 PIPE

以下代码是 `@al.register_custom_op` 注册类中的配置片段。为突出 `pipe`，省略了其他必需的
注册字段，因此不是完整可运行用例：

```python
@al.register_custom_op
class vector_custom_op:
    # CustomOp 使用一个 PIPE
    pipe = al.PIPE.PIPE_V


@al.register_custom_op
class vector_custom_macro:
    # CustomMacro 依次指定输入、输出 PIPE
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
```

### 4.2 同步接口中的 PIPE

以下代码是 Triton Kernel 内的调用片段。`sync_block_set` 在发送端作用域中设置同步信号，
`sync_block_wait` 在接收端作用域中等待同一个信号：

```python
with al.scope(core_mode="cube"):
    al.sync_block_set(
        "cube",
        "vector",
        0,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )

with al.scope(core_mode="vector"):
    al.sync_block_wait(
        "cube",
        "vector",
        0,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
```

## 5. 编译输出结果

`CustomOp` 配置生成的关键 IR 属性为：

```mlir
hivm.pipe = #hivm.pipe<PIPE_V>
```

`CustomMacro` 配置生成的关键 IR 属性为：

```mlir
hivm.pipe_in = #hivm.pipe<PIPE_MTE2>
hivm.pipe_out = #hivm.pipe<PIPE_V>
```
