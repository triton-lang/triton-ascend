# al.sync_block_set 接口文档

## 1. 硬件背景

昇腾 AIC+AIV **分离模式**下，Cube 核与 Vector 核并行工作。当两类核通过共享内存（GM / L1 / UB）传递数据时，需要在发送方完成写入、接收方开始读取之间插入同步。`al.sync_block_set` 与 [`al.sync_block_wait`](sync_block_wait.md) 配合构成一对**点对点（producer-consumer）信号量同步**原语：

- 每个 `event_id` 对应一个硬件计数器，初值为 0；
- 执行 `sync_block_set` 时该计数器加 1（表示数据已就绪）；
- 执行 `sync_block_wait` 时若计数器为 0 则阻塞，直到计数器大于 0 后将其减 1，然后继续执行后续指令。

`al.sync_block_set` 必须由**发送方（producer）核**在对应 core type 的 `al.scope` 内调用，声明"我这一侧的写入已完成，可以安全地被对端读取"。

## 2. 接口说明

```python
class PIPE(enum.Enum):
    PIPE_S     = ...   # 标量流水（如 GetValue）
    PIPE_V     = ...   # 矢量计算流水 / L0C→UB 搬运
    PIPE_M     = ...   # 矩阵计算流水
    PIPE_MTE1  = ...   # L1→L0A、L1→L0B 搬运流水
    PIPE_MTE2  = ...   # GM→L1、GM→L0A/L0B、GM→UB 搬运流水
    PIPE_MTE3  = ...   # UB→GM、UB→L1 搬运流水
    PIPE_ALL   = ...   # 所有流水
    PIPE_FIX   = ...   # L0C→GM、L0C→L1 搬运流水


def sync_block_set(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: PIPE | None = None,
    receiver_pipe: PIPE | None = None,
    _semantic=None,
) -> None:
```

### 2.1 入参

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `sender` | `str` | 是 | — | 发送端核心类型，仅支持 `"cube"` 或 `"vector"` |
| `receiver` | `str` | 是 | — | 接收端核心类型，仅支持 `"cube"` 或 `"vector"`，且必须与 `sender` 不同 |
| `event_id` | `int` | 是 | — | 同步标记 ID，取值范围 `[0, 15]`，每个 ID 对应一个独立计数器 |
| `sender_pipe` | `al.PIPE` | 否 | 自动选择 | 发送端流水线类型。`sender="cube"` 默认 `PIPE_FIX`；`sender="vector"` 默认 `PIPE_MTE3` |
| `receiver_pipe` | `al.PIPE` | 否 | `PIPE_MTE2` | 接收端流水线类型。默认 `PIPE_MTE2`（GM→片上搬运流水） |
| `_semantic` | - | 内部 | — | JIT 编译器自动传参，用户不要手动传 |

### 2.2 PIPE 枚举说明

| 枚举值 | 含义 |
|--------|------|
| `PIPE_S` | 标量流水线（如 Tensor GetValue 函数） |
| `PIPE_V` | 矢量计算流水 / L0C→UB 数据搬运流水 |
| `PIPE_M` | 矩阵计算流水 |
| `PIPE_MTE1` | L1→L0A、L1→L0B 数据搬运流水 |
| `PIPE_MTE2` | GM→L1、GM→L0A、GM→L0B、GM→UB 数据搬运流水 |
| `PIPE_MTE3` | UB→GM、UB→L1 数据搬运流水 |
| `PIPE_ALL` | 所有流水线 |
| `PIPE_FIX` | L0C→GM、L0C→L1 数据搬运流水 |

### 2.3 返回值

无返回值。

## 3. 约束说明

- `sender` 和 `receiver` 必须为 `"cube"` / `"vector"`，且**二者不能相同**（同一 core type 内的核间同步请使用 `al.sync_block_all`）。
- `event_id` 必须在 `[0, 15]` 范围内，否则触发断言失败。
- `sync_block_set` 必须在与 `sender` 核心类型一致的 `al.scope(core_mode=sender)` 内调用；否则会因流水/核心不匹配导致运行时错误。
- 必须与同 `(sender, receiver, event_id)` 的 `sync_block_wait` 配对使用，否则接收端会永久阻塞。
- `sender_pipe` 和 `receiver_pipe` 必须同时显式指定或同时省略（不能只指定一个），否则会触发 `TypeError`。
- 不同 `event_id` 相互独立，可在同一 kernel 中使用多个 ID 做多路生产者-消费者同步。

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def cube_to_vector_sync_kernel():
    """Cube 核完成 L0C→GM 的 fix 搬运后，通知 Vector 核通过 MTE3 读取。"""
    with al.scope(core_mode="cube"):
        # ... Cube 完成计算，把结果从 L0C 写出 ...
        al.sync_block_set(
            sender="cube",
            receiver="vector",
            event_id=0,
            sender_pipe=al.PIPE.PIPE_MTE1,
            receiver_pipe=al.PIPE.PIPE_MTE3,
        )

    with al.scope(core_mode="vector"):
        al.sync_block_wait(
            sender="cube",
            receiver="vector",
            event_id=0,
            sender_pipe=al.PIPE.PIPE_MTE1,
            receiver_pipe=al.PIPE.PIPE_MTE3,
        )
        # Vector 核在这之后安全地读取 Cube 写入的数据
        ...


@triton.jit
def vector_to_cube_sync_kernel():
    """Vector 核写入数据后通知 Cube 核读取（V/FIX 流水配对）。"""
    with al.scope(core_mode="vector"):
        # ... Vector 计算/搬运完成 ...
        al.sync_block_set(
            "vector", "cube", event_id=1,
            sender_pipe=al.PIPE.PIPE_V, receiver_pipe=al.PIPE.PIPE_FIX,
        )

    with al.scope(core_mode="cube"):
        al.sync_block_wait(
            "vector", "cube", event_id=1,
            sender_pipe=al.PIPE.PIPE_V, receiver_pipe=al.PIPE.PIPE_FIX,
        )
        # Cube 核现在可以消费 Vector 写入的数据
        ...


@triton.jit
def multi_event_id_kernel():
    """同一条 Cube→Vector 路径上用两个 event_id 同步两路不同流水的数据。"""
    with al.scope(core_mode="cube"):
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        al.sync_block_set("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        # ... 消费 MTE1 那一路数据 ...
        al.sync_block_wait("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        # ... 消费 MTE2 那一路数据 ...
```

## 5. 编译输出结果

以 `cube → vector, event_id=0, sender_pipe=PIPE_MTE1, receiver_pipe=PIPE_MTE3` 为例：

```mlir
module {
  tt.func public @cube_to_vector_sync_kernel() attributes {noinline = false} {
    // Cube scope：producer 侧 set flag
    scope.scope : () -> () {
      %c0 = arith.constant 0 : i32
      %0  = arith.extui %c0 : i32 to i64
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}

    // Vector scope：consumer 侧 wait flag（见 sync_block_wait.md）
    scope.scope : () -> () {
      %c0 = arith.constant 0 : i32
      %0  = arith.extui %c0 : i32 to i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}

    tt.return
  }
}
```

### 输出要点说明

- `al.sync_block_set` 降低为 `hivm.hir.sync_block_set[<SENDER>, <SENDER_PIPE>, <RECEIVER_PIPE>] flag = <event_id>` 指令。
- `<SENDER>` 枚举与 `core_mode` 一致：Cube 核内调用时为 `<CUBE>`，Vector 核内调用时为 `<VECTOR>`（指令本身不关心 receiver 是谁，但参数会用于后端流水 ordering）。
- `event_id` 在 IR 中以 `i64` 类型的 SSA 值传递（通常由 `arith.constant` + `arith.extui` 产生），而非常量属性，便于循环中动态 ID 的场景。
- 该指令位于与 `sender` 对应的 `scope.scope` region 内，携带 `hivm.tcore_type` 属性。

## 6. 相关接口

- [`al.sync_block_wait`](sync_block_wait.md)：与本接口配对使用的等待端，必须使用相同 `(sender, receiver, event_id, sender_pipe, receiver_pipe)`
- [`al.sync_block_all`](sync_block_all.md)：广播式屏障同步（用于同构核之间或全片同步）
- [`al.scope`](scope.md)：声明代码块运行在 Cube/Vector 核心
