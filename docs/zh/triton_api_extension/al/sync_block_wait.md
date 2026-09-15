# al.sync_block_wait 接口文档

## 1. 硬件背景

昇腾 AIC+AIV **分离模式**下，Cube 核与 Vector 核并行工作。当两类核通过共享内存（GM / L1 / UB）传递数据时，需要在发送方完成写入、接收方开始读取之间插入同步。`al.sync_block_wait` 与 [`al.sync_block_set`](sync_block_set.md) 配合构成一对**点对点（producer-consumer）信号量同步**原语：

- 每个 `event_id` 对应一个硬件计数器，初值为 0；
- 发送方执行 `sync_block_set` 时该计数器加 1（表示数据已就绪）；
- 接收方执行 `sync_block_wait` 时若计数器为 0 则**阻塞等待**，直到计数器大于 0 后将其减 1，然后继续执行后续指令。

`al.sync_block_wait` 必须由**接收方（consumer）核**在对应 core type 的 `al.scope` 内调用，表示"我已准备好消费对端写入的数据"，实现生产者-消费者语义。

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


def sync_block_wait(
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
| `sender` | `str` | 是 | — | 发送端核心类型（等待谁的信号），仅支持 `"cube"` 或 `"vector"` |
| `receiver` | `str` | 是 | — | 接收端核心类型（谁在等），仅支持 `"cube"` 或 `"vector"`，且必须与 `sender` 不同 |
| `event_id` | `int` | 是 | — | 同步标记 ID，取值范围 `[0, 15]`，必须与配对的 `sync_block_set` 使用相同 ID |
| `sender_pipe` | `al.PIPE` | 否 | 自动选择 | 发送端流水线类型，需与配对 `sync_block_set` 保持一致。`sender="cube"` 默认 `PIPE_FIX`；`sender="vector"` 默认 `PIPE_MTE3` |
| `receiver_pipe` | `al.PIPE` | 否 | `PIPE_MTE2` | 接收端流水线类型。默认 `PIPE_MTE2` |
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

无返回值（阻塞直到对端 set 信号）。

## 3. 约束说明

- `sender` 和 `receiver` 必须为 `"cube"` / `"vector"`，且**二者不能相同**。
- `event_id` 必须在 `[0, 15]` 范围内。
- `sync_block_wait` 必须在与 `receiver` 核心类型一致的 `al.scope(core_mode=receiver)` 内调用。
- 必须与同 `(sender, receiver, event_id, sender_pipe, receiver_pipe)` 的 `sync_block_set` 配对使用；如果对端没有 set，wait 会**永久阻塞**导致硬件死锁。
- `sender_pipe` 和 `receiver_pipe` 必须同时显式指定或同时省略，否则触发 `TypeError`。
- 不同 `event_id` 相互独立，可在同一 kernel 中多路并行等待。
- 在同一个 receiver scope 内多次 wait 同一 event_id 会对应多次 set，相当于消费多个信号量（计数型信号量语义）。

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def cube_to_vector_sync_kernel(in_ptr, out_ptr, N: tl.constexpr):
    """典型的 Cube 写 → Vector 读流程。"""
    with al.scope(core_mode="cube"):
        # ... Cube 完成矩阵乘法，结果写入 GM/UB ...
        al.sync_block_set(
            "cube", "vector", event_id=0,
            sender_pipe=al.PIPE.PIPE_MTE1, receiver_pipe=al.PIPE.PIPE_MTE3,
        )

    with al.scope(core_mode="vector"):
        # Vector 端必须先 wait，确保 Cube 写入完成后才读
        al.sync_block_wait(
            "cube", "vector", event_id=0,
            sender_pipe=al.PIPE.PIPE_MTE1, receiver_pipe=al.PIPE.PIPE_MTE3,
        )
        offs = tl.arange(0, N)
        x = tl.load(in_ptr + offs)
        tl.store(out_ptr + offs, x + 1.0)


@triton.jit
def multi_event_id_kernel():
    """多路同步：Vector 端分别等待两路不同流水的数据。"""
    with al.scope(core_mode="cube"):
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        al.sync_block_set("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        # ... 消费第 0 路数据（MTE1 流水写出） ...
        al.sync_block_wait("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        # ... 消费第 1 路数据（MTE2 流水写出） ...
```

## 5. 编译输出结果

以 `vector → cube, event_id=1, sender_pipe=PIPE_V, receiver_pipe=PIPE_FIX` 为例（Vector 发，Cube 等）：

```mlir
module {
  tt.func public @vector_to_cube_sync_kernel() attributes {noinline = false} {
    // Vector scope：producer set
    scope.scope : () -> () {
      %c1 = arith.constant 1 : i32
      %0  = arith.extui %c1 : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}

    // Cube scope：consumer wait（本接口）
    scope.scope : () -> () {
      %c1 = arith.constant 1 : i32
      %0  = arith.extui %c1 : i32 to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = %0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}

    tt.return
  }
}
```

### 输出要点说明

- `al.sync_block_wait` 降低为 `hivm.hir.sync_block_wait[<RECEIVER>, <SENDER_PIPE>, <RECEIVER_PIPE>] flag = <event_id>` 指令。
- 方括号内的第一个枚举为**接收方**核心（当前指令所在 core），与 wait 的语义对应；指令携带的 `<SENDER_PIPE>` / `<RECEIVER_PIPE>` 用于硬件流水排序，确保 set 侧对应流水的写入对 wait 侧对应流水可见。
- `event_id` 在 IR 中以 `i64` 类型 SSA 值传递（通过 `arith.constant` + `arith.extui` 产生），支持动态 ID。
- wait 指令位于与 `receiver` 对应的 `scope.scope` region 内，携带 `hivm.tcore_type` 属性。
- 如果 wait 指令执行时 event 计数器为 0，硬件会在该指令处暂停该核，直到对应 set 到达后才继续向下执行。

## 6. 相关接口

- [`al.sync_block_set`](sync_block_set.md)：与本接口配对使用的信号发送端，必须使用相同 `(sender, receiver, event_id, sender_pipe, receiver_pipe)`
- [`al.sync_block_all`](sync_block_all.md)：广播式屏障同步，适合同构核之间一次性对齐
- [`al.scope`](scope.md)：声明代码块运行在 Cube/Vector 核心
