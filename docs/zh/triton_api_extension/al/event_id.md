# al.EVENT_ID

## 1. 概述

`al.EVENT_ID` 是定制宏算子同步槽位使用的 event 枚举，用于将 `set_flag` 与 `wait_flag`
绑定到同一个同步事件。它不是可直接调用的函数。

`al.EVENT_ID` 只用于 `al.SyncEventSlot.event`。它与 `al.sync_block_set`、
`al.sync_block_wait` 和 `al.sync_block_all` 接收的 `[0, 15]` 普通整数 `event_id`
属于不同的接口，不应混用。

## 2. 枚举值

```python
class EVENT_ID(enum.Enum):
    EVENT_ID0
    EVENT_ID1
    EVENT_ID2
    EVENT_ID3
    EVENT_ID4
    EVENT_ID5
    EVENT_ID6
    EVENT_ID7
```

这些枚举值通过 `al.EVENT_ID` 访问，例如 `al.EVENT_ID.EVENT_ID1`。

| 枚举值 | 编码 |
| --- | ---: |
| `EVENT_ID0` | 0 |
| `EVENT_ID1` | 1 |
| `EVENT_ID2` | 2 |
| `EVENT_ID3` | 3 |
| `EVENT_ID4` | 4 |
| `EVENT_ID5` | 5 |
| `EVENT_ID6` | 6 |
| `EVENT_ID7` | 7 |

一组同步由有方向的三元组 `(set_pipe, wait_pipe, event_id)` 标识。只有三项均匹配时，
set 和 wait 才属于同一同步事件。

## 3. 使用方法

显式固定 event：

```python
sync_event_slots = [
    al.SyncEventSlot(
        set_pipe=al.PIPE.PIPE_MTE2,
        wait_pipe=al.PIPE.PIPE_MTE1,
        sync=al.SYNC_HINT.WAIT,
        event=al.EVENT_ID.EVENT_ID1,
    )
]
```

生成的关键 IR 属性为：

```mlir
#hivm.sync_event_slot<
  #hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>
>
```

也可以省略 `event`，由编译器解析：

```python
al.SyncEventSlot(
    set_pipe=al.PIPE.PIPE_MTE2,
    wait_pipe=al.PIPE.PIPE_MTE1,
    sync=al.SYNC_HINT.WAIT,
)
```

当前实现对未固定 event 的处理如下：

- `WAIT` 和 `SET` 优先复用相同 pipe pair 已有 `set_flag` 使用的 event；没有可复用事件时回退到
  `EVENT_ID0`。
- `INTERNAL` 从该 pipe pair 的可用 event 中分配一个尚未占用的 ID。

这些是当前 GraphSyncSolver 的解析规则，不应依赖某个自动分配结果长期保持不变。
需要稳定 ABI 时应显式指定 event，并让 bitcode 使用编译器传入的值。

## 4. 设备侧参数

每个同步槽位会向设备侧 `symbol` 调用追加一个 `i64` event 参数。`i64` 是 ABI 参数类型，
不代表存在 64 个 event。设备实现必须使用该参数，不能假定所有调用都使用 event 0。

## 5. 约束说明

- 枚举定义了 0 到 7，但某个 pipe pair 的实际可用 event 数量可能更少；编译器和设备库可能预留
  部分 ID。因此不能假定 `EVENT_ID7` 对所有 pipe pair 都可用。
- 显式 event 会由编译器保留和校验。存在冲突或超出对应 pipe pair 可用范围时，编译会失败。
- 数字相同但 `set_pipe` 或 `wait_pipe` 不同的 set/wait 不是同一组同步。
- 如果不需要固定 ID，优先省略 `event`，由编译器解析；如果设备 ABI 要求固定 ID，必须确保声明、
  bitcode 和所有相关同步点一致。

## 6. 验证范围

仓库的定制宏算子编译用例覆盖了固定 `EVENT_ID1` 的 IR 生成；GraphSyncSolver 的 MLIR 用例覆盖了
固定、复用、回退和 `INTERNAL` 分配路径。这些测试验证编译器行为，不代表所有 event 和 pipe pair
都已在真实 NPU 上逐项验证。
