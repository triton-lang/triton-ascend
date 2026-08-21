# al.SYNC_HINT

## 1. 概述

`al.SYNC_HINT` 用于描述定制宏算子的同步动作位于宏内部还是宏外部。它不是可直接调用的同步函数，
而是 `al.SyncEventSlot` 的 `sync` 字段。

```python
al.SyncEventSlot(
    set_pipe=None,
    wait_pipe=None,
    sync=None,
    event=None,
)
```

一个同步槽位会把宏内部不可见的同步需求告诉编译器。编译器据此解析 event ID、在宏边界插入必要的
`set_flag` 或 `wait_flag`，并把 event ID 传给设备侧 `symbol` 实现。

## 2. 枚举值

```python
class SYNC_HINT(enum.Enum):
    WAIT
    SET
    INTERNAL
```

这些枚举值通过 `al.SYNC_HINT` 访问，例如 `al.SYNC_HINT.WAIT`。

三种取值都从“宏内部已经做了什么”的角度命名：

| 枚举值 | 宏内部的同步动作 | 编译器处理宏边界 |
| --- | --- | --- |
| `WAIT` | 宏内部执行 `wait_flag` | 在宏前复用或补充匹配的 `set_flag` |
| `SET` | 宏内部执行 `set_flag` | 在宏后补充匹配的 `wait_flag` |
| `INTERNAL` | 设备实现自行使用 event ID | 不在宏边界补 set/wait，只解析并传入 event ID |

`INTERNAL` 只表示编译器不补宏边界同步，不保证 bitcode 内部一定包含一组 set/wait；
设备侧实现仍需正确使用传入的 event ID。

## 3. 使用方法

以下配置声明：设备侧实现会在 `PIPE_MTE1` 上等待，编译器需要在宏前准备一条从
`PIPE_MTE2` 发出的匹配信号，并固定使用 `EVENT_ID1`。

```python
@al.register_custom_op
class custom_macro_with_wait:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "custom_macro_with_wait_impl"
    bitcode = "/path/to/custom_macro_with_wait_impl.bc"
    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_MTE1,
            sync=al.SYNC_HINT.WAIT,
            event=al.EVENT_ID.EVENT_ID1,
        )
    ]
```

对应的前端 IR 属性包含：

```mlir
#hivm.sync_event_slot<
  #hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>
>
```

其他两种写法示例：

```python
# 宏内部 set，编译器在宏后补 wait。
al.SyncEventSlot(
    set_pipe=al.PIPE.PIPE_V,
    wait_pipe=al.PIPE.PIPE_MTE3,
    sync=al.SYNC_HINT.SET,
)

# 宏内部自行处理同步，编译器只提供 event ID。
al.SyncEventSlot(sync=al.SYNC_HINT.INTERNAL)
```

## 4. event 参数传递

每个 `SyncEventSlot` 最终对应设备侧 `symbol` 调用末尾的一个 `i64` 同步参数。
参数顺序为普通输入、输出、临时缓冲区，最后是各槽位的同步参数。bitcode 中的函数签名必须接收这些
参数，并在对应的内部 set/wait 中使用它们。

## 5. 约束说明

- `sync_event_slots` 只适用于 `pipe` 为两个 `al.PIPE` 的定制宏算子。
- `WAIT` 和 `SET` 必须指定 `set_pipe` 与 `wait_pipe`；`INTERNAL` 可以省略这两个字段，
  此时使用宏算子的 `pipe_in` 和 `pipe_out` 解析 event。
- Python 接口中省略 `SyncEventSlot.sync` 时默认为 `WAIT`。为避免与底层 IR 的默认语义混淆，
  建议始终显式填写 `sync`。
- 编译器不会检查 bitcode 内部是否真的执行了声明的同步动作；声明必须与设备侧实现一致。

## 6. 验证范围

仓库的 Python 编译用例验证了 `WAIT`、`SET` 和固定 event 可以生成同步槽位属性；
GraphSyncSolver 的 MLIR 用例验证了三种 hint 的宏边界处理。目前这些用例不等同于真实 bitcode 的
NPU 运行验证。
