# al.sync_block_wait

## 1. 概述

`al.sync_block_wait` 在接收端等待 Cube Core 与 Vector Core 之间的同步信号。
它必须与发送端的
[`al.sync_block_set`](./sync_block_set.md) 配对使用。

当对应 event 的计数器为 0 时，wait 会阻塞；计数器大于 0 时，wait 将其减 1，
随后允许接收端的后续指令继续执行。

## 2. 接口说明

```python
al.sync_block_wait(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: al.PIPE | None = None,
    receiver_pipe: al.PIPE | None = None,
) -> None
```

内部注入参数不属于用户接口，因此未在签名中列出。

### 参数

| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sender` | `str` | 无 | 发送端，只能是 `"cube"` 或 `"vector"` |
| `receiver` | `str` | 无 | 接收端，只能是 `"cube"` 或 `"vector"`，且必须与 `sender` 不同 |
| `event_id` | `int` | 无 | 同步标记 ID，取值范围为 `[0, 15]` |
| `sender_pipe` | [`al.PIPE`](./pipe.md) 或 `None` | `None` | 与 set 一致的发送端流水线 |
| `receiver_pipe` | [`al.PIPE`](./pipe.md) 或 `None` | `None` | 接收端执行等待的流水线 |

### 返回值

无。

## 3. 默认流水线

`sender_pipe` 和 `receiver_pipe` 同时省略时，接口根据发送端选择默认值：

| `sender` | `sender_pipe` | `receiver_pipe` |
| --- | --- | --- |
| `"cube"` | `PIPE_FIX` | `PIPE_MTE2` |
| `"vector"` | `PIPE_MTE3` | `PIPE_MTE2` |

如需显式指定，两项必须同时传入，并且必须与配对的 set 完全一致。

## 4. 使用方法

```python
import triton
import triton.language.extra.cann.extension as al


@triton.jit
def cube_to_vector_sync():
    with al.scope(core_mode="cube"):
        # Cube 侧生产数据并写入 GM。
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        # Vector 侧在 wait 之后读取并消费数据。
```

使用默认流水线时，对应的关键 IR 为：

```mlir
hivm.hir.sync_block_wait[
  <VECTOR>, <PIPE_FIX>, <PIPE_MTE2>
] flag = %event
```

显式指定流水线的写法：

```python
al.sync_block_wait(
    "cube",
    "vector",
    5,
    al.PIPE.PIPE_MTE1,
    al.PIPE.PIPE_MTE3,
)
```

## 5. 约束说明

- wait 的 sender、receiver、event ID 和 pipe pair 必须与配对的 set 完全相同。
- `sender` 与 `receiver` 只能取 `"cube"`、`"vector"`，并且不能相同。
- 静态 `event_id` 必须位于 `[0, 15]`。
- 这里的 `event_id` 是普通整数，不是定制宏算子使用的
  [`al.EVENT_ID`](./event_id.md) 枚举。
- wait 应位于与 `receiver` 类型匹配的 `al.scope` 中，并放在接收端消费数据之前。
- 只有一个 pipe 参数为 `None` 时会触发类型错误；两项应同时省略或同时提供。
- 不匹配或缺失 set 的 wait 可能让 Kernel 一直等待。不要在 NPU 上运行预期死锁的负例。

## 6. 验证范围

仓库的 `test_sync_block.py` 使用同一组 sender、receiver、event 和 pipe pair 执行 set/wait，
并在 NPU 上校验 Cube 矩阵计算与 Vector 指数计算的结果。该用例只覆盖其中一组配置，
不代表所有 event 和 pipe 组合都已逐项验证。
