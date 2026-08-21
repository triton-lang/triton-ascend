# al.sync_block_all

## 1. 概述

`al.sync_block_all` 让当前 Kernel 调度范围内参与执行的 Cube Core、Vector Core 或
Vector Sub Block 到达同一个同步点。它用于参与者访问共享 GM 且存在 RAW、WAR 或 WAW
数据依赖的场景。

该接口是集合屏障，不是
[`al.sync_block_set`](./sync_block_set.md) 与
[`al.sync_block_wait`](./sync_block_wait.md) 的单向生产者/消费者配对。

## 2. 接口说明

```python
al.sync_block_all(mode: str, event_id: int) -> None
```

### 参数

| 参数名 | 类型 | 含义 |
| --- | --- | --- |
| `mode` | `str` | 同步范围，只能取下表中的四个字符串 |
| `event_id` | `int` | 同步标记 ID，取值范围为 `[0, 15]` |

| `mode` | 同步范围 |
| --- | --- |
| `"all_cube"` | 当前调度范围内参与执行的所有 Cube Core |
| `"all_vector"` | 当前调度范围内参与执行的所有 Vector Core |
| `"all"` | 当前调度范围内参与执行的所有 Cube Core 与 Vector Core |
| `"all_sub_vector"` | 当前 AI Core 上参与执行的所有 Vector Sub Block |

### 返回值

无。

## 3. 使用方法

下面的 Kernel 展示四种模式的编译写法：

```python
import triton
import triton.language.extra.cann.extension as al


@triton.jit
def sync_all_modes():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)
```

对应的关键 IR 为：

```mlir
hivm.hir.sync_block[<ALL_CUBE>, 8 : index]
  tcube_pipe = <PIPE_ALL>
hivm.hir.sync_block[<ALL_VECTOR>, 9 : index]
  tvector_pipe = <PIPE_ALL>
hivm.hir.sync_block[<ALL>, 10 : index]
  tcube_pipe = <PIPE_ALL> tvector_pipe = <PIPE_ALL>
hivm.hir.sync_block[<ALL_SUB_VECTOR>, 11 : index]
  tvector_pipe = <PIPE_ALL>
```

## 4. 约束说明

- `mode` 必须是 `"all_cube"`、`"all_vector"`、`"all"` 或
  `"all_sub_vector"`。
- `event_id` 必须是 `[0, 15]` 范围内的整数。
- 这里的 `event_id` 是普通整数，不是定制宏算子使用的
  [`al.EVENT_ID`](./event_id.md) 枚举。
- 当前后端中，`"all_cube"`、`"all_vector"` 和 `"all_sub_vector"` 会保留用户传入的
  `event_id`；`"all"` 在后续分解时使用编译器内部的 10、11、12、13 号 flag，不把用户值作为
  最终同步 flag。不要使用 `"all"` 的 `event_id` 与其他显式同步操作建立配对关系。
- 这里的“所有”仅指当前 Kernel 调度范围内的参与者，不是整张设备、跨 Kernel 或多卡全局屏障。
- 参与同步的数据必须先移动到参与者都可见的 GM。
- 同步范围内的所有参与者必须以一致的控制流到达同一屏障；参与者缺失或到达次数不一致可能导致
  Kernel 一直等待。

## 5. 验证范围

仓库现有 `test_sync_block_all.py` 覆盖四种模式的 Python 到 HIVM IR 编译结果，
但没有执行真实 NPU 屏障功能校验。因此该页面不把编译成功表述为硬件同步已验证。
