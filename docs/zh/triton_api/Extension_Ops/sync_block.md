# triton.language.sync_block_set
## 1. 函数概述

显式的核心间同步指令，用于协调 Cube-Vector 架构中不同核心间的执行顺序和数据一致性。

## 2. `sync_block_set` 操作

### 2.1 函数概述

生产者核心完成任务后，向消费者发送同步信号。

```python
triton.language.sync_block_set(sender, receiver, event_id, _builder=None)
```

### 2.2 规格

#### 2.2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `sender` | `str` | 必需 | 发送方核心类型："cube" 或 "vector" |
| `receiver` | `str` | 必需 | 接收方核心类型："cube" 或 "vector" |
| `event_id` | `int` | 必需 | 事件ID，用于区分不同的同步点 |
| `_builder` | - | `None` | 保留参数，暂不支持外部调用 |

#### 2.2.2 特殊限制说明

1. `sender` 和 `receiver` 不能相同，不能自己给自己发信号
2. `event_id` 必须在 0-15 范围内（共16个独立事件）

## 3. `sync_block_wait` 操作

### 3.1 函数概述

消费者核心等待生产者的同步信号。

```python
triton.language.sync_block_wait(sender, receiver, event_id, _builder=None)
```

### 3.2 规格

#### 3.2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `sender` | `str` | 必需 | 发送方核心类型："cube" 或 "vector" |
| `receiver` | `str` | 必需 | 接收方核心类型："cube" 或 "vector" |
| `event_id` | `int` | 必需 | 等待的事件ID |
| `_builder` | - | `None` | 保留参数，暂不支持外部调用 |

#### 3.2.2 特殊限制说明

1. `sender` 和 `receiver` 不能相同
2. `event_id` 必须与对应 `sync_block_set` 使用的 ID 一致

## 4. `sync_block_all` 操作

### 4.1 函数概述

全局屏障同步，让所有指定类型的核心同步到同一点。

```python
triton.language.sync_block_all(mode, event_id, _builder=None)
```

### 4.2 规格

#### 4.2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `mode` | `str` | 必需 | 同步模式："all_cube"、"all_vector" 或 "all" |
| `event_id` | `int` | 必需 | 全局同步事件ID |
| `_builder` | - | `None` | 保留参数，暂不支持外部调用 |

#### 4.2.2 特殊限制说明

1. `mode` 必须为 "all_cube"、"all_vector" 或 "all" 之一
2. `event_id` 必须在 0-15 范围内

## 5. 使用方法

### 5.1 基础使用示例

```python
import triton
import triton.language as tl
import triton.language.ascend as al

@triton.jit
def sync_example():
    # Cube 核心计算并通知 Vector
    with al.Scope(core_mode="cube"):
        # ... 执行 Cube 计算 ...
        tl.sync_block_set("cube", "vector", 0)

    # Vector 核心等待 Cube 完成
    with al.Scope(core_mode="vector"):
        tl.sync_block_wait("cube", "vector", 0)
        # ... 执行 Vector 计算 ...
```

### 5.2 Flash Attention 流水线示例

```python
@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, ...):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    with al.Scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            tl.sync_block_set("cube", "vector", 0)
            tl.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            tl.sync_block_set("cube", "vector", 2)

    with al.Scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            tl.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            tl.sync_block_set("vector", "cube", 1)
            tl.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    with al.Scope(core_mode="cube"):
        tl.sync_block_all("all", 0)

    tl.store(o_ptr + offsets, acc)
```
