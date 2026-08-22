# al.EVENT_ID 接口文档

## 1. 硬件背景

Ascend NPU 的不同流水线可以通过 `set_flag` 和 `wait_flag` 配合完成同步。发送端设置事件，
等待端等待同一个事件，二者的流水线和 event ID 必须对应。

`CustomMacro` 是同时描述输入、输出流水线的设备侧操作。它可以通过 `sync_event_slots`
声明设备实现中的同步关系，其中每个 `al.SyncEventSlot` 表示一个同步槽位。
`al.EVENT_ID` 用于为这个槽位指定 event ID。

## 2. 接口说明

### 2.1 枚举定义

```python
class EVENT_ID(enum.Enum):
    EVENT_ID0 = ascend_ir.EVENT.EVENT_ID0
    EVENT_ID1 = ascend_ir.EVENT.EVENT_ID1
    EVENT_ID2 = ascend_ir.EVENT.EVENT_ID2
    EVENT_ID3 = ascend_ir.EVENT.EVENT_ID3
    EVENT_ID4 = ascend_ir.EVENT.EVENT_ID4
    EVENT_ID5 = ascend_ir.EVENT.EVENT_ID5
    EVENT_ID6 = ascend_ir.EVENT.EVENT_ID6
    EVENT_ID7 = ascend_ir.EVENT.EVENT_ID7
```

这些枚举值通过 `al.EVENT_ID` 访问，例如 `al.EVENT_ID.EVENT_ID1`。

### 2.2 使用方式

`al.EVENT_ID` 用于 `al.SyncEventSlot` 的 `event` 参数。一个同步槽位还会指定设置事件的
`set_pipe`、等待事件的 `wait_pipe`，以及 CustomMacro 设备实现执行哪一侧同步操作。

`event` 是可选参数。只有需要明确指定 event ID 时，才需要传入 `al.EVENT_ID` 枚举值。

## 3. 约束说明

- `al.EVENT_ID` 只能用于 `CustomMacro` 的 `sync_event_slots`，不能用于普通 `CustomOp`。
- 同一组 `set_flag` 和 `wait_flag` 必须使用相同的流水线组合和 event ID。
- 声明的同步槽位必须与 CustomMacro 的设备侧实现保持一致。
- `al.EVENT_ID` 的取值范围是 0～7，只用于 `SyncEventSlot.event`。`al.sync_block_set`、
  `al.sync_block_wait` 和 `al.sync_block_all` 的 `event_id` 参数接收 0～15 的普通整数，
  不能把 `al.EVENT_ID` 枚举值直接传给这些参数。

## 4. 用例示例

以下代码是 `@al.register_custom_op` 注册类中的配置片段。

```python
@al.register_custom_op
class vector_custom_macro:
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_V,
            sync=al.SYNC_HINT.WAIT,
            event=al.EVENT_ID.EVENT_ID1,
        )
    ]
```

这里的 `EVENT_ID1` 表示该同步槽位使用编号 1。`SYNC_HINT.WAIT` 表示 CustomMacro 的设备实现
执行 `wait_flag`，与之配对的 `set_flag` 也必须使用相同的流水线组合和 event ID。

## 5. 编译输出结果

上述配置生成的关键 IR 属性为：

```mlir
sync_event_slots = [
  #hivm.sync_event_slot<
    #hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, wait, <EVENT_ID1>
  >
]
```
