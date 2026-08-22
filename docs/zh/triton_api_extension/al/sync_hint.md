# al.SYNC_HINT 接口文档

## 1. 硬件背景

`CustomMacro` 可以跨两条流水线工作，因此设备侧实现与 `CustomMacro` 外部之间可能需要配合同步。
`al.SYNC_HINT` 用于说明设备侧实现已经完成哪一侧的同步动作，编译器的同步处理阶段据此衔接
另一侧的同步。

## 2. 接口说明

### 2.1 枚举定义

```python
class SYNC_HINT(enum.Enum):
    WAIT = ascend_ir.SYNC_HINT.wait
    SET = ascend_ir.SYNC_HINT.set
    INTERNAL = ascend_ir.SYNC_HINT.internal
```

这些枚举值通过 `al.SYNC_HINT` 访问，例如 `al.SYNC_HINT.WAIT`。

### 2.2 枚举值说明

| 枚举值 | 设备侧实现中的同步动作 | 编译器在 CustomMacro 边界的处理 |
| --- | --- | --- |
| `WAIT` | 内部执行等待 | 复用已有的匹配信号；没有时在 CustomMacro 前补充设置信号 |
| `SET` | 内部设置信号 | 在 CustomMacro 之后补充匹配的等待 |
| `INTERNAL` | 内部自行处理同步 | 不在边界补充设置或等待 |

`al.SYNC_HINT` 通过 `al.SyncEventSlot` 的 `sync` 字段使用：

```python
al.SyncEventSlot(
    set_pipe=None,
    wait_pipe=None,
    sync=None,
    event=None,
)
```

一个 `SyncEventSlot` 描述一组同步关系。其中，`set_pipe` 是设置信号的一侧，`wait_pipe`
是等待信号的一侧，`sync` 用于说明设备侧实现承担哪一个同步动作。

## 3. 约束说明

- `sync_event_slots` 只适用于 `pipe` 配置了两条流水线的 `CustomMacro`。
- 使用 `WAIT` 或 `SET` 时，必须同时指定 `set_pipe` 和 `wait_pipe`；使用 `INTERNAL` 时可以省略。
- Python 接口省略 `sync` 时默认使用 `WAIT`。为使配置含义清楚，建议显式填写 `sync`。
- `SYNC_HINT` 的声明必须与设备侧实现中的同步动作一致。

## 4. 用例示例

以下代码是 `@al.register_custom_op` 注册类中的配置片段。

```python
@al.register_custom_op
class custom_macro_with_wait:
    # 两个 PIPE 表示这是一个 CustomMacro
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)

    sync_event_slots = [
        al.SyncEventSlot(
            set_pipe=al.PIPE.PIPE_MTE2,
            wait_pipe=al.PIPE.PIPE_MTE1,
            # 设备侧实现内部等待，同步处理阶段复用或补充匹配的设置信号
            sync=al.SYNC_HINT.WAIT,
        )
    ]
```

如果设备侧实现内部设置信号，则将 `sync` 改为 `al.SYNC_HINT.SET`，同步处理阶段会在
`CustomMacro` 之后补充匹配的等待。

## 5. 编译输出结果

上面的配置会在 `CustomMacro` 上生成关键同步属性：

```mlir
sync_event_slots = [
  #hivm.sync_event_slot<
    #hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait
  >
]
```

其中 `wait` 表示设备侧实现内部执行等待，后续编译阶段据此处理 `CustomMacro` 边界的同步。
