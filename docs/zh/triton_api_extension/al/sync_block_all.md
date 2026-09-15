# al.sync_block_all 接口文档

## 1. 硬件背景

当多个核（Cube 核之间、Vector 核之间、或 Cube↔Vector）共同访问同一块全局内存（GM）时，可能产生 **RAW（读后写）/ WAR（写后读）/ WAW（写后写）** 数据冒险。`al.sync_block_all` 用于在指定范围内插入**全局屏障（barrier）**同步指令，使屏障前后的内存访问严格按序完成，避免数据读写错误。

与 `al.sync_block_set`/`al.sync_block_wait` 的信号量式点对点同步不同，`al.sync_block_all` 是 **广播式屏障**：一次调用同步当前 core type 下的所有核（或全部 Cube+Vector 核），不需要 event_id 配对。

## 2. 接口说明

```python
def sync_block_all(mode: str, event_id: int, _semantic=None) -> None:
```

### 2.1 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `mode` | `str` | 是 | 同步范围，取值见下文表格 |
| `event_id` | `int` | 是 | 同步标记 ID，取值范围 `[0, 15]`（硬件同步信号量槽位） |
| `_semantic` | - | 内部参数 | JIT 编译器自动传参，用户不要手动传 |

#### mode 取值

| 取值 | 含义 | IR 属性 | 涉及的硬件流水线 |
|------|------|---------|------------------|
| `"all_cube"` | 同步所有 Cube 核（同一 AIC 组内） | `<ALL_CUBE>` | `tcube_pipe = <PIPE_ALL>` |
| `"all_vector"` | 同步所有 Vector 核（同一 AIC 组内） | `<ALL_VECTOR>` | `tvector_pipe = <PIPE_ALL>` |
| `"all"` | 同时同步所有 Cube 核和所有 Vector 核 | `<ALL>` | `tcube_pipe = <PIPE_ALL>` 且 `tvector_pipe = <PIPE_ALL>` |
| `"all_sub_vector"` | 同步 Vector 子块之间（同一 AIV 内的并行子块） | `<ALL_SUB_VECTOR>` | `tvector_pipe = <PIPE_ALL>` |

### 2.2 返回值

无返回值（插入屏障指令）。

## 3. 约束说明

- `mode` 必须是 `"all_cube"` / `"all_vector"` / `"all"` / `"all_sub_vector"` 之一。
- `event_id` 必须为 `0 ~ 15` 的整数（硬件仅提供 16 个独立同步槽位）。
- `sync_block_all` 是**屏障**语义：所有参与核到达该点后才一起继续执行，不依赖配对的 set/wait 调用。
- 调用位置建议位于两个存在数据依赖的计算/访存阶段之间（例如：GM 写完成后、GM 读之前）。

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def sync_all_demo_kernel():
    """在同一 kernel 中分别对 Cube / Vector / 全部核 / Sub-Vector 进行屏障同步。"""
    # Cube 核之间屏障
    al.sync_block_all("all_cube", event_id=8)

    # Vector 核之间屏障
    al.sync_block_all("all_vector", event_id=9)

    # 所有 Cube + Vector 核屏障
    al.sync_block_all("all", event_id=10)

    # Vector 子块之间屏障
    al.sync_block_all("all_sub_vector", event_id=11)
```

## 5. 编译输出结果

```mlir
module {
  tt.func public @sync_all_demo_kernel() attributes {noinline = false} {
    // all_cube：tcube_pipe 参与同步
    hivm.hir.sync_block[<ALL_CUBE>, 8 : index]
        tcube_pipe = <PIPE_ALL>

    // all_vector：tvector_pipe 参与同步
    hivm.hir.sync_block[<ALL_VECTOR>, 9 : index]
        tvector_pipe = <PIPE_ALL>

    // all：tcube_pipe + tvector_pipe 同时参与
    hivm.hir.sync_block[<ALL>, 10 : index]
        tcube_pipe = <PIPE_ALL> tvector_pipe = <PIPE_ALL>

    // all_sub_vector：Vector 子块间同步（仍在 tvector_pipe）
    hivm.hir.sync_block[<ALL_SUB_VECTOR>, 11 : index]
        tvector_pipe = <PIPE_ALL>

    tt.return
  }
}
```

### 输出要点说明

- `al.sync_block_all` 在 IR 中降低为 `hivm.hir.sync_block[<mode>, event_id : index]` 指令。
- `mode` 通过方括号内的枚举属性表达：`<ALL_CUBE>` / `<ALL_VECTOR>` / `<ALL>` / `<ALL_SUB_VECTOR>`。
- 参与同步的硬件流水线由属性 `tcube_pipe` / `tvector_pipe` 指定，`al.sync_block_all` 默认是 `PIPE_ALL`（所有流水）。
- `event_id` 在 IR 中以 `index` 类型常量形式出现（如 `8 : index`）。

## 6. 相关接口

- [`al.sync_block_set`](sync_block_set.md)：Cube→Vector 或 Vector→Cube 的点对点信号量 set
- [`al.sync_block_wait`](sync_block_wait.md)：与 `sync_block_set` 配对的等待端
- [`al.scope`](scope.md)：跨 Cube/Vector scope 时经常需要用 `sync_block_all`/`sync_block_set` 做数据依赖保证
