# UpdateConditionInfoPass

## 1. 概述与核心职责
`UpdateConditionInfoPass` 是 `AddControlFlowCondition` 管线的第 5 步（Step 5）。
其职责是为带有 `ssbuffer.if` 属性的条件分支（`scf.IfOp`）生成控制条件，包括：
- 跨核依赖条件（Cube 与 Vector 之间的缓冲区空间与数据到达控制）。
- 核内核间依赖条件（Intra-core buffers 读写控制）。
- Tensor-typed `iter_args` 的流转控制。
- 循环计数器边界控制（`counter < upperBound`）与流水线优化条件（FlowOpt）。
- 更新循环的 yield 终结符，使变量迭代状态在循环迭代间正确传递。

---

## 2. 旧版方案：基于 SSBuffer 共享内存的跨核同步 (Legacy)

### 2.1 机制原理
旧方案依赖硬件提供的 SSBuffer（Scalar Synchronization Buffer）共享内存区域进行 Cube 与 Vector 核之间的标量数据通信与轮询。

### 2.2 内存布局与寻址
- **分配与清零 (`allocSSBuffer`)**:
  - 在 `scope.scope` 外为每个跨核依赖组分配 4 字节的槽位，基地址硬编码：
    - Vector 0: 基址 `0`，步长 4B。
    - Vector 1: 基址 `1024`（`VECTOR_SSBUF_OFFSET`），步长 4B。
  - 函数入口通过 `memref.store` 将槽位初始化为 0。
- **Cube 侧寻址**:
  - Cube 核心需同时监控两个 Vector 槽位（`kVectorCount = 2`），分别加载 Vector 0 和 Vector 1 的指针。
- **Vector 侧寻址 (`computeVectorSSBufferMemrefs`)**:
  - 动态调用 `hivm.hir.get_sub_block_idx` 获取当前子核 ID。
  - 计算地址偏移：`addr = subId * 1024 + groupIdx * 4`，通过 `PointerCastOp` 转为 memref。

### 2.3 轮询与 Volatile 协议
- 所有 SSBuffer 的 load/store 操作均需包裹 `annotation.mark` 并附带 `kMemrefExtVolatile` 属性以防止编译器死代码消除或指令重排。
- **输入条件 (Consumer)**:
  - 读取 SSBuffer 计数，判断 `token > 0`。
  - Cube 侧：`load(Vec0) > 0 && load(Vec1) > 0`。
  - Then-Block 执行：volatile load -> `subi 1` -> volatile store。
- **输出条件 (Producer)**:
  - 读取 SSBuffer 计数，判断 `token < bufferCapacity`。
  - Cube 侧：`load(Vec0) < cap && load(Vec1) < cap`。
  - Then-Block 执行：volatile load -> `addi 1` -> volatile store。

### 2.4 旧版弊端
1. **硬件访存开销**: 标量控制流频繁进行 SSBuffer volatile load/store 轮询，造成严重的内存延迟与总线冲突。
2. **硬件特定依赖**: 绑定 Ascend SSBuffer 内存模型与 sub-block 指令，无法纯标量静态调度。
3. **指令膨胀**: 产生大量地址计算、类型转换与 volatile 标注指令。

---

## 3. 新版方案：无 SSBuffer / 纯标量锁步仿真 (Lockstep Emulation)

### 3.1 核心思想
Cube 和 Vector 的主循环具有对称、确定的迭代空间（上下界及步长完全相同），其标量依赖完全可以通过**本地状态机锁步推进**来模拟，**无需任何跨核通信、SSBuffer 内存读写或 PIPE_S 硬件同步**。

### 3.2 循环状态统一 (`iter_args` 扩展)
Cube 和 Vector 的 `scf.for` 循环均被重构，携带完全相同的锁步控制状态：
```text
scf.for ... iter_args(
    %user_args...,             // 用户原本的计算与张量参数
    %token_0, %token_1, ...,   // 跨核 buffer 虚拟 token（初值 0）
    %cube_counters...,         // Cube 各 IfOp 的计数器（Cube 为本地，Vector 为 shadow）
    %vec_counters...           // Vector 各 IfOp 的计数器（Vector 为本地，Cube 为 shadow）
)
```

### 3.3 拓扑序：Cube If 在前，Vector If 在后
在**两核的循环体内**，所有 `scf.IfOp` 均遵循固定的执行顺序：
```text
Cube Scope Loop:
  [Real Cube_If_1]   -> ... -> [Real Cube_If_N]    // 真实计算 + 推进 Cube 计数器与 tokens
  [Dummy Vector_If_1]-> ... -> [Dummy Vector_If_M] // 空计算体 + 仅推进 Vector 计数器与 tokens
  scf.yield

Vector Scope Loop:
  [Dummy Cube_If_1]  -> ... -> [Dummy Cube_If_N]   // 空计算体 + 仅推进 Cube 计数器与 tokens
  [Real Vector_If_1] -> ... -> [Real Vector_If_M]  // 真实计算 + 推进 Vector 计数器与 tokens
  scf.yield
```
- **SSA 支配关系**: 在 Vector 循环中，Dummy Cube IfOps 插入在循环体最前端（`front()`），从而能够产生本轮迭代产生的 tokens，支配后续 Real Vector IfOps 的消费条件。
- 在 Cube 循环中，Dummy Vector IfOps 插入在 terminator 前，顺序推进。

### 3.4 Dummy IfOp 构造规范
- **条件谓词**:
  - 计算与真实 IfOp 完全相同的条件（基于本地 shadow 计数器和 token `iter_args`）。
  - `combined_cond = andi(crossCoreTokensCond, counter < upperBound)`.
- **Then-Block (命中分支)**:
  - **无任何计算负载**（无 DMA、算子、访存）。
  - 仅做标量更新：生产 token `+1`，消费 token `-1`，计数器 `+step`。
  - 通过 `scf.yield` 返回更新后的值。
- **Else-Block (旁路分支)**:
  - 直接原样通过 `scf.yield` 转发输入时的 token 与 counter，不做任何改变。

### 3.5 Selective Yielding 与最新 SSA 追踪
- **原则**: 每个 IfOp（无论是 Real 还是 Dummy）只 yield 自身修改的变量和计数器：`[orig_results..., used_vars..., counter]`。
- Else-block 仅转发自身对应的 `used_vars` 与 `counter`，绝对不 dump 全量 `iter_args`。
- 通过 `controlVarToLatestValue` 维护 `原 iter_arg -> 最新 SSA 结果` 的映射表。
- 循环末尾 `scf.yield` 统一从 `controlVarToLatestValue` 取出最新 SSA 值写入终结符。

---

## 4. 新旧方案对比

| 维度 | 旧版方案 (SSBuffer) | 新版方案 (Lockstep Scalar) |
|---|---|---|
| **通信机制** | SSBuffer 内存 volatile 读写轮询 | 零通信，本地 `iter_args` 标量状态机推进 |
| **硬件同步** | 依赖 SSBuffer 与跨核流水同步 | 完全不需要 SSBuffer，也不新增 PIPE_S |
| **IfOp 结构** | 单核仅包含本核 IfOp | 两核均包含对方的 Dummy IfOp，且 Cube 始终在前 |
| **Then/Else 动作** | 读写物理内存槽位 | 纯 SSA 标量加减（`arith.addi`/`subi`） |
| **执行延迟** | 受核间内存总线与轮询开销制约 | 零访存开销，完全消除内存等待 |
| **鲁棒性** | 多核并发时序敏感 | 确定性（Deterministic）无竞争 |

---

## 5. 调试踩坑与经验 (Lessons Learned)

### 5.1 `info->cntArgs` 悬垂指针 (UAF Crash)
- **现象**: 重写后下游 `UpdateLoopIterTimesPass` 发生 `SIGSEGV`，堆栈处于 `mlir::DictionaryAttr::contains`。
- **原因**:
  - `info->cntArgs` 是跨 Pass 共享的 `DenseMap<scf::IfOp, Value>`。
  - 本 Pass 将旧的 `rawCubeIfOps`/`rawVectorIfOps` 替换为新构建的 `newRealIfOp` 并 `erase()` 了旧的 `ifOp`。
  - 如果未清空 `info->cntArgs`，下游 Pass 遍历该 map 访问已销毁的 `ifOp` 指针时崩溃。
- **解决办法**:
  - 在 Pass 开始处理循环前显式调用 `info->cntArgs.clear()`。
  - 在生成 `newRealIfOp` 与 `newDummyIfOp` 时，将新的操作句柄注册回 `info->cntArgs`。

### 5.2 循环重建的 SSA 映射传递
- 重构 `scf.for` 添加额外 `init_args` 时，旧循环体的 BlockArgument 与新循环体不同。必须使用 `migrateBody` 正确建立 `replaceAllUsesWith`，并同步迁移 `info->blockCounters`、`info->innerDepConds`、`info->tensorIterArgDepsMap` 中的键到新 loop op。
