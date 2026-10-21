# UpdateLoopOpsPass (Step 4)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.cpp`
- **目的**: 重构主循环（`scf.for` / `scf.while`），在 `iter_args` 中追加分配控制所需的初始标量状态：
  - 各 Block 的循环迭代计数器（`blockCounters`）。
  - 核内依赖计数器（`innerDepConds`）。
  - 张量型 `iter_args` 伴生控制变量。
  - （可选配置下）插入核间 `PIPE_S` 同步指令。

## 2. 核心处理逻辑
1. **统计所需追加参数数量 (`computeMainLoopExtraArgs`)**:
   - `numBlockCounters`: 每个独特的 `block_id` 分配一个迭代计数器。
   - `numInnerDepConds`: 核内核间依赖槽位数量。
   - `numTensorIterArgs`: 张量型循环参数对应的消费者控制信号数量。
2. **构建初始值 (`buildMainLoopExtraInitArgs`)**:
   - `blockCounters`: `scf.for` 初始值为 `lowerBound`；`scf.while` 初始值为 `0 : i32`。
   - `innerDepConds`: 初始值为 `0 : i32`。
   - `tensorIterArgs`: 初始值为 `1 : i32`（表示初始状态就绪）。
3. **创建新循环并迁移体 (`createForOpAndMigrateBody`)**:
   - 调用 `createNewForOpWithExtras`，将 extra inits 追加到末尾。
   - 迁移旧循环体所有操作，构建新的 `scf.yield` 将新参数自环 yield。
   - 记录索引范围到 `info->blockCounters[newLoop]` 和 `info->innerDepConds[newLoop]`。
4. **历史遗留同步说明**:
   - 此 Pass 可能会在某些配置下发射 `hivm.hir.sync_block_set/wait[<CORE>, <PIPE_S>, <PIPE_S>] flag = 15`。
   - 在新版标量锁步架构下，这些 `PIPE_S` 保持不动，`UpdateConditionInfoPass` 中不需要也不再添加任何新的同步指令。
