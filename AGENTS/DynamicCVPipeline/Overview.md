# Dynamic CV Pipeline: Overview

## 1. 目标与背景
Dynamic CV Pipeline 用于在华为昇腾（Ascend NPU）架构上实现 Cube 核心（矩阵计算单元）与 Vector 核心（矢量计算单元）的高效流水线解耦与协同。

Cube 与 Vector 在物理硬件上彼此独立，但在算子计算中存在紧密的数据依赖（例如 Vector 准备数据 ->搬运到 CBUF -> Cube 计算 -> 搬运到 UB -> Vector 进一步后处理）。
通过将单核/混合指令流切分为 Cube Scope 与 Vector Scope，并在此基础上建立带缓冲（Double/Multi-buffering）的控制流水线，可以使 Cube 和 Vector 重叠执行，掩盖访存与搬运延迟。

---

## 2. 顶层流水线流程
在 `third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp` 中定义的总管线阶段如下：

1. **`SeparateMemoryFromComputePass`**: 分离全局内存（GM）加载与核心计算。
2. **`DecoupleComputeAndMemory`**: 循环体多缓存发射、异步加载提升与多缓冲配置。
3. **`SplitDataflow`**: 划分核间数据流，分离 Cube Scope 与 Vector Scope，标记主循环（`ssbuffer.main_loop`）。
4. **`PlanComputeBlock`**: 计算块（Compute Block）分类与规划，划分 `block_id`。
5. **`ComputeBlockOpt`**: 计算块优化（合并小块、定点管线优化、UB 使用优化）。
6. **`AddControlFlowCondition`**: 构建控制流条件管线（子 Pass 详解见下节）。
7. **`RemoveAttributes`**: 清理临时 `ssbuffer.*` 标记属性，恢复规范 MLIR 结构。

---

## 3. 控制流条件子管线 (`AddControlFlowCondition`)
在 `AddControlFlowConditionPass` 内部顺序执行以下 7 个子 Pass：

```text
AddControlFlowConditionPass
  ├── Step 0: CloneOpsPass (CloneOps.md)
  ├── Step 1: ProcessArgsPass (ProcessArgs.md)
  ├── Step 2: CreateIfOpsPass (CreateIfOps.md)
  ├── Step 3: InitDependentMapPass (InitDependentMap.md)
  ├── Step 4: UpdateLoopOpsPass (UpdateLoopOps.md)
  ├── Step 5: UpdateConditionInfoPass (UpdateConditionInfo.md)
  └── Step 6: UpdateLoopIterTimesPass (UpdateLoopIterTimes.md)
```

---

## 4. 核心数据流与控制流协议
- **Scope 结构**:
  - `hivm.tcore_type = #hivm.tcore_type<CUBE>` 包裹 Cube 核心操作。
  - `hivm.tcore_type = #hivm.tcore_type<VECTOR>` 包裹 Vector 核心操作。
- **主循环标注**:
  - 核心循环打上 `ssbuffer.main_loop = 0 : i64` 属性。
- **Block ID 机制**:
  - 每个算子携带 `ssbuffer.block_id = <id>`，标识其归属的流水步。
  - 后续据此包裹为 `scf.IfOp` 分支。
- **锁步（Lockstep）执行模型**:
  - 弃用 SSBuffer 跨核硬件共享内存，改用纯标量 `iter_args` 驱动两核在无同步开销下确知对方推进状态。
