# CloneOpsPass (Step 0)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CloneOps.cpp`
- **目的**: 消除不同 `block_id` 之间的算子共享。
  在之前的 Pass 中，部分无副作用或视图算子（例如 `tensor.empty`、常量、`memref.memory_space_cast`）可能被多个属于不同 `block_id` 的计算节点共享。
  如果不进行克隆，后续将同一个算子归属于多个 `scf.IfOp` 会破坏 SSA 单一归属原则并引入错误的控制依赖。

## 2. 核心处理逻辑
1. **收集共享算子**:
   - 遍历 `scope.scope` 内主循环的所有操作。
   - 检查每个算子被哪些不同的 `ssbuffer.block_id` 所使用。
   - 若同一个算子的结果被多个 `block_id` 的算子作为操作数消费，则标记为待克隆算子。
2. **按 Block ID 克隆**:
   - 为消费它的每一个独立 `block_id` 克隆一份专有算子副本。
   - 为克隆后的算子打上对应的 `ssbuffer.block_id`。
   - 替换消费者的操作数为克隆版本的结果。
3. **消除跨 Block 依赖**:
   - 使得每个 `block_id` 内的算子 def-use 链尽可能闭包在自身内部或仅通过循环 `iter_args` 显式传递。
