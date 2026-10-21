# ProcessArgsPass (Step 1)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.cpp`
- **目的**: 消除跨 `block_id` 的循环 `iter_args` 共享。
  若多个不同 `block_id` 的计算块读取了相同的循环 `iter_arg`，当这些计算块被包裹入独立的条件分支 `scf.IfOp` 后，彼此更新的状态会互相覆盖或冲突。

## 2. 核心处理逻辑
1. **分析共享参数 (`collectArgIndexToBlockIds`)**:
   - 分析每个 `iter_arg` 被哪些 `block_id` 访问。
   - 若某个 `iter_arg` 被超过一个 `block_id` 访问，则判定为共享参数。
2. **循环参数复制与裂变**:
   - 重构主循环（`scf.for` 或 `scf.while`），将共享参数裂变为多个专有参数（每个 `block_id` 独占一份副本）。
   - 复制初始值 `init_args`。
3. **循环体内部改写**:
   - 将各个 `block_id` 内部对共享参数的使用，重定向到其专属的 BlockArgument。
   - 相应更新循环的 `scf.yield`。
4. **`whileBlockArgMap` 记录**:
   - 对于 `scf.while` 循环，建立新老参数在 `before` 与 `after` 区域中的映射关系，供后续条件克隆使用。
