# CreateIfOpsPass (Step 2)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.cpp`
- **目的**: 按照 `ssbuffer.block_id` 将循环体内部的操作打包封装为初始的条件分支（`scf.IfOp`），并打上 `{ssbuffer.if = <block_id>}`。

## 2. 核心处理逻辑
1. **收集与分组**:
   - 遍历主循环体内的所有直接操作。
   - 排除非计算控制类的操作（如循环终结符 `scf.yield`）。
   - 按照 `ssbuffer.block_id` 将算子分组存入不同的块序列中。
2. **包裹为 `scf.if %true`**:
   - 对每个 `block_id` 分组，在原位置创建一个初始条件为 `%true` 的 `scf.IfOp`。
   - 将该分组中的所有操作移入 `then` 区域。
   - 若操作产生在块外被其他操作使用的结果，将这些结果作为 `scf.if` 的返回值并在 `then` 末尾通过 `scf.yield` 导出。
   - 给新生成的 `scf.IfOp` 添加属性 `{ssbuffer.if = <block_id>}`。
3. **统计 Block 数量**:
   - 将循环包含的 IfOp/Block 数量写入 `info->blockCounterNums[loopOp]`，作为后续分配迭代计数器的依据。
