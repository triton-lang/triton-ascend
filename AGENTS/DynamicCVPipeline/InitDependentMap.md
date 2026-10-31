# InitDependentMapPass (Step 3)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.cpp`
- **目的**: 基于内存读写关系和算子属性，提取跨核（Cross-Core）和核内（Intra-Core）的数据依赖拓扑，构建 `ifBlockDAG` 与流水线参数。

## 2. 核心数据结构与提取流程
1. **跨核依赖 (`crossCoreDependentMap`)**:
   - 检查操作上的 `ssbuffer.crossCoreDeps = [consumer_id, producer_id]` 属性。
   - 提取跨越 Cube 和 Vector 作用域的生产者-消费者关系：
     `info->crossCoreDependentMap`: `consumerOp -> SmallVector<SmallVector<producerOp>>`。
2. **核内依赖 (`intraCoreDependentMap`)**:
   - 检查操作上的 `ssbuffer.intraDeps = [consumer_id, producer_id]` 属性。
   - 建立单核循环内部不同计算块之间的生产-消费依赖：
     `info->intraCoreDependentMap[loopOp]`: `consumerOp -> SmallVector<producerOp>`。
3. **DAG 依赖图 (`ifBlockDAG`)**:
   - 将每个算子的依赖上溯到其包含的 `scf.IfOp`。
   - 构建 `scf.IfOp` 级别的有向无环图 `info->ifBlockDAG`，用于确定执行先后与拓扑排序。
4. **流水线流优化候选对 (`flowOptIfOpPairs`)**:
   - 识别符合流水线阶段加速的生产-消费块对，记录在 `info->flowOptIfOpPairs[targetIf] = sourceIf`。
   - 记录最大缓冲区深度 `info->crossCoreBufferCount` 和 `info->intraCoreBufferCount`。
