# UpdateLoopIterTimesPass (Step 6)

## 1. 概述与目的
- **代码位置**: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.cpp`
- **目的**: 
  1. 依据流水线缓冲深度和 DAG 阶段，缩放 `scf.for` 的总迭代上限（`upperBound`），使流水线有足够的迭代次数完成填充（Prologue）、稳态运行和排空（Epilogue）。
  2. 将 IfOp 内部对原循环归纳变量（Induction Variable, IV）的使用，替换为各 IfOp 专属的独立计数器（`cntArgs[ifOp]`）。

## 2. 核心处理逻辑
1. **替换 IV 为独立计数器 (`replaceForOpCounterInIfOps`)**:
   - 遍历每个主循环内部携带 `ssbuffer.if` 的 IfOp。
   - 从 `info->cntArgs[ifOp]` 获取该 IfOp 的专属计数器 SSA 值。
   - 将 IfOp 内部所有对原循环 IV 的使用替换为该 `cntVal`。
   - **关键依赖**: 强依赖 `info->cntArgs` 中记录的 `scf.IfOp` 必须是有效指针，若为销毁的悬垂指针则会在调用 `ifOp->hasAttr` 时崩溃。
2. **计算迭代放大因子 (`calculateFactor`)**:
   - 结合核内依赖 `intraCoreDependentMap` 与跨核依赖 `crossCoreDependentMap`。
   - 根据所需的总缓冲数 `requiredBuffers` 计算流水步长扩展：
     $$\text{factor} = \lceil \frac{\text{iterCount} \times \text{requiredBuffers}}{x} \rceil + \text{ifCount}$$
3. **重构循环上限 (`extendForOpIterationCount`)**:
   - 新上限公式：
     $$\text{newUpperBound} = \text{lowerBound} + \text{step} \times \text{factor}$$
   - 生成新的 `scf.for`，替换原循环。
