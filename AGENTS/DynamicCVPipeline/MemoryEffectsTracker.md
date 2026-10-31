# MemoryEffectsTracker

跨 pass 共享的内存依赖图（`include/DynamicCVPipeline/Common/MemoryEffectsTracker.h` + `lib/DynamicCVPipeline/Common/MemoryEffectsTracker.cpp`）。`analyzeOp` 按程序序扫块，`collectOuterEffects` 把每个算子的 memref 效果归并到 `MemSlot`，`applyEffects` 把效果写回槽状态，`collectPreds` / `recordEdges` 据此产出四个查询：`getMemUsers`（来自 `defs`）、`getExecBefore` / `getExecAfter`（来自 `preds`）、`getRealDependency`（同样来自 `preds`）。

消费者：`DataDependencyAnalysis`、`RefineArgsBlockId`、`FlagIdReuse`、`SplitIfByBlockId`、`UBUsageOptPass`、`CloneOps`、`DependencyHelper`、`Common/Utils.cpp`（`getUpstreamOpsWithMemoryDeps`，供 OpClassifier 的 CUBE/VECTOR 传播）。

## 陷阱

- **无值效果 = 全局屏障，不是“没有效果”**: `findAliasSlots(nullptr)` 把无值 `EffectInstance` 与**所有** slot 判为别名，于是 `applyEffects` / `collectPreds` 直接退化成整块 barrier。两条最常见来源：（1）`hivm.hir.custom` / `custom_macro` 的 `getEffects` 只压无值 `Read`+`Write`（`DefaultResource`，带 `no_side_effect` 时为空），其精确 buffer 集合只能另走 `CustomOpAnalysis`（`Common/Analysis.h`，由 `PackMemref` 打包时分析）；（2）`scope::ScopeOp` 只有 `RecursiveMemoryEffects`、不实现 `MemoryEffectOpInterface`：`getEffectsRecursively` 只看见 body 内算子**自身**的接口效果（custom 即上面的无值效果），`recursive=false` 路径更直接 `raw == nullopt` -> `unknown = true`。要精确效果只能手动 iter block 逐算子调 `collectOuterEffects` 聚合（`collectOuterEffects` 已有 `scope::ScopeOp` 分支，为后续 standard load 单元准备；load scope 实测得到带值的 GM 读，而 packed custom scope 仍是 2 个无值 `Read`+`Write`）。**customOp 在本模块得 barrier 是有意设计**，不要为此再补 customOp 专用 Case（其精确 buffer 分析只在 `PackMemref` 打包时算一次）。
- **`MemSlot` 依赖边不可裁剪**: `applyEffects` 对每个 memref 结果槽无条件写 `s->lastWriter = op;`（`for (auto result : op->getOpResults())` 块内那处；另有 unknown / allocate / write 三处）。这条 RAW pred 经 `collectPreds` 同时喂给 `recordEdges`（`execBefore` / `execAfter` 边）与 `getRealDependency`（跨核 set/wait），看起来像"产出不算写"的合理精简，实测劣化性能（daebd90 顺手删过该行，是回滚原因之一）-> 改 `MemSlot` / `applyEffects` 前先确认 4 处 `lastWriter = op;` 仍在（`rg -n 'lastWriter = op;' .../Common/MemoryEffectsTracker.cpp`）。
