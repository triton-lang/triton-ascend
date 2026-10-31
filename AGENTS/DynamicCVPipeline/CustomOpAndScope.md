# CustomOp Support & Scope Packaging/Unpacking (CustomOpAndScope.md)

## 1. 概述与背景
- **相关 Pass 与文件**:
  - `PackMemrefPass` (`--ssbuf-standardize-op-pack-memref`): `third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/PackMemref.cpp`
    - CustomOp 分组（含 `CustomOpAnalysis` / `packScopeOp`）与 load 单元打包均在此 Pass 完成（由 `OpClassifierPass::groupCustomOps` 迁移而来，见第 5 节）。
  - `OpClassifierPass`: `third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/OpClassifier.cpp`（仅消费 scope 上已有的 `ssbuffer.core_type`，不再做分组）
  - `UnpackScopePass` (`--ssbuf-unpack-scopeop`): `third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/UnpackScopePass.cpp`
  - 分析与工具库:
    - `third_party/ascend/include/DynamicCVPipeline/Common/Analysis.h` / `CustomOpUtils.cpp`
    - `third_party/ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h` / `ScopeOpUtils.cpp`
- **背景与目的**:
  在 Ascend NPU 的 Dynamic CV Pipeline 中，部分特殊算子（如 `hivm.hir.custom`，例如 `tanh_fp32`）由标量/向量指令扩展而来，通常伴随局部的临时内存分配与转换（`memref.alloc -> bufferization.to_tensor -> customOp`，或 `ConvertLayoutOp`）。
  如果不将其视为整体，后续在 `PlanComputeBlock`、`MergeCubeBlock` 以及重排序阶段，各局部缓冲算子和 customOp 容易被错误割裂分配到不同 block，或者跨块调度导致生命周期错乱。
  因此，引入**临时 Scope 封装机制**：
  1. 在 `OpClassifierPass` 阶段识别 CustomOp 及其依赖的局部缓冲区，将其整组打包进 `scope.scope`，作为单一原子计算单元参与后续的分块与重排序；
  2. 在 `ComputeBlockOptPass` 末尾（`ReorderOpsByBlockId` 与 `RelocateMemrefDecl` 之后），通过 `UnpackScopePass` 将其平铺还原回原始父 Block，并将 Scope 最终确定的 `ssbuffer.block_id` 和 `ssbuffer.core_type` 传播并覆盖到内部展开的各个算子。

---

## 2. 核心架构与机制对比

### 旧版方案 vs 新版方案
| 维度 | 旧版方案 | 新版方案（基于 Scope 封装机制） |
| :--- | :--- | :--- |
| **CustomOp 处理** | 未系统支持，依赖默认的逐算子分类 | 通过 `CustomOpAnalysis` 统一分析读写 Buffer、GM/Local 分类及 CoreType |
| **局部 Buffer 关联** | `memref.alloc` 与 `to_tensor` 容易被其他 Pass 割裂或错误沉降 | `packScopeOp` 将 `[alloc, to_tensor, customOp]` 作为一个原子 `scope.scope` 绑定保护 |
| **块规划与排序** | CustomOp 内部中间算子被暴露给 DAG，易引入冗余跨块依赖 | Scope 作为整体节点参与拓扑排序与块合并，排序完成后由 `UnpackScopePass` 统一无损拆包 |
| **属性一致性** | 易出现内部 `alloc` 的 `block_id` 与消费方 `customOp` 不一致 | Scope 解包时统一将 Scope 的 `block_id` 与 `core_type` 覆盖刷写到内部算子 |

---

## 3. 详细处理流程

### Step 1: 分析与分类 (`CustomOpAnalysis`)
- **缓冲区追踪 (`collectBuffers`)**:
  通过 `traceMemDef(operand)` 穿透 `ViewLikeOpInterface` 和 `bufferization::ToTensorOp`：
  - 若来自函数入参 BlockArgument，归类为 `gmBuffers`；
  - 若来自 `memref::AllocOp`，归类为 `localBuffers`。
- **关联算子收集 (`collectRelevantOps`)**:
  - 模式 1 (`buf.hasOneUse()`): `alloc -> to_tensor -> customOp`；
  - 模式 2 (写 buffer): `alloc` 被 `customOp` 写入，随后由单一 `to_tensor` 消费；
  - 若 `to_tensor` 单一用户为 `hivm.ConvertLayoutOp`，一并纳入待打包集合（`ConvertLayoutOp` 是 **tensor/memref 双语义**：ODS 为 `TensorOrMemref:$source` / `TensorOrMemref:$result`，故 `to_tensor -> convert_layout` 的 tensor 形态合法；树里其它 UT 只出现 memref 形态，勿据此判定该分支不可达，UT 必须用 tensor 形态）。
  - 两个放弃打包的前置条件：本地 buffer 有**第二个** `to_tensor` 消费者（打包会撕开活跃值），或 operand 溯源到非 `func` 的 block arg（如 `scf.for` iter_arg，`collectBuffers` 直接 failure -> 该 customOp 整体不打包）。两例均有 UT 固定（`test_pack_memref.mlir` section 6/7）。
- **核心类型映射 (`determineCoreType`)**:
  读取 `hivm.tcore_type` 属性，对应映射为 `CUBE_ONLY`、`VECTOR_ONLY` 或 `CUBE_AND_VECTOR`。注意 `hivm::TCoreType` 实际有 4 个值（`CUBE=1`、`VECTOR=2`、`CUBE_OR_VECTOR=3`、`CUBE_AND_VECTOR=4`），`switch` 的 `default` 会把 **3 和 4 一起**归为 `CUBE_AND_VECTOR` -> 两者都会触发 `OpClassifierPass::handleCubeAndVector` 的双核分裂，`CUBE_OR_VECTOR` 因此被过度分裂（单核即可）。
- **"打 core_type" == 预分类，不是提示**: `PackMemrefPass::packCustomOpUnit` 只在 `ana.coreType != UNDETERMINED` 时落 `ssbuffer.core_type`（`PackMemref.cpp:122-126`：未识别 tcore_type 必须**不打**属性，才能交给 OpClassifier 按数据流染色），而 `OpClassifierPass::initializePass` 把**任何**带该属性的算子塞进 `preClassifiedOps`（此后不再染色），只有 `OP_CUBE_AND_VECTOR` 会被分裂，`OP_UNDETERMINED` 最终被末尾循环强制成 VECTOR -> "不打属性让 OpClassifier 染色"与"打 UNDETERMINED"并不等价，是两种不同结果。 **对 UT 的后果**：输入 IR 里预置 `ssbuffer.core_type` 的旧用例不再测依赖传播——预置即权威，`memref.copy` 不会再被下游 CUBE 染色（`test_plan_compute_block_subview_memory_deps.mlir` 因此必须去掉预置属性）。

### Step 2: 封装 Scope (`packScopeOp`)
- 在待打包算子集合末尾插入 `scope::ScopeOp`；
- 扫描逃逸出该集合的返回值，由 `scope::ReturnOp` 返回，外部消费方无缝替换为消费 `scope.scope` 的返回值；
- 将集合内的算子移入 `scope.scope` 的 body block。

### Step 3: 解包 Scope (`UnpackScopePass`)
- 遍历模块内的所有 `scope::ScopeOp`；
- 调用 `unpackScopeOp`：
  - 将 body 内除 terminator 外的所有算子逐一移动到 `scopeOp` 之前；
  - 用 `scope.return` 的操作数通过 `replaceAllUsesWith` 替换外部对 `scopeOp` 的引用；
  - 销毁 `scopeOp`；
- **属性继承与覆盖**:
  - 若内部算子属性为空或仅含 `kBlockId`（命中 `canReuseAttrs` 分支），直接设为 Scope 的属性集（在 MLIRContext 层面复用属性字典）；
  - 若内部算子包含其它方言固有属性（如 `customOp` 的 `pipe`、`symbol` 等），逐项合并/覆盖 Scope 的属性（确保 `block_id` 和 `core_type` 统一刷新）。

---

## 4. 调试陷阱与注意事项

1. **Pass 方言依赖声明 (`getDependentDialects`)**:
   - **陷阱**: 如果 Pass 会在执行期间动态创建当前 IR 中尚未存在的方言算子（例如 `OpClassifierPass` 动态构造 `scope.scope`），必须在 Pass 类中实现 `getDependentDialects(DialectRegistry &registry)` 并注册 `registry.insert<scope::ScopeDialect>()`。
   - **后果**: 若未声明，当以独立 Pass（如 `triton-opt --op-classifier`）测试输入尚未包含 `scope` 方言的 IR 时，会触发 MLIRContext 断言报错崩溃：`LLVM ERROR: Building op scope.scope but it isn't known in this MLIRContext`。
2. **`packScopeOp` 的支配关系与插入点约束**:
   - 打包算子集合必须满足 SSA 拓扑支配关系；
   - 插入点位于集合中最后一个算子（`ops.back()`），外部用户必须位于其后。若 customOp 结果直接作为 `scf.yield` 的操作数，必须确保插入点与块终止符之间的有效放置顺序。
3. **`hivm::CustomOp` 没有单返回值 trait**: ODS 只给了 `AttrSizedOperandSegments` / `MemoryEffectsOpInterface` / `ExtraBufferOpInterface` / `HIVMStructuredOpInterface` / `HIVMInferCoreTypeInterface`（`HIVMOps.td:766-776`，`let results = res;`），所以“取 customOp 的结果值”必须手写 `getNumResults() == 1` 守卫，不能假定 `getResult()`。多结果 op 的合法形状受 verifier 约束：`#tensor results == #output tensors`（否则 `expected the number of tensor results (N) to be equal to the number of output tensors (M)`），所以造多结果 UT 必须给足 `outs`。同族但**不同 op**：`hivm.hir.custom_macro`（`CustomMacroOp`）——`TypeSwitch<..., hivm::CustomOp>` 不会命中它。
4. **属性复用保护**:
   - 解包时不能盲目使用 `op->setAttrs(attrs)` 覆盖所有内部算子，否则会擦除 `hivm.hir.custom` 上的 `symbol`、`bitcode`、`hivm.pipe` 等专用方言属性。仅在算子原有属性为空或仅含旧 `block_id` 时可直接整组复用。

---

## 5. 职责迁移：CustomOp 分组从 OpClassifier 迁至 PackMemref

### 旧版方案 vs 新版方案
| 维度 | 旧版 | 新版 |
| :--- | :--- | :--- |
| **分组时机** | `OpClassifierPass::groupCustomOps`（PlanComputeBlock 阶段）动态创建 `scope.scope` | `PackMemrefPass`（StandardizeOp 阶段）统一创建 |
| **分类来源** | 分组后立即在内存 `opCoreTypes` 里写死，IR 上无持久标记 | 分组同时把 `ssbuffer.core_type` 落到 scope op 上 |
| **分类消费者** | 仅本 Pass 内部使用 | `OpClassifierPass` 读取该属性并**尊重**它，把 scope 当作原子边界，不再向下传播 |
| **依赖前置** | 无（per-pass 自洽） | 要求 StandardizeOp 阶段先于 PlanComputeBlock 阶段运行 |

### 统一入口：以 alloc 驱动 customOp 与 load 打包
- `PackMemrefPass::runOnOperation` 先分析所有 `hivm.hir.custom`，建立 `local memref.alloc -> CustomOp` 映射；随后遍历模块内所有 `memref.alloc`，对尚未进入任何 `scope::ScopeOp` 的 alloc 统一分派：
  1. 若该 alloc 是某个 customOp 的 local buffer，则打包该 customOp 单元（同一 customOp 只打包一次）；
  2. 否则尝试作为 load 单元打包（`packLoadUnit`）。
- 这样 customOp 与 load 两类单元都由 alloc 作为入口，避免“按 customOp 遍历 + 再按 alloc 遍历”两套入口互相踩踏。

### load 单元打包（def-use 驱动 + 状态返回）
- 针对 `%a = memref.alloc() -> <作用于 %a 的算子> -> %b = bufferization.to_tensor %a`（`%a` 可能经 `subview` / `reinterpret_cast`），在 `PackMemrefPass::packLoadUnit` 中打包为 `scope::ScopeOp` 并打上 `ssbuffer.load`（`CVPipeline::kScopeLoadAttr`）。
- **沿 def-use 链寻找**（不再 `parentOp->walk` / `allocOp->getNextNode()` 线性扫描）：
  - `findToTensorsOver`：从 alloc 的 memref 出发，经 view-like op 反向追踪唯一的 `bufferization.to_tensor`；
  - `collectMemrefUsesBefore`：从该 memref 出发沿 SSA 用户遍历，取所在 block 的顶层祖先，并只保留位于 to_tensor 之前的算子。
- `<作用于 %a 的算子>` 上溯到所在 block 的顶层算子后，必须满足「是带 `hivm::ExtractLoadStoreAttr` 的 loop」或「其嵌套 user 所在 region 仅含该单个算子」，否则放弃打包。
- 需随包的 `subview` / `reinterpret_cast` / 标量通过 `ScalarClosure` 收集：调用 `closure.getScalarOps()` 获得按支配序排好的 `ArrayRef`。**有 scope 外消费者时：克隆体插在原算子位置、随包进入 scope，原算子留在 scope 外继续服务外部消费者**（与 `ScalarClosure::capture` 同一约定，见 `PackMemref.cpp:365`）；无外部消费者则直接把原算子并入 scope。
- 克隆规则**沿 def-use 链向后传播**：被克隆算子的原算子留在 scope 外，仍要读自己的操作数，所以喂给它的闭包成员也必须克隆。`buildPackList` 用不动点求这个集合，rewire 只改「随包搬入」的消费者（`inUnit` 与 `inClosure` 减克隆集），被克隆的原算子保持原操作数 —— 否则原算子会读到已搬入 scope 的值，报 `operand #N does not dominate this use`。
- 闭包成员本身是已打包的 `ssbuffer.load` scope 时留在 scope 外（不嵌套、不克隆）：新的 scope 插在单元末尾，嵌套搬运会让被搬 scope 的结果支配不到它原有的消费者。
- `ScalarClosure` 若在闭包内触及另一个 `memref.alloc(alloca)`（正常另一个 memref 应为 func arg / GM buffer），**兜底处理：把该标量/视图算子留在 scope 外，其余照常打包，不报异常**（这不是 bug）。保留在 scope 外可保证支配关系（scope body 仍可引用它），并避免把两个 local buffer 强行合入同一单元。
- 返回值 `PackUnitStatus` 区分三种情况：`Packed`（成功）、`NotALoadUnit`（与 load 无关，静默忽略）、`Failed`（形似 load 单元但模式不一致，例如嵌套 user 所在 region 含多个算子）。`Failed` 会让 `PackMemrefPass` 调用 `signalPassFailure()`，由外层 `AddDynamicCVPipeline` 记录 fallback，而不是让 Unpacked buffer 被后续 Pass 错误调度。
- 失败不是 core dump：`signalPassFailure()` 是 graceful failure；`NotALoadUnit` 与“闭包含 foreign alloc”的兜底均不触发失败。
- `Failed` 路径用 `emitError` 定点报告，消息自带 offending op 文本；`triton-opt` 默认还会附一条 `note: see current operation`，但该 note 受 `-mlir-print-op-on-diagnostic` 控制，因此不要把 op 的可见性依赖于它。
- 若该 load 单元由 `ExtractedLoadOrStore` loop 驱动，则 scope 直接标记 `ssbuffer.core_type = "VECTOR"`（`ExtractedLoadOrStore` 恒为 Vector 核），其余 load scope 不带 core_type，交由 `OpClassifierPass` 依 matmul 消费关系判定为 CUBE。

### OpClassifier 对 load 的处理
- `OpClassifierPass` **不再识别 to_tensor 中的 load 语义**（已删除 `isExtractedLoadStoreRelated`）。`matchLoadPattern` 只做两件事：
  1. 若输入是带 `ssbuffer.load` 的 `scope::ScopeOp`，视为 load 单元并按 matmul 消费关系标记 CUBE；
  2. 若输入是裸 `bufferization.to_tensor`，仅作为上游异常输入的兜底（保留 `MayImplicitTransposeWithLastAxis` 保护），不再判断它是否来自 ExtractedLoadOrStore。
- 若 load 未被 `PackMemrefPass` 正确打包而走到兜底分支，属于 `PackMemrefPass` 的 bug，OpClassifier 不再代为修正。

### Scope 的原子性贯穿 PlanComputeBlock
- `OpClassifierPass::initializePass` 必须使用 `WalkOrder::PreOrder` 才能让 `WalkResult::skip()` 真正剪掉 scope body（`PostOrder` 下 skip 不生效，会误把 body 内算子也纳入分类，甚至触发 CUBE_AND_VECTOR 克隆而撕碎 scope）。
- `PlanCubeBlockPass` / `PlanVectorBlockPass` 遍历 Block 时，若 `block->getParentOp()` 是 `scope::ScopeOp` 则直接 `WalkResult::skip()`，不向 body 内分配 block_id。
- `getCoreTypeOfSimpleOpOrCf` 对 `scope::ScopeOp` 直接返回其自身 `ssbuffer.core_type`，不再降到 body 推导（body 内算子没有 core_type 标记）。

### 原子性只延伸到 PlanComputeBlock：ComputeBlockOpt 的适配 / 删除结果
- `ComputeBlockOpt/` 对 `scope::ScopeOp` 的依赖点共三条，均已处理（统一用 `CVPipeline::isLoadScope`，索引见 `AGENTS/Utils.md`）：
  - `ExpLoadPatternPass`：`coreOps` 的 load 部分改为“载体”概念（`loadOps`）——packed 时为整个 scope，未打包兜底时仍为 `to_tensor/copy/alloc`；搬块时 scope 只走 `bm.updateBlockId`，**不得** `updateBlockIdWithInner`（否则 scope body 被打上 block_id，破坏原子性）。UT：`ComputeBlockOpt/exp_load_pattern.mlir`（packed + raw 两例）。
  - `MergeComputeBlockPass`：跨 CUBE 克隆的载体从“`to_tensor` 是 CubePre 的 block 成员”改为 `isTensorCarrier`（`to_tensor` 或 load scope），克隆时整 scope 一起克隆，且 `cloneOpCrossCubeDep` 打 id 时跳过 scope body。UT：`ComputeBlockOpt/merge_compute_block_scope.mlir`。
- **已删除 `UnifyAllocBlockPass`**：`alloc + scf.if{linalg.fill} + subview/copy` 现在整体就是一个 scope，block_id 统一是结构上自明的。删除前它在 packed IR 上是**真实回归**（不只是空转）：`getCommonBlockId` 在 scope body 里拿不到 copy 的 block_id（“There are not CopyOp!”）→ pass 返回 failure → `setFallbackAttr(ERRCODE_FAILED)` → 整个 `--compute-block-opt` fallback；同一 kernel 不打包（raw）则正常。
- **已删除 `MoveLoadIntoUserPass`**：实测（同一 kernel raw vs packed 的 A/B）raw 下三个 load 载体全部与 consumer 不同块（matmul 3≠2、addf 4≠5），packed 下全部相等 -> planner 已把 scope 与它的 consumer 放进同一 block，该 pass 无工作可做；另 5 个真实 kernel（masked load / 循环累加 / if 分支 / gather / gather+matmul）在 packed 下 carrier-consumer 也全部同块。
- **残留风险（已知、接受）**：被 PackMemref 跳过、仍留在裸形状的 GM load 单元不再被 unify / 不再被搬进 user block。按现有约定“读到 GM 却未打包”属于 PackMemref 的 bug，所以这里不再保留兼顾两套形状的代码。
- 探针（判断“某个分支是否仍依赖 op 是 block 成员”而无需改 pass）：把载体 op 的 `ssbuffer.block_id`（或它的 `ssbuffer.load`）去掉再跑单 pass —— `merge_compute_block_scope.mlir` 去掉 scope 的 `ssbuffer.load` 后，block 10 的克隆从 2 处塌成 1 处。


### handleCubeAndVector 的拆分与收敛（已实现）
- **复制判据**：仅 `opCoreTypes[op] == OP_CUBE_AND_VECTOR`，无其他条件。上游 / packed scope 的单核标识（`CUBE` / `VECTOR`）是权威，**永不复制**；其结果的跨核消费由依赖链承担，不由 OpClassifier 从使用者反向推导。
- 上游没提供 core type 的算子：Step 1–6 的传播决定类型；若两核都用，传播会把它置为 `CUBE_AND_VECTOR`，随后被复制。
- **pre-classified 在复制前不被任何步骤修改**：Step 1–6 的传播（含 `propagateVectorUpstream` 跳过 frozen）都不得写 frozen op；`handleCubeAndVector` 只读 `opCoreTypes` 拍一次快照。frozen 的 `CUBE`/`VECTOR` 不是 `CUBE_AND_VECTOR`，故不进入复制集。
- **TODO（单核跨核消费，最小修改暂不做）**：单核 pre-classified op 被另一核直接消费时当前不复制；若将来需要，可在依赖链上做反向驱动的复制。`@shared_load_scope_pre_classified_cube/_vector`、`@cube_scope_with_cav_downstream` 固定这一行为。
- 拆分路径（`splitOperationForCubeAndVector`）：先递归处理 `CUBE_AND_VECTOR` operand，克隆体按 `opToVectorClone` 重映射 operand；随后原算子 → `CUBE_ONLY`、克隆体 → `VECTOR_ONLY`，VECTOR 使用者重定向到克隆体，并在 `CloneOpMap` 双向登记。
- **传播 seed**：`propagateCubeUpstream` 的 seed = `cubeSeeds` ∪ 已打 `CUBE` 标签的 pre-classified 算子；`propagateVectorUpstream` 本身就以所有 `VECTOR_ONLY` 算子为 seed，天然包含已打标签的算子。
- **Step 8 的 frozen 防护已移除（TODO）**：`processYieldOperation` 只处理 `scf.yield` / `scf.if` / `scf.for`，它们不会被 PackMemref 打成 packed scope，因此 pre-classified op 到不了这里；若将来 pre-classified op 可能是 scf op，需把守卫加回。

### 历史挂死（反例，必须避免）
- **Symptom**: `triton-opt --plan-compute-block` 在 Step 7 前纯 CPU 挂死，无 IR 变化。
- **Cause**: 拆分入口先把 `op` 插入 `processedOps`，随后调用的辅助函数又对**同一集合**做去重，必然立即 return → scope 永不拆分 → 收集条件恒真 → 循环的 `changed` 恒真。
- **Fix**: 去重集合**只由一个函数维护**；现改为“复制前只读快照 + 单次遍历”，从结构上消除死循环。
- **Scope**: `handleCubeAndVector` / `splitOperationForCubeAndVector`。
- **Verification**: `test_plan_compute_block_shared_load_scope.mlir` 通过且不挂死。

### 调试陷阱
- `ScalarClosure` 持有 `ArrayRef<Operation*>` / `Block*`，不可将其实例跨作用域带出；外部（如 `PackMemrefPass`）应在其存活范围内通过 `getScalarOps()` 取用排序结果。
- `ScalarClosure::isBefore` 对“同一 multi-region op 的兄弟 region”（如单个 `scf.if` 的 then/else）之间的两个 op 判为无序并返回 false（兄弟 region 之间没有 SSA 支配序）。这违反 `ScalarClosure` 的前置条件：**由调用方保证不会收集这种集合**，实现不断言、不报错。
- `packScopeOp` 不校验支配关系，调用方必须保证打包列表按 SSA 拓扑序、且所有逃逸值的使用者位于 `ops.back()` 之后。
- 删除 OpClassifier 中前置的 `preClassifiedOps` 判定会让已有 `ssbuffer.core_type` 的算子被重新分类；对 packed scope 而言这是有意为之的“权威标记”，写 UT 时不要给待分类算子预留 VECTOR/CUBE 属性。（例外：`CUBE_AND_VECTOR` 仍会被拆分，因其语义就是两核各一份。）
