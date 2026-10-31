# Dynamic CV Pipeline Utilities & Helper Functions (Utils.md)

This document indexes reusable auxiliary functions and helpers across `DynamicCVPipeline`, noting their current implementation locations for easy reuse or future refactoring into shared headers.

C++ helpers only. Local CLI scripts (build / test / refactor tooling) are indexed in `AGENTS/LocalTooling.md`.

---

## 1. Loop Construction & Migration Helpers
Located in:
- Header: `third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h`
- Implementation: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/Utils.cpp`

### `createNewForOpWithExtras`
```cpp
scf::ForOp createNewForOpWithExtras(scf::ForOp oldForOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:363`
- **Purpose**: Creates a new `scf.for` with matching bounds/step/attributes, with `extraInitArgs` appended to the end of original `initArgs`. Returns `oldForOp` unchanged if `extraInitArgs` is empty.

### `createNewWhileOpWithExtras`
```cpp
scf::WhileOp createNewWhileOpWithExtras(scf::WhileOp oldWhileOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:386`
- **Purpose**: Creates a new `scf.while` with extra inits appended, allocating empty before and after blocks.

### `createMainLoopOpWithExtras`
```cpp
Operation *createMainLoopOpWithExtras(Operation *oldOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:426`
- **Purpose**: Polymorphic dispatcher for `createNewForOpWithExtras` or `createNewWhileOpWithExtras`.

### `migrateBody`
```cpp
void migrateBody(Block *oldBlock, Block *newBlock);
```
- **Location**: `Utils.cpp:294`
- **Purpose**: Rewires `oldBlock` arguments to `newBlock` matching arguments via `replaceAllUsesWith`, and splices all non-terminator operations into `newBlock`.

### `migrateWhileBodies`
```cpp
void migrateWhileBodies(scf::WhileOp oldWhileOp, scf::WhileOp newWhileOp);
```
- **Location**: `Utils.cpp:307`
- **Purpose**: Calls `migrateBody` on both `before` and `after` regions of an `scf.while`.

### `buildNewYieldOp`
```cpp
LogicalResult buildNewYieldOp(Block *oldBlock, Block *newBlock, Operation *newOp, ArrayRef<Value> extraYieldValues);
```
- **Location**: `Utils.cpp:314`
- **Purpose**: Copies original yield operands from `oldBlock`'s terminator, appends `extraYieldValues`, creates new `scf::YieldOp` at end of `newBlock`, and erases old yield.

### `replaceOpResultUses`
```cpp
void replaceOpResultUses(Operation *oldOp, Operation *newOp);
```
- **Location**: `Utils.cpp:333`
- **Purpose**: Replaces all uses of `oldOp` results with matching results from `newOp.getResults().take_front(numOldResults)`.

---

## 2. Block & Attribute Identification Helpers

### `getIfBlockId`
```cpp
static int getIfBlockId(scf::IfOp ifOp, int &outBlockId);
```
- **Location**: `UpdateConditionInfo.cpp:77`
- **Purpose**: Reads `IntegerAttr` with name `CVPipeline::kIf` (`"ssbuffer.if"`). Returns error if attribute is missing.

### `collectSSBufferIfOps`
```cpp
static void collectSSBufferIfOps(scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps);
```
- **Location**: `UpdateConditionInfo.cpp:88`
- **Purpose**: Walks `forOp` body and collects all `scf::IfOp` with attribute `kIf`.

### `countUniqueIfBlockIds`
```cpp
int countUniqueIfBlockIds(Operation *loopOp);
```
- **Location**: `Utils.cpp:215`
- **Purpose**: Walks operations inside `loopOp`, collecting unique `ssbuffer.if` block IDs.

### `isMainLoopOp`
```cpp
inline bool isMainLoopOp(Operation *op);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h:176`
- **Purpose**: Checks if an op is `scf::ForOp` or `scf::WhileOp` with attribute `CVPipeline::kMainLoop`.

### `hasFallbackAttr` / `setFallbackAttr`
```cpp
inline bool hasFallbackAttr(ModuleOp module);
inline void setFallbackAttr(ModuleOp module, int errorCode);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h:80-95`
- **Purpose**: Checks or sets fallback flag on `ModuleOp` when dynamic CV pipeline compilation fails.

---

## 3. SSA Mapping & Constant Helpers

### `getLatestValue`
```cpp
static Value getLatestValue(const DenseMap<Value, Value> &map, Value key);
```
- **Location**: `UpdateConditionInfo.cpp:71`
- **Purpose**: Safely resolves the most recent SSA value for an iter_arg or control variable from `controlVarToLatestValue`. Returns `key` if no updated SSA value exists.

### `createIntConst`
```cpp
static Value createIntConst(OpBuilder &builder, Location loc, int64_t val, Type type);
```
- **Location**: `UpdateConditionInfo.cpp:76`
- **Purpose**: Helper to create `arith::ConstantIntOp` matching the bitwidth of `type` (defaults to 32 if not integer type). Extracts result `Value` directly.

---

## 4. Graph & Buffer Analysis Helpers

### `buildBufferDependencyMappings`
```cpp
static int buildBufferDependencyMappings(
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &buffers,
    DenseMap<Operation *, SmallVector<int>> &consumerToGroups,
    DenseMap<Operation *, SmallVector<int>> &outputToGroups);
```
- **Location**: `UpdateConditionInfo.cpp:189`
- **Purpose**: Generates reverse lookup tables (`consumerToGroups` and `outputToGroups`) from buffer dependency structures for $O(1)$ query during IR traversal.

### `buildOutputGroups`
```cpp
int buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar,
    SmallVector<OutputGroupInfo> &outputGroups);
```
- **Location**: `UpdateConditionInfo.cpp:365`
- **Purpose**: Deduplicates and groups producer operations and their associated control variables to generate capacity constraints (`var < group.outputs.size()`).

### `dfsTopologicalSort`
```cpp
static LogicalResult dfsTopologicalSort(
    Operation *op, DenseSet<Operation *> &visited, DenseSet<Operation *> &inStack,
    const DenseSet<Operation *> &ops, DenseMap<Operation *, int> *opOrder,
    SmallVectorImpl<Operation *> &sorted);
```
- **Location**: `Utils.cpp:89`
- **Purpose**: Topologically sorts operations within an `scf.for` body based on data dependencies, reporting cycles if present.

### `filterCrossCoreMapByForOp`
```cpp
static ConsumerProducerMap filterCrossCoreMapByForOp(scf::ForOp forOp, ConsumerProducerMap &crossCoreMap);
```
- **Location**: `UpdateLoopIterTimes.cpp:61`
- **Purpose**: Filters cross-core producer-consumer entries to only those where the consumer op resides within `forOp`.

### `getOtherScopeMainloop`
```cpp
static scf::ForOp getOtherScopeMainloop(ModuleOp module, bool currentIsCube, bool currentIsVector, int mainLoopId);
```
- **Location**: `UpdateLoopIterTimes.cpp:76`
- **Purpose**: Locates the counterpart main loop on the opposite core scope sharing the same `mainLoopId`.

---

## 4. Scope Packaging & Memory Alias Utilities

### `packScopeOp`
```cpp
scope::ScopeOp packScopeOp(llvm::ArrayRef<Operation *> ops);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h:42` / `ScopeOpUtils.cpp:37`
- **Purpose**: Packs a list of operations into a newly created `scope::ScopeOp` at the insertion point of the last op (`ops.back()`). Identifies values escaping the packed set and routes them through `scope::ReturnOp`, redirecting external consumers. Returns `nullptr` on failure.

### `unpackScopeOp`
```cpp
llvm::LogicalResult unpackScopeOp(scope::ScopeOp scopeOp, llvm::SmallVectorImpl<Operation *> *movedOps = nullptr);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h:48` / `ScopeOpUtils.cpp:84`
- **Purpose**: Moves all operations inside `scopeOp` body to the parent block immediately before `scopeOp`, replaces uses of `scopeOp` results with the terminator's operands, and erases `scopeOp`. Optionally populates `movedOps`.

### `isLoadScope`
```cpp
inline bool isLoadScope(Operation *op);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h:49`
- **Purpose**: 判断 `op` 是否为 PackMemref 产出的原子 load 单元（带 `kScopeLoadAttr` 的 `scope::ScopeOp`）。
- **Consumers**: `ExpLoadPatternPass::matchExpLoadPattern`（load 单元作为 `coreOps` 成员）、`MergeComputeBlockPass::isTensorCarrier` / `cloneOpCrossCubeDep`（跨 CUBE 克隆载体 + 克隆后只给 scope 打 block_id）。

### `ScalarClosure`
```cpp
class ScalarClosure; // mlir::CVPipeline
llvm::ArrayRef<Operation *> ScalarClosure::getScalarOps();
```
- **Location**:
  - 类声明：`third_party/ascend/include/DynamicCVPipeline/Common/ScalarClosure.h`
  - 类实现（`collectOuterDependency` / `collectScalarClosure` / `collect` / 构造函数 / `isBefore` / `getScalarOps` / `capture`）：`third_party/ascend/lib/DynamicCVPipeline/Common/ScalarClosure.cpp`
- **Purpose**: `ScalarClosure` 沿 SSA 逐级收集「直到第一个张量依赖」之前的标量/视图算子（`subview` / `reinterpret_cast` / 索引标量），结果存于 private 的 `scalarOps`。
  `getScalarOps()` 返回按支配序排序的 `ArrayRef` 并重写内部 `SetVector`；`capture()` 也内部调用它。
- **排序实现（当前）**: `getScalarOps()` 把 `scalarOps` 拷贝成 `SmallVector`，用 `llvm::sort(..., isBefore)` 排序后重建 `SetVector`（保证重复调用得到稳定、已排序的底层存储）。`isBefore(a, b)` 是成对比较：先用 `lowestCommonAncestorBlock`（双指针法，O(1) 额外空间）求最近公共祖先 block，再用 `commonBlock->findAncestorOpInBlock(*op)` 把双方映射到该 block 的祖先并按 block 内序比较；若两 op 的祖先是同一个，则退化为各自 block 内序。
- **多 region 约束（前置条件，由调用方负责）**: 上述块序无法比较“同一个多 region op 的不同 region（如一个 `scf.if` 的 then/else）”内的两个 op —— 它们映射到同一祖先、没有 SSA 支配关系。`ScalarClosure` 只从 `block`/`parentBlock` 收集，天然不触发；外部若传入任意 op 集合必须自行保证。该情况下 `isBefore` 两个方向都返回 `false`，即 `false` 重载了“`b` 在 `a` 前”与“无序”两种语义。**不做运行时检查、不 assert、无 pre-order 兜底**（早期的 `findAncestorsInCommonBlock` / `buildPreOrderIndex` 兜底已删除）。
- **生命周期**: `ScalarClosure` 持有 `ArrayRef<Operation*>` / `Block*`，传入的 `ops` 容器与 block 必须比该实例存活更久；调用方应在同一作用域内 `collect()` 后立即通过 `getScalarOps()` 取用。
- **未来计划**: 拟改为自由函数 `makeClosure` / `captureClosure`（`capturedOps`、`eraseDeadOriginals`），详见 [Closure.md](DynamicCVPipeline/Closure.md)。**当前未实施。**

### `getAliasSource` / `traceMemDef`
```cpp
Value getAliasSource(Value val);
inline Value traceMemDef(Value val);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h:335-343` / `Utils.cpp:548`
- **Purpose**:
  - `getAliasSource`: Resolves the underlying memory source through `ViewLikeOpInterface` (via `getViewSource()`) and `bufferization::ToTensorOp` (via `getBuffer()`).
  - `traceMemDef`: Repeatedly follows `getAliasSource` in a loop until reaching the root definition (e.g., `memref.alloc` or function `BlockArgument`).

### `CustomOpAnalysis::get`
```cpp
static llvm::FailureOr<CustomOpAnalysis> CustomOpAnalysis::get(hivm::CustomOp customOp);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Analysis.h:81` / `CustomOpUtils.cpp:144`
- **Purpose**: Analyzes `hivm::CustomOp` to inspect memory effect buffers (separating GM buffers from local buffers), determines execution core type (CUBE / VECTOR), and collects closely-coupled operations (alloc, to_tensor, convert_layout) for unified scheduling.


---

## 4. SplitIfByBlockId hoist safety helpers
Located in: `third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/SplitIfByBlockId.cpp` (all `static`, `SplitIf` namespace scope).

### `isInternalValue`
```cpp
static bool isInternalValue(Value val, scf::IfOp ifOp);
```
- **Location**: `SplitIfByBlockId.cpp:1716`
- **Purpose**: True when `val` is produced by an op nested in `ifOp`'s regions, or is a block argument of such a region. Used to distinguish "computed inside the if" from "defined outside".

### `isDisposablePlaceholder`
```cpp
static bool isDisposablePlaceholder(Value val, scf::IfOp ifOp);
```
- **Location**: `SplitIfByBlockId.cpp:1740`
- **Purpose**: True when an internal value is a provably disposable else-branch placeholder emitted by `createPlaceholderValue` (`arith.constant` / `tensor.empty` / `linalg.fill` / `memref.alloc(a)`, or a `memref.reinterpret_cast` over one). Rejects `ensureLocalValue` ViewLike clones and Scene 3/4 absorbed real ops, which carry semantics.

### `canSafelyRedirectToUpstreamIf`
```cpp
static bool canSafelyRedirectToUpstreamIf(scf::IfOp upstreamIfOp, unsigned upResIdx,
                                          scf::IfOp curIfOp, Value elseVal,
                                          DominanceInfo &domInfo);
```
- **Location**: `SplitIfByBlockId.cpp:1773`
- **Purpose**: Validates that an earlier split-if's else-yield may be patched in place with `elseVal` for result `upResIdx`: same block / condition / `ssbuffer.splitted_if` tag, valid index, type match, `elseVal` properly dominates the upstream else-yield, and the current operand is either `elseVal` or a disposable placeholder.

### `ifBodiesHaveRealWork`
```cpp
static bool ifBodiesHaveRealWork(scf::IfOp ifOp);
```
- **Location**: `SplitIfByBlockId.cpp:1806`
- **Purpose**: True when a then/else body contains anything besides its terminator and disposable placeholders. Guards the Case-B full-hoist path from erasing live computation.

### `canSafelyHoistRedirectedOutput`
```cpp
static bool canSafelyHoistRedirectedOutput(scf::IfOp ifOp, scf::YieldOp thenYield,
                                           scf::YieldOp elseYield);
```
- **Location**: `SplitIfByBlockId.cpp:1832`
- **Purpose**: Combined safety gate for `hoistAndUnifyRedirectedOutput`. Returns false when any hoisted slot would drop a real else value, patch an unrelated upstream if, break dominance/type, or (Case B) erase live body work. Callers must leave the IR untouched when it returns false.
