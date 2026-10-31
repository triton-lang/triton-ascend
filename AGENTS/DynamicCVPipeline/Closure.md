# Closure 统一设计与决策 (Closure.md)

> 本文记录 `DynamicCVPipeline` 中「标量/memref 依赖闭包」的**未来统一计划**。
> 讨论来源：`ask_oracle.md`（用户提议 A/B/C + 专家意见 + pi agent 梳理）。
> **状态：未来计划，未实施。当前代码仍为 `ScalarClosure` 类 + 各处手写物化；本文件仅记录方向。**

---

## 1. 背景

`SplitIfByBlockId` 与 `PackMemref` 都需要同一件事：

1. 沿 SSA 收集「到第一个 `tensor` 依赖为止」的 memref/scalar 算子（下称 **闭包**）；
2. 把闭包**物化**到目标区域：克隆依赖、改写内部 operand、排序；
3. 目标区域之外若有使用者，保留原算子。

现状问题：
- 收集逻辑已由 `ScalarClosure` 复用，但**物化**在两处各写一遍；
- `ScalarClosure::capture(int blockId)` 只克隆、不 rewire，且耦合 `ssbuffer.block_id`；
- `PackMemref::packLoadUnit` 手写物化存在**两个 SSA dominance bug**（见第 4 节）。

---

## 2. 决策 `[by user + pi agent]`

| 编号 | 决策 | 说明 |
| :-- | :-- | :-- |
| **D1** | **取消 `ScalarClosure` 类，改为自由函数 `makeClosure` / `captureClosure`** | 无多态、无跨调用状态；`isBefore` 本身无状态。自由函数消除 `ScalarClosure` 持有 `ArrayRef ops` / `Block*` 的生命周期隐患。 |
| **D2** | `capture` 结果字段改名 `clonedOps` → **`capturedOps`** | 语义为 `allOps \ roots`，即「本次物化进目标区域的算子」，不区分是复制（clone）还是移动（clone+erase）。 |
| **D3** | **不引入 `shouldCollect` 收集钩子** | `memref.alloc/alloca` 跳过是闭包的**定义性安全不变量**，硬编码保留；`PackMemref` 的 foreign-alloc 排除留在调用方（过滤 `makeClosure` 的结果）。 |
| **D4** | **通用工具移除 `blockId`** | `ssbuffer.block_id` 是 Pass 级属性，由调用方对 `capturedOps` 打标。 |
| **D5** | **统一物化算法：向内克隆 + 内部 rewire + 可选 `eraseDeadOriginals`** | `eraseDeadOriginals=true` 时，「克隆后删除已无使用者的原算子」等价于「把原算子移入 scope」，同时修复第 4 节两个 dominance bug。 |

---

## 3. 目标 API

```cpp
// Common/Closure.h
namespace mlir::CVPipeline {

struct ClosureResult {
  /// roots ∪ capturedOps，按支配序排列，内部 operand 已完成 rewire。
  SmallVector<Operation *> allOps;
  /// 本次物化出的算子（allOps \ roots）。
  SmallVector<Operation *> capturedOps;
  bool empty() const { return capturedOps.empty(); }
};

/// 收集可达的非 tensor defining op（跳过 tensor 边界与 memref.alloc/alloca），
/// 仅接受位于 `block`（`includeParent` 时含 `parentBlock`）的算子，返回支配序列表。
SmallVector<Operation *> makeClosure(Block *block, ArrayRef<Operation *> roots,
                                     bool includeParent);

/// 跨嵌套 block 的支配序比较（双指针 LCA + 公共 block 祖先 + block 内序）。
bool isBefore(Operation *a, Operation *b);

/// 物化闭包：克隆 `closureOps` 中不属于 `roots` 的算子，插入并 rewire
/// `roots ∪ capturedOps` 的 operand，排序返回；`eraseDeadOriginals=true` 时
/// 逆拓扑删除因克隆而失效的原算子。
ClosureResult captureClosure(Block *block, ArrayRef<Operation *> roots,
                             ArrayRef<Operation *> closureOps,
                             bool eraseDeadOriginals = false);

} // namespace mlir::CVPipeline
```

---

## 4. `PackMemref` 现状的两个 dominance bug

根因：现有实现走「**原算子移入 scope，外部使用者改指向 scope 外克隆**」的反向路径，原算子最终的支配位置不确定。

### Bug 1：外部使用者位于 `scalarOp` 与 `allocOp` 之间
```
%s1  = arith.addi %a, %b      // 原 scalarOp
%ext = arith.muli %s1, %c     // 外部使用者，位于 allocOp 之前
%alloc = memref.alloc()       // allocOp（clone 统一插在它前面）
```
clone 插在 `%alloc` 前 → `%ext` 引用了排在它后面的 clone → 破坏 SSA dominance。

### Bug 2：闭包内链 `S1 -> S2`，外部只用到 `S2`
```
S1 -> S2 -> UnitOp
ExtOp 只用 S2
```
- `S1` 无外部使用者 → 不克隆，被移入 scope；
- `S2` 有外部使用者 → 在 `allocOp` 前克隆 `cloned_S2`，仍引用原始 `S1`；
- `packScopeOp` 把 `S1` 移入 scope 后只能 yield 成 scope 返回值，导致位于 `allocOp` 前的 `cloned_S2` 引用了 `to_tensor` 处的 scope 返回值 → 破坏 SSA dominance。

### 统一后的修法
采用 D5：「统一把闭包算子克隆进目标区域 + 只 rewire 内部使用」，则：
- 有外部使用者的原算子留在原地不动（外部继续用原件）；
- 无使用者的原算子由 `eraseDeadOriginals` 逆拓扑删除（等价于移入 scope）；
- 不再存在「外部使用者在原算子与插入点之间」或「跨层引用 scope 返回值」的问题。

---

## 5. 调用方用法

### 5.1 `SplitIfByBlockId`
```cpp
ScalarClosure closure{block, allOps, /*includeParent=*/true};  // 现状
// 改为：
auto closureOps = makeClosure(block, allOps, /*includeParent=*/true);
auto res = captureClosure(block, allOps, closureOps, /*eraseDeadOriginals=*/false);
if (res.empty())
  return;
for (Operation *op : res.capturedOps)
  op->setAttr(CVPipeline::kBlockId, /* group.blockId */);
// 用 res.allOps 重建 group.ops / group.nestedIfs
```
行为与现状一致（`eraseDeadOriginals=false` + 无 filter 时完全等价）。

### 5.2 `PackMemref::packLoadUnit`
```cpp
auto closureOps = makeClosure(block, unitOps, /*includeParent=*/false);
// 调用方过滤 foreign-alloc 消费者（保持现有行为；不做收集钩子）
llvm::erase_if(closureOps, [&](Operation *op) { return touchesForeignAlloc(op); });
auto res = captureClosure(block, unitOps, closureOps, /*eraseDeadOriginals=*/true);
auto scopeOp = packScopeOp(res.allOps);
```
不再手写 clone/rewire 循环，且修复 Bug 1/2。

### 5.3 `ExpLoadPatternPass` / `FixpipeOptPass`
只用到收集结果（插入 `SetVector`），改为：
```cpp
for (Operation *op : makeClosure(block, coreOps, /*includeParent=*/false))
  matchedOps.insert(op);
```

---

## 6. 实施顺序与状态

**现状（未实施）**：`ScalarClosure` 类保持原样（`collect()` / `getScalarOps()` / `isBefore()` / `capture(int blockId)`）；`PackMemref` 仍手写物化（因此仍有第 4 节两个 dominance 隐患）。本节各步骤仅作为未来计划。

1. [ ] 新增 `Common/Closure.{h,cpp}`（自由函数），删除 `ScalarClosure.{h,cpp}`；更新 4 个调用方 include。
2. [ ] `SplitIfByBlockId` 切到 `makeClosure` / `captureClosure(false)`，UT 全绿。
3. [ ] `PackMemref` 切到 `makeClosure` + 调用方 foreign-alloc 过滤 + `captureClosure(true)`，UT 全绿（同时修复第 4 节两个 bug）。
4. [ ] `ExpLoadPatternPass` / `FixpipeOptPass` 切到 `makeClosure`。
5. [ ] 更新 `AGENTS/Utils.md`（`ScalarClosure` 条目改为自由函数）与本文件。
6. [ ] 回归：`PlanComputeBlock` / `StandardizeOp` / `SplitDataflow` / `ComputeBlockOpt` 全部 UT。

### 待确认
- `captureClosure` 的插入点规则：为保证 dominance，克隆体在 `block` 内时插到原算子之后，否则插到 `block` 内最早的 root 之前；需用 UT 固定行为。
- `eraseDeadOriginals` 用 `mlir::isOpTriviallyDead` 判定（保守：删不掉就保留，不影响正确性）。
- 有外部使用者的闭包算子会「外部保留原件 + scope 内保留克隆」，语义等价但重复计算；当前两处调用方均可接受。
