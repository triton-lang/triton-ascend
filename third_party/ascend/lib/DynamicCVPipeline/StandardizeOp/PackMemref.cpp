/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/Analysis.h"
#include "ascend/include/DynamicCVPipeline/Common/ScalarClosure.h"
#include "ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "pack-memref";
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    llvm::dbgs() << '[' << DEBUG_TYPE << "] ";                                 \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

/// Outcome of a single packing attempt. Reporting goes through the pass-level
/// `--debug-only=pack-memref` trace rather than a diagnostic, because an
/// unpacked allocation keeps the OpClassifierPass fallback and therefore never
/// makes the pass fail.
enum class PackUnitStatus {
  /// The allocation was packed into a scope.
  Packed,

  /// The allocation is not a load unit (or does not form one consistently); it
  /// is left untouched and the reason is logged.
  Skipped,
};

// The block directly containing `op` holds nothing but `op` (terminators
// excepted).
static bool isOnlyOpInItsBlock(Operation *op) {
  Block *block = op->getBlock();
  if (!block) {
    return false;
  }
  for (Operation &nested : *block) {
    if (&nested == op || nested.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    return false;
  }
  return true;
}

namespace {

class PackMemrefPass
    : public PassWrapper<PackMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackMemrefPass);

  PackMemrefPass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<scope::ScopeDialect, hivm::HIVMDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pack-memref";
  }
};

} // namespace

// Group hivm.hir.custom together with its own local buffers into a scope unit.
static void packCustomOpUnit(hivm::CustomOp customOp);
// Pack `memref.alloc -> <ops on it> -> bufferization.to_tensor` load units.
static PackUnitStatus packLoadUnit(memref::AllocOp allocOp);
// Two-pass module driver: custom-op units first, then the remaining load units.
static void packAllCustomOpUnits(ModuleOp moduleOp);
static void packAllLoadUnits(ModuleOp moduleOp);

void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  packAllCustomOpUnits(moduleOp);
  packAllLoadUnits(moduleOp);
}

// Pack every hivm.hir.custom together with its local buffers. Each custom op
// owns its buffers, so claim those allocations before load-unit packing looks
// at them.
static void packAllCustomOpUnits(ModuleOp moduleOp) {
  SmallVector<hivm::CustomOp> customOps;
  moduleOp.walk(
      [&](hivm::CustomOp customOp) { customOps.push_back(customOp); });
  for (hivm::CustomOp customOp : customOps) {
    packCustomOpUnit(customOp);
  }
}

// Packs the remaining allocations as load units. Every allocation that does not
// form a load unit consistently is skipped with a `--debug-only=pack-memref`
// trace: leaving it unpacked is safe, because OpClassifierPass still recognises
// the raw `bufferization.to_tensor` chain.
static void packAllLoadUnits(ModuleOp moduleOp) {
  SmallVector<memref::AllocOp> allocs;
  moduleOp.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });

  for (memref::AllocOp allocOp : allocs) {
    // Buffers already swallowed by custom-op packing are done.
    if (allocOp->getParentOfType<scope::ScopeOp>()) {
      continue;
    }
    packLoadUnit(allocOp);
  }
}

/// Pack a custom op together with its buffers into one `scope::ScopeOp`
/// carrying the analysed core type.
static void packCustomOpUnit(hivm::CustomOp customOp) {
  auto anaRes = CustomOpAnalysis::get(customOp);
  if (llvm::failed(anaRes)) {
    LDBG("warning: failed to analyze " << customOp);
    return;
  }
  auto &ana = anaRes.value();
  LDBG("packing " << ana.customOp);

  auto scopeOp = packScopeOp(ana.relevantOps);
  if (!scopeOp) {
    // Leaving the custom op unpacked is safe: its buffers are either picked up
    // later as load units by `packAllLoadUnits`, or fall back to the raw
    // `bufferization.to_tensor` path in OpClassifierPass::matchLoadPattern.
    LDBG("failed to pack scope: " << customOp << "; fallback to default");
    return;
  }
  // A load-type custom op is also a load unit; tag it so OpClassifier can
  // recognize the load semantics, exactly as `packLoadUnit` does.
  if (ana.isLoad) {
    scopeOp->setAttr(CVPipeline::kScopeLoadAttr,
                     UnitAttr::get(scopeOp->getContext()));
  }
  // Stamping is not a hint: OpClassifierPass treats any `ssbuffer.core_type` as
  // authoritative and never re-colours it, so an undetermined core type must be
  // left off for the op to be classified by its data flow.
  if (ana.coreType != UNDETERMINED) {
    scopeOp->setAttr(
        kCoreType,
        StringAttr::get(scopeOp->getContext(), coreTypeToString(ana.coreType)));
  }
}

/// Collects every `bufferization.to_tensor` that (transitively) reads `base`
/// through view-like ops. The traversal follows the def-use chain instead of
/// walking the enclosing block, so a memref yielded out of a region is not
/// expected here: region results are not followed, hence their consumers are
/// not collected.
static SmallVector<bufferization::ToTensorOp> findToTensorsOver(Value base) {
  SmallVector<bufferization::ToTensorOp> result;
  SmallVector<Value> worklist{base};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (Operation *user : cur.getUsers()) {
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
        result.push_back(toTensor);
        continue;
      }
      if (isa<ViewLikeOpInterface>(user)) {
        for (Value res : user->getResults()) {
          worklist.push_back(res);
        }
      }
    }
  }
  return result;
}

namespace {
/// A top-level op in the packing block, together with the concrete nested op
/// that consumed the memref.
struct MemrefUse {
  Operation *top;
  Operation *user;
};
} // namespace

/// Follows the def-use chain from `base` and collects the top-level consumers
/// in `block` that precede `boundary` (the to_tensor that closes the unit).
static SmallVector<MemrefUse>
collectMemrefUsesBefore(Value base, Block *block,
                        bufferization::ToTensorOp boundary) {
  SmallVector<MemrefUse> uses;
  llvm::DenseSet<Operation *> seenTop;
  SmallVector<Value> worklist{base};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (Operation *user : cur.getUsers()) {
      // Do not cross the to_tensor boundary.
      if (user == boundary.getOperation()) {
        continue;
      }
      Operation *top = block->findAncestorOpInBlock(*user);
      if (!top || top->getBlock() != block || !top->isBeforeInBlock(boundary)) {
        continue;
      }
      if (seenTop.insert(top).second) {
        uses.push_back({top, user});
      }
      for (Value res : user->getResults()) {
        worklist.push_back(res);
      }
    }
  }
  return uses;
}

/// Finds the single `bufferization.to_tensor` that closes the load unit rooted
/// at `base`. Fails when there is no such boundary, or when the boundary is not
/// a direct child of `block`: a nested boundary cannot move into the scope
/// together with the allocation, and ordering it against the operations of
/// `block` would break the same-block precondition of `isBeforeInBlock`.
static FailureOr<bufferization::ToTensorOp> findLoadUnitBoundary(Value base,
                                                                 Block *block) {
  SmallVector<bufferization::ToTensorOp> candidates = findToTensorsOver(base);
  if (candidates.size() != 1) {
    LDBG("skip " << base << ": expected one to_tensor, found "
                 << candidates.size());
    return failure();
  }
  bufferization::ToTensorOp boundary = candidates.front();
  if (boundary->getBlock() != block) {
    LDBG("skip " << base << ": to_tensor " << boundary
                 << " is not in the block of the allocation");
    return failure();
  }
  // `findToTensorsOver` only follows view-like chains, so this cross-check can
  // only fire for a chain `traceMemDef` cannot resolve. Such a unit is left
  // unverified and falls back to the OpClassifierPass raw-to_tensor path.
  if (traceMemDef(boundary.getBuffer()) != base) {
    LDBG("skip " << base << ": to_tensor buffer does not trace back to it");
    return failure();
  }
  return boundary;
}

/// Collects every consumer of `base` that precedes `boundary` in `block` and
/// rejects a consumer that is neither an ExtractedLoadOrStore loop nor a
/// trivial single-op region. `isExtractedLoadStore` is set when the unit is
/// driven by such a loop, which pins the unit to VECTOR. Fails when a consumer
/// has an unrecognised shape; the caller then leaves the unit unpacked instead
/// of aborting the pipeline.
static FailureOr<SmallVector<MemrefUse>>
validateLoadUnitUses(Value base, Block *block,
                     bufferization::ToTensorOp boundary,
                     bool &isExtractedLoadStore) {
  SmallVector<MemrefUse> uses = collectMemrefUsesBefore(base, block, boundary);
  isExtractedLoadStore = false;
  for (const MemrefUse &use : uses) {
    if (isa<scf::ForOp>(use.top) &&
        use.top->hasAttr(hivm::ExtractLoadStoreAttr)) {
      isExtractedLoadStore = true;
      continue;
    }
    // `use.top != use.user` means the consumer is nested in a region of the
    // top-level op; only a region holding nothing but that consumer can travel
    // into the scope unchanged.
    if (use.top != use.user && !isOnlyOpInItsBlock(use.user)) {
      LDBG("skip load unit: expected an ExtractedLoadOrStore loop or a trivial "
           "single-op region; offending op: "
           << *use.user);
      return failure();
    }
  }
  return uses;
}

/// True when the unit reads at least one buffer that comes from outside the
/// kernel, i.e. a `func.func` block argument (the GM convention shared with
/// `CustomOpAnalysis::collectBuffers`). A unit built only from local buffers is
/// a compute pattern, not matmul input staging: tagging it `ssbuffer.load`
/// would feed it to `OpClassifierPass::matchLoadPattern`, which reserves the
/// load semantics for the load scopes that stage matmul inputs, so the caller
/// skips it.
static bool unitReadsGlobalMemory(ArrayRef<Operation *> unitOps) {
  for (Operation *unitOp : unitOps) {
    bool reachedGm = false;
    unitOp->walk([&](Operation *nestedOp) {
      if (reachedGm) {
        return;
      }
      for (Value operand : nestedOp->getOperands()) {
        if (!isa<MemRefType>(operand.getType())) {
          continue;
        }
        auto blockArg = dyn_cast<BlockArgument>(traceMemDef(operand));
        if (blockArg && isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
          reachedGm = true;
          return;
        }
      }
    });
    if (reachedGm) {
      return true;
    }
  }
  return false;
}

/// Builds the dominance-ordered op list that moves into the scope: the unit ops
/// plus the closure scalars. A scalar that still has a consumer outside the
/// unit is cloned, the internal consumers are rewired to the clone, and the
/// original keeps dominating the external uses. A scalar reaching a foreign
/// local allocation stays outside, so two local buffers are never merged into
/// one unit. Precondition: every op of `unitOps` and `scalars` is a direct
/// child of the same block (the closure is collected with `includeParent =
/// false`), which is what makes the in-block sort valid.
static SmallVector<Operation *> buildPackList(ArrayRef<Operation *> unitOps,
                                              ArrayRef<Operation *> scalars) {
  llvm::DenseSet<Operation *> inUnit(unitOps.begin(), unitOps.end());

  llvm::DenseSet<Operation *> foreignAllocScalars;
  for (Operation *scalarOp : scalars) {
    for (Value operand : scalarOp->getOperands()) {
      Value src = traceMemDef(operand);
      if (!src || inUnit.contains(src.getDefiningOp())) {
        continue;
      }
      if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
              src.getDefiningOp())) {
        LDBG("load unit closure reached a foreign allocation "
             << src << "; leaving " << *scalarOp << " outside the scope");
        foreignAllocScalars.insert(scalarOp);
        break;
      }
    }
  }

  llvm::DenseSet<Operation *> inClosure;
  for (Operation *scalarOp : scalars) {
    if (!foreignAllocScalars.contains(scalarOp)) {
      inClosure.insert(scalarOp);
    }
  }

  OpBuilder builder(unitOps.front());
  IRMapping mapper;

  SmallVector<Operation *> packList(unitOps);
  for (Operation *scalarOp : scalars) {
    // Fallback: consumers of a foreign allocation stay outside the scope.
    if (foreignAllocScalars.contains(scalarOp)) {
      continue;
    }

    bool hasExternalUse = false;
    for (Value result : scalarOp->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (!inUnit.contains(user) && !inClosure.contains(user)) {
          hasExternalUse = true;
          break;
        }
      }
      if (hasExternalUse) {
        break;
      }
    }

    if (!hasExternalUse) {
      // No external consumer: move it straight into the scope.
      packList.push_back(scalarOp);
      continue;
    }

    // External consumers exist: they keep using the original, while a clone
    // travels into the scope for the internal consumers (same convention as
    // `ScalarClosure::capture`: the packed region takes the clone). The clone
    // is created at the original's position so it still dominates every
    // consumer it is swapped into.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(scalarOp);
    Operation *cloned = builder.clone(*scalarOp, mapper);
    for (unsigned i = 0; i < scalarOp->getNumResults(); ++i) {
      Value original = scalarOp->getResult(i);
      Value replacement = cloned->getResult(i);
      for (OpOperand &use : llvm::make_early_inc_range(original.getUses())) {
        Operation *user = use.getOwner();
        if (inUnit.contains(user) || inClosure.contains(user)) {
          use.set(replacement);
        }
      }
    }
    packList.push_back(cloned);
  }

  llvm::sort(packList,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return packList;
}

/// Packs `allocOp -> <ops operating on it> -> bufferization.to_tensor` (the
/// to_tensor possibly reached through a subview / reinterpret_cast) into one
/// `scope::ScopeOp` tagged with `ssbuffer.load`. When the unit is driven by an
/// ExtractedLoadOrStore loop the scope is stamped VECTOR right away
/// (ExtractedLoadOrStore never runs on CUBE). Any deviation leaves the
/// allocation untouched and merely traces the reason.
static PackUnitStatus packLoadUnit(memref::AllocOp allocOp) {
  Value base = allocOp.getMemref();
  Block *block = allocOp->getBlock();
  if (!block) {
    LDBG("skip " << allocOp << ": allocation has no parent block");
    return PackUnitStatus::Skipped;
  }

  FailureOr<bufferization::ToTensorOp> boundary =
      findLoadUnitBoundary(base, block);
  if (llvm::failed(boundary)) {
    return PackUnitStatus::Skipped;
  }

  bool isExtractedLoadStore = false;
  FailureOr<SmallVector<MemrefUse>> uses =
      validateLoadUnitUses(base, block, *boundary, isExtractedLoadStore);
  if (llvm::failed(uses)) {
    return PackUnitStatus::Skipped;
  }

  SmallVector<Operation *> unitOps{allocOp, *boundary};
  for (const MemrefUse &use : *uses) {
    if (!llvm::is_contained(unitOps, use.top)) {
      unitOps.push_back(use.top);
    }
  }
  llvm::sort(unitOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  if (!unitReadsGlobalMemory(unitOps)) {
    LDBG("skip " << base
                 << ": unit reads no GM data, so it is a compute pattern");
    return PackUnitStatus::Skipped;
  }

  // Honour ScalarClosure for subviews / reinterpret_cast / scalars.
  ScalarClosure closure(block, unitOps, false);
  closure.collect();

  SmallVector<Operation *> packList =
      buildPackList(unitOps, closure.getScalarOps());

  auto scopeOp = packScopeOp(packList);
  if (!scopeOp) {
    LDBG("skip load unit: scope packing failed; offending alloc: " << allocOp);
    return PackUnitStatus::Skipped;
  }

  scopeOp->setAttr(CVPipeline::kScopeLoadAttr,
                   UnitAttr::get(scopeOp->getContext()));
  // An ExtractedLoadOrStore loop always runs on VECTOR.
  if (isExtractedLoadStore) {
    scopeOp->setAttr(kCoreType,
                     StringAttr::get(scopeOp->getContext(),
                                     coreTypeToString(CoreType::VECTOR_ONLY)));
  }
  LDBG("packed load unit of " << base);
  return PackUnitStatus::Packed;
}

std::unique_ptr<OperationPass<ModuleOp>> CVPipeline::createPackMemrefPass() {
  return std::make_unique<PackMemrefPass>();
}
