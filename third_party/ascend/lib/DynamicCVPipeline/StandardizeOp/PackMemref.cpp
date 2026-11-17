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
#include <utility>
#include <variant>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/Analysis.h"
#include "ascend/include/DynamicCVPipeline/Common/IROrder.h"
#include "ascend/include/DynamicCVPipeline/Common/ScalarClosure.h"
#include "ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "pack-memref";
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    llvm::dbgs() << '[' << DEBUG_TYPE << "] ";                                 \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

/// Outcome of a single packing attempt.
/// Reported through the `--debug-only=pack-memref` trace, not a diagnostic.
/// An unpacked allocation keeps the OpClassifierPass fallback, so a skip
/// never fails the pass.
enum class PackUnitStatus {
  /// The allocation was packed into a scope.
  Packed,

  /// Not a load unit, or not a consistent one: logged and left untouched.
  Skipped,
};

/// True when the block of `op` holds nothing but `op` and its terminator.
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
    registry.insert<annotation::AnnotationDialect, scope::ScopeDialect,
                    hivm::HIVMDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pack-memref";
  }
};

} // namespace

static void packCustomOpUnit(hivm::CustomOp customOp);
static PackUnitStatus packLoadUnit(memref::AllocOp allocOp);
static SmallVector<Operation *> buildPackList(ArrayRef<Operation *> unitOps,
                                              ArrayRef<Operation *> scalars);
static void packAllCustomOpUnits(ModuleOp moduleOp);
static void packAllLoadUnits(ModuleOp moduleOp);

void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  packAllCustomOpUnits(moduleOp);
  packAllLoadUnits(moduleOp);
}

/// Packs every `hivm.hir.custom` with its local buffers.
/// Each custom op owns its buffers, so claim them before load-unit packing
/// looks at the same allocations.
static void packAllCustomOpUnits(ModuleOp moduleOp) {
  SmallVector<hivm::CustomOp> customOps;
  moduleOp.walk(
      [&](hivm::CustomOp customOp) { customOps.push_back(customOp); });
  for (hivm::CustomOp customOp : customOps) {
    packCustomOpUnit(customOp);
  }
}

/// Packs the remaining allocations as load units.
/// An allocation that does not form a load unit is skipped with a trace.
/// Leaving it unpacked is safe: OpClassifierPass still recognises the raw
/// `bufferization.to_tensor` chain.
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

/// Packs `hivm.hir.custom` with its buffers into one `scope::ScopeOp`.
/// The scope carries the analysed core type.
static void packCustomOpUnit(hivm::CustomOp customOp) {
  auto anaRes = CustomOpAnalysis::get(customOp);
  auto [opToMark, coreType] = std::visit(
      llvm::makeVisitor(
          [&customOp](CoreType ct) -> std::pair<Operation *, CoreType> {
            return std::make_pair(customOp, ct);
          },
          [](CustomOpAnalysis ana) -> std::pair<Operation *, CoreType> {
            LDBG("packing " << ana.customOp);
            // The analysed ops build the unit; the views they read travel with
            // them. A unit spanning blocks gets no closure: it would be
            // collected from the wrong block, and the scope placement assumes a
            // single block.
            SmallVector<Operation *> unitOps(ana.relevantOps);
            if (unitOps.empty()) {
              LDBG(
                  "custom op unit holds no analysable buffer; nothing to pack");
              return {ana.customOp, ana.coreType};
            }
            Block *block = unitOps.front()->getBlock();
            bool sameBlock = llvm::all_of(unitOps, [block](Operation *op) {
              return op->getBlock() == block;
            });
            SmallVector<Operation *> packList(unitOps);
            if (sameBlock) {
              llvm::sort(unitOps, isBeforeInIr);
              ScalarClosure closure(block, unitOps, false);
              closure.collect();
              SmallVector<Operation *> scalars(closure.getScalars());
              llvm::sort(unitOps, isBeforeInIr);
              packList = buildPackList(unitOps, scalars);
            } else {
              LDBG("custom op unit spans blocks; packing it without a closure");
            }

            auto scopeOp = packScopeOp(packList);
            if (!scopeOp) {
              // Leaving the custom op unpacked is safe: its buffers are either
              // picked up
              // later as load units by `packAllLoadUnits`, or fall back to the
              // raw `bufferization.to_tensor` path in
              // OpClassifierPass::matchLoadPattern.
              LDBG("failed to pack scope: " << ana.customOp
                                            << "; fallback to default");
              return std::make_pair(ana.customOp, ana.coreType);
            }

            // A load-type custom op is also a load unit; tag it so OpClassifier
            // can recognize the load semantics, exactly as `packLoadUnit` does.
            if (ana.isLoad) {
              scopeOp->setAttr(CVPipeline::kScopeLoadAttr,
                               UnitAttr::get(scopeOp->getContext()));
            }

            return std::make_pair(scopeOp, ana.coreType);
          }),
      anaRes);

  if (coreType != UNDETERMINED) {
    opToMark->setAttr(kCoreType, StringAttr::get(opToMark->getContext(),
                                                 coreTypeToString(coreType)));
  }
}

/// The single `bufferization::to_tensor` that transitively reads `base`.
/// Follows view-like ops only, and stops at the second hit.
/// Pre: a load unit has exactly one boundary.
/// Post: failure when there is none, or more than one.
/// Region results are not followed: a memref yielded out of a region does not
/// contribute its consumers.
static FailureOr<bufferization::ToTensorOp> findToTensorOver(Value base) {
  FailureOr<bufferization::ToTensorOp> found;
  SmallVector<Value> worklist{base};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (Operation *user : cur.getUsers()) {
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (succeeded(found)) {
          return failure();
        }
        found = toTensor;
        continue;
      }
      if (isa<ViewLikeOpInterface>(user)) {
        for (Value res : user->getResults()) {
          worklist.push_back(res);
        }
      }
    }
  }
  return found;
}

namespace {
/// A top-level op in the packing block and the nested op that used the memref.
struct MemrefUse {
  Operation *top;
  Operation *user;
};
} // namespace

/// Top-level consumers of `base` in `block` that precede `boundary`.
/// `base` is followed along the def-use chain; each consumer is reported once,
/// by the top-level op that encloses it.
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

/// The to_tensor that closes the load unit rooted at `base`.
/// Pre: `block` is the block of the allocation.
/// Post: failure when the boundary is missing, or not a direct child of
/// `block`: a nested boundary cannot move into the scope, and ordering it
/// against the ops of `block` would break `isBeforeInBlock`.
static FailureOr<bufferization::ToTensorOp> findLoadUnitBoundary(Value base,
                                                                 Block *block) {
  FailureOr<bufferization::ToTensorOp> candidate = findToTensorOver(base);
  if (failed(candidate)) {
    LDBG("skip " << base << ": expected exactly one to_tensor over it");
    return failure();
  }
  bufferization::ToTensorOp boundary = *candidate;
  if (boundary->getBlock() != block) {
    LDBG("skip " << base << ": to_tensor " << boundary
                 << " is not in the block of the allocation");
    return failure();
  }
  return boundary;
}

/// Collects the consumers of `base` that precede `boundary` in `block`.
/// Post: failure when a consumer is neither an ExtractedLoadOrStore loop nor a
/// trivial single-op region; the caller then leaves the unit unpacked.
/// `isExtractedLoadStore` -> set when such a loop drives the unit, which pins
/// the unit to VECTOR.
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
    // Nested consumer (`top != user`): only a region holding nothing but that
    // consumer can travel into the scope unchanged.
    if (use.top != use.user && !isOnlyOpInItsBlock(use.user)) {
      LDBG("skip load unit: expected an ExtractedLoadOrStore loop or a trivial "
           "single-op region; offending op: "
           << *use.user);
      return failure();
    }
  }
  return uses;
}

/// True when `operand` is GM and `op` reads it.
/// GM is the old rule: the operand traces back to a `func.func` block argument.
/// An op that implements the effect interface then has to report a Read on this
/// operand, so a write-only GM operand no longer counts. An op without one (a
/// hivm op) keeps the old rule as is.
static bool operandReadsGlobal(Operation *op, Value operand) {
  auto blockArg = dyn_cast<BlockArgument>(traceMemDef(operand));
  if (!blockArg || !isa<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
    return false;
  }
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects) {
    return true;
  }
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> onValue;
  effects.getEffectsOnValue(operand, onValue);
  return llvm::any_of(onValue, [](const auto &effect) {
    return isa_and_nonnull<MemoryEffects::Read>(effect.getEffect());
  });
}

/// True when the unit reads a buffer that comes from outside the kernel.
/// Such a buffer is a `func.func` block argument, the GM convention shared with
/// `CustomOpAnalysis::collectBuffers`.
///
/// A unit built only from local buffers is a compute pattern, not staging.
/// Tagging it `ssbuffer.load` would feed it to
/// `OpClassifierPass::matchLoadPattern`, which reserves load semantics for the
/// scopes that stage matmul inputs, so the caller skips it.
static bool unitReadsGlobalMemory(ArrayRef<Operation *> unitOps) {
  for (Operation *unitOp : unitOps) {
    // One bool answer, so the first GM read interrupts the walk.
    WalkResult read = unitOp->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (!isa<MemRefType>(operand.getType())) {
          continue;
        }
        if (operandReadsGlobal(nestedOp, operand)) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (read.wasInterrupted()) {
      return true;
    }
  }
  return false;
}

/// The dominance-ordered op list that moves into the scope.
/// It holds the unit ops plus their view and scalar dependencies.
/// Pre: `unitOps` and `scalars` are direct children of one block.
/// Pre: `scalars` is in dominance order (`includeParent = false`).
/// Post: an op with an outside consumer is cloned, not moved: the internal
/// consumers read the clone, the original keeps dominating the outside uses.
/// Post: an op that must stay put is skipped, its results are captured.
static SmallVector<Operation *> buildPackList(ArrayRef<Operation *> unitOps,
                                              ArrayRef<Operation *> scalars) {
  llvm::DenseSet<Operation *> inUnit(unitOps.begin(), unitOps.end());

  // A closure op that stays outside is captured by result, not moved in.
  // tag            | stays outside because
  // [ForeignAlloc] | it reaches a foreign local allocation
  // [PackedUnit]   | it is itself a `scope.scope` unit
  // [ForeignAlloc] keeps two local buffers apart.
  // [PackedUnit] keeps a packed unit atomic: nesting it would redirect its
  // results to an op created after the consumers they dominate.
  llvm::DenseSet<Operation *> staysOutside;
  for (Operation *scalarOp : scalars) {
    if (isa<scope::ScopeOp>(scalarOp)) {
      LDBG("closure depends on the packed unit "
           << scalarOp << "; leaving it outside the new scope");
      staysOutside.insert(scalarOp);
      continue;
    }
    for (Value operand : scalarOp->getOperands()) {
      Value src = traceMemDef(operand);
      if (!src || inUnit.contains(src.getDefiningOp())) {
        continue;
      }
      if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
              src.getDefiningOp())) {
        LDBG("closure reached a foreign allocation "
             << src << "; leaving " << *scalarOp << " outside the scope");
        staysOutside.insert(scalarOp);
        break;
      }
    }
  }

  llvm::DenseSet<Operation *> inClosure;
  for (Operation *scalarOp : scalars) {
    if (staysOutside.contains(scalarOp)) {
      continue;
    }
    inClosure.insert(scalarOp);
  }

  // Move or copy: an op moves in when every user of its results travels in.
  // If any user stays outside, the original keeps serving it and the scope
  // receives a clone instead. The rule cascades: a copied consumer's own
  // original also stays outside and keeps reading its operands, so every
  // producer feeding it is copied too.
  //
  // user                | travels in? | because
  // in the unit         | yes         | it is packed anyway
  // closure, moved in   | yes         | it travels with its consumers
  // closure, copied     | no          | its original stays outside
  // outside the closure | no          | it reads across the boundary
  //
  // `packLoadUnit` sorts `scalars` into IR order, so walking it backwards
  // visits every user before its producers and one pass reaches the fixpoint.
  llvm::DenseSet<Operation *> copiedOps;
  auto travelsIn = [&](Operation *user) {
    return inUnit.contains(user) ||
           (inClosure.contains(user) && !copiedOps.contains(user));
  };
  for (Operation *scalarOp : llvm::reverse(scalars)) {
    bool escapes = llvm::any_of(scalarOp->getResults(), [&](Value result) {
      return llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return !travelsIn(use.getOwner());
      });
    });
    if (escapes) {
      copiedOps.insert(scalarOp);
    }
  }

  OpBuilder builder(unitOps.front());
  IRMapping mapper;

  SmallVector<Operation *> packList(unitOps);
  for (Operation *scalarOp : scalars) {
    if (staysOutside.contains(scalarOp)) {
      continue;
    }

    // No outside consumer: move it straight into the scope.
    if (!copiedOps.contains(scalarOp)) {
      packList.push_back(scalarOp);
      continue;
    }

    // External consumers exist: they keep the original, the scope takes a
    // clone (same convention as `ScalarClosure::capture`). The clone sits at
    // the original's position so it dominates every consumer it replaces.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(scalarOp);
    Operation *cloned = builder.clone(*scalarOp, mapper);
    for (unsigned i = 0; i < scalarOp->getNumResults(); ++i) {
      Value original = scalarOp->getResult(i);
      Value replacement = cloned->getResult(i);
      for (OpOperand &use : llvm::make_early_inc_range(original.getUses())) {
        // The clone serves the users that move in; a copied op's original
        // keeps serving the ones that stay outside.
        if (travelsIn(use.getOwner())) {
          use.set(replacement);
        }
      }
    }
    packList.push_back(cloned);
  }

  llvm::sort(packList, isBeforeInIr);
  return packList;
}

/// Packs `memref.alloc -> <ops on it> -> bufferization.to_tensor` into one
/// `scope::ScopeOp` tagged `ssbuffer.load`.
/// The to_tensor may be reached through a subview or a reinterpret_cast.
/// Post: an ExtractedLoadOrStore unit or an implicit-transpose unit is also
/// stamped VECTOR.
/// Post: any deviation leaves the allocation untouched and traces the reason.
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

  llvm::SetVector<Operation *> unitOps;
  unitOps.insert(allocOp);
  unitOps.insert(*boundary);
  for (const MemrefUse &use : *uses) {
    unitOps.insert(use.top);
  }

  if (!unitReadsGlobalMemory(unitOps.getArrayRef())) {
    LDBG("skip " << base << ": unit reads no GM data, a compute pattern");
    return PackUnitStatus::Skipped;
  }

  // OpClassifier skips the packed body, so keep the implicit-transpose
  // restriction as an authoritative core type before hiding its annotations.
  bool hasImplicitTranspose =
      utils::getAnnotateOpWithAttr(base,
                                   hivm::kMayImplicitTransposeWithLastAxis)
          .has_value() ||
      utils::getAnnotateOpWithAttr((*boundary).getResult(),
                                   hivm::kMayImplicitTransposeWithLastAxis)
          .has_value();

  // Result annotations join the unit so they sort with the load.
  // The explicit insertion point keeps a late annotation from delaying the
  // scope past an existing tensor user.
  for (Operation *user : (*boundary).getResult().getUsers()) {
    if (isa<annotation::MarkOp>(user) && user->getBlock() == block) {
      unitOps.insert(user);
    }
  }

  // Collect the load unit's scalar and memref view dependencies.
  ScalarClosure closure(block, unitOps.getArrayRef(), true);
  closure.collect();
  SmallVector<Operation *> scalars(closure.getScalars());
  llvm::sort(scalars, isBeforeInIr);
  SmallVector<Operation *> packList =
      buildPackList(unitOps.getArrayRef(), scalars);

  auto scopeOp = packScopeOp(packList, *boundary);
  if (!scopeOp) {
    LDBG("skip load unit: scope packing failed; offending alloc: " << allocOp);
    return PackUnitStatus::Skipped;
  }

  scopeOp->setAttr(CVPipeline::kScopeLoadAttr,
                   UnitAttr::get(scopeOp->getContext()));
  // Extracted loads and implicit-transpose loads must stay on VECTOR.
  if (isExtractedLoadStore || hasImplicitTranspose) {
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
