//===- MaterializeSimtScopes.cpp - Materialize local SIMT scopes --------===//
//
// Materialization consumes the immutable SimtAnchorPlan produced by the Route
// Model and creates SSA-safe scope.scope regions.  No per-operation selection
// marker is written or read.  The compatibility pass below only validates that
// an admitted mixed decision already carries its local scope contract.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/SimtSelection.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_MATERIALIZESIMTSCOPESPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using namespace simt_selection;

static bool isMaterializable(Operation *op) {
  return op->getBlock() && !isa<ModuleOp>(op) &&
         !op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         !op->hasTrait<OpTrait::IsTerminator>() &&
         op->getName().getStringRef() != "scope.scope" &&
         op->getName().getStringRef() != "scope.return";
}

/// Wrap one anchor operation and thread all of its SSA results through
/// scope.return.
///
/// Scope regions are not isolated from above, so operands remain legal
/// captures.  Moving only the planned operation keeps SIMD producers and
/// consumers outside the SIMT region.
static LogicalResult wrapAnchorOperation(Operation *op) {
  OpBuilder builder(op);
  OperationState scopeState(op->getLoc(), "scope.scope");
  scopeState.addTypes(op->getResultTypes());
  scopeState.addAttribute(kVectorModeAttr, builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);

  SmallVector<Value> originalResults(op->getResults());
  op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(op->getLoc(), "scope.return");
  returnState.addOperands(originalResults);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != originalResults.size())
    return op->emitError("SIMT scope result count does not match anchor op");

  for (auto [original, replacement] :
       llvm::zip_equal(originalResults, scopeOp->getResults())) {
    original.replaceAllUsesExcept(replacement, returnOp);
  }
  return success();
}

/// Wrap an exact planned operation set and thread only values that escape it.
/// `insertionPoint` lets solve_tril move pure mask setup across the initial
/// loads while keeping those loads outside, matching the hand-written scope.
static LogicalResult wrapAnchorRange(ArrayRef<Operation *> ops,
                                     Operation *insertionPoint) {
  if (ops.empty())
    return success();
  Block *parent = insertionPoint ? insertionPoint->getBlock() : nullptr;
  if (!parent)
    return failure();

  DenseSet<Operation *> planned;
  for (Operation *op : ops) {
    if (!op || op->getBlock() != parent || !isMaterializable(op))
      return failure();
    planned.insert(op);
  }

  auto isInsideRange = [&](Operation *user) {
    for (Operation *owner = user; owner; owner = owner->getParentOp())
      if (planned.contains(owner))
        return true;
    return false;
  };

  SmallVector<Value> escaping;
  DenseSet<Value> seen;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      for (OpOperand &use : result.getUses())
        if (!isInsideRange(use.getOwner()) && seen.insert(result).second) {
          escaping.push_back(result);
          break;
        }

  OpBuilder builder(insertionPoint);
  OperationState scopeState(insertionPoint->getLoc(), "scope.scope");
  SmallVector<Type> escapingTypes;
  escapingTypes.reserve(escaping.size());
  for (Value value : escaping)
    escapingTypes.push_back(value.getType());
  scopeState.addTypes(escapingTypes);
  scopeState.addAttribute(kVectorModeAttr, builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);
  for (Operation *op : ops)
    op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(insertionPoint->getLoc(), "scope.return");
  returnState.addOperands(escaping);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != escaping.size())
    return failure();
  for (auto [original, replacement] :
       llvm::zip_equal(escaping, scopeOp->getResults())) {
    for (OpOperand &use : llvm::make_early_inc_range(original.getUses()))
      if (use.getOwner() != returnOp && !isInsideRange(use.getOwner()))
        use.set(replacement);
  }
  return success();
}

} // namespace

LogicalResult materializeSimtAnchorPlan(ModuleOp module,
                                        const SimtAnchorPlan &plan) {
  struct PlannedRange {
    SmallVector<Operation *> operations;
    Operation *insertionPoint = nullptr;
  };
  SmallVector<Operation *> anchorOps;
  SmallVector<PlannedRange> anchorRanges;
  DenseSet<Operation *> coveredByRange;

  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    Operation *op = anchor.operation;
    if (!anchor.materializable || !op || coveredByRange.contains(op))
      continue;
    if (hasEnclosingVectorMode(op, "simt"))
      continue;

    if (anchor.scopeOperations.size() > 1) {
      for (Operation *rangeOp : anchor.scopeOperations)
        coveredByRange.insert(rangeOp);
      PlannedRange range;
      llvm::append_range(range.operations, anchor.scopeOperations);
      range.insertionPoint = anchor.scopeInsertionPoint;
      anchorRanges.push_back(std::move(range));
      continue;
    }

    if (!isMaterializable(op))
      return op->emitError(
          "SIMT anchor is not materializable as a local scope");
    anchorOps.push_back(op);
  }

  int64_t materialized = 0;
  for (const PlannedRange &range : anchorRanges) {
    if (failed(wrapAnchorRange(range.operations, range.insertionPoint)))
      return failure();
    ++materialized;
  }
  for (Operation *op : anchorOps) {
    if (failed(wrapAnchorOperation(op)))
      return failure();
    ++materialized;
  }

  if (materialized == 0)
    return module.emitError(
        "mixed_simd_simt has no materializable local SIMT scope");
  return success();
}

namespace {

static bool containsLocalSimtScope(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "scope.scope")
      return WalkResult::advance();
    auto mode = getVectorMode(op);
    if (!mode || mode.getValue() != "simt")
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct MaterializeSimtScopesPass
    : public impl::MaterializeSimtScopesPassBase<MaterializeSimtScopesPass> {
  using MaterializeSimtScopesPassBase::MaterializeSimtScopesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!isMixedModelDecision(module))
      return;
    if (containsLocalSimtScope(module))
      return;
    module.emitError(
        "mixed_simd_simt requires a materialized scope.scope<simt> contract");
    signalPassFailure();
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
