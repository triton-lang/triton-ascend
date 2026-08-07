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

#include "TritonControlFlowOpt/ControlFlowAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;

namespace mlir::triton::controlflow {

namespace {

/// Control-flow kinds whose operand/result correspondence is understood by
/// both the analyzer and the mechanical rewrite.
static bool isSupportedControlFlow(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op);
}

static void setResultIdentity(AnalyzedValue &value, Value result,
                              ArrayRef<unsigned> componentIndices) {
  // A transferred component is represented by the SCF result outside the op;
  // invariant components retain their incoming symbolic identities.
  for (unsigned index : componentIndices)
    value.components[index].identity =
        ComponentIdentity::fromValue(result, index);
}

static ControlFlowSlotAnalysis
makeSlotAnalysis(unsigned oldIndex, unsigned componentCount,
                 ArrayRef<unsigned> componentIndices,
                 SmallVector<Type> componentTypes) {
  // Preserve the full classification for future Recomputed support while also
  // storing a compact ordered list used by the current signature rewrite.
  ControlFlowSlotAnalysis slot;
  slot.oldIndex = oldIndex;
  slot.componentKinds.assign(componentCount, ComponentTransferKind::Invariant);
  for (unsigned index : componentIndices)
    slot.componentKinds[index] = ComponentTransferKind::Transferred;
  slot.componentIndices.append(componentIndices.begin(),
                               componentIndices.end());
  slot.componentTypes = std::move(componentTypes);
  return slot;
}

} // namespace

//===----------------------------------------------------------------------===//
// Shared value analysis and nested traversal
//===----------------------------------------------------------------------===//

const ControlFlowOpAnalysis *
ControlFlowRewritePlan::lookup(Operation *op) const {
  auto it = operations.find(op);
  return it == operations.end() ? nullptr : &it->second;
}

const AnalyzedValue *
ControlFlowAnalysisContext::lookupValue(Value value) const {
  auto it = analyzedValues.find(value);
  return it == analyzedValues.end() ? nullptr : &it->second;
}

const ControlFlowOpAnalysis *
ControlFlowAnalysisContext::lookup(Operation *op) const {
  auto it = analyzedOps.find(op);
  return it == analyzedOps.end() ? nullptr : &it->second;
}

FailureOr<AnalyzedValue> ControlFlowAnalysisContext::analyzeValue(Value value) {
  if (const AnalyzedValue *known = lookupValue(value))
    return *known;

  // A control-flow result is meaningful only after all incoming states have
  // been merged. Analyze its owner first instead of letting a pointer policy
  // treat the opaque result as a new base.
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *owner = result.getOwner();
    if (isSupportedControlFlow(owner)) {
      if (failed(analyzeControlFlowOp(owner)))
        return failure();
      if (const AnalyzedValue *known = lookupValue(value))
        return *known;
      return failure();
    }
  }

  FailureOr<AnalyzedValue> result = policy.analyzeValue(value, *this);
  if (failed(result))
    return failure();
  analyzedValues.try_emplace(value, *result);
  return *result;
}

void ControlFlowAnalysisContext::bindRegionArgument(
    Value argument, const AnalyzedValue &initial,
    ArrayRef<unsigned> componentIndices) {
  // Only candidate loop components acquire a new identity at region entry.
  // Policy-owned invariants and non-carried components remain traceable to the
  // initial descriptor and can therefore be checked at the backedge.
  AnalyzedValue argumentState = initial;
  for (unsigned index : componentIndices)
    argumentState.components[index].identity =
        ComponentIdentity::fromValue(argument, index);
  analyzedValues[argument] = std::move(argumentState);
}

FailureOr<SmallVector<Type>> ControlFlowAnalysisContext::getTransferredTypes(
    const AnalyzedValue &lhs, const AnalyzedValue &rhs,
    ArrayRef<unsigned> componentIndices) const {
  // Type joining is policy-owned because tensor offsets may widen i32 to i64,
  // whereas block-pointer descriptor fields currently require exact equality.
  SmallVector<Type> types;
  types.reserve(componentIndices.size());
  for (unsigned index : componentIndices) {
    if (index >= lhs.components.size() || index >= rhs.components.size())
      return failure();
    FailureOr<Type> type = policy.joinComponentTypes(
        lhs.components[index].type, rhs.components[index].type);
    if (failed(type))
      return failure();
    types.push_back(*type);
  }
  return types;
}

LogicalResult
ControlFlowAnalysisContext::analyzeNestedOperations(Block *block,
                                                    bool &hasNestedRewrite) {
  // Walk in program order so an inner result requested by a later address
  // expression is already cached. Nested SCF is analyzed recursively and its
  // rewrite requirement is propagated to every enclosing supported SCF op.
  for (Operation &operation : block->without_terminator()) {
    if (isSupportedControlFlow(&operation)) {
      FailureOr<ControlFlowOpAnalysis> nested =
          analyzeControlFlowOp(&operation);
      if (failed(nested))
        return failure();
      hasNestedRewrite |= nested->needsRewrite();
      continue;
    }

    // SCF may be wrapped in an ordinary region-owning operation. Such regions
    // do not change pointer schemas, but ControlFlowRewrite does not yet clone
    // arbitrary region operations recursively. Reject an affected nested SCF
    // instead of reporting success and leaving it opaque in the rewritten IR.
    for (Region &region : operation.getRegions()) {
      for (Block &nestedBlock : region) {
        bool regionNeedsRewrite = false;
        if (failed(analyzeNestedOperations(&nestedBlock, regionNeedsRewrite)) ||
            regionNeedsRewrite)
          return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.for schema analysis
//===----------------------------------------------------------------------===//

FailureOr<ControlFlowOpAnalysis>
ControlFlowAnalysisContext::analyzeFor(Operation *operation) {
  auto forOp = cast<scf::ForOp>(operation);
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (forOp.getInitArgs().size() != forOp.getRegionIterArgs().size() ||
      yieldOp.getNumOperands() != forOp.getRegionIterArgs().size())
    return failure();

  SmallVector<std::optional<AnalyzedValue>> initialStates(
      forOp.getInitArgs().size());

  // Bind an abstract component state to each pointer region argument. The body
  // can then be analyzed recursively without constructing a replacement loop.
  for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!policy.isDecompositionTarget(iterArg))
      continue;
    FailureOr<AnalyzedValue> initial = analyzeValue(forOp.getInitArgs()[index]);
    if (failed(initial))
      return failure();
    FailureOr<SmallVector<unsigned>> candidates =
        policy.getLoopCandidateComponents(*initial);
    if (failed(candidates))
      return failure();
    initialStates[index] = *initial;
    bindRegionArgument(iterArg, *initial, *candidates);
  }

  ControlFlowOpAnalysis result;
  if (failed(analyzeNestedOperations(forOp.getBody(), result.hasNestedRewrite)))
    return failure();

  // Compare the abstract init, region-argument and yielded states only after
  // the complete body (including nested SCF) has been analyzed.
  for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!initialStates[index])
      continue;
    FailureOr<AnalyzedValue> next = analyzeValue(yieldOp.getOperand(index));
    const AnalyzedValue *argument = lookupValue(iterArg);
    if (failed(next) || !argument)
      return failure();
    FailureOr<SmallVector<unsigned>> transferred =
        policy.getLoopTransferredComponents(*initialStates[index], *argument,
                                            *next);
    if (failed(transferred))
      return failure();

    AnalyzedValue resultState = *initialStates[index];
    if (!transferred->empty()) {
      // The ordered component/type list is the complete contract consumed by
      // rewriteForOp; rewrite does not rediscover its signature on the fly.
      FailureOr<SmallVector<Type>> types =
          getTransferredTypes(*initialStates[index], *next, *transferred);
      if (failed(types))
        return failure();
      for (auto [component, type] : llvm::zip(*transferred, *types))
        resultState.components[component].type = type;
      result.slots.push_back(makeSlotAnalysis(index,
                                              resultState.components.size(),
                                              *transferred, std::move(*types)));
      setResultIdentity(resultState, forOp.getResult(index), *transferred);
    }
    analyzedValues[forOp.getResult(index)] = std::move(resultState);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// scf.while schema analysis
//===----------------------------------------------------------------------===//

FailureOr<ControlFlowOpAnalysis>
ControlFlowAnalysisContext::analyzeWhile(Operation *operation) {
  auto whileOp = cast<scf::WhileOp>(operation);
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  if (whileOp.getInits().size() != whileOp.getBeforeArguments().size() ||
      conditionOp.getArgs().size() != whileOp.getAfterArguments().size() ||
      yieldOp.getNumOperands() != whileOp.getBeforeArguments().size())
    return failure();

  SmallVector<std::optional<AnalyzedValue>> initialStates(
      whileOp.getInits().size());
  SmallVector<SmallVector<unsigned>> candidateIndices(
      whileOp.getInits().size());

  // Bind the initial descriptor to the before-region arguments first. The
  // condition and backedge are analyzed as two consecutive state transfers.
  for (auto [index, beforeArg] :
       llvm::enumerate(whileOp.getBeforeArguments())) {
    if (!policy.isDecompositionTarget(beforeArg))
      continue;
    FailureOr<AnalyzedValue> initial = analyzeValue(whileOp.getInits()[index]);
    if (failed(initial))
      return failure();
    FailureOr<SmallVector<unsigned>> candidates =
        policy.getLoopCandidateComponents(*initial);
    if (failed(candidates))
      return failure();
    initialStates[index] = *initial;
    candidateIndices[index] = *candidates;
    bindRegionArgument(beforeArg, *initial, *candidates);
  }

  ControlFlowOpAnalysis result;
  if (failed(analyzeNestedOperations(whileOp.getBeforeBody(),
                                     result.hasNestedRewrite)))
    return failure();

  // The after-region argument receives the value forwarded by scf.condition.
  // Bind it before visiting the after region so address expressions there can
  // recursively resolve the same abstract schema.
  SmallVector<std::optional<AnalyzedValue>> conditionStates(
      conditionOp.getArgs().size());
  for (auto [index, afterArg] : llvm::enumerate(whileOp.getAfterArguments())) {
    if (!initialStates[index])
      continue;
    FailureOr<AnalyzedValue> condition =
        analyzeValue(conditionOp.getArgs()[index]);
    if (failed(condition))
      return failure();
    conditionStates[index] = *condition;
    bindRegionArgument(afterArg, *condition, candidateIndices[index]);
  }

  if (failed(analyzeNestedOperations(whileOp.getAfterBody(),
                                     result.hasNestedRewrite)))
    return failure();

  for (auto [index, beforeArg] :
       llvm::enumerate(whileOp.getBeforeArguments())) {
    if (!initialStates[index])
      continue;
    FailureOr<AnalyzedValue> next = analyzeValue(yieldOp.getOperand(index));
    const AnalyzedValue *argument = lookupValue(beforeArg);
    if (failed(next) || !argument || !conditionStates[index])
      return failure();

    // The policy merge is monotone over {Invariant, Transferred}: once either
    // region changes a candidate component it must be present in the loop
    // signature. Re-running the union would not remove a transferred bit, so
    // this is the fixed point for the current two-state domain.
    FailureOr<SmallVector<unsigned>> fromCondition =
        policy.getLoopTransferredComponents(*initialStates[index], *argument,
                                            *conditionStates[index]);
    const AnalyzedValue *afterArgument =
        lookupValue(whileOp.getAfterArguments()[index]);
    if (!afterArgument || failed(fromCondition))
      return failure();
    FailureOr<SmallVector<unsigned>> fromBackedge =
        policy.getLoopTransferredComponents(*conditionStates[index],
                                            *afterArgument, *next);
    if (failed(fromBackedge))
      return failure();

    SmallVector<unsigned> transferred = *fromCondition;
    for (unsigned component : *fromBackedge) {
      if (!llvm::is_contained(transferred, component))
        transferred.push_back(component);
    }
    llvm::sort(transferred);

    AnalyzedValue resultState = *initialStates[index];
    if (!transferred.empty()) {
      FailureOr<SmallVector<Type>> types = getTransferredTypes(
          *initialStates[index], *conditionStates[index], transferred);
      if (failed(types))
        return failure();
      for (auto [component, type] : llvm::zip(transferred, *types)) {
        FailureOr<Type> joined =
            policy.joinComponentTypes(type, next->components[component].type);
        if (failed(joined))
          return failure();
        type = *joined;
        resultState.components[component].type = type;
      }
      result.slots.push_back(makeSlotAnalysis(index,
                                              resultState.components.size(),
                                              transferred, std::move(*types)));
      setResultIdentity(resultState, whileOp.getResult(index), transferred);
    }
    analyzedValues[whileOp.getResult(index)] = std::move(resultState);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// scf.if schema analysis
//===----------------------------------------------------------------------===//

FailureOr<ControlFlowOpAnalysis>
ControlFlowAnalysisContext::analyzeIf(Operation *operation) {
  auto ifOp = cast<scf::IfOp>(operation);
  ControlFlowOpAnalysis result;

  if (failed(
          analyzeNestedOperations(ifOp.thenBlock(), result.hasNestedRewrite)))
    return failure();
  if (ifOp.elseBlock() && failed(analyzeNestedOperations(
                              ifOp.elseBlock(), result.hasNestedRewrite)))
    return failure();

  if (ifOp.getNumResults() == 0)
    // A result-less if may still need rebuilding solely to rewrite nested SCF.
    return result;
  if (!ifOp.elseBlock())
    return failure();

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  if (thenYield.getNumOperands() != ifOp.getNumResults() ||
      elseYield.getNumOperands() != ifOp.getNumResults())
    return failure();

  // Each result position is independent. Only policy-matching pointer results
  // are expanded; all other result positions keep their original type/order.
  for (auto [index, opResult] : llvm::enumerate(ifOp.getResults())) {
    if (!policy.isDecompositionTarget(opResult))
      continue;
    FailureOr<AnalyzedValue> thenState =
        analyzeValue(thenYield.getOperand(index));
    FailureOr<AnalyzedValue> elseState =
        analyzeValue(elseYield.getOperand(index));
    if (failed(thenState) || failed(elseState))
      return failure();
    FailureOr<SmallVector<unsigned>> transferred =
        policy.getIfTransferredComponents(*thenState, *elseState);
    if (failed(transferred))
      return failure();

    AnalyzedValue resultState = *thenState;
    if (!transferred->empty()) {
      FailureOr<SmallVector<Type>> types =
          getTransferredTypes(*thenState, *elseState, *transferred);
      if (failed(types))
        return failure();
      for (auto [component, type] : llvm::zip(*transferred, *types))
        resultState.components[component].type = type;
      result.slots.push_back(makeSlotAnalysis(index,
                                              resultState.components.size(),
                                              *transferred, std::move(*types)));
      setResultIdentity(resultState, opResult, *transferred);
    }
    analyzedValues[opResult] = std::move(resultState);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Stage-wide caching, plan freezing and entry-point discovery
//===----------------------------------------------------------------------===//

FailureOr<ControlFlowOpAnalysis>
ControlFlowAnalysisContext::analyzeControlFlowOp(Operation *op) {
  if (const ControlFlowOpAnalysis *known = lookup(op))
    return *known;
  // The in-progress set prevents accidental cyclic re-entry when value
  // analysis asks to analyze the control-flow op that owns that value.
  if (!isSupportedControlFlow(op) || !operationsBeingAnalyzed.insert(op).second)
    return failure();

  FailureOr<ControlFlowOpAnalysis> result = failure();
  if (isa<scf::ForOp>(op))
    result = analyzeFor(op);
  else if (isa<scf::WhileOp>(op))
    result = analyzeWhile(op);
  else if (isa<scf::IfOp>(op))
    result = analyzeIf(op);

  operationsBeingAnalyzed.erase(op);
  if (failed(result))
    return failure();
  analyzedOps.try_emplace(op, *result);
  return *result;
}

ControlFlowRewritePlan ControlFlowAnalysisContext::takeRewritePlan() && {
  // analyzedValues intentionally dies with this context. Rewrite only needs
  // the position/type decisions and must not retain Value handles that may be
  // invalidated as earlier roots are replaced.
  return ControlFlowRewritePlan{std::move(analyzedOps)};
}

SmallVector<Operation *> collectOutermostControlFlowOps(ModuleOp module) {
  // Rewriting an outer root recursively replaces all affected descendants.
  // Returning nested ops as independent roots would leave stale pointers after
  // the parent is erased, so filter them here instead of relying on walk order.
  SmallVector<Operation *> roots;
  module.walk([&](Operation *operation) {
    if (!isSupportedControlFlow(operation))
      return;
    for (Operation *parent = operation->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isSupportedControlFlow(parent))
        return;
    }
    roots.push_back(operation);
  });
  return roots;
}

FailureOr<ControlFlowRewritePlan>
analyzeControlFlow(ModuleOp module, const ControlFlowAnalysisPolicy &policy) {
  SmallVector<Operation *> roots = collectOutermostControlFlowOps(module);
  ControlFlowAnalysisContext context(policy);

  // One context covers the complete decomposition stage. Consequently a
  // later root can reuse the merged state of a preceding sibling control-flow
  // op, and no IR is mutated until every root has proved analyzable.
  for (Operation *root : roots) {
    if (failed(context.analyzeControlFlowOp(root))) {
      root->emitError(
          "failed to analyze pointer components across control flow");
      return failure();
    }
  }

  return std::move(context).takeRewritePlan();
}

} // namespace mlir::triton::controlflow
