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

#include "TritonControlFlowOpt/TritonControlFlowOptPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <limits>

#define DEBUG_TYPE "triton-control-flow-opt"

using namespace mlir;
using namespace triton;

namespace {

static bool isSupportedReturn(Operation *op) {
  return isa<triton::ReturnOp, func::ReturnOp>(op);
}

static SmallVector<Block *> getCfgSuccessors(Block *block) {
  Operation *term = block->getTerminator();
  if (auto br = dyn_cast<cf::BranchOp>(term))
    return {br.getDest()};
  if (auto condBr = dyn_cast<cf::CondBranchOp>(term))
    return {condBr.getTrueDest(), condBr.getFalseDest()};
  return {};
}

static DenseMap<Block *, unsigned> computeDistances(Block *start) {
  DenseMap<Block *, unsigned> distances;
  SmallVector<Block *> worklist;

  distances[start] = 0;
  worklist.push_back(start);

  for (unsigned i = 0; i < worklist.size(); ++i) {
    Block *block = worklist[i];
    unsigned nextDistance = distances[block] + 1;
    for (Block *successor : getCfgSuccessors(block)) {
      if (successor->getParent() != start->getParent())
        continue;
      if (distances.count(successor))
        continue;
      distances[successor] = nextDistance;
      worklist.push_back(successor);
    }
  }

  return distances;
}

static FailureOr<Block *> findNearestCommonBlock(Block *lhs, Block *rhs,
                                                 Location loc,
                                                 bool emitDiagnostic = true) {
  DenseMap<Block *, unsigned> lhsDistances = computeDistances(lhs);
  DenseMap<Block *, unsigned> rhsDistances = computeDistances(rhs);

  Block *best = nullptr;
  unsigned bestMaxDistance = std::numeric_limits<unsigned>::max();
  unsigned bestTotalDistance = std::numeric_limits<unsigned>::max();

  for (auto &entry : lhsDistances) {
    Block *candidate = entry.first;
    auto rhsIt = rhsDistances.find(candidate);
    if (rhsIt == rhsDistances.end())
      continue;

    unsigned lhsDistance = entry.second;
    unsigned rhsDistance = rhsIt->second;
    unsigned maxDistance = std::max(lhsDistance, rhsDistance);
    unsigned totalDistance = lhsDistance + rhsDistance;
    if (maxDistance < bestMaxDistance ||
        (maxDistance == bestMaxDistance &&
         totalDistance < bestTotalDistance)) {
      best = candidate;
      bestMaxDistance = maxDistance;
      bestTotalDistance = totalDistance;
    }
  }

  if (!best && emitDiagnostic) {
    emitError(loc) << "unsupported non-tree control flow: branch arms do not "
                      "reach a common convergence block";
    return failure();
  }
  if (!best)
    return failure();

  return best;
}

static LogicalResult replaceBlockArguments(Block *block, ValueRange incoming,
                                           Location loc) {
  if (block->getNumArguments() != incoming.size()) {
    emitError(loc) << "invalid branch operand count while structuring "
                      "control flow: "
                   << incoming.size() << " operands for "
                   << block->getNumArguments() << " block arguments";
    return failure();
  }

  for (auto [arg, value] : llvm::zip(block->getArguments(), incoming))
    arg.replaceAllUsesWith(value);
  return success();
}

static void moveBlockBodyBefore(Block *block, OpBuilder &builder) {
  SmallVector<Operation *> movedOps = llvm::map_to_vector(
      block->without_terminator(), [](Operation &op) { return &op; });
  for (Operation *op : movedOps)
    op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
}

struct ReturnPathResult {
  SmallVector<Value> operands;
};

static FailureOr<SmallVector<Value>>
buildRegionPath(Block *block, ValueRange incoming, Block *stopBlock,
                OpBuilder &builder);

static FailureOr<ReturnPathResult>
buildReturnPath(Block *block, ValueRange incoming, OpBuilder &builder);

static FailureOr<scf::IfOp> buildTerminalValueIf(cf::CondBranchOp condBr,
                                                 OpBuilder &builder);

static FailureOr<scf::IfOp> buildStructuredIf(cf::CondBranchOp condBr,
                                              Block *joinBlock,
                                              OpBuilder &builder) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(joinBlock->getNumArguments());
  for (BlockArgument arg : joinBlock->getArguments())
    resultTypes.push_back(arg.getType());

  auto ifOp = builder.create<scf::IfOp>(
      condBr.getLoc(), resultTypes, condBr.getCondition(),
      /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    Operation *autoYield =
        resultTypes.empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.thenBlock());
    FailureOr<SmallVector<Value>> thenYield = buildRegionPath(
        condBr.getTrueDest(), condBr.getTrueDestOperands(), joinBlock, builder);
    if (failed(thenYield))
      return failure();
    if (thenYield->size() != resultTypes.size()) {
      condBr.emitError("then branch yields ")
          << thenYield->size() << " values, expected " << resultTypes.size();
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), *thenYield);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Operation *autoYield =
        resultTypes.empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.elseBlock());
    FailureOr<SmallVector<Value>> elseYield = buildRegionPath(
        condBr.getFalseDest(), condBr.getFalseDestOperands(), joinBlock,
        builder);
    if (failed(elseYield))
      return failure();
    if (elseYield->size() != resultTypes.size()) {
      condBr.emitError("else branch yields ")
          << elseYield->size() << " values, expected " << resultTypes.size();
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), *elseYield);
  }

  return ifOp;
}

static bool haveSameTypes(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    if (lhsValue.getType() != rhsValue.getType())
      return false;
  }
  return true;
}

static bool haveSameTypes(ValueRange values, ArrayRef<Type> types) {
  if (values.size() != types.size())
    return false;
  for (auto [value, type] : llvm::zip(values, types)) {
    if (value.getType() != type)
      return false;
  }
  return true;
}

static bool haveSameTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (lhsType != rhsType)
      return false;
  }
  return true;
}

static Operation *createReturnLike(OpBuilder &builder, Location loc,
                                   Operation *sampleReturn,
                                   ValueRange operands) {
  OperationState state(loc, sampleReturn->getName());
  state.addOperands(operands);
  state.addAttributes(sampleReturn->getAttrs());
  return builder.create(state);
}

static FailureOr<SmallVector<Type>>
collectReturnPathTypes(Block *block, SmallPtrSetImpl<Block *> &visiting) {
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic terminal control flow";

  Operation *term = block->getTerminator();
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    FailureOr<SmallVector<Type>> result =
        collectReturnPathTypes(br.getDest(), visiting);
    visiting.erase(block);
    return result;
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (succeeded(nestedJoin)) {
      FailureOr<SmallVector<Type>> result =
          collectReturnPathTypes(*nestedJoin, visiting);
      visiting.erase(block);
      return result;
    }

    FailureOr<SmallVector<Type>> thenTypes =
        collectReturnPathTypes(condBr.getTrueDest(), visiting);
    FailureOr<SmallVector<Type>> elseTypes =
        collectReturnPathTypes(condBr.getFalseDest(), visiting);
    visiting.erase(block);
    if (failed(thenTypes) || failed(elseTypes))
      return failure();
    if (!haveSameTypes(*thenTypes, *elseTypes)) {
      condBr.emitError("terminal branch return types do not match");
      return failure();
    }
    return *thenTypes;
  }

  if (isSupportedReturn(term)) {
    SmallVector<Type> types;
    for (Value operand : term->getOperands())
      types.push_back(operand.getType());
    visiting.erase(block);
    return types;
  }

  visiting.erase(block);
  return term->emitError()
         << "unsupported terminator while analyzing terminal control flow";
}

static Operation *findReturnOnPath(Block *block,
                                   SmallPtrSetImpl<Block *> &visited) {
  if (!visited.insert(block).second)
    return nullptr;

  Operation *term = block->getTerminator();
  if (isSupportedReturn(term))
    return term;
  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() != block->getParent())
      continue;
    if (Operation *returnOp = findReturnOnPath(successor, visited))
      return returnOp;
  }
  return nullptr;
}

static SmallVector<Value> mapValues(ValueRange values, IRMapping &mapping) {
  SmallVector<Value> mapped;
  mapped.reserve(values.size());
  for (Value value : values)
    mapped.push_back(mapping.lookupOrDefault(value));
  return mapped;
}

static FailureOr<SmallVector<Value>>
buildClonedTerminalPath(Block *block, ValueRange incoming, OpBuilder &builder,
                        IRMapping mapping,
                        SmallPtrSetImpl<Block *> &visiting);

static FailureOr<SmallVector<Value>>
buildClonedTerminalTerminator(Operation *term, OpBuilder &builder,
                              IRMapping mapping,
                              SmallPtrSetImpl<Block *> &visiting) {
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> incoming =
        mapValues(br.getDestOperands(), mapping);
    return buildClonedTerminalPath(br.getDest(), incoming, builder, mapping,
                                   visiting);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    SmallPtrSet<Block *, 16> thenVisiting;
    FailureOr<SmallVector<Type>> thenTypes =
        collectReturnPathTypes(condBr.getTrueDest(), thenVisiting);
    SmallPtrSet<Block *, 16> elseVisiting;
    FailureOr<SmallVector<Type>> elseTypes =
        collectReturnPathTypes(condBr.getFalseDest(), elseVisiting);
    if (failed(thenTypes) || failed(elseTypes))
      return failure();
    if (!haveSameTypes(*thenTypes, *elseTypes)) {
      condBr.emitError("terminal branch return types do not match");
      return failure();
    }

    auto ifOp = builder.create<scf::IfOp>(
        condBr.getLoc(), *thenTypes,
        mapping.lookupOrDefault(condBr.getCondition()),
        /*withElseRegion=*/true);

    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *autoYield =
          thenTypes->empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
      if (autoYield)
        builder.setInsertionPoint(autoYield);
      else
        builder.setInsertionPointToStart(ifOp.thenBlock());
      SmallVector<Value> incoming =
          mapValues(condBr.getTrueDestOperands(), mapping);
      FailureOr<SmallVector<Value>> thenReturn = buildClonedTerminalPath(
          condBr.getTrueDest(), incoming, builder, mapping, visiting);
      if (failed(thenReturn))
        return failure();
      if (!haveSameTypes(*thenReturn, *thenTypes)) {
        condBr.emitError("then terminal branch returns incompatible values");
        return failure();
      }
      if (!autoYield)
        builder.create<scf::YieldOp>(condBr.getLoc(), *thenReturn);
    }

    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *autoYield =
          thenTypes->empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
      if (autoYield)
        builder.setInsertionPoint(autoYield);
      else
        builder.setInsertionPointToStart(ifOp.elseBlock());
      SmallVector<Value> incoming =
          mapValues(condBr.getFalseDestOperands(), mapping);
      FailureOr<SmallVector<Value>> elseReturn = buildClonedTerminalPath(
          condBr.getFalseDest(), incoming, builder, mapping, visiting);
      if (failed(elseReturn))
        return failure();
      if (!haveSameTypes(*elseReturn, *thenTypes)) {
        condBr.emitError("else terminal branch returns incompatible values");
        return failure();
      }
      if (!autoYield)
        builder.create<scf::YieldOp>(condBr.getLoc(), *elseReturn);
    }

    return SmallVector<Value>(ifOp->getResults().begin(),
                              ifOp->getResults().end());
  }

  if (isSupportedReturn(term))
    return mapValues(term->getOperands(), mapping);

  return term->emitError()
         << "unsupported terminator while structuring terminal control flow";
}

static FailureOr<SmallVector<Value>>
buildClonedTerminalPath(Block *block, ValueRange incoming, OpBuilder &builder,
                        IRMapping mapping,
                        SmallPtrSetImpl<Block *> &visiting) {
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic terminal control flow";

  if (block->getNumArguments() != incoming.size()) {
    visiting.erase(block);
    return block->getTerminator()->emitError()
           << "invalid branch operand count while structuring terminal "
              "control flow";
  }

  for (auto [arg, value] : llvm::zip(block->getArguments(), incoming))
    mapping.map(arg, value);

  for (Operation &op : block->without_terminator())
    builder.clone(op, mapping);

  FailureOr<SmallVector<Value>> result =
      buildClonedTerminalTerminator(block->getTerminator(), builder, mapping,
                                    visiting);
  visiting.erase(block);
  return result;
}

static bool hasNonTreeCondBranch(Region &body) {
  for (Block &block : body) {
    auto condBr = dyn_cast<cf::CondBranchOp>(block.getTerminator());
    if (!condBr)
      continue;
    if (failed(findNearestCommonBlock(condBr.getTrueDest(),
                                      condBr.getFalseDest(), condBr.getLoc(),
                                      /*emitDiagnostic=*/false)))
      return true;
  }
  return false;
}

static LogicalResult structureTerminalReturnBody(Operation *funcOp,
                                                 Region &body) {
  Block &entryBlock = body.front();
  Operation *entryTerm = entryBlock.getTerminator();
  SmallPtrSet<Block *, 16> visited;
  Operation *sampleReturn = findReturnOnPath(&entryBlock, visited);
  if (!sampleReturn) {
    return funcOp->emitError()
           << "unsupported non-tree control flow: no terminal return found";
  }

  OpBuilder builder(entryTerm);
  IRMapping mapping;
  SmallPtrSet<Block *, 16> visiting;
  FailureOr<SmallVector<Value>> returnOperands =
      buildClonedTerminalTerminator(entryTerm, builder, mapping, visiting);
  if (failed(returnOperands))
    return failure();

  createReturnLike(builder, entryTerm->getLoc(), sampleReturn, *returnOperands);

  SmallVector<Block *> eraseBlocks;
  for (Block &block : llvm::drop_begin(body.getBlocks()))
    eraseBlocks.push_back(&block);

  entryTerm->erase();
  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

static FailureOr<scf::IfOp> buildTerminalValueIf(cf::CondBranchOp condBr,
                                                 OpBuilder &builder) {
  SmallPtrSet<Block *, 16> thenVisiting;
  FailureOr<SmallVector<Type>> thenTypes =
      collectReturnPathTypes(condBr.getTrueDest(), thenVisiting);
  SmallPtrSet<Block *, 16> elseVisiting;
  FailureOr<SmallVector<Type>> elseTypes =
      collectReturnPathTypes(condBr.getFalseDest(), elseVisiting);
  if (failed(thenTypes) || failed(elseTypes))
    return failure();
  if (!haveSameTypes(*thenTypes, *elseTypes)) {
    condBr.emitError("terminal branch return types do not match");
    return failure();
  }

  auto ifOp = builder.create<scf::IfOp>(
      condBr.getLoc(), *thenTypes, condBr.getCondition(),
      /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard branchGuard(builder);
    Operation *autoYield =
        thenTypes->empty() ? ifOp.thenBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.thenBlock());
    FailureOr<ReturnPathResult> thenReturn = buildReturnPath(
        condBr.getTrueDest(), condBr.getTrueDestOperands(), builder);
    if (failed(thenReturn))
      return failure();
    if (!haveSameTypes(thenReturn->operands, *thenTypes)) {
      condBr.emitError("then terminal branch returns incompatible values");
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), thenReturn->operands);
  }

  {
    OpBuilder::InsertionGuard branchGuard(builder);
    Operation *autoYield =
        thenTypes->empty() ? ifOp.elseBlock()->getTerminator() : nullptr;
    if (autoYield)
      builder.setInsertionPoint(autoYield);
    else
      builder.setInsertionPointToStart(ifOp.elseBlock());
    FailureOr<ReturnPathResult> elseReturn = buildReturnPath(
        condBr.getFalseDest(), condBr.getFalseDestOperands(), builder);
    if (failed(elseReturn))
      return failure();
    if (!haveSameTypes(elseReturn->operands, *thenTypes)) {
      condBr.emitError("else terminal branch returns incompatible values");
      return failure();
    }
    if (!autoYield)
      builder.create<scf::YieldOp>(condBr.getLoc(), elseReturn->operands);
  }

  return ifOp;
}

static FailureOr<ReturnPathResult>
buildReturnPath(Block *block, ValueRange incoming, OpBuilder &builder) {
  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();
  moveBlockBodyBefore(block, builder);

  if (auto br = dyn_cast<cf::BranchOp>(term))
    return buildReturnPath(br.getDest(), br.getDestOperands(), builder);

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (succeeded(nestedJoin)) {
      FailureOr<scf::IfOp> nestedIf =
          buildStructuredIf(condBr, *nestedJoin, builder);
      if (failed(nestedIf))
        return failure();

      SmallVector<Value> nestedResults((*nestedIf)->getResults().begin(),
                                       (*nestedIf)->getResults().end());
      return buildReturnPath(*nestedJoin, nestedResults, builder);
    }

    FailureOr<scf::IfOp> terminalIf = buildTerminalValueIf(condBr, builder);
    if (failed(terminalIf)) {
      condBr.emitError() << "unsupported non-tree control flow: branch arms do "
                            "not both terminate with compatible returns";
      return failure();
    }

    ReturnPathResult result;
    result.operands.assign((*terminalIf)->getResults().begin(),
                           (*terminalIf)->getResults().end());
    return result;
  }

  if (isSupportedReturn(term)) {
    ReturnPathResult result;
    result.operands.assign(term->getOperands().begin(),
                           term->getOperands().end());
    return result;
  }

  return term->emitError()
         << "unsupported terminator while structuring terminal control flow";
}

static FailureOr<SmallVector<Value>>
buildRegionPath(Block *block, ValueRange incoming, Block *stopBlock,
                OpBuilder &builder) {
  if (block == stopBlock)
    return SmallVector<Value>(incoming.begin(), incoming.end());

  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();
  moveBlockBodyBefore(block, builder);

  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> operands(br.getDestOperands().begin(),
                                br.getDestOperands().end());
    if (br.getDest() == stopBlock)
      return operands;
    return buildRegionPath(br.getDest(), operands, stopBlock, builder);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> nestedJoin = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc());
    if (failed(nestedJoin))
      return failure();

    FailureOr<scf::IfOp> nestedIf =
        buildStructuredIf(condBr, *nestedJoin, builder);
    if (failed(nestedIf))
      return failure();

    SmallVector<Value> nestedResults((*nestedIf)->getResults().begin(),
                                     (*nestedIf)->getResults().end());
    if (*nestedJoin == stopBlock)
      return nestedResults;
    return buildRegionPath(*nestedJoin, nestedResults, stopBlock, builder);
  }

  if (isSupportedReturn(term)) {
    return term->emitError()
           << "unsupported early return while structuring control flow";
  }

  return term->emitError()
         << "unsupported terminator while structuring control flow";
}

static LogicalResult appendStructuredBlock(Block *block, ValueRange incoming,
                                           OpBuilder &builder,
                                           Operation *anchorTerminator);

static LogicalResult appendStructuredTerminator(Operation *term,
                                                OpBuilder &builder,
                                                Operation *anchorTerminator) {
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    return appendStructuredBlock(br.getDest(), br.getDestOperands(), builder,
                                 anchorTerminator);
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    FailureOr<Block *> joinBlock = findNearestCommonBlock(
        condBr.getTrueDest(), condBr.getFalseDest(), condBr.getLoc(),
        /*emitDiagnostic=*/false);
    if (failed(joinBlock)) {
      SmallPtrSet<Block *, 16> visited;
      Operation *sampleReturn = findReturnOnPath(condBr.getTrueDest(), visited);
      if (!sampleReturn) {
        visited.clear();
        sampleReturn = findReturnOnPath(condBr.getFalseDest(), visited);
      }
      if (!sampleReturn) {
        return condBr.emitError()
               << "unsupported non-tree control flow: branch arms do not "
                  "reach a common convergence block";
      }

      FailureOr<scf::IfOp> terminalIf =
          buildTerminalValueIf(condBr, builder);
      if (failed(terminalIf))
        return failure();

      SmallVector<Value> returnOperands((*terminalIf)->getResults().begin(),
                                        (*terminalIf)->getResults().end());
      createReturnLike(builder, condBr.getLoc(), sampleReturn, returnOperands);
      return success();
    }

    FailureOr<scf::IfOp> ifOp =
        buildStructuredIf(condBr, *joinBlock, builder);
    if (failed(ifOp))
      return failure();

    return appendStructuredBlock(*joinBlock, (*ifOp)->getResults(), builder,
                                 anchorTerminator);
  }

  if (isSupportedReturn(term)) {
    term->moveBefore(anchorTerminator);
    return success();
  }

  return term->emitError()
         << "unsupported entry terminator while structuring control flow";
}

static LogicalResult appendStructuredBlock(Block *block, ValueRange incoming,
                                           OpBuilder &builder,
                                           Operation *anchorTerminator) {
  Operation *term = block->getTerminator();
  if (failed(replaceBlockArguments(block, incoming, term->getLoc())))
    return failure();

  moveBlockBodyBefore(block, builder);
  return appendStructuredTerminator(term, builder, anchorTerminator);
}

static LogicalResult validateSupportedCfg(Region &body) {
  for (Block &block : body) {
    Operation *term = block.getTerminator();
    if (!isa<cf::BranchOp, cf::CondBranchOp>(term) && !isSupportedReturn(term))
      return term->emitError()
             << "unsupported terminator in multi-block function";
  }
  return success();
}

static void collectReachableBlocks(Block *block,
                                   SmallPtrSetImpl<Block *> &reachable) {
  if (!reachable.insert(block).second)
    return;

  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() == block->getParent())
      collectReachableBlocks(successor, reachable);
  }
}

static void eraseUnreachableBlocks(Region &body) {
  if (body.empty() || body.hasOneBlock())
    return;

  SmallPtrSet<Block *, 16> reachable;
  collectReachableBlocks(&body.front(), reachable);

  SmallVector<Block *> eraseBlocks;
  for (Block &block : body) {
    if (!reachable.contains(&block))
      eraseBlocks.push_back(&block);
  }
  if (eraseBlocks.empty())
    return;

  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();
}

static LogicalResult rejectCyclicCfg(Block *block,
                                     SmallPtrSetImpl<Block *> &visiting,
                                     SmallPtrSetImpl<Block *> &visited) {
  if (visited.contains(block))
    return success();
  if (!visiting.insert(block).second)
    return block->getTerminator()->emitError()
           << "unsupported cyclic control flow in multi-block function";

  for (Block *successor : getCfgSuccessors(block)) {
    if (successor->getParent() == block->getParent() &&
        failed(rejectCyclicCfg(successor, visiting, visited)))
      return failure();
  }

  visiting.erase(block);
  visited.insert(block);
  return success();
}

static LogicalResult structureFunctionBody(Operation *funcOp, Region &body) {
  if (body.empty() || body.hasOneBlock())
    return success();

  eraseUnreachableBlocks(body);
  if (body.hasOneBlock())
    return success();

  if (failed(validateSupportedCfg(body)))
    return failure();

  SmallPtrSet<Block *, 16> visiting;
  SmallPtrSet<Block *, 16> visited;
  if (failed(rejectCyclicCfg(&body.front(), visiting, visited)))
    return failure();

  if (hasNonTreeCondBranch(body))
    return structureTerminalReturnBody(funcOp, body);

  Block &entryBlock = body.front();
  Operation *entryTerm = entryBlock.getTerminator();
  if (isSupportedReturn(entryTerm)) {
    return funcOp->emitError()
           << "multi-block function entry cannot terminate with return";
  }

  SmallVector<Block *> eraseBlocks;
  for (Block &block : llvm::drop_begin(body.getBlocks()))
    eraseBlocks.push_back(&block);

  OpBuilder builder(entryTerm);
  if (failed(appendStructuredTerminator(entryTerm, builder, entryTerm)))
    return failure();

  entryTerm->erase();
  for (Block *block : eraseBlocks) {
    for (Operation &op : *block)
      op.dropAllReferences();
  }
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

enum class PtrKind { Tensor, Block };

struct TensorPtrInfo {
  Type resultType;
  Value base;
  Value offset;
  Value scalarOffset;
  bool scalarBase = false;
};

struct BlockPtrInfo {
  Type resultType;
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  DenseI32ArrayAttr order;
};

struct CFPtrInfo {
  PtrKind kind;
  TensorPtrInfo tensor;
  BlockPtrInfo block;
};

struct RewriteEnv {
  IRMapping valueMapping;
  DenseMap<Value, CFPtrInfo> pointerComponents;
};

struct LoopPointerInfo {
  unsigned oldIndex = 0;
  CFPtrInfo initInfo;
  SmallVector<unsigned> newIndices;
  SmallVector<Value> ivDeltas;
};

enum class IfComponentKind {
  TensorOffset,
  BlockShape,
  BlockStride,
  BlockOffset
};

struct IfComponent {
  IfComponentKind kind;
  unsigned dim = 0;
  Type type;
};

struct IfPointerInfo {
  unsigned oldIndex = 0;
  CFPtrInfo thenInfo;
  CFPtrInfo elseInfo;
  SmallVector<IfComponent> components;
};

static bool isTensorPointerType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

static bool isBlockPointerType(Type type) {
  auto ptrType = dyn_cast<triton::PointerType>(type);
  return ptrType && isa<RankedTensorType>(ptrType.getPointeeType());
}

static bool isControlFlowPointerType(Type type) {
  return isTensorPointerType(type) || isBlockPointerType(type);
}

static Value createZeroLike(OpBuilder &builder, Location loc, Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
    if (!elementType)
      return nullptr;
    auto attr = DenseElementsAttr::get(
        tensorType, builder.getIntegerAttr(elementType, 0));
    return builder.create<arith::ConstantOp>(loc, attr);
  }

  if (type.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, 0);

  if (auto intType = dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantIntOp>(loc, 0, intType);

  return nullptr;
}

static Value createZeroOffset(OpBuilder &builder, Location loc, Type ptrType) {
  Type i32 = builder.getI32Type();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptrType))
    return createZeroLike(builder, loc,
                          RankedTensorType::get(tensorType.getShape(), i32));
  return createZeroLike(builder, loc, i32);
}

static bool isScalarIntegerLike(Type type) {
  return type.isIndex() || isa<IntegerType>(type);
}

static Type getTensorElementType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType();
  return type;
}

static Value createZeroScalarOffset(OpBuilder &builder, Location loc,
                                    Type offsetType) {
  return createZeroLike(builder, loc, getTensorElementType(offsetType));
}

static Value castIntegerLike(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (value.getType() == targetType)
    return value;

  Type sourceType = value.getType();
  if ((sourceType.isIndex() && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(sourceType) && targetType.isIndex()))
    return builder.create<arith::IndexCastOp>(loc, targetType, value);

  auto sourceInt = dyn_cast<IntegerType>(sourceType);
  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (sourceInt && targetInt) {
    if (sourceInt.getWidth() < targetInt.getWidth())
      return builder.create<arith::ExtSIOp>(loc, targetType, value);
    if (sourceInt.getWidth() > targetInt.getWidth())
      return builder.create<arith::TruncIOp>(loc, targetType, value);
    return nullptr;
  }

  auto sourceTensor = dyn_cast<RankedTensorType>(sourceType);
  auto targetTensor = dyn_cast<RankedTensorType>(targetType);
  if (!sourceTensor || !targetTensor ||
      sourceTensor.getShape() != targetTensor.getShape())
    return nullptr;

  auto sourceElement = dyn_cast<IntegerType>(sourceTensor.getElementType());
  auto targetElement = dyn_cast<IntegerType>(targetTensor.getElementType());
  if (!sourceElement || !targetElement)
    return nullptr;

  if (sourceElement.getWidth() == targetElement.getWidth())
    return value;
  if (sourceElement.getWidth() < targetElement.getWidth())
    return builder.create<arith::ExtSIOp>(loc, targetType, value);
  return builder.create<arith::TruncIOp>(loc, targetType, value);
}

static FailureOr<Type> getWiderIntegerLikeType(Type lhs, Type rhs) {
  if (lhs == rhs)
    return lhs;

  if (lhs.isIndex() && rhs.isIndex())
    return lhs;
  if (lhs.isIndex() && isa<IntegerType>(rhs))
    return lhs;
  if (isa<IntegerType>(lhs) && rhs.isIndex())
    return rhs;

  auto lhsInt = dyn_cast<IntegerType>(lhs);
  auto rhsInt = dyn_cast<IntegerType>(rhs);
  if (lhsInt && rhsInt)
    return lhsInt.getWidth() >= rhsInt.getWidth() ? lhs : rhs;

  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (!lhsTensor || !rhsTensor ||
      lhsTensor.getShape() != rhsTensor.getShape())
    return failure();

  Type lhsElement = lhsTensor.getElementType();
  Type rhsElement = rhsTensor.getElementType();
  if (lhsElement == rhsElement)
    return lhs;
  if (lhsElement.isIndex() && isa<IntegerType>(rhsElement))
    return lhs;
  if (isa<IntegerType>(lhsElement) && rhsElement.isIndex())
    return rhs;

  auto lhsElementInt = dyn_cast<IntegerType>(lhsElement);
  auto rhsElementInt = dyn_cast<IntegerType>(rhsElement);
  if (!lhsElementInt || !rhsElementInt)
    return failure();
  return lhsElementInt.getWidth() >= rhsElementInt.getWidth() ? lhs : rhs;
}

static Value createAdd(OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  if (lhs.getType() != rhs.getType()) {
    rhs = castIntegerLike(builder, loc, rhs, lhs.getType());
    if (!rhs)
      return nullptr;
  }
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static Value createAddWithWiderType(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs) {
  if (!lhs || !rhs)
    return nullptr;

  FailureOr<Type> targetType =
      getWiderIntegerLikeType(lhs.getType(), rhs.getType());
  if (failed(targetType))
    return nullptr;

  lhs = castIntegerLike(builder, loc, lhs, *targetType);
  rhs = castIntegerLike(builder, loc, rhs, *targetType);
  if (!lhs || !rhs)
    return nullptr;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static FailureOr<Value>
getUniformScalarOffset(Value offset, OpBuilder &builder, Location loc) {
  Type offsetType = offset.getType();
  if (isScalarIntegerLike(offsetType))
    return offset;

  auto tensorType = dyn_cast<RankedTensorType>(offsetType);
  if (!tensorType || !isScalarIntegerLike(tensorType.getElementType()))
    return failure();

  if (auto splatOp = offset.getDefiningOp<triton::SplatOp>()) {
    if (splatOp.getSrc().getType() == tensorType.getElementType())
      return splatOp.getSrc();
  }

  Attribute constAttr;
  if (!matchPattern(offset, m_Constant(&constAttr)))
    return failure();

  auto denseAttr = dyn_cast<DenseElementsAttr>(constAttr);
  if (!denseAttr || !denseAttr.isSplat())
    return failure();

  if (auto intAttr =
          dyn_cast<IntegerAttr>(denseAttr.getSplatValue<Attribute>())) {
    return builder
        .create<arith::ConstantIntOp>(
            loc, intAttr.getValue().getSExtValue(),
            cast<IntegerType>(tensorType.getElementType()))
        .getResult();
  }

  return failure();
}

static bool hasStructuredScalarOffset(const TensorPtrInfo &parts) {
  return parts.scalarBase && parts.scalarOffset &&
         isa<RankedTensorType>(parts.offset.getType());
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  Attribute lhsAttr;
  Attribute rhsAttr;
  if (!matchPattern(lhs, m_Constant(&lhsAttr)) ||
      !matchPattern(rhs, m_Constant(&rhsAttr)))
    return false;
  return lhsAttr == rhsAttr;
}

static Value materializeTensorOffset(OpBuilder &builder, Location loc,
                                     const TensorPtrInfo &parts) {
  if (!hasStructuredScalarOffset(parts))
    return parts.offset;

  auto offsetType = cast<RankedTensorType>(parts.offset.getType());
  Value scalarOffset =
      castIntegerLike(builder, loc, parts.scalarOffset,
                      offsetType.getElementType());
  if (!scalarOffset)
    return nullptr;

  Value splatOffset =
      builder.create<triton::SplatOp>(loc, offsetType, scalarOffset);
  return createAdd(builder, loc, parts.offset, splatOffset);
}

static Value createMul(OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  if (lhs.getType() != rhs.getType()) {
    rhs = castIntegerLike(builder, loc, rhs, lhs.getType());
    if (!rhs)
      return nullptr;
  }
  return builder.create<arith::MulIOp>(loc, lhs, rhs);
}

static Value remapValue(Value value, const RewriteEnv &env) {
  if (Value mapped = env.valueMapping.lookupOrNull(value))
    return mapped;
  return value;
}

static FailureOr<CFPtrInfo> analyzePtr(Value value, const RewriteEnv &env,
                                       OpBuilder &builder, Location loc);

static FailureOr<TensorPtrInfo>
analyzeTensorPtr(Value value, const RewriteEnv &env, OpBuilder &builder,
                 Location loc) {
  if (auto it = env.pointerComponents.find(value);
      it != env.pointerComponents.end() && it->second.kind == PtrKind::Tensor)
    return it->second.tensor;

  value = remapValue(value, env);

  if (auto addPtrOp = value.getDefiningOp<triton::AddPtrOp>()) {
    FailureOr<CFPtrInfo> baseInfo =
        analyzePtr(addPtrOp.getPtr(), env, builder, loc);
    if (failed(baseInfo) || (*baseInfo).kind != PtrKind::Tensor)
      return failure();

    TensorPtrInfo tensor = (*baseInfo).tensor;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(addPtrOp);
    Value offset = addPtrOp.getOffset();
    if (hasStructuredScalarOffset(tensor)) {
      FailureOr<Value> scalarDelta =
          getUniformScalarOffset(offset, builder, addPtrOp.getLoc());
      if (succeeded(scalarDelta)) {
        Value newScalarOffset = createAddWithWiderType(
            builder, addPtrOp.getLoc(), tensor.scalarOffset, *scalarDelta);
        if (!newScalarOffset)
          return failure();
        tensor.scalarOffset = newScalarOffset;
        tensor.resultType = value.getType();
        return tensor;
      }

      Value newLaneOffset = createAddWithWiderType(
          builder, addPtrOp.getLoc(), tensor.offset, offset);
      if (!newLaneOffset)
        return failure();
      tensor.offset = newLaneOffset;
      tensor.resultType = value.getType();
      return tensor;
    }

    Value newOffset = createAddWithWiderType(builder, addPtrOp.getLoc(),
                                             tensor.offset, offset);
    if (!newOffset)
      return failure();
    tensor.offset = newOffset;
    tensor.resultType = value.getType();
    return tensor;
  }

  if (auto splatOp = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<triton::PointerType>(splatOp.getSrc().getType()))
      return failure();
    TensorPtrInfo parts;
    parts.resultType = value.getType();
    parts.base = splatOp.getSrc();
    parts.scalarBase = true;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(splatOp);
    parts.offset = createZeroOffset(builder, splatOp.getLoc(), value.getType());
    if (!parts.offset)
      return failure();
    parts.scalarOffset = createZeroScalarOffset(builder, splatOp.getLoc(),
                                                parts.offset.getType());
    if (!parts.scalarOffset)
      return failure();
    return parts;
  }

  if (!isTensorPointerType(value.getType()))
    return failure();

  TensorPtrInfo parts;
  parts.resultType = value.getType();
  parts.base = value;
  parts.scalarBase = false;
  OpBuilder::InsertionGuard guard(builder);
  if (Operation *defOp = value.getDefiningOp())
    builder.setInsertionPointAfter(defOp);
  else if (auto blockArg = dyn_cast<BlockArgument>(value))
    builder.setInsertionPointToStart(blockArg.getOwner());
  parts.offset = createZeroOffset(builder, loc, value.getType());
  if (!parts.offset)
    return failure();
  return parts;
}

static FailureOr<BlockPtrInfo>
analyzeBlockPtr(Value value, const RewriteEnv &env, OpBuilder &builder,
                Location loc) {
  if (auto it = env.pointerComponents.find(value);
      it != env.pointerComponents.end() && it->second.kind == PtrKind::Block)
    return it->second.block;

  value = remapValue(value, env);

  if (auto makePtrOp = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    BlockPtrInfo parts;
    parts.resultType = value.getType();
    parts.base = makePtrOp.getBase();
    parts.shape.assign(makePtrOp.getShape().begin(), makePtrOp.getShape().end());
    parts.strides.assign(makePtrOp.getStrides().begin(),
                         makePtrOp.getStrides().end());
    parts.offsets.assign(makePtrOp.getOffsets().begin(),
                         makePtrOp.getOffsets().end());
    parts.order = makePtrOp.getOrderAttr();
    return parts;
  }

  if (auto advanceOp = value.getDefiningOp<triton::AdvanceOp>()) {
    FailureOr<CFPtrInfo> baseInfo =
        analyzePtr(advanceOp.getPtr(), env, builder, loc);
    if (failed(baseInfo) || (*baseInfo).kind != PtrKind::Block)
      return failure();
    BlockPtrInfo block = (*baseInfo).block;
    if (block.offsets.size() != advanceOp.getOffsets().size())
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(advanceOp);
    for (auto [idx, delta] : llvm::enumerate(advanceOp.getOffsets())) {
      Value newOffset =
          createAdd(builder, advanceOp.getLoc(), block.offsets[idx], delta);
      if (!newOffset)
        return failure();
      block.offsets[idx] = newOffset;
    }
    block.resultType = value.getType();
    return block;
  }

  return failure();
}

static FailureOr<CFPtrInfo> analyzePtr(Value value, const RewriteEnv &env,
                                       OpBuilder &builder, Location loc) {
  if (auto it = env.pointerComponents.find(value);
      it != env.pointerComponents.end())
    return it->second;

  value = remapValue(value, env);

  if (isBlockPointerType(value.getType())) {
    FailureOr<BlockPtrInfo> block = analyzeBlockPtr(value, env, builder, loc);
    if (failed(block))
      return failure();
    CFPtrInfo info{PtrKind::Block};
    info.block = *block;
    return info;
  }

  if (isTensorPointerType(value.getType())) {
    FailureOr<TensorPtrInfo> tensor = analyzeTensorPtr(value, env, builder, loc);
    if (failed(tensor))
      return failure();
    CFPtrInfo info{PtrKind::Tensor};
    info.tensor = *tensor;
    return info;
  }

  return failure();
}

static Value rebuildTensorPtr(OpBuilder &builder, Location loc,
                              const TensorPtrInfo &parts, Value base,
                              Value offset) {
  Value ptrBase = base;
  if (parts.scalarBase && isTensorPointerType(parts.resultType))
    ptrBase = builder.create<triton::SplatOp>(loc, parts.resultType, base);
  TensorPtrInfo materializedParts = parts;
  materializedParts.offset = offset;
  Value materializedOffset =
      materializeTensorOffset(builder, loc, materializedParts);
  if (!materializedOffset)
    return nullptr;
  return builder.create<triton::AddPtrOp>(loc, parts.resultType, ptrBase,
                                          materializedOffset);
}

static Value rebuildBlockPtr(OpBuilder &builder, Location loc,
                             const BlockPtrInfo &parts, Value base,
                             ArrayRef<Value> shape, ArrayRef<Value> strides,
                             ArrayRef<Value> offsets) {
  return builder.create<triton::MakeTensorPtrOp>(
      loc, parts.resultType, base, ValueRange(shape), ValueRange(strides),
      ValueRange(offsets), parts.order);
}

static Value rebuildPtr(OpBuilder &builder, Location loc,
                        const CFPtrInfo &info) {
  if (info.kind == PtrKind::Tensor)
    return rebuildTensorPtr(builder, loc, info.tensor, info.tensor.base,
                            info.tensor.offset);
  return rebuildBlockPtr(builder, loc, info.block, info.block.base,
                         info.block.shape, info.block.strides,
                         info.block.offsets);
}

static void recordPointer(Value oldPtr, const CFPtrInfo &info, Value rebuiltPtr,
                          RewriteEnv &env) {
  env.pointerComponents[oldPtr] = info;
  env.pointerComponents[rebuiltPtr] = info;
  env.valueMapping.map(oldPtr, rebuiltPtr);
}

static SmallVector<Value> getLoopComponentValues(const CFPtrInfo &info) {
  if (info.kind == PtrKind::Tensor) {
    if (hasStructuredScalarOffset(info.tensor))
      return {info.tensor.scalarOffset};
    return {info.tensor.offset};
  }
  return info.block.offsets;
}

static SmallVector<Type> getLoopComponentTypes(const CFPtrInfo &info) {
  SmallVector<Type> types;
  for (Value value : getLoopComponentValues(info))
    types.push_back(value.getType());
  return types;
}

static CFPtrInfo withLoopComponentValues(CFPtrInfo info,
                                         ArrayRef<Value> values) {
  if (info.kind == PtrKind::Tensor) {
    if (values.size() != 1)
      return info;
    if (hasStructuredScalarOffset(info.tensor)) {
      info.tensor.scalarOffset = values[0];
      return info;
    }
    info.tensor.offset = values[0];
    return info;
  }

  if (values.size() == info.block.offsets.size())
    info.block.offsets.assign(values.begin(), values.end());
  return info;
}

static bool areLoopCompatible(const CFPtrInfo &initInfo,
                              const CFPtrInfo &nextInfo) {
  if (initInfo.kind != nextInfo.kind)
    return false;
  if (initInfo.kind == PtrKind::Tensor) {
    if (initInfo.tensor.resultType != nextInfo.tensor.resultType ||
        initInfo.tensor.base != nextInfo.tensor.base ||
        initInfo.tensor.scalarBase != nextInfo.tensor.scalarBase)
      return false;
    if (hasStructuredScalarOffset(initInfo.tensor) !=
        hasStructuredScalarOffset(nextInfo.tensor))
      return false;
    if (hasStructuredScalarOffset(initInfo.tensor) &&
        !areEquivalentValues(initInfo.tensor.offset, nextInfo.tensor.offset))
      return false;
    return haveSameTypes(getLoopComponentValues(initInfo),
                         getLoopComponentValues(nextInfo));
  }

  return initInfo.block.resultType == nextInfo.block.resultType &&
         initInfo.block.base == nextInfo.block.base &&
         initInfo.block.order == nextInfo.block.order &&
         initInfo.block.shape == nextInfo.block.shape &&
         initInfo.block.strides == nextInfo.block.strides &&
         initInfo.block.offsets.size() == nextInfo.block.offsets.size() &&
         haveSameTypes(getLoopComponentValues(initInfo),
                       getLoopComponentValues(nextInfo));
}

static bool isConstantIndex(Value value, int64_t expected) {
  auto constOp = value.getDefiningOp<arith::ConstantIndexOp>();
  return constOp && constOp.value() == expected;
}

static bool isRangeFromZeroByOne(scf::ForOp forOp) {
  return isConstantIndex(forOp.getLowerBound(), 0) &&
         isConstantIndex(forOp.getStep(), 1);
}

static bool isDefinedOutside(Operation *scope, Value value) {
  if (Operation *defOp = value.getDefiningOp())
    return !scope->isAncestor(defOp);

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;

  Operation *ownerOp = blockArg.getOwner()->getParentOp();
  return ownerOp != scope && (!ownerOp || !scope->isAncestor(ownerOp));
}

static FailureOr<SmallVector<Value>>
matchSimpleForIvDeltas(scf::ForOp forOp, const LoopPointerInfo &info,
                       Value yieldOperand) {
  if (!isRangeFromZeroByOne(forOp) || info.initInfo.kind != PtrKind::Block)
    return failure();

  auto advanceOp = yieldOperand.getDefiningOp<triton::AdvanceOp>();
  if (!advanceOp ||
      advanceOp.getPtr() != forOp.getRegionIterArgs()[info.oldIndex])
    return failure();

  if (advanceOp.getOffsets().size() != info.initInfo.block.offsets.size())
    return failure();

  SmallVector<Value> deltas;
  for (auto [initOffset, delta] :
       llvm::zip(info.initInfo.block.offsets, advanceOp.getOffsets())) {
    if (!isScalarIntegerLike(initOffset.getType()) ||
        !isScalarIntegerLike(delta.getType()) ||
        !isDefinedOutside(forOp, delta))
      return failure();
    deltas.push_back(delta);
  }
  return deltas;
}

static FailureOr<CFPtrInfo>
withForIvClosedFormComponents(const LoopPointerInfo &info, Value iv,
                              OpBuilder &builder, Location loc,
                              const RewriteEnv &env) {
  if (info.ivDeltas.empty())
    return failure();

  SmallVector<Value> initComponents = getLoopComponentValues(info.initInfo);
  if (initComponents.size() != info.ivDeltas.size())
    return failure();

  SmallVector<Value> components;
  components.reserve(initComponents.size());
  for (auto [initComponent, delta] :
       llvm::zip(initComponents, info.ivDeltas)) {
    Type componentType = initComponent.getType();
    if (!isScalarIntegerLike(componentType))
      return failure();

    Value typedIv = castIntegerLike(builder, loc, iv, componentType);
    Value typedDelta =
        castIntegerLike(builder, loc, remapValue(delta, env), componentType);
    if (!typedIv || !typedDelta)
      return failure();

    Value scaledDelta = createMul(builder, loc, typedIv, typedDelta);
    Value component = createAdd(builder, loc, initComponent, scaledDelta);
    if (!scaledDelta || !component)
      return failure();
    components.push_back(component);
  }

  return withLoopComponentValues(info.initInfo, components);
}

static LoopPointerInfo *findLoopInfo(SmallVectorImpl<LoopPointerInfo> &infos,
                                     unsigned oldIndex) {
  for (LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const LoopPointerInfo *findLoopInfo(ArrayRef<LoopPointerInfo> infos,
                                           unsigned oldIndex) {
  for (const LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static SmallVector<Value>
collectForComponents(const LoopPointerInfo &info, scf::ForOp forOp,
                     bool useResults) {
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices)
    values.push_back(useResults ? forOp.getResult(newIndex)
                                : forOp.getRegionIterArgs()[newIndex]);
  return values;
}

static SmallVector<Value>
collectWhileComponents(const LoopPointerInfo &info, scf::WhileOp whileOp,
                       bool useResults, bool useAfterArgs) {
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices) {
    if (useResults)
      values.push_back(whileOp.getResult(newIndex));
    else if (useAfterArgs)
      values.push_back(whileOp.getAfterArguments()[newIndex]);
    else
      values.push_back(whileOp.getBeforeArguments()[newIndex]);
  }
  return values;
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env);

static LogicalResult materializePointerResult(Operation &bodyOp,
                                              Operation *clonedOp,
                                              OpBuilder &builder,
                                              RewriteEnv &env) {
  if (!isa<triton::AddPtrOp, triton::AdvanceOp>(bodyOp))
    return success();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(clonedOp);

  for (auto [oldResult, clonedResult] :
       llvm::zip(bodyOp.getResults(), clonedOp->getResults())) {
    if (!isControlFlowPointerType(oldResult.getType()))
      continue;

    FailureOr<CFPtrInfo> info =
        analyzePtr(clonedResult, env, builder, oldResult.getLoc());
    if (failed(info))
      continue;

    Value rebuilt = rebuildPtr(builder, oldResult.getLoc(), *info);
    if (!rebuilt)
      return failure();
    recordPointer(oldResult, *info, rebuilt, env);
  }

  return success();
}

static LogicalResult rewriteBodyOps(Block *oldBlock, OpBuilder &builder,
                                    RewriteEnv &env) {
  for (Operation &bodyOp : oldBlock->without_terminator()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(bodyOp) &&
        succeeded(rewriteControlFlowOp(&bodyOp, builder, env)))
      continue;
    Operation *clonedOp = builder.clone(bodyOp, env.valueMapping);
    if (failed(materializePointerResult(bodyOp, clonedOp, builder, env)))
      return failure();
  }
  return success();
}

static LogicalResult rewriteForOp(scf::ForOp forOp, OpBuilder &builder,
                                  RewriteEnv &env) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(forOp.getContext());
  analysisBuilder.setInsertionPoint(forOp);

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!isControlFlowPointerType(iterArg.getType()))
      continue;
    if (idx >= forOp.getInitArgs().size() || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<CFPtrInfo> initInfo =
        analyzePtr(forOp.getInitArgs()[idx], env, analysisBuilder,
                   forOp.getLoc());
    if (failed(initInfo))
      continue;
    pointerInfos.push_back(
        LoopPointerInfo{static_cast<unsigned>(idx), *initInfo, {}});
  }

  if (pointerInfos.empty())
    return failure();

  for (LoopPointerInfo &info : pointerInfos) {
    FailureOr<SmallVector<Value>> deltas =
        matchSimpleForIvDeltas(forOp, info,
                               yieldOp.getOperand(info.oldIndex));
    if (succeeded(deltas))
      info.ivDeltas = *deltas;
  }

  SmallVector<Value> newInitArgs;
  SmallVector<unsigned> oldToNewStart(forOp.getInitArgs().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    oldToNewStart[idx] = newInitArgs.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component : getLoopComponentValues(info->initInfo)) {
        info->newIndices.push_back(newInitArgs.size());
        newInitArgs.push_back(component);
      }
      continue;
    }
    newInitArgs.push_back(remapValue(initArg, env));
  }

  bool bodyOk = true;
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), remapValue(forOp.getLowerBound(), env),
      remapValue(forOp.getUpperBound(), env), remapValue(forOp.getStep(), env),
      newInitArgs,
      [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange args) {
        RewriteEnv bodyEnv = env;
        bodyEnv.valueMapping.map(forOp.getInductionVar(), iv);

        for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            CFPtrInfo argInfo = withLoopComponentValues(info->initInfo, values);
            FailureOr<CFPtrInfo> closedFormInfo =
                withForIvClosedFormComponents(*info, iv, bodyBuilder, loc,
                                              bodyEnv);
            if (succeeded(closedFormInfo))
              argInfo = *closedFormInfo;
            Value rebuilt = rebuildPtr(bodyBuilder, loc, argInfo);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordPointer(oldArg, argInfo, rebuilt, bodyEnv);
            continue;
          }
          bodyEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(forOp.getBody(), bodyBuilder, bodyEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<CFPtrInfo> nextInfo =
                analyzePtr(oldOperand, bodyEnv, bodyBuilder,
                           yieldOp.getLoc());
            if (failed(nextInfo) ||
                !areLoopCompatible(info->initInfo, *nextInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component : getLoopComponentValues(*nextInfo))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, bodyEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newForOp->setAttrs(forOp->getAttrs());

  if (!bodyOk) {
    newForOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newForOp);
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      CFPtrInfo resultInfo = withLoopComponentValues(
          info->initInfo,
          collectForComponents(*info, newForOp, /*useResults=*/true));
      Value rebuilt = rebuildPtr(builder, oldResult.getLoc(), resultInfo);
      if (!rebuilt) {
        newForOp.erase();
        return failure();
      }
      recordPointer(oldResult, resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newForOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

static LogicalResult rewriteWhileOp(scf::WhileOp whileOp, OpBuilder &builder,
                                    RewriteEnv &env) {
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(whileOp.getContext());
  analysisBuilder.setInsertionPoint(whileOp);

  for (auto [idx, beforeArg] : llvm::enumerate(whileOp.getBeforeArguments())) {
    if (!isControlFlowPointerType(beforeArg.getType()))
      continue;
    if (idx >= whileOp.getInits().size() || idx >= conditionOp.getArgs().size()
        || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<CFPtrInfo> initInfo =
        analyzePtr(whileOp.getInits()[idx], env, analysisBuilder,
                   whileOp.getLoc());
    if (failed(initInfo))
      continue;
    pointerInfos.push_back(
        LoopPointerInfo{static_cast<unsigned>(idx), *initInfo, {}});
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Value> newInits;
  SmallVector<Type> newResultTypes;
  SmallVector<unsigned> oldToNewStart(whileOp.getInits().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(whileOp.getInits())) {
    oldToNewStart[idx] = newInits.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component : getLoopComponentValues(info->initInfo)) {
        info->newIndices.push_back(newInits.size());
        newInits.push_back(component);
        newResultTypes.push_back(component.getType());
      }
      continue;
    }
    newInits.push_back(remapValue(initArg, env));
    newResultTypes.push_back(whileOp.getResult(idx).getType());
  }

  bool bodyOk = true;
  auto newWhileOp = builder.create<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv beforeEnv = env;
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getBeforeArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            CFPtrInfo argInfo = withLoopComponentValues(info->initInfo, values);
            Value rebuilt = rebuildPtr(bodyBuilder, loc, argInfo);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordPointer(oldArg, argInfo, rebuilt, beforeEnv);
            continue;
          }
          beforeEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(whileOp.getBeforeBody(), bodyBuilder,
                                  beforeEnv)))
          bodyOk = false;

        SmallVector<Value> newConditionArgs;
        for (auto [idx, oldArg] : llvm::enumerate(conditionOp.getArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<CFPtrInfo> conditionInfo =
                analyzePtr(oldArg, beforeEnv, bodyBuilder,
                           conditionOp.getLoc());
            if (failed(conditionInfo) ||
                !areLoopCompatible(info->initInfo, *conditionInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newConditionArgs.push_back(args[newIndex]);
              continue;
            }
            for (Value component : getLoopComponentValues(*conditionInfo))
              newConditionArgs.push_back(component);
            continue;
          }
          newConditionArgs.push_back(remapValue(oldArg, beforeEnv));
        }

        bodyBuilder.create<scf::ConditionOp>(
            conditionOp.getLoc(),
            remapValue(conditionOp.getCondition(), beforeEnv),
            newConditionArgs);
      },
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv afterEnv = env;
        for (auto [idx, oldArg] : llvm::enumerate(whileOp.getAfterArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            CFPtrInfo argInfo = withLoopComponentValues(info->initInfo, values);
            Value rebuilt = rebuildPtr(bodyBuilder, loc, argInfo);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordPointer(oldArg, argInfo, rebuilt, afterEnv);
            continue;
          }
          afterEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(whileOp.getAfterBody(), bodyBuilder,
                                  afterEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<CFPtrInfo> nextInfo =
                analyzePtr(oldOperand, afterEnv, bodyBuilder,
                           yieldOp.getLoc());
            if (failed(nextInfo) ||
                !areLoopCompatible(info->initInfo, *nextInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component : getLoopComponentValues(*nextInfo))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, afterEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newWhileOp->setAttrs(whileOp->getAttrs());

  if (!bodyOk) {
    newWhileOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newWhileOp);
  for (auto [idx, oldResult] : llvm::enumerate(whileOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      CFPtrInfo resultInfo = withLoopComponentValues(
          info->initInfo,
          collectWhileComponents(*info, newWhileOp, /*useResults=*/true,
                                 /*useAfterArgs=*/false));
      Value rebuilt = rebuildPtr(builder, oldResult.getLoc(), resultInfo);
      if (!rebuilt) {
        newWhileOp.erase();
        return failure();
      }
      recordPointer(oldResult, resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newWhileOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

static LogicalResult
addIfTensorComponents(const TensorPtrInfo &thenParts,
                      const TensorPtrInfo &elseParts,
                      SmallVectorImpl<IfComponent> &components) {
  if (thenParts.base != elseParts.base)
    return failure();

  bool thenStructured = hasStructuredScalarOffset(thenParts);
  bool elseStructured = hasStructuredScalarOffset(elseParts);
  if (thenStructured != elseStructured)
    return failure();

  if (thenStructured) {
    if (!areEquivalentValues(thenParts.offset, elseParts.offset) ||
        thenParts.scalarOffset.getType() != elseParts.scalarOffset.getType())
      return failure();
    components.push_back(
        {IfComponentKind::TensorOffset, 0,
         thenParts.scalarOffset.getType()});
    return success();
  }

  if (thenParts.offset.getType() != elseParts.offset.getType())
    return failure();
  components.push_back(
      {IfComponentKind::TensorOffset, 0, thenParts.offset.getType()});
  return success();
}

static LogicalResult
addIfBlockComponents(const BlockPtrInfo &thenParts,
                     const BlockPtrInfo &elseParts,
                     SmallVectorImpl<IfComponent> &components) {
  if (thenParts.resultType != elseParts.resultType ||
      thenParts.order != elseParts.order ||
      thenParts.shape.size() != elseParts.shape.size() ||
      thenParts.strides.size() != elseParts.strides.size() ||
      thenParts.offsets.size() != elseParts.offsets.size())
    return failure();

  if (thenParts.base != elseParts.base)
    return failure();

  for (auto [idx, values] :
       llvm::enumerate(llvm::zip(thenParts.shape, elseParts.shape))) {
    Value thenValue = std::get<0>(values);
    Value elseValue = std::get<1>(values);
    if (thenValue == elseValue)
      continue;
    if (thenValue.getType() != elseValue.getType())
      return failure();
    components.push_back(
        {IfComponentKind::BlockShape, static_cast<unsigned>(idx),
         thenValue.getType()});
  }

  for (auto [idx, values] :
       llvm::enumerate(llvm::zip(thenParts.strides, elseParts.strides))) {
    Value thenValue = std::get<0>(values);
    Value elseValue = std::get<1>(values);
    if (thenValue == elseValue)
      continue;
    if (thenValue.getType() != elseValue.getType())
      return failure();
    components.push_back(
        {IfComponentKind::BlockStride, static_cast<unsigned>(idx),
         thenValue.getType()});
  }

  for (auto [idx, values] :
       llvm::enumerate(llvm::zip(thenParts.offsets, elseParts.offsets))) {
    Value thenValue = std::get<0>(values);
    Value elseValue = std::get<1>(values);
    if (thenValue == elseValue)
      continue;
    if (thenValue.getType() != elseValue.getType())
      return failure();
    components.push_back({IfComponentKind::BlockOffset,
                          static_cast<unsigned>(idx), thenValue.getType()});
  }
  return success();
}

static bool hasIfComponent(ArrayRef<IfComponent> components,
                           IfComponentKind kind, unsigned dim = 0) {
  return llvm::any_of(components, [&](const IfComponent &component) {
    return component.kind == kind && component.dim == dim;
  });
}

static void addIfComponentIfMissing(SmallVectorImpl<IfComponent> &components,
                                    IfComponent component) {
  if (!hasIfComponent(components, component.kind, component.dim))
    components.push_back(component);
}

static LogicalResult
addLoopCarriedIfComponents(const CFPtrInfo &info,
                           SmallVectorImpl<IfComponent> &components) {
  if (info.kind == PtrKind::Tensor) {
    Type offsetType = hasStructuredScalarOffset(info.tensor)
                          ? info.tensor.scalarOffset.getType()
                          : info.tensor.offset.getType();
    addIfComponentIfMissing(
        components, {IfComponentKind::TensorOffset, 0, offsetType});
    return success();
  }

  for (auto [idx, offset] : llvm::enumerate(info.block.offsets)) {
    addIfComponentIfMissing(
        components,
        {IfComponentKind::BlockOffset, static_cast<unsigned>(idx),
         offset.getType()});
  }
  return success();
}

// Nested loop results may look like their init pointer during planning, but
// the real result is produced by the loop-carried components after rewriting.
static bool isNestedForResult(Value value, const RewriteEnv &env) {
  Value mapped = remapValue(value, env);
  auto result = dyn_cast<OpResult>(mapped);
  return result && isa<scf::ForOp>(result.getOwner());
}

static FailureOr<CFPtrInfo>
analyzePtrForIfPlanning(Value value, const RewriteEnv &env,
                        OpBuilder &builder, Location loc);

static FailureOr<CFPtrInfo>
analyzeNestedIfResultForPlanning(scf::IfOp ifOp, unsigned resultIndex,
                                 const RewriteEnv &env, OpBuilder &builder,
                                 Location loc) {
  if (!ifOp.elseBlock() || resultIndex >= ifOp.getNumResults())
    return failure();

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  if (resultIndex >= thenYield.getNumOperands() ||
      resultIndex >= elseYield.getNumOperands())
    return failure();

  FailureOr<CFPtrInfo> thenInfo = analyzePtrForIfPlanning(
      thenYield.getOperand(resultIndex), env, builder, loc);
  FailureOr<CFPtrInfo> elseInfo = analyzePtrForIfPlanning(
      elseYield.getOperand(resultIndex), env, builder, loc);
  if (failed(thenInfo) || failed(elseInfo) ||
      thenInfo->kind != elseInfo->kind)
    return failure();

  SmallVector<IfComponent> components;
  if (thenInfo->kind == PtrKind::Tensor) {
    if (thenInfo->tensor.resultType != elseInfo->tensor.resultType ||
        thenInfo->tensor.scalarBase != elseInfo->tensor.scalarBase)
      return failure();
    if (failed(addIfTensorComponents(thenInfo->tensor, elseInfo->tensor,
                                     components)))
      return failure();
  } else if (failed(addIfBlockComponents(thenInfo->block, elseInfo->block,
                                         components))) {
    return failure();
  }

  return *thenInfo;
}

static FailureOr<CFPtrInfo>
analyzeNestedForResultForPlanning(scf::ForOp forOp, unsigned resultIndex,
                                  const RewriteEnv &env, OpBuilder &builder,
                                  Location loc) {
  if (resultIndex >= forOp.getNumResults() ||
      resultIndex >= forOp.getInitArgs().size() ||
      resultIndex >= forOp.getRegionIterArgs().size())
    return failure();

  scf::YieldOp yieldOp =
      dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp || resultIndex >= yieldOp.getNumOperands())
    return failure();

  FailureOr<CFPtrInfo> initInfo = analyzePtrForIfPlanning(
      forOp.getInitArgs()[resultIndex], env, builder, loc);
  if (failed(initInfo))
    return failure();

  RewriteEnv bodyEnv = env;
  bodyEnv.pointerComponents[forOp.getRegionIterArgs()[resultIndex]] =
      *initInfo;

  FailureOr<CFPtrInfo> nextInfo = analyzePtrForIfPlanning(
      yieldOp.getOperand(resultIndex), bodyEnv, builder, yieldOp.getLoc());
  if (failed(nextInfo) || !areLoopCompatible(*initInfo, *nextInfo))
    return failure();

  return *initInfo;
}

static FailureOr<CFPtrInfo>
analyzePtrForIfPlanning(Value value, const RewriteEnv &env,
                        OpBuilder &builder, Location loc) {
  if (auto it = env.pointerComponents.find(value);
      it != env.pointerComponents.end())
    return it->second;

  Value mapped = remapValue(value, env);
  if (auto result = dyn_cast<OpResult>(mapped)) {
    if (auto nestedIf = dyn_cast<scf::IfOp>(result.getOwner())) {
      FailureOr<CFPtrInfo> nestedInfo = analyzeNestedIfResultForPlanning(
          nestedIf, result.getResultNumber(), env, builder, loc);
      if (succeeded(nestedInfo))
        return nestedInfo;
    }
    if (auto nestedFor = dyn_cast<scf::ForOp>(result.getOwner())) {
      FailureOr<CFPtrInfo> nestedInfo = analyzeNestedForResultForPlanning(
          nestedFor, result.getResultNumber(), env, builder, loc);
      if (succeeded(nestedInfo))
        return nestedInfo;
    }
  }

  return analyzePtr(value, env, builder, loc);
}

static Value getComponentValue(const CFPtrInfo &info,
                               const IfComponent &component) {
  if (info.kind == PtrKind::Tensor) {
    if (component.kind == IfComponentKind::TensorOffset) {
      if (hasStructuredScalarOffset(info.tensor) &&
          info.tensor.scalarOffset.getType() == component.type)
        return info.tensor.scalarOffset;
      return info.tensor.offset;
    }
    return nullptr;
  }

  switch (component.kind) {
  case IfComponentKind::BlockShape:
    return info.block.shape[component.dim];
  case IfComponentKind::BlockStride:
    return info.block.strides[component.dim];
  case IfComponentKind::BlockOffset:
    return info.block.offsets[component.dim];
  default:
    return nullptr;
  }
}

static const IfPointerInfo *
findIfInfo(ArrayRef<IfPointerInfo> infos, unsigned oldIndex) {
  for (const IfPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static FailureOr<CFPtrInfo>
makeIfResultInfo(const IfPointerInfo &info, ArrayRef<Value> componentValues) {
  unsigned componentIndex = 0;
  CFPtrInfo resultInfo = info.thenInfo;
  if (info.thenInfo.kind == PtrKind::Tensor) {
    for (const IfComponent &component : info.components) {
      Value value = componentValues[componentIndex++];
      switch (component.kind) {
      case IfComponentKind::TensorOffset:
        if (hasStructuredScalarOffset(resultInfo.tensor) &&
            resultInfo.tensor.scalarOffset.getType() == value.getType())
          resultInfo.tensor.scalarOffset = value;
        else
          resultInfo.tensor.offset = value;
        break;
      default:
        return failure();
      }
    }
    return resultInfo;
  }

  for (const IfComponent &component : info.components) {
    Value value = componentValues[componentIndex++];
    switch (component.kind) {
    case IfComponentKind::BlockShape:
      resultInfo.block.shape[component.dim] = value;
      break;
    case IfComponentKind::BlockStride:
      resultInfo.block.strides[component.dim] = value;
      break;
    case IfComponentKind::BlockOffset:
      resultInfo.block.offsets[component.dim] = value;
      break;
    default:
      return failure();
    }
  }
  return resultInfo;
}

static LogicalResult rewriteIfOp(scf::IfOp ifOp, OpBuilder &builder,
                                 RewriteEnv &env) {
  if (!ifOp.elseBlock() || ifOp->getNumResults() == 0)
    return failure();

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  SmallVector<IfPointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(ifOp.getContext());
  analysisBuilder.setInsertionPoint(ifOp);

  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (!isControlFlowPointerType(result.getType()))
      continue;
    if (thenYield.getOperand(idx) == elseYield.getOperand(idx))
      continue;

    FailureOr<CFPtrInfo> thenInfo = analyzePtrForIfPlanning(
        thenYield.getOperand(idx), env, analysisBuilder, thenYield.getLoc());
    FailureOr<CFPtrInfo> elseInfo = analyzePtrForIfPlanning(
        elseYield.getOperand(idx), env, analysisBuilder, elseYield.getLoc());
    if (failed(thenInfo) || failed(elseInfo) ||
        (*thenInfo).kind != (*elseInfo).kind)
      continue;

    IfPointerInfo info;
    info.oldIndex = idx;
    info.thenInfo = *thenInfo;
    info.elseInfo = *elseInfo;
    if (info.thenInfo.kind == PtrKind::Tensor) {
      if (info.thenInfo.tensor.resultType != info.elseInfo.tensor.resultType ||
          info.thenInfo.tensor.scalarBase != info.elseInfo.tensor.scalarBase)
        continue;
      if (failed(addIfTensorComponents(info.thenInfo.tensor,
                                       info.elseInfo.tensor,
                                       info.components)))
        continue;
    } else {
      if (failed(addIfBlockComponents(info.thenInfo.block,
                                      info.elseInfo.block, info.components)))
        continue;
    }
    if (isNestedForResult(thenYield.getOperand(idx), env) ||
        isNestedForResult(elseYield.getOperand(idx), env)) {
      if (failed(addLoopCarriedIfComponents(info.thenInfo, info.components)))
        continue;
    }
    pointerInfos.push_back(info);
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Type> newResultTypes;
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      for (const IfComponent &component : info->components)
        newResultTypes.push_back(component.type);
      continue;
    }
    newResultTypes.push_back(result.getType());
  }

  bool bodyOk = true;
  auto buildBranch = [&](OpBuilder &branchBuilder, Location loc,
                         bool isThen) -> LogicalResult {
    RewriteEnv branchEnv = env;
    Block *oldBlock = isThen ? ifOp.thenBlock() : ifOp.elseBlock();
    scf::YieldOp oldYield = isThen ? thenYield : elseYield;
    if (failed(rewriteBodyOps(oldBlock, branchBuilder, branchEnv)))
      return failure();

    SmallVector<Value> newYieldOperands;
    for (auto [idx, oldOperand] : llvm::enumerate(oldYield.getOperands())) {
      if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
        FailureOr<CFPtrInfo> branchInfo =
            analyzePtr(oldOperand, branchEnv, branchBuilder,
                       oldYield.getLoc());
        if (failed(branchInfo) || branchInfo->kind != info->thenInfo.kind)
          return failure();
        for (const IfComponent &component : info->components) {
          Value value = getComponentValue(*branchInfo, component);
          if (!value || value.getType() != component.type)
            return failure();
          newYieldOperands.push_back(value);
        }
        continue;
      }
      newYieldOperands.push_back(remapValue(oldOperand, branchEnv));
    }
    branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
    return success();
  };

  auto newIfOp = builder.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, remapValue(ifOp.getCondition(), env),
      true);
  newIfOp->setAttrs(ifOp->getAttrs());

  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.thenBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.thenBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.thenBlock());
    }
    if (failed(buildBranch(builder, ifOp.getLoc(), /*isThen=*/true)))
      bodyOk = false;
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.elseBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.elseBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    if (failed(buildBranch(builder, ifOp.getLoc(), /*isThen=*/false)))
      bodyOk = false;
  }

  if (!bodyOk) {
    newIfOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newIfOp);
  unsigned newResultIndex = 0;
  for (auto [idx, oldResult] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      SmallVector<Value> componentValues;
      for (unsigned i = 0; i < info->components.size(); ++i)
        componentValues.push_back(newIfOp.getResult(newResultIndex++));
      FailureOr<CFPtrInfo> resultInfo =
          makeIfResultInfo(*info, componentValues);
      if (failed(resultInfo)) {
        newIfOp.erase();
        return failure();
      }
      Value rebuilt = rebuildPtr(builder, oldResult.getLoc(), *resultInfo);
      if (!rebuilt) {
        newIfOp.erase();
        return failure();
      }
      recordPointer(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newIfOp.getResult(newResultIndex++));
  }

  return success();
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp, builder, env);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp, builder, env);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp, builder, env);
  return failure();
}

static SmallVector<Value> collectReplacements(Operation *op,
                                              const RewriteEnv &env) {
  SmallVector<Value> replacements;
  replacements.reserve(op->getNumResults());
  for (Value result : op->getResults())
    replacements.push_back(remapValue(result, env));
  return replacements;
}

static LogicalResult tryDecoupleControlFlowOp(Operation *op,
                                              IRRewriter &rewriter) {
  RewriteEnv env;
  rewriter.setInsertionPoint(op);
  if (failed(rewriteControlFlowOp(op, rewriter, env)))
    return failure();

  SmallVector<Value> replacements = collectReplacements(op, env);
  if (replacements.size() != op->getNumResults() ||
      llvm::any_of(replacements, [](Value value) { return !value; }))
    return failure();
  rewriter.replaceOp(op, replacements);
  return success();
}

static bool decoupleControlFlowTree(Operation *op, IRRewriter &rewriter) {
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op) &&
      succeeded(tryDecoupleControlFlowOp(op, rewriter)))
    return true;

  bool changed = false;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : llvm::make_early_inc_range(block))
        changed |= decoupleControlFlowTree(&nestedOp, rewriter);
    }
  }
  return changed;
}

} // namespace

namespace mlir::triton {

void TritonControlFlowOptPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, scf::SCFDialect,
                  triton::TritonDialect>();
}

void TritonControlFlowOptPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SmallVector<Operation *> funcs;
  moduleOp.walk([&](Operation *op) {
    if (isa<triton::FuncOp, func::FuncOp>(op))
      funcs.push_back(op);
  });

  for (Operation *op : funcs) {
    if (op->getParentOp() == nullptr)
      continue;
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      if (!funcOp.isDeclaration() &&
          failed(structureFunctionBody(funcOp, funcOp.getBody()))) {
        signalPassFailure();
        return;
      }
      continue;
    }

    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (!funcOp.isDeclaration() &&
          failed(structureFunctionBody(funcOp, funcOp.getBody()))) {
        signalPassFailure();
        return;
      }
    }
  }

  IRRewriter rewriter(moduleOp.getContext());
  bool changed = decoupleControlFlowTree(moduleOp, rewriter);
  LLVM_DEBUG({
    llvm::dbgs() << "TritonControlFlowOpt decoupling changed module: "
                 << changed << "\n";
  });

  if (failed(verify(moduleOp)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createTritonControlFlowOptPass() {
  return std::make_unique<TritonControlFlowOptPass>();
}

} // namespace mlir::triton
