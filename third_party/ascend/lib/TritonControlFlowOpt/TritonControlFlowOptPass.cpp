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

struct TensorPtrParts {
  Type resultType;
  Value base;
  Value offset;
  bool scalarBase = false;
};

struct BlockPtrParts {
  Type resultType;
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  DenseI32ArrayAttr order;
};

struct PtrParts {
  PtrKind kind;
  TensorPtrParts tensor;
  BlockPtrParts block;
};

struct ForPointerInfo {
  unsigned oldIndex = 0;
  PtrParts initParts;
  SmallVector<Value> deltaValues;
  SmallVector<unsigned> newIndices;
};

struct WhilePointerInfo {
  unsigned oldIndex = 0;
  PtrParts initParts;
  SmallVector<Value> conditionDeltaValues;
  SmallVector<Value> yieldDeltaValues;
  SmallVector<unsigned> newIndices;
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
  PtrParts thenParts;
  PtrParts elseParts;
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

static unsigned getBlockPointerRank(Type type) {
  auto ptrType = cast<triton::PointerType>(type);
  auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
  return tensorType.getRank();
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

static FailureOr<TensorPtrParts>
analyzeTensorPtr(Value value, OpBuilder &builder, Location loc) {
  if (auto addPtrOp = value.getDefiningOp<triton::AddPtrOp>()) {
    FailureOr<TensorPtrParts> baseParts =
        analyzeTensorPtr(addPtrOp.getPtr(), builder, loc);
    if (failed(baseParts))
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(addPtrOp);
    Value offset = addPtrOp.getOffset();
    if ((*baseParts).offset.getType() != offset.getType()) {
      (*baseParts).offset =
          castIntegerLike(builder, addPtrOp.getLoc(), (*baseParts).offset,
                          offset.getType());
      if (!(*baseParts).offset)
        return failure();
    }
    Value newOffset =
        createAdd(builder, addPtrOp.getLoc(), (*baseParts).offset, offset);
    if (!newOffset)
      return failure();
    (*baseParts).offset = newOffset;
    (*baseParts).resultType = value.getType();
    return *baseParts;
  }

  if (auto splatOp = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<triton::PointerType>(splatOp.getSrc().getType()))
      return failure();
    TensorPtrParts parts;
    parts.resultType = value.getType();
    parts.base = splatOp.getSrc();
    parts.scalarBase = true;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(splatOp);
    parts.offset = createZeroOffset(builder, splatOp.getLoc(), value.getType());
    if (!parts.offset)
      return failure();
    return parts;
  }

  if (!isTensorPointerType(value.getType()))
    return failure();

  TensorPtrParts parts;
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

static FailureOr<BlockPtrParts>
analyzeBlockPtr(Value value, OpBuilder &builder, Location loc) {
  if (auto makePtrOp = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    BlockPtrParts parts;
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
    FailureOr<BlockPtrParts> baseParts =
        analyzeBlockPtr(advanceOp.getPtr(), builder, loc);
    if (failed(baseParts))
      return failure();
    if ((*baseParts).offsets.size() != advanceOp.getOffsets().size())
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(advanceOp);
    for (auto [idx, delta] : llvm::enumerate(advanceOp.getOffsets())) {
      Value newOffset =
          createAdd(builder, advanceOp.getLoc(), (*baseParts).offsets[idx],
                    delta);
      if (!newOffset)
        return failure();
      (*baseParts).offsets[idx] = newOffset;
    }
    (*baseParts).resultType = value.getType();
    return *baseParts;
  }

  return failure();
}

static FailureOr<PtrParts> analyzePtr(Value value, OpBuilder &builder,
                                      Location loc) {
  if (isBlockPointerType(value.getType())) {
    FailureOr<BlockPtrParts> block = analyzeBlockPtr(value, builder, loc);
    if (failed(block))
      return failure();
    PtrParts parts{PtrKind::Block};
    parts.block = *block;
    return parts;
  }

  if (isTensorPointerType(value.getType())) {
    FailureOr<TensorPtrParts> tensor = analyzeTensorPtr(value, builder, loc);
    if (failed(tensor))
      return failure();
    PtrParts parts{PtrKind::Tensor};
    parts.tensor = *tensor;
    return parts;
  }

  return failure();
}

static Value rebuildTensorPtr(OpBuilder &builder, Location loc,
                              const TensorPtrParts &parts, Value base,
                              Value offset) {
  Value ptrBase = base;
  if (parts.scalarBase && isTensorPointerType(parts.resultType))
    ptrBase = builder.create<triton::SplatOp>(loc, parts.resultType, base);
  return builder.create<triton::AddPtrOp>(loc, parts.resultType, ptrBase,
                                          offset);
}

static Value rebuildBlockPtr(OpBuilder &builder, Location loc,
                             const BlockPtrParts &parts, Value base,
                             ArrayRef<Value> shape, ArrayRef<Value> strides,
                             ArrayRef<Value> offsets) {
  return builder.create<triton::MakeTensorPtrOp>(
      loc, parts.resultType, base, ValueRange(shape), ValueRange(strides),
      ValueRange(offsets), parts.order);
}

static Value rebuildPtr(OpBuilder &builder, Location loc, const PtrParts &parts,
                        ArrayRef<Value> values) {
  if (parts.kind == PtrKind::Tensor) {
    if (values.size() == 1)
      return rebuildTensorPtr(builder, loc, parts.tensor, parts.tensor.base,
                              values[0]);
    if (values.size() == 2)
      return rebuildTensorPtr(builder, loc, parts.tensor, values[0], values[1]);
    return nullptr;
  }

  unsigned rank = parts.block.offsets.size();
  if (values.size() == rank)
    return rebuildBlockPtr(builder, loc, parts.block, parts.block.base,
                           parts.block.shape, parts.block.strides, values);

  return nullptr;
}

static SmallVector<Value>
collectForRebuildValues(const ForPointerInfo &info, scf::ForOp forOp,
                        bool useResults) {
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices) {
    values.push_back(useResults ? forOp.getResult(newIndex)
                                : forOp.getRegionIterArgs()[newIndex]);
  }
  return values;
}

static SmallVector<Value>
collectWhileRebuildValues(const WhilePointerInfo &info, scf::WhileOp whileOp,
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

static FailureOr<SmallVector<Value>>
getBlockPtrDelta(Value value, Value iterArg, OpBuilder &builder, Location loc) {
  unsigned rank = getBlockPointerRank(iterArg.getType());
  if (value == iterArg) {
    SmallVector<Value> zeros;
    zeros.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      zeros.push_back(createZeroLike(builder, loc, builder.getI32Type()));
    return zeros;
  }

  auto advanceOp = value.getDefiningOp<triton::AdvanceOp>();
  if (!advanceOp || advanceOp.getOffsets().size() != rank)
    return failure();

  FailureOr<SmallVector<Value>> baseDelta =
      getBlockPtrDelta(advanceOp.getPtr(), iterArg, builder, loc);
  if (failed(baseDelta))
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(advanceOp);
  for (auto [idx, step] : llvm::enumerate(advanceOp.getOffsets())) {
    Value newDelta =
        createAdd(builder, advanceOp.getLoc(), (*baseDelta)[idx], step);
    if (!newDelta)
      return failure();
    (*baseDelta)[idx] = newDelta;
  }
  return *baseDelta;
}

static FailureOr<SmallVector<Value>>
getTensorPtrDelta(Value value, Value iterArg, OpBuilder &builder,
                  Location loc) {
  if (value == iterArg) {
    Value zero = createZeroOffset(builder, loc, iterArg.getType());
    if (!zero)
      return failure();
    return SmallVector<Value>{zero};
  }

  auto addPtrOp = value.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return failure();

  FailureOr<SmallVector<Value>> baseDelta =
      getTensorPtrDelta(addPtrOp.getPtr(), iterArg, builder, loc);
  if (failed(baseDelta) || baseDelta->size() != 1)
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(addPtrOp);
  Value delta = (*baseDelta)[0];
  Value step = addPtrOp.getOffset();
  if (delta.getType() != step.getType()) {
    delta = castIntegerLike(builder, addPtrOp.getLoc(), delta, step.getType());
    if (!delta)
      return failure();
  }
  Value newDelta = createAdd(builder, addPtrOp.getLoc(), delta, step);
  if (!newDelta)
    return failure();
  return SmallVector<Value>{newDelta};
}

static FailureOr<SmallVector<Value>>
getPointerDelta(Value value, Value iterArg, OpBuilder &builder, Location loc) {
  if (isBlockPointerType(iterArg.getType()))
    return getBlockPtrDelta(value, iterArg, builder, loc);
  if (isTensorPointerType(iterArg.getType()))
    return getTensorPtrDelta(value, iterArg, builder, loc);
  return failure();
}

static ForPointerInfo *findForInfo(SmallVectorImpl<ForPointerInfo> &infos,
                                   unsigned oldIndex) {
  for (ForPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const ForPointerInfo *
findForInfo(ArrayRef<ForPointerInfo> infos, unsigned oldIndex) {
  for (const ForPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static WhilePointerInfo *
findWhileInfo(SmallVectorImpl<WhilePointerInfo> &infos, unsigned oldIndex) {
  for (WhilePointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const WhilePointerInfo *
findWhileInfo(ArrayRef<WhilePointerInfo> infos, unsigned oldIndex) {
  for (const WhilePointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static LogicalResult tryDecoupleFor(scf::ForOp forOp, IRRewriter &rewriter) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<ForPointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(forOp.getContext());
  analysisBuilder.setInsertionPoint(forOp);

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!isControlFlowPointerType(iterArg.getType()))
      continue;
    if (idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<PtrParts> initParts =
        analyzePtr(forOp.getInitArgs()[idx], analysisBuilder, forOp.getLoc());
    if (failed(initParts))
      continue;

    FailureOr<SmallVector<Value>> delta = getPointerDelta(
        yieldOp.getOperand(idx), iterArg, analysisBuilder, yieldOp.getLoc());
    if (failed(delta))
      continue;

    pointerInfos.push_back(
        ForPointerInfo{static_cast<unsigned>(idx), *initParts, *delta, {}});
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Value> newInitArgs;
  SmallVector<unsigned> oldToNewStart(forOp.getInitArgs().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    oldToNewStart[idx] = newInitArgs.size();
    if (ForPointerInfo *info = findForInfo(pointerInfos, idx)) {
      if (info->initParts.kind == PtrKind::Tensor) {
        info->newIndices.push_back(newInitArgs.size());
        newInitArgs.push_back(info->initParts.tensor.offset);
      } else {
        for (Value offset : info->initParts.block.offsets) {
          info->newIndices.push_back(newInitArgs.size());
          newInitArgs.push_back(offset);
        }
      }
      continue;
    }
    newInitArgs.push_back(initArg);
  }

  rewriter.setInsertionPoint(forOp);
  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), iv);

        for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (const ForPointerInfo *info = findForInfo(pointerInfos, idx)) {
            SmallVector<Value> rebuildValues;
            for (unsigned newIndex : info->newIndices)
              rebuildValues.push_back(args[newIndex]);
            Value rebuilt = rebuildPtr(builder, loc, info->initParts,
                                       rebuildValues);
            mapping.map(oldArg, rebuilt);
            continue;
          }
          mapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        for (Operation &bodyOp : forOp.getBody()->without_terminator())
          builder.clone(bodyOp, mapping);

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const ForPointerInfo *info = findForInfo(pointerInfos, idx)) {
            for (auto [componentIdx, delta] :
                 llvm::enumerate(info->deltaValues)) {
              Value current = args[info->newIndices[componentIdx]];
              Value mappedDelta = mapping.lookupOrDefault(delta);
              newYieldOperands.push_back(
                  createAdd(builder, yieldOp.getLoc(), current, mappedDelta));
            }
            continue;
          }
          newYieldOperands.push_back(mapping.lookupOrDefault(oldOperand));
        }

        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newForOp->setAttrs(forOp->getAttrs());

  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> replacements;
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (const ForPointerInfo *info = findForInfo(pointerInfos, idx)) {
      SmallVector<Value> rebuildValues =
          collectForRebuildValues(*info, newForOp, /*useResults=*/true);
      replacements.push_back(
          rebuildPtr(rewriter, oldResult.getLoc(), info->initParts,
                     rebuildValues));
      continue;
    }
    replacements.push_back(newForOp.getResult(oldToNewStart[idx]));
  }

  if (llvm::any_of(replacements, [](Value value) { return !value; }))
    return failure();

  rewriter.replaceOp(forOp, replacements);
  return success();
}

static LogicalResult tryDecoupleWhile(scf::WhileOp whileOp,
                                      IRRewriter &rewriter) {
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<WhilePointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(whileOp.getContext());
  analysisBuilder.setInsertionPoint(whileOp);

  for (auto [idx, beforeArg] : llvm::enumerate(whileOp.getBeforeArguments())) {
    if (!isControlFlowPointerType(beforeArg.getType()))
      continue;
    if (idx >= conditionOp.getArgs().size() || idx >= yieldOp.getNumOperands() ||
        idx >= whileOp.getInits().size())
      return failure();

    FailureOr<PtrParts> initParts =
        analyzePtr(whileOp.getInits()[idx], analysisBuilder, whileOp.getLoc());
    if (failed(initParts))
      continue;

    FailureOr<SmallVector<Value>> conditionDelta =
        getPointerDelta(conditionOp.getArgs()[idx], beforeArg, analysisBuilder,
                        conditionOp.getLoc());
    if (failed(conditionDelta))
      continue;

    Value afterArg = whileOp.getAfterArguments()[idx];
    FailureOr<SmallVector<Value>> yieldDelta =
        getPointerDelta(yieldOp.getOperand(idx), afterArg, analysisBuilder,
                        yieldOp.getLoc());
    if (failed(yieldDelta))
      continue;

    pointerInfos.push_back(WhilePointerInfo{static_cast<unsigned>(idx),
                                            *initParts, *conditionDelta,
                                            *yieldDelta, {}});
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Value> newInits;
  SmallVector<Type> newResultTypes;
  SmallVector<unsigned> oldToNewStart(whileOp.getInits().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(whileOp.getInits())) {
    oldToNewStart[idx] = newInits.size();
    if (WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
      if (info->initParts.kind == PtrKind::Tensor) {
        info->newIndices.push_back(newInits.size());
        newInits.push_back(info->initParts.tensor.offset);
        newResultTypes.push_back(info->initParts.tensor.offset.getType());
      } else {
        for (Value offset : info->initParts.block.offsets) {
          info->newIndices.push_back(newInits.size());
          newInits.push_back(offset);
          newResultTypes.push_back(offset.getType());
        }
      }
      continue;
    }
    newInits.push_back(initArg);
    newResultTypes.push_back(whileOp.getResult(idx).getType());
  }

  rewriter.setInsertionPoint(whileOp);
  auto newWhileOp = rewriter.create<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        IRMapping mapping;
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getBeforeArguments())) {
          if (const WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
            SmallVector<Value> rebuildValues;
            for (unsigned newIndex : info->newIndices)
              rebuildValues.push_back(args[newIndex]);
            Value rebuilt = rebuildPtr(builder, loc, info->initParts,
                                       rebuildValues);
            mapping.map(oldArg, rebuilt);
            continue;
          }
          mapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        for (Operation &bodyOp : whileOp.getBeforeBody()->without_terminator())
          builder.clone(bodyOp, mapping);

        SmallVector<Value> newConditionArgs;
        for (auto [idx, oldArg] : llvm::enumerate(conditionOp.getArgs())) {
          if (const WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
            for (auto [componentIdx, delta] :
                 llvm::enumerate(info->conditionDeltaValues)) {
              Value current = args[info->newIndices[componentIdx]];
              Value mappedDelta = mapping.lookupOrDefault(delta);
              newConditionArgs.push_back(createAdd(
                  builder, conditionOp.getLoc(), current, mappedDelta));
            }
            continue;
          }
          newConditionArgs.push_back(mapping.lookupOrDefault(oldArg));
        }

        builder.create<scf::ConditionOp>(
            conditionOp.getLoc(),
            mapping.lookupOrDefault(conditionOp.getCondition()),
            newConditionArgs);
      },
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        IRMapping mapping;
        for (auto [idx, oldArg] : llvm::enumerate(whileOp.getAfterArguments())) {
          if (const WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
            SmallVector<Value> rebuildValues;
            for (unsigned newIndex : info->newIndices)
              rebuildValues.push_back(args[newIndex]);
            Value rebuilt = rebuildPtr(builder, loc, info->initParts,
                                       rebuildValues);
            mapping.map(oldArg, rebuilt);
            continue;
          }
          mapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        for (Operation &bodyOp : whileOp.getAfterBody()->without_terminator())
          builder.clone(bodyOp, mapping);

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
            for (auto [componentIdx, delta] :
                 llvm::enumerate(info->yieldDeltaValues)) {
              Value current = args[info->newIndices[componentIdx]];
              Value mappedDelta = mapping.lookupOrDefault(delta);
              newYieldOperands.push_back(
                  createAdd(builder, yieldOp.getLoc(), current, mappedDelta));
            }
            continue;
          }
          newYieldOperands.push_back(mapping.lookupOrDefault(oldOperand));
        }

        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newWhileOp->setAttrs(whileOp->getAttrs());

  rewriter.setInsertionPointAfter(newWhileOp);
  SmallVector<Value> replacements;
  for (auto [idx, oldResult] : llvm::enumerate(whileOp.getResults())) {
    if (const WhilePointerInfo *info = findWhileInfo(pointerInfos, idx)) {
      SmallVector<Value> rebuildValues = collectWhileRebuildValues(
          *info, newWhileOp, /*useResults=*/true, /*useAfterArgs=*/false);
      replacements.push_back(
          rebuildPtr(rewriter, oldResult.getLoc(), info->initParts,
                     rebuildValues));
      continue;
    }
    replacements.push_back(newWhileOp.getResult(oldToNewStart[idx]));
  }

  if (llvm::any_of(replacements, [](Value value) { return !value; }))
    return failure();

  rewriter.replaceOp(whileOp, replacements);
  return success();
}

static LogicalResult
addIfTensorComponents(const TensorPtrParts &thenParts,
                      const TensorPtrParts &elseParts,
                      SmallVectorImpl<IfComponent> &components) {
  if (thenParts.base != elseParts.base)
    return failure();
  if (thenParts.offset.getType() != elseParts.offset.getType())
    return failure();
  components.push_back(
      {IfComponentKind::TensorOffset, 0, thenParts.offset.getType()});
  return success();
}

static LogicalResult
addIfBlockComponents(const BlockPtrParts &thenParts,
                     const BlockPtrParts &elseParts,
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

static Value getThenComponentValue(const IfPointerInfo &info,
                                   const IfComponent &component) {
  if (info.thenParts.kind == PtrKind::Tensor) {
    if (component.kind == IfComponentKind::TensorOffset)
      return info.thenParts.tensor.offset;
    return nullptr;
  }

  switch (component.kind) {
  case IfComponentKind::BlockShape:
    return info.thenParts.block.shape[component.dim];
  case IfComponentKind::BlockStride:
    return info.thenParts.block.strides[component.dim];
  case IfComponentKind::BlockOffset:
    return info.thenParts.block.offsets[component.dim];
  default:
    return nullptr;
  }
}

static Value getElseComponentValue(const IfPointerInfo &info,
                                   const IfComponent &component) {
  if (info.elseParts.kind == PtrKind::Tensor) {
    if (component.kind == IfComponentKind::TensorOffset)
      return info.elseParts.tensor.offset;
    return nullptr;
  }

  switch (component.kind) {
  case IfComponentKind::BlockShape:
    return info.elseParts.block.shape[component.dim];
  case IfComponentKind::BlockStride:
    return info.elseParts.block.strides[component.dim];
  case IfComponentKind::BlockOffset:
    return info.elseParts.block.offsets[component.dim];
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

static Value rebuildIfPointer(OpBuilder &builder, Location loc,
                              const IfPointerInfo &info,
                              ArrayRef<Value> componentValues) {
  unsigned componentIndex = 0;
  if (info.thenParts.kind == PtrKind::Tensor) {
    Value base = info.thenParts.tensor.base;
    Value offset = nullptr;
    for (const IfComponent &component : info.components) {
      Value value = componentValues[componentIndex++];
      if (component.kind == IfComponentKind::TensorOffset)
        offset = value;
    }
    return rebuildTensorPtr(builder, loc, info.thenParts.tensor, base, offset);
  }

  Value base = info.thenParts.block.base;
  SmallVector<Value> shape = info.thenParts.block.shape;
  SmallVector<Value> strides = info.thenParts.block.strides;
  SmallVector<Value> offsets = info.thenParts.block.offsets;
  for (const IfComponent &component : info.components) {
    Value value = componentValues[componentIndex++];
    switch (component.kind) {
    case IfComponentKind::BlockShape:
      shape[component.dim] = value;
      break;
    case IfComponentKind::BlockStride:
      strides[component.dim] = value;
      break;
    case IfComponentKind::BlockOffset:
      offsets[component.dim] = value;
      break;
    default:
      return nullptr;
    }
  }
  return rebuildBlockPtr(builder, loc, info.thenParts.block, base, shape,
                         strides, offsets);
}

static LogicalResult tryDecoupleIf(scf::IfOp ifOp, IRRewriter &rewriter) {
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

    FailureOr<PtrParts> thenParts = analyzePtr(
        thenYield.getOperand(idx), analysisBuilder, thenYield.getLoc());
    FailureOr<PtrParts> elseParts = analyzePtr(
        elseYield.getOperand(idx), analysisBuilder, elseYield.getLoc());
    if (failed(thenParts) || failed(elseParts) ||
        (*thenParts).kind != (*elseParts).kind)
      continue;

    IfPointerInfo info;
    info.oldIndex = idx;
    info.thenParts = *thenParts;
    info.elseParts = *elseParts;
    if (info.thenParts.kind == PtrKind::Tensor) {
      if (info.thenParts.tensor.resultType != info.elseParts.tensor.resultType ||
          info.thenParts.tensor.scalarBase != info.elseParts.tensor.scalarBase)
        continue;
      if (failed(addIfTensorComponents(info.thenParts.tensor,
                                       info.elseParts.tensor,
                                       info.components)))
        continue;
    } else {
      if (failed(addIfBlockComponents(info.thenParts.block,
                                      info.elseParts.block, info.components)))
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

  auto buildBranch = [&](OpBuilder &builder, Location loc, bool isThen) {
    IRMapping mapping;
    Block *oldBlock = isThen ? ifOp.thenBlock() : ifOp.elseBlock();
    scf::YieldOp oldYield = isThen ? thenYield : elseYield;
    for (Operation &bodyOp : oldBlock->without_terminator())
      builder.clone(bodyOp, mapping);

    SmallVector<Value> newYieldOperands;
    for (auto [idx, oldOperand] : llvm::enumerate(oldYield.getOperands())) {
      if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
        for (const IfComponent &component : info->components) {
          Value value = isThen ? getThenComponentValue(*info, component)
                               : getElseComponentValue(*info, component);
          newYieldOperands.push_back(mapping.lookupOrDefault(value));
        }
        continue;
      }
      newYieldOperands.push_back(mapping.lookupOrDefault(oldOperand));
    }
    builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
  };

  rewriter.setInsertionPoint(ifOp);
  auto newIfOp = rewriter.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, ifOp.getCondition(), true);
  newIfOp->setAttrs(ifOp->getAttrs());

  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (newResultTypes.empty()) {
      newIfOp.thenBlock()->getTerminator()->erase();
      rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
    } else {
      rewriter.setInsertionPointToStart(newIfOp.thenBlock());
    }
    buildBranch(rewriter, ifOp.getLoc(), /*isThen=*/true);
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (newResultTypes.empty()) {
      newIfOp.elseBlock()->getTerminator()->erase();
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
    } else {
      rewriter.setInsertionPointToStart(newIfOp.elseBlock());
    }
    buildBranch(rewriter, ifOp.getLoc(), /*isThen=*/false);
  }

  rewriter.setInsertionPointAfter(newIfOp);
  SmallVector<Value> replacements;
  unsigned newResultIndex = 0;
  for (auto [idx, oldResult] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      SmallVector<Value> componentValues;
      for (unsigned i = 0; i < info->components.size(); ++i)
        componentValues.push_back(newIfOp.getResult(newResultIndex++));
      replacements.push_back(
          rebuildIfPointer(rewriter, oldResult.getLoc(), *info,
                           componentValues));
      continue;
    }
    replacements.push_back(newIfOp.getResult(newResultIndex++));
  }

  if (llvm::any_of(replacements, [](Value value) { return !value; }))
    return failure();

  rewriter.replaceOp(ifOp, replacements);
  return success();
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

  SmallVector<Operation *> targets;
  moduleOp.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))
      targets.push_back(op);
  });

  LLVM_DEBUG({
    llvm::dbgs() << "TritonControlFlowOpt collected " << targets.size()
                 << " structured control-flow targets for later decoupling\n";
  });

  IRRewriter rewriter(moduleOp.getContext());
  for (Operation *op : targets) {
    if (op->getParentOp() == nullptr)
      continue;

    LogicalResult result = failure();
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      result = tryDecoupleFor(forOp, rewriter);
    else if (auto whileOp = dyn_cast<scf::WhileOp>(op))
      result = tryDecoupleWhile(whileOp, rewriter);
    else if (auto ifOp = dyn_cast<scf::IfOp>(op))
      result = tryDecoupleIf(ifOp, rewriter);

    (void)result;
  }

  if (failed(verify(moduleOp)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createTritonControlFlowOptPass() {
  return std::make_unique<TritonControlFlowOptPass>();
}

} // namespace mlir::triton
