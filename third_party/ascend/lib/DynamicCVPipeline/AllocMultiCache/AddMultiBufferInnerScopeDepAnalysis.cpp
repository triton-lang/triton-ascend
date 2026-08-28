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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScopeDepAnalysis.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <climits>

static constexpr const char *DEBUG_TYPE = "AddMultiBufferInnerScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace hivm;
using namespace annotation;
using namespace triton;
using namespace CVPipeline;

using BufferPair = std::pair<Value, Value>;
using BufferMap = DenseMap<Value, SmallVector<BufferPair>>;

namespace mlir {
namespace triton {

// True if op is nested strictly inside the main loop.
static bool isOpInMainLoop(Operation *op, const MainLoop &mainLoop) {
  return op && mainLoop.getOperation()->isProperAncestor(op);
}

std::optional<int64_t> getOutermostSsbufferId(Operation *op) {
  std::optional<int64_t> result;
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->hasAttr(CVPipeline::kMainLoop))
      return result.has_value() ? result : -1;

    if (current->getNumRegions() >= 2)
      return CVPipeline::getOpBlockId(current);

    // Otherwise remember the deepest id seen; the parent walk will
    // overwrite it if a closer-to-boundary op carries one.
    if (auto curId = CVPipeline::getOpBlockId(current); curId.has_value())
      result = curId;
  }
  return result;
}

void collectNestedOps(Block *block, SmallVector<Operation *> &ops) {
  for (auto &op : *block) {
    ops.push_back(&op);
    for (auto &region : op.getRegions()) {
      for (auto &innerBlock : region) {
        collectNestedOps(&innerBlock, ops);
      }
    }
  }
}

// Collect the values an op depends on: its direct operands plus the values its
// nested regions capture from above.
static void collectOpDependencies(Operation *op, SmallVector<Value> &deps) {
  for (Value v : op->getOperands()) {
    deps.push_back(v);
  }
  if (op->getNumRegions() > 0) {
    llvm::SetVector<Value> above;
    mlir::getUsedValuesDefinedAbove(op->getRegions(), above);
    for (Value v : above) {
      deps.push_back(v);
    }
  }
}

// yieldFn is used for normal op case, getNestedResults is used for nested if
// case
static void createConditionalYield(
    OpBuilder &builder, Location loc, bool hasResults,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    Operation *op,
    std::function<SmallVector<Value>()> getNestedResults = nullptr) {
  if (hasResults && getNestedResults) {
    builder.create<mlir::scf::YieldOp>(loc, getNestedResults());
  } else if (hasResults && yieldFn && op) {
    builder.create<mlir::scf::YieldOp>(loc, yieldFn(builder, loc, op));
  } else {
    builder.create<mlir::scf::YieldOp>(loc);
  }
}

// Build if-else chain for N==2 (simple nested structure)
static int buildIfChainTwoBuffers(
    OpBuilder &builder, Location loc, Value indexVal,
    SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
    SmallVector<Operation *> &outIfOps,
    function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    mlir::TypeRange types, bool hasResults, int blockId) {
  // Create condition: index == 0
  Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  Value firstCond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, indexVal, zero);
  auto firstIf =
      builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);

  newOps.push_back(zero.getDefiningOp());
  newOps.push_back(firstCond.getDefiningOp());
  newOps.push_back(firstIf);
  outIfOps.push_back(firstIf);

  // Tag counter operations with block_id
  if (blockId >= 0) {
    zero.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    firstCond.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
  }

  // Then branch: use buffer[0]
  builder.setInsertionPointToStart(&firstIf.getThenRegion().front());
  Operation *op0 = createOpFn(builder, loc, buffers[0].second);
  if (!op0)
    return -1;
  newOps.push_back(op0);
  createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

  // Else branch: use buffer[1]
  builder.setInsertionPointToStart(&firstIf.getElseRegion().front());
  Operation *op1 = createOpFn(builder, loc, buffers[1].second);
  if (!op1)
    return -1;
  newOps.push_back(op1);
  createConditionalYield(builder, loc, hasResults, yieldFn, op1, nullptr);

  builder.setInsertionPointAfter(firstIf);
  return 0;
}

// Build if-else chain for N>2 (if-else-if-else chain with proper yield
// passthrough)
static int buildIfChainMultiBuffers(
    OpBuilder &builder, Location loc, Value indexVal,
    SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
    SmallVector<Operation *> &outIfOps,
    function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
    function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
    mlir::TypeRange types, bool hasResults, int blockId) {
  int N = buffers.size();

  // Create rootIf (idx == 0)
  Value zeroVal = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  Value firstCond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, indexVal, zeroVal);
  if (blockId >= 0) {
    zeroVal.getDefiningOp()->setAttr(kBlockId,
                                     builder.getI32IntegerAttr(blockId));
    firstCond.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
  }
  newOps.push_back(zeroVal.getDefiningOp());
  newOps.push_back(firstCond.getDefiningOp());

  auto rootIf =
      builder.create<mlir::scf::IfOp>(loc, types, firstCond, true, true);
  if (!rootIf)
    return -1;
  newOps.push_back(rootIf);
  outIfOps.push_back(rootIf);

  // Then branch of rootIf: use buffer[0]
  builder.setInsertionPointToStart(&rootIf.getThenRegion().front());
  Operation *op0 = createOpFn(builder, loc, buffers[0].second);
  if (!op0)
    return -1;
  newOps.push_back(op0);
  createConditionalYield(builder, loc, hasResults, yieldFn, op0, nullptr);

  // Build the nested if chain in rootIf's else region
  Block *currentElseBlock = &rootIf.getElseRegion().front();

  for (int i = 1; i < N - 1; ++i) {
    // Set insertion to current else block
    builder.setInsertionPointToStart(currentElseBlock);

    // Create condition for idx == i
    Value iVal = builder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
    Value cond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, indexVal, iVal);
    if (blockId >= 0) {
      iVal.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
      cond.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
    }
    newOps.push_back(iVal.getDefiningOp());
    newOps.push_back(cond.getDefiningOp());

    // Create nested if for this level
    auto nestedIf =
        builder.create<mlir::scf::IfOp>(loc, types, cond, true, true);
    if (!nestedIf)
      return -1;
    newOps.push_back(nestedIf);

    // Then branch: use buffer[i]
    builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());
    Operation *op = createOpFn(builder, loc, buffers[i].second);
    if (!op)
      return -1;
    newOps.push_back(op);
    createConditionalYield(builder, loc, hasResults, yieldFn, op, nullptr);

    // Set insertion to end of currentElseBlock and add yield
    builder.setInsertionPointToEnd(currentElseBlock);
    createConditionalYield(builder, loc, hasResults, yieldFn, nullptr,
                           [&nestedIf]() { return nestedIf.getResults(); });

    // Move to nestedIf's else block for next iteration
    currentElseBlock = &nestedIf.getElseRegion().front();
  }

  // Final else: use buffer[N-1]
  builder.setInsertionPointToStart(currentElseBlock);
  Operation *opLast = createOpFn(builder, loc, buffers[N - 1].second);
  if (!opLast)
    return -1;
  newOps.push_back(opLast);
  createConditionalYield(builder, loc, hasResults, yieldFn, opLast, nullptr);

  builder.setInsertionPointAfter(rootIf);
  return 0;
}

// Build if-else chain for buffer selection: if (idx==0) -> buf[0] else ... else
// -> buf[N-1]
static int
buildIfChain(OpBuilder &builder, Location loc, Value indexVal,
             SmallVector<BufferPair> &buffers, SmallVector<Operation *> &newOps,
             SmallVector<Operation *> &outIfOps,
             function_ref<Operation *(OpBuilder &, Location, Value)> createOpFn,
             function_ref<Value(OpBuilder &, Location, Operation *)> yieldFn,
             std::optional<mlir::TypeRange> resultTypes = std::nullopt,
             int blockId = -1) {
  int N = buffers.size();
  auto types = resultTypes.value_or(mlir::TypeRange{});
  bool hasResults = !types.empty();

  if (N == 2) {
    return buildIfChainTwoBuffers(builder, loc, indexVal, buffers, newOps,
                                  outIfOps, createOpFn, yieldFn, types,
                                  hasResults, blockId);
  }
  return buildIfChainMultiBuffers(builder, loc, indexVal, buffers, newOps,
                                  outIfOps, createOpFn, yieldFn, types,
                                  hasResults, blockId);
}

// Depth-first build of the scalar op slice feeding `root`. Recursion stops at
// tensor operands
static void buildScalarSlice(Value root, const MainLoop &mainLoop,
                             SmallVector<Operation *> &sliceInOrder,
                             DenseSet<Operation *> &visited,
                             llvm::SetVector<Value> &boundaryTensors) {
  Operation *def = root.getDefiningOp();
  if (!def || !isOpInMainLoop(def, mainLoop)) {
    return;
  }
  if (!visited.insert(def).second) {
    return;
  }

  SmallVector<Value> deps;
  collectOpDependencies(def, deps);
  for (Value dep : deps) {
    if (isa<TensorType>(dep.getType())) {
      // Tensor boundary: let it travel through the normal tensor path.
      boundaryTensors.insert(dep);
      continue;
    }
    Operation *depDef = dep.getDefiningOp();
    if (!depDef || !isOpInMainLoop(depDef, mainLoop)) {
      continue; // block arg or loop-invariant value: reference it directly
    }
    buildScalarSlice(dep, mainLoop, sliceInOrder, visited, boundaryTensors);
  }
  sliceInOrder.push_back(def);
}

// Find the ancestor of `op` that is a direct child of `block`.
static Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block) {
    op = op->getParentOp();
  }
  return op;
}

// Rematerialize the scalar slice of `root` into each of its cross-block
// consumer blocks and rewire those consumers to the local copy. Returns true on
// rewrite.
static bool
rematerializeScalarDep(Value root, int producerId, const MainLoop &mainLoop,
                       const SmallVector<Operation *> &sliceInOrder) {
  Block *body = mainLoop.getBody();

  // Group cross-block users by their block id.
  llvm::MapVector<int, SmallVector<Operation *>> usersByBlock;
  for (Operation *user : root.getUsers()) {
    Operation *bodyAnc = getAncestorInBlock(user, body);
    if (!bodyAnc) {
      continue;
    }
    auto userId = getOpBlockId(user);
    if (!userId.has_value()) {
      userId = getOpBlockId(bodyAnc);
    }
    if (!userId.has_value() || *userId == producerId) {
      continue;
    }
    usersByBlock[*userId].push_back(user);
  }
  if (usersByBlock.empty()) {
    return false;
  }

  bool changed = false;
  for (auto &entry : usersByBlock) {
    int userBlockId = entry.first;
    SmallVector<Operation *> &users = entry.second;

    // Insert the rematerialized slice before the earliest consumer.
    Operation *insertPt = nullptr;
    for (Operation *user : users) {
      Operation *anc = getAncestorInBlock(user, body);
      if (!anc) {
        continue;
      }
      if (!insertPt || anc->isBeforeInBlock(insertPt)) {
        insertPt = anc;
      }
    }
    if (!insertPt) {
      continue;
    }

    OpBuilder builder(insertPt);
    IRMapping map;
    for (Operation *op : sliceInOrder) {
      Operation *cloned = builder.clone(*op, map);
      cloned->walk([&](Operation *o) {
        o->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
      });
    }

    Value clonedRoot = map.lookupOrDefault(root);
    if (clonedRoot == root) {
      continue;
    }
    for (Operation *user : users) {
      user->replaceUsesOfWith(root, clonedRoot);
    }
    changed = true;
  }
  return changed;
}

// Scan the main loop for cross-block scalar dependencies whose data originates
// from a tensor, and rematerialize the scalar portion into each consumer block
// so the tensor part can use the normal tensor-dependency buffering.
static void rematerializeTensorRootedScalarDeps(const MainLoop &mainLoop) {
  Block *body = mainLoop.getBody();
  if (!body) {
    return;
  }

  SmallVector<Operation *> allOps;
  collectNestedOps(body, allOps);

  // Collect candidate roots (deduplicated) before mutating the IR.
  llvm::SetVector<Value> roots;
  for (Operation *op : allOps) {
    auto userId = getOpBlockId(op);
    if (!userId.has_value()) {
      continue;
    }
    for (Value operand : op->getOperands()) {
      if (isa<ShapedType>(operand.getType())) {
        continue; // only scalar operands can be cross-block scalar deps
      }
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !isOpInMainLoop(defOp, mainLoop)) {
        continue;
      }
      auto producerId = getOpBlockId(defOp);
      if (!producerId.has_value() || *producerId == *userId) {
        continue;
      }
      roots.insert(operand);
    }
  }

  for (Value root : roots) {
    // If this .value() failed, it must be a bug in above codes.
    auto producerId = getOpBlockId(root.getDefiningOp()).value();

    SmallVector<Operation *> sliceInOrder;
    DenseSet<Operation *> visited;
    llvm::SetVector<Value> boundaryTensors;
    buildScalarSlice(root, mainLoop, sliceInOrder, visited, boundaryTensors);

    // Pure scalar/memref chains reach no tensor, so no buffer is needed;
    // leave the original defining ops untouched.
    if (boundaryTensors.empty()) {
      continue;
    }

    rematerializeScalarDep(root, producerId, mainLoop, sliceInOrder);
  }
}

// Collect a single dependency value to depValueMap. Same-block check uses
// outermost id so inner ops of a multi-region op (e.g. subview at block 3
// inside ifOp at block 4) are not treated as cross-block consumers of a
// same-block producer.
//
// i1Found is set to true when the operand is a tensor with element type i1,
// signaling the caller to fall back (set ERRCODE_IGNORED + signalPassFailure)
// rather than process the dep through the multi-buffer pipeline. The operand
// is intentionally NOT added to depValueMap in that case.
// i1 return is done temporarily.
static void collectDepValue(Value operand, Block *body, Operation *currentOp,
                            DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value>> &depValueMap,
                            Value groupKey, bool &i1Found) {
  if (auto barg = dyn_cast<BlockArgument>(operand)) {
    if (barg.getOwner() == body &&
        !llvm::is_contained(depValueMap[groupKey], barg))
      depValueMap[groupKey].push_back(barg);
    return;
  }

  if (!outputToBlockId.count(operand))
    return;

  auto currentOutermost = getOutermostSsbufferId(currentOp);
  auto operandOutermost = getOutermostSsbufferId(operand.getDefiningOp());

  if (currentOutermost.has_value() && currentOutermost == operandOutermost)
    return;

  // i1 tensor deps: trigger fallback only for cross-block deps that are
  // actually about to be multi-buffered. Same-block i1 tensors (e.g. a
  // condition operand of an arith.select inside the same block) are
  // filtered out by the same-block check above and never enter the
  // multi-buffer pipeline, so they do not need the fallback.
  if (auto shapedType = dyn_cast<ShapedType>(operand.getType())) {
    if (shapedType.getElementType().isInteger(1)) {
      i1Found = true;
      return;
    }
  }

  if (!llvm::is_contained(depValueMap[groupKey], operand))
    depValueMap[groupKey].push_back(operand);
}

// Collect all ops with ssbuffer.id from allOps, grouped by id
// Returns 0=success, -1=invalid negative block id from upstream pass
static int
groupOpsBySsbufferId(SmallVector<Operation *> &allOps,
                     llvm::MapVector<int, SmallVector<Operation *>> &opsById) {
  llvm::MapVector<Value, Operation *> opsByValue;
  for (Operation *op : allOps) {
    auto id = getOpBlockId(op);
    if (!id.has_value()) {
      continue;
    }
    for (auto res : op->getResults()) {
      opsByValue[res] = op;
    }
  }
  // Deduplicate: a multi-result op (e.g. scf.if) is inserted N times in
  // opsByValue (once per result) and would otherwise appear N times in
  // opsById, leading to repeated processing of the same op.
  DenseSet<Operation *> seen;
  for (auto &p : opsByValue) {
    Operation *op = p.second;
    if (!seen.insert(op).second)
      continue;
    auto id = getOpBlockId(op);
    if (!id.has_value()) {
      continue;
    }
    opsById[*id].push_back(op);
  }
  // Also register ops that carry ssbuffer.block_id but have no results
  for (Operation *op : allOps) {
    auto id = getOpBlockId(op);
    if (!id.has_value() || !op->getResults().empty())
      continue;
    if (seen.insert(op).second)
      opsById[*id].push_back(op);
  }
  return 0;
}

// True when operand is produced by an op with a block_id and lives in a
// different logical block from the consumer (mirrors the same-block check
// in collectDepValue).
static bool
isCrossBlockDepOperand(Operation *consumerOp, Value operand,
                       const DenseMap<Value, int> &outputToBlockId) {
  if (!outputToBlockId.count(operand))
    return false;
  auto consumerOutermost = getOutermostSsbufferId(consumerOp);
  auto operandOutermost = getOutermostSsbufferId(operand.getDefiningOp());
  return !(consumerOutermost.has_value() &&
           consumerOutermost == operandOutermost);
}

// Invoke callback for each cross-block dep operand yielded by a multi-region
// op (e.g. scf.if, scf.while). Skips ops with fewer than 2 regions, empty
// regions, and regions whose terminator is not scf.yield.
static void
forEachYieldedCrossBlockDep(Operation *op,
                            const DenseMap<Value, int> &outputToBlockId,
                            llvm::function_ref<void(Value)> callback) {
  if (op->getNumRegions() < 2)
    return;
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto yieldOp = dyn_cast<scf::YieldOp>(region.back().getTerminator());
    if (!yieldOp)
      continue;
    for (Value operand : yieldOp->getOperands()) {
      if (isCrossBlockDepOperand(op, operand, outputToBlockId))
        callback(operand);
    }
  }
}

// Returns 0=success (including normal skip when blocks empty), -1=invalid
// negative block id Returns 0=success (including normal skip when blocks
// empty), -1=invalid negative block id. i1Found is set to true when any tensor
// dep collected here has element type i1; the caller is expected to abort and
// trigger fallback in that case.
int collectInnerBlockInfo(const MainLoop &loop,
                          DenseMap<Value, InnerBlockInfo> &blocks,
                          DenseMap<Value, SmallVector<Value>> &depValueMap,
                          SmallVector<Operation *> &allOps, bool &i1Found) {
  depValueMap.clear();
  Block *body = loop.getBody();
  if (!body)
    return 0;

  collectNestedOps(body, allOps);

  llvm::MapVector<int, SmallVector<Operation *>> opsById;
  if (groupOpsBySsbufferId(allOps, opsById) != 0)
    return -1;
  if (opsById.empty())
    return 0;

  // Build mapping from output to block id
  DenseMap<Value, int> outputToBlockId;
  for (auto &p : opsById)
    for (Operation *op : p.second)
      for (auto res : op->getResults())
        outputToBlockId[res] = p.first;

  // Collect dependency values for each block. Inner ops of multi-region ops
  // (e.g. scf.if) are included so their scalar deps get tracked; cross-block
  // judgment still attributes them to the ifOp via getOutermostSsbufferId.
  for (auto &p : opsById) {
    Operation *keyOp = nullptr;
    for (Operation *op : p.second) {
      if (!op->getResults().empty()) {
        keyOp = op;
        break;
      }
    }
    if (!keyOp)
      continue;
    Value groupKey = keyOp->getResult(0);
    InnerBlockInfo bi;
    bi.blockId = groupKey;
    bi.ops = p.second;
    blocks[groupKey] = bi;

    for (Operation *op : bi.ops)
      for (Value operand : op->getOperands())
        collectDepValue(operand, body, op, outputToBlockId, depValueMap,
                        groupKey, i1Found);
  }

  // Additional pass: collect deps from yield ops of multi-region consumers
  // (e.g. scf.if), treating the multi-region op as the dep consumer.
  for (auto &blockPair : blocks) {
    Value blockKey = blockPair.first;
    for (Operation *op : blockPair.second.ops) {
      forEachYieldedCrossBlockDep(op, outputToBlockId, [&](Value operand) {
        if (!llvm::is_contained(depValueMap[blockKey], operand))
          depValueMap[blockKey].push_back(operand);
      });
    }
  }

  return 0;
}

// Check if a yieldOp is already processed in blocks
static bool isYieldAlreadyProcessed(scf::YieldOp yieldOp,
                                    DenseMap<Value, InnerBlockInfo> &blocks) {
  for (auto &p : blocks) {
    if (llvm::is_contained(p.second.ops, yieldOp.getOperation())) {
      return true;
    }
  }
  return false;
}

// Process yield op that is not in blocks: add parent multi-region op as
// consumer Generic version: supports any op with >= 2 regions (scf.if,
// scf.while, etc.)
static void
processYieldNotInBlocks(scf::YieldOp yieldOp,
                        DenseMap<Value, InnerBlockInfo> &blocks,
                        DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  if (isYieldAlreadyProcessed(yieldOp, blocks))
    return;

  Operation *parentOp = yieldOp->getParentOp();
  // Generic: check if parent op has >= 2 regions
  if (!parentOp || parentOp->getNumRegions() < 2)
    return;

  for (Value operand : yieldOp->getOperands()) {
    // Add parent multi-region op as consumer for yield operands
    // This handles cases where depVal is only used in yield (not as direct
    // operand)
    depUserMap[operand].push_back(parentOp);
  }
}

DenseMap<Value, SmallVector<Operation *>>
buildDepUserMap(DenseMap<Value, InnerBlockInfo> &blocks,
                SmallVector<Operation *> &allOps,
                DenseMap<Value, SmallVector<Value>> &depValueMap) {
  DenseMap<Value, SmallVector<Operation *>> depUserMap;

  // First pass: process operations in blocks
  for (auto &p : blocks)
    for (Operation *op : p.second.ops)
      for (Value operand : op->getOperands())
        depUserMap[operand].push_back(op);

  // Second pass: process yield operations that are not in blocks (e.g., INT_MIN
  // block_id) Generic version: supports any multi-region op's yield
  for (Operation *op : allOps) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      processYieldNotInBlocks(yieldOp, blocks, depUserMap);
    }
  }

  return depUserMap;
}

// Check if depVal matches the special pattern with linalg::FillOp
static bool isEmptyFillPattern(Value depVal) {
  Operation *defOp = depVal.getDefiningOp();
  auto fillOp = dyn_cast<linalg::FillOp>(defOp);
  if (!fillOp)
    return false;

  if (fillOp.getOutputs().empty())
    return false;

  Value outs = fillOp.getOutputs()[0];
  if (!outs)
    return false;
  Operation *outsDef = outs.getDefiningOp();
  if (!outsDef)
    return false;
  return isa<tensor::EmptyOp>(outsDef) ||
         isa<bufferization::AllocTensorOp>(outsDef);
}

// Check if depVal is the result of a bufferization.alloc_tensor
static bool isAllocTensorPattern(Value depVal) {
  return isa_and_nonnull<bufferization::AllocTensorOp>(depVal.getDefiningOp());
}

static bool
hasMemrefDepValue(DenseMap<Value, SmallVector<Value>> &depValueMap) {
  for (auto &p : depValueMap) {
    Value groupKey = p.first;
    Operation *producer = groupKey.getDefiningOp();
    for (Value depVal : p.second) {
      if (isa<MemRefType>(depVal.getType())) {
        Operation *depOp = depVal.getDefiningOp();
        LDBG("MEMREF DEP: producer=<" +
             std::string(producer ? producer->getName().getStringRef().str()
                                  : "(block-arg)") +
             ">, depVal=<" +
             std::string(depOp ? depOp->getName().getStringRef().str()
                               : "(block-arg)") +
             ">");
        return true;
      }
    }
  }
  return false;
}

// If `clonedDepVals` is non-null, depVals that were actually cloned
static int cloneDepsToConsumers(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap, OpBuilder &builder,
    llvm::function_ref<bool(Value)> patternCheck,
    llvm::function_ref<Value(IRMapping &, OpBuilder &, Value depVal,
                             int userBlockId, ArrayRef<Operation *> users)>
        cloneFn,
    DenseSet<Value> *clonedDepVals = nullptr) {
  llvm::DenseSet<Value> seenVals;

  for (auto &blockPair : blocks) {
    auto depIt = depValueMap.find(blockPair.first);
    if (depIt == depValueMap.end())
      continue;

    for (Value depVal : depIt->second) {
      Operation *defOp = depVal.getDefiningOp();
      if (!defOp)
        continue;
      if (!seenVals.insert(depVal).second)
        continue;
      if (!patternCheck(depVal))
        continue;
      if (defOp->getParentOp() != loop.getOperation())
        continue;

      auto producerId = getOpBlockId(defOp);
      if (!producerId.has_value())
        continue;

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;

      // Group users by their consumer block_id, skipping users in the
      // producer's own block and users that no longer reference depVal.
      DenseMap<int, SmallVector<Operation *>> opsByBlockId;
      for (Operation *user : userIt->second) {
        auto userBlockId = getOpBlockId(user);
        if (!userBlockId.has_value() || *userBlockId == producerId)
          continue;
        bool stillUses = false;
        for (OpOperand &opnd : user->getOpOperands()) {
          if (opnd.get() == depVal) {
            stillUses = true;
            break;
          }
        }
        if (!stillUses)
          continue;
        opsByBlockId[*userBlockId].push_back(user);
      }

      for (auto &p : opsByBlockId) {
        int userBlockId = p.first;
        auto &users = p.second;
        if (users.empty())
          continue;

        Operation *firstUser = users.front();
        builder.setInsertionPoint(firstUser);
        IRMapping mapper;
        Value newVal = cloneFn(mapper, builder, depVal, userBlockId, users);
        if (!newVal)
          continue;
        if (clonedDepVals)
          clonedDepVals->insert(depVal);
        for (Operation *user : users) {
          user->replaceUsesOfWith(depVal, newVal);
        }
      }
    }
  }
  return 0;
}

// Phase 1: clone empty+fill (and any `ins` defining ops sharing the empty's
// parentOp) to consumer blocks; runs before dep collection because the
// cloned fill's `ins` chain may reach a producer-side tensor that Phase 2
// must see.
static int cloneEmptyFillsInBlocks(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap,
    OpBuilder &globalBuilder, DenseSet<Value> *clonedDepVals = nullptr) {
  return cloneDepsToConsumers(
      loop, blocks, depValueMap, depUserMap, globalBuilder, isEmptyFillPattern,
      [](IRMapping &mapper, OpBuilder &builder, Value depVal, int userBlockId,
         ArrayRef<Operation *> users) -> Value {
        auto fillOp = cast<linalg::FillOp>(depVal.getDefiningOp());
        // outs may come from either tensor::EmptyOp (the original case) or
        // bufferization.alloc_tensor (the new case). Both are "fresh
        // tensor" sources and treated identically below.
        Operation *origAllocLike = fillOp.getOutputs()[0].getDefiningOp();

        // Collect the `ins` operands whose defining op shares the parentOp
        // with the empty/alloc_tensor.
        SmallVector<Value> insToClone;
        Operation *allocParent = origAllocLike->getParentOp();
        for (Value insVal : fillOp.getInputs()) {
          Operation *insDef = insVal.getDefiningOp();
          if (!insDef || insDef->getParentOp() != allocParent)
            continue;
          insToClone.push_back(insVal);
        }

        Operation *newAllocLike = builder.clone(*origAllocLike, mapper);
        newAllocLike->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        mapper.map(origAllocLike->getResult(0), newAllocLike->getResult(0));

        for (Value insVal : insToClone) {
          Operation *insDef = insVal.getDefiningOp();
          Operation *newIns = builder.clone(*insDef, mapper);
          newIns->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
          mapper.map(insVal, newIns->getResult(0));
        }

        Operation *newFill = builder.clone(*fillOp, mapper);
        newFill->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        return newFill->getResult(0);
      },
      clonedDepVals);
}

static Value getIterCount(OpBuilder &builder, const MainLoop &loop,
                          Location loc, SmallVector<Operation *> *newOps,
                          int blockId = -1) {
  auto i32Type = builder.getI32Type();

  if (loop.isWhile()) {
    assert(loop.iterCounter &&
           "whileOp main_loop requires a global iteration counter");
    return loop.iterCounter;
  }

  auto forOp = cast<scf::ForOp>(loop.getOperation());
  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Type ivType = iv.getType();

  // Check if lower bound is a constant zero
  bool lbIsZero = false;
  if (auto constOp = lb.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
      lbIsZero = (intAttr.getInt() == 0);
  }

  Value iterIdx;
  if (lbIsZero) {
    // Optimization: if lb is 0, use iv directly (or iv/step if step != 1)
    bool stepIsOne = false;
    if (auto constOp = step.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
        stepIsOne = intAttr.getInt() == 1;
    if (stepIsOne) {
      iterIdx = iv;
    } else {
      iterIdx = builder.create<mlir::arith::DivUIOp>(loc, iv, step);
      if (newOps)
        newOps->push_back(iterIdx.getDefiningOp());
      if (blockId >= 0) {
        iterIdx.getDefiningOp()->setAttr(kBlockId,
                                         builder.getI32IntegerAttr(blockId));
      }
    }
  } else {
    // General case: (iv - lb) / step
    Value diff = builder.create<mlir::arith::SubIOp>(loc, iv, lb);
    iterIdx = builder.create<mlir::arith::DivUIOp>(loc, diff, step);
    if (newOps) {
      newOps->push_back(diff.getDefiningOp());
      newOps->push_back(iterIdx.getDefiningOp());
    }
    if (blockId >= 0) {
      diff.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
      iterIdx.getDefiningOp()->setAttr(kBlockId,
                                       builder.getI32IntegerAttr(blockId));
    }
  }

  // Cast to i32 if necessary
  if (ivType == i32Type)
    return iterIdx;

  Value result;
  constexpr int bits32 = 32;
  if (ivType.isIndex()) {
    result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
  } else if (auto intType = dyn_cast<mlir::IntegerType>(ivType)) {
    // Extend or truncate integer types to i32
    if (intType.getWidth() < bits32)
      result = builder.create<mlir::arith::ExtSIOp>(loc, i32Type, iterIdx);
    else if (intType.getWidth() > bits32)
      result = builder.create<mlir::arith::TruncIOp>(loc, i32Type, iterIdx);
    else
      return iterIdx;
  } else {
    result = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, iterIdx);
  }
  if (newOps)
    newOps->push_back(result.getDefiningOp());
  if (blockId >= 0) {
    result.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
  }
  return result;
}

// Compute buffer index: iterCount % N
static Value computeBufferIndex(OpBuilder &builder, const MainLoop &loop,
                                Location loc, int N,
                                SmallVector<Operation *> *newOps,
                                int blockId = -1) {
  Value iterCount = getIterCount(builder, loop, loc, newOps, blockId);
  Value Nval = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  Value bufIdx = builder.create<mlir::arith::RemSIOp>(loc, iterCount, Nval);
  if (newOps) {
    newOps->push_back(Nval.getDefiningOp());
    newOps->push_back(bufIdx.getDefiningOp());
  }
  // Tag counter operations with block_id
  if (blockId >= 0) {
    MLIRContext *ctx = builder.getContext();
    Nval.getDefiningOp()->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    bufIdx.getDefiningOp()->setAttr(kBlockId,
                                    builder.getI32IntegerAttr(blockId));
  }
  return bufIdx;
}

// Find the last op in `anchorOp->getBlock()` whose `ssbuffer.block_id`
// attribute matches `blockId`. Returns nullptr if anchorOp is null or has no
// block, or if no such op is found.
static Operation *findLastOpWithBlockIdInBlock(Operation *anchorOp,
                                               int blockId) {
  if (!anchorOp)
    return nullptr;
  Block *block = anchorOp->getBlock();
  if (!block)
    return nullptr;
  Operation *result = nullptr;
  for (Operation &op : *block) {
    if (auto id = getOpBlockId(&op); id.has_value() && *id == blockId)
      result = &op;
  }
  return result;
}

// Builds the producer-side buffer selection for `depVal`.
//
// Two parts:
//
//   1) Select chain (counter + cmpis + nested `arith.select`) is anchored
//      BEFORE `firstAnchor`. For buffer_num==2 it builds
//          select(remsi==1, buf[1], buf[0])
//      for buffer_num==3 it builds
//          select(remsi==2, buf[2], select(remsi==1, buf[1], buf[0]))
//      and so on; the innermost default is buf[0] (the remsi==0 case).
//
//   2) `bufferization.materialize_in_destination %depVal in restrict writable
//      %sel` is anchored AFTER `matAnchor`.
//
// intraDeps = [groupId, crossCoreProducerId] is set on the materialize op.
// intra_buffer is no longer needed (no scf.if wrapper).
static SmallVector<Operation *>
insertProducerLogic(OpBuilder &builder, Value depVal,
                    SmallVector<BufferPair> &buffers, Operation *firstAnchor,
                    Operation *matAnchor, const MainLoop &loop,
                    int groupId = -1) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = depVal.getLoc();

  // Build the destination memref.
  //   N==1: buffer[0] directly.
  //   N>=2: nested arith.select chain at firstAnchor.
  Value dstBuf;
  OpBuilder selBuilder(builder);
  selBuilder.setInsertionPoint(firstAnchor);

  Value bufIdx = computeBufferIndex(selBuilder, loop, loc, N, &newOps);

  // sel_0 = buf[0]                          ; default for remsi==0
  // sel_i = arith.select(remsi == i, buf[i], sel_(i-1))   for i in [1, N-1]
  Value selectedBuf = buffers[0].second;
  for (int i = 1; i < N; ++i) {
    Value iVal = selBuilder.create<mlir::arith::ConstantIntOp>(loc, i, 32);
    Value cmp = selBuilder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, bufIdx, iVal);
    newOps.push_back(iVal.getDefiningOp());
    newOps.push_back(cmp.getDefiningOp());
    selectedBuf = selBuilder.create<mlir::arith::SelectOp>(
        loc, cmp, buffers[i].second, selectedBuf);
    newOps.push_back(selectedBuf.getDefiningOp());
  }
  dstBuf = selectedBuf;

  // Insert materialize_in_destination right after matAnchor.
  OpBuilder matBuilder(builder);
  matBuilder.setInsertionPointAfter(matAnchor);
  MLIRContext *ctx = builder.getContext();
  auto matOp =
      matBuilder.create<mlir::bufferization::MaterializeInDestinationOp>(
          loc, mlir::TypeRange{}, depVal, dstBuf, mlir::UnitAttr::get(ctx),
          mlir::UnitAttr::get(ctx));
  newOps.push_back(matOp);
  if (groupId >= 0) {
    matOp->setAttr(kIntraDeps,
                   builder.getI32ArrayAttr({groupId, crossCoreProducerId}));
  }

  return newOps;
}

// A shared ring buffer is still needed by other compute blocks. Exposing it
// through independent `to_tensor restrict writable` operations both violates
// the restrict contract and lets bufferization overwrite a later consumer's
// input (for example, m0 can become exp(-inf - m0) before alpha1 reads m0).
// For explicitly selected outer loops, give each consumer a private snapshot.
// The allocation/copy belong to the consumer block, so they run only after its
// readiness test and the ring credit is returned after the whole block runs.
static mlir::bufferization::ToTensorOp
createToTensorOp(OpBuilder &builder, Location loc, mlir::Type tensorType,
                 Value buffer, const MainLoop &loop, int blockId) {
  if (loop->hasAttr(kOuterEventControlCount)) {
    auto memrefType = cast<MemRefType>(buffer.getType());
    auto snapshot = builder.create<memref::AllocOp>(loc, memrefType);
    auto copy = builder.create<hivm::CopyOp>(loc, TypeRange{}, buffer,
                                             snapshot.getResult());
    if (blockId >= 0) {
      auto attr = builder.getI32IntegerAttr(blockId);
      snapshot->setAttr(kBlockId, attr);
      copy->setAttr(kBlockId, attr);
    }
    buffer = snapshot.getResult();
  }
  return builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensorType, buffer, mlir::UnitAttr::get(builder.getContext()),
      mlir::UnitAttr::get(builder.getContext()));
}

// Insert buffer selection logic (scf.if + to_tensor) at the start of the given
// region. Returns the scf::IfOp that performs the buffer selection, or nullptr
// on failure.
static Operation *
insertBufferSelectionInRegion(OpBuilder &builder, Region &region, Location loc,
                              Value depVal, SmallVector<BufferPair> &buffers,
                              const MainLoop &loop, int blockId) {
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());

  // Insert at the start of the region
  builder.setInsertionPointToStart(&region.front());

  // Compute buffer index
  Value readIdx =
      computeBufferIndex(builder, loop, loc, buffers.size(), nullptr, blockId);

  // Build buffer selection if-else chain
  SmallVector<Operation *> newIfOps;
  SmallVector<Operation *> outIfOps;
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newIfOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer, loop, blockId);
      },
      [&](OpBuilder &b, Location l, Operation *op) -> Value {
        return cast<mlir::bufferization::ToTensorOp>(op).getResult();
      },
      tensorType, blockId);
  if (ret != 0)
    return nullptr;

  if (newIfOps.empty())
    return nullptr;

  // Tag all operations in newIfOps with block_id
  for (auto *op : newIfOps) {
    op->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
    op->setAttr(kIntraBuffer, builder.getUnitAttr());
  }

  // The main scf.if is the first one in outIfOps
  return outIfOps.front();
}

static int insertConsumerLogic(OpBuilder &builder, Value depVal,
                               SmallVector<BufferPair> &buffers,
                               const MainLoop &loop,
                               SmallVector<Operation *> &outIfOps,
                               int groupId = -1, int blockId = -1) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = builder.getInsertionPoint()->getLoc();

  if (N == kBufferCountOne) {
    auto memrefType = cast<MemRefType>(buffers[0].second.getType());
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());
    Operation *consumerOp = createToTensorOp(builder, loc, tensorType,
                                             buffers[0].second, loop, blockId);
    outIfOps.push_back(consumerOp);
    if (groupId >= 0) {
      consumerOp->setAttr(kIntraDeps, builder.getI32ArrayAttr({groupId, 0}));
    }
    return 0;
  }

  Value readIdx = computeBufferIndex(builder, loop, loc, N, &newOps, blockId);
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());
  mlir::TypeRange resultTypes(tensorType);
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer, loop, blockId);
      },
      [&](OpBuilder &b, Location l, Operation *op) -> Value {
        return cast<mlir::bufferization::ToTensorOp>(op).getResult();
      },
      resultTypes, blockId);
  if (ret != 0) {
    return ret;
  }
  if (groupId >= 0 && !outIfOps.empty()) {
    outIfOps.front()->setAttr(kIntraDeps,
                              builder.getI32ArrayAttr({groupId, 0}));
  }
  return 0;
}

static void addIntraBufferAttr(SmallVector<Operation *> &ops,
                               OpBuilder &builder) {
  for (auto *op : ops) {
    if (isa<scf::IfOp>(op) || isa<hivm::CopyOp>(op) ||
        isa<bufferization::ToTensorOp>(op)) {
      op->setAttr(kIntraBuffer, builder.getUnitAttr());
    }
  }
}

// Process normal consumer for a block: generate one buffer selection and share
// among all ops in the block
static int processNormalConsumerBlock(OpBuilder &consumedBuilder, Value depVal,
                                      SmallVector<BufferPair> &buffers,
                                      const MainLoop &loop,
                                      SmallVector<Operation *> &opsInBlock,
                                      int userBlockId, int groupId,
                                      OpBuilder &globalBuilder) {
  SmallVector<Operation *> resultIfOps;
  int ret = insertConsumerLogic(consumedBuilder, depVal, buffers, loop,
                                resultIfOps, groupId, userBlockId);
  if (ret != 0)
    return -1;

  if (resultIfOps.empty())
    return 0;

  addBlockAttrForOps(resultIfOps, userBlockId, globalBuilder);
  if (buffers.size() > kBufferCountOne) {
    for (auto *op : resultIfOps) {
      if (isa<scf::IfOp>(op)) {
        op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
      }
    }
  } else {
    addIntraBufferAttr(resultIfOps, globalBuilder);
  }

  Operation *resultIf = resultIfOps.back();
  Value selectedBuffer = resultIf->getResult(0);

  // Replace operands of all ops in this block
  for (Operation *opInBlock : opsInBlock) {
    for (OpOperand &use : opInBlock->getOpOperands()) {
      if (use.get() == depVal)
        use.set(selectedBuffer);
    }
  }
  return 0;
}

static void addBlockAttrForOps(SmallVector<Operation *> &newOps, int blockId,
                               OpBuilder &builder) {
  auto attr = builder.getI32IntegerAttr(blockId);
  for (auto *op : newOps)
    op->setAttr(kBlockId, attr);
}

// Handle all regions (except region 0) when depVal is used in that region's
// yield Generic version: supports any op with >= 2 regions (scf.if, scf.while,
// etc.) For scf.if: handles else region (region index 1) For scf.while: handles
// after region (region index 1) For any op with 3+ regions: handles all regions
// from index 1 onwards
static int processMultiRegionAllYields(OpBuilder &consumedBuilder, Value depVal,
                                       SmallVector<BufferPair> &buffers,
                                       const MainLoop &loop, Operation *depUser,
                                       int userBlockId, int groupId) {
  // Generic: check if op has >= 2 regions
  if (depUser->getNumRegions() < 2)
    return 0;

  // Iterate through all regions (from region 1 onwards, excluding region 0)
  for (size_t i = 1; i < depUser->getNumRegions(); ++i) {
    Region &region = depUser->getRegion(i);
    if (region.empty())
      continue;

    auto yieldOp = dyn_cast<scf::YieldOp>(region.back().getTerminator());
    if (!yieldOp)
      continue;

    for (OpOperand &operand : yieldOp->getOpOperands()) {
      if (operand.get() != depVal)
        continue;

      Operation *selectIf = insertBufferSelectionInRegion(
          consumedBuilder, region, yieldOp.getLoc(), depVal, buffers, loop,
          userBlockId);
      if (!selectIf)
        return -1;

      if (groupId >= 0) {
        selectIf->setAttr(kIntraDeps,
                          consumedBuilder.getI32ArrayAttr({groupId, 0}));
      }

      operand.set(selectIf->getResult(0));
      return 0; // Only handle one
    }
  }
  return 0;
}

// Check if depUser is a multi-region op and depVal is not a direct operand
// Returns true when op was added as consumer due to yield using depVal
// Generic version: supports any op with >= 2 regions (scf.if, scf.while, etc.)
static bool isMultiRegionConsumerFromYield(Operation *depUser, Value depVal) {
  // Check if op has >= 2 regions (generic for any multi-region op)
  if (depUser->getNumRegions() < 2)
    return false;

  for (OpOperand &operand : depUser->getOpOperands()) {
    if (operand.get() == depVal)
      return false; // depVal is a direct operand
  }
  return true; // depVal comes from yield
}

// Find the first op in `anchorOp->getBlock()` whose `ssbuffer.block_id`
// attribute matches `blockId`. Returns nullptr if anchorOp is null or has no
// block, or if no such op is found.
static Operation *findFirstOpWithBlockIdInBlock(Operation *anchorOp,
                                                int blockId) {
  if (!anchorOp)
    return nullptr;
  Block *block = anchorOp->getBlock();
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (auto id = getOpBlockId(&op); id.has_value() && *id == blockId)
      return &op;
  }
  return nullptr;
}

// Process producer and consumer for a single dependency value
static int processDepVal(Value depVal, const MainLoop &loop,
                         BufferMap &bufferMap,
                         DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                         OpBuilder &globalBuilder, int producerId,
                         int groupId) {
  Operation *depDefinedOp = depVal.getDefiningOp();
  if (!depDefinedOp)
    return 0;

  SmallVector<BufferPair> &buffers = bufferMap[depVal];

  auto userIt = depUserMap.find(depVal);
  if (userIt == depUserMap.end())
    return 0;
  SmallVector<Operation *> depUsers = userIt->second;

  // Read the module-level `ssbuffer.insertionOptimization` attribute inline so
  // processDepVal can be called multiple times in the same pass run and stay
  // in sync with whatever the Python caller last wrote onto the ModuleOp.
  bool enableOpt = true;
  if (mlir::ModuleOp mod = loop->getParentOfType<mlir::ModuleOp>())
    enableOpt = mod->hasAttr(CVPipeline::kInsertionOptimization);

  // Create producer.
  // When enable_buffer_insert_optimization is on:
  //   - select chain stays tightly before depDefinedOp (no region-start
  //     relocation — firstAnchor remains depDefinedOp).
  //   - materialize_in_destination is moved to the END of depDefinedOp's
  //     block_id=X region (after lastInRegion)
  Operation *firstAnchor = depDefinedOp;
  Operation *producerAnchor = depDefinedOp;
  if (enableOpt) {
    if (auto prodId = getOpBlockId(depDefinedOp); prodId.has_value()) {
      if (Operation *lastWithBlockId =
              findLastOpWithBlockIdInBlock(depDefinedOp, *prodId))
        producerAnchor = lastWithBlockId; // materialize moves to region end
    }
  }
  OpBuilder producedBuffers(loop.getContext());
  SmallVector<Operation *> producerNewOps =
      insertProducerLogic(producedBuffers, depVal, buffers, firstAnchor,
                          producerAnchor, loop, groupId);
  addBlockAttrForOps(producerNewOps, producerId, globalBuilder);

  // Single pass: process both normal ops and multi-region ops
  // For normal ops: generate one buffer selection per block_id and share among
  // all ops in that block For multi-region ops: process independently
  DenseMap<int, SmallVector<Operation *>> opsByBlockId;
  DenseMap<int, Value> processedBlockSelections;

  for (Operation *depUser : depUsers) {
    auto userBlockId = getOutermostSsbufferId(depUser);
    if (!userBlockId.has_value() || *userBlockId == producerId)
      continue;

    if (isMultiRegionConsumerFromYield(depUser, depVal)) {
      // Multi-region op: process independently
      OpBuilder consumedBuilder(loop.getContext());
      // When enable_buffer_insert_optimization is on, place the consumer chain
      // at the start of depUser's block_id=X region (before the first op with
      // that block_id). Otherwise keep "right before depUser".
      Operation *consumerAnchor = depUser;
      if (enableOpt) {
        if (auto userId = getOpBlockId(depUser); userId.has_value()) {
          if (Operation *firstInRegion =
                  findFirstOpWithBlockIdInBlock(depUser, *userId))
            consumerAnchor = firstInRegion;
        }
      }
      consumedBuilder.setInsertionPoint(consumerAnchor);

      if (int ret =
              processMultiRegionAllYields(consumedBuilder, depVal, buffers,
                                          loop, depUser, *userBlockId, groupId))
        return ret;
    } else {
      // Normal op: collect by block_id for batch processing
      opsByBlockId[*userBlockId].push_back(depUser);
    }
  }

  // Re-group by Block* before processing: getOutermostSsbufferId may merge
  // ops from different Regions (e.g. then/else of an scf.if) into the same
  // block_id, but a single buffer selection's result is only visible in its
  // own Region, so sharing across Regions would break SSA dominance.
  for (auto &blockPair : opsByBlockId) {
    int userBlockId = blockPair.first;
    SmallVector<Operation *> &opsInBlock = blockPair.second;
    if (opsInBlock.empty())
      continue;

    DenseMap<Block *, SmallVector<Operation *>> opsByBlock;
    for (Operation *op : opsInBlock) {
      Block *blk = op->getBlock();
      if (!blk)
        continue;
      opsByBlock[blk].push_back(op);
    }

    for (auto &regionPair : opsByBlock) {
      SmallVector<Operation *> &opsInRegion = regionPair.second;
      if (opsInRegion.empty())
        continue;

      Operation *firstOp = opsInRegion.front();
      OpBuilder consumedBuilder(loop.getContext());
      // When enable_buffer_insert_optimization is on, place the consumer chain
      // at the start of the dep user's block_id=X region (before the first op
      // with that block_id). Otherwise keep "right before firstOp".
      Operation *consumerAnchor = firstOp;
      if (enableOpt) {
        if (auto userId = getOpBlockId(firstOp); userId.has_value()) {
          if (Operation *firstInRegion =
                  findFirstOpWithBlockIdInBlock(firstOp, *userId))
            consumerAnchor = firstInRegion;
        }
      }
      consumedBuilder.setInsertionPoint(consumerAnchor);

      if (int ret = processNormalConsumerBlock(consumedBuilder, depVal, buffers,
                                               loop, opsInRegion, userBlockId,
                                               groupId, globalBuilder))
        return ret;
    }
  }

  return 0;
}



// Clone bufferization.alloc_tensor to each consumer block
int cloneAllocTensorsInBlocks(
    const MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap,
    OpBuilder &globalBuilder) {
  return cloneDepsToConsumers(
      loop, blocks, depValueMap, depUserMap, globalBuilder,
      isAllocTensorPattern,
      [](IRMapping &mapper, OpBuilder &builder, Value depVal, int userBlockId,
         ArrayRef<Operation *> users) -> Value {
        auto origAlloc =
            cast<bufferization::AllocTensorOp>(depVal.getDefiningOp());
        Operation *newAlloc = builder.clone(*origAlloc, mapper);
        newAlloc->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
        return newAlloc->getResult(0);
      });
}

// Phase 1 driver: collect deps, build user map, clone empty+fill pattern,
// rematerialize scalar deps rooted in a tensor. Mirrors the original inline
// sequence in addInnerMultiBuffer lines 1999-2022; exposed here so the main
// file can call it as a single unit.
int runDepAnalysisAndClone(MainLoop &mainLoop, OpBuilder &globalBuilder,
                           bool &i1Found, bool &memrefFound,
                           DenseMap<Value, InnerBlockInfo> &blocks,
                           DenseMap<Value, SmallVector<Value>> &depValueMap,
                           SmallVector<Operation *> &allOps,
                           DenseSet<Value> &phase1ClonedDepVals) {
  if (collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found) !=
      0)
    return -1;

  if (blocks.empty())
    return -1;

  // Memref-type dep values are not supported here.
  if (hasMemrefDepValue(depValueMap)) {
    LDBG("ERROR: Memref type dependent values found in user IR, fallback");
    memrefFound = true;
    return -1;
  }

  // Phase 1: build initial depUserMap and clone empty+fill patterns. We use
  // a fresh user map built from the initial allOps so the clone can find
  // consumer-block users; the cloned fills will rewrite those users' uses.
  DenseMap<Value, SmallVector<Operation *>> initialDepUserMap =
      buildDepUserMap(blocks, allOps, depValueMap);
  if (cloneEmptyFillsInBlocks(mainLoop, blocks, depValueMap, initialDepUserMap,
                              globalBuilder, &phase1ClonedDepVals) != 0)
    return -1;

  rematerializeTensorRootedScalarDeps(mainLoop);

  return 0;
}

} // namespace triton
} // namespace mlir
