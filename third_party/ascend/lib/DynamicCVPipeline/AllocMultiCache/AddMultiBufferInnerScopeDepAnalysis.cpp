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
int
collectInnerBlockInfo(const MainLoop &loop,
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
    for (Value depVal : p.second) {
      if (isa<MemRefType>(depVal.getType()))
        return true;
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

// Clone bufferization.alloc_tensor to each consumer block
int
cloneAllocTensorsInBlocks(const MainLoop &loop,
                          DenseMap<Value, InnerBlockInfo> &blocks,
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
int runDepAnalysisAndClone(
    MainLoop &mainLoop, OpBuilder &globalBuilder, bool &i1Found,
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