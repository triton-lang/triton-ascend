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

// Buffer count constants
constexpr int kBufferCountOne = 1;

namespace mlir {
namespace triton {

// Check if forOp has main_loop attribute
static bool hasMainLoopAttr(scf::ForOp forOp) {
  if (forOp->hasAttr(kMainLoop)) {
    return true;
  }
  if (auto *term = forOp.getBody()->getTerminator())
    return term->hasAttr(kMainLoop);
  return false;
}

// Collect main_loop forOps in a single block
static int collectMainLoopsInBlock(Block &block,
                                   SmallVector<scf::ForOp> &mainLoopForOps) {
  int count = 0;
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      if (hasMainLoopAttr(forOp)) {
        mainLoopForOps.push_back(forOp);
        count++;
      }
    }
  }
  return count;
}

// Recursively collect main_loop forOps, returns count of collected items
static int
collectMainLoopsRecursively(Region &region,
                            SmallVector<scf::ForOp> &mainLoopForOps) {
  int totalCount = 0;
  for (Block &block : region) {
    totalCount += collectMainLoopsInBlock(block, mainLoopForOps);
    for (Operation &op : block) {
      for (auto &nestedRegion : op.getRegions())
        totalCount += collectMainLoopsRecursively(nestedRegion, mainLoopForOps);
    }
  }
  return totalCount;
}

struct InnerBlockInfo {
  Value blockId;
  SmallVector<Operation *> ops;
};

// Effective block_id for cross-block dep judgment: prefer the enclosing
// multi-region op's block_id (inner ops are attributed to the op itself,
// not their own innermost id), else the innermost recorded block_id walking
// up to the main_loop boundary.
static std::optional<int64_t> getOutermostSsbufferId(Operation *op) {
  std::optional<int64_t> result;
  for (Operation *current = op; current; current = current->getParentOp()) {
    // Any multi-region op (scf.if, scf.while, ...) acts as a logical
    // block boundary: its block_id overrides anything inside it.
    if (current->getNumRegions() >= 2)
      return getOpBlockId(current);

    // main_loop is an attribute, not exclusive to forOp. Take the
    // boundary's id only if nothing was recorded on the way up.
    if (current->hasAttr(kMainLoop))
      return result.has_value() ? result : -1;

    // Otherwise remember the deepest id seen; the parent walk will
    // overwrite it if a closer-to-boundary op carries one.
    if (auto curId = getOpBlockId(current); curId.has_value())
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

// Get priority of forOp (lower value means higher priority)
// Priority order: main_loop (1) > block_id (2) > iter_args (3) > none (0)
// This is used to select the most relevant main loop when multiple candidates
// exist
static int getForOpPriority(scf::ForOp f) {
  constexpr int priorityMainLoop = 1;
  constexpr int priorityBlockId = 2;
  constexpr int priorityIterArgs = 3;

  // Check if forOp itself has main_loop attribute
  bool hasMainloop = f->hasAttr(kMainLoop);
  bool bodyHasMainloop = false;
  bool bodyHasBlockId = false;

  // Check terminator for main_loop and block_id attributes
  if (auto *term = f.getBody()->getTerminator()) {
    bodyHasMainloop = term->hasAttr(kMainLoop);
    bodyHasBlockId = term->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
  }

  bool opHasBlockId = f->getAttrOfType<IntegerAttr>(kBlockId) != nullptr;
  bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

  if (hasMainloop || bodyHasMainloop) {
    return priorityMainLoop;
  }
  if (opHasBlockId || bodyHasBlockId) {
    return priorityIterArgs;
  }
  if (hasIterArgs) {
    return priorityIterArgs;
  }
  return 0;
}

scf::ForOp findMainloopInScope(scope::ScopeOp scope) {
  SmallVector<Operation *> allOps;
  collectNestedOps(&scope.getBodyRegion().front(), allOps);

  scf::ForOp mainLoopForOp;
  int bestPriority = INT_MAX;

  for (Operation *op : allOps) {
    auto f = dyn_cast<scf::ForOp>(op);
    if (!f)
      continue;

    int priority = getForOpPriority(f);
    if (priority > 0 && priority < bestPriority) {
      mainLoopForOp = f;
      bestPriority = priority;
    }
  }
  return mainLoopForOp;
}

// Collect a single dependency value to depValueMap. Same-block check uses
// outermost id so inner ops of a multi-region op (e.g. subview at block 3
// inside ifOp at block 4) are not treated as cross-block consumers of a
// same-block producer.
static void collectDepValue(Value operand, Block *body, Operation *currentOp,
                            DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value>> &depValueMap,
                            Value groupKey) {
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
  if (!llvm::is_contained(depValueMap[groupKey], operand))
    depValueMap[groupKey].push_back(operand);
}

// Recursively find nested main_loop
static scf::ForOp findNestedMainloopInForOp(scf::ForOp forOp) {
  SmallVector<Operation *> allOps;
  collectNestedOps(forOp.getBody(), allOps);

  for (Operation *op : allOps) {
    auto nestedFor = dyn_cast<scf::ForOp>(op);
    if (!nestedFor)
      continue;
    if (nestedFor->hasAttr(kMainLoop))
      return nestedFor;
  }
  return {};
}

bool isInsideMainLoopForOp(Operation *op) {
  Operation *parent = op->getParentOp();
  if (!parent) {
    return false;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
    return forOp->hasAttr(kMainLoop);
  }
  return false;
}

bool isInsideMainLoopForOpTraverse(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kMainLoop)) {
        return true;
      }
    }
    parent = parent->getParentOp();
  }
  return false;
}

// Collect all ops with ssbuffer.id from allOps, grouped by id
// Returns 0=success, -1=invalid negative block ID from upstream pass
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
  // opsByValue and would produce duplicated dep_marks like [1, 1] otherwise.
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
// negative block ID
static int
collectInnerBlockInfo(scf::ForOp forOp, DenseMap<Value, InnerBlockInfo> &blocks,
                      DenseMap<Value, SmallVector<Value>> &depValueMap,
                      SmallVector<Operation *> &allOps) {
  depValueMap.clear();
  Block *body = forOp.getBody();
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
    Value groupKey = p.second.front()->getResult(0);
    InnerBlockInfo bi;
    bi.blockId = groupKey;
    bi.ops = p.second;
    blocks[groupKey] = bi;

    for (Operation *op : bi.ops)
      for (Value operand : op->getOperands())
        collectDepValue(operand, body, op, outputToBlockId, depValueMap,
                        groupKey);
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

// Check if depVal matches the special pattern: linalg::FillOp whose outs comes
// from a tensor::EmptyOp. When this pattern is detected, the pass can avoid
// allocating a multi-buffer (alloc + copy + select + to_tensor) by cloning
// the empty+fill ops to the consumer's position instead.
static bool isEmptyFillPattern(Value depVal) {
  Operation *defOp = depVal.getDefiningOp();
  auto fillOp = dyn_cast<linalg::FillOp>(defOp);
  if (!fillOp)
    return false;

  if (fillOp.getOutputs().empty())
    return false;

  Value outs = fillOp.getOutputs()[0];
  if (!outs || !isa_and_nonnull<tensor::EmptyOp>(outs.getDefiningOp()))
    return false;

  return true;
}

SmallVector<Value>
collectBufferValues(DenseMap<Value, SmallVector<Value>> &depValueMap) {
  SmallVector<Value> valueList;
  SmallVector<Operation *> seenOps;

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      Operation *op = depVal.getDefiningOp();
      if (!op || llvm::is_contained(seenOps, op))
        continue;
      seenOps.push_back(op);

      auto shapedType = dyn_cast<ShapedType>(depVal.getType());
      if (!shapedType)
        continue;

      // Skip tensor::EmptyOp - it should only get dep_mark, not buffer
      // allocation
      if (isa<tensor::EmptyOp>(op))
        continue;

      // Skip tensor::EmptyOp + linalg::FillOp pattern - it gets cloned
      // to the consumer's position instead of being multi-buffered
      if (isEmptyFillPattern(depVal))
        continue;

      valueList.push_back(depVal);
    }
  }

  return valueList;
}

SmallVector<Value>
collectScalarDeps(DenseMap<Value, SmallVector<Value>> &depValueMap,
                  DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  SmallVector<Value> scalarValueList;

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      if (isa<BlockArgument>(depVal))
        continue;

      Operation *depDefinedOp = depVal.getDefiningOp();
      if (!depDefinedOp)
        continue;

      if (isa<ShapedType>(depVal.getType())) {
        // tensor::EmptyOp should be treated like scalar, add dep_mark
        if (!isa<tensor::EmptyOp>(depDefinedOp))
          continue;
        // Check if definingOp's parentOp is a main_loop forOp
        auto *parentOp = depDefinedOp->getParentOp();
        if (!parentOp || !parentOp->hasAttr(kMainLoop))
          continue;
      }

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;

      auto producerId = getOpBlockId(depDefinedOp);
      if (!producerId.has_value()) {
        continue;
      }

      SmallVector<Operation *> depUsers = userIt->second;
      bool hasCrossBlockUser = false;
      for (Operation *depUser : depUsers) {
        auto userId = getOutermostSsbufferId(depUser);
        if (!userId.has_value() || *userId != *producerId) {
          hasCrossBlockUser = true;
          break;
        }
      }

      if (hasCrossBlockUser)
        scalarValueList.push_back(depVal);
    }
  }

  return scalarValueList;
}

// True if op is nested strictly inside the main loop.
static bool isOpInMainLoop(Operation *op, scf::ForOp mainLoop) {
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
static void buildScalarSlice(Value root, scf::ForOp mainLoop,
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
rematerializeScalarDep(Value root, int producerId, scf::ForOp mainLoop,
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
static void rematerializeTensorRootedScalarDeps(scf::ForOp mainLoop) {
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

    // Pure scalar/memref chains keep the existing dep_mark handling.
    if (boundaryTensors.empty()) {
      continue;
    }

    rematerializeScalarDep(root, producerId, mainLoop, sliceInOrder);
  }
}

// Compute iteration index: (iv - lb) / step, used for buffer selection in
// double buffering
static Value getIterCount(OpBuilder &builder, mlir::scf::ForOp forOp,
                          Location loc, SmallVector<Operation *> *newOps,
                          int blockId = -1) {
  auto i32Type = builder.getI32Type();
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

// Compute buffer index: iterCount % N
static Value computeBufferIndex(OpBuilder &builder, mlir::scf::ForOp forOp,
                                Location loc, int N,
                                SmallVector<Operation *> *newOps,
                                int blockId = -1) {
  Value iterCount = getIterCount(builder, forOp, loc, newOps, blockId);
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

static SmallVector<Operation *>
insertProducerLogic(OpBuilder &builder, Value depVal,
                    SmallVector<BufferPair> &buffers, mlir::scf::ForOp forOp) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = depVal.getLoc();
  // Single buffer producer logic
  if (N == kBufferCountOne) {
    Operation *producerOp = builder.create<hivm::CopyOp>(
        loc, mlir::TypeRange{}, depVal, buffers[0].second);
    if (!producerOp)
      return newOps;
    newOps.push_back(producerOp);
    return newOps;
  }

  Value bufIdx = computeBufferIndex(builder, forOp, loc, N, &newOps);
  SmallVector<Operation *> dummyOutIfOps;
  if (buildIfChain(
          builder, loc, bufIdx, buffers, newOps, dummyOutIfOps,
          [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
            return b.create<hivm::CopyOp>(l, mlir::TypeRange{}, depVal, buffer);
          },
          nullptr) != 0) {
    return {};
  }
  return newOps;
}

// Handle consumer when N==1 (directly return buffer)
static Operation *handleSingleBufferConsumer(OpBuilder &builder, Location loc,
                                             SmallVector<BufferPair> &buffers) {
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());
  return builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensorType, buffers[0].second,
      mlir::UnitAttr::get(builder.getContext()),
      mlir::UnitAttr::get(builder.getContext()));
}

// Helper function to create ToTensorOp
static mlir::bufferization::ToTensorOp createToTensorOp(OpBuilder &builder,
                                                        Location loc,
                                                        mlir::Type tensorType,
                                                        Value buffer) {
  return builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensorType, buffer, mlir::UnitAttr::get(builder.getContext()),
      mlir::UnitAttr::get(builder.getContext()));
}

static int insertConsumerLogic(OpBuilder &builder, Value depVal,
                               SmallVector<BufferPair> &buffers,
                               mlir::scf::ForOp forOp,
                               SmallVector<Operation *> &outIfOps,
                               int groupId = -1, int blockId = -1) {
  SmallVector<Operation *> newOps;
  int N = buffers.size();
  Location loc = builder.getInsertionPoint()->getLoc();

  if (N == kBufferCountOne) {
    Operation *consumerOp = handleSingleBufferConsumer(builder, loc, buffers);
    outIfOps.push_back(consumerOp);
    if (groupId >= 0) {
      consumerOp->setAttr("ssbuffer.intraDeps",
                          builder.getI32ArrayAttr({groupId, 0}));
    }
    return 0;
  }

  Value readIdx = computeBufferIndex(builder, forOp, loc, N, &newOps, blockId);
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());
  mlir::TypeRange resultTypes(tensorType);
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer);
      },
      [&](OpBuilder &b, Location l, Operation *op) -> Value {
        return cast<mlir::bufferization::ToTensorOp>(op).getResult();
      },
      resultTypes, blockId);
  if (ret != 0) {
    return ret;
  }
  if (groupId >= 0 && !outIfOps.empty()) {
    outIfOps.front()->setAttr("ssbuffer.intraDeps",
                              builder.getI32ArrayAttr({groupId, 0}));
  }
  return 0;
}

static void addBlockAttrForOps(SmallVector<Operation *> &newOps, int blockId,
                               OpBuilder &builder) {
  auto attr = builder.getI32IntegerAttr(blockId);
  for (auto *op : newOps)
    op->setAttr(kBlockId, attr);
}

// Add dep_mark attribute to operation
static void addDepMarkAttr(Operation *op, int depMark, OpBuilder &builder) {
  if (auto existingAttr =
          op->getAttrOfType<mlir::ArrayAttr>("ssbuffer.dep_mark")) {
    SmallVector<int> marks;
    for (auto attr : existingAttr)
      marks.push_back(cast<mlir::IntegerAttr>(attr).getInt());
    marks.push_back(depMark);
    op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr(marks));
  } else {
    op->setAttr("ssbuffer.dep_mark", builder.getI32ArrayAttr({depMark}));
  }
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

// Collect cross-block user operations
static SmallVector<Operation *>
collectCrossBlockUsers(Value depVal, int producerId,
                       DenseMap<Value, SmallVector<Operation *>> &depUserMap) {
  SmallVector<Operation *> crossBlockUsers;

  auto userIt = depUserMap.find(depVal);
  if (userIt == depUserMap.end())
    return crossBlockUsers;

  for (Operation *depUser : userIt->second) {
    auto userId = getOutermostSsbufferId(depUser);
    if ((!userId.has_value() || *userId != producerId) &&
        isInsideMainLoopForOpTraverse(depUser))
      crossBlockUsers.push_back(depUser);
  }
  return crossBlockUsers;
}

// Insert buffer selection logic (scf.if + to_tensor) at the start of the given
// region. Returns the scf::IfOp that performs the buffer selection, or nullptr
// on failure.
static Operation *
insertBufferSelectionInRegion(OpBuilder &builder, Region &region, Location loc,
                              Value depVal, SmallVector<BufferPair> &buffers,
                              mlir::scf::ForOp forOp, int blockId) {
  auto memrefType = mlir::cast<mlir::MemRefType>(buffers[0].second.getType());
  auto tensorType = mlir::RankedTensorType::get(memrefType.getShape(),
                                                memrefType.getElementType());

  // Insert at the start of the region
  builder.setInsertionPointToStart(&region.front());

  // Compute buffer index
  Value readIdx =
      computeBufferIndex(builder, forOp, loc, buffers.size(), nullptr, blockId);

  // Build buffer selection if-else chain
  SmallVector<Operation *> newIfOps;
  SmallVector<Operation *> outIfOps;
  int ret = buildIfChain(
      builder, loc, readIdx, buffers, newIfOps, outIfOps,
      [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
        return createToTensorOp(b, l, tensorType, buffer);
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

static void
markScalarDeps(SmallVector<Value> &scalarValueList,
               DenseMap<Value, SmallVector<Operation *>> &depUserMap,
               OpBuilder &builder, int startDepMark) {
  int nextDepMark = startDepMark;

  for (Value depVal : scalarValueList) {
    Operation *depDefinedOp = depVal.getDefiningOp();
    if (!depDefinedOp)
      continue;

    if (!isInsideMainLoopForOp(depDefinedOp))
      continue;

    auto producerId = getOpBlockId(depDefinedOp);
    if (!producerId.has_value())
      continue;
    auto crossBlockUsers =
        collectCrossBlockUsers(depVal, *producerId, depUserMap);
    if (crossBlockUsers.empty())
      continue;

    int depMark = nextDepMark++;
    // Add depmark to producer
    addDepMarkAttr(depDefinedOp, depMark, builder);
    // Add depmark to consumer
    for (Operation *depUser : crossBlockUsers) {
      addDepMarkAttr(depUser, depMark, builder);
    }
  }
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

// Process normal consumer for a block: generate one buffer selection and share
// among all ops in the block
static int processNormalConsumerBlock(OpBuilder &consumedBuilder, Value depVal,
                                      SmallVector<BufferPair> &buffers,
                                      mlir::scf::ForOp mainLoopForOp,
                                      SmallVector<Operation *> &opsInBlock,
                                      int userBlockId, int groupId,
                                      OpBuilder &globalBuilder) {
  SmallVector<Operation *> resultIfOps;
  int ret = insertConsumerLogic(consumedBuilder, depVal, buffers, mainLoopForOp,
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

// Handle all regions (except region 0) when depVal is used in that region's
// yield Generic version: supports any op with >= 2 regions (scf.if, scf.while,
// etc.) For scf.if: handles else region (region index 1) For scf.while: handles
// after region (region index 1) For any op with 3+ regions: handles all regions
// from index 1 onwards
static int processMultiRegionAllYields(OpBuilder &consumedBuilder, Value depVal,
                                       SmallVector<BufferPair> &buffers,
                                       mlir::scf::ForOp mainLoopForOp,
                                       Operation *depUser, int userBlockId,
                                       int groupId) {
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
          consumedBuilder, region, yieldOp.getLoc(), depVal, buffers,
          mainLoopForOp, userBlockId);
      if (!selectIf)
        return -1;

      if (groupId >= 0) {
        selectIf->setAttr("ssbuffer.intraDeps",
                          consumedBuilder.getI32ArrayAttr({groupId, 0}));
      }

      operand.set(selectIf->getResult(0));
      return 0; // Only handle one
    }
  }
  return 0;
}

// Process producer and consumer for a single dependency value
static int processDepVal(Value depVal, mlir::scf::ForOp mainLoopForOp,
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

  // Create producer
  OpBuilder producedBuffers(mainLoopForOp.getContext());
  producedBuffers.setInsertionPointAfter(depDefinedOp);
  SmallVector<Operation *> producerNewOps =
      insertProducerLogic(producedBuffers, depVal, buffers, mainLoopForOp);
  addBlockAttrForOps(producerNewOps, producerId, globalBuilder);
  if (buffers.size() > kBufferCountOne) {
    for (auto *op : producerNewOps) {
      if (isa<scf::IfOp>(op)) {
        op->setAttr(kIntraBuffer, globalBuilder.getUnitAttr());
      }
    }
  } else {
    addIntraBufferAttr(producerNewOps, globalBuilder);
  }

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
      OpBuilder consumedBuilder(mainLoopForOp.getContext());
      consumedBuilder.setInsertionPoint(depUser);

      if (int ret = processMultiRegionAllYields(consumedBuilder, depVal,
                                                buffers, mainLoopForOp, depUser,
                                                *userBlockId, groupId))
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
      OpBuilder consumedBuilder(mainLoopForOp.getContext());
      consumedBuilder.setInsertionPoint(firstOp);

      if (int ret = processNormalConsumerBlock(
              consumedBuilder, depVal, buffers, mainLoopForOp, opsInRegion,
              userBlockId, groupId, globalBuilder))
        return ret;
    }
  }

  return 0;
}

// Special handling for the tensor::EmptyOp + linalg::FillOp pattern:
// Clone the empty+fill pair into each consumer block (different from
// producerId), and replace depVal usage in those consumers with the new
// linalg::FillOp result. Skips the normal multi-buffer (alloc + hivm.copy +
// scf.if + to_tensor) path because cloning is cheaper for this pattern.
//
// When linalg.fill's `ins` operand's defining op lives in the same parentOp as
// the tensor::EmptyOp (i.e. it is producer-side and not yet across blocks),
// clone that defining op together so the cloned fill's `ins` references the
// clone instead of leaving a cross-block reference.
static int
cloneEmptyFillToConsumers(Value depVal, int producerId,
                          DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                          OpBuilder &builder) {
  auto fillOp = cast<linalg::FillOp>(depVal.getDefiningOp());
  auto origEmpty =
      cast<tensor::EmptyOp>(fillOp.getOutputs()[0].getDefiningOp());

  // Collect the `ins` operands whose defining op shares the parentOp with the
  // tensor::EmptyOp. These will be cloned together with empty+fill so the
  // cloned fill's `ins` becomes a local reference instead of a cross-block one.
  SmallVector<Value> insToClone;
  Operation *emptyParent = origEmpty->getParentOp();
  for (Value insVal : fillOp.getInputs()) {
    Operation *insDef = insVal.getDefiningOp();
    if (!insDef)
      continue;
    if (insDef->getParentOp() != emptyParent)
      continue;
    insToClone.push_back(insVal);
  }

  auto userIt = depUserMap.find(depVal);
  if (userIt == depUserMap.end())
    return 0;

  // Group users by their consumer block_id (skip users in the producer's own
  // block)
  DenseMap<int, SmallVector<Operation *>> opsByBlockId;
  for (Operation *user : userIt->second) {
    auto userBlockId = getOpBlockId(user);
    if (!userBlockId.has_value() || *userBlockId == producerId)
      continue;
    // Only keep users that still use depVal (not already replaced)
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

    // Clone tensor::EmptyOp and tag with consumer's block_id
    IRMapping mapper;
    Operation *newEmpty = builder.clone(*origEmpty, mapper);
    newEmpty->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));

    // Map origEmpty's result to the new empty so the cloned fill's outs
    // operand is rewired to point at the cloned empty.
    mapper.map(origEmpty->getResult(0), newEmpty->getResult(0));

    // Clone the `ins` defining ops that share the empty's parentOp, and
    // pre-register their value mapping so the cloned fill uses the cloned
    // scalar instead of leaving a cross-block reference.
    for (Value insVal : insToClone) {
      Operation *insDef = insVal.getDefiningOp();
      Operation *newIns = builder.clone(*insDef, mapper);
      newIns->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));
      mapper.map(insVal, newIns->getResult(0));
    }

    // Clone linalg::FillOp and tag with consumer's block_id
    Operation *newFill = builder.clone(*fillOp, mapper);
    newFill->setAttr(kBlockId, builder.getI32IntegerAttr(userBlockId));

    // Replace depVal with the new fill's result for all users in this block
    Value newResult = newFill->getResult(0);
    for (Operation *user : users) {
      user->replaceUsesOfWith(depVal, newResult);
    }
  }

  return 0;
}

// First pass: clone tensor::EmptyOp + linalg::FillOp patterns to each
// consumer block. Runs BEFORE the rest of dep collection because cloning
// the fill can introduce fresh cross-block references (e.g. the cloned
// fill's `ins` chain may now reach a producer-side tensor that needs
// multi-buffering), and the second pass needs to see them.
//
// Returns 0 on success, -1 on failure.
static int
cloneEmptyFillsInBlocks(scf::ForOp mainLoopForOp,
                        DenseMap<Value, InnerBlockInfo> &blocks,
                        DenseMap<Value, SmallVector<Value>> &depValueMap,
                        DenseMap<Value, SmallVector<Operation *>> &depUserMap,
                        OpBuilder &globalBuilder) {
  SmallVector<Operation *> seenOps;

  for (auto &blockPair : blocks) {
    auto depIt = depValueMap.find(blockPair.first);
    if (depIt == depValueMap.end())
      continue;

    // Walk the dep list carefully: cloneEmptyFillToConsumers mutates
    // depUserMap (via replaceUsesOfWith) but does not invalidate the
    // depValueMap iterator (no insert/erase of depValueMap). We just
    // dedupe by defining op to avoid re-processing the same fill.
    for (Value depVal : depIt->second) {
      Operation *defOp = depVal.getDefiningOp();
      if (!defOp || llvm::is_contained(seenOps, defOp))
        continue;

      if (!isEmptyFillPattern(depVal))
        continue;

      // Skip if parentOp is not the main_loop forOp (clone logic
      // currently expects the empty/fill to be inside main_loop).
      if (defOp->getParentOp() != mainLoopForOp.getOperation())
        continue;

      auto producerId = getOpBlockId(defOp);
      if (!producerId.has_value())
        continue;

      seenOps.push_back(defOp);
      if (cloneEmptyFillToConsumers(depVal, *producerId, depUserMap,
                                    globalBuilder) != 0)
        return -1;
    }
  }
  return 0;
}

// Process cross-block tensor dependencies for double buffering
static int processTensorDependencies(
    mlir::scf::ForOp mainLoopForOp, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap, BufferMap &bufferMap,
    OpBuilder &globalBuilder, int &groupId) {
  SmallVector<Operation *> seenOps;

  for (auto &blockPair : blocks) {
    Value blockKey = blockPair.first;
    auto depIt = depValueMap.find(blockKey);
    if (depIt == depValueMap.end())
      continue;

    SmallVector<Value> &depValues = depIt->second;

    for (Value depVal : depValues) {
      // Skip if already processed
      if (llvm::is_contained(seenOps, depVal.getDefiningOp()))
        continue;
      seenOps.push_back(depVal.getDefiningOp());

      // Validate dependency value (skip BlockArgument, null definingOp,
      // non-ShapedType)
      if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp() ||
          !isa<ShapedType>(depVal.getType()))
        continue;

      // Skip tensor::EmptyOp - it should only get dep_mark, not buffer
      // allocation
      if (isa<tensor::EmptyOp>(depVal.getDefiningOp()))
        continue;

      // Check if definingOp's parentOp is the main_loop forOp
      auto *parentOp = depVal.getDefiningOp()->getParentOp();
      if (parentOp != mainLoopForOp.getOperation())
        continue;

      // The empty+fill pattern has already been cloned by
      // cloneEmptyFillsInBlocks (run before dep collection). Skip
      // any remaining occurrences here.
      if (isEmptyFillPattern(depVal))
        continue;

      auto userIt = depUserMap.find(depVal);
      if (userIt == depUserMap.end())
        continue;
      SmallVector<Operation *> depUsers = userIt->second;

      auto producerId = getOpBlockId(depVal.getDefiningOp());
      if (!producerId.has_value()) {
        continue;
      }

      // Check if all users are in the same block
      bool allUsersSameBlock = true;
      for (Operation *depUser : depUsers) {
        auto userId = getOutermostSsbufferId(depUser);
        if (!userId.has_value() || *userId != *producerId) {
          allUsersSameBlock = false;
          break;
        }
      }
      if (allUsersSameBlock)
        continue;

      // Process cross-block dependency with double buffering
      if (processDepVal(depVal, mainLoopForOp, bufferMap, depUserMap,
                        globalBuilder, *producerId, groupId) != 0)
        return -1;
      groupId++;
    }
  }
  return 0;
}

static BufferMap insertBuffersBeforeFor(mlir::scf::ForOp forOp,
                                        SmallVector<Value> &valueList,
                                        OpBuilder &builder, int groupId) {
  BufferMap bufferMap;
  Block *parentBlock = forOp->getBlock();
  OpBuilder insertedBuffers(builder.getContext());
  insertedBuffers.setInsertionPoint(parentBlock, forOp->getIterator());

  using BufferCountManager = mlir::triton::BufferCountManager;
  int bufNum = BufferCountManager::getInstance().getBufferCountByType(
      BufferCountManager::DepType::IntraCore);

  for (Value depVal : valueList) {
    ShapedType shapedType = cast<ShapedType>(depVal.getType());
    Type elemType = shapedType.getElementType();
    AddressSpace addrSpace = AddressSpace::UB;

    SmallVector<BufferPair> buffers;
    for (int i = 0; i < bufNum; ++i) {
      MemRefType memrefType = MemRefType::get(
          shapedType.getShape(), elemType, MemRefLayoutAttrInterface{},
          AddressSpaceAttr::get(insertedBuffers.getContext(), addrSpace));

      auto allocOp =
          insertedBuffers.create<memref::AllocOp>(forOp.getLoc(), memrefType);

      auto genericType = MemRefType::get(shapedType.getShape(), elemType,
                                         MemRefLayoutAttrInterface{}, 0u);

      auto casted = insertedBuffers.create<memref::MemorySpaceCastOp>(
          forOp.getLoc(), genericType, allocOp.getResult());

      casted->setAttr("ssbuffer.intraDeps",
                      insertedBuffers.getI32ArrayAttr({groupId, 1}));

      buffers.push_back({casted.getResult(), casted.getResult()});
    }

    bufferMap[depVal] = buffers;
    groupId++;
  }

  return bufferMap;
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

static int addInnerMultiBuffer(mlir::scf::ForOp mainLoopForOp,
                               OpBuilder &builder, scope::ScopeOp vectorScope,
                               int &groupId) {
  OpBuilder globalBuilder(mainLoopForOp.getContext());

  // Two-phase dep collection for empty+fill cloning:
  //   Phase 1 (initial): collect deps, build user map, then clone the
  //                      tensor::EmptyOp + linalg::FillOp pattern into each
  //                      consumer's block. The cloned fill's `ins` chain can
  //                      introduce fresh cross-block references (e.g. a
  //                      producer-side tensor referenced by the cloned
  //                      tensor.extract / arith.extf).
  //   Scalar rematerialize: trace each cloned fill's `ins` chain. If the
  //                        chain reaches a producer-side tensor (via
  //                        tensor.extract / arith.extf), build a fresh
  //                        block-local chain that reads from a new
  //                        multi-buffer added in Phase 2.
  //   Phase 2 (re-collect): re-run collectInnerBlockInfo / buildDepUserMap
  //                        so the new cross-block refs (from clone + scalar
  //                        rematerialize) are visible to the multi-buffer
  //                        pipeline. The multi-buffer transfer structure
  //                        (allocs, scf.if, intraDeps, intra_buffer) is
  //                        created here and is not affected by the reorder.
  DenseMap<Value, InnerBlockInfo> blocks;
  DenseMap<Value, SmallVector<Value>> depValueMap;
  SmallVector<Operation *> allOps;
  if (collectInnerBlockInfo(mainLoopForOp, blocks, depValueMap, allOps) != 0)
    return -1;

  if (blocks.empty())
    return -1;

  // Phase 1: build initial depUserMap and clone empty+fill patterns. We use
  // a fresh user map built from the initial allOps so the clone can find
  // consumer-block users; the cloned fills will rewrite those users' uses.
  DenseMap<Value, SmallVector<Operation *>> initialDepUserMap =
      buildDepUserMap(blocks, allOps, depValueMap);
  if (cloneEmptyFillsInBlocks(mainLoopForOp, blocks, depValueMap,
                              initialDepUserMap, globalBuilder) != 0)
    return -1;

  // Break tensor-rooted cross-block scalar dependencies AFTER the empty+fill
  // clone. The clone can introduce fresh scalar refs (e.g. a cloned
  // arith.extf whose operand is a producer-side tensor.extract) that need to
  // be rematerialized to a block-local chain reading from a Phase-2
  // multi-buffer. Running this before the clone leaves these refs invisible.
  rematerializeTensorRootedScalarDeps(mainLoopForOp);

  // Phase 2: re-collect deps now that cloned ops (and rematerialized scalar
  // chains) have created new cross-block references. depValueMap and allOps
  // are cleared inside collectInnerBlockInfo, so we re-discover everything
  // from scratch.
  blocks.clear();
  depValueMap.clear();
  allOps.clear();
  if (collectInnerBlockInfo(mainLoopForOp, blocks, depValueMap, allOps) != 0)
    return -1;

  if (blocks.empty())
    return -1;

  // Memref-type dep values are not supported here; fail loudly so downstream
  // passes don't see an unmarked-but-skipped scope.
  if (hasMemrefDepValue(depValueMap)) {
    LDBG("ERROR: Memref type dependent values found!");
    return -1;
  }

  auto depUserMap = buildDepUserMap(blocks, allOps, depValueMap);

  auto valueList = collectBufferValues(depValueMap);
  auto bufferMap =
      insertBuffersBeforeFor(mainLoopForOp, valueList, builder, groupId);

  auto scalarValueList = collectScalarDeps(depValueMap, depUserMap);

  markScalarDeps(scalarValueList, depUserMap, globalBuilder, 1);

  if (processTensorDependencies(mainLoopForOp, blocks, depValueMap, depUserMap,
                                bufferMap, globalBuilder, groupId) != 0) {
    return -1;
  }

  return 0;
}

void AddMultiBufferInnerScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<mlir::annotation::AnnotationDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect>();
}

void AddMultiBufferInnerScopePass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module.getContext());

  LDBG("Enter pass.");

  module.walk([&](scope::ScopeOp scope) -> WalkResult {
    // Step 1: Check if scope has coreType attribute
    auto coreTypeAttr =
        scope->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      return WalkResult::advance();

    // Step 2: Check if core type is VECTOR
    hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
    if (coreType != hivm::TCoreType::VECTOR) {
      LDBG("Not vector scope");
      return WalkResult::advance();
    }

    // Step 3: Collect all forOps with main_loop attribute
    SmallVector<scf::ForOp> mainLoopForOps;
    int foundCount =
        collectMainLoopsRecursively(scope.getBodyRegion(), mainLoopForOps);
    if (foundCount < 0) {
      LDBG("collectMainLoopsRecursively failed");
      signalPassFailure();
      return WalkResult::interrupt();
    }
    if (foundCount == 0)
      return WalkResult::advance();

    // Step 4: Process each main_loop forOp
    int groupId = 0;
    for (scf::ForOp mainLoopForOp : mainLoopForOps) {
      scf::ForOp nestedMainloop = findNestedMainloopInForOp(mainLoopForOp);
      if (nestedMainloop) {
        LDBG("Nested main_loop found, this is not allowed");
        signalPassFailure();
        return WalkResult::interrupt();
      }
      if (addInnerMultiBuffer(mainLoopForOp, builder, scope, groupId) != 0) {
        LDBG("addInnerMultiBuffer failed");
        signalPassFailure();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferInnerScopePass() {
  return std::make_unique<AddMultiBufferInnerScopePass>();
}

void registerAddMultiBufferInnerScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddMultiBufferInnerScopePass();
  });
}

} // namespace triton
} // namespace mlir
