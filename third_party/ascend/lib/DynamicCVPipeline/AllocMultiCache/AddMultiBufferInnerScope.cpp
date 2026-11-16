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
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScopeDepAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
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

// Collect main_loop loops (forOp / whileOp) in a single block
static int collectMainLoopsInBlock(Block &block,
                                   SmallVector<Operation *> &mainLoops) {
  int count = 0;
  for (Operation &op : block) {
    if (isMainLoopOp(&op)) {
      mainLoops.push_back(&op);
      count++;
    }
  }
  return count;
}

// Recursively collect main_loop loops, returns count of collected items
static int collectMainLoopsRecursively(Region &region,
                                       SmallVector<Operation *> &mainLoops) {
  int totalCount = 0;
  for (Block &block : region) {
    totalCount += collectMainLoopsInBlock(block, mainLoops);
    for (Operation &op : block) {
      for (auto &nestedRegion : op.getRegions())
        totalCount += collectMainLoopsRecursively(nestedRegion, mainLoops);
    }
  }
  return totalCount;
}

// Recursively find a nested main_loop (forOp / whileOp) inside `loop`'s body
static Operation *findNestedMainloop(const MainLoop &loop) {
  SmallVector<Operation *> allOps;
  collectNestedOps(loop.getBody(), allOps);

  for (Operation *op : allOps) {
    if (isa<scf::ForOp, scf::WhileOp>(op) && op->hasAttr(kMainLoop))
      return op;
  }
  return {};
}

bool isInsideMainLoopForOp(Operation *op) {
  return isMainLoopOp(op->getParentOp());
}

bool isInsideMainLoopForOpTraverse(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMainLoopOp(parent))
      return true;
  }
  return false;
}

SmallVector<Value>
collectBufferValues(DenseMap<Value, SmallVector<Value>> &depValueMap,
                    const DenseSet<Value> &clonedDepVals) {
  SmallVector<Value> valueList;
  llvm::DenseSet<Value> seenVals;

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      Operation *op = depVal.getDefiningOp();
      if (!op)
        continue;
      if (!seenVals.insert(depVal).second)
        continue;

      auto shapedType = dyn_cast<ShapedType>(depVal.getType());
      if (!shapedType)
        continue;

      // Skip tensor::EmptyOp - it carries no producer-consumer value flow;
      // cloned per consumer in Phase 1 or otherwise harmless.
      if (isa<tensor::EmptyOp>(op))
        continue;

      // Skip depVals that were already cloned by cloneEmptyFillsInBlocks
      if (clonedDepVals.contains(depVal))
        continue;

      // Skip bufferization.alloc_tensor
      if (isa<bufferization::AllocTensorOp>(op))
        continue;

      valueList.push_back(depVal);
    }
  }

  return valueList;
}

static BufferMap insertBuffersBeforeLoop(const MainLoop &loop,
                                         SmallVector<Value> &valueList,
                                         OpBuilder &builder, int groupId) {
  BufferMap bufferMap;
  Block *parentBlock = loop.getBlock();
  OpBuilder insertedBuffers(builder.getContext());
  insertedBuffers.setInsertionPoint(parentBlock, loop.getIterator());

  BufferCountManager bufferCountMgr(loop.getOperation());
  int bufNum = bufferCountMgr.getBufferCountByType(
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
          insertedBuffers.create<memref::AllocOp>(loop.getLoc(), memrefType);

      auto genericType = MemRefType::get(shapedType.getShape(), elemType,
                                         MemRefLayoutAttrInterface{}, 0u);

      auto casted = insertedBuffers.create<memref::MemorySpaceCastOp>(
          loop.getLoc(), genericType, allocOp.getResult());

      buffers.push_back({casted.getResult(), casted.getResult()});
    }

    bufferMap[depVal] = buffers;
    groupId++;
  }

  return bufferMap;
}

// Build the before-region of the new whileOp
static void buildBeforeRegion(scf::WhileOp oldWhile, OpBuilder &bb, Location bl,
                              ValueRange iterArgs) {
  Block *oldBefore = oldWhile.getBeforeBody();
  IRMapping mapper;
  unsigned numOrig = oldBefore->getNumArguments();
  for (unsigned i = 0; i < numOrig; ++i)
    mapper.map(oldBefore->getArgument(i), iterArgs[i]);

  Operation *oldCond = nullptr;
  for (Operation &op : *oldBefore) {
    if (isa<scf::ConditionOp>(&op)) {
      oldCond = &op;
      continue;
    }
    bb.clone(op, mapper);
  }
  if (!oldCond)
    return;

  SmallVector<Value> newCondOps;
  for (Value operand : oldCond->getOperands()) {
    Value mapped = mapper.lookupOrNull(operand);
    newCondOps.push_back(mapped ? mapped : operand);
  }
  newCondOps.push_back(iterArgs[numOrig]);
  Value condValue = newCondOps.front();
  ArrayRef<Value> carriedValues = ArrayRef<Value>(newCondOps).drop_front();
  bb.create<scf::ConditionOp>(bl, condValue, carriedValues);
}

// Build the after-region of the new whileOp. The counter add-1 is not
// created here; it is inserted later by insertWhileCounterOps at the right
// position. counterIterArgOut is yielded as a placeholder and gets replaced.
static void buildAfterRegion(scf::WhileOp oldWhile, OpBuilder &ab, Location al,
                             ValueRange iterArgs, Value &counterIterArgOut) {
  Block *oldAfter = oldWhile.getAfterBody();
  unsigned numOrig = oldAfter->getNumArguments();
  counterIterArgOut = iterArgs[numOrig];

  IRMapping mapper;
  for (unsigned i = 0; i < numOrig; ++i)
    mapper.map(oldAfter->getArgument(i), iterArgs[i]);

  Operation *oldYield = nullptr;
  for (Operation &op : *oldAfter) {
    if (isa<scf::YieldOp>(&op)) {
      oldYield = &op;
      continue;
    }
    ab.clone(op, mapper);
  }
  if (!oldYield)
    return;

  SmallVector<Value> newYieldOps;
  for (Value operand : oldYield->getOperands()) {
    Value mapped = mapper.lookupOrNull(operand);
    newYieldOps.push_back(mapped ? mapped : operand);
  }
  newYieldOps.push_back(counterIterArgOut);
  ab.create<scf::YieldOp>(al, newYieldOps);
}

// Create a pass-managed global iteration counter for a whileOp main_loop
static std::pair<Value, scf::WhileOp>
setupWhileIterArgCounter(const MainLoop &loop, OpBuilder &builder) {
  auto oldWhile = cast<scf::WhileOp>(loop.getOperation());
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();
  Type i32Type = builder.getI32Type();

  // Init 0 for the new counter iter_arg, inserted before oldWhile so the
  // new whileOp can replace it in-place.
  OpBuilder preBuilder(ctx);
  preBuilder.setInsertionPoint(oldWhile);
  Value zero = preBuilder.create<arith::ConstantIntOp>(loc, 0, 32);

  // Old inits/result-types + i32 counter appended at the end.
  SmallVector<Value> newInits(oldWhile.getInits().begin(),
                              oldWhile.getInits().end());
  newInits.push_back(zero);
  SmallVector<Type> newResultTypes(oldWhile.getResultTypes().begin(),
                                   oldWhile.getResultTypes().end());
  newResultTypes.push_back(i32Type);

  // Captured by the after-builder when the do-region is constructed;
  // returned to the caller as the live iteration count.
  Value counterIterArg;

  OpBuilder cb(ctx);
  cb.setInsertionPoint(oldWhile);
  auto newWhile = cb.create<scf::WhileOp>(
      loc, newResultTypes, newInits,
      [&](OpBuilder &bb, Location bl, ValueRange iterArgs) {
        buildBeforeRegion(oldWhile, bb, bl, iterArgs);
      },
      [&](OpBuilder &ab, Location al, ValueRange iterArgs) {
        buildAfterRegion(oldWhile, ab, al, iterArgs, counterIterArg);
      });

  // Move attrs
  for (auto attr : oldWhile->getAttrs())
    newWhile->setAttr(attr.getName(), attr.getValue());
  newWhile->setAttr(kIterCounter, cb.getUnitAttr());

  for (unsigned i = 0, e = oldWhile.getNumResults(); i < e; ++i)
    oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
  oldWhile.erase();

  return {counterIterArg, newWhile};
}

// Insert the iterCounter add-1 (and its constant 1) right after the last op
// sharing the block_id of the first op that consumes mainLoop.iterCounter.
// Must run after processTensorDependencies (dispatch ops land there later).
// Fallback when no counter user exists: last op's block_id, mirroring the
// legacy buildAfterRegion placement.
static void insertWhileCounterOps(const MainLoop &mainLoop) {
  if (!mainLoop.isWhile() || !mainLoop.iterCounter)
    return;

  auto whileOp = cast<scf::WhileOp>(mainLoop.getOperation());
  Block *doBlock = whileOp.getAfterBody();
  if (!doBlock)
    return;

  std::optional<int> counterBlockId;
  for (Operation &op : *doBlock) {
    if (llvm::is_contained(op.getOperands(), mainLoop.iterCounter)) {
      counterBlockId = getOpBlockId(&op);
      break;
    }
  }
  if (!counterBlockId) {
    for (Operation &op : llvm::reverse(*doBlock)) {
      if (auto id = getOpBlockId(&op); id.has_value()) {
        counterBlockId = id;
        break;
      }
    }
    if (!counterBlockId)
      return;
  }

  Operation *lastWithBlockId = nullptr;
  for (Operation &op : *doBlock) {
    auto id = getOpBlockId(&op);
    if (id.has_value() && *id == *counterBlockId)
      lastWithBlockId = &op;
  }

  Location loc = lastWithBlockId->getLoc();
  OpBuilder builder(mainLoop.getContext());
  builder.setInsertionPointAfter(lastWithBlockId);
  IntegerAttr blockIdAttr = builder.getI32IntegerAttr(*counterBlockId);

  Value one = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(mainLoop.iterCounter.getType(), 1));
  one.getDefiningOp()->setAttr(kBlockId, blockIdAttr);

  Value nextCounter =
      builder.create<arith::AddIOp>(loc, mainLoop.iterCounter, one);
  Operation *iterAddOp = nextCounter.getDefiningOp();
  iterAddOp->setAttr(kBlockId, blockIdAttr);
  iterAddOp->setAttr(kIterCounter, builder.getUnitAttr());

  // Replace the placeholder counterIterArgOut operand in the yield with
  // nextCounter so the iter_arg increments each iteration.
  Operation *yieldOp = doBlock->getTerminator();
  SmallVector<Value> newOperands(yieldOp->getOperands().begin(),
                                 yieldOp->getOperands().end());
  bool replaced = false;
  for (Value &operand : newOperands) {
    if (operand == mainLoop.iterCounter) {
      operand = nextCounter;
      replaced = true;
      break;
    }
  }
  if (replaced)
    yieldOp->setOperands(newOperands);
}

static int addInnerMultiBuffer(MainLoop mainLoop, OpBuilder &builder,
                               scope::ScopeOp vectorScope, int &groupId,
                               bool &i1Found) {
  OpBuilder globalBuilder(mainLoop.getContext());

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

  // whileOp: bufNum>1 needs pre-created counter (no implicit iter count);
  // bufNum==1 skips to avoid dead iter_arg.
  if (mainLoop.isWhile()) {
    BufferCountManager bufferCountMgr(mainLoop.getOperation());
    int bufNum = bufferCountMgr.getBufferCountByType(
        BufferCountManager::DepType::IntraCore);
    if (bufNum > kBufferCountOne) {
      auto [counter, newWhile] =
          setupWhileIterArgCounter(mainLoop, globalBuilder);
      // Update mainLoop so subsequent collections target the new whileOp
      // (the old one was erased inside setupWhileIterArgCounter).
      mainLoop.op = newWhile;
      mainLoop.body = newWhile.getAfterBody();
      mainLoop.iterCounter = counter;
    }
  }

  DenseSet<Value> phase1ClonedDepVals;
  if (runDepAnalysisAndClone(mainLoop, globalBuilder, i1Found, blocks,
                             depValueMap, allOps, phase1ClonedDepVals) != 0)
    return -1;

  // Phase 2: re-collect deps now that cloned ops (and rematerialized scalar
  // chains) have created new cross-block references. depValueMap and allOps
  // are cleared inside collectInnerBlockInfo, so we re-discover everything
  // from scratch.
  blocks.clear();
  depValueMap.clear();
  allOps.clear();
  if (collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found) !=
      0)
    return -1;

  // Phase 2 may surface i1 tensor deps that the clone introduced (e.g. a
  // cloned scalar chain reaching a producer-side i1 tensor). Abort here too.
  if (i1Found) {
    LDBG("i1 tensor dep found in Phase 2, falling back");
    return -1;
  }

  if (blocks.empty())
    return -1;

  // Drop memref-typed deps from Phase 2's collection
  int droppedMemrefDeps = 0;
  for (auto &p : depValueMap) {
    llvm::erase_if(p.second, [&droppedMemrefDeps](Value v) {
      if (isa<MemRefType>(v.getType())) {
        ++droppedMemrefDeps;
        return true;
      }
      return false;
    });
  }
  if (droppedMemrefDeps > 0)
    LDBG("Dropped " + std::to_string(droppedMemrefDeps) +
         " clone-induced memref deps from Phase 2 depValueMap");

  auto depUserMap = buildDepUserMap(blocks, allOps, depValueMap);

  // Clone bufferization.alloc_tensor deps to each consumer's block.
  if (cloneAllocTensorsInBlocks(mainLoop, blocks, depValueMap, depUserMap,
                                globalBuilder) != 0)
    return -1;
  auto valueList = collectBufferValues(depValueMap, phase1ClonedDepVals);
  LLVM_DEBUG(
      llvm::dbgs()
      << "[addInnerMultiBuffer] before insertBuffersBeforeLoop, valueList.size="
      << valueList.size() << "\n");

  auto bufferMap =
      insertBuffersBeforeLoop(mainLoop, valueList, builder, groupId);

  LLVM_DEBUG(llvm::dbgs()
             << "[addInnerMultiBuffer] before processTensorDependencies\n");
  if (processTensorDependencies(mainLoop, blocks, depValueMap, depUserMap,
                                bufferMap, globalBuilder, groupId,
                                phase1ClonedDepVals) != 0) {
    return -1;
  }

  // WhileOp only: insert the counter add-1 now that the dispatch ops that
  // use mainLoop.iterCounter have been emitted.
  insertWhileCounterOps(mainLoop);

  LLVM_DEBUG(llvm::dbgs() << "[addInnerMultiBuffer] DONE\n");
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

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpBuilder builder(module.getContext());

  // Scan existing kIntraDeps to start groupId after the max already used
  // (e.g. by InterCoreTransferAndSyncPass for C2C fixpipe transfers).
  int groupId = 0;
  module.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(CVPipeline::kIntraDeps)) {
      if (!attr.empty()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr[0])) {
          groupId = std::max(groupId, static_cast<int>(intAttr.getInt()) + 1);
        }
      }
    }
  });

  LDBG("Enter pass.");

  auto walkResult = module.walk([&](scope::ScopeOp scope) -> WalkResult {
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

    // Step 3: Collect all main_loop loops (forOp / whileOp) in the scope
    SmallVector<Operation *> mainLoops;
    int foundCount =
        collectMainLoopsRecursively(scope.getBodyRegion(), mainLoops);
    if (foundCount < 0) {
      LDBG("collectMainLoopsRecursively failed");
      return WalkResult::interrupt();
    }
    if (foundCount == 0)
      return WalkResult::advance();

    // Step 4: Process each main_loop
    for (Operation *loopOp : mainLoops) {
      MainLoop mainLoop(loopOp);
      if (findNestedMainloop(mainLoop)) {
        LDBG("Nested main_loop found, this is not allowed");
        return WalkResult::interrupt();
      }
      // i1Found is reset per main_loop so it only triggers fallback for
      // the current scope's deps.
      bool i1Found = false;
      int ret = addInnerMultiBuffer(mainLoop, builder, scope, groupId, i1Found);
      if (i1Found) {
        LDBG("i1 tensor dep found, setting fallback attribute");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
        return WalkResult::interrupt();
      }
      if (ret != 0) {
        LDBG(
            "addInnerMultiBuffer failed for main_loop; signaling pass failure");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }

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
