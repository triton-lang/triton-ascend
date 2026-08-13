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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferCubeScope.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <climits>
#include <optional>

static constexpr const char *DEBUG_TYPE = "AddMultiBufferCubeScope";
#define LDBG(...)                                                              \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

// ---- Helpers ---------------------------------------------------------------

// Read `ssbuffer.cube_buf_count` from the module. Defaults to 2 when absent.
static int getCubeBufCount(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(CVPipeline::kCubeBufCount))
    return static_cast<int>(attr.getInt());
  return 2;
}

// Parse `ssbuffer.cubeBuffer = [groupId, role]` into a (group, role) pair.
// Returns nullopt if the op lacks the attribute or has a malformed value.
static std::optional<std::pair<int, int>>
getCubeBufferGroup(Operation *op) {
  auto arr = op->getAttrOfType<ArrayAttr>(CVPipeline::kCubeBuffer);
  if (!arr || arr.size() != 2)
    return std::nullopt;
  auto g = dyn_cast<IntegerAttr>(arr[0]);
  auto r = dyn_cast<IntegerAttr>(arr[1]);
  if (!g || !r)
    return std::nullopt;
  return std::make_pair(static_cast<int>(g.getInt()),
                        static_cast<int>(r.getInt()));
}

// Compute iteration count = (iv - lb) / step, cast to i32. Mirrors
// AddMultiBufferInnerScope::getIterCount so the iter counter logic stays
// consistent across the pipeline.
static Value computeIterCount(OpBuilder &builder, scf::ForOp forOp,
                              SmallVector<Operation *> *newOps) {
  Location loc = forOp.getLoc();
  auto i32Type = builder.getI32Type();
  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();

  bool lbIsZero = false;
  if (auto constOp = lb.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      lbIsZero = (intAttr.getInt() == 0);

  Value iterIdx;
  if (lbIsZero) {
    bool stepIsOne = false;
    if (auto constOp = step.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        stepIsOne = (intAttr.getInt() == 1);
    if (stepIsOne) {
      iterIdx = iv;
    } else {
      iterIdx = builder.create<arith::DivUIOp>(loc, iv, step);
      if (newOps)
        newOps->push_back(iterIdx.getDefiningOp());
    }
  } else {
    Value diff = builder.create<arith::SubIOp>(loc, iv, lb);
    iterIdx = builder.create<arith::DivUIOp>(loc, diff, step);
    if (newOps) {
      newOps->push_back(diff.getDefiningOp());
      newOps->push_back(iterIdx.getDefiningOp());
    }
  }

  if (iv.getType() == i32Type)
    return iterIdx;
  if (iv.getType().isIndex()) {
    Value cast = builder.create<arith::IndexCastOp>(loc, i32Type, iterIdx);
    if (newOps)
      newOps->push_back(cast.getDefiningOp());
    return cast;
  }
  return iterIdx;
}

// Compute buffer index = iterCount % N. Mirrors AddMultiBufferInnerScope's
// computeBufferIndex.
static Value computeBufferIndex(OpBuilder &builder, scf::ForOp forOp,
                                int N, SmallVector<Operation *> *newOps) {
  Location loc = forOp.getLoc();
  Value iterCount = computeIterCount(builder, forOp, newOps);
  Value Nval = builder.create<arith::ConstantIntOp>(loc, N, 32);
  Value rem = builder.create<arith::RemSIOp>(loc, iterCount, Nval);
  if (newOps) {
    newOps->push_back(Nval.getDefiningOp());
    newOps->push_back(rem.getDefiningOp());
  }
  return rem;
}

// Walk the forOp's region recursively and collect every scf::ForOp that
// carries (or whose terminator carries) `ssbuffer.main_loop`. Mirrors
// AddMultiBufferInnerScope::collectMainLoopsRecursively.
static void collectMainLoopsInBlock(
    Block &block, SmallVector<scf::ForOp> &mainLoopForOps) {
  for (Operation &op : block) {
    auto forOp = dyn_cast<scf::ForOp>(&op);
    if (!forOp)
      continue;
    bool isMain = forOp->hasAttr(CVPipeline::kMainLoop);
    if (!isMain) {
      if (Operation *term = forOp.getBody()->getTerminator())
        isMain = term->hasAttr(CVPipeline::kMainLoop);
    }
    if (isMain)
      mainLoopForOps.push_back(forOp);
  }
}

static int collectMainLoopsRecursively(Region &region,
                                      SmallVector<scf::ForOp> &mainLoopForOps) {
  int total = 0;
  for (Block &block : region) {
    size_t before = mainLoopForOps.size();
    collectMainLoopsInBlock(block, mainLoopForOps);
    total += (mainLoopForOps.size() - before);
    for (Operation &op : block) {
      for (Region &nested : op.getRegions())
        total += collectMainLoopsRecursively(nested, mainLoopForOps);
    }
  }
  return total;
}

// Determine the buffer Value associated with the given op+role.
//   - role=1 on hivm::FixpipeOp → outs operand (last operand)
//   - role=0 on memref::MemorySpaceCastOp → input operand
//   - other combinations → fall back to the first memref operand
// Returns nullptr if no suitable operand is found.
static Value getOpBufferOperand(Operation *op, int role) {
  if (auto fixpipe = dyn_cast<hivm::FixpipeOp>(op)) {
    return fixpipe.getDst();
  }
  if (auto memcast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
    return memcast.getSource();
  }
  // Fallback: first memref-typed operand.
  for (OpOperand &operand : op->getOpOperands()) {
    if (isa<MemRefType>(operand.get().getType()))
      return operand.get();
  }
  return nullptr;
}

// Build an scf.if chain selecting between two buffer Values based on the
// iteration index (`idx`). For N==2, a single scf.if is emitted (then/else).
// For N>2, a nested if-else chain is built (mirrors buildIfChain from
// AddMultiBufferInnerScope). The returned scf.if:
//   - has `buffers` results type when `hasResult` is true (consumer chain
//     returns the selected buffer value);
//   - has no results when `hasResult` is false (producer fixpipe has no
//     result, just emits the wrapped op).
// `cloneFn(b, buffer)` must clone the wrapped op into the given builder
// using `buffer` as its replacement target. The clone's results (when
// `hasResult` is true) are yielded from the if-branch.
static scf::IfOp buildBufferSelectIf(
    OpBuilder &builder, Location loc, Value idx,
    ArrayRef<Value> buffers, bool hasResult, Type resultType,
    function_ref<Operation *(OpBuilder &, Location, Value)> cloneFn) {
  int N = static_cast<int>(buffers.size());

  auto buildBranch = [&](OpBuilder &b, Value buffer) -> Operation * {
    return cloneFn(b, loc, buffer);
  };

  // Result type defaults to the buffer type when caller doesn't specify one.
  Type ifResultType = resultType ? resultType : buffers.front().getType();

  if (N == 2) {
    // then-branch: idx == 0 → buffer[0]
    // else-branch: otherwise → buffer[1]
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Value cond =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idx, zero);
    SmallVector<Type, 1> resultTypesBuf;
    if (hasResult)
      resultTypesBuf.push_back(ifResultType);
    TypeRange resultTypes(resultTypesBuf);
    auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, cond, /*addThen=*/true, /*addElse=*/true);

    // then
    {
      OpBuilder tb(builder.getContext());
      tb.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Operation *cloned = buildBranch(tb, buffers[0]);
      if (hasResult && cloned)
        tb.create<scf::YieldOp>(loc, cloned->getResult(0));
      else
        tb.create<scf::YieldOp>(loc);
    }
    // else
    {
      OpBuilder eb(builder.getContext());
      eb.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Operation *cloned = buildBranch(eb, buffers[1]);
      if (hasResult && cloned)
        eb.create<scf::YieldOp>(loc, cloned->getResult(0));
      else
        eb.create<scf::YieldOp>(loc);
    }
    return ifOp;
  }

  // N > 2: nested if-else-if chain
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value rootCond = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, idx, zero);
  SmallVector<Type, 1> resultTypesBuf;
  if (hasResult)
    resultTypesBuf.push_back(ifResultType);
  TypeRange resultTypes(resultTypesBuf);
  auto rootIf = builder.create<scf::IfOp>(loc, resultTypes, rootCond, /*addThen=*/true, /*addElse=*/true);

  // then: idx == 0 → buffer[0]
  {
    OpBuilder tb(builder.getContext());
    tb.setInsertionPointToStart(&rootIf.getThenRegion().front());
    Operation *cloned = buildBranch(tb, buffers[0]);
    if (hasResult && cloned)
      tb.create<scf::YieldOp>(loc, cloned->getResult(0));
    else
      tb.create<scf::YieldOp>(loc);
  }

  // Build the else chain incrementally.
  Block *currentElse = &rootIf.getElseRegion().front();
  for (int i = 1; i < N - 1; ++i) {
    OpBuilder cb(builder.getContext());
    cb.setInsertionPoint(currentElse, currentElse->end());
    Value iVal = cb.create<arith::ConstantIntOp>(loc, i, 32);
    Value cond = cb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, idx,
                                         iVal);
    auto nestedIf = cb.create<scf::IfOp>(loc, resultTypes, cond, /*addThen=*/true, /*addElse=*/true);
    // then → buffer[i]
    {
      OpBuilder tb(builder.getContext());
      tb.setInsertionPointToStart(&nestedIf.getThenRegion().front());
      Operation *cloned = buildBranch(tb, buffers[i]);
      if (hasResult && cloned)
        tb.create<scf::YieldOp>(loc, cloned->getResult(0));
      else
        tb.create<scf::YieldOp>(loc);
    }
    // current else-block yields the nestedIf's results
    cb.create<scf::YieldOp>(loc, nestedIf.getResults());
    // Move on to the nested else for the next iteration
    currentElse = &nestedIf.getElseRegion().front();
  }

  // Final else: buffer[N-1]
  {
    OpBuilder fb(builder.getContext());
    fb.setInsertionPoint(currentElse, currentElse->end());
    Operation *cloned = buildBranch(fb, buffers[N - 1]);
    if (hasResult && cloned)
      fb.create<scf::YieldOp>(loc, cloned->getResult(0));
    else
      fb.create<scf::YieldOp>(loc);
  }

  return rootIf;
}

// For each (groupId, buffer) pair, allocate N memref.alloc buffers shaped
// like the shared original buffer, then wrap the producer (writes the
// buffer) and consumer (reads the buffer) with an scf.if chain selecting
// buffer[iter % N]. Returns 0 on success.
static int processCubeBufferGroup(ModuleOp module, scf::ForOp mainLoop,
                                  Operation *producer, Operation *consumer,
                                  int N) {
  Value prodBuf = getOpBufferOperand(producer, 1);
  Value consBuf = getOpBufferOperand(consumer, 0);
  if (!prodBuf || !consBuf) {
    LDBG("Could not extract buffer operand from producer/consumer. Skip.");
    return 0;
  }
  if (prodBuf != consBuf) {
    LDBG("Producer and consumer reference different buffers; skip group.");
    return 0;
  }

  // The buffer must be a memref defined by a memref.alloc (or its
  // memory_space_cast descendant) so we know its shape and address space.
  Operation *allocOp = prodBuf.getDefiningOp();
  if (!allocOp) {
    LDBG("Buffer is not produced by an op (block argument); skip.");
    return 0;
  }
  Value origMemref;
  if (auto alloc = dyn_cast<memref::AllocOp>(allocOp)) {
    origMemref = alloc.getResult();
  } else if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(allocOp)) {
    origMemref = cast.getResult();
  } else {
    LDBG("Buffer is not a memref.alloc or memory_space_cast; skip.");
    return 0;
  }

  auto origType = dyn_cast<MemRefType>(origMemref.getType());
  if (!origType) {
    LDBG("Buffer type is not MemRefType; skip.");
    return 0;
  }

  // Insert N new memref.alloc ops at the start of the main_loop body, with
  // the same shape and address space as the original buffer.
  Block *body = mainLoop.getBody();
  OpBuilder allocBuilder(body, body->begin());
  SmallVector<Value> buffers;
  buffers.reserve(N);
  for (int i = 0; i < N; ++i) {
    auto newAlloc = allocBuilder.create<memref::AllocOp>(
        mainLoop.getLoc(), origType);
    // Mirror the original op's block_id (if any) so downstream passes that
    // inspect block_id continue to work.
    if (auto origId = allocOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId))
      newAlloc->setAttr(CVPipeline::kBlockId, origId);
    buffers.push_back(newAlloc.getResult());
  }

  // Compute producer's own iter counter (inserted right before producer).
  // Producer and consumer each get their own counter — independent SSA values,
  // even though the underlying computation is identical. This leaves room for
  // future phase-offset tricks (e.g. consumer reads iter%N while producer
  // writes (iter+1)%N) without rewiring producers/consumers apart.
  OpBuilder prodIdxBuilder(producer);
  prodIdxBuilder.setInsertionPoint(producer);
  SmallVector<Operation *> prodIterOps;
  Value prodIdx = computeBufferIndex(prodIdxBuilder, mainLoop, N, &prodIterOps);
  if (auto origId = allocOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
    for (Operation *op : prodIterOps)
      op->setAttr(CVPipeline::kBlockId, origId);
  }

  // Wrap the producer: clone the producer op in each if-branch, replacing its
  // buffer operand with buffers[i]. The producer has no result.
  auto wrapProducer = [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
    IRMapping map;
    map.map(prodBuf, buffer);
    return b.clone(*producer, map);
  };
  OpBuilder prodBuilder(producer);
  prodBuilder.setInsertionPoint(producer);
  scf::IfOp prodIf =
      buildBufferSelectIf(prodBuilder, producer->getLoc(), prodIdx, buffers,
                          /*hasResult=*/false, /*resultType=*/Type(), wrapProducer);
  // Tag the producer if with the original block_id.
  if (auto origId = allocOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId))
    prodIf->setAttr(CVPipeline::kBlockId, origId);
  producer->erase();

  // Compute consumer's own iter counter (inserted right before consumer).
  OpBuilder consIdxBuilder(consumer);
  consIdxBuilder.setInsertionPoint(consumer);
  SmallVector<Operation *> consIterOps;
  Value consIdx = computeBufferIndex(consIdxBuilder, mainLoop, N, &consIterOps);
  if (auto origId = allocOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
    for (Operation *op : consIterOps)
      op->setAttr(CVPipeline::kBlockId, origId);
  }

  // Wrap the consumer: clone the consumer op in each if-branch, replacing its
  // buffer operand with buffers[i]. The consumer has one result that gets
  // yielded and used as the if's result.
  auto wrapConsumer = [&](OpBuilder &b, Location l, Value buffer) -> Operation * {
    IRMapping map;
    map.map(consBuf, buffer);
    return b.clone(*consumer, map);
  };
  OpBuilder consBuilder(consumer);
  consBuilder.setInsertionPoint(consumer);
  scf::IfOp consIf =
      buildBufferSelectIf(consBuilder, consumer->getLoc(), consIdx, buffers,
                          /*hasResult=*/true,
                          /*resultType=*/consumer->getResult(0).getType(),
                          wrapConsumer);
  if (auto origId = allocOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId))
    consIf->setAttr(CVPipeline::kBlockId, origId);
  // Rewire downstream uses of consumer's result to the new if's result.
  Value consumerResult = consumer->getResult(0);
  consumerResult.replaceAllUsesWith(consIf.getResult(0));
  consumer->erase();

  // Erase the original alloc (and its memory_space_cast if any) — its users
  // have all been rewired to the new buffers.
  if (allocOp->use_empty()) {
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(allocOp)) {
      Operation *innerAlloc = cast.getSource().getDefiningOp();
      cast->erase();
      if (innerAlloc && innerAlloc->use_empty())
        innerAlloc->erase();
    } else {
      allocOp->erase();
    }
  }

  LDBG("Group processed: " << N << " buffers created, producer/consumer "
                            << "wrapped with scf.if.");
  return 0;
}

// Process a single CUBE scope's main_loop. Walks all ops in the loop body
// and groups them by ssbuffer.cubeBuffer's group id. Within each group we
// pair producers and consumers by their buffer operand (not just by role),
// so a (fixpipe, memref.memory_space_cast) pair sharing the same buffer is
// treated as one chain — even if both ops carry the same role label.
static int processMainLoop(ModuleOp module, scf::ForOp mainLoop) {
  int N = getCubeBufCount(module);
  if (N < 2) {
    LDBG("cube_buf_count=" << N << " (<2); skip multi-buffer for this loop.");
    return 0;
  }

  // Collect ops with cubeBuffer; group them by (group id, buffer operand).
  // Each entry holds the ops touching the same buffer; we then split them
  // into producer (fixpipe) / consumer (memspace_cast) by op kind.
  llvm::MapVector<int, llvm::MapVector<Value, SmallVector<Operation *, 2>>>
      groupToBuffers;
  mainLoop.walk([&](Operation *op) {
    auto info = getCubeBufferGroup(op);
    if (!info)
      return;
    auto [groupId, role] = *info;
    (void)role; // role is informational; pairing is by buffer + op kind.
    Value buf = getOpBufferOperand(op, /*role=*/1);
    if (!buf)
      buf = getOpBufferOperand(op, /*role=*/0);
    if (!buf)
      return;
    groupToBuffers[groupId][buf].push_back(op);
  });

  // For each (groupId, buffer), build (producers, consumers) by op kind.
  llvm::MapVector<int, std::pair<SmallVector<Operation *>,
                                 SmallVector<Operation *>>>
      groups;
  for (auto &ge : groupToBuffers) {
    int groupId = ge.first;
    for (auto &be : ge.second) {
      SmallVector<Operation *, 2> &ops = be.second;
      SmallVector<Operation *> prods, cons;
      for (Operation *op : ops) {
        if (isa<hivm::FixpipeOp>(op))
          prods.push_back(op);
        else if (isa<memref::MemorySpaceCastOp>(op))
          cons.push_back(op);
      }
      auto &entry = groups[groupId];
      entry.first.append(prods.begin(), prods.end());
      entry.second.append(cons.begin(), cons.end());
    }
  }

  if (groups.empty()) {
    LDBG("No ssbuffer.cubeBuffer-tagged ops in main_loop; nothing to do.");
    return 0;
  }

  // Flatten to per-(group, buffer, producer, consumer) tuples and process.
  LDBG("Found " << groups.size() << " cubeBuffer group(s); N=" << N);
  for (auto &ge : groups) {
    int groupId = ge.first;
    auto &podsAndCons = ge.second;
    // Re-derive per-buffer pairing from the joined lists: walk the producer
    // list and try to match each fixpipe with a memspacecast by buffer.
    SmallVector<Operation *> &prods = podsAndCons.first;
    SmallVector<Operation *> &cons = podsAndCons.second;
    for (Operation *p : prods) {
      Value pBuf = getOpBufferOperand(p, 1);
      if (!pBuf)
        continue;
      Operation *matched = nullptr;
      for (Operation *c : cons) {
        Value cBuf = getOpBufferOperand(c, 0);
        if (cBuf == pBuf) {
          matched = c;
          break;
        }
      }
      if (!matched) {
        LDBG("No consumer matches producer in group " << groupId << "; skip.");
        continue;
      }
      if (processCubeBufferGroup(module, mainLoop, p, matched, N) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

// ---- Pass entry ------------------------------------------------------------

void AddMultiBufferCubeScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::hivm::HIVMDialect,
                  mlir::scope::ScopeDialect>();
}

void AddMultiBufferCubeScopePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpBuilder builder(module.getContext());
  LDBG("Enter pass.");

  auto walkResult = module.walk([&](scope::ScopeOp scope) -> WalkResult {
    auto coreTypeAttr =
        scope->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      return WalkResult::advance();
    hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
    if (coreType != hivm::TCoreType::CUBE) {
      LDBG("Skipping non-CUBE scope.");
      return WalkResult::advance();
    }

    SmallVector<scf::ForOp> mainLoopForOps;
    collectMainLoopsRecursively(scope.getBodyRegion(), mainLoopForOps);
    if (mainLoopForOps.empty()) {
      LDBG("No main_loop forOp in CUBE scope.");
      return WalkResult::advance();
    }

    for (scf::ForOp mainLoop : mainLoopForOps) {
      if (processMainLoop(module, mainLoop) != 0) {
        LDBG("processMainLoop failed; signaling failure.");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("Process successfully.");
}

std::unique_ptr<OperationPass<ModuleOp>>
createAddMultiBufferCubeScopePass() {
  return std::make_unique<AddMultiBufferCubeScopePass>();
}

void registerAddMultiBufferCubeScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddMultiBufferCubeScopePass();
  });
}

} // namespace triton
} // namespace mlir