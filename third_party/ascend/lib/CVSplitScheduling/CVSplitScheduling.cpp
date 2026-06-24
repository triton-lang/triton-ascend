#include "ascend/include/CVSplitScheduling/CVSplitScheduling.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <functional>

#define DEBUG_TYPE "cv-split-scheduling"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

// ============================================================================
// CV-Split Scheduling
// ----------------------------------------------------------------------------
// Splits the innermost loop of a fused kernel into two co-running engine scopes
// — a CUBE scope (matmul / fixpipe) and a VECTOR scope (elementwise / softmax) —
// with explicit cross-engine buffers and synchronization, so the Ascend cube
// and vector units overlap instead of running serially.
//
// Pipeline (driven by CVSplitSchedulingPass::processFunction):
//   1.  findInnermostLoop            locate the loop to split
//   1b. hasStoresInBody              bail if the body already stores (not fusable)
//   2.  loopUnrollByFactor           unroll by `unroll-factor` to expose ILP
//   3.  classifyAllOps               tag each op CUBE or VECTOR (matmul-seeded,
//                                    data-feeders pulled into CUBE, rest VECTOR)
//   4-7 DependencyLevelScheduler     graph -> BFS levels -> purity check ->
//                                    reorder so same-engine work is contiguous
//   7.5 unfusePVMatmuls              undo matmul(p,v,acc) fusion that entangles
//                                    the engines
//   8.  insertCrossScopeTransfers    materialize C->V (fixpipe->UB) and V->C
//                                    (NZ pack->L1) buffers + sync_block_set/wait
//   9.  createScopeSeparation        clone CUBE ops into a CUBE scope, wrap the
//                                    loop+epilogue in a VECTOR scope, strip the
//                                    wrong-engine ops from each, then ROW_SPLIT
//                                    re-tile the VECTOR scope across both veccores
//
// Generality: the pass is engine-pattern driven, not kernel-name driven — it
// keys off op semantics (matmul == CUBE, float elementwise == VECTOR) and bails
// out cleanly (leaving the IR untouched) whenever an assumption does not hold:
// no innermost loop, stores already present, unroll-factor <= 1, no CUBE ops, or
// CUBE/VECTOR work that the level scheduler finds entangled. Flash-Attention is
// the validated driver kernel; other fused cube+vector loops that satisfy the
// same structural contract are handled by the same path, and anything else
// falls through unmodified.
// ============================================================================

namespace {

enum class EngineType { CUBE, VECTOR, UNKNOWN };

static StringRef engineTypeToStr(EngineType e) {
  switch (e) {
  case EngineType::CUBE:    return "CUBE";
  case EngineType::VECTOR:  return "VECTOR";
  default:                  return "UNKNOWN";
  }
}

// ============================================================================
// Stage 1: Find innermost loop
// ============================================================================
static scf::ForOp findInnermostLoop(func::FuncOp funcOp) {
  scf::ForOp innermost = nullptr;
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedFor = false;
    forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
    if (!hasNestedFor)
      innermost = forOp;
  });
  return innermost;
}

// ============================================================================
// Stage 1b: Bail-out check — no stores in loop body
// ============================================================================
static bool hasStoresInBody(scf::ForOp forOp) {
  bool found = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (isa<memref::StoreOp>(op) || isa<tensor::InsertSliceOp>(op) ||
        isa<bufferization::MaterializeInDestinationOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    if (op->getName().getStringRef().contains("store") &&
        !isa<scf::YieldOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// ============================================================================
// Stage 3: Engine classification
// ============================================================================
static EngineType classifyOp(Operation *op,
                             DenseMap<Operation *, EngineType> &cache) {
  auto it = cache.find(op);
  if (it != cache.end())
    return it->second;

  if (isa<linalg::MatmulOp>(op) || isa<linalg::MatmulTransposeBOp>(op) ||
      isa<linalg::BatchMatmulOp>(op)) {
    cache[op] = EngineType::CUBE;
    return EngineType::CUBE;
  }

  StringRef opName = op->getName().getStringRef();
  if (opName.contains("dot") || opName.contains("matmul")) {
    cache[op] = EngineType::CUBE;
    return EngineType::CUBE;
  }

  if (isa<hivm::FixpipeOp>(op)) {
    cache[op] = EngineType::CUBE;
    return EngineType::CUBE;
  }

  if (auto syncSetOp = dyn_cast<hivm::SyncBlockSetOp>(op)) {
    auto coreType = syncSetOp.getTcoreType().getTcoretype();
    cache[op] = coreType == hivm::TCoreType::CUBE ? EngineType::CUBE : EngineType::VECTOR;
    return cache[op];
  }
  if (auto syncWaitOp = dyn_cast<hivm::SyncBlockWaitOp>(op)) {
    auto coreType = syncWaitOp.getTcoreType().getTcoretype();
    cache[op] = coreType == hivm::TCoreType::CUBE ? EngineType::CUBE : EngineType::VECTOR;
    return cache[op];
  }
  if (isa<hivm::CopyOp>(op)) {
    cache[op] = EngineType::VECTOR;
    return EngineType::VECTOR;
  }
  if (isa<hivm::ConvertLayoutOp>(op)) {
    cache[op] = EngineType::CUBE;
    return EngineType::CUBE;
  }

  cache[op] = EngineType::VECTOR;
  return EngineType::VECTOR;
}

static void classifyAllOps(Block *body,
                           DenseMap<Operation *, EngineType> &classification) {
  SmallVector<Operation *> cubeSeeds;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    EngineType ty = classifyOp(&op, classification);
    if (ty == EngineType::CUBE)
      cubeSeeds.push_back(&op);
  }

  DenseSet<Operation *> visited;
  std::queue<Operation *> worklist;
  for (auto *seed : cubeSeeds)
    worklist.push(seed);

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop();
    if (!visited.insert(op).second)
      continue;

    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != body || isa<scf::YieldOp>(defOp))
        continue;

      bool isDataFeeder =
          isa<bufferization::ToTensorOp>(defOp) ||
          isa<linalg::TransposeOp>(defOp) ||
          isa<linalg::FillOp>(defOp) ||
          isa<memref::AllocOp>(defOp) ||
          isa<memref::CopyOp>(defOp) ||
          isa<memref::SubViewOp>(defOp) ||
          isa<memref::ReinterpretCastOp>(defOp) ||
          isa<memref::MemorySpaceCastOp>(defOp) ||
          isa<tensor::ExtractSliceOp>(defOp) ||
          defOp->getName().getStringRef().contains("transpose") ||
          defOp->getName().getStringRef().contains("to_tensor") ||
          defOp->getName().getStringRef().contains("convert_layout");

      if (isDataFeeder && classification[defOp] != EngineType::CUBE) {
        classification[defOp] = EngineType::CUBE;
        worklist.push(defOp);
      }
    }
  }

  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    if (classification.find(&op) == classification.end())
      classification[&op] = EngineType::VECTOR;
  }
}

// Diagnostic: log the per-op CUBE/VECTOR classification and return the CUBE op
// count (the pass bails when there are no CUBE ops to separate).
static int logClassification(
    Block *body, const DenseMap<Operation *, EngineType> &classification) {
  int nCube = 0, nVec = 0;
  for (auto &kv : classification) {
    if (kv.second == EngineType::CUBE) ++nCube;
    else ++nVec;
  }
  llvm::errs() << "[cv-split] Classification: " << nCube << "C " << nVec << "V\n";
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op)) continue;
    auto it = classification.find(&op);
    llvm::errs() << "[cv-split]   "
                 << (it != classification.end()
                     ? engineTypeToStr(it->second) : "??")
                 << " " << op.getName() << "\n";
  }
  return nCube;
}

// ============================================================================
// Stage 4: Dependency graph
// ============================================================================
static void buildDependencyGraph(
    Block *body,
    DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    DenseMap<Operation *, SmallVector<Operation *>> &successors) {
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    for (Value operand : op.getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != body || isa<scf::YieldOp>(defOp))
        continue;
      predecessors[&op].push_back(defOp);
      successors[defOp].push_back(&op);
    }
  }

  // Add memory dependency edges: memref.copy → bufferization.to_tensor
  // memref.copy writes to an alloc via side effect (no SSA result).
  // to_tensor reads from the same alloc. Without this edge, BFS leveling
  // can place to_tensor BEFORE copy, causing reads of uninitialized data.
  for (Operation &op : *body) {
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    if (!copyOp) continue;
    Value dst = copyOp.getTarget();
    for (Operation *user : dst.getUsers()) {
      if (user == &op || user->getBlock() != body) continue;
      if (isa<bufferization::ToTensorOp>(user)) {
        predecessors[user].push_back(&op);
        successors[&op].push_back(user);
      }
    }
  }

  // Similarly: memref.copy also depends on reinterpret_cast of its SOURCE.
  // The source reinterpret_cast depends on address arithmetic that computes
  // the GM offset. Ensure copy is ordered after its source address is ready.
  // (This is already captured by SSA edges since copy USES the source value.)
}

// ============================================================================
// Stage 5: BFS levelization
// ============================================================================
static SmallVector<Operation *> collectRoots(
    Block *body,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  SmallVector<Operation *> roots;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    auto it = predecessors.find(&op);
    if (it == predecessors.end() || it->second.empty())
      roots.push_back(&op);
  }
  return roots;
}

static int bfsLevelize(
    Block *body,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    const SmallVector<Operation *> &roots,
    DenseMap<Operation *, int> &levels) {
  for (auto *root : roots)
    levels[root] = 0;

  int maxLevel = 0;

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op))
        continue;

      auto it = predecessors.find(&op);
      if (it == predecessors.end()) {
        if (!levels.count(&op)) {
          levels[&op] = 0;
          changed = true;
        }
        continue;
      }

      int requiredLevel = 0;
      for (auto *pred : it->second) {
        auto predIt = levels.find(pred);
        if (predIt != levels.end())
          requiredLevel = std::max(requiredLevel, predIt->second + 1);
      }

      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end() || requiredLevel > lvlIt->second) {
        levels[&op] = requiredLevel;
        maxLevel = std::max(maxLevel, requiredLevel);
        changed = true;
      }
    }
  }

  return maxLevel;
}

// ============================================================================
// Stage 6: Level purity check
// ============================================================================
static bool checkLevelPurity(
    Block *body,
    const DenseMap<Operation *, int> &levels,
    const DenseMap<Operation *, EngineType> &classification,
    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    int maxLevel) {
  for (int lvl = 0; lvl <= maxLevel; ++lvl) {
    SmallVector<Operation *> cubeOps, vectorOps;
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(&op)) continue;
      auto lvlIt = levels.find(&op);
      if (lvlIt == levels.end() || lvlIt->second != lvl) continue;
      auto classIt = classification.find(&op);
      if (classIt == classification.end()) continue;
      if (classIt->second == EngineType::CUBE)
        cubeOps.push_back(&op);
      else
        vectorOps.push_back(&op);
    }

    if (cubeOps.empty() || vectorOps.empty())
      continue;

    llvm::errs() << "[cv-split] Level " << lvl << " mixed ("
                 << cubeOps.size() << "C, " << vectorOps.size() << "V)\n";

    DenseSet<Operation *> cubeSet(cubeOps.begin(), cubeOps.end());
    DenseSet<Operation *> vectorSet(vectorOps.begin(), vectorOps.end());

    for (auto *cOp : cubeOps) {
      auto predIt = predecessors.find(cOp);
      if (predIt == predecessors.end()) continue;
      for (auto *pred : predIt->second) {
        if (vectorSet.count(pred)) {
          llvm::errs() << "[cv-split] Level " << lvl
                       << ": CUBE depends on VECTOR -- bail\n";
          return false;
        }
      }
    }
    for (auto *vOp : vectorOps) {
      auto predIt = predecessors.find(vOp);
      if (predIt == predecessors.end()) continue;
      for (auto *pred : predIt->second) {
        if (cubeSet.count(pred)) {
          llvm::errs() << "[cv-split] Level " << lvl
                       << ": VECTOR depends on CUBE -- bail\n";
          return false;
        }
      }
    }
  }
  return true;
}

// ============================================================================
// Stage 7: Reorder by BFS level
// ============================================================================
static void reorderByLevel(Block *body,
                           const DenseMap<Operation *, int> &levels) {
  SmallVector<Operation *> ops;
  for (Operation &op : *body) {
    if (!isa<scf::YieldOp>(&op))
      ops.push_back(&op);
  }

  llvm::stable_sort(ops, [&](Operation *a, Operation *b) {
    int la = 0, lb = 0;
    auto itA = levels.find(a);
    if (itA != levels.end()) la = itA->second;
    auto itB = levels.find(b);
    if (itB != levels.end()) lb = itB->second;
    return la < lb;
  });

  Operation *yield = body->getTerminator();
  for (auto *op : ops)
    op->moveBefore(yield);
}

// ============================================================================
// Dependency-level scheduler
// ----------------------------------------------------------------------------
// Orchestrates stages 4-7 over an (already unrolled) loop body:
//   1. build a def->use dependency graph (SSA edges + memref.copy->to_tensor
//      memory edges),
//   2. assign every op a BFS "level" = longest dependency depth from a root,
//   3. verify the levels are cleanly separable (no level contains a CUBE op and
//      a VECTOR op that depend on each other),
//   4. reorder the body by level so same-engine work is grouped, ready to be
//      split into a CUBE scope and a VECTOR scope.
// run() returns false (leaving the body untouched) when the work is entangled
// and cannot be cleanly separated, so the caller can bail safely.
// ============================================================================
struct DependencyLevelScheduler {
  DenseMap<Operation *, SmallVector<Operation *>> predecessors;
  DenseMap<Operation *, SmallVector<Operation *>> successors;
  DenseMap<Operation *, int> levels;
  int maxLevel = 0;

  bool run(Block *body,
           const DenseMap<Operation *, EngineType> &classification) {
    buildDependencyGraph(body, predecessors, successors);

    SmallVector<Operation *> roots = collectRoots(body, predecessors);
    llvm::errs() << "[cv-split] " << roots.size() << " roots\n";

    maxLevel = bfsLevelize(body, predecessors, roots, levels);
    llvm::errs() << "[cv-split] " << (maxLevel + 1) << " BFS levels\n";
    logLevelHistogram(body, classification);

    if (!checkLevelPurity(body, levels, classification, predecessors,
                          maxLevel)) {
      llvm::errs() << "[cv-split] Purity check failed, bail\n";
      return false;
    }
    llvm::errs() << "[cv-split] Purity OK\n";

    reorderByLevel(body, levels);
    llvm::errs() << "[cv-split] Reordered by level\n";
    return true;
  }

private:
  // Per-level CUBE/VECTOR op-count breakdown (diagnostic only).
  void logLevelHistogram(
      Block *body,
      const DenseMap<Operation *, EngineType> &classification) const {
    for (int lvl = 0; lvl <= maxLevel; ++lvl) {
      int nC = 0, nV = 0;
      for (Operation &op : *body) {
        if (isa<scf::YieldOp>(&op)) continue;
        auto lvlIt = levels.find(&op);
        if (lvlIt == levels.end() || lvlIt->second != lvl) continue;
        auto clsIt = classification.find(&op);
        if (clsIt == classification.end()) continue;
        if (clsIt->second == EngineType::CUBE) ++nC; else ++nV;
      }
      llvm::errs() << "[cv-split]   L" << lvl << ": "
                   << nC << "C " << nV << "V\n";
    }
  }
};

// ============================================================================
// Stage 8: Insert cross-scope transfers and synchronization
//
// For each SSA value that crosses from CUBE→VECTOR or VECTOR→CUBE:
//   C→V: alloc UB buffer, insert fixpipe + sync_block_set/wait, replace uses
//   V→C: alloc L1 buffer (NZ), insert copy + sync_block_set/wait, replace uses
//
// This runs BEFORE scope separation so both scopes see the shared buffers.
// ============================================================================

struct CrossScopeTransfer {
  Value value;
  Operation *producer;
  SmallVector<Operation *> consumers;
  enum Direction { CUBE_TO_VECTOR, VECTOR_TO_CUBE } direction;
};

static SmallVector<CrossScopeTransfer> findCrossScopeValues(
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {
  SmallVector<CrossScopeTransfer> transfers;

  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    auto prodIt = classification.find(&op);
    if (prodIt == classification.end())
      continue;
    EngineType prodType = prodIt->second;

    // C→V: Only transfer results of linalg.matmul (QK and PV dot products)
    // V→C: Only transfer values that DIRECTLY feed into linalg.matmul as operands
    //       (these are the P values after softmax+cast)
    //
    // Reference pattern for K=4:
    //   4× QK matmul results (C→V, fixpipe, flags 0-3)
    //   4× P inputs to PV matmul (V→C, copy UB→L1, flags 4-7)
    //   4× PV matmul results (C→V, fixpipe, flags 8-11)

    if (prodType == EngineType::CUBE && isa<linalg::MatmulOp>(&op)) {
      // C→V: matmul result consumed by VECTOR ops
      for (Value result : op.getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        SmallVector<Operation *> crossUsers;
        for (Operation *user : result.getUsers()) {
          if (user->getBlock() != body) continue;
          if (isa<scf::YieldOp>(user)) continue;
          auto consIt = classification.find(user);
          if (consIt == classification.end()) continue;
          if (consIt->second == EngineType::VECTOR)
            crossUsers.push_back(user);
        }
        if (!crossUsers.empty())
          transfers.push_back({result, &op, crossUsers,
                               CrossScopeTransfer::CUBE_TO_VECTOR});
      }
    } else if (prodType == EngineType::VECTOR) {
      // V→C: only if a VECTOR result feeds linalg.matmul as LHS (operand 0) or RHS (operand 1)
      // NOT operand 2 (the output/accumulator init)
      for (Value result : op.getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        SmallVector<Operation *> crossUsers;
        for (Operation *user : result.getUsers()) {
          if (user->getBlock() != body) continue;
          if (!isa<linalg::MatmulOp>(user)) continue;
          auto consIt = classification.find(user);
          if (consIt == classification.end()) continue;
          if (consIt->second != EngineType::CUBE) continue;
          // Check it's operand 0 or 1 (LHS/RHS), not 2 (init/accumulator)
          for (unsigned i = 0; i < 2; ++i) {
            if (user->getOperand(i) == result) {
              crossUsers.push_back(user);
              break;
            }
          }
        }
        if (!crossUsers.empty())
          transfers.push_back({result, &op, crossUsers,
                               CrossScopeTransfer::VECTOR_TO_CUBE});
      }
    }
  }
  return transfers;
}

// Per-pass attribute bundle shared by the transfer emitters: the core-type and
// pipe attributes are constant for the whole pass, and `loop` is where the
// shared buffers are allocated (just before the inner loop).
struct TransferEmitContext {
  MLIRContext *ctx;
  Location loc;
  scf::ForOp loop;
  hivm::TCoreTypeAttr cubeCoreAttr;
  hivm::TCoreTypeAttr vecCoreAttr;
  hivm::PipeAttr pipeFixAttr;
  hivm::PipeAttr pipeVAttr;
  hivm::PipeAttr pipeMte3Attr;
  hivm::PipeAttr pipeMte1Attr;
};

// alloc + annotation.mark{effects=["write","read"]}. One shared buffer per
// transfer, like the reference FA kernel. (The target IR carries only
// `effects`; the hivm.tightly_coupled_buffer<N> attribute is intentionally
// omitted.)
static memref::AllocOp createAnnotatedAlloc(OpBuilder &builder, Location loc,
                                            MemRefType allocType) {
  auto allocOp = builder.create<memref::AllocOp>(loc, allocType);
  auto markOp = builder.create<annotation::MarkOp>(loc, allocOp.getResult());
  auto writeAttr = builder.getStringAttr("write");
  auto readAttr = builder.getStringAttr("read");
  markOp->setAttr("effects", builder.getArrayAttr({writeAttr, readAttr}));
  return allocOp;
}

// Depth-2 ping/pong buffer pool. Transfers that share an identical buffer type
// (e.g. all the unrolled qk_ub fixpipe targets, or all the P L1 packs) reuse a
// rotating set of `depth` physical allocations instead of one fresh buffer per
// unrolled stage. This mirrors the manual kernel's qk_ub_0/1, pv_ub_0/1,
// p_l1_0/1 double buffering and keeps peak UB/L1 bounded so unroll>=2 fits in
// the 248 KB UB. Reuse serializes stage i and stage i+depth on the same buffer
// (WAR), which BiShengIR's GraphSyncSolver covers via the existing
// sync_block_set/wait flags — exactly the depth-2 software pipeline the manual
// kernel uses.
struct PingPongPool {
  llvm::StringMap<SmallVector<memref::AllocOp, 2>> slots;
  llvm::StringMap<unsigned> useCount;
  // One hoisted ND view (memory_space_cast of convert_layout) per L1 P buffer,
  // matching the manual kernel which emits a single convert_layout per p_l1
  // buffer before the KV loop and reuses it (fresh to_tensor per matmul).
  // Multiple convert_layout views of the same cbuf buffer confuse BiShengIR's
  // L1 NZ tracking and yield a misaligned / zero-burst matmul operand load.
  llvm::DenseMap<Operation *, Value> ndView;
  unsigned depth = 2;

  // Allocation to use for the next transfer of `allocType`: a new physical
  // buffer only while the rotating set is smaller than `depth`, otherwise the
  // round-robin reuse. `builder`'s insertion point must already be set (before
  // the loop) for the create case.
  memref::AllocOp getOrCreate(OpBuilder &builder, Location loc,
                              MemRefType allocType) {
    std::string sig;
    llvm::raw_string_ostream os(sig);
    os << allocType;
    (void)os.str();
    auto &vec = slots[sig];
    unsigned slot = useCount[sig]++ % depth;
    if (slot < vec.size())
      return vec[slot];
    auto allocOp = createAnnotatedAlloc(builder, loc, allocType);
    vec.push_back(allocOp);
    return allocOp;
  }
};

// CUBE -> VECTOR: the matmul (L0C) result is fixpipe'd to a shared UB buffer,
// CUBE signals via sync_block_set, VECTOR waits and reads it back as a tensor.
// ROW_SPLIT: the UB buffer is half height (16 rows); the fixpipe sends 16 rows
// to each veccore's private UB so both veccores stay busy (2x throughput). The
// VECTOR scope is re-tiled to 16 rows per veccore in a later stage.
static void emitCubeToVectorTransfer(const TransferEmitContext &c,
                                     CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, int flagId,
                                     PingPongPool &pool) {
  Type elemType = tensorType.getElementType();
  ArrayRef<int64_t> shape = tensorType.getShape();

  OpBuilder builder(c.ctx);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);

  auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  SmallVector<int64_t, 4> ubShape(shape.begin(), shape.end());
  bool rowSplit = (ubShape[0] % 2 == 0);
  if (rowSplit) ubShape[0] /= 2;
  auto allocType = MemRefType::get(ubShape, elemType, nullptr, ubAddrSpace);
  auto halfTensorType = RankedTensorType::get(ubShape, elemType);

  // Ping/pong shared alloc before the loop (depth-2 reuse across unrolled
  // stages instead of one buffer per transfer).
  builder.setInsertionPoint(c.loop);
  auto sharedAllocOp = pool.getOrCreate(builder, c.loc, allocType);

  // fixpipe after the producer (inside loop body) -> writes the shared buffer.
  builder.setInsertionPointAfter(xfer.producer);
  auto dmaModeAttr = hivm::FixpipeDMAModeAttr::get(c.ctx, hivm::FixpipeDMAMode::NZ2ND);
  auto dualDstAttr = hivm::FixpipeDualDstModeAttr::get(c.ctx,
      rowSplit ? hivm::FixpipeDualDstMode::ROW_SPLIT
               : hivm::FixpipeDualDstMode::NO_DUAL);
  builder.create<hivm::FixpipeOp>(c.loc, mlir::TypeRange{},
      xfer.value,                    // src (full 32-row tile from dot)
      sharedAllocOp.getResult(),     // dst (16-row shared UB alloc)
      mlir::ValueRange{}, dmaModeAttr,
      dualDstAttr, nullptr, nullptr, nullptr, mlir::ArrayAttr{}, nullptr);

  // CUBE signals VECTOR.
  builder.create<hivm::SyncBlockSetOp>(c.loc, c.cubeCoreAttr, c.pipeFixAttr, c.pipeVAttr, flagAttr);

  // Consumer side: wait + read the shared buffer back as a 16-row tensor.
  Operation *firstConsumer = xfer.consumers.front();
  for (auto *cons : xfer.consumers)
    if (cons->isBeforeInBlock(firstConsumer))
      firstConsumer = cons;
  builder.setInsertionPoint(firstConsumer);

  builder.create<hivm::SyncBlockWaitOp>(c.loc, c.vecCoreAttr, c.pipeFixAttr, c.pipeVAttr, flagAttr);

  auto plainMemrefType = MemRefType::get(ubShape, elemType);
  auto castOp = builder.create<memref::MemorySpaceCastOp>(c.loc, plainMemrefType, sharedAllocOp.getResult());
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      c.loc, halfTensorType, castOp.getResult(), /*restrict=*/true, /*writable=*/true);

  for (auto *consumer : xfer.consumers)
    consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());

  llvm::errs() << "[cv-split]   C→V transfer #" << flagId
               << ": " << xfer.producer->getName()
               << " → " << ubShape[0] << "x" << ubShape[1]
               << " UB buffer (" << (rowSplit ? "ROW_SPLIT" : "NO_DUAL") << ")\n";
}

// VECTOR -> CUBE: a softmax/cast result is NZ-packed and copied UB->L1 into a
// shared L1 buffer, VECTOR signals via sync_block_set, CUBE waits and reads it
// back through a convert_layout (NZ fractal -> ND view) for matmul consumption.
// NZ packing applies only when both dims are multiples of 16; otherwise the L1
// buffer keeps the flat [M, N] layout.
static void emitVectorToCubeTransfer(const TransferEmitContext &c,
                                     CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, int flagId,
                                     int markAllocIndex, PingPongPool &pool) {
  Type elemType = tensorType.getElementType();
  ArrayRef<int64_t> shape = tensorType.getShape();

  OpBuilder builder(c.ctx);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);

  int64_t M = shape[0];
  int64_t N = shape[1];
  auto l1AddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);

  // NZ-fractal L1 layout: ND [M, N] is stored as [N/16, M/16, 16, 16] (B16
  // fractal) when both dims are multiples of 16; otherwise fall back to flat.
  bool useNZ = (M % 16 == 0) && (N % 16 == 0);
  int64_t N16 = N / 16, M16 = M / 16;
  SmallVector<int64_t, 4> l1Shape =
      useNZ ? SmallVector<int64_t, 4>{N16, M16, 16, 16}
            : SmallVector<int64_t, 4>{M, N};
  auto l1AllocType = MemRefType::get(l1Shape, elemType, nullptr, l1AddrSpace);

  // Ping/pong shared L1 alloc before the loop (depth-2 reuse across unrolled
  // stages instead of one buffer per transfer).
  builder.setInsertionPoint(c.loop);
  auto sharedL1AllocOp = pool.getOrCreate(builder, c.loc, l1AllocType);

  // Inside loop body after producer: (NZ pack) -> to_memref -> cast -> copy.
  builder.setInsertionPointAfter(xfer.producer);
  auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  Value packedTensor = xfer.value;
  SmallVector<int64_t, 4> srcShape =
      useNZ ? SmallVector<int64_t, 4>{N16, M16, 16, 16}
            : SmallVector<int64_t, 4>{M, N};

  if (useNZ) {
    // ND [M,N] -> NZ [N/16, M/16, 16, 16] via reshape -> transpose -> reshape.
    auto i64Ty = builder.getI64Type();
    auto s3Type = RankedTensorType::get({3}, i64Ty);
    auto s3Const = builder.create<arith::ConstantOp>(c.loc, s3Type,
        DenseElementsAttr::get(s3Type, ArrayRef<int64_t>{M, N16, 16}));
    auto resh1Type = RankedTensorType::get({M, N16, 16}, elemType);
    auto resh1 = builder.create<tensor::ReshapeOp>(c.loc, resh1Type,
        xfer.value, s3Const.getResult());
    auto emptyT = builder.create<tensor::EmptyOp>(c.loc,
        ArrayRef<int64_t>{N16, M, 16}, elemType);
    auto transp = builder.create<linalg::TransposeOp>(c.loc, resh1.getResult(),
        emptyT.getResult(), ArrayRef<int64_t>{1, 0, 2});
    auto s4Type = RankedTensorType::get({4}, i64Ty);
    auto s4Const = builder.create<arith::ConstantOp>(c.loc, s4Type,
        DenseElementsAttr::get(s4Type, ArrayRef<int64_t>{N16, M16, 16, 16}));
    auto nzTensorType = RankedTensorType::get({N16, M16, 16, 16}, elemType);
    auto resh2 = builder.create<tensor::ReshapeOp>(c.loc, nzTensorType,
        transp->getResult(0), s4Const.getResult());
    packedTensor = resh2.getResult();
  }

  auto srcMemrefType = MemRefType::get(srcShape, elemType);
  auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(
      c.loc, srcMemrefType, packedTensor);
  auto ubMemrefType = MemRefType::get(srcShape, elemType, nullptr, ubAddrSpace);
  auto ubCastOp = builder.create<memref::MemorySpaceCastOp>(
      c.loc, ubMemrefType, toMemrefOp.getResult());

  // UB -> L1 copy (same NZ/flat shape on both sides).
  builder.create<hivm::CopyOp>(c.loc, mlir::TypeRange{},
      ubCastOp.getResult(),           // src (UB memref)
      sharedL1AllocOp.getResult());   // dst (shared L1 memref)

  // VECTOR signals CUBE.
  builder.create<hivm::SyncBlockSetOp>(c.loc, c.vecCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr, flagAttr);

  // Consumer (CUBE) side: wait + convert_layout (NZ -> ND view) for matmul.
  Operation *firstConsumer = xfer.consumers.front();
  for (auto *cons : xfer.consumers)
    if (cons->isBeforeInBlock(firstConsumer))
      firstConsumer = cons;
  builder.setInsertionPoint(firstConsumer);

  builder.create<hivm::SyncBlockWaitOp>(c.loc, c.cubeCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr, flagAttr);

  // ONE convert_layout (NZ -> ND view) + memory_space_cast per shared L1
  // buffer, reused across the unrolled stages that share that buffer. The
  // manual kernel emits the convert_layout once per p_l1 buffer and only
  // re-reads it with a fresh to_tensor per matmul. Emitting a convert_layout
  // per unrolled stage produces several aliasing ND views of the same cbuf
  // buffer, which BiShengIR mis-tracks into a misaligned / zero-burst L1->L0
  // load. The view ops stay INSIDE the loop body (before the first consumer of
  // the first stage that uses this buffer) — hoisting them above the loop
  // breaks the MIX-kernel AIC/AIV split (SplitMixKernel can't get out-operands
  // for a scope-level convert_layout).
  Operation *l1Key = sharedL1AllocOp.getOperation();
  Value ndViewVal = pool.ndView.lookup(l1Key);
  if (!ndViewVal) {
    // Place at the FRONT of the loop body so the single view dominates every
    // consumer regardless of the order transfers are processed in (the reuse
    // for later stages must be dominated by this definition).
    OpBuilder viewBuilder(c.ctx);
    scf::ForOp loopMut = c.loop;
    viewBuilder.setInsertionPointToStart(loopMut.getBody());
    auto ndLayout = hivm::DataLayoutAttr::get(c.ctx, hivm::DataLayout::ND);
    auto ndL1Type = MemRefType::get(shape, elemType, nullptr, l1AddrSpace);
    auto convertOp = viewBuilder.create<hivm::ConvertLayoutOp>(
        c.loc, ndL1Type, sharedL1AllocOp.getResult(), ndLayout, ndLayout,
        DenseI64ArrayAttr::get(c.ctx, shape), ValueRange{});
    auto plainMemrefType = MemRefType::get(shape, elemType);
    auto castOp = viewBuilder.create<memref::MemorySpaceCastOp>(
        c.loc, plainMemrefType, convertOp.getResult());
    ndViewVal = castOp.getResult();
    pool.ndView[l1Key] = ndViewVal;
  }
  // Fresh to_tensor per consumer group (after the wait), like the manual.
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      c.loc, tensorType, ndViewVal, true, true);

  for (auto *consumer : xfer.consumers)
    consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());

  llvm::errs() << "[cv-split]   V→C transfer #" << flagId
               << " (tightly_coupled=" << markAllocIndex << ")"
               << ": " << xfer.producer->getName()
               << " → " << M << "x" << N << " L1 buffer\n";
}

static void insertCrossScopeTransfers(
    scf::ForOp loop,
    Block *body,
    const DenseMap<Operation *, EngineType> &classification) {

  MLIRContext *ctx = loop.getContext();
  Location loc = loop.getLoc();

  auto transfers = findCrossScopeValues(body, classification);
  if (transfers.empty()) {
    llvm::errs() << "[cv-split] No cross-scope transfers needed\n";
    return;
  }

  // Sort transfers for clean flag numbering: C→V QK first, then V→C P, then C→V PV
  // QK = C→V with smaller shape; PV = C→V with larger shape; P = V→C
  std::stable_sort(transfers.begin(), transfers.end(),
      [](const CrossScopeTransfer &a, const CrossScopeTransfer &b) {
        if (a.direction != b.direction)
          return a.direction == CrossScopeTransfer::CUBE_TO_VECTOR;
        if (a.direction == CrossScopeTransfer::CUBE_TO_VECTOR) {
          auto aType = dyn_cast<RankedTensorType>(a.value.getType());
          auto bType = dyn_cast<RankedTensorType>(b.value.getType());
          if (aType && bType) {
            int64_t aSize = aType.getNumElements();
            int64_t bSize = bType.getNumElements();
            if (aSize != bSize) return aSize < bSize;
          }
        }
        return false;
      });

  llvm::errs() << "[cv-split] Found " << transfers.size()
               << " cross-scope value transfers\n";

  TransferEmitContext ec{
      ctx, loc, loop,
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE),
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_V),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
      hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE1)};

  // Per-channel flag counters. C->V (PIPE_FIX/PIPE_V) and V->C (PIPE_MTE3/
  // PIPE_MTE1) are independent hardware sync channels (the HW key is the
  // (set_pipe, wait_pipe, event_id) triple), so each gets its own 0.. range.
  // The backend WAIT.INTRA.BLOCK intrinsic encodes the flag as a 4-bit
  // immediate (valid 0..15); a single shared counter overflowed it at unroll-8
  // (flags 0-23, "Cannot select" for >=16). Splitting per channel keeps
  // unroll-8 in range (C->V 0-15, V->C 0-7) without reusing a flag inside a
  // channel (never a set before its wait -> no set_flag hazard).
  int cvFlagCounter = 0;  // CUBE -> VECTOR (PIPE_FIX / PIPE_V)
  int vcFlagCounter = 0;  // VECTOR -> CUBE (PIPE_MTE3 / PIPE_MTE1)
  int markAllocIndex = 0; // ordinal of the shared buffer across all transfers

  // Depth-2 ping/pong pool shared by all transfers: same-typed buffers (all
  // unrolled qk_ub, all pv_ub, all P L1) rotate over 2 physical allocations.
  PingPongPool pool;

  for (auto &xfer : transfers) {
    auto tensorType = dyn_cast<RankedTensorType>(xfer.value.getType());
    if (!tensorType) {
      llvm::errs() << "[cv-split]   Skipping non-tensor transfer: "
                   << xfer.value.getType() << "\n";
      continue;
    }
    if (tensorType.getRank() < 2) {
      llvm::errs() << "[cv-split]   Skipping rank-" << tensorType.getRank()
                   << " tensor\n";
      continue;
    }

    if (xfer.direction == CrossScopeTransfer::CUBE_TO_VECTOR) {
      emitCubeToVectorTransfer(ec, xfer, tensorType, cvFlagCounter++, pool);
    } else {
      emitVectorToCubeTransfer(ec, xfer, tensorType, vcFlagCounter++,
                               markAllocIndex, pool);
    }
    ++markAllocIndex;
  }

  llvm::errs() << "[cv-split] Inserted " << transfers.size()
               << " transfers with " << cvFlagCounter << " C->V + "
               << vcFlagCounter << " V->C sync flags, "
               << markAllocIndex << " tightly-coupled pairs\n";
}

// ============================================================================
// Stage 7.5: Unfuse PV matmuls (split matmul(p,v,acc*alpha) into
//   pv = matmul(p,v,zeros) + combined = arith.addf(pv, acc*alpha))
// This is needed because triton's combine pass fuses arith.addf(matmul(...,0), x)
// into matmul(..., x), creating an unresolvable CUBE→VECTOR→CUBE chain through
// the accumulator. Unfusing makes the PV matmul independent of the accumulator.
// ============================================================================
static void unfusePVMatmuls(Block *body,
                            DenseMap<Operation *, EngineType> &classification) {
  SmallVector<linalg::MatmulOp> toUnfuse;
  for (Operation &op : *body) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(&op);
    if (!matmulOp) continue;
    auto classIt = classification.find(&op);
    if (classIt == classification.end() || classIt->second != EngineType::CUBE)
      continue;

    // Check if outs (operand index getDpsInitOperand(0)) is non-zero
    // The outs value is the DPS init — if it's a constant zero, skip
    Value outsVal = matmulOp.getDpsInitOperand(0)->get();
    auto outsType = dyn_cast<RankedTensorType>(outsVal.getType());
    if (!outsType) continue;

    // Check if outs is produced by a VECTOR op (e.g. arith.mulf for acc*alpha)
    Operation *outsDef = outsVal.getDefiningOp();
    if (!outsDef) continue;
    auto outsClassIt = classification.find(outsDef);
    if (outsClassIt == classification.end()) continue;
    if (outsClassIt->second != EngineType::VECTOR) continue;

    // This is a fused PV matmul with VECTOR-produced accumulator init
    toUnfuse.push_back(matmulOp);
  }

  if (toUnfuse.empty()) return;

  llvm::errs() << "[cv-split] Unfusing " << toUnfuse.size()
               << " PV matmuls with VECTOR outs\n";

  for (auto matmulOp : toUnfuse) {
    OpBuilder builder(matmulOp);
    Location loc = matmulOp.getLoc();

    Value outsVal = matmulOp.getDpsInitOperand(0)->get();
    auto outsType = cast<RankedTensorType>(outsVal.getType());

    // Create zero init tensor
    auto zeroAttr = builder.getZeroAttr(outsType.getElementType());
    auto zeroConst = builder.create<arith::ConstantOp>(loc, outsType,
        DenseElementsAttr::get(outsType, zeroAttr));

    // Replace outs with zeros in the matmul
    matmulOp.getDpsInitOperand(0)->set(zeroConst.getResult());

    // Insert arith.addf after matmul: combined = matmul_result + original_outs
    builder.setInsertionPointAfter(matmulOp);
    Value matResult = matmulOp.getResult(0);
    auto addOp = builder.create<arith::AddFOp>(loc, matResult, outsVal);

    // Replace all uses of the original matmul result (except the addf itself)
    matResult.replaceAllUsesExcept(addOp.getResult(), addOp);

    // Classify new ops
    classification[zeroConst] = EngineType::VECTOR;
    classification[addOp] = EngineType::VECTOR;
  }
}

// ============================================================================
// Stage 11.5: Bind the loop-invariant matmul LHS (Q in flash-attention) into a
// dedicated L1 (cbuf) buffer, matching the manual kernel.
//
// In the manual kernel Q is staged once into a persistent cbuf buffer
// (mem_unique) and the QK matmul reads it directly from L1:
//   %alloc_q = memref.alloc() : memref<MxKxf16, cbuf>
//   annotation.mark %alloc_q {mem_unique}
//   annotation.mark %alloc_q {effects = ["write","read"]}
//   annotation.mark %q keys = ["bind_buffer"] values = [%alloc_q : cbuf]
//
// Without this, our QK matmul reads Q from a loop-invariant *plain* memref and
// BiShengIR inserts an implicit GM/UB->L1 stage whose descriptor is misaligned
// (runtime fixp_addr_misal / zero-burst MOV_SRC_TO_FB on the cube). Binding Q to
// a cbuf buffer (as the manual does) gives the matmul an aligned NZ L1 operand.
//
// We target only the loop-invariant LHS (defined OUTSIDE the innermost loop):
// that is Q for QK. The PV matmul LHS (P) is loop-variant (produced by the
// vector scope) and already lives in cbuf via convert_layout, so it is skipped.
static void bindLoopInvariantMatmulLhsToCbuf(func::FuncOp funcOp) {
  // Map each distinct loop-invariant LHS value to the CUBE scope that consumes
  // it. The alloc + bind must live INSIDE that scope (before its loop) so the
  // whole Q-staging is self-contained on the cube/AIC side after the MIX split
  // — placing it at function level breaks dominance in buildFinalHIVMPipelines.
  llvm::MapVector<Value, scope::ScopeOp> lhsToScope;
  funcOp.walk([&](Operation *op) {
    Value lhs;
    if (auto m = dyn_cast<linalg::MatmulTransposeBOp>(op))
      lhs = m.getInputs()[0];
    else if (auto m = dyn_cast<linalg::MatmulOp>(op))
      lhs = m.getInputs()[0];
    else
      return;

    auto tt = dyn_cast<RankedTensorType>(lhs.getType());
    if (!tt || tt.getRank() != 2 || !tt.getElementType().isF16())
      return;

    Operation *def = lhs.getDefiningOp();
    if (!def)
      return; // block argument: not a stage-able buffer

    auto enclosingFor = op->getParentOfType<scf::ForOp>();
    if (!enclosingFor)
      return;
    // Loop-variant LHS (e.g. P, produced inside the loop) -> already cbuf-backed.
    if (enclosingFor->isProperAncestor(def))
      return;

    auto cubeScope = op->getParentOfType<scope::ScopeOp>();
    if (!cubeScope)
      return; // only handle matmuls that ended up inside a scope
    // The LHS def must dominate the scope (it is defined before/outside it).
    if (cubeScope->isProperAncestor(def))
      return;

    if (!lhsToScope.count(lhs))
      lhsToScope.insert({lhs, cubeScope});
  });

  for (auto &kv : lhsToScope) {
    Value q = kv.first;
    scope::ScopeOp cubeScope = kv.second;
    Block *scopeBody = &cubeScope.getBodyRegion().front();

    // GM source behind Q. We re-load Q from global memory INSIDE the cube scope
    // (matching the manual): GM -> fresh plain buffer -> bind to cbuf. Binding a
    // value captured from outside the scope breaks dominance after the MIX
    // split, and copying from Q's existing (now-cbuf-bound) staging buffer would
    // lower to an unsupported cbuf->cbuf copy. Copying straight from the GM
    // reinterpret_cast avoids both.
    Value srcMemref;
    if (auto toTensor = q.getDefiningOp<bufferization::ToTensorOp>()) {
      Value qMem = toTensor.getMemref();
      // Trace back through the GM->staging memref.copy to the GM source.
      for (Operation *user : qMem.getUsers()) {
        if (auto cp = dyn_cast<memref::CopyOp>(user)) {
          if (cp.getTarget() == qMem) {
            srcMemref = cp.getSource();
            break;
          }
        }
      }
    }
    if (!srcMemref)
      continue; // unexpected producer; leave Q as-is (safe fallback)

    OpBuilder builder(cubeScope.getContext());
    Location loc = q.getLoc();
    auto tt = cast<RankedTensorType>(q.getType());

    // Persistent cbuf buffer for Q (mem_unique). It MUST be allocated at
    // FUNCTION scope (before the CUBE scope), alongside the P/V cbuf buffers,
    // NOT inside the cube scope. SplitMixKernel clones the function into an AIC
    // and an AIV part and drops each scope's body from the other clone; if the
    // Q cbuf alloc lives inside the CUBE scope it vanishes from the AIV clone,
    // the two clones then disagree on cbuf (L1) layout, and the cross-engine P
    // buffer lands at mismatched L1 addresses in the cube vs vector code -> the
    // matmul's L1->FB operand load reads a misaligned address (degenerate
    // MOV_SRC_TO_FB, fixp_addr_misal at runtime). Hoisting it to function scope
    // (matching the manual kernel) keeps the L1 layout identical in both clones.
    //
    // It must also be the FIRST cbuf buffer (before the P/V NZ buffers), matching
    // the manual's Q,P,V order: PlanMemory lays cbuf out in allocation order, and
    // the QK matmul reads Q from L1 into the cube feature buffer. If Q lands at a
    // large L1 offset (P+V before it ~= 64 KB) the offset no longer fits the FB
    // load's immediate offset field, so hivmc emits offset mode 2 (register
    // offset) which the simulator's dmamov_decode_to_fb rejects ("Invalid offset
    // mode: 2"). Placing Q first keeps it at L1 offset 0.
    Operation *firstCbufAlloc = nullptr;
    for (Operation &o : *cubeScope->getBlock()) {
      auto a = dyn_cast<memref::AllocOp>(&o);
      if (!a)
        continue;
      auto mt = dyn_cast<MemRefType>(a.getType());
      if (!mt)
        continue;
      auto as = dyn_cast_or_null<hivm::AddressSpaceAttr>(mt.getMemorySpace());
      if (as && as.getAddressSpace() == hivm::AddressSpace::L1) {
        firstCbufAlloc = &o;
        break;
      }
    }
    if (firstCbufAlloc)
      builder.setInsertionPoint(firstCbufAlloc);
    else
      builder.setInsertionPoint(cubeScope);
    auto cbufAS = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
    auto cbufType =
        MemRefType::get(tt.getShape(), tt.getElementType(), nullptr, cbufAS);
    auto cbufAlloc = builder.create<memref::AllocOp>(loc, cbufType);

    auto muMark = builder.create<annotation::MarkOp>(loc, cbufAlloc.getResult());
    muMark->setAttr("mem_unique", builder.getUnitAttr());

    auto effMark = builder.create<annotation::MarkOp>(loc, cbufAlloc.getResult());
    effMark->setAttr("effects", builder.getArrayAttr({builder.getStringAttr("write"),
                                                      builder.getStringAttr("read")}));

    // Everything below (the fresh Q load, bind_buffer, cbuf read view, matmul
    // operand rewrite) stays INSIDE the cube scope — those are cube-only and
    // correctly dropped from the AIV clone.
    builder.setInsertionPointToStart(scopeBody);

    // Fresh in-scope Q load (plain memref) + to_tensor, then bind it to the
    // cbuf buffer (this fills the persistent cbuf with Q, kept ND [M,K]).
    auto plainType = MemRefType::get(tt.getShape(), tt.getElementType());
    auto qAlloc = builder.create<memref::AllocOp>(loc, plainType);
    builder.create<memref::CopyOp>(loc, srcMemref, qAlloc.getResult());
    auto qBindTensor = builder.create<bufferization::ToTensorOp>(
        loc, tt, qAlloc.getResult(), /*restrict=*/true, /*writable=*/true);

    builder.create<annotation::MarkOp>(loc, qBindTensor.getResult(),
                                       ValueRange{cbufAlloc.getResult()},
                                       builder.getStrArrayAttr({"bind_buffer"}));

    // Read Q back from the cbuf buffer via memory_space_cast for the matmul
    // operand — matching the manual kernel (and our P operand path). Feeding the
    // bound plain tensor directly makes BiShengIR insert an nd2nz + multi_buffer
    // for Q (loop-invariant Q must NOT be double-buffered), which overflows UB.
    // A plain cbuf->ND memory_space_cast read keeps Q single-buffered ND [M,K].
    auto qCastView = builder.create<memref::MemorySpaceCastOp>(
        loc, plainType, cbufAlloc.getResult());
    auto qReadTensor = builder.create<bufferization::ToTensorOp>(
        loc, tt, qCastView.getResult(), /*restrict=*/true, /*writable=*/true);

    // Rewrite Q uses inside the cube scope (the QK matmuls) to the cbuf read.
    q.replaceUsesWithIf(qReadTensor.getResult(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return cubeScope->isProperAncestor(owner);
    });

    // Drop the now-dead original (pre-loop) Q load chain so its UB staging
    // buffer is reclaimed: the to_tensor, the GM->staging memref.copy, and the
    // staging alloc. Keeping it would double Q's UB footprint (and the copy has
    // side effects, so DCE will not remove it on its own).
    if (auto origToTensor = q.getDefiningOp<bufferization::ToTensorOp>()) {
      if (origToTensor->use_empty()) {
        Value qMem = origToTensor.getMemref();
        origToTensor->erase();
        Operation *fillCopy = nullptr;
        for (Operation *user : qMem.getUsers()) {
          if (auto cp = dyn_cast<memref::CopyOp>(user))
            if (cp.getTarget() == qMem) { fillCopy = cp; break; }
        }
        if (fillCopy)
          fillCopy->erase();
        if (Operation *allocDef = qMem.getDefiningOp())
          if (isa<memref::AllocOp>(allocDef) && allocDef->use_empty())
            allocDef->erase();
      }
    }

    llvm::errs() << "[cv-split] Staged loop-invariant matmul LHS into cbuf "
                 << tt << " (inside CUBE scope, bind_buffer)\n";
  }
}

//
// 1. Move all function body ops into a scope::ScopeOp (VECTOR)
// 2. Clone that scope to create a second one (CUBE)
// 3. Strip wrong-type ops from each scope using our classification
// 4. Neutralize yield slots for stripped ops with zero/alloc placeholders
//
// This gives bishengir two isolated scopes it can compile independently.
// ============================================================================

static Value buildNeutralPlaceholder(OpBuilder &builder, Type type, Location loc) {
  if (auto memrefTy = dyn_cast<MemRefType>(type))
    return builder.create<memref::AllocOp>(loc, memrefTy).getResult();

  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasStaticShape() && !isa<MemRefType>(type)) {
      if (Attribute elemZero = builder.getZeroAttr(shapedTy.getElementType())) {
        auto zeroDense = DenseElementsAttr::get(shapedTy, elemZero);
        return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(zeroDense)).getResult();
      }
    }
  }

  if (Attribute zeroAttr = builder.getZeroAttr(type)) {
    if (auto typedZero = dyn_cast<TypedAttr>(zeroAttr))
      return builder.create<arith::ConstantOp>(loc, type, typedZero).getResult();
  }

  return Value();
}

// Re-classify all ops in a scope's loop body using full BFS propagation.
// This is the same algorithm as classifyAllOps but also handles transfer ops
// (sync, fixpipe, convert_layout, copy) that were inserted after the initial
// classification in Stage 3.
static void classifyOpsInScopeBody(Block *body,
    DenseMap<Operation *, EngineType> &scopeClassification) {
  // Seed with CUBE ops + explicit engine classification for transfer ops
  SmallVector<Operation *> cubeSeeds;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;

    EngineType ty = EngineType::VECTOR;

    if (isa<linalg::MatmulOp>(&op) || isa<linalg::MatmulTransposeBOp>(&op) ||
        isa<linalg::BatchMatmulOp>(&op))
      ty = EngineType::CUBE;
    else if (op.getName().getStringRef().contains("dot") ||
             op.getName().getStringRef().contains("matmul"))
      ty = EngineType::CUBE;
    else if (isa<hivm::FixpipeOp>(&op))
      ty = EngineType::CUBE;
    else if (isa<hivm::ConvertLayoutOp>(&op))
      ty = EngineType::CUBE;
    else if (auto syncSetOp = dyn_cast<hivm::SyncBlockSetOp>(&op)) {
      auto coreType = syncSetOp.getTcoreType().getTcoretype();
      ty = coreType == hivm::TCoreType::CUBE ? EngineType::CUBE : EngineType::VECTOR;
    } else if (auto syncWaitOp = dyn_cast<hivm::SyncBlockWaitOp>(&op)) {
      auto coreType = syncWaitOp.getTcoreType().getTcoretype();
      ty = coreType == hivm::TCoreType::CUBE ? EngineType::CUBE : EngineType::VECTOR;
    } else if (isa<hivm::CopyOp>(&op))
      ty = EngineType::VECTOR;

    scopeClassification[&op] = ty;
    if (ty == EngineType::CUBE)
      cubeSeeds.push_back(&op);
  }

  // BFS backward: propagate CUBE to data feeders (multi-hop)
  DenseSet<Operation *> visited;
  std::queue<Operation *> worklist;
  for (auto *seed : cubeSeeds)
    worklist.push(seed);

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop();
    if (!visited.insert(op).second)
      continue;

    for (Value operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != body || isa<scf::YieldOp>(defOp))
        continue;

      bool isDataFeeder =
          isa<bufferization::ToTensorOp>(defOp) ||
          isa<linalg::TransposeOp>(defOp) ||
          isa<linalg::FillOp>(defOp) ||
          isa<memref::AllocOp>(defOp) ||
          isa<memref::CopyOp>(defOp) ||
          isa<memref::SubViewOp>(defOp) ||
          isa<memref::ReinterpretCastOp>(defOp) ||
          isa<memref::MemorySpaceCastOp>(defOp) ||
          isa<tensor::ExtractSliceOp>(defOp) ||
          isa<hivm::ConvertLayoutOp>(defOp) ||
          isa<annotation::MarkOp>(defOp) ||
          defOp->getName().getStringRef().contains("transpose") ||
          defOp->getName().getStringRef().contains("to_tensor") ||
          defOp->getName().getStringRef().contains("convert_layout");

      if (isDataFeeder && scopeClassification[defOp] != EngineType::CUBE) {
        scopeClassification[defOp] = EngineType::CUBE;
        worklist.push(defOp);
      }
    }
  }

  // Forward pass: promote memref.copy to CUBE if its destination alloc is CUBE
  // (memref.copy has no SSA result, so BFS can't reach it backward)
  // Also promote the SOURCE of a CUBE copy (typically reinterpret_cast) to CUBE.
  for (Operation &op : *body) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
      Value dst = copyOp.getTarget();
      if (auto *dstDef = dst.getDefiningOp()) {
        auto dstIt = scopeClassification.find(dstDef);
        if (dstIt != scopeClassification.end() &&
            dstIt->second == EngineType::CUBE) {
          scopeClassification[&op] = EngineType::CUBE;
          // Promote source (e.g. reinterpret_cast) to CUBE too
          Value src = copyOp.getSource();
          if (auto *srcDef = src.getDefiningOp()) {
            if (srcDef->getBlock() == body)
              scopeClassification[srcDef] = EngineType::CUBE;
          }
        }
      }
    }
  }

  // Also promote annotation::MarkOp to CUBE if it marks a CUBE alloc or fixpipe operand
  for (Operation &op : *body) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(&op)) {
      if (auto *defOp = markOp.getOperand(0).getDefiningOp()) {
        auto defIt = scopeClassification.find(defOp);
        if (defIt != scopeClassification.end() &&
            defIt->second == EngineType::CUBE) {
          scopeClassification[&op] = EngineType::CUBE;
        }
        // Also check if the marked alloc feeds fixpipe
        for (Value res : defOp->getResults()) {
          for (Operation *user : res.getUsers()) {
            if (isa<hivm::FixpipeOp>(user)) {
              scopeClassification[&op] = EngineType::CUBE;
              break;
            }
          }
        }
      }
    }
  }

  // Default: anything not yet classified → VECTOR
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op))
      continue;
    if (scopeClassification.find(&op) == scopeClassification.end())
      scopeClassification[&op] = EngineType::VECTOR;
  }
}

static void stripWrongTypeOps(scope::ScopeOp scopeOp, EngineType keepType,
    const DenseMap<Operation *, EngineType> &scopeClassification) {
  llvm::errs() << "[cv-split] Stripping " 
               << (keepType == EngineType::CUBE ? "VECTOR" : "CUBE")
               << " ops from " << engineTypeToStr(keepType) << " scope\n";

  Block &scopeBlock = scopeOp.getBodyRegion().front();

  // First pass: strip wrong-type ops at the scope's top-level block
  auto *terminator = scopeBlock.getTerminator();
  SmallVector<Operation *> topLevelToErase;
  for (Operation &op : scopeBlock) {
    if (&op == terminator) continue;
    if (isa<scf::ForOp>(&op)) continue;
    if (isa<arith::ConstantOp>(&op)) continue;
    // Shared allocs/marks stay in both scopes
    if (isa<memref::AllocOp>(&op) || isa<annotation::MarkOp>(&op)) {
      if (op.hasAttr("effects") || op.hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name))
        continue;
      auto coreAttr = op.getAttrOfType<StringAttr>("ssbuffer.core_type");
      if (!coreAttr) continue;
      StringRef tag = coreAttr.getValue();
      StringRef keepStr = keepType == EngineType::CUBE ? "CUBE" : "VECTOR";
      if (tag == keepStr) continue;
      topLevelToErase.push_back(&op);
      continue;
    }
    // Keep scalar/index ops (address math, loop control)
    bool allScalar = true;
    for (Value res : op.getResults()) {
      Type t = res.getType();
      if (!t.isIntOrIndexOrFloat() || isa<RankedTensorType>(t)) {
        allScalar = false;
        break;
      }
    }
    if (allScalar && op.getNumResults() > 0) continue;
    // Keep address infrastructure only if classified as keepType
    if (isa<memref::ReinterpretCastOp, memref::CopyOp, memref::MemorySpaceCastOp>(&op)) {
      auto it = scopeClassification.find(&op);
      EngineType opType = (it != scopeClassification.end()) ? it->second : EngineType::VECTOR;
      if (opType == keepType || opType == EngineType::UNKNOWN)
        continue;
      topLevelToErase.push_back(&op);
      continue;
    }

    // Use scope-level classification (from BFS)
    auto it = scopeClassification.find(&op);
    EngineType opType = (it != scopeClassification.end()) ? it->second : EngineType::VECTOR;
    if (opType != keepType && opType != EngineType::UNKNOWN) {
      topLevelToErase.push_back(&op);
    }
  }

  for (Operation *op : llvm::reverse(topLevelToErase)) {
    OpBuilder builder(op);
    for (Value result : op->getResults()) {
      if (!result.use_empty()) {
        Value placeholder = buildNeutralPlaceholder(builder, result.getType(), op->getLoc());
        if (placeholder)
          result.replaceAllUsesWith(placeholder);
        else
          result.dropAllUses();
      }
    }
    op->erase();
  }

  // Second pass: strip wrong-type ops inside scf.for loop bodies
  int loopErased = 0;
  scopeOp.walk([&](scf::ForOp forOp) {
    Block *body = forOp.getBody();
    auto *yieldOp = body->getTerminator();

    SmallVector<Operation *> toErase;
    for (Operation &op : *body) {
      if (&op == yieldOp) continue;
      if (isa<arith::ConstantOp>(&op)) continue;
      // Shared allocs/marks stay in both scopes
      if (isa<memref::AllocOp>(&op) || isa<annotation::MarkOp>(&op)) {
        if (op.hasAttr("effects") || op.hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name))
          continue;
        auto coreAttr = op.getAttrOfType<StringAttr>("ssbuffer.core_type");
        if (!coreAttr) continue;
        StringRef tag = coreAttr.getValue();
        StringRef keepStr = keepType == EngineType::CUBE ? "CUBE" : "VECTOR";
        if (tag == keepStr) continue;
        toErase.push_back(&op);
        continue;
      }
      // Keep scalar/index ops
      bool allScalar = true;
      for (Value res : op.getResults()) {
        Type t = res.getType();
        if (!t.isIntOrIndexOrFloat() || isa<RankedTensorType>(t)) {
          allScalar = false;
          break;
        }
      }
      if (allScalar && op.getNumResults() > 0) continue;
      // Keep address infrastructure only if classified as keepType
      if (isa<memref::ReinterpretCastOp, memref::CopyOp, memref::MemorySpaceCastOp>(&op)) {
        auto it = scopeClassification.find(&op);
        EngineType opType = (it != scopeClassification.end()) ? it->second : EngineType::VECTOR;
        if (opType == keepType || opType == EngineType::UNKNOWN)
          continue;
        toErase.push_back(&op);
        continue;
      }

      // Use scope-level BFS classification
      auto it = scopeClassification.find(&op);
      EngineType opType = (it != scopeClassification.end()) ? it->second : EngineType::VECTOR;
      if (opType != keepType && opType != EngineType::UNKNOWN)
        toErase.push_back(&op);
    }

    for (Operation *op : llvm::reverse(toErase)) {
      OpBuilder builder(op);
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          Value placeholder = buildNeutralPlaceholder(builder, result.getType(), op->getLoc());
          if (placeholder)
            result.replaceAllUsesWith(placeholder);
          else
            result.dropAllUses();
        }
      }
      op->erase();
    }
    loopErased += toErase.size();
  });

  llvm::errs() << "[cv-split]   Erased " << topLevelToErase.size() << " top-level + "
               << loopErased << " loop ops from " << engineTypeToStr(keepType) << " scope\n";

  // Final cleanup: remove dead allocs whose only users are annotation.mark
  SmallVector<Operation *> deadOps;
  scopeOp.walk([&](memref::AllocOp allocOp) {
    Value result = allocOp.getResult();
    bool allUsersDead = true;
    SmallVector<Operation *> markUsers;
    for (Operation *user : result.getUsers()) {
      if (isa<annotation::MarkOp>(user)) {
        markUsers.push_back(user);
      } else {
        allUsersDead = false;
        break;
      }
    }
    if (allUsersDead && result.use_empty()) {
      deadOps.push_back(allocOp);
    } else if (allUsersDead && !markUsers.empty()) {
      for (auto *m : markUsers) deadOps.push_back(m);
      deadOps.push_back(allocOp);
    }
  });
  for (Operation *op : llvm::reverse(deadOps)) {
    if (isa<annotation::MarkOp>(op))
      op->erase();
  }
  for (Operation *op : llvm::reverse(deadOps)) {
    if (isa<memref::AllocOp>(op) && op->use_empty())
      op->erase();
  }
  if (!deadOps.empty())
    llvm::errs() << "[cv-split]   Cleaned up " << deadOps.size() << " dead alloc/mark ops\n";
}

// ============================================================================
// Single-veccore output guard.
// With NO_DUAL fixpipe, the cube delivers the whole MxN tile to one sub-block's
// UB; the other veccore's UB is stale. Both veccores still execute the VECTOR
// scope, so we must ensure only the veccore that owns valid data writes the
// result to GM. We wrap the epilogue store(s) in `if (get_sub_block_idx == 0)`.
// (Half vector throughput; correctness-first, does not match the ROW_SPLIT
// target which uses both veccores.)
// ============================================================================
static void guardStoresToSubBlock0(scope::ScopeOp vecScope) {
  Block &block = vecScope.getBodyRegion().front();

  SmallVector<Operation *> stores;
  for (Operation &op : block) {
    if (isa<bufferization::MaterializeInDestinationOp>(&op))
      stores.push_back(&op);
  }
  if (stores.empty()) {
    llvm::errs() << "[cv-split]   guardStores: no materialize_in_destination found\n";
    return;
  }

  Operation *firstStore = stores.front();
  Location loc = firstStore->getLoc();
  Operation *terminator = block.getTerminator();

  // Collect the contiguous epilogue tail [firstStore .. terminator) so that
  // any intervening producers (e.g. the truncf feeding the 2nd store) move too.
  SmallVector<Operation *> tail;
  for (Operation *op = firstStore; op != terminator;
       op = op->getNextNode())
    tail.push_back(op);

  OpBuilder builder(firstStore);
  auto sbid = builder.create<hivm::GetSubBlockIdxOp>(loc, builder.getI64Type());
  auto zero = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  auto cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      sbid.getResult(), zero.getResult());
  auto ifOp = builder.create<scf::IfOp>(loc, cond.getResult(),
      /*withElseRegion=*/false);

  Operation *thenTerm = ifOp.thenBlock()->getTerminator();
  for (Operation *op : tail)
    op->moveBefore(thenTerm);

  llvm::errs() << "[cv-split]   guardStores: wrapped " << stores.size()
               << " store(s) (+ " << (tail.size() - stores.size())
               << " tail ops) in if(get_sub_block_idx==0)\n";
}

static void retileVectorScopeForRowSplit(scope::ScopeOp vecScope);
static void wrapSimdScopes(scope::ScopeOp vecScope);
static void rowLoopifyVectorScope(scope::ScopeOp vecScope);
static void serializeVectorScopeByInstance(scope::ScopeOp vecScope);
static void rewriteNZTransposePacks(scope::ScopeOp vecScope);
static void coalesceAdjacentSimdScopes(scope::ScopeOp vecScope);
static void sinkCubeLoadChainsToMatmul(scope::ScopeOp cubeScope);
static void hoistVectorStateToMemUnique(scope::ScopeOp vecScope);

// Sink each cube matmul's operand load chain to immediately before the matmul.
// See the call site (createScopeSeparation step 6b) for the rationale.
//
// For each linalg.matmul in the cube loop body (program order), gather the
// backward "load chain" of ops feeding its tensor operands -- restricted to the
// same block and to load-chain op types (reinterpret_cast / alloc / memref.copy
// / to_tensor / transpose). An op is only moved if every use of it is internal
// to the chain or the matmul itself, so shared operands (the loop-invariant Q
// cbuf, the P pack consumed by all PV matmuls, accumulator fill tensors) are
// left in place. The collected ops are moved (in their existing relative order,
// preserving topological validity) to just before the matmul.
static void sinkCubeLoadChainsToMatmul(scope::ScopeOp cubeScope) {
  auto isChainType = [](Operation *o) {
    return isa<memref::ReinterpretCastOp, memref::AllocOp, memref::CopyOp,
               bufferization::ToTensorOp, linalg::TransposeOp>(o);
  };

  cubeScope.walk([&](scf::ForOp forOp) {
    Block *body = forOp.getBody();

    SmallVector<Operation *> matmuls;
    for (Operation &op : *body)
      if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op))
        matmuls.push_back(&op);

    for (Operation *mm : matmuls) {
      // 1. Build the candidate chain via a worklist over operands. memref.copy
      //    writes its dst alloc by side effect (no SSA result), so when we reach
      //    an alloc we also pull in the copy that writes it and that copy's
      //    source (reinterpret_cast).
      SetVector<Operation *> chain;
      SmallVector<Value> worklist(mm->operand_begin(), mm->operand_end());

      auto enqueueOperands = [&](Operation *op) {
        for (Value v : op->getOperands())
          worklist.push_back(v);
      };

      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != body || !isChainType(def))
          continue;
        if (!chain.insert(def))
          continue;
        enqueueOperands(def);
        // For an alloc, find the memref.copy in this block that writes it.
        if (isa<memref::AllocOp>(def)) {
          for (Operation *user : def->getResult(0).getUsers()) {
            auto copy = dyn_cast<memref::CopyOp>(user);
            if (copy && copy->getBlock() == body &&
                copy.getTarget() == def->getResult(0)) {
              if (chain.insert(copy))
                enqueueOperands(copy);
            }
          }
        }
      }

      // 2. Keep only ops that are safe to move: every user of the op's results
      //    must be inside the chain or be this matmul. (memref.copy has no
      //    results, so it is always safe once its alloc is in the chain.)
      auto safeToMove = [&](Operation *op) {
        if (isa<memref::CopyOp>(op))
          return chain.contains(op->getOperand(1).getDefiningOp());
        for (Operation *user : op->getUsers())
          if (user != mm && !chain.contains(user))
            return false;
        return true;
      };

      SmallVector<Operation *> movable;
      for (Operation *op : chain)
        if (safeToMove(op))
          movable.push_back(op);

      // 3. Move in existing program order so relative topological order holds.
      llvm::sort(movable, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      for (Operation *op : movable)
        op->moveBefore(mm);
    }
  });
}

static void createScopeSeparation(
    func::FuncOp funcOp, scf::ForOp innerLoop,
    DenseMap<Operation *, EngineType> &classification) {

  MLIRContext *ctx = funcOp.getContext();
  Location loc = innerLoop.getLoc();

  // Strategy (matching reference FA kernel pattern):
  // - CUBE scope: clone of inner loop with only CUBE ops (before VECTOR scope)
  // - VECTOR scope: original inner loop + ALL subsequent ops in the parent block
  //   (epilogue: normalize + store). Both go inside the VECTOR scope.
  // - Preamble (q load, ptr setup) stays in parent block before both scopes.
  // - Shared allocs stay at the function level (outside any scope).
  //
  // This ensures:
  //   parent_block {
  //     preamble...
  //     scope(CUBE) { inner_loop_clone { cube ops } }
  //     scope(VECTOR) { inner_loop { vector ops }; epilogue... }
  //   }

  Block *parentBlock = innerLoop->getBlock();
  OpBuilder builder(ctx);

  // Collect epilogue: all ops AFTER the inner loop in the parent block
  // (up to but not including the terminator)
  auto *terminator = parentBlock->getTerminator();
  SmallVector<Operation *> epilogueOps;
  bool afterLoop = false;
  for (Operation &op : *parentBlock) {
    if (&op == terminator) break;
    if (afterLoop) {
      epilogueOps.push_back(&op);
    }
    if (&op == innerLoop.getOperation()) {
      afterLoop = true;
    }
  }

  // Step 1: Create CUBE scope (placed BEFORE the inner loop)
  builder.setInsertionPoint(innerLoop);
  auto cubeScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type>{});
  cubeScope.getBodyRegion().emplaceBlock();
  cubeScope->setAttr("noinline", UnitAttr::get(ctx));

  // Clone the inner loop into CUBE scope
  Block *cubeBlock = &cubeScope.getBodyRegion().front();
  OpBuilder cubeBuilder(cubeBlock, cubeBlock->end());
  IRMapping cubeMapping;
  cubeBuilder.clone(*innerLoop.getOperation(), cubeMapping);
  cubeBuilder.create<scope::ReturnOp>(loc);

  // Step 2: Create VECTOR scope (wraps the original inner loop + epilogue)
  builder.setInsertionPoint(innerLoop);
  auto vecScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type>{});
  vecScope.getBodyRegion().emplaceBlock();
  vecScope->setAttr("noinline", UnitAttr::get(ctx));

  Block *vecBlock = &vecScope.getBodyRegion().front();
  // Move original inner loop into VECTOR scope
  innerLoop->remove();
  vecBlock->push_back(innerLoop.getOperation());
  // Move epilogue ops into VECTOR scope
  for (Operation *op : epilogueOps) {
    op->remove();
    vecBlock->push_back(op);
  }
  OpBuilder vecBuilder(vecBlock, vecBlock->end());
  vecBuilder.create<scope::ReturnOp>(loc);

  // Step 3: Set core type attributes (match target: only tcore_type + noinline)
  cubeScope->setAttr(hivm::TCoreTypeAttr::name,
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE));
  vecScope->setAttr(hivm::TCoreTypeAttr::name,
      hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR));

  // Step 4: Classify ops in each scope using BFS (includes transfer ops inserted in Stage 8)
  // This replaces the stale Stage-3 classification that pre-dates transfer insertion.
  DenseMap<Operation *, EngineType> cubeBodyClassification;
  cubeScope.walk([&](scf::ForOp forOp) {
    classifyOpsInScopeBody(forOp.getBody(), cubeBodyClassification);
  });

  DenseMap<Operation *, EngineType> vecBodyClassification;
  vecScope.walk([&](scf::ForOp forOp) {
    classifyOpsInScopeBody(forOp.getBody(), vecBodyClassification);
  });

  // Log classification summary for each scope
  int cubeCubeOps = 0, cubeVecOps = 0;
  for (auto &kv : cubeBodyClassification) {
    if (kv.second == EngineType::CUBE) ++cubeCubeOps;
    else ++cubeVecOps;
  }
  llvm::errs() << "[cv-split] CUBE scope BFS classification: "
               << cubeCubeOps << "C " << cubeVecOps << "V\n";
  int vecCubeOps = 0, vecVecOps = 0;
  for (auto &kv : vecBodyClassification) {
    if (kv.second == EngineType::CUBE) ++vecCubeOps;
    else ++vecVecOps;
  }
  llvm::errs() << "[cv-split] VECTOR scope BFS classification: "
               << vecCubeOps << "C " << vecVecOps << "V\n";

  // Step 5: Strip wrong-type ops from each scope using BFS-propagated classification
  stripWrongTypeOps(cubeScope, EngineType::CUBE, cubeBodyClassification);
  stripWrongTypeOps(vecScope, EngineType::VECTOR, vecBodyClassification);

  // Step 6: Hoist convert_layout ops out of the CUBE scope's loop.
  // These are view reshapes on L1 buffers (NZ→ND) that don't depend on loop
  // iteration state — they can be computed once before the loop starts.
  cubeScope.walk([&](scf::ForOp forOp) {
    Block *loopBody = forOp.getBody();
    SmallVector<hivm::ConvertLayoutOp> toHoist;
    SmallVector<memref::MemorySpaceCastOp> castsToHoist;
    for (Operation &op : *loopBody) {
      if (auto cvtOp = dyn_cast<hivm::ConvertLayoutOp>(&op)) {
        // Only hoist if its input is defined OUTSIDE the loop (shared L1 alloc)
        Value input = cvtOp.getOperand(0);
        if (input.getDefiningOp() &&
            input.getDefiningOp()->getBlock() != loopBody) {
          toHoist.push_back(cvtOp);
        }
      }
    }
    // Also hoist memory_space_cast that directly consumes a hoisted convert_layout
    for (auto cvtOp : toHoist) {
      for (Operation *user : cvtOp.getResult().getUsers()) {
        if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(user)) {
          if (castOp->getBlock() == loopBody)
            castsToHoist.push_back(castOp);
        }
      }
    }
    // Move them before the loop (inside the scope block, before the scf.for)
    for (auto cvtOp : toHoist)
      cvtOp->moveBefore(forOp);
    for (auto castOp : castsToHoist)
      castOp->moveBefore(forOp);
  });

  // Step 6b: Sink each cube matmul's operand load chain to immediately before
  // the matmul. The BFS level scheduler (stage 7) groups every unrolled K/V
  // load at the top of the cube loop body, so all 8 cbuf staging buffers stay
  // simultaneously live. PlanMemory then assigns them high L1 offsets and the
  // matmul's FB operand load falls back to register offset (mode 2), which the
  // simulator's dmamov_decode_to_fb path rejects. Interleaving the loads (the
  // manual kernel allocates one K tile right before each matmul) keeps only the
  // in-flight operands live -> low static offsets -> immediate offset mode.
  sinkCubeLoadChainsToMatmul(cubeScope);

  // Step 7: ROW_SPLIT re-tile of the VECTOR scope (16 rows per veccore, both
  // veccores active). Replaces the single-veccore NO_DUAL guard.
  retileVectorScopeForRowSplit(vecScope);

  // Stage 8: (experimental, default OFF) wrap pure elementwise vector compute
  // in vector_mode="simd" scopes. MEASURED RESULT (N=8192, 1 core): this is a
  // NO-OP — BiShengIR emits byte-for-byte identical microcode (same 221,285
  // cycles, same 57250 RV_VLDI / 35939 RV_VSTI) with or without these scopes,
  // and also with --enable-vf-fusion=true --vf-fusion-mode=ub-aware-op on top.
  // Reason: the bottleneck is register reuse, not vectorisation per se — the
  // whole-tensor (16xN) ops spill every intermediate to UB. Only the reference
  // per-row scf.for softmax-core (a row stays in VF registers across
  // mul→max→sub→exp→sum) collapses the loads. Kept as scaffolding for that
  // future per-row transform; enable with TRITON_CVSPLIT_SIMD=1.
  bool enableSimd = false;
  if (const char *e = std::getenv("TRITON_CVSPLIT_SIMD"))
    enableSimd = (StringRef(e) == "1" || StringRef(e) == "true");
  bool enableRowLoop = false;
  if (const char *e = std::getenv("TRITON_CVSPLIT_ROWLOOP"))
    enableRowLoop = (StringRef(e) == "1" || StringRef(e) == "true");
  // Stage 8 and stage 8b are ALTERNATIVE vectorisation strategies: stage 8
  // wraps whole-tensor [M,N] elementwise runs in simd scopes, while stage 8b
  // (row-loopify) rewrites the same runs into per-row [1,N] scf.for loops and
  // does its OWN simd-scope wrapping. Running both buries the arith ops inside
  // stage-8 scopes where row-loopify's block-level segmentation can't see them
  // (only ~1 loop emitted), leaving the softmax whole-tensor [64,N] -> the UB
  // overflow. When row-loopify is on, skip the whole-tensor wrap so the per-row
  // form (tiny [1,N] UB temps, matching the manual) wins.
  if (enableSimd && !enableRowLoop)
    wrapSimdScopes(vecScope);

  // Stage 8b: (experimental, default OFF) rewrite the softmax elementwise+reduce
  // runs as per-row scf.for loops wrapped in vector_mode="simd" scopes (the
  // target_optimized.ir shape) so each row stays VF-register-resident across
  // the sub->exp->reduce chain instead of spilling to UB. Enable with
  // TRITON_CVSPLIT_ROWLOOP=1.
  if (enableRowLoop) {
    // De-pipeline the (software-pipelined) vector scope into per-instance order
    // first, so each instance's softmax is one contiguous segment and its
    // qk_scale stays internal to a single simd scope (bounded UB live set).
    serializeVectorScopeByInstance(vecScope);
        rowLoopifyVectorScope(vecScope);
        // Replace the V->C NZ-pack transpose (a permuting DMA the simulator
        // rejects) with a per-row contiguous insert loop, matching the manual.
        rewriteNZTransposePacks(vecScope);
        // Merge the per-stage softmax + pack simd scopes into one scope per
        // cube/vector handoff (manual emits ~10 scopes; without this we emit
        // ~15, which makes SplitMixKernel mis-tag a cube load into the AIV
        // clone -> degenerate MOV_SRC_TO_FB / fixp_addr_misal at runtime).
        coalesceAdjacentSimdScopes(vecScope);
        // Pin the loop-carried accumulator (acc) into a dedicated mem_unique UB
        // buffer with per-instance bind_buffer read/modify/write, instead of
        // threading it as a live scf.for iter_arg. This matches the manual
        // reference (alloc_4): mem_unique is a whole-function allocation signal
        // that pins PlanMemory to a deterministic, fixed-offset layout shared
        // by the AIC/AIV clones, keeping the cube-side matmul L1->FB operand
        // load immediate-encodable (mode 0/1) rather than register-indirect
        // (mode 2 -> simulator dmamov_decode_to_fb assert). Escape:
        // TRITON_CVSPLIT_HOIST_STATE=1 (opt-in).
        hoistVectorStateToMemUnique(vecScope);
      }

  llvm::errs() << "[cv-split] Scope separation done: CUBE scope then VECTOR scope "
               << "(inside parent block, matching reference pattern)\n";
}

// ============================================================================
// ROW_SPLIT vector re-tile.
// The fixpipe ROW_SPLIT delivers 16 rows (M/2) to each veccore's UB. The VECTOR
// scope was cloned at the original 32-row tile size, so we re-tile its whole
// softmax DAG to 16 rows per veccore:
//   - leading (M) dimension 32 -> 16 on all vector tensors / UB memrefs
//   - external vector-only init constants (fills/empties) halved
//   - V->C P pack rebuilt as 16x32 -> 2x1x16x16, copied into the veccore's
//     subview [0, sub_block_idx, 0, 0] of the shared 2x2x16x16 L1 buffer
//   - output store offset += sub_block_idx * 16 * leadingStride, sizes M=16
// Both veccores then do useful work (2x vector throughput), matching the target.
// ============================================================================
// Halve the leading (M) dim of a [Mfull, ..] tensor/memref to Mfull/2 (the
// per-veccore band under ROW_SPLIT). Generic over the tile size: BLOCK_M=32 ->
// 16, BLOCK_M=64 -> 32, BLOCK_M=128 -> 64 (matching target_optimized.ir, whose
// vector scope runs at BLOCK_M/2 = 64 rows). Types whose leading dim != Mfull
// pass through unchanged.
static Type retileRowHalve(Type t, int64_t Mfull) {
  int64_t half = Mfull / 2;
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    auto sh = rt.getShape();
    if (!sh.empty() && sh[0] == Mfull) {
      SmallVector<int64_t> ns(sh.begin(), sh.end());
      ns[0] = half;
      return RankedTensorType::get(ns, rt.getElementType());
    }
  } else if (auto mt = dyn_cast<MemRefType>(t)) {
    auto sh = mt.getShape();
    if (!sh.empty() && sh[0] == Mfull) {
      SmallVector<int64_t> ns(sh.begin(), sh.end());
      ns[0] = half;
      return MemRefType::get(ns, mt.getElementType(), mt.getLayout(),
                             mt.getMemorySpace());
    }
  }
  return t;
}

// Detect the full tile row count (BLOCK_M) of a freshly-cloned VECTOR scope:
// the leading dim of the final output store source ([BLOCK_M, HEAD_DIM]).
// Falls back to the max rank>=2 leading dim, then to 32. Must run BEFORE retile.
static int64_t detectTileRows(scope::ScopeOp vecScope) {
  int64_t M = 0;
  vecScope.walk([&](bufferization::MaterializeInDestinationOp m) {
    if (auto rt = dyn_cast<RankedTensorType>(m.getSource().getType()))
      if (rt.getRank() >= 1 && !rt.isDynamicDim(0))
        M = std::max(M, rt.getShape()[0]);
  });
  if (M == 0)
    vecScope.walk([&](Operation *op) {
      for (Value r : op->getResults())
        if (auto rt = dyn_cast<RankedTensorType>(r.getType()))
          if (rt.getRank() >= 2 && !rt.isDynamicDim(0))
            M = std::max(M, rt.getShape()[0]);
    });
  return M ? M : 32;
}

// V->C pack chain detached in step 2 and rebuilt per-veccore in step 6: the
// softmax P tensor (pSrc), the shared L1 destination (l1Alloc), and the op the
// rebuilt pack is inserted before (anchor).
struct VectorToCubePack {
  Value pSrc;
  Value l1Alloc;
  Operation *anchor;
};

// Step 1: emit get_sub_block_idx at the top of the scope and return it as an
// index value (0 or 1 — which of the core's two veccores is executing).
static Value emitSubBlockIndex(scope::ScopeOp vecScope, Location loc) {
  Block &block = vecScope.getBodyRegion().front();
  OpBuilder topB(&block, block.begin());
  auto sbid = topB.create<hivm::GetSubBlockIdxOp>(loc, topB.getI64Type());
  return topB
      .create<arith::IndexCastOp>(loc, topB.getIndexType(), sbid.getResult())
      .getResult();
}

// Step 2: detach the whole-tile V->C pack chain
// (truncf -> reshape -> transpose -> reshape -> to_memref -> cast -> copy),
// recording each P source / L1 alloc / insertion anchor so step 6 can rebuild a
// per-veccore pack. The truncf (the actual P value) is kept; the rest is erased.
static SmallVector<VectorToCubePack>
detachVectorToCubePacks(scope::ScopeOp vecScope) {
  Block &block = vecScope.getBodyRegion().front();
  SmallVector<VectorToCubePack> packs;
  SmallVector<Operation *> toErase;
  block.walk([&](hivm::CopyOp copy) {
    Value src = copy.getOperand(0);   // UB memspacecast
    Value dst = copy.getOperand(1);   // L1 alloc
    Operation *anchor = copy->getNextNode();
    SmallVector<Operation *> chain;
    chain.push_back(copy);
    Operation *cur = src.getDefiningOp(); // memspacecast
    Value pSrc;
    // walk: memspacecast <- to_memref <- reshape2 <- transpose <- reshape1 <- truncf
    while (cur) {
      chain.push_back(cur);
      if (isa<arith::TruncFOp>(cur)) { pSrc = cur->getResult(0); break; }
      if (cur->getNumOperands() == 0) break;
      cur = cur->getOperand(0).getDefiningOp();
    }
    // pSrc is the truncf result; keep truncf (pop it from erase list)
    if (pSrc && isa<arith::TruncFOp>(chain.back()))
      chain.pop_back();
    packs.push_back({pSrc, dst, anchor});
    for (auto *o : chain) toErase.push_back(o);
  });
  for (Operation *o : toErase) {
    o->dropAllUses();
    o->erase();
  }
  return packs;
}

// Step 3: clone function-level 32-row init fills/empties as 16-row and rewrite
// only the vector-scope uses (these inits are shared with the CUBE scope, so we
// must not retile them in place). Returns the number of clones created.
static unsigned cloneExternalInitsAsHalfHeight(scope::ScopeOp vecScope,
                                               Location loc, int64_t Mfull) {
  // These are shared with the CUBE scope (e.g. a 32x32 empty feeds both the
  // matmul init and the vector scale fill), so we must NOT retile them in
  // place; instead clone a 16-row version and rewrite only the vector uses.
  DenseSet<Operation *> vecOps;
  vecScope.walk([&](Operation *o) { vecOps.insert(o); });
  OpBuilder cb(vecScope);
  DenseMap<Value, Value> cloneMap;
  unsigned clonedCount = 0;
  vecScope.walk([&](Operation *op) {
    for (OpOperand &opd : op->getOpOperands()) {
      Value v = opd.get();
      Operation *d = v.getDefiningOp();
      if (!d || vecOps.count(d)) continue;            // external defs only
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt || rt.getRank() == 0 || rt.getShape()[0] != Mfull) continue;
      auto it = cloneMap.find(v);
      Value repl;
      if (it != cloneMap.end()) {
        repl = it->second;
      } else {
        auto ntt = cast<RankedTensorType>(retileRowHalve(rt, Mfull));
        if (auto fill = dyn_cast<linalg::FillOp>(d)) {
          Value ne = cb.create<tensor::EmptyOp>(loc, ntt.getShape(), ntt.getElementType());
          repl = cb.create<linalg::FillOp>(loc, fill.getInputs(),
                                           ValueRange{ne}).getResult(0);
        } else if (isa<tensor::EmptyOp>(d)) {
          repl = cb.create<tensor::EmptyOp>(loc, ntt.getShape(), ntt.getElementType());
        } else {
          continue;
        }
        cloneMap[v] = repl;
        ++clonedCount;
      }
      opd.set(repl);
    }
  });
  return clonedCount;
}

// Steps 4 & 4.5: retile every in-scope op result (and loop-carried arg) from 32
// to 16 rows, then repair any DPS tensor.empty init whose type drifted from its
// retiled result. (arith.constant splats get their value attr rebuilt;
// reinterpret_cast is handled in steps 5/6 and skipped here.)
static void retileVectorScopeOps(scope::ScopeOp vecScope, int64_t Mfull) {
  // ---- 4. Generic re-tile of every vector op (skip reinterpret_cast) ----
  vecScope.walk([&](Operation *op) {
    if (isa<memref::ReinterpretCastOp>(op)) return;
    // arith.constant: rebuild the splat value attr to match the retiled type
    // (e.g. the dead PV zero-init clone left in the vector scope).
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      Type nt = retileRowHalve(c.getType(), Mfull);
      if (nt != c.getType()) {
        if (auto dense = dyn_cast<DenseElementsAttr>(c.getValue())) {
          if (dense.isSplat()) {
            auto nst = cast<ShapedType>(nt);
            c.setValueAttr(DenseElementsAttr::get(nst, dense.getSplatValue<Attribute>()));
            c.getResult().setType(nt);
          }
        }
      }
      return;
    }
    for (Value r : op->getResults())
      r.setType(retileRowHalve(r.getType(), Mfull));
    if (auto f = dyn_cast<scf::ForOp>(op)) {
      Block *b = f.getBody();
      for (unsigned i = 1; i < b->getNumArguments(); ++i)
        b->getArgument(i).setType(
            retileRowHalve(b->getArgument(i).getType(), Mfull));
    }
  });

  // ---- 4.5. Safety net: fix any in-scope DPS *empty* init whose type no longer
  // matches its (retiled) result (a tensor.empty is a pure destination, so a
  // fresh correctly-typed empty is always safe; do NOT touch fill/reduce inits
  // which carry meaningful identity values). ----
  vecScope.walk([&](Operation *op) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dps) return;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      OpOperand *io = dps.getDpsInitOperand(i);
      Value init = io->get();
      if (!init.getDefiningOp<tensor::EmptyOp>()) continue;
      Type rt = op->getResult(i).getType();
      if (init.getType() == rt) continue;
      auto rtt = cast<RankedTensorType>(rt);
      OpBuilder b(op);
      io->set(b.create<tensor::EmptyOp>(op->getLoc(), rtt.getShape(),
                                        rtt.getElementType()).getResult());
    }
  });
}

// Step 5: shift each output store to this veccore's 16-row band — offset +=
// sub_block_idx * 16 * leadingStride, size M = 16. Returns the store count.
static unsigned retileOutputStores(scope::ScopeOp vecScope, Value sbidx,
                                   Location loc, int64_t Mfull) {
  int64_t half = Mfull / 2;
  // ---- 5. Output stores: offset += sbid*half*leadingStride, sizes M=half ----
  SmallVector<bufferization::MaterializeInDestinationOp> mats;
  vecScope.walk([&](bufferization::MaterializeInDestinationOp m) { mats.push_back(m); });
  for (auto m : mats) {
    auto ric = m.getDest().getDefiningOp<memref::ReinterpretCastOp>();
    if (!ric) continue;
    // Build the per-veccore reinterpret_cast right before the materialize (it is
    // inside the scope, so sub_block_idx dominates it). The original ric may be
    // a loop-invariant op hoisted into the parent block, where sbid is not in
    // scope.
    OpBuilder b(m);
    auto offsets = ric.getMixedOffsets();
    auto sizes = ric.getMixedSizes();
    auto strides = ric.getMixedStrides();
    int64_t leadStride = 1;
    if (auto sAttr = dyn_cast<Attribute>(strides[0]))
      leadStride = cast<IntegerAttr>(sAttr).getInt();
    Value origOff = getValueOrCreateConstantIndexOp(b, loc, offsets[0]);
    Value step = b.create<arith::ConstantIndexOp>(loc, half * leadStride);
    Value add = b.create<arith::MulIOp>(loc, sbidx, step);
    Value newOff = b.create<arith::AddIOp>(loc, origOff, add);
    sizes[0] = b.getIndexAttr(half);
    auto newType = cast<MemRefType>(retileRowHalve(ric.getType(), Mfull));
    auto newRic = b.create<memref::ReinterpretCastOp>(
        loc, newType, ric.getSource(), getAsOpFoldResult(newOff), sizes, strides);
    ric.replaceAllUsesWith(newRic.getResult());
    ric.erase();
  }
  return mats.size();
}

// Step 6: rebuild each detached V->C pack as a per-veccore pack: 16x32 ->
// 2x1x16x16, copied into this veccore's subview [0, sub_block_idx, 0, 0] of the
// shared L1 buffer.
static void rebuildVectorToCubePacks(ArrayRef<VectorToCubePack> packs,
                                     Value sbidx,
                                     hivm::AddressSpaceAttr ubAddrSpace,
                                     Location loc) {
  // ---- 6. Regenerate V->C packs: 16x32 -> 2x1x16x16 -> subview[0,sbid,0,0] ----
  for (auto &p : packs) {
    if (!p.pSrc) continue;
    auto pType = cast<RankedTensorType>(p.pSrc.getType()); // 16x32xf16
    int64_t M = pType.getShape()[0];     // 16
    int64_t N = pType.getShape()[1];     // 32
    int64_t N16 = N / 16, M16 = M / 16;  // 2, 1
    Type elemType = pType.getElementType();
    OpBuilder b(p.anchor);
    auto i64Ty = b.getI64Type();
    // reshape [M,N] -> [M, N16, 16]
    auto s3Type = RankedTensorType::get({3}, i64Ty);
    auto s3 = b.create<arith::ConstantOp>(loc, s3Type,
        DenseElementsAttr::get(s3Type, ArrayRef<int64_t>{M, N16, 16}));
    auto resh1Type = RankedTensorType::get({M, N16, 16}, elemType);
    auto resh1 = b.create<tensor::ReshapeOp>(loc, resh1Type, p.pSrc, s3.getResult());
    // transpose [M,N16,16] -> [N16,M,16]
    auto emptyT = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{N16, M, 16}, elemType);
    auto transp = b.create<linalg::TransposeOp>(loc, resh1.getResult(),
        emptyT.getResult(), ArrayRef<int64_t>{1, 0, 2});
    // reshape [N16,M,16] -> [N16,M16,16,16]
    auto s4Type = RankedTensorType::get({4}, i64Ty);
    auto s4 = b.create<arith::ConstantOp>(loc, s4Type,
        DenseElementsAttr::get(s4Type, ArrayRef<int64_t>{N16, M16, 16, 16}));
    auto nzType = RankedTensorType::get({N16, M16, 16, 16}, elemType);
    auto resh2 = b.create<tensor::ReshapeOp>(loc, nzType, transp->getResult(0), s4.getResult());
    // to_memref + cast to UB
    auto memT = MemRefType::get({N16, M16, 16, 16}, elemType);
    auto toMem = b.create<bufferization::ToMemrefOp>(loc, memT, resh2.getResult());
    auto ubMemT = MemRefType::get({N16, M16, 16, 16}, elemType, nullptr, ubAddrSpace);
    auto cast = b.create<memref::MemorySpaceCastOp>(loc, ubMemT, toMem.getResult());
    // subview of L1 alloc [0, sbid*M16, 0, 0] [N16,M16,16,16]: each veccore owns
    // M16 = (rows/veccore)/16 fractal-row blocks, so veccore `sbid` writes the
    // band starting at block sbid*M16 (NOT bare sbid — that only coincided for
    // BLOCK_M=32 where M16==1, and overlapped/misaligned for BLOCK_M>=64).
    Value m16c = b.create<arith::ConstantIndexOp>(loc, M16);
    Value off1 = b.create<arith::MulIOp>(loc, sbidx, m16c);
    SmallVector<OpFoldResult, 4> offs{b.getIndexAttr(0), off1,
                                      b.getIndexAttr(0), b.getIndexAttr(0)};
    SmallVector<OpFoldResult, 4> szs{b.getIndexAttr(N16), b.getIndexAttr(M16),
                                     b.getIndexAttr(16), b.getIndexAttr(16)};
    SmallVector<OpFoldResult, 4> strs{b.getIndexAttr(1), b.getIndexAttr(1),
                                      b.getIndexAttr(1), b.getIndexAttr(1)};
    auto subview = b.create<memref::SubViewOp>(loc, p.l1Alloc, offs, szs, strs);
    b.create<hivm::CopyOp>(loc, mlir::TypeRange{}, cast.getResult(), subview.getResult());
  }
}

// Re-tile the VECTOR scope for ROW_SPLIT so both veccores do useful work (2x
// vector throughput): 16 rows per veccore, addressed by get_sub_block_idx,
// matching the target IR. Runs the six steps in order; see each helper.
static void retileVectorScopeForRowSplit(scope::ScopeOp vecScope) {
  Location loc = vecScope.getLoc();
  auto ubAddrSpace = OpBuilder(vecScope.getContext())
      .getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);

  // Detect the tile height (BLOCK_M) before any retile mutates the types, so we
  // retile Mfull -> Mfull/2 generically (32->16, 64->32, 128->64).
  int64_t Mfull = detectTileRows(vecScope);

  Value sbidx = emitSubBlockIndex(vecScope, loc);
  SmallVector<VectorToCubePack> packs = detachVectorToCubePacks(vecScope);
  unsigned clonedCount = cloneExternalInitsAsHalfHeight(vecScope, loc, Mfull);
  retileVectorScopeOps(vecScope, Mfull);
  unsigned nStores = retileOutputStores(vecScope, sbidx, loc, Mfull);
  rebuildVectorToCubePacks(packs, sbidx, ubAddrSpace, loc);

  llvm::errs() << "[cv-split]   ROW_SPLIT re-tile (BLOCK_M=" << Mfull << " -> "
               << (Mfull / 2) << "/veccore): " << packs.size()
               << " V->C packs, " << nStores << " stores, "
               << clonedCount << " ext consts cloned\n";
}

// ============================================================================
// Per-row loopification (TRITON_CVSPLIT_ROWLOOP, experimental, default OFF).
// ----------------------------------------------------------------------------
// The flat simd scopes above are a no-op because the bottleneck is register
// reuse, not vectorisation: our whole-tensor softmax spills every intermediate
// to UB (~57k RV_VLDI + ~36k RV_VSTI). The reference (target_optimized.ir)
// instead runs the softmax core as a per-row `scf.for %row = 0..M`: each
// iteration extract_slice's one row (tensor<1xN>), runs the whole
// sub->exp->reduce chain on that row register-resident, and insert_slice's the
// result back. This collapses the UB round-trips.
//
// buildRowLoop() rewrites one maximal elementwise(+reduce) run into that form:
//   acc_k = tensor.empty (full [M, ...] shape, one per live-out)
//   res_k = scf.for %r = 0..M iter_args(acc_k) {
//             <live-in>_row = extract_slice <live-in>[%r, ..]   (M-leading dims)
//             <clone run ops, every [M,..] type -> [1,..]>
//             acc_k = insert_slice <run result row> into acc_k[%r, ..]
//             yield acc_k
//           }
// then external uses of the run's live-outs are redirected to res_k.
// ============================================================================

static bool isVFEligibleSimd(Operation *op, bool includeReduce);

// Map a per-tile tensor type [M, ...] to its single-row form [1, ...].
static Type leadingDimTo1(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt || rt.getRank() == 0)
    return t;
  SmallVector<int64_t> sh(rt.getShape().begin(), rt.getShape().end());
  sh[0] = 1;
  return RankedTensorType::get(sh, rt.getElementType());
}

// True if `v` is a tensor whose leading dim is exactly M (a "row tensor" we
// slice per iteration). Scalars / [1,..] broadcasts / non-M tensors pass
// through unchanged.
static bool isRowTensor(Value v, int64_t M) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  return rt && rt.getRank() >= 1 && rt.getShape()[0] == M;
}

// extract_slice / insert_slice offset-size-stride for row %iv of a [M, ..] type.
static void rowSlice(OpBuilder &b, RankedTensorType rt, Value iv,
                     SmallVector<OpFoldResult> &offsets,
                     SmallVector<OpFoldResult> &sizes,
                     SmallVector<OpFoldResult> &strides) {
  offsets.assign(rt.getRank(), b.getIndexAttr(0));
  offsets[0] = iv;
  sizes.clear();
  for (int64_t d : rt.getShape())
    sizes.push_back(b.getIndexAttr(d));
  sizes[0] = b.getIndexAttr(1);
  strides.assign(rt.getRank(), b.getIndexAttr(1));
}

// Determine the common row count M of a run (the shared leading dim of its
// row-tensor results). Returns 0 if the run has no consistent static M > 1.
static int64_t runRowCount(ArrayRef<Operation *> run) {
  int64_t M = 0;
  for (Operation *o : run)
    for (Value r : o->getResults()) {
      auto rt = dyn_cast<RankedTensorType>(r.getType());
      if (!rt || rt.getRank() == 0 || rt.isDynamicDim(0))
        continue;
      int64_t d0 = rt.getShape()[0];
      if (d0 <= 1)
        continue;
      if (M == 0)
        M = d0;
      else if (M != d0)
        return 0; // inconsistent leading dims -> not safely loopifiable
    }
  return M;
}

// Rewrite one run into a per-row scf.for. Returns the new loop (or nullptr if
// the run is not loopifiable). Erases the original run ops.
static scf::ForOp buildRowLoop(ArrayRef<Operation *> run, int64_t M) {
  DenseSet<Operation *> runSet(run.begin(), run.end());
  Location loc = run.front()->getLoc();
  OpBuilder b(run.front());

  // Live-outs: run results consumed outside the run (program order).
  SmallVector<Value> liveOuts;
  for (Operation *o : run)
    for (Value res : o->getResults()) {
      for (OpOperand &u : res.getUses())
        if (!runSet.count(u.getOwner())) {
          liveOuts.push_back(res);
          break;
        }
    }
  if (liveOuts.empty())
    return nullptr;

  // One full-shape accumulator per live-out.
  SmallVector<Value> initArgs;
  for (Value v : liveOuts) {
    auto rt = cast<RankedTensorType>(v.getType());
    initArgs.push_back(
        b.create<tensor::EmptyOp>(loc, rt.getShape(), rt.getElementType()));
  }

  // Target uses an i32 induction var + arith.index_cast (not an index loop) so
  // BiShengIR's VF layout analysis recognises the per-row loop; the casted
  // index value is used for the extract/insert_slice offsets. See bishengir
  // test Dialect/HIVMAVE/vector-layout-analyze-1.mlir (scf.for : i32 + cast).
  Value lb = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value ub = b.create<arith::ConstantIntOp>(loc, M, 32);
  Value step = b.create<arith::ConstantIntOp>(loc, 1, 32);
  auto forOp = b.create<scf::ForOp>(loc, lb, ub, step, initArgs);

  OpBuilder bb(forOp.getBody(), forOp.getBody()->begin());
  Value iv = bb.create<arith::IndexCastOp>(loc, bb.getIndexType(),
                                           forOp.getInductionVar());

  // Slice every row-tensor live-in to [1, ..]; pass everything else through.
  IRMapping map;
  DenseSet<Value> handled;
  for (Operation *o : run)
    for (Value opd : o->getOperands()) {
      if (!handled.insert(opd).second)
        continue;
      Operation *d = opd.getDefiningOp();
      if (d && runSet.count(d))
        continue; // produced inside the run
      if (!isRowTensor(opd, M))
        continue; // scalar / broadcast-invariant -> used as-is
      auto rt = cast<RankedTensorType>(opd.getType());
      SmallVector<OpFoldResult> offs, sizes, strides;
      rowSlice(bb, rt, iv, offs, sizes, strides);
      map.map(opd, bb.create<tensor::ExtractSliceOp>(loc, opd, offs, sizes, strides));
    }

  // Clone the run on the sliced operands, retyping [M,..] results to [1,..].
  for (Operation *o : run) {
    Operation *c = bb.clone(*o, map);
    for (Value r : c->getResults())
      r.setType(leadingDimTo1(r.getType()));
    for (auto [orig, cloned] : llvm::zip(o->getResults(), c->getResults()))
      map.map(orig, cloned);
  }

  // insert_slice each live-out row into its accumulator, then yield.
  SmallVector<Value> yields;
  for (auto [i, v] : llvm::enumerate(liveOuts)) {
    Value rowRes = map.lookup(v);
    Value acc = forOp.getRegionIterArg(i);
    auto rt = cast<RankedTensorType>(v.getType());
    SmallVector<OpFoldResult> offs, sizes, strides;
    rowSlice(bb, rt, iv, offs, sizes, strides);
    yields.push_back(
        bb.create<tensor::InsertSliceOp>(loc, rowRes, acc, offs, sizes, strides));
  }
  bb.create<scf::YieldOp>(loc, yields);

  // Redirect external uses to the loop results, then erase the original run.
  for (auto [i, v] : llvm::enumerate(liveOuts))
    v.replaceAllUsesWith(forOp.getResult(i));
  for (Operation *o : llvm::reverse(run))
    o->erase();

  return forOp;
}

// Wrap a row loop (and any ops it solely depends on within the block) in an
// outlined vector_mode="simd" scope, so BiShengIR lowers the row body to packed
// VF micro-ops. The loop's iter-arg inits stay outside the scope (live-ins).
static void wrapInSimdScope(scf::ForOp forOp) {
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  SmallVector<Type> retTypes(forOp.getResultTypes().begin(),
                             forOp.getResultTypes().end());
  auto scopeOp = b.create<scope::ScopeOp>(loc, TypeRange(retTypes));
  Block *body = &scopeOp.getBodyRegion().emplaceBlock();

  forOp->moveBefore(body, body->end());
  OpBuilder rb(body, body->end());
  rb.create<scope::ReturnOp>(loc, forOp.getResults());

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceUsesWithIf(
        scopeOp.getResult(i), [&](OpOperand &u) {
          return !scopeOp->isProperAncestor(u.getOwner());
        });

  scopeOp->setAttr("vector_mode", StringAttr::get(ctx, "simd"));
  scopeOp->setAttr("outline", BoolAttr::get(ctx, true));
  scopeOp->setAttr("noinline", UnitAttr::get(ctx));
}

// Wrap a contiguous range of ops [firstOp, anchorOp) in ONE outlined
// vector_mode="simd" scope (so all the segment's per-row loops + whole-tensor
// narrow ops share a single VF scope, exactly like target_optimized.ir's
// softmax-core scope; values used after the range become scope results).
static void wrapRangeInSimdScope(Operation *firstOp, Operation *anchorOp) {
  MLIRContext *ctx = firstOp->getContext();
  Location loc = firstOp->getLoc();
  SmallVector<Operation *> ops;
  for (Operation *o = firstOp; o && o != anchorOp; o = o->getNextNode())
    ops.push_back(o);
  if (ops.empty())
    return;
  DenseSet<Operation *> opSet(ops.begin(), ops.end());

  SmallVector<Value> outs;
  for (Operation *o : ops)
    for (Value r : o->getResults())
      for (OpOperand &u : r.getUses())
        if (!opSet.count(u.getOwner())) {
          outs.push_back(r);
          break;
        }

  OpBuilder b(firstOp);
  SmallVector<Type> retTypes;
  for (Value v : outs)
    retTypes.push_back(v.getType());
  auto scopeOp = b.create<scope::ScopeOp>(loc, TypeRange(retTypes));
  Block *body = &scopeOp.getBodyRegion().emplaceBlock();
  for (Operation *o : ops)
    o->moveBefore(body, body->end());
  OpBuilder rb(body, body->end());
  rb.create<scope::ReturnOp>(loc, outs);
  for (auto [i, v] : llvm::enumerate(outs))
    v.replaceUsesWithIf(scopeOp.getResult(i), [&](OpOperand &u) {
      return !scopeOp->isProperAncestor(u.getOwner());
    });
  scopeOp->setAttr("vector_mode", StringAttr::get(ctx, "simd"));
  scopeOp->setAttr("outline", BoolAttr::get(ctx, true));
  scopeOp->setAttr("noinline", UnitAttr::get(ctx));
}

// ============================================================================
// Coalesce adjacent sibling simd scopes into one (match the manual's flat
// VECTOR layout).
//
// row-loopify (stage 8b) + the NZ-pack rewrite each emit a *separate*
// vector_mode="simd" scope per softmax stage and per pack copy, so our VECTOR
// scope ends up with ~15 nested simd scopes vs the manual's ~10. The manual
// builds the packed P *inside* the softmax scope's second loop, so each
// cube/vector handoff is one contiguous simd scope. The extra nesting makes the
// downstream MIX split (CVPipelining -> SplitMixKernel) mis-tag a cube
// matmul-operand load into the AIV clone -> a nullified, degenerate
// MOV_SRC_TO_FB (XD=XN=XM, burst=0) that the simulator faults on as
// fixp_addr_misal. Merging runs of adjacent simd scopes (separated only by pure
// tensor filler ops) back into one scope reproduces the manual's flat structure
// so the split tags cleanly.
// ============================================================================
static bool isSimdScope(Operation *op) {
  auto s = dyn_cast<scope::ScopeOp>(op);
  return s && s->hasAttr("vector_mode");
}

// Pure tensor-domain ops that may be absorbed between two simd scopes when
// coalescing (no memory effects, tensor results only). Anything else
// (memref.*, hivm.*, bufferization.to_tensor/to_memref, bare scf.for, sync,
// copy, fixpipe, convert_layout, memory_space_cast) is a hard boundary that
// must stay outside / between scopes.
static bool isCoalesceFiller(Operation *op) {
  if (isa<tensor::EmptyOp, tensor::ReshapeOp, tensor::ExtractSliceOp,
          tensor::InsertSliceOp, tensor::CollapseShapeOp,
          tensor::ExpandShapeOp, linalg::FillOp, linalg::BroadcastOp,
          linalg::ReduceOp, linalg::TransposeOp>(op))
    return true;
  if (Dialect *d = op->getDialect()) {
    StringRef ns = d->getNamespace();
    if (ns == "arith" || ns == "math")
      return true;
  }
  return false;
}

// Flatten a simd scope back into its parent block: rewire each result to the
// inner returned value, move the body ops (sans terminator) before the scope,
// then erase the now-empty scope. SSA stays valid because the body ops were
// already ordered before the scope's return and the scope's results were only
// used after the scope.
static void unwrapScope(scope::ScopeOp s) {
  Block &body = s.getBodyRegion().front();
  auto ret = cast<scope::ReturnOp>(body.getTerminator());
  for (unsigned k = 0; k < s.getNumResults(); ++k)
    s.getResult(k).replaceAllUsesWith(ret.getOperand(k));
  while (&body.front() != ret)
    body.front().moveBefore(s);
  ret.erase();
  s.erase();
}

static void coalesceAdjacentSimdScopes(scope::ScopeOp vecScope) {
  SmallVector<Block *> blocks;
  blocks.push_back(&vecScope.getBodyRegion().front());
  vecScope.walk([&](scf::ForOp f) { blocks.push_back(f.getBody()); });

  unsigned merged = 0;
  for (Block *blk : blocks) {
    // Re-scan from the front after each merge (the moves/erases invalidate the
    // remaining iteration order).
    bool changed = true;
    while (changed) {
      changed = false;
      Operation *runStart = nullptr;
      Operation *lastScope = nullptr;
      for (Operation &op : *blk) {
        if (isSimdScope(&op)) {
          if (!runStart)
            runStart = &op;
          lastScope = &op;
          continue;
        }
        if (runStart && isCoalesceFiller(&op))
          continue; // tentative filler between scopes
        // Hard boundary: close the pending run.
        if (runStart && lastScope != runStart)
          break; // mergeable run [runStart .. lastScope] found
        runStart = nullptr;
        lastScope = nullptr;
      }
      if (!runStart || !lastScope || lastScope == runStart)
        continue;

      Operation *anchorAfter = lastScope->getNextNode();
      Operation *beforeRun = runStart->getPrevNode();
      SmallVector<scope::ScopeOp> toUnwrap;
      for (Operation *o = runStart; o && o != anchorAfter;
           o = o->getNextNode())
        if (auto s = dyn_cast<scope::ScopeOp>(o))
          if (s->hasAttr("vector_mode"))
            toUnwrap.push_back(s);
      for (scope::ScopeOp s : toUnwrap)
        unwrapScope(s);
      Operation *first = beforeRun ? beforeRun->getNextNode() : &blk->front();
      wrapRangeInSimdScope(first, anchorAfter);
      ++merged;
      changed = true;
    }
  }
  llvm::errs() << "[cv-split]   coalesced " << merged
               << " adjacent simd-scope run(s)\n";
}

// ============================================================================
// De-pipeline (serialize) the VECTOR scope by QK instance.
//
// The upstream `.ttadapter` for the unrolled FA inner loop is software-
// pipelined: all K instances' `qk*scale` tiles are computed up front and each
// is consumed (subtract/exp) many ops later, so every instance's full-tile
// `qk_scale` is simultaneously live. Combined with the depth-2 ping/pong on the
// shared `qk_ub` buffer (alloc reused by instance i and i+2), this both blows
// the UB budget (K live [M,N]f32 tiles) and makes rematerialising the scale
// unsafe (the buffer holds a newer instance's data by the time the late
// consumer runs).
//
// The manual reference (target_optimized.ir) is instead strictly per-instance:
// instance i's entire softmax (scale, rowmax, m-update, sub, exp, pack, rowsum)
// runs and fully consumes qk_ub_i before instance i+1 touches anything. That
// keeps `qk_scale` internal to one simd scope per instance, so PlanMemory
// reuses a single physical scratch across all instances.
//
// We reproduce that by re-serialising the loop body so each instance's ops are
// contiguous, in instance order. We assign every op an instance index (the QK
// transfer it derives from) and stable-sort by (instance, original position).
// Because instanceOf(op) = max(instanceOf(operands)) and seeds flow only
// downstream, the sort is a valid topological order: data dependencies are
// preserved and full serialisation only *adds* ordering already implied by the
// cross-engine sync flags, so it cannot introduce a sync hazard.
// ============================================================================
static void serializeVectorScopeByInstance(scope::ScopeOp vecScope) {
  bool enabled = true;
  if (const char *e = std::getenv("TRITON_CVSPLIT_SERIALIZE"))
    enabled = !(StringRef(e) == "0" || StringRef(e) == "false");
  if (!enabled)
    return;

  // The pipelined softmax lives at the top level of the KV-loop body block(s).
  SmallVector<Block *> blocks;
  vecScope.walk([&](scf::ForOp f) { blocks.push_back(f.getBody()); });

  auto flagOf = [](Operation *op) -> int {
    if (auto w = dyn_cast<hivm::SyncBlockWaitOp>(op))
      if (auto a = w.getStaticFlagIdAttr())
        return (int)a.getInt();
    if (auto s = dyn_cast<hivm::SyncBlockSetOp>(op))
      if (auto a = s.getStaticFlagIdAttr())
        return (int)a.getInt();
    return -1;
  };

  unsigned serialized = 0;
  for (Block *blk : blocks) {
    // Only act on a genuinely pipelined region (>=2 cross-engine waits).
    int nWait = 0;
    for (Operation &op : *blk)
      if (isa<hivm::SyncBlockWaitOp>(op))
        ++nWait;
    if (nWait < 2)
      continue;

    Operation *term = blk->getTerminator();
    SmallVector<Operation *> ops;
    for (Operation &op : *blk)
      if (&op != term)
        ops.push_back(&op);

    DenseMap<Operation *, int> inst;

    // Pass A: seed instances from the cross-engine sync flags. In the VECTOR
    // scope every sync_block_wait is a C->V wait and every sync_block_set is a
    // V->C set; the static flag id equals the transfer (instance) index. The
    // QK tensor is read by the bufferization.to_tensor immediately following
    // its wait, so propagate that wait's flag onto the next plain to_tensor.
    int pendingSeed = -1;
    for (Operation *op : ops) {
      if (isa<hivm::SyncBlockWaitOp>(op)) {
        int f = flagOf(op);
        if (f >= 0) { inst[op] = f; pendingSeed = f; }
        continue;
      }
      if (isa<hivm::SyncBlockSetOp>(op)) {
        int f = flagOf(op);
        if (f >= 0) inst[op] = f;
        continue;
      }
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(op)) {
        bool hasTensorOperand = false;
        for (Value v : op->getOperands())
          if (isa<RankedTensorType>(v.getType())) { hasTensorOperand = true; break; }
        if (!hasTensorOperand && pendingSeed >= 0) {
          inst[op] = pendingSeed;
          pendingSeed = -1;
        }
      }
    }

    // Pass B: propagate downstream. SSA defs precede uses inside a block, so a
    // single forward sweep computes instanceOf = max(seed, operand instances).
    DenseSet<Operation *> opSet(ops.begin(), ops.end());
    for (Operation *op : ops) {
      int s = inst.count(op) ? inst[op] : 0;
      for (Value v : op->getOperands())
        if (Operation *d = v.getDefiningOp())
          if (opSet.count(d))
            s = std::max(s, inst.lookup(d));
      inst[op] = s;
    }

    // Priority topological sort: emit by (instance, original order) but never
    // before an in-block operand. A plain sort-by-instance can place an op
    // ahead of a value it consumes (e.g. a scope whose instance is lower than
    // an accumulator to_tensor it reads, which got seeded by the preceding
    // wait flag), which violates SSA dominance. Kahn's algorithm with a
    // (instance, originalIndex) priority preserves the per-instance grouping
    // wherever the data dependencies allow it and stays valid everywhere else.
    DenseMap<Operation *, unsigned> idxOf;
    for (auto [i, op] : llvm::enumerate(ops))
      idxOf[op] = (unsigned)i;

    DenseMap<Operation *, int> indeg;
    DenseMap<Operation *, SmallVector<Operation *>> succs;
    for (Operation *op : ops)
      indeg[op] = 0;
    for (Operation *op : ops) {
      DenseSet<Operation *> preds;
      // Direct operands AND values captured inside nested regions. scope.scope
      // ops take no operands but their bodies implicitly reference block values
      // (e.g. the accumulator to_tensor a rescale scope consumes); missing those
      // edges lets the scope be ordered before the value it reads.
      auto addOperand = [&](Value v) {
        if (Operation *d = v.getDefiningOp())
          if (opSet.count(d) && d != op)
            preds.insert(d);
      };
      for (Value v : op->getOperands())
        addOperand(v);
      op->walk([&](Operation *nested) {
        for (Value v : nested->getOperands())
          addOperand(v);
      });
      for (Operation *d : preds) {
        succs[d].push_back(op);
        ++indeg[op];
      }
    }

    auto cmp = [&](Operation *a, Operation *b) {
      int ia = inst.lookup(a), ib = inst.lookup(b);
      if (ia != ib) return ia > ib;          // min-heap on instance
      return idxOf[a] > idxOf[b];            // then original order
    };
    std::priority_queue<Operation *, SmallVector<Operation *>, decltype(cmp)>
        ready(cmp);
    for (Operation *op : ops)
      if (indeg[op] == 0)
        ready.push(op);

    SmallVector<Operation *> sorted;
    sorted.reserve(ops.size());
    while (!ready.empty()) {
      Operation *op = ready.top();
      ready.pop();
      sorted.push_back(op);
      for (Operation *s : succs[op])
        if (--indeg[s] == 0)
          ready.push(s);
    }

    if (sorted.size() == ops.size()) {
      for (Operation *op : sorted)
        op->moveBefore(term);
      ++serialized;
    }
  }
  if (serialized)
    llvm::errs() << "[cv-split]   serialized " << serialized
                 << " pipelined vector block(s) into per-instance order\n";
}

// ============================================================================
// Pin the loop-carried softmax state into mem_unique UB buffers.
//
// Our VECTOR scope threads the softmax state (m_i running max, l_i running sum,
// acc output accumulator) through the outer KV loop as scf.for iter_args, so
// each stays a live SSA tensor that PlanMemory places dynamically. The manual
// reference (target_optimized.ir) instead allocates each state once in a
// dedicated `mem_unique` UB buffer (m_i->alloc, acc->alloc_4, l_i->alloc_12..15)
// and every per-instance update does a read/modify/write on it (read via
// memory_space_cast + to_tensor, write via `bind_buffer`). `mem_unique` is a
// whole-function allocation-discipline signal: pinning all state gives
// PlanMemory a deterministic fixed-offset layout (shared across the AIC/AIV
// clones), which is what keeps the cube-side matmul L1->FB operand load
// immediate-encodable (mode 0/1) rather than register-indirect (mode 2/3 ->
// simulator dmamov_decode_to_fb assert). The manual pins 7 mem_unique buffers
// (Q + 6 state); ours pins only Q -- this step adds the 6 state buffers.
//
// CRITICAL: the manual does NOT remove the iter_arg -- its VECTOR loop still
// carries m_i as `iter_args(%arg15 = %1)` (init = -inf fill). `bind_buffer` is
// an *additional* storage-pinning annotation layered on top of the SSA
// iter_arg dataflow; it tells bufferization which fixed buffer materializes the
// value, it does not replace the dataflow. So we must NOT touch iter_args /
// reads / loop structure (doing so broke dominance in bishengir's bind-buffer
// pass: a pre-loop init write got LICM-hoisted above the per-block buffer alloc).
//
// Per tensor iter_arg k (m_i / l_i / acc):
//   - alloc %buf_k : memref<...xf32, ub> at function scope, mark {mem_unique}
//     + effects
//   - identify the per-instance updated values via region-aware
//     forward(blockArg) INTERSECT backward(yieldOperand), filtered to the
//     iter_arg's exact type (so look-alikes such as pv:64x64xf32 or
//     alpha/m_new:64xf32 are excluded -- they are not forward-reachable from
//     this state's block-arg)
//   - mark each such updated value {bind_buffer %buf_k}; leave iter_args, reads,
//     and the loop structure untouched.
// Opt-in: enable with TRITON_CVSPLIT_HOIST_STATE=1 (default OFF; acc-only is
// known insufficient -- the full set must match the manual to flip the mode).
// ============================================================================
static void hoistVectorStateToMemUnique(scope::ScopeOp vecScope) {
  bool enabled = false;
  if (const char *e = std::getenv("TRITON_CVSPLIT_HOIST_STATE"))
    enabled = (StringRef(e) == "1" || StringRef(e) == "true");
  if (!enabled)
    return;

  // The outer KV loop is the (single) scf.for directly inside the vector scope.
  scf::ForOp loop;
  for (Operation &op : vecScope.getBodyRegion().front())
    if (auto f = dyn_cast<scf::ForOp>(&op)) {
      loop = f;
      break;
    }
  if (!loop)
    return;

  // Hoist targets: every tensor (ranked f32) iter_arg (m_i, l_i, acc); the i32
  // loop counters are left as iter_args.
  SmallVector<int> targets;
  for (auto [i, a] : llvm::enumerate(loop.getRegionIterArgs())) {
    auto tt = dyn_cast<RankedTensorType>(a.getType());
    if (tt && tt.getElementType().isF32())
      targets.push_back((int)i);
  }
  if (targets.empty())
    return;

  MLIRContext *ctx = vecScope.getContext();
  Location loc = loop.getLoc();
  Block *loopBody = loop.getBody();
  Operation *term = loopBody->getTerminator();

  // mem_unique state buffers MUST be allocated at function scope (before the
  // first scope), exactly like the manual reference and the Q-cbuf hoist:
  // SplitMixKernel clones the function into AIC/AIV halves and drops each
  // scope's body from the other clone, and buildFinalHIVMPipelines moves
  // function-level allocs -- a buffer allocated *inside* the VECTOR scope then
  // fails dominance ("operand #1 does not dominate this use"). Allocating it
  // before the first (CUBE) scope keeps it dominating every use in both clones.
  Operation *funcInsertPt = vecScope.getOperation();
  if (Block *parentBlk = vecScope->getBlock())
    for (Operation &op : *parentBlk)
      if (isa<scope::ScopeOp>(&op)) {
        funcInsertPt = &op;
        break;
      }

  // Region-aware forward reachability: SSA uses, with terminator operands
  // mapped to their parent op's same-index result (and scf.for iter_arg), so a
  // value used inside a scope/loop flows precisely to the matching result.
  auto forwardSet = [&](Value seed) {
    llvm::DenseSet<Value> F;
    SmallVector<Value> wl{seed};
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!F.insert(v).second)
        continue;
      for (OpOperand &u : v.getUses()) {
        Operation *O = u.getOwner();
        if (O->hasTrait<OpTrait::IsTerminator>()) {
          unsigned idx = u.getOperandNumber();
          Operation *parent = O->getParentOp();
          if (parent && idx < parent->getNumResults())
            wl.push_back(parent->getResult(idx));
          if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent))
            if (idx < forOp.getRegionIterArgs().size())
              wl.push_back(forOp.getRegionIterArgs()[idx]);
        } else {
          for (Value r : O->getResults())
            wl.push_back(r);
        }
      }
    }
    return F;
  };

  // Region-aware backward reachability: operands, with region-op results mapped
  // to their terminator operand at the same index (descend into scope/for).
  auto backwardSet = [&](Value goal) {
    llvm::DenseSet<Value> B;
    SmallVector<Value> wl{goal};
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!B.insert(v).second)
        continue;
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        if (auto forOp =
                dyn_cast_or_null<scf::ForOp>(ba.getOwner()->getParentOp())) {
          unsigned argno = ba.getArgNumber();
          if (argno >= 1) {
            unsigned idx = argno - 1;
            if (idx < forOp.getInitArgs().size())
              wl.push_back(forOp.getInitArgs()[idx]);
            if (auto y = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator()))
              if (idx < y.getNumOperands())
                wl.push_back(y.getOperand(idx));
          }
        }
        continue;
      }
      auto res = cast<OpResult>(v);
      Operation *def = res.getOwner();
      unsigned idx = res.getResultNumber();
      bool isRegionOp =
          isa<scope::ScopeOp>(def) || isa<scf::ForOp>(def) || isa<scf::IfOp>(def);
      if (isRegionOp) {
        for (Region &reg : def->getRegions())
          if (!reg.empty())
            if (Operation *t = reg.front().getTerminator())
              if (idx < t->getNumOperands())
                wl.push_back(t->getOperand(idx));
        if (auto forOp = dyn_cast<scf::ForOp>(def))
          if (idx < forOp.getInitArgs().size())
            wl.push_back(forOp.getInitArgs()[idx]);
        continue;
      }
      for (Value o : def->getOperands())
        wl.push_back(o);
    }
    return B;
  };

  (void)ctx;
  unsigned pinned = 0;
  for (int k : targets) {
    BlockArgument arg = loop.getRegionIterArgs()[k];
    Value yieldVal = cast<scf::YieldOp>(term).getOperand(k);
    auto tt = cast<RankedTensorType>(arg.getType());

    // The carrier chain: values of this iter_arg's exact type that lie on the
    // dataflow path from the block-arg to the yield operand. The per-instance
    // updated state values are exactly the loop-body-level child results on
    // this chain (the block-arg itself is the carried-in value, not pinned).
    llvm::DenseSet<Value> fwd = forwardSet(arg);
    llvm::DenseSet<Value> bwd = backwardSet(yieldVal);
    SmallVector<Value> checkpoints;
    for (Operation &op : loopBody->without_terminator())
      for (Value r : op.getResults())
        if (r.getType() == tt && fwd.contains(r) && bwd.contains(r))
          checkpoints.push_back(r);
    if (checkpoints.empty())
      continue;

    // mem_unique UB buffer at function scope (before the first scope). We do
    // NOT touch the loop's iter_args or any reads: exactly like the manual
    // (target_optimized.ir), the softmax state stays a loop-carried SSA tensor
    // and we only pin its storage to a fixed mem_unique buffer by marking each
    // per-instance updated value with `bind_buffer`. This gives PlanMemory the
    // deterministic fixed-offset layout that keeps the cube-side L1->FB operand
    // load immediate-encodable, without perturbing dataflow/dominance.
    OpBuilder b(funcInsertPt);
    auto ubAS = b.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
    auto bufTy = MemRefType::get(tt.getShape(), tt.getElementType(), nullptr, ubAS);
    auto buf = b.create<memref::AllocOp>(loc, bufTy).getResult();
    b.create<annotation::MarkOp>(loc, buf)->setAttr("mem_unique",
                                                    b.getUnitAttr());
    b.create<annotation::MarkOp>(loc, buf)->setAttr(
        "effects", b.getArrayAttr({b.getStringAttr("write"),
                                   b.getStringAttr("read")}));

    for (Value cp : checkpoints) {
      OpBuilder ab(loop);
      ab.setInsertionPointAfter(cp.getDefiningOp());
      ab.create<annotation::MarkOp>(loc, cp, ValueRange{buf},
                                    ab.getStrArrayAttr({"bind_buffer"}));
    }
    ++pinned;
  }

  if (pinned)
    llvm::errs() << "[cv-split]   pinned " << pinned
                 << " softmax state tensor(s) to mem_unique UB buffers "
                 << "(bind_buffer on per-instance updates; iter_args kept)\n";
}

// ============================================================================
// Rewrite the V->C NZ-pack transpose into a per-row insert loop.
//
// The cross-engine pack lowers P[M,N]f16 -> reshape [M,N16,16] -> transpose
// [N16,M,16] (perm [1,0,2]) -> reshape [N16,M16,16,16] -> copy to L1. The
// `linalg.transpose` is a genuine data permutation; when bufferized + DMA'd to
// the strided L1 subview the simulator rejects the resulting MTE addressing
// ("dmamov: Invalid offset mode: 2"). The manual reference never transposes:
// it builds p_nz [N16,M,16] per row inside the softmax loop.
//
// We reproduce the manual's contiguous pack by replacing the transpose with a
// per-row scf.for: row m of the [M,N16,16] input is a [1,N16,16] slice, which
// (because the leading dim is 1) reshapes for free to [N16,1,16] and is
// inserted at p_nz[:, m, :]. The result is the same [N16,M,16] tensor with no
// permuting DMA, so the downstream reshape->copy matches the manual exactly.
// ============================================================================
static void rewriteNZTransposePacks(scope::ScopeOp vecScope) {
  SmallVector<linalg::TransposeOp> targets;
  vecScope.walk([&](linalg::TransposeOp t) {
    ArrayRef<int64_t> perm = t.getPermutation();
    auto inT = dyn_cast<RankedTensorType>(t.getInput().getType());
    if (inT && inT.getRank() == 3 && perm.size() == 3 && perm[0] == 1 &&
        perm[1] == 0 && perm[2] == 2)
      targets.push_back(t);
  });

  for (linalg::TransposeOp t : targets) {
    OpBuilder b(t);
    Location loc = t.getLoc();
    auto inT = cast<RankedTensorType>(t.getInput().getType());     // [M, N16, K]
    auto outT = cast<RankedTensorType>(t->getResult(0).getType()); // [N16, M, K]
    int64_t M = inT.getShape()[0], N16 = inT.getShape()[1], K = inT.getShape()[2];
    Type elemTy = inT.getElementType();
    int64_t N = N16 * K;

    // Prefer slicing the original rank-2 P[M,N] tile (the source of the
    // [M,N16,K] reshape feeding the transpose). The manual packs from a clean
    // [1,N] row -> reshape [N16,1,K]; slicing the rank-3 [M,N16,K] and
    // collapsing introduces a collapse_shape that BiShengIR folds into an
    // invalid expand_shape (unit-dim move). Only use the rank-2 path when the
    // source really is [M, N].
    Value src2d;
    if (auto rin = t.getInput().getDefiningOp<tensor::ReshapeOp>()) {
      Value s = rin.getSource();
      auto st = dyn_cast<RankedTensorType>(s.getType());
      if (st && st.getRank() == 2 && st.getShape()[0] == M &&
          st.getShape()[1] == N)
        src2d = s;
    }

    Value init = b.create<tensor::EmptyOp>(loc, outT.getShape(), elemTy);
    Value lb = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value ub = b.create<arith::ConstantIntOp>(loc, M, 32);
    Value step = b.create<arith::ConstantIntOp>(loc, 1, 32);
    auto forOp = b.create<scf::ForOp>(loc, lb, ub, step, ValueRange{init});

    OpBuilder bb(forOp.getBody(), forOp.getBody()->begin());
    Value iv = bb.create<arith::IndexCastOp>(loc, bb.getIndexType(),
                                             forOp.getInductionVar());
    auto f32Ty = bb.getF32Type();
    auto i64Ty = bb.getI64Type();
    auto s3 = RankedTensorType::get({3}, i64Ty);
    auto shp = bb.create<arith::ConstantOp>(
        loc, s3, DenseElementsAttr::get(s3, ArrayRef<int64_t>{N16, 1, K}));

    // Build a clean rank-2 [1, N] row (no collapse_shape).
    Value row;
    if (src2d) {
      SmallVector<OpFoldResult, 2> o2{iv, bb.getIndexAttr(0)};
      SmallVector<OpFoldResult, 2> z2{bb.getIndexAttr(1), bb.getIndexAttr(N)};
      SmallVector<OpFoldResult, 2> s2(2, bb.getIndexAttr(1));
      row = bb.create<tensor::ExtractSliceOp>(loc, src2d, o2, z2, s2);
    } else {
      SmallVector<OpFoldResult, 3> offs{iv, bb.getIndexAttr(0),
                                        bb.getIndexAttr(0)};
      SmallVector<OpFoldResult, 3> szs{bb.getIndexAttr(1), bb.getIndexAttr(N16),
                                       bb.getIndexAttr(K)};
      SmallVector<OpFoldResult, 3> strs(3, bb.getIndexAttr(1));
      auto slice =
          bb.create<tensor::ExtractSliceOp>(loc, t.getInput(), offs, szs, strs);
      SmallVector<ReassociationIndices, 2> reassoc{{0}, {1, 2}};
      row = bb.create<tensor::CollapseShapeOp>(loc, slice.getResult(), reassoc);
    }
    // Match the manual: reshape [1,N]f32 -> [N16,1,K]f32 then trunc back.
    Value rowF32 = row;
    if (elemTy != f32Ty)
      rowF32 = bb.create<arith::ExtFOp>(
          loc, RankedTensorType::get({1, N}, f32Ty), row);
    auto reshT = RankedTensorType::get({N16, 1, K}, f32Ty);
    Value resh =
        bb.create<tensor::ReshapeOp>(loc, reshT, rowF32, shp.getResult());
    if (elemTy != f32Ty)
      resh = bb.create<arith::TruncFOp>(
          loc, RankedTensorType::get({N16, 1, K}, elemTy), resh);
    // insert into acc[:, m, :]
    SmallVector<OpFoldResult, 3> ioffs{bb.getIndexAttr(0), iv,
                                       bb.getIndexAttr(0)};
    SmallVector<OpFoldResult, 3> iszs{bb.getIndexAttr(N16), bb.getIndexAttr(1),
                                      bb.getIndexAttr(K)};
    SmallVector<OpFoldResult, 3> istrs(3, bb.getIndexAttr(1));
    auto ins = bb.create<tensor::InsertSliceOp>(
        loc, resh, forOp.getBody()->getArgument(1), ioffs, iszs, istrs);
    bb.create<scf::YieldOp>(loc, ins.getResult());

    t->getResult(0).replaceAllUsesWith(forOp.getResult(0));
    t.erase();
    wrapInSimdScope(forOp);
  }
  if (!targets.empty())
    llvm::errs() << "[cv-split]   rewrote " << targets.size()
                 << " NZ-pack transpose(s) into per-row insert loops\n";
}

// ---- stage-based grouping helpers ----------------------------------------
// Leaf ops that may sit inside a reorderable segment as pure inits/sources.
static bool isReorderLeaf(Operation *op) {
  return isa<tensor::EmptyOp, linalg::FillOp, arith::ConstantOp>(op);
}

// A "wide" (row-parallel) op operates on a rank>=2 tensor whose leading dim is
// M: its body is independent per row. A linalg.reduce over the column dim is
// also wide (its *input* is [M, ...]); it anchors a row loop. Narrow ops (all
// results rank<=1, e.g. running max/sum/correction on [M]) are NOT wide and
// stay whole-tensor outside the row loops, matching target_optimized.ir.
static bool isWideRowOp(Operation *op, int64_t M) {
  if (auto red = dyn_cast<linalg::ReduceOp>(op)) {
    if (red.getInputs().empty())
      return false;
    auto t = dyn_cast<RankedTensorType>(red.getInputs()[0].getType());
    return t && t.getRank() >= 2 && !t.isDynamicDim(0) && t.getShape()[0] == M;
  }
  for (Value r : op->getResults()) {
    auto t = dyn_cast<RankedTensorType>(r.getType());
    if (t && t.getRank() >= 2 && !t.isDynamicDim(0) && t.getShape()[0] == M)
      return true;
  }
  return false;
}

// Stage of a member op = max over in-segment operands of (operand stage, +1 if
// the operand is produced by a reduce). A reduce therefore pushes everything
// downstream into the next stage -> stage 0 = {scale-mul, rowmax}, stage 1 =
// {sub, exp, pack, rowsum}, exactly the two target row loops.
static int getSegStage(Operation *op, const DenseSet<Operation *> &seg,
                       DenseMap<Operation *, int> &memo) {
  auto it = memo.find(op);
  if (it != memo.end())
    return it->second;
  memo[op] = 0; // DAG guard
  int s = 0;
  for (Value v : op->getOperands()) {
    Operation *d = v.getDefiningOp();
    if (!d || !seg.count(d))
      continue;
    int ds = getSegStage(d, seg, memo) + (isa<linalg::ReduceOp>(d) ? 1 : 0);
    s = std::max(s, ds);
  }
  memo[op] = s;
  return s;
}

static int64_t blockRowCount(Block *blk) {
  for (Operation &op : *blk)
    for (Value r : op.getResults()) {
      auto t = dyn_cast<RankedTensorType>(r.getType());
      if (t && t.getRank() >= 2 && !t.isDynamicDim(0) && t.getShape()[0] > 1)
        return t.getShape()[0];
    }
  return 0;
}

// Process one barrier-delimited segment of pure VF/leaf ops: reorder by stage
// (narrow before wide within a stage), then row-loopify each stage's contiguous
// wide ops into a <=2-3 iter_arg map->reduce loop wrapped in a simd scope.
static unsigned rowLoopifySegment(ArrayRef<Operation *> seg, Operation *anchor,
                                  int64_t M) {
  // Row count is per-segment: a block can mix leading dims (16-row softmax,
  // 64-row setup, packed reshapes). Pick the most common rank>=2 leading dim.
  {
    DenseMap<int64_t, int> hist;
    for (Operation *op : seg)
      for (Value r : op->getResults()) {
        auto t = dyn_cast<RankedTensorType>(r.getType());
        if (t && t.getRank() >= 2 && !t.isDynamicDim(0) && t.getShape()[0] > 1)
          ++hist[t.getShape()[0]];
      }
    int best = 0;
    for (auto &kv : hist)
      if (kv.second > best) { best = kv.second; M = kv.first; }
  }
  if (M <= 1)
    return 0;

  // Anchor the segment region by the (non-erased) barriers around it so we can
  // wrap the whole thing in one scope after loopifying.
  Operation *segStart = seg.front()->getPrevNode();
  Block *blk = seg.front()->getBlock();

  DenseSet<Operation *> segSet(seg.begin(), seg.end());
  DenseMap<Operation *, int> stageMemo;

  struct Entry { Operation *op; int stage; int cls; unsigned idx; };
  SmallVector<Entry> order;
  for (auto [i, op] : llvm::enumerate(seg)) {
    int cls = isReorderLeaf(op) ? 0 : (isWideRowOp(op, M) ? 2 : 1);
    order.push_back({op, getSegStage(op, segSet, stageMemo), cls, (unsigned)i});
  }
  // Bail if any wide op feeds a narrow op of the SAME stage (would break the
  // narrow-before-wide contiguous layout) -> leave this segment whole-tensor.
  for (Operation *op : seg) {
    if (isReorderLeaf(op) || isWideRowOp(op, M))
      continue;
    int sNarrow = stageMemo[op];
    for (Value v : op->getOperands()) {
      Operation *d = v.getDefiningOp();
      if (d && segSet.count(d) && isWideRowOp(d, M) &&
          !isa<linalg::ReduceOp>(d) && stageMemo[d] == sNarrow)
        return 0;
    }
  }

  llvm::stable_sort(order, [](const Entry &a, const Entry &b) {
    if (a.stage != b.stage) return a.stage < b.stage;
    if (a.cls != b.cls) return a.cls < b.cls;
    return a.idx < b.idx;
  });

  // Physically lay the segment out in stage order, just before the barrier.
  for (Entry &e : order)
    e.op->moveBefore(anchor);

  // Collect contiguous wide runs of equal stage (after the sort, all wide ops
  // of a stage are adjacent).
  SmallVector<SmallVector<Operation *>> stageRuns;
  int curStage = -1;
  for (Entry &e : order) {
    if (e.cls != 2)
      continue;
    if (stageRuns.empty() || e.stage != curStage) {
      stageRuns.push_back({});
      curStage = e.stage;
    }
    stageRuns.back().push_back(e.op);
  }

  // Within a stage, the software pipeline interleaves several independent tile
  // chains. Split each stage into weakly-connected components (producer ->
  // consumer edges among the stage's wide ops) so each row loop carries one
  // tile's map->reduce (<=2-3 iter_args), matching target_optimized.ir instead
  // of a single fat loop with one accumulator per tile.
  SmallVector<SmallVector<Operation *>> wideRuns;
  for (auto &stageRun : stageRuns) {
    DenseMap<Operation *, unsigned> idx;
    for (auto [i, op] : llvm::enumerate(stageRun))
      idx[op] = i;
    SmallVector<unsigned> uf(stageRun.size());
    std::iota(uf.begin(), uf.end(), 0u);
    std::function<unsigned(unsigned)> find = [&](unsigned x) {
      while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
      return x;
    };
    for (auto [i, op] : llvm::enumerate(stageRun))
      for (Value v : op->getOperands())
        if (Operation *d = v.getDefiningOp()) {
          auto it = idx.find(d);
          if (it != idx.end())
            uf[find(i)] = find(it->second);
        }
    DenseMap<unsigned, unsigned> rootToRun;
    for (auto [i, op] : llvm::enumerate(stageRun)) {
      unsigned r = find(i);
      auto it = rootToRun.find(r);
      if (it == rootToRun.end()) {
        rootToRun[r] = wideRuns.size();
        wideRuns.push_back({op});
      } else {
        wideRuns[it->second].push_back(op);
      }
    }
  }

  unsigned looped = 0;
  for (auto &run : wideRuns) {
    bool hasReduce = llvm::any_of(run, [](Operation *o) {
      return isa<linalg::ReduceOp>(o);
    });
    if (run.size() < 2 && !hasReduce)
      continue; // a lone elementwise op is cheaper whole-tensor
    if (buildRowLoop(run, M))
      ++looped;
  }
  if (looped == 0)
    return 0;

  // The segment region is everything between the (preserved) barriers.
  Operation *regionFirst = segStart ? segStart->getNextNode() : &blk->front();

  // Between consecutive row loops, emit the intra-VF store->load barrier the
  // target uses so BiShengIR keeps the handoff tensor VF-register-resident
  // instead of a full UB round-trip. BiShengIR's ProcessMembar pass reads the
  // attribute key "SYNC_IN_VF" (value "VST_VLD" -> membarType 1); see
  // bishengir test Dialect/HIVMAVE/{process-membar,vector-layout-analyze-1}.mlir.
  bool seenLoop = false;
  for (Operation *o = regionFirst; o && o != anchor; o = o->getNextNode()) {
    if (!isa<scf::ForOp>(o))
      continue;
    if (seenLoop) {
      OpBuilder mb(o);
      Location loc = o->getLoc();
      Value c0 = mb.create<arith::ConstantIntOp>(loc, 0, 64);
      auto mk = mb.create<annotation::MarkOp>(loc, c0);
      mk->setAttr("SYNC_IN_VF", mb.getStringAttr("VST_VLD"));
    }
    seenLoop = true;
  }

  // Wrap loops + whole-tensor narrow ops of this segment in ONE simd scope.
  wrapRangeInSimdScope(regionFirst, anchor);
  return looped;
}

// Driver: split each VECTOR-scope block into barrier-delimited segments of pure
// VF/leaf ops and stage-loopify each, reproducing target_optimized.ir's
// per-row map->reduce SIMD loops (small iter_arg count, narrow ops whole-tensor).
static void rowLoopifyVectorScope(scope::ScopeOp vecScope) {
  SmallVector<Block *> blocks;
  blocks.push_back(&vecScope.getBodyRegion().front());
  vecScope.walk([&](scf::ForOp f) { blocks.push_back(f.getBody()); });

  unsigned looped = 0;
  for (Block *blk : blocks) {
    int64_t M = blockRowCount(blk);
    if (M <= 1)
      continue;

    // Gather (segment, anchor) pairs first; loopify after so moves/erases don't
    // disturb the scan.
    struct Seg { SmallVector<Operation *> ops; Operation *anchor; };
    SmallVector<Seg> segs;
    SmallVector<Operation *> cur;
    for (Operation &op : *blk) {
      bool member = isVFEligibleSimd(&op, /*includeReduce=*/true) ||
                    isReorderLeaf(&op);
      if (member) {
        cur.push_back(&op);
      } else {
        if (cur.size() >= 2) segs.push_back({cur, &op});
        cur.clear();
      }
    }
    // (a trailing segment before the terminator)
    if (cur.size() >= 2)
      segs.push_back({cur, blk->getTerminator()});

    for (auto &s : segs)
      looped += rowLoopifySegment(s.ops, s.anchor, M);
  }
  llvm::errs() << "[cv-split]   row-loopify(stage): " << looped
               << " per-row simd loops emitted\n";
}

// ============================================================================
// SIMD (VF) scope wrapping.
// Profiling the cvsplit vector core shows the softmax runs almost entirely as
// scalar register traffic (~57k RV_VLDI + ~36k RV_VSTI, only ~1k VF) because
// our pass emits whole-tensor linalg/arith ops but no `vector_mode="simd"`
// scopes. BiShengIR only lowers vector compute to packed VF micro-ops when it
// is wrapped in a `scope.scope {vector_mode="simd", outline=true}` (this is
// exactly what the manual+SIMD reference kernel and AscendC do). The reference
// `correction` scope is whole-tensor `tensor<Nxf32>` with no row loop, proving
// BiShengIR vectorises whole-tensor elementwise ops inside a simd scope.
//
// We therefore wrap every maximal contiguous run of pure float-tensor
// elementwise ops (mulf/subf/addf/maximumf/divf/truncf/exp/log/broadcast) in
// the VECTOR scope into its own outlined simd scope, returning the values used
// downstream. Ops that must stay scalar/structured (reduce, reshape/transpose
// pack, UB casts, sync, copy, materialize, index math) act as run boundaries
// and stay outside the simd scopes — matching the reference, where sync/copy
// live in the enclosing VECTOR scope and only the compute is simd-outlined.
// ============================================================================
static bool isVFEligibleSimd(Operation *op, bool includeReduce) {
  bool ok = isa<arith::MulFOp, arith::AddFOp, arith::SubFOp, arith::MaximumFOp,
                arith::MinimumFOp, arith::DivFOp, arith::TruncFOp, arith::ExtFOp,
                arith::NegFOp, math::ExpOp, math::Exp2Op, math::LogOp,
                linalg::BroadcastOp>(op);
  // The reference softmax-core simd scope ends each row pass with the
  // reduction (rowmax / rowsum), so optionally let a linalg.reduce that yields
  // a float tensor join the VF run too.
  if (!ok && includeReduce && isa<linalg::ReduceOp>(op))
    ok = true;
  if (!ok)
    return false;
  if (op->getNumResults() == 0)
    return false;
  for (Value r : op->getResults()) {
    auto t = dyn_cast<RankedTensorType>(r.getType());
    if (!t || !isa<FloatType>(t.getElementType()))
      return false;
  }
  return true;
}

static void wrapSimdScopes(scope::ScopeOp vecScope) {
  MLIRContext *ctx = vecScope.getContext();
  // Minimum run length worth outlining (a scope boundary forces its returned
  // tensors through UB, so tiny runs are not worth the boundary traffic).
  constexpr unsigned kMinRun = 3;

  // Optionally absorb linalg.reduce (rowmax/rowsum) into the VF runs, matching
  // the reference softmax-core scope which ends with the reduction.
  bool includeReduce = false;
  if (const char *e = std::getenv("TRITON_CVSPLIT_SIMD_REDUCE"))
    includeReduce = !(StringRef(e) == "0" || StringRef(e) == "false");

  // Gather candidate blocks: the VECTOR scope top block + every scf.for body.
  SmallVector<Block *> blocks;
  blocks.push_back(&vecScope.getBodyRegion().front());
  vecScope.walk([&](scf::ForOp f) { blocks.push_back(f.getBody()); });

  // Collect maximal contiguous runs of VF-eligible ops (op pointers are stable
  // across the later moves since runs are disjoint).
  SmallVector<SmallVector<Operation *>> runs;
  for (Block *blk : blocks) {
    SmallVector<Operation *> cur;
    for (Operation &op : *blk) {
      if (isVFEligibleSimd(&op, includeReduce)) {
        cur.push_back(&op);
      } else {
        if (cur.size() >= kMinRun)
          runs.push_back(cur);
        cur.clear();
      }
    }
    if (cur.size() >= kMinRun)
      runs.push_back(cur);
  }

  unsigned wrapped = 0;
  for (auto &run : runs) {
    Location loc = run.front()->getLoc();
    DenseSet<Operation *> runSet(run.begin(), run.end());
    // Values produced in the run that are consumed outside it -> scope returns.
    SmallVector<Value> rets;
    SmallVector<Type> retTypes;
    for (Operation *o : run) {
      for (Value res : o->getResults()) {
        bool usedOutside = false;
        for (OpOperand &u : res.getUses()) {
          if (!runSet.count(u.getOwner())) {
            usedOutside = true;
            break;
          }
        }
        if (usedOutside) {
          rets.push_back(res);
          retTypes.push_back(res.getType());
        }
      }
    }

    OpBuilder b(run.front());
    auto scopeOp = b.create<scope::ScopeOp>(loc, TypeRange(retTypes));
    Block *body = &scopeOp.getBodyRegion().emplaceBlock();
    for (Operation *o : run)
      o->moveBefore(body, body->end());
    OpBuilder rb(body, body->end());
    rb.create<scope::ReturnOp>(loc, ValueRange(rets));

    // Redirect external (non-scope) uses to the scope results.
    for (unsigned i = 0; i < rets.size(); ++i) {
      Value oldV = rets[i];
      Value newV = scopeOp.getResult(i);
      oldV.replaceUsesWithIf(newV, [&](OpOperand &u) {
        return !scopeOp->isProperAncestor(u.getOwner());
      });
    }

    scopeOp->setAttr("vector_mode", StringAttr::get(ctx, "simd"));
    scopeOp->setAttr("outline", BoolAttr::get(ctx, true));
    scopeOp->setAttr("noinline", UnitAttr::get(ctx));
    ++wrapped;
  }

  llvm::errs() << "[cv-split]   SIMD wrap: emitted " << wrapped
               << " vector_mode=simd scopes\n";
}

// ============================================================================
// Pass entry point
// ============================================================================
class CVSplitSchedulingPass
    : public ::impl::CVSplitSchedulingBase<CVSplitSchedulingPass> {
public:
  explicit CVSplitSchedulingPass(const CVSplitSchedulingOptions &options) {
    this->compileOn91095 = options.compileOn91095;
    this->unrollFactor = options.unrollFactor;
  }

  void runOnOperation() override {
    if (!compileOn91095) {
      llvm::errs() << "[cv-split] Not A5 target, skipping\n";
      return;
    }

    ModuleOp moduleOp = getOperation();
    llvm::errs() << "\n[cv-split] ============================\n"
                 << "[cv-split]  CVSplitScheduling START\n"
                 << "[cv-split]  unrollFactor=" << unrollFactor << "\n"
                 << "[cv-split] ============================\n\n";

    // Dump IR BEFORE the pass
    llvm::errs() << "[cv-split] === IR DUMP BEFORE CV-SPLIT PASS ===\n";
    moduleOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR DUMP BEFORE ===\n\n";

    moduleOp.walk([&](func::FuncOp funcOp) {
      processFunction(funcOp);
    });

    llvm::errs() << "\n[cv-split] ============================\n"
                 << "[cv-split]  CVSplitScheduling END\n"
                 << "[cv-split] ============================\n\n";

    // Dump IR AFTER the pass
    llvm::errs() << "[cv-split] === IR DUMP AFTER CV-SPLIT PASS ===\n";
    moduleOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR DUMP AFTER ===\n\n";
  }

private:
  void processFunction(func::FuncOp funcOp) {
    llvm::errs() << "[cv-split] Function: " << funcOp.getName() << "\n";

    // Stage 1: Find innermost loop
    scf::ForOp loop = findInnermostLoop(funcOp);
    if (!loop) {
      llvm::errs() << "[cv-split] No innermost loop, skip\n";
      return;
    }
    llvm::errs() << "[cv-split] Found innermost loop\n";

    // Stage 1b: No-store check
    if (hasStoresInBody(loop)) {
      llvm::errs() << "[cv-split] Loop has stores, bail\n";
      return;
    }

    // Stage 2: Unroll
    int K = unrollFactor;
    if (K <= 1) {
      llvm::errs() << "[cv-split] unrollFactor<=1, skip\n";
      return;
    }

    LogicalResult unrollResult = loopUnrollByFactor(loop, K);
    if (failed(unrollResult)) {
      llvm::errs() << "[cv-split] Unroll failed, bail\n";
      return;
    }
    llvm::errs() << "[cv-split] Unrolled by " << K << "\n";

    loop = findInnermostLoop(funcOp);
    if (!loop) {
      llvm::errs() << "[cv-split] Lost loop after unroll, bail\n";
      return;
    }

    Block *body = loop.getBody();

    // Stage 3: Classification
    DenseMap<Operation *, EngineType> classification;
    classifyAllOps(body, classification);
    if (logClassification(body, classification) == 0) {
      llvm::errs() << "[cv-split] No CUBE ops, skip\n";
      return;
    }

    // Stages 4-7: build the dependency graph, assign BFS levels, verify the
    // CUBE/VECTOR work is cleanly separable, and reorder the body by level.
    DependencyLevelScheduler scheduler;
    if (!scheduler.run(body, classification))
      return;

    // Dump IR before scope separation
    llvm::errs() << "[cv-split] === IR BEFORE SCOPE SEPARATION ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR BEFORE ===\n\n";

    // Stage 7.5: Unfuse PV matmuls (split matmul(p,v,acc*alpha) into pv + addf)
    unfusePVMatmuls(body, classification);

    // Stage 8: Insert cross-scope transfers (BEFORE scope separation)
    llvm::errs() << "[cv-split] === Stage 8: cross-scope transfers ===\n";
    insertCrossScopeTransfers(loop, body, classification);
    llvm::errs() << "[cv-split] Stage 8 complete\n";

    // Dump IR after transfers, before scope separation
    llvm::errs() << "[cv-split] === IR AFTER TRANSFERS ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END IR AFTER TRANSFERS ===\n\n";

    // Stage 9: Scope separation (like DynamicCVPipeline/SeparateCVScope)
    llvm::errs() << "[cv-split] === Stage 9: scope separation ===\n";
    createScopeSeparation(funcOp, loop, classification);
    llvm::errs() << "[cv-split] Stage 9 complete\n";

    // Stage 11.5: bind the loop-invariant matmul LHS (Q) into a cbuf buffer so
    // the QK matmul reads an aligned NZ L1 operand (matches the manual kernel
    // and avoids the misaligned implicit GM/UB->L1 stage of a plain memref).
    bindLoopInvariantMatmulLhsToCbuf(funcOp);

    // Stage 10: Ensure function has mix_mode attribute (it should already)
    // Note: do NOT add hivm.func_core_type=MIX — that triggers SplitMixKernel
    // which conflicts with our already-scoped IR. The scope::ScopeOp attrs +
    // mix_mode="mix" are sufficient for BiShengIR to handle the scopes.
    if (!funcOp->hasAttr("mix_mode"))
      funcOp->setAttr("mix_mode", StringAttr::get(funcOp.getContext(), "mix"));
    llvm::errs() << "[cv-split] Function attributes set on " << funcOp.getName() << "\n";

    // Stage 11: Set module attribute to disable auto-tiling
    // Without this, BiShengIR's auto-tile pass creates invalid pointer_casts
    // inside our scoped loops (they're not IsolatedFromAbove).
    if (auto moduleOp = funcOp->getParentOfType<ModuleOp>()) {
      moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                        UnitAttr::get(funcOp.getContext()));
    }

    // Dump function IR after scope separation
    llvm::errs() << "[cv-split] === FUNCTION IR AFTER SCOPE SEPARATION ===\n";
    funcOp.print(llvm::errs());
    llvm::errs() << "\n[cv-split] === END FUNCTION IR ===\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCVSplitSchedulingPass(
    const CVSplitSchedulingOptions &options) {
  return std::make_unique<CVSplitSchedulingPass>(options);
}
