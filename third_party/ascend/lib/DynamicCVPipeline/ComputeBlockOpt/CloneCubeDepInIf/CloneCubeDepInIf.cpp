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
#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"

static constexpr const char *DEBUG_TYPE = "clone-cube-dep-in-if";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;
using namespace SplitIf;

namespace {

/// A maximal run of consecutive ops in MLIR order inside a single if-region
/// that share the same block_id AND whose core_type is CUBE.
struct CubeBlock {
  int blockId;
  SmallVector<Operation *> ops;
};

/// One candidate if that SplitIfByBlockId would later split.
struct CandidateIf {
  scf::IfOp ifOp;
  SmallVector<CubeBlock> thenCubes;
  SmallVector<CubeBlock> elseCubes;
  bool thenNeedsSplit = false;
  bool elseNeedsSplit = false;
};

/// Pretty-print a CubeBlock for debug.
static std::string formatCubeBlock(const CubeBlock &c) {
  std::string s;
  s += "{bid=" + std::to_string(c.blockId) + " ops=[";
  bool first = true;
  for (Operation *op : c.ops) {
    if (!first) {
      s += ", ";
    }
    first = false;
    if (op->getName().getStringRef().size() > 32) {
      s += op->getName().getStringRef().substr(0, 32).str();
    } else {
      s += op->getName().getStringRef().str();
    }
  }
  s += "]}";
  return s;
}

} // namespace

/// Scan a region (then-block or else-block) and produce maximal CUBE runs.
static SmallVector<CubeBlock> collectCubeBlocksInRegion(Block &block) {
  SmallVector<CubeBlock> cubes;
  auto *parentOp = block.getParentOp();
  if (!parentOp) {
    return cubes;
  }

  for (auto &op : block) {
    if (isa<scf::YieldOp>(op)) {
      continue;
    }

    auto bid = CVPipeline::getOpBlockId(&op);
    if (!bid.has_value() || *bid == -1) {
      continue;
    }

    CoreType ct = CVPipeline::getOpCoreType(&op);
    if (ct != CoreType::CUBE_ONLY) {
      continue;
    }

    if (!cubes.empty() && cubes.back().blockId == *bid &&
        !cubes.back().ops.empty()) {
      cubes.back().ops.push_back(&op);
    } else {
      cubes.push_back({*bid, {&op}});
    }
  }

  return cubes;
}

/// Build the candidate if we plan to process.
static CandidateIf getCandidate(scf::IfOp ifOp) {
  CandidateIf cand;
  cand.ifOp = ifOp;

  cand.thenCubes = collectCubeBlocksInRegion(*ifOp.thenBlock());
  Block *elseBlk = ifOp.elseBlock();
  if (elseBlk) {
    cand.elseCubes = collectCubeBlocksInRegion(*elseBlk);
  }

  auto distinctCount = [](const SmallVector<CubeBlock> &cubes) {
    llvm::SmallDenseSet<int> ids;
    for (auto &c : cubes) {
      ids.insert(c.blockId);
    }
    return static_cast<unsigned>(ids.size());
  };
  cand.thenNeedsSplit = distinctCount(cand.thenCubes) >= 2;
  cand.elseNeedsSplit = distinctCount(cand.elseCubes) >= 2;

  return cand;
}

/// Check whether any op in `laterOps` references (via SSA use-def) any op in
/// `earlierOps` (transitively through nested regions).
static bool
laterCubeDependsOnEarlier(const llvm::SmallDenseSet<Operation *> &earlierOps,
                          ArrayRef<Operation *> laterOps) {
  if (earlierOps.empty()) {
    return false;
  }

  auto operandRefsEarlier = [&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      Operation *defOp = operand.get().getDefiningOp();
      if (defOp && earlierOps.contains(defOp)) {
        return true;
      }
    }
    for (auto &region : op->getRegions()) {
      bool found = false;
      region.walk([&](Operation *nestedOp) {
        if (found) {
          return;
        }
        for (auto &operand : nestedOp->getOpOperands()) {
          Operation *defOp = operand.get().getDefiningOp();
          if (defOp && earlierOps.contains(defOp)) {
            found = true;
            return;
          }
        }
      });
      if (found) {
        return true;
      }
    }
    return false;
  };

  for (auto *op : laterOps) {
    if (operandRefsEarlier(op)) {
      return true;
    }
  }
  return false;
}

/// Clone a single op using an IRMapping seeded with all previously cloned
/// values. Mirrors CloneOps::cloneOpWithMapping.
static Operation *
cloneOpWithMapping(Operation *op, OpBuilder &builder,
                   llvm::DenseMap<Value, Value> &valueMap) {
  IRMapping mapper;
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
  }
  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }
  return cloned;
}

/// Update op operands through valueMap, recursing through nested regions.
/// Mirrors CloneOps::updateCloneMapping.
static LogicalResult
updateCloneMapping(Operation *op, llvm::DenseMap<Value, Value> &valueMap,
                   const llvm::DenseSet<Value> &yieldValues) {
  if (!op) {
    return failure();
  }
  for (OpOperand &operand : op->getOpOperands()) {
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }
    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        LDBG("[Error]: type mismatch in value mapping: " << v.getType() << " vs "
                                                        << it->second.getType());
        return failure();
      }
      operand.set(it->second);
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }
  return success();
}

/// Like CloneOps::cloneOpsForBlock: clone every op from `earlierCubes` (in
/// MLIR order) into the position right before `insertBefore` and rewrite
/// `laterOps`'s operands to point at the cloned values. Inserted cloned ops
/// are tagged with kBlockId = `laterBlockId` and kClone = original block_id.
static LogicalResult
cloneEarlierCubesInto(ArrayRef<CubeBlock> earlierCubes, int laterBlockId,
                      ArrayRef<Operation *> laterOps, Operation *insertBefore,
                      const llvm::DenseSet<Value> &yieldValues,
                      SmallVectorImpl<Operation *> &newlyInserted) {
  if (earlierCubes.empty() || laterOps.empty() || insertBefore == nullptr) {
    return success();
  }

  // Collect all earlier ops in MLIR order across cubes.
  SmallVector<Operation *> toClone;
  for (auto &c : earlierCubes) {
    for (auto *op : c.ops) {
      toClone.push_back(op);
    }
  }
  if (toClone.empty()) {
    return success();
  }
  llvm::sort(toClone, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });

  LDBG("cloneEarlierCubesInto: laterBlockId=" << laterBlockId
                                              << " earlierCubes="
                                              << earlierCubes.size()
                                              << " toClone=" << toClone.size());

  OpBuilder builder(insertBefore);
  builder.setInsertionPoint(insertBefore);

  llvm::DenseMap<Value, Value> valueMap;
  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
    cloned->setAttr(CVPipeline::kBlockId,
                    builder.getI32IntegerAttr(laterBlockId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(
          CVPipeline::kClone,
          builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
    }
    LDBG("  cloned " << op->getName() << " (orig block_id="
                     << CVPipeline::getOpBlockId(op).value_or(-1)
                     << ") -> new block_id=" << laterBlockId);
    newlyInserted.push_back(cloned);
  }

  // Rewrite laterOps' operands to point at cloned values.
  for (Operation *op : laterOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }
  LDBG("cloneEarlierCubesInto: done, valueMap size=" << valueMap.size());

  return success();
}

/// Returns true if `root`'s effect can reach a non-cloned op (transitively
/// through SSA use chains restricted to ops inside `scope`). For ops with
/// no side effects on the SSA chain this returns false. For side-effecting
/// ops it walks the effects of `root`, finds the values that are written/
/// allocated (i.e. the side-effect targets), then BFS-walks every user of
/// every such value to detect a non-cloned consumer.
static bool effectReachesNonClonedOp(Operation *root) {
  // Pure: no side effect to leak.
  if (mlir::isMemoryEffectFree(root)) {
    return false;
  }

  // Collect the values that root produces side effects on. For MemWrite
  // / MemAlloc / MemFree effects, the .getValue() is the affected resource.
  SmallVector<Value> affected;
  for (auto &effect :
       mlir::getEffectsRecursively(root).value_or(
           llvm::SmallVector<mlir::MemoryEffects::EffectInstance>{})) {
    Value v = effect.getValue();
    if (!v) {
      continue;
    }
    // Track any value the op side-effects on: writes (memref.copy,
    // memref.store, linalg.fill), allocations (memref.alloc), reads
    // (bufferization.to_tensor), and frees (memref.dealloc). If the
    // value flows only to other cloned ops, the side effect is dead.
    if (isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free,
            MemoryEffects::Read>(effect.getEffect())) {
      affected.push_back(v);
    }
  }

  // If the op has no side effect on a known value (only Resource
  // effects), be conservative and assume it may be observed externally.
  if (affected.empty()) {
    return true;
  }

  // BFS over users of every affected value, stopping at the first
  // non-cloned user.
  llvm::SmallPtrSet<Operation *, 32> visited;
  SmallVector<Operation *> worklist;
  for (Value v : affected) {
    for (Operation *user : v.getUsers()) {
      worklist.push_back(user);
    }
  }
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    if (!cur->hasAttr(CVPipeline::kClone)) {
      return true;
    }
    // Continue along the user chain in case the effect value flows
    // through more cloned ops to a final non-cloned consumer.
    for (Operation *next : cur->getUsers()) {
      worklist.push_back(next);
    }
  }
  return false;
}

/// Cleanup helper: a cloned op is *potentially* erasable when:
///   * it carries the kClone attr,
///   * it is not a terminator,
///   * for side-effecting ops: its effect does not reach any non-cloned
///     op in the same region (i.e. the memory it produces/writes is only
///     consumed by other cloned ops),
///   * for ops with SSA results: every user of every result is itself a
///     cloned op or the result is unused (the cascade into side-effect
///     chains is handled by the fixpoint loop).
static bool isInitiallyErasable(Operation *op) {
  if (!op->hasAttr(CVPipeline::kClone)) {
    return false;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  // Side-effect check (covers memref.alloc / memref.copy / linalg.fill /
  // memref.store / bufferization.to_tensor's declared MemRead, etc.).
  // We treat anything that declares side effects as live UNLESS its
  // effect is provably confined to the cloned chain.
  if (effectReachesNonClonedOp(op)) {
    return false;
  }
  // SSA-result check: a result whose only consumer is a non-cloned op
  // must be kept. This is redundant with the effect check for memref
  // effects but covers pure ops whose result is read by non-cloned ops.
  for (auto result : op->getResults()) {
    if (result.use_empty()) {
      continue;
    }
    for (Operation *user : result.getUsers()) {
      if (!user->hasAttr(CVPipeline::kClone)) {
        return false;
      }
    }
  }
  return true;
}

/// Process one side (then or else) of a candidate if. Mirrors
/// CloneOps::cloneOpsInMainLoop's reverse-order strategy: walk CUBE blocks
/// from last to first; for each CUBE block whose ops depend on any earlier
/// CUBE block, clone every op from earlier CUBE blocks in front of this
/// block and rewire uses.
static LogicalResult processSide(Block *block,
                                 MutableArrayRef<CubeBlock> cubes) {
  if (cubes.size() < 2 || block == nullptr) {
    return success();
  }

  llvm::DenseSet<Value> yieldValues;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(block->getTerminator())) {
    for (Value operand : yieldOp.getOperands()) {
      yieldValues.insert(operand);
    }
  }

  LDBG("processSide: " << cubes.size() << " cubes");
  for (size_t k = 0; k < cubes.size(); ++k) {
    LDBG("  cube[" << k << "] " << formatCubeBlock(cubes[k]));
  }

  // Process CUBE blocks in reverse order. For each later cube, if it
  // depends on any earlier cube, clone every op in earlier cubes into
  // its front.
  for (size_t i = cubes.size(); i-- > 0;) {
    if (i == 0) {
      continue;
    }
    ArrayRef<CubeBlock> earlierCubes = ArrayRef<CubeBlock>(cubes).take_front(i);

    llvm::SmallDenseSet<Operation *> earlierOps;
    for (auto &c : earlierCubes) {
      for (auto *op : c.ops) {
        earlierOps.insert(op);
      }
    }

    bool depends = laterCubeDependsOnEarlier(earlierOps, cubes[i].ops);
    LDBG("processSide: i=" << i << " laterBid=" << cubes[i].blockId
                           << " depends=" << depends);
    if (!depends) {
      continue;
    }

    Operation *insertBefore = cubes[i].ops.front();
    SmallVector<Operation *> newlyInserted;
    if (failed(cloneEarlierCubesInto(earlierCubes, cubes[i].blockId,
                                     cubes[i].ops, insertBefore, yieldValues,
                                     newlyInserted))) {
      return failure();
    }
  }

  return success();
}

/// Cleanup pass: erase cloned ops whose entire SSA-use chain is closed
/// inside the cloned set (i.e. no non-cloned op references any result of
/// any op in the chain). The closure is computed bottom-up via a
/// fixpoint. This mirrors the conservative side of CloneOps cleanup,
/// avoiding "operation destroyed but still has uses" verifier errors.
static void cleanupSide(Block *block) {
  if (block == nullptr) {
    return;
  }

  // 1. Initially-erasable set.
  llvm::DenseSet<Operation *> erasable;
  for (auto &op : *block) {
    if (isa<scf::YieldOp>(op)) {
      continue;
    }
    if (isInitiallyErasable(&op)) {
      erasable.insert(&op);
    }
  }
  // Also consider cloned ops nested inside region-bearing ops (e.g. the
  // body of a cloned scf.if). We treat each region recursively, but
  // iterating just the outer block is sufficient because we never clone
  // region-bearing ops in processSide — every op in earlierCubes is a
  // simple op. We still walk one level deep defensively for nested
  // cloned ops that may appear in user-written `scf.if` inside a CUBE
  // block (rare but possible).
  for (auto &op : *block) {
    if (!op.hasAttr(CVPipeline::kClone)) {
      continue;
    }
    for (Region &region : op.getRegions()) {
      for (Block &subBlock : region) {
        for (auto &nestedOp : subBlock) {
          if (isInitiallyErasable(&nestedOp)) {
            erasable.insert(&nestedOp);
          }
        }
      }
    }
  }

  LDBG("cleanupSide: initiallyErasable size=" << erasable.size());

  // 2. Fixpoint: an op is erasable only if (a) every user of every SSA
  //    result is itself erasable, AND (b) for side-effecting ops, the
  //    effect chain does not reach a non-cloned (or non-erasable) op.
  //    A non-cloned consumer acts as a "live" anchor; a previously-
  //    erasable-but-now-removed consumer also exposes upstream effects
  //    that may no longer have any cloned consumer.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> snapshot(erasable.begin(), erasable.end());
    for (Operation *op : snapshot) {
      if (!op->getBlock()) {
        continue;
      }
      // SSA-result check.
      bool pin = false;
      for (auto result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!erasable.contains(user)) {
            pin = true;
            break;
          }
        }
        if (pin) {
          break;
        }
      }
      if (pin) {
        if (erasable.erase(op)) {
          changed = true;
        }
        continue;
      }
      // Side-effect check: re-evaluate effect reachability, but treat
      // any non-erasable op in the user chain as an external anchor
      // (equivalent to non-cloned). The pure helper above already
      // handles effect-reaches-non-cloned; we additionally exclude ops
      // that have been removed from erasable.
      if (!mlir::isMemoryEffectFree(op)) {
        SmallVector<Value> affected;
        for (auto &effect :
             mlir::getEffectsRecursively(op).value_or(
                 llvm::SmallVector<mlir::MemoryEffects::EffectInstance>{})) {
          Value v = effect.getValue();
          if (!v) {
            continue;
          }
          if (isa<MemoryEffects::Write, MemoryEffects::Allocate,
                  MemoryEffects::Free, MemoryEffects::Read>(
                  effect.getEffect())) {
            affected.push_back(v);
          }
        }
        bool leak = affected.empty(); // conservative: no value ⇒ unknown
        if (!leak) {
          llvm::SmallPtrSet<Operation *, 32> visited;
          SmallVector<Operation *> worklist;
          for (Value v : affected) {
            for (Operation *user : v.getUsers()) {
              worklist.push_back(user);
            }
          }
          while (!worklist.empty() && !leak) {
            Operation *cur = worklist.pop_back_val();
            if (!visited.insert(cur).second) {
              continue;
            }
            if (!cur->hasAttr(CVPipeline::kClone) ||
                !erasable.contains(cur)) {
              leak = true;
              break;
            }
            for (Operation *next : cur->getUsers()) {
              worklist.push_back(next);
            }
          }
        }
        if (leak) {
          if (erasable.erase(op)) {
            changed = true;
          }
        }
      }
    }
  }
  LDBG("cleanupSide: post-fixpoint erasable size=" << erasable.size());

  // 3. Erase bottom-up (consumers before producers).
  SmallVector<Operation *> toErase(erasable.begin(), erasable.end());
  llvm::sort(toErase, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  std::reverse(toErase.begin(), toErase.end());

  for (Operation *op : toErase) {
    if (!op->getBlock()) {
      continue;
    }
    // Sanity: still no live non-erasable user.
    bool safe = true;
    for (auto result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!erasable.contains(user)) {
          safe = false;
          LDBG("cleanupSide: REFUSE to erase " << op->getName()
                                                << " (live user: " << user->getName()
                                                << ")");
          break;
        }
      }
      if (!safe) {
        break;
      }
    }
    if (!safe) {
      continue;
    }
    // Sanity: side-effect no longer leaks (defensive).
    if (effectReachesNonClonedOp(op)) {
      LDBG("cleanupSide: REFUSE to erase " << op->getName()
                                            << " (effect leaks to non-cloned)");
      continue;
    }
    LDBG("cleanupSide: erasing " << op->getName());
    op->erase();
  }
}

static LogicalResult processCandidate(CandidateIf &cand) {
  if (cand.thenNeedsSplit) {
    if (failed(processSide(cand.ifOp.thenBlock(), cand.thenCubes))) {
      return failure();
    }
    cleanupSide(cand.ifOp.thenBlock());
  }
  if (cand.elseNeedsSplit) {
    if (failed(processSide(cand.ifOp.elseBlock(), cand.elseCubes))) {
      return failure();
    }
    cleanupSide(cand.ifOp.elseBlock());
  }
  return success();
}

namespace {

class CloneCubeDepInIfPass
    : public PassWrapper<CloneCubeDepInIfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CloneCubeDepInIfPass)

  CloneCubeDepInIfPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "clone-cube-dep-in-if";
  }

  llvm::StringRef getDescription() const final {
    return "Clone CUBE-block dependency chains inside scf.if ops so that each "
           "CUBE block owns its computations and is independent across the "
           "if split performed by SplitIfByBlockId.";
  }
};

} // namespace

void CloneCubeDepInIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before:\n" << module << "\n----------");

  WalkResult walkRes = module->walk([&](scf::IfOp ifOp) -> WalkResult {
    CandidateIf cand = getCandidate(ifOp);
    if (!cand.thenNeedsSplit && !cand.elseNeedsSplit) {
      return WalkResult::advance();
    }
    LDBG("Processing if: " << ifOp);
    LDBG("  thenNeedsSplit=" << cand.thenNeedsSplit
                              << " elseNeedsSplit=" << cand.elseNeedsSplit
                              << " thenCubes=" << cand.thenCubes.size()
                              << " elseCubes=" << cand.elseCubes.size());
    if (failed(processCandidate(cand))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkRes.wasInterrupted()) {
    LDBG("Clone cube deps in if failed, fallback to original");
    setFallbackAttr(module, ERRCODE_FAILED);
    return;
  }

  LDBG("After: \n" << module << "\n----------");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCloneCubeDepInIfPass() {
  return std::make_unique<CloneCubeDepInIfPass>();
}

} // namespace triton
} // namespace mlir