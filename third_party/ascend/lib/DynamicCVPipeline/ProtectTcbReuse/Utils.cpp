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

#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "protect-tcb-reuse"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
struct mainLoopPair {
  Operation *vectorMainLoop = nullptr;
  Operation *cubeMainLoop = nullptr;
};

static std::optional<hivm::TCoreType> getCore(Operation *mainLoop) {
  auto scope = mainLoop->getParentOfType<scope::ScopeOp>();
  auto coreType =
      scope->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
  return coreType.getTcoretype();
}

// This protection is optional: leave IR unchanged and let dynamic CV proceed.
static void skipProtection(ModuleOp module, StringRef protectionName,
                           StringRef reason) {
  for (auto function : module.getOps<func::FuncOp>()) {
    LOG_DEBUG("kernel '" << function.getSymName() << "' is not protected by "
                         << protectionName << " synchronization: " << reason);
  }
}

static std::optional<mainLoopPair> findCandidateMainLoop(
    scf::ForOp outerLoop,
    const llvm::DenseMap<int64_t, mainLoopPair> &mainLoopPairs,
    tcbReuse::LoopPredicate matchesCandidate) {
  Operation *firstMainLoop = nullptr;
  outerLoop.walk<WalkOrder::PreOrder>([&](Operation *operation) -> WalkResult {
    if (!CVPipeline::isMainLoopOp(operation)) {
      return WalkResult::advance();
    }
    firstMainLoop = operation;
    return WalkResult::interrupt();
  });
  if (!firstMainLoop || !matchesCandidate(outerLoop, firstMainLoop)) {
    return std::nullopt;
  }
  auto id =
      firstMainLoop->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop).getInt();
  return mainLoopPairs.lookup(id);
}

static FailureOr<SmallVector<mainLoopPair>>
collectProtectionPlans(ModuleOp module, StringRef protectionName,
                       tcbReuse::LoopPredicate matchesCandidate) {
  llvm::DenseMap<int64_t, mainLoopPair> mainLoopPairs;
  llvm::MapVector<Operation *, SmallVector<int64_t>> outerLoopIds;
  auto walkResult = module.walk([&](Operation *mainLoop) -> WalkResult {
    if (!CVPipeline::isMainLoopOp(mainLoop)) {
      return WalkResult::advance();
    }
    if (isa<scf::WhileOp>(mainLoop)) {
      return WalkResult::interrupt();
    }
    auto core = getCore(mainLoop);
    if (core != hivm::TCoreType::VECTOR && core != hivm::TCoreType::CUBE) {
      return WalkResult::advance();
    }
    auto id =
        mainLoop->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop).getInt();
    auto &pair = mainLoopPairs[id];
    if (core == hivm::TCoreType::VECTOR) {
      pair.vectorMainLoop = mainLoop;
    } else {
      pair.cubeMainLoop = mainLoop;
    }
    // Include every containing loop up to the function boundary, even
    // when a scope separates a main loop from an enclosing loop.
    for (auto outerLoop = mainLoop->getParentOfType<scf::ForOp>(); outerLoop;
         outerLoop = outerLoop->getParentOfType<scf::ForOp>()) {
      outerLoopIds[outerLoop.getOperation()].push_back(id);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    skipProtection(module, protectionName, "while main loop is not supported");
    return failure();
  }
  SmallVector<mainLoopPair> candidates;
  // Insertion order follows main-loop discovery, preserving flag assignment.
  for (const auto &[outerOperation, vectorIds] : outerLoopIds) {
    if (getCore(outerOperation) != hivm::TCoreType::VECTOR) {
      continue;
    }
    if (vectorIds.size() != 2) {
      continue;
    }
    // Find fisrt main_loop
    auto outerLoop = cast<scf::ForOp>(outerOperation);
    auto candidateMainLoop =
        findCandidateMainLoop(outerLoop, mainLoopPairs, matchesCandidate);
    if (!candidateMainLoop) {
      continue;
    }
    auto *firstVectorMainLoop = candidateMainLoop->vectorMainLoop;

    auto cubeOuterLoop =
        candidateMainLoop->cubeMainLoop->getParentOfType<scf::ForOp>();
    // Split C/V loop nests retain their enclosing for-loop levels.
    for (auto vectorOuterLoop =
             firstVectorMainLoop->getParentOfType<scf::ForOp>();
         vectorOuterLoop != outerLoop;
         vectorOuterLoop = vectorOuterLoop->getParentOfType<scf::ForOp>()) {
      cubeOuterLoop = cubeOuterLoop->getParentOfType<scf::ForOp>();
    }
    candidateMainLoop->vectorMainLoop = CVPipeline::getAncestorInBlock(
        firstVectorMainLoop, outerLoop.getBody());
    candidateMainLoop->cubeMainLoop = CVPipeline::getAncestorInBlock(
        candidateMainLoop->cubeMainLoop, cubeOuterLoop.getBody());
    candidates.push_back(*candidateMainLoop);
  }
  return candidates;
}

static FailureOr<SmallVector<int64_t>>
findAvailableFlags(ModuleOp module, size_t requiredCount) {
  llvm::SmallSet<int64_t, 16> usedFlags;
  bool hasDynamicFlag = false;
  module.walk([&](Operation *operation) {
    if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(operation) &&
        !operation->getAttrOfType<IntegerAttr>("static_flag_id") &&
        !operation->getAttrOfType<IntegerAttr>("flag")) {
      hasDynamicFlag = true;
    }
    if (auto flagId = operation->getAttrOfType<IntegerAttr>("static_flag_id")) {
      usedFlags.insert(flagId.getInt());
    }
    if (auto flagId = operation->getAttrOfType<IntegerAttr>("flag")) {
      usedFlags.insert(flagId.getInt());
    }
    if (auto sync = dyn_cast<hivm::SyncBlockOp>(operation)) {
      if (auto flagId = sync.getFlagId()) {
        usedFlags.insert(flagId->getInt());
      }
    }
  });
  if (hasDynamicFlag) {
    return failure();
  }
  // Preserve ascending allocation, including holes below reserved flag 15.
  // Boundary syncs have no transfer_id and are not copied per buffer slot.
  SmallVector<int64_t> flags;
  for (int64_t flagId = 0;
       flagId <= FlagIdManager::MAX_FLAG_ID && flags.size() < requiredCount;
       ++flagId) {
    if (!usedFlags.contains(flagId)) {
      flags.push_back(flagId);
    }
  }
  if (flags.size() != requiredCount) {
    return failure();
  }
  return flags;
}

static void insertSynchronization(ModuleOp module, ArrayRef<mainLoopPair> plans,
                                  ArrayRef<int64_t> flags,
                                  hivm::PIPE sourcePipe) {
  OpBuilder builder(module.getContext());
  auto source = hivm::PipeAttr::get(module.getContext(), sourcePipe);
  auto destination =
      hivm::PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
  auto vectorCore =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto cubeCore =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  // Serial main loops share one handshake per outer iteration.
  for (auto [pair, flag] : llvm::zip(plans, flags)) {
    auto flagId = builder.getI64IntegerAttr(flag);
    builder.setInsertionPoint(pair.vectorMainLoop);
    builder.create<hivm::SyncBlockSetOp>(
        pair.vectorMainLoop->getParentOp()->getLoc(), vectorCore, source,
        destination, flagId);
    builder.setInsertionPoint(pair.cubeMainLoop);
    builder.create<hivm::SyncBlockWaitOp>(
        pair.cubeMainLoop->getParentOp()->getLoc(), cubeCore, source,
        destination, flagId);
  }
}
} // namespace

void mlir::triton::tcbReuse::protectTcbReuse(ModuleOp module,
                                             StringRef protectionName,
                                             hivm::PIPE sourcePipe,
                                             LoopPredicate matchesCandidate) {
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  auto plans = collectProtectionPlans(module, protectionName, matchesCandidate);
  if (failed(plans) || plans->empty()) {
    return;
  }
  auto flags = findAvailableFlags(module, plans->size());
  if (failed(flags)) {
    skipProtection(module, protectionName,
                   "no free flag for tcb reuse protection");
    return;
  }
  // Commit only after every function and resource has been validated.
  insertSynchronization(module, *plans, *flags, sourcePipe);
}
