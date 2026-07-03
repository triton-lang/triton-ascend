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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

static constexpr const char *DEBUG_TYPE = "analyze-scope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {
static bool isVectorScope(scope::ScopeOp scopeOp)
{
    auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) {
        return false;
    }
    return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool checkTransferInteraction(mlir::Operation* op) {
  bool hasCVInteraction = false;
  // check v->c data interaction
  if (isa<hivm::CopyOp>(op)) {
    hasCVInteraction = true;
  }
  // check c->v data interaction
  // user with "ssbuffer.add_from_matmul" is not a real c->v data interaction
  if (isa<bufferization::ToTensorOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (!user->hasAttr(CVPipeline::kAddFromMatmul)) {
        hasCVInteraction = true;
        break;
      }
      for (Operation *userUser : user->getUsers()) {
        if (!isa<scf::YieldOp>(userUser)) {
          hasCVInteraction = true;
          break;
        }
      }
    }
  }

  return hasCVInteraction;
}

static bool checkVecScopeMainLoop(ModuleOp module) {
  bool hasMainLoopFor = false;
  bool allForSatisfy = true;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    if (!isVectorScope(scopeOp)) {
      return WalkResult::advance();
    }

    scopeOp.walk([&](scf::ForOp forOp) -> WalkResult {
      if (!forOp->hasAttr(CVPipeline::kMainLoop)) {
        return WalkResult::advance();
      }

      hasMainLoopFor = true;
      bool hasCVInteraction = false;

      forOp.walk([&](mlir::Operation *op) -> WalkResult {
        if (op == forOp) {
          return WalkResult::advance();
        }

        // ops with "ssbuffer.transfer_id" are injected in SplitDataflowPass for data transfer
        if (op->hasAttr(CVPipeline::kTransferId)) {
          hasCVInteraction = checkTransferInteraction(op);
          if (hasCVInteraction) {
            return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      });

      // As long as there is a forOp with "ssbuffer.main_loop" not a real mainloop, 
      // the processing conditions are not met, need to skip.
      if (!hasCVInteraction) {
        allForSatisfy = false;
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (!allForSatisfy) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return hasMainLoopFor && allForSatisfy;
}

// For every main_loop id, gather all forOps sharing that id and count the
// hivm.hir.copy and hivm.hir.fixpipe ops within them. Only when ALL main_loop
// ids have either count equal to zero (every id has only copy or only
// fixpipe, none has both), the dynamic CV pipeline cannot be applied and we
// fall back to the original workflow.
//   - hivm::CopyOp    typically appears in VECTOR scope main_loops
//   - hivm::FixpipeOp typically appears in CUBE scope main_loops
// Nested regions inside the main_loop forOp are also walked, and scf.yield
// terminators are skipped.
static bool isMainLoopOnlyCopyOrFixpipe(ModuleOp module)
{
  // main_loop id -> (countCopy, countFixpipe)
  llvm::DenseMap<int, std::pair<int, int>> idToCounts;

  module.walk([&](scf::ForOp forOp) -> WalkResult {
    auto mainLoopAttr = forOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (!mainLoopAttr) {
      return WalkResult::advance();
    }

    int id = mainLoopAttr.getInt();
    auto &counts = idToCounts[id];

    forOp.walk([&](mlir::Operation *op) -> WalkResult {
      // Skip the forOp itself (the walk visits it first)
      if (op == forOp) {
        return WalkResult::advance();
      }
      // Skip yield terminators (they are not real ops)
      if (isa<scf::YieldOp>(op)) {
        return WalkResult::advance();
      }
      if (isa<hivm::CopyOp>(op)) {
        ++counts.first;
      } else if (isa<hivm::FixpipeOp>(op)) {
        ++counts.second;
      }
      return WalkResult::advance();
    });

    return WalkResult::advance();
  });

  // Only trigger fallback when EVERY main_loop id has only copy or only
  // fixpipe (one count is zero). If at least one id has both copy and
  // fixpipe ops, the dynamic CV pipeline still be applicable.
  if (idToCounts.empty()) {
    return false;
  }

  for (const auto &entry : idToCounts) {
    if (entry.second.first != 0 && entry.second.second != 0) {
      return false;
    }
  }

  return true;
}

static LogicalResult verifyMainLoop(ModuleOp module)
{
  // Only skip if ALL forOps lack main_loop attr
  bool hasMainLoopForOp = false;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      hasMainLoopForOp = true;
    }
  });

  if (!hasMainLoopForOp) {
    LDBG("[INFO]: No cycle of multiple iterations, the DynamicCVPipeline pass will be interrupted, and resumed to the original workflow.");
    CVPipeline::setFallbackAttr(module);
    return failure();
  }

  if (!checkVecScopeMainLoop(module)) {
    LDBG("[INFO]: No op beside matmul add in vector main loop.");
    CVPipeline::setFallbackAttr(module);
    return failure();
  };

  if (isMainLoopOnlyCopyOrFixpipe(module)) {
    LDBG("[INFO]: All main_loop only contains hivm.hir.copy or hivm.hir.fixpipe ops.");
    CVPipeline::setFallbackAttr(module);
    return failure();
  }

  return success();
}

} // namespace

void AnalyzeScopePass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeScope:\n" << module << "\n");

  if (failed(verifyMainLoop(module))) {
    signalPassFailure();
    return;
  }

  LDBG("After AnalyzeScope:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeScopePass()
{
  return std::make_unique<AnalyzeScopePass>();
}

} // namespace triton
} // namespace mlir