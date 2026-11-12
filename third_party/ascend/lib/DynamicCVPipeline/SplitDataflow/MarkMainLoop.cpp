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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "mark-main-loop";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

// Pass Entry Point
void MarkMainLoopPass::runOnOperation() {
  LOG_DEBUG("\n--- enter MarkMainLoopPass --->\n");
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  int mainLoopIdCounter = 0;
  SmallVector<Operation *> mainLoops;

  const bool outerMainLoop = CVPipeline::hasOuterMainLoop(module);
  if (outerMainLoop) {
    // Explicit selection is valid only after outer-loop normalization succeeds.
    // Do not silently replace an unfulfilled request with the innermost loop.
    auto requestCheck = module.walk([&](Operation *op) -> WalkResult {
      if (op != module.getOperation() &&
          op->hasAttr(CVPipeline::kOuterMainLoopHint)) {
        op->emitError("outer main-loop request was not normalized before "
                      "main-loop selection");
        return WalkResult::interrupt();
      }
      if (op->hasAttr(CVPipeline::kExplicitMainLoop)) {
        if (!isa<scf::ForOp, scf::WhileOp>(op)) {
          op->emitError("explicit main-loop selection requires a loop");
          return WalkResult::interrupt();
        }
        for (Operation *parent = op->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (parent->hasAttr(CVPipeline::kExplicitMainLoop)) {
            op->emitError(
                "nested explicit main-loop selections are unsupported");
            return WalkResult::interrupt();
          }
        }
        mainLoops.push_back(op);
      }
      return WalkResult::advance();
    });
    if (requestCheck.wasInterrupted()) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
      signalPassFailure();
      return;
    }

    for (Operation *loop : mainLoops) {
      loop->walk([&](Operation *nested) {
        if (nested != loop)
          nested->removeAttr(CVPipeline::kMainLoop);
      });
    }
  }

  // Find all candidate main loops (ForOp + WhileOp)
  auto isL1Fixpipe = [](Operation *op) -> bool {
    auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
    if (!fixpipeOp)
      return false;
    auto dstType = dyn_cast<MemRefType>(fixpipeOp.getDst().getType());
    if (!dstType)
      return false;
    auto addrSpaceAttr =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstType.getMemorySpace());
    return addrSpaceAttr &&
           addrSpaceAttr.getAddressSpace() == hivm::AddressSpace::L1;
  };

  module.walk([&](Operation *op) {
    if (!isa<hivm::FixpipeOp, hivm::CopyOp>(op))
      return;

    if (isL1Fixpipe(op))
      return;

    if (outerMainLoop) {
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (parent->hasAttr(CVPipeline::kExplicitMainLoop) ||
            parent->hasAttr(CVPipeline::kOuterEventFallback))
          return;
      }
    }

    if (auto forOp = op->getParentOfType<scf::ForOp>())
      mainLoops.push_back(forOp);
    if (auto whileOp = op->getParentOfType<scf::WhileOp>())
      mainLoops.push_back(whileOp);
  });

  for (Operation *loopOp : mainLoops) {
    if (!loopOp->hasAttr(CVPipeline::kMainLoop)) {
      // Add attribute with integer value (current counter ID)
      loopOp->setAttr(
          CVPipeline::kMainLoop,
          Builder(module.getContext()).getI32IntegerAttr(mainLoopIdCounter));
      mainLoopIdCounter++;
    }
  }

  // Remove main_loop attribute from outer loops if nested loops both have it
  // Keep only the innermost main_loop
  SmallVector<Operation *> allMainLoops;
  module.walk([&](Operation *loopOp) {
    if (CVPipeline::isMainLoopOp(loopOp)) {
      allMainLoops.push_back(loopOp);
    }
  });

  for (Operation *loopOp : allMainLoops) {
    // Check if there's any nested loop with main_loop attribute
    bool hasNestedMainLoop = false;
    loopOp->walk([&](Operation *nestedLoopOp) {
      if (nestedLoopOp != loopOp && CVPipeline::isMainLoopOp(nestedLoopOp)) {
        hasNestedMainLoop = true;
      }
    });
    // Remove attribute from outer loop if inner loop also has it
    if (hasNestedMainLoop) {
      loopOp->removeAttr(CVPipeline::kMainLoop);
    }
  }

  LOG_DEBUG("--- exit MarkMainLoopPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createMarkMainLoopPass() {
  return std::make_unique<MarkMainLoopPass>();
}
} // namespace triton
} // namespace mlir
