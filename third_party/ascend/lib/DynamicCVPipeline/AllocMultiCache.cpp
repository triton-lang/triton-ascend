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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AllocMultiCache";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::outs() << __VA_ARGS__;                                               \
    llvm::outs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;

// Pure tensor seeds used only by the serial fallback do not need to be live
// on the scheduled path. Sink them after Cube/Vector scope separation.
static void sinkOuterFallbackFills(ModuleOp module) {
  module.walk([](scf::ForOp loop) {
    if (!loop->hasAttr(CVPipeline::kOuterEventControlCount))
      return;
    auto branch = loop->getParentOfType<scf::IfOp>();
    if (!branch || branch.getElseRegion().empty() ||
        loop->getParentRegion() != &branch.getThenRegion())
      return;
    auto &fallback = branch.getElseRegion();
    if (!llvm::any_of(fallback.front().getOps<scf::ForOp>(), [](scf::ForOp op) {
          return op->hasAttr(CVPipeline::kOuterEventFallback);
        }))
      return;
    for (auto fill : llvm::make_early_inc_range(
             branch->getBlock()->getOps<linalg::FillOp>())) {
      if (!fill->isBeforeInBlock(branch) || !fill.hasPureTensorSemantics() ||
          fill->getNumResults() != 1)
        continue;
      auto empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
      Value result = fill->getResult(0);
      if (!empty || !empty.getDynamicSizes().empty() || result.use_empty() ||
          !llvm::all_of(result.getUsers(), [&](Operation *user) {
            return fallback.isAncestor(user->getParentRegion());
          }))
        continue;
      fill->moveBefore(&fallback.front(), fallback.front().begin());
    }
  });
}

// Run the pass
void AllocMultiCachePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  if (CVPipeline::hasOuterMainLoop(module))
    sinkOuterFallbackFills(module);
  OpPassManager pm(module.getOperationName());
  LDBG("Enter pass.");
  LDBG("before innerscope:\n" << module << "\n");
  // Step 1:Inner multibuffer
  pm.addPass(createAddMultiBufferInnerScopePass());

  // Step 2: Outer multibuffer
  pm.addPass(createAddMultiBufferOuterScopePass());

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }

  LDBG("Process successfully");
  LDBG(llvm::StringRef("after innerscope:\n") << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAllocMultiCachePass() {
  return std::make_unique<AllocMultiCachePass>();
}

} // namespace triton
} // namespace mlir
