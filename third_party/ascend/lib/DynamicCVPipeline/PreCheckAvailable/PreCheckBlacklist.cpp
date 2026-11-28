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

#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"

using namespace mlir;
using namespace triton;

// The blacklist operations that should skip SSBUFFER
static const llvm::SmallVector<llvm::StringRef> kBlacklistOpNames = {
    "scope.scope",
};

static constexpr const char *DEBUG_TYPE = "pre-check-blacklist";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

// Sliced-tile accumulation idiom: a scf.for carrying tensor iter_args whose
// body slices a carried tensor itself (tensor.extract_slice of the
// loop-carried value, or tensor.insert_slice into it). Kernels written this
// way (e.g. the HEAD_DIM >= 256 fused-attention accumulator update, which
// sub-tiles by 1/4 of BLOCK_M) tile their accumulators manually to fit the
// vector-core UB budget of the legacy SSBUFFER plan. The dynamic CV
// pipeline instead plans whole-tile cross-core UB buffers and multi-buffers
// them, which exceeds UB capacity at the BiShengIR regbase stage ("ub
// overflow") and fails the compile. Decline such kernels so they keep the
// legacy plan their tiling was designed against.
static bool hasSlicedTileAccumulation(ModuleOp module) {
  bool found = false;
  module->walk([&](scf::ForOp forOp) -> WalkResult {
    for (BlockArgument iterArg : forOp.getRegionIterArgs()) {
      if (!isa<TensorType>(iterArg.getType())) {
        continue;
      }
      forOp.getBody()->walk([&](Operation *op) -> WalkResult {
        if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
          if (extractOp.getSource() == iterArg) {
            found = true;
            return WalkResult::interrupt();
          }
        } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
          if (insertOp.getDest() == iterArg) {
            found = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (found) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

void PreCheckBlacklistPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<scope::ScopeDialect>();
}

void PreCheckBlacklistPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  Operation *foundBlacklistOp = nullptr;
  llvm::StringRef foundOpName;

  // Check for all blacklist operations
  module.walk([&](Operation *op) -> WalkResult {
    llvm::StringRef opName = op->getName().getStringRef();
    if (llvm::is_contained(kBlacklistOpNames, opName)) {
      foundBlacklistOp = op;
      foundOpName = opName;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!foundBlacklistOp) {
    if (hasSlicedTileAccumulation(module)) {
      LDBG("SSBUFFER will be skipped because a scf.for carries tensor "
           "iter_args that are sliced in-body (sliced-tile accumulation); "
           "the kernel is tiled against the legacy UB budget.");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
      return;
    }
    LDBG("No blacklist operations found, passed.");
    return;
  }

  LDBG("SSBUFFER will be skipped because "
       << foundOpName
       << " operation was found, which indicates that it has been optimized "
          "for the Ascend.");
  CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckBlacklistPass() {
  return std::make_unique<PreCheckBlacklistPass>();
}

} // namespace triton
} // namespace mlir
