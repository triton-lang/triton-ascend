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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
// body performs a read-modify-write of a carried tensor through *matching*
// slices - a tensor.extract_slice of the loop-carried value and a
// tensor.insert_slice into the same carried value at the same window
// (offsets, sizes and strides all equal), where the slice is a proper
// sub-tile of the carried tensor and the inserted value is not the extracted
// value itself (that would be a pure copy). Kernels tile their accumulators
// this way to fit the vector-core UB budget of the legacy SSBUFFER plan
// (e.g. the HEAD_DIM >= 256 fused-attention accumulator update, sub-tiled by
// 1/4 of BLOCK_M: extract [BLOCK_M/4, HEAD_DIM], accumulate, insert back at
// the same offset). The dynamic CV pipeline instead plans whole-tile
// cross-core UB buffers and multi-buffers them, which exceeds UB capacity at
// the BiShengIR regbase stage ("ub overflow") and fails the compile. Decline
// such kernels so they keep the legacy plan their tiling was designed
// against.
//
// Only the paired same-window read-modify-write matches. Slicing a carried
// tensor in any other shape is NOT manual UB-driven sub-tiling and must keep
// the dynamic pipeline, e.g.:
//   - insert-only moving windows (index_select-style gather filling
//     buf[scatterRow, :] row by row, the window offset comes from the data);
//   - extract-only reads, or slicing of tiles that are not loop-carried.
static bool isSameOFR(OpFoldResult a, OpFoldResult b) {
  if (a == b) {
    return true; // same SSA value, or the same uniqued attribute
  }
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  return ca && cb && *ca == *cb;
}

static bool isSameWindow(ArrayRef<OpFoldResult> extractWindow,
                         ArrayRef<OpFoldResult> insertWindow) {
  if (extractWindow.size() != insertWindow.size()) {
    return false;
  }
  for (auto [e, i] : llvm::zip(extractWindow, insertWindow)) {
    if (!isSameOFR(e, i)) {
      return false;
    }
  }
  return true;
}

// The slice must be a strict sub-tile of the carried tensor (at least one
// static dimension strictly smaller, none larger), not a whole-tile
// reshuffle.
static bool isProperSubtile(Type carriedType, Type sliceType) {
  auto carried = dyn_cast<RankedTensorType>(carriedType);
  auto slice = dyn_cast<RankedTensorType>(sliceType);
  if (!carried || !slice || carried.getRank() != slice.getRank()) {
    return false;
  }
  ArrayRef<int64_t> carriedShape = carried.getShape();
  ArrayRef<int64_t> sliceShape = slice.getShape();
  bool strictlySmaller = false;
  for (size_t d = 0; d < carriedShape.size(); ++d) {
    if (ShapedType::isDynamic(carriedShape[d]) ||
        ShapedType::isDynamic(sliceShape[d])) {
      continue;
    }
    if (sliceShape[d] > carriedShape[d]) {
      return false;
    }
    if (sliceShape[d] < carriedShape[d]) {
      strictlySmaller = true;
    }
  }
  return strictlySmaller;
}

// Whether the (extractOp, insertOp) pair is a same-window read-modify-write
// of the loop-carried iterArg: read a sub-tile, compute something new from
// it, write the result back to the very same window.
static bool isAccumulationPair(BlockArgument iterArg,
                               tensor::ExtractSliceOp extractOp,
                               tensor::InsertSliceOp insertOp) {
  if (insertOp.getSource() == extractOp.getResult()) {
    return false; // pure copy through, no computation in between
  }
  if (!isProperSubtile(iterArg.getType(), extractOp.getResult().getType())) {
    return false;
  }
  return isSameWindow(extractOp.getMixedOffsets(),
                      insertOp.getMixedOffsets()) &&
         isSameWindow(extractOp.getMixedSizes(), insertOp.getMixedSizes()) &&
         isSameWindow(extractOp.getMixedStrides(),
                      insertOp.getMixedStrides());
}

static bool hasSlicedTileAccumulation(ModuleOp module) {
  bool found = false;
  module->walk([&](scf::ForOp forOp) -> WalkResult {
    for (BlockArgument iterArg : forOp.getRegionIterArgs()) {
      if (!isa<TensorType>(iterArg.getType())) {
        continue;
      }
      SmallVector<tensor::ExtractSliceOp> extracts;
      SmallVector<tensor::InsertSliceOp> inserts;
      forOp.getBody()->walk([&](Operation *op) {
        if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
          if (extractOp.getSource() == iterArg) {
            extracts.push_back(extractOp);
          }
        } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
          if (insertOp.getDest() == iterArg) {
            inserts.push_back(insertOp);
          }
        }
      });
      for (const auto &extractOp : extracts) {
        for (const auto &insertOp : inserts) {
          if (isAccumulationPair(iterArg, extractOp, insertOp)) {
            found = true;
            return WalkResult::interrupt();
          }
        }
      }
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
           "iter_args that are read-modify-written through matching "
           "extract/insert_slice windows (sliced-tile accumulation); the "
           "kernel is tiled against the legacy UB budget.");
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
