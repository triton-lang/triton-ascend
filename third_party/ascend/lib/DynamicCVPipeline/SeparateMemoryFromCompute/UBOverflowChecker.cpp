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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace triton;

#define DEBUG_TYPE "UBOverflow"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static annotation::MarkOp findMarkOp(memref::AllocOp allocOp) {
  for (auto *user : allocOp.getResult().getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user))
      return markOp;
  }
  return nullptr;
}

int triton::getAlignUnit(Type elementType) {
  unsigned width = elementType.getIntOrFloatBitWidth();
  if (width == 0 || width > UBConstants::ALIGN_UNIT_BITS)
    return 1;
  return static_cast<int>(UBConstants::ALIGN_UNIT_BITS / width);
}

SmallVector<BufferInfo> triton::collectBuffers(ModuleOp module) {
  SmallVector<BufferInfo> buffers;

  module.walk([&](scope::ScopeOp scopeOp) {
    bool isCube = false;
    bool isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector)))
      return WalkResult::advance();
    if (!isVector)
      return WalkResult::advance();

    scopeOp.walk([&](memref::AllocOp allocOp) {
      BufferInfo buf;
      buf.allocOp = allocOp;
      buf.forOp = allocOp->getParentOfType<scf::ForOp>();

      annotation::MarkOp markOp = findMarkOp(allocOp);
      if (markOp) {
        if (markOp->hasAttr(hivm::MultiBufferAttr::name)) {
          buf.kind = BufferInfo::Kind::Annot;
          buf.markOp = markOp;
          buf.fromHint = markOp->hasAttr(CVPipeline::kGMLoadHintAttr);
          if (buf.fromHint)
            markOp->removeAttr(CVPipeline::kGMLoadHintAttr);
        } else {
          buf.kind = BufferInfo::Kind::Unannot;
        }
      } else {
        buf.kind = BufferInfo::Kind::Unannot;
      }

      buffers.push_back(buf);
    });

    return WalkResult::advance();
  });

  LOG_DEBUG("collected " << buffers.size() << " buffers");
  return buffers;
}

void triton::computeBufferSize(BufferInfo &buf) {
  auto memrefType = mlir::cast<MemRefType>(buf.allocOp.getResult().getType());
  ArrayRef<int64_t> shape = memrefType.getShape();
  Type elementType = memrefType.getElementType();
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();

  for (auto dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      LOG_DEBUG("dynamic dim encountered, skipping size computation for "
                << buf.allocOp);
      return;
    }
  }

  if (shape.empty() || shape[0] == 0) {
    LOG_DEBUG("zero-sized or rank-0 buffer, skipping size computation");
    return;
  }

  int64_t numElements = 1;
  for (auto dim : shape) {
    if (numElements > std::numeric_limits<int64_t>::max() / dim) {
      LOG_DEBUG("overflow in element count computation, clamping");
      numElements = std::numeric_limits<int64_t>::max();
      break;
    }
    numElements *= dim;
  }
  buf.originalSize = numElements * bitWidth;

  // TileAndBindSubBlock: dim0 = ceil(dim0 / K_SUB_BLOCK_DIM)
  int64_t reducedDim0 = (shape[0] + UBConstants::K_SUB_BLOCK_DIM - 1) /
                        UBConstants::K_SUB_BLOCK_DIM;
  int64_t reducedElements = numElements * reducedDim0 / shape[0];
  buf.reducedSize = reducedElements * bitWidth;

  // EnableStrideAlign: align last dim to 32B
  int64_t lastDim = (shape.size() == 1) ? reducedDim0 : shape.back();
  int alignUnit = getAlignUnit(elementType);

  if (lastDim % alignUnit != 0) {
    int64_t alignedLastDim = (lastDim + alignUnit - 1) / alignUnit * alignUnit;
    buf.alignedSize =
        (buf.reducedSize * alignedLastDim + lastDim - 1) / lastDim;
    buf.alignedSize =
        llvm::alignTo(buf.alignedSize, UBConstants::ALIGN_UNIT_BITS);
  } else {
    buf.alignedSize = buf.reducedSize;
  }
}

UBEstimateResult triton::checkUBOverflow(ModuleOp module) {
  auto buffers = collectBuffers(module);
  for (auto &buf : buffers)
    computeBufferSize(buf);

  UBEstimateResult result;
  int64_t total = 0;
  for (auto &buf : buffers) {
    if (buf.alignedSize <= 0)
      continue;
    int64_t n = 1;
    if (buf.kind == BufferInfo::Kind::Annot && buf.markOp)
      if (auto attr = buf.markOp->getAttrOfType<IntegerAttr>(
              hivm::MultiBufferAttr::name))
        n = attr.getInt();
    // Conservative: assume full expansion (alignedSize × N). Will be
    // refined with PlanMemory-based for-loop tree model later.
    total += buf.alignedSize * n;
  }

  result.totalBits = total;

  LOG_DEBUG("UB = " << result.totalBits << " bits");
  return result;
}

LogicalResult triton::pruneMultiBufferMarks(ModuleOp module) {
  auto buffers = collectBuffers(module);
  for (auto &buf : buffers)
    computeBufferSize(buf);

  int64_t total = 0;
  SmallVector<std::pair<annotation::MarkOp, int64_t>> annotMarks;
  for (auto &buf : buffers) {
    if (buf.alignedSize <= 0)
      continue;
    int64_t n = 1;
    if (buf.kind == BufferInfo::Kind::Annot && buf.markOp)
      if (auto attr = buf.markOp->getAttrOfType<IntegerAttr>(
              hivm::MultiBufferAttr::name))
        n = attr.getInt();
    total += buf.alignedSize * n;

    if (buf.kind == BufferInfo::Kind::Annot && buf.markOp &&
        buf.alignedSize > 0 && !buf.fromHint) {
      annotMarks.push_back({buf.markOp, buf.alignedSize});
    }
  }

  LOG_DEBUG("initial UB = " << total << " bits, max = "
                            << UBConstants::UB_SPACE_SIZE_BITS << " bits");

  if (total <= UBConstants::UB_SPACE_SIZE_BITS) {
    LOG_DEBUG("safe, no pruning needed");
    return success();
  }

  if (annotMarks.empty()) {
    LOG_DEBUG("no computable multi_buffer marks to prune");
    return success();
  }

  llvm::sort(annotMarks,
             [](const auto &a, const auto &b) { return a.second > b.second; });

  LOG_DEBUG("collected " << annotMarks.size() << " annot marks");

  int deleted = 0;
  for (auto &[markOp, size] : annotMarks) {
    markOp->removeAttr(hivm::MultiBufferAttr::name);
    ++deleted;

    auto result = checkUBOverflow(module);
    LOG_DEBUG("after deleting mark #" << deleted << " (size " << size
                                      << " bits): UB = " << result.totalBits
                                      << " bits");

    if (result.totalBits <= UBConstants::UB_SPACE_SIZE_BITS)
      break;
  }

  return success();
}

void UBOverflowCheckerPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG("Enter UBOverflowChecker pass");

  if (failed(pruneMultiBufferMarks(module))) {
    LOG_DEBUG("pruneMultiBufferMarks failed (non-fatal)");
  }

  LOG_DEBUG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUBOverflowCheckerPass() {
  return std::make_unique<UBOverflowCheckerPass>();
}

void registerUBOverflowCheckerPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createUBOverflowCheckerPass();
  });
}

} // namespace triton
} // namespace mlir
