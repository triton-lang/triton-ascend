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

#include "ascend/include/CVSplitScheduling/PreCheck.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "cv-split-pre-check"

using namespace mlir;
using mlir::triton::cv_split::kNzTileSize;
namespace {

static SmallVector<scf::ForOp> collectInnermostLoops(func::FuncOp funcOp) {
  SmallVector<scf::ForOp> loops;
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedLoop = false;
    forOp.getBody()->walk([&](scf::ForOp) {
      hasNestedLoop = true;
      return WalkResult::interrupt();
    });
    if (!hasNestedLoop)
      loops.push_back(forOp);
  });
  return loops;
}

static bool containsStore(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (isa<memref::StoreOp, tensor::InsertSliceOp,
            bufferization::MaterializeInDestinationOp>(op))
      return WalkResult::interrupt();

    if (op->getName().getStringRef().contains("store"))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool containsBranching(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (isa<RegionBranchOpInterface>(op) || op->getNumSuccessors() != 0)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static LogicalResult checkStaticTensorType(Type type) {
  if (!isa<TensorType>(type))
    return success();

  auto rankedType = dyn_cast<RankedTensorType>(type);
  return success(rankedType && rankedType.hasStaticShape());
}

static LogicalResult checkStaticTensorShapes(scf::ForOp forOp) {
  auto checkValue = [](Value value) {
    return checkStaticTensorType(value.getType());
  };

  for (Value initArg : forOp.getInitArgs())
    if (failed(checkValue(initArg)))
      return failure();

  for (Value result : forOp.getResults())
    if (failed(checkValue(result)))
      return failure();

  for (BlockArgument argument : forOp.getBody()->getArguments())
    if (failed(checkValue(argument)))
      return failure();

  WalkResult walkResult = forOp.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (failed(checkValue(operand)))
        return WalkResult::interrupt();

    for (Value result : op->getResults())
      if (failed(checkValue(result)))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static LogicalResult checkMatmulOutsAreRankedTensors(scf::ForOp forOp) {
  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    Value outs = matmulOp.getDpsInitOperand(0)->get();
    if (!isa<RankedTensorType>(outs.getType()))
      return failure();
  }
  return success();
}

static bool containsMatmul(scf::ForOp forOp) {
  return llvm::any_of(*forOp.getBody(),
                      [](Operation &op) { return isa<linalg::MatmulOp>(op); });
}

static LogicalResult checkMatmulDimensionsAre16Aligned(scf::ForOp forOp) {
  auto is16AlignedRank2Tensor = [](Type type) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || tensorType.getRank() != 2)
      return false;
    for (int64_t dim : tensorType.getShape())
      if (dim <= 0 || dim % kNzTileSize != 0)
        return false;
    return true;
  };

  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    for (Value operand : matmulOp->getOperands())
      if (!is16AlignedRank2Tensor(operand.getType()))
        return failure();
    for (Value result : matmulOp->getResults())
      if (!is16AlignedRank2Tensor(result.getType()))
        return failure();
  }
  return success();
}

// ROW_SPLIT assigns M/2 rows to each vector core. The subsequent V->C NZ pack
// groups those per-core rows in blocks of 16, so the original M must be a
// multiple of 2 * 16 = 32.
static LogicalResult checkMatmulRowsAre32Aligned(scf::ForOp forOp) {
  for (Operation &op : *forOp.getBody()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp)
      continue;

    auto resultType =
        dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 ||
        resultType.getShape()[0] <= 0 ||
        resultType.getShape()[0] % (2 * kNzTileSize) != 0)
      return failure();
  }
  return success();
}

static LogicalResult checkStaticDivisibleTripCount(scf::ForOp forOp,
                                                   int unrollFactor) {
  std::optional<int64_t> lowerBound =
      getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> upperBound =
      getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0 ||
      *upperBound <= *lowerBound)
    return failure();

  int64_t distance;
  if (llvm::SubOverflow(*upperBound, *lowerBound, distance))
    return failure();

  int64_t tripCount = distance / *step + (distance % *step != 0);
  return success(tripCount >= unrollFactor && tripCount % unrollFactor == 0);
}

} // namespace

FailureOr<scf::ForOp>
mlir::triton::cv_split::preCheckCVSplitScheduling(func::FuncOp funcOp,
                                                  int unrollFactor) {
  if (unrollFactor != 2 && unrollFactor != 4 && unrollFactor != 8) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] unsupported unroll factor: "
               << unrollFactor << "\n");
    return failure();
  }

  SmallVector<scf::ForOp> candidates = collectInnermostLoops(funcOp);
  if (candidates.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] expected one innermost loop in "
               << funcOp.getName() << ", found " << candidates.size() << "\n");
    return failure();
  }

  // this is the chosen for loop
  scf::ForOp candidate = candidates.front();

  if (failed(checkStaticTensorShapes(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains an unranked "
                  "or dynamically shaped tensor\n");
    return failure();
  }

  if (!containsMatmul(candidate)) {
    LLVM_DEBUG(llvm::dbgs() << "[cv-split-pre-check] candidate loop must "
                               "contain at least one linalg.matmul\n");
    return failure();
  }

  if (failed(checkMatmulOutsAreRankedTensors(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul outs must be a ranked "
                  "tensor\n");
    return failure();
  }

  if (failed(checkMatmulDimensionsAre16Aligned(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul operands and results "
                  "must be rank-2 tensors with dimensions divisible by 16\n");
    return failure();
  }

  if (failed(checkMatmulRowsAre32Aligned(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] linalg.matmul output rows must be "
                  "divisible by 32 for two-core ROW_SPLIT\n");
    return failure();
  }

  if (failed(checkStaticDivisibleTripCount(candidate, unrollFactor))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop must have a static, "
                  "positive trip count divisible by the unroll factor\n");
    return failure();
  }

  if (containsBranching(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains branching\n");
    return failure();
  }

  if (containsStore(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains a store\n");
    return failure();
  }

  return candidate;
}
