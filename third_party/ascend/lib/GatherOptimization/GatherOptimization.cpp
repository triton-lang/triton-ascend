/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "GatherOptimization/Passes.h"
#include "TritonToUnstructure/OffsetAnalysis.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gather-optimization"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_GATHEROPTIMIZATION
#include "ascend/include/GatherOptimization/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using AxisInfo = PtrOffsetInfo::AxisInfo;

namespace {
struct GatherOptimizationPass
    : public mlir::triton::impl::GatherOptimizationBase<
          GatherOptimizationPass> {
  void runOnOperation() override;
};
} // namespace

bool GatherOptimizationConversionPattern::isIntegerTensorType(
    mlir::Type type, int &numDimensionsOut) const {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isInteger()) {
    return false;
  }
  numDimensionsOut = tensorType.getShape().size();
  return true;
}

ArrayRef<int64_t>
GatherOptimizationConversionPattern::getShapeFromType(mlir::Type type) const {
  ArrayRef<int64_t> shape;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (tensorType) {
    shape = tensorType.getShape();
  }
  return shape;
}

template <typename TIOp>
mlir::Value reduce(mlir::Value inputTensor, mlir::Location loc,
                   PatternRewriter &rewriter) {
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(inputTensor.getType());
  if (!tensorType) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[reduce] input must be a RankedTensorType\n";
    });
    return nullptr;
  }
  auto elementType = tensorType.getElementType();
  auto shape = tensorType.getShape();
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[reduce] reduce tensor of rank { " << shape.size() << " }: { "
       << inputTensor << " }\n";
  });
  int64_t totalElements = 1;
  for (int64_t dim : shape) {
    totalElements *= dim;
  }

  auto flatTensorType =
      mlir::RankedTensorType::get({totalElements}, elementType);
  auto flatTensor =
      rewriter.create<triton::ReshapeOp>(loc, flatTensorType, inputTensor);

  auto reduceOp = rewriter.create<triton::ReduceOp>(
      loc, mlir::ValueRange{flatTensor.getResult()}, 0);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    mlir::SmallVector<mlir::Type, 2> argTypes = {elementType, elementType};
    mlir::SmallVector<mlir::Location, 2> argLocs = {loc, loc};
    mlir::Block *block = rewriter.createBlock(
        &reduceOp.getRegion(), reduceOp.getRegion().end(), argTypes, argLocs);

    auto mathOp = rewriter.create<TIOp>(loc, block->getArgument(0),
                                        block->getArgument(1));

    rewriter.create<triton::ReduceReturnOp>(loc, mathOp.getResult());
  }
  return reduceOp.getResults()[0];
}

/*
  Analyze if an operation matches the conditions and extract the variables
  needed for the rewrite.
*/
bool GatherOptimizationConversionPattern::analyze(
    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Operation *analyzedOp,
    mlir::Value &ourIndices, int &indexRank, llvm::SmallVector<int64_t> &shape,
    mlir::Value &rowOffset, mlir::Value &srcPtr, int &gatherAxis) const {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[analyze] analyzing op: { " << *analyzedOp << " }\n";
  });

  std::function<bool(Operation *)> isFullyUnstructured =
      [&](Operation *op) -> bool {
    // Find first op with fully unstructured result
    if (auto addIOp = dyn_cast<arith::AddIOp>(op)) {
      return offsetMap[addIOp.getLhs()].isUnstructured() &&
             offsetMap[addIOp.getRhs()].isUnstructured();
    }
    return offsetMap[op->getResult(0)].isUnstructured();
  };
  std::function<bool(Operation *)> isSplatPointer = [&](Operation *op) -> bool {
    // Find src pointer
    if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
      auto src = splatOp.getSrc();
      return isa<mlir::BlockArgument>(src) &&
             isa<triton::PointerType>(src.getType());
    }
    return false;
  };
  // Find indices tensor
  std::function<bool(Operation *)> stopStructured = [&](Operation *op) -> bool {
    // Any op that has structured result is not built from our indices
    return offsetMap[op->getResult(0)].isStructured();
  };
  auto indicesOp = findPrecedingOpWithCondition(analyzedOp, isFullyUnstructured,
                                                stopStructured);
  if (!indicesOp) {
    return false;
  }
  ourIndices = indicesOp->getResult(0);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[analyze] indices source op: { " << ourIndices << " }\n";
  });

  if (!isIntegerTensorType(ourIndices.getType(), indexRank) || indexRank > 5) {
    return false;
  }

  // Find src shape analysis starting op
  std::function<bool(Operation *)> stopIndices = [&](Operation *op) -> bool {
    // Stop analyzing branch if it's indices
    return op == indicesOp;
  };
  std::function<bool(Operation *)> isStructuredNotScalar =
      [&](Operation *op) -> bool {
    // Finds first structured non-scalarlike op for src grid analysis
    return offsetMap[op->getResult(0)].isStructured() &&
           !offsetMap[op->getResult(0)].isScalarLike();
  };
  auto srcAnalysisStart = findPrecedingOpWithCondition(
      analyzedOp, isStructuredNotScalar, stopIndices);
  if (!srcAnalysisStart) {
    return false;
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[analyze] root op for src shape analysis: { " << *srcAnalysisStart
       << " }\n";
  });

  std::function<bool(Operation *)> isNotUnstructuredNotSplat =
      [&](Operation *op) -> bool {
    // Find tensor that has useful axis info
    return !offsetMap[op->getResult(0)].isUnstructured() && !isSplatPointer(op);
  };
  /* Determine gather axis. Gather axis will be the only one scalar-like in
   * not-unstructured pointer base */
  gatherAxis = 0;
  auto baseNotUnstructuredOp = findPrecedingOpWithCondition(
      analyzedOp, isNotUnstructuredNotSplat, stopIndices);
  if (!baseNotUnstructuredOp) {
    return false;
  }
  auto baseInfo = offsetMap[baseNotUnstructuredOp->getResult(0)];
  auto baseInfoStructured = baseInfo.getStructured();
  for (int i = 1; i < baseInfo.getRank(); i++) {
    if (baseInfoStructured[i] == AxisInfo::scalar) {
      // 1 element axis -> stride = 1 (no multiplication)
      continue;
    }
    if (baseInfoStructured[i] == AxisInfo::scalarlike) {
      if (gatherAxis != 0) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << "[analyze] more than one axis could potentially be the gather "
                "axis\n";
        });
        return false;
      }
      gatherAxis = i;
    }
  }
  if (gatherAxis == 0) {
    // Gather axis not found (* might be 1 element)
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[analyze] gather axis not found\n";
    });
    return false;
  }
  if (gatherAxis != indexRank - 1) {
    // For now handle only last dimension gather patterns
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[analyze] gather over non-last axis: { " << gatherAxis << " }\n";
    });
    return false;
  }

  auto srcSplat =
      findPrecedingOpWithCondition(analyzedOp, isSplatPointer, stopIndices);
  if (!srcSplat) {
    return false;
  }
  srcPtr = dyn_cast<triton::SplatOp>(srcSplat).getSrc();

  // Prepare grid for loading source
  auto indexShape = getShapeFromType(ourIndices.getType());

  shape.push_back(indexShape[0]);

  int checkRank = indexRank - 1;
  std::function<bool(Operation *)> stopSplatOrNotLastDimExp =
      [&](Operation *op) -> bool {
    // Stop analyzing branch when condition is true
    // (we won't get proper shapes from it)
    if (auto expDimsOp = dyn_cast<triton::ExpandDimsOp>(op)) {
      auto axisAttr = expDimsOp.getAxisAttr();
      return axisAttr.getInt() != checkRank;
    }
    return isSplatPointer(op);
  };
  int addExpDim = 0; // (n mutiplications in a row "cancel out" 1-sized
                     // dimensions added by n expansions in a row)
  // Recursively look for operations shaping source
  std::function<bool(Operation *)> isSrcShapeFound =
      [&](Operation *op) -> bool {
    // Analyze source shape until all dimensions are found
    auto parentType = addExpDim;
    if (auto expDimsOp = dyn_cast<triton::ExpandDimsOp>(op)) {
      if (addExpDim) {
        addExpDim--;
      }
      checkRank--;
      if (checkRank == 0 ||
          findPrecedingOpWithCondition(op, isSrcShapeFound,
                                       stopSplatOrNotLastDimExp)) {
        if (parentType == 0) {
          shape.push_back(1);
        }
        return true;
      }
      checkRank++;
    } else if (auto mulIOp = dyn_cast<arith::MulIOp>(op)) {
      addExpDim++;
      if (findPrecedingOpWithCondition(op, isSrcShapeFound,
                                       stopSplatOrNotLastDimExp)) {
        uint64_t dim;
        arith::ConstantOp constLast = nullptr;

        constLast = mulIOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!constLast) {
          constLast = mulIOp.getLhs().getDefiningOp<arith::ConstantOp>();
        }
        if (!constLast) {
          LLVM_DEBUG({
            auto &os = llvm::dbgs();
            os << "[analyze] no const found in analyzed MulI: {" << mulIOp
               << "}\n";
          });
          addExpDim = parentType;
          return false;
        }
        if (auto denseAttr =
                mlir::dyn_cast<mlir::DenseElementsAttr>(constLast.getValue())) {
          if (!denseAttr.getElementType().isInteger() || !denseAttr.isSplat()) {
            addExpDim = parentType;
            return false;
          }
          dim = denseAttr.getSplatValue<mlir::IntegerAttr>().getInt();
        } else if (auto intAttr = dyn_cast<IntegerAttr>(constLast.getValue())) {
          dim = intAttr.getInt();
        } else {
          LLVM_DEBUG({
            auto &os = llvm::dbgs();
            os << "[analyze] unsupported constant type in MulI\n";
          });
          addExpDim = parentType;
          return false;
        }
        shape.push_back(dim);
        return true;
      }
    } else if (llvm::any_of(op->getOperands(), [](auto val) {
                 return isa<mlir::BlockArgument>(val);
               })) {
      return true;
    }
    if (!findPrecedingOpWithCondition(op, isSrcShapeFound,
                                      stopSplatOrNotLastDimExp)) {
      addExpDim = parentType;
      return false;
    }
    return true;
  };

  if (findOperandDefinitionWithCondition(srcAnalysisStart->getResult(0),
                                         isSrcShapeFound,
                                         stopSplatOrNotLastDimExp) == nullptr) {
    if (checkRank != 1) {
      return false;
    }
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[analyze] src shape: {";
    for (int i = 0; i < shape.size(); i++) {
      os << " " << shape[i];
    }
    os << " }\n";
  });
  // Skip gather axis comparison
  if (shape.size() != indexShape.size()) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[analyze] rank mismatch { src: " << shape.size()
         << ", idx: " << indexShape.size() << " }\n";
    });
    return false;
  }
  for (int d = 0; d < shape.size(); d++) {
    if (d != gatherAxis && indexShape[d] > shape[d]) {
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "[analyze] indices shape mismatch: {";
        for (int i = 0; i < indexShape.size(); i++) {
          os << " " << indexShape[i];
        }
        os << " }\n";
      });
      return false;
    }
  }

  std::function<bool(Operation *)> isRank0 = [&](Operation *op) -> bool {
    // Find unmodified row offsets
    return offsetMap[op->getResult(0)].getRank() == 0 &&
           !isa<arith::MulIOp>(op);
  };
  if (auto rank0 = findPrecedingOpWithCondition(srcAnalysisStart, isRank0,
                                                stopIndices)) {
    rowOffset = rank0->getResult(0);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[analyze] rowOffset: { " << rowOffset << " }\n";
  });

  return true;
}

/*
  Rewrite a load operation if it matches gather pattern so it uses gather if all
  indices are in proper range.
*/
bool GatherOptimizationConversionPattern::tryOptimise(
    triton::LoadOp loadOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[tryOptimise] optimizing load op: { " << loadOp << " }\n";
  });
  auto addPtrOp = loadOp.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp) {
    // Loaded offsets must be a sum of srcPtr and indices
    return false;
  }

  auto loadTensorType = mlir::dyn_cast<mlir::TensorType>(loadOp.getType());
  if (!loadTensorType) {
    return false;
  }
  auto loadShape = loadTensorType.getShape();
  auto ptrTensorType =
      mlir::dyn_cast<mlir::TensorType>(loadOp.getPtr().getType());
  if (!ptrTensorType) {
    return false;
  }
  auto ptrShape = ptrTensorType.getShape();
  if (loadShape != ptrShape) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] result and ptr shapes do not match - skip gather "
            "optimisation\n";
    });
    return false;
  }

  // Create offset map for analysis
  llvm::DenseMap<Value, PtrOffsetInfo> offsetMap;
  parse(addPtrOp, loadOp.getLoc(), rewriter, offsetMap);

  auto addPtrInfo = offsetMap[addPtrOp];
  if (!addPtrInfo.isUnstructured() || addPtrInfo.getRank() == 1) {
    // For rank == 1 gather is identical to 'index select'
    // For higher ranks 'index select' has 1 structured axis
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] pointer is not unstructured or fits 'index select' "
            "pattern - skip gather optimisation\n";
    });
    // Only fully unstructured indices are of interest to us
    return false;
  }

  // Analyze pointer and retrieve necessary data
  mlir::Value ourIndices, srcPtr, rowOffset = nullptr;
  int rank, gatherAxis;
  llvm::SmallVector<int64_t> srcShape;
  if (!analyze(offsetMap, loadOp.getPtr().getDefiningOp(), ourIndices, rank,
               srcShape, rowOffset, srcPtr, gatherAxis)) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] op not qualified for gather optimisation\n";
    });
    return false;
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[tryOptimise] start rewrite\n";
  });

  // Transform
  auto indicesTensorType =
      mlir::dyn_cast<mlir::TensorType>(ourIndices.getType());
  auto indicesShape = indicesTensorType.getShape();
  auto indexType = indicesTensorType.getElementType();

  auto loadElementType = loadTensorType.getElementType();
  auto elementTypeBitWidth = loadTensorType.getElementTypeBitWidth();

  auto loc = loadOp.getLoc();

  // Check for correct idx values, we need to make sure that
  // -srcShape[gatherAxis] <= idx < srcShape[gatherAxis]
  auto minIndex = reduce<arith::MinSIOp>(ourIndices, loc, rewriter);
  if (!minIndex) {
    return false;
  }
  auto minAllowedIndex = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(indexType, -srcShape[gatherAxis]));
  auto minCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                minIndex, minAllowedIndex);

  auto maxIndex = reduce<arith::MaxSIOp>(ourIndices, loc, rewriter);
  if (!maxIndex) {
    return false;
  }
  auto maxAllowedIndex = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(indexType, srcShape[gatherAxis]));
  auto maxCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                maxIndex, maxAllowedIndex);
  // AND those two checks
  auto cond = rewriter.create<arith::AndIOp>(loc, maxCond.getResult(),
                                             minCond.getResult());

  auto outputTensor = rewriter.create<tensor::EmptyOp>(
      loc, indicesShape, loadTensorType.getElementType());

  auto ifOp = rewriter.create<scf::IfOp>(loc, /*yieldType*/ loadOp.getType(),
                                         cond, /*hasElse=*/true);
  {
    mlir::OpBuilder thenBuilder =
        ifOp.getThenBodyBuilder(rewriter.getListener());
    // "tt.gather" does not accept indices dims smaller than source dims for
    // non-gather dimensions
    llvm::SmallVector<int64_t> gatherStrides(srcShape.size()),
        gatherOffsets(srcShape.size()), gatherShape(srcShape.size());
    // Create grid to load src
    mlir::Value grid = nullptr;
    {
      llvm::SmallVector<int64_t> srcStrides(srcShape.size());
      {
        int64_t s = 1;
        for (int i = srcShape.size() - 1; i >= 0; i--) {
          srcStrides[i] = s;
          s *= srcShape[i];
        }
      }
      auto i32Type = thenBuilder.getI32Type();
      auto gridTy = mlir::RankedTensorType::get(srcShape, i32Type);
      for (int i = 0; i < srcShape.size(); i++) {
        gatherOffsets[i] = 0;
        gatherStrides[i] = 1;
        gatherShape[i] = (i == gatherAxis) ? srcShape[i] : indicesShape[i];
        auto rangeTy = mlir::RankedTensorType::get({gatherShape[i]}, i32Type);
        mlir::Value range = thenBuilder.create<triton::MakeRangeOp>(
            loc, rangeTy, 0, gatherShape[i]);
        mlir::Value expanded;
        // Incorporate row offset
        if ((i == 0) && (rowOffset)) {
          auto offset =
              thenBuilder.create<triton::SplatOp>(loc, rangeTy, rowOffset);
          expanded = thenBuilder.create<arith::AddIOp>(loc, range, offset);
        } else {
          expanded = range;
        }
        // Expand to match target rank
        for (int d = 0; d < i; d++) {
          llvm::SmallVector<int64_t> expandedShape(
              cast<mlir::RankedTensorType>(expanded.getType()).getShape());
          expandedShape.insert(expandedShape.begin(), 1);
          auto expandedTy = mlir::RankedTensorType::get(expandedShape, i32Type);
          expanded = thenBuilder.create<triton::ExpandDimsOp>(
              loc, expandedTy, expanded, /*axis=*/0);
        }
        for (int d = i + 1; d < srcShape.size(); d++) {
          auto curShape =
              cast<mlir::RankedTensorType>(expanded.getType()).getShape();
          llvm::SmallVector<int64_t> expandedShape(curShape.begin(),
                                                   curShape.end());
          expandedShape.push_back(1);
          auto expandedTy = mlir::RankedTensorType::get(expandedShape, i32Type);
          expanded = thenBuilder.create<triton::ExpandDimsOp>(
              loc, expandedTy, expanded, /*axis=*/curShape.size());
        }
        // Broadcast to target shape
        mlir::Value bcast =
            thenBuilder.create<triton::BroadcastOp>(loc, gridTy, expanded);
        // Apply stride
        mlir::Value strideVal = thenBuilder.create<arith::ConstantIntOp>(
            loc, i32Type, srcStrides[i]);
        mlir::Value strideSplat =
            thenBuilder.create<triton::SplatOp>(loc, gridTy, strideVal);
        mlir::Value strideMulI =
            thenBuilder.create<arith::MulIOp>(loc, bcast, strideSplat);
        // Accumulate
        if (grid) {
          grid = thenBuilder.create<arith::AddIOp>(loc, grid, strideMulI);
        } else {
          grid = strideMulI;
        }
      }
    }
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] grid for loading src: { " << grid << " }\n";
    });

    // Normalize negative values: idx = idx + (srcShape[rank - 1] & (idx >>
    // (bitWidth - 1)))
    auto bitWidth = indexType.getIntOrFloatBitWidth();
    auto cShift = thenBuilder.create<arith::ConstantOp>(
        loc, thenBuilder.getIntegerAttr(indexType, bitWidth - 1));
    auto cShiftTensor =
        thenBuilder.create<triton::SplatOp>(loc, ourIndices.getType(), cShift);

    auto shifted =
        thenBuilder.create<arith::ShRSIOp>(loc, ourIndices, cShiftTensor);
    auto srcColsTensor = thenBuilder.create<triton::SplatOp>(
        loc, indicesTensorType, maxAllowedIndex);
    auto mask = thenBuilder.create<arith::AndIOp>(loc, srcColsTensor, shifted);

    auto indexNormalized =
        thenBuilder.create<arith::AddIOp>(loc, ourIndices, mask);

    auto newSplat = thenBuilder.create<triton::SplatOp>(
        loc, RankedTensorType::get(srcShape, srcPtr.getType()), srcPtr);
    auto addPtr = thenBuilder.create<triton::AddPtrOp>(
        loc, newSplat.getType(), newSplat.getResult(), grid);
    auto load = thenBuilder.create<triton::LoadOp>(
        loc, addPtr, mlir::Value(), mlir::Value(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());

    load->setAttr("gather.optimised.load",
                  mlir::StringAttr::get(load->getContext(), "source"));
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] load src op: { " << load << " }\n";
    });
    auto input = load.getResult();

    if (gatherShape != srcShape) {
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "[tryOptimise] adjust src to match indices shape (mitigate "
              "\"tt.gather\" error)\n";
      });
      auto resultTy = mlir::RankedTensorType::get(gatherShape, loadElementType);
      input = thenBuilder.create<mlir::tensor::ExtractSliceOp>(
          loc, resultTy, load, ValueRange{}, ValueRange{}, ValueRange{},
          gatherOffsets, gatherShape, gatherStrides);
    }

    auto gather = thenBuilder.create<triton::GatherOp>(
        loc, loadTensorType, input, indexNormalized, rank - 1);

    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "[tryOptimise] optimized gather op (last axis): { " << gather
         << " }\n";
    });

    auto copyOp = thenBuilder.create<linalg::CopyOp>(
        loc, TypeRange{outputTensor.getType()}, ValueRange{gather},
        ValueRange{outputTensor});
    thenBuilder.create<scf::YieldOp>(loc, copyOp->getResult(0));
  }
  {
    mlir::OpBuilder elseBuilder =
        ifOp.getElseBodyBuilder(rewriter.getListener());

    mlir::IRMapping mapping;
    Operation *clonedLoad = elseBuilder.clone(*loadOp, mapping);
    clonedLoad->setAttr(
        "gather.optimised.load",
        mlir::StringAttr::get(clonedLoad->getContext(), "fallback"));
    auto copyOp = elseBuilder.create<linalg::CopyOp>(
        loc, TypeRange{outputTensor.getType()},
        ValueRange{clonedLoad->getResult(0)}, ValueRange{outputTensor});
    elseBuilder.create<scf::YieldOp>(loc, copyOp->getResult(0));
  }
  rewriter.replaceOp(loadOp, ifOp.getResult(0));

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[tryOptimise] optimisation succeeded\n\n";
  });
  return true;
}

LogicalResult
mlir::triton::GatherOptimizationConversionPattern::matchAndRewrite(
    mlir::triton::LoadOp loadOp, PatternRewriter &rewriter) const {
  if (loadOp->getAttr(
          "gather.optimised.load")) { // load was created/rejected by this pass
    return failure();
  }
  if (tryOptimise(loadOp, rewriter)) {
    return success();
  } else {
    loadOp->setAttr("gather.optimised.load",
                    mlir::StringAttr::get(loadOp->getContext(), "no match"));
    return failure();
  }
}

void GatherOptimizationPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.add<triton::GatherOptimizationConversionPattern>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createGatherOptimizationPass() {
  return std::make_unique<GatherOptimizationPass>();
}
