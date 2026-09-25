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

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToUnstructure/OffsetAnalysis.h"

#include "ascend/include/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "gather-optimization-rule"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

using AxisInfo = PtrOffsetInfo::AxisInfo;

//===----------------------------------------------------------------------===//
// Read-only mirror of OffsetAnalysis's classification (OffsetAnalysis::parse
// mutates IR, so it can't be used here). triton::LoadOp is the one opacity
// boundary; AddPtrOp is not, since callers walk into nested addptrs. .getPtr()
// is always null on the results; the base pointer comes from
// isSplatOfBlockArgPointer instead.
class ReadOnlyOffsetClassifier {
public:
  PtrOffsetInfo classify(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    // Recursion can grow the DenseMap. Neither callers nor this function may
    // retain references into its buckets across another classify() call.
    cache.try_emplace(value, unstructuredLike(value));
    PtrOffsetInfo result = classifyUncached(value);
    cache[value] = result;
    return result;
  }

private:
  llvm::DenseMap<Value, PtrOffsetInfo> cache;

  static PtrOffsetInfo unstructuredLike(Value value) {
    PtrOffsetInfo info;
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
      info.setUnstructured(tensorType.getRank());
    return info;
  }

  // Mirrors OffsetAnalysis::parseMulI.
  static PtrOffsetInfo combineMulI(const PtrOffsetInfo &lhs,
                                   const PtrOffsetInfo &rhs) {
    PtrOffsetInfo info;
    info.setScalarLike(lhs.isScalarLike() && rhs.isScalarLike());
    auto &lhsStructured = lhs.getStructured();
    auto &rhsStructured = rhs.getStructured();
    size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
    auto &dstStructured = info.getStructuredRef();
    dstStructured.resize(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
      if (lhs.isScalarLike())
        dstStructured[i] = rhsStructured[i];
      else if (rhs.isScalarLike())
        dstStructured[i] = lhsStructured[i];
      else
        dstStructured[i] = AxisInfo::unstructured;
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseBinaryOp.
  static PtrOffsetInfo combineBinaryLike(const PtrOffsetInfo &lhs,
                                         const PtrOffsetInfo &rhs) {
    PtrOffsetInfo info;
    bool scalarLike = lhs.isScalarLike() && rhs.isScalarLike();
    info.setScalarLike(scalarLike);
    int rank = static_cast<int>(lhs.getStructured().size());
    if (scalarLike)
      info.setStructured(rank, AxisInfo::scalarlike);
    else
      info.setUnstructured(rank);
    return info;
  }

  // Mirrors OffsetAnalysis::parseExpandDims (minus the companion offset).
  static PtrOffsetInfo combineExpandDims(const PtrOffsetInfo &src,
                                         unsigned axis) {
    PtrOffsetInfo info;
    info.setScalarLike(src.isScalarLike());
    auto &srcStructured = src.getStructured();
    auto &dstStructured = info.getStructuredRef();
    dstStructured.resize(srcStructured.size() + 1);
    size_t j = 0;
    for (size_t i = 0; i < dstStructured.size(); i++) {
      if (i == axis) {
        dstStructured[i] = AxisInfo::scalar;
      } else {
        dstStructured[i] = srcStructured[j];
        j++;
      }
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseBroadcast (minus the companion offset).
  static PtrOffsetInfo combineBroadcast(const PtrOffsetInfo &src,
                                        RankedTensorType srcType,
                                        RankedTensorType dstType) {
    // tt.broadcast also permits identical shapes. Preserve the source tags
    // without calling getBroadcastDims, which requires an expanded dimension.
    if (srcType.getShape() == dstType.getShape())
      return src;
    PtrOffsetInfo info;
    info.setScalarLike(src.isScalarLike());
    auto broadcastDim = ConverterUtils::getBroadcastDims(srcType, dstType);
    auto &srcStructured = src.getStructured();
    auto &dstStructured = info.getStructuredRef();
    auto dstShape = dstType.getShape();
    dstStructured.resize(srcStructured.size());
    for (size_t i = 0; i < dstStructured.size(); i++) {
      if (llvm::find(broadcastDim, i) != broadcastDim.end() && dstShape[i] != 1)
        dstStructured[i] = AxisInfo::scalarlike;
      else
        dstStructured[i] = srcStructured[i];
    }
    return info;
  }

  // Mirrors OffsetAnalysis::parseSplat's tagging.
  static PtrOffsetInfo splatTagging(RankedTensorType dstType) {
    PtrOffsetInfo info;
    auto &dstStructured = info.getStructuredRef();
    for (auto dim : dstType.getShape())
      dstStructured.push_back(dim == 1 ? AxisInfo::scalar
                                       : AxisInfo::scalarlike);
    info.setScalarLike(true);
    return info;
  }

  // Mirrors OffsetAnalysis::parseConstantOp.
  static PtrOffsetInfo classifyConstant(Value value) {
    if (isa<RankedTensorType>(value.getType())) {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      auto elements = constant
                          ? dyn_cast<DenseElementsAttr>(constant.getValue())
                          : DenseElementsAttr();
      if (!elements || !elements.isSplat())
        return unstructuredLike(value);
    }
    PtrOffsetInfo info;
    info.setScalarLike(true);
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
      auto &dstStructured = info.getStructuredRef();
      for (auto dim : tensorType.getShape())
        dstStructured.push_back(dim == 1 ? AxisInfo::scalar
                                         : AxisInfo::scalarlike);
    }
    return info;
  }

  PtrOffsetInfo classifyUncached(Value value) {
    if (Operation *defOp = value.getDefiningOp()) {
      // Mirrors parseAddPtr's classification.
      if (auto addPtr = dyn_cast<triton::AddPtrOp>(defOp))
        return combineInfo(classify(addPtr.getPtr()),
                           classify(addPtr.getOffset()));
      if (isa<triton::LoadOp>(defOp))
        return unstructuredLike(value);

      if (auto addI = dyn_cast<arith::AddIOp>(defOp))
        return combineInfo(classify(addI.getLhs()), classify(addI.getRhs()));
      if (auto subI = dyn_cast<arith::SubIOp>(defOp)) {
        const PtrOffsetInfo &lhs = classify(subI.getLhs());
        const PtrOffsetInfo &rhs = classify(subI.getRhs());
        PtrOffsetInfo info = combineInfo(lhs, rhs);
        if (!(lhs.isStructured() && rhs.isScalarLike()))
          info.setUnstructured(info.getRank());
        return info;
      }
      if (auto mulI = dyn_cast<arith::MulIOp>(defOp))
        return combineMulI(classify(mulI.getLhs()), classify(mulI.getRhs()));
      if (isa<arith::RemSIOp, arith::DivSIOp, arith::MulFOp, arith::DivFOp,
              arith::AddFOp, arith::SubFOp, arith::MinNumFOp, arith::MaxNumFOp,
              arith::MaxSIOp, arith::MinSIOp, arith::CmpIOp, arith::AndIOp,
              arith::OrIOp>(defOp))
        return combineBinaryLike(classify(defOp->getOperand(0)),
                                 classify(defOp->getOperand(1)));
      if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(defOp))
        return combineExpandDims(classify(expandDims.getSrc()),
                                 expandDims.getAxis());
      if (auto broadcast = dyn_cast<triton::BroadcastOp>(defOp)) {
        auto srcType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
        auto dstType =
            dyn_cast<RankedTensorType>(broadcast.getResult().getType());
        if (!srcType || !dstType)
          return unstructuredLike(value);
        return combineBroadcast(classify(broadcast.getSrc()), srcType, dstType);
      }
      if (auto splat = dyn_cast<triton::SplatOp>(defOp)) {
        if (auto dstType =
                dyn_cast<RankedTensorType>(splat.getResult().getType()))
          return splatTagging(dstType);
        return unstructuredLike(value);
      }
      if (auto indexCast = dyn_cast<arith::IndexCastOp>(defOp))
        return classify(indexCast.getIn());
      if (auto extSI = dyn_cast<arith::ExtSIOp>(defOp))
        return classify(extSI.getIn());
      if (auto bitcast = dyn_cast<triton::BitcastOp>(defOp))
        return classify(bitcast.getSrc());
      if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantFloatOp>(
              defOp))
        return classifyConstant(value);
      if (isa<triton::MakeRangeOp>(defOp)) {
        PtrOffsetInfo info;
        info.setStructured(1);
        return info;
      }
      if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(defOp)) {
        PtrOffsetInfo info;
        info.setScalarLike(true);
        return info;
      }

      // Everything else: mirrors OffsetAnalysis's own dispatch fallback.
      return unstructuredLike(value);
    }

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (parentOp && isa<FunctionOpInterface>(parentOp))
        return unstructuredLike(value);
      if (auto loopOp = dyn_cast_or_null<LoopLikeOpInterface>(parentOp)) {
        if (OpOperand *initArgOperand = loopOp.getTiedLoopInit(blockArg))
          return classify(initArgOperand->get());
      }
      // Induction variable, or an unresolvable tied-init: unstructured.
      return unstructuredLike(value);
    }

    return unstructuredLike(value);
  }
};

//===----------------------------------------------------------------------===//
// Candidate matching
//===----------------------------------------------------------------------===//

// Marks a tt.load this rule generated (the source/fallback load in a
// scf.if), so findCandidates skips it.
constexpr llvm::StringLiteral kGatherOptimisedLoadAttr =
    "gather.optimised.load";
constexpr llvm::StringLiteral kIndependentAxisTensorizeMarkerAttr =
    "hacc.independent_axis_tensorize";
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";

struct GatherCandidate {
  triton::LoadOp loadOp;
  triton::AddPtrOp addPtrOp;
  Value indices;
  Value srcPtr;
  Value
      sourceBase; // Original element offsets, invariant along the gather axis.
  // A scalar or a tensor broadcast across the gather axis; null if unmasked.
  Value sourceMask;
  int indexRank = 0;
  int gatherAxis = 0;
  SmallVector<int64_t> srcShape;
};

bool isSplatOfBlockArgPointer(Operation *op) {
  auto splatOp = dyn_cast_or_null<triton::SplatOp>(op);
  if (!splatOp)
    return false;
  Value src = splatOp.getSrc();
  return isa<BlockArgument>(src) && isa<triton::PointerType>(src.getType());
}

bool isIntegerTensorType(Type type, int &rankOut) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isInteger())
    return false;
  rankOut = tensorType.getShape().size();
  return true;
}

// Value types covered by this rule's current lowering and cost model.
bool isSupportedGatherValueType(Type elementType) {
  return elementType.isF16() || elementType.isF32() || elementType.isBF16();
}

// Recover a scalar or tensor that broadcasts across the gather axis. Keep
// the original SSA value so both pointer offsets and row masks retain their
// exact computation instead of being reconstructed from a shape heuristic.
Value findAxisInvariantValue(Value value, ArrayRef<int64_t> srcShape,
                             int gatherAxis) {
  while (auto broadcast = value.getDefiningOp<triton::BroadcastOp>())
    value = broadcast.getSrc();
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getEncoding() || !type.hasStaticShape() ||
      type.getRank() != static_cast<int64_t>(srcShape.size()) ||
      type.getDimSize(gatherAxis) != 1)
    return {};
  for (size_t axis = 0; axis < srcShape.size(); ++axis)
    if (type.getDimSize(axis) != 1 && type.getDimSize(axis) != srcShape[axis])
      return {};
  return value;
}

std::optional<uint64_t> getByteWidth(Type type) {
  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(type))
    bitWidth = integerType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(type))
    bitWidth = floatType.getWidth();
  else
    return std::nullopt;
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return static_cast<uint64_t>(bitWidth / 8);
}

bool checkedMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

// UB allocations are padded to a 32-byte block on the last axis.
constexpr uint64_t kUbBlockBytes = 32;

uint64_t roundUpToBlock(uint64_t bytes) {
  return (bytes + kUbBlockBytes - 1) / kUbBlockBytes * kUbBlockBytes;
}

// Total bytes for a tensor of this shape/element size, with its last axis
// padded to a 32-byte block (matching how it's actually laid out in UB).
// Returns nullopt on overflow.
std::optional<uint64_t> tileBytes(ArrayRef<int64_t> shape, uint64_t elemBytes) {
  if (shape.empty())
    return elemBytes;
  uint64_t outerElements = 1;
  for (int64_t dim : shape.drop_back())
    if (!checkedMulU64(outerElements, static_cast<uint64_t>(dim),
                       outerElements))
      return std::nullopt;
  uint64_t lastAxisBytes = 0;
  if (!checkedMulU64(static_cast<uint64_t>(shape.back()), elemBytes,
                     lastAxisBytes))
    return std::nullopt;
  uint64_t paddedLastAxisBytes = roundUpToBlock(lastAxisBytes);
  uint64_t total = 0;
  if (!checkedMulU64(outerElements, paddedLastAxisBytes, total))
    return std::nullopt;
  return total;
}

// Fit against real UB-overflow measurements, exact to the bit: usage =
// 2x the source tile + 3x the index footprint at the load element size +
// 4x at the index element size (mixed float16/int32 shapes pin these two
// terms down separately).
constexpr uint64_t kSrcTileMultiplier = 2;
constexpr uint64_t kIndexFootprintAtLoadElemMultiplier = 3;
constexpr uint64_t kIndexFootprintAtIndexElemMultiplier = 4;

bool exceedsUbCapacity(ArrayRef<int64_t> srcShape,
                       ArrayRef<int64_t> indicesShape, Type loadElementType,
                       Type indexElementType, unsigned ubCapacityBytes) {
  if (ubCapacityBytes == 0)
    return false;
  std::optional<uint64_t> loadElemBytes = getByteWidth(loadElementType);
  std::optional<uint64_t> indexElemBytes = getByteWidth(indexElementType);
  if (!loadElemBytes || !indexElemBytes)
    return true; // Unknown element size: don't risk it.

  std::optional<uint64_t> srcTileBytes = tileBytes(srcShape, *loadElemBytes);
  std::optional<uint64_t> idxAtLoadElemBytes =
      tileBytes(indicesShape, *loadElemBytes);
  std::optional<uint64_t> idxAtIndexElemBytes =
      tileBytes(indicesShape, *indexElemBytes);
  if (!srcTileBytes || !idxAtLoadElemBytes || !idxAtIndexElemBytes)
    return true;
  uint64_t srcContribution = 0, idxContribution1 = 0, idxContribution2 = 0,
           total = 0;
  if (!checkedMulU64(*srcTileBytes, kSrcTileMultiplier, srcContribution) ||
      !checkedMulU64(*idxAtLoadElemBytes, kIndexFootprintAtLoadElemMultiplier,
                     idxContribution1) ||
      !checkedMulU64(*idxAtIndexElemBytes, kIndexFootprintAtIndexElemMultiplier,
                     idxContribution2) ||
      !checkedAddU64(srcContribution, idxContribution1, total) ||
      !checkedAddU64(total, idxContribution2, total))
    return true;

  // The backend supplies its conservative graph UB budget (80% of raw UB).
  // Consume that budget directly, without restoring raw capacity or applying
  // a second safety factor.
  return total > ubCapacityBytes;
}

// The index tensor is always i32, so its tile row is a whole number of
// 32-byte UB blocks exactly when idxLast is a multiple of 8 -- independent
// of the gathered value's element size. fp16 data separates the index tile
// from the value tile and shows the index tile is what the DMA cares about.
constexpr int64_t kIndexBlockElems = 8;

// idx_last threshold below which tt.gather is not expected to beat the
// scalar baseline, indexed by [iteration bucket][src_last bucket]. Derived
// from measured ScalarLoop/tt.gather ratios across rank 2-5, gate-free:
// 0 fires below ratio 0.98 on 1760 pooled points, both dtypes.
constexpr int kGatherBenefitThreshold[4][9] = {
    {10, 8, 8, 6, 5, 4, 3, 2, 2}, // T <= 4
    {8, 8, 7, 5, 4, 3, 2, 2, 2},  // 5 <= T <= 16
    {8, 7, 7, 5, 4, 2, 2, 2, 2},  // 17 <= T <= 64
    {8, 7, 7, 4, 3, 2, 2, 2, 2},  // T > 64
};

// 2-byte value types (f16/bf16): half the bytes/element means it amortises
// slower, so every threshold moves up ~2 src_last buckets. T <= 4 is
// hardened, not measured -- this harness can't produce a 2-byte T <= 4
// case above 0.26ms to calibrate it.
constexpr int kGatherBenefitThresholdHalf[4][9] = {
    {14, 12, 12, 10, 9, 6, 5, 4, 4}, // T <= 4
    {12, 10, 10, 7, 6, 5, 4, 3, 2},  // 5 <= T <= 16
    {12, 10, 10, 7, 6, 5, 4, 3, 2},  // 17 <= T <= 64
    {12, 10, 10, 7, 6, 5, 4, 3, 2},  // T > 64
};

int gatherSrcBucket(int64_t srcLast) {
  if (srcLast <= 3)
    return 0;
  if (srcLast == 4)
    return 1;
  if (srcLast <= 7)
    return 2;
  if (srcLast <= 9)
    return 3;
  if (srcLast <= 15)
    return 4;
  if (srcLast <= 23)
    return 5;
  if (srcLast <= 31)
    return 6;
  // A 2-byte source tile only pays for a small index tile once it's wide
  // enough to hide the gather's own tile load: srcLast 32-44 measure
  // 0.86-1.05 at idxLast==2 where 48-64 measure 1.04-1.29.
  if (srcLast <= 47)
    return 7;
  return 8;
}

int gatherIterBucket(int64_t iterations) {
  if (iterations <= 4)
    return 0;
  if (iterations <= 16)
    return 1;
  if (iterations <= 64)
    return 2;
  return 3;
}

// True if rewriting to tt.gather is expected to be at least as fast as the
// scalar baseline it replaces. srcLast/idxLast are the gather axis sizes;
// iterations counts the enclosing row loop; rowStep is the number of rows
// processed per iteration (distinct from the Gather-side UB budget above).
bool exceedsBenefitThreshold(int64_t srcLast, int64_t idxLast,
                             int64_t iterations, int64_t rowStep,
                             uint64_t elemBytes) {
  const bool isHalf = elemBytes < 4;
  if (iterations <= 0)
    return false;

  // A width-1 source gather axis is a degenerate case with its own, much
  // later crossing point: tt.gather still pays a full tile load and the
  // gather op while the scalar baseline has almost nothing to do.
  if (srcLast == 1) {
    // At 2 bytes/element nothing below idxLast 25 ever reaches parity (best
    // 0.974 at 274 iterations); from 32 upwards the rewrite wins 6-27%. No
    // measurement exists at <=4 iterations, where the branch is 5-10% worse
    // at every other idxLast, so that corner stays closed.
    if (isHalf)
      return iterations >= 5 && idxLast >= 32;
    // A block-aligned index tile crosses early and stays there.
    if (idxLast % kIndexBlockElems == 0)
      return idxLast >= 16;
    // Few iterations: the scalar baseline pays its per-iteration overhead
    // only a handful of times and keeps winning far longer.
    if (iterations <= 4)
      return idxLast >= 32;
    if (iterations >= 17 || rowStep == 1)
      return idxLast >= 16;
    return idxLast >= 19;
  }

  if (idxLast % kIndexBlockElems == 0) // idx tile is whole 32-byte DMA blocks
    return true;
  if (rowStep == 1) // scalar baseline pays full per-iteration overhead/row
    return true;

  const auto &table =
      isHalf ? kGatherBenefitThresholdHalf : kGatherBenefitThreshold;
  return table[gatherIterBucket(iterations)][gatherSrcBucket(srcLast)] <=
         idxLast;
}

// Extracts the single splat integer constant multiplied in `mulI`.
std::optional<int64_t> extractMulIConstant(arith::MulIOp mulI) {
  arith::ConstantOp constOp = mulI.getRhs().getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    constOp = mulI.getLhs().getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
    if (!denseAttr.getElementType().isInteger() || !denseAttr.isSplat())
      return std::nullopt;
    return denseAttr.getSplatValue<IntegerAttr>().getInt();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getInt();
  return std::nullopt;
}

// Saturate only after computing ceil((upper - lower) / step). Unsigned
// subtraction handles the full signed-bound range without signed overflow;
// adding the remainder bit cannot overflow a positive-step quotient.
int64_t constantTripCount(int64_t lower, int64_t upper, int64_t step) {
  if (step <= 0)
    return 1;
  if (upper <= lower)
    return 0;
  uint64_t distance =
      static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower);
  uint64_t stride = static_cast<uint64_t>(step);
  uint64_t count = distance / stride + (distance % stride != 0);
  return static_cast<int64_t>(std::min(
      count, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

// Recover a nonnegative affine coefficient of this loop's IV in the row
// offsets. Unknown arithmetic and cancellation must not make an unrelated
// (or invariant) row appear to be driven by a long enclosing loop.
std::optional<uint64_t>
rowIvCoefficient(Value value, scf::ForOp loop,
                 DenseMap<Value, std::optional<uint64_t>> &cache) {
  auto found = cache.find(value);
  if (found != cache.end())
    return found->second;
  auto compute = [&]() -> std::optional<uint64_t> {
    if (value == loop.getInductionVar())
      return 1;
    if (loop.isDefinedOutsideOfLoop(value))
      return 0;
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<arith::ConstantOp, triton::MakeRangeOp, triton::GetProgramIdOp>(op))
      return 0;
    if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(op))
      return rowIvCoefficient(op->getOperand(0), loop, cache);
    if (isa<arith::AddIOp, arith::SubIOp>(op)) {
      auto lhs = rowIvCoefficient(op->getOperand(0), loop, cache);
      auto rhs = rowIvCoefficient(op->getOperand(1), loop, cache);
      if (!lhs || !rhs)
        return std::nullopt;
      uint64_t result;
      if (isa<arith::SubIOp>(op)) {
        if (*lhs < *rhs)
          return std::nullopt;
        result = *lhs - *rhs;
      } else if (!checkedAddU64(*lhs, *rhs, result)) {
        return std::nullopt;
      }
      return result;
    }
    if (auto mul = dyn_cast<arith::MulIOp>(op)) {
      auto factor = extractMulIConstant(mul);
      if (!factor || *factor <= 0)
        return std::nullopt;
      Value carrier = mul.getRhs().getDefiningOp<arith::ConstantOp>()
                          ? mul.getLhs()
                          : mul.getRhs();
      auto coefficient = rowIvCoefficient(carrier, loop, cache);
      uint64_t result;
      if (!coefficient || !checkedMulU64(*coefficient, *factor, result))
        return std::nullopt;
      return result;
    }
    return std::nullopt;
  };
  auto coefficient = compute();
  // Row offsets are i32. A coefficient that wraps to zero is not evidence
  // that the loop advances the row; reject large coefficients outright.
  if (coefficient && *coefficient > std::numeric_limits<int32_t>::max())
    coefficient = std::nullopt;
  cache.try_emplace(value, coefficient);
  return coefficient;
}

int64_t findRowIterations(Operation *anchor, Value sourceBase,
                          int64_t rowStep) {
  auto loop = anchor->getParentOfType<scf::ForOp>();
  if (!loop)
    return 1;
  auto step = getConstantIntValue(loop.getStep());
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  if (!step || *step <= 0 || *step != rowStep || !lower || !upper)
    return 1;
  DenseMap<Value, std::optional<uint64_t>> cache;
  auto coefficient = rowIvCoefficient(sourceBase, loop, cache);
  if (!coefficient || *coefficient == 0)
    return 1;
  return constantTripCount(*lower, *upper, *step);
}

// Searches backward from searchRoot (anchored on the pointer side), so a
// tt.expand_dims shared with the indices side is never ambiguous.
std::optional<int64_t> findAxisDimension(Operation *searchRoot,
                                         Operation *indicesOp, int64_t axis) {
  auto isAxisExpandDims = [axis](Value v) -> bool {
    auto expandDims = v.getDefiningOp<triton::ExpandDimsOp>();
    return expandDims && static_cast<int64_t>(expandDims.getAxis()) == axis;
  };
  std::function<bool(Operation *)> isMulIOfThisAxis =
      [&](Operation *op) -> bool {
    auto mulI = dyn_cast<arith::MulIOp>(op);
    return mulI &&
           (isAxisExpandDims(mulI.getLhs()) || isAxisExpandDims(mulI.getRhs()));
  };
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
  };
  // Inclusive of searchRoot itself: it is frequently the axis's own muli.
  Operation *mulI = findOperandDefinitionWithCondition(
      searchRoot->getResult(0), isMulIOfThisAxis, leavesSourceRegion);
  if (!mulI)
    return std::nullopt;
  return extractMulIConstant(cast<arith::MulIOp>(mulI));
}

// Last resort: an unmultiplied tt.expand_dims means its multiplier
// canonicalized away as arith.muli(x, 1), so the dimension is 1.
std::optional<int64_t> findFoldedUnitAxisDimension(Operation *searchRoot,
                                                   Operation *indicesOp,
                                                   int64_t axis) {
  std::function<bool(Operation *)> isThisAxisExpandDims =
      [axis](Operation *op) -> bool {
    auto expandDims = dyn_cast<triton::ExpandDimsOp>(op);
    return expandDims && static_cast<int64_t>(expandDims.getAxis()) == axis;
  };
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op);
  };
  if (findOperandDefinitionWithCondition(
          searchRoot->getResult(0), isThisAxisExpandDims, leavesSourceRegion))
    return 1;
  return std::nullopt;
}

// Fallback for axis 1 when there is no tt.expand_dims to anchor on: its
// position can be a plain scalar computation instead of a tensor one (see
// graph-optimize-gather-scalar-row.mlir).
struct ScalarAxisMatch {
  int64_t dimension;
};

std::optional<ScalarAxisMatch>
findScalarAxisDimension(Operation *searchRoot, Operation *indicesOp,
                        ReadOnlyOffsetClassifier &classifier) {
  // Scalar only, so gatherAxis's own tensor-typed MulI can't shadow this.
  // The non-constant operand must not be scalarLike either: a tile-bound
  // computation like "program_id * row_blk" is scalarLike (pure program_id
  // and constants), while the real per-row counter always incorporates
  // something that genuinely varies per row, which is never scalarLike.
  std::function<bool(Operation *)> isMulIWithConstant =
      [&](Operation *op) -> bool {
    auto mulI = dyn_cast<arith::MulIOp>(op);
    if (!mulI || isa<RankedTensorType>(mulI.getType()))
      return false;
    bool rhsIsConst =
        static_cast<bool>(mulI.getRhs().getDefiningOp<arith::ConstantOp>());
    bool lhsIsConst =
        static_cast<bool>(mulI.getLhs().getDefiningOp<arith::ConstantOp>());
    if (!rhsIsConst && !lhsIsConst)
      return false;
    Value carrier = rhsIsConst ? mulI.getLhs() : mulI.getRhs();
    return !classifier.classify(carrier).isScalarLike();
  };
  // Also stop outside searchRoot's own block (e.g. a hoisted loop-tile bound).
  std::function<bool(Operation *)> leavesSourceRegion =
      [&](Operation *op) -> bool {
    return op == indicesOp || isSplatOfBlockArgPointer(op) ||
           op->getBlock() != searchRoot->getBlock();
  };
  Operation *mulIOp = findOperandDefinitionWithCondition(
      searchRoot->getResult(0), isMulIWithConstant, leavesSourceRegion);
  if (!mulIOp)
    return std::nullopt;
  auto mulI = cast<arith::MulIOp>(mulIOp);
  std::optional<int64_t> dim = extractMulIConstant(mulI);
  if (!dim)
    return std::nullopt;
  return ScalarAxisMatch{*dim};
}

// Matches a tt.load against the gather pattern and recovers its shape,
// using ReadOnlyOffsetClassifier rather than the mutating
// OffsetAnalysis::parse.
std::optional<GatherCandidate>
analyzeGatherCandidate(triton::LoadOp loadOp, unsigned ubCapacityBytes) {
  // Mapping runs before Gather and records successful rewrites on the module.
  // Gather's local UB estimate does not account for the combined backend
  // allocation. Keep indirect loads after either mapping rewrite. Checking
  // here also protects plan revalidation and application, not just discovery.
  auto module = loadOp->getParentOfType<ModuleOp>();
  if (module && (module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
                 module->hasAttr(kPersistentTaskStripMiningMarkerAttr)))
    return std::nullopt;

  // Volatile accesses cannot be replaced by a different set of reads.
  if (loadOp.getIsVolatile() || !loadOp.getBoundaryCheck().empty())
    return std::nullopt;
  auto addPtrOp = loadOp.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp || !isSplatOfBlockArgPointer(addPtrOp.getPtr().getDefiningOp()))
    return std::nullopt;

  auto loadTensorType = dyn_cast<RankedTensorType>(loadOp.getType());
  auto ptrTensorType = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
  if (!loadTensorType || !ptrTensorType || !loadTensorType.hasStaticShape() ||
      loadTensorType.getEncoding() || ptrTensorType.getEncoding() ||
      loadTensorType.getRank() < 2 || loadTensorType.getRank() > 5 ||
      loadTensorType.getShape() != ptrTensorType.getShape())
    return std::nullopt;
  if (!isSupportedGatherValueType(loadTensorType.getElementType()))
    return std::nullopt;

  // Classify only the offset side of the outer addptr: a plain tt.splat base
  // pointer never tags below what the offset side gives combineInfo anyway.
  ReadOnlyOffsetClassifier classifier;
  Value offsetValue = addPtrOp.getOffset();
  const PtrOffsetInfo &offsetInfo = classifier.classify(offsetValue);
  if (!offsetInfo.isUnstructured())
    return std::nullopt;
  if (offsetInfo.getRank() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] rank 1 is 'index "
                               "select', handled elsewhere\n");
    return std::nullopt;
  }

  Operation *analyzedOp = addPtrOp.getOperation();

  std::function<bool(Operation *)> isFullyUnstructured =
      [&](Operation *op) -> bool {
    if (auto addIOp = dyn_cast<arith::AddIOp>(op))
      return classifier.classify(addIOp.getLhs()).isUnstructured() &&
             classifier.classify(addIOp.getRhs()).isUnstructured();
    return classifier.classify(op->getResult(0)).isUnstructured();
  };
  std::function<bool(Operation *)> stopStructured = [&](Operation *op) -> bool {
    return classifier.classify(op->getResult(0)).isStructured();
  };
  Operation *indicesOp = findPrecedingOpWithCondition(
      analyzedOp, isFullyUnstructured, stopStructured);
  if (!indicesOp)
    return std::nullopt;

  GatherCandidate candidate;
  candidate.loadOp = loadOp;
  candidate.addPtrOp = addPtrOp;
  candidate.indices = indicesOp->getResult(0);

  if (!isIntegerTensorType(candidate.indices.getType(), candidate.indexRank))
    return std::nullopt;
  auto indicesType = cast<RankedTensorType>(candidate.indices.getType());
  // The grid and the calibrated index-alignment model currently use i32.
  if (!indicesType.getElementType().isInteger(32) ||
      indicesType.getEncoding() ||
      indicesType.getShape() != loadTensorType.getShape())
    return std::nullopt;
  if (candidate.indexRank > 5) {
    LLVM_DEBUG(llvm::dbgs()
               << "[GatherOptimization] rank " << candidate.indexRank
               << " exceeds the supported maximum of 5\n");
    return std::nullopt;
  }

  std::function<bool(Operation *)> stopIndices = [&](Operation *op) -> bool {
    return op == indicesOp;
  };
  std::function<bool(Operation *)> isStructuredNotScalar =
      [&](Operation *op) -> bool {
    const PtrOffsetInfo &info = classifier.classify(op->getResult(0));
    return info.isStructured() && !info.isScalarLike();
  };
  Operation *srcAnalysisStart = findPrecedingOpWithCondition(
      analyzedOp, isStructuredNotScalar, stopIndices);
  if (!srcAnalysisStart)
    return std::nullopt;

  std::function<bool(Operation *)> isNotUnstructuredNotSplat =
      [&](Operation *op) -> bool {
    return !classifier.classify(op->getResult(0)).isUnstructured() &&
           !isSplatOfBlockArgPointer(op);
  };
  Operation *baseNotUnstructuredOp = findPrecedingOpWithCondition(
      analyzedOp, isNotUnstructuredNotSplat, stopIndices);
  if (!baseNotUnstructuredOp)
    return std::nullopt;

  const PtrOffsetInfo &baseInfo =
      classifier.classify(baseNotUnstructuredOp->getResult(0));
  const SmallVector<AxisInfo> &baseStructured = baseInfo.getStructured();

  // The gather axis is the one tagged scalarlike in the pointer base's own
  // structure: indices vary there, but the base pointer's value doesn't.
  candidate.gatherAxis = 0;
  for (int i = 1; i < baseInfo.getRank(); i++) {
    if (baseStructured[i] == AxisInfo::scalar)
      continue; // 1-element axis: no multiplication, contributes nothing.
    if (baseStructured[i] == AxisInfo::scalarlike) {
      if (candidate.gatherAxis != 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[GatherOptimization] more than one axis could be the "
                      "gather axis\n");
        return std::nullopt;
      }
      candidate.gatherAxis = i;
    }
  }
  if (candidate.gatherAxis == 0) {
    // Size-1 axes tag scalar, not scalarlike, so several can be candidates
    // here with nothing to tell them apart; only accept an unambiguous one.
    SmallVector<int> scalarCandidates;
    for (int i = 1; i < baseInfo.getRank(); i++) {
      if (baseStructured[i] == AxisInfo::scalar)
        scalarCandidates.push_back(i);
    }
    if (scalarCandidates.size() == 1) {
      candidate.gatherAxis = scalarCandidates[0];
    } else if (scalarCandidates.size() > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[GatherOptimization] size-1 gather-axis candidates are "
                    "ambiguous\n");
    }
  }
  if (candidate.gatherAxis == 0)
    return std::nullopt;
  if (candidate.gatherAxis != candidate.indexRank - 1) {
    LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] detected gather axis "
                            << candidate.gatherAxis
                            << ", but only the last axis is in scope\n");
    return std::nullopt;
  }

  candidate.srcPtr =
      addPtrOp.getPtr().getDefiningOp<triton::SplatOp>().getSrc();

  ArrayRef<int64_t> indexShape =
      cast<RankedTensorType>(candidate.indices.getType()).getShape();

  // As in PR #1027, the inferred source shape assumes that each active
  // source row is fully readable. Index bounds and row masks below do not
  // establish the allocation size of the raw source pointer.
  // Axis 0 (the row dimension) comes from the indices' own shape.
  candidate.srcShape.push_back(indexShape[0]);
  for (int axis = 1; axis < candidate.indexRank; axis++) {
    std::optional<int64_t> dim =
        findAxisDimension(srcAnalysisStart, indicesOp, axis);
    if (!dim && axis == 1) {
      if (std::optional<ScalarAxisMatch> scalarMatch = findScalarAxisDimension(
              srcAnalysisStart, indicesOp, classifier)) {
        dim = scalarMatch->dimension;
      }
    }
    if (!dim)
      dim = findFoldedUnitAxisDimension(srcAnalysisStart, indicesOp, axis);
    if (!dim) {
      LLVM_DEBUG(llvm::dbgs() << "[GatherOptimization] could not recover "
                                 "source dimension for axis "
                              << axis << "\n");
      return std::nullopt;
    }
    candidate.srcShape.push_back(*dim);
  }

  if (candidate.srcShape.size() != indexShape.size())
    return std::nullopt;
  uint64_t sourceElements = 1, indexElements = 1;
  for (size_t d = 0; d < candidate.srcShape.size(); d++) {
    if (candidate.srcShape[d] <= 0 || indexShape[d] <= 0 ||
        !checkedMulU64(sourceElements, candidate.srcShape[d], sourceElements) ||
        !checkedMulU64(indexElements, indexShape[d], indexElements) ||
        sourceElements > std::numeric_limits<int32_t>::max() ||
        indexElements > std::numeric_limits<int32_t>::max())
      return std::nullopt;
    if (static_cast<int>(d) != candidate.gatherAxis &&
        indexShape[d] != candidate.srcShape[d])
      return std::nullopt;
  }

  // Prove the exact last-axis address decomposition. Do not rebuild the row
  // offsets: a range starting at a nonzero value or an extra constant bias
  // must keep pointing to the same row after the rewrite.
  auto offsetAdd = addPtrOp.getOffset().getDefiningOp<arith::AddIOp>();
  if (!offsetAdd)
    return std::nullopt;
  Value base;
  if (offsetAdd.getLhs() == candidate.indices)
    base = offsetAdd.getRhs();
  else if (offsetAdd.getRhs() == candidate.indices)
    base = offsetAdd.getLhs();
  else
    return std::nullopt;
  candidate.sourceBase =
      findAxisInvariantValue(base, candidate.srcShape, candidate.gatherAxis);
  if (!candidate.sourceBase)
    return std::nullopt;

  if (Value mask = loadOp.getMask()) {
    candidate.sourceMask =
        findAxisInvariantValue(mask, candidate.srcShape, candidate.gatherAxis);
    if (!candidate.sourceMask)
      return std::nullopt;
  }

  if (exceedsUbCapacity(
          candidate.srcShape, loadTensorType.getShape(),
          loadTensorType.getElementType(),
          cast<TensorType>(candidate.indices.getType()).getElementType(),
          ubCapacityBytes)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[GatherOptimization] estimated UB usage exceeds budget\n");
    return std::nullopt;
  }

  // The matched pattern's own row-dim size is the scalar baseline's
  // per-iteration row count (rowStep). Unknown loop bounds or row dependence
  // use the most conservative, single-iteration benefit bucket.
  std::optional<uint64_t> loadElemBytes =
      getByteWidth(loadTensorType.getElementType());
  if (!loadElemBytes)
    return std::nullopt;
  int64_t rowStep = candidate.srcShape[0];
  int64_t iterations =
      findRowIterations(candidate.addPtrOp, candidate.sourceBase, rowStep);
  if (!exceedsBenefitThreshold(candidate.srcShape.back(), indexShape.back(),
                               iterations, rowStep, *loadElemBytes)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[GatherOptimization] rewrite not expected to beat the "
                  "scalar baseline\n");
    return std::nullopt;
  }

  return candidate;
}

// Used by revalidate(): a stale plan must match the same pattern it
// originally found, not just any gather pattern on its anchor load.
bool candidatesMatch(const GatherCandidate &stored,
                     const GatherCandidate &fresh) {
  return stored.loadOp == fresh.loadOp && stored.addPtrOp == fresh.addPtrOp &&
         stored.indices == fresh.indices && stored.srcPtr == fresh.srcPtr &&
         stored.sourceBase == fresh.sourceBase &&
         stored.sourceMask == fresh.sourceMask &&
         stored.indexRank == fresh.indexRank &&
         stored.gatherAxis == fresh.gatherAxis &&
         stored.srcShape == fresh.srcShape;
}

template <typename TIOp>
Value reduce(Value inputTensor, Location loc, IRRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!tensorType)
    return nullptr;
  auto elementType = tensorType.getElementType();
  int64_t totalElements = 1;
  for (int64_t dim : tensorType.getShape())
    totalElements *= dim;

  auto flatTensorType = RankedTensorType::get({totalElements}, elementType);
  auto flatTensor =
      rewriter.create<triton::ReshapeOp>(loc, flatTensorType, inputTensor);

  auto reduceOp = rewriter.create<triton::ReduceOp>(
      loc, ValueRange{flatTensor.getResult()}, 0);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Type, 2> argTypes = {elementType, elementType};
    SmallVector<Location, 2> argLocs = {loc, loc};
    Block *block = rewriter.createBlock(
        &reduceOp.getRegion(), reduceOp.getRegion().end(), argTypes, argLocs);
    auto mathOp = rewriter.create<TIOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
    rewriter.create<triton::ReduceReturnOp>(loc, mathOp.getResult());
  }
  return reduceOp.getResults()[0];
}

//===----------------------------------------------------------------------===//
// Plan / Rule
//===----------------------------------------------------------------------===//

class GatherOptimizationPlan final : public RewritePlan {
public:
  GatherOptimizationPlan(GatherCandidate candidate, unsigned epoch,
                         unsigned ubCapacityBytes)
      : loadOp(candidate.loadOp),
        loadOperation(candidate.loadOp.getOperation()),
        indicesElements(computeIndicesElements(candidate)),
        candidate(std::move(candidate)), epoch(epoch),
        ubCapacityBytes(ubCapacityBytes) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::GatherOptimization;
  }

  // Larger gathers are worth rewriting first.
  unsigned getBenefit() const override {
    constexpr int64_t maxBenefit = std::numeric_limits<unsigned>::max();
    return static_cast<unsigned>(
        std::min<int64_t>(indicesElements, maxBenefit));
  }

  Operation *getAnchor() const override { return loadOperation; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !loadOp)
      return failure();
    if (loadOp->getParentOfType<triton::FuncOp>().getOperation() !=
        context.getFunction().getOperation())
      return failure();
    if (loadOp->getAttr(kGatherOptimisedLoadAttr))
      return failure();
    std::optional<GatherCandidate> fresh =
        analyzeGatherCandidate(loadOp, ubCapacityBytes);
    return success(fresh && candidatesMatch(candidate, *fresh));
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Re-derive before mutating, like revalidate(): a stale plan must never
    // be able to mutate the IR.
    std::optional<GatherCandidate> fresh =
        analyzeGatherCandidate(loadOp, ubCapacityBytes);
    if (!fresh || !candidatesMatch(candidate, *fresh))
      return failure();
    return buildGatherRewrite(*fresh, rewriter);
  }

private:
  static int64_t computeIndicesElements(const GatherCandidate &candidate) {
    auto indicesType = dyn_cast<RankedTensorType>(candidate.indices.getType());
    if (!indicesType)
      return 0;
    int64_t elements = 1;
    for (int64_t dim : indicesType.getShape())
      elements *= dim;
    return elements;
  }

  static LogicalResult buildGatherRewrite(const GatherCandidate &candidate,
                                          IRRewriter &rewriter) {
    triton::LoadOp loadOp = candidate.loadOp;
    auto loadTensorType = cast<TensorType>(loadOp.getType());
    auto indicesTensorType = cast<TensorType>(candidate.indices.getType());
    auto indexType = indicesTensorType.getElementType();
    auto loadElementType = loadTensorType.getElementType();
    ArrayRef<int64_t> srcShape = candidate.srcShape;
    int gatherAxis = candidate.gatherAxis;
    Value srcPtr = candidate.srcPtr;
    Value ourIndices = candidate.indices;

    // Both reduce<> calls below share this precondition; check it once here,
    // before either creates anything, rather than relying on them failing
    // identically (true today only because they share the same input).
    if (!isa<RankedTensorType>(ourIndices.getType()))
      return failure();

    Location loc = loadOp.getLoc();
    rewriter.setInsertionPoint(loadOp);
    Operation *previous = loadOp->getPrevNode();
    auto rollback = llvm::make_scope_exit([&] {
      while (Operation *created = loadOp->getPrevNode()) {
        if (created == previous)
          break;
        rewriter.eraseOp(created);
      }
    });

    if (loadOp.getMask()) {
      // Inactive lanes must neither force fallback nor feed invalid indices
      // to tt.gather. Their final value is restored from the original other.
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(cast<RankedTensorType>(indicesTensorType),
                                      rewriter.getIntegerAttr(indexType, 0)));
      ourIndices = rewriter.create<arith::SelectOp>(loc, loadOp.getMask(),
                                                    ourIndices, zero);
    }

    auto minIndex = reduce<arith::MinSIOp>(ourIndices, loc, rewriter);
    auto maxIndex = reduce<arith::MaxSIOp>(ourIndices, loc, rewriter);
    if (!minIndex || !maxIndex)
      return failure();

    // Match the v2 / PR #1027 indexing contract: indices in [-width, width)
    // use row-local wraparound in the Gather branch. This intentionally differs
    // from a raw negative pointer offset; an out-of-range tile still executes
    // the unchanged original load in the fallback branch.
    auto minAllowedIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, -srcShape[gatherAxis]));
    auto minCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, minIndex, minAllowedIndex);
    auto maxAllowedIndex = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, srcShape[gatherAxis]));
    auto maxCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, maxIndex, maxAllowedIndex);
    auto cond = rewriter.create<arith::AndIOp>(loc, maxCond.getResult(),
                                               minCond.getResult());

    auto ifOp = rewriter.create<scf::IfOp>(loc, /*yieldType=*/loadOp.getType(),
                                           cond, /*hasElse=*/true);
    {
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      // idx += width & (idx >> (bits - 1)): add width only to negative indices.
      // Use the sanitized indices so masked-off lanes stay at zero.
      auto shift = thenBuilder.create<arith::ConstantOp>(
          loc, thenBuilder.getIntegerAttr(
                   indexType, indexType.getIntOrFloatBitWidth() - 1));
      auto shifts =
          thenBuilder.create<triton::SplatOp>(loc, indicesTensorType, shift);
      auto sign = thenBuilder.create<arith::ShRSIOp>(loc, ourIndices, shifts);
      auto widths = thenBuilder.create<triton::SplatOp>(loc, indicesTensorType,
                                                        maxAllowedIndex);
      auto adjustment = thenBuilder.create<arith::AndIOp>(loc, widths, sign);
      auto normalizedIndices =
          thenBuilder.create<arith::AddIOp>(loc, ourIndices, adjustment);

      auto i32Type = thenBuilder.getI32Type();
      auto gridType = RankedTensorType::get(srcShape, i32Type);
      Value base;
      if (candidate.sourceBase.getType().isInteger(32))
        base = thenBuilder.create<triton::SplatOp>(loc, gridType,
                                                   candidate.sourceBase);
      else if (candidate.sourceBase.getType() == gridType)
        base = candidate.sourceBase;
      else
        base = thenBuilder.create<triton::BroadcastOp>(loc, gridType,
                                                       candidate.sourceBase);
      auto rangeType = RankedTensorType::get({srcShape[gatherAxis]}, i32Type);
      Value columns = thenBuilder.create<triton::MakeRangeOp>(
          loc, rangeType, 0, srcShape[gatherAxis]);
      for (int axis = 0; axis < gatherAxis; ++axis) {
        SmallVector<int64_t> shape(
            cast<RankedTensorType>(columns.getType()).getShape());
        shape.insert(shape.begin(), 1);
        columns = thenBuilder.create<triton::ExpandDimsOp>(
            loc, RankedTensorType::get(shape, i32Type), columns, 0);
      }
      if (columns.getType() != gridType)
        columns =
            thenBuilder.create<triton::BroadcastOp>(loc, gridType, columns);
      Value grid = thenBuilder.create<arith::AddIOp>(loc, base, columns);

      auto newSplat = thenBuilder.create<triton::SplatOp>(
          loc, RankedTensorType::get(srcShape, srcPtr.getType()), srcPtr);
      auto addPtr = thenBuilder.create<triton::AddPtrOp>(
          loc, newSplat.getType(), newSplat.getResult(), grid);
      Value sourceMask, sourceOther;
      if (candidate.sourceMask) {
        auto maskType =
            RankedTensorType::get(srcShape, thenBuilder.getI1Type());
        if (candidate.sourceMask.getType().isInteger(1))
          sourceMask = thenBuilder.create<triton::SplatOp>(
              loc, maskType, candidate.sourceMask);
        else if (candidate.sourceMask.getType() == maskType)
          sourceMask = candidate.sourceMask;
        else
          sourceMask = thenBuilder.create<triton::BroadcastOp>(
              loc, maskType, candidate.sourceMask);
        auto sourceType = RankedTensorType::get(srcShape, loadElementType);
        sourceOther = thenBuilder.create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(
                     sourceType, thenBuilder.getZeroAttr(loadElementType)));
      }
      auto load = thenBuilder.create<triton::LoadOp>(
          loc, addPtr, sourceMask, sourceOther, loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile());
      load->setAttr(kGatherOptimisedLoadAttr,
                    StringAttr::get(load->getContext(), "source"));
      Value input = load.getResult();

      auto gather = thenBuilder.create<triton::GatherOp>(
          loc, loadTensorType, input, normalizedIndices,
          candidate.indexRank - 1);
      Value result = gather.getResult();
      if (loadOp.getMask() && loadOp.getOther())
        result = thenBuilder.create<arith::SelectOp>(loc, loadOp.getMask(),
                                                     result, loadOp.getOther());
      thenBuilder.create<scf::YieldOp>(loc, result);
    }
    {
      OpBuilder elseBuilder = ifOp.getElseBodyBuilder(rewriter.getListener());
      IRMapping mapping;
      Operation *clonedLoad = elseBuilder.clone(*loadOp, mapping);
      clonedLoad->setAttr(
          kGatherOptimisedLoadAttr,
          StringAttr::get(clonedLoad->getContext(), "fallback"));
      elseBuilder.create<scf::YieldOp>(loc, clonedLoad->getResult(0));
    }

    Operation *first =
        previous ? previous->getNextNode() : &loadOp->getBlock()->front();
    for (Operation *created = first; created != loadOp.getOperation();
         created = created->getNextNode())
      if (failed(mlir::verify(created)))
        return failure();
    rollback.release();
    rewriter.replaceOp(loadOp, ifOp.getResult(0));
    return success();
  }

  triton::LoadOp loadOp;
  Operation *loadOperation;
  int64_t indicesElements;
  GatherCandidate candidate;
  unsigned epoch;
  unsigned ubCapacityBytes;
};

class GatherOptimizationRule final : public GraphOptimizationRule {
public:
  explicit GatherOptimizationRule(unsigned ubCapacityBytes)
      : ubCapacityBytes(ubCapacityBytes) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::GatherOptimization;
  }

  // Walks raw IR directly; no GraphOptimizationContext analysis needed.
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::LoadOp loadOp) {
      // A load that doesn't match yet needs no "rejected" marker: this
      // re-walks the whole function every iteration anyway.
      if (loadOp->getAttr(kGatherOptimisedLoadAttr))
        return;
      std::optional<GatherCandidate> candidate =
          analyzeGatherCandidate(loadOp, ubCapacityBytes);
      if (!candidate)
        return;
      plans.push_back(std::make_unique<GatherOptimizationPlan>(
          std::move(*candidate), context.getEpoch(), ubCapacityBytes));
    });
    return success();
  }

private:
  unsigned ubCapacityBytes;
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createGatherOptimizationRule(unsigned ubCapacityBytes) {
  return std::make_unique<GatherOptimizationRule>(ubCapacityBytes);
}
