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

#include "TritonToGraph/PermutationAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// This static assertion fixes the compose order in source as well as in the
// public documentation: apply [2, 0, 1], then [1, 2, 0] == identity.
constexpr int32_t kComposeBefore[] = {2, 0, 1};
constexpr int32_t kComposeAfter[] = {1, 2, 0};
static_assert(kComposeBefore[kComposeAfter[0]] == 0 &&
                  kComposeBefore[kComposeAfter[1]] == 1 &&
                  kComposeBefore[kComposeAfter[2]] == 2,
              "Permutation::compose applies the left operand before the "
              "right operand");

struct ParsedRange {
  ProofOutcome outcome;
  triton::MakeRangeOp range;
};

struct ParsedRankTwoAxis {
  ProofOutcome outcome;
  StaticAccessAxis axis;
};

bool haveSameShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getShape() == rhs.getShape();
}

bool isUnencodedRankedTensor(RankedTensorType type) {
  return type.hasStaticShape() && !type.getEncoding();
}

bool isI32Tensor(RankedTensorType type) {
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  return elementType && elementType.getWidth() == 32;
}

bool getStaticI32Constant(Value value, int64_t &result) {
  auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType || integerType.getWidth() != 32)
    return false;

  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer || integer.getType() != integerType)
    return false;

  result = integer.getValue().getSExtValue();
  return true;
}

bool getSignedI32RangeBounds(triton::MakeRangeOp range, int64_t &start,
                             int64_t &end) {
  if (!range)
    return false;

  IntegerAttr startAttr = range.getStartAttr();
  IntegerAttr endAttr = range.getEndAttr();
  auto startType = startAttr ? dyn_cast<IntegerType>(startAttr.getType())
                             : IntegerType();
  auto endType = endAttr ? dyn_cast<IntegerType>(endAttr.getType())
                         : IntegerType();
  if (!startAttr || !endAttr || !startType || !endType ||
      startType.getWidth() != 32 || endType.getWidth() != 32)
    return false;

  start = startAttr.getInt();
  end = endAttr.getInt();
  return true;
}

bool multiplyWouldOverflow(int64_t lhs, int64_t rhs) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  if (lhs == 0 || rhs == 0)
    return false;
  if (lhs == -1)
    return rhs == kMin;
  if (rhs == -1)
    return lhs == kMin;
  if (lhs > 0) {
    if (rhs > 0)
      return lhs > kMax / rhs;
    return rhs < kMin / lhs;
  }
  if (rhs > 0)
    return lhs < kMin / rhs;
  return lhs < kMax / rhs;
}

bool addWouldOverflow(int64_t lhs, int64_t rhs) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (rhs > 0)
    return lhs > kMax - rhs;
  return lhs < kMin - rhs;
}

ParsedRange parseNormalizedRange(Value offset) {
  if (auto range = offset.getDefiningOp<triton::MakeRangeOp>())
    return {ProofOutcome::proven(), range};

  return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
          triton::MakeRangeOp()};
}

ParsedRankTwoAxis parseRankTwoAxis(Value value,
                                   RankedTensorType fullOffsetType) {
  auto broadcast = value.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast || broadcast.getResult() != value)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto broadcastType = dyn_cast<RankedTensorType>(value.getType());
  if (!broadcastType || !isUnencodedRankedTensor(broadcastType) ||
      !isI32Tensor(broadcastType) || !haveSameShape(broadcastType, fullOffsetType))
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto multiply = broadcast.getSrc().getDefiningOp<arith::MulIOp>();
  if (!multiply || multiply.getResult() != broadcast.getSrc())
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};
  if (multiply.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {ProofOutcome::rejected(ProofReason::OverflowFlags), {}};

  auto expanded = multiply.getLhs().getDefiningOp<triton::ExpandDimsOp>();
  auto strideSplat = multiply.getRhs().getDefiningOp<triton::SplatOp>();
  if (!expanded || !strideSplat || expanded.getResult() != multiply.getLhs() ||
      strideSplat.getResult() != multiply.getRhs())
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto expandedType = dyn_cast<RankedTensorType>(expanded.getResult().getType());
  auto strideType = dyn_cast<RankedTensorType>(strideSplat.getResult().getType());
  if (!expandedType || !strideType || !isUnencodedRankedTensor(expandedType) ||
      !isUnencodedRankedTensor(strideType) || !isI32Tensor(expandedType) ||
      !isI32Tensor(strideType) || expandedType != strideType ||
      multiply.getResult().getType() != expandedType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  int64_t stride = 0;
  if (!getStaticI32Constant(strideSplat.getSrc(), stride))
    return {ProofOutcome::rejected(ProofReason::DynamicStride), {}};

  auto range = expanded.getSrc().getDefiningOp<triton::MakeRangeOp>();
  if (!range || range.getResult() != expanded.getSrc())
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto rangeType = dyn_cast<RankedTensorType>(range.getResult().getType());
  if (!rangeType || !isUnencodedRankedTensor(rangeType) || !isI32Tensor(rangeType) ||
      rangeType.getRank() != 1 || rangeType.getShape().front() <= 1)
    return {ProofOutcome::rejected(ProofReason::UnsupportedRank), {}};

  int64_t rangeStart = 0;
  int64_t rangeEnd = 0;
  if (!getSignedI32RangeBounds(range, rangeStart, rangeEnd))
    return {ProofOutcome::rejected(ProofReason::InvalidMakeRange), {}};
  const int64_t extent = rangeType.getShape().front();
  // V1 deliberately fixes the lane origin at zero.  This makes the rebuilt
  // range proof structural and avoids silently folding an affine base term.
  if (rangeStart != 0 || rangeEnd <= rangeStart || rangeEnd - rangeStart != extent)
    return {ProofOutcome::rejected(ProofReason::InvalidMakeRange), {}};

  const unsigned insertedAxis = expanded.getAxis();
  if (insertedAxis >= 2)
    return {ProofOutcome::rejected(ProofReason::InvalidAxisProvenance), {}};
  const unsigned outputAxis = 1 - insertedAxis;

  if (expandedType.getRank() != 2 || expandedType.getShape()[outputAxis] != extent ||
      expandedType.getShape()[insertedAxis] != 1)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  return {ProofOutcome::proven(),
          StaticAccessAxis{range, rangeStart, rangeEnd, stride, outputAxis}};
}

StaticAccessProof rejectAccess(ProofOutcome outcome) {
  return {outcome, std::nullopt};
}

bool isBarrierLike(Operation *operation) {
  return operation->getName().getStringRef().contains("barrier");
}

} // namespace

llvm::StringRef cfg::getProofReasonMessage(ProofReason reason) {
  switch (reason) {
  case ProofReason::None:
    return "proof established";
  case ProofReason::InvalidPermutationRank:
    return "permutation rank must be non-zero and representable";
  case ProofReason::InvalidPermutationAxis:
    return "permutation contains an out-of-range axis";
  case ProofReason::DuplicatePermutationAxis:
    return "permutation contains a duplicate axis";
  case ProofReason::PermutationRankMismatch:
    return "permutations have different ranks";
  case ProofReason::ShapeRankMismatch:
    return "shape rank does not match permutation rank";
  case ProofReason::InvalidOldAxis:
    return "old axis is out of range";
  case ProofReason::NullValue:
    return "analysis received a null value";
  case ProofReason::UnresolvedPointer:
    return "pointer has no normalized defining operation";
  case ProofReason::UnsupportedPointerType:
    return "pointer is not a ranked tensor of pointers";
  case ProofReason::UnsupportedPointerForm:
    return "pointer is not normalized as tt.splat plus tt.addptr";
  case ProofReason::UnsupportedRank:
    return "access rank is outside the supported static subset";
  case ProofReason::UnsupportedEncoding:
    return "encoded tensors are outside the pre-layout static subset";
  case ProofReason::NonSquareShape:
    return "rank-2 access must have a static square shape";
  case ProofReason::UnsupportedIndexElementType:
    return "access indices must be unencoded i32 tensors";
  case ProofReason::DynamicShape:
    return "access shape is dynamic";
  case ProofReason::UnsupportedAffineOffset:
    return "offset is outside the supported static affine form";
  case ProofReason::DynamicStride:
    return "stride is not a static integer";
  case ProofReason::NegativeStride:
    return "stride is negative";
  case ProofReason::ZeroStride:
    return "stride is zero";
  case ProofReason::DuplicateStride:
    return "strides are duplicated";
  case ProofReason::InvalidAxisProvenance:
    return "axis provenance is out of range";
  case ProofReason::DuplicateAxisProvenance:
    return "multiple access axes have the same provenance";
  case ProofReason::DuplicateRangeSource:
    return "multiple access axes reuse the same tt.make_range source";
  case ProofReason::InvalidMakeRange:
    return "tt.make_range does not match its static result shape";
  case ProofReason::OffsetOverflow:
    return "static affine offset arithmetic overflows";
  case ProofReason::OverflowFlags:
    return "affine arithmetic has overflow flags that V1 cannot preserve";
  case ProofReason::NonInjectiveLanes:
    return "lane address map is not proven injective";
  case ProofReason::MaskedAccess:
    return "masked or predicated access is unsupported";
  case ProofReason::BoundaryCheck:
    return "boundary or padding behavior is unsupported";
  case ProofReason::VolatileLoad:
    return "volatile load is unsupported";
  case ProofReason::NullOperation:
    return "analysis received a null operation";
  case ProofReason::DifferentBlocks:
    return "protected interval crosses MLIR blocks";
  case ProofReason::InvalidProtectedInterval:
    return "protected interval endpoints are not in program order";
  case ProofReason::RegionOperation:
    return "protected interval contains a region operation";
  case ProofReason::CallOperation:
    return "protected interval contains a call";
  case ProofReason::BarrierOperation:
    return "protected interval contains a barrier";
  case ProofReason::UnknownMemoryEffect:
    return "protected interval contains an operation with unknown memory effects";
  case ProofReason::InterveningMemoryEffect:
    return "protected interval contains a memory effect";
  case ProofReason::DifferentAccessBase:
    return "accesses do not share the same proven SSA base";
  case ProofReason::OverlappingAccessRange:
    return "static access ranges overlap";
  case ProofReason::UnsupportedInterveningMemoryAccess:
    return "intervening memory access is unsupported or cannot be proven static";
  }
  return "unknown proof reason";
}

ProofOutcome Permutation::validate(llvm::ArrayRef<int32_t> perm) {
  if (perm.empty() ||
      perm.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return ProofOutcome::rejected(ProofReason::InvalidPermutationRank);

  llvm::SmallVector<unsigned char, 4> seen(perm.size(), 0);
  for (int32_t oldAxis : perm) {
    if (oldAxis < 0 || static_cast<size_t>(oldAxis) >= perm.size())
      return ProofOutcome::rejected(ProofReason::InvalidPermutationAxis);
    if (seen[oldAxis])
      return ProofOutcome::rejected(ProofReason::DuplicatePermutationAxis);
    seen[oldAxis] = 1;
  }
  return ProofOutcome::proven();
}

FailureOr<Permutation> Permutation::create(llvm::ArrayRef<int32_t> perm) {
  if (!validate(perm).isProven())
    return failure();
  return Permutation(perm);
}

Permutation Permutation::inverse() const {
  llvm::SmallVector<int32_t, 4> oldToNew(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    oldToNew[newToOld[newAxis]] = static_cast<int32_t>(newAxis);
  return Permutation(oldToNew);
}

FailureOr<Permutation> Permutation::compose(
    const Permutation &after) const {
  if (rank() != after.rank())
    return failure();

  llvm::SmallVector<int32_t, 4> combined;
  combined.reserve(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    combined.push_back(newToOld[after.newToOld[newAxis]]);
  return Permutation(combined);
}

int32_t Permutation::mapOldAxisToNew(int32_t oldAxis) const {
  if (oldAxis < 0 || static_cast<unsigned>(oldAxis) >= rank())
    return -1;
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis) {
    if (newToOld[newAxis] == oldAxis)
      return static_cast<int32_t>(newAxis);
  }
  return -1;
}

FailureOr<llvm::SmallVector<int64_t>>
Permutation::permuteShape(llvm::ArrayRef<int64_t> shape) const {
  if (shape.size() != rank())
    return failure();

  llvm::SmallVector<int64_t> result;
  result.reserve(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    result.push_back(shape[newToOld[newAxis]]);
  return result;
}

ProofOutcome StaticAccessAnalysis::proveLaneInjectivity(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<unsigned> axisProvenance) {
  constexpr int64_t kSignedI32Max = std::numeric_limits<int32_t>::max();
  if (shape.empty() || shape.size() != strides.size() ||
      shape.size() != axisProvenance.size())
    return ProofOutcome::rejected(ProofReason::UnsupportedRank);

  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < 0)
      return ProofOutcome::rejected(ProofReason::DynamicShape);
    if (shape[axis] == 0)
      return ProofOutcome::rejected(ProofReason::UnsupportedRank);
    if (strides[axis] < 0)
      return ProofOutcome::rejected(ProofReason::NegativeStride);
    if (strides[axis] == 0)
      return ProofOutcome::rejected(ProofReason::ZeroStride);
    if (strides[axis] > kSignedI32Max)
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    if (axisProvenance[axis] >= shape.size())
      return ProofOutcome::rejected(ProofReason::InvalidAxisProvenance);

    for (unsigned previous = 0; previous < axis; ++previous) {
      if (axisProvenance[previous] == axisProvenance[axis])
        return ProofOutcome::rejected(
            ProofReason::DuplicateAxisProvenance);
      if (strides[previous] == strides[axis])
        return ProofOutcome::rejected(ProofReason::DuplicateStride);
    }
  }

  llvm::SmallVector<unsigned, 4> axes;
  axes.reserve(shape.size());
  for (unsigned axis = 0; axis < shape.size(); ++axis)
    axes.push_back(axis);
  std::sort(axes.begin(), axes.end(), [&](unsigned lhs, unsigned rhs) {
    return strides[lhs] < strides[rhs];
  });

  int64_t reachableSpan = 0;
  for (unsigned axis : axes) {
    if (shape[axis] == 1)
      continue;
    if (strides[axis] <= reachableSpan)
      return ProofOutcome::rejected(ProofReason::NonInjectiveLanes);

    const int64_t extent = shape[axis] - 1;
    if (multiplyWouldOverflow(extent, strides[axis]))
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    const int64_t contribution = extent * strides[axis];
    // The IR offsets are i32 tensors.  i64 host arithmetic is used only to
    // prove the expression safely; it must not authorize an address map that
    // would overflow the actual TTIR arithmetic.  Existing tensor-size limits
    // keep current fixtures below this bound, but the check is intentional for
    // future configurations as well.
    if (contribution > kSignedI32Max ||
        reachableSpan > kSignedI32Max - contribution)
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    if (addWouldOverflow(reachableSpan, contribution))
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    reachableSpan += contribution;
  }

  return ProofOutcome::proven();
}

StaticAccessProof StaticAccessAnalysis::analyzePointer(Value pointer) const {
  if (!pointer)
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullValue));

  auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
  if (!pointerType)
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerType));
  if (pointerType.getRank() != 1 && pointerType.getRank() != 2)
    return rejectAccess(ProofOutcome::rejected(ProofReason::UnsupportedRank));
  if (!pointerType.hasStaticShape())
    return rejectAccess(ProofOutcome::rejected(ProofReason::DynamicShape));
  if (pointerType.getEncoding())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedEncoding));

  auto pointerElement = dyn_cast<triton::PointerType>(pointerType.getElementType());
  if (!pointerElement || triton::isTensorPointerType(pointerType.getElementType()))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerType));

  auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr) {
    if (!pointer.getDefiningOp())
      return rejectAccess(
          ProofOutcome::unknown(ProofReason::UnresolvedPointer));
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  }

  auto baseSplat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!baseSplat)
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  Type baseType = baseSplat.getSrc().getType();
  if (!isa<triton::PointerType>(baseType) ||
      triton::isTensorPointerType(baseType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));

  auto baseSplatType =
      dyn_cast<RankedTensorType>(baseSplat.getResult().getType());
  auto offsetType = dyn_cast<RankedTensorType>(addPtr.getOffset().getType());
  if (!baseSplatType || !offsetType || !baseSplatType.hasStaticShape() ||
      !offsetType.hasStaticShape() || !haveSameShape(pointerType, baseSplatType) ||
      !haveSameShape(pointerType, offsetType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  if (baseSplatType.getEncoding() || offsetType.getEncoding())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedEncoding));

  if (pointerType.getRank() == 1) {
    ParsedRange parsedOffset = parseNormalizedRange(addPtr.getOffset());
    if (!parsedOffset.outcome.isProven())
      return rejectAccess(parsedOffset.outcome);

    auto rangeType =
        dyn_cast<RankedTensorType>(parsedOffset.range.getResult().getType());
    if (!rangeType || rangeType.getRank() != 1 || !rangeType.hasStaticShape() ||
        rangeType.getEncoding() || !haveSameShape(pointerType, rangeType))
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::UnsupportedRank));

    int64_t rangeStart = 0;
    int64_t rangeEnd = 0;
    if (!getSignedI32RangeBounds(parsedOffset.range, rangeStart, rangeEnd))
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));
    if (rangeEnd <= rangeStart ||
        rangeType.getShape().front() != rangeEnd - rangeStart)
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));

    StaticAccess access;
    access.pointer = pointer;
    access.offset = addPtr.getOffset();
    access.base = baseSplat.getSrc();
    access.shape.push_back(rangeType.getShape().front());
    access.strides.push_back(1);
    access.axisProvenance.push_back(0);
    access.axes.push_back(
        StaticAccessAxis{parsedOffset.range, rangeStart, rangeEnd, 1, 0});

    ProofOutcome injectivity = proveLaneInjectivity(
        access.shape, access.strides, access.axisProvenance);
    if (!injectivity.isProven())
      return rejectAccess(injectivity);

    const int64_t lastRangeValue = rangeEnd - 1;
    access.firstOffset = rangeStart;
    access.lastOffset = lastRangeValue;
    access.lanesInjective = true;
    return {ProofOutcome::proven(), std::move(access)};
  }

  if (!isI32Tensor(offsetType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedIndexElementType));
  if (pointerType.getShape()[0] <= 1 ||
      pointerType.getShape()[0] != pointerType.getShape()[1])
    return rejectAccess(ProofOutcome::rejected(ProofReason::NonSquareShape));

  auto offsetAdd = addPtr.getOffset().getDefiningOp<arith::AddIOp>();
  if (!offsetAdd || offsetAdd.getResult() != addPtr.getOffset())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset));
  if (offsetAdd.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return rejectAccess(ProofOutcome::rejected(ProofReason::OverflowFlags));

  ParsedRankTwoAxis firstAxis =
      parseRankTwoAxis(offsetAdd.getLhs(), offsetType);
  ParsedRankTwoAxis secondAxis =
      parseRankTwoAxis(offsetAdd.getRhs(), offsetType);
  if (!firstAxis.outcome.isProven())
    return rejectAccess(firstAxis.outcome);
  if (!secondAxis.outcome.isProven())
    return rejectAccess(secondAxis.outcome);

  SmallVector<StaticAccessAxis, 4> axes(2);
  bool seenOutputAxis[2] = {false, false};
  for (const StaticAccessAxis &axis : {firstAxis.axis, secondAxis.axis}) {
    if (axis.outputAxis >= 2 || seenOutputAxis[axis.outputAxis])
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::DuplicateAxisProvenance));
    seenOutputAxis[axis.outputAxis] = true;
    axes[axis.outputAxis] = axis;
  }
  if (!seenOutputAxis[0] || !seenOutputAxis[1])
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::InvalidAxisProvenance));
  if (!axes[0].range || !axes[1].range ||
      axes[0].range.getOperation() == axes[1].range.getOperation())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::DuplicateRangeSource));

  StaticAccess access;
  access.pointer = pointer;
  access.offset = addPtr.getOffset();
  access.base = baseSplat.getSrc();
  access.shape.append(pointerType.getShape().begin(), pointerType.getShape().end());
  access.axes = std::move(axes);
  for (unsigned axis = 0; axis < 2; ++axis) {
    const StaticAccessAxis &parsedAxis = access.axes[axis];
    if (parsedAxis.rangeEnd != access.shape[axis])
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));
    access.strides.push_back(parsedAxis.stride);
    access.axisProvenance.push_back(axis);
  }

  ProofOutcome injectivity =
      proveLaneInjectivity(access.shape, access.strides, access.axisProvenance);
  if (!injectivity.isProven())
    return rejectAccess(injectivity);

  int64_t lastOffset = 0;
  for (unsigned axis = 0; axis < 2; ++axis) {
    const int64_t extent = access.shape[axis] - 1;
    if (multiplyWouldOverflow(extent, access.strides[axis]))
      return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));
    const int64_t contribution = extent * access.strides[axis];
    if (addWouldOverflow(lastOffset, contribution))
      return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));
    lastOffset += contribution;
  }

  access.firstOffset = 0;
  access.lastOffset = lastOffset;
  access.lanesInjective = true;
  return {ProofOutcome::proven(), std::move(access)};
}

StaticAccessProof StaticAccessAnalysis::analyzeLoad(triton::LoadOp load) const {
  if (!load.getOperation())
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullOperation));
  if (load.getMask() || load.getOther())
    return rejectAccess(ProofOutcome::rejected(ProofReason::MaskedAccess));
  if (!load.getBoundaryCheck().empty() || load.getPadding())
    return rejectAccess(ProofOutcome::rejected(ProofReason::BoundaryCheck));
  if (load.getIsVolatile())
    return rejectAccess(ProofOutcome::rejected(ProofReason::VolatileLoad));
  return analyzePointer(load.getPtr());
}

StaticAccessProof StaticAccessAnalysis::analyzeStore(
    triton::StoreOp store) const {
  if (!store.getOperation())
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullOperation));
  if (store.getMask())
    return rejectAccess(ProofOutcome::rejected(ProofReason::MaskedAccess));
  if (!store.getBoundaryCheck().empty())
    return rejectAccess(ProofOutcome::rejected(ProofReason::BoundaryCheck));
  return analyzePointer(store.getPtr());
}

ProofOutcome StaticAccessAnalysis::proveSameBaseDisjoint(
    const StaticAccess &lhs, const StaticAccess &rhs) const {
  if (!lhs.base || !rhs.base || lhs.base != rhs.base)
    return ProofOutcome::rejected(ProofReason::DifferentAccessBase);
  if (!lhs.lanesInjective || !rhs.lanesInjective ||
      lhs.firstOffset > lhs.lastOffset || rhs.firstOffset > rhs.lastOffset)
    return ProofOutcome::rejected(
        ProofReason::UnsupportedInterveningMemoryAccess);
  if (lhs.lastOffset < rhs.firstOffset || rhs.lastOffset < lhs.firstOffset)
    return ProofOutcome::proven();
  return ProofOutcome::rejected(ProofReason::OverlappingAccessRange);
}

ProofOutcome ProtectedIntervalAnalysis::proveNoMemoryEffects(
    Operation *first, Operation *last) const {
  if (!first || !last)
    return ProofOutcome::rejected(ProofReason::NullOperation);
  if (first == last)
    return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
  if (first->getBlock() != last->getBlock())
    return ProofOutcome::rejected(ProofReason::DifferentBlocks);

  for (Operation *operation = first->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (operation == last)
      return ProofOutcome::proven();
    if (operation->getNumRegions() != 0)
      return ProofOutcome::rejected(ProofReason::RegionOperation);
    if (isa<CallOpInterface>(operation))
      return ProofOutcome::rejected(ProofReason::CallOperation);
    if (isBarrierLike(operation))
      return ProofOutcome::rejected(ProofReason::BarrierOperation);

    if (auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      llvm::SmallVector<
          SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
          effects;
      memoryEffects.getEffects(effects);
      if (!effects.empty())
        return ProofOutcome::rejected(ProofReason::InterveningMemoryEffect);
    }
    if (!isMemoryEffectFree(operation))
      return ProofOutcome::rejected(ProofReason::UnknownMemoryEffect);
  }

  return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
}

ProofOutcome ProtectedIntervalAnalysis::proveNoConflictingLoadStoreEffects(
    Operation *first, Operation *last,
    llvm::ArrayRef<StaticAccess> protectedAccesses) const {
  if (!first || !last)
    return ProofOutcome::rejected(ProofReason::NullOperation);
  if (protectedAccesses.empty() || first == last)
    return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
  if (first->getBlock() != last->getBlock())
    return ProofOutcome::rejected(ProofReason::DifferentBlocks);

  StaticAccessAnalysis accessAnalysis;
  auto proveDisjoint = [&](const StaticAccessProof &proof) {
    if (!proof.isProven())
      return ProofOutcome::rejected(
          ProofReason::UnsupportedInterveningMemoryAccess);
    for (const StaticAccess &protectedAccess : protectedAccesses) {
      ProofOutcome outcome = accessAnalysis.proveSameBaseDisjoint(
          *proof.access, protectedAccess);
      if (!outcome.isProven())
        return outcome;
    }
    return ProofOutcome::proven();
  };

  for (Operation *operation = first->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (operation == last)
      return ProofOutcome::proven();
    if (operation->getNumRegions() != 0)
      return ProofOutcome::rejected(ProofReason::RegionOperation);
    if (isa<CallOpInterface>(operation))
      return ProofOutcome::rejected(ProofReason::CallOperation);
    if (isBarrierLike(operation))
      return ProofOutcome::rejected(ProofReason::BarrierOperation);

    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      ProofOutcome outcome = proveDisjoint(accessAnalysis.analyzeLoad(load));
      if (!outcome.isProven())
        return outcome;
      continue;
    }
    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      ProofOutcome outcome =
          proveDisjoint(accessAnalysis.analyzeStore(store));
      if (!outcome.isProven())
        return outcome;
      continue;
    }

    if (auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      llvm::SmallVector<
          SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
          effects;
      memoryEffects.getEffects(effects);
      if (!effects.empty())
        return ProofOutcome::rejected(ProofReason::InterveningMemoryEffect);
    }
    if (!isMemoryEffectFree(operation))
      return ProofOutcome::rejected(ProofReason::UnknownMemoryEffect);
  }

  return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
}
