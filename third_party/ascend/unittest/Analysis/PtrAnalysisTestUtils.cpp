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

#include "PtrAnalysisTestUtils.h"
#include "TritonMemoryAccess/OpFoldResultUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
namespace mlir::triton::ptr::test {
bool canMaterializeAddressTensor(RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getEncoding())
    return false;
  // Bound the product before multiplying: both total tensor size and overflow
  // matter, even when each individual axis could form a legal make_range.
  int64_t elements = 1;
  constexpr int64_t limit = OpTrait::impl::maxTensorNumElements;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || extent > limit / elements)
      return false;
    elements *= extent;
  }
  return llvm::isPowerOf2_64(elements);
}

FailureOr<Value> materializeAxisRange(RankedTensorType target, unsigned axis,
                                      OpBuilder &builder, Location loc) {
  if (!canMaterializeAddressTensor(target) ||
      axis >= static_cast<unsigned>(target.getRank()))
    return failure();
  int64_t extent = target.getShape()[axis];
  if (extent <= 0 || extent > std::numeric_limits<int32_t>::max())
    return failure();
  auto rangeType = RankedTensorType::get({extent}, builder.getI32Type());
  Value range = builder.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  auto elementRangeType =
      RankedTensorType::get({extent}, target.getElementType());
  auto casted = detail::foldOrCreateCast(range, elementRangeType,
                                         detail::CastKind::Signed, {64, 64},
                                         builder, loc);
  if (failed(casted))
    return failure();
  auto converted = detail::materialize(*casted, elementRangeType, builder, loc);
  if (failed(converted))
    return failure();
  Value expanded = *converted;
  for (unsigned dim = 0; dim < static_cast<unsigned>(target.getRank()); ++dim)
    if (dim != axis)
      expanded = builder.create<triton::ExpandDimsOp>(loc, expanded, dim);
  if (expanded.getType() != target)
    expanded = builder.create<triton::BroadcastOp>(loc, target, expanded);
  return expanded;
}

} // namespace mlir::triton::ptr::test
namespace mlir::triton::ptr::test {
namespace {
bool isIndex(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isIndex();
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType().isIndex();
  return false;
}

} // namespace

FailureOr<OffsetInfo> buildAddressView(const PtrInfo &ptr, OpBuilder &builder,
                                       Location loc) {
  const auto &offsets = ptr.offsets;
  if (offsets.fullOffsets)
    return offsets;
  // This test-only oracle constructs a legal Triton tensor. Production keeps
  // exact terms narrow and materializes only individual index lanes.
  Type oracleType = offsets.valueType;
  if (auto tensor = dyn_cast<RankedTensorType>(oracleType))
    oracleType = tensor.clone(builder.getI64Type());
  else
    oracleType = builder.getI64Type();
  if (!offsets.addressTerms.empty()) {
    OpFoldResult full;
    for (auto term : offsets.addressTerms) {
      Type type = dyn_cast<Value>(term)
                      ? cast<Value>(term).getType()
                      : cast<TypedAttr>(cast<Attribute>(term)).getType();
      Type target =
          isa<RankedTensorType>(type) ? oracleType : Type(builder.getI64Type());
      auto widened =
          detail::foldOrCreateCast(term, target, detail::CastKind::Signed,
                                   AnalysisOptions{64, 64}, builder, loc);
      if (failed(widened))
        return failure();
      auto value = detail::materialize(*widened, target, builder, loc);
      if (failed(value))
        return failure();
      if (target != oracleType) {
        auto splat = detail::splatTo(*value, cast<RankedTensorType>(oracleType),
                                     builder, loc);
        if (failed(splat))
          return failure();
        value = *splat;
      }
      if (!full)
        full = *value;
      else {
        auto sum =
            detail::foldOrCreateBinary(detail::BinaryKind::Add, full, *value,
                                       oracleType, {}, builder, loc);
        if (failed(sum))
          return failure();
        full = *sum;
      }
    }
    OffsetInfo info = offsets;
    info.valueType = oracleType;
    info.fullOffsets = full;
    info.addressTerms.clear();
    return info;
  }
  if (!offsets.hasStructuredForm() || !isIndex(*offsets.scalarOffset) ||
      llvm::any_of(detail::axisStrides(offsets),
                   [](OpFoldResult stride) { return !isIndex(stride); }))
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(oracleType);
  if (!tensorType || offsets.shape != llvm::to_vector(tensorType.getShape()) ||
      !canMaterializeAddressTensor(tensorType))
    return failure();

  auto toI64 = [&](OpFoldResult input) -> FailureOr<Value> {
    auto casted = detail::foldOrCreateCast(input, builder.getI64Type(),
                                           detail::CastKind::Signed, {64, 64},
                                           builder, loc);
    if (failed(casted))
      return failure();
    return detail::materialize(*casted, builder.getI64Type(), builder, loc);
  };
  auto uniform = toI64(*offsets.scalarOffset);
  if (failed(uniform))
    return failure();
  auto splat = detail::splatTo(*uniform, tensorType, builder, loc);
  if (failed(splat))
    return failure();
  Value complete = *splat;
  for (unsigned dim = 0; dim < offsets.shape.size(); ++dim) {
    if (offsets.axes[dim].kind == AxisKind::Invariant)
      continue;
    auto lane = materializeAxisRange(tensorType, dim, builder, loc);
    auto stride = toI64(offsets.axes[dim].stride);
    if (failed(lane) || failed(stride))
      return failure();
    auto strideTensor = detail::splatTo(*stride, tensorType, builder, loc);
    if (failed(strideTensor))
      return failure();
    Value term = builder.create<arith::MulIOp>(loc, *lane, *strideTensor);
    complete = builder.create<arith::AddIOp>(loc, complete, term);
  }
  OffsetInfo result = offsets;
  result.valueType = oracleType;
  result.fullOffsets = complete;
  return result;
}

} // namespace mlir::triton::ptr::test
namespace mlir::triton::ptr::test {
namespace {
bool isNumericHelper(Operation *op) {
  if (op->getName().getDialectNamespace() == "arith")
    return true;
  return isa<tensor::ExtractOp, tensor::ExtractSliceOp, tensor::CastOp,
             triton::ReshapeOp, triton::MakeRangeOp, triton::SplatOp,
             triton::BroadcastOp, triton::ExpandDimsOp>(op);
}

} // namespace
void eraseDeadPtrArithmetic(Operation *scope, RewriterBase &rewriter,
                            ArrayRef<Operation *> pointerHelpers) {
  if (!scope)
    return;
  SmallVector<Operation *> worklist;
  llvm::DenseSet<Operation *> erased;
  auto eligible = [&](Operation *op) {
    return isNumericHelper(op) || (isa<triton::BitcastOp>(op) &&
                                   llvm::is_contained(pointerHelpers, op));
  };
  scope->walk([&](Operation *op) {
    if (op != scope && eligible(op))
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (erased.contains(op))
      continue;
    if (!op->getBlock() || !eligible(op) || !isOpTriviallyDead(op))
      continue;
    SmallVector<Operation *> producers;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (scope->isAncestor(def) && eligible(def))
          producers.push_back(def);
    erased.insert(op);
    rewriter.eraseOp(op);
    worklist.append(producers);
  }
}

} // namespace mlir::triton::ptr::test
