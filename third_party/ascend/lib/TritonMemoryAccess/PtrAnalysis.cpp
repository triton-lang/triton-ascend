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

#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ScopeExit.h"
#include <limits>

namespace mlir::triton::ptr {

//===----------------------------------------------------------------------===//
// Analysis session and caches
//===----------------------------------------------------------------------===//

Operation *PtrAnalysis::getCacheOwner(Value value) const {
  for (Operation *op = value.getParentBlock()
                           ? value.getParentBlock()->getParentOp()
                           : nullptr;
       op; op = op->getParentOp())
    if (loopCaches.contains(op))
      return op;
  return nullptr;
}

PtrAnalysis::CacheMap &PtrAnalysis::getCacheMap(Operation *owner) {
  return owner ? loopCaches.find(owner)->second.back() : rootCache;
}

const PtrInfo *PtrAnalysis::findCached(Value value) const {
  if (!value)
    return nullptr;
  for (Operation *op = getCacheOwner(value); op; op = op->getParentOp()) {
    auto loop = loopCaches.find(op);
    if (loop == loopCaches.end())
      continue;
    const auto &cache = loop->second.back();
    if (auto found = cache.find(value); found != cache.end())
      return &found->second;
  }
  auto found = rootCache.find(value);
  return found == rootCache.end() ? nullptr : &found->second;
}

FailureOr<PtrInfo> PtrAnalysis::analyze(Value value) {
  if (!value)
    return failure();
  if (const PtrInfo *info = findCached(value))
    return *info;
  // Recursive loop queries can grow or temporarily replace maps. Save only
  // the owner key, then reacquire its current map after parsing.
  Operation *owner = getCacheOwner(value);
  OpBuilder::InsertionGuard guard(builder);
  auto info = isControlFlowValue(value) ? parseControlFlow(value)
                                        : parseOperation(value);
  if (failed(info))
    return failure();
  return getCacheMap(owner).try_emplace(value, std::move(*info)).first->second;
}

LogicalResult PtrAnalysis::setInfo(Value value, const PtrInfo &info) {
  if (!loopCaches.empty() || failed(detail::validateInfo(value, info)))
    return failure();
  // info may refer to an entry moved by insertion into rootCache.
  PtrInfo copy = info;
  rootCache.insert_or_assign(value, std::move(copy));
  return success();
}

LogicalResult PtrAnalysis::reset(ArrayRef<std::pair<Value, PtrInfo>> known) {
  if (!loopCaches.empty())
    return failure();
  CacheMap replacement;
  for (const auto &[value, info] : known) {
    if (failed(detail::validateInfo(value, info)))
      return failure();
    replacement.insert_or_assign(value, info);
  }
  rootCache = std::move(replacement);
  return success();
}

void PtrAnalysis::popLoopCache(Operation *loop) {
  auto found = loopCaches.find(loop);
  assert(found != loopCaches.end() && !found->second.empty());
  found->second.pop_back();
  if (found->second.empty())
    loopCaches.erase(found);
}

LogicalResult checkInsertionPoint(OpBuilder &builder,
                                  ArrayRef<Value> dependencies) {
  auto insertion = builder.saveInsertionPoint();
  Block *block = insertion.getBlock();
  if (!block)
    return failure();
  auto point = insertion.getPoint();
  DominanceInfo dominance;
  for (Value dependency : dependencies) {
    if (!dependency || !dependency.getParentBlock())
      return failure();
    if (Operation *def = dependency.getDefiningOp()) {
      if (!dominance.properlyDominates(def->getBlock(), def->getIterator(),
                                       block, point, /*enclosingOk=*/false))
        return failure();
    } else if (dependency.getParentBlock() != block &&
               !dominance.dominates(dependency.getParentBlock(), block)) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Ordinary operations
//===----------------------------------------------------------------------===//

using namespace detail;

// Local expression queries used by the recursive rules.
namespace {
// fptosi and integer reduction results need only axis invariance from float
// intermediates. Keep these queries local to preserve OffsetInfo's integer
// origin/stride contract; broader reuse can justify integration into parse.
// Deterministic arithmetic preserves input axis invariance, including in
// reduction combiners.
bool isInvariancePreservingArithmetic(Operation &op) {
  return isa<arith::ConstantOp>(op) ||
         (op.getName().getDialectNamespace() == "arith" &&
          op.hasTrait<OpTrait::Elementwise>() && op.getNumRegions() == 0);
}

// Each combiner operation must preserve the inputs' axis invariance.
bool hasInvariancePreservingCombiner(triton::ReduceOp op) {
  return llvm::all_of(op.getCombineOp().front().without_terminator(),
                      isInvariancePreservingArithmetic);
}

SmallVector<bool>
inferInvariantAxes(Value value, PtrAnalysis &analysis,
                   llvm::DenseMap<Value, SmallVector<bool>> &invarianceCache);

// A uniform displacement only proves equal addresses with a scalar base.
// Masked-out lanes must have a defined, equally invariant replacement value.
SmallVector<bool> inferLoadInvariantAxes(
    triton::LoadOp op, PtrAnalysis &analysis,
    llvm::DenseMap<Value, SmallVector<bool>> &invarianceCache) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  SmallVector<bool> axes(type ? type.getRank() : 0, false);
  if (!type || op.getIsVolatile() || !isOrdinaryPtrType(op.getPtr().getType()))
    return axes;

  Value mask = op.getMask();
  if (mask) {
    auto constant = getIntegerConstant(mask, 1);
    if (succeeded(constant) && constant->isAllOnes())
      mask = {};
    else if (!op.getOther())
      return axes;
  }
  auto pointer = analysis.analyze(op.getPtr());
  if (failed(pointer) || !pointer->hasScalarBase())
    return axes;
  for (unsigned i = 0; i < axes.size(); ++i)
    axes[i] = pointer->offsets.axes[i].kind == AxisKind::Invariant;
  if (mask) {
    auto maskAxes = inferInvariantAxes(mask, analysis, invarianceCache);
    auto otherAxes =
        inferInvariantAxes(op.getOther(), analysis, invarianceCache);
    for (unsigned i = 0; i < axes.size(); ++i)
      axes[i] = axes[i] && maskAxes[i] && otherAxes[i];
  }
  return axes;
}

// Float intermediates need only per-axis equality facts. Integer inputs still
// use the same analysis session. This local memo avoids rewalking shared float
// expressions; it owns no SSA helpers and has no session lifecycle.
// An unanalyzable input contributes no invariant axes.
SmallVector<bool>
inferInvariantAxes(Value value, PtrAnalysis &analysis,
                   llvm::DenseMap<Value, SmallVector<bool>> &invarianceCache) {
  if (auto it = invarianceCache.find(value); it != invarianceCache.end())
    return it->second;
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  SmallVector<bool> invariantAxes(tensorType ? tensorType.getRank() : 0, false);
  if (!tensorType)
    return invariantAxes;
  auto remember = [&]() {
    invarianceCache.try_emplace(value, invariantAxes);
    return invariantAxes;
  };
  if (!isa<FloatType>(tensorType.getElementType())) {
    auto info = analysis.analyze(value);
    if (succeeded(info))
      for (unsigned i = 0; i < invariantAxes.size(); ++i)
        invariantAxes[i] = info->offsets.axes[i].kind == AxisKind::Invariant;
    return remember();
  }
  if (auto load = value.getDefiningOp<triton::LoadOp>()) {
    invariantAxes = inferLoadInvariantAxes(load, analysis, invarianceCache);
  } else if (auto branch = value.getDefiningOp<scf::IfOp>()) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    auto thenAxes = inferInvariantAxes(branch.thenYield().getOperand(index),
                                       analysis, invarianceCache);
    auto elseAxes = inferInvariantAxes(branch.elseYield().getOperand(index),
                                       analysis, invarianceCache);
    for (unsigned i = 0; i < invariantAxes.size(); ++i)
      invariantAxes[i] = thenAxes[i] && elseAxes[i];
  } else if (value.getDefiningOp<triton::SplatOp>()) {
    std::fill(invariantAxes.begin(), invariantAxes.end(), true);
  } else if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      if (dense.isSplat())
        std::fill(invariantAxes.begin(), invariantAxes.end(), true);
  } else if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    auto sourceInvariantAxes =
        inferInvariantAxes(broadcast.getSrc(), analysis, invarianceCache);
    auto sourceShape =
        cast<RankedTensorType>(broadcast.getSrc().getType()).getShape();
    for (unsigned i = 0; i < invariantAxes.size(); ++i)
      invariantAxes[i] = sourceShape[i] == 1 || sourceInvariantAxes[i];
  } else if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    invariantAxes =
        inferInvariantAxes(expand.getSrc(), analysis, invarianceCache);
    invariantAxes.insert(invariantAxes.begin() + expand.getAxis(), true);
  } else if (Operation *op = value.getDefiningOp();
             op && (isInvariancePreservingArithmetic(*op) ||
                    isa<triton::ClampFOp>(op))) {
    std::fill(invariantAxes.begin(), invariantAxes.end(), true);
    for (Value operand : op->getOperands()) {
      auto sourceInvariantAxes =
          inferInvariantAxes(operand, analysis, invarianceCache);
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (sourceInvariantAxes.size() != invariantAxes.size())
        return SmallVector<bool>(invariantAxes.size(), false);
      for (unsigned i = 0; i < invariantAxes.size(); ++i)
        invariantAxes[i] = invariantAxes[i] && sourceInvariantAxes[i];
    }
  }
  return remember();
}

} // namespace

FailureOr<PtrInfo> PtrAnalysis::parseOperation(Value value) {
  // Validate the supported integer and pointer types before dispatching.
  if (!value)
    return failure();
  bool integer = succeeded(detail::getIntegerWidth(value.getType()));
  bool ordinary = detail::isOrdinaryPtrType(value.getType());
  bool block = detail::isBlockPtrType(value.getType());
  if (!integer && !ordinary && !block) {
    emitError(value.getLoc())
        << "unsupported type passed to pointer analysis: " << value.getType();
    return failure();
  }
  // Analyze splat sources even for singleton results, preserving bound origins.
  if (auto op = value.getDefiningOp<triton::SplatOp>())
    return parseSplat(op);
  // A scalar or singleton tensor is already an exact dynamic origin/stride.
  // Reuse its final SSA instead of rebuilding its arithmetic in index;
  // larger tensors need axis analysis below.
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (integer &&
      (isa<IntegerType, IndexType>(value.getType()) ||
       (tensor && tensor.hasStaticShape() && tensor.getNumElements() == 1))) {
    builder.setInsertionPointAfterValue(value);
    return buildIntegerInfo(value, builder);
  }
  // Block pointers derive coordinates and strides from their descriptors.
  if (block) {
    if (auto op = value.getDefiningOp<triton::MakeTensorPtrOp>())
      return parseMakeTensorPtr(op);
    if (auto op = value.getDefiningOp<triton::AdvanceOp>())
      return parseAdvance(op);
    return failure();
  }
  // Shape and selection rules apply to both offsets and ordinary pointers.
  if (auto op = value.getDefiningOp<triton::BroadcastOp>())
    return parseBroadcast(op);
  if (auto op = value.getDefiningOp<triton::ExpandDimsOp>())
    return parseExpandDims(op);
  if (auto op = value.getDefiningOp<triton::ReshapeOp>())
    return parseReshape(op);
  if (auto op = value.getDefiningOp<tensor::ExtractSliceOp>())
    return parseExtractSlice(op);
  if (auto op = value.getDefiningOp<tensor::InsertOp>())
    return parseInsert(op);
  if (auto op = value.getDefiningOp<tensor::InsertSliceOp>())
    return parseInsertSlice(op);
  if (auto op = value.getDefiningOp<tensor::CastOp>())
    return parseTensorCast(op);
  if (auto op = value.getDefiningOp<arith::SelectOp>())
    return parseSelect(op);
  // Ordinary pointers combine scalar or per-lane bases with element offsets.
  if (ordinary) {
    if (auto op = value.getDefiningOp<triton::AddPtrOp>())
      return parseAddPtr(op);
    if (auto op = value.getDefiningOp<triton::BitcastOp>())
      return parseBitcast(op);
    return buildPointerInfo(value, builder);
  }
  // Integer tensor rules infer origins and relationships between lanes.
  if (auto op = value.getDefiningOp<triton::MakeRangeOp>()) {
    return parseMakeRange(op);
  }
  if (auto *op = value.getDefiningOp()) {
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
            arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::AndIOp,
            arith::OrIOp, arith::XOrIOp, arith::CmpIOp>(op))
      return parseBinaryArithmetic(value);
    // Integer casts share width checks; distinguish unsigned extension
    // semantics.
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>(op)) {
      bool unsignedCast = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op);
      return parseCast(value, op->getOperand(0), unsignedCast);
    }
  }
  if (auto op = value.getDefiningOp<arith::FPToSIOp>())
    return parseFloatToInt(op);
  if (auto op = value.getDefiningOp<triton::LoadOp>())
    return parseLoad(op);
  if (auto op = value.getDefiningOp<triton::ReduceOp>())
    return parseReduce(value, op);
  // Other integer producers may still provide facts through the value itself.
  builder.setInsertionPointAfterValue(value);
  return buildIntegerInfo(value, builder);
}

// Shape and selection rules.
FailureOr<PtrInfo> PtrAnalysis::parseSplat(triton::SplatOp op) {
  // Analyze the scalar source so its known facts and origin are retained.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Integer splats use the source origin for every result lane.
  if (!sourceInfo->base) {
    if (!sourceInfo->offsets.getShape().empty())
      return failure();
    OpFoldResult scalar = sourceInfo->offsets.scalarOffset;
    if (!scalar)
      scalar = op.getSrc();
    return PtrInfo{{}, buildUniformOffsetInfo(op.getResult(), builder, scalar)};
  }

  // Pointer splats broadcast the element offset and retain the scalar base.
  builder.setInsertionPoint(op);
  PtrInfo info = *sourceInfo;
  assert(info.offsets.valueType.isIndex() &&
         "scalar pointer offsets must have index type");
  auto targetType =
      cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto full = materializeFullOffset(info.offsets,
                                    addressStorageType(targetType, builder),
                                    builder, op.getLoc());
  if (failed(full))
    return failure();

  // Every new tensor axis is invariant because all pointers came from one SSA.
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = *full;
  info.offsets.axes.assign(targetType.getRank(),
                           makeInvariantAxisInfo(builder));
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseBroadcast(triton::BroadcastOp op) {
  // Obtain the source facts before expanding its singleton dimensions.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Transform per-lane bases and offsets with the same lane mapping.
    builder.setInsertionPoint(op);
    if (!info.hasScalarBase()) {
      if (info.base == op.getSrc())
        info.base = op.getResult();
      else
        info.base = builder.create<triton::BroadcastOp>(
            op.getLoc(), op.getType(), info.base);
    }
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::BroadcastOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset);
  }

  // Broadcast the copied axes, then record the result type and complete value.
  broadcastAxes(info.offsets.axes, sourceInfo->offsets.getShape(),
                targetType.getShape(), makeInvariantAxisInfo(builder));
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = full;
  setUniformOrigin(info.offsets, full, builder);
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseExpandDims(triton::ExpandDimsOp op) {
  // Obtain the source facts before inserting the singleton dimension.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Insert the same dimension into per-lane bases and numeric offsets.
    builder.setInsertionPoint(op);
    if (!info.hasScalarBase()) {
      if (info.base == op.getSrc())
        info.base = op.getResult();
      else
        info.base = builder.create<triton::ExpandDimsOp>(
            op.getLoc(), op.getType(), info.base, op.getAxis());
    }
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::ExpandDimsOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset,
        op.getAxis());
  }

  // Insert the corresponding invariant axis into either descriptor.
  unsigned axis = op.getAxis();
  info.offsets.axes.insert(info.offsets.axes.begin() + axis,
                           makeInvariantAxisInfo(builder));
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = full;
  setUniformOrigin(info.offsets, full, builder);
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseReshape(triton::ReshapeOp op) {
  // allow_reorder has no fixed lane correspondence. Normalization is a caller
  // decision; analysis must not change the source op or choose two
  // permutations.
  if (op.getAllowReorder())
    return failure();

  // Obtain the source facts under the fixed lane ordering.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Reshape both base lanes and offsets with the original lane ordering.
    builder.setInsertionPoint(op);
    if (!info.hasScalarBase()) {
      if (info.base == op.getSrc())
        info.base = op.getResult();
      else
        info.base = builder.create<triton::ReshapeOp>(op.getLoc(), op.getType(),
                                                      info.base, false,
                                                      op.getEfficientLayout());
    }
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::ReshapeOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset,
        false, op.getEfficientLayout());
  }

  // Preserve uniformity and axes whose coordinates only gain or lose unit dims.
  info.offsets = reshapeOffsetInfo(full, targetType, info.offsets, builder);
  setUniformOrigin(info.offsets, full, builder);
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseExtractSlice(tensor::ExtractSliceOp op) {
  // Analyze the source; dynamic result extents still admit complete offsets.
  auto targetType = op.getType();
  auto sourceInfo = analyze(op.getSource());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Record which source dimensions disappear in a rank-reducing slice.
  auto mask = op.getDroppedDims();
  SmallVector<bool> dropped;
  for (unsigned i = 0; i < op.getSourceType().getRank(); ++i)
    dropped.push_back(mask.test(i));

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  if (info.base) {
    // Apply the same slice to base lanes and numeric offsets.
    builder.setInsertionPoint(op);
    if (!info.hasScalarBase()) {
      if (info.base == op.getSource())
        info.base = op.getResult();
      else
        info.base = builder.create<tensor::ExtractSliceOp>(
            op.getLoc(), op.getType(), info.base, op.getMixedOffsets(),
            op.getMixedSizes(), op.getMixedStrides());
    }
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<tensor::ExtractSliceOp>(
        op.getLoc(),
        cast<RankedTensorType>(addressStorageType(targetType, builder)),
        *inputOffset, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
  }

  // Update the origin and axes using the same starts, steps and dropped axes.
  auto offsets =
      sliceOffsetInfo(full, targetType, info.offsets, op.getMixedOffsets(),
                      op.getMixedStrides(), dropped, builder, op.getLoc());
  if (failed(offsets))
    return failure();
  info.offsets = std::move(*offsets);
  setUniformOrigin(info.offsets, full, builder);
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseInsert(tensor::InsertOp op) {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer inputs; insertion reuses the destination's shape.
  auto source = analyze(op.getScalar()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Insert the source offset into the destination offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getScalar().getType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertOp>(op.getLoc(), src, dst,
                                                op.getIndices());
      },
      builder);
}

FailureOr<PtrInfo> PtrAnalysis::parseInsertSlice(tensor::InsertSliceOp op) {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer tensors; the original slice carries dynamic extents.
  auto source = analyze(op.getSource()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Replace the same slice in the complete offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getSourceType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertSliceOp>(
            op.getLoc(), src, dst, op.getMixedOffsets(), op.getMixedSizes(),
            op.getMixedStrides());
      },
      builder);
}

FailureOr<PtrInfo> PtrAnalysis::parseTensorCast(tensor::CastOp op) {
  // Axis facts require a known rank; verified casts preserve all coordinates.
  auto sourceType = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto targetType = dyn_cast<RankedTensorType>(op.getType());
  if (!sourceType || !targetType)
    return failure();

  // Reuse the source base, origin and axis strides.
  auto sourceInfo = analyze(op.getSource());
  if (failed(sourceInfo))
    return failure();
  PtrInfo info = *sourceInfo;

  // Integer results already carry the cast. Pointer offsets need the same
  // shape cast, preserving their numeric element type.
  if (!info.base) {
    info.offsets.fullOffsets = op.getResult();
  } else if (sourceType != targetType) {
    builder.setInsertionPoint(op);
    if (!info.hasScalarBase()) {
      if (info.base == op.getSource())
        info.base = op.getResult();
      else
        info.base =
            builder.create<tensor::CastOp>(op.getLoc(), targetType, info.base);
    }
    Value full = dyn_cast<Value>(info.offsets.fullOffsets);
    if (!full)
      full = builder.create<arith::ConstantOp>(
          op.getLoc(),
          cast<TypedAttr>(cast<Attribute>(info.offsets.fullOffsets)));
    Type storageType = targetType.clone(
        cast<RankedTensorType>(full.getType()).getElementType());
    info.offsets.fullOffsets =
        builder.create<tensor::CastOp>(op.getLoc(), storageType, full)
            .getResult();
  }
  info.offsets.valueType =
      info.base ? addressValueType(targetType, builder) : targetType;

  // Fixed-size sub-axis partitions require a static extent in the result.
  for (auto [axis, size] : llvm::zip(info.offsets.axes, targetType.getShape()))
    if (ShapedType::isDynamic(size))
      axis.subAxes.clear();
  if (auto full = dyn_cast<Value>(info.offsets.fullOffsets)) {
    setUniformOrigin(info.offsets, full, builder);
  } else if (auto origin = getIntegerConstant(info.offsets.fullOffsets, 64);
             succeeded(origin)) {
    setUniformOrigin(info.offsets, op.getResult(), builder,
                     builder.getIndexAttr(origin->getSExtValue()));
  }
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> PtrAnalysis::parseSelect(arith::SelectOp op) {
  // A scalar pointer selection can serve directly as the resulting base.
  if (isa<triton::PointerType>(op.getType()))
    return buildPointerInfo(op.getResult(), builder);

  // Obtain both branch results before combining their offset facts.
  auto lhsInfo = analyze(op.getTrueValue());
  auto rhsInfo = analyze(op.getFalseValue());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();

  // Combine branch facts; return pointer or fully structured results directly.
  if (!lhsInfo->base)
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPoint(op);
  auto result = combineSelectInfo(op, *lhsInfo, *rhsInfo, builder);
  if (failed(result) || result->base || result->offsets.hasStructuredForm())
    return result;

  // Condition invariance may refine the integer result; unavailable facts
  // leave the already known branch result intact.
  auto conditionInfo = analyze(op.getCondition());
  if (failed(conditionInfo))
    return result;
  // A tensor condition can be uniform on some axes too. Keep only equality
  // facts shared by all three inputs, then extract the original result if
  // every axis is invariant.
  bool scalarCondition = conditionInfo->offsets.getShape().empty();
  for (unsigned i = 0; i < result->offsets.axes.size(); ++i) {
    bool conditionInvariantOnAxis =
        scalarCondition ||
        conditionInfo->offsets.axes[i].kind == AxisKind::Invariant;
    if (conditionInvariantOnAxis &&
        lhsInfo->offsets.axes[i].kind == AxisKind::Invariant &&
        rhsInfo->offsets.axes[i].kind == AxisKind::Invariant)
      result->offsets.axes[i].kind = AxisKind::Invariant;
  }

  // Recover a scalar origin when every result axis is invariant.
  setUniformOrigin(result->offsets, op.getResult(), builder);
  return result;
}

// Integer rules.
PtrInfo PtrAnalysis::parseMakeRange(triton::MakeRangeOp op) {
  // The verifier guarantees a nonempty 1D i32 range matching the signed bounds.
  int64_t start = op.getStartAttr().getInt();
  // Represent the range as its starting offset plus a unit-stride axis.
  OffsetInfo info;
  info.valueType = op.getType();
  info.fullOffsets = op.getResult();
  info.axes = {AxisInfo(AxisKind::Structured, builder.getIndexAttr(1))};
  info.scalarOffset = builder.getIndexAttr(start);
  return PtrInfo{{}, std::move(info)};
}

FailureOr<PtrInfo> PtrAnalysis::parseBinaryArithmetic(Value value) {
  // Obtain both operands' origins and axis facts through the current session.
  Operation *op = value.getDefiningOp();
  auto lhsInfo = analyze(op->getOperand(0));
  auto rhsInfo = analyze(op->getOperand(1));
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  const auto &lhs = *lhsInfo, &rhs = *rhsInfo;
  builder.setInsertionPointAfter(op);

  // Add or subtract the origins and corresponding axis contributions.
  // Preserve the legacy no-overflow assumption for index descriptors. Dynamic
  // origins and strides do not need a separate propagated range proof.
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto offsets = addOffsetInfo(value, lhs.offsets, rhs.offsets,
                                 isa<arith::SubIOp>(op), builder, op->getLoc());
    if (failed(offsets))
      return failure();
    return PtrInfo{{}, std::move(*offsets)};
  }

  // A uniform multiplier scales the other operand's origin and axis strides.
  if (isa<arith::MulIOp>(op)) {
    bool lhsUniform = lhs.offsets.isUniform();
    if (lhsUniform || rhs.offsets.isUniform()) {
      const auto &varying = lhsUniform ? rhs.offsets : lhs.offsets;
      const auto &scale = lhsUniform ? lhs.offsets : rhs.offsets;
      auto offsets = scaleOffsetInfo(value, varying, scale.scalarOffset,
                                     builder, op->getLoc());
      if (failed(offsets))
        return failure();
      return PtrInfo{{}, std::move(*offsets)};
    }
  }

  // Other cases retain the complete value and common invariant axes.
  auto fallback = combineElementwise(value, lhs, rhs, builder);
  if (failed(fallback))
    return failure();

  // Division and remainder may recover a stronger decomposition.
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return refineDivRem(value, lhs, rhs, *fallback, true, builder);
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return refineDivRem(value, lhs, rhs, *fallback, false, builder);
  return fallback;
}

FailureOr<PtrInfo> PtrAnalysis::parseCast(Value value, Value input,
                                          bool unsignedCast) {
  // Obtain the source facts and compare the integer widths of the cast.
  auto sourceInfo = analyze(input);
  if (failed(sourceInfo))
    return failure();
  const auto &sourceOffsets = sourceInfo->offsets;
  builder.setInsertionPointAfterValue(value);
  auto targetWidth = getIntegerWidth(value.getType());
  auto sourceWidth = getIntegerWidth(sourceOffsets.valueType);
  if (failed(targetWidth) || failed(sourceWidth))
    return failure();

  // Check whether every represented lane keeps the same mathematical value.
  bool preservesValue = *targetWidth == *sourceWidth ||
                        (*targetWidth > *sourceWidth && !unsignedCast);
  if (!preservesValue) {
    auto bounds = constantDecompositionRange(sourceOffsets, *sourceWidth);
    preservesValue =
        bounds && (*targetWidth > *sourceWidth
                       ? bounds->isAllNonNegative()
                       : bounds->getSignedMin().isSignedIntN(*targetWidth) &&
                             bounds->getSignedMax().isSignedIntN(*targetWidth));
  }
  if (preservesValue) {
    // Signed widening keeps index descriptors. Other casts need the local
    // check above because they can change the value represented by a lane.
    auto offsets = sourceOffsets;
    offsets.valueType = value.getType();
    offsets.fullOffsets = value;
    setUniformOrigin(offsets, value, builder);
    return PtrInfo{{}, std::move(offsets)};
  }

  // Otherwise retain the exact cast result and only the invariant axis facts.
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets))
    return failure();
  for (unsigned i = 0; i < sourceOffsets.axes.size(); ++i)
    if (sourceOffsets.axes[i].kind == AxisKind::Invariant)
      offsets->axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(*offsets, value, builder);
  return PtrInfo{{}, std::move(*offsets)};
}

FailureOr<PtrInfo> PtrAnalysis::parseFloatToInt(arith::FPToSIOp op) {
  // First use any constant or uniform facts available from the integer result.
  auto info = buildIntegerInfo(op.getResult(), builder);
  if (failed(info) || info->offsets.getShape().empty())
    return info;

  // Floating arithmetic contributes axis equality, without integer strides.
  llvm::DenseMap<Value, SmallVector<bool>> invarianceCache;
  auto invariantAxes = inferInvariantAxes(op.getIn(), *this, invarianceCache);
  for (unsigned i = 0; i < invariantAxes.size(); ++i)
    if (invariantAxes[i])
      info->offsets.axes[i].kind = AxisKind::Invariant;
  // Extract the integer result, preserving the actual conversion semantics.
  setUniformOrigin(info->offsets, op.getResult(), builder);
  return info;
}

FailureOr<PtrInfo> PtrAnalysis::parseLoad(triton::LoadOp op) {
  auto info = buildIntegerInfo(op.getResult(), builder);
  if (failed(info) || info->offsets.isUniform())
    return info;
  llvm::DenseMap<Value, SmallVector<bool>> invarianceCache;
  auto invariantAxes = inferLoadInvariantAxes(op, *this, invarianceCache);
  for (unsigned i = 0; i < invariantAxes.size(); ++i)
    if (invariantAxes[i])
      info->offsets.axes[i] = makeInvariantAxisInfo(builder);
  setUniformOrigin(info->offsets, op.getResult(), builder);
  return info;
}

namespace {
bool isIntegerSum(triton::ReduceOp op) {
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return false;
  Block &body = op.getCombineOp().front();
  if (!llvm::hasSingleElement(body.without_terminator()))
    return false;
  auto add = dyn_cast<arith::AddIOp>(body.front());
  return add && body.getTerminator()->getOperand(0) == add.getResult() &&
         ((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getLhs() == body.getArgument(1) &&
           add.getRhs() == body.getArgument(0)));
}

FailureOr<OffsetInfo> sumOffsetInfo(Value value, const OffsetInfo &source,
                                    unsigned axis, OpBuilder &builder) {
  if (!source.hasStructuredForm() ||
      llvm::any_of(source.getShape(), [](int64_t size) { return size <= 0; }))
    return failure();
  int64_t count = source.getShape()[axis];
  __int128 triangle = __int128(count) * (count - 1) / 2;
  if (triangle > std::numeric_limits<int64_t>::max())
    return failure();

  // Index descriptors use 64-bit arithmetic. A narrow source sum must not
  // wrap before sign extension: bound every possible partial sum as well as
  // the final result, without introducing helper IR for an unproved case.
  unsigned width = *getIntegerWidth(value.getType());
  if (width < 64) {
    auto range = constantDecompositionRange(source, width);
    if (!range ||
        __int128(count) * range->getSignedMin().getSExtValue() <
            -(__int128(1) << (width - 1)) ||
        __int128(count) * range->getSignedMax().getSExtValue() >
            (__int128(1) << (width - 1)) - 1)
      return failure();
  }

  auto result = buildOffsetInfo(value, builder);
  assert(succeeded(result) && "expected an integer sum result");
  builder.setInsertionPointAfterValue(value);
  ConcreteArithmetic arithmetic{builder.getIndexType(), builder,
                                value.getLoc()};
  OpFoldResult length = builder.getIndexAttr(count);
  auto origin = arithmetic.multiply(source.scalarOffset, length);
  auto reduced = arithmetic.multiply(source.axes[axis].stride,
                                     builder.getIndexAttr(int64_t(triangle)));
  if (failed(origin) || failed(reduced))
    return failure();
  origin = arithmetic.add(*origin, *reduced, false);
  if (failed(origin))
    return failure();
  result->scalarOffset = *origin;
  result->axes.clear();
  for (auto [i, inputAxis] : llvm::enumerate(source.axes)) {
    if (i == axis)
      continue;
    auto stride = arithmetic.multiply(inputAxis.stride, length);
    if (failed(stride))
      return failure();
    result->axes.push_back(buildAxisInfo(*stride));
  }
  return std::move(*result);
}
} // namespace

FailureOr<PtrInfo> PtrAnalysis::parseReduce(Value value, triton::ReduceOp op) {
  // Seed result facts; tensor axes require a supported arithmetic combiner.
  auto info = buildIntegerInfo(value, builder);
  if (failed(info) || info->offsets.getShape().empty() ||
      !hasInvariancePreservingCombiner(op))
    return info;

  if (isIntegerSum(op)) {
    auto source = analyze(op.getSrcs().front());
    if (succeeded(source)) {
      auto sum = sumOffsetInfo(value, source->offsets, op.getAxis(), builder);
      if (succeeded(sum))
        return PtrInfo{{}, std::move(*sum)};
    }
  }

  llvm::DenseMap<Value, SmallVector<bool>> invarianceCache;
  SmallVector<bool> invariantAxes(info->offsets.axes.size(), true);
  // Results can depend on every input (e.g. argmin). Intersect input facts,
  // then remove the reduced dimension; never copy a pre-reduction stride.
  for (Value operand : op.getSrcs()) {
    auto sourceInvariantAxes =
        inferInvariantAxes(operand, *this, invarianceCache);
    sourceInvariantAxes.erase(sourceInvariantAxes.begin() + op.getAxis());
    for (unsigned i = 0; i < invariantAxes.size(); ++i)
      invariantAxes[i] = invariantAxes[i] && sourceInvariantAxes[i];
  }

  // Keep common invariant axes and singleton dimensions, then recover an
  // origin.
  for (unsigned i = 0; i < invariantAxes.size(); ++i)
    if (invariantAxes[i] || info->offsets.getShape()[i] == 1)
      info->offsets.axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(info->offsets, value, builder);
  return info;
}

// Ordinary pointer rules.
FailureOr<PtrInfo> PtrAnalysis::parseAddPtr(triton::AddPtrOp op) {
  // Analyze the pointer and its integer increment independently.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  auto sourceInfo = analyze(op.getOffset());
  if (failed(sourceInfo))
    return failure();

  // Descriptor fields are already index. The complete offset is widened
  // separately before accumulation, preserving each addptr increment.
  builder.setInsertionPoint(op);
  auto info =
      addPtrOffsets(op.getResult(), *parentInfo, sourceInfo->offsets, builder);
  if (failed(info))
    return failure();
  return PtrInfo(std::move(*info));
}

FailureOr<PtrInfo> PtrAnalysis::parseBitcast(triton::BitcastOp op) {
  // A scalar cast is already an exact base, including across element widths.
  // Do not reinterpret offsets accumulated before that scalar boundary.
  if (isa<triton::PointerType>(op.getType()))
    return buildPointerInfo(op.getResult(), builder);

  // Reusing offsets requires identical shapes and encodings. The verifier
  // permits compatible dynamic shapes and an encoding on only one side.
  auto sourceType = cast<RankedTensorType>(op.getSrc().getType());
  auto targetType = cast<RankedTensorType>(op.getType());
  if (sourceType.getShape() != targetType.getShape() ||
      sourceType.getEncoding() != targetType.getEncoding())
    return failure();
  // The verifier guarantees pointer elements with the same address space.
  auto sourcePtrType = cast<triton::PointerType>(sourceType.getElementType());
  auto targetPtrType = cast<triton::PointerType>(targetType.getElementType());

  // Restrict element-unit conversion to the supported storage widths.
  auto storageWidth = [](Type type) -> unsigned {
    if (!isa<IntegerType, FloatType>(type))
      return 0;
    unsigned width = type.getIntOrFloatBitWidth();
    return width == 8 || width == 16 || width == 32 || width == 64 ? width : 0;
  };
  unsigned sourceElementWidth = storageWidth(sourcePtrType.getPointeeType());
  unsigned targetElementWidth = storageWidth(targetPtrType.getPointeeType());
  if (!sourceElementWidth || !targetElementWidth)
    return failure();

  // Recover the base and element offsets of the source pointer tensor.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  assert(!sourceInfo->block && "a pointer tensor cannot carry block metadata");
  PtrInfo info = *sourceInfo;
  builder.setInsertionPoint(op);

  // Equal-width casts reuse offsets; other casts must change their units.
  if (sourceElementWidth != targetElementWidth) {
    // Rescaling uses a dense factor tensor and therefore needs static extents.
    if (!targetType.hasStaticShape())
      return failure();
    uint64_t ratio = std::max(sourceElementWidth, targetElementWidth) /
                     std::min(sourceElementWidth, targetElementWidth);
    bool widenElement = sourceElementWidth < targetElementWidth;
    // Widening needs exact division: offset_new * sizeof(new) must equal
    // offset_old * sizeof(old). Test the complete offset, never round terms.
    if (widenElement && !canDivideDecomposition(info.offsets, ratio) &&
        !isKnownMultiple(info.offsets.fullOffsets, ratio))
      return failure();

    // Materialize the complete offset and the element-size conversion factor.
    auto full = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(full))
      return failure();
    auto fullType = cast<RankedTensorType>((*full).getType());
    Value factor = builder.create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(fullType, int64_t(ratio)));
    if (widenElement) {
      // Divide the full offset and retain exactly divisible descriptor fields.
      info.offsets.fullOffsets =
          builder.create<arith::DivSIOp>(op.getLoc(), *full, factor)
              .getResult();
      divideDecomposition(info.offsets, ratio, builder, op.getLoc());
    } else {
      // Smaller elements scale up both the complete offset and its descriptor.
      Value scaled = builder.create<arith::MulIOp>(op.getLoc(), *full, factor);
      auto offsets =
          scaleOffsetInfo(scaled, info.offsets, builder.getIndexAttr(ratio),
                          builder, op.getLoc());
      if (failed(offsets))
        return failure();
      info.offsets = std::move(*offsets);
    }
  }

  // Express scalar or per-lane bases in the destination pointer type.
  Type baseType = info.hasScalarBase() ? Type(targetPtrType) : Type(targetType);
  if (info.base.getType() != baseType)
    info.base =
        builder.create<triton::BitcastOp>(op.getLoc(), baseType, info.base);
  return PtrInfo(std::move(info));
}

// Block pointer rules.
FailureOr<PtrInfo> PtrAnalysis::parseMakeTensorPtr(triton::MakeTensorPtrOp op) {
  // Start from an ordinary scalar pointer with an index-typed base offset.
  auto baseInfo = analyze(op.getBase());
  if (failed(baseInfo))
    return failure();
  if (baseInfo->block || !baseInfo->offsets.valueType.isIndex())
    return failure();

  // Resolve each starting coordinate to its uniform index origin.
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    if (!sourceInfo->offsets.isUniform())
      return failure();
    coordinates.push_back(sourceInfo->offsets.scalarOffset);
  }

  // Combine the base, coordinates and strides into the block descriptor.
  builder.setInsertionPoint(op);
  return makeBlockInfo(op, *baseInfo, coordinates, builder);
}

FailureOr<PtrInfo> PtrAnalysis::parseAdvance(triton::AdvanceOp op) {
  // An advance needs the existing block coordinates and strides.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  if (!parentInfo->block)
    return failure();

  // Resolve each coordinate increment to its uniform index origin.
  SmallVector<OpFoldResult> coordinateDeltas;
  for (Value delta : op.getOffsets()) {
    auto deltaInfo = analyze(delta);
    if (failed(deltaInfo))
      return failure();
    if (!deltaInfo->offsets.isUniform())
      return failure();
    coordinateDeltas.push_back(deltaInfo->offsets.scalarOffset);
  }

  // Update the coordinates and add their stride-weighted offset contribution.
  builder.setInsertionPoint(op);
  return advanceBlockInfo(op, *parentInfo, coordinateDeltas, builder);
}

//===----------------------------------------------------------------------===//
// Numeric control flow
//===----------------------------------------------------------------------===//

namespace {
using KnownInfo = std::pair<Value, PtrInfo>;

// Scalars and singleton tensors already have exact uniform facts. Dynamic or
// empty tensors cannot supply the coordinates used by this representation.
bool isNumericCarrier(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getNumElements() > 1 &&
         succeeded(detail::getIntegerWidth(type));
}

// Across SCF edges retain only kinds and constant strides, never region-local
// SSA fields. A null stride in this internal state means no known constant.
SmallVector<AxisInfo> analyzeAxes(Value value, PtrAnalysis &analysis) {
  auto info = analysis.analyze(value);
  OpBuilder &builder = analysis.getBuilder();
  SmallVector<AxisInfo> axes;
  for (auto [i, size] :
       llvm::enumerate(cast<RankedTensorType>(value.getType()).getShape())) {
    AxisKind kind =
        succeeded(info) ? info->offsets.axes[i].kind : AxisKind::Unknown;
    if (size == 1)
      kind = AxisKind::Invariant;
    OpFoldResult stride;
    if (kind == AxisKind::Invariant) {
      stride = builder.getIndexAttr(0);
    } else if (kind == AxisKind::Structured) {
      auto constant =
          detail::getIntegerConstant(info->offsets.axes[i].stride, 64);
      if (succeeded(constant))
        stride = builder.getIndexAttr(constant->getSExtValue());
    }
    axes.emplace_back(kind, stride);
  }
  return axes;
}

// Invariant is the zero-stride subset of Structured. Kinds only weaken and a
// constant can only be lost, so the combined state still has finite height.
bool joinAxes(SmallVectorImpl<AxisInfo> &current, ArrayRef<AxisInfo> incoming) {
  bool changed = false;
  for (auto [axis, next] : llvm::zip_equal(current, incoming)) {
    AxisKind joined = AxisKind::Invariant;
    if (axis.kind == AxisKind::Unknown || next.kind == AxisKind::Unknown)
      joined = AxisKind::Unknown;
    else if (axis.kind == AxisKind::Structured ||
             next.kind == AxisKind::Structured)
      joined = AxisKind::Structured;
    OpFoldResult stride;
    if (joined != AxisKind::Unknown && axis.stride == next.stride)
      stride = axis.stride;
    // Other carriers and nested loops must be revisited when a constant is
    // lost even if the axis remains Structured.
    changed |= axis.kind != joined || axis.stride != stride;
    axis = AxisInfo(joined, stride);
  }
  return changed;
}

// These samples represent an assumed or proved axis class; they never prove
// that an arbitrary tensor is affine. Keep the current SSA as the exact value,
// so origins and strides may change on every iteration.
PtrInfo carrierInfo(Value value, ArrayRef<AxisInfo> axes, OpBuilder &builder) {
  auto offsets = buildOffsetInfo(value, builder);
  assert(succeeded(offsets) && "expected a supported numeric carrier");
  bool needsOrigin = llvm::any_of(axes,
                                  [](const AxisInfo &axis) {
                                    return axis.kind == AxisKind::Structured;
                                  }) ||
                     llvm::all_of(axes, [](const AxisInfo &axis) {
                       return axis.kind == AxisKind::Invariant;
                     });
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(value);
  Location loc = value.getLoc();
  SmallVector<Value> indices;
  Value origin;
  auto extract = [&]() -> Value {
    Value scalar = builder.create<tensor::ExtractOp>(loc, value, indices);
    if (!scalar.getType().isIndex())
      scalar = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                  scalar);
    return scalar;
  };
  if (needsOrigin) {
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    indices.assign(axes.size(), zero);
    origin = extract();
    offsets->scalarOffset = origin;
  }
  for (auto [i, axis] : llvm::enumerate(axes)) {
    OpFoldResult stride = axis.stride;
    if (!stride && axis.kind == AxisKind::Structured) {
      Value saved = indices[i];
      indices[i] = builder.create<arith::ConstantIndexOp>(loc, 1);
      stride =
          builder.create<arith::SubIOp>(loc, extract(), origin).getResult();
      indices[i] = saved;
    }
    if (!stride)
      stride = builder.getIndexAttr(0);
    offsets->axes[i] = AxisInfo(axis.kind, stride);
  }
  return PtrInfo{{}, std::move(*offsets)};
}

struct Carrier {
  Value argument;
  Value incoming;
  SmallVector<AxisInfo> axes;
};

SmallVector<KnownInfo, 0> assumptions(ArrayRef<Carrier> first,
                                      ArrayRef<Carrier> second,
                                      OpBuilder &builder) {
  SmallVector<KnownInfo, 0> known;
  for (ArrayRef<Carrier> carriers : {first, second})
    for (const Carrier &carrier : carriers)
      known.emplace_back(carrier.argument,
                         carrierInfo(carrier.argument, carrier.axes, builder));
  return known;
}

bool joinIncoming(SmallVectorImpl<Carrier> &carriers, PtrAnalysis &analysis) {
  bool changed = false;
  for (Carrier &carrier : carriers)
    changed |= joinAxes(carrier.axes, analyzeAxes(carrier.incoming, analysis));
  return changed;
}

} // namespace

FailureOr<PtrInfo> PtrAnalysis::parseFor(Value value, scf::ForOp loop) {
  SmallVector<Carrier> carriers;
  auto yields =
      cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands();
  for (auto [arg, init, yielded] :
       llvm::zip_equal(loop.getRegionIterArgs(), loop.getInitArgs(), yields))
    if (isNumericCarrier(arg))
      carriers.push_back({arg, yielded, analyzeAxes(init, *this)});

  // A_{n+1} = join(A_n, transfer(A_n)); A_0 already contains the initial
  // values, including the zero-trip result. Every pass has a fresh cache.
  bool changed;
  do {
    loopCaches[loop].emplace_back();
    auto cleanup = llvm::make_scope_exit([&] { popLoopCache(loop); });
    for (auto &[argument, info] : assumptions(carriers, {}, builder))
      getCacheMap(loop).insert_or_assign(argument, std::move(info));
    changed = joinIncoming(carriers, *this);
  } while (changed);

  Value argument = value;
  if (auto result = dyn_cast<OpResult>(value))
    argument = loop.getRegionIterArgs()[result.getResultNumber()];
  for (const Carrier &carrier : carriers)
    if (carrier.argument == argument)
      return carrierInfo(value, carrier.axes, builder);
  llvm_unreachable("queried for value must be a numeric carrier");
}

FailureOr<PtrInfo> PtrAnalysis::parseWhile(Value value, scf::WhileOp loop) {
  SmallVector<Carrier> before, after;
  for (auto [arg, init, yielded] :
       llvm::zip_equal(loop.getBeforeArguments(), loop.getInits(),
                       loop.getYieldOp().getOperands()))
    if (isNumericCarrier(arg))
      before.push_back({arg, yielded, analyzeAxes(init, *this)});

  // condition arguments feed both the after region and the loop results.
  // Their count, order and types need not match the before region.
  {
    loopCaches[loop].emplace_back();
    auto cleanup = llvm::make_scope_exit([&] { popLoopCache(loop); });
    for (auto &[argument, info] : assumptions(before, {}, builder))
      getCacheMap(loop).insert_or_assign(argument, std::move(info));
    for (auto [arg, condition] : llvm::zip_equal(
             loop.getAfterArguments(), loop.getConditionOp().getArgs()))
      if (isNumericCarrier(arg))
        after.push_back({arg, condition, analyzeAxes(condition, *this)});
  }

  bool changed;
  do {
    loopCaches[loop].emplace_back();
    auto cleanup = llvm::make_scope_exit([&] { popLoopCache(loop); });
    for (auto &[argument, info] : assumptions(before, after, builder))
      getCacheMap(loop).insert_or_assign(argument, std::move(info));
    changed = joinIncoming(before, *this);
    changed |= joinIncoming(after, *this);
  } while (changed);

  Value argument = value;
  if (auto result = dyn_cast<OpResult>(value))
    argument = loop.getAfterArguments()[result.getResultNumber()];
  for (ArrayRef<Carrier> carriers :
       {ArrayRef<Carrier>(before), ArrayRef<Carrier>(after)})
    for (const Carrier &carrier : carriers)
      if (carrier.argument == argument)
        return carrierInfo(value, carrier.axes, builder);
  llvm_unreachable("queried while value must be a numeric carrier");
}

FailureOr<PtrInfo> PtrAnalysis::parseIf(OpResult value, scf::IfOp branch) {
  unsigned i = value.getResultNumber();
  auto axes = analyzeAxes(branch.thenYield().getOperand(i), *this);
  joinAxes(axes, analyzeAxes(branch.elseYield().getOperand(i), *this));
  // A scalar condition chooses a whole tensor. Never export branch-local
  // numerical fields into the parent region.
  return carrierInfo(value, axes, builder);
}

bool PtrAnalysis::isControlFlowValue(Value value) {
  if (!value || !isNumericCarrier(value))
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<scf::ForOp, scf::WhileOp>(arg.getOwner()->getParentOp());
  return isa<scf::ForOp, scf::WhileOp, scf::IfOp>(value.getDefiningOp());
}

FailureOr<PtrInfo> PtrAnalysis::parseControlFlow(Value value) {
  Operation *op = value.getDefiningOp();
  if (auto arg = dyn_cast<BlockArgument>(value))
    op = arg.getOwner()->getParentOp();
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return parseFor(value, loop);
  if (auto loop = dyn_cast<scf::WhileOp>(op))
    return parseWhile(value, loop);
  return parseIf(cast<OpResult>(value), cast<scf::IfOp>(op));
}

//===----------------------------------------------------------------------===//
// Component preservation
//===----------------------------------------------------------------------===//

namespace {
ChangeInfo changesFor(const PtrInfo &layout, bool changed) {
  ChangeInfo result;
  result.base = changed;
  result.uniform = result.residual = !layout.block && changed;
  unsigned rank = layout.offsets.axes.size();
  result.strides.assign(rank, changed);
  if (layout.block) {
    result.parentShape.assign(rank, changed);
    result.coordinates.assign(rank, changed);
  }
  result.order = layout.block && changed;
  return result;
}

// The fields refer to the same fixed query/layout. True can only accumulate.
bool joinChanges(ChangeInfo &current, const ChangeInfo &incoming) {
  bool changed = false;
  auto join = [&](bool &field, bool next) {
    changed |= next && !field;
    field |= next;
  };
  join(current.base, incoming.base);
  join(current.uniform, incoming.uniform);
  join(current.residual, incoming.residual);
  join(current.order, incoming.order);
  auto joinVector = [&](SmallVectorImpl<bool> &fields, ArrayRef<bool> next) {
    for (auto [field, value] : llvm::zip_equal(fields, next))
      join(field, value);
  };
  joinVector(current.strides, incoming.strides);
  joinVector(current.parentShape, incoming.parentShape);
  joinVector(current.coordinates, incoming.coordinates);
  return changed;
}

ChangeInfo compareComponents(const PtrInfo &info, const PtrInfo &layout) {
  ChangeInfo result = changesFor(layout, true);
  if (bool(info.block) != bool(layout.block) ||
      info.hasScalarBase() != layout.hasScalarBase())
    return result;
  result.base = info.base != layout.base;
  for (unsigned i = 0; i < result.strides.size(); ++i) {
    const auto &lhs = info.offsets.axes[i], &rhs = layout.offsets.axes[i];
    if (lhs.kind != AxisKind::Unknown && rhs.kind != AxisKind::Unknown)
      result.strides[i] = !detail::equalConstant(lhs.stride, rhs.stride);
  }
  if (layout.block) {
    result.base = info.block->descriptorBase != layout.block->descriptorBase;
    result.order = info.block->order != layout.block->order;
    for (unsigned i = 0; i < result.coordinates.size(); ++i) {
      result.coordinates[i] = !detail::equalConstant(
          info.block->coordinateOffsets[i], layout.block->coordinateOffsets[i]);
      result.parentShape[i] = !detail::equalConstant(
          info.block->parentShape[i], layout.block->parentShape[i]);
    }
  } else if (info.offsets.hasStructuredForm() &&
             layout.offsets.hasStructuredForm()) {
    // Only a complete affine form has its origin as the uniform contribution
    // and no irregular part. Unknown axes do not imply zero residuals.
    result.uniform = !detail::equalConstant(info.offsets.scalarOffset,
                                            layout.offsets.scalarOffset);
    result.residual = false;
  }
  return result;
}

struct ChangeCarrier {
  Value argument;
  Value incoming;
  ChangeInfo changes;
};

// A new loop hypothesis invalidates its body facts, not facts defined outside
// it. This local memo is scoped to a single reference, unlike the numeric
// cache.
llvm::DenseMap<Value, ChangeInfo>
externalChanges(const llvm::DenseMap<Value, ChangeInfo> &cache,
                Operation *loop) {
  llvm::DenseMap<Value, ChangeInfo> result;
  for (const auto &[value, info] : cache) {
    Value source = value;
    Operation *owner = source.getParentBlock()->getParentOp();
    if (!loop->isAncestor(owner))
      result.insert({value, info});
  }
  return result;
}
} // namespace

FailureOr<ChangeInfo> PtrAnalysis::analyzeChanges(Value value, Value reference,
                                                  Value initial) {
  if (!initial)
    initial = reference;
  if (!value || !reference || !initial ||
      value.getType() != reference.getType() ||
      initial.getType() != reference.getType() ||
      (!detail::isOrdinaryPtrType(reference.getType()) &&
       !detail::isBlockPtrType(reference.getType())))
    return failure();
  auto layout = analyze(initial);
  if (failed(layout))
    return failure();
  llvm::DenseMap<Value, ChangeInfo> cache;
  if (auto argument = dyn_cast<BlockArgument>(reference)) {
    Operation *loop = argument.getOwner()->getParentOp();
    if (isa<scf::ForOp, scf::WhileOp>(loop) &&
        !loop->isAncestor(initial.getParentBlock()->getParentOp())) {
      // Compare all carried fields to this fixed initial value first. This
      // includes the target argument: if its field varies, equality to an
      // unrelated argument's initial field says nothing about current values.
      llvm::DenseMap<Value, ChangeInfo> invariants;
      computeLoopChanges(loop, initial, initial, *layout, invariants);
      auto target = invariants.find(reference);
      if (target != invariants.end()) {
        for (const auto &[candidate, changes] : invariants) {
          auto other = dyn_cast<BlockArgument>(candidate);
          if (candidate == reference || !other ||
              other.getOwner()->getParentOp() != loop)
            continue;
          ChangeInfo relative = changes;
          joinChanges(relative, target->second);
          cache.try_emplace(candidate, std::move(relative));
        }
      }
    }
  }
  return computeChanges(value, reference, initial, *layout, cache);
}

void PtrAnalysis::computeLoopChanges(Operation *loop, Value reference,
                                     Value initial, const PtrInfo &layout,
                                     llvm::DenseMap<Value, ChangeInfo> &cache) {
  SmallVector<ChangeCarrier> before, after;
  auto query = [&](Value value, llvm::DenseMap<Value, ChangeInfo> &local) {
    return computeChanges(value, reference, initial, layout, local);
  };
  auto queryInitial = [&](Value value) {
    ChangeInfo result = query(value, cache);
    if (reference != initial || value == initial)
      return result;
    auto info = analyze(value);
    if (failed(info))
      return result;
    // With a fixed reference, compare the completed initial expressions too:
    // different addptr chains can still have identical final strides. Keep
    // preservation proved by either this comparison or the transfer rules,
    // which can also prove zero increments on otherwise unknown axes.
    ChangeInfo concrete = compareComponents(*info, layout);
    // An opaque current pointer may become a fresh base with zero offsets.
    // Those fields cannot refine facts expressed relative to the old base.
    if (concrete.base)
      return result;
    result.base &= concrete.base;
    result.uniform &= concrete.uniform;
    result.residual &= concrete.residual;
    result.order &= concrete.order;
    for (auto [field, other] :
         llvm::zip_equal(result.strides, concrete.strides))
      field &= other;
    for (auto [field, other] :
         llvm::zip_equal(result.parentShape, concrete.parentShape))
      field &= other;
    for (auto [field, other] :
         llvm::zip_equal(result.coordinates, concrete.coordinates))
      field &= other;
    cache.insert_or_assign(value, result);
    return result;
  };
  if (auto op = dyn_cast<scf::ForOp>(loop)) {
    auto yields =
        cast<scf::YieldOp>(op.getBody()->getTerminator()).getOperands();
    for (auto [arg, init, yielded] :
         llvm::zip_equal(op.getRegionIterArgs(), op.getInitArgs(), yields))
      if (arg.getType() == reference.getType())
        before.push_back({arg, yielded, queryInitial(init)});
  } else {
    auto whileOp = cast<scf::WhileOp>(loop);
    for (auto [arg, init, yielded] :
         llvm::zip_equal(whileOp.getBeforeArguments(), whileOp.getInits(),
                         whileOp.getYieldOp().getOperands()))
      if (arg.getType() == reference.getType())
        before.push_back({arg, yielded, queryInitial(init)});
    auto local = externalChanges(cache, loop);
    for (const auto &carrier : before)
      local.insert_or_assign(carrier.argument, carrier.changes);
    for (auto [arg, condition] : llvm::zip_equal(
             whileOp.getAfterArguments(), whileOp.getConditionOp().getArgs()))
      if (arg.getType() == reference.getType())
        after.push_back({arg, condition, query(condition, local)});
  }

  auto external = externalChanges(cache, loop);
  bool changed;
  do {
    auto local = external;
    for (ArrayRef<ChangeCarrier> carriers :
         {ArrayRef<ChangeCarrier>(before), ArrayRef<ChangeCarrier>(after)})
      for (const auto &carrier : carriers)
        local.insert_or_assign(carrier.argument, carrier.changes);
    changed = false;
    for (auto *carriers : {&before, &after})
      for (auto &carrier : *carriers)
        changed |= joinChanges(carrier.changes, query(carrier.incoming, local));
  } while (changed);

  for (ArrayRef<ChangeCarrier> carriers :
       {ArrayRef<ChangeCarrier>(before), ArrayRef<ChangeCarrier>(after)})
    for (const auto &carrier : carriers)
      cache.insert_or_assign(carrier.argument, carrier.changes);
  if (auto op = dyn_cast<scf::ForOp>(loop)) {
    for (auto [result, argument] :
         llvm::zip_equal(op.getResults(), op.getRegionIterArgs()))
      if (result.getType() == reference.getType())
        cache.insert_or_assign(result, cache.lookup(argument));
  } else {
    auto whileOp = cast<scf::WhileOp>(loop);
    for (auto [result, argument] :
         llvm::zip_equal(whileOp.getResults(), whileOp.getAfterArguments()))
      if (result.getType() == reference.getType())
        cache.insert_or_assign(result, cache.lookup(argument));
  }
}

ChangeInfo
PtrAnalysis::computeChanges(Value value, Value reference, Value initial,
                            const PtrInfo &layout,
                            llvm::DenseMap<Value, ChangeInfo> &cache) {
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  auto remember = [&](ChangeInfo info) {
    cache.insert_or_assign(value, info);
    return info;
  };
  if (value == reference || value == initial)
    return remember(changesFor(layout, false));
  auto unknown = changesFor(layout, true);
  if (value.getType() != reference.getType())
    return remember(std::move(unknown));
  // Only the enclosing-loop fixed point may supply a fact for another
  // parameter. Equal initial values alone do not establish current equality.
  if (isa<BlockArgument>(value))
    return remember(std::move(unknown));
  auto query = [&](Value input) {
    return computeChanges(input, reference, initial, layout, cache);
  };

  if (auto op = value.getDefiningOp<triton::AddPtrOp>()) {
    ChangeInfo result = query(op.getPtr());
    auto delta = analyze(op.getOffset());
    if (failed(delta)) {
      result.uniform = result.residual = true;
      std::fill(result.strides.begin(), result.strides.end(), true);
      return remember(std::move(result));
    }
    const OffsetInfo &offset = delta->offsets;
    if (offset.isUniform()) {
      // Each increment is widened before address accumulation. Adding a
      // uniform increment cannot change the residual, even on opaque axes.
      result.uniform |= !detail::zero(offset.scalarOffset);
    } else {
      auto width = detail::getIntegerWidth(op.getOffset().getType());
      bool separable =
          offset.hasStructuredForm() &&
          (*width == 64 || detail::decompositionFitsSigned(offset, *width));
      if (separable) {
        result.uniform |= !detail::zero(offset.scalarOffset);
        for (unsigned i = 0; i < result.strides.size(); ++i) {
          bool changed = !detail::zero(offset.axes[i].stride);
          if (layout.offsets.axes[i].kind == AxisKind::Unknown)
            result.residual |= changed;
          else
            result.strides[i] |= changed;
        }
      } else {
        result.uniform = result.residual = true;
        std::fill(result.strides.begin(), result.strides.end(), true);
      }
    }
    return remember(std::move(result));
  }
  if (auto op = value.getDefiningOp<triton::AdvanceOp>()) {
    ChangeInfo result = query(op.getPtr());
    for (auto [i, delta] : llvm::enumerate(op.getOffsets())) {
      auto info = analyze(delta);
      result.coordinates[i] |=
          failed(info) || !detail::zero(info->offsets.scalarOffset);
    }
    return remember(std::move(result));
  }
  if (auto op = value.getDefiningOp<arith::SelectOp>()) {
    ChangeInfo result = query(op.getTrueValue());
    joinChanges(result, query(op.getFalseValue()));
    if (isa<RankedTensorType>(op.getCondition().getType())) {
      auto condition = analyze(op.getCondition());
      if (failed(condition) || !condition->offsets.isUniform()) {
        // Selecting different uniform offsets lane by lane can create a new
        // residual. Per-branch preservation alone does not describe that mix.
        result.uniform = result.residual = true;
        std::fill(result.strides.begin(), result.strides.end(), true);
      }
    }
    return remember(std::move(result));
  }
  if (auto op = value.getDefiningOp<scf::IfOp>()) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    ChangeInfo result = query(op.thenYield().getOperand(index));
    joinChanges(result, query(op.elseYield().getOperand(index)));
    return remember(std::move(result));
  }

  // The same solver handles nested results and enclosing argument invariants.
  if (Operation *op = value.getDefiningOp();
      isa<scf::ForOp, scf::WhileOp>(op)) {
    computeLoopChanges(op, reference, initial, layout, cache);
    return cache.lookup(value);
  }

  // Compare available concrete fields for independent expressions, notably two
  // make_tensor_ptr operations sharing all operands except one shape field.
  // These are actual values, not numeric guesses for current loop parameters.
  auto info = analyze(value);
  if (failed(info))
    return remember(std::move(unknown));
  return remember(compareComponents(*info, layout));
}

} // namespace mlir::triton::ptr
