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

#include "TritonMemoryAccess/PointerAnalysis.h"
#include "PointerAnalysisInternal.h"
#include "TritonMemoryAccess/PointerAnalysisDriver.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::pointer {

bool OffsetComponents::hasAffineForm() const {
  return valueType && uniformOffset && axes.size() == shape.size() &&
         strides.size() == shape.size() &&
         llvm::none_of(axes,
                       [](AxisKind axis) { return axis == AxisKind::Unknown; });
}

bool OffsetComponents::isUniform() const {
  return hasAffineForm() && llvm::all_of(axes, [](AxisKind axis) {
           return axis == AxisKind::Invariant;
         });
}

std::optional<OpFoldResult>
OffsetComponents::getKnownStride(unsigned axis) const {
  if (!valueType || axis >= shape.size() || axis >= axes.size() ||
      axis >= strides.size() || !strides[axis] ||
      axes[axis] == AxisKind::Unknown)
    return std::nullopt;
  return strides[axis];
}

std::optional<OpFoldResult> OffsetComponents::getAffineOrigin() const {
  return hasAffineForm() ? std::optional<OpFoldResult>(uniformOffset)
                         : std::nullopt;
}

FailureOr<OffsetComponents> makeOpaqueOffset(Value value, OpBuilder &builder,
                                             ArithmeticDomain domain) {
  if (!value || (!isa<IntegerType, IndexType>(value.getType()) &&
                 !isa<RankedTensorType>(value.getType())))
    return failure();
  Type element = detail::getIntegerElementType(value.getType());
  if ((!element.isSignlessInteger() && !element.isIndex()) ||
      (domain == ArithmeticDomain::ElementAddress &&
       !element.isSignlessInteger(64)))
    return failure();
  OffsetComponents result;
  result.valueType = value.getType();
  result.domain = domain;
  result.shape = detail::getLogicalShape(value.getType());
  result.completeOffset = value;
  result.axes.assign(result.shape.size(), AxisKind::Unknown);
  result.strides.assign(result.shape.size(), builder.getZeroAttr(element));
  if (!isa<RankedTensorType>(value.getType())) {
    result.uniformOffset = value;
  } else if (result.shape.empty()) {
    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      result.uniformOffset = splat.getSrc();
    } else {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(value);
      result.uniformOffset =
          builder.create<tensor::ExtractOp>(value.getLoc(), value, ValueRange{})
              .getResult();
    }
  } else {
    result.uniformOffset = builder.getZeroAttr(element);
  }
  return result;
}

namespace {

bool validOfrType(OpFoldResult value, Type expected) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType() == expected;
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType() == expected;
  return false;
}

bool validOffset(Type input, const OffsetComponents &components,
                 ArithmeticDomain expectedDomain, AnalysisOptions options,
                 bool allowAffineOnly = false) {
  Type resultType = input;
  if (expectedDomain == ArithmeticDomain::ElementAddress) {
    if (auto tensor = dyn_cast<RankedTensorType>(input))
      resultType = RankedTensorType::get(
          tensor.getShape(), IntegerType::get(input.getContext(), 64),
          tensor.getEncoding());
    else
      resultType = IntegerType::get(input.getContext(), 64);
  }
  Type scalarType = detail::getIntegerElementType(resultType);
  if (components.domain != expectedDomain ||
      components.valueType != resultType ||
      components.shape != detail::getLogicalShape(resultType) ||
      components.axes.size() != components.shape.size() ||
      components.strides.size() != components.shape.size() ||
      failed(detail::getIntegerWidth(resultType, options)) ||
      (components.completeOffset
           ? !validOfrType(components.completeOffset, resultType)
           : !allowAffineOnly || !components.hasAffineForm()) ||
      !validOfrType(components.uniformOffset, scalarType))
    return false;
  for (OpFoldResult stride : components.strides)
    if (!validOfrType(stride, scalarType))
      return false;
  return true;
}

bool validPointer(Value boundary, const PointerComponents &components,
                  AnalysisOptions options) {
  if (!boundary || !components.base ||
      !isa<triton::PointerType>(components.base.getType()))
    return false;
  Type pointerElement = boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(pointerElement))
    pointerElement = tensor.getElementType();
  auto pointerType = dyn_cast<triton::PointerType>(pointerElement);
  auto baseType = dyn_cast<triton::PointerType>(components.base.getType());
  if (!pointerType || !baseType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType && isa<ShapedType>(pointerType.getPointeeType()))
    return false;
  Type elementType =
      blockType ? blockType.getElementType() : pointerType.getPointeeType();
  if (components.elementType != elementType ||
      baseType.getPointeeType() != elementType ||
      components.addressSpace != pointerType.getAddressSpace() ||
      baseType.getAddressSpace() != components.addressSpace ||
      (blockType ? !detail::validBlockPointerBinding(boundary, components)
                 : components.block.has_value()))
    return false;
  Type source = blockType ? Type(blockType) : boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    source = RankedTensorType::get(tensor.getShape(),
                                   IntegerType::get(source.getContext(), 64),
                                   tensor.getEncoding());
  else
    source = IntegerType::get(source.getContext(), 64);
  return validOffset(source, components.offsets,
                     ArithmeticDomain::ElementAddress, options,
                     /*allowAffineOnly=*/bool(blockType));
}

bool isNumericHelper(Operation *op) {
  if (op->getName().getDialectNamespace() == "arith")
    return true;
  return isa<tensor::ExtractOp, triton::MakeRangeOp, triton::SplatOp,
             triton::BroadcastOp, triton::ExpandDimsOp>(op);
}

} // namespace

namespace detail {
bool validOffsetBinding(Value boundary, const OffsetComponents &components,
                        ArithmeticDomain expectedDomain,
                        AnalysisOptions options) {
  return boundary &&
         validOffset(boundary.getType(), components, expectedDomain, options);
}

bool validPointerBinding(Value boundary, const PointerComponents &components,
                         AnalysisOptions options) {
  return supportsAddressAnalysis(options) &&
         validPointer(boundary, components, options);
}
} // namespace detail

struct PointerAnalysis::Impl {
  explicit Impl(OpBuilder &builder, AnalysisOptions options)
      : options(options), driver(builder, detail::PointerRules(options)) {}
  AnalysisOptions options;
  detail::AnalysisDriver<detail::PointerRules> driver;
};

PointerAnalysis::PointerAnalysis(OpBuilder &builder, AnalysisOptions options)
    : impl(std::make_unique<Impl>(builder, options)) {}
PointerAnalysis::~PointerAnalysis() = default;
PointerAnalysis::PointerAnalysis(PointerAnalysis &&) noexcept = default;
PointerAnalysis &
PointerAnalysis::operator=(PointerAnalysis &&) noexcept = default;

FailureOr<OffsetComponents> PointerAnalysis::analyzeOffset(Value value) {
  auto result = impl->driver.analyze({value, detail::SourceInteger});
  if (failed(result) || !result->offset)
    return failure();
  return *result->offset;
}

FailureOr<OffsetComponents> PointerAnalysis::analyzeAddressDelta(Value value) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result = impl->driver.analyze({value, detail::AddressDelta});
  if (failed(result) || !result->offset)
    return failure();
  return *result->offset;
}

FailureOr<PointerComponents> PointerAnalysis::analyzePointer(Value value) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result = impl->driver.analyze({value, detail::PointerValue});
  if (failed(result) || !result->pointer)
    return failure();
  return *result->pointer;
}

FailureOr<BlockAdvanceDelta>
PointerAnalysis::analyzeBlockAdvanceDelta(triton::AdvanceOp op) {
  if (!op || !detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result =
      impl->driver.analyze({op.getResult(), detail::AdvanceDeltaRequest});
  if (failed(result) || !result->advanceDelta)
    return failure();
  return *result->advanceDelta;
}

const PointerComponents *PointerAnalysis::findCachedPointer(Value value) const {
  auto *result = impl->driver.lookupCompleted({value, detail::PointerValue});
  return result && result->pointer ? &*result->pointer : nullptr;
}

const BlockAdvanceDelta *
PointerAnalysis::findCachedBlockAdvanceDelta(triton::AdvanceOp op) const {
  if (!op)
    return nullptr;
  auto *result = impl->driver.lookupCompleted(
      {op.getResult(), detail::AdvanceDeltaRequest});
  return result && result->advanceDelta ? &*result->advanceDelta : nullptr;
}

const OffsetComponents *PointerAnalysis::findCachedOffset(Value value) const {
  auto *result = impl->driver.lookupCompleted({value, detail::SourceInteger});
  return result ? detail::getOffset(*result) : nullptr;
}

const OffsetComponents *
PointerAnalysis::findCachedAddressDelta(Value value) const {
  if (!detail::supportsAddressAnalysis(impl->options))
    return nullptr;
  auto *result = impl->driver.lookupCompleted({value, detail::AddressDelta});
  return result ? detail::getOffset(*result) : nullptr;
}

LogicalResult PointerAnalysis::bindOffset(Value boundary,
                                          const OffsetComponents &components) {
  if (!detail::validOffsetBinding(
          boundary, components, ArithmeticDomain::SourceInteger, impl->options))
    return failure();
  detail::AnalysisResult result;
  result.offset = components;
  FailureOr<unsigned> width =
      detail::getIntegerWidth(boundary.getType(), impl->options);
  result.integerRange = llvm::ConstantRange::getFull(*width);
  impl->driver.bindBoundary({boundary, detail::SourceInteger},
                            std::move(result));
  return success();
}

LogicalResult
PointerAnalysis::bindPointer(Value boundary,
                             const PointerComponents &components) {
  if (!detail::validPointerBinding(boundary, components, impl->options))
    return failure();
  detail::AnalysisResult result;
  result.pointer = components;
  impl->driver.bindBoundary({boundary, detail::PointerValue},
                            std::move(result));
  return success();
}

LogicalResult
PointerAnalysis::remapAndForgetScope(Operation *oldScope,
                                     const IRMapping &resultMapping) {
  return impl->driver.remapAndForgetScope(oldScope, resultMapping);
}

void PointerAnalysis::clear() { impl->driver.clear(); }

void eraseDeadPointerArithmetic(Operation *scope, RewriterBase &rewriter) {
  if (!scope)
    return;
  SmallVector<Operation *> worklist;
  llvm::DenseSet<Operation *> erased;
  scope->walk([&](Operation *op) {
    if (op != scope && isNumericHelper(op))
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (erased.contains(op))
      continue;
    if (!op->getBlock() || !isNumericHelper(op) || !isOpTriviallyDead(op))
      continue;
    SmallVector<Operation *> producers;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (scope->isAncestor(def) && isNumericHelper(def))
          producers.push_back(def);
    erased.insert(op);
    rewriter.eraseOp(op);
    worklist.append(producers);
  }
}

} // namespace mlir::triton::pointer
