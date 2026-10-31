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

#include "TritonMemoryAccess/PtrAnalysisRules/ControlFlowPtrRules.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
using namespace mlir::triton;
namespace mlir::triton::ptr {
namespace {
namespace pd = detail;
using Result = ControlFlowPtrRules::Result;
Type addressType(Type type, OpBuilder &b) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(b.getI64Type());
  return b.getI64Type();
}
Type componentType(OpFoldResult value) {
  if (auto v = dyn_cast<Value>(value))
    return v.getType();
  return cast<TypedAttr>(cast<Attribute>(value)).getType();
}
ComponentInfo fromValue(Value value, AnalysisOptions options) {
  ComponentInfo info{value.getType(), {value, 0}, OpFoldResult(value), {}};
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    auto width = pd::getIntegerWidth(value.getType(), options);
    if (succeeded(width)) {
      auto constant = pd::getIntegerConstant(attr, *width);
      if (succeeded(constant) && constant->isZero())
        info.identity = {};
    }
  }
  return info;
}
struct SymbolicArithmetic {
  Value owner;
  AnalysisOptions options;
  OpBuilder builder;
  unsigned nextField = 1;
  SymbolicArithmetic(Value owner, AnalysisOptions options)
      : owner(owner), options(options), builder(owner.getContext()) {}
  ComponentInfo constant(Type type, int64_t value) {
    auto scalar = pd::getIntegerElementType(type);
    auto width = pd::getIntegerWidth(type, options);
    assert(succeeded(width));
    auto attr = pd::scalarIntegerAttr(scalar, llvm::APInt(*width, value, true),
                                      builder);
    OpFoldResult concrete = attr;
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      concrete = DenseElementsAttr::get(tensor, attr);
    return {type,
            value == 0 ? ComponentIdentity{}
                       : ComponentIdentity{owner, nextField++},
            concrete,
            {}};
  }
  FailureOr<llvm::APInt> constantValue(const ComponentInfo &info) {
    auto width = pd::getIntegerWidth(info.type, options);
    if (failed(width) || !info.concrete)
      return failure();
    return pd::getIntegerConstant(*info.concrete, *width);
  }
  ComponentInfo expression(Type type, ComponentExpression::Kind kind,
                           ArrayRef<ComponentInfo> inputs, unsigned axis = 0) {
    auto recipe = std::make_shared<ComponentExpression>();
    recipe->kind = kind;
    recipe->inputs.assign(inputs.begin(), inputs.end());
    recipe->axis = axis;
    return {type, {owner, nextField++}, std::nullopt, std::move(recipe)};
  }
  ComponentInfo shape(const ComponentInfo &source, Type type,
                      ComponentExpression::Kind kind, unsigned axis = 0) {
    if (source.type == type)
      return source;
    if (source.isZero())
      return constant(type, 0);
    return expression(type, kind, {source}, axis);
  }
  ComponentInfo cast(const ComponentInfo &source, Type type,
                     bool isUnsigned = false) {
    if (source.type == type)
      return source;
    if (source.isZero())
      return constant(type, 0);
    auto number = constantValue(source);
    auto width = pd::getIntegerWidth(type, options);
    if (succeeded(number) && succeeded(width)) {
      auto value = isUnsigned ? number->zextOrTrunc(*width)
                              : number->sextOrTrunc(*width);
      auto info = constant(type, 0);
      auto attr = pd::scalarIntegerAttr(pd::getIntegerElementType(type), value,
                                        builder);
      info.concrete = isa<RankedTensorType>(type)
                          ? OpFoldResult(DenseElementsAttr::get(
                                mlir::cast<RankedTensorType>(type), attr))
                          : OpFoldResult(attr);
      if (!value.isZero())
        info.identity = {owner, nextField++};
      return info;
    }
    return expression(type,
                      isUnsigned ? ComponentExpression::UnsignedCast
                                 : ComponentExpression::SignedCast,
                      {source});
  }
  FailureOr<ComponentInfo> binary(ComponentInfo lhs, ComponentInfo rhs,
                                  ComponentExpression::Kind kind) {
    Type type = isa<RankedTensorType>(lhs.type) ? lhs.type : rhs.type;
    if (lhs.type != type)
      lhs = shape(lhs, type, ComponentExpression::Splat);
    if (rhs.type != type)
      rhs = shape(rhs, type, ComponentExpression::Splat);
    if (lhs.type != rhs.type)
      return failure();
    if (kind == ComponentExpression::Add ||
        kind == ComponentExpression::Subtract) {
      if (rhs.isZero())
        return lhs;
      if (kind == ComponentExpression::Add && lhs.isZero())
        return rhs;
    } else if (lhs.isZero() || rhs.isZero())
      return constant(type, 0);
    auto l = constantValue(lhs), r = constantValue(rhs);
    if (kind == ComponentExpression::Multiply) {
      if (succeeded(l) && l->isOne())
        return rhs;
      if (succeeded(r) && r->isOne())
        return lhs;
    }
    if (succeeded(l) && succeeded(r)) {
      auto value = kind == ComponentExpression::Add        ? *l + *r
                   : kind == ComponentExpression::Subtract ? *l - *r
                                                           : *l * *r;
      auto info = constant(type, 0);
      auto attr = pd::scalarIntegerAttr(pd::getIntegerElementType(type), value,
                                        builder);
      info.concrete = isa<RankedTensorType>(type)
                          ? OpFoldResult(DenseElementsAttr::get(
                                mlir::cast<RankedTensorType>(type), attr))
                          : OpFoldResult(attr);
      if (!value.isZero())
        info.identity = {owner, nextField++};
      return info;
    }
    return expression(type, kind, {lhs, rhs});
  }
  FailureOr<ComponentInfo> add(const ComponentInfo &lhs,
                               const ComponentInfo &rhs, bool sub) {
    return binary(lhs, rhs,
                  sub ? ComponentExpression::Subtract
                      : ComponentExpression::Add);
  }
  FailureOr<ComponentInfo> multiply(const ComponentInfo &lhs,
                                    const ComponentInfo &rhs) {
    return binary(lhs, rhs, ComponentExpression::Multiply);
  }
};
SymbolicOffsetInfo opaqueOffset(ComponentInfo full, SymbolicArithmetic &math) {
  SymbolicOffsetInfo info;
  info.valueType = full.type;
  info.shape = pd::getLogicalShape(full.type);
  info.fullOffsets = full;
  auto zero = math.constant(pd::getIntegerElementType(full.type), 0);
  bool scalar = !isa<RankedTensorType>(full.type);
  info.terms.origin = scalar ? full : zero;
  info.terms.coefficients.assign(info.shape.size(), zero);
  info.terms.irregular = scalar ? math.constant(full.type, 0) : full;
  return info;
}
SymbolicOffsetInfo zeroOffset(Type type, SymbolicArithmetic &math) {
  return opaqueOffset(math.constant(type, 0), math);
}
bool decompositionFits(const SymbolicOffsetInfo &source, unsigned width,
                       bool isUnsigned) {
  if (!source.terms.irregular->isZero() || !source.terms.origin->concrete)
    return false;
  OffsetInfo concrete;
  concrete.valueType = source.valueType;
  concrete.shape = source.shape;
  concrete.scalarOffset = *source.terms.origin->concrete;
  auto nonnegative = [&](OpFoldResult value) {
    auto number = pd::getIntegerConstant(value, width);
    return succeeded(number) && !number->isNegative();
  };
  if (isUnsigned && !nonnegative(*concrete.scalarOffset))
    return false;
  for (const auto &coefficient : source.terms.coefficients) {
    if (!coefficient->concrete ||
        (isUnsigned && !nonnegative(*coefficient->concrete)))
      return false;
    concrete.axes.emplace_back(AxisKind::Structured, *coefficient->concrete);
  }
  return pd::decompositionFitsSigned(concrete, width);
}
FailureOr<SymbolicOffsetInfo> castOffset(const SymbolicOffsetInfo &source,
                                         Type type, bool isUnsigned,
                                         SymbolicArithmetic &math) {
  auto sourceWidth = pd::getIntegerWidth(source.valueType, math.options);
  auto targetWidth = pd::getIntegerWidth(type, math.options);
  if (failed(sourceWidth) || failed(targetWidth))
    return failure();
  if (source.valueType == type)
    return source;
  auto full = math.cast(source.fullOffsets, type, isUnsigned);
  // Truncation is a ring homomorphism. Widening a nonuniform source can undo
  // modular cancellation, so retain the exact whole expression as residual.
  if (!source.isUniform() && *sourceWidth < *targetWidth &&
      !decompositionFits(source, *sourceWidth, isUnsigned))
    return opaqueOffset(full, math);
  auto info = source;
  info.valueType = type;
  info.fullOffsets = full;
  auto scalar = pd::getIntegerElementType(type);
  info.terms.origin = math.cast(*source.terms.origin, scalar, isUnsigned);
  for (auto &coefficient : info.terms.coefficients)
    coefficient = math.cast(*coefficient, scalar, false);
  info.terms.irregular = math.cast(*source.terms.irregular, type, isUnsigned);
  return info;
}
FailureOr<SymbolicOffsetInfo> addOffsets(const SymbolicOffsetInfo &lhs,
                                         const SymbolicOffsetInfo &rhs,
                                         bool subtract,
                                         SymbolicArithmetic &math) {
  auto terms = addOffsetTerms(lhs.terms, rhs.terms, subtract, math);
  auto full = math.add(lhs.fullOffsets, rhs.fullOffsets, subtract);
  if (failed(terms) || failed(full))
    return failure();
  auto info = lhs;
  info.fullOffsets = *full;
  info.terms = std::move(*terms);
  return info;
}
FailureOr<SymbolicOffsetInfo> shapeOffset(const SymbolicOffsetInfo &source,
                                          Type type,
                                          ComponentExpression::Kind kind,
                                          unsigned axis,
                                          SymbolicArithmetic &math) {
  auto info = source;
  info.valueType = type;
  info.shape = pd::getLogicalShape(type);
  info.fullOffsets = math.shape(source.fullOffsets, type, kind, axis);
  info.terms.irregular = math.shape(*source.terms.irregular, type, kind, axis);
  auto zero = math.constant(pd::getIntegerElementType(type), 0);
  if (kind == ComponentExpression::Splat)
    info.terms.coefficients.assign(info.shape.size(), zero);
  else {
    auto invariant = [&] { return std::optional<ComponentInfo>(zero); };
    auto axes =
        kind == ComponentExpression::ExpandDims
            ? expandAxisFacts(ArrayRef<std::optional<ComponentInfo>>(
                                  source.terms.coefficients),
                              axis, invariant)
            : broadcastAxisFacts(ArrayRef<std::optional<ComponentInfo>>(
                                     source.terms.coefficients),
                                 ArrayRef<int64_t>(source.shape),
                                 ArrayRef<int64_t>(info.shape), invariant);
    if (failed(axes))
      return failure();
    info.terms.coefficients = std::move(*axes);
  }
  return info;
}
// Validate each immutable recipe once, without copying the expression DAG.
struct ComponentValidator {
  llvm::DenseSet<const ComponentExpression *> checked, active;
  LogicalResult check(const ComponentInfo &info) {
    if (!info.type ||
        (info.concrete && componentType(*info.concrete) != info.type))
      return failure();
    if (info.isZero()) {
      if (info.identity.field || info.expression ||
          !pd::getIntegerElementType(info.type).isIntOrIndex())
        return failure();
      if (info.concrete) {
        auto zero = pd::getIntegerConstant(*info.concrete, 64);
        if (failed(zero) || !zero->isZero())
          return failure();
      }
    }
    if (!info.expression)
      return success();
    auto *recipe = info.expression.get();
    if (checked.contains(recipe))
      return success();
    if (!active.insert(recipe).second)
      return failure();
    auto restore = llvm::make_scope_exit([&] { active.erase(recipe); });
    auto kind = recipe->kind;
    if (kind < ComponentExpression::Add ||
        kind > ComponentExpression::ExpandDims ||
        recipe->inputs.size() !=
            (kind <= ComponentExpression::Multiply ? 2u : 1u))
      return failure();
    for (const auto &input : recipe->inputs)
      if (failed(check(input)))
        return failure();
    checked.insert(recipe);
    return success();
  }
  LogicalResult checkOffset(const SymbolicOffsetInfo &info,
                            bool hasFull = true) {
    if ((hasFull && failed(check(info.fullOffsets))) || !info.terms.origin ||
        !info.terms.irregular || failed(check(*info.terms.origin)) ||
        failed(check(*info.terms.irregular)))
      return failure();
    for (const auto &coefficient : info.terms.coefficients)
      if (!coefficient || failed(check(*coefficient)))
        return failure();
    return success();
  }
};
} // namespace
ComponentInfo ComponentInfo::boundary(Value source, unsigned field, Type type) {
  assert(source && type);
  return {type, {source, field}, std::nullopt, {}};
}
bool SymbolicOffsetInfo::isUniform() const {
  return terms.irregular && terms.irregular->isZero() &&
         llvm::all_of(terms.coefficients,
                      [](const auto &v) { return v && v->isZero(); });
}
FailureOr<Result> ControlFlowPtrRules::parseOperation(Value value,
                                                      Analyze analyze) const {
  if (options.addressBitWidth != 64 &&
      (pd::isOrdinaryPtrType(value.getType()) ||
       pd::isBlockPtrType(value.getType())))
    return Result{};
  // Check before dispatch: casts and shape rules can create zero components
  // without visiting a leaf, and index has no implicit host bit width.
  if (pd::getIntegerElementType(value.getType()).isIntOrIndex() &&
      failed(pd::getIntegerWidth(value.getType(), options)))
    return Result{};
  auto *op = value.getDefiningOp();
  if (op) {
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
      return parseArithmetic(value, analyze);
    if (isa<SplatOp, BroadcastOp, ExpandDimsOp>(op))
      return parseShape(value, analyze);
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>(op))
      return parseCast(value, analyze);
    if (auto add = dyn_cast<AddPtrOp>(op))
      return parseAddPtr(add, analyze);
    if (auto make = dyn_cast<MakeTensorPtrOp>(op))
      return parseMakeBlock(make, analyze);
    if (auto advance = dyn_cast<AdvanceOp>(op))
      return parseAdvance(advance, analyze);
  }
  SymbolicArithmetic math(value, options);
  if (pd::isOrdinaryPtrType(value.getType())) {
    auto offsets = zeroOffset(addressType(value.getType(), math.builder), math);
    return Result(SymbolicPtrInfo{value.getType(), fromValue(value, options),
                                  offsets, std::nullopt});
  }
  if (failed(pd::getIntegerWidth(value.getType(), options)))
    return Result{};
  auto info = opaqueOffset(fromValue(value, options), math);
  // A dense splat has one scalar contribution, not an irregular tensor.
  auto number = math.constantValue(info.fullOffsets);
  if (!info.shape.empty() && succeeded(number)) {
    auto attr = pd::scalarIntegerAttr(
        pd::getIntegerElementType(value.getType()), *number, math.builder);
    ComponentInfo origin{attr.getType(), {value, 1}, OpFoldResult(attr), {}};
    if (number->isZero())
      origin.identity = {};
    info.terms.origin = origin;
    info.terms.irregular = math.constant(value.getType(), 0);
  }

  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    info.terms.origin = math.constant(
        pd::getIntegerElementType(value.getType()), range.getStart());
    info.terms.coefficients[0] =
        math.constant(pd::getIntegerElementType(value.getType()), 1);
    info.terms.irregular = math.constant(value.getType(), 0);
  }
  return Result(info);
}
FailureOr<Result> ControlFlowPtrRules::parseArithmetic(Value value,
                                                       Analyze analyze) const {
  auto *op = value.getDefiningOp();
  auto lhsInfo = analyze(op->getOperand(0)),
       rhsInfo = analyze(op->getOperand(1));
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  auto *lhs = std::get_if<SymbolicOffsetInfo>(&*lhsInfo);
  auto *rhs = std::get_if<SymbolicOffsetInfo>(&*rhsInfo);
  if (!lhs || !rhs)
    return Result{};
  SymbolicArithmetic math(value, options);
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto info = addOffsets(*lhs, *rhs, isa<arith::SubIOp>(op), math);
    if (failed(info))
      return failure();
    return Result(*info);
  }
  auto *source = lhs;
  auto *scale = rhs;
  if (!scale->isUniform())
    std::swap(source, scale);
  if (!scale->isUniform())
    return Result(opaqueOffset(fromValue(value, options), math));
  auto terms = scaleOffsetTerms(source->terms, *scale->terms.origin, math);
  auto full = math.multiply(source->fullOffsets, scale->fullOffsets);
  if (failed(terms) || failed(full))
    return failure();
  auto info = *source;
  info.terms = *terms;
  info.fullOffsets = *full;
  return Result(info);
}
FailureOr<Result> ControlFlowPtrRules::parseShape(Value value,
                                                  Analyze analyze) const {
  auto *op = value.getDefiningOp();
  auto sourceInfo = analyze(op->getOperand(0));
  if (failed(sourceInfo))
    return failure();
  SymbolicArithmetic math(value, options);
  auto kind = isa<SplatOp>(op)       ? ComponentExpression::Splat
              : isa<BroadcastOp>(op) ? ComponentExpression::Broadcast
                                     : ComponentExpression::ExpandDims;
  unsigned axis = isa<ExpandDimsOp>(op) ? cast<ExpandDimsOp>(op).getAxis() : 0;
  if (auto *integer = std::get_if<SymbolicOffsetInfo>(&*sourceInfo)) {
    auto info = shapeOffset(*integer, value.getType(), kind, axis, math);
    if (failed(info))
      return failure();
    return Result(*info);
  }
  if (auto *ptr = std::get_if<SymbolicPtrInfo>(&*sourceInfo)) {
    if (ptr->block)
      return Result{};
    auto info = *ptr;
    info.ptrType = value.getType();
    auto offsets =
        shapeOffset(info.offsets, addressType(value.getType(), math.builder),
                    kind, axis, math);
    if (failed(offsets))
      return failure();
    info.offsets = *offsets;
    if (isa<RankedTensorType>(info.base.type))
      info.base = math.shape(info.base, value.getType(), kind, axis);
    return Result(info);
  }
  return Result{};
}
FailureOr<Result> ControlFlowPtrRules::parseCast(Value value,
                                                 Analyze analyze) const {
  auto *op = value.getDefiningOp();
  auto sourceInfo = analyze(op->getOperand(0));
  if (failed(sourceInfo))
    return failure();
  auto *source = std::get_if<SymbolicOffsetInfo>(&*sourceInfo);
  if (!source)
    return Result{};
  SymbolicArithmetic math(value, options);
  auto info = castOffset(*source, value.getType(),
                         isa<arith::ExtUIOp, arith::IndexCastUIOp>(op), math);
  return succeeded(info) ? Result(*info) : Result{};
}
FailureOr<Result> ControlFlowPtrRules::parseAddPtr(AddPtrOp op,
                                                   Analyze analyze) const {
  auto parentInfo = analyze(op.getPtr());
  auto sourceInfo = analyze(op.getOffset());
  if (failed(parentInfo) || failed(sourceInfo))
    return failure();
  auto *parent = std::get_if<SymbolicPtrInfo>(&*parentInfo);
  auto *source = std::get_if<SymbolicOffsetInfo>(&*sourceInfo);
  if (!parent || !source || parent->block ||
      (failed(pd::getIntegerWidth(source->valueType, options)) ||
       *pd::getIntegerWidth(source->valueType, options) > 64))
    return Result{};
  SymbolicArithmetic math(op.getResult(), options);
  auto delta = castOffset(*source, addressType(source->valueType, math.builder),
                          false, math);
  if (failed(delta))
    return Result{};
  auto offsets = addOffsets(parent->offsets, *delta, false, math);
  if (failed(offsets))
    return failure();
  auto info = *parent;
  info.offsets = *offsets;
  return Result(info);
}
FailureOr<Result> ControlFlowPtrRules::parseMakeBlock(MakeTensorPtrOp op,
                                                      Analyze analyze) const {
  auto baseInfo = analyze(op.getBase());
  if (failed(baseInfo))
    return failure();
  auto *base = std::get_if<SymbolicPtrInfo>(&*baseInfo);
  if (!base || base->block || !base->offsets.shape.empty())
    return Result{};
  SymbolicArithmetic math(op.getResult(), options);
  SymbolicBlockInfo block;
  for (Value coordinate : op.getOffsets()) {
    auto input = analyze(coordinate);
    if (failed(input))
      return failure();
    auto *integer = std::get_if<SymbolicOffsetInfo>(&*input);
    if (!integer || !integer->isUniform() ||
        (failed(pd::getIntegerWidth(integer->valueType, options)) ||
         *pd::getIntegerWidth(integer->valueType, options) > 64))
      return Result{};
    auto delta = castOffset(
        *integer, addressType(integer->valueType, math.builder), false, math);
    if (failed(delta))
      return Result{};
    block.coordinates.push_back(*delta->terms.origin);
  }
  for (Value v : op.getShape())
    block.parentShape.push_back(fromValue(v, options));
  for (Value v : op.getStrides())
    block.strides.push_back(
        math.cast(fromValue(v, options), math.builder.getI64Type()));
  block.order.assign(op.getOrder().begin(), op.getOrder().end());
  auto tensor =
      cast<RankedTensorType>(cast<PointerType>(op.getType()).getPointeeType());
  auto offsets = zeroOffset(tensor.clone(math.builder.getI64Type()), math);
  auto origin = addScaledCoordinateTerms(
      *base->offsets.terms.origin, ArrayRef<ComponentInfo>(block.coordinates),
      ArrayRef<ComponentInfo>(block.strides), math);
  if (failed(origin))
    return failure();
  offsets.terms.origin = *origin;
  offsets.terms.coefficients.assign(block.strides.begin(), block.strides.end());
  // Block full offsets are intentionally deferred; descriptor fields suffice.
  offsets.fullOffsets = {};
  return Result(SymbolicPtrInfo{op.getType(), base->base, offsets, block});
}
FailureOr<Result> ControlFlowPtrRules::parseAdvance(AdvanceOp op,
                                                    Analyze analyze) const {
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  auto *parent = std::get_if<SymbolicPtrInfo>(&*parentInfo);
  if (!parent || !parent->block)
    return Result{};
  SymbolicArithmetic math(op.getResult(), options);
  SmallVector<ComponentInfo> deltas;
  for (Value offset : op.getOffsets()) {
    auto sourceInfo = analyze(offset);
    if (failed(sourceInfo))
      return failure();
    auto *source = std::get_if<SymbolicOffsetInfo>(&*sourceInfo);
    if (!source || !source->isUniform() ||
        (failed(pd::getIntegerWidth(source->valueType, options)) ||
         *pd::getIntegerWidth(source->valueType, options) > 64))
      return Result{};
    auto delta = castOffset(
        *source, addressType(source->valueType, math.builder), false, math);
    if (failed(delta))
      return Result{};
    deltas.push_back(*delta->terms.origin);
  }
  auto coordinates = advanceCoordinateTerms(
      ArrayRef<ComponentInfo>(parent->block->coordinates),
      ArrayRef<ComponentInfo>(deltas), math);
  auto origin = addScaledCoordinateTerms(
      *parent->offsets.terms.origin, ArrayRef<ComponentInfo>(deltas),
      ArrayRef<ComponentInfo>(parent->block->strides), math);
  if (failed(coordinates) || failed(origin))
    return failure();
  auto info = *parent;
  info.block->coordinates = *coordinates;
  info.offsets.terms.origin = *origin;
  return Result(info);
}
LogicalResult ControlFlowPtrRules::validateBinding(Value value,
                                                   const Result &info) const {
  const auto *integer = std::get_if<SymbolicOffsetInfo>(&info);
  const auto *ptr = std::get_if<SymbolicPtrInfo>(&info);
  if (!integer && !ptr)
    return failure();
  if (integer && integer->valueType != value.getType())
    return failure();
  if (ptr && ptr->ptrType != value.getType())
    return failure();
  const auto &offsets = integer ? *integer : ptr->offsets;
  if (failed(pd::getIntegerWidth(offsets.valueType, options)))
    return failure();
  if (ptr) {
    if (options.addressBitWidth != 64 ||
        !pd::getIntegerElementType(offsets.valueType).isInteger(64))
      return failure();
    Type type = value.getType();
    if (ptr->block) {
      if (!pd::isBlockPtrType(type))
        return failure();
      type = cast<PointerType>(type).getPointeeType();
    } else if (!pd::isOrdinaryPtrType(type))
      return failure();
    OpBuilder builder(value.getContext());
    if (offsets.valueType != addressType(type, builder))
      return failure();
    Type baseType = ptr->base.type;
    if (auto tensor = dyn_cast<RankedTensorType>(baseType)) {
      if (ptr->block || baseType != value.getType())
        return failure();
      baseType = tensor.getElementType();
    }
    auto basePtr = dyn_cast<PointerType>(baseType);
    if (!basePtr || isa<ShapedType>(basePtr.getPointeeType()))
      return failure();
    auto expected =
        ptr->block
            ? cast<PointerType>(value.getType())
            : cast<PointerType>(pd::getIntegerElementType(value.getType()));
    Type element = ptr->block
                       ? pd::getIntegerElementType(expected.getPointeeType())
                       : expected.getPointeeType();
    if (basePtr.getPointeeType() != element ||
        basePtr.getAddressSpace() != expected.getAddressSpace())
      return failure();
  }

  if (offsets.shape != pd::getLogicalShape(offsets.valueType) ||
      offsets.terms.coefficients.size() != offsets.shape.size() ||
      !offsets.terms.origin || !offsets.terms.irregular)
    return failure();
  Type scalar = pd::getIntegerElementType(offsets.valueType);
  if (!scalar || offsets.terms.origin->type != scalar ||
      offsets.terms.irregular->type != offsets.valueType)
    return failure();
  for (const auto &coefficient : offsets.terms.coefficients)
    if (!coefficient || coefficient->type != scalar)
      return failure();
  if ((!ptr || !ptr->block) && offsets.fullOffsets.type != offsets.valueType)
    return failure();
  if (ptr && ptr->block) {
    const auto &block = *ptr->block;
    auto rank = offsets.shape.size();
    if (block.coordinates.size() != rank || block.strides.size() != rank ||
        block.parentShape.size() != rank || block.order.size() != rank)
      return failure();
    for (unsigned i = 0; i < rank; ++i)
      if (!block.coordinates[i].type.isInteger(64) ||
          !block.strides[i].type.isInteger(64) ||
          !block.parentShape[i].type.isIntOrIndex() ||
          !block.strides[i].sameIdentity(*offsets.terms.coefficients[i]))
        return failure();
    if (offsets.fullOffsets.type)
      return failure();
    SmallVector<bool> seen(rank, false);
    for (auto i : block.order) {
      if (i < 0 || i >= static_cast<int32_t>(rank) || seen[i])
        return failure();
      seen[i] = true;
    }
  }
  ComponentValidator validator;
  if (failed(validator.checkOffset(offsets, !ptr || !ptr->block)))
    return failure();
  if (ptr) {
    if (failed(validator.check(ptr->base)))
      return failure();
    if (ptr->block)
      for (const auto *fields :
           {&ptr->block->coordinates, &ptr->block->parentShape,
            &ptr->block->strides})
        for (const auto &field : *fields)
          if (failed(validator.check(field)))
            return failure();
  }
  return success();
}
FailureOr<OpFoldResult>
ComponentMaterializer::materialize(const ComponentInfo &info) {
  OpBuilder::InsertionGuard guard(builder);
  if (!anchor || !anchor.getDefiningOp())
    return failure();
  builder.setInsertionPoint(anchor.getDefiningOp());
  if (!info.type || (pd::getIntegerElementType(info.type).isIntOrIndex() &&
                     failed(pd::getIntegerWidth(info.type, options))))
    return failure();
  std::optional<OpFoldResult> leaf;
  if (resolver && !info.isZero())
    leaf = resolver(info.identity, info.type);
  if (!leaf && info.concrete)
    leaf = info.concrete;
  if (!leaf && info.isZero()) {
    if (failed(pd::getIntegerWidth(info.type, options)))
      return failure();
    SymbolicArithmetic math(anchor, options);
    leaf = math.constant(info.type, 0).concrete;
  }
  if (leaf) {
    if (componentType(*leaf) != info.type)
      return failure();
    SmallVector<Value> dependencies;
    if (auto value = dyn_cast<Value>(*leaf))
      dependencies.push_back(value);
    if (failed(checkInsertionPoint(builder, dependencies)))
      return failure();
    return *leaf;
  }
  if (!info.expression)
    return failure();
  auto found = cache.find(info.expression);
  if (found != cache.end())
    return found->second;
  SmallVector<OpFoldResult> inputs;
  for (const auto &input : info.expression->inputs) {
    auto value = materialize(input);
    if (failed(value))
      return failure();
    inputs.push_back(*value);
  }
  // Recursive recipes emit before this same anchor, so their results are
  // available here without walking every field of a symbolic result.
  auto kind = info.expression->kind;
  FailureOr<OpFoldResult> result = failure();
  if (kind <= ComponentExpression::Multiply)
    result = pd::foldOrCreateBinary(
        kind == ComponentExpression::Add        ? pd::BinaryKind::Add
        : kind == ComponentExpression::Subtract ? pd::BinaryKind::Subtract
                                                : pd::BinaryKind::Multiply,
        inputs[0], inputs[1], info.type, options, builder, anchor.getLoc());
  else if (kind == ComponentExpression::SignedCast ||
           kind == ComponentExpression::UnsignedCast)
    result = pd::foldOrCreateCast(inputs[0], info.type,
                                  kind == ComponentExpression::SignedCast
                                      ? pd::CastKind::Signed
                                      : pd::CastKind::Unsigned,
                                  options, builder, anchor.getLoc());
  else {
    auto input = pd::materialize(inputs[0], info.expression->inputs[0].type,
                                 builder, anchor.getLoc());
    if (failed(input))
      return failure();
    if (kind == ComponentExpression::Splat)
      result = OpFoldResult(
          builder.create<SplatOp>(anchor.getLoc(), info.type, *input)
              .getResult());
    else if (kind == ComponentExpression::Broadcast)
      result = OpFoldResult(
          builder.create<BroadcastOp>(anchor.getLoc(), info.type, *input)
              .getResult());
    else
      result =
          OpFoldResult(builder
                           .create<ExpandDimsOp>(anchor.getLoc(), info.type,
                                                 *input, info.expression->axis)
                           .getResult());
  }
  if (succeeded(result))
    cache.emplace(info.expression, *result);
  return result;
}
} // namespace mlir::triton::ptr
