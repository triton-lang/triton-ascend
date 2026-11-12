// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::cfg {
namespace {
bool hasSafeConstantPowExponent(triton::ExternElementwiseOp call) {
  auto constant = call.getOperand(1).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto value = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!value || !value.isSplat())
    return false;
  double exponent = value.getSplatValue<APFloat>().convertToDouble();
  // SIMT pow casts abs(exponent) to signed i32. Both endpoints are
  // excluded: abs(INT32_MIN) is not representable. NaN/Inf fail the same
  // comparisons. Keep the original literal; do not rewrite pow's math.
  return exponent > -2147483648.0 && exponent < 2147483648.0;
}

BatchInactivePolicy inactivePolicy(Operation *op,
                                   const BatchOpDescriptor &descriptor,
                                   unsigned operand) {
  // The registered f32 tanh entry lowers to floating min/max, exp and div.
  // None has immediate integer UB on an inactive lane, including NaN/Inf.
  // pow's base also enters only floating arithmetic or bit reinterpretation;
  // its exponent retains sanitization unless its literal is cast-safe.
  // Unknown external signatures retain the conservative descriptor policy.
  if (auto call = dyn_cast<triton::ExternElementwiseOp>(op))
    if ((call.getSymbol() == "__hmf_tanhf" ||
         (call.getSymbol() == "__hmf_pow_fp32" &&
          (operand == 0 || hasSafeConstantPowExponent(call)))) &&
        call.getPure() && call.getLibname().empty() &&
        call.getLibpath().empty())
      return BatchInactivePolicy::IEEE;
  return descriptor.inactiveInputs;
}
} // namespace
bool needsBatchActiveMask(Operation *op, const BatchOpDescriptor &descriptor) {
  for (unsigned index = 0; index < op->getNumOperands(); ++index) {
    auto policy = inactivePolicy(op, descriptor, index);
    if (policy != BatchInactivePolicy::Preserve &&
        policy != BatchInactivePolicy::IEEE)
      return true;
  }
  return false;
}

SmallVector<Value> prepareBatchElementwiseOperands(
    Operation *op, const BatchOpDescriptor &descriptor, ValueRange operands,
    Value activeMask, Location loc, IRRewriter &rewriter) {
  SmallVector<Value> result(operands.begin(), operands.end());
  if (!needsBatchActiveMask(op, descriptor))
    return result;
  assert(activeMask && "poison-sensitive operands require a live mask");
  for (auto [index, operand] : llvm::enumerate(operands)) {
    auto policy = inactivePolicy(op, descriptor, index);
    if (policy == BatchInactivePolicy::Preserve ||
        policy == BatchInactivePolicy::IEEE)
      continue;
    int64_t fill = 0;
    if (policy == BatchInactivePolicy::Positive ||
        (policy == BatchInactivePolicy::Divide && index == 1))
      fill = 1;
    auto type = cast<RankedTensorType>(operand.getType());
    Type element = type.getElementType();
    TypedAttr value;
    if (isa<IntegerType>(element))
      value = rewriter.getIntegerAttr(element, fill);
    else
      value = rewriter.getFloatAttr(element, static_cast<double>(fill));
    Value safe = rewriter.create<arith::ConstantOp>(
        loc, type, DenseElementsAttr::get(type, value));
    // A select of the RESULT would be too late for integer division, shifts,
    // or float-to-int conversion: sanitize the operands before evaluation.
    auto select =
        rewriter.create<arith::SelectOp>(loc, activeMask, operand, safe);
    select->setAttr(kBatchSafeInputAttr, rewriter.getUnitAttr());
    result[index] = select;
  }
  return result;
}
Operation *rewriteBatchElementwise(Operation *op,
                                   const BatchOpDescriptor &descriptor,
                                   ValueRange operands,
                                   RankedTensorType resultType, Location loc,
                                   IRRewriter &rewriter) {
  if (descriptor.name == "arith.ceildivui") {
    // q + (r != 0) is exact throughout the original unsigned domain. It
    // cannot overflow: b == 1 implies r == 0; b > 1 leaves room for +1.
    Value quotient =
        rewriter.create<arith::DivUIOp>(loc, operands[0], operands[1]);
    Value remainder =
        rewriter.create<arith::RemUIOp>(loc, operands[0], operands[1]);
    TypedAttr zero = rewriter.getIntegerAttr(resultType.getElementType(), 0);
    Value zeros = rewriter.create<arith::ConstantOp>(
        loc, resultType, DenseElementsAttr::get(resultType, zero));
    Value nonzero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, remainder, zeros);
    Value increment = rewriter.create<arith::ExtUIOp>(loc, resultType, nonzero);
    return rewriter.create<arith::AddIOp>(loc, quotient, increment)
        .getOperation();
  }
  IRMapping mapping;
  for (auto [original, value] : llvm::zip(op->getOperands(), operands))
    mapping.map(original, value);
  Operation *result = rewriter.clone(*op, mapping);
  result->getResult(0).setType(resultType);
  return result;
}
} // namespace mlir::triton::cfg
