// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::cfg {
namespace {
const BatchOpDescriptor descriptors[] = {
#define BATCH_OP(name, spec, model, attrs, inactive, enabled)                  \
  {name,                                                                       \
   #spec[0],                                                                   \
   BatchModel::model,                                                          \
   BatchAttributePolicy::attrs,                                                \
   BatchInactivePolicy::inactive,                                              \
   enabled},
#include "TritonToGraph/BlockBatching/BatchOps.def"
#undef BATCH_OP
};
static_assert(sizeof(descriptors) / sizeof(descriptors[0]) == 185);

bool allowsAttribute(BatchAttributePolicy policy, llvm::StringRef name) {
  switch (policy) {
  case BatchAttributePolicy::None:
    return false;
  case BatchAttributePolicy::Constant:
    return name == "value";
  case BatchAttributePolicy::Axis:
    return name == "axis";
  case BatchAttributePolicy::Range:
    return name == "start" || name == "end";
  case BatchAttributePolicy::Memory:
    return name == "cache" || name == "evict" || name == "isVolatile" ||
           name == "operandSegmentSizes" || name == "boundaryCheck" ||
           name == "padding" || name == "was_bool_to_int8";
  case BatchAttributePolicy::Rounding:
    return name == "rounding" || name == "fp_downcast_rounding";
  case BatchAttributePolicy::Clamp:
    return name == "propagateNan";
  case BatchAttributePolicy::Compare:
    return name == "predicate" || name == "fastmath";
  case BatchAttributePolicy::Overflow:
    return name == "overflowFlags";
  case BatchAttributePolicy::FastMath:
    return name == "fastmath" || name == "roundingmode";
  case BatchAttributePolicy::Reshape:
    return name == "allow_reorder" || name == "efficient_layout";
  case BatchAttributePolicy::Transpose:
    return name == "order";
  case BatchAttributePolicy::Scan:
    return name == "axis" || name == "reverse";
  case BatchAttributePolicy::External:
    return name == "symbol" || name == "libname" || name == "libpath" ||
           name == "pure";
  case BatchAttributePolicy::Dot:
    return name == "inputPrecision" || name == "maxNumImpreciseAcc";
  }
  llvm_unreachable("unknown batch attribute policy");
}
} // namespace

llvm::ArrayRef<BatchOpDescriptor> getBatchOpRegistry() { return descriptors; }
const BatchOpDescriptor *lookupBatchOp(llvm::StringRef name) {
  for (const auto &descriptor : descriptors)
    if (descriptor.name == name)
      return &descriptor;
  return nullptr;
}
bool isBatchOpSupported(llvm::StringRef name) {
  const auto *descriptor = lookupBatchOp(name);
  return descriptor && descriptor->implemented;
}
llvm::StringRef getBatchModelName(BatchModel model) {
  switch (model) {
#define MODEL(value)                                                           \
  case BatchModel::value:                                                      \
    return #value;
    MODEL(Domain)
    MODEL(Constant)
    MODEL(Elementwise)
    MODEL(Shape)
    MODEL(Memory)
    MODEL(Reduction)
    MODEL(Scan)
    MODEL(ControlFlow)
    MODEL(IndexCast)
    MODEL(Structural) MODEL(Deferred) MODEL(Rejected) MODEL(Dot)
#undef MODEL
  }
  llvm_unreachable("unknown batch model");
}
llvm::StringRef getBatchAttributePolicyName(BatchAttributePolicy policy) {
  switch (policy) {
#define POLICY(value)                                                          \
  case BatchAttributePolicy::value:                                            \
    return #value;
    POLICY(None)
    POLICY(Constant)
    POLICY(Axis)
    POLICY(Range)
    POLICY(Memory)
    POLICY(Rounding)
    POLICY(Clamp)
    POLICY(Compare)
    POLICY(Overflow)
    POLICY(FastMath)
    POLICY(Reshape) POLICY(Transpose) POLICY(Scan) POLICY(External) POLICY(Dot)
#undef POLICY
  }
  llvm_unreachable("unknown batch attribute policy");
}
llvm::StringRef getBatchInactivePolicyName(BatchInactivePolicy policy) {
  switch (policy) {
#define POLICY(value)                                                          \
  case BatchInactivePolicy::value:                                             \
    return #value;
    POLICY(Preserve)
    POLICY(IEEE)
    POLICY(Zero)
    POLICY(Positive) POLICY(Divide) POLICY(Shift) POLICY(FloatToInt)
#undef POLICY
  }
  llvm_unreachable("unknown batch inactive-input policy");
}
LogicalResult verifyBatchAttributes(Operation *op,
                                    const BatchOpDescriptor &descriptor) {
  if (descriptor.attributes == BatchAttributePolicy::External) {
    auto call = dyn_cast<triton::ExternElementwiseOp>(op);
    if (!call || !call.getPure() || !call.getLibname().empty() ||
        !call.getLibpath().empty())
      return failure();
    auto symbol = call.getSymbol();
    unsigned arity = symbol == "__hmf_pow_fp32" ? 2
                     : symbol == "__hmf_tanhf"  ? 1
                                                : 0;
    if (!arity || call.getSrcs().size() != arity)
      return failure();
    auto f32 = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        type = tensor.getElementType();
      return type.isF32();
    };
    if (!f32(call.getType()))
      return failure();
    for (Value operand : call.getSrcs())
      if (!f32(operand.getType()))
        return failure();
  }
  if (op->hasAttr("was_bool_to_int8") && !isBatchNormalizedBoolLoad(op))
    return failure();
  for (NamedAttribute attr : op->getAttrs()) {
    // This lowering-only hint does not change select semantics. Rewritten IR
    // is checked again, so recognize the exact marker on selects only.
    if (attr.getName() == kBatchSafeInputAttr && isa<arith::SelectOp>(op) &&
        isa<UnitAttr>(attr.getValue()))
      continue;
    if (!allowsAttribute(descriptor.attributes, attr.getName().strref()))
      return failure();
    // Non-default fast-math/overflow promises need their own proof of the new
    // execution domain. Do not silently drop or expand these promises.
    if (attr.getName() == "fastmath") {
      auto flags = dyn_cast<arith::FastMathFlagsAttr>(attr.getValue());
      if (!flags || flags.getValue() != arith::FastMathFlags::none)
        return failure();
    }
    if (attr.getName() == "overflowFlags") {
      auto flags = dyn_cast<arith::IntegerOverflowFlagsAttr>(attr.getValue());
      if (!flags || flags.getValue() != arith::IntegerOverflowFlags::none)
        return failure();
    }
    // Rounding on extf/truncf is not covered by the first scalar model. The
    // Triton fp_to_fp model retains its registered rounding attribute.
    if (attr.getName() == "roundingmode")
      return failure();
  }
  return success();
}
} // namespace mlir::triton::cfg
