//===- TritonAscendOps.cpp - TritonAscend dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>


using namespace mlir;
using namespace mlir::triton;


namespace mlir::triton::ascend {

void GatherOutToUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
{
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

void IndirectLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

//-- IndexSelectSimdOp --
LogicalResult IndexSelectSimdOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get operands using adaptor
  IndexSelectSimdOpAdaptor adaptor(operands, attributes, properties, regions);
  
  // Get element type from src pointer
  Type elemType;
  if (auto ptrType = dyn_cast<triton::PointerType>(adaptor.getSrc().getType())) {
    elemType = ptrType.getPointeeType();
  } else {
    return failure();
  }
  
  // Get index shape to determine the size of dim
  auto indicesType = dyn_cast<RankedTensorType>(adaptor.getIndex().getType());
  if (!indicesType)
    return failure();
  int64_t numIndices = indicesType.getShape()[0];
  
  // Use adaptor to get attributes - this is the compatible way
  int32_t dim = adaptor.getDim();
  auto readShapeAttr = adaptor.getReadShape();
  
  // Build result shape: read_shape but with dim replaced by numIndices
  SmallVector<int64_t> resultShape;
  for (size_t i = 0; i < readShapeAttr.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      resultShape.push_back(numIndices);
    } else {
      resultShape.push_back(readShapeAttr[i]);
    }
  }
  
  // Create result tensor type
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elemType));
  
  return success();
}

// FlipOp
LogicalResult FlipOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location, ValueRange operands,
                                       DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
                                       SmallVectorImpl<Type> &inferredReturnTypes)
{
    auto inputTy = dyn_cast<RankedTensorType>(operands[0].getType());
    if (!inputTy) {
        if (location)
            return emitOptionalError(location, "expected ranked tensor for flip input");
        return failure();
    }
    inferredReturnTypes.push_back(inputTy);
    return success();
}

//-- SortOp --
LogicalResult SortOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
    {
    if (operands.size() != 1) {
        return emitOptionalError(location, "expected exactly one operand for SortOp");
    }

    if (!isa<RankedTensorType>(operands[0].getType())) {
        return emitOptionalError(location, "operand must be a ranked tensor type for SortOp");
    }

    Value src = operands[0];
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcShape = srcTy.getShape();
    auto srcEnc = srcTy.getEncoding();

    if (srcShape.empty()) {
    return emitOptionalError(location, "input tensor must have rank >= 1");
    }

    Type sortedTy = RankedTensorType::get(srcShape, srcTy.getElementType(), srcEnc);

    inferredReturnTypes.push_back(sortedTy);

    return success();
}

} // namespace mlir::triton::ascend
