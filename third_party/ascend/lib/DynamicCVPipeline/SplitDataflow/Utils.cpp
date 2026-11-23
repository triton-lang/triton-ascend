
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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;

void triton::setOpBlockId(mlir::Operation *op, int blockId) {
  static constexpr int kIntegerBitWidth = 32;
  op->setAttr(
      CVPipeline::kBlockId,
      IntegerAttr::get(IntegerType::get(op->getContext(), kIntegerBitWidth),
                       blockId));
}

int triton::getOpLoopDepth(mlir::Operation *op) {
  if (auto loopDepthAttr =
          op->getAttrOfType<IntegerAttr>(CVPipeline::kLoopDepthAttr)) {
    return loopDepthAttr.getInt();
  }
  return 0;
}

void triton::setOpLoopDepth(mlir::Operation *op) {
  static constexpr int kIntegerBitWidth = 32;
  int loopDepth = 1;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      ++loopDepth;
    }
  }
  op->setAttr(
      CVPipeline::kLoopDepthAttr,
      IntegerAttr::get(IntegerType::get(op->getContext(), kIntegerBitWidth),
                       loopDepth));
}

void triton::setOpCoreType(mlir::Operation *op, llvm::StringRef coreType) {
  op->setAttr(CVPipeline::kCoreType,
              StringAttr::get(op->getContext(), coreType));
}

bool triton::isScalarDependency(mlir::Value depValue) {
  return isa<FloatType, IntegerType>(depValue.getType());
}

bool triton::is1DTensorDependency(mlir::Value depValue) {
  static constexpr int SHAPE_1D_LENGTH = 1;
  auto tensorTy = dyn_cast<TensorType>(depValue.getType());
  if (tensorTy && tensorTy.getRank() == SHAPE_1D_LENGTH) {
    return true;
  }
  return false;
}

bool triton::isOutlineReuseBlacklistKernel(ModuleOp module) {
  return false;
  std::vector<std::string> blacklist{
      // "kernel_da_fwd_u",
      // "chunk_kda_bwd_kernel_intra",
      // "flex_attention_kernel",
  };

  bool isBlacklistKernel = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (llvm::is_contained(blacklist, func.getName().str())) {
      isBlacklistKernel = true;
      break;
    }
  }
  return isBlacklistKernel;
}

// Helper: ssbuffer.core_type
llvm::StringRef triton::getSsbufferCoreType(Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(CVPipeline::kCoreType)) {
    return attr.getValue();
  }
  return "";
}

// Helper: Get CoreType from op and index
llvm::StringRef triton::getCoreTypeWithIndex(Operation *op, int index) {
  llvm::StringRef typeStr = getSsbufferCoreType(op);
  if (typeStr.contains(", ")) {
    llvm::SmallVector<llvm::StringRef> types;
    typeStr.split(types, ", ", -1, false);
    if (index < types.size()) {
      return types[index].trim();
    }
    // LOG_DEBUG("Warning: Core type string has multiple types but value is not
    // "
    //           "an OpResult or index out of range.\n");
    return "";
  }

  return typeStr;
}
