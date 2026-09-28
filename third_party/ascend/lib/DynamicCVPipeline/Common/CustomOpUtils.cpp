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

#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Analysis.h"

static constexpr const char *DEBUG_TYPE = "custom-op-utils";
#define DBGS(...) LLVM_DEBUG(llvm::dbgs() << __VA_ARGS__)
#define LOG_DEBUG(...) DBGS("[" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace CVPipeline;

void CustomOpAnalysis::determineCoreType() {
  auto hivmCoreTypeOpt = customOp.getCoreType();
  if (hivmCoreTypeOpt.has_value()) {
    switch (hivmCoreTypeOpt.value()) {
    case hivm::TCoreType::CUBE:
      coreType = CUBE_ONLY;
      break;
    case hivm::TCoreType::VECTOR:
      coreType = VECTOR_ONLY;
      break;
    case hivm::TCoreType::CUBE_AND_VECTOR:
    case hivm::TCoreType::CUBE_OR_VECTOR:
      coreType = CUBE_AND_VECTOR;
    }
  }
}

llvm::LogicalResult CustomOpAnalysis::collectBuffers() {
  // Instead of using Dps, the following logic handles more scenarios
  for (auto operand : customOp->getOperands()) {
    auto type = operand.getType();
    LOG_DEBUG("Trying to collect: " << operand << "\n");
    if (!llvm::isa<TensorType, MemRefType>(type)) {
      continue;
    }
    Value sourceMemref = traceMemDef(operand);
    LOG_DEBUG("Source: " << sourceMemref << "\n");
    if (!sourceMemref) {
      LOG_DEBUG("No source memref: " << operand << "\n");
      continue;
    }
    if (auto barg = llvm::dyn_cast<BlockArgument>(sourceMemref)) {
      if (!llvm::dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
        LOG_DEBUG("Found memref in block args but is not function arg: "
                  << *barg.getOwner()->getParentOp() << "\n");
        return llvm::failure();
      }
      LOG_DEBUG("GM buffer: " << operand << "\n");
      gmBuffers.push_back(barg);
    } else if (isa<memref::AllocOp>(sourceMemref.getDefiningOp())) {
      LOG_DEBUG("Local buffer: " << operand << "\n");
      localBuffers.push_back(sourceMemref);
    }
  }
  return llvm::success();
}

void CustomOpAnalysis::collectRelaventOps() {
  for (auto buf : localBuffers) {
    bufferization::ToTensorOp toTensorOp;
    if (buf.hasOneUse()) {
      // pattern: alloc->to_tensor->customOp
      toTensorOp =
          llvm::dyn_cast<bufferization::ToTensorOp>(*buf.getUsers().begin());
      if (!toTensorOp) {
        continue;
      }
      auto tensorRes = toTensorOp.getResult();
      if (!tensorRes.hasOneUse() || *tensorRes.getUsers().begin() != customOp) {
        continue;
      }
      relaventOps.push_back(buf.getDefiningOp());
      relaventOps.push_back(toTensorOp);
      relaventOps.push_back(customOp);
    } else {
      // pattern: alloc -- filled by customOp -> to_tensor
      for (auto *user : buf.getUsers()) {
        if (user == customOp) {
          continue;
        }
        auto currToTensorOp = llvm::dyn_cast<bufferization::ToTensorOp>(user);
        if (!currToTensorOp || toTensorOp) {
          // 1. not to_tensor or customOp
          // 2. or has another to_tensor
          // skip this buffer
          toTensorOp = nullptr;
          break;
        }
        toTensorOp = currToTensorOp;
      }
      if (!toTensorOp) {
        continue;
      }
      relaventOps.push_back(buf.getDefiningOp());
      relaventOps.push_back(customOp);
      relaventOps.push_back(toTensorOp);
    }
    if (!toTensorOp->hasOneUse()) {
      continue;
    }
    if (auto convertLayout = llvm::dyn_cast<hivm::ConvertLayoutOp>(
            *toTensorOp->getUsers().begin())) {
      relaventOps.push_back(convertLayout);
    }
  }
}

void CustomOpAnalysis::determineLoadOrUnknown() {
  PipeAnalysis pipeAna{pipe};
  if (pipeAna.from.contains(hivm::AddressSpace::GM)) {
    isLoad = true;
  } else if (pipeAna.to.contains(hivm::AddressSpace::GM)) {
    isLoad = false;
  } else {
    isUnknownPipe = true;
  }
}

FailureOr<CustomOpAnalysis> CustomOpAnalysis::get(hivm::CustomOp customOp) {
  LOG_DEBUG("Analysing: " << customOp << "\n");
  CustomOpAnalysis result(customOp);

  result.determineCoreType();
  if (result.collectBuffers().failed()) {
    LOG_DEBUG("Failed to analyze: " << customOp << "\n");
    return llvm::failure();
  }
  result.collectRelaventOps();
  result.determineLoadOrUnknown();
  return result;
}
