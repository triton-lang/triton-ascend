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

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {
StringRef stringifyEngineType(EngineType engineType) {
  switch (engineType) {
  case EngineType::CUBE:
    return "CUBE";
  case EngineType::VECTOR:
    return "VECTOR";
  }
  llvm_unreachable("unknown CV-split engine type");
}
} // namespace

void setOpEngineTypeAttr(Operation *op, EngineType engineType) {
  if (!op)
    return;
  StringRef coreType = stringifyEngineType(engineType);
  op->setAttr(CVPipeline::kCoreType,
              StringAttr::get(op->getContext(), coreType));
}

LogicalResult runDCVPClassifier(ModuleOp module) {
  PassManager pm(module.getContext(), module.getOperationName());
  pm.addPass(createOpClassifierPass());

  if (failed(pm.run(module)))
    return failure();
  return success();
}

FailureOr<Classification> readDCVPClassification(Block *body) {
  if (!body)
    return failure();
  Classification classification;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;

    switch (CVPipeline::getOpCoreType(&op)) {
    case CVPipeline::CUBE_ONLY:
      classification[&op] = EngineType::CUBE;
      break;
    case CVPipeline::VECTOR_ONLY:
      classification[&op] = EngineType::VECTOR;
      break;
    default:
      op.emitError("DCVP did not produce a single-core classification");
      return failure();
    }
  }

  return classification;
}

bool checkCoreClassifications(Block *body,
                              const Classification &classification) {
  if (!body)
    return false;
  int nCube = 0;
  int nVector = 0;
  for (const auto &entry : classification) {
    switch (entry.second) {
    case EngineType::CUBE:
      ++nCube;
      break;
    case EngineType::VECTOR:
      ++nVector;
      break;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] Classification: " << nCube << "C " << nVector
                 << "V\n";
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(op))
        continue;

      auto it = classification.find(&op);
      llvm::dbgs() << "[cv-split]   ";
      if (it == classification.end())
        llvm::dbgs() << "??";
      else
        llvm::dbgs() << stringifyEngineType(it->second);
      llvm::dbgs() << " " << op.getName() << "\n";
    }
  });

  return nCube > 0 && nVector > 0;
}

void removeDCVPClassificationAttrs(ModuleOp module) {
  module.walk([](Operation *op) { op->removeAttr(CVPipeline::kCoreType); });
}

} // namespace mlir::triton::cv_split
