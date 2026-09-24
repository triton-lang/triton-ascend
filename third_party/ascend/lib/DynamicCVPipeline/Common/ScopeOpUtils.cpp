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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h"

using namespace mlir;
using namespace CVPipeline;

namespace mlir::CVPipeline {

scope::ScopeOp packScopeOp(llvm::ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return nullptr;
  }
  Operation *insertionPoint = ops.back();

  DenseSet<Operation *> allOps;
  for (auto *op : ops) {
    op->walk([&](Operation *subOp) { allOps.insert(subOp); });
  }

  llvm::SetVector<Value> escapedValues;
  for (auto *op : ops) {
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (allOps.contains(user)) {
          continue;
        }
        escapedValues.insert(result);
        break;
      }
    }
  }

  OpBuilder builder(insertionPoint);
  builder.setInsertionPoint(insertionPoint);
  Location loc = insertionPoint->getLoc();
  ValueRange redirectValues(escapedValues.getArrayRef());
  TypeRange types = redirectValues.getTypes();
  auto scopeOp = builder.create<scope::ScopeOp>(loc, types);
  for (auto [origVal, scopeRes] :
       llvm::zip(redirectValues, scopeOp->getResults())) {
    origVal.replaceUsesWithIf(scopeRes, [&](OpOperand &operand) {
      return !allOps.contains(operand.getOwner());
    });
  }

  auto *block = &scopeOp.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToEnd(block);
  auto returnOp =
      builder.create<scope::ReturnOp>(loc, escapedValues.getArrayRef());
  for (auto *op : ops) {
    op->moveBefore(returnOp);
  }
  return scopeOp;
}

llvm::LogicalResult
unpackScopeOp(scope::ScopeOp scopeOp,
              llvm::SmallVectorImpl<Operation *> *movedOps) {
  auto *block = scopeOp.getBody();
  if (!block) {
    return llvm::success();
  }
  for (auto &op : llvm::make_early_inc_range(block->without_terminator())) {
    op.moveBefore(scopeOp);
    if (movedOps) {
      movedOps->push_back(&op);
    }
  }
  auto *term = block->getTerminator();
  if (!term) {
    return llvm::failure();
  }
  scopeOp.replaceAllUsesWith(term->getOperands());
  if (!scopeOp->getUses().empty()) {
    return llvm::failure();
  }
  scopeOp->erase();
  return llvm::success();
}

} // namespace mlir::CVPipeline
