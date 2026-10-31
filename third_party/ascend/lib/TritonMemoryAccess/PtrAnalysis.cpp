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
#include "mlir/IR/Dominance.h"

namespace mlir::triton::ptr {

namespace {
LogicalResult checkInsertionPoint(OpBuilder::InsertPoint insertion,
                                  ArrayRef<Value> dependencies) {
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
} // namespace

LogicalResult checkInsertionPoint(OpBuilder &builder,
                                  ArrayRef<Value> dependencies) {
  return checkInsertionPoint(builder.saveInsertionPoint(), dependencies);
}

} // namespace mlir::triton::ptr
