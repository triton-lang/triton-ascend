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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::cv_split {

enum class EngineType { CUBE, VECTOR };
/// Maps operations to their assigned execution engine. Operation pointers are
/// non-owning handles: they remain pointer-valid when moved, but the mapping's
/// block-level meaning must be revalidated after a move and pointers become
/// dangling when their operations are erased.
using Classification = llvm::DenseMap<Operation *, EngineType>;

/// Stamps a newly-created operation with the same core ownership attribute
/// emitted by DynamicCVPipeline's classifier.
void setOpEngineTypeAttr(Operation *op, EngineType engineType);

/// Runs DynamicCVPipeline's operation classifier once on `module`.
LogicalResult runDCVPClassifier(ModuleOp module);

/// Imports the DCVP classifications already stamped on operations directly
/// contained in `body`.
FailureOr<Classification> readDCVPClassification(Block *body);

/// Logs the candidate body's classifications and returns true when both the
/// CUBE and VECTOR subcores have work.
bool checkCoreClassifications(Block *body, const Classification &classification);

/// Removes the temporary core ownership attribute emitted by DCVP.
void removeDCVPClassificationAttrs(ModuleOp module);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
