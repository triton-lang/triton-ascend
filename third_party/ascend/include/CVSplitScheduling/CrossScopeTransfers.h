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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

struct VectorToCubeTransferChain {
    Value pSrc;
    Value l1Alloc;
    Operation *anchor;
    SmallVector<Operation *> operationsToErase;
};

struct CrossScopeTransferInfo {
    int64_t blockM;
    SmallVector<VectorToCubeTransferChain> vectorToCubeChains;
};

FailureOr<CrossScopeTransferInfo> insertCrossScopeTransfers(scf::ForOp loop, const Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
