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
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

/// Describes the IR emitted for one VECTOR-to-CUBE transfer. The values and
/// operation pointers are non-owning handles into the loop being transformed.
struct VectorToCubeTransferChain {
    /// VECTOR-produced tensor that must be packed for CUBE consumption.
    Value pSrc;
    /// Shared L1 destination allocated for the packed tensor.
    Value l1Alloc;
    /// `hivm::SyncBlockSetOp` before which the final copy is committed.
    Operation *anchor;
    /// Pack operations still owned by the IR. Scope separation must erase them
    /// in reverse order before rebuilding a per-vector-core pack.
    llvm::SmallVector<Operation *> operationsToErase;
};

/// Cross-scope transfer metadata consumed by scope separation.
struct CrossScopeTransferInfo {
    /// Full row count derived from the leading CUBE-to-VECTOR transfer
    /// dimension. Scope separation halves it to M/2 rows per vector core.
    int64_t blockM;
    llvm::SmallVector<VectorToCubeTransferChain> vectorToCubeChains;
};

/// Materializes CUBE-to-VECTOR and VECTOR-to-CUBE data movement and
/// synchronization for `loop`.
///
/// `classification` assigns each operation to its execution engine.
/// `transferPhaseEnds` maps a VECTOR-engine operation whose result is consumed
/// by CUBE to the last VECTOR operation that must finish before its final
/// UB-to-L1 copy and ready signal may be committed.
/// The returned handles remain owned by the mutated IR and must be consumed
/// before the referenced operations are erased or reordered.
/// `blockM`, derived internally from the leading transfer, must be positive and
/// divisible by 32 so scope separation can split it evenly across two AIVs.
FailureOr<CrossScopeTransferInfo>
insertCrossScopeTransfers(scf::ForOp loop, const Classification &classification,
                          const llvm::DenseMap<Operation *, Operation *> &transferPhaseEnds,
                          unsigned interCoreBufferDepth, uint64_t privateBufferUbBudgetBytes = 0,
                          bool promotePrivateBufferPools = false);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
