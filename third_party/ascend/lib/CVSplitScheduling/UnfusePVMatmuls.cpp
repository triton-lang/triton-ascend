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

#include "ascend/include/CVSplitScheduling/UnfusePVMatmuls.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

// Split matmul(p, v, acc * alpha) into two operations:
// (1) pv = matmul(p, v, zeros) and (2) combined = arith.addf(pv, acc * alpha)
// This is needed because triton's combine pass fuses arith.addf(matmul(...,0), x)
// into matmul(..., x), creating an unresolvable CUBE→VECTOR→CUBE chain through
// the accumulator. Unfusing makes the PV matmul independent of the accumulator.
LogicalResult unfusePVMatmuls(Block *body, Classification &classification)
{
    SmallVector<linalg::MatmulOp> toUnfuse;
    for (Operation &op : *body) {
        auto matmulOp = dyn_cast<linalg::MatmulOp>(&op);
        if (!matmulOp)
            continue;

        // The outs value is the DPS init.
        Value outsVal = matmulOp.getDpsInitOperand(0)->get();

        // Check if outs is produced by a VECTOR op (e.g. arith.mulf for acc*alpha)
        Operation *outsDef = outsVal.getDefiningOp();
        if (!outsDef || outsDef->getBlock() != body)
            continue;
        auto outsClassIt = classification.find(outsDef);
        if (outsClassIt == classification.end()) {
            matmulOp.emitError("missing classification for matmul accumulator producer");
            return failure();
        }
        if (outsClassIt->second != EngineType::VECTOR)
            continue;

        // This is a fused PV matmul with VECTOR-produced accumulator init
        toUnfuse.push_back(matmulOp);
    }

    if (toUnfuse.empty())
        return success();

    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Unfusing " << toUnfuse.size() << " PV matmuls with VECTOR outs\n");

    for (auto matmulOp : toUnfuse) {
        OpBuilder builder(matmulOp);
        Location loc = matmulOp.getLoc();

        Value outsVal = matmulOp.getDpsInitOperand(0)->get();
        auto outsType = dyn_cast<RankedTensorType>(outsVal.getType());
        if (!outsType) {
            matmulOp.emitError("expected a ranked tensor matmul accumulator");
            return failure();
        }

        // Create zero init tensor
        auto zeroAttr = builder.getZeroAttr(outsType.getElementType());
        auto zeroConst = builder.create<arith::ConstantOp>(loc, outsType, DenseElementsAttr::get(outsType, zeroAttr));

        // Replace outs with zeros in the matmul
        matmulOp.getDpsInitOperand(0)->set(zeroConst.getResult());

        // Insert arith.addf after matmul: combined = matmul_result + original_outs
        builder.setInsertionPointAfter(matmulOp);
        Value matResult = matmulOp.getResult(0);
        auto addOp = builder.create<arith::AddFOp>(loc, matResult, outsVal);

        // Replace all uses of the original matmul result (except the addf itself)
        matResult.replaceAllUsesExcept(addOp.getResult(), addOp);

        // Classify new ops
        classification[zeroConst] = EngineType::CUBE;
        classification[addOp] = EngineType::VECTOR;
        setOpEngineTypeAttr(zeroConst, EngineType::CUBE);
        setOpEngineTypeAttr(addOp, EngineType::VECTOR);
    }

    return success();
}

} // namespace mlir::triton::cv_split
