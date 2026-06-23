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

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace CVPipeline {

ComputeBlockIdManager::ComputeBlockIdManager(Operation *root)
{
    cntComputeBlockId = 0;
    blockIdToOps.clear();
    opToBlockId.clear();
    root->walk([&](Operation *op) {
        if (auto blockIdAttr = op->getAttrOfType<IntegerAttr>(kBlockId)) {
            if (auto blockId = blockIdAttr.getInt()) {
                opToBlockId[op] = blockId;
                blockIdToOps[blockId].push_back(op);
                cntComputeBlockId = std::max(cntComputeBlockId, static_cast<int>(blockId));
            }
        }
    });
    cntComputeBlockId++; // ensure new id is unique
}

bool ComputeBlockIdManager::isWholeCubeReady(Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree)
{
    auto id = getBlockIdByOp(seedOp);
    if (id == -1) {
        return (indegree[seedOp] == 0);
    }
    auto cubeBlock = getOpsByBlockId(id);
    for (auto op : cubeBlock) {
        if (!indegree.contains(op)) {
            continue;
        }
        if (indegree[op] != 0) {
            return false;
        }
    }
    return true;
}

bool ComputeBlockIdManager::isSameBlock(Operation *a, Operation *b)
{
    if (getBlockIdByOp(a) == getBlockIdByOp(b) && getBlockIdByOp(a) != -1) {
        return true;
    }
    return false;
}

int ComputeBlockIdManager::getNextId()
{
    return cntComputeBlockId++;
}

void ComputeBlockIdManager::updateBlockId(Operation *op, int blockId)
{
    // Force Update.
    MLIRContext *ctx = op->getContext();
    if (blockId == -1) {
        op->removeAttr(kBlockId);
    } else {
        op->setAttr(kBlockId, IntegerAttr::get(IntegerType::get(ctx, blockIdWidth), blockId));
    }
    auto it = opToBlockId.find(op);
    if (it != opToBlockId.end()) {
        int preBlockId = it->second;
        if (preBlockId != -1) {
            auto &vec = blockIdToOps[preBlockId];
            auto vecIt = llvm::find(vec, op);
            if (vecIt != vec.end()) {
                vec.erase(vecIt);
            }
        }
    }
    opToBlockId[op] = blockId;
    blockIdToOps[blockId].push_back(op);
}

llvm::SmallVector<Operation *> ComputeBlockIdManager::getOpsByBlockId(int blockId)
{
    if (blockId == -1) {
        return {};
    }

    auto it = blockIdToOps.find(blockId);
    if (it == blockIdToOps.end()) {
        return {};
    }
    return llvm::SmallVector<Operation *>(it->second.begin(), it->second.end());
}

int ComputeBlockIdManager::getBlockIdByOp(Operation *op)
{
    auto it = opToBlockId.find(op);
    if (it != opToBlockId.end()) {
        return it->second;
    }
    return -1;
}

llvm::LogicalResult ComputeBlockIdManager::markAndRecord(Operation *op, int blockId)
{
    // When we call mark, we assume the op have no record in manager.
    MLIRContext *ctx = op->getContext();
    op->setAttr(kBlockId, IntegerAttr::get(IntegerType::get(ctx, blockIdWidth), blockId));
    auto itOld = opToBlockId.find(op);
    if (itOld != opToBlockId.end() && itOld->second != -1) {
        llvm::errs() << "Error: Operation already has a block id. Op: " << *op << ", old block id: " << itOld->second
                     << ", new block id: " << blockId << "\n";
        return llvm::failure();
    }

    opToBlockId[op] = blockId;
    blockIdToOps[blockId].push_back(op);
    return llvm::success();
}

llvm::LogicalResult ComputeBlockIdManager::markOpBlockId(Operation *op)
{
    int blockId = getNextId();
    return markAndRecord(op, blockId);
}

llvm::LogicalResult ComputeBlockIdManager::markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops)
{
    if (ops.empty()) {
        return llvm::success();
    }
    int id = getNextId();
    for (Operation *op : ops) {

        if (llvm::failed(markAndRecord(op, id))) {
            return llvm::failure();
        }
    }
    return llvm::success();
}

void ComputeBlockIdManager::reset()
{
    cntComputeBlockId = 0;
    blockIdToOps.clear();
    opToBlockId.clear();
}

} // namespace CVPipeline
} // namespace mlir
