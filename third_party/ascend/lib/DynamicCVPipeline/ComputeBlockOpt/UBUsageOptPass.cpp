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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>



#define DEBUG_TYPE "ub-usage-opt"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg<<"\n")
    
using namespace mlir;
using namespace triton;
using namespace mlir::triton;
namespace mlir {
namespace triton {
class UBUsageOptPass : public PassWrapper<UBUsageOptPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UBUsageOptPass)

    UBUsageOptPass() = default;
    void runOnOperation() override;

    llvm::StringRef getArgument() const final {
      return "ub-usage-opt";
    }
};
}}

static int getValueSizeInBytes(Value value)
{
    Type type = value.getType();
    auto getElemBytes = [](Type elemType) -> int64_t {
        if (elemType.isIntOrFloat()) {
            unsigned bits = elemType.getIntOrFloatBitWidth();
            return std::max<int64_t>(1, bits / 8);
        }
        return 1;
    };

    if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
        if (!rankedTensorType.hasStaticShape()) {
            return 1;
        }
        int64_t numElements = 1;
        for (int64_t dim : rankedTensorType.getShape()) {
            if (dim < 0) {
                return 1;
            }
            numElements *= dim;
        }
        return static_cast<int>(std::max<int64_t>(1, numElements * getElemBytes(rankedTensorType.getElementType())));
    }

    if (auto memRefType = dyn_cast<MemRefType>(type)) {
        if (!memRefType.hasStaticShape()) {
            return 1;
        }
        int64_t numElements = 1;
        for (int64_t dim : memRefType.getShape()) {
            if (dim < 0) {
                return 1;
            }
            numElements *= dim;
        }
        return static_cast<int>(std::max<int64_t>(1, numElements * getElemBytes(memRefType.getElementType())));
    }

    if (auto vectorType = dyn_cast<VectorType>(type)) {
        return static_cast<int>(
            std::max<int64_t>(1, vectorType.getNumElements() * getElemBytes(vectorType.getElementType())));
    }

    return static_cast<int>(getElemBytes(type));
}

void buildUBUsageGraph(Block *block, DenseMap<Operation *, int> &op2nodeId, DenseMap<int, Operation *> &nodeId2op,
                               SmallVector<SmallVector<int>> &linkOut, SmallVector<SmallVector<int>> &linkIn,
                               SmallVector<int> &linkSize, SmallVector<int> &linkStart, SmallVector<int> &linkEnd,
                               SmallVector<int> &nodeBlockId, SmallVector<int> &nodeCoreType, SmallVector<int> &nodeArgs, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    op2nodeId.clear();
    nodeId2op.clear();
    linkOut.clear();
    linkIn.clear();
    linkSize.clear();
    linkStart.clear();
    linkEnd.clear();
    nodeBlockId.clear();
    nodeCoreType.clear();
    nodeArgs.clear();

    DenseMap<int, int> cubeBlockId2nodeId;
    const int cubeCoreType = static_cast<int>(CVPipeline::CoreType::CUBE_ONLY);

    auto getOrCreateNodeId = [&](Operation *op) -> int {
        auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
        int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
        int blockId = bm.getBlockIdByOp(op);
        bool canShrink = (coreType == cubeCoreType && blockId != -1);

        if (canShrink) {
            auto it = cubeBlockId2nodeId.find(blockId);
            if (it != cubeBlockId2nodeId.end()) {
                op2nodeId[op] = it->second;
                return it->second;
            }
        }

        int nodeId = static_cast<int>(nodeBlockId.size());
        op2nodeId[op] = nodeId;
        nodeId2op[nodeId] = op;
        nodeBlockId.push_back(blockId);
        nodeCoreType.push_back(coreType);
        nodeArgs.push_back(-1);
        linkOut.emplace_back();
        linkIn.emplace_back();

        if (canShrink) {
            cubeBlockId2nodeId[blockId] = nodeId;
        }
        return nodeId;
    };

    for (Operation &op : *block) {
        getOrCreateNodeId(&op);
    }

    DenseMap<std::pair<int, int>, bool> visited;
    auto addEdge = [&](int src, int dst, int sizeInBytes) {
        if(visited.contains(std::make_pair(src, dst))){
            return;
        }
        int edgeId = static_cast<int>(linkSize.size());
        linkSize.push_back(sizeInBytes);
        linkStart.push_back(src);
        linkEnd.push_back(dst);
        linkOut[src].push_back(edgeId);
        linkIn[dst].push_back(edgeId);
        visited[std::make_pair(src, dst)] = true;
    };

    Operation *terminator = block->getTerminator();
    if (terminator) {
        unsigned maxArgIdx = std::min<unsigned>(block->getNumArguments(), terminator->getNumOperands());
        for (unsigned argIdx = 0; argIdx < maxArgIdx; ++argIdx) {
            Value yielded = terminator->getOperand(argIdx);
            Operation *defOp = yielded.getDefiningOp();
            if (!defOp) {
                continue;
            }
            Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
            if (!defInBlock || defInBlock->getBlock() != block) {
                continue;
            }
            int nodeId = getOrCreateNodeId(defInBlock);
            if (nodeArgs[nodeId] == -1) {
                nodeArgs[nodeId] = static_cast<int>(argIdx);
            }
        }
    }

    block->walk([&](Operation *blockOp){
        int dstNode = getOrCreateNodeId(blockOp);
        blockOp->walk([&](Operation *op){
            for (Value operand : op->getOperands()) {
                Operation *srcInBlock = nullptr;
                bool fromArgEdge = false;
                if (Operation *defOp = operand.getDefiningOp()) {
                    srcInBlock = CVPipeline::getAncestorInBlock(defOp, block);
                } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                    if (blockArg.getOwner() == block && terminator &&
                        blockArg.getArgNumber() < terminator->getNumOperands())
                    {
                        Value yielded = terminator->getOperand(blockArg.getArgNumber());
                        if (Operation *yieldDefOp = yielded.getDefiningOp()) {
                            srcInBlock = CVPipeline::getAncestorInBlock(yieldDefOp, block);
                            fromArgEdge = true;
                        }
                    }
                }
                if (!srcInBlock || srcInBlock->getBlock() != block) {
                    continue;
                }
                int srcNode = getOrCreateNodeId(srcInBlock);
                int edgeSize = getValueSizeInBytes(operand);
                if (fromArgEdge) {
                    edgeSize *= 2;
                }
                addEdge(srcNode, dstNode, edgeSize);
            }
    
            for (auto memDef : memGraph.getExecBefore(op)) {
                Operation *srcInBlock = nullptr;
                srcInBlock = CVPipeline::getAncestorInBlock(memDef, block);
                if (!srcInBlock || srcInBlock->getBlock() != block) {
                    continue;
                }
                int srcNode = getOrCreateNodeId(srcInBlock);
                addEdge(srcNode, dstNode, 0);
            }
        });
    });
}

bool isActiveEndNode(int srcNode, int endNode, const SmallVector<SmallVector<int>> &linkIn,
                             const SmallVector<int> &linkStart, const SmallVector<int> &nodeBlockId,
                             const SmallVector<int> &nodeCoreType)
{
    int nodeNum = static_cast<int>(nodeBlockId.size());
    if (nodeCoreType[endNode] != nodeCoreType[srcNode]) {
        return false;
    }
    if (nodeBlockId[endNode] == -1) {
        return false;
    }
    if(nodeBlockId[srcNode] == nodeBlockId[endNode]) {
        return false;
    }
    for (int inEdgeId : linkIn[endNode]) {
        int inStart = linkStart[inEdgeId];
        if (nodeBlockId[inStart] != nodeBlockId[srcNode]) {
            return false;
        }
    }
    return true;
}

static SmallVector<SmallVector<int>> collectNeedUbOpts(const SmallVector<SmallVector<int>> &linkOut,
                                                        const SmallVector<SmallVector<int>> &linkIn,
                                                        const SmallVector<int> &linkStart,
                                                        const SmallVector<int> &linkEnd,
                                                        const SmallVector<int> &nodeBlockId,
                                                        const SmallVector<int> &nodeCoreType)
{
    SmallVector<SmallVector<int>> needUbOpts;
    int maxBlockId = -1;
    for (int blockId : nodeBlockId) {
        maxBlockId = std::max(maxBlockId, blockId);
    }
    if (maxBlockId >= 0) {
        needUbOpts.resize(static_cast<size_t>(maxBlockId + 1));
    }

    for (int i = 0, nodeNum = static_cast<int>(nodeBlockId.size()); i < nodeNum; ++i) {
        int srcBlockId = nodeBlockId[i];
        int srcCoreType = nodeCoreType[i];
        bool canOptimize = false;
        for (int outEdgeId : linkOut[i]) {
            int dstNode = linkEnd[outEdgeId];
            if (isActiveEndNode(i, dstNode, linkIn, linkStart, nodeBlockId, nodeCoreType))
            {
                canOptimize = true;
                break;
            }
        }
        if (canOptimize && srcBlockId >= 0 ) {
            needUbOpts[srcBlockId].push_back(i);
        }
    }
    return needUbOpts;
}

static int sumIncomingLinkSize(int nodeId, const SmallVector<SmallVector<int>> &linkIn, const SmallVector<int> &linkSize)
{
    int totalSize = 0;
    for (int edgeId : linkIn[nodeId]) {
        totalSize += linkSize[edgeId];
    }
    return totalSize;
}

bool findUniqueDependentNode(int curNode, int optBlockId, const SmallVector<SmallVector<int>> &linkOut,
                                     const SmallVector<SmallVector<int>> &linkIn, const SmallVector<int> &linkStart,
                                     const SmallVector<int> &linkEnd, const SmallVector<int> &linkSize,
                                     const SmallVector<int> &nodeBlockId, int &uniqueNextNode, int &edgeSizeToNext)
{
    if (linkOut[curNode].size() != 1) return false;
    auto edgeId = linkOut[curNode][0];
    uniqueNextNode = linkEnd[edgeId];
    if (nodeBlockId[uniqueNextNode] == optBlockId || nodeBlockId[uniqueNextNode] == -1) {
        return false;
    }
    bool onlyDependsOnCur = true;
    for (int inEdgeId : linkIn[uniqueNextNode]) {
        int inStart = linkStart[inEdgeId];
        if (inStart != curNode) {
            onlyDependsOnCur = false;
            break;
        }
    }
    if (!onlyDependsOnCur) {
        return false;
    }
    edgeSizeToNext = linkSize[edgeId];
    return true;
}

static DenseMap<int, int> collectRecordChange(const SmallVector<SmallVector<int>> &needUbOpts,
                                              const SmallVector<SmallVector<int>> &linkOut,
                                              const SmallVector<SmallVector<int>> &linkIn,
                                              const SmallVector<int> &linkSize, const SmallVector<int> &linkStart,
                                              const SmallVector<int> &linkEnd, const SmallVector<int> &nodeBlockId,
                                              const SmallVector<int> &nodeCoreType)
{
    DenseMap<int, int> recordChange;
    int nodeNum = static_cast<int>(nodeBlockId.size());
    for (int optBlockId = 0, blockNum = static_cast<int>(needUbOpts.size()); optBlockId < blockNum; ++optBlockId) {
        for (int optNode : needUbOpts[optBlockId]) {
            SmallVector<int> activateSet;
            for (int outEdgeId : linkOut[optNode]) {
                int dstNode = linkEnd[outEdgeId];
                if (isActiveEndNode(optNode, dstNode, linkIn, linkStart, nodeBlockId, nodeCoreType))
                {
                    if (std::find(activateSet.begin(), activateSet.end(), dstNode) == activateSet.end()) {
                        activateSet.push_back(dstNode);
                    }
                }
            }

            for (int activateNode : activateSet) {
                int originUBSize = sumIncomingLinkSize(activateNode, linkIn, linkSize);
                int minUBSize = originUBSize;
                SmallVector<int> chain;
                chain.push_back(activateNode);
                int bestCutPointIdx = -1;

                while (true) {
                    int curNode = chain.back();
                    int uniqueNextNode = -1;
                    int edgeSizeToNext = 0;
                    if (!findUniqueDependentNode(curNode, optBlockId, linkOut, linkIn, linkStart, linkEnd, linkSize,
                                                 nodeBlockId, uniqueNextNode, edgeSizeToNext))
                    {
                        break;
                    }

                    chain.push_back(uniqueNextNode);
                    if (edgeSizeToNext < minUBSize) {
                        minUBSize = edgeSizeToNext;
                        bestCutPointIdx = static_cast<int>(chain.size()) - 1;
                    }
                }
                if (bestCutPointIdx > 0) {
                    for (int i = 0; i < bestCutPointIdx; ++i) {
                        recordChange[chain[i]] = optBlockId;
                    }
                }
            }
        }
    }
    return recordChange;
}

llvm::LogicalResult UBUsageOptimization(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    if (!isa<scf::ForOp>(block->getParentOp())){
        return llvm::success();
    } 
    DenseMap<Operation *, int> op2nodeId;
    DenseMap<int, Operation *> nodeId2op;
    SmallVector<SmallVector<int>> linkOut;
    SmallVector<SmallVector<int>> linkIn;
    SmallVector<int> linkSize;
    SmallVector<int> linkStart;
    SmallVector<int> linkEnd;
    SmallVector<int> nodeBlockId;
    SmallVector<int> nodeCoreType;
    SmallVector<int> nodeArgs;
    buildUBUsageGraph(block, op2nodeId, nodeId2op, linkOut, linkIn, linkSize, linkStart, linkEnd, nodeBlockId,
                      nodeCoreType, nodeArgs, memGraph);
    SmallVector<SmallVector<int>> needUbOpts =
        collectNeedUbOpts(linkOut, linkIn, linkStart, linkEnd, nodeBlockId, nodeCoreType);
    int candidateCnt = 0;
    for (const auto &nodes : needUbOpts) {
        candidateCnt += static_cast<int>(nodes.size());
    }
    LOG_DEBUG("Find " << candidateCnt << " op maybe need UB optimization\n");
    llvm::DenseMap<int, int> recordChange =
        collectRecordChange(needUbOpts, linkOut, linkIn, linkSize, linkStart, linkEnd, nodeBlockId, nodeCoreType);
    LOG_DEBUG("Need change blockId for " << recordChange.size() << " nodes\n");

    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    for (const auto &it : recordChange) {
        int nodeId = it.first;
        int optBlockId = it.second;
        return bm.markOpBlockId(nodeId2op[nodeId], optBlockId);
    }
    return llvm::success();
}

void mlir::triton::UBUsageOptPass::runOnOperation()
{
    LOG_DEBUG("--- Pass: UBUsageOpt ---\n");

    ModuleOp module = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

    llvm::SmallVector<Block *> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });

    for (Block *block : blocks) {
        if (UBUsageOptimization(block, memDepGraph).failed()) {
            module.emitError("UB usage optimization failed.");
            return signalPassFailure();
        }
    }

    LOG_DEBUG("=== Pass UBUsageOpt complete ===\n");
}


namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUBUsageOptPass()
{
    return std::make_unique<UBUsageOptPass>();
}

} // namespace triton
} // namespace mlir