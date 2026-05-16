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
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <queue>

#define DEBUG_TYPE "ub-usage-opt"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

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

    llvm::StringRef getArgument() const final { return "ub-usage-opt"; }

  private:
    int getValueSizeInBytes(Value value);
    void buildUBUsageGraph(Block *block, DenseMap<Operation *, int> &op2nodeId, DenseMap<int, Operation *> &nodeId2op,
                           SmallVector<SmallVector<int>> &linkOut, SmallVector<SmallVector<int>> &linkIn,
                           SmallVector<int> &linkSize, SmallVector<int> &linkStart, SmallVector<int> &linkEnd,
                           SmallVector<int> &nodeBlockId, SmallVector<int> &nodeCoreType, SmallVector<int> &nodeArgs,
                           const CVPipeline::MemoryDependenceGraph &memGraph);
    llvm::LogicalResult UBUsageOptimization(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph);
};
} // namespace triton
} // namespace mlir

int UBUsageOptPass::getValueSizeInBytes(Value value)
{
    Type type = value.getType();
    auto getElemBytes = [](Type elemType) -> int64_t {
        if (elemType.isIntOrFloat()) {
            unsigned bits = elemType.getIntOrFloatBitWidth();
            return std::max<int64_t>(1, bits / 8);
        }
        return 1;
    };
    // Tensor
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
    // Memref
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
    // Vector
    if (auto vectorType = dyn_cast<VectorType>(type)) {
        return static_cast<int>(
            std::max<int64_t>(1, vectorType.getNumElements() * getElemBytes(vectorType.getElementType())));
    }
    // Index
    if (auto idxTy = dyn_cast<IndexType>(value.getType())) {
        DataLayout dataLayout(getOperation());
        unsigned bitWidth = dataLayout.getTypeSizeInBits(IndexType::get(getOperation()->getContext()));
        return bitWidth / 8;
    }
    return static_cast<int>(getElemBytes(type));
}

void UBUsageOptPass::buildUBUsageGraph(Block *block, DenseMap<Operation *, int> &op2nodeId,
                                       DenseMap<int, Operation *> &nodeId2op, SmallVector<SmallVector<int>> &linkOut,
                                       SmallVector<SmallVector<int>> &linkIn, SmallVector<int> &linkSize,
                                       SmallVector<int> &linkStart, SmallVector<int> &linkEnd,
                                       SmallVector<int> &nodeBlockId, SmallVector<int> &nodeCoreType,
                                       SmallVector<int> &nodeArgs, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    DenseMap<int, int> cubeBlockId2nodeId;
    const int cubeCoreType = static_cast<int>(CVPipeline::CoreType::CUBE_ONLY);

    auto getOrCreateNodeId = [&](Operation *op) -> int {
        if (op2nodeId.find(op) != op2nodeId.end()) {
            return op2nodeId.at(op);
        }
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

    DenseMap<std::pair<int, int>, bool> visited;
    auto addEdge = [&](int src, int dst, int sizeInBytes) {
        if (visited.contains(std::make_pair(src, dst))) {
            return;
        }
        if (src == dst) {
            // self-cycle can only be args dependency.
            LOG_DEBUG("Find one self-cycle: id edge from " << src << " to " << dst << " size = " << sizeInBytes
                                                           << "\n");
            LOG_DEBUG("op edge from " << *nodeId2op[src] << " to " << *nodeId2op[dst] << "\n");
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

    for (Operation &blockOp : *block) {
        int dstNode = getOrCreateNodeId(&blockOp);
        blockOp.walk([&](Operation *op) {
            for (Value operand : op->getOperands()) {
                Operation *srcInBlock = nullptr;
                bool fromArgEdge = false;
                if (Operation *defOp = operand.getDefiningOp()) {
                    srcInBlock = CVPipeline::getAncestorInBlock(defOp, block);
                    if (srcInBlock && srcInBlock == &blockOp) {
                        // To avoid unnesessary self-cycle.
                        // some ops in block(op) used parent(blockop) args can lead one self-cycle.
                        continue;
                    }
                } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                    if (blockArg.getOwner() == block && terminator) {
                        unsigned numArgs = block->getNumArguments();
                        unsigned numYieldOperands = terminator->getNumOperands();
                        // for op offset=1, while op offset=0
                        int offset = (int)numArgs - (int)numYieldOperands;
                        int argIdx = (int)blockArg.getArgNumber() - offset;

                        if (argIdx >= 0 && argIdx < (int)numYieldOperands) {
                            Value yielded = terminator->getOperand(argIdx);
                            if (Operation *yieldDefOp = yielded.getDefiningOp()) {
                                srcInBlock = CVPipeline::getAncestorInBlock(yieldDefOp, block);
                                fromArgEdge = true;
                            }
                        }
                    }
                }
                if (!srcInBlock || srcInBlock->getBlock() != block) {
                    continue;
                }
                // To avoid unnesessary self-cycle.
                // Two ops in the same cube block lead to self-cycle, so continue it.
                auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
                int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
                int srcBlockId = bm.getBlockIdByOp(srcInBlock);
                int dstBlockId = bm.getBlockIdByOp(&blockOp);
                if (coreType == cubeCoreType && srcBlockId == dstBlockId) {
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
                if (srcInBlock && srcInBlock == &blockOp) {
                    // To avoid unnesessary self-cycle.
                    // some ops in block(op) used parent(blockop) args can lead one self-cycle.
                    continue;
                }
                // To avoid unnesessary self-cycle.
                // Two ops in the same cube block lead to self-cycle, so continue it.
                auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
                int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
                int srcBlockId = bm.getBlockIdByOp(srcInBlock);
                int dstBlockId = bm.getBlockIdByOp(&blockOp);
                if (coreType == cubeCoreType && srcBlockId == dstBlockId) {
                    continue;
                }

                int srcNode = getOrCreateNodeId(srcInBlock);
                addEdge(srcNode, dstNode, 0);
            }
        });
    }
}

SmallVector<int> findDependency(int targetNdoe, int preNode, const SmallVector<SmallVector<int>> &linkIn,
                                const SmallVector<int> &linkStart)
{
    SmallVector<int> dependNodes;
    DenseSet<int> visited;
    std::queue<int> queue;
    queue.push(targetNdoe);
    visited.insert(targetNdoe);
    while (!queue.empty()) {
        int curNode = queue.front();
        queue.pop();
        LOG_DEBUG("curNode = " << curNode << "\n");
        if (curNode == preNode) {
            continue; // we find other linkin of b in A->B, so pass A.
        }

        for (int inEdgeId : linkIn[curNode]) {
            int inStart = linkStart[inEdgeId];
            LOG_DEBUG("instart = " << inStart << "\n");
            if (visited.insert(inStart).second) {
                dependNodes.push_back(inStart);
                queue.push(inStart);
            }
        }
    }
    return dependNodes;
}

bool isActiveEndNode(int srcNode, int endNode, const SmallVector<SmallVector<int>> &linkIn,
                     const SmallVector<int> &linkStart, const SmallVector<int> &nodeBlockId,
                     const SmallVector<int> &nodeCoreType, DenseMap<int, Operation *> nodeId2op)
{
    int nodeNum = static_cast<int>(nodeBlockId.size());
    if (nodeCoreType[endNode] != nodeCoreType[srcNode]) {
        return false;
    }
    if (nodeBlockId[endNode] == -1) {
        // meet scf.yield/return..
        return false;
    }
    if (nodeBlockId[srcNode] == nodeBlockId[endNode]) {
        // same cmpute block not leaf
        return false;
    }
    // To avoid cycles: we only collect 2 kind node:
    // 1. endNode only from src compute block. A->endnode->...
    // 2. endNode's srcs donnot depend any compute block.  A->endnode;args/const/->src1->endnode->...
    // In fact the second kind includes first;
    // BFS/DFS search expanding dependNodes transitively
    auto dependNodes = findDependency(endNode, srcNode, linkIn, linkStart);
    for (int node : dependNodes) {
        if (nodeBlockId[node] != nodeBlockId[endNode] && nodeBlockId[node] != nodeBlockId[srcNode]) {
            return false;
        }
    }
    return true;
}

static SmallVector<SmallVector<int>>
collectNeedUbOpts(const SmallVector<SmallVector<int>> &linkOut, const SmallVector<SmallVector<int>> &linkIn,
                  const SmallVector<int> &linkStart, const SmallVector<int> &linkEnd,
                  const SmallVector<int> &nodeBlockId, const SmallVector<int> &nodeCoreType,
                  DenseMap<int, Operation *> nodeId2op)
{
    SmallVector<SmallVector<int>> needUbOpts;
    int maxBlockId = -1; // FIXME: We need to count how many compute block in the graph, but use max is not accurate.
    for (int blockId : nodeBlockId) {
        maxBlockId = std::max(maxBlockId, blockId);
    }
    if (maxBlockId >= 0) {
        needUbOpts.resize(static_cast<size_t>(maxBlockId + 1));
    }
    for (int i = 0; i < nodeBlockId.size(); ++i) {
        int srcBlockId = nodeBlockId[i];
        int srcCoreType = nodeCoreType[i];
        if (srcCoreType != CVPipeline::CoreType::VECTOR_ONLY) {
            // only vector block need ub optimization.
            continue;
        }
        bool canOptimize = false;
        for (int outEdgeId : linkOut[i]) {
            int dstNode = linkEnd[outEdgeId];
            if (isActiveEndNode(i, dstNode, linkIn, linkStart, nodeBlockId, nodeCoreType, nodeId2op)) {
                canOptimize = true;
                break;
            }
        }
        if (canOptimize && srcBlockId >= 0) {
            needUbOpts[srcBlockId].push_back(i);
        }
    }
    return needUbOpts;
}

static int sumIncomingLinkSize(int nodeId, const SmallVector<SmallVector<int>> &linkIn,
                               const SmallVector<int> &linkSize)
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
                             const SmallVector<int> &nodeBlockId, int &uniqueNextNode)
{
    // only find A link single chain.
    if (linkOut[curNode].size() != 1)
        return false;
    // sigle chian must be in one compute block.
    auto edgeId = linkOut[curNode][0];
    uniqueNextNode = linkEnd[edgeId];
    if (nodeBlockId[uniqueNextNode] != nodeBlockId[curNode]) {
        return false;
    }
    // uniqueNextNode's dependent is all in nodeBlockId[curNode]
    bool onlyDependsOnCur = true;
    auto dependNodes = findDependency(uniqueNextNode, curNode, linkIn, linkStart);
    for (auto node : dependNodes) {
        if (nodeBlockId[node] != nodeBlockId[curNode]) {
            return false;
        }
    }
    return true;
}

static DenseMap<int, int> collectRecordChange(const SmallVector<SmallVector<int>> &needUbOpts,
                                              const SmallVector<SmallVector<int>> &linkOut,
                                              const SmallVector<SmallVector<int>> &linkIn,
                                              const SmallVector<int> &linkSize, const SmallVector<int> &linkStart,
                                              const SmallVector<int> &linkEnd, const SmallVector<int> &nodeBlockId,
                                              const SmallVector<int> &nodeCoreType,
                                              DenseMap<int, Operation *> nodeId2op)
{
    DenseMap<int, int> recordChange;
    int nodeNum = static_cast<int>(nodeBlockId.size());
    for (int optBlockId = 0, blockNum = static_cast<int>(needUbOpts.size()); optBlockId < blockNum; ++optBlockId) {
        for (int optNode : needUbOpts[optBlockId]) {
            LOG_DEBUG("optNode: " << *nodeId2op.at(optNode) << "\n");
            SmallVector<int> activateSet;
            for (int outEdgeId : linkOut[optNode]) {
                int dstNode = linkEnd[outEdgeId];
                if (isActiveEndNode(optNode, dstNode, linkIn, linkStart, nodeBlockId, nodeCoreType, nodeId2op)) {
                    if (std::find(activateSet.begin(), activateSet.end(), dstNode) == activateSet.end()) {
                        activateSet.push_back(dstNode);
                    }
                }
            }
            LOG_DEBUG("activateSet Size:" << activateSet.size() << "\n");

            for (int activateNode : activateSet) {
                int originUBSize = sumIncomingLinkSize(activateNode, linkIn, linkSize);
                int minUBSize = originUBSize;
                SmallVector<int> chain;
                chain.push_back(activateNode);
                int bestCutPointIdx = -1;

                // find the activate chain.
                while (true) {
                    int curNode = chain.back();
                    int uniqueNextNode = -1;
                    if (!findUniqueDependentNode(curNode, optBlockId, linkOut, linkIn, linkStart, linkEnd, linkSize,
                                                 nodeBlockId, uniqueNextNode))
                    {
                        break;
                    }

                    chain.push_back(uniqueNextNode);
                }
                for (auto i = 0; i < chain.size(); i++) {
                    auto nowUBSize = 0;
                    LOG_DEBUG("now chain op = " << *nodeId2op.at(chain[i]) << "\n");
                    for (auto cutEdgeId : linkOut[chain[i]]) {

                        nowUBSize += linkSize[cutEdgeId];
                    }
                    if (nowUBSize < minUBSize) {
                        bestCutPointIdx = i + 1;
                        minUBSize = nowUBSize;
                    }
                }

                if (bestCutPointIdx > 0) {
                    // need to change not only chain[i], and chain[i]'s dependency.
                    for (int i = 0; i < bestCutPointIdx; ++i) {
                        recordChange[chain[i]] = optBlockId;
                        auto chainPreNode = i - 1 < 0 ? optNode : chain[i - 1];
                        auto dependNodes = findDependency(chain[i], chainPreNode, linkIn, linkStart);
                        for (auto node : dependNodes) {
                            if (nodeBlockId[node] != optBlockId) {
                                recordChange[node] = optBlockId;
                            }
                        }
                    }
                }
            }
        }
    }
    return recordChange;
}

namespace {

struct DependencyCycleDetector {
    llvm::DenseSet<mlir::Operation *> &okSet; // node in okSet will become one compute block;
    llvm::DenseSet<mlir::Operation *> visited;
    const CVPipeline::MemoryDependenceGraph &memGraph;
    Block *block;
    void clear() { visited.clear(); }
    bool operator()(Operation *cur);
    bool dfs(Operation *cur) { return (*this)(cur); };

    DependencyCycleDetector(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
                            llvm::DenseSet<mlir::Operation *> &okSet)
        : block(block), memGraph(memGraph), okSet(okSet)
    {
    }
};

} // namespace

bool DependencyCycleDetector::operator()(Operation *cur)
{
    if (okSet.contains(cur)) {
        return true;
    }
    if (!visited.insert(cur).second) {
        return false;
    }

    SmallVector<Operation *> allusers;
    allusers.append(cur->getUsers().begin(), cur->getUsers().end());
    for (auto *memUser : memGraph.getExecAfter(cur)) {
        allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
        auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
        auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
        if (bm.getBlockIdByOp(userInBlock) == -1) {
            if (dfs(userInBlock)) {
                return true;
            }
        } else {
            for (auto *nx : bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock))) {
                if (dfs(nx)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
    * Check if adding willaddOps to targetBlockId will create cycle.
    * Walk from every op in targetBlockId and willaddOps.
    * if reach other blockid ops and dfs find any targetBlockId op, then there is cycle.
*/
static bool willCreateCycle(llvm::SmallVectorImpl<Operation *> &willaddOps, Block *block,
                            const CVPipeline::MemoryDependenceGraph &memGraph, int targetBlockId)
{
    // Step1: Init, Add willaddOps to targetBlockId.
    // OkSet is new block, includes two part: 1. original ops in targetBlockId. 2. willaddOps.
    llvm::DenseSet<mlir::Operation *> okSet;  
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    for (auto op : bm.getOpsByBlockId(targetBlockId)) {
        okSet.insert(op);
    }
    llvm::DenseMap<mlir::Operation *, int> originBlockId;
    for (auto op : willaddOps) {
        okSet.insert(op);
        // For backtracing
        originBlockId[op] = bm.getBlockIdByOp(op);
        bm.updateBlockId(op, targetBlockId);
    }
    DependencyCycleDetector dfs = {block, memGraph, okSet};

    // Step2: Walk from every op in okSet
    auto ret = false;
    for (mlir::Operation *okOp : okSet) {
        SmallVector<Operation *> allusers;
        allusers.append(okOp->getUsers().begin(), okOp->getUsers().end());
        for (auto *memUser : memGraph.getExecAfter(okOp)) {
            allusers.push_back(memUser);
        }
        for (auto *user : allusers) {
            auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
            if (okSet.contains(userInBlock)) {
                continue;
            }
            if (bm.getBlockIdByOp(userInBlock) == -1) {
                dfs.clear();
                if (dfs(userInBlock)) {
                    ret = true;
                    break;
                }
                continue;
            }
            auto opsUsedBlockId = bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock));
            for (auto *userOp : opsUsedBlockId) {
                dfs.clear();
                if (dfs(userOp)) {
                    ret = true;
                    break;
                }
            }
        }
        if (ret) {
            //early stop if find cycle.
            break;
        }
    }
    
    // Step3: Backtrace blockId change.
    for (auto op : willaddOps) {
        bm.updateBlockId(op, originBlockId[op]);
    }
    return ret;
}

bool applyRecordChange(DenseMap<int, int> &recordChange, DenseMap<int, Operation *> &nodeId2op,
                       const CVPipeline::MemoryDependenceGraph &memGraph)
{
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();

    // Get ever blockid should be add which nodeId in recordChange.
    DenseMap<int, SmallVector<int>> blockWilladd;
    for (const auto &it : recordChange) {
        int nodeId = it.first;
        int optBlockId = it.second;
        blockWilladd[optBlockId].push_back(nodeId);
    }
    bool hasError = false;
    for (auto it : blockWilladd) {
        int targetBlockId = it.first;
        auto willaddNodes = it.second;
        llvm::SmallVector<Operation *> willaddOps;
        for (int nodeId : willaddNodes) {
            willaddOps.push_back(nodeId2op[nodeId]);
        }
        if (willCreateCycle(willaddOps, willaddOps[0]->getBlock(), memGraph, targetBlockId)) {
            LOG_DEBUG("Find cycle when apply change for blockId: " << targetBlockId << "\n");
            for (auto nodeId : willaddNodes) {
                LOG_DEBUG("  - " << *nodeId2op[nodeId]<<"\n");
            }
            hasError = true;
            continue;
        }

        for (auto op : willaddOps) {
            bm.updateBlockId(op, targetBlockId);
        }
    }

    return hasError;
}

llvm::LogicalResult UBUsageOptPass::UBUsageOptimization(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    if (!isa<scf::ForOp>(block->getParentOp())) {
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
        collectNeedUbOpts(linkOut, linkIn, linkStart, linkEnd, nodeBlockId, nodeCoreType, nodeId2op);
    int candidateCnt = 0;
    for (const auto &nodes : needUbOpts) {
        candidateCnt += static_cast<int>(nodes.size());
        for (auto opid : nodes) {
            LOG_DEBUG("maybe need opt: " << *nodeId2op[opid] << "\t id = " << opid << "\n");
        }
    }
    LOG_DEBUG("Find " << candidateCnt << " op maybe need UB optimization\n");
    llvm::DenseMap<int, int> recordChange = collectRecordChange(needUbOpts, linkOut, linkIn, linkSize, linkStart,
                                                                linkEnd, nodeBlockId, nodeCoreType, nodeId2op);
    LOG_DEBUG("Need change blockId for " << recordChange.size() << " nodes\n");

    if (applyRecordChange(recordChange, nodeId2op, memGraph)) {
        // FIXME: it shouldn't happen....
        llvm::errs() << "Some skiped when apply UB usage optimization changes.\n";
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
            llvm::errs() << "UB usage optimization failed in block:\n";
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
