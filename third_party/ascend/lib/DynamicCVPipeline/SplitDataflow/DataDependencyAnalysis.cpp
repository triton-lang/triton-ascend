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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "data-dependency-analysis";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace mlir::CVPipeline;

// Attribute name constants
static constexpr const char *kBlockIdAttr = "ssbuffer.block_id";
static constexpr const char *kCoreTypeAttr = "ssbuffer.core_type";
static constexpr const char *kTransferIdAttr = "ssbuffer.transfer_id";
static constexpr const char *ssbufferCoreTypeCubeAttr = "CUBE";
static constexpr const char *ssbufferCoreTypeVectorAttr = "VECTOR";

// Helper: ssbuffer.core_type
llvm::StringRef getSsbufferCoreType(Operation *op)
{
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(kCoreTypeAttr)) {
        return attr.getValue();
    }
    return "";
}

// Helper: Get CoreType from op and index
llvm::StringRef getCoreTypeWithIndex(Operation *op, int index)
{
    llvm::StringRef typeStr = getSsbufferCoreType(op);
    if (typeStr.contains(", ")) {
        llvm::SmallVector<llvm::StringRef> types;
        typeStr.split(types, ", ", -1, false);
        if (index < types.size()) {
            return types[index].trim();
        }
        LOG_DEBUG("Warning: Core type string has multiple types but value is not an OpResult or index out of range.\n");
        return "";
    }

    return typeStr;
}

// Helper: Check if operation is control flow
bool DataDependencyAnalysisPass::isControlFlowOp(mlir::Operation *op)
{
    if (!op)
        return false;
    return isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::WhileOp>(op) || isa<scf::YieldOp>(op);
}

// Helper: Check if value is a valid tensor for dependency analysis
// Returns true if value is TensorType and not defined by EmptyOp/FillOp
bool DataDependencyAnalysisPass::isValidTensorForDependency(mlir::Value value)
{
    if (!dyn_cast<mlir::TensorType>(value.getType())) {
        return false;
    }
    Operation *defOp = value.getDefiningOp();

    // EmptyOp/FillOp can be processed both by CUBE and VECTOR
    // so they should not be data dependency
    if (defOp && isa<tensor::EmptyOp, linalg::FillOp>(defOp)) {
        return false;
    }
    auto tensorTy = dyn_cast<TensorType>(value.getType());
    static constexpr int NdShapeLength = 2;
    if (!tensorTy || tensorTy.getRank() != NdShapeLength) {
        return false;
    }
    return true;
}

// Helper: Check if value is a BlockArgument
bool DataDependencyAnalysisPass::isOuterOpArg(mlir::Value value)
{
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        mlir::Block *ownerBlock = blockArg.getOwner();
        return true;
    }
    return false;
}

// Helper: Build and record BlockInfo
void DataDependencyAnalysisPass::collectBlockInfo(DataDependencyInfo &info, int blockId,
    llvm::SmallVector<mlir::Operation *> &ops)
{
    if (ops.empty()) {
        LOG_DEBUG("Warning: Block ID " << blockId << " has no operations.\n");
        return;
    }

    BlockInfo blockInfo;
    blockInfo.blockId = blockId;
    blockInfo.isCube = false;

    // In cases with one or more core_types
    // as long as there is a cube, it is necessary to check the dataflow.
    StringRef coreType = getSsbufferCoreType(ops[0]);
    if (coreType.contains(ssbufferCoreTypeCubeAttr)) {
        blockInfo.isCube = true;
    }

    blockInfo.isControl = false;
    if (isControlFlowOp(ops[0])) {
        blockInfo.isControl = true;
    }

    llvm::DenseSet<mlir::Operation *> opSet(ops.begin(), ops.end());

    for (auto *op : ops) {
        blockInfo.Operations.push_back(op);
        for (auto operand : op->getOperands()) {
            mlir::Operation *defOp = operand.getDefiningOp();
            // If defOp is not null and defOp is not in current ops set, it's an external input
            if (!defOp || opSet.find(defOp) == opSet.end()) {
                blockInfo.inputs.insert(operand);
            }
        }
        for (auto result : op->getResults()) {
            // If any user is not in the current ops set, it's an external output
            bool hasExternalUser = false;
            for (mlir::Operation *user : result.getUsers()) {
                if (opSet.find(user) == opSet.end()) {
                    hasExternalUser = true;
                    break;
                }
            }
            if (hasExternalUser) {
                blockInfo.outputs.push_back(result);
            }
        }
    }

    info.getBlockInfoMap()[blockInfo.blockId] = blockInfo;

    LOG_DEBUG("Block_ID=" << blockInfo.blockId << "Processed!\n");
}

// Block Information Collection
void DataDependencyAnalysisPass::createBlockInfoMap(DataDependencyInfo &info)
{
    int currentId = -2;
    static constexpr int startCurrId = -2;
    llvm::SmallVector<mlir::Operation *> currentOps;

    module.walk([&](mlir::Operation *op) {
        auto opBlockIdOpt = CVPipeline::getOpBlockId(op);
        if (opBlockIdOpt) {
            int opBlockId = static_cast<int>(*opBlockIdOpt);
            // When the id changes, the block ends && Exclude the initial state
            if (opBlockId != currentId && currentId != startCurrId) {
                collectBlockInfo(info, currentId, currentOps);
                currentOps.clear();
            }
            currentId = opBlockId;
            currentOps.push_back(op);
        }
    });
    // Process the last group
    if (!currentOps.empty()) {
        collectBlockInfo(info, currentId, currentOps);
    }
}

void DataDependencyAnalysisPass::collectDepInfo(mlir::Value depvalue, DependencyType dependencyType,
    llvm::SmallVector<DependencyInfo> &dependencies, int iniProdId, int iniConsId, DataDependencyInfo &info)
{
    DependencyInfo depInfo;
    depInfo.type = dependencyType;
    depInfo.value = depvalue;
    LOG_DEBUG("try finding common level block IDs\n");
    depInfo.iniProducerBlockId = iniProdId;
    depInfo.iniConsumerBlockId = iniConsId;
    std::pair<int, int> commonLevelIds = findCommonLevelBlockIds(info, iniProdId, iniConsId);
    if (commonLevelIds.first == -1 || commonLevelIds.second == -1) {
        LOG_DEBUG("Could not find common level block IDs for producer and consumer blocks");
        signalPassFailure();
    }

    depInfo.producerBlockId = commonLevelIds.first;
    depInfo.consumerBlockId = commonLevelIds.second;

    dependencies.push_back(depInfo);
}


// Collects users of iterArg that have a different core type than initCoreType.
llvm::SmallVector<mlir::Operation *> DataDependencyAnalysisPass::collectDiffCoreTypeUsers(
    mlir::BlockArgument iterArg, llvm::StringRef initCoreType)
{
    llvm::SmallVector<mlir::Operation *> diffUsers;

    for (mlir::Operation *user : iterArg.getUsers()) {
        if (isa<scf::YieldOp>(user)) {
            continue;
        }
        if (isControlFlowOp(user)) {
            LOG_DEBUG("cannot process nested iterarg!");
            signalPassFailure();
        }

        auto userCoreType = getCoreTypeWithIndex(user, 0);
        if (userCoreType != initCoreType && !userCoreType.empty()) {
            diffUsers.push_back(user);
        }
    }

    return diffUsers;
}

// Inserts a producer block at the beginning of the for loop body and records
// cross-core-type dependencies for each user in diffUsers.
void DataDependencyAnalysisPass::insertProducerAndRecordDeps(scf::ForOp forOp,
    mlir::BlockArgument iterArg, llvm::StringRef initCoreType,
    llvm::SmallVector<mlir::Operation *> &diffUsers, DataDependencyInfo &info)
{
    auto &v2cDependencies = info.getV2CDependencies();
    auto &c2vDependencies = info.getC2VDependencies();

    int newId = CVPipeline::getAvailableBlockId(module);
    OpBuilder builder(forOp);
    Block &bodyBlock = forOp.getRegion().front();
    builder.setInsertionPointToStart(&bodyBlock);
    Location loc = forOp.getLoc();
    auto constOp = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    constOp->setAttr(kBlockIdAttr, IntegerAttr::get(IntegerType::get(builder.getContext(), 32), newId));
    constOp->setAttr(kCoreTypeAttr, StringAttr::get(builder.getContext(), initCoreType));

    for (auto &user : diffUsers) {
        auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
        if (!userBlockIdOpt) {
            LOG_DEBUG("Warning: User block ID not found for iterArg user.\n");
            continue;
        }
        int userBlockId = static_cast<int>(*userBlockIdOpt);

        // Determine dependency type based on initCoreType
        DependencyType depType;
        if (initCoreType == ssbufferCoreTypeVectorAttr) {
            depType = DependencyType::VectorToCube;
        } else if (initCoreType == ssbufferCoreTypeCubeAttr) {
            depType = DependencyType::CubeToVector;
        } else {
            LOG_DEBUG("Warning: Unknown initCoreType: " << initCoreType << "\n");
            continue;
        }

        // Record dependency
        DependencyInfo depInfo;
        depInfo.type = depType;
        depInfo.value = iterArg;
        depInfo.iniProducerBlockId = newId;
        depInfo.iniConsumerBlockId = userBlockId;
        depInfo.producerBlockId = newId;
        depInfo.consumerBlockId = userBlockId;

        if (depType == DependencyType::VectorToCube) {
            v2cDependencies.push_back(depInfo);
        } else {
            c2vDependencies.push_back(depInfo);
        }

        LOG_DEBUG("Recorded iterArg dependency: " << initCoreType << " -> "
            << (depType == DependencyType::VectorToCube ? ssbufferCoreTypeCubeAttr : ssbufferCoreTypeVectorAttr)
            << ", producerBlockId=" << newId << ", consumerBlockId=" << userBlockId << "\n");
    }
}

// Process iterArg dependencies for all scf.for operations in the module.
// This function iterates through all for loops and checks each iterArg to determine
// if there are cross-core-type data dependencies.
void DataDependencyAnalysisPass::processIterArgDependencies()
{
    auto &info = getAnalysis<DataDependencyInfo>();

    // Step1: Collect all scf.for operations in the module
    llvm::SmallVector<scf::ForOp> forOps;
    module.walk([&](scf::ForOp forOp) {
        forOps.push_back(forOp);
    });
    LOG_DEBUG("Processing iterArg dependencies, found " << forOps.size() << " scf.for ops\n");

    // Step2: Process each iterArg of each scf.for operation
    for (scf::ForOp forOp : forOps) {
        size_t numIterArgs = forOp.getInitArgs().size();

        for (int iterArgIndex = 0; iterArgIndex < numIterArgs; ++iterArgIndex) {
            mlir::Value initValue = forOp.getInits()[iterArgIndex];
            // Skip non-tensor values as they don't require inter-core synchronization
            if (!isValidTensorForDependency(initValue)) {
                LOG_DEBUG("iterarg: "<< initValue <<"is not valid tensor for dependency!");
                continue;
            }

            mlir::BlockArgument iterArg = forOp.getRegionIterArg(iterArgIndex);
            mlir::Value yieldedValue = forOp.getYieldedValues()[iterArgIndex];

            Operation *initDefOp = initValue.getDefiningOp();
            if (!initDefOp) {
                LOG_DEBUG("warning: nested iterarg!");
                continue;
            }

            auto initDefResult = dyn_cast<mlir::OpResult>(initValue);
            auto initCoreType = getCoreTypeWithIndex(initDefOp, initDefResult ? initDefResult.getResultNumber() : 0);
            auto yieldCoreType = getCoreTypeWithIndex(forOp, iterArgIndex);

            // Only process if init and yield have matching core types
            // Mismatch indicates a more complex dependency pattern that requires special handling
            if (initCoreType != yieldCoreType) {
                LOG_DEBUG("iterarg init core_type conflicts with yield");
                signalPassFailure();
            }

            auto diffUsers = collectDiffCoreTypeUsers(iterArg, initCoreType);
            if (!diffUsers.empty()) {
                insertProducerAndRecordDeps(forOp, iterArg, initCoreType, diffUsers, info);
            }
        }
    }
}

// Analyze V->C
void DataDependencyAnalysisPass::analyzeExternalInputs(DataDependencyInfo &info)
{
    auto &blockInfoMap = info.getBlockInfoMap();
    auto &v2cDependencies = info.getV2CDependencies();

    LOG_DEBUG("Analyzing external inputs for Cube blocks...\n");
    for (auto &[id, blockInfo] : blockInfoMap) {
        if (!blockInfo.isCube || blockInfo.isControl || blockInfo.inputs.empty())
            continue;
        LOG_DEBUG("Analyzing external inputs for Cube Block ID: " << id << "\n");
        for (mlir::Value input : blockInfo.inputs) {
            // Check if input is a value which can be produced by CUBE
            if (!isValidTensorForDependency(input)) {
                LOG_DEBUG("Warning: [v->c] Input value is not a valid tensor for dependency analysis.\n");
                continue;
            }
            // Check if input is a blockarg.
            if (isOuterOpArg(input)) {
                LOG_DEBUG("Warning: [v->c] Input value is a function/scf parameter.\n");
                continue;
            }

            Operation *defOp = input.getDefiningOp();
            auto defReuslt = dyn_cast<mlir::OpResult>(input);
            auto coreType = getCoreTypeWithIndex(defOp, defReuslt ? defReuslt.getResultNumber() : 0);
            if (coreType == "") {
                LOG_DEBUG("Warning: [v->c] Input value has no core type attribute.\n");
                continue;
            }

            // Case 1: Cube -> C->C special case
            if (coreType == ssbufferCoreTypeCubeAttr) {
                continue;
            }
            // Case 2: Vector -> V->C dependency
            if (coreType == ssbufferCoreTypeVectorAttr) {
                LOG_DEBUG("Found external input with VECTOR core type: " << input << "\n");
                auto producerIdOpt = CVPipeline::getOpBlockId(input.getDefiningOp());
                if (!producerIdOpt) {
                    LOG_DEBUG("Warning: [v->c] Producer block ID not found for input value.\n");
                    continue;
                }
                int producerId = static_cast<int>(*producerIdOpt);
                collectDepInfo(input, DependencyType::VectorToCube, v2cDependencies, producerId, blockInfo.blockId,
                    info);
            }
        }
    }
    LOG_DEBUG("External input analysis complete.\n");
}

// Analyze C->V
void DataDependencyAnalysisPass::analyzeExternalOutputs(DataDependencyInfo &info)
{
    auto &blockInfoMap = info.getBlockInfoMap();
    auto &c2vDependencies = info.getC2VDependencies();

    LOG_DEBUG("Analyzing external outputs for Cube blocks...\n");
    for (auto &[id, blockInfo] : blockInfoMap) {
        if (!blockInfo.isCube || blockInfo.outputs.empty())
            continue;

        for (mlir::Value output : blockInfo.outputs) {
            // Check if output is a value which can be produced by CUBE
            if (!isValidTensorForDependency(output)) {
                LOG_DEBUG("Warning: [c->v] Output value is not a valid tensor for dependency analysis.\n");
                continue;
            }

            auto opResult = dyn_cast<OpResult>(output);
            unsigned resultIndex = opResult.getResultNumber();
            StringRef resultCoreType = getCoreTypeWithIndex(output.getDefiningOp(), resultIndex);
            if (resultCoreType != ssbufferCoreTypeCubeAttr) {
                continue;
            }

            // Check who is using this output
            llvm::DenseSet<int> handledBlockIds;
            for (mlir::Operation *user : output.getUsers()) {
                int outputIndex = 0;
                if (isControlFlowOp(user)) {
                    for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                        if (user->getOperand(i) == output) {
                            outputIndex = i;
                            break;
                        }
                    }
                }
                auto userCoreType = getCoreTypeWithIndex(user, outputIndex);
                if ((userCoreType == "")) {
                    LOG_DEBUG("Warning: [c->v] Input value has no core type attribute.\n");
                    continue;
                }
                if (userCoreType == ssbufferCoreTypeVectorAttr) {
                    LOG_DEBUG("Found external output used by VECTOR core type: " << output << "\n");
                    auto consumerIdOpt = CVPipeline::getOpBlockId(user);
                    if (!consumerIdOpt) {
                        LOG_DEBUG("Warning: [c->v] Consumer block ID not found for user operation.\n");
                        continue;
                    }
                    int consumerId = static_cast<int>(*consumerIdOpt);
                    auto inserted = handledBlockIds.insert(consumerId).second;
                    if (inserted) {
                      collectDepInfo(output, DependencyType::CubeToVector, c2vDependencies, blockInfo.blockId, consumerId,
                          info);
                    }
                }
                // If user belongs to Cube block, this C->C dependency was handled
                // in the Input analysis phase, so here we only handle C->V.
            }
        }
    }
    LOG_DEBUG("External output analysis complete.\n");
}

void DataDependencyAnalysisPass::analyzeMemoryEffect(DataDependencyInfo &info)
{
    auto &memoryDependencies = info.getMemoryDependencies();
    LOG_DEBUG("\n=== start mem dep analysis ===\n");

    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

    module.walk([&](mlir::Operation *op) {
        auto currBlockIdOpt = CVPipeline::getOpBlockId(op);
        llvm::StringRef currCoreType = getSsbufferCoreType(op);
        if (!currBlockIdOpt || currCoreType.empty()) {
            return;
        }
        int currBlockId = static_cast<int>(*currBlockIdOpt);

        for (mlir::Operation *predOp : memDepGraph.getExecBefore(op)) {
            auto predBlockIdOpt = CVPipeline::getOpBlockId(predOp);
            llvm::StringRef predCoreType = getSsbufferCoreType(predOp);
            if (!predBlockIdOpt || predCoreType == currCoreType || predCoreType.empty()) {
                continue;
            }
            int predBlockId = static_cast<int>(*predBlockIdOpt);

            auto [producerBlockId, consumerBlockId] = findCommonLevelBlockIds(info, predBlockId, currBlockId);
            if (producerBlockId == -1 || consumerBlockId == -1) {
                LOG_DEBUG("Could not find common level block IDs for producer and consumer blocks");
                signalPassFailure();
            }
            if (producerBlockId == consumerBlockId) {
                continue;
            }

            DependencyInfo depInfo;

            if (predCoreType == ssbufferCoreTypeCubeAttr) {
                depInfo.type = DependencyType::CubeToVector;
            } else if (predCoreType == ssbufferCoreTypeVectorAttr) {
                depInfo.type = DependencyType::VectorToCube;
            }
            depInfo.producerBlockId = producerBlockId;
            depInfo.consumerBlockId = consumerBlockId;
            depInfo.iniProducerBlockId = predBlockId;
            depInfo.iniConsumerBlockId = currBlockId;

            memoryDependencies.push_back(depInfo);
            LOG_DEBUG("=mem dep analysis= "
                << "producer Block: " << predBlockId << "consumer Block: " << currBlockId << "\n");
        }
    });
    LOG_DEBUG("=== mem dep analysis complete ===\n");
}

// Producer/Consumer Hierarchy Analysis
std::pair<int, int> DataDependencyAnalysisPass::findCommonLevelBlockIds(DataDependencyInfo &info, int producerBlockId,
    int consumerBlockId)
{
    auto &blockInfoMap = info.getBlockInfoMap();

    LOG_DEBUG("start findCommonLevelBlockIds...\n");

    // Step 1: Get corresponding BlockInfo from Map
    auto pIt = blockInfoMap.find(producerBlockId);
    auto cIt = blockInfoMap.find(consumerBlockId);
    // Defensive programming: if corresponding Block info not found, return original ID or error code
    if (pIt == blockInfoMap.end() || cIt == blockInfoMap.end()) {
        return { producerBlockId, consumerBlockId };
    }

    BlockInfo &pInfo = pIt->second;
    BlockInfo &cInfo = cIt->second;

    // Take the first operation of each Block as representative to check hierarchy
    // (Assumes all operations in a Block are closely related in hierarchy)
    mlir::Operation *producerOp = pInfo.Operations[0];
    mlir::Operation *consumerOp = cInfo.Operations[0];

    mlir::Block *pBlock = producerOp->getBlock();
    mlir::Block *cBlock = consumerOp->getBlock();

    // Case 1: In the same MLIR Block
    if (pBlock == cBlock) {
        return { producerBlockId, consumerBlockId };
    }

    // Case 2: In different Blocks, find Lowest Common Ancestor (LCA)
    // Step 1: Collect producer's ancestor chain
    llvm::SmallVector<mlir::Operation *> pAncestors;
    pAncestors.push_back(producerOp);
    mlir::Operation *current = producerOp->getParentOp();
    while (current) {
        pAncestors.push_back(current);
        current = current->getParentOp();
    }

    // Step 2: Walk up consumer's ancestors, using current and before for rolling
    mlir::Operation *before = consumerOp; // Initialize as consumerOp itself
    current = consumerOp;                 // Initialize as parent

    while (current) {
        // --- Found common ancestor ---
        auto it = std::find(pAncestors.begin(), pAncestors.end(), current);
        if (it != pAncestors.end()) {
            size_t pIndex = std::distance(pAncestors.begin(), it);
            if (pIndex == 0) {
                break;
            }
            mlir::Operation *pPrevOp = pAncestors[pIndex - 1];
            auto pPrevIdOpt = CVPipeline::getOpBlockId(pPrevOp);
            auto cPrevIdOpt = CVPipeline::getOpBlockId(before);
            int pPrevId = pPrevIdOpt ? static_cast<int>(*pPrevIdOpt) : -1;
            int cPrevId = cPrevIdOpt ? static_cast<int>(*cPrevIdOpt) : -1;
            if (!pPrevIdOpt) {
                LOG_DEBUG("Warning: Producer ancestor operation has no block ID attribute.\n");
            }
            if (!cPrevIdOpt) {
                LOG_DEBUG("Warning: Consumer ancestor operation has no block ID attribute.\n");
            }
            return { pPrevId, cPrevId };
        }

        // Rolling: continue upward
        before = current;
        current = current->getParentOp();
    }
    LOG_DEBUG("Warning: No common ancestor found for producer block " << producerBlockId << " and consumer block " <<
        consumerBlockId << "\n");
    return { -1, -1 };
}

void DataDependencyAnalysisPass::runOnOperation()
{
    LOG_DEBUG("\n--- enter DataDependencyAnalysisPass --->\n");
    module = getOperation();

    auto &info = getAnalysis<DataDependencyInfo>();

    // Step 1: Collect block information (populate blockInfoMap)
    createBlockInfoMap(info);

    // Step 2: Analyze iter_args dependencies
    processIterArgDependencies();

    // Step 3: Analyze dependencies (populate v2c, c2v lists)
    analyzeExternalInputs(info);
    analyzeExternalOutputs(info);

    // Step 4: Analyze memory dependencies (PIPE_S sync)
    analyzeMemoryEffect(info);

    info.setValid(true);

    LOG_DEBUG("DataDependencyAnalysisPass: Analysis complete.\n");
    LOG_DEBUG("  V->C dependencies: " << info.getV2CDependencies().size() << "\n");
    LOG_DEBUG("  C->V dependencies: " << info.getC2VDependencies().size() << "\n");
    LOG_DEBUG("  Memory dependencies: " << info.getMemoryDependencies().size() << "\n");

    LOG_DEBUG("\n--- exit DataDependencyAnalysisPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createDataDependencyAnalysisPass()
{
    return std::make_unique<DataDependencyAnalysisPass>();
}

} // namespace triton
} // namespace mlir
