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
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "triton/Analysis/Alias.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
#define DEBUG_TYPE "data-dependency-analysis"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

using namespace mlir::triton;
using namespace mlir::CVPipeline;

// Helper: Get CoreType from op and index
// ==========================================
llvm::StringRef DataDependencyAnalysisPass::getCoreType(Operation* op, int index) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("ssbuffer.core_type");
  if (!attr) return "";

  llvm::StringRef typeStr = attr.getValue();

  if (typeStr.contains(", ")) {
    llvm::SmallVector<llvm::StringRef> types;
    typeStr.split(types, ", ", /*MaxSplit=*/ -1, /*KeepEmpty=*/ false);
    if (index < types.size()) {
      return types[index].trim();
    }

    LOG_DEBUG("Warning: Core type string has multiple types but value is not an OpResult or index out of range.\n");
    return "";
  }

  return typeStr;
}

// ==========================================
// Helper: Get BlockId
// ==========================================
int DataDependencyAnalysisPass::getBlockId(mlir::Value val) {
  auto *op = val.getDefiningOp();
  if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
    return attr.getInt();
  }
  return -1;
}

// ==========================================
// Helper: Check if operation is control flow
// ==========================================
bool DataDependencyAnalysisPass::isControlFlowOp(mlir::Operation *op) {
  if (!op) return false;
  return isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::WhileOp>(op) || isa<scf::YieldOp>(op);
  // TODO: Add more control flow operation types as needed
}

// ==========================================
// Helper: Build and record BlockInfo
// ==========================================
void DataDependencyAnalysisPass::collectBlockInfo(DataDependencyInfo& info, int blockId, llvm::SmallVector<mlir::Operation*>& ops) {
  if (ops.empty()) {
    LOG_DEBUG("Warning: Block ID " << blockId << " has no operations.\n");
    return;
  }

  BlockInfo blockInfo;
  blockInfo.blockId = blockId;
  blockInfo.isCube = false;
  if (auto typeAttr = ops[0]->getAttrOfType<StringAttr>("ssbuffer.core_type")) {
    StringRef coreType = typeAttr.getValue();
    if (coreType.contains("CUBE")) {
      blockInfo.isCube = true;
    }
  }

  blockInfo.isControl = false;
  if (isControlFlowOp(ops[0])) {
    blockInfo.isControl = true;
  }

  llvm::DenseSet<mlir::Operation*> opSet(ops.begin(), ops.end());

  for (auto *op : ops) {
    blockInfo.Operations.push_back(op);
    for (auto operand : op->getOperands()) {
      mlir::Operation *defOp = operand.getDefiningOp();

      // If defOp is null (BlockArgument) or defOp is not in current ops set
      if (!defOp || opSet.find(defOp) == opSet.end()) {
        blockInfo.inputs.push_back(operand);
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

  LOG_DEBUG("Processed Block: ID=" << blockInfo.blockId
                          << " Type=" << (blockInfo.isCube ? "CUBE" : "VECTOR")
                          << " OpsCount=" << ops.size() << "\n");
}

// ==========================================
// Block Information Collection
// ==========================================
void DataDependencyAnalysisPass::createBlockInfoMap(DataDependencyInfo& info) {
  int currentId = -1;
  llvm::SmallVector<mlir::Operation*> currentOps;

  module.walk([&](mlir::Operation* op) {
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")){

      int opBlockId = idAttr.getInt();

      if (opBlockId != currentId) {
        if (currentId != -1) {
          collectBlockInfo(info, currentId, currentOps);
          currentOps.clear();
        }
        currentId = opBlockId;
      }

      currentOps.push_back(op);
    }
  });
  // Process the last group
  if (!currentOps.empty()) {
    collectBlockInfo(info, currentId, currentOps);
  }
}

// ==========================================
// External Input Analysis (V->C)
// ==========================================
void DataDependencyAnalysisPass::analyzeExternalInputs(DataDependencyInfo& info) {
  auto& blockInfoMap = info.getBlockInfoMap();
  auto& v2cDependencies = info.getV2CDependencies();
  auto& c2cDependencies = info.getC2CDependencies();

  // Iterate through all Cube compute blocks
  LOG_DEBUG("Analyzing external inputs for Cube blocks...\n");
  for (auto& [id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.isControl || blockInfo.inputs.empty()) continue;
    LOG_DEBUG("Analyzing external inputs for Cube Block ID: " << id << "\n");
    for (mlir::Value input : blockInfo.inputs) {
      // TODO: Add logic to check if input is a func.func blockarg.
      // If so, it's a function input parameter - skip for now (needs further thought)
      if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(input)) {
        LOG_DEBUG("Warning: [v->c] Input value is a function parameter.\n");
        continue;
      }
      //
      if (!dyn_cast<mlir::TensorType>(input.getType())) {
        LOG_DEBUG("Warning: ExternalInput is not TensorType\n");
        continue;
      }
      if (isa<tensor::EmptyOp, linalg::FillOp>(input.getDefiningOp())) {
        LOG_DEBUG("Warning: [v->c] Input value is defined by tensor::EmptyOp/linalg::FillOp.\n");
        continue;
      }
      Operation* defOp = input.getDefiningOp();
      auto defReuslt = dyn_cast<mlir::OpResult>(input);
      auto coreType = getCoreType(defOp, defReuslt ? defReuslt.getResultNumber() : 0);
      if (coreType == "") {
        LOG_DEBUG("Warning: [v->c] Input value has no core type attribute.\n");
        continue;
      }

      // Case 1: Cube -> C->C special case
      if (coreType == "CUBE") {
        analyzeCubeToCubeDependencies(info, id, input);
      }
      // Case 2: Vector -> V->C dependency
      else if (coreType == "VECTOR") {

        LOG_DEBUG("Found external input with VECTOR core type: " << input << "\n");
        auto producerId = getBlockId(input);
        if (producerId != -1) {
          DependencyInfo depInfo;
          depInfo.type = DependencyType::VectorToCube;
          depInfo.value = input;
          LOG_DEBUG("try finding common level block IDs\n");
          depInfo.iniProducerBlockId = producerId;
          depInfo.iniConsumerBlockId = blockInfo.blockId;
          std::pair<int, int> commonLevelIds = findCommonLevelBlockIds(info, producerId, blockInfo.blockId);
          LOG_DEBUG("found!\n");
          assert(commonLevelIds.first != -1 && commonLevelIds.second != -1 &&
                 "Could not find common level block IDs for producer and consumer blocks");
          depInfo.producerBlockId = commonLevelIds.first;
          depInfo.consumerBlockId = commonLevelIds.second;
          v2cDependencies.push_back(depInfo);
        } else {
          LOG_DEBUG("Warning: [v->c] Producer block ID not found for input value.\n");
        }
      }
    }
  }
  LOG_DEBUG("External input analysis complete.\n");
}

// ==========================================
// External Output Analysis (C->V)
// ==========================================
void DataDependencyAnalysisPass::analyzeExternalOutputs(DataDependencyInfo& info) {
  auto& blockInfoMap = info.getBlockInfoMap();
  auto& c2vDependencies = info.getC2VDependencies();

  LOG_DEBUG("Analyzing external outputs for Cube blocks...\n");
  // Iterate through all Cube compute blocks
  for (auto& [id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.outputs.empty()) continue;
    // Iterate through all outputs of this Cube block
    for (mlir::Value output : blockInfo.outputs) {
      if (!dyn_cast<mlir::TensorType>(output.getType())) {
        LOG_DEBUG("Warning: ExternalOutput is not TensorType\n");
        continue;
      }
      if (isa<tensor::EmptyOp, linalg::FillOp>(output.getDefiningOp())) {
        LOG_DEBUG("Warning: [c->v] output value is defined by tensor::EmptyOp/linalg::FillOp.\n");
        continue;
      }
      auto opResult = dyn_cast<OpResult>(output);
      unsigned resultIndex = opResult.getResultNumber();
      StringRef resultCoreType = getCoreType(output.getDefiningOp(), resultIndex);
      if (resultCoreType != "CUBE") {
        continue;
      }
      // Check who is using this output
      for (mlir::Operation* user : output.getUsers()) {
        int outputIndex = 0;
        if (auto forOp = mlir::dyn_cast<scf::ForOp>(user)) {
          bool isDependencyValid = false;
          // Check if output is an iter_arg of forOp
          for (size_t i = 0; i < forOp.getNumRegionIterArgs(); i++) {
            if (forOp.getRegionIterArg(i) == output) {
              outputIndex = i;  // Update to index in iter_args
              isDependencyValid = true;
              LOG_DEBUG("[DEBUG] Output is iter_arg #" << i << " of scf.ForOp\n");
              break;
            }
          }
          // Only consider dependency valid when output is an iter_arg of forOp
          if (!isDependencyValid) {
            LOG_DEBUG("[DEBUG] Output is NOT an iter_arg, skipping this dependency\n");
            continue;  // Skip this user
          }
        } else if (isControlFlowOp(user)) {
          for (unsigned i = 0; i < user->getNumOperands(); ++i) {
            if (user->getOperand(i) == output) {
              outputIndex = i;
              break;
            }
          }
        }
        auto userCoreType = getCoreType(user, outputIndex);
        if ((userCoreType == "")) {
          LOG_DEBUG("Warning: [c->v] Input value has no core type attribute.\n");
          continue;
        } else if (userCoreType == "VECTOR") {
          LOG_DEBUG("Found external output used by VECTOR core type: " << output << "\n");
          auto consumerId = getBlockId(user->getResult(0));
          if (consumerId != -1) {
            DependencyInfo depInfo;
            depInfo.type = DependencyType::CubeToVector;
            depInfo.value = output;
            LOG_DEBUG("try finding common level block IDs\n");
            depInfo.iniProducerBlockId = blockInfo.blockId;
            depInfo.iniConsumerBlockId = consumerId;
            std::pair<int, int> commonLevelIds = findCommonLevelBlockIds(info, blockInfo.blockId, consumerId);
            assert(commonLevelIds.first != -1 && commonLevelIds.second != -1 &&
                   "Could not find common level block IDs for producer and consumer blocks");
            LOG_DEBUG("found!\n");
            depInfo.producerBlockId = commonLevelIds.first;
            depInfo.consumerBlockId = commonLevelIds.second;

            c2vDependencies.push_back(depInfo);
          } else {
            LOG_DEBUG("Warning: [c->v] Consumer block ID not found for user operation.\n");
          }
        }

        // Note: If user belongs to Cube block, this C->C dependency was handled
        // in the Input analysis phase, so here we only handle C->V.
      }
    }
  }
  LOG_DEBUG("External output analysis complete.\n");
}

// ==========================================
// C->C Special Case Analysis
// ==========================================
void DataDependencyAnalysisPass::analyzeCubeToCubeDependencies(DataDependencyInfo& info, int consumerBlockId, mlir::Value operand) {
  auto& c2cDependencies = info.getC2CDependencies();

  for (mlir::Operation* user : operand.getUsers()) {
    int outputIndex = 0;
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(user)) {
      bool isDependencyValid = false;
      // Check if output is an iter_arg of forOp
      for (size_t i = 0; i < forOp.getNumRegionIterArgs(); i++) {
        if (forOp.getRegionIterArg(i) == operand) {
          outputIndex = i;  // Update to index in iter_args
          isDependencyValid = true;
          LOG_DEBUG("[DEBUG] Output is iter_arg #" << i << " of scf.ForOp\n");
          break;
        }
      }
      // Only consider dependency valid when output is an iter_arg of forOp
      if (!isDependencyValid) {
        LOG_DEBUG("[DEBUG] Output is NOT an iter_arg, skipping this dependency\n");
        continue;  // Skip this user
      }
    } else if (isControlFlowOp(user)) {
      for (unsigned i = 0; i < user->getNumOperands(); ++i) {
        if (user->getOperand(i) == operand) {
          outputIndex = i;
          break;
        }
      }
    }
    auto userCoreType = getCoreType(user, outputIndex);
    if (userCoreType == "") {
      LOG_DEBUG("Warning: [c->c] User operation has no core type attribute.\n");
      continue;
    }
    if (userCoreType == "CUBE") {
      DependencyInfo depInfo;
      depInfo.type = DependencyType::CubeToCube;
      depInfo.value = operand;
      depInfo.consumerBlockId = consumerBlockId;

      auto producerId = getBlockId(operand);
      if (producerId != -1) {
        depInfo.producerBlockId = producerId;
      }
    }
  }
}

// ==========================================
// PIPE_S Memory Effect Analysis
// ==========================================

// 辅助函数：获取操作的核心类型
static llvm::StringRef getOpCoreType(mlir::Operation* op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("ssbuffer.core_type");
  if (!attr) {
    return "";
  }
  return attr.getValue();
}

void DataDependencyAnalysisPass::analyzeMemoryEffect(DataDependencyInfo& info) {
  auto& memoryDependencies = info.getMemoryDependencies();

  LOG_DEBUG("\n=== 开始 PIPE_S 内存依赖分析 ===\n");

  // 构建 MemoryDependenceGraph 获取内存依赖
  auto& aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
  MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

  // 遍历所有具有 block_id 属性的操作，检查其内存依赖
  module.walk([&](mlir::Operation* op) {
    // 跳过没有 block_id 属性的操作
    if (!op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      return;
    }

    // 跳过 assert 操作
    if (llvm::isa<triton::AssertOp>(op)) {
      return;
    }

    llvm::StringRef currCoreType = getOpCoreType(op);

    // 跳过没有核心类型的操作（标量操作）
    if (currCoreType.empty()) {
      return;
    }

    int currBlockId = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id").getInt();

    // 使用 getExecBefore 获取必须在当前操作之前执行的操作（内存依赖前驱）
    for (mlir::Operation* predOp : memDepGraph.getExecBefore(op)) {
      // 跳过没有 block_id 属性的前驱操作
      if (!predOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
        continue;
      }

      llvm::StringRef predCoreType = getOpCoreType(predOp);

      // 跳过相同核心类型（没有跨核心依赖）
      if (predCoreType == currCoreType) {
        continue;
      }

      // 跳过没有核心类型的前驱操作
      if (predCoreType.empty()) {
        continue;
      }

      int predBlockId = predOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id").getInt();

      // 使用已有函数找到公共层级的 block ID
      auto [producerBlockId, consumerBlockId] = findCommonLevelBlockIds(info, predBlockId, currBlockId);

      if (producerBlockId == -1 || consumerBlockId == -1) {
        op->emitWarning("无法找到公共层级的 block ID。\n");
        continue;
      }

      // 跳过生产者和消费者相同的情况
      if (producerBlockId == consumerBlockId) {
        continue;
      }

      // 创建此内存依赖的 DependencyInfo
      DependencyInfo depInfo;

      // 根据核心类型确定依赖类型
      if (predCoreType == "CUBE") {
        depInfo.type = DependencyType::CubeToVector;
      } else if (predCoreType == "VECTOR") {
        depInfo.type = DependencyType::VectorToCube;
      }

      depInfo.producerBlockId = producerBlockId;
      depInfo.consumerBlockId = consumerBlockId;
      depInfo.iniProducerBlockId = predBlockId;
      depInfo.iniConsumerBlockId = currBlockId;

      memoryDependencies.push_back(depInfo);

      LOG_DEBUG("[PIPE_S 内存依赖] "
                   << "生产者 Block: " << predBlockId
                   << " (" << predCoreType << ") -> "
                   << "消费者 Block: " << currBlockId
                   << " (" << currCoreType << ")\n");
    }
  });

  LOG_DEBUG("=== PIPE_S 内存依赖分析完成 ===\n");
  LOG_DEBUG("发现 " << memoryDependencies.size() << " 个内存依赖。\n");
}

// ==========================================
// Producer/Consumer Hierarchy Analysis
// ==========================================
std::pair<int, int> DataDependencyAnalysisPass::findCommonLevelBlockIds(
    DataDependencyInfo& info, int producerBlockId, int consumerBlockId) {
  auto& blockInfoMap = info.getBlockInfoMap();

  LOG_DEBUG("start findCommonLevelBlockIds...\n");

  // Step 1: Get corresponding BlockInfo from Map
  auto pIt = blockInfoMap.find(producerBlockId);
  auto cIt = blockInfoMap.find(consumerBlockId);

  // Defensive programming: if corresponding Block info not found, return original ID or error code
  if (pIt == blockInfoMap.end() || cIt == blockInfoMap.end()) {
    return {producerBlockId, consumerBlockId};
  }

  BlockInfo& pInfo = pIt->second;
  BlockInfo& cInfo = cIt->second;

  // Take the first operation of each Block as representative to check hierarchy
  // (Assumes all operations in a Block are closely related in hierarchy)
  mlir::Operation* producerOp = pInfo.Operations[0];
  mlir::Operation* consumerOp = cInfo.Operations[0];

  mlir::Block* pBlock = producerOp->getBlock();
  mlir::Block* cBlock = consumerOp->getBlock();

  // Case 1: In the same MLIR Block
  if (pBlock == cBlock) {
    return {producerBlockId, consumerBlockId};
  }

  // Case 2: In different Blocks, find Lowest Common Ancestor (LCA)

  // Step 1: Collect producer's ancestor chain
  llvm::SmallVector<mlir::Operation*> pAncestors;
  pAncestors.push_back(producerOp);
  mlir::Operation* current = producerOp->getParentOp();
  while (current) {
    pAncestors.push_back(current);
    current = current->getParentOp();
  }

  // Step 2: Walk up consumer's ancestors, using current and before for rolling
  mlir::Operation* before = consumerOp; // Initialize as consumerOp itself
  current = consumerOp;  // Initialize as parent

  while (current) {
    // --- Found common ancestor ---
    auto it = std::find(pAncestors.begin(), pAncestors.end(), current);
    if (it != pAncestors.end()) {
      size_t pIndex = std::distance(pAncestors.begin(), it);
      mlir::Operation* pPrevOp = pAncestors[pIndex - 1];
      int pPrevId = -1;
      int cPrevId = -1;
      if (auto attr = pPrevOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
        pPrevId = attr.getInt();
      } else {
        LOG_DEBUG("Warning: Producer ancestor operation has no block ID attribute.\n");
      }
      if (auto attr = before->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
        cPrevId = attr.getInt();
      } else {
        LOG_DEBUG("Warning: Consumer ancestor operation has no block ID attribute.\n");
      }
      return {pPrevId, cPrevId};
    }

    // Rolling: continue upward
    before = current;
    current = current->getParentOp();
  }
  LOG_DEBUG("Warning: No common ancestor found for producer block " << producerBlockId
               << " and consumer block " << consumerBlockId << ". Returning original IDs.\n");
  return {-1, -1};
}

// ==========================================
// Pass Entry Point
// ==========================================
void DataDependencyAnalysisPass::runOnOperation() {
  LOG_DEBUG("\n--- enter DataDependencyAnalysisPass --->\n");
  module = getOperation();

  // 获取 DataDependencyInfo（MLIR Analysis 机制）
  auto& info = getAnalysis<DataDependencyInfo>();

  // Step 1: Collect block information (populate blockInfoMap)
  createBlockInfoMap(info);

  // Step 2: Analyze dependencies (populate v2c, c2v, c2c lists)
  analyzeExternalInputs(info);
  analyzeExternalOutputs(info);

  // Step 3: Analyze memory dependencies (PIPE_S sync)
  analyzeMemoryEffect(info);

  // 标记数据有效
  info.setValid(true);

  LOG_DEBUG("DataDependencyAnalysisPass: Analysis complete.\n");
  LOG_DEBUG("  V->C dependencies: " << info.getV2CDependencies().size() << "\n");
  LOG_DEBUG("  C->V dependencies: " << info.getC2VDependencies().size() << "\n");
  LOG_DEBUG("  C->C dependencies: " << info.getC2CDependencies().size() << "\n");
  LOG_DEBUG("  Memory dependencies: " << info.getMemoryDependencies().size() << "\n");

  // IR validity verification
  if (failed(module.verify())) {
    module.emitError("DataDependencyAnalysis: generated invalid IR");
    signalPassFailure();
  }
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createDataDependencyAnalysisPass()
{
    return std::make_unique<DataDependencyAnalysisPass>();
}

void registerDataDependencyAnalysisPasses()
{
  registerPass([]() -> std::unique_ptr<mlir::Pass> { return createDataDependencyAnalysisPass(); });
}

} // namespace triton
} // namespace mlir