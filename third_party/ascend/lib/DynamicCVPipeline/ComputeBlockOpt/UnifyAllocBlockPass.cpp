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
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "unify-alloc-block"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct FillInfo {
  linalg::FillOp fillOp;
  scf::IfOp parentIf;
  bool needsSplit;
};

struct CycleDfs {
  llvm::DenseSet<mlir::Operation *> &okSet;
  llvm::DenseSet<mlir::Operation *> visited;
  const CVPipeline::MemoryDependenceGraph &memGraph;
  Block *block;
  void clear() { visited.clear(); }
  bool operator()(Operation *cur);
  bool dfs(Operation *cur) { return (*this)(cur); };
  CycleDfs(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
           llvm::DenseSet<mlir::Operation *> &okSet)
      : block(block), memGraph(memGraph), okSet(okSet) {}
};

bool CycleDfs::operator()(Operation *cur) {
  if (okSet.contains(cur)) {
    LOG_DEBUG("[CycleDfs] Cycle found, back to okSet: " << cur->getName());
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
    if (okSet.contains(userInBlock)) {
      LOG_DEBUG("[CycleDfs] Cycle found, userInBlock in okSet: " << userInBlock->getName());
      return true;
    }
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    int userBlockId = bm.getBlockIdByOp(userInBlock);
    if (userBlockId == -1) {
      if (dfs(userInBlock)) {
        return true;
      }
    } else {
      for (auto *nx : bm.getOpsByBlockId(userBlockId)) {
        if (dfs(nx)) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * @brief 检测将 alloc、fill 和 parentIf 统一到目标 block_id 是否会形成环
 *
 * 此函数用于判断将 memref.alloc、scf.if 内的 linalg.fill 和对应的 scf.if
 * 合并到同一个 block_id 是否会破坏依赖图的无环性质。
 *
 * @param allocOp 要检测的 memref.alloc 操作
 * @param fillInfo 包含 fillOp 和 parentIf 的结构
 * @param memGraph 内存依赖图，用于分析 RAW/WAW/WAR 依赖
 * @param targetBlockId 统一后的目标 block_id
 * @return bool 如果统一会形成环返回 true，否则返回 false
 */
static bool willCreateCycle(memref::AllocOp allocOp, FillInfo &fillInfo,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            int targetBlockId) {
  auto *block = allocOp->getBlock();
  LOG_DEBUG("[Cycle] Starting detection with targetBlockId: " << targetBlockId);

  // 构建"安全集合"：包含 targetBlockId 对应的所有 ops
  // 如果 DFS 能从某操作回到安全集合中的任意节点，说明统一会形成环
  llvm::DenseSet<Operation *> okSet;

  auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();

  // 将 targetBlockId 对应的所有 ops 加入 okSet
  LOG_DEBUG("[Cycle] targetBlockId: " << targetBlockId << " has ops:");
  for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
    okSet.insert(op);
    LOG_DEBUG("insert  - " << *op);
  }

  // 将三个操作加入 okSet
  LOG_DEBUG("[Cycle] Adding three ops to okSet:");
  LOG_DEBUG("  - allocOp: " << *allocOp.getOperation());
  okSet.insert(allocOp.getOperation());
  LOG_DEBUG("  - fillOp: " << *fillInfo.fillOp.getOperation());
  okSet.insert(fillInfo.fillOp.getOperation());
  LOG_DEBUG("  - parentIf: " << *fillInfo.parentIf.getOperation());
  okSet.insert(fillInfo.parentIf.getOperation());

  // 初始化 DFS 检测器
  CycleDfs dfs(block, memGraph, okSet);
  bool hasCycle = false;

  LOG_DEBUG("[Cycle] Starting DFS from okSet, size=" << okSet.size());

  // 遍历安全集合中的每个操作作为起点，检测是否存在路径回到安全集合
  for (mlir::Operation *okOp : okSet) {
    LOG_DEBUG("[Cycle] DFS from okOp: " << *okOp);
    SmallVector<Operation *> allusers;
    allusers.append(okOp->getUsers().begin(), okOp->getUsers().end());
    // 考虑内存依赖带来的隐式执行顺序
    for (auto *memUser : memGraph.getExecAfter(okOp)) {
      allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
      auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
      // 用户已在安全集合中，跳过（不会形成环）
      if (okSet.contains(userInBlock)) {
        continue;
      }
      LOG_DEBUG("[Cycle]   checking user: " << *userInBlock);
      auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
      int userBlockId = bm.getBlockIdByOp(userInBlock);
      // 用户不在任何 block 中，直接 DFS 该操作
      if (userBlockId == -1) {
        dfs.clear();
        if (dfs(userInBlock)) {
          LOG_DEBUG("[Cycle] Cycle detected from user: " << *userInBlock);
          hasCycle = true;
          break;
        }
      } else {
        // 用户属于某个 block_id，DFS 该 block 内所有操作
        LOG_DEBUG("[Cycle]   user in blockId: " << userBlockId);
        for (auto *userOp : bm.getOpsByBlockId(userBlockId)) {
          dfs.clear();
          if (dfs(userOp)) {
            LOG_DEBUG("[Cycle] Cycle detected from userOp: " << *userOp);
            hasCycle = true;
            break;
          }
        }
      }
    }
    if (hasCycle) {
      break;
    }
  }

  LOG_DEBUG("[Cycle] DFS complete, hasCycle=" << (hasCycle ? "true" : "false"));

  return hasCycle;
}

static std::optional<int> lookupOpBlockId(Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        return attr.getInt();
    return -1;
}

static void markOpBlockId(Operation *op, int blockId) {
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    bm.updateBlockId(op, blockId);
    // op->setAttr("ssbuffer.block_id", mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), blockId));
}

/**
 * @brief Collect direct users of alloc result
 *
 * This function collects all operations that directly use the alloc result,
 * excluding linalg.fill operations because linalg.fill uses BlockArgument
 * (i.e., the outs parameter) rather than SSA value dependency.
 *
 * @param allocResult The result value of memref.alloc
 * @return SmallVector<Operation*> List of direct user operations
 *
 * @note linalg.fill is a DestinationStyleOp where:
 *       - ins(%v : f16) is the input value
 *       - outs(%alloc : memref) is the target memory location (BlockArgument)
 *       Therefore, linalg.fill does not appear in allocResult.getUsers().
 */
static SmallVector<Operation *> collectDirectUsers(Value allocResult) {
  SmallVector<Operation *> directUsers;
  for (Operation *user : allocResult.getUsers()) {
    if (!isa<linalg::FillOp>(user)) {
      LOG_DEBUG("[collectDirectUsers] Found direct user: " << *user); //user->getName()
      directUsers.push_back(user);
    }
  }
  return directUsers;
}

/**
 * @brief Get common block_id from a list of operations
 *
 * Checks whether all operations have the same block_id.
 * If they are the same, returns that block_id; otherwise returns std::nullopt.
 *
 * @param ops List of operations to check
 * @return std::optional<int> Returns common block_id if all are the same,
 *         otherwise returns std::nullopt
 */
static std::optional<int> getCommonBlockId(ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return std::nullopt;
  }

  int commonId = -1;
  for (Operation *op : ops) {
    auto optBlockId = lookupOpBlockId(op);
    if (!optBlockId.has_value()) {
      LOG_DEBUG("[getCommonBlockId] op without block_id: " << *op);
      return std::nullopt;
    }
    int blockId = *optBlockId;

    if (commonId == -1) {
      commonId = blockId;
    } else if (commonId != blockId) {
      LOG_DEBUG("[getCommonBlockId] block_id mismatch: " << *op << " has " << blockId << " != " << commonId);
      return std::nullopt;
    }
  }
  return commonId;
}

/**
 * @brief Find linalg.fill operation that uses alloc as outs inside scf.if
 *
 * This function searches for linalg.fill operations that satisfy:
 * 1. Use the given alloc result as its outs parameter
 * 2. Located inside an scf.if operation (then branch only)
 * 3. The scf.if has no else region (withElseRegion=false)
 *
 * @param allocResult The alloc result value to search for
 * @return FillInfo Structure containing fillOp and parentIf if found
 */
static FillInfo findFillOpInSCFIf(Value allocResult) {
  FillInfo info;
  for (Operation *user : allocResult.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp) {
      continue;
    }

    auto parentIf = fillOp->getParentOfType<scf::IfOp>();
    if (!parentIf) {
      continue;
    }

    if (!parentIf.getElseRegion().empty()) {
      continue;
    }

    Block *parentBlock = fillOp->getBlock();
    if (parentBlock != &parentIf.getThenRegion().front()) {
      continue;
    }

    if (fillOp.getDpsInits()[0] == allocResult) {
      // LOG_DEBUG("[findFillOpInSCFIf] Found fill in scf.if: " << fillOp.getOperation()->getName()
      //            << ", parentIf: " << parentIf.getOperation()->getName()
      //            << ", parentIf block_id=" << bm.getBlockIdByOp(parentIf.getOperation()));
      info.fillOp = fillOp;
      info.parentIf = parentIf;
      return info;
    }
  }
  LOG_DEBUG("[findFillOpInSCFIf] No fill found for alloc");
  return info;
}

/**
 * @brief Check if scf.if needs to be split
 *
 * Determines whether the scf.if operation containing linalg.fill needs to be split.
 * Split is needed when the if branch contains multiple operations
 * (not just linalg.fill).
 *
 * @param info FillInfo structure containing fillOp and parentIf
 * @return bool Returns true if split is needed, false otherwise
 *
 * @note Split logic:
 *       - If branch only has linalg.fill (+ scf.yield terminator), no split needed
 *       - If branch has other operations besides linalg.fill, split needed
 */
static bool needsSplitIf(const FillInfo &info) {
  if (!info.fillOp || !info.parentIf) {
    return false;
  }

  Block *fillBlock = info.fillOp->getBlock();
  int opCount = 0;
  for (auto &op : fillBlock->without_terminator()) {
    (void)op;
    opCount++;
  }
  return opCount > 1;
}

/**
 * @brief Split scf.if into two separate scf.if blocks
 *
 * When an scf.if branch contains multiple operations (linalg.fill + other ops),
 * this function splits it into two scf.if blocks:
 * - One containing only linalg.fill (will be unified)
 * - One containing other operations (keeps original block_id)
 *
 * @param info FillInfo structure containing fillOp and parentIf
 * @return FillInfo Updated FillInfo pointing to the new fill-only scf.if
 *
 * @note Split pattern:
 *       Before:
 *         scf.if %cond {
 *           linalg.fill {block_id=8} ins(%v) outs(%alloc)  // keep
 *           arith.addf {block_id=12} %x, %y               // move to new scf.if
 *         } {hivm.unlikely_condition}
 *
 *       After:
 *         scf.if %cond {
 *           linalg.fill {block_id=8} ins(%v) outs(%alloc)  // keep
 *         } {hivm.unlikely_condition}
 *
 *         scf.if %cond {
 *           arith.addf {block_id=12} %x, %y               // new scf.if
 *         } {hivm.unlikely_condition}
 */
static FillInfo splitSCFIfIfNeeded(FillInfo &info) {
  if (!needsSplitIf(info)) {
    return info;
  }

  Block *originalBlock = info.fillOp->getBlock();
  Operation *fillOp = info.fillOp.getOperation();
  scf::IfOp originalIf = info.parentIf;
  Value cond = originalIf.getCondition();
  Location loc = originalIf.getLoc();

  SmallVector<Operation *> otherOps;
  for (auto &op : originalBlock->without_terminator()) {
    if (&op != fillOp) {
      otherOps.push_back(&op);
    }
  }

  if (otherOps.empty()) {
    return info;
  }

  DictionaryAttr originalAttrs = originalIf->getAttrDictionary();

  OpBuilder builder(originalIf);

  fillOp->moveBefore(originalIf.getOperation()->getNextNode());
  builder.setInsertionPointAfter(fillOp);

  auto newFillIf = builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);
  if (originalAttrs) {
    for (auto attr : originalAttrs) {
      newFillIf->setAttr(attr.getName(), attr.getValue());
    }
  }

  fillOp->moveBefore(newFillIf.getThenRegion().front().getTerminator());

  // LOG_DEBUG("[splitSCFIfIfNeeded] Created new scf.if: " << newFillIf.getOperation()->getName()
  //            << ", block_id=" << bm.getBlockIdByOp(newFillIf.getOperation()));
  info.parentIf = newFillIf;
  return info;
}

/**
 * @brief Try to unify block_id for a single alloc operation
 *
 * @param allocOp The memref.alloc operation to process
 * @param memGraph Memory dependence graph for cycle detection
 * @return bool Returns true if unification was performed, false otherwise
 */
static bool tryUnifyForAlloc(memref::AllocOp allocOp, const CVPipeline::MemoryDependenceGraph &memGraph) {
  // Step1: Collect direct users (excluding linalg.fill)
  Value allocResult = allocOp.getResult();
  SmallVector<Operation *> directUsers = collectDirectUsers(allocResult);
  if (directUsers.empty()) {
    return false;
  }

  // Step2: Check if all direct users have the same block_id
  std::optional<int> targetBlockId = getCommonBlockId(directUsers);
  LOG_DEBUG("[getCommonBlockId]!!!!!!!!!!!!!!!!!!" << targetBlockId);
  if (!targetBlockId.has_value()) {
    return false;
  }

  // Step3: Find linalg.fill inside scf.if that uses this alloc
  FillInfo fillInfo = findFillOpInSCFIf(allocResult);
  if (!fillInfo.fillOp) {
    return false;
  }

  // LOG_DEBUG("[tryUnifyForAlloc] After findFillOpInSCFIf: fill="
  //            << bm.getBlockIdByOp(fillInfo.fillOp.getOperation())
  //            << ", parentIf=" << bm.getBlockIdByOp(fillInfo.parentIf.getOperation()));

  // Step4: Split if scf.if contains multiple operations
  if (needsSplitIf(fillInfo)) {
    fillInfo = splitSCFIfIfNeeded(fillInfo);
    // LOG_DEBUG("[tryUnifyForAlloc] After splitSCFIfIfNeeded: fill="
    //           << bm.getBlockIdByOp(fillInfo.fillOp.getOperation())
    //           << ", parentIf=" << bm.getBlockIdByOp(fillInfo.parentIf.getOperation()));
  }

  // Step5: Cycle detection - check if unification would create cycle
  if (willCreateCycle(allocOp, fillInfo, memGraph, *targetBlockId)) {
    LOG_DEBUG("[Cycle detection] find cycle!");
    return false;
  }

  // Step6: Unify block_id of alloc, scf.if, and linalg.fill
  markOpBlockId(allocOp, *targetBlockId);
  markOpBlockId(fillInfo.fillOp, *targetBlockId);
  markOpBlockId(fillInfo.parentIf, *targetBlockId);
  return true;
}

} // anonymous namespace

class UnifyAllocBlockPass
    : public PassWrapper<UnifyAllocBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnifyAllocBlockPass)

  UnifyAllocBlockPass() = default;

  StringRef getArgument() const override { return "unify-alloc-block"; }

  StringRef getDescription() const override {
    return "Unify block_id for memref.alloc, scf.if with linalg.fill, and "
           "memref.subview operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);

    int processedCount = 0;
    int successCount = 0;

    module.walk([&](memref::AllocOp allocOp) {
      processedCount++;
      if (tryUnifyForAlloc(allocOp, memGraph)) {
        successCount++;
      }
    });

    LOG_DEBUG("[UnifyAllocBlockPass] Processed: " << processedCount << " allocs, unified: " << successCount);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnifyAllocBlockPass() {
  return std::make_unique<UnifyAllocBlockPass>();
}

void registerUnifyAllocBlockPass() {
  PassRegistration<UnifyAllocBlockPass> reg;
}

} // namespace triton
} // namespace mlir