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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-args-in-forOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;

namespace {

// Check if a value is a tensor-type iter_arg and return its index, -1
// otherwise.
static int getTensorIterArgIndex(Value v, ArrayRef<Value> iterArgs) {
  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    if (v == iterArgs[i]) {
      if (isa<RankedTensorType>(iterArgs[i].getType())) {
        return i;
      }
    }
  }
  return -1;
}

// Data collected for each tensor iter_arg: first block_id that uses it, and set
// of all block_ids
struct TensorArgBlockInfo {
  int firstBlockId = -1;
  llvm::DenseSet<int> blockIds;
};

// Collect block info for all tensor-type iter_args in the given body.
static llvm::DenseMap<unsigned, TensorArgBlockInfo>
collectTensorArgBlockInfo(ArrayRef<Value> iterArgs, Block *body) {
  llvm::DenseMap<unsigned, TensorArgBlockInfo> result;

  if (!body) {
    return result;
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) {
      continue;
    }
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      int argIdx = getTensorIterArgIndex(operand.get(), iterArgs);
      if (argIdx >= 0) {
        auto &info = result[argIdx];
        info.blockIds.insert(blockId);
        if (info.firstBlockId < 0) {
          info.firstBlockId = blockId;
        }
      }
    }
  }

  return result;
}

// Check if any tensor-type iter_arg appears in different block_ids.
static bool checkMultiBlockUse(
    const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo) {
  for (auto &p : argBlockInfo) {
    if (p.second.blockIds.size() > 1) {
      return true;
    }
  }
  return false;
}

// Check if the first block_id differs from the block_id of yield's defining op.
static bool checkUseUpdateMismatch(
    ArrayRef<Value> iterArgs, Block *body,
    const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo) {
  if (!body) {
    return false;
  }
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    if (!isa<RankedTensorType>(iterArgs[i].getType())) {
      continue;
    }

    auto it = argBlockInfo.find(i);
    if (it == argBlockInfo.end()) {
      continue;
    }

    Operation *defOp = yieldOp.getOperand(i).getDefiningOp();
    if (!defOp) {
      continue;
    }

    auto defBlockIdAttr =
        defOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!defBlockIdAttr) {
      continue;
    }

    if (it->second.firstBlockId != defBlockIdAttr.getInt()) {
      return true;
    }
  }
  return false;
}

// Main check function that combines both conditions
static bool hasTensorArgInDifferentBlockIds(ArrayRef<Value> iterArgs,
                                            Block *body) {
  auto argBlockInfo = collectTensorArgBlockInfo(iterArgs, body);
  return checkMultiBlockUse(argBlockInfo) ||
         checkUseUpdateMismatch(iterArgs, body, argBlockInfo);
}

} // namespace

bool checkTensorArgsInMainLoop(ModuleOp module) {
  bool shouldReturn = false;

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }

    SmallVector<Value> iterArgs;
    Block *body = nullptr;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      for (auto arg : forOp.getRegionIterArgs())
        iterArgs.push_back(arg);
      body = forOp.getBody();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      // For whileOp, analyze the do-region (after body), where iter_args are
      // used and updated by scf.yield.
      for (auto arg : whileOp.getRegionIterArgs())
        iterArgs.push_back(arg);
      body = whileOp.getAfterBody();
    } else {
      return WalkResult::advance();
    }

    if (hasTensorArgInDifferentBlockIds(iterArgs, body)) {
      shouldReturn = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return shouldReturn;
}

void AnalyzeArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeArgs:\n" << module << "\n");

  BufferCountManager bufferCountMgr(module);
  int intraBufNum = bufferCountMgr.getBufferCountByType(
      BufferCountManager::DepType::IntraCore);
  int interBufNum = bufferCountMgr.getBufferCountByType(
      BufferCountManager::DepType::InterCore);

  if (intraBufNum == 3 && interBufNum == 2 &&
      checkTensorArgsInMainLoop(module)) {
    LDBG("[INFO]: Found tensor iter_args dependency in mainloop!");
    CVPipeline::setFallbackAttr(module,
                                CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED);
    return;
  }

  LDBG("After AnalyzeArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeArgsPass() {
  return std::make_unique<AnalyzeArgsPass>();
}

} // namespace triton
} // namespace mlir
