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

static constexpr llvm::StringLiteral containedFunc[]{
    "chunk_fwd_mesa_cg_dim64_kernel",
    "pre_process_bwd_kernel_merged",
    "pre_process_fwd_kernel_merged",
    "fused_chunk_ttt_linear_bwd_kernel_h",
};

static LogicalResult isInterceptedModule(ModuleOp module) {
  bool intercepted = false;

  module.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(containedFunc, funcOp.getSymName())) {
      return WalkResult::advance();
    }
    intercepted = true;
    return WalkResult::interrupt();
  });

  if (!intercepted) {
    return success();
  }

  return failure();
}

// Check if a value is a tensor-type iter_arg and return its index, -1
// otherwise.
static int getTensorIterArgIndex(Value v, scf::ForOp forOp) {
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    if (v == forOp.getRegionIterArgs()[i]) {
      if (isa<RankedTensorType>(forOp.getRegionIterArgs()[i].getType())) {
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

// Collect block info for all tensor-type iter_args in the forOp body
static llvm::DenseMap<unsigned, TensorArgBlockInfo>
collectTensorArgBlockInfo(scf::ForOp forOp) {
  llvm::DenseMap<unsigned, TensorArgBlockInfo> result;

  Block *body = forOp.getBody();
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
      int argIdx = getTensorIterArgIndex(operand.get(), forOp);
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
      LDBG("[INFO]: Found tensor iter_arg using in multi block_ids!\n");
      return true;
    }
  }
  return false;
}

// Check if the first block_id differs from the block_id of yield's defining op.
static bool checkUseUpdateMismatch(
    scf::ForOp forOp,
    const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo) {
  Block *body = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    if (!isa<RankedTensorType>(forOp.getRegionIterArgs()[i].getType())) {
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
      LDBG("[INFO]: Found tensor iter_arg using and updating in different "
           "block_ids!\n");
      return true;
    }
  }
  return false;
}

// Main check function that combines both conditions
static bool hasTensorArgInDifferentBlockIds(scf::ForOp forOp) {
  auto argBlockInfo = collectTensorArgBlockInfo(forOp);
  return checkMultiBlockUse(argBlockInfo) ||
         checkUseUpdateMismatch(forOp, argBlockInfo);
}

} // namespace

bool checkTensorArgsInMainLoop(ModuleOp module) {
  bool shouldReturn = false;

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }

    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      return WalkResult::advance();
    }

    if (hasTensorArgInDifferentBlockIds(forOp)) {
      shouldReturn = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return shouldReturn;
}

// Check VECTOR main_loop forOps: if a tensor arith.subf op has both operands
// from linalg.broadcast ops sharing the same source, and the subf op's
// block_id differs from the source op's block_id, fallback.
static bool checkSubfBroadcastMismatchInVectorMainLoop(ModuleOp module) {
  bool shouldFallback = false;

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      return WalkResult::advance();
    }

    scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) {
      return WalkResult::advance();
    }
    auto tcoreAttr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
    if (!tcoreAttr || tcoreAttr.getTcoretype() != hivm::TCoreType::VECTOR) {
      return WalkResult::advance();
    }

    Block *body = forOp.getBody();
    if (!body) {
      return WalkResult::advance();
    }

    for (Operation &bodyOp : body->without_terminator()) {
      auto subfOp = dyn_cast<arith::SubFOp>(&bodyOp);
      if (!subfOp || !isa<RankedTensorType>(subfOp.getType())) {
        continue;
      }

      Operation *lhsDef = subfOp.getLhs().getDefiningOp();
      Operation *rhsDef = subfOp.getRhs().getDefiningOp();
      if (!lhsDef || !rhsDef) {
        continue;
      }

      auto lhsBroadcast = dyn_cast<linalg::BroadcastOp>(lhsDef);
      auto rhsBroadcast = dyn_cast<linalg::BroadcastOp>(rhsDef);
      if (!lhsBroadcast || !rhsBroadcast) {
        continue;
      }

      Value lhsInput = lhsBroadcast.getInput();
      Value rhsInput = rhsBroadcast.getInput();
      if (lhsInput != rhsInput) {
        continue;
      }

      Operation *sourceOp = lhsInput.getDefiningOp();
      if (!sourceOp) {
        continue;
      }

      auto subfBlockIdAttr =
          subfOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      auto sourceBlockIdAttr =
          sourceOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      if (!subfBlockIdAttr || !sourceBlockIdAttr) {
        continue;
      }

      if (subfBlockIdAttr.getInt() != sourceBlockIdAttr.getInt()) {
        LDBG("[INFO]: Found subf and broadcast source with different "
             "block_ids in VECTOR main_loop!\n");
        shouldFallback = true;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return shouldFallback;
}

void AnalyzeArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeArgs:\n" << module << "\n");

  if (failed(isInterceptedModule(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }

  if (checkTensorArgsInMainLoop(module) &&
      checkSubfBroadcastMismatchInVectorMainLoop(module)) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
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
