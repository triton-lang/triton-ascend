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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CloneOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/OuterEventSchedule.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/SplittedIfWrapSsbufIf.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfoLegacy.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimesLegacy.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOpsLegacy.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <memory>

static constexpr const char *DEBUG_TYPE = "AddControlFlowCondition";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Check if the module should be skipped for control flow condition processing
static LogicalResult verifyControlFlowPrerequisites(ModuleOp module) {
  // Check if scopeOp has ssbuffer.skip
  bool hasSkipAttr = false;
  module.walk([&](Operation *op) {
    auto scopeOp = dyn_cast<scope::ScopeOp>(op);
    if (!scopeOp) {
      return;
    }
    if (scopeOp->hasAttr("ssbuffer.skip")) {
      hasSkipAttr = true;
    }
  });
  if (hasSkipAttr) {
    LDBG("scopeOp has ssbuffer.skip, skip processing.");
    return failure();
  }

  return success();
}

static bool isOuterVectorPayload(Operation *op) {
  auto loop = op->getParentOfType<scf::ForOp>();
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  auto core =
      scopeOp ? scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type")
              : hivm::TCoreTypeAttr();
  return loop && loop->hasAttr(CVPipeline::kOuterEventControlCount) && core &&
         core.getTcoretype() == hivm::TCoreType::VECTOR;
}

// Keep a full tile multiplication inside its mutually exclusive consumers.
// This lets the backend fuse score scaling with mask/select and reduction.
static void sinkOuterScoreMultiplication(ModuleOp module) {
  SmallVector<arith::MulFOp> candidates;
  module.walk([&](arith::MulFOp mul) {
    auto type = dyn_cast<RankedTensorType>(mul.getType());
    if (isOuterVectorPayload(mul) && type && type.getRank() == 2 &&
        !mul.getResult().use_empty())
      candidates.push_back(mul);
  });

  for (arith::MulFOp mul : candidates) {
    Block *block = mul->getBlock();
    auto branch = dyn_cast_or_null<scf::IfOp>(
        block->findAncestorOpInBlock(**mul.getResult().getUsers().begin()));
    if (!branch || branch.getElseRegion().empty())
      continue;

    SmallVector<OpOperand *> thenUses, elseUses;
    bool feedsReduction = false;
    for (OpOperand &use : mul.getResult().getUses()) {
      Operation *owner = use.getOwner();
      if (branch.getThenRegion().isAncestor(owner->getParentRegion()))
        thenUses.push_back(&use);
      else if (branch.getElseRegion().isAncestor(owner->getParentRegion()))
        elseUses.push_back(&use);
      feedsReduction |= isa<linalg::ReduceOp>(owner);
    }
    if (!feedsReduction || thenUses.empty() || elseUses.empty() ||
        thenUses.size() + elseUses.size() !=
            static_cast<size_t>(std::distance(mul.getResult().use_begin(),
                                              mul.getResult().use_end())))
      continue;

    auto sink = [&](Block &dest, ArrayRef<OpOperand *> uses) {
      OpBuilder builder = OpBuilder::atBlockBegin(&dest);
      Value result = builder.clone(*mul.getOperation())->getResult(0);
      for (OpOperand *use : uses)
        use->set(result);
    };
    sink(branch.getThenRegion().front(), thenUses);
    sink(branch.getElseRegion().front(), elseUses);
    mul.erase();
  }
}

// Keep row recurrence initialization in the consuming VF instead of forcing
// a tensor-storage join at every event. A select preserves the chosen bits.
static void selectOuterRowSeeds(ModuleOp module) {
  SmallVector<scf::IfOp> candidates;
  module.walk([&](scf::IfOp branch) {
    if (!isOuterVectorPayload(branch) || branch.getNumResults() == 0 ||
        branch.getElseRegion().empty() ||
        !llvm::hasSingleElement(branch.getElseRegion().front()))
      return;
    bool valid = llvm::all_of(branch.getResultTypes(), [](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      return tensor && tensor.hasStaticShape() && tensor.getRank() == 1 &&
             tensor.getNumElements() > 0 && tensor.getElementType().isF32();
    });
    valid &= llvm::all_of(branch.getThenRegion().front(), [](Operation &op) {
      return isa<tensor::EmptyOp, arith::ConstantOp, scf::YieldOp>(op) ||
             (isa<linalg::FillOp>(op) && isMemoryEffectFree(&op));
    });
    for (Value value : branch.thenYield().getOperands()) {
      auto fill = value.getDefiningOp<linalg::FillOp>();
      if (!fill || !fill.hasPureTensorSemantics() ||
          !fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
        valid = false;
        break;
      }
      auto scalar = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      if (!scalar || !isa<FloatAttr>(scalar.getValue()))
        valid = false;
    }
    if (valid)
      candidates.push_back(branch);
  });
  for (scf::IfOp branch : candidates) {
    OpBuilder builder(branch);
    SmallVector<Value> replacements;
    for (auto [initial, previous] :
         llvm::zip(branch.thenYield().getOperands(),
                   branch.elseYield().getOperands())) {
      auto type = cast<RankedTensorType>(initial.getType());
      auto fill = initial.getDefiningOp<linalg::FillOp>();
      auto scalar = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      auto dense =
          DenseElementsAttr::get(type, ArrayRef<Attribute>{scalar.getValue()});
      Value seed =
          builder.create<arith::ConstantOp>(branch.getLoc(), type, dense);
      auto maskType =
          RankedTensorType::get(type.getShape(), builder.getI1Type());
      Value mask = builder.create<tensor::SplatOp>(branch.getLoc(), maskType,
                                                   branch.getCondition());
      replacements.push_back(builder.create<arith::SelectOp>(
          branch.getLoc(), mask, seed, previous));
    }
    branch.replaceAllUsesWith(replacements);
    branch.erase();
  }
}

// Publish a packed tile from its producing branch. Returning it through an
// scf.if storage join otherwise adds a full UB-to-UB copy before the L1 copy.
static void sinkOuterTilePublication(ModuleOp module) {
  SmallVector<scf::IfOp> branches;
  module.walk([&](scf::IfOp branch) { branches.push_back(branch); });
  for (scf::IfOp branch : llvm::reverse(branches)) {
    if (!isOuterVectorPayload(branch) || branch.getElseRegion().empty())
      continue;
    Operation *wait = branch->getNextNode();
    Operation *copy = wait ? wait->getNextNode() : nullptr;
    Operation *publish = copy ? copy->getNextNode() : nullptr;
    if (!publish)
      continue;
    for (auto indexed : llvm::enumerate(branch.getResults())) {
      unsigned index = indexed.index();
      Value result = indexed.value();
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
          !type.getElementType().isBF16() || result.use_empty())
        continue;
      unsigned copies = 0;
      auto valid = [&](Operation *root, unsigned kind) {
        unsigned effects = 0;
        return !root->walk([&](Operation *op) {
                      if (auto guard = dyn_cast<scf::IfOp>(op))
                        return guard.getNumResults() == 0
                                   ? WalkResult::advance()
                                   : WalkResult::interrupt();
                      if (auto yield = dyn_cast<scf::YieldOp>(op))
                        return yield.getNumOperands() == 0
                                   ? WalkResult::advance()
                                   : WalkResult::interrupt();
                      if ((kind == 0 && isa<hivm::SyncBlockWaitOp>(op)) ||
                          (kind == 2 && isa<hivm::SyncBlockSetOp>(op))) {
                        ++effects;
                        return WalkResult::advance();
                      }
                      if (kind == 1) {
                        if (auto transfer = dyn_cast<hivm::CopyOp>(op)) {
                          auto dest =
                              dyn_cast<MemRefType>(transfer.getDst().getType());
                          auto space =
                              dest ? dyn_cast_or_null<hivm::AddressSpaceAttr>(
                                         dest.getMemorySpace())
                                   : hivm::AddressSpaceAttr();
                          if (transfer.getSrc() != result ||
                              transfer.getNumResults() || !space ||
                              space.getAddressSpace() != hivm::AddressSpace::L1)
                            return WalkResult::interrupt();
                          ++copies;
                          ++effects;
                          return WalkResult::advance();
                        }
                      }
                      return isMemoryEffectFree(op) && op->getNumRegions() == 0
                                 ? WalkResult::advance()
                                 : WalkResult::interrupt();
                    })
                    .wasInterrupted() &&
               effects > 0;
      };
      if (!valid(wait, 0) || !valid(copy, 1) || !valid(publish, 2) ||
          copies != static_cast<unsigned>(
                        std::distance(result.use_begin(), result.use_end())))
        continue;
      bool otherResultUsed = false;
      for (Operation *root : {wait, copy, publish})
        root->walk([&](Operation *op) {
          for (Value operand : op->getOperands())
            if (auto definition = dyn_cast<OpResult>(operand))
              if (definition.getOwner() == branch && operand != result)
                otherResultUsed = true;
        });
      if (otherResultUsed)
        continue;
      for (Region *region :
           {&branch.getThenRegion(), &branch.getElseRegion()}) {
        auto yield = cast<scf::YieldOp>(region->front().getTerminator());
        OpBuilder builder(yield);
        IRMapping mapping;
        mapping.map(result, yield.getOperand(index));
        for (Operation *root : {wait, copy, publish})
          builder.clone(*root, mapping);
      }
      publish->erase();
      copy->erase();
      wait->erase();
      break;
    }
  }
}

void AddControlFlowConditionPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter add controlflow condition pass.\n");
  LDBG("before AddControlFlowCondition:");
  LLVM_DEBUG(module.dump());

  if (failed(verifyControlFlowPrerequisites(module))) {
    return;
  }

  PassManager pm(&getContext(), module.getOperationName());
  ControlFlowConditionInfo info;

  // Step0: Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  pm.addPass(createCloneOpsPass());

  // Step1: Process shared iter_args in for ops to eliminate arg sharing across
  // block_ids
  std::unique_ptr<ProcessArgsPass> processArgsPass(new ProcessArgsPass());
  processArgsPass->setConditionInfo(&info);
  pm.addPass(std::move(processArgsPass));

  // Step2: Create if ops based on block_id
  std::unique_ptr<CreateIfOpsPass> createIfOpsPass(new CreateIfOpsPass());
  createIfOpsPass->setConditionInfo(&info);
  pm.addPass(std::move(createIfOpsPass));

  // Step3: Initialize crossCoreDependentMap, intraCoreDependentMap, and build
  // if block DAG
  std::unique_ptr<InitDependentMapPass> initDependentMapPass(
      new InitDependentMapPass());
  initDependentMapPass->setConditionInfo(&info);
  pm.addPass(std::move(initDependentMapPass));

  // The kernels of legacyKernel keep the control flow conditions of the
  // upstream implementation: they are not rewritten by the normal preload
  // scheme. Every other kernel goes through the normal preload scheme.
  // A module with an outer main loop keeps the upstream conditions too, see
  // the useLegacyConditions computation below.
  constexpr llvm::StringLiteral legacyKernel[] = {
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu_kernel",
      "kernel_da_bwd_q_u",
      "kernel_sdpa_bwd_q",
      "_sdpa_infer_kernel",
      "paged_prefill_kernel",
      "flex_attention_backward_dkdv_kernel",
      "flex_attention_backward_dq_kernel",
      "flex_attention_kernel",
      "kernel_sda_bwd_q_down",
      "kernel_sda_bwd_q_up",
      "paged_decode_fd_kernel",
      "_swa_bwd_dkdv_kernel",
      "kernel_da_bwd_d",
      "kernel_da_bwd_kv_ul",
      "kernel_da_bwd_kv_ur",
      "_hstu_attn_fwd",
      "kernel_sda_bwd_d",
      "kernel_sda_bwd_kv_left",
      "kernel_sda_bwd_kv_right",
      "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
      "chunk_bwd_kernel_dv_local",
      "fwd_kernel",
      "merge_fwd_bwd_kernel",
      "parallel_path_fwd_kernel",
      "_jagged_dense_flash_attention_bwd_dv_db_dq_kernel",
      "parallel_nsa_compression_bwd_kernel_dq",
      "parallel_nsa_compression_bwd_kernel_dkv",
      "kernel_sdpa_bwd_kv"};
  // The outer main loop scheduling feature is built on the control flow
  // scheme of the upstream implementation: its static ordinal schedule and its
  // native credit handshake are not modelled by the normal preload scheme.
  // A module with an outer main loop keeps the upstream conditions.
  bool useLegacyConditions = CVPipeline::hasOuterMainLoop(module);
  for (auto &op : module.getOps()) {
    auto funcOp = llvm::dyn_cast<func::FuncOp>(&op);
    if (funcOp && llvm::is_contained(legacyKernel, funcOp.getSymName())) {
      useLegacyConditions = true;
      break;
    }
  }

  if (useLegacyConditions) {
    // Step4: Update for ops with block counters and inner dependency
    // conditions, and insert PIPE_S inter-core synchronization
    std::unique_ptr<UpdateLoopOpsPassLegacy> legacyUpdateLoopOpsPass(
        new UpdateLoopOpsPassLegacy());
    legacyUpdateLoopOpsPass->setConditionInfo(&info);
    pm.addPass(std::move(legacyUpdateLoopOpsPass));

    // Step5:Update the conditions of ifOp based on the intraCoreDependentMap
    // and crossCoreDependentMap
    std::unique_ptr<UpdateConditionInfoPassLegacy> legacyUpdatePass(
        new UpdateConditionInfoPassLegacy());
    legacyUpdatePass->setConditionInfo(&info);
    pm.addPass(std::move(legacyUpdatePass));

    // Step6: Update for loop iteration times based on intraCoreDependentMap
    std::unique_ptr<UpdateLoopIterTimesPassLegacy>
        legacyUpdateLoopIterTimesPass(new UpdateLoopIterTimesPassLegacy());
    legacyUpdateLoopIterTimesPass->setConditionInfo(&info);
    pm.addPass(std::move(legacyUpdateLoopIterTimesPass));
  } else {
    // Step4: Update for ops with block counters and inner dependency
    // conditions, and insert PIPE_S inter-core synchronization
    std::unique_ptr<UpdateLoopOpsPass> updateLoopOpsPass(
        new UpdateLoopOpsPass());
    updateLoopOpsPass->setConditionInfo(&info);
    pm.addPass(std::move(updateLoopOpsPass));

    // Step5:Update the conditions of ifOp based on the intraCoreDependentMap
    // and crossCoreDependentMap
    auto updatePass = std::make_unique<UpdateConditionInfoPass>();
    updatePass->setConditionInfo(&info);
    pm.addPass(std::move(updatePass));

    // Step6: Update for loop iteration times based on intraCoreDependentMap
    std::unique_ptr<UpdateLoopIterTimesPass> updateLoopIterTimesPass(
        new UpdateLoopIterTimesPass());
    updateLoopIterTimesPass->setConditionInfo(&info);
    pm.addPass(std::move(updateLoopIterTimesPass));
  }

  // Step7: Use the splitted_if condition to wrap each ssbuffer.if.
  std::unique_ptr<SplittedIfWrapSsbufIfPass> splittedIfWrapSsbufIfPass(
      new SplittedIfWrapSsbufIfPass());
  splittedIfWrapSsbufIfPass->setConditionInfo(&info);
  pm.addPass(std::move(splittedIfWrapSsbufIfPass));

  if (failed(runPipeline(pm, module))) {
    LDBG("Pass failed!");
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  } else if (CVPipeline::hasOuterMainLoop(module)) {
    issueOuterProducerBeforeWait(module);
    sinkOuterScoreMultiplication(module);
    selectOuterRowSeeds(module);
    sinkOuterTilePublication(module);
  }

  LDBG("Exit add controlflow condition pass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass() {
  return std::make_unique<AddControlFlowConditionPass>();
}

void registerAddControlFlowConditionPasses() {
  registerPass(createCloneOpsPass);
  registerPass(createCreateIfOpsPass);
  registerPass(createProcessArgsPass);
  registerPass(createUpdateLoopOpsPass);
  registerPass(createUpdateLoopIterTimesPass);
  registerPass(createAddControlFlowConditionPass);
}
} // namespace triton
} // namespace mlir
