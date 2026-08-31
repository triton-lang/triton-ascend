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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "exp-subf-pattern";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace arith;
using namespace math;

namespace {

static bool matchExpFromSubf(arith::SubFOp subfOp, math::ExpOp &expOp,
                             CVPipeline::ComputeBlockIdManager &bm) {

  auto subfResult = subfOp.getResult();
  if (!subfOp->hasOneUse()) {
    LOG_DEBUG("SubF has multiple users, skipping");
    return false;
  }

  Operation *user = *subfResult.getUsers().begin();
  expOp = dyn_cast<math::ExpOp>(user);
  if (!expOp) {
    LOG_DEBUG("SubF user is not ExpOp, skipping");
    return false;
  }
  if (expOp->getBlock() != subfOp->getBlock()) {
    LOG_DEBUG("SubF and ExpOp not in the same Block, skipping");
    return false;
  }
  return true;
}

static bool matchExtfFromSubf(arith::SubFOp subfOp,
                              SmallVector<arith::ExtFOp, 2> &extfOps,
                              CVPipeline::ComputeBlockIdManager &bm) {
  auto lhs = subfOp.getLhs();
  auto rhs = subfOp.getRhs();

  auto lhsDef = lhs.getDefiningOp<arith::ExtFOp>();
  auto rhsDef = rhs.getDefiningOp<arith::ExtFOp>();

  if (!lhsDef || !rhsDef) {
    LOG_DEBUG(
        "SubF operands are not both from ExtFOp, skipping extended pattern");
    return false;
  }
  if (lhsDef->getBlock() != subfOp->getBlock() ||
      rhsDef->getBlock() != subfOp->getBlock()) {
    LOG_DEBUG("SubF and ExtfOp not in the same Block, skipping");
    return false;
  }

  auto lhsInType = lhsDef.getIn().getType();
  auto rhsInType = rhsDef.getIn().getType();
  auto lhsOutType = lhsDef.getOut().getType();
  auto rhsOutType = rhsDef.getOut().getType();

  if (!lhsInType.isF16() || !rhsInType.isF16()) {
    LOG_DEBUG("ExtF input types are not both f16, skipping extended pattern");
    return false;
  }

  if (!lhsOutType.isF32() || !rhsOutType.isF32()) {
    LOG_DEBUG("ExtF output types are not both f32, skipping extended pattern");
    return false;
  }

  SmallVector<Operation *, 2> lhsUsers;
  for (Operation *user : lhsDef.getResult().getUsers()) {
    lhsUsers.push_back(user);
  }
  if (lhsUsers.size() != 1 || lhsUsers[0] != subfOp) {
    LOG_DEBUG("Left ExtF has multiple users or not used by subf, skipping");
    return false;
  }

  SmallVector<Operation *, 2> rhsUsers;
  for (Operation *user : rhsDef.getResult().getUsers()) {
    rhsUsers.push_back(user);
  }
  if (rhsUsers.size() != 1 || rhsUsers[0] != subfOp) {
    LOG_DEBUG("Right ExtF has multiple users or not used by subf, skipping");
    return false;
  }

  extfOps.push_back(lhsDef);
  extfOps.push_back(rhsDef);
  return true;
}

static void processSubfOp(arith::SubFOp subfOp,
                          CVPipeline::ComputeBlockIdManager &bm) {
  int subfBlockId = bm.getBlockIdByOp(subfOp);

  math::ExpOp expOp;
  if (!matchExpFromSubf(subfOp, expOp, bm)) {
    return;
  }

  int expBlockId = bm.getBlockIdByOp(expOp);
  if (expBlockId != subfBlockId) {
    LOG_DEBUG("Updating exp blockId from " << expBlockId << " to "
                                           << subfBlockId);
    bm.updateBlockId(expOp, subfBlockId);
  } else {
    LOG_DEBUG("Exp blockId already matches subf, skipping update");
  }

  SmallVector<arith::ExtFOp, 2> extfOps;
  if (matchExtfFromSubf(subfOp, extfOps, bm)) {
    for (auto extfOp : extfOps) {
      int extfBlockId = bm.getBlockIdByOp(extfOp);
      if (extfBlockId != subfBlockId) {
        LOG_DEBUG("Updating extf blockId from " << extfBlockId << " to "
                                                << subfBlockId);
        bm.updateBlockId(extfOp, subfBlockId);
      } else {
        LOG_DEBUG("ExtF blockId already matches subf, skipping update");
      }
    }
  }
}

} // anonymous namespace

class ExpSubfPatternPass
    : public PassWrapper<ExpSubfPatternPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpSubfPatternPass)

  ExpSubfPatternPass() = default;

  StringRef getArgument() const override { return "exp-subf-pattern"; }

  StringRef getDescription() const override {
    return "Match exp-subf pattern and unify blockId for exp and optional extf "
           "operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LOG_DEBUG("Input mlir:\n " << module);
    CVPipeline::ComputeBlockIdManager bm(module);

    module.walk([&](arith::SubFOp subfOp) {
      LOG_DEBUG("Process subf: " << subfOp << "\n");
      processSubfOp(subfOp, bm);
    });
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpSubfPatternPass() {
  return std::make_unique<ExpSubfPatternPass>();
}

} // namespace triton
} // namespace mlir
