// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/BatchAnalyzer.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::triton::cfg {
namespace {

// Accept a positive constant step or num_programs times a positive constant.
// The latter covers cyclic loops such as range(pid, N, num_programs): it is
// not constant, but the grid has at least one program. This is deliberately
// limited to those forms, not a general symbolic positivity proof.
bool provenPositiveStep(Value step) {
  if (step.getDefiningOp<triton::GetNumProgramsOp>())
    return true;
  // Sign extension preserves positive values, so inspect its operand.
  // Casting tl.program_id(0) to tl.int64 can wrap all loop bounds in
  // extsi; those wrappers must not prevent a positivity proof.
  if (auto widened = step.getDefiningOp<arith::ExtSIOp>())
    return provenPositiveStep(widened.getIn());
  if (auto product = step.getDefiningOp<arith::MulIOp>())
    return provenPositiveStep(product.getLhs()) &&
           provenPositiveStep(product.getRhs());
  auto constant = step.getDefiningOp<arith::ConstantOp>();
  auto amount =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  return amount && !amount.getValue().isNegative() &&
         !amount.getValue().isZero();
}

} // namespace
LogicalResult BatchAnalyzer::analyzeVariation(
    int axis, const llvm::DenseSet<Operation *> &replicatedLoads) {
  workspace.beginVariationScope();
  unsigned budget = 8192;
  auto reject = [&](Operation *op, llvm::StringRef why) {
    if (reason.empty()) {
      reason = why.str();
      operation = op;
    }
    return false;
  };
  auto known = [&](Value v, Operation *op) {
    return workspace.known(v) || reject(op, "VALUE_VARIATION_UNKNOWN");
  };
  auto record = [&](Value v, bool varies) {
    workspace.recordVariation(v, varies);
  };
  auto block = [&](auto &&self, Block &body, unsigned depth) -> bool {
    if (depth > 8)
      return reject(body.getParentOp(), "REGION_DEPTH_EXCEEDED");
    for (Operation &op : body) {
      if (!budget--)
        return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
      for (Value value : op.getOperands())
        if (!known(value, &op))
          return false;
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (workspace.isVarying(branch.getCondition()))
          return reject(&op, "VARYING_CONTROL_CONDITION");
        if (!branch.getThenRegion().hasOneBlock() ||
            (!branch.getElseRegion().empty() &&
             !branch.getElseRegion().hasOneBlock()))
          return reject(&op, "UNSUPPORTED_REGION_SHAPE");
        if (!self(self, branch.getThenRegion().front(), depth + 1))
          return false;
        if (!branch.getElseRegion().empty() &&
            !self(self, branch.getElseRegion().front(), depth + 1))
          return false;
        if (branch.getNumResults()) {
          auto thenYield =
              cast<scf::YieldOp>(branch.getThenRegion().front().back());
          auto elseYield =
              cast<scf::YieldOp>(branch.getElseRegion().front().back());
          for (auto [result, a, b] :
               llvm::zip(branch.getResults(), thenYield.getOperands(),
                         elseYield.getOperands()))
            record(result, workspace.isVarying(a) || workspace.isVarying(b));
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        // Delegate program-dependent bounds to the form and coordinate checks
        // in BatchLoopForm.h and BatchLoopSchedule.h. They recognize the
        // supported distributions. Any varying bound or step is owned by gate
        // 0, so accepting it here cannot bypass the later legality checks.
        if (!provenPositiveStep(loop.getStep()))
          return reject(&op, "UNPROVEN_POSITIVE_LOOP_STEP");
        if (loop.getNumRegionIterArgs() > 128)
          return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
        // A program-axis loop's induction value varies across semantic lanes;
        // an ordinary loop's does not.
        //
        // The rebuilt scf.for still uses scalar bounds and a scalar induction
        // variable. This flag describes its body: lane u processes iv + u*R,
        // so derived indices need a lane axis for the wide-memory bridge.
        // retileBatchLoopBounds computes scalar bounds separately from this
        // map.
        //
        // Use the same criterion as gate 0: any bound or step varies with the
        // selected program axis.
        record(loop.getInductionVar(),
               workspace.isVarying(loop.getLowerBound()) ||
                   workspace.isVarying(loop.getUpperBound()) ||
                   workspace.isVarying(loop.getStep()));
        for (auto [arg, init] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
          record(arg, workspace.isVarying(init));
        bool stable = false;
        for (unsigned round = 0; round <= loop.getNumRegionIterArgs();
             ++round) {
          if (!self(self, *loop.getBody(), depth + 1))
            return false;
          stable = true;
          auto yield = cast<scf::YieldOp>(loop.getBody()->back());
          for (auto [arg, value] :
               llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
            if (workspace.isVarying(value) && !workspace.isVarying(arg)) {
              record(arg, true);
              stable = false;
            }
          }
          if (stable)
            break;
        }
        if (!stable)
          return reject(&op, "LOOP_VARIATION_DID_NOT_CONVERGE");
        for (auto [result, arg] :
             llvm::zip(loop.getResults(), loop.getRegionIterArgs()))
          record(result, workspace.isVarying(arg));
        continue;
      }
      if (isa<triton::ReduceOp, triton::ScanOp>(op))
        for (Value capture : getBatchCombinerCaptures(&op)) {
          if (!known(capture, &op))
            return false;
          if (workspace.isVarying(capture))
            return reject(&op, "UNSUPPORTED_REGION_CAPTURE");
        }
      bool varying = llvm::any_of(
          op.getOperands(), [&](Value v) { return workspace.isVarying(v); });
      if (auto load = dyn_cast<triton::LoadOp>(op);
          load && (load.getIsVolatile() || replicatedLoads.contains(&op)))
        varying = true;
      if (auto pid = dyn_cast<triton::GetProgramIdOp>(op))
        varying = static_cast<int>(pid.getAxis()) == axis;
      for (Value result : op.getResults())
        record(result, varying);
    }
    return true;
  };
  for (Value arg : function.getArguments())
    record(arg, false);
  return success(block(block, function.getBody().front(), 0));
}

bool cvLoopNeedsZeroTripGuard(scf::ForOp loop) {
  auto recurrent = [&](auto &&self, Value value, Value iterArg,
                       unsigned depth) -> bool {
    if (value == iterArg)
      return true;
    if (depth > 16)
      return false;
    if (auto dot = value.getDefiningOp<triton::DotOp>())
      return dot.getC() == iterArg;
    if (auto pointer = value.getDefiningOp<triton::AddPtrOp>())
      return pointer.getPtr() == iterArg;
    if (auto branch = value.getDefiningOp<scf::IfOp>()) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      for (Region &region : branch->getRegions()) {
        if (!region.hasOneBlock())
          return false;
        Value yielded = region.front().getTerminator()->getOperand(index);
        if (!self(self, yielded, iterArg, depth + 1))
          return false;
      }
      return true;
    }
    return false;
  };
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [result, arg, end] : llvm::zip(
           loop.getResults(), loop.getRegionIterArgs(), yield.getOperands())) {
    if (!result.use_empty() && !recurrent(recurrent, end, arg, 0))
      return true;
  }
  return false;
}
} // namespace mlir::triton::cfg
