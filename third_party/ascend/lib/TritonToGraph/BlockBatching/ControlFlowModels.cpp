// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::triton::cfg {
namespace {

// 常数正步长，或者 num_programs × 正常数。后者是成块发的步长
// （for r in range(pid, N, num_programs)），它不是编译期常量但恒为正：网格大小
// 至少是 1。只认这两种形状，不做一般的符号正性证明。
bool provenPositiveStep(Value step) {
  if (step.getDefiningOp<triton::GetNumProgramsOp>())
    return true;
  // 符号扩展保值：正数扩宽之后还是同一个正数，所以可以直接看进去。少了这一条，
  // 凡是源码写过 tl.program_id(0).cast(tl.int64) 的 kernel 整条循环都抬到 i64，
  // 三个界全裹着 extsi，一个都证不出来——正例 #7 就是这么被拒的。
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

struct Analysis {
  BatchControlFlowPlan plan;
  int axis;
  unsigned budget = 8192;
  const llvm::DenseSet<Operation *> &replicatedLoads;
  Analysis(int axis, const llvm::DenseSet<Operation *> &loads)
      : axis(axis), replicatedLoads(loads) {}
  bool reject(Operation *op, llvm::StringRef reason) {
    if (plan.reason.empty()) {
      plan.reason = reason.str();
      plan.operation = op;
    }
    return false;
  }
  bool known(Value value, Operation *op) {
    return plan.varying.count(value) || reject(op, "VALUE_VARIATION_UNKNOWN");
  }
  void record(Value value, bool varying) {
    plan.varying[value] = plan.isVarying(value) || varying;
  }
  bool block(Block &body, unsigned depth = 0) {
    if (depth > 8)
      return reject(body.getParentOp(), "REGION_DEPTH_EXCEEDED");
    for (Operation &op : body) {
      if (!budget--)
        return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
      for (Value value : op.getOperands())
        if (!known(value, &op))
          return false;
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (plan.isVarying(branch.getCondition()))
          return reject(&op, "VARYING_CONTROL_CONDITION");
        if (!branch.getThenRegion().hasOneBlock() ||
            (!branch.getElseRegion().empty() &&
             !branch.getElseRegion().hasOneBlock()))
          return reject(&op, "UNSUPPORTED_REGION_SHAPE");
        if (!block(branch.getThenRegion().front(), depth + 1))
          return false;
        if (!branch.getElseRegion().empty() &&
            !block(branch.getElseRegion().front(), depth + 1))
          return false;
        if (branch.getNumResults()) {
          auto thenYield =
              cast<scf::YieldOp>(branch.getThenRegion().front().back());
          auto elseYield =
              cast<scf::YieldOp>(branch.getElseRegion().front().back());
          for (auto [result, a, b] :
               llvm::zip(branch.getResults(), thenYield.getOperands(),
                         elseYield.getOperands()))
            record(result, plan.isVarying(a) || plan.isVarying(b));
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        // 界随程序轴变化不再在这里拒。认不认得出这条循环、能不能按 lane
        // 重分块， 是关卡 0（BatchLoopForm.h）和关卡
        // 1（BatchLoopSchedule.h）的事：它们
        // 看得见两种分布的形态，这里看不见。三个界里任何一个变都归关卡 0 接管，
        // 所以这里放行不会留下无人认领的形状。
        if (!provenPositiveStep(loop.getStep()))
          return reject(&op, "UNPROVEN_POSITIVE_LOOP_STEP");
        if (loop.getNumRegionIterArgs() > 128)
          return reject(&op, "ANALYSIS_BUDGET_EXCEEDED");
        // 程序轴循环的循环变量按 lane 变化，普通循环的不变。
        //
        // 看着矛盾：合并之后 scf.for 的界只能是标量，循环变量当然也是标量。但这
        // 一位管的不是界，是**体**——lane u 这一圈处理的行是 iv +
        // u*R，由它派生的 下标因此各 lane 不同，语义体必须给它们加上 lane
        // 轴，否则桥折不成一条宽 访存。界由 retileBatchLoopBounds
        // 另算，走的不是这条映射 （BatchRewrite.cpp 的循环重建）。
        //
        // 判据和关卡 0 一致：三个界里任一随程序轴变化。
        record(loop.getInductionVar(),
               plan.isVarying(loop.getLowerBound()) ||
                   plan.isVarying(loop.getUpperBound()) ||
                   plan.isVarying(loop.getStep()));
        for (auto [arg, init] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
          record(arg, plan.isVarying(init));
        bool stable = false;
        for (unsigned round = 0; round <= loop.getNumRegionIterArgs();
             ++round) {
          if (!block(*loop.getBody(), depth + 1))
            return false;
          stable = true;
          auto yield = cast<scf::YieldOp>(loop.getBody()->back());
          for (auto [arg, value] :
               llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
            if (plan.isVarying(value) && !plan.isVarying(arg)) {
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
          record(result, plan.isVarying(arg));
        continue;
      }
      if (isa<triton::ReduceOp, triton::ScanOp>(op))
        for (Value capture : getBatchCombinerCaptures(&op)) {
          if (!known(capture, &op))
            return false;
          if (plan.isVarying(capture))
            return reject(&op, "UNSUPPORTED_REGION_CAPTURE");
        }
      bool varying = llvm::any_of(op.getOperands(),
                                  [&](Value v) { return plan.isVarying(v); });
      if (auto load = dyn_cast<triton::LoadOp>(op);
          load && (load.getIsVolatile() || replicatedLoads.contains(&op)))
        varying = true;
      if (auto pid = dyn_cast<triton::GetProgramIdOp>(op))
        varying = static_cast<int>(pid.getAxis()) == axis;
      for (Value result : op.getResults())
        record(result, varying);
    }
    return true;
  }
};
} // namespace
BatchControlFlowPlan
analyzeBatchControlFlow(triton::FuncOp function, int axis,
                        const llvm::DenseSet<Operation *> &replicatedLoads) {
  Analysis analysis(axis, replicatedLoads);
  for (Value arg : function.getArguments())
    analysis.record(arg, false);
  analysis.block(function.getBody().front());
  return std::move(analysis.plan);
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
