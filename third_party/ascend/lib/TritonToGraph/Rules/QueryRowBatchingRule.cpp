/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/QueryRowBatching.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "TritonToGraph/RowLifting.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

static int64_t tensorStorageBytes(RankedTensorType tensor) {
  Type element = tensor.getElementType();
  const int64_t bytes = getElementStorageBytes(element);
  int64_t size = 0;
  if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
    return std::numeric_limits<int64_t>::max();
  return size;
}

static std::optional<RowLiftCandidate> matchQueryRows(
    triton::FuncOp fn, uint64_t l0aBudgetBytes, uint64_t l0cBudgetBytes,
    uint64_t coreNum, uint64_t ubBudgetBytes,
    std::optional<int64_t> forcedRows = std::nullopt, bool allowSplit = true) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  auto grid = module->getAttrOfType<IntegerAttr>("hacc.grid_num_tiles");

  if ((grid && (grid.getInt() <= 0 || grid.getInt() > INT32_MAX)) ||
      !llvm::hasSingleElement(fn.getBody()) || !fn.isPublic() ||
      llvm::count_if(module.getOps<triton::FuncOp>(),
                     [](triton::FuncOp entry) { return entry.isPublic(); }) !=
          1 ||
      module->hasAttr(kCoalesceFactorAttr) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return std::nullopt;

  triton::GetProgramIdOp pid;

  scf::ForOp loop;

  bool output = false;

  unsigned pids = 0, loops = 0, dots = 0, stores = 0;

  unsigned blockPointers = 0, reduces = 0;

  int64_t maxDotRows = 1;

  int64_t maxTensorBytes = 1;

  int64_t maxL0ABytes = 1;

  int64_t maxL0CBytes = 1;

  bool safe = true;

  DenseSet<Value> lifted;

  auto reject = [&](Operation *bad, StringRef why) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: reject " << why << " ("
                            << bad->getName() << ")\n");
    safe = false;
  };

  fn.walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool dependent = llvm::any_of(
        op->getOperands(), [&](Value value) { return lifted.contains(value); });
    if (dependent)
      lifted.insert(op->getResults().begin(), op->getResults().end());

    for (Type type : op->getResultTypes()) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        Type element = tensor.getElementType();
        int64_t bytes = isa<triton::PointerType>(element)
                            ? 8
                            : (element.getIntOrFloatBitWidth() + 7) / 8;
        int64_t size;
        if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
          reject(op, "tensor size overflow");
        else
          maxTensorBytes = std::max(maxTensorBytes, size);
      }
    }

    if (auto p = dyn_cast<triton::GetProgramIdOp>(op)) {

      if (p.getAxisAsInt() == 0) {
        pid = p;
        ++pids;
        lifted.insert(p.getResult());
      }
      return;
    }

    if (auto f = dyn_cast<scf::ForOp>(op)) {
      loop = f;
      ++loops;

      lifted.insert(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
      lifted.insert(f.getResults().begin(), f.getResults().end());
      if (f->getBlock() != &fn.getBody().front())
        reject(f, "loop not in entry block");
      return;
    }

    if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      ++blockPointers;

      if (dependent && !canExpandTensorPointer(makePtr, true))
        reject(makePtr, "unexpanded tensor pointer");
      return;
    }

    if (auto d = dyn_cast<triton::DotOp>(op)) {
      ++dots;

      auto a = dyn_cast<RankedTensorType>(d.getA().getType());
      auto b = dyn_cast<RankedTensorType>(d.getB().getType());
      if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
          !(a.getElementType().isF16() || a.getElementType().isBF16()) ||
          (dependent &&
           (!lifted.contains(d.getA()) || lifted.contains(d.getB()))))
        reject(d, "dot operand or shape");
      else {
        maxDotRows = std::max(maxDotRows, a.getShape()[0]);
        maxL0ABytes = std::max(maxL0ABytes, tensorStorageBytes(a));
        if (auto acc = dyn_cast<RankedTensorType>(d.getC().getType()))
          maxL0CBytes = std::max(maxL0CBytes, tensorStorageBytes(acc));
        if (auto res = dyn_cast<RankedTensorType>(d.getD().getType()))
          maxL0CBytes = std::max(maxL0CBytes, tensorStorageBytes(res));
      }
      return;
    }

    if (auto l = dyn_cast<triton::LoadOp>(op)) {
      if (l.getIsVolatile()) {
        reject(l, "volatile load");
        return;
      }

      return;
    }

    if (isa<triton::StoreOp>(op)) {
      ++stores;
      if (op->getBlock() == &fn.getBody().front())
        output = true;
      return;
    }

    if (isa<triton::ReduceOp>(op)) {
      ++reduces;
      return;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (pid && !liftsRowInvariantIf(ifOp, pid.getResult()))
        reject(ifOp, "row-dependent scf.if");
      return;
    }

    if (isa<triton::FuncOp, triton::ReturnOp, scf::YieldOp,
            triton::ReduceReturnOp, triton::MakeRangeOp, triton::SplatOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::AddPtrOp,
            triton::TransOp, triton::ReshapeOp, triton::AdvanceOp,
            triton::AssertOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if ((ns != "arith" && ns != "math") || op->getNumRegions())
      reject(op, "unsupported op");
  });

  LLVM_DEBUG(llvm::dbgs() << "query-row: safe=" << safe << " pids=" << pids
                          << " loops=" << loops << " dots=" << dots
                          << " stores=" << stores << " blockPtrs="
                          << blockPointers << " reduces=" << reduces << "\n");

  if (!safe || pids != 1 || loops != 1 || dots == 0 || stores == 0 || !output)
    return std::nullopt;
  if (loop.getNumResults() == 0 ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0))
    return std::nullopt;

  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return std::nullopt;

  if (!canUseLastRowLoopBound(loop, pid.getResult(),
                              grid ? grid.getInt() : 0)) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: reject unsafe shared loop bound\n");
    return std::nullopt;
  }

  RowLiftCandidate candidate;
  candidate.function = fn;
  candidate.anchor = pid.getOperation();
  candidate.runtimeGrid = !grid;
  candidate.grid = grid ? grid.getInt() : 0;

  candidate.features.reduce = true;
  candidate.features.loopInvariantIf = true;
  candidate.features.tensorPointer = true;
  candidate.features.maskedLoad = true;
  candidate.features.maskedStore = true;
  candidate.features.flatCompute = true;
  candidate.features.groupedLoopBounds = true;

  candidate.features.splitWideAccumulator = allowSplit;

  RowCountPolicy policy;
  policy.splitWideAccumulator = candidate.features.splitWideAccumulator;
  policy.maxL0ABytes = maxL0ABytes;
  policy.maxL0CBytes = maxL0CBytes;
  policy.l0ABudgetBytes = l0aBudgetBytes;
  policy.l0CBudgetBytes = l0cBudgetBytes;
  policy.ubBudgetBytes = ubBudgetBytes;
  policy.coreNum = coreNum;
  candidate.rowsPerProgram = selectRowsPerProgram(
      grid ? grid.getInt() : 0, maxDotRows, maxTensorBytes, 0, policy);

  if (forcedRows)
    candidate.rowsPerProgram = std::min(candidate.rowsPerProgram, *forcedRows);
  if (candidate.rowsPerProgram <= 1)
    return std::nullopt;

  candidate.splitValueBudgetBytes =
      allowSplit ? maxDotRows * candidate.rowsPerProgram * kVectorRegisterBytes
                 : std::numeric_limits<int64_t>::max();

  if (!canSplitWideAccumulators(loop, candidate.rowsPerProgram,
                                candidate.features.splitWideAccumulator,
                                candidate.splitValueBudgetBytes)) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: reject rewrite pre-check\n");
    return std::nullopt;
  }

  return candidate;
}

static LogicalResult rewriteQueryRowBatchInPlace(
    ModuleOp module, triton::FuncOp function, const ResourceSnapshot &resources,
    std::optional<int64_t> forcedRows = std::nullopt, bool allowSplit = true) {

  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  if (resources.hasL0Budgets()) {
    l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
    l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }

  auto candidate = matchQueryRows(
      function, l0aBudgetBytes, l0cBudgetBytes, resources.coreNum,
      resources.getSafeUBBudget().value_or(0), forcedRows, allowSplit);
  if (!candidate)
    return failure();

  IRRewriter rw(module.getContext());

  ScopedDiagnosticHandler diagnostics(
      module.getContext(), [](Diagnostic &diag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "query-row: skipped candidate: " << diag << "\n");
        return success();
      });

  if (candidate->runtimeGrid &&
      failed(addProgramGridHiddenExtentArguments(function))) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: hidden extent args failed\n");
    return failure();
  }

  RowLiftBuilder builder(rw, *candidate);
  if (!builder.run(function)) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: builder.run failed\n");
    return failure();
  }
  if (failed(verify(module))) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: verify failed\n");
    return failure();
  }

  if (candidate->runtimeGrid) {
    ProgramGridTransformContract contract;
    contract.transforms.push_back(
        {0, 0, candidate->rowsPerProgram, false, false});
    return setProgramGridTransformContract(module, contract);
  }

  module->setAttr(kCoalesceFactorAttr,
                  rw.getI32IntegerAttr(candidate->rowsPerProgram));
  module->setAttr(kCoalesceAxisAttr, rw.getI32IntegerAttr(0));
  module->setAttr(kCoalesceGridCeilDivAttr, rw.getI32IntegerAttr(1));
  return success();
}

static bool fitsQueryRowBudgets(ModuleOp module, triton::FuncOp function,
                                const ResourceSnapshot &resources,
                                int64_t factor) {
  ScopedDiagnosticHandler diagnostics(
      module.getContext(), [](Diagnostic &diag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "query-row: candidate cleanup failed: " << diag << "\n");
        return success();
      });
  PassManager cleanup(module.getContext(), ModuleOp::getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  if (failed(cleanup.run(module)) || failed(verify(module)))
    return false;
  if (auto budget = resources.getSafeUBBudget()) {
    LiveByteEstimate estimate = estimatePeakLiveBytes(function);
    if (::getenv("TRITON_GRAPH_UB_DEBUG"))
      llvm::errs() << "graph-ub[query-row]: R=" << factor
                   << " liftedPeak=" << estimate.peakLiveBytes
                   << " known=" << estimate.known
                   << " replication=" << kUBBufferReplication
                   << " ubBudget=" << *budget << "\n";
    if (!estimate.known || estimate.overflow ||
        estimate.peakLiveBytes > *budget / kUBBufferReplication)
      return false;
  }
  return !exceedsCubeBudgets(function, resources);
}

class QueryRowBatchingPlan final : public RewritePlan {
  RowLiftCandidate candidate;
  unsigned epoch;
  OwningOpRef<ModuleOp> prepared;

  bool allowSplit;

  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;

  uint64_t ubBudgetBytes = 0;
  uint64_t coreNum = 0;

  QueryRowBatchingPlan(RowLiftCandidate candidate, unsigned epoch,
                       OwningOpRef<ModuleOp> prepared, uint64_t l0aBudgetBytes,
                       uint64_t l0cBudgetBytes, uint64_t ubBudgetBytes,
                       uint64_t coreNum, bool allowSplit)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)),
        allowSplit(allowSplit), l0aBudgetBytes(l0aBudgetBytes),
        l0cBudgetBytes(l0cBudgetBytes), ubBudgetBytes(ubBudgetBytes),
        coreNum(coreNum) {}

public:
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::QueryRowBatching;
  }

  unsigned getBenefit() const override { return 1; }

  Operation *getAnchor() const override { return candidate.function; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();

    auto current = matchQueryRows(candidate.function, l0aBudgetBytes,
                                  l0cBudgetBytes, coreNum, ubBudgetBytes,
                                  candidate.rowsPerProgram, allowSplit);
    return success(current && current->anchor == candidate.anchor &&
                   current->rowsPerProgram == candidate.rowsPerProgram &&
                   current->runtimeGrid == candidate.runtimeGrid);
  }

  static std::unique_ptr<QueryRowBatchingPlan>
  create(RowLiftCandidate candidate, unsigned epoch,
         const ResourceSnapshot &resources, bool allowSplit) {
    uint64_t l0aBudgetBytes = 0;
    uint64_t l0cBudgetBytes = 0;
    if (resources.hasL0Budgets()) {
      l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
      l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
    }
    const uint64_t ubBudgetBytes = resources.getSafeUBBudget().value_or(0);

    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    int64_t factor = candidate.rowsPerProgram;
    while (factor >= 2) {
      candidate.rowsPerProgram = factor;

      OwningOpRef<ModuleOp> sandbox(
          ModuleOp::create(candidate.function.getLoc()));
      sandbox->getOperation()->setAttrs(module->getAttrs());
      sandbox->getBody()->push_back(candidate.function->clone());
      auto cloned = cast<triton::FuncOp>(&sandbox->getBody()->front());
      LLVM_DEBUG(llvm::dbgs()
                 << "query-row: trying factor " << factor
                 << (allowSplit ? " (split)" : " (no split)") << "\n");
      if (succeeded(rewriteQueryRowBatchInPlace(*sandbox, cloned, resources,
                                                factor, allowSplit)) &&
          fitsQueryRowBudgets(*sandbox, cloned, resources, factor))
        return std::unique_ptr<QueryRowBatchingPlan>(new QueryRowBatchingPlan(
            candidate, epoch, std::move(sandbox), l0aBudgetBytes,
            l0cBudgetBytes, ubBudgetBytes, resources.coreNum, allowSplit));

      factor /= 2;
    }
    LLVM_DEBUG(llvm::dbgs() << "query-row: no factor rewrites\n");
    return nullptr;
  }

  LogicalResult apply(IRRewriter &) override {

    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    auto replacement = cast<triton::FuncOp>(&prepared->getBody()->front());
    if (candidate.runtimeGrid) {
      if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                      replacement)))
        return failure();
      module->setAttr(
          kProgramGridTransformsAttr,
          prepared->getOperation()->getAttr(kProgramGridTransformsAttr));
      return success();
    }

    candidate.function.getBody().takeBody(replacement.getBody());
    ModuleOp preparedModule = *prepared;
    module->setAttr(kCoalesceFactorAttr,
                    preparedModule->getAttr(kCoalesceFactorAttr));
    module->setAttr(kCoalesceAxisAttr,
                    preparedModule->getAttr(kCoalesceAxisAttr));
    module->setAttr(kCoalesceGridCeilDivAttr,
                    preparedModule->getAttr(kCoalesceGridCeilDivAttr));
    return success();
  }
};

class QueryRowBatchingRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::QueryRowBatching;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (auto plan = createQueryRowBatchingPlan(
            context.getFunction(), context.getEpoch(),
            context.getResourceCostAnalysis().getResourceSnapshot()))
      plans.push_back(std::move(plan));
    return success();
  }
};

} // namespace

LogicalResult

cfg::materializeQueryRowBatchingCandidate(ModuleOp module,
                                          triton::FuncOp function,
                                          const ResourceSnapshot &resources) {
  if (!module || !function || function->getParentOfType<ModuleOp>() != module)
    return failure();

  auto plan = createQueryRowBatchingPlan(function, 0, resources);
  if (!plan)
    return failure();
  IRRewriter rw(module.getContext());
  return plan->apply(rw);
}

std::unique_ptr<RewritePlan>
cfg::createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                                const ResourceSnapshot &resources) {

  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return nullptr;

  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  if (resources.hasL0Budgets()) {
    l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
    l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }

  for (bool allowSplit : {true, false}) {
    auto tierCandidate = matchQueryRows(
        function, l0aBudgetBytes, l0cBudgetBytes, resources.coreNum,
        resources.getSafeUBBudget().value_or(0), std::nullopt, allowSplit);
    if (!tierCandidate)
      continue;
    if (auto plan = QueryRowBatchingPlan::create(*tierCandidate, epoch,
                                                 resources, allowSplit))
      return plan;
  }
  return nullptr;
}

std::unique_ptr<GraphOptimizationRule> cfg::createQueryRowBatchingRule() {
  return std::make_unique<QueryRowBatchingRule>();
}
