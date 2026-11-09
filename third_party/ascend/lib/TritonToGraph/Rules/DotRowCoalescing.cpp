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

#include "TritonToGraph/DotRowCoalescing.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "TritonToGraph/RowLifting.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// Internal RowCoalescing pattern for a counted, independent row matmul loop.
// No kernel name, Python argument name, or public tuning option is inspected.
// A static x-grid selects an exact divisor. Without it, use the existing
// original-grid hidden arguments and mask the final partial group. Indirect
// scalar loads deliberately remain on the original path.

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();

  return std::nullopt;
}

struct RowAddressRange {
  int64_t stride = 0, lo = 0, hi = 0;
};

// Scalar offsets independent of x translate every lane by the same amount.
// Ignore that common translation, but retain the constant x coefficient and
// tensor lane span. Unknown products/casts fail closed.
static std::optional<RowAddressRange> analyzeRowAddress(Value v, Value pid) {
  if (v == pid)
    return RowAddressRange{1, 0, 0};

  if (!isa<RankedTensorType>(v.getType()) && !dependsOnValue(v, pid))
    return RowAddressRange{};

  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (auto range = dyn_cast<triton::MakeRangeOp>(op))
    return RowAddressRange{0, range.getStart(), range.getEnd() - 1};

  if (isa<arith::ExtSIOp, triton::SplatOp, triton::ExpandDimsOp,
          triton::BroadcastOp>(op))
    return analyzeRowAddress(op->getOperand(0), pid);

  if (isa<triton::AddPtrOp, arith::AddIOp>(op)) {
    auto a = analyzeRowAddress(op->getOperand(0), pid),
         b = analyzeRowAddress(op->getOperand(1), pid);
    if (!a || !b)
      return std::nullopt;

    RowAddressRange out;
    if (__builtin_add_overflow(a->stride, b->stride, &out.stride) ||
        __builtin_add_overflow(a->lo, b->lo, &out.lo) ||
        __builtin_add_overflow(a->hi, b->hi, &out.hi))
      return std::nullopt;

    return out;
  }

  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    auto c = getConstantIntValue(mul.getRhs());
    Value other = mul.getLhs();
    if (!c) {
      c = getConstantIntValue(mul.getLhs());
      other = mul.getRhs();
    }

    if (!c || *c <= 0)
      return std::nullopt;

    auto a = analyzeRowAddress(other, pid);
    if (!a)
      return std::nullopt;

    RowAddressRange out;
    if (__builtin_mul_overflow(a->stride, *c, &out.stride) ||
        __builtin_mul_overflow(a->lo, *c, &out.lo) ||
        __builtin_mul_overflow(a->hi, *c, &out.hi))
      return std::nullopt;

    return out;
  }

  return std::nullopt;
}

// The row-coalescing engine is shared; this rule only owns the matcher and
// plan.
using RowCandidate = RowLiftCandidate;

static int64_t tensorStorageBytes(RankedTensorType tensor) {
  Type element = tensor.getElementType();
  const int64_t bytes = isa<triton::PointerType>(element)
                            ? 8
                            : (element.getIntOrFloatBitWidth() + 7) / 8;
  int64_t size = 0;
  if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
    return std::numeric_limits<int64_t>::max();
  return size;
}

static std::optional<RowCandidate> matchDotRows(triton::FuncOp fn,
                                                uint64_t l0aBudgetBytes,
                                                uint64_t l0cBudgetBytes,
                                                uint64_t coreNum) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  auto grid = module->getAttrOfType<IntegerAttr>("hacc.grid_num_tiles");
  if ((grid && (grid.getInt() <= 0 || grid.getInt() > INT32_MAX)) ||
      !llvm::hasSingleElement(fn.getBody()) || !fn.isPublic() ||
      llvm::count_if(module.getOps<triton::FuncOp>(),
                     [](triton::FuncOp entry) { return entry.isPublic(); }) !=
          1 ||
      module->hasAttr(kCoalesceFactorAttr) ||
      module->hasAttr("hacc.program_grid_transforms"))
    return std::nullopt;

  LLVM_DEBUG(llvm::dbgs() << "dot-row: grid gate passed\n");

  triton::GetProgramIdOp pid;
  scf::ForOp loop;
  triton::StoreOp output;
  unsigned pids = 0, loops = 0, dots = 0, stores = 0;
  int64_t maxDotRows = 1;
  int64_t maxTensorBytes = 1;
  // Cube-tile demands of the original dots; the rewrite widens the lhs and
  // accumulator M dimension by the factor while the rhs stays shared.
  int64_t maxL0ABytes = 1;
  int64_t maxL0CBytes = 1;
  bool safe = true;
  DenseSet<Value> lifted;

  // Propagate row dependence while checking operations in definition order.
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
          safe = false;
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
      // Loop states are per-row even when their initial values are uniform.
      lifted.insert(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
      lifted.insert(f.getResults().begin(), f.getResults().end());
      if (f->getBlock() != &fn.getBody().front())
        safe = false;
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
        safe = false;
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
      // A scalar metadata load is the varlen/indirect case, not this pattern.
      if (!isa<RankedTensorType>(l.getType()) || l.getIsVolatile() ||
          (dependent && op->getParentOfType<scf::ForOp>()))
        safe = false;
      return;
    }

    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      output = store;
      ++stores;
      if (op->getBlock() != &fn.getBody().front())
        safe = false;
      return;
    }

    // Reduction regions are outside this pattern's supported rewrite scope.
    if (isa<triton::ReduceOp>(op)) {
      safe = false;
      return;
    }

    if (isa<triton::FuncOp, triton::ReturnOp, scf::YieldOp, triton::MakeRangeOp,
            triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
            triton::AddPtrOp, triton::TransOp, triton::ReshapeOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if ((ns != "arith" && ns != "math") || op->getNumRegions())
      safe = false;
  });

  LLVM_DEBUG(llvm::dbgs() << "dot-row: safe=" << safe << " pids=" << pids
                          << " loops=" << loops << " dots=" << dots
                          << " stores=" << stores << "\n");
  if (!safe || pids != 1 || loops != 1 || dots == 0 || stores != 1 ||
      loop.getNumResults() == 0 ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0) ||
      !canUseLastRowLoopBound(loop, pid.getResult(), grid ? grid.getInt() : 0))
    return std::nullopt;

  for (Value init : loop.getInitArgs())
    if (!isa<RankedTensorType>(init.getType()))
      return std::nullopt;

  auto address = analyzeRowAddress(output.getPtr(), pid.getResult());
  if (!address || address->stride <= 0 || address->hi < address->lo ||
      static_cast<uint64_t>(address->hi) - static_cast<uint64_t>(address->lo) >=
          static_cast<uint64_t>(address->stride))
    return std::nullopt;

  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return std::nullopt;

  RowCountPolicy policy;
  policy.targetDotRows = 128;
  policy.maxL0ABytes = maxL0ABytes;
  policy.maxL0CBytes = maxL0CBytes;
  policy.l0ABudgetBytes = l0aBudgetBytes;
  policy.l0CBudgetBytes = l0cBudgetBytes;
  policy.coreNum = coreNum;
  int64_t rows =
      selectRowsPerProgram(grid ? grid.getInt() : 0, maxDotRows, maxTensorBytes,
                           /*peakLiveBytes=*/0, policy);
  if (rows <= 1)
    return std::nullopt;

  RowCandidate candidate;
  candidate.function = fn;
  candidate.anchor = pid.getOperation();
  candidate.rowsPerProgram = rows;
  candidate.runtimeGrid = !grid;
  candidate.grid = grid ? grid.getInt() : 0;
  return candidate;
}

class DotRowPlan final : public RewritePlan {
  RowCandidate candidate;
  unsigned epoch;
  OwningOpRef<ModuleOp> prepared;
  // Safe L0 budgets captured at plan creation; zero means the target's cube
  // capacities are unknown and the factor stays unconstrained.
  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  uint64_t coreNum = 0;

  DotRowPlan(RowCandidate candidate, unsigned epoch,
             OwningOpRef<ModuleOp> prepared, uint64_t l0aBudgetBytes,
             uint64_t l0cBudgetBytes, uint64_t coreNum)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)),
        l0aBudgetBytes(l0aBudgetBytes), l0cBudgetBytes(l0cBudgetBytes),
        coreNum(coreNum) {}

public:
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::RowCoalescing;
  }

  unsigned getBenefit() const override { return 1; }

  Operation *getAnchor() const override { return candidate.function; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();

    auto current = matchDotRows(candidate.function, l0aBudgetBytes,
                                l0cBudgetBytes, coreNum);
    return success(current && current->anchor == candidate.anchor &&
                   current->rowsPerProgram == candidate.rowsPerProgram &&
                   current->runtimeGrid == candidate.runtimeGrid);
  }

  static std::unique_ptr<DotRowPlan>
  create(RowCandidate candidate, unsigned epoch, uint64_t l0aBudgetBytes,
         uint64_t l0cBudgetBytes, uint64_t coreNum) {
    // Prepare and verify privately before offering a plan to the scheduler.
    // An unsupported rewrite must leave the input and its diagnostics
    // untouched.
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    OwningOpRef<ModuleOp> sandbox(
        ModuleOp::create(candidate.function.getLoc()));
    sandbox->getOperation()->setAttrs(module->getAttrs());

    IRMapping mapping;
    sandbox->getBody()->push_back(candidate.function->clone(mapping));

    RowCandidate copy = candidate;
    copy.function = cast<triton::FuncOp>(&sandbox->getBody()->front());
    copy.anchor =
        mapping.lookup(candidate.anchor->getResult(0)).getDefiningOp();

    IRRewriter rw(module.getContext());
    ScopedDiagnosticHandler diagnostics(
        module.getContext(), [](Diagnostic &diag) {
          LLVM_DEBUG(llvm::dbgs()
                     << "dot-row: skipped candidate: " << diag << "\n");
          return success();
        });
    if (copy.runtimeGrid &&
        failed(addProgramGridHiddenExtentArguments(copy.function)))
      return nullptr;

    RowLiftBuilder builder(rw, copy);
    if (!builder.run(copy.function) || failed(verify(*sandbox)))
      return nullptr;

    if (copy.runtimeGrid) {
      ProgramGridTransformContract contract;
      contract.transforms.push_back({0, 0, copy.rowsPerProgram, false, false});
      if (failed(setProgramGridTransformContract(*sandbox, contract)))
        return nullptr;
    }

    return std::unique_ptr<DotRowPlan>(
        new DotRowPlan(candidate, epoch, std::move(sandbox), l0aBudgetBytes,
                       l0cBudgetBytes, coreNum));
  }

  LogicalResult apply(IRRewriter &rw) override {
    // Commit the verified body and, for runtime grids, its hidden arguments.
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
    module->setAttr(kCoalesceFactorAttr,
                    rw.getI32IntegerAttr(candidate.rowsPerProgram));
    module->setAttr(kCoalesceAxisAttr, rw.getI32IntegerAttr(0));
    module->setAttr(kCoalesceGridCeilDivAttr, rw.getI32IntegerAttr(1));
    return success();
  }
};

} // namespace

std::unique_ptr<RewritePlan>
cfg::createDotRowCoalescingPlan(triton::FuncOp function, unsigned epoch,
                                const ResourceSnapshot &resources) {
  // Dot row coalescing is supported only on Ascend 950.
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return nullptr;

  // Zero budgets (unknown cube capacities) leave the factor unconstrained.
  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  if (resources.hasL0Budgets()) {
    l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
    l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }

  auto candidate =
      matchDotRows(function, l0aBudgetBytes, l0cBudgetBytes, resources.coreNum);
  if (!candidate)
    return nullptr;

  return DotRowPlan::create(*candidate, epoch, l0aBudgetBytes, l0cBudgetBytes,
                            resources.coreNum);
}
