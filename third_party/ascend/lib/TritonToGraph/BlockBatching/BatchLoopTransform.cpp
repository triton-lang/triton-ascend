// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopTransform.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::cfg {
LogicalResult groupBatchLoopIterations(scf::ForOp loop, int64_t factor) {
  // Recheck the actual IR immediately before the upstream utility, which
  // performs host arithmetic for static bounds. Keep direct callers safe too.
  BatchLoopFact fact;
  auto type = dyn_cast<IntegerType>(loop.getInductionVar().getType());
  fact.bits = type ? type.getWidth() : 64;
  auto bound = [&](Value value) -> BatchExprRef {
    llvm::APInt constant;
    if (!matchPattern(value, m_ConstantInt(&constant)))
      return BatchExpr::leaf(BatchExpr::Argument, 0, fact.bits);
    if (!constant.isSignedIntN(64))
      return {};
    return BatchExpr::leaf(BatchExpr::Constant, constant.getSExtValue(),
                           fact.bits);
  };
  fact.lower = bound(loop.getLowerBound());
  fact.upper = bound(loop.getUpperBound());
  fact.step = bound(loop.getStep());
  if (!fact.lower || !fact.upper || !fact.step ||
      batchLoopIterationArithmetic(fact, factor) == BatchLoopArithmetic::Unsafe)
    return failure();
  MLIRContext *context = loop.getContext();
  auto grouped = loopUnrollByFactor(loop, factor);
  if (failed(grouped))
    return failure();
  // Match the existing Triton loop-unroll epilogue contract.
  if (grouped->epilogueLoopOp)
    (*grouped->epilogueLoopOp)
        ->setAttr("tt.num_stages",
                  IntegerAttr::get(IntegerType::get(context, 32), 1));
  return success();
}
namespace {

// Observed kernel bounds use only a few operations; this budget protects
// against pathological expression graphs.
constexpr unsigned kBudget = 256;

// Return whether value's defining graph may depend on programId. Return
// false only when independence is proved; the caller then remaps that
// entire subtree without cloning operations.
//
// Uncertainty returns true so the main path still applies its region and
// budget checks. Assuming independence could miss a program ID captured
// inside a region and leave the original ID in a rewritten bound.
bool mayReadProgramId(Value value, Value programId, unsigned &budget) {
  if (value == programId)
    return true;
  if (!budget)
    return true;
  --budget;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false; // Function argument, independent of pid.
  if (definition->getNumRegions())
    return true;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return mayReadProgramId(operand, programId, budget);
  });
}

// Rebuild value's defining graph, substituting folded for programId.
// Return null on budget exhaustion or an unsafe operation so the caller
// rejects the entire bound, never a partially substituted expression.
Value substitute(OpBuilder &builder, Value value, Value programId, Value folded,
                 BatchValueRemap remap, unsigned &budget) {
  if (value == programId)
    return folded;
  if (!budget)
    return nullptr;
  --budget;
  // Remap program-independent subtrees through the caller instead of
  // cloning them. Retaining the original values would leave dangling
  // references when the original function body is erased.
  unsigned probe = budget;
  if (!mayReadProgramId(value, programId, probe))
    return remap(value);
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return remap(value); // Independent values, such as function arguments.
  // An operation with regions can capture programId without listing it as
  // an operand. Check regions before concluding that unchanged operands
  // imply independence; otherwise a bound could retain the original ID.
  // This check must precede that inference, not just the final clone.
  if (definition->getNumRegions())
    return nullptr;
  SmallVector<Value> operands;
  bool changed = false;
  for (Value operand : definition->getOperands()) {
    Value rebuilt =
        substitute(builder, operand, programId, folded, remap, budget);
    if (!rebuilt)
      return nullptr;
    changed |= rebuilt != operand;
    operands.push_back(rebuilt);
  }
  if (!changed)
    return remap(value); // Probe budget exhausted; defer to the caller.
  // Cloning a memory read changes access multiplicity. Reject it unless
  // equivalence is proved.
  if (!isMemoryEffectFree(definition))
    return nullptr;
  Operation *clone = builder.clone(*definition);
  clone->setOperands(operands);
  // Select the matching result index; value need not be the first result.
  return clone->getResult(cast<OpResult>(value).getResultNumber());
}

} // namespace

BatchResult<std::array<Value, 3>>
retileBatchLoopBounds(OpBuilder &builder, scf::ForOp loop,
                      const BatchLogicalAxis &axis, Value programId,
                      Value foldedProgramId, BatchValueRemap remap) {
  // Both distributions express bounds in terms of the program ID: cyclic
  // starts at p*R; blocked spans p*R to min((p+1)*R, N). Substituting the
  // folded lane ID recovers that lane's segment without a distribution
  // branch. A blocked tail may change a lane's trip count; body masks handle
  // that difference.
  using Result = BatchResult<std::array<Value, 3>>;
  auto decline = [] {
    return Result::decline({"LOOP_RETILING_FAILED", "retile", {}, {}, {}, {}});
  };
  (void)axis;
  if (!programId || !foldedProgramId ||
      programId.getType() != foldedProgramId.getType())
    return decline();

  unsigned budget = kBudget;
  Value lower = substitute(builder, loop.getLowerBound(), programId,
                           foldedProgramId, remap, budget);
  Value upper = substitute(builder, loop.getUpperBound(), programId,
                           foldedProgramId, remap, budget);
  // Do not substitute the program ID in the step, but still remap its
  // original-IR values. tt.get_num_programs maps to the hidden original-grid
  // argument, so the step uses G, not G/F. Multiplying it by F again would
  // produce G*R*F and skip (F-1)/F of the rows.
  Value step = remap(loop.getStep());
  if (!lower || !upper || !step)
    return decline();
  return Result::success({lower, upper, step});
}

} // namespace mlir::triton::cfg
