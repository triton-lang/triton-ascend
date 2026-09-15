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

// Atomic mask recovery eliminates identity-valued branches in atomic inputs.
//
// Encoding an inactive lane as an identity value, such as zero for ADD,
// still permits an atomic access to global memory, with unnecessary traffic
// and contention. Generated kernels often express these values with tl.where.
//
// This graph rule recovers the predicate before memory-access lowering:
//   atomic(ptr, where(active, value, identity), mask)
// becomes:
//   atomic(ptr, value, mask & active).
// It follows nested selects and supported shape operations, preserves shared
// value users, and obtains each RMW's identity from the existing table (e.g.
// zero for ADD, +infinity for floating MIN, and all ones for AND). Pure
// identity atomics are removed. Arithmetic and bitcasts are not traversed
// because they need not preserve the identity of the atomic operation.
//
// The recovered mask lets native predication or downstream mask lowering skip
// inactive accesses. Surviving atomics keep their operation,
// memory semantics, and scope.
//
// Only unused-result atomics are eligible. Elimination assumes numerical
// reductions whose identity updates are not synchronization events and whose
// exact signed-zero, NaN, and denormal effects need not be preserved.

#include "TritonToGraph/GraphOptimizationRule.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

#include <utility>

namespace mlir::triton::cfg {

namespace {

static bool isAtomicShapeOp(Operation *op) {
  return op && isa<triton::BroadcastOp, triton::SplatOp, triton::ExpandDimsOp,
                   triton::ReshapeOp, triton::TransOp>(op);
}

static bool isAtomicIdentity(Value value, TypedAttr identity) {
  while (isAtomicShapeOp(value.getDefiningOp()))
    value = value.getDefiningOp()->getOperand(0);
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  Attribute attr = constant.getValue();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (!dense.isSplat())
      return false;
    attr = dense.getSplatValue<Attribute>();
  }
  // Compare with the selected RMW's identity. Numeric FP equality handles
  // +/-infinity for MIN/MAX and both signs of zero for FADD uniformly.
  if (auto fp = dyn_cast<FloatAttr>(attr))
    if (auto expected = dyn_cast<FloatAttr>(identity))
      return fp.getValue().compare(expected.getValue()) == APFloat::cmpEqual;
  return attr == identity;
}

static Value cloneAtomicShapeOp(Operation *op, Value source, Type type,
                                IRRewriter &rewriter) {
  IRMapping mapping;
  mapping.map(op->getOperand(0), source);
  auto *clone = rewriter.clone(*op, mapping);
  clone->getResult(0).setType(type);
  return clone->getResult(0);
}

static Value invertAtomicPredicate(Value predicate, IRRewriter &rewriter) {
  auto *def = predicate.getDefiningOp();
  if (isAtomicShapeOp(def))
    return cloneAtomicShapeOp(
        def, invertAtomicPredicate(def->getOperand(0), rewriter),
        predicate.getType(), rewriter);
  Value one = rewriter.create<arith::ConstantOp>(
      predicate.getLoc(), rewriter.getOneAttr(predicate.getType()));
  return rewriter.create<arith::XOrIOp>(predicate.getLoc(), predicate, one);
}

// Follow only select/shape definitions of the atomic value. For example,
// select(a, identity, select(b, v, identity)) yields (v, !a & b).
// Do not traverse arithmetic, loads or casts: their results need not preserve
// the identity (e.g. 0 * NaN). Other users of the selects remain untouched.
// Return (value, mask); a null mask means no identity branch was found.
static std::pair<Value, Value> recoverAtomicContribution(Value value,
                                                         TypedAttr identity,
                                                         IRRewriter &rewriter) {
  Type maskType = triton::getI1SameShape(value.getType());
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    bool trueIsIdentity = isAtomicIdentity(select.getTrueValue(), identity);
    bool falseIsIdentity = isAtomicIdentity(select.getFalseValue(), identity);
    if (!trueIsIdentity && !falseIsIdentity)
      return {value, nullptr};
    Value predicate = select.getCondition();
    if (trueIsIdentity)
      predicate = invertAtomicPredicate(predicate, rewriter);
    if (predicate.getType() != maskType)
      predicate =
          rewriter.create<triton::SplatOp>(value.getLoc(), maskType, predicate);
    auto [innerValue, innerMask] = recoverAtomicContribution(
        trueIsIdentity ? select.getFalseValue() : select.getTrueValue(),
        identity, rewriter);
    if (innerMask)
      predicate =
          rewriter.create<arith::AndIOp>(value.getLoc(), predicate, innerMask);
    return {innerValue, predicate};
  }
  auto *def = value.getDefiningOp();
  if (isAtomicShapeOp(def)) {
    auto [innerValue, innerMask] =
        recoverAtomicContribution(def->getOperand(0), identity, rewriter);
    if (innerMask)
      return {cloneAtomicShapeOp(def, innerValue, value.getType(), rewriter),
              cloneAtomicShapeOp(def, innerMask, maskType, rewriter)};
  }
  return {value, nullptr};
}

static FailureOr<TypedAttr> matchAtomicMask(triton::AtomicRMWOp op) {
  if (!op.getResult().use_empty())
    return failure();
  OpBuilder builder(op.getContext());
  auto identity = getAtomicRMWIdentityAttr(op.getAtomicRmwOp(),
                                           op.getVal().getType(), builder);
  if (failed(identity))
    return failure();
  Value value = op.getVal();
  while (isAtomicShapeOp(value.getDefiningOp()))
    value = value.getDefiningOp()->getOperand(0);
  if (isAtomicIdentity(value, *identity))
    return identity;
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    if (isAtomicIdentity(select.getTrueValue(), *identity) ||
        isAtomicIdentity(select.getFalseValue(), *identity))
      return identity;
  return failure();
}

class AtomicMaskPlan final : public RewritePlan {
public:
  AtomicMaskPlan(triton::AtomicRMWOp op, unsigned epoch)
      : op(op), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::AtomicMaskCanonicalization;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return op; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    return success(op->getParentOfType<triton::FuncOp>() ==
                       context.getFunction() &&
                   succeeded(matchAtomicMask(op)));
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    auto identity = matchAtomicMask(op);
    if (failed(identity))
      return failure();
    rewriter.setInsertionPoint(op);
    auto src = op.getVal();

    if (isAtomicIdentity(src, *identity)) {
      rewriter.eraseOp(op);
    } else {
      auto [value, predicate] =
          recoverAtomicContribution(src, *identity, rewriter);
      assert(predicate && "candidate must contain an identity branch");
      if (Value mask = op.getMask(); mask && !matchPattern(mask, m_One()))
        predicate =
            rewriter.createOrFold<arith::AndIOp>(op.getLoc(), mask, predicate);
      rewriter.modifyOpInPlace(op, [&, value = value, predicate = predicate] {
        op.getValMutable().assign(value);
        op.getMaskMutable().assign(predicate);
      });
    }

    // Retire only unused select/shape/constant producers of this atomic.
    // Shared values and unrelated IR stay intact; other atomics require their
    // own plan and consume their own rewrite budget.
    llvm::SmallSetVector<Operation *, 16> dead;
    if (Operation *def = src.getDefiningOp())
      dead.insert(def);
    while (!dead.empty()) {
      Operation *def = dead.pop_back_val();
      if (!def->use_empty() || (!isAtomicShapeOp(def) &&
                                !isa<arith::SelectOp, arith::ConstantOp>(def)))
        continue;
      for (Value operand : def->getOperands())
        if (Operation *producer = operand.getDefiningOp())
          dead.insert(producer);
      rewriter.eraseOp(def);
    }
    return success();
  }

private:
  triton::AtomicRMWOp op;
  unsigned epoch;
};

class AtomicMaskRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::AtomicMaskCanonicalization;
  }
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }
  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::AtomicRMWOp op) {
      if (succeeded(matchAtomicMask(op)))
        plans.push_back(
            std::make_unique<AtomicMaskPlan>(op, context.getEpoch()));
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> createAtomicMaskCanonicalizationRule() {
  return std::make_unique<AtomicMaskRule>();
}

} // namespace mlir::triton::cfg
