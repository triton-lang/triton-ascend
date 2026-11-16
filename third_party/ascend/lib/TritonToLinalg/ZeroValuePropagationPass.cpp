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
//===----------------------------------------------------------------------===//
// ZeroValuePropagationPass
//
// TTIR-level zero-value propagation + algebraic folding. It runs at the very
// head of the `ttir_to_linalg` pipeline, i.e. while `tt.splat`,
// `tt.broadcast`, `tt.expand_dims` and `tt.reduce` still carry explicit shapes.
// Folding here removes the arithmetic *and* the [BV, BK] UB tile that the
// statement would otherwise materialize, which a post-bufferization pass
// (NPUIR) can no longer do.
//
// Two kinds of rewrites:
//
//   * bit-exact identities, always applied:
//       x * 1 -> x            (exact for nan/inf/-0 as well)
//       x + 0 -> x
//       x - 0 -> x
//       reduce_add(zero_tensor) -> zero_tensor
//       splat/broadcast/expand_dims(zero) -> zero
//
//   * `x * 0 -> 0`, which holds only for finite `x`. Applied when the other
//     operand is a compile-time finite constant, or when
//     `allow-fp-mul-zero-fold` is set. This is the only rule that is not
//     bit-exact, and the only one the option controls; the standalone pass
//     defaults it to off, while the Ascend backend exposes it as the
//     `allow_fp_mul_zero_fold` launch option, on by default. Relaxed folds emit
//     a warning so the assumption stays visible.
//
// NaN / Inf
// ---------
// `x * 0` is the only rule that can change NaN/Inf propagation: IEEE says
// `0 * inf = nan` and `0 * nan = nan`, while the fold produces a plain `0`.
// Every other rule preserves NaN and Inf bit-for-bit, so it is always applied.
//
// Signed zero
// -----------
// `x + 0 -> x` and `x - 0 -> x` differ from IEEE only at `x = -0.0`, and only
// in the sign of the resulting zero:
//     (-0) + (+0) = +0        (-0) - (-0) = +0
// i.e. the folded value keeps the magnitude and flips only a zero's sign, which
// is observable solely through `1/x`, `signbit`, `copysign` and `atan2`. Those
// folds are therefore applied for both constant signs.
//
// Tracking which zeros are provably `+0` and refusing the rest was implemented
// and then reverted: on the recurrent gating kernels the prediction is
// `b_v -= reduce_sum(b_h * k)` fed by an all-zero `b_h`, so rejecting the
// sign-unknown case keeps the entire `exp`/`log`/softplus decay chain alive for
// no observable benefit (measured 1.6x slower).
//
// `x * 0 -> 0` is a separate case: the product's sign is `sign(x) xor sign(0)`
// and `x`'s sign is unknown, so a folded product is recorded as "zero, sign
// unknown" and no rule ever treats it as `+0`.
//
// Integers
// --------
// `arith.muli` / `arith.addi` / `arith.subi` and integer `reduce_add` get the
// same zero rules, but unconditionally and with no option: two's complement
// integers have no NaN, no Inf and no signed zero, so `x * 0 -> 0`,
// `x + 0 -> x` and `x - 0 -> x` are exact for every `x`. `i1` goes through the
// same path with `0 == false`, which is what lets a `false` mask literal
// survive `tt.splat` / `tt.broadcast` as a zero; combining masks uses
// `arith.andi`, which is not a zero-value identity and is left alone.
//
// Scope is deliberately local: only values that are provably zero by
// construction are tracked, and the map never crosses a region or a function
// boundary on its own (loop-carried values are simply unknown). No inference is
// made about loads, selects or masked values.
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonToLinalg/ZeroValuePropagationPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "zero-value-propagation"

using namespace mlir;
using namespace mlir::triton;

namespace {

/// True when the attribute is a float constant whose every element is zero,
/// including `-0.0`: the pass folds `x + 0 -> x` and `x - 0 -> x` for both
/// signs of the constant (see the "Signed zero" note at the top of the file).
///
/// The null check is load-bearing: `getConstantAttr` returns an empty
/// attribute for "not a constant", and `dyn_cast` on an empty `Attribute` hits
/// an assertion in assertion-enabled builds.
bool isZeroFloatAttr(Attribute attr) {
  if (!attr)
    return false;
  if (auto f = dyn_cast<FloatAttr>(attr))
    return f.getValue().isZero();
  if (auto d = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (d.empty())
      return false;
    for (APFloat v : d.getValues<APFloat>()) {
      if (!v.isZero())
        return false;
    }
    return true;
  }
  return false;
}

/// True when the attribute is an integer constant whose every element is zero.
///
/// `IntegerAttr` also matches `BoolAttr`, i.e. the `i1` case where zero is
/// `false`. That is deliberate: a mask literal reaches a tensor through
/// `tt.splat` / `tt.broadcast` of a scalar `false`, and tracking it keeps the
/// zero-ness available to the integer folds below. `index` constants are
/// `IntegerAttr` too, and folding them is equally exact.
///
/// Integer values need none of the caveats that the float helpers carry: there
/// is no NaN, no Inf and no signed zero, so `x * 0 -> 0` always holds.
bool isZeroIntAttr(Attribute attr) {
  if (!attr)
    return false;
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return i.getValue().isZero();
  if (auto d = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (d.empty())
      return false;
    for (APInt v : d.getValues<APInt>()) {
      if (!v.isZero())
        return false;
    }
    return true;
  }
  return false;
}

/// True for a float or an integer constant whose every element is zero.
bool isZeroAttr(Attribute attr) {
  return isZeroFloatAttr(attr) || isZeroIntAttr(attr);
}

/// True when the attribute is a float constant whose every element is finite.
bool isFiniteFloatAttr(Attribute attr) {
  if (!attr)
    return false;
  if (auto f = dyn_cast<FloatAttr>(attr)) {
    const APFloat &v = f.getValue();
    return !v.isNaN() && !v.isInfinity();
  }
  if (auto d = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (d.empty())
      return false;
    for (APFloat v : d.getValues<APFloat>()) {
      if (v.isNaN() || v.isInfinity())
        return false;
    }
    return true;
  }
  return false;
}

/// True when the attribute is a float constant whose every element is one.
bool isOneFloatAttr(Attribute attr) {
  if (!attr)
    return false;
  if (auto f = dyn_cast<FloatAttr>(attr))
    return f.getValue().isExactlyValue(1.0);
  if (auto d = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (d.empty())
      return false;
    for (APFloat v : d.getValues<APFloat>()) {
      if (!v.isExactlyValue(1.0))
        return false;
    }
    return true;
  }
  return false;
}

/// Returns the attribute behind an `arith.constant`, or a null attribute when
/// `v` is produced by anything else. Float and integer constants both come
/// through here; callers pick the matching predicate.
Attribute getConstantAttr(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return cst.getValue();
  return Attribute();
}

/// An op whose only effect is producing its results, so it may be dropped once
/// they are unused. `tt.load` reads memory, so it is handled separately.
bool isDeadRemovable(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return !load.getIsVolatile();
  return isMemoryEffectFree(op);
}

/// Per-module state and rewrites for the zero-value propagation. All folds are
/// recorded during a pre-order walk and materialized afterwards, because
/// rewriting an op while the walk is in progress invalidates the walk.
struct ZeroValuePropagator {
  explicit ZeroValuePropagator(bool allowFpMulZeroFold)
      : allowFpMulZeroFold(allowFpMulZeroFold) {}

  // Whether `x * 0 -> 0` may be folded without proving `x` finite.
  bool allowFpMulZeroFold;
  // value -> replacement value, for statements that are exactly equal to one
  // of their operands (`x + 0`, `x - 0`, `x * 1`).
  llvm::DenseMap<Value, Value> alias;
  // Values proven to be all-zero, sign unknown (it may be `+0.0` or `-0.0`).
  // The sign is deliberately not tracked: `x + 0` / `x - 0` are folded for
  // both, they can only differ from IEEE in the sign of a zero result.
  llvm::DenseSet<Value> zeroVals;
  // How many `x * 0 -> 0` folds relied on `allowFpMulZeroFold` rather than on
  // `x` being a finite constant.
  unsigned relaxedMulFolds = 0;

  Value resolve(Value v) {
    while (alias.count(v))
      v = alias.lookup(v);
    return v;
  }

  bool isZero(Value v) { return zeroVals.contains(resolve(v)); }

  void visit(Operation *op);
  void foldMulF(arith::MulFOp mul);
  bool allowMulByZeroFold(Attribute otherAttr);
  void foldMulI(arith::MulIOp mul);
  void foldAddF(arith::AddFOp add);
  void foldSubF(arith::SubFOp sub);
  void foldAddI(arith::AddIOp add);
  void foldSubI(arith::SubIOp sub);
  void foldReduce(triton::ReduceOp reduce);
  void applyAliases();
  void eraseDeadCode(ModuleOp module);
  void publishRelaxedFoldCount(ModuleOp module);
};

/// Dispatch one op of the walk to the matching fold. All-zero constants and
/// shape-only ops only extend `zeroVals`; everything else is either a foldable
/// arithmetic op or ignored.
void ZeroValuePropagator::visit(Operation *op) {
  // All-zero constants, in scalar (`f32`, `i32`, `i1`, ...) and tensor
  // (`dense<0.0>`, `dense<0>`) form. `-0.0` counts as a zero here, which is
  // what enables the `x + 0 -> x` and `x - 0 -> x` folds below.
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    if (isZeroAttr(cst.getValue()))
      zeroVals.insert(cst.getResult());
    return;
  }
  // Shape-only ops preserve the payload, and a zero stays a zero.
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(op)) {
    if (isZero(op->getOperand(0)))
      zeroVals.insert(op->getResult(0));
    return;
  }
  if (auto mul = dyn_cast<arith::MulFOp>(op)) {
    foldMulF(mul);
    return;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    foldMulI(mul);
    return;
  }
  if (auto add = dyn_cast<arith::AddFOp>(op)) {
    foldAddF(add);
    return;
  }
  if (auto sub = dyn_cast<arith::SubFOp>(op)) {
    foldSubF(sub);
    return;
  }
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    foldAddI(add);
    return;
  }
  if (auto sub = dyn_cast<arith::SubIOp>(op)) {
    foldSubI(sub);
    return;
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op))
    foldReduce(reduce);
}

void ZeroValuePropagator::foldMulF(arith::MulFOp mul) {
  Value lhs = mul.getLhs();
  Value rhs = mul.getRhs();
  Attribute lhsAttr = getConstantAttr(resolve(lhs));
  Attribute rhsAttr = getConstantAttr(resolve(rhs));
  bool lhsZero = isZero(lhs);
  bool rhsZero = isZero(rhs);
  // `0 * 0` is exactly a zero for every combination of signs, and it cannot
  // produce a NaN (`0 * inf` needs a non-zero `inf`), so this branch needs no
  // flag check. The sign of the result is unknown, so it is never treated as
  // `+0`.
  if (lhsZero && rhsZero) {
    zeroVals.insert(mul.getResult());
    return;
  }
  // `0 * x -> 0`. The other operand can never be zero here -- that is the
  // both-zero branch above -- so no zero test is needed.
  if (lhsZero || rhsZero) {
    if (allowMulByZeroFold(lhsZero ? rhsAttr : lhsAttr))
      zeroVals.insert(mul.getResult());
    return;
  }
  // `x * 1 -> x` is exact, but only without a broadcast: the identity does not
  // hold when the result shape differs from the operand type.
  if (lhsAttr && isOneFloatAttr(lhsAttr) &&
      lhs.getType() == mul.getResult().getType()) {
    alias[mul.getResult()] = rhs;
    return;
  }
  if (rhsAttr && isOneFloatAttr(rhsAttr) &&
      rhs.getType() == mul.getResult().getType())
    alias[mul.getResult()] = lhs;
}

/// Decide whether `0 * x -> 0` may be folded, given the constant form of the
/// non-zero operand. The fold is exact only for finite `x`, because IEEE
/// defines `0 * inf = nan` and `0 * nan = nan`. Two ways to be allowed here:
///   * `x` is a compile-time constant that is finite, or
///   * the caller opted into the relaxation via `allow-fp-mul-zero-fold`
///     (off for the standalone pass; the Ascend launch option defaults to on).
/// Only the second case relies on an unproven assumption, so only it is
/// counted in `relaxedMulFolds`.
bool ZeroValuePropagator::allowMulByZeroFold(Attribute otherAttr) {
  bool otherIsFinite = isFiniteFloatAttr(otherAttr);
  if (!otherIsFinite && !allowFpMulZeroFold)
    return false;
  // The result sign is `sign(x) xor sign(zero)`, and `x` has no tracked sign,
  // so the caller records `zero` but never `+0`.
  if (!otherIsFinite)
    ++relaxedMulFolds;
  return true;
}

void ZeroValuePropagator::foldMulI(arith::MulIOp mul) {
  // Integer counterpart of the `arith.mulf` case above. No flag check needed:
  // two's complement integers have no NaN, no Inf and no signed zero, so
  // `x * 0 -> 0` (and `0 * 0`) is exact for every `x` and is never counted in
  // `relaxedMulFolds`. `x * 1 -> x` is not repeated here: it is not a
  // zero-value rule and arith canonicalization already covers it.
  if (isZero(mul.getLhs()) || isZero(mul.getRhs()))
    zeroVals.insert(mul.getResult());
}

void ZeroValuePropagator::foldAddF(arith::AddFOp add) {
  // `x + 0 -> x` preserves NaN and Inf exactly, and is exact for every `x`
  // when the constant is `-0.0`. With `+0.0` it maps `x = -0.0` to `+0.0`
  // (`(-0) + (+0) = +0`). The fold is still applied for both signs: a zero's
  // sign is only observable through `1/x`, `signbit` or `atan2`, and rejecting
  // the `+0` case would also reject the `h = 0 + outer(k, v)` statement, i.e.
  // the zero-state tile this pass exists to remove.
  if (isZero(add.getLhs())) {
    alias[add.getResult()] = add.getRhs();
    return;
  }
  if (isZero(add.getRhs()))
    alias[add.getResult()] = add.getLhs();
}

void ZeroValuePropagator::foldSubF(arith::SubFOp sub) {
  // `x - 0 -> x`. Exact for every `x` -- including NaN, Inf and `-0.0` -- when
  // the constant is `+0.0`. With `-0.0` it becomes `x + (+0)`, which maps
  // `x = -0.0` to `+0.0`; the difference is the sign of a zero, so the fold is
  // accepted for both signs (see "Signed zero" above). Note `(+0) - (+0)` also
  // lands here and correctly folds to `+0`.
  if (isZero(sub.getRhs()))
    alias[sub.getResult()] = sub.getLhs();
}

void ZeroValuePropagator::foldAddI(arith::AddIOp add) {
  // `x + 0 -> x`, exact for every integer. There is no analogue of the
  // signed-zero caveat above: an integer zero has no sign to flip.
  if (isZero(add.getLhs())) {
    alias[add.getResult()] = add.getRhs();
    return;
  }
  if (isZero(add.getRhs()))
    alias[add.getResult()] = add.getLhs();
}

void ZeroValuePropagator::foldSubI(arith::SubIOp sub) {
  // `x - 0 -> x`, exact for every integer, `INT_MIN` included (subtracting
  // zero cannot overflow). `0 - x -> x` is deliberately not folded: the result
  // would be `-x`, which overflows at `INT_MIN`.
  if (isZero(sub.getRhs()))
    alias[sub.getResult()] = sub.getLhs();
}

void ZeroValuePropagator::foldReduce(triton::ReduceOp reduce) {
  if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1)
    return;
  Operation *combiner = reduce.getSingleCombiner();
  if (!combiner || !isa<arith::AddFOp, arith::AddIOp>(combiner))
    return;
  // Summing a tensor whose elements are all zero gives a zero. There can be no
  // NaN/Inf in it (they are not zero), so this fold is exact for NaN/Inf; only
  // the sign of the zero is at stake, and `+0 + -0 = +0` keeps the result a
  // zero either way. For integers the same holds trivially, including `i1`
  // where `arith.addi` wraps modulo 2.
  if (isZero(reduce->getOperand(0)))
    zeroVals.insert(reduce->getResult(0));
}

/// Materialize the operand-equal folds recorded during the walk.
void ZeroValuePropagator::applyAliases() {
  for (auto &entry : alias) {
    Value from = entry.first;
    Value to = resolve(entry.second);
    if (from != to)
      from.replaceAllUsesWith(to);
  }
}

/// Drop everything the folds made dead, including the scalar `tt.load`s of the
/// decay chain. Iterated to a fixpoint because erasing an op can make its
/// producers dead.
void ZeroValuePropagator::eraseDeadCode(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<Operation *, 32> dead;
    module.walk([&](Operation *op) {
      if (!isDeadRemovable(op))
        return;
      if (llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); }))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[zero-value-propagation] erase: " << *op << "\n");
      op->erase();
    }
    changed = !dead.empty();
  }
}

/// Report the number of folds that relied on the IEEE relaxation, so the
/// driver can surface it to the user. It is published as a module attribute
/// rather than through `emitWarning` on purpose: in this stack MLIR
/// diagnostics below `error` are filtered out unless
/// `MLIR_ENABLE_DIAGNOSTICS=warnings` is set (see
/// `setupTritonDiagnosticHandler` in python/src/ir.cc), which would make a
/// warning about changed NaN/Inf semantics invisible by default.
/// `backend/compiler.py` reads the attribute after the pass manager runs and
/// raises a Python warning. One aggregated value per module, so a kernel with
/// many such statements does not spam the log.
void ZeroValuePropagator::publishRelaxedFoldCount(ModuleOp module) {
  if (relaxedMulFolds == 0)
    return;
  module->setAttr("tt.fp_mul_zero_folds",
                  IntegerAttr::get(IntegerType::get(module.getContext(), 32),
                                   relaxedMulFolds));
}

struct ZeroValuePropagationPass
    : public PassWrapper<ZeroValuePropagationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroValuePropagationPass)

  ZeroValuePropagationPass() = default;
  explicit ZeroValuePropagationPass(bool allowFpMulZeroFold) {
    this->allowFpMulZeroFold = allowFpMulZeroFold;
  }
  // `Option` is not copy-constructible, so the implicit one is deleted and
  // `clonePass` would not compile. Copy the option values explicitly.
  ZeroValuePropagationPass(const ZeroValuePropagationPass &other)
      : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }

  // Named after the multiplication on purpose: every other fold in this pass is
  // a bit-exact identity and is applied unconditionally. This option guards the
  // single rule that is not, `x * 0 -> 0`, whose validity depends on `x`.
  Option<bool> allowFpMulZeroFold{
      *this, "allow-fp-mul-zero-fold",
      llvm::cl::desc(
          "Allow folding `x * 0 -> 0` when `x` is not proven finite. Only this "
          "rule is affected: IEEE defines `0 * inf = nan` and `0 * nan = nan`, "
          "so the fold changes the result for non-finite multiplicands. Off by "
          "default for the standalone pass; the Ascend backend exposes it as "
          "the `allow_fp_mul_zero_fold` launch option, which defaults to on."),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "zero-value-propagation"; }

  StringRef getDescription() const final {
    return "Propagate all-zero values at TTIR level and fold the algebraic "
           "identities that follow.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The folds are recorded during the walk and materialized afterwards:
    // rewriting an op while the walk is in progress invalidates the walk.
    ZeroValuePropagator propagator(allowFpMulZeroFold);
    module.walk<WalkOrder::PreOrder>(
        [&](Operation *op) { propagator.visit(op); });

    propagator.applyAliases();
    propagator.eraseDeadCode(module);

    LLVM_DEBUG(llvm::dbgs() << "[zero-value-propagation] folded "
                            << propagator.alias.size() << " statement(s)\n");

    // Surface the folds that relied on the IEEE relaxation.
    propagator.publishRelaxedFoldCount(module);
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createZeroValuePropagationPass(bool allowFpMulZeroFold) {
  return std::make_unique<ZeroValuePropagationPass>(allowFpMulZeroFold);
}

} // namespace triton
} // namespace mlir
