// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAffine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <utility>
namespace mlir::triton::cfg {
namespace {
using E = BatchExprRef;
using K = BatchExpr;
} // namespace
bool BatchAffineForm::Coefficient::addScaled(const Coefficient &other,
                                             int64_t scale) {
  int64_t value;
  if (llvm::MulOverflow(other.constant, scale, value) ||
      llvm::AddOverflow(constant, value, constant))
    return false;
  for (auto [atom, coefficient] : other.parameters) {
    if (llvm::MulOverflow(coefficient, scale, value) ||
        llvm::AddOverflow(parameters[atom], value, parameters[atom]))
      return false;
    if (!parameters[atom])
      parameters.erase(atom);
  }
  return parameters.size() <= 32;
}
std::optional<BatchAffineForm::Coefficient>
BatchAffineForm::Coefficient::multiply(const Coefficient &other) const {
  if (!parameters.empty() && !other.parameters.empty())
    return std::nullopt;
  Coefficient result;
  bool ok = parameters.empty() ? result.addScaled(other, constant)
                               : result.addScaled(*this, other.constant);
  return ok ? std::optional<Coefficient>(result) : std::nullopt;
}
std::optional<BatchAffineForm::Coefficient>
BatchAffineForm::Coefficient::divide(int64_t divisor) const {
  if (divisor <= 0 || constant % divisor)
    return std::nullopt;
  Coefficient result{constant / divisor, {}};
  for (auto [atom, coefficient] : parameters) {
    if (coefficient % divisor)
      return std::nullopt;
    result.parameters[atom] = coefficient / divisor;
  }
  return result;
}
BatchExprRef BatchAffineForm::Coefficient::expression() const {
  BatchExprRef result = BatchExpr::leaf(BatchExpr::Constant, constant, 64);
  for (auto [atom, coefficient] : parameters) {
    auto [kind, id, bits] = atom;
    BatchExprRef term = BatchExpr::leaf(kind, id, bits);
    if (bits != 64)
      term = BatchExpr::node(BatchExpr::ExtSI, 64, term, {}, bits);
    if (coefficient != 1)
      term = BatchExpr::node(
          BatchExpr::Mul, 64, term,
          BatchExpr::leaf(BatchExpr::Constant, coefficient, 64));
    if (result->kind == BatchExpr::Constant && result->value == 0)
      result = term;
    else
      result = BatchExpr::node(BatchExpr::Add, 64, result, term);
  }
  return result;
}

bool BatchAffineForm::symbolic() const {
  return !offset.parameters.empty() ||
         llvm::any_of(terms, [](const auto &term) {
           return !term.second.parameters.empty();
         });
}

std::optional<BatchAffineForm>
BatchAffineForm::fromExpr(const E &e, unsigned &budget, BatchLeafMask leaves,
                          Profile profile, int axis) {
  if (!e || !budget)
    return std::nullopt;
  --budget;
  if (e->kind == K::Constant)
    return BatchAffineForm{Coefficient{e->value, {}}, {}};
  if (profile == Profile::Parameter && e->kind == K::Argument &&
      (leaves & batchLeaf(e->kind))) {
    if (e->bits != 32 && e->bits != 64)
      return std::nullopt;
    Coefficient value;
    value.parameters[{e->kind, e->value, e->bits}] = 1;
    return BatchAffineForm{value, {}};
  }
  // Induction 在这里和坐标同级：normalizeBatchAccessOnLogicalAxis 已经把循环
  // 变量换成「下界 + 圈内偏移」，剩下的这个叶子代表圈内偏移，各份相同。
  const bool division =
      e->kind == K::ProgramQuotient || e->kind == K::ProgramRemainder;
  if ((leaves & batchLeaf(e->kind)) &&
      (profile == Profile::Integer || !division)) {
    BatchAffineForm value;
    value.terms[{e->kind, e->value, 0}] = Coefficient{1, {}};
    return value;
  }
  // Quotient/remainder of another program axis is uniform within this
  // batch. Preserve its divisor and existing proof rather than treating it
  // as a linear function of the selected axis.
  if (profile == Profile::Parameter && division &&
      (leaves & batchLeaf(e->kind))) {
    if (e->value == axis || e->extent <= 0)
      return std::nullopt;
    BatchAffineForm value;
    value.terms[{e->kind, e->value, e->extent}] = Coefficient{1, {}};
    return value;
  }
  if (e->kind == K::ExtSI)
    return fromExpr(e->lhs, budget, leaves, profile, axis);
  if (e->kind != K::Add && e->kind != K::Sub && e->kind != K::Mul)
    return std::nullopt;
  auto a = fromExpr(e->lhs, budget, leaves, profile, axis);
  auto b = fromExpr(e->rhs, budget, leaves, profile, axis);
  if (!a || !b)
    return std::nullopt;
  if (e->kind == K::Mul) {
    if (!a->terms.empty() && !b->terms.empty())
      return std::nullopt;
    if (b->terms.empty())
      std::swap(a, b);
    auto offset = b->offset.multiply(a->offset);
    if (!offset)
      return std::nullopt;
    b->offset = *offset;
    for (auto &[key, coefficient] : b->terms) {
      auto product = coefficient.multiply(a->offset);
      if (!product)
        return std::nullopt;
      coefficient = *product;
    }
    for (auto it = b->terms.begin(); it != b->terms.end();)
      if (profile == Profile::Parameter && it->second.isZero())
        it = b->terms.erase(it);
      else
        ++it;
    return b;
  }
  int64_t sign = e->kind == K::Sub ? -1 : 1;
  if (!a->offset.addScaled(b->offset, sign))
    return std::nullopt;
  for (const auto &[key, coefficient] : b->terms) {
    if (!a->terms[key].addScaled(coefficient, sign))
      return std::nullopt;
    if (profile == Profile::Parameter && a->terms[key].isZero())
      a->terms.erase(key);
  }
  return profile == Profile::Integer || a->terms.size() <= 32 ? a
                                                              : std::nullopt;
}
bool batchExprEquivalent(const BatchExprRef &a, const BatchExprRef &b) {
  // 共享子树在界表达式里很常见（段首段尾读的是同一个 SSA 值），先比指针既是
  // 快路径，也让递归深度跟着 DAG 的共享结构走而不是展开后的树。
  if (a == b)
    return true;
  if (!a || !b)
    return false;
  return a->kind == b->kind && a->bits == b->bits && a->value == b->value &&
         a->extent == b->extent && a->rangeOrigin == b->rangeOrigin &&
         batchExprEquivalent(a->lhs, b->lhs) &&
         batchExprEquivalent(a->rhs, b->rhs);
}
} // namespace mlir::triton::cfg
