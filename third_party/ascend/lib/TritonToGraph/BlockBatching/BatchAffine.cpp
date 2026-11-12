// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAffine.h"
#include "llvm/Support/MathExtras.h"
#include <utility>
namespace mlir::triton::cfg {
namespace {
using E = BatchExprRef;
using K = BatchExpr;
} // namespace
std::optional<BatchLinearForm> batchLinearForm(const E &e, unsigned &budget,
                                               BatchLeafMask leaves) {
  if (!e || !budget)
    return std::nullopt;
  --budget;
  if (e->kind == K::Constant)
    return BatchLinearForm{e->value, {}};
  if (batchLeaf(e->kind) & leaves)
    return BatchLinearForm{0, {{{e->kind, e->value}, 1}}};
  // Signed extension is exact under the existing no-wrap guard. Unsigned
  // extension and truncation require additional range facts: KEEP for now.
  if (e->kind == K::ExtSI)
    return batchLinearForm(e->lhs, budget, leaves);
  if (e->kind != K::Add && e->kind != K::Sub && e->kind != K::Mul)
    return std::nullopt;
  auto a = batchLinearForm(e->lhs, budget, leaves);
  auto b = batchLinearForm(e->rhs, budget, leaves);
  if (!a || !b)
    return std::nullopt;
  if (e->kind == K::Mul) {
    if (!a->terms.empty() && !b->terms.empty())
      return std::nullopt;
    if (b->terms.empty())
      std::swap(a, b);
    int64_t scale = a->constant;
    if (llvm::MulOverflow(b->constant, scale, b->constant))
      return std::nullopt;
    for (auto &[key, value] : b->terms)
      if (llvm::MulOverflow(value, scale, value))
        return std::nullopt;
    return b;
  }
  if (e->kind == K::Sub) {
    if (llvm::MulOverflow(b->constant, int64_t(-1), b->constant))
      return std::nullopt;
    for (auto &[key, value] : b->terms)
      if (llvm::MulOverflow(value, int64_t(-1), value))
        return std::nullopt;
  }
  if (llvm::AddOverflow(a->constant, b->constant, a->constant))
    return std::nullopt;
  // Zero coefficients are kept rather than erased: callers decide whether a
  // vanished term means "absent" or "present with weight 0".
  for (auto [key, value] : b->terms)
    if (llvm::AddOverflow(a->terms[key], value, a->terms[key]))
      return std::nullopt;
  return a;
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
