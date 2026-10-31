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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::triton::ptr {

/// Check actual dependencies at the caller's current insertion point.
LogicalResult checkInsertionPoint(OpBuilder &builder,
                                  ArrayRef<Value> dependencies);

/// A session on stable source IR, with one result type and recursive entry.
/// Rules provides parse(Value, context), validateBinding(Value, Result), and
/// clearCaches(). Parsers place helpers next to their source definitions so
/// cached results dominate later uses. End the session before
/// rewriting/deleting source IR or running CSE/canonicalization; appending pure
/// helpers is allowed.
template <typename Rules> class PtrAnalysis {
public:
  using Result = typename Rules::Result;
  PtrAnalysis(OpBuilder &builder, Rules rules)
      : builder(builder), rules(std::move(rules)) {}
  PtrAnalysis(const PtrAnalysis &) = delete;
  PtrAnalysis &operator=(const PtrAnalysis &) = delete;
  OpBuilder &getBuilder() const { return builder; }
  const Rules &getRules() const { return rules; }

  FailureOr<Result> analyze(Value value) {
    if (!value)
      return failure();
    if (const Result *info = findCached(value))
      return *info;
    if (!active.insert(value).second)
      return failure();
    auto restore = llvm::make_scope_exit([&] { active.erase(value); });
    OpBuilder::InsertionGuard guard(builder);
    auto info = rules.parse(value, *this);
    if (failed(info))
      return failure();
    return cache.try_emplace(value, std::move(*info)).first->second;
  }
  /// Borrow only until the next analyze/bind/clear or session destruction.
  /// Copy a result before recursive queries: DenseMap growth moves entries.
  /// Result copies still reference the original SSA and require stable IR.
  const Result *findCached(Value value) const {
    auto bound = bindings.find(value);
    if (bound != bindings.end())
      return &bound->second;
    auto found = cache.find(value);
    return found == cache.end() ? nullptr : &found->second;
  }
  /// The caller proves equivalence; Rules checks the field contract. Every
  /// successful bind clears computed results, preserving all other bindings.
  /// Bound SSA fields must be available wherever the bound source is used;
  /// parsers do not move helpers to accommodate fields defined later.
  LogicalResult bind(Value value, const Result &info) {
    if (!value || !active.empty() || failed(rules.validateBinding(value, info)))
      return failure();
    // info may itself be borrowed from cache or bindings.
    Result boundInfo = info;
    clearComputed();
    bindings.insert_or_assign(value, std::move(boundInfo));
    return success();
  }
  void clear() {
    assert(active.empty() && "cannot clear during analysis");
    clearComputed();
    bindings.clear();
  }

private:
  void clearComputed() {
    cache.clear();
    rules.clearCaches();
  }
  OpBuilder &builder;
  Rules rules;
  llvm::DenseSet<Value> active;
  llvm::DenseMap<Value, Result> cache, bindings;
};

} // namespace mlir::triton::ptr
#endif
