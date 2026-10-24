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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_DRIVER_H
#define TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_DRIVER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <map>
#include <utility>

namespace mlir::triton::pointer::detail {

/// An extension can add requests without changing the shared traversal.
enum RequestKind : unsigned {
  SourceInteger,
  AddressDelta,
  PointerValue,
  AdvanceDeltaRequest,
  FirstExtensionRequest,
};

struct AnalysisRequest {
  Value value;
  unsigned kind = SourceInteger;
};

struct RequestLess {
  bool operator()(const AnalysisRequest &lhs,
                  const AnalysisRequest &rhs) const {
    auto less = std::less<const void *>();
    if (lhs.value != rhs.value)
      return less(lhs.value.getAsOpaquePointer(),
                  rhs.value.getAsOpaquePointer());
    return lhs.kind < rhs.kind;
  }
};

inline bool isOwnedBy(Value value, Operation *scope) {
  if (!scope || !value)
    return false;
  Operation *owner = value.getDefiningOp();
  if (!owner)
    owner = cast<BlockArgument>(value).getOwner()->getParentOp();
  return owner && (owner == scope || scope->isAncestor(owner));
}

/// Rules must expose Result, collectInputs, transfer, and mapResultValues.
/// Only this class traverses the producer graph; callers bind boundaries with
/// bindBoundary().
template <typename Rules> class AnalysisDriver {
public:
  using Result = typename Rules::Result;
  using Cache = std::map<AnalysisRequest, Result, RequestLess>;

  AnalysisDriver(OpBuilder &builder, Rules rules)
      : builder(builder), rules(std::move(rules)) {}

  FailureOr<Result> analyze(AnalysisRequest request) {
    if (!request.value)
      return failure();
    if (auto *completed = lookupCompleted(request))
      return *completed;
    FailureOr<SmallVector<AnalysisRequest>> inputs =
        rules.collectInputs(request);
    if (failed(inputs))
      return failure();
    SmallVector<Result, 1> operands;
    operands.reserve(inputs->size());
    for (AnalysisRequest input : *inputs) {
      FailureOr<Result> operand = analyze(input);
      if (failed(operand))
        return failure();
      operands.push_back(std::move(*operand));
    }
    OpBuilder::InsertionGuard guard(builder);
    // Component helpers may be later than the original producer. Include every
    // value exposed by the rules (including extension fields) in the placement
    // constraint, so subsequent requests never insert before their operands.
    const bool beforePointer =
        request.kind == PointerValue || request.kind == AdvanceDeltaRequest;
    Operation *producer = request.value.getDefiningOp();
    Block *block = producer ? producer->getBlock()
                            : cast<BlockArgument>(request.value).getOwner();
    auto point = producer ? producer->getIterator() : block->begin();
    if (producer && !beforePointer)
      ++point;
    SmallVector<Value> dependencies;
    if (!beforePointer)
      dependencies.push_back(request.value);
    for (const Result &operand : operands) {
      Result copy = operand;
      if (failed(
              rules.mapResultValues(copy, [&](Value value) -> FailureOr<Value> {
                dependencies.push_back(value);
                return value;
              })))
        return failure();
    }
    if (!beforePointer) {
      for (Value dependency : dependencies) {
        Operation *def = dependency.getDefiningOp();
        if (def && def->getBlock() == block && point != block->end() &&
            (def == &*point || point->isBeforeInBlock(def))) {
          if (def->hasTrait<OpTrait::IsTerminator>())
            return failure();
          point = std::next(def->getIterator());
        }
      }
    }
    DominanceInfo dominance;
    for (Value dependency : dependencies) {
      if (point != block->end()) {
        if (!dominance.properlyDominates(dependency, &*point))
          return failure();
      } else if (dependency.getParentBlock() != block &&
                 !dominance.dominates(dependency.getParentBlock(), block)) {
        return failure();
      }
    }
    builder.setInsertionPoint(block, point);
    FailureOr<Result> result = rules.transfer(request, operands, builder);
    if (failed(result))
      return failure();
    auto [it, inserted] = cache.emplace(request, std::move(*result));
    ++visits[request];
    return it->second;
  }

  const Result *lookupCompleted(AnalysisRequest request) const {
    auto found = bindings.find(request);
    if (found != bindings.end())
      return &found->second;
    found = cache.find(request);
    return found == cache.end() ? nullptr : &found->second;
  }

  /// A new, unvisited request cannot be an input to a completed result: every
  /// dependency is requested through analyze(). Preserve those results while
  /// callers prepare further independent boundaries. Replacing an observed
  /// request still invalidates all derived entries, not other bindings.
  void bindBoundary(AnalysisRequest request, Result result) {
    if (lookupCompleted(request)) {
      cache.clear();
      visits.clear();
    }
    bindings.insert_or_assign(request, std::move(result));
  }

  unsigned getVisitCount(AnalysisRequest request) const {
    auto found = visits.find(request);
    return found == visits.end() ? 0 : found->second;
  }

  LogicalResult remapAndForgetScope(Operation *oldScope,
                                    const IRMapping &mapping) {
    if (!oldScope)
      return failure();
    auto mapValue = [&](Value value) -> FailureOr<Value> {
      if (!value)
        return failure();
      Value replacement = mapping.lookupOrNull(value);
      if (!replacement) {
        if (isOwnedBy(value, oldScope))
          return failure();
        return value;
      }
      if (replacement.getType() != value.getType())
        return failure();
      return replacement;
    };
    auto remap = [&](const Cache &source, Cache &target) -> LogicalResult {
      for (const auto &[key, original] : source) {
        if (isOwnedBy(key.value, oldScope))
          continue;
        FailureOr<Value> newKey = mapValue(key.value);
        if (failed(newKey))
          return failure();
        Result copy = original;
        if (failed(rules.mapResultValues(copy, mapValue)))
          return failure();
        if (!target.emplace(AnalysisRequest{*newKey, key.kind}, std::move(copy))
                 .second)
          return failure();
      }
      return success();
    };
    Cache remappedCache, remappedBindings;
    if (failed(remap(cache, remappedCache)) ||
        failed(remap(bindings, remappedBindings)))
      return failure();
    cache.swap(remappedCache);
    bindings.swap(remappedBindings);
    visits.clear();
    return success();
  }

  void clear() {
    cache.clear();
    bindings.clear();
    visits.clear();
  }

private:
  OpBuilder &builder;
  Rules rules;
  Cache cache;
  Cache bindings;
  std::map<AnalysisRequest, unsigned, RequestLess> visits;
};

} // namespace mlir::triton::pointer::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_DRIVER_H
