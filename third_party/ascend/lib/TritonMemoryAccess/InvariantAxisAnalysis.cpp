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

#include "TritonMemoryAccess/InvariantAxisAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::ptr {

namespace {

bool sameInvariantAxes(const PtrInfo &lhs, const PtrInfo &rhs) {
  if (lhs.offsets.axes.size() != rhs.offsets.axes.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.offsets.axes, rhs.offsets.axes)) {
    if ((left.kind == AxisKind::Invariant) !=
        (right.kind == AxisKind::Invariant))
      return false;
  }
  return true;
}

} // namespace

InvariantAxisAnalysis::AxisMask
InvariantAxisAnalysis::getAxisMask(const PtrInfo &info, unsigned rank) {
  AxisMask mask(rank, false);
  if (info.offsets.axes.size() != rank)
    return mask;
  for (unsigned i = 0; i < rank; ++i)
    mask[i] = info.offsets.axes[i].kind == AxisKind::Invariant;
  return mask;
}

FailureOr<PtrInfo> InvariantAxisAnalysis::joinAxes(Value result,
                                                   const PtrInfo &lhs,
                                                   const PtrInfo &rhs) {
  if (!result || lhs.isEmpty() || rhs.isEmpty() ||
      lhs.offsets.axes.size() != rhs.offsets.axes.size())
    return failure();
  PtrInfo joined = lhs;
  Type scalar = builder.getIndexType();
  joined.offsets.scalarOffset = {};
  for (unsigned i = 0; i < joined.offsets.axes.size(); ++i) {
    bool invariant = lhs.offsets.axes[i].kind == AxisKind::Invariant &&
                     rhs.offsets.axes[i].kind == AxisKind::Invariant;
    joined.offsets.axes[i] =
        AxisInfo(invariant ? AxisKind::Invariant : AxisKind::Unknown,
                 builder.getIntegerAttr(scalar, 0));
  }
  return joined;
}

FailureOr<PtrInfo>
InvariantAxisAnalysis::analyzeIfResult(scf::IfOp ifOp, unsigned resultIndex) {
  if (resultIndex >= ifOp.getNumResults())
    return failure();
  auto thenInfo = analyzePtr(ifOp.thenYield().getOperand(resultIndex));
  auto elseInfo = analyzePtr(ifOp.elseYield().getOperand(resultIndex));
  if (failed(thenInfo) || failed(elseInfo))
    return failure();
  return joinAxes(ifOp.getResult(resultIndex), *thenInfo, *elseInfo);
}

FailureOr<PtrInfo> InvariantAxisAnalysis::analyzeForIterArg(scf::ForOp forOp,
                                                            unsigned index) {
  if (index >= forOp.getRegionIterArgs().size())
    return failure();
  auto initInfo = analyzePtr(forOp.getInitArgs()[index]);
  if (failed(initInfo) || initInfo->isEmpty())
    return failure();

  BlockArgument iterArg = forOp.getRegionIterArgs()[index];
  PtrInfo candidate = *initInfo;
  auto saved = bindings.find(iterArg);
  bool hadSaved = saved != bindings.end();
  PtrInfo savedInfo;
  if (hadSaved)
    savedInfo = saved->second;
  auto restore = llvm::make_scope_exit([&] {
    if (hadSaved)
      bindings[iterArg] = std::move(savedInfo);
    else
      bindings.erase(iterArg);
  });

  // Start from the initial iter_arg facts and intersect every backedge until
  // no invariant axis can be removed. The lattice is finite (one bit per axis).
  while (true) {
    bindings.insert_or_assign(iterArg, candidate);
    auto yieldedInfo = analyzePtr(forOp.getYieldedValues()[index]);
    if (failed(yieldedInfo))
      return failure();
    auto joined = joinAxes(forOp.getResult(index), candidate, *yieldedInfo);
    if (failed(joined))
      return failure();
    if (sameInvariantAxes(candidate, *joined))
      break;
    candidate = std::move(*joined);
    // Body expressions were analyzed under the previous assumption. Rebuild
    // them after weakening the iter_arg facts.
    cache.clear();
  }

  // Keep the fixed point available to other uses in the loop body.
  bindings.insert_or_assign(iterArg, candidate);
  restore.release();
  return candidate;
}

FailureOr<PtrInfo>
InvariantAxisAnalysis::analyzeForResult(scf::ForOp forOp,
                                        unsigned resultIndex) {
  return analyzeForIterArg(forOp, resultIndex);
}

FailureOr<PtrInfo> InvariantAxisAnalysis::analyzePtr(Value value) {
  if (!value)
    return failure();
  OpBuilder::InsertionGuard insertionGuard(builder);
  if (auto found = bindings.find(value); found != bindings.end())
    return found->second;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (forOp) {
      auto iterArgs = forOp.getRegionIterArgs();
      auto iter = llvm::find(iterArgs, blockArg);
      if (iter != iterArgs.end())
        return analyzeForIterArg(forOp, std::distance(iterArgs.begin(), iter));
    }
  }

  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto info =
        analyzeForResult(forOp, cast<OpResult>(value).getResultNumber());
    if (succeeded(info))
      cache.try_emplace(value, *info);
    return info;
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto info = analyzeIfResult(ifOp, cast<OpResult>(value).getResultNumber());
    if (succeeded(info))
      cache.try_emplace(value, *info);
    return info;
  }

  if (!active.insert(value).second)
    return failure();
  auto activeGuard = llvm::make_scope_exit([&] { active.erase(value); });

  // Pointer element-size changes do not affect zero/nonzero axis variation.
  // Reuse the source facts even for bitcasts whose storage widths are outside
  // CommonPtrRules' exact-offset conversion set (notably ptr<i1> to ptr<i8>).
  if (auto bitcast = value.getDefiningOp<triton::BitcastOp>()) {
    auto sourceType = dyn_cast<RankedTensorType>(bitcast.getSrc().getType());
    auto targetType = dyn_cast<RankedTensorType>(value.getType());
    if (sourceType && targetType &&
        sourceType.getShape() == targetType.getShape() &&
        isa<triton::PointerType>(sourceType.getElementType()) &&
        isa<triton::PointerType>(targetType.getElementType())) {
      auto sourceInfo = analyzePtr(bitcast.getSrc());
      if (succeeded(sourceInfo)) {
        cache.try_emplace(value, *sourceInfo);
        return cache.find(value)->second;
      }
    }
  }

  auto analyze = [&](Value input) { return analyzePtr(input); };
  auto info = rules.parseOperation(value, analyze, builder);
  if (failed(info) || info->isEmpty())
    return failure();
  PtrInfo result = *info;
  cache.try_emplace(value, std::move(result));
  return cache.find(value)->second;
}

FailureOr<InvariantAxisAnalysis::AxisMask>
InvariantAxisAnalysis::analyze(Value value) {
  if (!value)
    return failure();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return AxisMask{};
  auto info = analyzePtr(value);
  if (failed(info))
    return failure();
  return getAxisMask(*info, type.getRank());
}

} // namespace mlir::triton::ptr
