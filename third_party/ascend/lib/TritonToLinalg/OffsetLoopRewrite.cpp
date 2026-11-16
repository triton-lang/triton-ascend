/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "TritonToLinalg/LoadStoreConverter.h"
#include "Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
static bool
isUsedWithCondition(Value v, std::function<bool(OpOperand *)> cond,
                    int depth = 0,
                    llvm::SmallSetVector<Value, 8> *visited = nullptr) {
  // Scalar uses do not require decomposition of their tensor producers.
  // Do not require upstream tensor carriers to be decomposed through them.
  if (isa<IntegerType>(v.getType()))
    return false;

  llvm::SmallSetVector<Value, 8> localVisited;
  if (!visited) {
    visited = &localVisited;
  }

  if (visited->contains(v)) {
    return false;
  }
  visited->insert(v);

  for (auto &use : v.getUses()) {
    auto *user = use.getOwner();
    if (user->hasAttr(ConverterUtils::discreteAttrName) ||
        isa<tensor::ExtractOp>(user))
      continue;
    if (cond(&use))
      return true;
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user);
        loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
      Value tiedArg = loopOp.getTiedLoopRegionIterArg(&use);
      if (tiedArg && isUsedWithCondition(tiedArg, cond, depth + 1, visited))
        return true;
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user);
               yieldOp && !isa<scf::WhileOp>(user->getParentOp())) {
      if (depth && isUsedWithCondition(yieldOp->getParentOp()->getResult(
                                           use.getOperandNumber()),
                                       cond, depth - 1, visited))
        return true;
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user);
               conditionOp && use.getOperandNumber() > 0) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      if (depth &&
          isUsedWithCondition(whileOp->getResult(use.getOperandNumber() - 1),
                              cond, depth - 1, visited))
        return true;
      if (isUsedWithCondition(
              whileOp.getAfterArguments()[use.getOperandNumber() - 1], cond,
              depth, visited))
        return true;
    }
    for (auto res : user->getResults()) {
      if (isUsedWithCondition(res, cond, depth, visited))
        return true;
    }
  }
  return false;
}

// A loop-carried value may be consumed through a region argument, through a
// while after-argument, or only after the loop result. Check every semantic
// view of the same carried slot so an identity tensor.cast after the loop
// cannot hide an AddPtr/load/store use from the decomposition decision.
static bool isLoopCarriedValueUsedWithCondition(
    LoopLikeOpInterface loopOp, unsigned index,
    const std::function<bool(OpOperand *)> &condition) {
  if (index >= loopOp.getRegionIterArgs().size() ||
      index >= loopOp->getNumResults())
    return false;
  if (isUsedWithCondition(loopOp.getRegionIterArgs()[index], condition))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (index < whileOp.getAfterArguments().size() &&
        isUsedWithCondition(whileOp.getAfterArguments()[index], condition))
      return true;
  }
  return isUsedWithCondition(loopOp->getResult(index), condition);
}

// The existing loop rewrite used a separate offset for every dimension. The
// public result has one origin and one stride per axis, in that order.
struct OffsetSlot {
  unsigned index;
  mlir::triton::ptr::OffsetInfo argument;
  SmallVector<Value> initial;
  SmallVector<Value> yielded;
  SmallVector<Value> condition;
  mlir::triton::ptr::OffsetInfo afterArgument;
};

SmallVector<Value> materializeFields(const mlir::triton::ptr::OffsetInfo &info,
                                     OpBuilder &builder, Location loc) {
  SmallVector<Value> fields;
  fields.push_back(
      getValueOrCreateConstantIndexOp(builder, loc, info.scalarOffset));
  for (const auto &axis : info.axes)
    fields.push_back(
        getValueOrCreateConstantIndexOp(builder, loc, axis.stride));
  return fields;
}

Value rebuildOffset(RankedTensorType type,
                    const mlir::triton::ptr::OffsetInfo &info,
                    ValueRange fields, OpBuilder &builder, Location loc) {
  auto elementType = type.getElementType();
  Value origin = *castIntegerLike(builder, loc, fields.front(), elementType);
  Value result = builder.create<triton::SplatOp>(loc, type, origin);
  for (auto [dim, axis] : llvm::enumerate(info.axes)) {
    auto constant =
        mlir::triton::ptr::detail::getIntegerConstant(axis.stride, 64);
    if (succeeded(constant) && constant->isZero())
      continue;
    int64_t size = type.getDimSize(dim);
    Value range = builder.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({size}, builder.getI32Type()), 0, size);
    auto axisType = RankedTensorType::get({size}, elementType);
    range = *castIntegerLike(builder, loc, range, axisType);
    if (failed(constant) || !constant->isOne()) {
      Value stride = succeeded(constant)
                         ? builder.create<arith::ConstantIndexOp>(
                               loc, constant->getSExtValue())
                         : fields[dim + 1];
      stride = *castIntegerLike(builder, loc, stride, elementType);
      stride = builder.create<triton::SplatOp>(loc, axisType, stride);
      range = builder.create<arith::MulIOp>(loc, range, stride);
    }
    for (unsigned d = 0; d < type.getRank(); ++d)
      if (d != dim)
        range = builder.create<triton::ExpandDimsOp>(loc, range, d);
    range = builder.create<triton::BroadcastOp>(loc, type, range);
    result = builder.create<arith::AddIOp>(loc, result, range);
  }
  return result;
}

LogicalResult rewriteOffsetLoop(LoopLikeOpInterface loop,
                                IRRewriter &rewriter) {
  if (!isa<scf::ForOp, scf::WhileOp>(loop.getOperation()) ||
      loop->hasAttr("ExtractedLoadOrStore"))
    return success();
  auto whileOp = dyn_cast<scf::WhileOp>(loop.getOperation());
  // Before/after signatures may differ. Only slots present on every edge can
  // use the same scalar component layout.
  if (whileOp &&
      whileOp.getBeforeArguments().size() != whileOp.getAfterArguments().size())
    return success();
  SmallVector<OffsetSlot, 0> slots;
  {
    mlir::triton::ptr::PtrAnalysis analysis(rewriter);
    auto owned = loop->getAttrOfType<DenseI32ArrayAttr>(
        controlflow::kPointerDescriptorBoundaryAttr);
    for (auto [index, init] : llvm::enumerate(loop.getInits())) {
      auto type = dyn_cast<RankedTensorType>(init.getType());
      if (!type || !type.hasStaticShape() ||
          !isa<IntegerType>(type.getElementType()) ||
          type.getElementType().isInteger(1) ||
          (owned &&
           llvm::is_contained(owned.asArrayRef(), static_cast<int32_t>(index))))
        continue;
      if (!isLoopCarriedValueUsedWithCondition(loop, index, [](OpOperand *use) {
            auto *user = use->getOwner();
            return isa<triton::AddPtrOp>(user) ||
                   (isa<triton::LoadOp>(user) &&
                    use->getOperandNumber() == 1) ||
                   (isa<triton::StoreOp>(user) && use->getOperandNumber() == 2);
          }))
        continue;
      Value arg = loop.getRegionIterArgs()[index];
      Value yield = loop.getYieldedValues()[index];
      auto initInfo = analysis.analyze(init);
      auto argInfo = analysis.analyze(arg);
      auto yieldInfo = analysis.analyze(yield);
      if (failed(initInfo) || failed(argInfo) || failed(yieldInfo) ||
          !initInfo->offsets.hasStructuredForm() ||
          !argInfo->offsets.hasStructuredForm() ||
          !yieldInfo->offsets.hasStructuredForm())
        continue;
      OffsetSlot slot;
      slot.index = index;
      slot.argument = argInfo->offsets;
      if (whileOp) {
        auto after = analysis.analyze(whileOp.getAfterArguments()[index]);
        auto condition =
            analysis.analyze(whileOp.getConditionOp().getArgs()[index]);
        if (failed(after) || failed(condition) ||
            !after->offsets.hasStructuredForm() ||
            !condition->offsets.hasStructuredForm() ||
            after->offsets.valueType != type)
          continue;
        slot.afterArgument = after->offsets;
        rewriter.setInsertionPoint(whileOp.getConditionOp());
        slot.condition =
            materializeFields(condition->offsets, rewriter, loop.getLoc());
      }
      rewriter.setInsertionPoint(loop);
      slot.initial =
          materializeFields(initInfo->offsets, rewriter, loop.getLoc());
      rewriter.setInsertionPoint(whileOp ? whileOp.getYieldOp().getOperation()
                                         : cast<scf::ForOp>(loop.getOperation())
                                               .getBody()
                                               ->getTerminator());
      slot.yielded =
          materializeFields(yieldInfo->offsets, rewriter, loop.getLoc());
      slots.push_back(std::move(slot));
    }
  }
  if (slots.empty())
    return success();

  // Retain the old mapping/clone algorithm, but all analysis above has finished
  // before either SCF region is rebuilt.
  SmallVector<int64_t> indices(loop.getInits().size(), -1);
  SmallVector<Value> inits;
  for (auto [i, init] : llvm::enumerate(loop.getInits())) {
    unsigned originalIndex = i;
    if (llvm::any_of(slots, [&](const auto &slot) {
          return slot.index == originalIndex;
        }))
      continue;
    indices[i] = inits.size();
    inits.push_back(init);
  }
  unsigned fieldStart = inits.size();
  for (const auto &slot : slots)
    llvm::append_range(inits, slot.initial);

  auto commonBodyBuilder = [&](OpBuilder &b, Location loc, Block *body,
                               ValueRange args, bool before, Value iv = {}) {
    IRMapping mapping;
    auto oldArgs = body->getArguments();
    if (iv) {
      mapping.map(oldArgs.front(), iv);
      oldArgs = oldArgs.drop_front();
    }
    for (auto [oldArg, index] : llvm::zip_equal(oldArgs, indices))
      if (index >= 0)
        mapping.map(oldArg, args[index]);
    unsigned cursor = fieldStart;
    for (const auto &slot : slots) {
      auto fields = args.slice(cursor, slot.initial.size());
      const auto &info =
          whileOp && !before ? slot.afterArgument : slot.argument;
      Value rebuilt =
          rebuildOffset(cast<RankedTensorType>(oldArgs[slot.index].getType()),
                        info, fields, b, loc);
      mapping.map(oldArgs[slot.index], rebuilt);
      if (auto origin = dyn_cast<Value>(info.scalarOffset))
        mapping.map(origin, fields.front());
      for (auto [dim, axis] : llvm::enumerate(info.axes))
        if (auto stride = dyn_cast<Value>(axis.stride))
          mapping.map(stride, fields[dim + 1]);
      cursor += fields.size();
    }
    for (Operation &bodyOp : body->without_terminator()) {
      // Analysis samples of the original tensor are now scalar arguments.
      // Cloning their definitions would overwrite those mappings.
      if (bodyOp.getNumResults() &&
          llvm::all_of(bodyOp.getResults(),
                       [&](Value value) { return mapping.contains(value); }))
        continue;
      b.clone(bodyOp, mapping);
    }
    Operation *terminator = body->getTerminator();
    ValueRange operands = terminator->getOperands();
    auto condition = dyn_cast<scf::ConditionOp>(terminator);
    if (condition)
      operands = condition.getArgs();
    SmallVector<Value> yielded;
    for (auto [operand, index] : llvm::zip_equal(operands, indices))
      if (index >= 0)
        yielded.push_back(mapping.lookupOrDefault(operand));
    for (const auto &slot : slots)
      for (Value field : condition ? slot.condition : slot.yielded)
        yielded.push_back(mapping.lookupOrDefault(field));
    if (condition)
      b.create<scf::ConditionOp>(
          loc, mapping.lookupOrDefault(condition.getCondition()), yielded);
    else
      b.create<scf::YieldOp>(loc, yielded);
  };
  rewriter.setInsertionPoint(loop);
  Operation *replacement;
  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
    replacement = rewriter.create<scf::ForOp>(
        loop.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), inits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          commonBodyBuilder(b, loc, forOp.getBody(), args, true, iv);
        });
  } else {
    SmallVector<Type> types;
    for (auto [type, index] :
         llvm::zip_equal(whileOp.getResultTypes(), indices))
      if (index >= 0)
        types.push_back(type);
    types.append(inits.size() - fieldStart, rewriter.getIndexType());
    replacement = rewriter.create<scf::WhileOp>(
        loop.getLoc(), types, inits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          commonBodyBuilder(b, loc, whileOp.getBeforeBody(), args, true);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) {
          commonBodyBuilder(b, loc, whileOp.getAfterBody(), args, false);
        });
  }
  replacement->setAttrs(loop->getAttrs());
  if (auto marker = loop->getAttrOfType<DenseI32ArrayAttr>(
          controlflow::kPointerDescriptorBoundaryAttr)) {
    SmallVector<int32_t> mapped;
    for (int32_t slot : marker.asArrayRef())
      mapped.push_back(indices[slot]);
    replacement->setAttr(controlflow::kPointerDescriptorBoundaryAttr,
                         rewriter.getDenseI32ArrayAttr(mapped));
  }
  rewriter.setInsertionPointAfter(replacement);
  SmallVector<Value> results(loop->getNumResults());
  for (auto [i, index] : llvm::enumerate(indices))
    if (index >= 0)
      results[i] = replacement->getResult(index);
  unsigned cursor = fieldStart;
  for (const auto &slot : slots) {
    const auto &info = whileOp ? slot.afterArgument : slot.argument;
    results[slot.index] = rebuildOffset(
        cast<RankedTensorType>(loop->getResult(slot.index).getType()), info,
        replacement->getResults().slice(cursor, slot.initial.size()), rewriter,
        loop.getLoc());
    cursor += slot.initial.size();
  }
  rewriter.replaceOp(loop, results);
  return success();
}
} // namespace

LogicalResult LoadStoreConverter::rewriteOffsetLoops(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  WalkResult result = module.walk([&](LoopLikeOpInterface loop) {
    return failed(rewriteOffsetLoop(loop, rewriter)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}
