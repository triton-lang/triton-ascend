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
#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
bool isScalarPointer(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && !isa<ShapedType>(pointer.getPointeeType());
}

// Use the same complete address on every SCF edge. This works for changing
// bases and selected pointers as well as base-plus-offset recurrences.
SmallVector<Value> addressOperands(ValueRange values, IRMapping &mapping,
                                   OpBuilder &builder, Location loc) {
  SmallVector<Value> result;
  for (Value value : values) {
    Value mapped = mapping.lookupOrDefault(value);
    if (isScalarPointer(value.getType()))
      mapped =
          builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(), mapped);
    result.push_back(mapped);
  }
  return result;
}

void cloneBody(Block *source, ValueRange arguments, OpBuilder &builder,
               Location loc, IRMapping &mapping) {
  for (auto [oldArg, argument] :
       llvm::zip_equal(source->getArguments(), arguments)) {
    Value value = argument;
    if (isScalarPointer(oldArg.getType()))
      value = builder.create<triton::IntToPtrOp>(loc, oldArg.getType(), value);
    mapping.map(oldArg, value);
  }
  for (Operation &op : source->without_terminator())
    builder.clone(op, mapping);
  Operation *terminator = source->getTerminator();
  if (auto condition = dyn_cast<scf::ConditionOp>(terminator))
    builder.create<scf::ConditionOp>(
        loc, mapping.lookupOrDefault(condition.getCondition()),
        addressOperands(condition.getArgs(), mapping, builder, loc));
  else
    builder.create<scf::YieldOp>(
        loc, addressOperands(terminator->getOperands(), mapping, builder, loc));
}

struct PointerSlot {
  unsigned index;
  Value initialPointer;
  mlir::triton::ptr::PtrInfo layout;
  SmallVector<Value> initial, before, after, yielded, condition;
};

// The legacy boundary rewrite transported offsets or block coordinates.
// Prove that the remaining descriptor fields are invariant before retaining
// that compact representation. Block coordinates can exceed i32 after advance;
// only a completely invariant block descriptor can reuse its original value.
// Changing block descriptors are decomposed by the preceding CFO pass.
bool preservesLayout(const mlir::triton::ptr::ChangeInfo &changes, bool block) {
  if (changes.base)
    return false;
  return !block || (!changes.order &&
                    llvm::none_of(changes.coordinates,
                                  [](bool changed) { return changed; }) &&
                    llvm::none_of(changes.strides,
                                  [](bool changed) { return changed; }) &&
                    llvm::none_of(changes.parentShape,
                                  [](bool changed) { return changed; }));
}

SmallVector<Value> pointerFields(const mlir::triton::ptr::PtrInfo &info,
                                 OpBuilder &builder, Location loc) {
  assert(!info.block && "invariant blocks transport no numerical fields");
  auto tensor = cast<RankedTensorType>(info.offsets.valueType);
  return {*mlir::triton::ptr::detail::materializeFullOffset(
      info.offsets, tensor.clone(builder.getI64Type()), builder, loc)};
}

Value rebuildPointer(Type type, const mlir::triton::ptr::PtrInfo &layout,
                     ValueRange fields, OpBuilder &builder, Location loc) {
  assert(!layout.block && "invariant blocks reuse their original SSA value");
  Value base = builder.create<triton::SplatOp>(loc, type, layout.base);
  return builder.create<triton::AddPtrOp>(loc, type, base, fields.front());
}

// These temporary casts label the numerical fields of an unchanged source
// block argument. The clone maps their results directly to new arguments and
// omits the casts. No half-built SCF operation is analyzed.
LogicalResult seedArgument(Value argument, PointerSlot &slot,
                           SmallVectorImpl<Value> &fields,
                           mlir::triton::ptr::PtrAnalysis &analysis,
                           IRRewriter &rewriter) {
  if (slot.layout.block)
    return analysis.setInfo(argument, slot.layout);
  rewriter.setInsertionPointToStart(argument.getParentBlock());
  SmallVector<Type> types{
      cast<RankedTensorType>(argument.getType()).clone(rewriter.getI64Type())};
  auto placeholders = rewriter.create<UnrealizedConversionCastOp>(
      argument.getLoc(), types, argument);
  llvm::append_range(fields, placeholders.getResults());
  mlir::triton::ptr::PtrInfo info = slot.layout;
  info.offsets = *mlir::triton::ptr::buildOffsetInfo(fields.front(), rewriter);
  info.offsets.valueType =
      mlir::triton::ptr::detail::addressValueType(argument.getType(), rewriter);
  return analysis.setInfo(argument, info);
}

LogicalResult rewritePointerLoop(LoopLikeOpInterface loop, IRRewriter &rewriter,
                                 bool &changed) {
  auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
  auto whileOp = dyn_cast<scf::WhileOp>(loop.getOperation());
  if (!forOp && !whileOp)
    return success();
  if (whileOp &&
      !llvm::equal(whileOp.getOperandTypes(), whileOp.getResultTypes()))
    return success();
  SmallVector<PointerSlot, 0> slots;
  {
    mlir::triton::ptr::PtrAnalysis proof(rewriter);
    for (auto [index, initial] : llvm::enumerate(loop.getInits())) {
      bool tensorPointer =
          mlir::triton::ptr::detail::isOrdinaryPtrType(initial.getType()) &&
          isa<RankedTensorType>(initial.getType());
      bool blockPointer =
          mlir::triton::ptr::detail::isBlockPtrType(initial.getType());
      if (!tensorPointer && !blockPointer)
        continue;
      auto layout = proof.analyze(initial);
      Value reference = whileOp ? whileOp.getAfterArguments()[index]
                                : loop.getRegionIterArgs()[index];
      auto backedge = proof.analyzeChanges(loop.getYieldedValues()[index],
                                           reference, initial);
      if (failed(layout) || !layout->hasScalarBase() || failed(backedge) ||
          !preservesLayout(*backedge, blockPointer))
        continue;
      if (whileOp) {
        auto condition =
            proof.analyzeChanges(whileOp.getConditionOp().getArgs()[index],
                                 whileOp.getBeforeArguments()[index], initial);
        if (failed(condition) || !preservesLayout(*condition, blockPointer))
          continue;
      }
      PointerSlot slot;
      slot.index = index;
      slot.initialPointer = initial;
      slot.layout = *layout;
      rewriter.setInsertionPoint(loop);
      if (!blockPointer)
        slot.initial = pointerFields(*layout, rewriter, loop.getLoc());
      slots.push_back(std::move(slot));
    }
  }
  if (slots.empty())
    return success();
  {
    mlir::triton::ptr::PtrAnalysis analysis(rewriter);
    for (auto &slot : slots) {
      if (failed(seedArgument(loop.getRegionIterArgs()[slot.index], slot,
                              slot.before, analysis, rewriter)))
        return failure();
      if (whileOp &&
          failed(seedArgument(whileOp.getAfterArguments()[slot.index], slot,
                              slot.after, analysis, rewriter)))
        return failure();
    }
    for (auto &slot : slots) {
      if (slot.layout.block)
        continue;
      auto yielded = analysis.analyze(loop.getYieldedValues()[slot.index]);
      if (failed(yielded) || !yielded->hasScalarBase() || yielded->block ||
          yielded->base != slot.layout.base)
        return loop.emitError(
            "pointer backedge requires CFO descriptor decomposition");
      rewriter.setInsertionPoint(whileOp ? whileOp.getYieldOp().getOperation()
                                         : forOp.getBody()->getTerminator());
      slot.yielded = pointerFields(*yielded, rewriter, loop.getLoc());
      if (whileOp) {
        auto condition =
            analysis.analyze(whileOp.getConditionOp().getArgs()[slot.index]);
        if (failed(condition) || !condition->hasScalarBase() ||
            condition->block || condition->base != slot.layout.base)
          return loop.emitError(
              "pointer condition requires CFO descriptor decomposition");
        rewriter.setInsertionPoint(whileOp.getConditionOp());
        slot.condition = pointerFields(*condition, rewriter, loop.getLoc());
      }
    }
  }
  SmallVector<int64_t> indices(loop.getInits().size(), -1);
  SmallVector<Value> inits;
  for (auto [index, initial] : llvm::enumerate(loop.getInits())) {
    unsigned originalIndex = index;
    if (llvm::any_of(slots, [&](const auto &slot) {
          return slot.index == originalIndex;
        }))
      continue;
    indices[index] = inits.size();
    inits.push_back(initial);
  }
  unsigned fieldStart = inits.size();
  for (const auto &slot : slots)
    llvm::append_range(inits, slot.initial);
  auto bodyBuilder = [&](Block *body, ValueRange arguments, OpBuilder &b,
                         Location loc, bool before, Value iv = {}) {
    IRMapping mapping;
    auto oldArguments = body->getArguments();
    if (iv) {
      mapping.map(oldArguments.front(), iv);
      oldArguments = oldArguments.drop_front();
    }
    for (auto [argument, index] : llvm::zip_equal(oldArguments, indices))
      if (index >= 0)
        mapping.map(argument, arguments[index]);
    unsigned cursor = fieldStart;
    for (const auto &slot : slots) {
      auto fields = arguments.slice(cursor, slot.initial.size());
      Value rebuilt = slot.layout.block
                          ? slot.initialPointer
                          : rebuildPointer(oldArguments[slot.index].getType(),
                                           slot.layout, fields, b, loc);
      mapping.map(oldArguments[slot.index], rebuilt);
      mapping.map(before ? ValueRange(slot.before) : ValueRange(slot.after),
                  fields);
      cursor += fields.size();
    }
    for (Operation &op : body->without_terminator()) {
      if (op.getNumResults() && llvm::all_of(op.getResults(), [&](Value v) {
            return mapping.contains(v);
          }))
        continue;
      b.clone(op, mapping);
    }
    auto condition = dyn_cast<scf::ConditionOp>(body->getTerminator());
    ValueRange operands =
        condition ? condition.getArgs() : body->getTerminator()->getOperands();
    SmallVector<Value> values;
    for (auto [operand, index] : llvm::zip_equal(operands, indices))
      if (index >= 0)
        values.push_back(mapping.lookupOrDefault(operand));
    for (const auto &slot : slots)
      for (Value value : condition ? slot.condition : slot.yielded)
        values.push_back(mapping.lookupOrDefault(value));
    if (condition)
      b.create<scf::ConditionOp>(
          loc, mapping.lookupOrDefault(condition.getCondition()), values);
    else
      b.create<scf::YieldOp>(loc, values);
  };
  rewriter.setInsertionPoint(loop);
  Operation *replacement;
  if (forOp) {
    replacement = rewriter.create<scf::ForOp>(
        loop.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), inits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          bodyBuilder(forOp.getBody(), args, b, loc, true, iv);
        });
  } else {
    SmallVector<Type> types;
    for (Value value : inits)
      types.push_back(value.getType());
    replacement = rewriter.create<scf::WhileOp>(
        loop.getLoc(), types, inits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          bodyBuilder(whileOp.getBeforeBody(), args, b, loc, true);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) {
          bodyBuilder(whileOp.getAfterBody(), args, b, loc, false);
        });
  }
  replacement->setAttrs(loop->getAttrs());
  if (auto marker = loop->getAttrOfType<DenseI32ArrayAttr>(
          controlflow::kPointerDescriptorBoundaryAttr)) {
    SmallVector<int32_t> mapped;
    for (int32_t index : marker.asArrayRef())
      if (indices[index] >= 0)
        mapped.push_back(indices[index]);
    replacement->setAttr(controlflow::kPointerDescriptorBoundaryAttr,
                         rewriter.getDenseI32ArrayAttr(mapped));
  }
  rewriter.setInsertionPointAfter(replacement);
  SmallVector<Value> results(loop->getNumResults());
  for (auto [index, mapped] : llvm::enumerate(indices))
    if (mapped >= 0)
      results[index] = replacement->getResult(mapped);
  unsigned cursor = fieldStart;
  for (const auto &slot : slots) {
    results[slot.index] =
        slot.layout.block
            ? slot.initialPointer
            : rebuildPointer(
                  loop->getResult(slot.index).getType(), slot.layout,
                  replacement->getResults().slice(cursor, slot.initial.size()),
                  rewriter, loop.getLoc());
    cursor += slot.initial.size();
  }
  rewriter.replaceOp(loop, results);
  changed = true;
  return success();
}
} // namespace

LogicalResult replacePtrArguments(triton::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  // Postorder preserves nested rewrites when the outer body is cloned. No
  // analysis cache or retired operations survive a replacement.
  funcOp.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))
      return;
    bool needed = llvm::any_of(op->getResultTypes(), isScalarPointer);
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op))
      needed |= llvm::any_of(loop.getInits(), [](Value v) {
        return isScalarPointer(v.getType());
      });
    if (!needed)
      return;
    rewriter.setInsertionPoint(op);
    IRMapping mapping;
    SmallVector<Type> types;
    for (Type type : op->getResultTypes())
      types.push_back(isScalarPointer(type) ? rewriter.getI64Type() : type);
    Operation *replacement = nullptr;
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      replacement = rewriter.create<scf::ForOp>(
          op->getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(),
          addressOperands(loop.getInitArgs(), mapping, rewriter, op->getLoc()),
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            SmallVector<Value> allArgs{iv};
            llvm::append_range(allArgs, args);
            cloneBody(loop.getBody(), allArgs, b, loc, mapping);
          });
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      replacement = rewriter.create<scf::WhileOp>(
          op->getLoc(), types,
          addressOperands(loop.getInits(), mapping, rewriter, op->getLoc()),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            IRMapping before;
            cloneBody(loop.getBeforeBody(), args, b, loc, before);
          },
          [&](OpBuilder &b, Location loc, ValueRange args) {
            IRMapping after;
            cloneBody(loop.getAfterBody(), args, b, loc, after);
          });
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      replacement = rewriter.create<scf::IfOp>(
          op->getLoc(), branch.getCondition(),
          [&](OpBuilder &b, Location loc) {
            IRMapping thenMapping;
            cloneBody(branch.thenBlock(), {}, b, loc, thenMapping);
          },
          [&](OpBuilder &b, Location loc) {
            IRMapping elseMapping;
            cloneBody(branch.elseBlock(), {}, b, loc, elseMapping);
          });
    }
    replacement->setAttrs(op->getAttrs());
    rewriter.setInsertionPointAfter(replacement);
    SmallVector<Value> results;
    for (auto [old, value] :
         llvm::zip_equal(op->getResults(), replacement->getResults())) {
      Value result = value;
      if (isScalarPointer(old.getType()))
        result = rewriter.create<triton::IntToPtrOp>(op->getLoc(),
                                                     old.getType(), result);
      results.push_back(result);
    }
    rewriter.replaceOp(op, results);
  });
  bool changed;
  do {
    changed = false;
    auto result = funcOp.walk([&](LoopLikeOpInterface loop) {
      return failed(rewritePointerLoop(loop, rewriter, changed))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  } while (changed);
  return success();
}
