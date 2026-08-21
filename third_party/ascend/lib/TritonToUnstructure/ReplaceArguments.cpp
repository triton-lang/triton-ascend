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
#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-replace-arguments"

using namespace mlir;
using namespace triton;

namespace {

static bool isScalarPointerType(Type type) {
  auto pointerType = dyn_cast<triton::PointerType>(type);
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

static bool isPointerDescriptorBoundarySlot(Operation *loop, unsigned slot) {
  if (!loop)
    return false;
  auto slots = dyn_cast_or_null<DenseI32ArrayAttr>(
      loop->getAttr(controlflow::kPointerDescriptorBoundaryAttr));
  return slots &&
         llvm::is_contained(slots.asArrayRef(), static_cast<int32_t>(slot));
}

// Only scalar pointers owned by the CFO descriptor schema cross a loop as a
// complete i64 address. Ordinary scalar pointers retain the established
// base-plus-relative-offset representation.
static bool useCompleteScalarAddress(Operation *loop, unsigned slot,
                                     Type type) {
  return loop && isa<scf::ForOp, scf::WhileOp>(loop) &&
         isScalarPointerType(type) &&
         isPointerDescriptorBoundarySlot(loop, slot);
}

// A scalar-pointer SCF slot is represented by one complete i64 address on all
// structural edges. Region-local pointer users are rebuilt immediately, so no
// pointer type crosses the control-flow boundary.
static Value rebuildScalarPointer(Value address, Type pointerType,
                                  OpBuilder &builder, Location loc) {
  return builder.create<triton::IntToPtrOp>(loc, pointerType, address);
}

static Value materializeScalarPointerAddress(Value pointer, OpBuilder &builder,
                                             Location loc) {
  return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(), pointer);
}

static bool hasScalarPointerBase(const PtrOffsetInfo &info) {
  return info.getPtr() && isScalarPointerType(info.getPtr().getType());
}

} // namespace

void replaceOperands(MutableArrayRef<OpOperand> oprs, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto it = oprs.begin(); it != oprs.end(); ++it) {
    auto &opr = *it;
    auto operand = opr.get();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      const PtrOffsetInfo &info = offsetMap.at(operand);
      // A complete opaque tensor base cannot be represented as an offset from
      // one scalar source. Keep this structural edge unchanged instead of
      // creating a type-mismatched loop signature.
      if (hasScalarPointerBase(info))
        opr.set(info.getOffset());
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(operand.getType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        for (auto offset : offsetMap.at(operand).getOffsets()) {
          it->set(offset);
          ++it;
        }
        --it;
      } else {
        // source == operand marks a complete scalar address whose source may be
        // selected or loop-carried at runtime. Keep it as a pointer instead of
        // replacing it with an offset relative to one statically chosen base.
        if (offsetMap.at(operand).getPtr() == operand)
          continue;
        opr.set(offsetMap.at(operand).getOffset());
      }
    }
  }
}

void replaceArgs(ValueRange args, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    auto arg = *it;
    if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      const PtrOffsetInfo &info = offsetMap.at(arg);
      // The relative-offset representation is legal only with one scalar
      // element-pointer base. Opaque tensor bases remain pointer values and
      // are handled lane-wise (or rejected later by the boundary validator).
      if (!hasScalarPointerBase(info))
        continue;

      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      Value src = info.getPtr();
      rewriter.replaceAllUsesWith(arg, tempVar);
      arg.setType(RankedTensorType::get(tensorType.getShape(),
                                        rewriter.getIntegerType(64)));
      src = rewriter.create<triton::SplatOp>(arg.getLoc(), tempVar.getType(),
                                             src);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), src, arg);
    } else if (auto ptrType = dyn_cast<triton::PointerType>(arg.getType())) {
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      if (!isa<RankedTensorType>(ptrType.getPointeeType()) &&
          offsetMap.at(arg).getPtr() == arg)
        continue;

      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      rewriter.replaceAllUsesWith(arg, tempVar);
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        auto srcOp =
            offsetMap.at(arg).getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        arg.setType(rewriter.getIntegerType(32));
        SmallVector<Value> newOffsets;
        for (auto offset : offsetMap.at(arg).getOffsets()) {
          newOffsets.push_back(*it);
          ++it;
        }
        --it;
        rewriter.replaceOpWithNewOp<triton::MakeTensorPtrOp>(
            tempVar.getDefiningOp(), tempVar.getType(), srcOp.getBase(),
            srcOp.getShape(), srcOp.getStrides(), newOffsets, srcOp.getOrder());
      } else {
        auto src = offsetMap.at(arg).getPtr();
        arg.setType(rewriter.getIntegerType(64));
        rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
            tempVar.getDefiningOp(), tempVar.getType(), src, arg);
      }
    }
  }
}

void convertTensorPtrPre(Operation *op, RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Preorder start\n" << *op << "\n";
  });
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    replaceArgs(whileOp.getBeforeArguments(), rewriter, offsetMap);
    replaceOperands(whileOp.getInitsMutable(), rewriter, offsetMap);
    replaceArgs(whileOp.getAfterArguments(), rewriter, offsetMap);
    replaceArgs(whileOp->getResults(), rewriter, offsetMap);
    replaceOperands(whileOp.getConditionOp().getArgsMutable(), rewriter,
                    offsetMap);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    replaceArgs(loopOp.getRegionIterArgs(), rewriter, offsetMap);
    replaceOperands(loopOp.getInitsMutable(), rewriter, offsetMap);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    replaceArgs(ifOp->getResults(), rewriter, offsetMap);
    replaceOperands(ifOp.thenYield().getResultsMutable(), rewriter, offsetMap);
    replaceOperands(ifOp.elseYield().getResultsMutable(), rewriter, offsetMap);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Preorder end\n" << *op << "\n";
  });
}

void convertTensorPtrPost(Operation *op, RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Postorder start\n" << *op << "\n";
  });
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    replaceOperands(whileOp.getYieldOp()->getOpOperands(), rewriter, offsetMap);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    replaceArgs(loopOp->getResults(), rewriter, offsetMap);
    replaceOperands(*loopOp.getYieldedValuesMutable(), rewriter, offsetMap);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[convertTensorPtr]: Postorder end\n" << *op << "\n";
  });
}

int getPtrTensorRank(Type type) {
  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      return tensorType.getRank();
    }
  }
  return 0;
}

SmallVector<Value> constructOperands(ValueRange operands, Value tempVar,
                                     IRMapping mapping, OpBuilder &builder,
                                     Operation *loop) {
  SmallVector<Value> newOperands;
  for (auto [slot, originalOperand] : llvm::enumerate(operands)) {
    Value mappedOperand = mapping.lookupOrDefault(originalOperand);
    if (useCompleteScalarAddress(loop, slot, originalOperand.getType()))
      mappedOperand = materializeScalarPointerAddress(mappedOperand, builder,
                                                      originalOperand.getLoc());
    newOperands.push_back(mappedOperand);
    auto numAppend = getPtrTensorRank(originalOperand.getType()) - 1;
    if (numAppend > 0)
      newOperands.append(numAppend, tempVar);
  }
  return newOperands;
}

SmallVector<Type> constructTypes(TypeRange types, Operation *loop) {
  SmallVector<Type> newTypes;
  for (auto [slot, type] : llvm::enumerate(types)) {
    newTypes.push_back(useCompleteScalarAddress(loop, slot, type)
                           ? IntegerType::get(type.getContext(), 64)
                           : type);
    if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        if (tensorType.getRank() > 0)
          newTypes.append(tensorType.getRank() - 1,
                          IntegerType::get(type.getContext(), 32));
      }
    }
  }
  return newTypes;
}

void replacePtrArguments(triton::FuncOp funcOp,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  Value tempVar = rewriter
                      .create<UnrealizedConversionCastOp>(
                          funcOp.getLoc(), rewriter.getI32Type(), ValueRange{})
                      ->getResult(0);
  std::function<WalkResult(Operation *)> convertTensorPtr = [&](Operation *op) {
    IRMapping mapping;
    Operation *newOp = nullptr;
    // Address carriers used by the replacement op must dominate it. Build the
    // replacement immediately before the old op and erase the old op last.
    rewriter.setInsertionPoint(op);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      SmallVector<Value> newInitArgs = constructOperands(
          forOp.getInitArgs(), tempVar, mapping, rewriter, forOp);
      newOp = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newInitArgs,
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            mapping.map(forOp.getInductionVar(), iv);
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(forOp.getRegionIterArgs())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(forOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              mapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : forOp.getBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     forOp));
          });
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      SmallVector<Value> newInits = constructOperands(
          whileOp.getInits(), tempVar, mapping, rewriter, whileOp);
      newOp = rewriter.create<scf::WhileOp>(
          whileOp.getLoc(), constructTypes(whileOp->getResultTypes(), whileOp),
          newInits,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(whileOp.getBeforeArguments())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(whileOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              mapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : whileOp.getBeforeBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto conditionOp = whileOp.getConditionOp();
            b.create<scf::ConditionOp>(
                conditionOp.getLoc(),
                mapping.lookup(conditionOp.getCondition()),
                constructOperands(conditionOp.getArgs(), tempVar, mapping, b,
                                  whileOp));
          },
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto newArgIter = args.begin();
            for (auto [slot, oldArg] :
                 llvm::enumerate(whileOp.getAfterArguments())) {
              Value mappedArg = *newArgIter;
              if (useCompleteScalarAddress(whileOp, slot, oldArg.getType()))
                mappedArg =
                    rebuildScalarPointer(mappedArg, oldArg.getType(), b, loc);
              mapping.map(oldArg, mappedArg);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : whileOp.getAfterBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = whileOp.getYieldOp();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     whileOp));
          });
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op);
               ifOp && ifOp->getNumResults() > 0) {
      newOp = rewriter.create<scf::IfOp>(
          ifOp.getLoc(), ifOp.getCondition(),
          [&](OpBuilder &b, Location loc) {
            for (auto &bodyOp : ifOp.thenBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = ifOp.thenYield();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     /*loop=*/nullptr));
          },
          [&](OpBuilder &b, Location loc) {
            for (auto &bodyOp : ifOp.elseBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = ifOp.elseYield();
            b.create<scf::YieldOp>(yieldOp.getLoc(),
                                   constructOperands(yieldOp.getOperands(),
                                                     tempVar, mapping, b,
                                                     /*loop=*/nullptr));
          });
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
      llvm_unreachable("Unsupported loop op");
    }
    if (newOp) {
      newOp->setAttrs(op->getAttrs());
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Converting\n" << *op << "\nto\n" << *newOp << "\n";
      });
      auto resIter = newOp->result_begin();
      for (auto [slot, res] : llvm::enumerate(op->getResults())) {
        Value replacement = *resIter;
        if (useCompleteScalarAddress(op, slot, res.getType())) {
          rewriter.setInsertionPointAfter(newOp);
          replacement = rebuildScalarPointer(replacement, res.getType(),
                                             rewriter, res.getLoc());
        }
        rewriter.replaceAllUsesWith(res, replacement);
        std::advance(resIter, std::max(getPtrTensorRank(res.getType()), 1));
      }
      rewriter.eraseOp(op);
      op = newOp;
      convertTensorPtrPre(op, rewriter, offsetMap);
      for (auto &region : op->getRegions())
        region.walk<WalkOrder::PreOrder>(convertTensorPtr);
      convertTensorPtrPost(op, rewriter, offsetMap);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  };

  funcOp->walk<WalkOrder::PreOrder>(convertTensorPtr);
}
