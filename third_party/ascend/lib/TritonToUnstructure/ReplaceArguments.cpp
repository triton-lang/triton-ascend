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

#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-replace-arguments"

using namespace mlir;
using namespace triton;

void replaceOperands(MutableArrayRef<OpOperand> oprs, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto it = oprs.begin(); it != oprs.end(); ++it) {
    auto &opr = *it;
    auto operand = opr.get();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      opr.set(offsetMap.at(operand).getOffset());
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
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      auto src = offsetMap.at(arg).getPtr();
      rewriter.replaceAllUsesWith(arg, tempVar);
      arg.setType(RankedTensorType::get(tensorType.getShape(),
                                        rewriter.getIntegerType(64)));
      src = rewriter.create<triton::SplatOp>(arg.getLoc(), tempVar.getType(),
                                             src);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), src, arg);
    } else if (auto ptrType = dyn_cast<triton::PointerType>(arg.getType())) {
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
      parse(arg, arg.getLoc(), rewriter, offsetMap);
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
                                     IRMapping mapping) {
  SmallVector<Value> newOperands;
  for (auto opr : operands) {
    opr = mapping.lookupOrDefault(opr);
    newOperands.push_back(opr);
    auto numAppend = getPtrTensorRank(opr.getType()) - 1;
    if (numAppend > 0)
      newOperands.append(numAppend, tempVar);
  }
  return newOperands;
}

SmallVector<Type> constructTypes(TypeRange types) {
  SmallVector<Type> newTypes;
  for (auto type : types) {
    newTypes.push_back(type);
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
    rewriter.setInsertionPointAfter(op);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      newOp = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(),
          constructOperands(forOp.getInitArgs(), tempVar, mapping),
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            mapping.map(forOp.getInductionVar(), iv);
            auto newArgIter = args.begin();
            for (auto oldArg : forOp.getRegionIterArgs()) {
              mapping.map(oldArg, *newArgIter);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : forOp.getBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            b.create<scf::YieldOp>(
                yieldOp.getLoc(),
                constructOperands(yieldOp.getOperands(), tempVar, mapping));
          });
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      newOp = rewriter.create<scf::WhileOp>(
          whileOp.getLoc(), constructTypes(whileOp->getResultTypes()),
          constructOperands(whileOp.getInits(), tempVar, mapping),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto newArgIter = args.begin();
            for (auto oldArg : whileOp.getBeforeArguments()) {
              mapping.map(oldArg, *newArgIter);
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
                constructOperands(conditionOp.getArgs(), tempVar, mapping));
          },
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto newArgIter = args.begin();
            for (auto oldArg : whileOp.getAfterArguments()) {
              mapping.map(oldArg, *newArgIter);
              std::advance(newArgIter,
                           std::max(getPtrTensorRank(oldArg.getType()), 1));
            }
            for (auto &bodyOp : whileOp.getAfterBody()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = whileOp.getYieldOp();
            b.create<scf::YieldOp>(
                yieldOp.getLoc(),
                constructOperands(yieldOp.getOperands(), tempVar, mapping));
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
            b.create<scf::YieldOp>(
                yieldOp.getLoc(),
                constructOperands(yieldOp.getOperands(), tempVar, mapping));
          },
          [&](OpBuilder &b, Location loc) {
            for (auto &bodyOp : ifOp.elseBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yieldOp = ifOp.elseYield();
            b.create<scf::YieldOp>(
                yieldOp.getLoc(),
                constructOperands(yieldOp.getOperands(), tempVar, mapping));
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
      for (auto res : op->getResults()) {
        rewriter.replaceAllUsesWith(res, *resIter);
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
