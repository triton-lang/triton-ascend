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

#include "ascend/include/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>

#define DEBUG_TYPE "TritonNPU-Utils"

namespace mlir {

static Value createConstIndexValueOp(const Location &loc, OpBuilder &b,
                                     int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIndexAttr(value)).getResult();
}

static std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg) {
  if (isa<Attribute>(arg)) {
    return getConstantIntValue(arg);
  }

  return std::nullopt;
}

namespace ConverterUtils {

std::optional<int64_t>
getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op) {
  SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();
  if (mixedStrides.empty()) {
    op->emitError("ReinterpretCastOp has no strides");
    return std::nullopt;
  }

  OpFoldResult lastStride = mixedStrides.back();

  if (op.getStaticStrides().back() > 0) {
    return op.getStaticStrides().back();
  } else if (isa<BlockArgument>(op.getStrides().back())) {
    auto u = op.getStrides().back();
    while (auto blkArg = dyn_cast<BlockArgument>(u)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blkArg.getOwner()->getParentOp())) {
        auto prt = forOp->getOperand(3 + blkArg.getArgNumber() - 1);
        u = prt;
      } else {
        u = nullptr;
        break;
      }
    }
    if (!u)
      return std::nullopt;
    lastStride = u;
  }

  if (auto attr = lastStride.dyn_cast<Attribute>()) {
    return getConstantOfAttr(lastStride);
  } else if (auto value = lastStride.dyn_cast<Value>()) {
    auto defOp = value.getDefiningOp();
    if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(defOp)) {
      int64_t constValue = constIndexOp.value();
      return constValue;
    } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(defOp)) {
      int64_t constValue = constIntOp.value();
      return constValue;
    }
  }
  return std::nullopt;
}

bool isaPermutedMemRefType(MemRefType memRefType) {
  auto [ptrStrides, ptrOffsets] = getStridesAndOffset(memRefType);
  LLVM_DEBUG({
    llvm::dbgs() << "---------- [BEG] ptrStrides ----------\n";
    for (auto stride : ptrStrides)
      llvm::dbgs() << stride << " ";
    llvm::dbgs() << "\n";
    llvm::dbgs() << "---------- [END] ptrStrides ----------\n";
  });

  switch (ptrStrides.size()) {
  case 0:
    return false;
  case 1:
    return false;
  default: {
    return ptrStrides[ptrStrides.size() - 1] != 1;
  }
  }
}

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(order);
  SmallVector<int64_t> originalShape(sourceType.getShape());
  SmallVector<int64_t> transposedShape(sourceRank);
  for (size_t i = 0; i < sourceRank; i++) {
    transposedShape[i] = originalShape[perm[i]];
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;
  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = mlir::TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

memref::SubViewOp makeSubViewOp(Value src,
                                const llvm::SmallVector<OpFoldResult> &offsets,
                                const llvm::SmallVector<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter) {
  auto srcType = cast<MemRefType>(src.getType());
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<memref::SubViewOp>(loc, dyn_cast<MemRefType>(dstType),
                                            src, offsets, sizes, strides);
}

tensor::ExtractSliceOp
makeExtractSliceOp(Value src, const llvm::SmallVector<OpFoldResult> &offsets,
                   const llvm::SmallVector<OpFoldResult> &sizes,
                   const Location &loc, ConversionPatternRewriter &rewriter) {
  auto srcType = cast<RankedTensorType>(src.getType());
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      tensor::ExtractSliceOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<tensor::ExtractSliceOp>(loc, dstType, src, offsets,
                                                 sizes, strides);
}

std::optional<Operation *> getFullShapeOp(Value val,
                                          ConversionPatternRewriter &rewriter) {
  assert(isa<BaseMemRefType>(val.getType()));

  if (isa<BlockArgument>(val)) {
    auto blockArg = dyn_cast<BlockArgument>(val);
    auto blockOp = blockArg.getOwner()->getParentOp();
    if (isa<scf::ForOp>(blockOp)) {
      auto forOp = dyn_cast<scf::ForOp>(blockOp);
      auto operand = forOp.getTiedLoopInit(blockArg)->get();
      return getFullShapeOp(operand, rewriter);
    } else {
      emitError(val.getLoc())
          << "getFullShapeOp() only support ReinterpretCastOp "
             "and scf.for's block argument, but got : "
          << val << "\n";
    }
    return std::nullopt;
  }

  if (!isa<memref::ReinterpretCastOp>(val.getDefiningOp())) {
    emitError(val.getLoc())
        << "getFullShapeOp() only support ReinterpretCastOp "
           "and scf.for's block argument, but got : "
        << val << "\n";
    return std::nullopt;
  }

  auto reCastOp = val.getDefiningOp<memref::ReinterpretCastOp>();
  if (reCastOp->hasAttr("tensor_ptr_full_shape"))
    return reCastOp;

  return getFullShapeOp(reCastOp.getSource(), rewriter);
}

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 const Location &loc, ConversionPatternRewriter &rewriter) {
  if (isa<triton::PointerType>(ptr.getType()))
    ptr = rewriter.getRemappedValue(ptr);

  auto shapedType = dyn_cast_if_present<ShapedType>(ptr.getType());
  assert(shapedType && shapedType.hasStaticShape());

  auto fullShapeOp = getFullShapeOp(ptr, rewriter);

  assert(fullShapeOp.has_value());
  SmallVector<OpFoldResult> boundarySize =
      getAsIndexOpFoldResult(rewriter.getContext(), shapedType.getShape());

  auto fullShapeReCast =
      dyn_cast<memref::ReinterpretCastOp>(fullShapeOp.value());
  OpFoldResult curPtrOffset;
  if (auto curReCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    curPtrOffset = curReCast.getConstifiedMixedOffset();
  } else if (isa<BlockArgument>(ptr) &&
             isa<scf::ForOp>(ptr.getParentBlock()->getParentOp())) {
    // Here's to process loop state where ptr is just from loop interator.
    // Following assertion corresponds to conversion result from `rewriteFor`
    auto blockArg = dyn_cast<BlockArgument>(ptr);
    auto forOp = dyn_cast<scf::ForOp>(ptr.getParentBlock()->getParentOp());
    auto initReCastOfLoop = forOp.getTiedLoopInit(blockArg)
                                ->get()
                                .getDefiningOp<memref::ReinterpretCastOp>();
    assert(initReCastOfLoop && initReCastOfLoop.getOffsets().size() == 1);
    Value initReCastOffset = initReCastOfLoop.getOffsets()[0];

    for (OpOperand &use : initReCastOffset.getUses()) {
      if (use.getOwner() == initReCastOfLoop)
        continue;
      else if (isa<memref::ReinterpretCastOp>(use.getOwner()))
        continue;
      else if (use.getOwner() == forOp)
        curPtrOffset = OpFoldResult(forOp.getTiedLoopRegionIterArg(&use));
      else
        llvm_unreachable("Illegal interation offset after rewriteFor");
    }
  } else {
    llvm_unreachable("Unsupported state when check tensor_ptr boundary");
  }

  assert(curPtrOffset);

  OpFoldResult offsetShift = subOpFoldResult(
      curPtrOffset, fullShapeReCast.getConstifiedMixedOffset(), loc, rewriter);

  for (int i = 0; i < shapedType.getRank(); ++i) {
    if (llvm::find(boundaryCheck, i) != boundaryCheck.end()) {
      auto fullShape = fullShapeReCast.getConstifiedMixedSizes()[i];

      OpFoldResult curOffset = divOpFoldResult(
          offsetShift, fullShapeReCast.getConstifiedMixedStrides()[i], loc,
          rewriter);
      OpFoldResult curLeftSize =
          maxOpFoldResult(subOpFoldResult(fullShape, curOffset, loc, rewriter),
                          rewriter.getIndexAttr(0), loc, rewriter);

      boundarySize[i] =
          minOpFoldResult(boundarySize[i], curLeftSize, loc, rewriter);

      offsetShift = remOpFoldResult(
          offsetShift, fullShapeReCast.getConstifiedMixedStrides()[i], loc,
          rewriter);
    }
  }

  return boundarySize;
}

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1 &&
             "Size of source broadcast dimension must be 1");
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "Cannot identify broadcast dimension");
  return broadcastDims;
}

// Dimensions of collapesd tensor is all unbroadcast dims
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) {
  SmallVector<int64_t> unbroadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] == srcShape[i]) {
      unbroadcastDims.emplace_back(srcShape[i]);
    }
  }
  return unbroadcastDims;
}

} // namespace ConverterUtils

namespace triton {

mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn) {
  LLVM_DEBUG(llvm::dbgs() << "[findFirstMatchingOperandDef] Current op: "
                          << *rootOp << "\n");
  mlir::Value lhs = nullptr;
  mlir::Value rhs = nullptr;
  if (auto op = dyn_cast<triton::AddPtrOp>(rootOp)) {
    lhs = op.getPtr();
    rhs = op.getOffset();
  } else if (auto op = dyn_cast<arith::AddIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::SubIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::MulIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::DivSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::RemSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<triton::SplatOp>(rootOp)) {
    lhs = op.getSrc();
  } else if (auto op = dyn_cast<triton::MakeRangeOp>(rootOp)) {
  } else {
    rootOp->emitRemark("Backtracing encounters unsupported Operation");
    return nullptr;
  }
  // Backtrace operands
  if (!lhs) {
    return nullptr;
  }
  auto lhsDef = lhs.getDefiningOp();
  mlir::Operation *targetOp;
  if (lhsDef) {
    if (condFn(lhsDef)) {
      targetOp = lhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(lhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  if (!rhs) {
    return nullptr;
  }
  auto rhsDef = rhs.getDefiningOp();
  if (rhsDef) {
    if (condFn(rhsDef)) {
      targetOp = rhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(rhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  return nullptr;
}

void traverseBackwardUpdateOperandChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    DenseSet<Operation *> &handledOperation) {

  if (!op || handledOperation.contains(op))
    return;

  handledOperation.insert(op);

  if (stopFn(op))
    return;

  if (conditionFn(op))
    actionFn(builder, op);

  DenseSet<Value> handledOperand;

  std::function<void(Value)> handler = [&](Value operand) {
    if (handledOperand.contains(operand))
      return;
    handledOperand.insert(operand);
    if (Operation *defOp = operand.getDefiningOp()) {
      traverseBackwardUpdateOperandChainIf(defOp, conditionFn, stopFn, actionFn,
                                           builder, handledOperation);
    } else {
      auto blockArgument = cast<BlockArgument>(operand);
      auto parentOp = blockArgument.getOwner()->getParentOp();
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp);
          whileOp && whileOp.getAfterBody() == blockArgument.getOwner()) {
        auto argNum = blockArgument.getArgNumber();
        auto conditionArg = whileOp.getConditionOp().getArgs()[argNum];
        handler(conditionArg);
      } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
        OpOperand *initArgOperand = loopOp.getTiedLoopInit(blockArgument);
        if (!initArgOperand)
          return;
        Value initArg = initArgOperand->get();
        handler(initArg);
        Value yieldedValue =
            loopOp.getTiedLoopYieldedValue(blockArgument)->get();
        if (yieldedValue != blockArgument)
          handler(yieldedValue);
      }
    }
  };

  for (Value operand : op->getOperands()) {
    handler(operand);
  }

  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    for (auto yieldedValue : loopOp.getYieldedValues())
      handler(yieldedValue);
  }
}

// Note: rootOp will also be processed.
void traverseBackwardUpdateOperandChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn) {

  OpBuilder builder(rootOp->getContext());
  DenseSet<Operation *> handledOperation;

  traverseBackwardUpdateOperandChainIf(rootOp, conditionFn, stopFn, actionFn,
                                       builder, handledOperation);
}

void traverseForwardUpdateUserChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  if (!op) {
    return;
  }

  if (stopFn(op)) {
    stopOps.insert(op);
    return;
  }

  if (conditionFn(op)) {
    actionFn(builder, op);
  }

  for (auto res : op->getResults()) {
    for (auto userOp : res.getUsers()) {
      traverseForwardUpdateUserChainIf(userOp, conditionFn, stopFn, actionFn,
                                       builder, stopOps);
    }
  }
}

// Note: rootOp will also be processed.
void traverseForwardUpdateUserChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  OpBuilder builder(rootOp->getContext());

  traverseForwardUpdateUserChainIf(rootOp, conditionFn, stopFn, actionFn,
                                   builder, stopOps);
}

bool isMetaUse(Operation *op) { return op->hasAttr("MetaUse"); }

bool isMixUse(Operation *op) { return op->hasAttr("MixUse"); }

IndirectLoadInterfaceOpType getIndirectLoadInterfaceOpType(Operation *op) {
  auto ty = IndirectLoadInterfaceOpType::Undefined;
  if (isMetaUse(op)) {
    if (isa<triton::LoadOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Load;
    } else if (isa<arith::FPToSIOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Calc;
    }
  }
  return ty;
}

bool opIsIndirectLoad(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Load;
}

bool opIsIndirectCalc(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Calc;
}

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder) {

  if (currentDim >= totalDims) {
    bodyBuilder(builder, loc, ivs, initArgs);
    return nullptr;
  }

  auto loop = builder.create<scf::ForOp>(
      loc, LBs[currentDim], UBs[currentDim], steps[currentDim], initArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange iterArgs) {
        ivs.push_back(iv);
        auto innerLoop = createNestedLoops(nestedBuilder, nestedLoc,
                                           currentDim + 1, totalDims, LBs, UBs,
                                           steps, ivs, iterArgs, bodyBuilder);
        if (innerLoop) {
          nestedBuilder.create<scf::YieldOp>(loc, innerLoop.getResults());
        }
      });

  return loop;
}

ModuleOp getModuleOpFromOperation(Operation *op) {
  Operation *parent = op;
  while (parent != nullptr && !isa<ModuleOp>(parent)) {
    parent = parent->getParentOp(); // 向上查找
  }
  return cast<ModuleOp>(parent); // 如果没找到会抛出异常
}

} // namespace triton

// TODO: imply these function below
OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() + rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;
  if (!rhsInt && lhsInt && lhsInt.value() == 0)
    return rhs;

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() - rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::SubIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() * rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
    if (lhsInt.value() == 1)
      return rhs;
  }
  if (rhsInt) {
    if (rhsInt.value() == 0)
      return rhs;
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot div 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() / rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  if (rhsInt) {
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::DivSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot remainder by 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() % rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::RemSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::min(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::MinSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::max(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  if (rhsInt) {
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  } else {
    lhsValue = convertToIndexIfNeeded(lhsValue, loc, b);
    assert(isa<IndexType>(lhsValue.getType()));
  }

  return b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue).getResult();
}

void addReduceWithIndexAttr(ReduceWithIndexParams params,
                            ConversionPatternRewriter &rewriter,
                            linalg::ReduceOp reduceOp) {
  const StringRef reduceRef = "reduce_mode";
  const StringRef tieBreakLeftRef = "tie_break_left";
  const StringRef unsignedSrcRef = "unsigned_src";

  const StringRef tieBreakStr =
      params.tieBreakType == TieBreakType::LEFT ? "true" : "false";
  const StringRef withIndexStr =
      params.withIndexType == ReduceWithIndexType::MAX ? "max_with_index"
                                                       : "min_with_index";
  const StringRef unsignedSrcStr = params.isUnsignedSrc ? "true" : "false";

  reduceOp->setAttr(reduceRef, rewriter.getStringAttr(withIndexStr));
  reduceOp->setAttr(tieBreakLeftRef, rewriter.getStringAttr(tieBreakStr));
  reduceOp->setAttr(unsignedSrcRef, rewriter.getStringAttr(unsignedSrcStr));
}

llvm::FailureOr<ReduceWithIndexParams>
getReduceWithIndexParams(triton::ReduceOp op) {
  auto tritonReduceBlock = op.getBody();
  auto *tritonYield = tritonReduceBlock->getTerminator();
  auto yieldValues = tritonYield->getOperands();
  constexpr int yieldValuesNum = 2;
  if (yieldValues.empty()) {
    return llvm::failure();
  }
  if (yieldValues.size() != yieldValuesNum) {
    return ReduceWithIndexParams{};
  }

  // Unify signed/unsigned and int/float predicate
  enum class Predicate { Undefined = 0, lt = 1, gt = 2, eq = 3 };
  enum class Signedness { NotApplicable = 0, Signed = 1, Unsigned = 2 };
  auto unifyPredicateI =
      [](arith::CmpIPredicate p) -> std::pair<Predicate, Signedness> {
    switch (p) {
    case arith::CmpIPredicate::slt:
      return {Predicate::lt, Signedness::Signed};
    case arith::CmpIPredicate::ult:
      return {Predicate::lt, Signedness::Unsigned};
    case arith::CmpIPredicate::sgt:
      return {Predicate::gt, Signedness::Signed};
    case arith::CmpIPredicate::ugt:
      return {Predicate::gt, Signedness::Unsigned};
    case arith::CmpIPredicate::eq:
      return {Predicate::eq, Signedness::NotApplicable};
    default:
      return {Predicate::Undefined, Signedness::NotApplicable};
    }
  };
  auto unifyPredicateF =
      [](arith::CmpFPredicate p) -> std::pair<Predicate, Signedness> {
    switch (p) {
    case arith::CmpFPredicate::OLT:
      return {Predicate::lt, Signedness::Signed};
    case arith::CmpFPredicate::ULT:
      return {Predicate::lt, Signedness::Unsigned};
    case arith::CmpFPredicate::OGT:
      return {Predicate::gt, Signedness::Signed};
    case arith::CmpFPredicate::UGT:
      return {Predicate::gt, Signedness::Unsigned};
    case arith::CmpFPredicate::OEQ:
      return {Predicate::eq, Signedness::Signed};
    case arith::CmpFPredicate::UEQ:
      return {Predicate::eq, Signedness::Unsigned};
    default:
      return {Predicate::Undefined, Signedness::NotApplicable};
    }
  };

  // Composite predicate to pick index of min (or max) element have to be
  // written in following form: (v means value and i means index)
  // For leftmost element:
  //    (new_v == old_v and new_i < old_i) or new_v < old_v
  //    new_v < old_v or (new_v == old_v and new_i < old_i)
  //    new_v < old_v // python3.11 ttir
  //    (new_v == old_v and new_i < old_i) or new_v > old_v
  //    new_v > old_v or (new_v == old_v and new_i < old_i)
  //    new_v > old_v // python3.11 ttir
  // For rightmost element:
  //    (new_v == old_v and new_i > old_i) or new_v < old_v
  //    new_v < old_v or (new_v == old_v and new_i > old_i)
  //    (new_v == old_v and new_i > old_i) or new_v > old_v
  //    new_v > old_v or (new_v == old_v and new_i > old_i)

  std::map<std::vector<Predicate>, std::pair<ReduceWithIndexType, TieBreakType>>
      m{
          // leftmost
          {{Predicate::eq, Predicate::lt, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::lt, Predicate::eq, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::lt}, {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::eq, Predicate::lt, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          {{Predicate::gt, Predicate::eq, Predicate::lt},
           {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          {{Predicate::gt}, {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          // rightmost
          {{Predicate::eq, Predicate::gt, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::RIGHT}},
          {{Predicate::lt, Predicate::eq, Predicate::gt},
           {ReduceWithIndexType::MIN, TieBreakType::RIGHT}},
          {{Predicate::eq, Predicate::gt, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::RIGHT}},
          {{Predicate::gt, Predicate::eq, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::RIGHT}},
      };

  std::vector<Predicate> preds;
  std::vector<Signedness> signednesses;
  // A better way is to trace the arith.select
  // Checking the operations one by one is hacky :(
  for (auto &op : tritonReduceBlock->without_terminator()) {
    Predicate pred = Predicate::Undefined;
    Signedness signedness = Signedness::NotApplicable;
    if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
      auto predi = cmpiOp.getPredicate();
      std::tie(pred, signedness) = unifyPredicateI(predi);
    }
    if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
      auto predf = cmpfOp.getPredicate();
      std::tie(pred, signedness) = unifyPredicateF(predf);
    }
    if (pred != Predicate::Undefined) {
      preds.push_back(pred);
      signednesses.push_back(signedness);
    }
  }

  // check if sequence of predicates matches any sequence for min/max
  // leftmost/rightmost
  if (m.find(preds) == m.end()) {
    return llvm::failure();
  }

  assert(!signednesses.empty());
  const bool isUnsignedSrc =
      signednesses[0] == Signedness::Unsigned ||
      signednesses[signednesses.size() - 1] == Signedness::Unsigned;
  return ReduceWithIndexParams{.withIndexType = m.at(preds).first,
                               .tieBreakType = m.at(preds).second,
                               .isUnsignedSrc = isUnsignedSrc};
}

// Fold layout constant info to attr, otherwise convert to index type value
OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder) {
  OpFoldResult constantFold = getAsOpFoldResult(value);
  if (llvm::isa<Attribute>(constantFold)) {
    assert(isa<IntegerAttr>(constantFold.get<Attribute>()));
    return constantFold;
  }

  if (!isa<IntegerType>(value.getType()))
    llvm_unreachable("Illegal data type when parse block data layout info");

  if (!isa<IndexType>(value.getType())) {
    if (value.getType().isInteger(/*width*/ 1))
      value = builder.create<arith::IndexCastUIOp>(
          value.getLoc(), builder.getIndexType(), value);
    else
      value = builder.create<arith::IndexCastOp>(value.getLoc(),
                                                 builder.getIndexType(), value);
  }

  return value;
}

// Specialize the Typeless Value (Zero, Min, Max) into a mlir TypedAttr
FailureOr<TypedAttr> specializeTypelessValueToAttr(TypelessValue value,
                                                   Type type, OpBuilder &b) {
  mlir::Type f16Ty = Float16Type::get(b.getContext());
  mlir::Type f32Ty = Float32Type::get(b.getContext());
  mlir::Type i8TySL = IntegerType::get(
      b.getContext(), 8, IntegerType::SignednessSemantics::Signless);
  mlir::Type i8TyS = IntegerType::get(b.getContext(), 8,
                                      IntegerType::SignednessSemantics::Signed);
  mlir::Type i8TyU = IntegerType::get(
      b.getContext(), 8, IntegerType::SignednessSemantics::Unsigned);
  mlir::Type i16TySL = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Signless);
  mlir::Type i16TyS = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Signed);
  mlir::Type i16TyU = IntegerType::get(
      b.getContext(), 16, IntegerType::SignednessSemantics::Unsigned);
  mlir::Type i32TySL = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Signless);
  mlir::Type i32TyS = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Signed);
  mlir::Type i32TyU = IntegerType::get(
      b.getContext(), 32, IntegerType::SignednessSemantics::Unsigned);
  mlir::Type i64TySL = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Signless);
  mlir::Type i64TyS = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Signed);
  mlir::Type i64TyU = IntegerType::get(
      b.getContext(), 64, IntegerType::SignednessSemantics::Unsigned);
  llvm::APFloat halfZero = llvm::APFloat::getZero(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfOne(llvm::APFloat::IEEEhalf(), 1);
  llvm::APFloat halfMax = llvm::APFloat::getInf(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEhalf(), true);
  llvm::APFloat floatZero = llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatOne(llvm::APFloat::IEEEsingle(), 1);
  llvm::APFloat floatMax = llvm::APFloat::getInf(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), true);
  auto toPtr = [](mlir::Type ty) { return ty.getAsOpaquePointer(); };

  std::map<std::pair<TypelessValue, const void *>,
           std::variant<int8_t, int16_t, int32_t, int64_t, uint32_t, uint64_t,
                        llvm::APFloat>>
      initMap = {
          {{TypelessValue::Zero, toPtr(f16Ty)}, halfZero},
          {{TypelessValue::Zero, toPtr(f32Ty)}, floatZero},
          {{TypelessValue::Zero, toPtr(i16TySL)}, (int16_t)0},
          {{TypelessValue::Zero, toPtr(i16TyS)}, (int16_t)0},
          {{TypelessValue::Zero, toPtr(i16TyU)}, (int16_t)0},
          {{TypelessValue::Zero, toPtr(i32TySL)}, 0},
          {{TypelessValue::Zero, toPtr(i32TyS)}, 0},
          {{TypelessValue::Zero, toPtr(i32TyU)}, 0},
          {{TypelessValue::Zero, toPtr(i64TySL)}, (int64_t)0},
          {{TypelessValue::Zero, toPtr(i64TyS)}, (int64_t)0},
          {{TypelessValue::Zero, toPtr(i64TyU)}, (int64_t)0},
          {{TypelessValue::Min, toPtr(f16Ty)}, halfMin},
          {{TypelessValue::Min, toPtr(f32Ty)}, floatMin},
          {{TypelessValue::Min, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::min()},
          {{TypelessValue::Min, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::min()},
          {{TypelessValue::Min, toPtr(i16TyU)},
           std::numeric_limits<uint16_t>::min()},
          {{TypelessValue::Min, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::min()},
          {{TypelessValue::Min, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::min()},
          {{TypelessValue::Min, toPtr(i32TyU)},
           std::numeric_limits<uint32_t>::min()},
          {{TypelessValue::Min, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::min()},
          {{TypelessValue::Min, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::min()},
          {{TypelessValue::Min, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::min()},
          {{TypelessValue::Max, toPtr(f16Ty)}, halfMax},
          {{TypelessValue::Max, toPtr(f32Ty)}, floatMax},
          {{TypelessValue::Max, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::max()},
          {{TypelessValue::Max, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::max()},
          {{TypelessValue::Max, toPtr(i16TyU)},
           std::numeric_limits<uint16_t>::max()},
          {{TypelessValue::Max, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::max()},
          {{TypelessValue::Max, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::max()},
          {{TypelessValue::Max, toPtr(i32TyU)},
           std::numeric_limits<uint32_t>::max()},
          {{TypelessValue::Max, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::max()},
          {{TypelessValue::Max, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::max()},
          {{TypelessValue::Max, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::max()},
      };

  std::pair<TypelessValue, const void *> key =
      std::make_pair(value, toPtr(type));
  if (initMap.find(key) == initMap.end())
    return failure();
  if (type.isInteger(8))
    return success(IntegerAttr::get(IntegerType::get(b.getContext(), 8),
                                    std::get<int8_t>(initMap.at(key))));
  if (type.isInteger(16))
    return success(IntegerAttr::get(IntegerType::get(b.getContext(), 16),
                                    std::get<int16_t>(initMap.at(key))));
  if (type.isInteger(32))
    return success(IntegerAttr::get(IntegerType::get(b.getContext(), 32),
                                    std::get<int32_t>(initMap.at(key))));
  if (type.isInteger(64))
    return success(IntegerAttr::get(IntegerType::get(b.getContext(), 64),
                                    std::get<int64_t>(initMap.at(key))));
  if (isa<Float16Type>(type))
    return success(
        FloatAttr::get(f16Ty, std::get<llvm::APFloat>(initMap.at(key))));
  if (isa<Float32Type>(type))
    return success(
        FloatAttr::get(f32Ty, std::get<llvm::APFloat>(initMap.at(key))));
  return failure();
}

// Specialize the Typeless Value (Zero, Min, Max) into a mlir constant value
FailureOr<Value> specializeTypelessValueToConstant(TypelessValue value,
                                                   Type type, Location loc,
                                                   OpBuilder &b) {
  std::function<mlir::Type(mlir::Type)> getElemType = [&](mlir::Type ty) {
    if (auto ptrType = dyn_cast<triton::PointerType>(getElementTypeOrSelf(ty)))
      return getElemType(ptrType.getPointeeType());
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(ty))
      return getElemType(tensorType.getElementType());
    return ty;
  };

  if (value == TypelessValue::Undefined)
    return failure();
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    auto elemType = getElemType(tensorType);
    FailureOr<TypedAttr> typedAttr =
        specializeTypelessValueToAttr(value, elemType, b);
    if (failed(typedAttr))
      return failure();
    auto otherTensorType =
        RankedTensorType::get(tensorType.getShape(), elemType);
    auto denseAttr = DenseElementsAttr::get(otherTensorType, *typedAttr);
    return b.create<mlir::arith::ConstantOp>(loc, denseAttr).getResult();
  }
  if (mlir::isa<mlir::FloatType>(type) || mlir::isa<mlir::IntegerType>(type)) {
    FailureOr<TypedAttr> typedAttr =
        specializeTypelessValueToAttr(value, type, b);
    if (failed(typedAttr))
      return failure();
    return b.create<mlir::arith::ConstantOp>(loc, *typedAttr).getResult();
  }
  return failure();
}

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  Attribute attr;
  if (auto val = dyn_cast<Value>(ofr)) {
    if (!val.getDefiningOp<arith::ConstantOp>())
      return std::nullopt;
    attr = cast<IntegerAttr>(val.getDefiningOp<arith::ConstantOp>().getValue());
  } else {
    attr = dyn_cast<Attribute>(ofr);
  }
  if (attr && isa<IntegerAttr>(attr))
    return dyn_cast<IntegerAttr>(attr).getInt();
  return std::nullopt;
}

Value materializeValue(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>()) {
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(intVal.value()));
  }
  assert(intVal.has_value());
  return Value();

  // return builder.create<arith::ConstantIndexOp>(
  //     loc, dyn_cast<IntegerAttr>(attr).getInt());
}

bool isZero(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 0;
}

bool isOne(const OpFoldResult ofr) {
  auto staticOfr = getIntAttr(ofr);
  return staticOfr.has_value() && staticOfr.value() == 1;
}

Value convertToIndexIfNeeded(Value input, const Location &loc, OpBuilder &b) {
  auto inputType = input.getType();
  if (auto intType = dyn_cast<IntegerType>(inputType)) {
    if (intType.isInteger(32) || intType.isInteger(64)) {
      return b.create<arith::IndexCastOp>(loc, b.getIndexType(), input);
    }
  }
  return input;
}
} // namespace mlir
