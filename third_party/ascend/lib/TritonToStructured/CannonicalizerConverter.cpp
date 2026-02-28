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

#include "TritonToStructured/CannonicalizerConverter.h"

#include <cassert>
#include <numeric>
#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "llvm/Support/Debug.h"

#include "TritonToStructured/PtrAnalysis.h"
#include "TritonToStructured/TritonToStructuredPass.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#define DEBUG_TYPE "triton-cannonicalizer-converter"

namespace CannonicalizerConverter {
using namespace mlir;
using namespace triton;

// Match and rewrite pattern for optimizing cmp.ne (select(cond, 1, 0), 0) ->
// cond This pattern transforms:
//   %select = arith.select %cond, %true_val, %false_val
//   %cmp = arith.cmpi ne, %select, %zero
// Where:
//   - %true_val is a constant splat tensor of 1s
//   - %false_val is a constant splat tensor of 0s
//   - %zero is a constant splat tensor of 0s
// Into:
//   %cond  (directly replace the cmp with the select condition)
//
// This optimization is valid because:
//   select(cond, 1, 0) != 0
//   is equivalent to: cond != 0
//   Since the result of select is either 1 or 0, the only way it's not equal to
//   0 is when it's 1, which happens exactly when cond is true.
//
// Example:
//   Input IR:
//     %39 = arith.cmpi slt, %15, %cst_14 : tensor<128xi32>
//     %40 = arith.select %39, %cst_13, %cst_12 : tensor<128xi1>,
//     tensor<128xi32> %41 = arith.cmpi ne, %40, %cst_12 : tensor<128xi32>
//   Where cst_13 is constant dense<1> and cst_12 is constant dense<0>
//   Output IR:
//     %39 = arith.cmpi slt, %15, %cst_14 : tensor<128xi32>
LogicalResult CmpConverter::matchAndRewrite(arith::CmpIOp cmpOp,
                                            PatternRewriter &rewriter) const {
  // Only handle "not equal" comparison
  auto cmpType = cmpOp.getPredicate();
  if (cmpType != arith::CmpIPredicate::ne) {
    return failure();
  }

  Value rhs = cmpOp.getRhs();
  Value lhs = cmpOp.getLhs();

  // 1. Check if RHS is a constant zero
  APInt rhsValue;
  if (!matchPattern(rhs, m_ConstantInt(&rhsValue))) {
    return failure(); // RHS is not a constant
  }

  if (!rhsValue.isZero()) {
    return failure(); // RHS is not zero
  }

  // 2. Check if LHS is defined by a select operation
  auto selectOp = lhs.getDefiningOp<arith::SelectOp>();
  if (!selectOp) {
    return failure();
  }

  // 3. Check if select's true and false values are constants
  DenseElementsAttr trueAttr;
  DenseElementsAttr falseAttr;
  if (!matchPattern(selectOp.getTrueValue(), m_Constant(&trueAttr)) ||
      !matchPattern(selectOp.getFalseValue(), m_Constant(&falseAttr))) {
    return failure(); // Either true or false value is not constant
  }

  // 4. Check if true value is all 1s and false value is all 0s
  if (!trueAttr.isSplat() || !trueAttr.getSplatValue<APInt>().isOne() ||
      !falseAttr.isSplat() || !falseAttr.getSplatValue<APInt>().isZero()) {
    return failure();
  }

  // 5. Optimization matched, replace cmp with select's condition
  rewriter.replaceOp(cmpOp, selectOp.getCondition());
  return success();
}

// Detect when both operands of the cmpOp are triton::SplatOp. If so,
// replace the original comparison by comparing the underlying scalar values
// and then splatting (broadcasting) the scalar comparison result back to the
// original tensor shape.
// Example:
//   Input IR:
//     %splat_lhs = tt.splat %val1 : tensor<128xi32>
//     %splat_rhs = tt.splat %val2 : tensor<128xi32>
//     %cmp = arith.cmpi slt, %splat_lhs, %splat_rhs : tensor<128xi32>
//   Output IR:
//     %cmp_scalar = arith.cmpi slt, %val1, %val2
//     %splat_cmp = tt.splat %cmp_scalar : tensor<128xi1>
LogicalResult
SplatCmpConverter::matchAndRewrite(arith::CmpIOp cmpOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto lhs = cmpOp.getLhs();
  auto rhs = cmpOp.getRhs();
  auto lhsSplatOp = lhs.getDefiningOp<triton::SplatOp>();
  auto rhsSplatOp = rhs.getDefiningOp<triton::SplatOp>();
  if (!lhsSplatOp || !rhsSplatOp) {
    return failure();
  }
  auto lhsSrc = lhsSplatOp.getSrc();
  auto rhsSrc = rhsSplatOp.getSrc();
  auto newCmpOp = rewriter.create<arith::CmpIOp>(
      cmpOp.getLoc(), cmpOp.getPredicate(), lhsSrc, rhsSrc);
  auto cmpType = dyn_cast<RankedTensorType>(cmpOp.getType());
  if (!cmpType) {
    return failure();
  }
  auto splatType =
      RankedTensorType::get(cmpType.getShape(), newCmpOp.getType());
  auto splatOp = rewriter.create<triton::SplatOp>(cmpOp.getLoc(), splatType,
                                                  newCmpOp.getResult());
  rewriter.replaceOp(cmpOp, splatOp.getResult());
  return success();
}

// Transform a for loop that uses pointer iteration arguments into one that uses
// integer offsets instead. This pattern handles the specific case where:
// 1. The loop has pointer iteration arguments of type like
// tensor<1024x!tt.ptr<f32>>
// 2. Each pointer is used in a load/store operation and then incremented by
//    a constant offset via tt.addptr
// 3. The updated pointer (from addptr) is yielded back as the next iteration
// value
//
// The transformation converts:
//   scf.for iter_args(%ptr = %base_ptr) {
//     %val = tt.load %ptr
//     tt.store %other_ptr, %val
//     %new_ptr = tt.addptr %ptr, %offset
//     scf.yield %new_ptr
//   }
//
// Into:
//   scf.for iter_args(%offset_int = 0) {
//     %splat_offset = tt.splat %offset_int
//     %current_ptr = tt.addptr %base_ptr, %splat_offset
//     %val = tt.load %current_ptr
//     tt.store %other_ptr, %val
//     %new_offset = arith.addi %offset_int, %const_offset
//     scf.yield %new_offset
//   }
//
LogicalResult PromotePointerIterArgsPattern::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  // 1. Check if the loop meets transformation conditions
  if (failed(matchLoop(forOp))) {
    return failure();
  }

  // 2. Collect pointer iteration arguments to be processed
  auto pointerArgsInfo = collectPointerIterArgs(forOp);
  if (pointerArgsInfo.empty()) {
    return failure();
  }

  // 3. Create new iteration argument types and initial values
  auto [newInitArgs, newIterArgTypes, indexMap] =
      createNewIterArgs(forOp, pointerArgsInfo, rewriter);

  // 4. Create the new for loop
  auto newForOp =
      createNewForLoop(forOp, newInitArgs, newIterArgTypes, rewriter);

  // 5. Rewrite the loop body
  if (failed(rewriteLoopBody(forOp, newForOp, pointerArgsInfo, indexMap,
                             rewriter))) {
    return failure();
  }

  // 6. Replace original loop results
  return replaceResults(forOp, newForOp, pointerArgsInfo, indexMap, rewriter);
}

LogicalResult PromotePointerIterArgsPattern::matchLoop(scf::ForOp forOp) const {
  auto lowerBound = forOp.getLowerBound();
  auto upperBound = forOp.getUpperBound();
  auto step = forOp.getStep();
  if (!matchPattern(lowerBound, m_Constant()) ||
      !matchPattern(upperBound, m_Constant()) ||
      !matchPattern(step, m_Constant())) {
    return failure();
  }
  return success();
}

SmallVector<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::collectPointerIterArgs(scf::ForOp forOp) const {
  SmallVector<PointerArgInfo> result;
  auto &loopBody = *forOp.getBody();

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (isPointerIterArg(iterArg)) {
      auto info = analyzePointerIterArg(iterArg, loopBody);
      if (info.has_value()) {
        info->oldIndex = static_cast<unsigned>(idx),
        info->basePointer = forOp.getInitArgs()[idx],
        result.push_back(info.value());
      }
    }
  }
  return result;
}

bool PromotePointerIterArgsPattern::isPointerIterArg(Value iterArg) const {
  auto ptrType = dyn_cast<TensorType>(iterArg.getType());
  return ptrType && isa<triton::PointerType>(ptrType.getElementType());
}

std::optional<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::analyzePointerIterArg(Value iterArg,
                                                     Block &loopBody) const {
  int memCount =
      0; // Count of memory operations (load/store) using this pointer
  int addPtrCount = 0;          // Count of addptr operations on this pointer
  Value addPtrResult = nullptr; // Result of the addptr operation
  Value offset = nullptr;       // Offset value used in addptr
  Value addPtrValue = nullptr;  // The addptr operation result value

  for (auto &op : loopBody) {
    TypeSwitch<Operation *>(&op)
        .Case<triton::LoadOp, triton::StoreOp>([&](auto memoryOp) {
          // Check if this memory operation uses the pointer we're analyzing
          if (memoryOp.getPtr() == iterArg)
            ++memCount;
        })
        .Case<triton::AddPtrOp>([&](auto addPtrOp) {
          // Check if this addptr operation updates the pointer we're analyzing
          if (addPtrOp.getPtr() == iterArg) {
            ++addPtrCount;
            addPtrResult = addPtrOp.getResult();
            offset = addPtrOp.getOffset();
            addPtrValue = addPtrOp.getResult();
          }
        })
        .Default([](auto) {}); // Ignore other operations
  }

  // Check the terminator to see if the addptr result is yielded
  auto yieldOp = dyn_cast<scf::YieldOp>(loopBody.getTerminator());
  if (!yieldOp)
    return std::nullopt;

  bool isYielded = false;
  for (auto operand : yieldOp.getOperands()) {
    if (operand == addPtrResult) {
      isYielded = true;
      break;
    }
  }

  // Pattern matched if:
  // 1. Exactly one addptr operation on this pointer
  // 2. At least one memory operation using this pointer
  // 3. The addptr result is yielded
  if (addPtrCount == 1 && memCount >= 1 && isYielded) {
    return PointerArgInfo{
        .oldIndex = 0,
        .basePointer = nullptr, // Will be set in collectPointerIterArgs
        .offsetValue = offset,
        .newIterArg = nullptr, // Will be set in createNewIterArgs
        .addPtrValue = addPtrValue};
  }
  return std::nullopt;
}

std::tuple<SmallVector<Value>, SmallVector<Type>, DenseMap<unsigned, unsigned>>
PromotePointerIterArgsPattern::createNewIterArgs(
    scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
    PatternRewriter &rewriter) const {
  SmallVector<Value> newInitArgs;
  SmallVector<Type> newIterArgTypes;
  DenseMap<unsigned, unsigned> indexMap;

  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      // Replace pointer with integer offset (initialized to 0)
      Value zero = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
      newInitArgs.push_back(zero);
      newIterArgTypes.push_back(rewriter.getIntegerType(32));
    } else {
      // Preserve original argument unchanged
      newInitArgs.push_back(forOp.getInitArgs()[i]);
      newIterArgTypes.push_back(forOp.getInitArgs()[i].getType());
    }

    // Identity mapping: argument count and order unchanged，
    // may change in future
    indexMap[i] = i;
  }

  return {newInitArgs, newIterArgTypes, indexMap};
}

scf::ForOp PromotePointerIterArgsPattern::createNewForLoop(
    scf::ForOp forOp, ArrayRef<Value> newInitArgs,
    ArrayRef<Type> newIterArgTypes, PatternRewriter &rewriter) const {
  return rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     newInitArgs);
}

LogicalResult PromotePointerIterArgsPattern::rewriteLoopBody(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  Block &oldBody = *oldForOp.getBody();
  Block &newBody = *newForOp.getBody();

  rewriter.setInsertionPointToStart(&newBody);

  // Create IR mapping that maps original values to their transformed
  // equivalents
  IRMapping mapping =
      createIRMapping(oldForOp, newForOp, pointerArgs, indexMap, rewriter);

  // Clone instructions from original loop body, applying the mapping
  return cloneInstructions(oldBody, newBody, pointerArgs, indexMap, mapping,
                           rewriter);
}

IRMapping PromotePointerIterArgsPattern::createIRMapping(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  IRMapping mapping;
  mapping.map(oldForOp.getInductionVar(), newForOp.getInductionVar());

  // Process iteration arguments
  for (unsigned i = 0; i < oldForOp.getRegionIterArgs().size(); ++i) {
    Value oldIterArg = oldForOp.getRegionIterArgs()[i];
    Value newIterArg = newForOp.getRegionIterArgs()[indexMap[i]];

    if (isPointerArgIndex(pointerArgs, i)) {
      // Update the PointerArgInfo with the new integer iteration argument
      for (auto &info : pointerArgs) {
        if (info.oldIndex == i) {
          info.newIterArg = newIterArg;
          break;
        }
      }

      // Map original pointer argument to a reconstructed pointer
      mapping.map(oldIterArg,
                  rebuildPointer(oldForOp, pointerArgs, i, rewriter));
    } else {
      // Direct mapping for non-pointer arguments
      mapping.map(oldIterArg, newIterArg);
    }
  }

  return mapping;
}

bool PromotePointerIterArgsPattern::isPointerArgIndex(
    ArrayRef<PointerArgInfo> pointerArgs, unsigned idx) const {
  for (auto &info : pointerArgs) {
    if (info.oldIndex == idx)
      return true;
  }
  return false;
}

Value PromotePointerIterArgsPattern::rebuildPointer(
    scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs, unsigned idx,
    PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;

  // Create splat operation to broadcast integer offset to tensor shape
  auto baseType = info->basePointer.getType();
  Value splatOffset = nullptr;
  if (auto rankedType = dyn_cast<RankedTensorType>(baseType)) {
    // Get the shape of the original tensor
    auto shape = rankedType.getShape();

    splatOffset = rewriter.create<triton::SplatOp>(
        forOp.getLoc(), RankedTensorType::get(shape, rewriter.getI32Type()),
        info->newIterArg);
  } else {
    return nullptr;
  }

  // Create addptr operation: base pointer + splatted offset
  return rewriter.create<triton::AddPtrOp>(forOp.getLoc(),
                                           info->basePointer.getType(),
                                           info->basePointer, splatOffset);
}

LogicalResult PromotePointerIterArgsPattern::cloneInstructions(
    Block &oldBody, Block &newBody, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  // Collect all operations from the old loop body except the terminator
  SmallVector<Operation *> toClone;
  for (auto &op : oldBody.without_terminator()) {
    toClone.push_back(&op);
  }

  // Build a set of addptr operations to skip (those that update pointer
  // iteration arguments)
  DenseSet<Value> addPtrOpsToSkip;
  for (const auto &info : pointerArgs) {
    if (info.addPtrValue) {
      addPtrOpsToSkip.insert(info.addPtrValue);
    }
  }

  // Clone all operations except the skipped addptr operations
  for (auto *op : toClone) {
    // Only skip addptr operations that are updating pointer iteration arguments
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      if (addPtrOpsToSkip.contains(addPtrOp.getResult())) {
        continue;
      }
    }
    rewriter.clone(*op, mapping);
  }

  // Handle the yield terminator separately
  auto yieldOp = dyn_cast<scf::YieldOp>(oldBody.getTerminator());
  if (!yieldOp) {
    return failure();
  }

  return cloneYieldOp(yieldOp, pointerArgs, indexMap, mapping, rewriter);
}

LogicalResult PromotePointerIterArgsPattern::cloneYieldOp(
    scf::YieldOp yieldOp, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  // Process each operand of the original yield operation
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      // For pointer arguments being promoted: create integer addition
      Value intResult = createIntegerAdd(i, pointerArgs, indexMap, rewriter);
      newOperands.push_back(intResult);
    } else {
      // For other arguments: use the value from the IR mapping
      newOperands.push_back(mapping.lookupOrDefault(yieldOp.getOperand(i)));
    }
  }

  // Validate that all new operands are non-null
  for (auto v : newOperands) {
    if (!v) {
      return failure();
    }
  }

  // Create the new yield operation in the transformed loop
  rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
  return success();
}

Value PromotePointerIterArgsPattern::createIntegerAdd(
    unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;

  // Try to extract constant offset value
  Attribute offsetAttr;
  if (matchPattern(info->offsetValue, m_Constant(&offsetAttr))) {
    Location loc = info->offsetValue.getLoc();

    // Case 1: Integer attribute (scalar constant)
    if (auto intAttr = dyn_cast<IntegerAttr>(offsetAttr)) {
      Value constOffset =
          rewriter.create<arith::ConstantIntOp>(loc, intAttr.getInt(), 32);
      return rewriter.create<arith::AddIOp>(loc, info->newIterArg, constOffset);
    }

    // Case 2: DenseElementsAttr (tensor constant)
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(offsetAttr)) {
      // Check if it's a splat (all elements are the same)
      if (denseAttr.isSplat()) {
        // For integer-type DenseElementsAttr
        if (denseAttr.getElementType().isInteger(32)) {
          auto splatValue = denseAttr.getSplatValue<APInt>();
          Value constOffset = rewriter.create<arith::ConstantIntOp>(
              loc, splatValue.getZExtValue(), 32);
          return rewriter.create<arith::AddIOp>(loc, info->newIterArg,
                                                constOffset);
        }
      } else {
        // If not a splat, but has only one element, we can still handle it
        if (denseAttr.getNumElements() == 1) {
          auto firstElement = *denseAttr.getValues<APInt>().begin();
          Value constOffset = rewriter.create<arith::ConstantIntOp>(
              loc, firstElement.getZExtValue(), 32);
          return rewriter.create<arith::AddIOp>(loc, info->newIterArg,
                                                constOffset);
        }
      }
    }
  }

  // Return nullptr if offset is not a constant (pattern only handles constant
  // offsets)
  return nullptr;
}

LogicalResult PromotePointerIterArgsPattern::replaceResults(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  SmallVector<Value> newResults;

  for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      Value ptrResult = reconstructPointer(
          oldForOp, i, newForOp.getResult(indexMap[i]), pointerArgs, rewriter);
      newResults.push_back(ptrResult);
    } else {
      newResults.push_back(newForOp.getResult(indexMap[i]));
    }
  }

  for (auto v : newResults) {
    if (!v) {
      return failure();
    }
  }
  rewriter.replaceOp(oldForOp, newResults);
  return success();
}

Value PromotePointerIterArgsPattern::reconstructPointer(
    scf::ForOp forOp, unsigned idx, Value intResult,
    ArrayRef<PointerArgInfo> pointerArgs, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;

  // Create splat operation to broadcast integer result to tensor shape
  auto baseType = info->basePointer.getType();
  Value splatOffset = nullptr;
  if (auto rankedType = dyn_cast<RankedTensorType>(baseType)) {
    // Get the shape of the original tensor
    auto shape = rankedType.getShape();

    splatOffset = rewriter.create<triton::SplatOp>(
        forOp.getLoc(), RankedTensorType::get(shape, rewriter.getI32Type()),
        intResult);
  } else {
    return nullptr;
  }

  // Create a tensor with the same shape, where all elements are the integer
  // result
  return rewriter.create<triton::AddPtrOp>(forOp.getLoc(),
                                           info->basePointer.getType(),
                                           info->basePointer, splatOffset);
}
} // namespace CannonicalizerConverter
