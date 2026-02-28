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
#ifndef TRITON_ADAPTER_CANNONICALIZERCONVERTER_H
#define TRITON_ADAPTER_CANNONICALIZERCONVERTER_H

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace CannonicalizerConverter {

using namespace mlir;
using namespace triton;

class CmpConverter : public OpRewritePattern<arith::CmpIOp> {
public:
  explicit CmpConverter(MLIRContext *context)
      : OpRewritePattern<arith::CmpIOp>(context) {}

  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override;
};

class SplatCmpConverter : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class PromotePointerIterArgsPattern : public OpRewritePattern<scf::ForOp> {
public:
  explicit PromotePointerIterArgsPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;

private:
  // Information about a pointer iteration argument to be promoted
  struct PointerArgInfo {
    unsigned oldIndex; // Original index in the iteration arguments
    Value basePointer; // Base pointer value passed as init arg
    Value offsetValue; // Offset value used in addptr operation
    Value newIterArg;  // New integer iteration argument
    Value addPtrValue; // The addptr operation result that updates the pointer
  };

  // Check if the loop meets basic transformation conditions
  LogicalResult matchLoop(scf::ForOp forOp) const;

  // Collect all pointer iteration arguments that match the promotion pattern
  SmallVector<PointerArgInfo> collectPointerIterArgs(scf::ForOp forOp) const;

  // Check if a value has pointer tensor type
  bool isPointerIterArg(Value iterArg) const;

  // Analyze a pointer iteration argument to determine if it matches the
  // promotion pattern
  std::optional<PointerArgInfo> analyzePointerIterArg(Value iterArg,
                                                      Block &loopBody) const;

  // Check if an index corresponds to a pointer argument being promoted
  bool isPointerArgIndex(ArrayRef<PointerArgInfo> pointerArgs,
                         unsigned idx) const;

  // Get pointer argument information for a specific index
  const PointerArgInfo *getPointerArgInfo(ArrayRef<PointerArgInfo> pointerArgs,
                                          unsigned idx) const;

  // Create a new for loop with updated iteration argument types
  scf::ForOp createNewForLoop(scf::ForOp forOp, ArrayRef<Value> newInitArgs,
                              ArrayRef<Type> newIterArgTypes,
                              PatternRewriter &rewriter) const;

  // Rewrite the loop body to use integer iteration arguments instead of
  // pointers
  LogicalResult rewriteLoopBody(scf::ForOp oldForOp, scf::ForOp newForOp,
                                SmallVector<PointerArgInfo> &pointerArgs,
                                DenseMap<unsigned, unsigned> &indexMap,
                                PatternRewriter &rewriter) const;

  // Create new iteration arguments by replacing pointers with integer offsets
  std::tuple<SmallVector<Value>, SmallVector<Type>,
             DenseMap<unsigned, unsigned>>
  createNewIterArgs(scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
                    PatternRewriter &rewriter) const;

  // Create IR mapping for cloning operations, rebuilding pointers from integer
  // offsets
  IRMapping createIRMapping(scf::ForOp oldForOp, scf::ForOp newForOp,
                            SmallVector<PointerArgInfo> &pointerArgs,
                            DenseMap<unsigned, unsigned> &indexMap,
                            PatternRewriter &rewriter) const;

  // Reconstruct a pointer value from base pointer and integer offset
  Value rebuildPointer(scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
                       unsigned idx, PatternRewriter &rewriter) const;

  // Clone instructions from old loop body to new loop body, skipping
  // transformed addptr ops
  LogicalResult cloneInstructions(Block &oldBody, Block &newBody,
                                  ArrayRef<PointerArgInfo> pointerArgs,
                                  DenseMap<unsigned, unsigned> &indexMap,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter) const;

  // Clone and transform the yield operation, converting pointer updates to
  // integer additions
  LogicalResult cloneYieldOp(scf::YieldOp yieldOp,
                             ArrayRef<PointerArgInfo> pointerArgs,
                             DenseMap<unsigned, unsigned> &indexMap,
                             IRMapping &mapping,
                             PatternRewriter &rewriter) const;

  // Create integer addition for pointer offset updates in the yield operation
  Value createIntegerAdd(unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
                         DenseMap<unsigned, unsigned> &indexMap,
                         PatternRewriter &rewriter) const;

  // Extract constant integer value from offset (handles both scalar and tensor
  // constants)
  std::optional<int64_t> extractConstantOffset(Value offsetValue) const;

  // Replace the original loop results with reconstructed pointers from integer
  // results
  LogicalResult replaceResults(scf::ForOp oldForOp, scf::ForOp newForOp,
                               ArrayRef<PointerArgInfo> pointerArgs,
                               DenseMap<unsigned, unsigned> &indexMap,
                               PatternRewriter &rewriter) const;

  // Reconstruct final pointer from integer result after the loop
  Value reconstructPointer(scf::ForOp forOp, unsigned idx, Value intResult,
                           ArrayRef<PointerArgInfo> pointerArgs,
                           PatternRewriter &rewriter) const;
};

} // namespace CannonicalizerConverter

#endif
