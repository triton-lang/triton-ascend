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

#include "TritonToStructured/TritonToStructuredPass.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToStructured/ImplicitPermute.h"
#include "TritonToStructured/MemOpConverter.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "triton-to-structured"

using namespace mlir;
using namespace triton;

namespace {

static LogicalResult matchExpandToTwos(RankedTensorType srcTy,
                                       RankedTensorType resTy, int64_t &numTwos,
                                       int64_t &flipDim) {
  if (resTy.getRank() <= srcTy.getRank())
    return failure(); // Expand only.
  auto reassoc = getReassociationIndicesForReshape(srcTy, resTy);
  if (!reassoc)
    return failure();
  ArrayRef<int64_t> resShape = resTy.getShape();
  for (int64_t d = 0; d < static_cast<int64_t>(reassoc->size()); ++d) {
    ArrayRef<int64_t> group = (*reassoc)[d];
    if (group.size() < 2)
      continue; // This source dim didn't split.
    bool allTwos = true;
    for (int64_t idx : group)
      if (resShape[idx] != 2) {
        allTwos = false;
        break;
      }
    if (allTwos) {
      flipDim = d;
      numTwos = static_cast<int64_t>(group.size());
      return success();
    }
  }
  return failure();
}

template <typename OpTy> static OpTy getSingleUser(Value value) {
  if (!value.hasOneUse())
    return OpTy();
  return dyn_cast<OpTy>(*value.user_begin());
}

static bool isXorReduce(triton::ReduceOp op) {
  if (!op->getNumRegions() || op->getRegion(0).getBlocks().size() != 1)
    return false;
  for (Operation &inner : op->getRegion(0).front().without_terminator())
    if (!isa<arith::XOrIOp>(inner))
      return false;
  return true;
}

static triton::ReduceOp findReduceXorUser(Value value, int64_t axis) {
  triton::ReduceOp found;
  for (auto *user : value.getUsers()) {
    auto reduce = dyn_cast<triton::ReduceOp>(user);
    if (reduce && static_cast<int64_t>(reduce.getAxis()) == axis &&
        isXorReduce(reduce)) {
      if (found)
        return triton::ReduceOp(); // Ambiguous: bail out.
      found = reduce;
    }
  }
  return found;
}

// `tl.flip` lowers to a bit-reversal idiom: reshape 2^k -> [2]^k, then a
// reduce-xor/expand_dims/broadcast/xori chain, then a collapse reshape. That
// first reshape becomes an unalignable memref.expand_shape (see #1638), so we
// recognize the idiom and rewrite it into the backend's existing (orphaned)
// native ascend.flip, which lowers via @triton_flip -> hfusion.flip ->
// hivm.vflip. vflip only flips the last (innermost) dim, so this pattern only
// fires for last-dim flips; any other dim keeps the bit-reversal idiom.
struct BitReversalFlipToAscendFlipPattern
    : public OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern<triton::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    int64_t numTwos = 0, flipDim = -1;
    if (failed(matchExpandToTwos(srcTy, resTy, numTwos, flipDim)))
      return failure();
    // hivm.vflip only flips the innermost dim; keep the bit-reversal idiom for
    // any other dim so non-last-dim tl.flip still compiles.
    if (flipDim != srcTy.getRank() - 1)
      return failure();

    auto srcBitcast = op.getSrc().getDefiningOp<triton::BitcastOp>();

    Value cur = op.getResult();
    SmallVector<Operation *> toErase{op};
    for (int64_t i = 0; i < numTwos; ++i) {
      int64_t axis = flipDim + i;
      auto reduce = findReduceXorUser(cur, axis);
      if (!reduce)
        return failure();
      auto expand = getSingleUser<triton::ExpandDimsOp>(reduce->getResult(0));
      if (!expand || static_cast<int64_t>(expand.getAxis()) != axis)
        return failure();
      auto broadcast = getSingleUser<triton::BroadcastOp>(expand->getResult(0));
      if (!broadcast)
        return failure();
      auto xori = getSingleUser<arith::XOrIOp>(broadcast->getResult(0));
      if (!xori)
        return failure();
      bool lhsCur = (xori.getLhs() == cur), rhsCur = (xori.getRhs() == cur);
      if (lhsCur == rhsCur)
        return failure(); // Exactly one operand == cur.
      cur = xori->getResult(0);
      toErase.push_back(reduce);
      toErase.push_back(expand);
      toErase.push_back(broadcast);
      toErase.push_back(xori);
    }

    auto finalReshape = getSingleUser<triton::ReshapeOp>(cur);
    if (!finalReshape || finalReshape.getResult().getType() != srcTy)
      return failure();

    auto dstBitcast =
        getSingleUser<triton::BitcastOp>(finalReshape.getResult());
    bool absorb = srcBitcast && dstBitcast;
    Value flipSrc = op.getSrc();
    RankedTensorType flipSrcTy = srcTy;
    if (absorb) {
      // Float path: build the flip on the original element type so its result
      // type matches the trailing bitcast's output (which we replace directly)
      flipSrc = srcBitcast.getSrc();
      flipSrcTy = cast<RankedTensorType>(flipSrc.getType());
    }

    rewriter.setInsertionPoint(op);
    auto flip = rewriter.create<triton::ascend::FlipOp>(
        op.getLoc(), flipSrcTy, flipSrc, static_cast<uint64_t>(flipDim));
    if (absorb) {
      rewriter.replaceOp(dstBitcast, flip.getResult());
      toErase.push_back(finalReshape);
      toErase.insert(toErase.begin(), srcBitcast);
    } else {
      rewriter.replaceOp(finalReshape, flip.getResult());
    }
    // Erase bottom-up (consumers before producers) since erasure is immediate.
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      rewriter.eraseOp(*it);
    return success();
  }
};

} // namespace

void TritonToStructuredPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, hivm::HIVMDialect,
                  annotation::AnnotationDialect,
                  triton::ascend::TritonAscendDialect>();
}

void TritonToStructuredPass::populateTritonToStructuredCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  // TODO enable this optimization after fixing the bisheng bug it causes in
  // current version
  // patterns.add<CannonicalizerConverter::CmpConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::PromotePointerIterArgsPattern>(
      patterns.getContext());
  patterns.add<CannonicalizerConverter::SimplifyTensorIterArgsPattern>(
      patterns.getContext());
  // Add addptr splat->broadcast hoisting converter
  patterns.add<CannonicalizerConverter::AddPtrSplatConverter>(
      patterns.getContext());
  // Move loads before broadcasts when safe
  patterns.add<CannonicalizerConverter::LoadBroadcastConverter>(
      patterns.getContext());
  // Move stores before broadcasts when safe
  patterns.add<CannonicalizerConverter::StoreBroadcastConverter>(
      patterns.getContext());
  // Lower `make_tensor_ptr` with all-zero strides to scalar load + broadcast.
  patterns.add<CannonicalizerConverter::ZeroStrideMakeTensorPtrConverter>(
      patterns.getContext());
  // Rewrite the frontend bit-reversal `tl.flip` idiom into `ascend.flip`.
  patterns.add<BitReversalFlipToAscendFlipPattern>(patterns.getContext());
}

void TritonToStructuredPass::populateTritonToStructuredPatterns(
    RewritePatternSet &patterns, bool optimizeDynamicOffset,
    bool enableMaskFallbackConversion) {
  patterns.add<MemOpConverter::LoadConverter>(patterns.getContext(),
                                              optimizeDynamicOffset,
                                              enableMaskFallbackConversion);
  patterns.add<MemOpConverter::StoreConverter>(patterns.getContext(),
                                               optimizeDynamicOffset,
                                               enableMaskFallbackConversion);
}

static bool moduleHasCubeOps(ModuleOp moduleOp) {
  bool existDot = false;
  moduleOp.walk([&](Operation *op) {
    if (isa<triton::DotOp, triton::DotScaledOp>(op)) {
      existDot = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return existDot;
}

LogicalResult
TritonToStructuredPass::processImplicitPermuteOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ImplicitPermute::LoadConverter>(
      patterns.getContext(), compileOn91095, moduleHasCubeOps(moduleOp),
      forceExplicitLoadTranspose);
  patterns.add<ImplicitPermute::StoreConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicRMWConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicCASConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::SplatCmpConverter>(
      patterns.getContext());

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    LLVM_DEBUG({ llvm::dbgs() << "ImplicitPermute: rewrite MemOp failed\n"; });
  }

  mlir::PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  return runPipeline(pm, getOperation());
}

LogicalResult
TritonToStructuredPass::processSplatBinaryOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CannonicalizerConverter::SplatCmpConverter>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp.emitWarning("Splat binary op processing failed");
    return failure();
  }
  return success();
}

void TritonToStructuredPass::runOnOperation() {
  auto moduleOp = getOperation();
  ConversionTarget target(getContext());
  RewritePatternSet canonicalizerPatterns(&getContext());

  this->populateTritonToStructuredCanonicalizationPatterns(
      canonicalizerPatterns);
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(canonicalizerPatterns)))) {
    moduleOp.emitWarning("Canonicalize failed");
  }

  RewritePatternSet tritonToStructuredPatterns(&getContext());
  populateTritonToStructuredPatterns(tritonToStructuredPatterns,
                                     optimizeDynamicOffset,
                                     enableMaskFallbackConversion);

  // Seed the worklist with memory ops while keeping newly generated operations
  // eligible for rewriting, including loads/stores created by cat splitting.
  SmallVector<Operation *> memoryOps;
  moduleOp.walk([&](Operation *op) {
    if (isa<triton::LoadOp, triton::StoreOp>(op))
      memoryOps.push_back(op);
  });
  FrozenRewritePatternSet memoryPatterns(std::move(tritonToStructuredPatterns));
  if (failed(applyOpPatternsGreedily(memoryOps, memoryPatterns))) {
    LLVM_DEBUG({ moduleOp->emitRemark("PtrAnalysis: rewrite MemOp failed"); });
  }

  if (failed(processImplicitPermuteOperations(moduleOp))) {
    LLVM_DEBUG(
        { llvm::dbgs() << "Failed to process implicit permute operations\n"; });
    signalPassFailure();
    return;
  }

  if (failed(processSplatBinaryOperations(moduleOp))) {
    moduleOp.emitWarning("Splat binary op processing failed");
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitWarning("Canonicalize failed");
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToStructuredPass(
    bool enableMaskFallbackConversion, bool optimizeDynamicOffset,
    bool compileOn91095, bool forceExplicitLoadTranspose) {
  return std::make_unique<TritonToStructuredPass>(
      enableMaskFallbackConversion, optimizeDynamicOffset, compileOn91095,
      forceExplicitLoadTranspose);
}
