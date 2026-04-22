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

#include "Utils/Utils.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "ascend/include/TritonToStructured/MemOpConverter.h"
#include "ascend/include/TritonToUnstructure/OffsetAnalysis.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "discrete-mask-access-conversion"

using namespace mlir;
using namespace hivm;

// File-scope flags set by DiscreteMaskAccessConversionPass::runOnOperation()
// before pattern application, so that OpRewritePattern subclasses can read
// them.
static bool compileOn91095Flag = false;
static bool forceSimtTemplateFlag = false;
static bool enableSyncBlockLockFlag = true;

LogicalResult isDiscreteMask(Operation *op, Value mask,
                             PatternRewriter &rewriter) {
  if (!mask)
    return failure();

  MaskState mstate;
  auto isContMask = mstate.parse(mask, op->getLoc(), rewriter);
  if (!isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }
  return success();
}

// Recursively collect all leaf operands of a nested arith::AndIOp tree.
// This function also normalizes masks by distributing broadcast over andi
//   broadcast(andi(a, b)) = andi(broadcast(a), broadcast(b))
// so that inner AND operands nested inside a broadcast are still reachable.
static void collectAndLeaves(Value mask, SmallVectorImpl<Value> &leaves,
                             Location loc, PatternRewriter &rewriter) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    collectAndLeaves(andOp.getLhs(), leaves, loc, rewriter);
    collectAndLeaves(andOp.getRhs(), leaves, loc, rewriter);
  } else if (auto broadcastOp = mask.getDefiningOp<triton::BroadcastOp>()) {
    // Distribute broadcast over andi so we can inspect each factor separately.
    if (auto innerAnd = broadcastOp.getSrc().getDefiningOp<arith::AndIOp>()) {
      Type dstType = mask.getType();
      Value broadcastA =
          rewriter.create<triton::BroadcastOp>(loc, dstType, innerAnd.getLhs())
              .getResult();
      Value broadcastB =
          rewriter.create<triton::BroadcastOp>(loc, dstType, innerAnd.getRhs())
              .getResult();
      collectAndLeaves(broadcastA, leaves, loc, rewriter);
      collectAndLeaves(broadcastB, leaves, loc, rewriter);
    } else {
      leaves.push_back(mask);
    }
  } else {
    leaves.push_back(mask);
  }
}

struct MaskDecomposition {
  // AND of all leaves that MaskState::parse() can analyze as a rectangle mask.
  // nullptr when no such leaves exist.
  Value contMask;
  // AND of all leaves that MaskState::parse() cannot analyze
  // (discrete/runtime). nullptr when no such leaves exist.
  Value discMask;
};

// Decompose an AND-tree mask into its continuous and discrete leaf components
// so that we can use contMask to bound GM accesses while discMask still drives
// the per-element selection.
static MaskDecomposition decomposeAndMask(Operation *op, Value mask,
                                          const Location &loc,
                                          PatternRewriter &rewriter) {
  SmallVector<Value> leaves;
  collectAndLeaves(mask, leaves, loc, rewriter);

  SmallVector<Value> contLeaves;
  SmallVector<Value> discLeaves;

  for (Value leaf : leaves) {
    MaskState st;
    if (st.parse(leaf, loc, rewriter).succeeded()) {
      if (st.isMask())
        contLeaves.push_back(leaf);
      else
        discLeaves.push_back(leaf);
    } else {
      discLeaves.push_back(leaf);
    }
  }

  Value contMask = nullptr;
  for (Value v : contLeaves)
    contMask =
        contMask ? rewriter.create<arith::AndIOp>(loc, contMask, v).getResult()
                 : v;

  Value discMask = nullptr;
  for (Value v : discLeaves)
    discMask =
        discMask ? rewriter.create<arith::AndIOp>(loc, discMask, v).getResult()
                 : v;

  return {contMask, discMask};
}

struct DiscreteMaskStoreConversion : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const final {
    auto mask = op.getMask();
    auto loc = op.getLoc();
    auto dst = op.getPtr();
    auto src = op.getValue();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    if (compileOn91095Flag && forceSimtTemplateFlag) {
      llvm::DenseMap<Value, PtrOffsetInfo> offsetMap;
      mlir::triton::parse(dst, loc, rewriter, offsetMap);

      if (!offsetMap.contains(dst)) {
        return failure();
      }

      auto &info = offsetMap[dst];
      Value basePtr = info.getPtr();
      Value totalOffset = info.getOffset();

      rewriter.create<triton::ascend::IndirectStoreOp>(loc, basePtr,
                                                       totalOffset, src, mask);
      rewriter.eraseOp(op);
      return success();
    }

    // When mask = contMask & discMask, use contMask to bound GM accesses and
    // discMask to select the final per-element value. This prevents the
    // unguarded full-load from reading past the tail-block boundary.
    auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
    if (contMask && discMask) {
      // insert sync_block_lock
      auto lockVar = MemOpConverter::createSyncBlockLockVar(rewriter, loc);
      if (enableSyncBlockLockFlag) {
        rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
      }
      auto safeLoad = rewriter.create<triton::LoadOp>(
          loc, dst, contMask, op.getCache(), op.getEvict(), false);
      auto selOp = rewriter.create<arith::SelectOp>(loc, discMask, src,
                                                    safeLoad.getResult());
      auto newStore = rewriter.create<triton::StoreOp>(
          loc, dst, selOp, contMask, op.getCache(), op.getEvict());
      newStore->setAttr(ConverterUtils::discreteMaskAttrName,
                        UnitAttr::get(rewriter.getContext()));
      if (enableSyncBlockLockFlag) {
        rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
      }
      rewriter.replaceOp(op, newStore);
      return success();
    }

    // Fallback: original full load + select (contMask absent, pure discrete).
    // insert sync_block_lock
    auto lockVar = MemOpConverter::createSyncBlockLockVar(rewriter, loc);
    if (enableSyncBlockLockFlag) {
      rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    }
    auto loadFromDstOp = rewriter.create<triton::LoadOp>(
        loc, dst, op.getCache(), op.getEvict(), false);
    auto selOp = rewriter.create<arith::SelectOp>(loc, mask, src,
                                                  loadFromDstOp.getResult());
    auto newStore = rewriter.create<triton::StoreOp>(
        loc, dst, selOp, op.getCache(), op.getEvict());
    newStore->setAttr(ConverterUtils::discreteMaskAttrName,
                      UnitAttr::get(rewriter.getContext()));
    if (enableSyncBlockLockFlag) {
      rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    }
    rewriter.replaceOp(op, newStore);
    return success();
  }
};

struct DiscreteMaskLoadConversion : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto other = op.getOther();
    auto mask = op.getMask();
    auto ptr = op.getPtr();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    const std::string isDiscreteMaskTag = "is_discrete_mask";
    op->setAttr(isDiscreteMaskTag, rewriter.getUnitAttr());

    if (compileOn91095Flag && forceSimtTemplateFlag)
      return failure();

    // When mask = contMask & discMask, load only the safe range defined by
    // contMask and use discMask for the per-element select, avoiding OOB reads.
    auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
    if (contMask && discMask) {
      if (!other) {
        FailureOr<Value> constant = specializeTypelessValueToConstant(
            TypelessValue::Zero, ptr.getType(), loc, rewriter);
        if (failed(constant)) {
          llvm_unreachable("Unsupported type for constant creation");
        }
        other = *constant;
      }
      auto safeLoad = rewriter.create<triton::LoadOp>(
          loc, ptr, contMask, op.getCache(), op.getEvict(), op.getIsVolatile());
      // Use combined mask to select the result, avoid the uninitialized memory
      // access.
      auto combinedMask =
          rewriter.create<arith::AndIOp>(loc, contMask, discMask);
      auto discreteMaskOp = rewriter.create<arith::SelectOp>(
          loc, combinedMask, safeLoad.getResult(), other);
      rewriter.replaceOp(op, discreteMaskOp);
      return success();
    }

    // Fallback: original full load + select (contMask absent, pure discrete).
    if (!other) {
      FailureOr<Value> constant = specializeTypelessValueToConstant(
          TypelessValue::Zero, ptr.getType(), loc, rewriter);
      if (failed(constant))
        llvm_unreachable("Unsupported type for constant creation");
      other = *constant;
    }

    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, ptr, op.getCache(), op.getEvict(), op.getIsVolatile());
    auto discreteMaskOp =
        rewriter.create<arith::SelectOp>(loc, mask, newLoadOp, other);
    rewriter.replaceOp(op, discreteMaskOp);
    return success();
  }
};

struct DiscreteMaskAtomicConversion
    : OpRewritePattern<mlir::triton::AtomicRMWOp> {
  using OpRewritePattern<mlir::triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto ptr = op.getPtr();
    auto src = op.getVal();
    auto mask = op.getMask();
    RMWOp rmwOp = op.getAtomicRmwOp();

    if (failed(isDiscreteMask(op, mask, rewriter)))
      return failure();

    const std::map<RMWOp, TypelessValue> initMap = {
        {RMWOp::FADD, TypelessValue::Zero},
        {RMWOp::ADD, TypelessValue::Zero},
        {RMWOp::UMAX, TypelessValue::Zero},
        {RMWOp::OR, TypelessValue::Zero},
        {RMWOp::MIN, TypelessValue::Max},
        {RMWOp::UMIN, TypelessValue::Max},
        {RMWOp::AND, TypelessValue::Max},
        {RMWOp::MAX, TypelessValue::Min},
        {RMWOp::XOR, TypelessValue::Zero},
        {RMWOp::XCHG, TypelessValue::Undefined},
    };
    assert(initMap.find(rmwOp) != initMap.end());
    auto typelessVal = initMap.at(rmwOp);
    if (typelessVal == TypelessValue::Undefined) {
      // Undefined default value atomic op will be decomposed in AscendNPU-IR
      op->setAttr(ConverterUtils::discreteMaskAttrName,
                  UnitAttr::get(rewriter.getContext()));
      return failure();
    }

    FailureOr<mlir::Value> fill = specializeTypelessValueToConstant(
        typelessVal, src.getType(), loc, rewriter);
    if (failed(fill))
      op->emitError("Unsupported atomic operation.");

    auto maskedValue = rewriter.create<arith::SelectOp>(loc, mask, src, *fill);
    auto newAtomicOp = rewriter.create<mlir::triton::AtomicRMWOp>(
        loc, src.getType(), rmwOp, ptr, maskedValue, mlir::Value(), op.getSem(),
        op.getScope());
    rewriter.replaceOp(op, newAtomicOp);
    return success();
  }
};

DiscreteMaskAccessConversionPass::DiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options)
    : DiscreteMaskAccessConversionBase(options) {}

void DiscreteMaskAccessConversionPass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  forceSimtTemplateFlag = this->forceSimtTemplate;
  enableSyncBlockLockFlag = this->enableSyncBlockLock;
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<DiscreteMaskLoadConversion, DiscreteMaskStoreConversion,
               DiscreteMaskAtomicConversion>(patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply discrete mask access patterns");
    signalPassFailure();
  }

  // Clean up dead analysis ops left behind by MaskState::parse().
  // These are trivially-dead auxiliary ops (constants, arithmetic) with no
  // users that parse() creates as side effects of mask analysis.
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    moduleOp->emitWarning(
        "DiscreteMaskAccessConversion: dead-code cleanup failed");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "==============================================\n";
    llvm::dbgs() << "After DiscreteMaskAccessConversionPass:\n" << moduleOp;
    llvm::dbgs() << "\n==============================================\n";
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDiscreteMaskAccessConversionPass(
    const DiscreteMaskAccessConversionOptions &options) {
  return std::make_unique<DiscreteMaskAccessConversionPass>(options);
}
