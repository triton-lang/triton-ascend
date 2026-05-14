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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"

#include "Utils/Utils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <optional>

using namespace mlir;
#define DEBUG_TYPE "inter-core-transfer-and-sync"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

using namespace mlir::triton;
using namespace hivm;

static uint64_t getElemBytesForAlign(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return (uint64_t)((ft.getWidth() + 7) / 8);
  if (auto it = dyn_cast<IntegerType>(t))
    return (uint64_t)((it.getWidth() + 7) / 8);
  if (isa<IndexType>(t))
    return 8ULL;
  if (auto ct = dyn_cast<ComplexType>(t))
    return 2ULL * getElemBytesForAlign(ct.getElementType());
  return 0ULL;
}

static uint64_t getBlockElemsFor32BAlign(Type elemType) {
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes <= 0)
    return -1;
  if (elemBytes >= kAlignBytes)
    return 1;
  if (kAlignBytes % elemBytes != 0)
    return -1;
  return kAlignBytes / elemBytes;
}

static void attachCommonTags(Operation *op, int blockId, StringRef coreType) {
  MLIRContext* ctx = op->getContext();
  op->setAttr("ssbuffer.block_id", IntegerAttr::get(IntegerType::get(ctx, 32), blockId));
  op->setAttr("ssbuffer.core_type", StringAttr::get(ctx, coreType));
}

static void attachTransferTags(Operation *op, int blockId, StringRef coreType, int transferId) {
  MLIRContext* ctx = op->getContext();
  op->setAttr("ssbuffer.block_id", IntegerAttr::get(IntegerType::get(ctx, 32), blockId));
  op->setAttr("ssbuffer.core_type", StringAttr::get(ctx, coreType));
  op->setAttr("ssbuffer.transfer_id", IntegerAttr::get(IntegerType::get(ctx, 32), transferId));
}

// Block Start/End Operation Retrieval
std::pair<mlir::Operation*, mlir::Operation*>
InterCoreTransferAndSyncPass::getBlockStartEnd(int targetId, mlir::ModuleOp module) {
  mlir::Operation* knownOpInBlock = nullptr;
  module.walk([&](mlir::Operation* op) {
    if (knownOpInBlock) {
      return;
    }
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("ssbuffer.block_id");
    if (attr && attr.getInt() == targetId) {
      knownOpInBlock = op;
    }
  });
  if (!knownOpInBlock) {
    return {nullptr, nullptr};
  }

  mlir::Block* block = knownOpInBlock->getBlock();
  if (!block) return {nullptr, nullptr};

  mlir::Operation* start = nullptr;
  mlir::Operation* end = nullptr;

  // Iterate through all operations in the current block
  for (auto& op : *block) {
    auto attr = op.getAttrOfType<mlir::IntegerAttr>("ssbuffer.block_id");
    if (!attr) {
      continue;
    }
    int blockId = attr.getInt();
    if (!start) {
      if (targetId == blockId) {
        start = &op;
        end = &op;
      }
    } else {
      if (targetId == blockId) {
        end = &op;
      } else {
        break;
      }
    }
  }
  return {start, end};
}

bool InterCoreTransferAndSyncPass::isOuterLayerDependency(
    size_t depIndex,
    mlir::Operation* currProdEnd,
    mlir::Operation* currConsStart,
    llvm::SmallVector<DependencyInfo>& memDependencies) {
  if (!currProdEnd || !currConsStart) {
    return false;
  }
  mlir::Block* currBlock = currProdEnd->getBlock();
  if (currBlock != currConsStart->getBlock()) {
    return false;
  }
  for (size_t i = 0; i < memDependencies.size(); ++i) {
    if (i == depIndex) {
      continue;
    }
    auto& otherDep = memDependencies[i];

    auto [otherProdStart, otherProdEnd] = getBlockStartEnd(otherDep.producerBlockId, module);
    auto [otherConsStart, otherConsEnd] = getBlockStartEnd(otherDep.consumerBlockId, module);

    if (!otherProdEnd || !otherConsStart) {
      continue;
    }

    if (otherProdEnd->getBlock() != currBlock || otherConsStart->getBlock() != currBlock) {
      continue;
    }

    // 情况1：内层嵌套判断
    // 满足：
    // otherProdEnd 不在 currProdEnd 前面（后 或 相等）
    // otherConsStart 不在 currConsStart 后面（前 或 相等）
    bool isOtherInsideCurrent =
        !otherProdEnd->isBeforeInBlock(currProdEnd) &&
        !currConsStart->isBeforeInBlock(otherConsStart);

    // 情况2：完全相同范围 → 只保留 index 最小的
    if (otherProdEnd == currProdEnd && otherConsStart == currConsStart) {
      if (i < depIndex) {
        return true; // 更小index存在 → 当前跳过
      }
    } else if (isOtherInsideCurrent) {
      return true;
    }
  }

  return false;
}


// Nd2NzNormalizer
SmallVector<int64_t, 2>
InterCoreTransferAndSyncPass::computeExpectedShape(Value value) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  int64_t M = tensorTy.getDimSize(0);
  int64_t N = tensorTy.getDimSize(1);

  // Compute bit width & Nwidth
  int64_t nWidth = getBlockElemsFor32BAlign(tensorTy.getElementType());

  // Calculate newM / newN using the formula
  int64_t blM = (M + 15) / 16;
  int64_t newM = blM * 16;

  int64_t blN = (N + nWidth - 1) / nWidth;
  int64_t newN = blN * nWidth;

  return {newM, newN}; // Return 2D shape
}

bool InterCoreTransferAndSyncPass::isShapeExpected(Value value, SmallVector<int64_t, 2>& expectedShape) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  ArrayRef<int64_t> currShape = tensorTy.getShape();
  return currShape.equals(expectedShape);
}

void InterCoreTransferAndSyncPass::rewriteMatmulWithNewShape(
    OpBuilder &builder,
    Operation* matmulOp,
    Location loc) {
  auto cidAttr = matmulOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int matmulOpBlockId = cidAttr.getInt();

  Value lhs = matmulOp->getOperands()[0];
  Value rhs = matmulOp->getOperands()[1];
  Value acc = matmulOp->getOperands()[2];
  Value originalResult = matmulOp->getResult(0);
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  auto accType = dyn_cast<RankedTensorType>(acc.getType());
  auto resType = dyn_cast<RankedTensorType>(originalResult.getType());
  ArrayRef<int64_t> accshape = accType.getShape();
  ArrayRef<int64_t> resshape = resType.getShape();
  SmallVector<int64_t, 2> expectedShape = {lhsType.getShape()[0], rhsType.getShape()[1]};
  auto expectedType = RankedTensorType::get(expectedShape, resType.getElementType());

  builder.setInsertionPoint(matmulOp);

  auto floatElemTy = cast<FloatType>(resType.getElementType());
  auto zeroConstOp = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(floatElemTy.getFloatSemantics()), floatElemTy);
  auto tensorEmptyOp = builder.create<tensor::EmptyOp>(
      loc, expectedShape, resType.getElementType());
  auto linalgFillOp = builder.create<linalg::FillOp>(
      loc, zeroConstOp.getResult(), tensorEmptyOp.getResult());

  attachCommonTags(zeroConstOp, matmulOpBlockId, "CUBE");
  attachCommonTags(tensorEmptyOp, matmulOpBlockId, "CUBE");
  attachCommonTags(linalgFillOp, matmulOpBlockId, "CUBE");

  Value newAccResult = linalgFillOp->getResult(0);

  builder.setInsertionPointAfter(matmulOp);
  matmulOp->setOperand(2, newAccResult);
  matmulOp->getResult(0).setType(expectedType);
  auto newMatmulOp = dyn_cast<linalg::MatmulOp>(matmulOp);
  Value newMatmulResult = newMatmulOp->getResult(0);
  LOG_DEBUG("newmatmulOp" << newMatmulOp << "\n");
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0), builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(accType.getShape()[0]),
                                      builder.getIndexAttr(accType.getShape()[1])};
  auto extractSliceOp = builder.create<tensor::ExtractSliceOp>(
      loc,
      newMatmulResult,
      offsets,
      sizes,
      strides
  );
  attachCommonTags(extractSliceOp, matmulOpBlockId, "CUBE");

  originalResult.replaceUsesWithIf(
      extractSliceOp.getResult(),
      [&](OpOperand &use) { return use.getOwner() != extractSliceOp.getOperation();
  });
  LOG_DEBUG("cubeValueMapping[originalResult]" << originalResult << "\n");
  LOG_DEBUG("cubeValueMapping[originalResult]extractSliceOp.getResult()   " << extractSliceOp.getResult() << "\n");
  cubeValueMapping[originalResult] = extractSliceOp.getResult();
}

void InterCoreTransferAndSyncPass::rewriteTransposeWithNewShape(
    OpBuilder &builder,
    Operation* transposeOp,
    Location loc) {
  Value inputvalue = transposeOp->getOperands()[0];
  Value outputvalue = transposeOp->getOperands()[0];

  auto inputTy = dyn_cast<RankedTensorType>(inputvalue.getType());
  Type elemType = inputTy.getElementType();
  SmallVector<int64_t, 2> newOutputShape = {inputTy.getShape()[1], inputTy.getShape()[0]};
  auto expectedType = RankedTensorType::get(newOutputShape, elemType);
  // Create new empty tensor with new shape
  auto tensorEmptyOp = builder.create<tensor::EmptyOp>(
      loc, newOutputShape, elemType);
  attachCommonTags(tensorEmptyOp, transposeOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id").getInt(), "CUBE");
  Value transposeOpResult = transposeOp->getResult(0);
  transposeOp->setOperand(1, tensorEmptyOp.getResult());
  transposeOp->getResult(0).setType(expectedType);
  transposeOpResult.replaceAllUsesWith(transposeOp->getResult(0));
}

mlir::Value InterCoreTransferAndSyncPass::normalizeIfNeeded(OpBuilder &builder, DependencyInfo& dep, Location loc,
                                               mlir::Value origValue, SmallVector<int64_t, 2> expectedShape, int originBlockId) {
  auto origTensorType = dyn_cast<RankedTensorType>(origValue.getType());

  int64_t iniM = origTensorType.getDimSize(0);
  int64_t iniN = origTensorType.getDimSize(1);
  Type elemType = origTensorType.getElementType();

  builder.setInsertionPointAfter(origValue.getDefiningOp());

  auto floatElemTy = cast<FloatType>(elemType);
  auto zeroConstOp = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(floatElemTy.getFloatSemantics()), floatElemTy);
  auto tensorEmptyOp = builder.create<tensor::EmptyOp>(
      loc, expectedShape, elemType);
  auto linalgFillOp = builder.create<linalg::FillOp>(
      loc, zeroConstOp.getResult(), tensorEmptyOp.getResult());
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0),builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> insertsizes = {builder.getIndexAttr(iniM),builder.getIndexAttr(iniN)};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),builder.getIndexAttr(1)};
  auto tensorInsertSliceOp = builder.create<tensor::InsertSliceOp>(
      loc,
      origValue,
      linalgFillOp->getResult(0),
      offsets,
      insertsizes,
      strides);

  attachCommonTags(zeroConstOp, originBlockId, "VECTOR");
  attachCommonTags(tensorEmptyOp, originBlockId, "VECTOR");
  attachCommonTags(linalgFillOp, originBlockId, "VECTOR");
  attachCommonTags(tensorInsertSliceOp, originBlockId, "VECTOR");
  int cId = dep.iniConsumerBlockId;
  for (Operation *user : origValue.getUsers()) {
    if (auto blockIdAttr = user->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      int userBlockId = blockIdAttr.getInt();
      if (userBlockId == cId) {
        user->replaceUsesOfWith(origValue, tensorInsertSliceOp.getResult());
        if (auto matmulOp = dyn_cast<linalg::MatmulOp>(user)) {
          rewriteMatmulWithNewShape(builder, matmulOp, loc);
        } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(user)) {
          rewriteTransposeWithNewShape(builder, transposeOp, loc);
          rewriteMatmulWithNewShape(builder, matmulOp, loc);
        }
      }
    }
  }
  cubeValueMapping[origValue] = tensorInsertSliceOp.getResult();
  return tensorInsertSliceOp.getResult();
}

void InterCoreTransferAndSyncPass::Nd2NzNormalize(OpBuilder &builder, DependencyInfo& dep, Location loc) {
  Value origValue = dep.value;
  Value newValue = origValue;
  // Step 0: Check if this Value has already been processed
  auto it = vecValueMapping.find(origValue);
  if (it == vecValueMapping.end()) {
    // Step 1: Compute expected shape
    SmallVector<int64_t, 2> expectedShape = computeExpectedShape(origValue);
    auto vidAttr = origValue.getDefiningOp()->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    int originBlockId = vidAttr.getInt();
    // Step 2: If shapes match, return original value
    if (!isShapeExpected(origValue, expectedShape)) {
      newValue = normalizeIfNeeded(builder, dep, loc, origValue, expectedShape, originBlockId);
    }
    auto srcTensorType = cast<RankedTensorType>(newValue.getType());
    int64_t M = srcTensorType.getDimSize(0);
    int64_t N = srcTensorType.getDimSize(1);
    Type elemType = srcTensorType.getElementType();

    int64_t blk = getBlockElemsFor32BAlign(elemType);

    SmallVector<int64_t, 3> shape3D = {M, N / blk, blk};
    // transpose: 64x8x8 -> 8x64x8
    SmallVector<int64_t, 3> shapeTrans = {N / blk, M, blk};
    // reshape2: 8x64x8 -> 8x4x16x8
    SmallVector<int64_t, 4> shapeFinal = {N/blk, M/16, 16, blk};

    auto type3D    = RankedTensorType::get(shape3D, elemType);
    auto typeTrans = RankedTensorType::get(shapeTrans, elemType);
    auto typeFinal = RankedTensorType::get(shapeFinal, elemType);
    builder.setInsertionPointAfter(newValue.getDefiningOp());
    auto reshape3Dcst = builder.create<arith::ConstantOp>(
        loc, builder.getI64TensorAttr(shape3D));
    auto reshape3DOp = builder.create<tensor::ReshapeOp>(
        loc, type3D, newValue, reshape3Dcst );
    // auto markOp3d = builder.create<annotation::MarkOp>(loc, reshape3DOp);
    // auto tilingDimAttr3d = builder.getDictionaryAttr(SmallVector<NamedAttribute>{
    //     NamedAttribute(builder.getStringAttr("1"), builder.getIndexAttr(1))});
    // markOp3d->setAttr("tiling_dim_mapping", tilingDimAttr3d);

    auto emptyTrans = builder.create<tensor::EmptyOp>(loc, shapeTrans, elemType);
    SmallVector<int64_t, 4> order = {1, 0, 2};
    auto transposeOp = builder.create<linalg::TransposeOp>(
        loc,
        reshape3DOp.getResult(),
        emptyTrans.getResult(),
        order);
    auto reshape4Dcst = builder.create<arith::ConstantOp>(
        loc, builder.getI64TensorAttr(shapeFinal));
    auto reshape4DOp = builder.create<tensor::ReshapeOp>(
        loc, typeFinal, transposeOp->getResult(0), reshape4Dcst);
    // auto markOp4d = builder.create<annotation::MarkOp>(loc, reshape4DOp);
    // auto tilingDimAttr4d = builder.getDictionaryAttr(SmallVector<NamedAttribute>{
    // NamedAttribute(builder.getStringAttr("1"), builder.getIndexAttr(1))});
    // markOp4d->setAttr("tiling_dim_mapping", tilingDimAttr4d);

    attachCommonTags(reshape3Dcst, originBlockId, "VECTOR");
    attachCommonTags(reshape3DOp, originBlockId, "VECTOR");
    // attachCommonTags(markOp3d, originBlockId, "VECTOR");
    attachCommonTags(emptyTrans, originBlockId, "VECTOR");
    attachCommonTags(transposeOp, originBlockId, "VECTOR");
    attachCommonTags(reshape4Dcst, originBlockId, "VECTOR");
    attachCommonTags(reshape4DOp, originBlockId, "VECTOR");
    // attachCommonTags(markOp4d, originBlockId, "VECTOR");
    LOG_DEBUG("[reshape3Dcst]: " << *reshape3Dcst << "\n");
    LOG_DEBUG("[reshape3DOp]: " << *reshape3DOp << "\n");
    // LOG_DEBUG("[markOp3d]: " << *markOp3d << "\n");
    LOG_DEBUG("[emptyTrans]: " << emptyTrans << "\n");
    LOG_DEBUG("[transposeOp]: " << *transposeOp << "\n");
    LOG_DEBUG("[reshape4Dcst]: " << *reshape4Dcst << "\n");
    LOG_DEBUG("[reshape4DOp]: " << *reshape4DOp << "\n");
    // LOG_DEBUG("[markOp4d]: " << *markOp4d << "\n");
    vecValueMapping[origValue] = reshape4DOp.getResult();
  }
}

// TransferExecutor
mlir::Operation* InterCoreTransferAndSyncPass::annotateTightlyCoupledBuffer(OpBuilder &builder, Operation* allocOp, Location loc) {
  builder.setInsertionPointAfter(allocOp);
  auto markAllocOp = builder.create<annotation::MarkOp>(
      loc, allocOp->getResult(0));
  auto writeAttr = builder.getStringAttr("write");
  auto readAttr  = builder.getStringAttr("read");
  auto effectsAttr = builder.getArrayAttr({writeAttr, readAttr});
  markAllocOp->setAttr("effects", effectsAttr);
  markAllocOp->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
                HIVMTightlyCoupledBufferAttr::get(builder.getContext(),
                                                  markAllocIndex));
  return markAllocOp;
}

Operation* InterCoreTransferAndSyncPass::findMainLoopforTransfer(Operation* endOp, Operation* startOp) {
  Operation* lca = endOp->getParentOp();
  assert(lca == startOp->getParentOp() &&
         "startOp and endOp are not in the same parent block, which is unexpected.");
  Operation* current = lca;
  while (current) {
    if (isa<scf::ForOp>(current)) {
      return current;
    }
    current = current->getParentOp();
  }
  return nullptr;
}

Operation* InterCoreTransferAndSyncPass::insertVectorToCubeTransfer(OpBuilder &builder, Value srcValue, Value normalizedValue, Operation* vectorEndOp, Operation* cubeStartOp, Location loc, int transferIndex, int iniConsumerId) {
  LOG_DEBUG("Inserting [Vector->Cube] transfer for value: " << srcValue << "\n");
  // Step 1: Get input information (2D tensor: MxN)
  auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
  auto normalizedTensorType = cast<RankedTensorType>(normalizedValue.getType());
  Type elemType = srcTensorType.getElementType();

  auto vidAttr = vectorEndOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int vecBlockId = vidAttr.getInt();
  auto cidAttr = cubeStartOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int cubeBlockId = cidAttr.getInt();

  auto cbufaddressSpaceAttr = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
  auto allocType = MemRefType::get(
      normalizedTensorType.getShape(),
      elemType,
      /*layout=*/nullptr,
      cbufaddressSpaceAttr
  );
  Operation* vecAllocOp;
  Operation* cubeAllocOp;
  // Allocate memory outside mainloop or inside func for data transfer/reception
  Operation* mainLoopOp = findMainLoopforTransfer(vectorEndOp, cubeStartOp);
  if (mainLoopOp) {
    builder.setInsertionPoint(mainLoopOp);
    vecAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markVecAllocOp = annotateTightlyCoupledBuffer(builder, vecAllocOp, loc);
    cubeAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markCubeAllocOp = annotateTightlyCoupledBuffer(builder, cubeAllocOp, loc);
    auto loopidAttr = mainLoopOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    int loopBlockId = loopidAttr.getInt();
    attachTransferTags(vecAllocOp, loopBlockId, "VECTOR", transferIndex);
    attachTransferTags(cubeAllocOp, loopBlockId, "CUBE", transferIndex);
    attachTransferTags(markVecAllocOp, loopBlockId, "VECTOR", transferIndex);
    attachTransferTags(markCubeAllocOp, loopBlockId, "CUBE", transferIndex);
    builder.setInsertionPointAfter(vectorEndOp);
  } else {
    builder.setInsertionPoint(cubeStartOp);
    cubeAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markCubeAllocOp = annotateTightlyCoupledBuffer(builder, cubeAllocOp, loc);
    builder.setInsertionPointAfter(vectorEndOp);
    vecAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markVecAllocOp = annotateTightlyCoupledBuffer(builder, vecAllocOp, loc);
    attachTransferTags(vecAllocOp, vecBlockId, "VECTOR", transferIndex);
    attachTransferTags(cubeAllocOp, cubeBlockId, "CUBE", transferIndex);
    attachTransferTags(markVecAllocOp, vecBlockId, "VECTOR", transferIndex);
    attachTransferTags(markCubeAllocOp, cubeBlockId, "CUBE", transferIndex);
  }
  markAllocIndex++;

  auto copyOp = builder.create<hivm::CopyOp>(
      loc,
      mlir::TypeRange{},
      normalizedValue,
      vecAllocOp->getResult(0));

  attachTransferTags(copyOp, vecBlockId, "VECTOR", transferIndex);

  LOG_DEBUG("[copyOp]: " << *copyOp << "\n");

  builder.setInsertionPoint(cubeStartOp);

  auto nzLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::nZ);
  auto ndLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
  auto newAllocType = MemRefType::get(
      srcTensorType.getShape(),
      elemType,
      /*layout=*/nullptr,
      cbufaddressSpaceAttr
  );
  auto convertLayoutOp = builder.create<hivm::ConvertLayoutOp>(
      loc,
      newAllocType,
      cubeAllocOp->getResult(0),
      /*srcLayout=*/nzLayout,  // srcLayout
      /*dstLayout=*/ndLayout   // dstLayout
  );
  auto plainMemrefType = MemRefType::get(srcTensorType.getShape(), elemType);
  auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
      loc,
      plainMemrefType,
      convertLayoutOp.getResult()
  );
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc,
      srcTensorType,
      memspaceCastOp.getResult(),
      /*restrict=*/true,
      /*writable=*/true
  );

  attachTransferTags(convertLayoutOp, cubeBlockId, "CUBE", transferIndex);
  attachTransferTags(memspaceCastOp, cubeBlockId, "CUBE", transferIndex);
  attachTransferTags(toTensorOp, cubeBlockId, "CUBE", transferIndex);
  LOG_DEBUG("[convertLayoutOp]: " << *convertLayoutOp << "\n");
  LOG_DEBUG("[memspaceCastOp]: " << *memspaceCastOp << "\n");
  LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");

  for (Operation* user : srcValue.getUsers()) {
    auto userIdAttr = user->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    int userBlockId = userIdAttr.getInt();
    if (userBlockId == iniConsumerId) {
      user->replaceUsesOfWith(srcValue, toTensorOp.getResult());
    }
  }
  return copyOp;
}

Operation* InterCoreTransferAndSyncPass::insertCubeToVectorTransfer(OpBuilder &builder, Value srcValue, Operation* cubeEndOp, Operation* vectorStartOp, Location loc, int transferIndex, int iniConsumerId) {
  LOG_DEBUG("Inserting [Cube->Vector] transfer for value: " << srcValue << "\n");
  auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
  int64_t M = srcTensorType.getDimSize(0);
  int64_t N = srcTensorType.getDimSize(1);
  Type elemType = srcTensorType.getElementType();

  auto cidAttr = srcValue.getDefiningOp()->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int cubeBlockId = cidAttr.getInt();
  auto vidAttr = vectorStartOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int vecBlockId = vidAttr.getInt();

  auto ubaddressSpaceAttr = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  auto allocType = MemRefType::get(
    {M, N},
    elemType,
    /*layout=*/nullptr,
    ubaddressSpaceAttr);

  Operation* cubeAllocOp;
  Operation* vecAllocOp;
  // Allocate memory outside mainloop
  Operation* mainLoopOp = findMainLoopforTransfer(cubeEndOp, vectorStartOp);
  if (mainLoopOp) {
    LOG_DEBUG("[mainLoopOp]" << *mainLoopOp << "\n");
    builder.setInsertionPoint(mainLoopOp);
    cubeAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markCubeAllocOp = annotateTightlyCoupledBuffer(builder, cubeAllocOp, loc);
    vecAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markVecAllocOp = annotateTightlyCoupledBuffer(builder, vecAllocOp, loc);
    auto loopidAttr = mainLoopOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    int loopBlockId = loopidAttr.getInt();
    attachTransferTags(cubeAllocOp, loopBlockId, "CUBE", transferIndex);
    attachTransferTags(vecAllocOp, loopBlockId, "VECTOR", transferIndex);
    attachTransferTags(markVecAllocOp, loopBlockId, "VECTOR", transferIndex);
    attachTransferTags(markCubeAllocOp, loopBlockId, "CUBE", transferIndex);
    builder.setInsertionPointAfter(cubeEndOp);
    LOG_DEBUG("[cubeAllocOp]" << *cubeAllocOp << "\n");
    LOG_DEBUG("[vecAllocOp]" << *vecAllocOp << "\n");
  } else {
    builder.setInsertionPoint(vectorStartOp);
    vecAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markVecAllocOp = annotateTightlyCoupledBuffer(builder, vecAllocOp, loc);
    builder.setInsertionPointAfter(cubeEndOp);
    cubeAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markCubeAllocOp = annotateTightlyCoupledBuffer(builder, cubeAllocOp, loc);
    attachTransferTags(cubeAllocOp, cubeBlockId, "CUBE", transferIndex);
    attachTransferTags(vecAllocOp, vecBlockId, "VECTOR", transferIndex);
    attachTransferTags(markVecAllocOp, vecBlockId, "VECTOR", transferIndex);
    attachTransferTags(markCubeAllocOp, cubeBlockId, "CUBE", transferIndex);
    LOG_DEBUG("[cubeAllocOp]" << *cubeAllocOp << "\n");
    LOG_DEBUG("[vecAllocOp]" << *vecAllocOp << "\n");
  }
  markAllocIndex++;

  FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(builder.getContext(), FixpipeDMAMode::NZ2ND);
  auto fixpipeOp = builder.create<hivm::FixpipeOp>(
      loc,
      mlir::TypeRange{}, // No return value
      srcValue,         // src
      cubeAllocOp->getResult(0),           // dst
      /*unit_flag_cond=*/mlir::ValueRange{},
      /*dma_mode=*/dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*pre_quant=*/nullptr,
      /*pre_relu=*/nullptr,
      /*channel_split=*/nullptr,
      /*unit_flag_mode=*/mlir::ArrayAttr{});
  attachTransferTags(fixpipeOp, cubeBlockId, "CUBE", transferIndex);
  LOG_DEBUG("[fixpipeOp]: " << *fixpipeOp << "\n");

  // Vector side: memspace_cast + to_tensor
  builder.setInsertionPoint(vectorStartOp);

  auto plainMemrefType = MemRefType::get({M, N}, elemType);
  auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
      loc, plainMemrefType, vecAllocOp->getResult(0));

  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc,
      srcTensorType,
      memspaceCastOp.getResult(),
      /*restrict=*/true,
      /*writable=*/true);

  attachTransferTags(memspaceCastOp, vecBlockId, "VECTOR", transferIndex);
  attachTransferTags(toTensorOp, vecBlockId, "VECTOR", transferIndex);
  LOG_DEBUG("[memspaceCastOp]: " << *memspaceCastOp << "\n");
  LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");

  for (Operation* user : srcValue.getUsers()) {
    auto userIdAttr = user->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    int userBlockId = userIdAttr.getInt();
    if (userBlockId == iniConsumerId) {
      user->replaceUsesOfWith(srcValue, toTensorOp.getResult());
    }
  }
  return fixpipeOp;
}

void InterCoreTransferAndSyncPass::insertInterCoreSync(OpBuilder &builder, Operation* transferOp, Operation* consumerStartOp, Operation* consumerEndOp, int flag, Location loc, int transferIndex) {
  LOG_DEBUG("Inserting inter-core synchronization for transferOp: " << *transferOp << "\n");
  auto cubeCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
  auto vecCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
  auto pipeFixAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
  auto pipeVAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
  auto pipeMte3Attr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
  auto pipeMte1Attr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
  auto pipeMAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_M);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  auto produceridAttr = transferOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int producerBlockId = produceridAttr.getInt();
  auto consumeridAttr = consumerStartOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  int consumerBlockId = consumeridAttr.getInt();

  Operation* mainLoopOp = findMainLoopforTransfer(transferOp, consumerStartOp);

  if (dyn_cast<hivm::FixpipeOp>(transferOp)) {
    builder.setInsertionPointAfter(transferOp);
    auto setOpForRead = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeFixAttr, pipeVAttr, flagId);
    attachTransferTags(setOpForRead, producerBlockId, "CUBE", transferIndex);
    builder.setInsertionPoint(consumerStartOp);
    auto waitOpForRead = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeFixAttr, pipeVAttr, flagId);
    attachTransferTags(waitOpForRead, consumerBlockId, "VECTOR", transferIndex);

    if (mainLoopOp) {
      builder.setInsertionPoint(transferOp);
      auto waitOpForWrite = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeVAttr, pipeFixAttr, flagId);
      attachTransferTags(waitOpForWrite, producerBlockId, "CUBE", transferIndex);
      builder.setInsertionPointAfter(consumerEndOp);
      auto setOpForWrite = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeVAttr, pipeFixAttr, flagId);
      attachTransferTags(setOpForWrite, consumerBlockId, "VECTOR", transferIndex);

      builder.setInsertionPoint(mainLoopOp);
      auto setOpForStart = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeVAttr, pipeFixAttr, flagId);
      builder.setInsertionPointAfter(mainLoopOp);
      auto waitOpForEnd = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeVAttr, pipeFixAttr, flagId);

      auto startEndIdAttr = mainLoopOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      int startEndBlockId = startEndIdAttr.getInt();
      attachTransferTags(setOpForStart, startEndBlockId, "VECTOR", transferIndex);
      attachTransferTags(waitOpForEnd , startEndBlockId, "CUBE", transferIndex);
    }
    return;
  } else if (dyn_cast<hivm::CopyOp>(transferOp)) {
    builder.setInsertionPointAfter(transferOp);
    auto setOpForRead = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeMte3Attr, pipeMte1Attr, flagId);
    attachTransferTags(setOpForRead, producerBlockId, "VECTOR", transferIndex);
    builder.setInsertionPoint(consumerStartOp);
    auto waitOpForRead = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeMte3Attr, pipeMte1Attr, flagId);
    attachTransferTags(waitOpForRead, consumerBlockId, "CUBE", transferIndex);

    if (mainLoopOp) {
      builder.setInsertionPoint(transferOp);
      auto waitOpForWrite = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
      attachTransferTags(waitOpForWrite, producerBlockId, "VECTOR", transferIndex);

      builder.setInsertionPointAfter(consumerEndOp);
      auto setOpForWrite = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
      attachTransferTags(setOpForWrite, consumerBlockId, "CUBE", transferIndex);

      builder.setInsertionPoint(mainLoopOp);
      auto setOpForStart = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
      builder.setInsertionPointAfter(mainLoopOp);
      auto waitOpForEnd  = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeMAttr, pipeMte3Attr, flagId);

      auto startEndIdAttr = mainLoopOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      int startEndBlockId = startEndIdAttr.getInt();
      attachTransferTags(setOpForStart, startEndBlockId, "CUBE", transferIndex);
      attachTransferTags(waitOpForEnd , startEndBlockId, "VECTOR", transferIndex);
    }
    return;
  }
}

void InterCoreTransferAndSyncPass::insertPipeSSync(OpBuilder &builder,
                                        Operation* producerOp,
                                        Operation* consumerOp,
                                        int flag,
                                        Location loc,
                                        bool isCubeToVector) {
  LOG_DEBUG("Inserting PIPE_S sync: "
               << (isCubeToVector ? "CUBE->VECTOR" : "VECTOR->CUBE")
               << ", flag = " << flag << "\n");

  // 确定 srcPipe 和 coreType
  // CUBE -> VECTOR: srcPipe = PIPE_FIX, srcCoreType = CUBE, dstCoreType = VECTOR
  // VECTOR -> CUBE: srcPipe = PIPE_MTE2, srcCoreType = VECTOR, dstCoreType = CUBE
  hivm::PIPE srcPipe = isCubeToVector ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE2;
  hivm::TCoreType srcCoreType = isCubeToVector ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
  hivm::TCoreType dstCoreType = isCubeToVector ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
  hivm::PIPE dstPipe = hivm::PIPE::PIPE_S;

  // 创建属性
  auto srcCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), srcCoreType);
  auto dstCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), dstCoreType);
  auto srcPipeAttr = PipeAttr::get(builder.getContext(), srcPipe);
  auto dstPipeAttr = PipeAttr::get(builder.getContext(), dstPipe);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  // 在生产者操作之后插入 set 操作
  builder.setInsertionPointAfter(producerOp);
  auto setOp = builder.create<SyncBlockSetOp>(loc, srcCoreAttr, srcPipeAttr, dstPipeAttr, flagId);

  // 在消费者操作之前插入 wait 操作
  builder.setInsertionPoint(consumerOp);
  auto waitOp = builder.create<SyncBlockWaitOp>(loc, dstCoreAttr, srcPipeAttr, dstPipeAttr, flagId);

  // 获取 block ID 并设置属性
  auto prodIdAttr = producerOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");
  auto consIdAttr = consumerOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id");

  if (prodIdAttr) {
    int prodBlockId = prodIdAttr.getInt();
    StringRef prodCoreType = isCubeToVector ? "CUBE" : "VECTOR";
    attachCommonTags(setOp, prodBlockId, prodCoreType);
  }

  if (consIdAttr) {
    int consBlockId = consIdAttr.getInt();
    StringRef consCoreType = isCubeToVector ? "VECTOR" : "CUBE";
    attachCommonTags(waitOp, consBlockId, consCoreType);
  }

  LOG_DEBUG("[PIPE_S setOp]: " << *setOp << "\n");
  LOG_DEBUG("[PIPE_S waitOp]: " << *waitOp << "\n");
}

// ==========================================
// V->C Transfer Logic
// ==========================================
LogicalResult InterCoreTransferAndSyncPass::handleVectorToCube(
    OpBuilder &builder,
    DependencyInfo& dep,
    llvm::DenseMap<mlir::Value, mlir::Value> vecvalueMapping,
    llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping,
    FlagIdManager& flagManager) {

  mlir::Value srcValue = dep.value;
  auto it = cubeValueMapping.find(srcValue);
  if (it != cubeValueMapping.end()) {
    srcValue = it->second;
  }
  Location loc = dep.value.getLoc();
  // Step 1: Shape normalization (automatically insert slice)
  Value normalizedVal = vecvalueMapping[dep.value];

  // Get start/end operations for V/C blocks
  auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
  auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

  Operation* transferOp = insertVectorToCubeTransfer(builder, srcValue, normalizedVal, prodEnd, consStart, loc, transferIndex, dep.iniConsumerBlockId);

  int flagId = flagManager.acquireId(prodStart);
  assert(flagId != -1 && "No available flag_id!!");

  auto [newProdStart, newProdEnd] = getBlockStartEnd(dep.producerBlockId, module);
  auto [newConsStart, newConsEnd] = getBlockStartEnd(dep.consumerBlockId, module);
  insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId, loc, transferIndex);

  transferIndex++;
  LOG_DEBUG("Inserted V->C transfer and sync: block " << dep.producerBlockId
            << " -> block " << dep.consumerBlockId << "\n");
  return success();
}

// ==========================================
// C->V Transfer Logic
// ==========================================
LogicalResult InterCoreTransferAndSyncPass::handleCubeToVector(
    OpBuilder &builder,
    DependencyInfo& dep,
    llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping,
    FlagIdManager& flagManager) {
  mlir::Value srcValue = dep.value;
  auto it = cubeValueMapping.find(srcValue);
  if (it != cubeValueMapping.end()) {
    srcValue = it->second;
  }
  Location loc = srcValue.getLoc();
  auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module); // C Block
  auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module); // V Block
  LOG_DEBUG("[newProdStart]" << *prodStart << "\n");
  LOG_DEBUG("[newProdEnd]" << *prodEnd << "\n");
  LOG_DEBUG("[newConsStart]" << *consStart << "\n");
  LOG_DEBUG("[newConsEnd]" << *consEnd << "\n");
  Operation* transferOp = insertCubeToVectorTransfer(builder, srcValue, prodEnd, consStart, loc, transferIndex, dep.iniConsumerBlockId);

  auto [newProdStart, newProdEnd] = getBlockStartEnd(dep.producerBlockId, module); // C Block
  auto [newConsStart, newConsEnd] = getBlockStartEnd(dep.consumerBlockId, module); // V Block
  int flagId = flagManager.acquireId(newProdStart);
  assert(flagId != -1 && "No available flag_id!!");

  insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId, loc, transferIndex);

  transferIndex++;
  LOG_DEBUG("Inserted C->V transfer and sync: block " << dep.producerBlockId
            << " -> block " << dep.consumerBlockId << "\n");
  return success();
}

// PIPE_S Memory Dependency
// PIPE_S 内存依赖处理
LogicalResult InterCoreTransferAndSyncPass::handleMemoryDependency(
    OpBuilder &builder,
    DependencyInfo& dep,
    size_t depIndex,
    llvm::SmallVector<DependencyInfo> memDependencies,
    FlagIdManager& flagManager) {
  LOG_DEBUG("Handling PIPE_S memory dependency...\n");

  // Get producer and consumer block start/end operations
  // 获取生产者和消费者 block 的起止操作
  auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
  auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

  if (!prodStart || !prodEnd || !consStart || !consEnd) {
    LOG_DEBUG("[ERROR] Failed to get block start/end operations.\n");
    return failure();
  }

  if (isOuterLayerDependency(depIndex, prodEnd, consStart, memDependencies)) {
    LOG_DEBUG("[PIPE_S] Skipping outer layer dependency: block " << dep.producerBlockId
             << " -> block " << dep.consumerBlockId << "\n");
    return success();
  }

  // Get flag ID
  // 获取 flag ID
  int flagId = flagManager.acquireId(prodStart);
  assert(flagId != -1 && "No available flag_id!!");

  // Determine sync direction: CUBE->VECTOR or VECTOR->CUBE
  // 确定同步方向：是 CUBE->VECTOR 还是 VECTOR->CUBE
  bool isCubeToVector = (dep.type == DependencyType::CubeToVector);

  // Get location info
  // 获取位置信息
  Location loc = prodEnd->getLoc();

  // Insert PIPE_S sync
  // 调用 insertPipeSSync 插入 PIPE_S 同步
  insertPipeSSync(builder, prodEnd, consStart, flagId, loc, isCubeToVector);

  transferIndex++;

  LOG_DEBUG("Inserted PIPE_S sync: block " << dep.producerBlockId
               << " -> block " << dep.consumerBlockId
               << ", flagId = " << flagId << "\n");

  return success();
}

// Main Processing
LogicalResult InterCoreTransferAndSyncPass::processDependencies(FlagIdManager& flagManager) {
  LOG_DEBUG("Starting InterCoreTransferAndSyncPass processDependencies...\n");
  OpBuilder builder(module.getContext());

  // 从 MLIR Analysis 获取数据依赖分析结果
  // 只传递数据 (DataDependencyInfo)，不传递 Pass
  auto& info = getAnalysis<DataDependencyInfo>();
  if (!info.isValid()) {
    LOG_DEBUG("Error: Data dependency analysis failed.\n");
    return failure();
  }

  llvm::SmallVector<DependencyInfo>& V2CDependencies = info.getV2CDependencies();
  LOG_DEBUG("[DEBUG] V2CDependencies size: " << V2CDependencies.size() << "\n");
  // Step 1: Handle V->C dependencies
  for (auto& dep : V2CDependencies) {
    Location loc = dep.value.getLoc();
    Nd2NzNormalize(builder, dep, loc);
  }
  llvm::DenseMap<mlir::Value, mlir::Value> vecvalueMapping = getVecValueMapping();
  llvm::DenseMap<mlir::Value, mlir::Value> cubevalueMapping = getCubeValueMapping();
  for (auto& dep : V2CDependencies) {
    LOG_DEBUG("[V->C] producerBlockId = " << dep.producerBlockId
            << ", consumerBlockId = " << dep.consumerBlockId << "\n");
    if (failed(handleVectorToCube(builder, dep, vecvalueMapping, cubevalueMapping, flagManager))) {
      LOG_DEBUG("[ERROR] V->C failed! producerBlockId = " << dep.producerBlockId
                  << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed V->C transfers and syncs.\n");

  llvm::SmallVector<DependencyInfo>& C2VDependencies = info.getC2VDependencies();
  LOG_DEBUG("[DEBUG] C2VDependencies size: " << C2VDependencies.size() << "\n");
  // Step 2: Handle C->V dependencies
  for (auto& dep : C2VDependencies) {
    LOG_DEBUG("[C->V] producerBlockId = " << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
    if (failed(handleCubeToVector(builder, dep, cubevalueMapping, flagManager))) {
    LOG_DEBUG("[ERROR] C->V failed!  producerBlockId = " << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed C->V transfers and syncs.\n");

  llvm::SmallVector<DependencyInfo>& memDependencies = info.getMemoryDependencies();
  LOG_DEBUG("[DEBUG] MemoryDependencies size: " << memDependencies.size() << "\n");

  for (size_t i = 0; i < memDependencies.size(); ++i) {
    auto& dep = memDependencies[i];
    LOG_DEBUG("[PIPE_S] value = " << dep.value
                 << " producerBlockId = " << dep.producerBlockId
                 << ", consumerBlockId = " << dep.consumerBlockId << "\n");
    if (failed(handleMemoryDependency(builder, dep, i, memDependencies, flagManager))) {
      LOG_DEBUG("[ERROR] PIPE_S failed! producerBlockId = " << dep.producerBlockId
                   << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed PIPE_S memory syncs.\n");
  LOG_DEBUG("=====================================================\n");
  LOG_DEBUG("InterCoreTransferAndSyncPass success!\n");

  return success();
}


// Declare dependent dialects
void InterCoreTransferAndSyncPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<
    func::FuncDialect,
    arith::ArithDialect,
    linalg::LinalgDialect,
    scf::SCFDialect,
    tensor::TensorDialect,
    bufferization::BufferizationDialect,
    memref::MemRefDialect,
    hivm::HIVMDialect,
    annotation::AnnotationDialect
  >();
}

// Pass Entry Point
void InterCoreTransferAndSyncPass::runOnOperation() {
  LOG_DEBUG("\n--- enter InterCoreTransferAndSyncPass --->\n");
  module = getOperation();

  // Phase 1: Initialize FlagIdManager as local variable
  FlagIdManager flagManager(module);

  // Phase 2: Execute transfer and sync insertion
  if (failed(processDependencies(flagManager))) {
    signalPassFailure();
    LOG_DEBUG("Error: Inter-core transfer and sync failed.\n");
    return;
  }

  // IR validity verification
  // IR 有效性验证
  if (failed(module.verify())) {
    module.emitError("InterCoreTransferAndSync: generated invalid IR");
    signalPassFailure();
  }
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createInterCoreTransferAndSyncPass()
{
    return std::make_unique<InterCoreTransferAndSyncPass>();
}

void registerInterCoreTransferAndSyncPasses()
{
  registerPass([]() -> std::unique_ptr<mlir::Pass> { return createInterCoreTransferAndSyncPass(); });
}

} // namespace triton
} // namespace mlir
