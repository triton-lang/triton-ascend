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

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/PermutationAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

struct StoreCandidate {
  Operation *operation;
  StaticAccess access;
  RankedTensorType pointerType;
  RankedTensorType valueType;
  RankedTensorType offsetType;
  Type pointeeType;
  uint64_t elementBytes;
  Value value;
  triton::CacheModifier cache;
  triton::EvictionPolicy evict;
  DictionaryAttr attributes;
};

struct UBPreloadRun {
  SmallVector<StoreCandidate, 4> addressOrderStores;
  SmallVector<Operation *, 4> programOrderStores;
  Operation *anchor = nullptr;
  int64_t firstOffset = 0;
  int64_t endExclusive = 0;
  int64_t totalElements = 0;
  uint64_t totalBytes = 0;
};

bool fitsI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

bool checkedAddI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs)
    return false;
  if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedSubI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs)
    return false;
  if (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs)
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> getByteWidth(Type type) {
  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    bitWidth = integerType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
  } else {
    return std::nullopt;
  }

  // i1 and sub-byte integer/float element types are deliberately excluded:
  // capacity is accounted in concrete addressable bytes, never rounded lanes.
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return static_cast<uint64_t>(bitWidth / 8);
}

bool isUnencodedStaticRankOneTensor(RankedTensorType type) {
  return type && type.hasStaticShape() && !type.getEncoding() &&
         type.getRank() == 1 && type.getShape().front() > 0;
}

bool isDirectFunctionBodyStore(triton::StoreOp store,
                               triton::FuncOp function, Block *block) {
  if (!store || !function || !block || store->getBlock() != block ||
      store->getNumRegions() != 0)
    return false;
  if (block->getParent() != &function.getBody())
    return false;
  triton::FuncOp parent = store->getParentOfType<triton::FuncOp>();
  return parent && parent.getOperation() == function.getOperation();
}

bool hasExpectedRankOneStoreTypes(const StoreCandidate &candidate) {
  if (!isUnencodedStaticRankOneTensor(candidate.pointerType) ||
      !isUnencodedStaticRankOneTensor(candidate.valueType) ||
      !isUnencodedStaticRankOneTensor(candidate.offsetType))
    return false;
  if (candidate.pointerType.getShape() != candidate.valueType.getShape() ||
      candidate.pointerType.getShape() != candidate.offsetType.getShape() ||
      candidate.pointerType.getEncoding() != candidate.valueType.getEncoding() ||
      candidate.pointerType.getEncoding() != candidate.offsetType.getEncoding())
    return false;

  auto pointerElement =
      dyn_cast<triton::PointerType>(candidate.pointerType.getElementType());
  if (!pointerElement || triton::isTensorPointerType(pointerElement) ||
      pointerElement.getPointeeType() != candidate.pointeeType ||
      candidate.valueType.getElementType() != candidate.pointeeType)
    return false;

  int64_t span = 0;
  if (!checkedSubI64(candidate.access.lastOffset,
                     candidate.access.firstOffset, span) ||
      !checkedAddI64(span, 1, span) || span <= 0 ||
      span != candidate.pointerType.getShape().front())
    return false;
  return true;
}

std::optional<StoreCandidate>
matchStore(triton::StoreOp store, triton::FuncOp function, Block *block) {
  if (!isDirectFunctionBodyStore(store, function, block) || store.getMask() ||
      !store.getBoundaryCheck().empty())
    return std::nullopt;

  auto pointerType = dyn_cast<RankedTensorType>(store.getPtr().getType());
  auto valueType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!pointerType || !valueType || !isUnencodedStaticRankOneTensor(pointerType) ||
      !isUnencodedStaticRankOneTensor(valueType))
    return std::nullopt;

  auto pointerElement =
      dyn_cast<triton::PointerType>(pointerType.getElementType());
  if (!pointerElement || triton::isTensorPointerType(pointerElement) ||
      pointerElement.getPointeeType() != valueType.getElementType() ||
      pointerType.getShape() != valueType.getShape() ||
      pointerType.getEncoding() != valueType.getEncoding())
    return std::nullopt;

  std::optional<uint64_t> elementBytes =
      getByteWidth(valueType.getElementType());
  if (!elementBytes)
    return std::nullopt;

  StaticAccessAnalysis accessAnalysis;
  StaticAccessProof proof = accessAnalysis.analyzeStore(store);
  if (!proof.isProven() || !proof.access->isRankOneContiguous() ||
      proof.access->pointer != store.getPtr() || !proof.access->base ||
      proof.access->shape.size() != 1 || proof.access->shape.front() !=
                                            pointerType.getShape().front())
    return std::nullopt;

  auto offsetType = dyn_cast<RankedTensorType>(proof.access->offset.getType());
  if (!offsetType)
    return std::nullopt;

  StoreCandidate candidate{store.getOperation(),
                           std::move(*proof.access),
                           pointerType,
                           valueType,
                           offsetType,
                           pointerElement.getPointeeType(),
                           *elementBytes,
                           store.getValue(),
                           store.getCache(),
                           store.getEvict(),
                           store->getAttrDictionary()};
  if (!hasExpectedRankOneStoreTypes(candidate))
    return std::nullopt;
  return candidate;
}

bool belongToSameBucket(const StoreCandidate &lhs,
                        const StoreCandidate &rhs) {
  // The base comparison is deliberately SSA identity only.  This rule never
  // consults alias analysis, so two different bases can never be coalesced.
  return lhs.operation->getBlock() == rhs.operation->getBlock() &&
         lhs.access.base == rhs.access.base &&
         lhs.pointeeType == rhs.pointeeType &&
         lhs.valueType.getElementType() == rhs.valueType.getElementType() &&
         lhs.pointerType.getEncoding() == rhs.pointerType.getEncoding() &&
         lhs.valueType.getEncoding() == rhs.valueType.getEncoding() &&
         lhs.cache == rhs.cache && lhs.evict == rhs.evict &&
         lhs.attributes == rhs.attributes;
}

bool isBeforeInProgramOrder(Operation *lhs, Operation *rhs) {
  return lhs && rhs && lhs != rhs && lhs->getBlock() == rhs->getBlock() &&
         lhs->isBeforeInBlock(rhs);
}

bool isLocalValueBeforeAnchor(Value value, Operation *anchor,
                              DominanceInfo &dominance) {
  if (!value || !anchor || !dominance.dominates(value, anchor))
    return false;

  if (auto blockArgument = dyn_cast<BlockArgument>(value))
    return blockArgument.getOwner() == anchor->getBlock();

  Operation *definition = value.getDefiningOp();
  return definition && definition->getBlock() == anchor->getBlock() &&
         definition->isBeforeInBlock(anchor);
}

bool valuesDominateAnchor(const UBPreloadRun &run, triton::FuncOp function) {
  if (!run.anchor || !function)
    return false;

  DominanceInfo dominance(function.getOperation());
  for (const StoreCandidate &candidate : run.addressOrderStores) {
    if (!candidate.value || !candidate.access.base ||
        !isLocalValueBeforeAnchor(candidate.value, run.anchor, dominance) ||
        !isLocalValueBeforeAnchor(candidate.access.base, run.anchor,
                                  dominance))
      return false;
  }
  return true;
}

std::optional<UBPreloadRun>
buildRun(ArrayRef<StoreCandidate> addressOrderStores,
         triton::FuncOp function, unsigned ubCapacityBytes) {
  if (!function || ubCapacityBytes == 0 || addressOrderStores.size() < 2)
    return std::nullopt;

  const StoreCandidate &first = addressOrderStores.front();
  const StoreCandidate &last = addressOrderStores.back();
  for (const StoreCandidate &candidate : addressOrderStores) {
    if (candidate.elementBytes != first.elementBytes)
      return std::nullopt;
  }
  int64_t endExclusive = 0;
  int64_t totalElements = 0;
  if (!checkedAddI64(last.access.lastOffset, 1, endExclusive) ||
      !checkedSubI64(endExclusive, first.access.firstOffset, totalElements) ||
      totalElements <= 0 || !fitsI32(first.access.firstOffset) ||
      !fitsI32(endExclusive))
    return std::nullopt;

  uint64_t totalBytes = 0;
  if (!checkedMulU64(static_cast<uint64_t>(totalElements),
                     first.elementBytes, totalBytes) ||
      totalBytes > static_cast<uint64_t>(ubCapacityBytes))
    return std::nullopt;

  UBPreloadRun run;
  run.addressOrderStores.append(addressOrderStores.begin(),
                                addressOrderStores.end());
  run.firstOffset = first.access.firstOffset;
  run.endExclusive = endExclusive;
  run.totalElements = totalElements;
  run.totalBytes = totalBytes;
  for (const StoreCandidate &candidate : run.addressOrderStores)
    run.programOrderStores.push_back(candidate.operation);
  std::stable_sort(run.programOrderStores.begin(), run.programOrderStores.end(),
                   isBeforeInProgramOrder);
  run.anchor = run.programOrderStores.back();

  if (!valuesDominateAnchor(run, function))
    return std::nullopt;

  // Check adjacent planned stores in program order, not address order.  Thus
  // another store in this run is always an endpoint of one of these checks,
  // never a spurious intervening memory effect.  Any non-planned load, store,
  // call, barrier, or unknown effect remains a rejection.
  ProtectedIntervalAnalysis intervalAnalysis;
  for (size_t index = 1; index < run.programOrderStores.size(); ++index) {
    if (!intervalAnalysis
             .proveNoMemoryEffects(run.programOrderStores[index - 1],
                                   run.programOrderStores[index])
             .isProven())
      return std::nullopt;
  }

  if (run.addressOrderStores.size() - 1 >
      static_cast<size_t>(std::numeric_limits<unsigned>::max()))
    return std::nullopt;
  return run;
}

SmallVector<UBPreloadRun, 4> findRuns(triton::FuncOp function,
                                      unsigned ubCapacityBytes) {
  SmallVector<UBPreloadRun, 4> runs;
  if (!function || ubCapacityBytes == 0)
    return runs;

  for (Block &block : function.getBody()) {
    SmallVector<StoreCandidate, 8> stores;
    for (Operation &operation : block) {
      auto store = dyn_cast<triton::StoreOp>(&operation);
      if (!store)
        continue;
      std::optional<StoreCandidate> candidate =
          matchStore(store, function, &block);
      if (candidate)
        stores.push_back(std::move(*candidate));
    }

    SmallVector<SmallVector<StoreCandidate, 4>, 4> buckets;
    for (StoreCandidate &candidate : stores) {
      auto bucket = std::find_if(
          buckets.begin(), buckets.end(), [&](const auto &existing) {
            return !existing.empty() &&
                   belongToSameBucket(existing.front(), candidate);
          });
      if (bucket == buckets.end()) {
        buckets.emplace_back();
        buckets.back().push_back(candidate);
      } else {
        bucket->push_back(candidate);
      }
    }

    for (SmallVector<StoreCandidate, 4> &bucket : buckets) {
      if (bucket.size() < 2)
        continue;

      std::stable_sort(bucket.begin(), bucket.end(),
                       [](const StoreCandidate &lhs,
                          const StoreCandidate &rhs) {
                         if (lhs.access.firstOffset != rhs.access.firstOffset)
                           return lhs.access.firstOffset < rhs.access.firstOffset;
                         if (lhs.access.lastOffset != rhs.access.lastOffset)
                           return lhs.access.lastOffset < rhs.access.lastOffset;
                         return isBeforeInProgramOrder(lhs.operation,
                                                       rhs.operation);
                       });

      // An overlap invalidates every candidate in this bucket.  In particular,
      // do not try to salvage a later non-overlapping sub-run.
      bool hasOverlap = false;
      for (size_t index = 1; index < bucket.size(); ++index) {
        if (bucket[index].access.firstOffset <=
            bucket[index - 1].access.lastOffset) {
          hasOverlap = true;
          break;
        }
      }
      if (hasOverlap)
        continue;

      size_t runBegin = 0;
      for (size_t index = 1; index <= bucket.size(); ++index) {
        bool endsRun = index == bucket.size();
        if (!endsRun) {
          int64_t expectedNextOffset = 0;
          endsRun = !checkedAddI64(bucket[index - 1].access.lastOffset, 1,
                                    expectedNextOffset) ||
                    expectedNextOffset != bucket[index].access.firstOffset;
        }

        if (!endsRun)
          continue;
        if (index - runBegin >= 2) {
          std::optional<UBPreloadRun> run =
              buildRun(ArrayRef<StoreCandidate>(bucket).slice(runBegin,
                                                               index - runBegin),
                       function, ubCapacityBytes);
          if (run)
            runs.push_back(std::move(*run));
        }
        runBegin = index;
      }
    }
  }
  return runs;
}

bool hasSameAddressOrder(const UBPreloadRun &run,
                         ArrayRef<Operation *> operations) {
  if (run.addressOrderStores.size() != operations.size())
    return false;
  for (size_t index = 0; index < operations.size(); ++index) {
    if (run.addressOrderStores[index].operation != operations[index])
      return false;
  }
  return true;
}

std::optional<UBPreloadRun>
findRunByOperations(triton::FuncOp function, unsigned ubCapacityBytes,
                    ArrayRef<Operation *> operations) {
  for (UBPreloadRun &run : findRuns(function, ubCapacityBytes)) {
    if (hasSameAddressOrder(run, operations))
      return run;
  }
  return std::nullopt;
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

bool recordVerifiedOperation(Operation *operation,
                             SmallVectorImpl<Operation *> &created) {
  if (!operation)
    return false;
  created.push_back(operation);
  return succeeded(mlir::verify(operation));
}

LogicalResult applyRun(IRRewriter &rewriter, const UBPreloadRun &run) {
  if (!run.anchor || run.addressOrderStores.size() < 2 ||
      run.programOrderStores.size() != run.addressOrderStores.size() ||
      run.totalElements <= 0 || !fitsI32(run.firstOffset) ||
      !fitsI32(run.endExclusive))
    return failure();

  const StoreCandidate &exemplar = run.addressOrderStores.front();
  SmallVector<int64_t, 1> packedShape = {run.totalElements};
  // clone() retains any type details carried by the original tensor.  V1 has
  // already required unencoded types.
  RankedTensorType packedValueType = exemplar.valueType.clone(packedShape);
  RankedTensorType packedPointerType = exemplar.pointerType.clone(packedShape);
  RankedTensorType packedOffsetType = exemplar.offsetType.clone(packedShape);
  if (!isUnencodedStaticRankOneTensor(packedValueType) ||
      !isUnencodedStaticRankOneTensor(packedPointerType) ||
      !isUnencodedStaticRankOneTensor(packedOffsetType) ||
      packedPointerType.getShape() != packedValueType.getShape() ||
      packedPointerType.getShape() != packedOffsetType.getShape())
    return failure();

  SmallVector<Operation *, 16> created;
  rewriter.setInsertionPoint(run.anchor);
  auto empty = rewriter.create<tensor::EmptyOp>(run.anchor->getLoc(),
                                                 packedValueType, ValueRange{});
  if (!recordVerifiedOperation(empty.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  Value packedValue = empty.getResult();
  int64_t packedOffset = 0;
  for (const StoreCandidate &candidate : run.addressOrderStores) {
    int64_t sliceLength = 0;
    if (!checkedSubI64(candidate.access.lastOffset,
                       candidate.access.firstOffset, sliceLength) ||
        !checkedAddI64(sliceLength, 1, sliceLength) || sliceLength <= 0 ||
        candidate.valueType.getShape().front() != sliceLength) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    SmallVector<OpFoldResult, 1> offsets = {
        rewriter.getIndexAttr(packedOffset)};
    SmallVector<OpFoldResult, 1> sizes = {rewriter.getIndexAttr(sliceLength)};
    SmallVector<OpFoldResult, 1> strides = {rewriter.getIndexAttr(1)};
    auto inserted = rewriter.create<tensor::InsertSliceOp>(
        run.anchor->getLoc(), candidate.value, packedValue, offsets,
        sizes, strides);
    if (!recordVerifiedOperation(inserted.getOperation(), created)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    packedValue = inserted.getResult();

    int64_t nextPackedOffset = 0;
    if (!checkedAddI64(packedOffset, sliceLength, nextPackedOffset)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    packedOffset = nextPackedOffset;
  }
  if (packedOffset != run.totalElements) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  auto range = rewriter.create<triton::MakeRangeOp>(
      run.anchor->getLoc(), packedOffsetType,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(run.firstOffset)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(run.endExclusive)));
  if (!recordVerifiedOperation(range.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }
  auto baseSplat = rewriter.create<triton::SplatOp>(
      run.anchor->getLoc(), packedPointerType, exemplar.access.base);
  if (!recordVerifiedOperation(baseSplat.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }
  auto pointer = rewriter.create<triton::AddPtrOp>(
      run.anchor->getLoc(), packedPointerType, baseSplat.getResult(),
      range.getResult());
  if (!recordVerifiedOperation(pointer.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  auto combinedStore = rewriter.create<triton::StoreOp>(
      run.anchor->getLoc(), pointer.getResult(), packedValue,
      exemplar.cache, exemplar.evict);
  // Bucket construction required complete attribute equality.  Copying the
  // full dictionary therefore preserves every store property and extra attr.
  combinedStore->setAttrs(exemplar.attributes);
  if (!recordVerifiedOperation(combinedStore.getOperation(), created)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  StaticAccessAnalysis accessAnalysis;
  StaticAccessProof rebuiltProof =
      accessAnalysis.analyzePointer(pointer.getResult());
  if (!rebuiltProof.isProven() || !rebuiltProof.access->isRankOneContiguous() ||
      rebuiltProof.access->base != exemplar.access.base ||
      rebuiltProof.access->firstOffset != run.firstOffset ||
      rebuiltProof.access->lastOffset != run.endExclusive - 1) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  // All operations have verified and the replacement pointer has re-proven
  // its static interval before any original memory effect is erased.
  for (Operation *operation : llvm::reverse(run.programOrderStores))
    rewriter.eraseOp(operation);
  return success();
}

class UBPreloadPlan final : public RewritePlan {
public:
  UBPreloadPlan(const UBPreloadRun &run, triton::FuncOp function,
                unsigned ubCapacityBytes, unsigned epoch)
      : function(function), anchor(run.anchor), ubCapacityBytes(ubCapacityBytes),
        epoch(epoch) {
    for (const StoreCandidate &candidate : run.addressOrderStores)
      addressOrderStores.push_back(candidate.operation);
  }

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::UBPreload;
  }

  unsigned getBenefit() const override {
    return addressOrderStores.size() < 2
               ? 0
               : static_cast<unsigned>(addressOrderStores.size() - 1);
  }

  Operation *getAnchor() const override { return anchor; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !function || !anchor ||
        context.getFunction().getOperation() != function.operator->())
      return failure();

    std::optional<UBPreloadRun> current = findRunByOperations(
        function, ubCapacityBytes, addressOrderStores);
    if (!current || current->anchor != anchor)
      return failure();
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    if (!function || !anchor)
      return failure();
    std::optional<UBPreloadRun> current = findRunByOperations(
        function, ubCapacityBytes, addressOrderStores);
    if (!current || current->anchor != anchor)
      return failure();
    return applyRun(rewriter, *current);
  }

private:
  triton::FuncOp function;
  Operation *anchor;
  SmallVector<Operation *, 4> addressOrderStores;
  unsigned ubCapacityBytes;
  unsigned epoch;
};

class UBPreloadRule final : public GraphOptimizationRule {
public:
  explicit UBPreloadRule(unsigned ubCapacityBytes)
      : ubCapacityBytes(ubCapacityBytes) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::UBPreload;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    for (const UBPreloadRun &run :
         findRuns(context.getFunction(), ubCapacityBytes)) {
      plans.push_back(std::make_unique<UBPreloadPlan>(
          run, context.getFunction(), ubCapacityBytes, context.getEpoch()));
    }
    return success();
  }

private:
  unsigned ubCapacityBytes;
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createUBPreloadRule(unsigned ubCapacityBytes) {
  return std::make_unique<UBPreloadRule>(ubCapacityBytes);
}
