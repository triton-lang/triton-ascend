/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
#include "AscendModel/Analysis/HardwareConfig.h"
#endif

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedScale(uint64_t value, uint64_t weight, uint64_t &result) {
  return checkedMul(value, weight, result);
}

bool checkedAccumulate(uint64_t value, uint64_t &total) {
  uint64_t next = 0;
  if (!checkedAdd(total, value, next))
    return false;
  total = next;
  return true;
}

bool isTensorValue(Value value) {
  return value && isa<RankedTensorType, UnrankedTensorType>(value.getType());
}

bool isUBResidentTensor(Value value) {
  if (!value)
    return false;
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  return tensor && tensor.hasStaticShape() &&
         isa<FloatType>(tensor.getElementType());
}

bool isStorageViewOperation(Operation *operation) {
  if (!operation)
    return false;
  const StringRef name = operation->getName().getStringRef();
  return name == "tt.expand_dims" || name == "tt.broadcast" ||
         name == "tt.reshape" || name == "tensor.cast" ||
         name == "tensor.collapse_shape" || name == "tensor.expand_shape";
}

bool isVirtualTensorOperation(Operation *operation) {
  if (!operation)
    return false;
  const StringRef name = operation->getName().getStringRef();
  return name == "tt.make_range" || name == "tt.splat" ||
         name == "arith.constant";
}

std::optional<unsigned> getElementBitWidth(Type elementType) {
  if (auto integer = dyn_cast<IntegerType>(elementType))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(elementType))
    return floating.getWidth();
  if (isa<IndexType>(elementType))
    return 64;
  if (isa<triton::PointerType>(elementType))
    return 64;
  return std::nullopt;
}

Value findCanonicalValue(Value value, const llvm::DenseMap<Value, Value> &map) {
  Value current = value;
  while (true) {
    auto it = map.find(current);
    if (it == map.end() || it->second == current)
      return current;
    current = it->second;
  }
}

uint64_t
getDefinitionOrder(Value value,
                   const llvm::DenseMap<Operation *, uint64_t> &order) {
  if (auto result = dyn_cast<OpResult>(value)) {
    auto it = order.find(result.getOwner());
    return it == order.end() ? 0 : it->second;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    auto it = order.find(parent);
    return it == order.end() ? 0 : it->second;
  }
  return 0;
}

bool isDefinedInside(Operation *valueOwner, Operation *containingOp) {
  return valueOwner && containingOp &&
         (valueOwner == containingOp || containingOp->isAncestor(valueOwner));
}

bool isDefinedInside(Value value, Operation *containingOp) {
  if (auto result = dyn_cast<OpResult>(value))
    return isDefinedInside(result.getOwner(), containingOp);
  if (auto argument = dyn_cast<BlockArgument>(value))
    return isDefinedInside(argument.getOwner()->getParentOp(), containingOp);
  return false;
}

bool isForLikeLoop(Operation *operation) {
  return operation && operation->getName().getStringRef() == "scf.for" &&
         operation->getNumRegions() == 1 && !operation->getRegion(0).empty();
}

Operation *getValueDefiningOperation(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return result.getOwner();
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner()->getParentOp();
  return nullptr;
}

struct ConditionalRegion {
  Operation *conditional = nullptr;
  unsigned region = 0;
};

std::optional<ConditionalRegion> getConditionalRegion(Value value) {
  Operation *operation = getValueDefiningOperation(value);
  while (operation) {
    Region *region = operation->getParentRegion();
    if (!region)
      return std::nullopt;
    Operation *parent = region->getParentOp();
    if (!parent)
      return std::nullopt;
    if (parent->getName().getStringRef() == "scf.if") {
      for (unsigned index = 0; index < parent->getNumRegions(); ++index)
        if (&parent->getRegion(index) == region)
          return ConditionalRegion{parent, index};
    }
    operation = parent;
  }
  return std::nullopt;
}

bool areMutuallyExclusive(Value lhs, Value rhs) {
  std::optional<ConditionalRegion> left = getConditionalRegion(lhs);
  std::optional<ConditionalRegion> right = getConditionalRegion(rhs);
  return left && right && left->conditional == right->conditional &&
         left->region != right->region;
}

void addCanonicalFactors(llvm::ArrayRef<unsigned> input,
                         llvm::SmallVectorImpl<unsigned> &output) {
  for (unsigned value : input) {
    if (value != 0)
      output.push_back(value);
  }
  if (output.empty())
    output.push_back(1);
  llvm::sort(output);
  output.erase(std::unique(output.begin(), output.end()), output.end());
}

bool addWeightedChange(uint64_t before, uint64_t after, uint64_t weight,
                       uint64_t &gain, uint64_t &penalty) {
  const uint64_t difference = before >= after ? before - after : after - before;
  uint64_t weighted = 0;
  if (!checkedScale(difference, weight, weighted))
    return false;
  return before >= after ? checkedAccumulate(weighted, gain)
                         : checkedAccumulate(weighted, penalty);
}

CandidateEvaluation reject(const CandidateCost &candidate,
                           ResourceCostRejectReason reason,
                           uint64_t safeUBBudget = 0,
                           uint64_t requiredParallelPrograms = 0,
                           const ResourceSnapshot *resources = nullptr) {
  CandidateEvaluation evaluation;
  evaluation.candidate = candidate;
  evaluation.reason = reason;
  evaluation.safeUBBudgetBytes = safeUBBudget;
  evaluation.requiredParallelPrograms = requiredParallelPrograms;
  if (resources && resources->hasL0Budgets()) {
    evaluation.safeL0ABudgetBytes = resources->getSafeL0ABudget().value_or(0);
    evaluation.safeL0BBudgetBytes = resources->getSafeL0BBudget().value_or(0);
    evaluation.safeL0CBudgetBytes = resources->getSafeL0CBudget().value_or(0);
  }
  evaluation.remark = formatCandidateRemark(evaluation);
  return evaluation;
}

} // namespace

const char *
cfg::getResourceCostRejectReasonName(ResourceCostRejectReason reason) {
  switch (reason) {
  case ResourceCostRejectReason::None:
    return "none";
  case ResourceCostRejectReason::UnknownResource:
    return "unknown_resource";
  case ResourceCostRejectReason::DynamicShape:
    return "dynamic_shape";
  case ResourceCostRejectReason::UnknownElementType:
    return "unknown_element_type";
  case ResourceCostRejectReason::Overflow:
    return "overflow";
  case ResourceCostRejectReason::UBOverflow:
    return "ub_overflow";
  case ResourceCostRejectReason::L0AOverflow:
    return "l0a_overflow";
  case ResourceCostRejectReason::L0BOverflow:
    return "l0b_overflow";
  case ResourceCostRejectReason::L0COverflow:
    return "l0c_overflow";
  case ResourceCostRejectReason::InvalidCandidate:
    return "invalid_candidate";
  }
  return "unknown_resource";
}

bool ResourceSnapshot::isKnown() const {
  return ubCapacityBytes != 0 && reservedUBBytes <= ubCapacityBytes &&
         ubSafetyPercent > 0 && ubSafetyPercent <= 100;
}

bool ResourceSnapshot::hasL0Budgets() const {
  const bool capacitiesKnown =
      l0aCapacityBytes != 0 && l0bCapacityBytes != 0 && l0cCapacityBytes != 0;
  return capacitiesKnown && l0SafetyPercent > 0 && l0SafetyPercent <= 100;
}

namespace {

std::optional<uint64_t> scaledSafeBudget(uint64_t availableBytes,
                                         unsigned safetyPercent) {
  if (safetyPercent == 0 || safetyPercent > 100)
    return std::nullopt;
  const uint64_t wholePercent = availableBytes / 100;
  const uint64_t remainder = availableBytes % 100;
  uint64_t budget = 0;
  if (!checkedMul(wholePercent, safetyPercent, budget))
    return std::nullopt;
  uint64_t remainderBudget = 0;
  if (!checkedMul(remainder, safetyPercent, remainderBudget))
    return std::nullopt;
  if (!checkedAdd(budget, remainderBudget / 100, budget))
    return std::nullopt;
  return budget;
}

} // namespace

std::optional<uint64_t> ResourceSnapshot::getSafeUBBudget() const {
  if (!isKnown())
    return std::nullopt;
  return scaledSafeBudget(ubCapacityBytes - reservedUBBytes, ubSafetyPercent);
}

std::optional<uint64_t> ResourceSnapshot::getSafeL0ABudget() const {
  if (l0aCapacityBytes == 0)
    return std::nullopt;
  return scaledSafeBudget(l0aCapacityBytes, l0SafetyPercent);
}

std::optional<uint64_t> ResourceSnapshot::getSafeL0BBudget() const {
  if (l0bCapacityBytes == 0)
    return std::nullopt;
  return scaledSafeBudget(l0bCapacityBytes, l0SafetyPercent);
}

std::optional<uint64_t> ResourceSnapshot::getSafeL0CBudget() const {
  if (l0cCapacityBytes == 0)
    return std::nullopt;
  return scaledSafeBudget(l0cCapacityBytes, l0SafetyPercent);
}

ResourceSnapshot
ResourceSnapshot::fromExplicit(uint64_t ubCapacity, unsigned ubSafety,
                               uint64_t reservedUB, uint64_t l0aCapacity,
                               uint64_t l0bCapacity, uint64_t l0cCapacity,
                               unsigned l0Safety) {
  ResourceSnapshot result;
  result.ubCapacityBytes = ubCapacity;
  result.reservedUBBytes = reservedUB;
  result.ubSafetyPercent = ubSafety;
  result.l0aCapacityBytes = l0aCapacity;
  result.l0bCapacityBytes = l0bCapacity;
  result.l0cCapacityBytes = l0cCapacity;
  result.l0SafetyPercent = l0Safety;
  return result;
}

ResourceSnapshot
ResourceSnapshot::fromHardwareConfig(const ascend::HardwareConfig &hardware,
                                     unsigned ubSafety, uint64_t reservedUB,
                                     unsigned l0Safety) {
#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
  const ascend::MemorySpace *ub = hardware.getMemorySpace("ub");
  if (!ub || ub->sizeBytes == 0 ||
      ub->sizeBytes > std::numeric_limits<uint64_t>::max())
    return {};
  uint64_t l0aCapacity = 0;
  uint64_t l0bCapacity = 0;
  uint64_t l0cCapacity = 0;
  if (const ascend::MemorySpace *l0a = hardware.getMemorySpace("l0a"))
    l0aCapacity = l0a->sizeBytes;
  if (const ascend::MemorySpace *l0b = hardware.getMemorySpace("l0b"))
    l0bCapacity = l0b->sizeBytes;
  if (const ascend::MemorySpace *l0c = hardware.getMemorySpace("l0c"))
    l0cCapacity = l0c->sizeBytes;
  return fromExplicit(static_cast<uint64_t>(ub->sizeBytes), ubSafety,
                      reservedUB, l0aCapacity, l0bCapacity, l0cCapacity,
                      l0Safety);
#else
  (void)hardware;
  (void)ubSafety;
  (void)reservedUB;
  (void)l0Safety;
  return {};
#endif
}

std::optional<uint64_t> cfg::getStaticTensorBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return std::nullopt;

  std::optional<unsigned> elementBits =
      getElementBitWidth(tensor.getElementType());
  if (!elementBits || *elementBits == 0)
    return std::nullopt;

  uint64_t elements = 1;
  for (int64_t dimension : tensor.getShape()) {
    if (dimension < 0)
      return std::nullopt;
    if (!checkedMul(elements, static_cast<uint64_t>(dimension), elements))
      return std::nullopt;
  }

  uint64_t totalBits = 0;
  if (!checkedMul(elements, *elementBits, totalBits))
    return std::nullopt;
  if (totalBits > std::numeric_limits<uint64_t>::max() - 7)
    return std::nullopt;
  return (totalBits + 7) / 8;
}

LiveByteEstimate cfg::estimatePeakLiveBytes(Operation *root) {
  LiveByteEstimate estimate;
  if (!root)
    return estimate;

  llvm::DenseMap<Operation *, uint64_t> order;
  llvm::SmallVector<Operation *, 64> operations;
  root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    order[operation] = operations.size();
    operations.push_back(operation);
  });

  llvm::DenseMap<Value, Value> canonicalValues;
  for (Operation *operation : operations) {
    if (!isForLikeLoop(operation))
      continue;

    Block &body = operation->getRegion(0).front();
    const unsigned initCount =
        operation->getNumOperands() < 3 ? 0 : operation->getNumOperands() - 3;
    if (body.getNumArguments() < initCount + 1 ||
        operation->getNumResults() != initCount)
      continue;

    for (unsigned index = 0; index < initCount; ++index)
      canonicalValues[body.getArgument(index + 1)] =
          findCanonicalValue(operation->getOperand(3 + index), canonicalValues);

    Operation *terminator = body.getTerminator();
    if (!terminator || terminator->getName().getStringRef() != "scf.yield" ||
        terminator->getNumOperands() != initCount)
      continue;
    for (unsigned index = 0; index < initCount; ++index)
      canonicalValues[operation->getResult(index)] =
          findCanonicalValue(terminator->getOperand(index), canonicalValues);
  }

  for (Operation *operation : operations) {
    if (!isStorageViewOperation(operation) || operation->getNumOperands() != 1)
      continue;
    Value source = operation->getOperand(0);
    if (!isTensorValue(source))
      continue;
    Value owner = findCanonicalValue(source, canonicalValues);
    for (Value result : operation->getResults())
      if (isTensorValue(result))
        canonicalValues[result] = owner;
  }

  struct IntervalState {
    Value value;
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t bytes = 0;
  };
  llvm::DenseMap<Value, IntervalState> intervals;

  auto addValue = [&](Value value) {
    if (!isTensorValue(value) || !isUBResidentTensor(value))
      return;
    Value canonical = findCanonicalValue(value, canonicalValues);
    if (intervals.count(canonical))
      return;

    if (Operation *owner = getValueDefiningOperation(canonical);
        isVirtualTensorOperation(owner))
      return;

    std::optional<uint64_t> bytes = getStaticTensorBytes(canonical.getType());
    if (!bytes) {
      estimate.reason = isa<UnrankedTensorType>(canonical.getType()) ||
                                (isa<RankedTensorType>(canonical.getType()) &&
                                 !cast<RankedTensorType>(canonical.getType())
                                      .hasStaticShape())
                            ? ResourceCostRejectReason::DynamicShape
                            : ResourceCostRejectReason::UnknownElementType;
      return;
    }
    const uint64_t begin = getDefinitionOrder(canonical, order);
    intervals.try_emplace(canonical,
                          IntervalState{canonical, begin, begin, *bytes});
  };

  for (Operation *operation : operations) {
    for (Value value : operation->getResults())
      addValue(value);
    for (Region &region : operation->getRegions()) {
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          addValue(argument);
    }
  }

  if (estimate.reason == ResourceCostRejectReason::DynamicShape ||
      estimate.reason == ResourceCostRejectReason::UnknownElementType)
    return estimate;

  for (Operation *operation : operations) {
    const uint64_t useOrder = order.lookup(operation);
    for (Value operand : operation->getOperands()) {
      Value canonical = findCanonicalValue(operand, canonicalValues);
      auto it = intervals.find(canonical);
      if (it != intervals.end())
        it->second.end = std::max(it->second.end, useOrder);
    }
  }

  for (Operation *loop : operations) {
    if (!isForLikeLoop(loop))
      continue;

    uint64_t loopEnd = order.lookup(loop);
    loop->getRegion(0).walk<WalkOrder::PreOrder>([&](Operation *nested) {
      loopEnd = std::max(loopEnd, order.lookup(nested));
    });
    loop->getRegion(0).walk<WalkOrder::PreOrder>([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        Value canonical = findCanonicalValue(operand, canonicalValues);
        auto it = intervals.find(canonical);
        if (it != intervals.end() && !isDefinedInside(canonical, loop))
          it->second.end = std::max(it->second.end, loopEnd);
      }
    });
  }

  llvm::SmallVector<uint64_t, 32> positions;
  positions.reserve(intervals.size() * 2);
  for (const auto &entry : intervals) {
    const IntervalState &state = entry.second;
    estimate.intervals.push_back(
        LiveTensorInterval{state.value, state.begin, state.end, state.bytes});
    positions.push_back(state.begin);
    if (state.end == std::numeric_limits<uint64_t>::max()) {
      estimate.overflow = true;
      estimate.reason = ResourceCostRejectReason::Overflow;
      return estimate;
    }
    positions.push_back(state.end + 1);
  }

  llvm::sort(positions);
  positions.erase(std::unique(positions.begin(), positions.end()),
                  positions.end());
  for (uint64_t position : positions) {
    llvm::SmallVector<const IntervalState *, 16> live;
    for (const auto &entry : intervals) {
      const IntervalState &state = entry.second;
      if (state.begin <= position && position <= state.end)
        live.push_back(&state);
    }
    llvm::sort(live, [](const IntervalState *lhs, const IntervalState *rhs) {
      if (lhs->bytes != rhs->bytes)
        return lhs->bytes > rhs->bytes;
      return lhs->begin < rhs->begin;
    });
    uint64_t liveBytes = 0;
    llvm::SmallVector<const IntervalState *, 16> selected;
    for (const IntervalState *state : live) {
      bool exclusive = false;
      for (const IntervalState *other : selected) {
        if (areMutuallyExclusive(state->value, other->value)) {
          exclusive = true;
          break;
        }
      }
      if (exclusive)
        continue;
      if (!checkedAdd(liveBytes, state->bytes, liveBytes)) {
        estimate.overflow = true;
        estimate.reason = ResourceCostRejectReason::Overflow;
        return estimate;
      }
      selected.push_back(state);
    }
    estimate.peakLiveBytes = std::max(estimate.peakLiveBytes, liveBytes);
  }

  estimate.known = true;
  estimate.reason = ResourceCostRejectReason::None;
  return estimate;
}

namespace {

ResourceCostRejectReason classifyUnknownTensorType(Type type) {
  if (isa<UnrankedTensorType>(type) ||
      (isa<RankedTensorType>(type) &&
       !cast<RankedTensorType>(type).hasStaticShape()))
    return ResourceCostRejectReason::DynamicShape;
  return ResourceCostRejectReason::UnknownElementType;
}

} // namespace

CubeByteEstimate cfg::estimatePeakCubeBytes(Operation *root) {
  CubeByteEstimate estimate;
  if (!root)
    return estimate;

  ResourceCostRejectReason failure = ResourceCostRejectReason::None;
  root->walk([&](Operation *operation) {
    Value lhs;
    Value rhs;
    Value accumulator;
    Value result;
    if (auto dot = dyn_cast<DotOp>(operation)) {
      lhs = dot.getA();
      rhs = dot.getB();
      accumulator = dot.getC();
      result = dot.getD();
    } else if (auto scaled = dyn_cast<DotScaledOp>(operation)) {
      lhs = scaled.getA();
      rhs = scaled.getB();
      accumulator = scaled.getC();
      result = scaled.getD();
    } else {
      return;
    }
    ++estimate.dotSiteCount;

    const auto record = [&](Value value, uint64_t &peak) {
      std::optional<uint64_t> bytes = getStaticTensorBytes(value.getType());
      if (bytes) {
        peak = std::max(peak, *bytes);
      } else if (failure == ResourceCostRejectReason::None) {
        failure = classifyUnknownTensorType(value.getType());
      }
    };
    // L0A holds the lhs (A matrix) tile, L0B the rhs (B matrix) tile, and
    // L0C the accumulator/result tiles, including loop-carried accumulation
    // across scf.for iterations (the acc operand reaches the next dot).
    record(lhs, estimate.peakL0ABytes);
    record(rhs, estimate.peakL0BBytes);
    record(accumulator, estimate.peakL0CBytes);
    record(result, estimate.peakL0CBytes);
  });

  if (estimate.dotSiteCount == 0 || failure == ResourceCostRejectReason::None) {
    estimate.known = true;
    estimate.reason = ResourceCostRejectReason::None;
    return estimate;
  }
  estimate.reason = failure;
  return estimate;
}

CandidateEvaluation
cfg::evaluateCandidateCost(const ResourceSnapshot &resources,
                           const CandidateCost &candidate,
                           const CostModelWeights &weights) {
  if (!resources.isKnown())
    return reject(candidate, ResourceCostRejectReason::UnknownResource, 0, 0,
                  &resources);

  std::optional<uint64_t> safeBudget = resources.getSafeUBBudget();
  if (!safeBudget)
    return reject(candidate, ResourceCostRejectReason::Overflow, 0, 0,
                  &resources);

  if (candidate.hasDynamicShape)
    return reject(candidate, ResourceCostRejectReason::DynamicShape,
                  *safeBudget, 0, &resources);
  if (candidate.hasUnknownResource || !candidate.hasPeakLiveBytes)
    return reject(candidate, ResourceCostRejectReason::UnknownResource,
                  *safeBudget, 0, &resources);
  if (candidate.plan.tensorizeFactor == 0 || candidate.plan.blockT == 0 ||
      candidate.plan.staticAxisFusionFactor == 0)
    return reject(candidate, ResourceCostRejectReason::InvalidCandidate,
                  *safeBudget, 0, &resources);
  if (candidate.estimatedPeakLiveBytes > *safeBudget)
    return reject(candidate, ResourceCostRejectReason::UBOverflow, *safeBudget,
                  0, &resources);

  // Cube-buffer checks: enforced only when the snapshot carries L0A/L0B/L0C
  // capacities and the candidate reports cube tile bytes. A candidate that
  // reports cube demand against a snapshot without L0 capacities is an
  // unknown resource and must not pass silently.
  if (candidate.hasPeakCubeBytes) {
    if (!resources.hasL0Budgets())
      return reject(candidate, ResourceCostRejectReason::UnknownResource,
                    *safeBudget, 0, &resources);
    const std::optional<uint64_t> safeL0A = resources.getSafeL0ABudget();
    const std::optional<uint64_t> safeL0B = resources.getSafeL0BBudget();
    const std::optional<uint64_t> safeL0C = resources.getSafeL0CBudget();
    if (!safeL0A || !safeL0B || !safeL0C)
      return reject(candidate, ResourceCostRejectReason::Overflow, *safeBudget,
                    0, &resources);
    if (candidate.peakL0ABytes > *safeL0A)
      return reject(candidate, ResourceCostRejectReason::L0AOverflow,
                    *safeBudget, 0, &resources);
    if (candidate.peakL0BBytes > *safeL0B)
      return reject(candidate, ResourceCostRejectReason::L0BOverflow,
                    *safeBudget, 0, &resources);
    if (candidate.peakL0CBytes > *safeL0C)
      return reject(candidate, ResourceCostRejectReason::L0COverflow,
                    *safeBudget, 0, &resources);
  }

  uint64_t gains = 0;
  uint64_t penalties = 0;
  if (!addWeightedChange(candidate.launchesBefore, candidate.launchesAfter,
                         weights.launchReduction, gains, penalties) ||
      !addWeightedChange(candidate.gmReadBytesBefore,
                         candidate.gmReadBytesAfter, weights.gmByte, gains,
                         penalties) ||
      !addWeightedChange(candidate.gmWriteBytesBefore,
                         candidate.gmWriteBytesAfter, weights.gmByte, gains,
                         penalties) ||
      !addWeightedChange(candidate.storeCountBefore, candidate.storeCountAfter,
                         weights.store, gains, penalties) ||
      !addWeightedChange(candidate.addressCalculationsBefore,
                         candidate.addressCalculationsAfter,
                         weights.addressCalculation, gains, penalties) ||
      !addWeightedChange(candidate.baselinePeakLiveBytes,
                         candidate.estimatedPeakLiveBytes, weights.liveByte,
                         gains, penalties) ||
      !addWeightedChange(candidate.tokenOnlyRepeatedBytesBefore,
                         candidate.tokenOnlyRepeatedBytesAfter,
                         weights.tokenOnlyRepeatedByte, gains, penalties) ||
      gains > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      penalties > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return reject(candidate, ResourceCostRejectReason::Overflow, *safeBudget, 0,
                  &resources);

  CandidateEvaluation evaluation;
  evaluation.candidate = candidate;
  evaluation.accepted = true;
  evaluation.reason = ResourceCostRejectReason::None;
  evaluation.benefitScore =
      static_cast<int64_t>(gains) - static_cast<int64_t>(penalties);
  evaluation.safeUBBudgetBytes = *safeBudget;
  if (resources.hasL0Budgets()) {
    evaluation.safeL0ABudgetBytes = resources.getSafeL0ABudget().value_or(0);
    evaluation.safeL0BBudgetBytes = resources.getSafeL0BBudget().value_or(0);
    evaluation.safeL0CBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }
  evaluation.requiredParallelPrograms = 0;
  evaluation.effectiveWorkPerProgram = 0;
  evaluation.physicalWavesBefore = 0;
  evaluation.physicalWavesAfter = 0;
  evaluation.persistentLoopTripsBefore = 0;
  evaluation.persistentLoopTripsAfter = 0;
  evaluation.tokenOnlyRepeatedBytesBefore =
      candidate.tokenOnlyRepeatedBytesBefore;
  evaluation.tokenOnlyRepeatedBytesAfter =
      candidate.tokenOnlyRepeatedBytesAfter;
  evaluation.remark = formatCandidateRemark(evaluation);
  return evaluation;
}

llvm::SmallVector<CandidatePlan>
cfg::enumerateCandidatePlans(llvm::ArrayRef<unsigned> tensorizeFactors,
                             llvm::ArrayRef<unsigned> blockTFactors,
                             llvm::ArrayRef<unsigned> staticAxisFusionFactors,
                             llvm::StringRef stableIdPrefix) {
  llvm::SmallVector<unsigned> tensorize;
  llvm::SmallVector<unsigned> blockT;
  llvm::SmallVector<unsigned> staticFusion;
  addCanonicalFactors(tensorizeFactors, tensorize);
  addCanonicalFactors(blockTFactors, blockT);
  addCanonicalFactors(staticAxisFusionFactors, staticFusion);

  llvm::SmallVector<CandidatePlan> plans;
  uint64_t ordinal = 0;
  for (unsigned tensorizeFactor : tensorize) {
    for (unsigned tokenFactor : blockT) {
      for (unsigned staticFactor : staticFusion) {
        CandidatePlan plan;
        plan.tensorizeFactor = tensorizeFactor;
        plan.blockT = tokenFactor;
        plan.staticAxisFusionFactor = staticFactor;
        plan.stableId =
            (llvm::Twine(stableIdPrefix) + ".t" + llvm::Twine(tensorizeFactor) +
             ".b" + llvm::Twine(tokenFactor) + ".f" + llvm::Twine(staticFactor))
                .str();
        plan.sourceOrdinal = ordinal++;
        plans.push_back(std::move(plan));
      }
    }
  }
  return plans;
}

void cfg::sortCandidateEvaluations(
    llvm::MutableArrayRef<CandidateEvaluation> values) {
  std::stable_sort(
      values.begin(), values.end(),
      [](const CandidateEvaluation &lhs, const CandidateEvaluation &rhs) {
        if (lhs.accepted != rhs.accepted)
          return lhs.accepted;
        if (lhs.benefitScore != rhs.benefitScore)
          return lhs.benefitScore > rhs.benefitScore;
        const CandidatePlan &left = lhs.candidate.plan;
        const CandidatePlan &right = rhs.candidate.plan;
        return std::tie(left.tensorizeFactor, left.blockT,
                        left.staticAxisFusionFactor, left.stableId,
                        left.sourceOrdinal) <
               std::tie(right.tensorizeFactor, right.blockT,
                        right.staticAxisFusionFactor, right.stableId,
                        right.sourceOrdinal);
      });
}

std::string cfg::formatCandidateRemark(const CandidateEvaluation &evaluation) {
  const CandidateCost &candidate = evaluation.candidate;
  std::string remark;
  llvm::raw_string_ostream stream(remark);
  stream << "resource-cost candidate=" << candidate.plan.stableId
         << " accepted=" << (evaluation.accepted ? "true" : "false")
         << " reason=" << getResourceCostRejectReasonName(evaluation.reason)
         << " score=" << evaluation.benefitScore
         << " tensorize_factor=" << candidate.plan.tensorizeFactor
         << " block_t=" << candidate.plan.blockT
         << " static_axis_fusion_factor="
         << candidate.plan.staticAxisFusionFactor
         << " logical_programs=" << candidate.logicalTasksBefore << "->"
         << candidate.logicalTasksAfter
         << " physical_blocks=" << candidate.actualProgramsBefore << "->"
         << candidate.actualProgramsAfter
         << " physical_waves=" << evaluation.physicalWavesBefore << "->"
         << evaluation.physicalWavesAfter << " legacy_auto_map="
         << (candidate.legacyAutoMapBefore ? "true" : "false") << "->"
         << (candidate.legacyAutoMapAfter ? "true" : "false")
         << " required_programs=" << evaluation.requiredParallelPrograms
         << " gm_read_bytes=" << candidate.gmReadBytesBefore << "->"
         << candidate.gmReadBytesAfter
         << " gm_write_bytes=" << candidate.gmWriteBytesBefore << "->"
         << candidate.gmWriteBytesAfter
         << " stores=" << candidate.storeCountBefore << "->"
         << candidate.storeCountAfter
         << " address_calculations=" << candidate.addressCalculationsBefore
         << "->" << candidate.addressCalculationsAfter
         << " peak_live_bytes=" << candidate.baselinePeakLiveBytes << "->"
         << candidate.estimatedPeakLiveBytes
         << " ub_budget_bytes=" << evaluation.safeUBBudgetBytes
         << " peak_l0a_bytes=" << candidate.peakL0ABytes
         << " peak_l0b_bytes=" << candidate.peakL0BBytes
         << " peak_l0c_bytes=" << candidate.peakL0CBytes
         << " l0_budget_bytes=" << evaluation.safeL0ABudgetBytes << "/"
         << evaluation.safeL0BBudgetBytes << "/"
         << evaluation.safeL0CBudgetBytes
         << " work_per_program=" << evaluation.effectiveWorkPerProgram
         << " persistent_loop_trips=" << evaluation.persistentLoopTripsBefore
         << "->" << evaluation.persistentLoopTripsAfter
         << " token_only_repeated_bytes="
         << evaluation.tokenOnlyRepeatedBytesBefore << "->"
         << evaluation.tokenOnlyRepeatedBytesAfter
         << " persistent=" << (candidate.persistent ? "true" : "false");
  return std::move(stream.str());
}

void cfg::emitCandidateRemark(Operation *anchor,
                              const CandidateEvaluation &evaluation) {
  if (anchor)
    anchor->emitRemark() << evaluation.remark;
}

std::string
cfg::formatProfilerCalibrationRecord(const ProfilerCalibrationRecord &record) {
  std::string formatted;
  llvm::raw_string_ostream stream(formatted);
  stream << "resource-cost-calibration schema=v1 kernel=" << record.kernelName
         << " profiler_artifact=" << record.profilerArtifact
         << " samples=" << record.sampleCount
         << " median_ns=" << record.medianNanoseconds
         << " p90_ns=" << record.p90Nanoseconds << " "
         << formatCandidateRemark(record.evaluation);
  return std::move(stream.str());
}

ResourceCostAnalysis::ResourceCostAnalysis(Operation *root,
                                           ResourceSnapshot resources)
    : resources(resources), liveBytes(estimatePeakLiveBytes(root)),
      cubeBytes(estimatePeakCubeBytes(root)) {}

CandidateEvaluation
ResourceCostAnalysis::evaluate(const CandidateCost &candidate) const {
  CandidateCost effectiveCandidate = candidate;
  if (!effectiveCandidate.hasPeakLiveBytes && liveBytes.known) {
    effectiveCandidate.hasPeakLiveBytes = true;
    effectiveCandidate.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
    effectiveCandidate.estimatedPeakLiveBytes = liveBytes.peakLiveBytes;
  }
  // Cube-buffer gating mirrors the unified-buffer contract: when the
  // snapshot carries L0 capacities, a candidate without its own cube
  // estimate inherits the baseline one, and an unanalyzable cube demand is
  // an unknown resource rather than a silent pass.
  if (resources.hasL0Budgets() && !effectiveCandidate.hasPeakCubeBytes) {
    if (cubeBytes.known) {
      effectiveCandidate.hasPeakCubeBytes = true;
      effectiveCandidate.peakL0ABytes = cubeBytes.peakL0ABytes;
      effectiveCandidate.peakL0BBytes = cubeBytes.peakL0BBytes;
      effectiveCandidate.peakL0CBytes = cubeBytes.peakL0CBytes;
    } else if (cubeBytes.reason == ResourceCostRejectReason::DynamicShape) {
      effectiveCandidate.hasDynamicShape = true;
    } else {
      effectiveCandidate.hasUnknownResource = true;
    }
  }
  if (!liveBytes.known) {
    if (liveBytes.reason == ResourceCostRejectReason::DynamicShape)
      effectiveCandidate.hasDynamicShape = true;
    else
      effectiveCandidate.hasUnknownResource = true;
  }
  CandidateEvaluation evaluation =
      evaluateCandidateCost(resources, effectiveCandidate);
  if (liveBytes.reason == ResourceCostRejectReason::Overflow) {
    evaluation.accepted = false;
    evaluation.reason = ResourceCostRejectReason::Overflow;
    evaluation.benefitScore = 0;
    evaluation.remark = formatCandidateRemark(evaluation);
  }
  return evaluation;
}
