//===- AtomicMemoryCostModel.cpp - Atomic workload pricing ----------------===//

#include "AscendModel/RouteModel/AtomicMemoryCostModel.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace mlir::ascend {

namespace {

/// Piecewise-linear interpolation over a positive-integer keyed curve. The
/// endpoints clamp, matching the measured-curve tables used elsewhere in the
/// profile.
double interpolateAtomicBody(const std::map<int64_t, double> &curve,
                             double elements, double fallback) {
  if (curve.empty() || elements <= 0.0)
    return fallback;
  const int64_t point = static_cast<int64_t>(elements);
  auto upper = curve.lower_bound(point);
  if (upper == curve.begin())
    return upper->second;
  if (upper == curve.end())
    return std::prev(upper)->second;
  if (upper->first == point)
    return upper->second;
  const auto lower = std::prev(upper);
  const double fraction = (elements - static_cast<double>(lower->first)) /
                          static_cast<double>(upper->first - lower->first);
  return lower->second + fraction * (upper->second - lower->second);
}

} // namespace

bool accumulateAtomicWorkload(StageResourceCycles &resources,
                              const AtomicWorkload &atomic,
                              const StageModeProfile &profile) {
  // The address class selects the slot. GM bitwise atomics have no native
  // hardware instruction: the HIVM normalization expands and/or/xor into a
  // locked read-modify-write, so the measured rate already carries that cost
  // and the generic lookup below applies without a special case.
  const llvm::StringRef addressClass =
      atomic.addressDependsOnLoadedIndex
          ? "loaded_index"
          : (atomic.provenContiguousRunWidth > 1.0 ? "contiguous"
                                                   : "unknown_address");
  auto rate =
      profile.atomicRates.find(atomic.profileKey() + "." + addressClass.str());
  if (rate == profile.atomicRates.end())
    rate = profile.atomicRates.find(addressClass);
  if (rate == profile.atomicRates.end())
    rate = profile.atomicRates.find(atomic.profileKey());
  if (rate == profile.atomicRates.end())
    rate = profile.atomicRates.find("default");
  if (rate == profile.atomicRates.end())
    return false;

  const StageAtomicRate &atomicRate = rate->second;
  const double linearBase =
      atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
      atomic.logicalElements / atomicRate.logicalElementsPerCycle;
  const double base =
      interpolateAtomicBody(atomicRate.bodySystemCyclesByLogicalElements,
                            atomic.logicalElements, linearBase);
  const double singleProgramRate =
      atomicRate.singlePhysicalProgramLogicalElementsPerCycle > 0.0
          ? atomicRate.singlePhysicalProgramLogicalElementsPerCycle
          : atomicRate.logicalElementsPerCycle;
  const double singleProgramBase =
      atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
      atomic.logicalElements / singleProgramRate;
  const double contention =
      atomic.contentionUnknown ? atomicRate.unknownContentionMultiplier : 1.0;
  resources.atomic += base * contention;
  resources.singlePhysicalProgramAtomic += singleProgramBase * contention;
  resources.hasSinglePhysicalProgramAtomicRate |=
      atomicRate.singlePhysicalProgramLogicalElementsPerCycle > 0.0;
  if (atomic.resultUsed) {
    // A measured used-result delta (read-back and critical section) overrides
    // the small dependency latency. It is additive on top of the unused cost,
    // so a dedicated `<slot>.result_used` entry carries only the delta.
    auto usedRate = profile.atomicRates.find(
        atomic.profileKey() + "." + addressClass.str() + ".result_used");
    if (usedRate == profile.atomicRates.end())
      usedRate = profile.atomicRates.find(atomic.profileKey() + ".result_used");
    if (usedRate != profile.atomicRates.end()) {
      const StageAtomicRate &used = usedRate->second;
      const double delta =
          atomic.logicalOperationInstances * used.operationStartupCycles +
          atomic.logicalElements / used.logicalElementsPerCycle;
      resources.atomic += delta * contention;
      resources.singlePhysicalProgramAtomic += delta * contention;
    } else {
      resources.atomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
      resources.singlePhysicalProgramAtomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
    }
  }
  return true;
}

void accumulateAtomicWorkloads(StageResourceCycles &resources,
                               llvm::ArrayRef<AtomicWorkload> atomics,
                               const StageModeProfile &profile) {
  // A float min/max atomic is lowered by Triton into a signed/unsigned integer
  // pair (max+umax or min+umin) that touches the same addresses. The target
  // executes it as one float atomic, so price the pair as a single f32 atomic
  // here instead of twice. This is a costing-time fold only; the IR is left
  // untouched.
  const size_t count = atomics.size();
  std::vector<bool> consumed(count, false);
  auto sameTarget = [](const AtomicWorkload &a, const AtomicWorkload &b) {
    return a.logicalElements == b.logicalElements &&
           a.addressDependsOnLoadedIndex == b.addressDependsOnLoadedIndex &&
           a.provenContiguousRunWidth == b.provenContiguousRunWidth;
  };
  auto isSigned = [](const std::string &kind) {
    return kind == "max" || kind == "min";
  };
  auto splitPartner = [&](const AtomicWorkload &a, const AtomicWorkload &b) {
    if (!sameTarget(a, b))
      return false;
    const bool maxPair = (a.kind == "max" && b.kind == "umax") ||
                         (a.kind == "umax" && b.kind == "max");
    const bool minPair = (a.kind == "min" && b.kind == "umin") ||
                         (a.kind == "umin" && b.kind == "min");
    return maxPair || minPair;
  };

  for (size_t i = 0; i < count; ++i) {
    if (consumed[i])
      continue;
    const AtomicWorkload &a = atomics[i];
    size_t partner = count;
    for (size_t j = i + 1; j < count; ++j) {
      if (!consumed[j] && splitPartner(a, atomics[j])) {
        partner = j;
        break;
      }
    }
    if (partner == count) {
      accumulateAtomicWorkload(resources, a, profile);
      consumed[i] = true;
      continue;
    }
    AtomicWorkload fused = a;
    fused.kind = isSigned(a.kind) ? a.kind : atomics[partner].kind;
    fused.dataType = "f32";
    accumulateAtomicWorkload(resources, fused, profile);
    consumed[i] = true;
    consumed[partner] = true;
  }
}

} // namespace mlir::ascend
