//===- StageCostModel.h - Interface for one Stage cost model ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include <optional>

namespace mlir::ascend {

/// Optional replacements for the legacy indirect-memory resource prices.
/// Values are per-iteration increments; nullopt retains the existing estimate.
struct StageMemoryCost {
  std::optional<double> load;
  std::optional<double> store;
  const char *loadPricing = nullptr;
  const char *storePricing = nullptr;
  // SIMD native target loop, including generated UB traffic and completion.
  // Only the proven exclusive predicate is covered; never erase producers.
  bool nativeSimdEnvelope = false;
  Operation *coveredPredicate = nullptr;
  Operation *coveredMemory = nullptr;
  std::optional<std::pair<double, double>> densityRange;
};

/// Extension point for newly calibrated Stage resources. Existing workload,
/// latency and SuperBlock accounting remain in StageCostModels.cpp.
class StageCostModel {
public:
  virtual ~StageCostModel() = default;
  virtual StageMemoryCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const = 0;
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H
