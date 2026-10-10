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
