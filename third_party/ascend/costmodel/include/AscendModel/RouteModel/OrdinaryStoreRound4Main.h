#ifndef ASCENDMODEL_ORDINARY_STORE_ROUND4_MAIN_H
#define ASCENDMODEL_ORDINARY_STORE_ROUND4_MAIN_H
#include "AscendModel/RouteModel/StageCostModels.h"
#include <optional>
#include <string>
namespace mlir::ascend {
struct OrdinaryStoreMainResult {
  std::optional<LogicalStage> auxiliaryStage;
  std::optional<double> targetSystemCycles;
  std::string reason = "candidate_default_off";
  std::string leaf;
};
OrdinaryStoreMainResult
ordinaryStoreMainCost(const LogicalStage &stage, const HardwareProfile &profile,
                      const StageImplementation &implementation,
                      bool operationOwnershipComplete);
} // namespace mlir::ascend
#endif
