//===- IndirectGatherMemoryCostModel.h -----------------------*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {

/// Owns the new indirect load/store formulas and applicability guards only.
class IndirectGatherMemoryCostModel final : public StageCostModel {
public:
  StageMemoryCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
