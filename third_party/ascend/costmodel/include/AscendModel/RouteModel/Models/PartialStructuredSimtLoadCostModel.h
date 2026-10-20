#ifndef ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADCOSTMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADCOSTMODEL_H
#include "AscendModel/RouteModel/StageCostModel.h"
namespace mlir::ascend {
class PartialStructuredSimtLoadCostModel final : public StageCostModel {
public:
  StageMemoryCost cost(const LogicalStage &, const HardwareProfile &,
                       const StageImplementation &) const override;
};
} // namespace mlir::ascend
#endif
