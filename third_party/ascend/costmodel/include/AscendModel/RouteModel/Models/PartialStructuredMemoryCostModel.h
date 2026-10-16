#ifndef ASCENDMODEL_PARTIALSTRUCTUREDMEMORYCOSTMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDMEMORYCOSTMODEL_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {
class PartialStructuredMemoryCostModel final : public StageCostModel {
public:
  StageMemoryCost cost(const LogicalStage &, const HardwareProfile &,
                       const StageImplementation &) const override;
};
} // namespace mlir::ascend
#endif
