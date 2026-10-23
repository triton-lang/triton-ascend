#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_ANALYSIS_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_ANALYSIS_H
#include "AscendModel/RouteModel/Reduction/ReductionWorkload.h"
#include "mlir/IR/Operation.h"
#include <optional>
namespace mlir::ascend {
std::optional<ReductionWorkload> analyzeReduction(Operation *operation);
} // namespace mlir::ascend
#endif
