#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_COSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_COSTMODEL_H
#include "AscendModel/RouteModel/Reduction/ReductionProfile.h"
#include "AscendModel/RouteModel/Reduction/ReductionWorkload.h"
#include <optional>
namespace mlir::ascend {
std::optional<double>
estimateTailAxisReductionCycles(const ReductionWorkload &work,
                                const ReductionCostProfile &profile,
                                ReductionMode mode, int64_t numWarps);
std::optional<double>
estimateFirstAxisReductionCycles(const ReductionWorkload &work,
                                 const ReductionProfile &profile,
                                 ReductionMode mode, int64_t warps);
} // namespace mlir::ascend
#endif
