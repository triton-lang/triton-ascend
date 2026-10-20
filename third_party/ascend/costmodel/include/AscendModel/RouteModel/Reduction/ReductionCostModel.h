#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_COSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_COSTMODEL_H
#include "AscendModel/RouteModel/Reduction/ReductionProfile.h"
#include "AscendModel/RouteModel/Reduction/ReductionWorkload.h"
#include <optional>
namespace mlir::ascend {
enum class ReductionAxisKind { First, Middle, Tail };
struct FittedReductionEstimate {
  double cycles;
  ReductionAxisKind axis;
};
std::optional<FittedReductionEstimate>
estimateReductionCycles(const ReductionWorkload &, const ReductionProfile &,
                        ReductionMode, int64_t warps);

std::optional<double>
estimateTailAxisReductionCycles(const ReductionWorkload &work,
                                const ReductionCostProfile &profile,
                                ReductionMode mode, int64_t numWarps);
std::optional<double>
estimateFirstAxisReductionCycles(const ReductionWorkload &work,
                                 const ReductionProfile &profile,
                                 ReductionMode mode, int64_t warps);
std::optional<double>
estimateMiddleAxisReductionCycles(const ReductionWorkload &work,
                                  const ReductionProfile &profile,
                                  ReductionMode mode, int64_t warps);
} // namespace mlir::ascend
#endif
