#ifndef ASCENDMODEL_REDUCTION_LAYOUT_MODEL_H
#define ASCENDMODEL_REDUCTION_LAYOUT_MODEL_H
#include "AscendModel/RouteModel/Reduction/ReductionWorkload.h"
#include "llvm/ADT/StringMap.h"
#include <array>
namespace mlir::ascend {
struct MiddleAxisGeometry {
  int64_t outer, reduction, inner;
};
struct ReductionLayoutProjection {
  std::array<int64_t, 3> sizePerThread, threadsPerWarp, warpsPerCTA;
  int64_t route;
};
std::optional<MiddleAxisGeometry>
normalizeMiddleAxis(const ReductionWorkload &);
std::optional<ReductionLayoutProjection>
projectReductionLayout(const MiddleAxisGeometry &, int64_t bits, int64_t warps);
llvm::StringMap<double> middleAxisFeatures(const MiddleAxisGeometry &,
                                           const ReductionLayoutProjection &,
                                           int64_t warps);
} // namespace mlir::ascend
#endif
