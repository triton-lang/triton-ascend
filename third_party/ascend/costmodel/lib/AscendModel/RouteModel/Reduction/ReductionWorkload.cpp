#include "AscendModel/RouteModel/Reduction/ReductionWorkload.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
using namespace mlir::ascend;

bool ReductionWorkload::isFiniteAndNonNegative() const {
  return !kind.empty() && !dataType.empty() && !shape.empty() && axis >= 0 &&
         static_cast<size_t>(axis) < shape.size() &&
         llvm::all_of(
             shape, [](int64_t extent) { return extent > 0; }) &&
         std::isfinite(instances) && instances >= 0.0;
}

llvm::json::Object ReductionWorkload::toJSON() const {
  llvm::json::Array jsonShape;
  for (int64_t extent : shape)
    jsonShape.push_back(extent);
  return llvm::json::Object{{"kind", kind},
                            {"data_type", dataType},
                            {"shape", std::move(jsonShape)},
                            {"axis", axis},
                            {"instances_per_iteration", instances}};
}
