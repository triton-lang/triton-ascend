#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_WORKLOAD_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_WORKLOAD_H
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <string>
#include <vector>
namespace mlir::ascend {
/// Route-independent semantic description of one dynamically executed
/// tt.reduce.  Mode-specific instruction/cycle costs are deliberately not
/// stored here: they are selected later from the target profile.
struct ReductionWorkload {
  std::string kind;
  std::string dataType;
  std::vector<int64_t> shape;
  int64_t axis = -1;
  double instances = 1.0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

} // namespace mlir::ascend
#endif
