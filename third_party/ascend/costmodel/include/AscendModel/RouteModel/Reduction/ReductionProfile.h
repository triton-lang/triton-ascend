#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_PROFILE_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_PROFILE_H
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>
namespace mlir::ascend {
enum class ReductionMode { SIMD, SIMT };
/// Target-calibrated numeric parameters for tail-axis tt.reduce formulas.
/// Keys are stable formula/route identifiers; keeping values in the hardware
/// profile allows recalibration without changing the evaluator source.
struct ReductionCostProfile {
  llvm::StringMap<std::vector<double>> parameters;

  bool isValid() const;
};

llvm::Expected<ReductionCostProfile>
parseReductionParameters(const llvm::json::Object &parameters);
class ReductionProfile {
public:
  static llvm::Expected<ReductionProfile> loadFromFile(llvm::StringRef path);
  const ReductionCostProfile &getModeProfile(ReductionMode mode) const;
  llvm::StringRef getTarget() const { return target; }
  llvm::StringRef getContentHash() const { return contentHash; }

private:
  std::string target, contentHash;
  ReductionCostProfile simd, simt;
};
} // namespace mlir::ascend
#endif
