#ifndef ASCENDMODEL_ROUTEMODEL_REDUCTION_PROFILE_H
#define ASCENDMODEL_ROUTEMODEL_REDUCTION_PROFILE_H
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>
namespace mlir::ascend {
enum class ReductionMode { SIMD, SIMT };
/// Parsed expression tree for first- and middle-axis formulas, including
/// numeric coefficients, feature references and operators.
struct ReductionExpression {
  enum class Kind { Constant, Feature, Operator };
  Kind kind = Kind::Constant;
  double constant = 0;
  std::string name;
  std::vector<ReductionExpression> arguments;
};
struct ReductionSegment {
  std::string id;
  int64_t laneWidth = 1;
  ReductionExpression condition;
  ReductionExpression expression;
};
/// A first- or middle-axis formula with its supported warps and conditional
/// segments. Parsed when the profile is loaded; evaluated for each workload.
struct ReductionFunction {
  std::vector<int64_t> supportedWarps;
  std::vector<ReductionSegment> segments;
  std::string layoutPolicy;
};
/// Target-calibrated numeric parameters for tail-axis tt.reduce formulas.
/// Formula structure is implemented in C++; this table stores only
/// coefficients. First- and middle-axis formulas instead use ReductionFunction
/// expression trees. Keys are stable formula/route identifiers; keeping values
/// in the hardware profile allows recalibration without changing the evaluator
/// source.
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
  const ReductionFunction *findFirstAxis(ReductionMode mode, llvm::StringRef op,
                                         llvm::StringRef dtype) const;
  const ReductionFunction *findMiddleAxis(ReductionMode mode,
                                          llvm::StringRef op,
                                          llvm::StringRef dtype) const;
  llvm::StringRef getTarget() const { return target; }
  llvm::StringRef getContentHash() const { return contentHash; }

private:
  std::string target, contentHash;
  ReductionCostProfile simd, simt;
  llvm::StringMap<ReductionFunction> firstAxisFunctions, middleAxisFunctions;
};
} // namespace mlir::ascend
#endif
