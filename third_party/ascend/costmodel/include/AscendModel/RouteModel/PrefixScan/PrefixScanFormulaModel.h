#ifndef ASCENDMODEL_ROUTEMODEL_PREFIXSCAN_PREFIXSCANFORMULAMODEL_H
#define ASCENDMODEL_ROUTEMODEL_PREFIXSCAN_PREFIXSCANFORMULAMODEL_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>

namespace mlir::ascend {

struct StageModeProfile;

struct PrefixScanFormulaDomain {
  std::string operation, dataType, branch;
  int64_t rank = 0, axis = 0, warps = 0;
};
struct PrefixScanFormula {
  std::string id;
  std::string projection = "legacy";
  std::vector<PrefixScanFormulaDomain> domains;
  std::map<std::string, double> terms;
};
struct PrefixScanEstimate {
  std::string formulaId;
  double cycles = 0.0;
};
llvm::Expected<std::vector<PrefixScanFormula>>
parsePrefixScanFormulas(const llvm::json::Array &formulas);
std::optional<PrefixScanEstimate>
estimatePrefixScanFormula(const StageWorkload::PrefixScanKey &scan,
                          unsigned warps, bool simd,
                          const std::vector<PrefixScanFormula> &formulas);
std::optional<PrefixScanEstimate>
estimateFittedScan(const StageWorkload::PrefixScanKey &scan,
                   const StageModeProfile &profile, bool simd,
                   bool allowTableFormulas);
} // namespace mlir::ascend
#endif
