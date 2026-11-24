#ifndef ASCENDMODEL_ROUTEMODEL_PREFIXSCAN_PREFIXSCANFORMULAMODEL_H
#define ASCENDMODEL_ROUTEMODEL_PREFIXSCAN_PREFIXSCANFORMULAMODEL_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <optional>

namespace mlir::ascend {

struct StageModeProfile;

/// Numeric split used by OR/AND/XOR formula domains.
struct PrefixScanFormulaCondition {
  std::string feature;
  double threshold = 0.0;
  /// true: feature <= threshold; false: feature > threshold.
  bool lessEqual = true;
};

/// Shared domain fields for arithmetic and bitwise scan formulas.
struct PrefixScanFormulaDomain {
  std::string operation, dataType, branch;
  int64_t rank = 0, axis = 0, warps = 0;
  /// Bitwise only: nullopt shares the domain between both scan directions.
  std::optional<bool> reverse;
  /// Bitwise only: every numeric condition must match.
  std::vector<PrefixScanFormulaCondition> conditions;
};
struct PrefixScanFormula {
  std::string id;
  /// Bitwise fits use source_work; arithmetic fits use legacy or row_major.
  std::string projection = "legacy";
  std::vector<PrefixScanFormulaDomain> domains;
  std::map<std::string, double> terms;
};
struct PrefixScanEstimate {
  std::string formulaId;
  double cycles = 0.0;
};
/// Read arithmetic or bitwise scan formulas.
/// Expand bitwise dispatch trees once for both SIMD and SIMT.
llvm::Expected<std::vector<PrefixScanFormula>>
parsePrefixScanFormulas(const llvm::json::Value &profile);
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
