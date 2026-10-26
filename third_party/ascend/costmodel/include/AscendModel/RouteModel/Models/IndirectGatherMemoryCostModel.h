//===- IndirectGatherMemoryCostModel.h -----------------------*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
#include "AscendModel/RouteModel/StageCostModel.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ascend {

/// Validated immutable parameter asset, shared by profiles and candidate
/// copies.
class IndirectMemoryParameters {
  llvm::StringMap<std::vector<double>> arrays;

public:
  std::string contentSha256;
  static llvm::Expected<std::shared_ptr<const IndirectMemoryParameters>>
  load(llvm::StringRef path);
  llvm::ArrayRef<double> get(llvm::StringRef key) const {
    return arrays.find(key)->second;
  }
};

/// Owns the new indirect load/store formulas and applicability guards only.
class IndirectGatherMemoryCostModel final : public StageCostModel {
public:
  StageMemoryCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;

  /// Complete single-AIV memory chain, including its fixed/helper budget.
  /// Never add this result to per-operation memory resource increments.
  static std::optional<double> nativeChainCost(const StagePartition &partition,
                                               const HardwareProfile &profile,
                                               const StageRoutePlan &plan);
  static double nativeChainCycles(const IndirectMemoryParameters &parameters,
                                  bool store, RankedTensorType type,
                                  int indexBits, int64_t warps, int64_t factor);
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_INDIRECTGATHERMEMORYCOSTMODEL_H
