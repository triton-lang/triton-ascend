//===- AtomicMemoryCostModel.h - Atomic workload pricing ------------------===//
//
// Atomic service pricing, kept out of StageCostModels so that StageCostModels
// only exposes a single entry point for it.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_ATOMICMEMORYCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_ATOMICMEMORYCOSTMODEL_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::ascend {

/// Price one atomic workload into `resources`. Returns false when no profile
/// slot matches, leaving the workload unpriced.
bool accumulateAtomicWorkload(StageResourceCycles &resources,
                              const AtomicWorkload &atomic,
                              const StageModeProfile &profile);

/// Price every atomic workload of a stage into `resources`.
void accumulateAtomicWorkloads(StageResourceCycles &resources,
                               llvm::ArrayRef<AtomicWorkload> atomics,
                               const StageModeProfile &profile);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_ATOMICMEMORYCOSTMODEL_H
