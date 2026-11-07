#ifndef ASCENDMODEL_ORDINARY_LOAD_MAIN_H
#define ASCENDMODEL_ORDINARY_LOAD_MAIN_H
#include "AscendModel/RouteModel/OrdinaryStoreRound4Main.h"
namespace mlir::ascend {
// The result describes one matched Load target plus the unchanged auxiliary
// workload. Its structure is shared with Store; it carries no Store lease.
using OrdinaryLoadMainResult = OrdinaryStoreMainResult;
OrdinaryLoadMainResult ordinaryLoadMainCost(
    const LogicalStage &, const HardwareProfile &, const StageImplementation &,
    bool operationOwnershipComplete);
} // namespace mlir::ascend
#endif
