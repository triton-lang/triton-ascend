// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef ASCENDMODEL_SPECIALIZEDMEMORYMAIN_H
#define ASCENDMODEL_SPECIALIZEDMEMORYMAIN_H
#include "AscendModel/RouteModel/SpecializedMemoryCostModel.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::ascend {
/// Prove a complete, single-program, three-pointer DS source/sink envelope.
/// Returns no geometry for an internal target or an unowned auxiliary op.
std::optional<SpecializedMemoryGeometry>
proveSpecializedMemoryFragment(ModuleOp module);
} // namespace mlir::ascend
#endif
