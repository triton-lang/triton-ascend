#ifndef TRITON_ASCEND_PROTON_TO_HIVM_PASSES_H
#define TRITON_ASCEND_PROTON_TO_HIVM_PASSES_H

#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace mlir::triton::proton {

std::unique_ptr<Pass> createAssignAscendProtonScopeIdsPass();

std::unique_ptr<Pass>
createLowerAscendProtonToHIVMPass(int32_t dataSegmentBytes = 4096);

void registerAscendProtonToHIVMPasses();

} // namespace mlir::triton::proton

#endif // TRITON_ASCEND_PROTON_TO_HIVM_PASSES_H
