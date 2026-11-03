#ifndef TRITON_ASCEND_TRITONTOTENSORVIEW_PASSES_H
#define TRITON_ASCEND_TRITONTOTENSORVIEW_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createTritonToTensorViewPass();
#define GEN_PASS_REGISTRATION
#include "ascend/include/TritonToTensorView/Passes.h.inc"
} // namespace triton
} // namespace mlir

#endif
