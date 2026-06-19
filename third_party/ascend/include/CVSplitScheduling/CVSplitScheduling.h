#ifndef TRITON_CV_SPLIT_SCHEDULING_H
#define TRITON_CV_SPLIT_SCHEDULING_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include <memory>

#define GEN_PASS_DECL_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"

using namespace mlir;

#define GEN_PASS_DEF_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCVSplitSchedulingPass(
    const CVSplitSchedulingOptions &options = {});

} // namespace triton
} // namespace mlir

#endif // TRITON_CV_SPLIT_SCHEDULING_H
