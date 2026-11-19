// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ascend/include/TritonToLinalg/ExecutionPolicy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::triton;

static bool isContraction(linalg::LinalgOp op) {
  return linalg::isaContractionOpInterface(op);
}

bool mlir::triton::canLookAhead(Operation *op) {
  if (!op->getNumResults() || op->hasTrait<OpTrait::IsTerminator>())
    return false;

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (!linalgOp.hasPureTensorSemantics() || isContraction(linalgOp) ||
        !isMemoryEffectFree(op))
      return false;

    // Clone structured computations as a unit. Check the payload for effects
    // and speculation safety, including any nested operation.
    return !op->walk([&](Operation *nested) {
                if (nested == op || isa<linalg::YieldOp>(nested))
                  return WalkResult::advance();
                return nested->getNumRegions() == 0 &&
                               isMemoryEffectFree(nested) &&
                               isSpeculatable(nested)
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              }).wasInterrupted();
  }

  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         isSpeculatable(op);
}

static bool isNumericTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getElementType().isIntOrIndexOrFloat();
}

ExecutionKind mlir::triton::classifyExecution(Operation *op,
                                              ExecutionMode mode) {
  if (mode != ExecutionMode::SIMD)
    return ExecutionKind::Unknown;

  // Payload ops describe the enclosing structured computation. Classify the
  // enclosing op rather than interpreting scalar payload types as Scalar.
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (!isa<ModuleOp, func::FuncOp, scf::ForOp, scf::IfOp>(parent))
      return ExecutionKind::Unknown;

  // The TA/backend SIMD contract classifies structured Linalg computations
  // as Vector, with contractions such as matmul assigned to Cube.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return isContraction(linalgOp) ? ExecutionKind::Cube
                                   : ExecutionKind::Vector;

  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) ||
      isa<arith::ConstantOp>(op) || !op->hasTrait<OpTrait::Elementwise>())
    return ExecutionKind::Unknown;

  if (llvm::any_of(op->getOperandTypes(), isNumericTensor) ||
      llvm::any_of(op->getResultTypes(), isNumericTensor))
    return ExecutionKind::Vector;

  auto isIntOrIndex = [](Type type) { return type.isIntOrIndex(); };
  if (isa<arith::ArithDialect>(op->getDialect()) && op->getNumOperands() &&
      op->getNumResults() &&
      llvm::all_of(op->getOperandTypes(), isIntOrIndex) &&
      llvm::all_of(op->getResultTypes(), isIntOrIndex))
    return ExecutionKind::Scalar;

  return ExecutionKind::Unknown;
}
