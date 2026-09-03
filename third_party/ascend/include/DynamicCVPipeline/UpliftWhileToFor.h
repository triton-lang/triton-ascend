//===- UpliftWhileToFor.h - Uplift scf.while to scf.for ---------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UPLIFT_WHILE_TO_FOR_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UPLIFT_WHILE_TO_FOR_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

/// Create a pass to uplift for-shaped scf.while loops to scf.for.
/// Same approach as mlir::hfusion::createUpliftWhileToForPass().
std::unique_ptr<Pass> createUpliftWhileToForPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UPLIFT_WHILE_TO_FOR_H
