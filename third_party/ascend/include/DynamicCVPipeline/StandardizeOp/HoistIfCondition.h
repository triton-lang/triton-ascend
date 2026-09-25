/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DYNAMIC_CVPIPELINE_HOIST_IF_CONDITION_H
#define TRITON_ADAPTER_DYNAMIC_CVPIPELINE_HOIST_IF_CONDITION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::CVSplit {

/// Pass that transforms a conditional scf.if inside an scf.for into a
/// two-pass structure: the first pass collects valid loop indices into a
/// buffer, and the second pass executes the original if-body only for
/// those valid indices.
class HoistIfConditionPass
    : public PassWrapper<HoistIfConditionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistIfConditionPass);

  HoistIfConditionPass() = default;
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override;

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-hoist-if-condition";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createHoistIfConditionPass();

} // namespace mlir::triton::CVSplit

#endif // TRITON_ADAPTER_DYNAMIC_CVPIPELINE_HOIST_IF_CONDITION_H
