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

#ifndef TRITON_ASCEND_ANALYZE_DATAFLOW_H
#define TRITON_ASCEND_ANALYZE_DATAFLOW_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// Pass for analyzing tensor args in main_loop forOps
class AnalyzeArgsPass : public PassWrapper<AnalyzeArgsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeArgsPass)

  AnalyzeArgsPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-args"; }
  llvm::StringRef getDescription() const override {
    return "Analyze tensor args in main_loop forOps";
  }
};

// Wrapper pass for AnalyzeDataFlow
class AnalyzeDataFlowPass : public PassWrapper<AnalyzeDataFlowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeDataFlowPass)

  AnalyzeDataFlowPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "analyze-data-flow"; }
  llvm::StringRef getDescription() const override {
    return "Analyze data flow and detect tensor args in different block_ids";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeArgsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeDataFlowPass();

void registerAnalyzeDataFlowPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_ANALYZE_DATAFLOW_H