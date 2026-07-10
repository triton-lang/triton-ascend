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

#ifndef TRITON_ADAPTER_FLOW_OPT_H
#define TRITON_ADAPTER_FLOW_OPT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

class FlowOptPass : public PassWrapper<FlowOptPass, OperationPass<ModuleOp>> {
public:
  FlowOptPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

  llvm::StringRef getArgument() const override { return "flow-opt"; }

private:
  // Create new IfOp with updated condition
  scf::IfOp createNewIfOpWithFlowOptCondition(scf::IfOp oldIfOp,
                                              Value newCondition);

  // Build the flow optimization condition
  Value buildFlowOptCondition(OpBuilder &builder, Location loc,
                              scf::IfOp firstIfOp, scf::ForOp forOp,
                              Value originalCondition);

  ControlFlowConditionInfo *info = nullptr;
};

std::unique_ptr<OperationPass<ModuleOp>> createFlowOptPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_FLOW_OPT_H
