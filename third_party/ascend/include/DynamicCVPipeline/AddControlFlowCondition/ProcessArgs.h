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

#ifndef TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#define TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

struct ControlFlowConditionInfo;

// For each shared iter_arg, we need to track:
// - Which block_ids use it
// - Who is the owner (first block_id in order)
// - For each non-owner block, what new iter_arg index to use
struct SharedArgInfo {
  int argIndex;
  Value iterArg;
  int ownerBlockId;
  int newArgIndex;
  int nonOwnerBlockId;

  SharedArgInfo(int arg, int owner, int newIdx, int nonOwner)
      : argIndex(arg), iterArg(Value()), ownerBlockId(owner),
        newArgIndex(newIdx), nonOwnerBlockId(nonOwner) {}
};

class ProcessArgsPass
    : public PassWrapper<ProcessArgsPass, OperationPass<ModuleOp>> {
public:
  ProcessArgsPass() = default;

  void runOnOperation() override;

  LogicalResult processSharedIterArgs(ModuleOp module);

  void setConditionInfo(ControlFlowConditionInfo *info_) { info = info_; }

  llvm::StringRef getArgument() const override { return "process-args"; }

  ControlFlowConditionInfo *info = nullptr;
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass();

} // namespace triton
} // namespace mlir
#endif // TRITON_ASCEND_SSBUF_PROCESS_ARGS_FOR_CONTROL_FLOW_H
