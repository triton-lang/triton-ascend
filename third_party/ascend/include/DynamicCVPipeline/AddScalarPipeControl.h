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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_SCALAR_PIPE_CONTROL_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_ADD_SCALAR_PIPE_CONTROL_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// Wraps every PIPE_S flag=15 synchronization inside the body of the
// innermost main_loop (scf.for / scf.while carrying ssbuffer.main_loop) into
// an `scf.if` guarded by a constant `true` condition. The guard is a
// placeholder: a later step replaces the constant with a real condition to
// make the scalar pipe synchronization conditional.
class AddScalarPipeControlPass
    : public PassWrapper<AddScalarPipeControlPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddScalarPipeControlPass);

  AddScalarPipeControlPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "add-scalar-pipe-control";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddScalarPipeControlPass();
void registerAddScalarPipeControlPasses();

} // namespace triton
} // namespace mlir

#endif
