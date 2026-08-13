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

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_CUBE_SCOPE_PASS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_CUBE_SCOPE_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// AddMultiBufferCubeScopePass for adding cube core intra-loop multi-buffer
// optimization driven by `ssbuffer.cubeBuffer = [group, role]` attributes on
// producers (currently hivm.hir.fixpipe, role=1) and consumers (currently
// memref.memory_space_cast, role=0). For each (group, role=1)/(group,
// role=0) pair inside the main_loop of a CUBE scope, this pass:
//   1. Allocates N memref.alloc buffers (N read from ssbuffer.cube_buf_count).
//   2. Wraps producer/consumer with an scf.if chain selecting buffer[i] based
//      on (iter_count % N).
class AddMultiBufferCubeScopePass
    : public PassWrapper<AddMultiBufferCubeScopePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferCubeScopePass)

  // Constructor
  AddMultiBufferCubeScopePass() = default;

  // Pass argument
  StringRef getArgument() const override {
    return "add_multi_buffer_cube_scope";
  }

  // Dependent dialects
  void getDependentDialects(DialectRegistry &registry) const override;

  // Run the pass
  void runOnOperation() override;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferCubeScopePass();

// Register the pass
void registerAddMultiBufferCubeScopePasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_CUBE_SCOPE_PASS_H