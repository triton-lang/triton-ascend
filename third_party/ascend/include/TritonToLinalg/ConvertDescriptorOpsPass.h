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

#ifndef TRITON_ADAPTER_CONVERSION_CONVERTDESCRIPTOROPSPASS_H
#define TRITON_ADAPTER_CONVERSION_CONVERTDESCRIPTOROPSPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_CONVERTDESCRIPTOROPS
#include "ascend/include/TritonToLinalg/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDescriptorOpsPass();

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;

// Lowers Triton tensor descriptor operations (tt.make_tensor_descriptor and
// tt.descriptor_load/store/gather/scatter/reduce) to plain Triton
// block-pointer, tt.load/tt.store, and scf.for based IR. This is a
// Triton-to-Triton desugaring pass, not a Triton-to-Linalg conversion: it
// runs as a preparation step before the main TritonToLinalg conversion.
class ConvertDescriptorOpsPass
    : public ::impl::ConvertDescriptorOpsBase<ConvertDescriptorOpsPass> {
public:
  ConvertDescriptorOpsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};

#endif // TRITON_ADAPTER_CONVERSION_CONVERTDESCRIPTOROPSPASS_H
