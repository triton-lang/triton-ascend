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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_DYNAMIC_CV_AUTO_BLOCKIFY_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_DYNAMIC_CV_AUTO_BLOCKIFY_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#define GEN_PASS_DECL_DYNAMICCVAUTOBLOCKIFYPASS
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"

using namespace mlir;

#define GEN_PASS_DEF_DYNAMICCVAUTOBLOCKIFYPASS
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createDynamicCVAutoBlockifyPass(
    const DynamicCVAutoBlockifyPassOptions &options = {});
} // namespace triton
} // namespace mlir

namespace {
using namespace mlir;
using namespace triton;

class DynamicCVAutoBlockifyPass
    : public ::impl::DynamicCVAutoBlockifyPassBase<DynamicCVAutoBlockifyPass> {
public:
    explicit DynamicCVAutoBlockifyPass(const DynamicCVAutoBlockifyPassOptions &options);
    void runOnOperation() override;
};

} // namespace

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_DYNAMIC_CV_AUTO_BLOCKIFY_H