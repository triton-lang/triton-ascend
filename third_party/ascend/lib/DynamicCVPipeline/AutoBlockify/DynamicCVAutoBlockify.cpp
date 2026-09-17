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

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "ascend/include/DynamicCVPipeline/DynamicCVAutoBlockify.h"
#include "ascend/include/DynamicCVPipeline/AutoBlockify/AutoBlockifyParallelLoop.h"
#include "ascend/include/DynamicCVPipeline/AutoBlockify/TritonGridArgsToHIVMOp.h"

static constexpr const char *DEBUG_TYPE = "dynamic-cv-auto-blockify";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

DynamicCVAutoBlockifyPass::DynamicCVAutoBlockifyPass(
    const DynamicCVAutoBlockifyPassOptions &options)
    : DynamicCVAutoBlockifyPassBase(options) {}

void DynamicCVAutoBlockifyPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter DynamicCVAutoBlockify pass\n" << module);

    PassManager pm(&getContext(), module.getOperationName());

    pm.addPass(createTritonGridArgsToHIVMOpPass());
    
    AutoBlockifyParallelLoopPassOptions autoBlockifyOptions;
    autoBlockifyOptions.aicoreNum = this->aicoreNum;
    pm.addPass(createAutoBlockifyParallelLoopPass(autoBlockifyOptions));

    if (failed(runPipeline(pm, module))) {
        signalPassFailure();
    }

    LDBG("Exit DynamicCVAutoBlockify pass.\n" << module);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createDynamicCVAutoBlockifyPass(
    const DynamicCVAutoBlockifyPassOptions &options)
{
    return std::make_unique<DynamicCVAutoBlockifyPass>(options);
}