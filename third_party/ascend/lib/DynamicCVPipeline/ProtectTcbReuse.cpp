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

#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "protect-tcb-reuse"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
class ProtectTcbReusePass
    : public PassWrapper<ProtectTcbReusePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectTcbReusePass)
  StringRef getArgument() const override { return "protect-tcb-reuse"; }
  StringRef getDescription() const override {
    return "Protect TCB reuse after Vector computation and output stores";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    if (CVPipeline::hasFallbackAttr(getOperation())) {
      return;
    }
    OpPassManager pipeline(ModuleOp::getOperationName());
    pipeline.addPass(createProtectVFtcbResusePass());
    pipeline.addPass(createProtectSTtcbResusePass());
    if (failed(runPipeline(pipeline, getOperation()))) {
      LOG_DEBUG("tcb reuse protection pipeline did not complete");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectTcbReusePass() {
  return std::make_unique<ProtectTcbReusePass>();
}

void mlir::triton::registerProtectTcbReusePasses() {
  registerPass(createProtectTcbReusePass);
  registerPass(createProtectVFtcbResusePass);
  registerPass(createProtectSTtcbResusePass);
}
