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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse.h"
#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse/Utils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
static bool hasVFBeforeMainLoop(scf::ForOp outerLoop,
                                Operation *firstMainLoop) {
  auto *firstMainLoopInOuterLoop =
      CVPipeline::getAncestorInBlock(firstMainLoop, outerLoop.getBody());
  return llvm::any_of(llvm::make_range(outerLoop.getBody()->begin(),
                                       firstMainLoopInOuterLoop->getIterator()),
                      [](Operation &bodyOperation) {
                        return CVPipeline::isTensorComputeOp(&bodyOperation);
                      });
}

class ProtectVFtcbResusePass
    : public PassWrapper<ProtectVFtcbResusePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectVFtcbResusePass)
  StringRef getArgument() const override { return "protect-vf-tcb-reuse"; }
  StringRef getDescription() const override {
    return "Protect pre-loop Vector computation before TCB reuse";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    tcbReuse::protectTcbReuse(getOperation(), "vf tcb reuse",
                              hivm::PIPE::PIPE_V, hasVFBeforeMainLoop);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectVFtcbResusePass() {
  return std::make_unique<ProtectVFtcbResusePass>();
}
