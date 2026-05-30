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

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/BuiltinOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "pre-check-scope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

void PreCheckScopePass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<scope::ScopeDialect>();
}

void PreCheckScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    scope::ScopeOp firstScopeOp = nullptr;

    module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
        firstScopeOp = scopeOp;
        return WalkResult::interrupt();
    });

    if (!firstScopeOp) {
        LDBG("The scope.scope operation is not found, passed.");
        return;
    }

    LDBG("SSBUFFER will be skipped because the scope.scope operation was found, "
        "which indicating that it has been optimized for the Ascend platform.");
    signalPassFailure();
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckScopePass()
{
    return std::make_unique<PreCheckScopePass>();
}

} // namespace triton
} // namespace mlir
