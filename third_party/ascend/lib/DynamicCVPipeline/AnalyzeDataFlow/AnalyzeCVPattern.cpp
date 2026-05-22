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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

static constexpr const char *DEBUG_TYPE = "analyze-cv-pattern";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {

static constexpr const char *kAttrMainLoop = "ssbuffer.main_loop";
static constexpr const char *kAttrTransferId = "ssbuffer.transfer_id";
static constexpr const char *kAttrAddFromMatmul = "ssbuffer.add_from_matmul";

struct VectorMainLoopTransferPattern {
    scf::ForOp mainLoop;
    SmallVector<bufferization::ToTensorOp> transferToTensors;
};

static bool isOpAlive(Operation *op)
{
    return op && !op->hasTrait<OpTrait::IsTerminator>();
}

static bool isVectorScope(scope::ScopeOp scopeOp)
{
    auto scopeTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    return scopeTypeAttr && scopeTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool forOpHasMainLoopAttr(scf::ForOp forOp)
{
    if (forOp->hasAttr(kAttrMainLoop)) {
        return true;
    }
    Operation *terminator = forOp.getBody()->getTerminator();
    return terminator && terminator->hasAttr(kAttrMainLoop);
}

static SmallVector<bufferization::ToTensorOp> collectTransferTaggedToTensors(scf::ForOp forOp)
{
    SmallVector<bufferization::ToTensorOp> toTensorOps;
    forOp.walk([&](bufferization::ToTensorOp toTensorOp) {
        if (toTensorOp->hasAttr(kAttrTransferId)) {
            toTensorOps.push_back(toTensorOp);
        }
    });
    return toTensorOps;
}

static bool hasOnlyTaggedAddFromMatmulUsers(bufferization::ToTensorOp toTensorOp)
{
    bool sawLiveUser = false;
    for (Operation *user : toTensorOp.getResult().getUsers()) {
        if (!isOpAlive(user)) {
            continue;
        }

        auto addFOp = dyn_cast<arith::AddFOp>(user);
        if (!addFOp || !addFOp->hasAttr(kAttrAddFromMatmul)) {
            return false;
        }
        sawLiveUser = true;
    }
    return sawLiveUser;
}

static SmallVector<VectorMainLoopTransferPattern> collectVectorMainLoopTransferPatterns(scope::ScopeOp vecScope)
{
    SmallVector<VectorMainLoopTransferPattern> patterns;
    if (!vecScope || !isVectorScope(vecScope)) {
        return patterns;
    }

    vecScope.walk([&](scf::ForOp forOp) {
        if (!forOpHasMainLoopAttr(forOp)) {
            return;
        }

        SmallVector<bufferization::ToTensorOp> transferToTensors = collectTransferTaggedToTensors(forOp);
        if (transferToTensors.empty()) {
            return;
        }

        for (bufferization::ToTensorOp toTensorOp : transferToTensors) {
            if (!hasOnlyTaggedAddFromMatmulUsers(toTensorOp)) {
                return;
            }
        }

        patterns.push_back({forOp, std::move(transferToTensors)});
    });

    return patterns;
}

static bool checkCVPatternCondition(ModuleOp module)
{
    bool shouldFallback = false;

    module.walk([&](scope::ScopeOp scopeOp) {
        if (!isVectorScope(scopeOp)) {
            return;
        }

        SmallVector<VectorMainLoopTransferPattern> patterns =
            collectVectorMainLoopTransferPatterns(scopeOp);

        if (!patterns.empty()) {
            shouldFallback = true;
            LDBG("[INFO]: Found VectorMainLoopTransferPattern in VECTOR scope, triggering fallback.");
            for (const VectorMainLoopTransferPattern &pattern : patterns) {
                LDBG("  matched main_loop with " << pattern.transferToTensors.size()
                     << " transfer-tagged bufferization.to_tensor ops");
            }
        }
    });

    return shouldFallback;
}

} // namespace

void AnalyzeCVPatternPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Before AnalyzeCVPattern:\n" << module << "\n");

    if (checkCVPatternCondition(module)) {
        setFallbackAttr(module);
        signalPassFailure();
        return;
    }

    LDBG("After AnalyzeCVPattern:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeCVPatternPass()
{
    return std::make_unique<AnalyzeCVPatternPass>();
}

} // namespace triton
} // namespace mlir