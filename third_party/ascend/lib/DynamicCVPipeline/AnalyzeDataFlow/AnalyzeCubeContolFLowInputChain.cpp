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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

static constexpr const char *DEBUG_TYPE = "analyze-cube-control-flow-input-chain";
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

static bool isCubeScope(scope::ScopeOp scopeOp)
{
    auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) {
        return false;
    }
    return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE;
}

static bool isControlFlowOp(Operation *op)
{
    return isa<scf::IfOp>(op) || isa<scf::ForOp>(op) || isa<scf::WhileOp>(op);
}

static bool hasDefiningChainWithReduceOrTensorSelect(Value val,
                                                     llvm::DenseSet<Value> &visited)
{
    if (visited.contains(val)) {
        return false;
    }
    visited.insert(val);

    Operation *defOp = val.getDefiningOp();
    while (defOp) {

        if (isa<linalg::ReduceOp>(defOp)) {
            LDBG("Found linalg.reduce in defining chain");
            return true;
        }

        if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
            if (isa<RankedTensorType>(selectOp.getType())) {
                LDBG("Found arith.select with tensor result in defining chain");
                return true;
            }
        }

        if (defOp->getNumOperands() == 0) {
            break;
        }

        for (Value operand : defOp->getOperands()) {
            if (hasDefiningChainWithReduceOrTensorSelect(operand, visited)) {
                return true;
            }
        }
        
        break;
    }

    return false;
}

static bool checkControlFlowOpInputs(Operation *cfOp)
{
    llvm::DenseSet<Value> visited;

    auto checkOperands = [&](ValueRange operands) {
        for (Value operand : operands) {
            if (hasDefiningChainWithReduceOrTensorSelect(operand, visited)) {
                return true;
            }
        }
        return false;
    };

    if (checkOperands(cfOp->getOperands())) {
        return true;
    }

    return false;
}

bool checkCubeControlFlowInputChain(ModuleOp module)
{
    bool shouldReturn = false;

    module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
        if (!isCubeScope(scopeOp)) {
            return WalkResult::advance();
        }
        LDBG("Found CUBE scope");

        scopeOp.walk([&](Operation *op) -> WalkResult {
            if (!isControlFlowOp(op)) {
                return WalkResult::advance();
            }
            LDBG("Found control flow op in CUBE scope");

            if (checkControlFlowOpInputs(op)) {
                shouldReturn = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        if (shouldReturn) {
            return WalkResult::interrupt();
        }

        return WalkResult::advance();
    });

    return shouldReturn;
}

} // namespace

void AnalyzeCubeControlFlowInputChainPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter AnalyzeCubeControlFlowInputChainPass.");

    if (checkCubeControlFlowInputChain(module)) {
        setFallbackAttr(module);
        signalPassFailure();
        return;
    }

    LDBG("Exit AnalyzeCubeControlFlowInputChainPass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeCubeContolFLowInputChainPass()
{
  return std::make_unique<AnalyzeCubeControlFlowInputChainPass>();
}

} // namespace triton
} // namespace mlir