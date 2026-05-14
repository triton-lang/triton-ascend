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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "fuse-adotbaddc"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg << "\n")
    
using namespace mlir;
using namespace triton;
using namespace mlir::triton;
namespace mlir {
namespace triton {
class FuseAdotBaddCPass : public PassWrapper<FuseAdotBaddCPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAdotBaddCPass)

    FuseAdotBaddCPass() = default;
    void runOnOperation() override;

    llvm::StringRef getArgument() const final {
      return "fuse-adotbaddc";
    }

    struct FuseInfo {
        linalg::MatmulOp matmul;
        arith::AddFOp addf;
        Value other;
    };

private:
    bool canFuse(FuseInfo &fuseInfo) {
        // Get the defining operation of 'other'
        Operation *defOp = fuseInfo.other.getDefiningOp();
        if (!defOp) {
            LOG_DEBUG("Other operand is not defined by any operation, cannot fuse.");
            return false;
        }

        // Check the addf's result is used by other matmul.
        // TODO: this can be remove after bishengir-compile support fixpipe to L1.
        auto addfResult = fuseInfo.addf.getResult();
        for (auto user : addfResult.getUsers()) {
            if (auto userMatmul = dyn_cast<linalg::MatmulOp>(user)) {
                return false;
            }
        }
        
        // Check if defOp dominates matmul
        auto matmul = fuseInfo.matmul;
        DominanceInfo dominance(matmul->getParentOp());
        if (!dominance.properlyDominates(defOp, matmul)) {
            LOG_DEBUG("Defining operation does not dominate matmul, cannot fuse.");
            return false;
        }
        
        // Case 1: other is result of a fill operation with zero value
        if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
            // FillOp has structure: fill ins(%value) outs(%tensor)
            // The first operand is the scalar fill value
            auto operands = fillOp->getOperands();
            if (!operands.empty()) {
                Value fillValue = operands[0];
                if (auto constOp = fillValue.getDefiningOp<arith::ConstantOp>()) {
                    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
                        return floatAttr.getValue().isZero();
                    }
                }
            }
            LOG_DEBUG("FillOp does not have a zero constant operand, cannot fuse.");
            return false;
        }
        
        // Case 2: other is result of a broadcast operation that broadcasts N-sized shape to 2D
        if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(defOp)) {
            // BroadcastOp.getInput() returns the input value, check its type
            Value inputValue = broadcastOp.getInput();
            
            auto inputType = inputValue.getType();
            auto outputType = fuseInfo.other.getType();
            
            if (auto inputTensor = dyn_cast<RankedTensorType>(inputType)) {
                if (auto outputTensor = dyn_cast<RankedTensorType>(outputType)) {
                    // Input should be 1D and output should be 2D
                    return inputTensor.getRank() == 1 && outputTensor.getRank() == 2;
                }
            }
            return false;
        }
        
        return false;
    }

    void performFusion(FuseInfo &fuseInfo) {
        OpBuilder builder(fuseInfo.addf);
        builder.setInsertionPoint(fuseInfo.matmul);
        auto newMatmul = builder.create<linalg::MatmulOp>(
            fuseInfo.matmul.getLoc(),
            ValueRange{fuseInfo.matmul.getInputs()[0], fuseInfo.matmul.getInputs()[1]},
            ValueRange{fuseInfo.other}
        );
        newMatmul->setAttrs(fuseInfo.matmul->getAttrs());
        
        fuseInfo.addf.getResult().replaceAllUsesWith(newMatmul.getResult(0));
        // Erase 
        fuseInfo.addf.erase();
        if (fuseInfo.matmul.getResult(0).use_empty()) {
            fuseInfo.matmul.erase();
        }
    }
};



void mlir::triton::FuseAdotBaddCPass::runOnOperation()
{
    ModuleOp module = getOperation();
    SmallVector<FuseInfo> fuseCandidates;
    LOG_DEBUG("== FuseAdotBaddC Pass Start ==\n");
    LOG_DEBUG(module);
    
    module.walk([&](Operation *op) {
        if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
            LOG_DEBUG("Found matmul: " << *matmul);
            auto users = matmul.getResult(0).getUsers();
            size_t userCount = 0;
            Operation *singleUser = nullptr;
            for (auto user : users) {
                userCount++;
                singleUser = user;
            }
            
            // Check conditions
            if (userCount != 1 || op->getBlock() != singleUser->getBlock() || !isa<arith::AddFOp>(singleUser)) {
                llvm::errs() << "Matmul has " << userCount << " users, expected exactly 1. ";
                llvm::errs() << "matumu = " << *matmul << "\n";
                llvm::errs() << "singleUser = " << *singleUser<< "\n";
                module->emitError() << "[" << DEBUG_TYPE << "] The previous pass split errors.";
                signalPassFailure();
                return;
            }
            
            auto addf = dyn_cast<arith::AddFOp>(singleUser);
            Value lhs = addf.getLhs();
            Value rhs = addf.getRhs();
            Value other;
            if (lhs == matmul.getResult(0)) {
                other = rhs;
            } else if (rhs == matmul.getResult(0)) {
                other = lhs;
            } else {
                module->emitError() << "[" << DEBUG_TYPE << "] The previous pass split errors.";
                signalPassFailure();
                return;
            }
            auto fuseInfo = FuseInfo{matmul, addf, other};
            if (canFuse(fuseInfo)) {
                fuseCandidates.push_back(fuseInfo);
            }
        }
    });
    
    LOG_DEBUG("== FuseAdotBaddC will Fuse " << fuseCandidates.size() << " matmul+addf patterns ==\n");
    // Perform fusion for all candidates
    for (auto &fuseInfo : fuseCandidates) {
        performFusion(fuseInfo);
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createFuseAdotBaddCPass()
{
    return std::make_unique<FuseAdotBaddCPass>();
}

} // namespace triton
} // namespace mlir