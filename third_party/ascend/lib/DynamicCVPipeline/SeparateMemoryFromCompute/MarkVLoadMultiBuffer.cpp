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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/MarkVLoadMultiBufferPass.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "mark-vload-multi-buffer";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

namespace {

constexpr llvm::StringLiteral kTensorKindAttr("tt.tensor_kind");
constexpr llvm::StringLiteral kMultiBufferAttr("hivm.multi_buffer");
constexpr int32_t kGmTensorKind = 0;
constexpr int32_t kVLoadBufferDepth = 3;
constexpr unsigned kMatmulRightOperandIndex = 1;

Value stripMemRefViews(Value value)
{
    while (auto definingOp = value.getDefiningOp()) {
        if (auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
            value = subviewOp.getSource();
            continue;
        }
        if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
            value = reinterpretCastOp.getSource();
            continue;
        }
        if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
            value = castOp.getSource();
            continue;
        }
        break;
    }
    return value;
}

bool isGmFunctionArgument(Value value)
{
    auto blockArg = dyn_cast<BlockArgument>(stripMemRefViews(value));
    if (!blockArg)
        return false;

    auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp)
        return false;

    auto attr = funcOp.getArgAttrOfType<IntegerAttr>(blockArg.getArgNumber(), kTensorKindAttr);
    return attr && attr.getInt() == kGmTensorKind;
}

memref::AllocOp getCopyTargetAlloc(memref::CopyOp copyOp)
{
    return stripMemRefViews(copyOp.getTarget()).getDefiningOp<memref::AllocOp>();
}

bool isDirectRightOperandOfMatmul(Value tensor)
{
    for (Operation *user : tensor.getUsers()) {
        auto matmulOp = dyn_cast<linalg::MatmulOp>(user);
        if (!matmulOp)
            continue;

        auto inputs = matmulOp.getDpsInputs();
        if (inputs.size() > kMatmulRightOperandIndex && inputs[kMatmulRightOperandIndex] == tensor)
            return true;
    }
    return false;
}

bool doesAllocFeedDirectMatmulRightOperand(memref::AllocOp allocOp)
{
    SmallVector<Value> worklist;
    llvm::SmallPtrSet<Operation *, 8> visited;
    worklist.push_back(allocOp.getMemref());

    while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        for (Operation *user : value.getUsers()) {
            if (!visited.insert(user).second)
                continue;

            if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user)) {
                if (isDirectRightOperandOfMatmul(toTensorOp.getResult()))
                    return true;
                continue;
            }

            if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
                worklist.push_back(subviewOp.getResult());
                continue;
            }

            if (auto castOp = dyn_cast<memref::CastOp>(user)) {
                worklist.push_back(castOp.getResult());
                continue;
            }

            if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(user))
                worklist.push_back(reinterpretCastOp.getResult());
        }
    }

    return false;
}

annotation::MarkOp getExistingMark(Value value)
{
    for (Operation *user : value.getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(user))
            return markOp;
    }
    return nullptr;
}

void markAllocWithMultiBufferDepth(memref::AllocOp allocOp, OpBuilder &builder)
{
    annotation::MarkOp markOp = getExistingMark(allocOp.getMemref());
    if (!markOp) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(allocOp);
        markOp = builder.create<annotation::MarkOp>(allocOp.getLoc(), allocOp.getMemref());
    }

    markOp->setAttr(kMultiBufferAttr, builder.getI32IntegerAttr(kVLoadBufferDepth));
}

} // namespace

void MarkVLoadMultiBufferPass::runOnOperation()
{
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    unsigned markedCount = 0;
    llvm::SmallPtrSet<Operation *, 8> markedAllocs;

    module.walk([&](memref::CopyOp copyOp) {
        if (!isGmFunctionArgument(copyOp.getSource()))
            return;

        memref::AllocOp allocOp = getCopyTargetAlloc(copyOp);
        if (!allocOp || !doesAllocFeedDirectMatmulRightOperand(allocOp))
            return;
        if (!markedAllocs.insert(allocOp.getOperation()).second)
            return;

        markAllocWithMultiBufferDepth(allocOp, builder);
        ++markedCount;
    });

    LDBG("Marked " << markedCount << " direct matmul right operand GM load alloc(s)");
}

void MarkVLoadMultiBufferPass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<annotation::AnnotationDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect>();
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMarkVLoadMultiBufferPass()
{
    return std::make_unique<MarkVLoadMultiBufferPass>();
}

void registerMarkVLoadMultiBufferPasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createMarkVLoadMultiBufferPass(); });
}

} // namespace triton
} // namespace mlir
