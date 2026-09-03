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

#include "TritonControlFlowOpt/HoistPointerChains.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace triton;

namespace {

/// Moves tt.ptr computation chains (tt.advance / tt.make_tensor_ptr /
/// tt.addptr) out of scope::ScopeOp regions. Unlike a clone, the original
/// operations inside the scope are erased and scope.return is repointed at the
/// chain's incoming pointer, so no dead pointer ops remain inside the region.
/// Running this before TritonControlFlowOpt lets the block/tensor pointer
/// decomposition trace pointer offsets without crossing scope boundaries.
static void hoistPointerChainsOutOfScopes(triton::FuncOp funcOp,
                                          RewriterBase &rewriter) {
  funcOp->walk([&](scope::ScopeOp scopeOp) {
    Block &block = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(block.getTerminator());

    for (unsigned i = 0; i < scopeOp.getNumResults(); ++i) {
      auto res = scopeOp.getResult(i);
      if (!isa<triton::PointerType>(res.getType()))
        continue;

      Value returnedVal = returnOp.getResults()[i];
      SetVector<Operation *> slice;
      SmallVector<Value> worklist;
      worklist.push_back(returnedVal);
      bool allOutsideDeps = true;

      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        auto *defOp = v.getDefiningOp();
        if (!defOp) {
          continue;
        }
        if (defOp->getParentRegion() != &scopeOp.getRegion()) {
          continue;
        }
        if (isa<triton::AdvanceOp, triton::MakeTensorPtrOp,
                triton::AddPtrOp>(defOp)) {
          slice.insert(defOp);
          for (auto operand : defOp->getOperands())
            worklist.push_back(operand);
        } else {
          allOutsideDeps = false;
          break;
        }
      }

      if (!allOutsideDeps || slice.empty())
        continue;

      // Locate the chain head's incoming pointer. Its type must match the
      // scope result so scope.return can be repointed at it directly (holds
      // for advance/addptr whose pointer operand has the same type).
      Value rootPtr;
      for (auto *op : slice) {
        Value ptrOperand;
        if (auto advance = dyn_cast<triton::AdvanceOp>(op))
          ptrOperand = advance.getPtr();
        else if (auto addPtr = dyn_cast<triton::AddPtrOp>(op))
          ptrOperand = addPtr.getPtr();
        else if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op))
          ptrOperand = makePtr.getBase();
        if (!ptrOperand)
          continue;
        if (!ptrOperand.getDefiningOp() ||
            !slice.contains(ptrOperand.getDefiningOp())) {
          if (ptrOperand.getType() == res.getType())
            rootPtr = ptrOperand;
          break;
        }
      }
      if (!rootPtr)
        continue;

      // Conservative guard: the chain's intermediate results must only be
      // consumed by the chain itself or by this scope.return operand.
      bool hasEscapingUse = false;
      for (auto *op : slice) {
        for (auto result : op->getResults()) {
          for (auto *user : result.getUsers()) {
            if (user == returnOp.getOperation())
              continue;
            if (slice.contains(user))
              continue;
            hasEscapingUse = true;
            break;
          }
          if (hasEscapingUse)
            break;
        }
        if (hasEscapingUse)
          break;
      }
      if (hasEscapingUse)
        continue;

      // 1. Rebuild the chain outside the scope (head first so the tail's
      //    pointer operand maps to the cloned head).
      rewriter.setInsertionPointAfter(scopeOp);
      IRMapping mapping;
      for (auto *op : llvm::reverse(slice))
        rewriter.clone(*op, mapping);
      Value clonedVal = mapping.lookup(returnedVal);

      // 2. Repoint scope.return at the incoming pointer.
      returnOp->setOperand(i, rootPtr);

      // 3. Erase the original chain inside the scope (tail first).
      for (auto *op : slice)
        rewriter.eraseOp(op);

      // 4. Point uses of the scope result at the cloned chain head.
      rewriter.replaceAllUsesWith(res, clonedVal);
    }
  });
}

} // namespace

namespace mlir::triton {

void TritonHoistPointerChainsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<scope::ScopeDialect, triton::TritonDialect>();
}

void TritonHoistPointerChainsPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(&getContext());
  module->walk([&](triton::FuncOp funcOp) {
    hoistPointerChainsOutOfScopes(funcOp, rewriter);
  });
}

std::unique_ptr<OperationPass<ModuleOp>> createTritonHoistPointerChainsPass() {
  return std::make_unique<TritonHoistPointerChainsPass>();
}

} // namespace mlir::triton
