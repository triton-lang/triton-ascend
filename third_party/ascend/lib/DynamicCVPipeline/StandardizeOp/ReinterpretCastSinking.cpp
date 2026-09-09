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
#include <memory>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/ReinterpretCastSinking.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "ReinterpretCastSinking";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

// Sink one reinterpret_cast: insert a clone immediately before every user.
// 1. Snapshot all result users.
// 2. Clone cast before each user.
// 3. Redirect each user to its clone.
// 4. Erase original when all uses redirected.
int sinkReinterpretCastOp(memref::ReinterpretCastOp reinterpretOp) {
  Value result = reinterpretOp.getResult();

  // Snapshot users before mutating uses.
  SmallVector<Operation *> users;
  for (Operation *user : result.getUsers())
    users.push_back(user);

  LOG_DEBUG("Processing " << reinterpretOp << " with " << users.size()
                          << " users");

  OpBuilder builder(reinterpretOp->getContext());
  int cloned = 0;

  // Clone per user: block may split into multiple block_ids;
  // CSE drops redundant clones.
  for (auto *user : users) {
    builder.setInsertionPoint(user);
    Value clonedResult =
        builder.clone(*reinterpretOp.getOperation())->getResult(0);
    user->replaceUsesOfWith(result, clonedResult);
    ++cloned;
  }

  // Erase the original once every use has been redirected.
  if (cloned > 0 && result.use_empty()) {
    LOG_DEBUG("Erasing original: " << reinterpretOp);
    reinterpretOp->erase();
  }

  return cloned;
}

} // namespace

namespace mlir::triton::CVSplit {

void ReinterpretCastSinkingPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Collect all reinterpret_cast ops first, since we'll be modifying the IR.
  SmallVector<memref::ReinterpretCastOp> opsToProcess;
  mod->walk([&](memref::ReinterpretCastOp op) { opsToProcess.push_back(op); });

  int totalCloned = 0;
  for (auto reinterpretOp : opsToProcess) {
    totalCloned += sinkReinterpretCastOp(reinterpretOp);
  }

  LOG_DEBUG("Processed " << opsToProcess.size() << " ops, cloned "
                         << totalCloned << " times");
}

std::unique_ptr<OperationPass<ModuleOp>> createReinterpretCastSinkingPass() {
  return std::make_unique<ReinterpretCastSinkingPass>();
}

} // namespace mlir::triton::CVSplit
