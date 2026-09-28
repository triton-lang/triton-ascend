
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

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"

static constexpr const char *DEBUG_TYPE = "SplitDataflow";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace CVPipeline;

namespace {

class UnpackScopePass
    : public PassWrapper<UnpackScopePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnpackScopePass)

  // Constructor
  UnpackScopePass() = default;

  // Run the pass
  void runOnOperation() override {
    auto module = getOperation();
    if (hasFallbackAttr(module)) {
      return;
    }
    auto result = module.walk([](scope::ScopeOp scopeOp) {
      llvm::SmallVector<Operation *> unpackedOps;
      auto attrs = scopeOp->getAttrs();
      if (unpackScopeOp(scopeOp, &unpackedOps).failed()) {
        return WalkResult::interrupt();
      }
      for (auto *op : unpackedOps) {
        auto canReuseAttrs = op->getAttrs().size() == 0;
        if (op->getAttrs().size() == 1) {
          canReuseAttrs = op->hasAttr(kBlockId);
        }
        if (canReuseAttrs) {
          // empty attrs, reuse global attr list (in MLIRContext)
          op->setAttrs(attrs);
        } else {
          for (const auto attr : attrs) {
            op->setAttr(attr.getName(), attr.getValue());
          }
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      setFallbackAttr(module, ERRCODE_FAILED);
    }
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-unpack-scopeop";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scope::ScopeDialect>();
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnpackScopePass() {
  return std::make_unique<UnpackScopePass>();
}

} // namespace mlir::triton
