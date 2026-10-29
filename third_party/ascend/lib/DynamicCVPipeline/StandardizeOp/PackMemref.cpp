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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

#include "DynamicCVPipeline/Common/Analysis.h"
#include "DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "pack-memref";
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    llvm::dbgs() << '[' << DEBUG_TYPE << "] ";                                 \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

namespace {

class PackMemrefPass
    : public PassWrapper<PackMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackMemrefPass);

  PackMemrefPass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<scope::ScopeDialect, hivm::HIVMDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pack-memref";
  }
};

} // namespace

// Group hivm.hir.custom together with its own local buffers into a scope unit.
static void packCustomOpUnit(hivm::CustomOp customOp);
// Module driver: every custom op is packed together with its buffers.
static void packAllCustomOpUnits(ModuleOp moduleOp);

void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  packAllCustomOpUnits(moduleOp);
}

// Pack every hivm.hir.custom together with its local buffers. Each custom op
// owns its buffers, so pack them before anything else looks at them.
static void packAllCustomOpUnits(ModuleOp moduleOp) {
  SmallVector<hivm::CustomOp> customOps;
  moduleOp.walk(
      [&](hivm::CustomOp customOp) { customOps.push_back(customOp); });
  for (hivm::CustomOp customOp : customOps) {
    packCustomOpUnit(customOp);
  }
}

/// Pack a custom op together with its buffers into one `scope::ScopeOp`
/// carrying the analysed core type.
static void packCustomOpUnit(hivm::CustomOp customOp) {
  auto anaRes = CustomOpAnalysis::get(customOp);
  if (llvm::failed(anaRes)) {
    LDBG("warning: failed to analyze " << customOp);
    return;
  }
  auto &ana = anaRes.value();
  LDBG("packing " << ana.customOp);

  auto scopeOp = packScopeOp(ana.relevantOps);
  if (!scopeOp) {
    // Leaving the custom op unpacked is safe: its buffers keep the raw
    // `bufferization.to_tensor` chain that
    // OpClassifierPass::matchToTensorPattern already recognises.
    LDBG("failed to pack scope: " << customOp << "; fallback to default");
    return;
  }
  // Stamping is not a hint: OpClassifierPass treats any `ssbuffer.core_type` as
  // authoritative and never re-colours it, so an undetermined core type must be
  // left off for the op to be classified by its data flow.
  if (ana.coreType != UNDETERMINED) {
    scopeOp->setAttr(
        kCoreType,
        StringAttr::get(scopeOp->getContext(), coreTypeToString(ana.coreType)));
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CVPipeline::createPackMemrefPass() {
  return std::make_unique<PackMemrefPass>();
}
