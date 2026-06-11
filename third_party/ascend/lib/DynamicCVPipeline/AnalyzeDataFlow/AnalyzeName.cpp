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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

static constexpr const char *DEBUG_TYPE = "analyze-name";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace mlir;
using namespace triton;

namespace {

static constexpr llvm::StringLiteral interceptrFunc[] {
  "parallel_nsa_kernel_topk",
  "chunk_transform_qk_fwd_kernel",
  "_attn_fwd",
  "_attn_bwd",
  "_kernel_matmul_fp8_row_non_persistent",
  "bmm_kernel",
  "lightning_indexer_grad_kernel",
  "backward_dkdv",
  "backward_dq",
  "backward_sum_o_do",
  "forward_kernel",
  "bwd_qkv_kernel",
  "parallel_rebased_bwd_kernel",
  "parallel_rebased_fwd_kernel",
  "parallel_path_bwd_dkv_kernel",
  "parallel_path_bwd_dq_kernel",
  "chunkwise_bwd_kernel_hdqgl",
  "chunk_mesa_net_fwd_kernel_h",
  "parallel_simple_gla_bwd_kernel",
  "chunkwise_fwd_kernel",
  "_parallel_rebased_bwd_dq",
  "parallel_based_bwd_kernel",
  "chunk_generalized_iplr_delta_rule_fwd_kernel_h",
  "chunk_dplr_fwd_kernel_h"
};

static LogicalResult verifyFuncNames(ModuleOp module)
{
  bool intercepted = false;

  module.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(interceptrFunc, funcOp.getSymName())) {
      return WalkResult::advance();
    }

    LDBG("[INFO]: DynamicCVPipeline is interrupted by function name: " << funcOp.getSymName());
    intercepted = true;
    return WalkResult::interrupt();
  });

  if (!intercepted) {
    return success();
  }

  return failure();
}

} // namespace

void AnalyzeNamePass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeName:\n" << module << "\n");

  if (failed(verifyFuncNames(module))) {
    CVPipeline::setFallbackAttr(module);
    signalPassFailure();
    return;
  }

  LDBG("After AnalyzeName:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeNamePass()
{
  return std::make_unique<AnalyzeNamePass>();
}

} // namespace triton
} // namespace mlir
