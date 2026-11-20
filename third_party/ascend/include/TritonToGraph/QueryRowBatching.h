/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
#define TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

struct ResourceSnapshot;

// Rule A calls this after turning the loop into a sequential scan in its
// private sandbox, so Rule B lifts the rows in place; success means the
// sandbox can commit both rewrites together.
LogicalResult
materializeQueryRowBatchingCandidate(ModuleOp module, triton::FuncOp function,
                                     const ResourceSnapshot &resources);

// Return a privately rewritten and verified plan for the Rule B scheduling
// path; nullptr when the shape is unsupported.
std::unique_ptr<RewritePlan>
createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                           const ResourceSnapshot &resources);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
