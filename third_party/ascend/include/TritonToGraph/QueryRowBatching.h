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

// The single scan → row-lift entry point.  SelectedBlockScanRule calls this on
// its own sandboxed copy right after turning the sparse loop into a sequential
// scan, so the row lift runs immediately instead of being queried and later
// scheduled on its own.  The scan rewrite therefore obtains the row lift's real
// result: success means the sandbox can commit both rewrites and the launch
// contract together, failure means the row lift cannot run on this shape and
// the scan rewrite must not commit. The rewrite is in place: the caller owns
// the sandbox, the module may be partially modified on failure and must not be
// the live IR.  Static grids publish the coalesce attributes on the module;
// runtime grids add the hidden extent argument and publish
// hacc.program_grid_transforms.  The L0A/L0C budgets in the resource snapshot
// bound the lifting factor; an unknown capacity (zero) leaves the factor
// unconstrained.
LogicalResult
materializeQueryRowBatchingCandidate(ModuleOp module, triton::FuncOp function,
                                     const ResourceSnapshot &resources);

// Return a privately rewritten and verified plan for query-row batching.  Used
// only by the graph-optimize scheduler; SelectedBlockScanRule uses the
// materialize entry above.  Both entries share one rewrite implementation, so
// the scheduled path and the materialize path cannot diverge.  Unsupported
// inputs return nullptr without touching the original IR.
std::unique_ptr<RewritePlan>
createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                           const ResourceSnapshot &resources);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
